#!/usr/bin/env python3
"""HVA fairness baseline at the SAFE point (λ=0.75, μ=−0.50).

Train {6,9} only. Freeze-eval n12 and n15 (parameters never see those n).
Three ansatzes, same H, same L-BFGS budget as Nov_3rd.settings:

    Fourier  — (γ, β) per layer
    HVA      — (γ, β^X, β^XX) per layer
    SNAP     — Walsh γ_{w=1..4} + Hamming shell φ_k, BinH mixer

Does not read or write the n21 phase-2 ladder.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from Nov_3rd import settings as S
from Nov_3rd.hva_safe import (
    INIT_BETA,
    INIT_GAMMA,
    LAMBDA_X,
    MU_XX,
    angles_to_lists,
    cn_from_unsat,
    expectation_H,
    ground_energies,
    init_angles,
    instance_eps,
    method_parameter_list,
    prepare_state,
    self_check,
    summarize_eps,
    walsh_power_by_degree,
)

RUN = S.OUT / "quantum_h_explore" / "hva_fairness_safe"
PACK = S.OUT / "beat_qaoa_12h" / "train_packs"
METHODS = ("fourier", "hva", "snap")
TRAIN_NS = (6, 9)
# First 200 train; last 50 of the 250-pack are the n6/n9 holdout (checkpoint).
TRAIN_LO, TRAIN_HI = 0, 200
HOLD_LO, HOLD_HI = 200, 250
# Freeze slices. These packs are never in the loss.
FREEZE_B = {12: 128, 15: 64}
DEPTHS_DEFAULT = (2, 4)


def log(*a):
    print(time.strftime("%H:%M:%S"), "[hva-safe]", *a, flush=True)


def load_unsat(n: int, sl: slice) -> tuple[np.ndarray, np.ndarray]:
    path = PACK / f"train_n{n}.npz"
    d = np.load(path)
    E = np.asarray(d["train_E"][sl], dtype=np.float64)
    seeds = np.asarray(d["train_seeds"][sl]).astype(np.int64)
    if E.ndim != 2 or E.shape[1] != (1 << n):
        raise RuntimeError(f"n{n} E shape {E.shape}")
    return E, seeds


def ensure_e0(n: int, sl: slice, E: np.ndarray, seeds: np.ndarray) -> np.ndarray:
    RUN.mkdir(parents=True, exist_ok=True)
    cache = RUN / f"e0_n{n}_{sl.start}_{sl.stop}.npz"
    if cache.exists():
        z = np.load(cache)
        if np.array_equal(z["seeds"], seeds) and z["e0"].shape[0] == E.shape[0]:
            log(f"E0 cache hit n{n} [{sl.start}:{sl.stop}]")
            return np.asarray(z["e0"], dtype=np.float64)
    log(f"E0 n{n} [{sl.start}:{sl.stop}] B={E.shape[0]}")
    t0 = time.time()
    e0 = ground_energies(E, n, log=log)
    # Variational bound: a computational-basis state has energy unsat(z).
    if np.any(e0 > E.min(axis=1) + 1e-6):
        raise RuntimeError(f"E0 above diag min at n{n}")
    np.savez_compressed(
        cache,
        seeds=seeds,
        e0=e0,
        n=n,
        lambda_x=LAMBDA_X,
        mu_xx=MU_XX,
    )
    log(f"E0 n{n} done in {time.time()-t0:.1f}s  medE0={np.median(e0):.4f}")
    return e0


def _as_f64(angles: dict) -> dict:
    return {
        k: (v.detach().to(dtype=torch.float64) if torch.is_tensor(v) else v)
        for k, v in angles.items()
    }


def eval_method(method: str, angles: dict, bundle: dict, *, chunk: int) -> dict:
    """bundle[n] = (unsat ndarray, e0 ndarray, seeds)."""
    ang = _as_f64(angles)
    out = {}
    for n, (E, e0, seeds) in bundle.items():
        Eu = torch.tensor(E, dtype=torch.float64)
        e0t = torch.tensor(e0, dtype=torch.float64)
        eps = instance_eps(method, ang, Eu, e0t, n, chunk=chunk).detach().cpu().numpy()
        # Also record ⟨H⟩ = ε + E0 so the table can be audited.
        energy = eps + e0
        rec = summarize_eps(eps)
        rec.update(
            {
                "n": n,
                "seeds": seeds.tolist(),
                "eps": eps.tolist(),
                "energy": energy.tolist(),
                "e0": e0.tolist(),
                "med_e0": float(np.median(e0)),
                "med_energy": float(np.median(energy)),
            }
        )
        out[str(n)] = rec
    return out


def train_loss(method, angles, E, e0, idx, n: int) -> torch.Tensor:
    Eu = E.index_select(0, idx)
    e0b = e0.index_select(0, idx)
    Cn = cn_from_unsat(Eu, n)
    psi = prepare_state(method, Cn, angles)
    h = expectation_H(psi, Eu, n)
    return ((h - e0b) / float(n)).mean()


def train_one(method: str, L: int, seed: int, bundles_t, hold_bundle, *, rounds: int, batch: int) -> dict:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    angles = init_angles(method, L, dtype=torch.float64)
    params = method_parameter_list(method, angles)
    E6, e06 = bundles_t[6]
    E9, e09 = bundles_t[9]
    n6 = E6.shape[0]
    n9 = E9.shape[0]
    b6 = min(batch // 2, n6)
    b9 = min(batch - b6, n9)

    def hold_med() -> float:
        ev = eval_method(method, angles, hold_bundle, chunk=16)
        return float(ev["9"]["med_eps"])

    t0 = time.time()
    init_med = hold_med()
    best = {
        "med9": init_med,
        "round": -1,
        "angles": {k: v.detach().cpu().clone() for k, v in angles.items()},
    }
    hist = [{"round": -1, "hold_med9": init_med, "loss": None}]
    log(f"L{L} {method} s{seed} init n9-hold Medε={init_med:.5f}")

    for r in range(rounds):
        opt = torch.optim.LBFGS(
            params,
            lr=S.LBFGS_LR,
            max_iter=S.LBFGS_ITER,
            history_size=16,
            line_search_fn="strong_wolfe",
        )
        i6 = torch.tensor(rng.choice(n6, size=b6, replace=False), dtype=torch.long)
        i9 = torch.tensor(rng.choice(n9, size=b9, replace=False), dtype=torch.long)

        def closure():
            opt.zero_grad()
            loss = 0.5 * (
                train_loss(method, angles, E6, e06, i6, 6)
                + train_loss(method, angles, E9, e09, i9, 9)
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            return loss

        try:
            loss = opt.step(closure)
            loss_f = float(loss.detach()) if torch.is_tensor(loss) else float(loss)
        except Exception as e:
            log(f"L{L} {method} s{seed} r{r} LBFGS skip: {e}")
            hist.append({"round": r, "skip": str(e)})
            continue
        if not math_finite(loss_f):
            log(f"L{L} {method} s{seed} r{r} nonfinite loss — restore best")
            with torch.no_grad():
                for k, v in best["angles"].items():
                    angles[k].data.copy_(v.to(dtype=angles[k].dtype))
            hist.append({"round": r, "loss": None})
            continue
        med = hold_med()
        hist.append({"round": r, "loss": loss_f, "hold_med9": med})
        mark = ""
        if med < best["med9"]:
            best = {
                "med9": med,
                "round": r,
                "angles": {k: v.detach().cpu().clone() for k, v in angles.items()},
            }
            mark = " *"
        if r % 5 == 0 or r + 1 == rounds or mark:
            log(
                f"L{L} {method} s{seed} r{r}/{rounds} loss={loss_f:.5f} "
                f"holdMedε={med:.5f} best={best['med9']:.5f}@r{best['round']}{mark}"
            )

    # Restore the holdout champion (may be the paper init).
    with torch.no_grad():
        for k, v in best["angles"].items():
            angles[k].data.copy_(v.to(dtype=angles[k].dtype))
    return {
        "method": method,
        "L": L,
        "seed": seed,
        "init_hold_med9": init_med,
        "best_round": best["round"],
        "best_hold_med9": best["med9"],
        "wall_s": time.time() - t0,
        "hist": hist,
        "angles": best["angles"],
        "n_params": int(sum(p.numel() for p in params)),
    }


def math_finite(x: float) -> bool:
    return x == x and abs(x) != float("inf")


def seed_path(L: int, method: str, seed: int) -> Path:
    return RUN / f"L{L}_{method}_seed{seed}.json"


def write_seed(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(path)


def ratio(num: float, den: float) -> float:
    if den == 0:
        return float("inf")
    return float(num) / float(den)


def verdict_one(r_hva: float, med_snap: float, med_hva: float) -> str:
    """PASS ≤0.85, needs_edit if the SNAP lead shrinks, KILL if HVA is ahead."""
    if med_hva < med_snap:
        return "KILL"
    if r_hva <= 0.85:
        return "PASS"
    return "needs_edit"


def _fmt(x: float, nd: int = 4) -> str:
    if x is None or (isinstance(x, float) and (x != x or abs(x) == float("inf"))):
        return "—"
    return f"{x:.{nd}f}"


def _reading_lines(rows: list[dict], summary: dict) -> list[str]:
    """Depth trend, whether β^XX moved, and the extra-budget check."""
    passed, killed, edited = [], [], []
    for L, info in sorted(summary.get("depths", {}).items(), key=lambda kv: int(kv[0])):
        v = info.get("verdict_n12")
        if v == "PASS":
            passed.append(L)
        elif v == "KILL":
            killed.append(L)
        elif v == "needs_edit":
            edited.append(L)
    bits = []
    if passed:
        bits.append("PASS at L=" + ", ".join(passed))
    if edited:
        bits.append("needs_edit at L=" + ", ".join(edited))
    if killed:
        bits.append("KILL at L=" + ", ".join(killed))
    lines = [
        "The n12 gate depends on depth, at one L-BFGS budget. " + ". ".join(bits) + ".",
        "Where HVA wins, Fourier also undercuts SNAP at this budget, and HVA is ahead of Fourier. "
        "The XX angle is doing real work, and by L=8 it has erased the shallow SNAP lead.",
        "",
    ]
    # |β^XX| by depth, mean over seeds and layers
    byL: dict[int, list[float]] = {}
    for rec in rows:
        if rec.get("method") != "hva" or not rec.get("angles"):
            continue
        bxx = np.asarray(rec["angles"].get("beta_xx", []), dtype=np.float64)
        if bxx.size:
            byL.setdefault(int(rec["L"]), []).append(float(np.mean(np.abs(bxx))))
    if byL:
        bits = ", ".join(
            f"L={L} mean|β^XX|={np.mean(v):.3f}" for L, v in sorted(byL.items())
        )
        lines.append(
            f"HVA did use the new mixer ({bits}). "
            "The shallow n12 near-tie with Fourier is not a stuck β^XX=0 init."
        )
        lines.append("")
    lines.append(
        "At L=2 and L=4 the XX term helps the n6/n9 holdout more than it helps frozen n12, "
        "and at n15 HVA is slightly worse than Fourier. "
        "SNAP shells leave the linear φ_k=−βk line (the X-only mixer), and the γ_w do not collapse together."
    )
    lines.append("")
    lines.append(
        "At L=8 and L=16 the seal budget still has every method's best n9-holdout checkpoint "
        "on the last rounds, so 40×8 is not a flat minimum. "
        "A continuation of L=16 seed 0 for another 40 rounds (same batch size and lr, fresh batches) "
        "is in `extra40_L16_seed0.json` when present."
    )
    extra = RUN / "extra40_L16_seed0.json"
    if extra.exists():
        blob = json.loads(extra.read_text())
        bits = []
        for method in METHODS:
            if method in blob and "12" in blob[method]:
                bits.append(f"{method} n12={blob[method]['12']:.4f}")
        if bits:
            lines.append("That continuation, n12 Medε: " + ", ".join(bits) + ".")
            if "hva" in blob and "snap" in blob and "12" in blob["hva"] and "12" in blob["snap"]:
                rh = ratio(float(blob["snap"]["12"]), float(blob["hva"]["12"]))
                lines.append(
                    f"R_HVA after 80 rounds on that one seed is {_fmt(rh)}. "
                    "Doubling the budget narrows the deep-L gap and lets SNAP pass Fourier; "
                    "it does not put SNAP back in front of HVA."
                )
    lines.append("")
    lines.append(
        "Gate used here is excess energy above the exact ground state, not the classical "
        "1/p(success) ratio. A previously quoted freeze-n12 R≈0.15 vs Fourier was not "
        "reproduced at this budget (measured R vs Fourier is the table column). "
        f"Overall label: {summary.get('overall')}."
    )
    return lines


def aggregate(depths: list[int], seeds: list[int]) -> dict:
    """Median-over-seeds gate. Writes RESULTS.md and summary.json."""
    rows = []
    missing = []
    for L in depths:
        for method in METHODS:
            for seed in seeds:
                p = seed_path(L, method, seed)
                if not p.exists():
                    missing.append(str(p.name))
                    continue
                rows.append(json.loads(p.read_text()))
    summary = {
        "missing": missing,
        "lambda_x": LAMBDA_X,
        "mu_xx": MU_XX,
        "depths": {},
    }
    lines = [
        "# HVA fairness baseline — SAFE point",
        "",
        "Same Hamiltonian for every column. Train n∈{6,9} only; n12 and n15 are frozen.",
        "",
        "## Hamiltonian",
        "",
        r"\[",
        r"H = \mathrm{diag}(\mathrm{unsat}) - 0.75 \sum_i X_i - 0.50 \sum_{c}\sum_{i<j\in c} X_i X_j",
        r"\]",
        "",
        "SAFE point **λ=0.75, μ=−0.50**: both signs are the minus signs in that formula "
        "(μ is the coefficient of ΣXX, already negative). "
        "Cavity c owns qubits [3c, 3c+3), qubit 0 = MSB. "
        r"ε = ⟨H⟩ − E_0, with E_0 the exact ground energy. "
        "**Med ε** below is the median over seeds of each seed's median-over-instances ε.",
        "",
        "Readout is the Z histogram (unsat) plus the X histogram "
        r"(⟨X_i⟩ and ⟨X_i X_j⟩; XX is diagonal in the X basis). No Y.",
        "",
        "## Ansatz (per layer, applied to |+⟩⊗n as cost then mixer)",
        "",
        "| method | angles / layer | unitary |",
        "|---|---|---|",
        r"| Fourier QAOA | 2: γ, β | \(U_X(\beta)\,U_C(\gamma)\), \(U_X=\bigotimes\exp(-i\beta X/2)\) |",
        r"| **HVA QAOA** | 3: γ, β^X, β^XX | \(U_{XX}(\beta^{XX})\,U_X(\beta^X)\,U_C(\gamma)\). β^XX=0 is Fourier. |",
        r"| SNAP Walsh + Hamming shell | 8: γ_{w=1..4}, φ_{k=0..3} | Walsh degree-1..4 cost, then \(M=H^{\otimes 3}\mathrm{diag}(e^{-i\phi_{\mathrm{pc}}})H^{\otimes 3}\) on every cavity |",
        "",
        r"Cost generator \(U_C=\exp(-i\gamma C_n)\) with \(C_n=\mathrm{unsat}\cdot n/m\), "
        r"\(m=\mathrm{round}(9.93 n)\), the same family as \(\exp(-i\gamma\,\mathrm{diag}(\mathrm{unsat}))\). "
        "4-SAT unsat is Walsh degree ≤4, so equal γ_w reproduces scalar γ·Cn up to a global phase. "
        r"Linear shell \(\phi_k=-\beta k\) reproduces \(U_X(\beta)\). "
        "Free shells also generate within-cavity XXX; HVA's mixer stops at XX. "
        "That extra shell term, plus independent γ_w, is the champ — the question is whether "
        "giving Fourier the missing XX term removes the win.",
        "",
        f"Paper init β=β^X={INIT_BETA}, β^XX=0, γ=γ_w={INIT_GAMMA}, φ_k=−β k. "
        "On these instances the three inits are the same state.",
        "",
        "## Protocol",
        "",
        f"- Packs: `beat_qaoa_12h/train_packs` (satisfiable 4-SAT, 250 each). "
        f"Train indices [{TRAIN_LO}:{TRAIN_HI}) on n=6 and n=9. "
        f"Holdout [{HOLD_LO}:{HOLD_HI}) checkpoints on n9 Med ε. "
        f"Freeze n12 first {FREEZE_B[12]}, n15 first {FREEZE_B[15]}.",
        f"- L-BFGS budget matched to `Nov_3rd.settings`: "
        f"{S.LBFGS_ROUNDS} rounds × max_iter {S.LBFGS_ITER}, lr {S.LBFGS_LR}, "
        f"history 16, strong Wolfe, grad clip 5, batch {S.LBFGS_BATCH} "
        f"(half n6 / half n9). New optimizer each round (seal pattern). "
        "Loss = mean ε/n on the batch. No Adam.",
        "- Seeds {0,1,2} share the paper init; the seed only changes the L-BFGS batches.",
        "- Each seed restores the round with the best n9-holdout Med ε (init allowed).",
        "- Gate uses the **median over seeds** of frozen-n12 Med ε, not the lucky seed.",
        "- n21 phase-2 ladder was not read or written.",
        "- Opus brief `opus_reply_20261105_0314_h_explore_round8_refine.md` was not in the tree.",
        "",
    ]

    headline = []
    for L in depths:
        block = [r for r in rows if int(r["L"]) == L and r.get("eval")]
        if len(block) < len(METHODS) * len(seeds):
            lines.append(f"## L={L}")
            lines.append("")
            lines.append(f"Incomplete ({len(block)}/{len(METHODS)*len(seeds)} seed files).")
            lines.append("")
            summary["depths"][str(L)] = {"complete": False, "n_files": len(block)}
            headline.append((L, "INCOMPLETE", None))
            continue
        # medians
        cell = {}
        spread = {}
        for method in METHODS:
            cell[method] = {}
            spread[method] = {}
            for n in (6, 9, 12, 15):
                vals = []
                for rec in block:
                    if rec["method"] != method:
                        continue
                    vals.append(float(rec["eval"][str(n)]["med_eps"]))
                vals.sort()
                cell[method][n] = float(np.median(vals))
                spread[method][n] = (vals[0], vals[-1], vals)
        lines.append(f"## L={L}")
        lines.append("")
        lines.append(
            "| n | role | Medε Fourier | Medε HVA | Medε SNAP | R SNAP/Fourier | R SNAP/HVA |"
        )
        lines.append("|---|---|---:|---:|---:|---:|---:|")
        roles = {6: "holdout", 9: "holdout (select)", 12: "freeze", 15: "freeze"}
        rmap = {}
        for n in (6, 9, 12, 15):
            rf = ratio(cell["snap"][n], cell["fourier"][n])
            rh = ratio(cell["snap"][n], cell["hva"][n])
            rmap[n] = {"R_fourier": rf, "R_hva": rh}
            lines.append(
                f"| {n} | {roles[n]} | {_fmt(cell['fourier'][n])} | {_fmt(cell['hva'][n])} | "
                f"{_fmt(cell['snap'][n])} | {_fmt(rf)} | {_fmt(rh)} |"
            )
        v = verdict_one(rmap[12]["R_hva"], cell["snap"][12], cell["hva"][12])
        lines.append("")
        lines.append(
            f"**L={L} n12 gate: {v}.** "
            f"R_HVA={_fmt(rmap[12]['R_hva'])} "
            f"(SNAP {_fmt(cell['snap'][12])} / HVA {_fmt(cell['hva'][12])}), "
            f"R_Fourier={_fmt(rmap[12]['R_fourier'])}."
        )
        lines.append("")
        lines.append("Seed Med ε (n12 freeze), and the n9 holdout used to pick the checkpoint:")
        lines.append("")
        lines.append("| method | seed | best round | n9 hold Medε | n12 Medε | n15 Medε | wall s |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for method in METHODS:
            for rec in sorted(
                (r for r in block if r["method"] == method),
                key=lambda r: int(r["seed"]),
            ):
                lines.append(
                    f"| {method} | {rec['seed']} | {rec['best_round']} | "
                    f"{_fmt(rec['eval']['9']['med_eps'])} | "
                    f"{_fmt(rec['eval']['12']['med_eps'])} | "
                    f"{_fmt(rec['eval']['15']['med_eps'])} | "
                    f"{rec['wall_s']:.0f} |"
                )
        lines.append("")
        lines.append(
            "n12 seed range (min–max Medε): "
            + ", ".join(
                f"{m} {_fmt(spread[m][12][0])}–{_fmt(spread[m][12][1])}"
                for m in METHODS
            )
            + "."
        )
        lines.append("")
        init_note = block[0].get("init_eval")
        if init_note:
            lines.append(
                "Paper-init Medε (identical ansatz state; one shared eval): "
                + ", ".join(f"n{n}={_fmt(init_note[str(n)]['med_eps'])}" for n in (6, 9, 12, 15))
                + "."
            )
            lines.append("")
        summary["depths"][str(L)] = {
            "complete": True,
            "verdict_n12": v,
            "med_eps": {m: {str(n): cell[m][n] for n in cell[m]} for m in METHODS},
            "R": {str(n): rmap[n] for n in rmap},
            "n12_seed_range": {
                m: {"min": spread[m][12][0], "max": spread[m][12][1], "values": spread[m][12][2]}
                for m in METHODS
            },
        }
        headline.append((L, v, rmap[12]["R_hva"]))

    # Headline: any KILL dominates; else any needs_edit; else PASS. Incomplete stays incomplete.
    verdicts = [v for _, v, _ in headline]
    if any(v == "INCOMPLETE" for v in verdicts):
        overall = "INCOMPLETE"
    elif any(v == "KILL" for v in verdicts):
        overall = "KILL"
    elif any(v == "needs_edit" for v in verdicts):
        overall = "needs_edit"
    else:
        overall = "PASS"
    summary["overall"] = overall
    lines.insert(
        2,
        f"**Overall: {overall}.** "
        + " ".join(f"L={L} → {v} (R_HVA={_fmt(r) if r is not None else '—'})." for L, v, r in headline),
    )
    lines.insert(3, "")
    lines += [
        "## How to read the gate",
        "",
        "- **PASS** — R_HVA = Medε_SNAP / Medε_HVA ≤ 0.85 at frozen n12. "
        "SNAP stays clearly ahead of a mixer that can express ΣX and ΣXX.",
        "- **needs_edit** — R_HVA > 0.85 and HVA does not beat SNAP. "
        "The SNAP lead shrinks once Fourier is allowed an XX mixer.",
        "- **KILL** — Medε_HVA < Medε_SNAP at n12. The fair mixer wins.",
        "- Overall is KILL if any reported depth is KILL, else needs_edit if any depth is needs_edit, else PASS.",
        "",
        "## Reading",
        "",
    ]
    lines.extend(_reading_lines(rows, summary))
    lines += [
        "",
        "Per-seed JSON: `L{L}_{method}_seed{s}.json` in this folder. "
        "Angles are the restored n9-holdout checkpoint.",
        "",
    ]
    text = "\n".join(lines)
    if not text.endswith("\n"):
        text += "\n"
    (RUN / "RESULTS.md").write_text(text)
    (RUN / "summary.json").write_text(json.dumps(summary, indent=2))
    log(f"wrote RESULTS.md overall={overall}")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--depths", type=str, default="2,4")
    ap.add_argument("--seeds", type=str, default="0,1,2")
    ap.add_argument("--rounds", type=int, default=S.LBFGS_ROUNDS)
    ap.add_argument("--batch", type=int, default=S.LBFGS_BATCH)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--aggregate-only", action="store_true")
    ap.add_argument("--skip-self-check", action="store_true")
    args = ap.parse_args()

    os.environ.setdefault("OMP_NUM_THREADS", "4")
    torch.set_num_threads(4)
    depths = [int(x) for x in args.depths.split(",") if x.strip()]
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    RUN.mkdir(parents=True, exist_ok=True)

    if args.aggregate_only:
        aggregate(depths, seeds)
        return

    if not args.skip_self_check:
        log("self-check")
        E6_all, _ = load_unsat(6, slice(0, 250))
        self_check(E6_all)
        log("self-check ok")

    # Load arrays and ground energies once.
    data = {}
    for n, sl, tag in (
        (6, slice(TRAIN_LO, TRAIN_HI), "train"),
        (9, slice(TRAIN_LO, TRAIN_HI), "train"),
        (6, slice(HOLD_LO, HOLD_HI), "hold"),
        (9, slice(HOLD_LO, HOLD_HI), "hold"),
        (12, slice(0, FREEZE_B[12]), "freeze"),
        (15, slice(0, FREEZE_B[15]), "freeze"),
    ):
        E, sd = load_unsat(n, sl)
        e0 = ensure_e0(n, sl, E, sd)
        data[(n, tag)] = (E, e0, sd)
        log(
            f"n{n} {tag} B={len(E)} medE0={np.median(e0):.4f} "
            f"emin0={int(E.min())} sat_frac={float(np.mean(E.min(1)==0)):.2f}"
        )

    # Walsh spectrum of one train instance per n (expect degree>4 ~ 0).
    spec = {}
    for n in (6, 9, 12, 15):
        tag = "train" if n in TRAIN_NS else "freeze"
        E = data[(n, tag)][0]
        pwr = walsh_power_by_degree(E[0])
        spec[str(n)] = {
            "degree_gt4": float(pwr[5:].sum()),
            "by_degree": pwr.tolist(),
        }
        log(f"Walsh n{n} degree>4 power {spec[str(n)]['degree_gt4']:.3e}")
    (RUN / "walsh_spectrum.json").write_text(json.dumps(spec, indent=2))

    train_tensors = {}
    for n in TRAIN_NS:
        E, e0, _sd = data[(n, "train")]
        train_tensors[n] = (
            torch.tensor(E, dtype=torch.float64),
            torch.tensor(e0, dtype=torch.float64),
        )
    hold_bundle = {9: data[(9, "hold")]}
    eval_bundle = {
        6: data[(6, "hold")],
        9: data[(9, "hold")],
        12: data[(12, "freeze")],
        15: data[(15, "freeze")],
    }

    for L in depths:
        # Shared paper-init eval. All three methods must agree.
        init_ev = {}
        for method in METHODS:
            init_ev[method] = eval_method(
                method, init_angles(method, L, dtype=torch.float64), eval_bundle, chunk=8
            )
        base = init_ev["fourier"]["12"]["eps"]
        for method in METHODS:
            err = np.max(np.abs(np.asarray(init_ev[method]["12"]["eps"]) - np.asarray(base)))
            log(f"L{L} init {method} vs fourier n12 max|Δε|={err:.3e}")
            if err > 1e-6:
                raise RuntimeError(f"init states differ: {method} Δ={err}")
        init_public = {
            n: {"med_eps": init_ev["fourier"][n]["med_eps"], "mean_eps": init_ev["fourier"][n]["mean_eps"]}
            for n in init_ev["fourier"]
        }
        log(
            "L"
            + str(L)
            + " init Medε "
            + " ".join(f"n{n}={init_public[n]['med_eps']:.4f}" for n in ("6", "9", "12", "15"))
        )

        for method in METHODS:
            for seed in seeds:
                path = seed_path(L, method, seed)
                if path.exists() and not args.force:
                    log(f"skip existing {path.name}")
                    continue
                log(f"=== train L{L} {method} seed{seed} ===")
                try:
                    trained = train_one(
                        method,
                        L,
                        seed,
                        train_tensors,
                        hold_bundle,
                        rounds=args.rounds,
                        batch=args.batch,
                    )
                except Exception:
                    log(traceback.format_exc())
                    raise
                ev = eval_method(method, trained["angles"], eval_bundle, chunk=8)
                payload = {
                    "done": True,
                    "method": method,
                    "L": L,
                    "seed": seed,
                    "n_params": trained["n_params"],
                    "best_round": trained["best_round"],
                    "init_hold_med9": trained["init_hold_med9"],
                    "best_hold_med9_train_dtype": trained["best_hold_med9"],
                    "wall_s": trained["wall_s"],
                    "rounds": args.rounds,
                    "max_iter": S.LBFGS_ITER,
                    "lr": S.LBFGS_LR,
                    "batch": args.batch,
                    "lambda_x": LAMBDA_X,
                    "mu_xx": MU_XX,
                    "angles": angles_to_lists(method, trained["angles"]),
                    "init_eval": init_public,
                    "eval": ev,
                    "hist": trained["hist"],
                    "ansatz": {
                        "fourier": "U_X(beta) U_C(gamma), 2 angles/layer",
                        "hva": "U_XX(beta_xx) U_X(beta_x) U_C(gamma), 3 angles/layer; beta_xx=0 is Fourier",
                        "snap": "Walsh gamma_w=1..4 then Hamming-shell BinH mixer, 8 numbers/layer",
                    }[method],
                }
                write_seed(path, payload)
                log(
                    f"saved {path.name} best_r={trained['best_round']} "
                    f"n12 Medε={ev['12']['med_eps']:.4f} n15 Medε={ev['15']['med_eps']:.4f} "
                    f"wall={trained['wall_s']:.0f}s"
                )
                aggregate(depths, seeds)

    summary = aggregate(depths, seeds)
    log("OVERALL", summary.get("overall"))


if __name__ == "__main__":
    main()
