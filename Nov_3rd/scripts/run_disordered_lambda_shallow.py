#!/usr/bin/env python3
"""Disordered transverse field at shallow depth. Gates stay the SAFE-point ansatzes.

    H = diag(unsat) − Σ_i λ_i X_i − 0.50 Σ_c Σ_{i<j in c} X_i X_j
    λ_i = 0.75 * (1 + δ_i),   δ_i = δ * u_i,   u_i ~ Unif[-1, 1]

One shared pattern u (seed 20261105), prefix-stable in n, scaled by
δ ∈ {0.15, 0.30}. Train n∈{6,9} only. Freeze n12. Compare Walsh SNAP,
HVA, and Fourier at L=2 and L=4.

Does not read or write the n21 phase-2 ladder.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

# BLAS stays at 1 thread so several E0 workers do not oversubscribe.
# Torch threads are set later, only for the circuit.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from Nov_3rd import settings as S
from Nov_3rd.hva_safe import (
    INIT_BETA,
    INIT_GAMMA,
    MU_XX,
    angles_to_lists,
    cn_from_unsat,
    expectation_H,
    ground_energy,
    init_angles,
    instance_eps,
    method_parameter_list,
    prepare_state,
    self_check,
    summarize_eps,
)

RUN = S.OUT / "quantum_h_explore" / "disordered_lambda_shallow"
PACK = S.OUT / "beat_qaoa_12h" / "train_packs"
METHODS = ("fourier", "hva", "snap")
DELTAS = (0.15, 0.30)
TRAIN_NS = (6, 9)
TRAIN_LO, TRAIN_HI = 0, 200
HOLD_LO, HOLD_HI = 200, 250
FREEZE_N = 12
FREEZE_B = 128
DISORDER_SEED = 20261105
N_PATTERN = 48
# Uniform-λ n12 Medε from the SAFE-point HVA fairness run (not remeasured).
UNIFORM_N12 = {
    2: {"fourier": 0.2667, "hva": 0.2608, "snap": 0.0855, "R_hva": 0.3280, "R_fourier": 0.3208},
    4: {"fourier": 0.0771, "hva": 0.0736, "snap": 0.0394, "R_hva": 0.5356, "R_fourier": 0.5112},
}


def log(*a):
    print(time.strftime("%H:%M:%S"), "[dlam]", *a, flush=True)


def tag_delta(delta: float) -> str:
    return f"d{delta:.2f}".replace(".", "p")


def unit_pattern() -> np.ndarray:
    """Shared u_i ~ Unif[-1, 1]. δ_i = δ u_i is then Unif[-δ, δ]."""
    return np.random.default_rng(DISORDER_SEED).uniform(-1.0, 1.0, size=N_PATTERN)


def positive_lambda(n: int, delta: float, unit: np.ndarray) -> np.ndarray:
    """λ_i = 0.75 (1 + δ u_i) > 0 for |δ u_i| < 1."""
    lam = 0.75 * (1.0 + float(delta) * np.asarray(unit[:n], dtype=np.float64))
    if np.any(lam <= 0):
        raise RuntimeError(f"nonpositive λ at δ={delta}: {lam}")
    return lam


def load_unsat(n: int, sl: slice) -> tuple[np.ndarray, np.ndarray]:
    path = PACK / f"train_n{n}.npz"
    d = np.load(path)
    E = np.asarray(d["train_E"][sl], dtype=np.float64)
    seeds = np.asarray(d["train_seeds"][sl]).astype(np.int64)
    if E.ndim != 2 or E.shape[1] != (1 << n):
        raise RuntimeError(f"n{n} E shape {E.shape}")
    return E, seeds


def _ground_one(task):
    unsat, n, lam = task
    return ground_energy(np.asarray(unsat, dtype=np.float64), int(n), lambda_x=np.asarray(lam, dtype=np.float64))


def ensure_e0(tag: str, n: int, sl: slice, E: np.ndarray, seeds: np.ndarray, coeff: np.ndarray, jobs: int) -> np.ndarray:
    RUN.mkdir(parents=True, exist_ok=True)
    cache = RUN / f"e0_{tag}_n{n}_{sl.start}_{sl.stop}.npz"
    if cache.exists():
        z = np.load(cache)
        same = (
            np.array_equal(z["seeds"], seeds)
            and z["e0"].shape[0] == E.shape[0]
            and np.allclose(z["coeff"], coeff)
        )
        if same:
            log(f"E0 cache hit {tag} n{n} [{sl.start}:{sl.stop}]")
            return np.asarray(z["e0"], dtype=np.float64)
    log(f"E0 {tag} n{n} [{sl.start}:{sl.stop}] B={E.shape[0]} jobs={jobs}")
    t0 = time.time()
    tasks = [(E[i], n, coeff) for i in range(E.shape[0])]
    if jobs <= 1 or E.shape[0] == 1:
        e0 = np.array([_ground_one(t) for t in tasks], dtype=np.float64)
    else:
        ctx = __import__("multiprocessing").get_context("spawn")
        with ProcessPoolExecutor(max_workers=jobs, mp_context=ctx) as ex:
            e0 = np.array(list(ex.map(_ground_one, tasks, chunksize=2)), dtype=np.float64)
    if np.any(e0 > E.min(axis=1) + 1e-6):
        raise RuntimeError(f"E0 above diag min at {tag} n{n}")
    np.savez_compressed(cache, seeds=seeds, e0=e0, n=n, coeff=coeff, mu_xx=MU_XX, delta_tag=tag)
    log(f"E0 {tag} n{n} done in {time.time()-t0:.1f}s  medE0={float(np.median(e0)):.4f}")
    return e0


def _as_f64(angles: dict) -> dict:
    return {k: (v.detach().to(dtype=torch.float64) if torch.is_tensor(v) else v) for k, v in angles.items()}


def eval_method(method: str, angles: dict, bundle: dict, coeff_of: dict, *, chunk: int) -> dict:
    """bundle[n] = (unsat ndarray, e0 ndarray). coeff_of[n] = X coefficients."""
    ang = _as_f64(angles)
    out = {}
    for n, (E, e0) in bundle.items():
        Eu = torch.tensor(E, dtype=torch.float64)
        e0t = torch.tensor(e0, dtype=torch.float64)
        eps = instance_eps(
            method, ang, Eu, e0t, n, chunk=chunk, lambda_x=coeff_of[n], mu_xx=MU_XX
        ).detach().cpu().numpy()
        rec = summarize_eps(eps)
        rec.update({"n": n, "med_e0": float(np.median(e0)), "med_energy": float(np.median(eps + e0))})
        out[str(n)] = rec
    return out


def train_loss(method, angles, E, e0, idx, n: int, coeff: np.ndarray) -> torch.Tensor:
    Eu = E.index_select(0, idx)
    e0b = e0.index_select(0, idx)
    Cn = cn_from_unsat(Eu, n)
    psi = prepare_state(method, Cn, angles)
    h = expectation_H(psi, Eu, n, lambda_x=coeff, mu_xx=MU_XX)
    return ((h - e0b) / float(n)).mean()


def math_finite(x: float) -> bool:
    return x == x and abs(x) != float("inf")


def train_one(method, L, seed, bundles_t, coeff_of, hold_bundle, *, rounds: int, batch: int) -> dict:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    angles = init_angles(method, L, dtype=torch.float64)
    params = method_parameter_list(method, angles)
    E6, e06 = bundles_t[6]
    E9, e09 = bundles_t[9]
    n6, n9 = E6.shape[0], E9.shape[0]
    b6 = min(batch // 2, n6)
    b9 = min(batch - b6, n9)

    def hold_med() -> float:
        ev = eval_method(method, angles, hold_bundle, coeff_of, chunk=16)
        return float(ev["9"]["med_eps"])

    t0 = time.time()
    init_med = hold_med()
    best = {"med9": init_med, "round": -1, "angles": {k: v.detach().cpu().clone() for k, v in angles.items()}}
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
                train_loss(method, angles, E6, e06, i6, 6, coeff_of[6])
                + train_loss(method, angles, E9, e09, i9, 9, coeff_of[9])
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


def seed_path(tag: str, L: int, method: str, seed: int) -> Path:
    return RUN / f"{tag}_L{L}_{method}_seed{seed}.json"


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(path)


def ratio(num: float, den: float) -> float:
    if den == 0:
        return float("inf")
    return float(num) / float(den)


def verdict_one(r_hva: float, med_snap: float, med_hva: float) -> str:
    """PASS if SNAP still leads by R≤0.85. KILL if HVA is strictly ahead."""
    if med_hva < med_snap:
        return "KILL"
    if r_hva <= 0.85:
        return "PASS"
    return "needs_edit"


def _fmt(x, nd: int = 4) -> str:
    if x is None or (isinstance(x, float) and (x != x or abs(x) == float("inf"))):
        return "—"
    return f"{float(x):.{nd}f}"


def _shell_nl(phi) -> float:
    arr = np.asarray(phi, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 4 or arr.shape[0] == 0:
        return float("nan")
    return float(np.mean(np.std(np.diff(arr, axis=1), axis=1)))


def _mean_abs(vals) -> float:
    a = np.asarray(vals, dtype=np.float64).reshape(-1)
    if a.size == 0:
        return float("nan")
    return float(np.mean(np.abs(a)))


def _cell(block, method: str, n: int, seeds: list[int]) -> tuple[float, list[float]]:
    vals = []
    for rec in block:
        if rec["method"] != method:
            continue
        vals.append(float(rec["eval"][str(n)]["med_eps"]))
    if len(vals) != len(seeds):
        raise RuntimeError(f"{method} n{n} has {len(vals)} seeds, expected {len(seeds)}")
    vals_sorted = sorted(vals)
    return float(np.median(vals_sorted)), vals_sorted


def aggregate(deltas: list[float], depths: list[int], seeds: list[int], unit: np.ndarray) -> dict:
    summary = {
        "deltas": list(deltas),
        "depths": list(depths),
        "disorder_seed": DISORDER_SEED,
        "mu_xx": MU_XX,
        "lambda0": 0.75,
        "uniform_n12_reference": UNIFORM_N12,
        "by_delta": {},
    }
    lines = [
        "# Disordered λ at shallow L",
        "",
        "Hypothesis: the shallow SNAP win at uniform λ may be Hamming-shell symmetry "
        "matching a permutation-symmetric field. This run breaks that match. "
        "The gates are the same three ansatzes as the uniform SAFE-point run.",
        "",
        "## Hamiltonian",
        "",
        r"\[",
        r"H = \mathrm{diag}(\mathrm{unsat}) - \sum_i \lambda_i X_i - 0.50 \sum_c \sum_{i<j\in c} X_i X_j",
        r"\]",
        "",
        r"\[",
        r"\lambda_i = 0.75\,(1+\delta_i),\qquad \delta_i = \delta\, u_i,\qquad u_i\sim\mathrm{Unif}[-1,1]",
        r"\]",
        "",
        f"One pattern u (NumPy seed {DISORDER_SEED}), shared by both strengths and by every n "
        "(qubit i uses u_i, so the n=6 pattern is the prefix of n=12). "
        "δ_i is Unif[−δ, δ]. XX stays the uniform cavity Rank-2 term with coefficient −0.50. "
        "All λ_i stay positive, so |+⟩ is still the ground state of the off-diagonal part.",
        "",
        "Readout is the Z histogram (unsat) plus the X histogram. "
        r"A site-dependent field is \(\sum_i \lambda_i\langle X_i\rangle\) from that same X histogram. No Y.",
        "",
        "## Ansatz (unchanged)",
        "",
        "| method | angles / layer | unitary |",
        "|---|---|---|",
        r"| Fourier QAOA | 2: γ, β | \(U_X(\beta)\,U_C(\gamma)\), same β on every qubit |",
        r"| HVA QAOA | 3: γ, β^X, β^XX | \(U_{XX}(\beta^{XX})\,U_X(\beta^X)\,U_C(\gamma)\). β^XX=0 is Fourier. |",
        r"| SNAP Walsh + Hamming shell | 8: γ_{w=1..4}, φ_{k=0..3} | Walsh degree-1..4 cost, then one shared shell on every cavity |",
        "",
        "No per-qubit angles were added. Disorder is only in H. "
        r"Cost family is still \(C_n=\mathrm{unsat}\cdot n/m\). "
        f"Paper init β=β^X={INIT_BETA}, β^XX=0, γ=γ_w={INIT_GAMMA}, φ_k=−β k, "
        "so the three methods start on the same state.",
        "",
        "## Protocol",
        "",
        f"- Packs: `beat_qaoa_12h/train_packs`. Train indices [{TRAIN_LO}:{TRAIN_HI}) on n=6 and n=9. "
        f"Checkpoint on n9 holdout [{HOLD_LO}:{HOLD_HI}). Freeze n12, first {FREEZE_B} instances. n15 is not in this gate.",
        f"- L-BFGS budget matched to `Nov_3rd.settings`: {S.LBFGS_ROUNDS} rounds × max_iter {S.LBFGS_ITER}, "
        f"lr {S.LBFGS_LR}, history 16, strong Wolfe, grad clip 5, batch {S.LBFGS_BATCH} (half n6 / half n9). "
        "New optimizer each round. Loss = mean ε/n. No Adam.",
        "- Seeds {0,1,2} share the paper init; the seed only reshuffles batches.",
        "- Each seed restores the round with the best n9-holdout Med ε.",
        "- The gate is the **median over seeds** of frozen-n12 Med ε.",
        "- n21 phase-2 ladder was not read or written.",
        "",
        "## Uniform reference (not remeasured)",
        "",
        "δ=0 numbers are the SAFE-point HVA fairness run "
        "(`hva_fairness_safe`, same packs, budget, and gates). "
        "They are the shell-symmetric baseline this disorder is trying to break.",
        "",
        "| L | Medε Fourier | Medε HVA | Medε SNAP | R SNAP/HVA |",
        "|---|---:|---:|---:|---:|",
    ]
    for L, rec in UNIFORM_N12.items():
        lines.append(
            f"| {L} | {_fmt(rec['fourier'])} | {_fmt(rec['hva'])} | {_fmt(rec['snap'])} | {_fmt(rec['R_hva'])} |"
        )
    lines += ["", "## λ patterns", ""]

    primary = []
    stress = []
    for delta in deltas:
        tag = tag_delta(delta)
        lam12 = positive_lambda(FREEZE_N, delta, unit)
        lines.append(f"### δ={delta:.2f} ({tag})")
        lines.append("")
        lines.append(
            "n=12 λ = " + ", ".join(f"{v:.4f}" for v in lam12) + "."
        )
        lines.append("")
        lines.append("| cavity | λ | range |")
        lines.append("|---|---|---:|")
        cav_rows = []
        for c in range(FREEZE_N // 3):
            sl = lam12[3 * c : 3 * c + 3]
            cav_rows.append(
                {
                    "cavity": c,
                    "lambda": [float(x) for x in sl],
                    "range": float(sl.max() - sl.min()),
                }
            )
            lines.append(
                f"| {c} | {', '.join(f'{v:.4f}' for v in sl)} | {_fmt(float(sl.max()-sl.min()))} |"
            )
        lines.append("")
        summary["by_delta"][tag] = {
            "delta": delta,
            "lambda_n12": [float(x) for x in lam12],
            "cavities_n12": cav_rows,
            "depths": {},
        }
        for L in depths:
            block = []
            missing = []
            for method in METHODS:
                for seed in seeds:
                    p = seed_path(tag, L, method, seed)
                    if not p.exists():
                        missing.append(p.name)
                        continue
                    block.append(json.loads(p.read_text()))
            lines.append(f"## δ={delta:.2f}  L={L}")
            lines.append("")
            if missing:
                lines.append(f"Incomplete, missing {len(missing)}: {', '.join(missing)}.")
                lines.append("")
                summary["by_delta"][tag]["depths"][str(L)] = {"complete": False, "missing": missing}
                (primary if abs(delta - 0.15) < 1e-9 else stress).append((delta, L, "INCOMPLETE", None))
                continue
            cell = {}
            spread = {}
            for method in METHODS:
                med, vals = _cell(block, method, FREEZE_N, seeds)
                cell[method] = {FREEZE_N: med}
                spread[method] = vals
                for n in (6, 9):
                    med_n, _vals_n = _cell(block, method, n, seeds)
                    cell[method][n] = med_n
            lines.append("| n | role | Medε Fourier | Medε HVA | Medε SNAP | R SNAP/Fourier | R SNAP/HVA |")
            lines.append("|---|---|---:|---:|---:|---:|---:|")
            roles = {6: "holdout", 9: "holdout (select)", 12: "freeze"}
            rmap = {}
            for n in (6, 9, 12):
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
                f"**δ={delta:.2f} L={L} n12 gate: {v}.** "
                f"R_HVA={_fmt(rmap[12]['R_hva'])} "
                f"(SNAP {_fmt(cell['snap'][12])} / HVA {_fmt(cell['hva'][12])}), "
                f"R_Fourier={_fmt(rmap[12]['R_fourier'])}."
            )
            lines.append("")
            lines.append("| method | seed | best round | n9 hold Medε | n12 Medε | wall s | mean\\|β^XX\\| or shell-nl |")
            lines.append("|---|---:|---:|---:|---:|---:|---:|")
            diag_bits = []
            for method in METHODS:
                for rec in sorted((r for r in block if r["method"] == method), key=lambda r: int(r["seed"])):
                    ang = rec.get("angles") or {}
                    if method == "hva":
                        extra = _mean_abs(ang.get("beta_xx", []))
                    elif method == "snap":
                        extra = _shell_nl(ang.get("phi", []))
                    else:
                        extra = float("nan")
                    diag_bits.append((method, extra))
                    lines.append(
                        f"| {method} | {rec['seed']} | {rec['best_round']} | "
                        f"{_fmt(rec['eval']['9']['med_eps'])} | {_fmt(rec['eval']['12']['med_eps'])} | "
                        f"{rec['wall_s']:.0f} | {_fmt(extra)} |"
                    )
            lines.append("")
            lines.append(
                "n12 seed range (min–max Medε): "
                + ", ".join(f"{m} {_fmt(spread[m][0])}–{_fmt(spread[m][-1])}" for m in METHODS)
                + "."
            )
            lines.append("")
            ref = UNIFORM_N12.get(L)
            if ref is not None:
                lines.append(
                    f"Versus uniform δ=0 at this L, n12 R_HVA was {_fmt(ref['R_hva'])} "
                    f"(SNAP {_fmt(ref['snap'])} / HVA {_fmt(ref['hva'])})."
                )
                lines.append("")
            summary["by_delta"][tag]["depths"][str(L)] = {
                "complete": True,
                "verdict_n12": v,
                "med_eps": {m: {str(n): cell[m][n] for n in (6, 9, 12)} for m in METHODS},
                "R_n12": rmap[12],
                "n12_seed_values": {m: spread[m] for m in METHODS},
            }
            bucket = primary if abs(delta - 0.15) < 1e-9 else stress
            bucket.append((delta, L, v, rmap[12]["R_hva"]))

    def _fold(items):
        vs = [v for *_a, v, _r in items]
        if not items:
            return "ABSENT"
        if any(v == "INCOMPLETE" for v in vs):
            return "INCOMPLETE"
        if any(v == "KILL" for v in vs):
            return "KILL"
        if any(v == "needs_edit" for v in vs):
            return "needs_edit"
        return "PASS"

    overall = _fold(primary)
    stress_overall = _fold(stress)
    summary["overall_delta_0p15"] = overall
    summary["stress_delta_0p30"] = stress_overall

    def _bits(items):
        if not items:
            return "no depths"
        return " ".join(
            f"L={L} → {v} (R_HVA={_fmt(r) if r is not None else '—'})."
            for _d, L, v, r in items
        )

    lines.insert(2, f"**Overall: {overall}.** Gate is δ=0.15, frozen n12. {_bits(primary)}")
    lines.insert(3, "")
    lines.insert(4, f"**Stress δ=0.30 (reported, does not relabel the δ=0.15 gate): {stress_overall}.** {_bits(stress)}")
    lines.insert(5, "")
    lines += [
        "## How to read the gate",
        "",
        "- **PASS** — at δ=0.15, R_HVA = Medε_SNAP / Medε_HVA ≤ 0.85 on frozen n12. "
        "The shallow SNAP lead survives a 15% site disorder.",
        "- **needs_edit** — at δ=0.15 the lead shrinks past 0.85 and HVA is not ahead. The win vanished.",
        "- **KILL** — at δ=0.15, Medε_HVA < Medε_SNAP on a shallow depth. "
        "HVA dominates once the field is disordered.",
        "- δ=0.30 is the stronger break of the same pattern. It is reported beside the gate and does not flip a δ=0.15 PASS into a KILL.",
        "- Overall uses every shallow depth that was run at δ=0.15. Any KILL there wins; else any needs_edit; else PASS.",
        "",
        "Per-seed JSON: `{d0p15,d0p30}_L{L}_{method}_seed{s}.json`. "
        "Angles are the restored n9-holdout checkpoint. "
        "`shell-nl` is the mean over layers of the std of consecutive φ differences "
        "(0 means the shell is exactly linear, φ_k = a + b k, the pure X mixer). "
        "`mean|β^XX|` is the mean absolute XX angle.",
        "",
    ]
    text = "\n".join(lines)
    if not text.endswith("\n"):
        text += "\n"
    (RUN / "RESULTS.md").write_text(text)
    (RUN / "summary.json").write_text(json.dumps(summary, indent=2))
    (RUN / "lambda_u.json").write_text(
        json.dumps({"seed": DISORDER_SEED, "u": unit.tolist(), "note": "delta_i = delta * u_i"}, indent=2)
    )
    log(f"wrote RESULTS.md overall={overall} stress={stress_overall}")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deltas", type=str, default="0.15,0.30")
    ap.add_argument("--depths", type=str, default="2,4")
    ap.add_argument("--seeds", type=str, default="0,1,2")
    ap.add_argument("--rounds", type=int, default=S.LBFGS_ROUNDS)
    ap.add_argument("--batch", type=int, default=S.LBFGS_BATCH)
    ap.add_argument("--jobs", type=int, default=4, help="E0 worker processes")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--aggregate-only", action="store_true")
    ap.add_argument("--skip-self-check", action="store_true")
    args = ap.parse_args()

    torch.set_num_threads(4)
    deltas = [float(x) for x in args.deltas.split(",") if x.strip()]
    depths = [int(x) for x in args.depths.split(",") if x.strip()]
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    unit = unit_pattern()
    RUN.mkdir(parents=True, exist_ok=True)
    log(
        "pattern u[:12]",
        " ".join(f"{v:.3f}" for v in unit[:12]),
        "threads",
        torch.get_num_threads(),
    )

    if args.aggregate_only:
        aggregate(deltas, depths, seeds, unit)
        return

    if not args.skip_self_check:
        log("self-check")
        E6_all, _ = load_unsat(6, slice(0, 1))
        self_check(E6_all)
        log("self-check ok")

    for delta in deltas:
        tag = tag_delta(delta)
        coeff = {n: -positive_lambda(n, delta, unit) for n in (6, 9, 12)}
        log(f"=== {tag} λ6={[round(float(-c), 4) for c in coeff[6]]} ===")
        data = {}
        for n, sl, role in (
            (6, slice(TRAIN_LO, TRAIN_HI), "train"),
            (9, slice(TRAIN_LO, TRAIN_HI), "train"),
            (6, slice(HOLD_LO, HOLD_HI), "hold"),
            (9, slice(HOLD_LO, HOLD_HI), "hold"),
            (12, slice(0, FREEZE_B), "freeze"),
        ):
            E, sd = load_unsat(n, sl)
            e0 = ensure_e0(tag, n, sl, E, sd, coeff[n], jobs=args.jobs)
            data[(n, role)] = (E, e0)
            log(f"{tag} n{n} {role} B={len(E)} medE0={float(np.median(e0)):.4f}")

        train_tensors = {}
        for n in TRAIN_NS:
            E, e0 = data[(n, "train")]
            train_tensors[n] = (torch.tensor(E, dtype=torch.float64), torch.tensor(e0, dtype=torch.float64))
        hold_bundle = {9: data[(9, "hold")]}
        eval_bundle = {6: data[(6, "hold")], 9: data[(9, "hold")], 12: data[(12, "freeze")]}

        for L in depths:
            init_ev = {}
            for method in METHODS:
                init_ev[method] = eval_method(
                    method,
                    init_angles(method, L, dtype=torch.float64),
                    eval_bundle,
                    coeff,
                    chunk=8,
                )
            Cn12 = cn_from_unsat(torch.tensor(data[(12, "freeze")][0][:1], dtype=torch.float64), 12)
            sf = prepare_state("fourier", Cn12, init_angles("fourier", L, dtype=torch.float64))
            for method in ("hva", "snap"):
                sm = prepare_state(method, Cn12, init_angles(method, L, dtype=torch.float64))
                fid = float(torch.sum(sf.conj() * sm, dim=-1).abs()[0])
                log(f"{tag} L{L} init fidelity fourier vs {method} = {fid:.8f}")
                if fid < 1 - 1e-6:
                    raise RuntimeError(f"init states differ: {method} fid={fid}")
            log(
                f"{tag} L{L} init Medε "
                + " ".join(f"n{n}={init_ev['fourier'][n]['med_eps']:.4f}" for n in ("6", "9", "12"))
            )
            for method in METHODS:
                for seed in seeds:
                    path = seed_path(tag, L, method, seed)
                    if path.exists() and not args.force:
                        log(f"skip existing {path.name}")
                        continue
                    log(f"=== train {tag} L{L} {method} seed{seed} ===")
                    try:
                        trained = train_one(
                            method, L, seed, train_tensors, coeff, hold_bundle,
                            rounds=args.rounds, batch=args.batch,
                        )
                    except Exception:
                        log(traceback.format_exc())
                        raise
                    ev = eval_method(method, trained["angles"], eval_bundle, coeff, chunk=4)
                    payload = {
                        "done": True,
                        "delta": delta,
                        "tag": tag,
                        "method": method,
                        "L": L,
                        "seed": seed,
                        "n_params": trained["n_params"],
                        "best_round": trained["best_round"],
                        "best_hold_med9": trained["best_hold_med9"],
                        "wall_s": trained["wall_s"],
                        "rounds": args.rounds,
                        "max_iter": S.LBFGS_ITER,
                        "lr": S.LBFGS_LR,
                        "batch": args.batch,
                        "mu_xx": MU_XX,
                        "lambda_positive": {str(n): (-coeff[n]).tolist() for n in (6, 9, 12)},
                        "coeff_X": {str(n): coeff[n].tolist() for n in (6, 9, 12)},
                        "angles": angles_to_lists(method, trained["angles"]),
                        "init_med_eps": {n: init_ev["fourier"][n]["med_eps"] for n in ("6", "9", "12")},
                        "eval": ev,
                        "hist": [
                            {k: h[k] for k in h if k in ("round", "loss", "hold_med9", "skip")}
                            for h in trained["hist"]
                        ],
                    }
                    write_json(path, payload)
                    log(
                        f"saved {path.name} best_r={trained['best_round']} "
                        f"n12 Medε={ev['12']['med_eps']:.4f} wall={trained['wall_s']:.0f}s"
                    )
                    aggregate(deltas, depths, seeds, unit)
        aggregate(deltas, depths, seeds, unit)

    summary = aggregate(deltas, depths, seeds, unit)
    log("OVERALL", summary.get("overall_delta_0p15"), "STRESS", summary.get("stress_delta_0p30"))


if __name__ == "__main__":
    main()
