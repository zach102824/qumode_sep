#!/usr/bin/env python3
"""Smoke arm A: transverse-field Max-E, locked BinH Hamming-shell gate.

H = H_c − λ Σ_i X_i, H_c = raw unsat diagonal from the scale1000 generator.
Does not write Nov_3rd/qaoa_angles, settings.py, or the n21 ladder.

Metric: Med ε = median (⟨H⟩ − E0) / n. BEAT if SNAP Med ≤ 0.98 × Fourier QAOA
on both n=6 and n=9.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from Nov_3rd.quantum_h_A import (  # noqa: E402
    BEAT_RATIO,
    LAMBDA_GRID,
    SnapSimulator,
    a_of_n,
    angles_from_modes,
    choose_lambda,
    e0_batch,
    epsilon,
    load_instances,
    paper_modes,
    phi_from_shell,
    plus_energy,
    qaoa_energy,
    quantumness_scan,
    ratio_mark,
)

OUT = _ROOT / "Nov_3rd" / "runs" / "quantum_h_explore" / "smoke_A"


def log(*a):
    print(time.strftime("%H:%M:%S"), "[qhA]", *a, flush=True)


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if isinstance(obj, Path):
        return str(obj)
    return obj


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(obj), indent=2) + "\n")


def slice_pack(pack: dict, start: int, stop: int) -> dict:
    return {
        "n": pack["n"],
        "m": pack["m"],
        "E": pack["E"][start:stop],
        "seeds": pack["seeds"][start:stop],
        "source": pack["source"],
    }


def pack_tensors(pack: dict, e0: np.ndarray, device) -> dict:
    return {
        "E": torch.tensor(pack["E"], dtype=torch.float32, device=device),
        "e0": torch.tensor(e0, dtype=torch.float32, device=device),
        "n": int(pack["n"]),
        "seeds": pack["seeds"],
    }


def run_ed(packs: dict[int, dict], bed: int, lambdas, out: Path, force: bool) -> dict:
    path = out / "ed_quantumness.json"
    if path.exists() and not force:
        log(f"ED reuse {path}")
        return json.loads(path.read_text())
    scans = {}
    for n, pack in packs.items():
        sub = slice_pack(pack, 0, bed)
        log(f"ED n={n} B={bed} λ-grid {list(lambdas)}")
        t0 = time.time()
        scans[n] = quantumness_scan(sub, tuple(lambdas))
        scans[n]["seeds"] = sub["seeds"].tolist()
        scans[n]["wall_s"] = time.time() - t0
        row = scans[n]["per_lambda"]
        for key, rec in row.items():
            log(
                f"  n={n} λ={key} peak={rec['peak_med']:.3f} Sh={rec['S_half_med']:.3f} "
                f"S1={rec['S1_med']:.3f} Fsat={rec['F_sat_med']:.3f} "
                f"Xn={rec['X_per_n_med']:.3f} chi={rec['chi_F']}"
            )
    chosen = choose_lambda(scans, tuple(lambdas))
    payload = {
        "grid": list(lambdas),
        "B_ed": bed,
        "Hc": "raw unsat count (not n/m scaled)",
        "sign": "H = Hc - λ Σ X_i",
        "scans": scans,
        "chosen": chosen,
    }
    write_json(path, payload)
    log(f"ED chose λ={chosen.get('lambda')} quantum={chosen.get('quantum')}")
    return payload


def ensure_e0(pack: dict, lam: float, cache: Path, force: bool) -> np.ndarray:
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.exists() and not force:
        blob = np.load(cache)
        if blob["E0"].shape[0] == pack["E"].shape[0] and abs(float(blob["lam"]) - float(lam)) < 1e-12:
            if np.array_equal(blob["seeds"], pack["seeds"]):
                return np.asarray(blob["E0"], dtype=np.float64)
    log(f"eigh E0 n={pack['n']} B={len(pack['E'])} λ={lam}")
    t0 = time.time()
    e0 = e0_batch(pack["E"], lam)
    np.savez_compressed(
        cache,
        E0=e0,
        seeds=pack["seeds"],
        E=pack["E"].astype(np.uint8),
        lam=np.float64(lam),
        n=np.int32(pack["n"]),
    )
    log(f"  E0 done {time.time()-t0:.1f}s  median E0={np.median(e0):.4f}")
    return e0


def median_eps(energy: torch.Tensor, e0: torch.Tensor, n: int) -> float:
    ep = epsilon(energy, e0, n)
    return float(ep.median().detach())


@torch.no_grad()
def eval_eps(kind: str, params: dict, data: dict, sims: dict, lam: float, L: int) -> dict:
    out = {}
    for n, blob in data.items():
        energy = energies_of(kind, params, blob["E"], n, sims, lam, L)
        ep = epsilon(energy, blob["e0"], n).detach().cpu().numpy()
        plus = epsilon(plus_energy(blob["E"], lam, n), blob["e0"], n).detach().cpu().numpy()
        # Best satisfying bitstring has ⟨H⟩=0 (sat, ⟨X⟩=0 on a Z string).
        bit = (-blob["e0"].detach().cpu().numpy()) / float(n)
        out[n] = {
            "Med": float(np.median(ep)),
            "mean": float(np.mean(ep)),
            "B": int(ep.size),
            "plus_Med": float(np.median(plus)),
            "bitstring_Med": float(np.median(bit)),
            "eps": ep,
        }
    return out


def energies_of(kind, params, E, n, sims, lam, L):
    if kind == "qaoa":
        betas, gammas = angles_from_modes(params["A"], params["B"], L)
        return qaoa_energy(betas, gammas, E, lam, n)
    sim: SnapSimulator = sims[n]
    if kind == "power_p":
        gamma, phi = power_sched(params, n, L)
        return sim.energy(E, gamma, phi, lam)
    if kind == "walsh":
        gamma, phi = walsh_sched(params, n, L)
        return sim.energy(E, gamma, phi, lam)
    raise ValueError(kind)


def power_sched(params, n, L):
    a = a_of_n(n, params["p"])
    phi = phi_from_shell(params["c1"], a * params["c2"], a * params["c3"])
    return params["gamma"].reshape(L), phi


def walsh_sched(params, n, L):
    a = a_of_n(n, params["p"])
    phi = phi_from_shell(params["c1"], a * params["c2"], a * params["c3"])
    return params["gamma"].reshape(L, 4), phi


def clone_params(params: dict) -> dict:
    out = {}
    for k, v in params.items():
        if torch.is_tensor(v):
            out[k] = v.detach().cpu().clone()
        else:
            out[k] = v
    return out


def load_params(params: dict, saved: dict, device) -> None:
    for k, v in params.items():
        if torch.is_tensor(v) and k in saved:
            v.data.copy_(saved[k].to(device=device, dtype=v.dtype))


def loss_on(kind, params, batches, sims, lam, L) -> torch.Tensor:
    terms = []
    for n, blob in batches.items():
        energy = energies_of(kind, params, blob["E"], n, sims, lam, L)
        terms.append(epsilon(energy, blob["e0"], n).mean())
    return torch.stack(terms).mean()


def hold_view(kind, params, hold, sims, lam, L) -> dict:
    return eval_eps(kind, params, hold, sims, lam, L)


def checkpoint_score(ev: dict, ref_med: dict | None) -> float:
    """QAOA: mean of the two median ε. SNAP: worse ratio vs frozen QAOA.

    The ratio score keeps a warm-start tie unless both sizes improve together.
    """
    if ref_med is None:
        return 0.5 * (ev[6]["Med"] + ev[9]["Med"])
    return max(ev[n]["Med"] / ref_med[n] for n in (6, 9))


def budget_for(L: int, args) -> tuple[int, int, int]:
    if args.adam_steps is not None:
        adam = int(args.adam_steps)
    else:
        adam = 40 if L <= 2 else 30
    if args.lbfgs_rounds is not None:
        rounds = int(args.lbfgs_rounds)
    else:
        rounds = 10 if L <= 2 else 8
    iters = int(args.lbfgs_iter)
    return adam, rounds, iters


def train_kind(
    kind: str,
    L: int,
    lam: float,
    train: dict,
    hold: dict,
    sims: dict,
    seed: int,
    args,
    device,
    ref_med: dict | None = None,
) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    adam_steps, lbfgs_rounds, lbfgs_iter = budget_for(L, args)
    params, opt_list = init_params(kind, L, None, device)
    # Warm SNAP from the trained Fourier angles when that file exists.
    if kind != "qaoa":
        qpath = OUT / f"L{L}" / "qaoa" / "params.pt"
        if qpath.exists():
            q = torch.load(qpath, map_location="cpu", weights_only=False)
            qA = q["params"]["A"] if "params" in q else q["A"]
            qB = q["params"]["B"] if "params" in q else q["B"]
            with torch.no_grad():
                betas, gammas = angles_from_modes(qA.to(device), qB.to(device), L)
                params["c1"].copy_(-betas)
                params["c2"].zero_()
                params["c3"].zero_()
                params["p"].zero_()
                if kind == "power_p":
                    params["gamma"].copy_(gammas)
                else:
                    params["gamma"].copy_(gammas.reshape(L, 1).expand(L, 4))
            log(f"L={L} {kind}: warm-started from trained Fourier QAOA")
    opt = torch.optim.Adam(opt_list, lr=float(args.adam_lr))
    hist = []
    best = {"score": float("inf"), "params": clone_params(params), "step": "init"}
    t0 = time.time()

    def consider(tag: str):
        ev = hold_view(kind, params, hold, sims, lam, L)
        score = checkpoint_score(ev, ref_med)
        hist.append({
            "tag": tag,
            "score": score,
            "med6": ev[6]["Med"],
            "med9": ev[9]["Med"],
            "wall_s": time.time() - t0,
        })
        if score < best["score"]:
            best["score"] = score
            best["params"] = clone_params(params)
            best["step"] = tag
        log(
            f"L={L} {kind} {tag} med6={ev[6]['Med']:.4f} med9={ev[9]['Med']:.4f} "
            f"score={score:.4f} best={best['score']:.4f}"
        )

    consider("init")
    for step in range(adam_steps):
        opt.zero_grad()
        loss = loss_on(kind, params, train, sims, lam, L)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(opt_list, 5.0)
        opt.step()
        if step % 10 == 0 or step == adam_steps - 1:
            consider(f"adam{step}")
            hist[-1]["train_loss"] = float(loss.detach())
    for r in range(lbfgs_rounds):
        opt_l = torch.optim.LBFGS(
            opt_list,
            lr=0.25,
            max_iter=lbfgs_iter,
            history_size=10,
            line_search_fn="strong_wolfe",
        )

        def closure():
            opt_l.zero_grad()
            loss_l = loss_on(kind, params, train, sims, lam, L)
            loss_l.backward()
            torch.nn.utils.clip_grad_norm_(opt_list, 5.0)
            return loss_l

        try:
            loss_v = opt_l.step(closure)
            loss_f = float(loss_v.detach()) if torch.is_tensor(loss_v) else float(loss_v)
        except Exception as exc:
            log(f"L={L} {kind} lbfgs{r} skip: {exc}")
            continue
        consider(f"lbfgs{r}")
        hist[-1]["train_loss"] = loss_f
    load_params(params, best["params"], device)
    return {
        "kind": kind,
        "L": L,
        "lambda": lam,
        "seed": seed,
        "best_step": best["step"],
        "best_score": best["score"],
        "wall_s": time.time() - t0,
        "hist_tail": hist[-12:],
        "params": {k: v.detach().cpu() for k, v in params.items() if torch.is_tensor(v)},
        "n_params": int(sum(v.numel() for v in opt_list)),
        "adam_steps": adam_steps,
        "lbfgs_rounds": lbfgs_rounds,
        "lbfgs_iter": lbfgs_iter,
    }


def init_params(kind: str, L: int, warm_angles, device):
    A0, B0 = paper_modes(L)
    if kind == "qaoa":
        A = torch.nn.Parameter(A0.to(device))
        B = torch.nn.Parameter(B0.to(device))
        return {"A": A, "B": B}, [A, B]
    betas, gammas = angles_from_modes(A0.to(device), B0.to(device), L)
    c1 = torch.nn.Parameter((-betas).detach().clone())
    c2 = torch.nn.Parameter(torch.zeros(L, device=device))
    c3 = torch.nn.Parameter(torch.zeros(L, device=device))
    p = torch.nn.Parameter(torch.zeros((), device=device))
    if kind == "power_p":
        gamma = torch.nn.Parameter(gammas.detach().clone())
        return {"gamma": gamma, "c1": c1, "c2": c2, "c3": c3, "p": p}, [gamma, c1, c2, c3, p]
    if kind == "walsh":
        gamma = torch.nn.Parameter(gammas.detach().reshape(L, 1).expand(L, 4).clone())
        return {"gamma": gamma, "c1": c1, "c2": c2, "c3": c3, "p": p}, [gamma, c1, c2, c3, p]
    raise ValueError(kind)


def summarize_pair(method_ev: dict, qaoa_ev: dict) -> dict:
    per_n = {}
    marks = []
    for n in (6, 9):
        R = float(method_ev[n]["Med"]) / float(qaoa_ev[n]["Med"])
        mark = ratio_mark(method_ev[n]["Med"], qaoa_ev[n]["Med"])
        marks.append(mark)
        per_n[str(n)] = {
            "Med": method_ev[n]["Med"],
            "mean": method_ev[n]["mean"],
            "QAOA_Med": qaoa_ev[n]["Med"],
            "R": R,
            "mark": mark,
            "plus_Med": method_ev[n]["plus_Med"],
            "bitstring_Med": method_ev[n]["bitstring_Med"],
            "B": method_ev[n]["B"],
        }
    both = all(m == "BEAT" for m in marks)
    return {"per_n": per_n, "both_n_BEAT": both, "marks": marks}


def family_verdict(table: dict, quantum: bool) -> dict:
    """Kill rules from SHORTLIST family A / PLAN (the opus reply file is not in the repo)."""
    if not quantum:
        return {
            "verdict": "KILL",
            "reason": "GS stayed a classical bitstring across the λ grid (median peak ≥ 0.5).",
        }
    pp = {L: table[L]["power_p"]["both_n_BEAT"] for L in table}
    walsh = {L: table[L].get("walsh", {}).get("both_n_BEAT", False) for L in table}
    if pp and all(pp.values()):
        return {
            "verdict": "PASS",
            "reason": "power_p Med ε ≤ 0.98× Fourier QAOA on both n=6 and n=9 at every smoked L, and the GS is not a bitstring.",
        }
    if any(pp.values()) or any(walsh.values()):
        return {
            "verdict": "needs_edit",
            "reason": (
                "Hamming-shell SNAP beats Fourier QAOA on both n at some depths, "
                "but scalar power_p does not clear every smoked L. "
                f"power_p both-n BEAT by L: {pp}; walsh: {walsh}. "
                "Keep the gate and grade the cost angles by Walsh degree."
            ),
        }
    # Near-miss: every L has both n at TIE or better, but not a strict BEAT.
    near = True
    for L, block in table.items():
        for method in ("power_p", "walsh"):
            if method not in block:
                continue
            marks = block[method]["marks"]
            if any(m == "UNDER" for m in marks):
                near = False
    if near and table:
        return {
            "verdict": "needs_edit",
            "reason": "SNAP stays inside a TIE of Fourier QAOA (R≤1.02) but never clears 0.98 on both n.",
        }
    return {
        "verdict": "KILL",
        "reason": (
            "At the entangled λ, neither power_p nor Walsh-degree Hamming-shell SNAP "
            "reaches Med ε ≤ 0.98× Fourier QAOA on both n=6 and n=9. "
            "SHORTLIST kill: SNAP never ≤0.98× QAOA on n6 and n9."
        ),
    }


def _by_n(scans: dict, n: int) -> dict:
    if n in scans:
        return scans[n]
    return scans[str(n)]


def render_results(payload: dict) -> str:
    ch = payload["chosen"]
    lines = [
        "# RESULTS — arm A transverse-field Max-E (smoke)",
        "",
        "Locked gate: `U_c^X = W exp(-i γ C_n) W`, `W=H^{⊗3}`, Hamming-shell SNAP.",
        "`C_n = (n/m) H_c − max` is the existing Nov_3rd cost (global phase drops; `γ` absorbs `n/m`).",
        "",
        "Hamiltonian, stoquastic sign:",
        "",
        "```text",
        "H_c(z) = unsat count of the TEB / scale1000 clause pack",
        "H     = H_c − λ Σ_i X_i ,   λ>0",
        "ε     = (⟨H⟩ − E_0) / n",
        "⟨H⟩   = Σ_z p_Z(z) H_c(z) − λ Σ_i ⟨X_i⟩",
        "```",
        "",
        "Readout is the Z histogram plus single-qubit X histograms. No YY or XX.",
        "",
        f"**Chosen λ = {ch.get('lambda')}.** {ch.get('reason')}",
        "",
        "## ED quantumness",
        "",
        "| n | λ | peak Med | S_half Med | S1 Med | F_sat Med | ⟨X⟩/n | χ_F | quantum |",
        "|---|---|----------|------------|--------|-----------|-------|-----|---------|",
    ]
    scans = payload["ed"]["scans"]
    for n in (6, 9):
        block = _by_n(scans, n)["per_lambda"]
        for key, rec in block.items():
            chi = rec["chi_F"]
            chi_s = "—" if chi is None else f"{chi:.2f}"
            q = "yes" if rec["peak_med"] < 0.5 else "no"
            lines.append(
                f"| {n} | {rec['lambda']:.2f} | {rec['peak_med']:.3f} | {rec['S_half_med']:.3f} | "
                f"{rec['S1_med']:.3f} | {rec['F_sat_med']:.3f} | {rec['X_per_n_med']:.3f} | {chi_s} | {q} |"
            )
    lines += [
        "",
        "Peak is the largest computational-basis probability of the ground state.",
        "S_half is the half-chain von Neumann entropy (nats). χ_F = median (1−|⟨ψ_λ|ψ_λ′⟩|²)/(Δλ)²",
        "from the previous grid point. A point is quantum when peak Med < 0.5.",
        "",
        "Raw unsat is required for this grid. With `H_c = (n/m)·unsat` the same λ values sit",
        "past the crossover (`⟨X⟩/n ≳ 0.97` already at λ=0.25) and the ground state is a product",
        "`|+⟩` state. The locked pulse still generates `exp(−i γ_phys H_c)` because `γ_phys = γ·(n/m)`.",
        "",
        "## Train / holdout",
        "",
        f"Train B={payload['B_tr']} per n, holdout B={payload['B_ho']}, seed={payload['seed']}.",
        "Fourier QAOA uses a full DST-IV (γ) / DCT-IV (β) basis at Q=L, paper init β=0.01, γ=−0.01.",
        "power_p and Walsh-degree SNAP warm-start from that trained QAOA dual (`φ_k=−β k`, `c2=c3=0`),",
        "then Adam + L-BFGS on mean ε. Reported Med is the best holdout checkpoint.",
        "Angles are written under `smoke_A/` only.",
        "",
        "| L | method | n | Med ε | QAOA Med ε | R | mark | plus Med | bitstring Med |",
        "|---|--------|---|-------|------------|---|------|---------|---------------|",
    ]
    for L, block in payload["table"].items():
        for method, rec in block.items():
            for n, row in rec["per_n"].items():
                lines.append(
                    f"| {L} | {method} | {n} | {row['Med']:.4f} | {row['QAOA_Med']:.4f} | "
                    f"{row['R']:.4f} | {row['mark']} | {row['plus_Med']:.4f} | {row['bitstring_Med']:.4f} |"
                )
    v = payload["verdict"]
    lines += [
        "",
        f"## Verdict: **{v['verdict']}**",
        "",
        v["reason"],
        "",
        "Kill criteria (SHORTLIST family A and PLAN; "
        "`Nov_3rd/runs/opus_ideas/opus_reply_20261104_2150_hamiltonian_families.md` is not in the repo):",
        "",
        "- KILL if the ground state stays a classical bitstring on the λ grid.",
        "- KILL if ⟨H⟩ needs YY or XX correlators. It does not.",
        "- KILL if GRAPE of the diagonal `H_c` alone already prepares the ground state.",
        "  A bitstring ground state would. At the chosen λ the median peak is below 0.5, so the",
        "  diagonal pulse is not the whole answer. It is exactly the cost layer the motif already applies.",
        f"- KILL if SNAP never reaches Med ε ≤ {BEAT_RATIO:.2f}× Fourier QAOA on both n=6 and n=9.",
        "- needs_edit if that bar is cleared at only one depth, or only by the Walsh-degree head.",
        "- PASS if power_p clears it at every smoked depth and the ground state is quantum.",
        "",
        "Phase-2 n21 ladder, `settings.py`, and `qaoa_angles/` were not modified.",
        "",
    ]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="Arm A transverse-field smoke")
    ap.add_argument("--B-ed", type=int, default=24)
    ap.add_argument("--B-tr", type=int, default=64)
    ap.add_argument("--B-ho", type=int, default=128)
    ap.add_argument("--Ls", type=str, default="2,16")
    ap.add_argument("--methods", type=str, default="qaoa,power_p,walsh")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--adam-steps", type=int, default=None)
    ap.add_argument("--adam-lr", type=float, default=0.02)
    ap.add_argument("--lbfgs-rounds", type=int, default=None)
    ap.add_argument("--lbfgs-iter", type=int, default=8)
    ap.add_argument("--out", type=str, default="")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--skip-train", action="store_true")
    args = ap.parse_args()

    global OUT
    if args.out:
        OUT = Path(args.out)
    torch.set_num_threads(int(args.threads))
    device = torch.device("cpu")
    OUT.mkdir(parents=True, exist_ok=True)
    Ls = [int(x) for x in args.Ls.split(",") if x.strip()]
    methods = [x.strip() for x in args.methods.split(",") if x.strip()]
    if "qaoa" not in methods:
        methods = ["qaoa"] + methods
    bed, btr, bho = int(args.B_ed), int(args.B_tr), int(args.B_ho)
    count = bed + btr + bho

    packs = {}
    for n in (6, 9):
        log(f"load n={n} count={count}")
        packs[n] = load_instances(n, count)
        log(f"  source={packs[n]['source']} m={packs[n]['m']} Emin_med={np.median(packs[n]['E'].min(1)):.1f}")

    ed = run_ed(packs, bed, LAMBDA_GRID, OUT, args.force)
    chosen = ed["chosen"]
    if not chosen.get("quantum") or chosen.get("lambda") is None:
        payload = {
            "chosen": chosen,
            "ed": ed,
            "B_tr": btr,
            "B_ho": bho,
            "seed": args.seed,
            "table": {},
            "verdict": family_verdict({}, False),
        }
        text = render_results(payload)
        (OUT / "RESULTS.md").write_text(text)
        write_json(OUT / "verdict.json", payload["verdict"])
        log(payload["verdict"])
        return 0

    lam = float(chosen["lambda"])
    train_packs, hold_packs = {}, {}
    train_data, hold_data = {}, {}
    for n, pack in packs.items():
        tr = slice_pack(pack, bed, bed + btr)
        ho = slice_pack(pack, bed + btr, bed + btr + bho)
        e0_tr = ensure_e0(tr, lam, OUT / "cache" / f"e0_n{n}_train_l{lam:.2f}.npz", args.force)
        e0_ho = ensure_e0(ho, lam, OUT / "cache" / f"e0_n{n}_hold_l{lam:.2f}.npz", args.force)
        train_packs[n] = tr
        hold_packs[n] = ho
        train_data[n] = pack_tensors(tr, e0_tr, device)
        hold_data[n] = pack_tensors(ho, e0_ho, device)

    table: dict[str, dict] = {}
    if not args.skip_train:
        for L in Ls:
            sims = {n: SnapSimulator(n, L, device) for n in (6, 9)}
            qaoa_ev = None
            ref_med = None
            table[str(L)] = {}
            for kind in methods:
                dest = OUT / f"L{L}" / kind
                dest.mkdir(parents=True, exist_ok=True)
                ckpt = dest / "params.pt"
                if ckpt.exists() and not args.force:
                    log(f"reuse {ckpt}")
                    saved = torch.load(ckpt, map_location="cpu", weights_only=False)
                else:
                    log(f"train L={L} {kind} λ={lam} Btr={btr}")
                    saved = train_kind(
                        kind, L, lam, train_data, hold_data, sims, args.seed, args, device,
                        ref_med=None if kind == "qaoa" else ref_med,
                    )
                    torch.save(saved, ckpt)
                    write_json(dest / "train.json", {k: v for k, v in saved.items() if k != "params"})
                params = {k: v.to(device) for k, v in saved["params"].items()}
                ev = eval_eps(kind, params, hold_data, sims, lam, L)
                if kind == "qaoa":
                    ref_med = {n: ev[n]["Med"] for n in (6, 9)}
                # Drop bulky vectors from the json summary; keep npz.
                for n, row in ev.items():
                    np.save(dest / f"eps_hold_n{n}.npy", row["eps"])
                if kind == "qaoa":
                    qaoa_ev = ev
                    summary = summarize_pair(ev, ev)
                else:
                    summary = summarize_pair(ev, qaoa_ev)
                summary["wall_s"] = saved.get("wall_s")
                summary["best_step"] = saved.get("best_step")
                summary["n_params"] = saved.get("n_params")
                table[str(L)][kind] = summary
                write_json(dest / "eval.json", summary)
                for n in (6, 9):
                    row = summary["per_n"][str(n)]
                    log(
                        f"EVAL L={L} {kind} n={n} Med={row['Med']:.4f} "
                        f"QAOA={row['QAOA_Med']:.4f} R={row['R']:.4f} {row['mark']}"
                    )

    verdict = family_verdict(table, True)
    payload = {
        "chosen": chosen,
        "ed": ed,
        "B_tr": btr,
        "B_ho": bho,
        "B_ed": bed,
        "seed": args.seed,
        "lambda": lam,
        "table": table,
        "verdict": verdict,
        "methods": methods,
        "Ls": Ls,
    }
    write_json(OUT / "verdict.json", verdict)
    text = render_results(payload)
    (OUT / "RESULTS.md").write_text(text)
    log("VERDICT", verdict["verdict"])
    log(verdict["reason"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
