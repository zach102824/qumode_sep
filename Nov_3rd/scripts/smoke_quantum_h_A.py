#!/usr/bin/env python3
"""Smoke family A / Opus rank-1: transverse-field Max-E3LIN2.

Does not write qaoa_angles/ or train69_ladder_fresh250/.
Objective is ⟨H⟩ from the Z-histogram and the X-histogram.
"""
from __future__ import annotations

import argparse
import json
import math
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
from Nov_3rd.quantum_h_tf import (
    FockSim,
    SCALE1000_BASE,
    ed_observables,
    fit_fourier,
    fourier_schedule,
    load_scale_holdout,
    load_train_instances,
    paper_fourier_init,
    phi_powers,
    self_check,
    stack_diags,
)

TRAIN_PACKS = S.OUT / "beat_qaoa_12h" / "train_packs"
LAMBDAS_DEFAULT = (0.25, 0.5, 0.75, 1.0, 1.25, 1.5)
PAPER_BETA = 0.01
PAPER_GAMMA = -0.01


def log(*a):
    print(time.strftime("%H:%M:%S"), "[smokeA]", *a, flush=True)


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    if isinstance(o, (np.floating, float)):
        return float(o)
    if isinstance(o, (np.integer, int)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, torch.Tensor):
        return o.detach().cpu().tolist()
    if o is None:
        return None
    return o


def _med(xs) -> float:
    return float(np.median(np.asarray(xs, dtype=np.float64)))


def _mean(xs) -> float:
    return float(np.mean(np.asarray(xs, dtype=np.float64)))


def summarize_ed(rows: list[dict]) -> dict:
    def col(k):
        return [r[k] for r in rows]

    peak = col("peak")
    gap = col("prod_gap")
    return {
        "B": len(rows),
        "med_peak": _med(peak),
        "mean_peak": _mean(peak),
        "frac_peak_lt_0.5": float(np.mean(np.asarray(peak) < 0.5)),
        "med_prod_gap": _med(gap),
        "mean_prod_gap": _mean(gap),
        "frac_prod_gap_ge_0.02": float(np.mean(np.asarray(gap) >= 0.02)),
        "med_entropy": _med(col("entropy")),
        "med_chi": _med(col("chi")),
        "med_gap": _med(col("gap")),
        "med_E0": _med(col("E0")),
        "med_E0_over_n": _med([r["E0"] / r.get("n", 1) for r in rows]),
        "med_E_cl": _med(col("E_cl")),
        "med_E_prod": _med(col("E_prod")),
        "med_sx": _med(col("sx")),
        "quantum_pass": bool(_med(peak) < 0.5 and _med(gap) >= 0.02),
    }


def attach_n(rows, n):
    for r in rows:
        r["n"] = int(n)
    return rows


def scan_lambdas(insts, n, lambdas) -> dict:
    out = {}
    per = {}
    for lam in lambdas:
        t0 = time.time()
        rows = attach_n([ed_observables(ins, lam, product=True) for ins in insts], n)
        summ = summarize_ed(rows)
        summ["wall_s"] = time.time() - t0
        # large-λ density e ≈ -λ - (m/n)/(6λ), using each instance's m
        analytic = [-float(lam) - (ins.m / float(n)) / (6.0 * float(lam)) for ins in insts]
        summ["analytic_large_lam_e"] = float(np.median(analytic))
        summ["med_m"] = float(np.median([ins.m for ins in insts]))
        out[float(lam)] = summ
        per[float(lam)] = rows
        log(
            f"ED n={n} λ={lam:.2f} peak={summ['med_peak']:.3f} "
            f"prodΔ={summ['med_prod_gap']:.3f} S={summ['med_entropy']:.3f} "
            f"χ={summ['med_chi']:.3f} E0/n={summ['med_E0_over_n']:.3f} "
            f"pass={summ['quantum_pass']} ({summ['wall_s']:.1f}s)"
        )
    return {"summary": out, "per_instance": per}


def choose_lambda(scans: dict, lambdas) -> dict:
    """One smoke λ near 0.9 λ_c on both n, preferring the quantumness gate."""
    lams = [float(x) for x in lambdas]
    lc = {}
    lc_chi = {}
    for n, sc in scans.items():
        summ = sc["summary"]
        lc[n] = max(lams, key=lambda l: summ[l]["med_entropy"])
        lc_chi[n] = max(lams, key=lambda l: summ[l]["med_chi"])
    best = None
    for lam in lams:
        score = 0.0
        passes = {}
        for n, lc_n in lc.items():
            row = scans[n]["summary"][lam]
            passes[n] = bool(row["quantum_pass"])
            target = 0.9 * lc_n
            score += abs(lam - target)
            if not (0.8 * lc_n - 1e-9 <= lam <= lc_n + 1e-9):
                score += 0.75
            if not passes[n]:
                score += 5.0
        # prefer a λ that passes on every n
        if all(passes.values()):
            score -= 0.25
        if best is None or score < best[0] - 1e-12:
            best = (score, lam, passes)
    assert best is not None
    return {
        "lambda": best[1],
        "score": best[0],
        "passes_at_choice": {str(k): v for k, v in best[2].items()},
        "lambda_c_entropy": {str(k): v for k, v in lc.items()},
        "lambda_c_chi": {str(k): v for k, v in lc_chi.items()},
    }


def _metrics_from_energy(e: torch.Tensor, e0: torch.Tensor, n: int, lam: float) -> dict:
    e_init = -float(lam) * int(n)
    e_d = e.detach()
    eps = (e_d - e0) / float(n)
    denom = (e_init - e0).clamp_min(1e-12)
    f = (e_init - e_d) / denom
    eps_np = eps.cpu().numpy()
    f_np = f.cpu().numpy()
    return {
        "mean_H": float(e_d.mean()),
        "med_H": float(e_d.median()),
        "med_eps": float(np.median(eps_np)),
        "mean_eps": float(np.mean(eps_np)),
        "med_f": float(np.median(f_np)),
        "mean_f": float(np.mean(f_np)),
        "E_init": e_init,
    }


def _decode(method: str, theta: torch.Tensor, L: int):
    if method == "fourier_qaoa":
        q = L
        gamma = fourier_schedule(theta[:q], L)
        beta = fourier_schedule(theta[q:], L)
        k = torch.arange(4, device=theta.device, dtype=theta.dtype)
        phi = (-beta.reshape(L, 1)) * k.reshape(1, 4)
        return gamma, phi
    if method == "power_p":
        g, c1, c2, c3 = theta.reshape(4, L)
        return g, phi_powers(c1, c2, c3)
    raise ValueError(method)


def _init_theta(method: str, L: int, seed: int, n: int, dtype, device) -> torch.Tensor:
    rng = np.random.default_rng(50_000 + 1000 * seed + 10 * L + n)
    if method == "fourier_qaoa":
        ug, ub = paper_fourier_init(L, L)
        # reconstruction check of the un-jittered paper point
        raw = np.concatenate([ug, ub])
        jit = rng.normal(0.0, 0.01, size=raw.shape)
        return torch.tensor(raw + jit, dtype=dtype, device=device)
    if method == "power_p":
        gamma = np.full(L, PAPER_GAMMA) + rng.normal(0.0, 0.01, size=L)
        c1 = np.full(L, -PAPER_BETA) + rng.normal(0.0, 0.01, size=L)
        c2 = rng.normal(0.0, 0.01, size=L)
        c3 = rng.normal(0.0, 0.01, size=L)
        return torch.tensor(np.concatenate([gamma, c1, c2, c3]), dtype=dtype, device=device)
    raise ValueError(method)


def n_params(method: str, L: int) -> int:
    if method == "fourier_qaoa":
        return 2 * L  # q = L Fourier coeffs for β and for γ
    if method == "power_p":
        return 4 * L  # γ, c1, c2, c3
    raise ValueError(method)


def power_theta_from_fourier(theta_f: torch.Tensor, L: int) -> torch.Tensor:
    """Dual map: γ and φ_j=−β j, with c2=c3=0. SNAP then contains that QAOA point."""
    q = int(L)
    gamma = fourier_schedule(theta_f[:q], L)
    beta = fourier_schedule(theta_f[q:], L)
    c1 = -beta
    c2 = torch.zeros(L, dtype=theta_f.dtype, device=theta_f.device)
    c3 = torch.zeros(L, dtype=theta_f.dtype, device=theta_f.device)
    return torch.cat([gamma.detach(), c1.detach(), c2, c3])


def train_one(
    *,
    method: str,
    n: int,
    L: int,
    seed: int,
    sim: FockSim,
    Hc_tr: torch.Tensor,
    Hc_ho: torch.Tensor,
    e0_tr: torch.Tensor,
    e0_ho: torch.Tensor,
    lam: float,
    rounds: int,
    max_iter: int,
    lr: float,
    theta0: torch.Tensor | None = None,
) -> dict:
    dtype = Hc_tr.dtype
    device = Hc_tr.device
    if theta0 is None:
        theta = torch.nn.Parameter(_init_theta(method, L, seed, n, dtype, device))
    else:
        theta = torch.nn.Parameter(theta0.detach().to(dtype=dtype, device=device).clone())
    t0 = time.time()

    def energies(Hc):
        gamma, phi = _decode(method, theta, L)
        return sim.energy(Hc, gamma, phi, lam)

    with torch.no_grad():
        init_tr = _metrics_from_energy(energies(Hc_tr), e0_tr, n, lam)
        init_ho = _metrics_from_energy(energies(Hc_ho), e0_ho, n, lam)

    best = {
        "ho_med_eps": float(init_ho["med_eps"]),
        "round": -1,
        "theta": theta.detach().clone(),
        "train": init_tr,
        "hold": init_ho,
    }
    hist = [{"round": -1, "train_med_eps": init_tr["med_eps"], "hold_med_eps": init_ho["med_eps"]}]
    stall = 0
    for r in range(rounds):
        opt = torch.optim.LBFGS(
            [theta],
            lr=lr,
            max_iter=max_iter,
            history_size=16,
            line_search_fn="strong_wolfe",
        )

        def closure():
            opt.zero_grad(set_to_none=True)
            e = energies(Hc_tr)
            loss = e.mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([theta], 1e3)
            return loss

        try:
            loss = opt.step(closure)
            loss_f = float(loss.detach())
        except Exception as exc:
            log(f"  LBFGS skip {method} n={n} L={L} seed={seed} r={r}: {exc}")
            stall += 1
            if stall >= 4:
                break
            continue
        with torch.no_grad():
            tr = _metrics_from_energy(energies(Hc_tr), e0_tr, n, lam)
            ho = _metrics_from_energy(energies(Hc_ho), e0_ho, n, lam)
        hist.append(
            {
                "round": r,
                "loss": loss_f,
                "train_med_eps": tr["med_eps"],
                "hold_med_eps": ho["med_eps"],
                "hold_med_f": ho["med_f"],
            }
        )
        if ho["med_eps"] < best["ho_med_eps"] - 1e-6:
            best = {
                "ho_med_eps": float(ho["med_eps"]),
                "round": r,
                "theta": theta.detach().clone(),
                "train": tr,
                "hold": ho,
            }
            stall = 0
        else:
            stall += 1
        if r % 5 == 0 or r + 1 == rounds:
            log(
                f"  {method} n={n} L={L} s={seed} r={r} "
                f"train ε={tr['med_eps']:.4f} hold ε={ho['med_eps']:.4f} "
                f"f={ho['med_f']:.3f} best@{best['round']}"
            )
        if stall >= 8 and r >= 6:
            log(f"  early stop {method} n={n} L={L} seed={seed} at r={r}")
            break

    with torch.no_grad():
        theta.copy_(best["theta"])
        hold = _metrics_from_energy(energies(Hc_ho), e0_ho, n, lam)
        train_m = _metrics_from_energy(energies(Hc_tr), e0_tr, n, lam)
    gamma, phi = _decode(method, theta.detach(), L)
    return {
        "method": method,
        "n": n,
        "L": L,
        "seed": seed,
        "lam": float(lam),
        "n_params": n_params(method, L),
        "rounds_ran": int(hist[-1]["round"]) if hist else 0,
        "best_round": int(best["round"]),
        "init_hold": init_ho,
        "train": train_m,
        "hold": hold,
        "history_tail": hist[-6:],
        "gamma": gamma.detach().cpu().tolist(),
        "phi": phi.detach().cpu().tolist(),
        "theta": theta.detach().cpu().tolist(),
        "wall_s": time.time() - t0,
        "optimizer": "LBFGS",
        "lr": lr,
        "max_iter": max_iter,
        "selection": "best holdout median ε during training (warm start included)",
        "warm_start": theta0 is not None,
    }


def aggregate_method(rows: list[dict]) -> dict:
    eps = [r["hold"]["med_eps"] for r in rows]
    fs = [r["hold"]["med_f"] for r in rows]
    med_eps = float(np.median(eps))
    med_f = float(np.median(fs))
    std_eps = float(np.std(eps, ddof=1)) if len(eps) > 1 else 0.0
    return {
        "seeds": [r["seed"] for r in rows],
        "per_seed_med_eps": eps,
        "per_seed_med_f": fs,
        "med_eps": med_eps,
        "std_eps": std_eps,
        "med_f": med_f,
        "best_seed_med_eps": float(np.min(eps)),
        "n_params": rows[0]["n_params"],
    }


def ratio_table(by_key: dict) -> dict:
    """by_key[(n,L)][method] = aggregate."""
    out = {}
    for (n, L), methods in by_key.items():
        q = methods["fourier_qaoa"]
        s = methods["power_p"]
        R = float(s["med_eps"] / q["med_eps"]) if q["med_eps"] > 1e-8 else float("inf")
        adv = float(q["med_eps"] - s["med_eps"])
        if R <= 0.98:
            verdict = "BEAT"
        elif R <= 1.02:
            verdict = "TIE"
        else:
            verdict = "UNDER"
        noise = max(q["std_eps"], s["std_eps"])
        out[f"n{n}_L{L}"] = {
            "n": n,
            "L": L,
            "med_eps_qaoa": q["med_eps"],
            "med_eps_snap": s["med_eps"],
            "std_eps_qaoa": q["std_eps"],
            "std_eps_snap": s["std_eps"],
            "med_f_qaoa": q["med_f"],
            "med_f_snap": s["med_f"],
            "R": R,
            "adv_eps": adv,
            "verdict": verdict,
            "within_one_seed_std": bool(abs(q["med_eps"] - s["med_eps"]) <= noise + 1e-12),
            "saturated": bool(q["med_f"] > 0.99 and s["med_f"] > 0.99),
            "n_params_qaoa": q["n_params"],
            "n_params_snap": s["n_params"],
        }
    return out


def kill_rules(table: dict) -> dict:
    """Opus smoke kills, evaluated at L=16."""
    cells = [v for v in table.values() if int(v["L"]) == 16]
    by_n = {int(c["n"]): c for c in cells}
    have_both = 6 in by_n and 9 in by_n
    within_both = bool(have_both and all(by_n[n]["within_one_seed_std"] for n in (6, 9)))
    sat_both = bool(have_both and all(by_n[n]["saturated"] for n in (6, 9)))
    shrink = False
    shrink_detail = ""
    if 6 in by_n and 9 in by_n:
        a6 = by_n[6]["adv_eps"]
        a9 = by_n[9]["adv_eps"]
        noise = by_n[6]["std_eps_snap"] + by_n[9]["std_eps_snap"]
        # Clear finite-size shrink of a positive SNAP advantage.
        shrink = bool(a6 > 0 and a9 < a6 - max(noise, 1e-4) and a9 < 0.85 * a6)
        shrink_detail = f"adv n6={a6:.5f} n9={a9:.5f} noise={noise:.5f}"
    fired = []
    if within_both:
        fired.append("L16_within_one_seed_std_both_n")
    if shrink:
        fired.append("advantage_shrinks_n6_to_n9")
    if sat_both:
        fired.append("saturated_f_gt_0.99_both_methods_both_n")
    return {
        "fired": fired,
        "within_one_seed_std_both_n": within_both,
        "advantage_shrinks": shrink,
        "shrink_detail": shrink_detail,
        "saturated_both_n": sat_both,
        "kill": bool(fired),
    }


def family_verdict(choice: dict, table: dict, kills: dict, quantum_ok: bool) -> str:
    if not quantum_ok:
        return "KILL"
    if kills.get("kill"):
        return "KILL"
    l16 = [v for v in table.values() if int(v["L"]) == 16]
    if len(l16) < 2:
        return "needs_edit"
    beat_both = all(v["verdict"] == "BEAT" for v in l16)
    if beat_both:
        return "PASS"
    return "needs_edit"


def fmt(x, nd=4):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "nan"
    return f"{x:.{nd}f}"


def yesno(v) -> str:
    return "yes" if v in (True, 1) else "no"


def render_md(payload: dict) -> str:
    ch = payload["choice"]
    lines = []
    lines.append("# RESULTS — family A transverse-field Max-E3LIN2")
    lines.append("")
    lines.append(f"**Verdict: {payload['verdict']}**")
    lines.append("")
    lines.append("## Hamiltonian")
    lines.append("")
    lines.append("Opus sign, rank-1 transverse-field Max-E3LIN2:")
    lines.append("")
    lines.append(r"\[")
    lines.append(r"H = \sum_{(a,b,c)\in E} J_{abc} Z_a Z_b Z_c \;-\; \lambda \sum_i X_i")
    lines.append(r"\]")
    lines.append("")
    lines.append(
        "Packs store 4-SAT unsat tables, not 3-local `cls_s`. "
        "Each instance keeps cross-cavity triples taken from its 4-clauses "
        "(variables not all in one 3-qubit cavity). "
        "`J=±1` is the product of those three literal signs, first occurrence wins. "
        "Greedy selection caps degree at 3 so `m=n` when the geometry allows it. "
        "Train seeds are the existing `beat_qaoa_12h/train_packs` rows; "
        "regenerated unsat `E` matches the pack. "
        "Holdout seeds use the scale1000 base `54_000_000_000` "
        "(`regen_scale1000.py`); the scale1000 npz is not in this checkout. "
        "Classical `qaoa_angles/` and `train69_ladder_fresh250/` were not written."
    )
    lines.append("")
    lines.append(
        "Gate sequence is the locked Nov_3rd motif: "
        r"`|0⟩ → [U_c^X → Hamming-shell SNAP]^L → W` with BinH `W=H^{⊗3}`. "
        r"`U_c^X = W exp(-iγ H_c) W`. "
        "The field is not inside `U_c`. "
        r"⟨H⟩ = ⟨H_c⟩ from the Z-histogram (after the final W) "
        r"minus λ Σ ⟨X_i⟩ from the X-histogram (final W omitted), "
        r"⟨X_i⟩ = 2 p(+)-1."
    )
    lines.append("")
    lines.append(
        "Walsh-degree γ is a null knob here: `H_c` is pure degree 3, so the "
        "degree-1/2/4 Walsh pieces are identically zero and graded γ collapses "
        "to scalar γ. It was not trained."
    )
    lines.append("")
    lines.append("## Exact diagonalization")
    lines.append("")
    ent = ch["lambda_c_entropy"]
    chi = ch["lambda_c_chi"]
    lines.append(
        f"Smoke λ = **{ch['lambda']}**. "
        f"Entropy-peak λ_c: n6={ent.get('6', ent.get(6))}, n9={ent.get('9', ent.get(9))}. "
        f"Fidelity-susceptibility peak: n6={chi.get('6', chi.get(6))}, n9={chi.get('9', chi.get(9))}."
    )
    lines.append("")
    qlines = ["Quantumness on train+holdout at the smoke λ (median gate):", ""]
    for n, block in payload["quantum_at_choice"].items():
        both = block["all"]
        qlines.append(
            f"- n={n}: med peak={fmt(both['med_peak'],3)}, "
            f"med product gap={fmt(both['med_prod_gap'],3)}, "
            f"fraction with product gap≥2%={fmt(both['frac_prod_gap_ge_0.02'],2)}, "
            f"med cavity entropy={fmt(both['med_entropy'],3)} bits, "
            f"pass={yesno(block['pass'])}."
        )
    lines.extend(qlines)
    lines.append("")
    lines.append(
        "Gate: median bitstring peak of the ground state < 0.5 and "
        "median product-state gap (E_prod−E_0)/|E_0| ≥ 2%. "
        "Product states are XZ-plane, ⟨Z⟩=cos θ, ⟨X⟩=sin θ."
    )
    lines.append("")
    lines.append("| n | λ | med peak | med prod gap | med S (bits) | med χ | med E0/n | large-λ e | pass |")
    lines.append("|--:|--:|---------:|-------------:|-------------:|------:|---------:|----------:|:----:|")
    for n, sc in payload["scans"].items():
        for lam, row in sc["summary"].items():
            lines.append(
                f"| {n} | {float(lam):.2f} | {fmt(row['med_peak'],3)} | {fmt(row['med_prod_gap'],3)} | "
                f"{fmt(row['med_entropy'],3)} | {fmt(row['med_chi'],3)} | {fmt(row['med_E0_over_n'],3)} | "
                f"{fmt(row['analytic_large_lam_e'],3)} | {yesno(row['quantum_pass'])} |"
            )
    lines.append("")
    lines.append("## Train")
    lines.append("")
    lines.append(
        f"Fourier QAOA: Zhou sine modes, q=L (full schedule, {2}*L coefficients), "
        "mixer is the linear Hamming shell φ_j=−β j, i.e. the QAOA point "
        r"`exp(-i β X/2)`."
    )
    lines.append(
        "power_p SNAP: φ_k = c1 k + c2 k² + c3 k³ on each cavity (4L parameters) "
        "plus scalar γ. Trained per n, so the cross-n exponent p is idle "
        "(c2, c3 already absorb a per-n scale). "
        "Fourier init is the Boulebnane–Montanaro point (β,γ)=(0.01,−0.01) plus N(0,0.01²) seed jitter. "
        "power_p is warm-started from that seed's trained Fourier dual "
        "(γ, c1=−β, c2=c3=0) and polished; the warm checkpoint is kept if polish "
        "does not improve holdout median ε. Optimizer is L-BFGS."
    )
    lines.append("")
    lines.append(
        f"B_train={payload['B_train']}, B_hold={payload['B_hold']}, "
        f"seeds={payload['seeds']}, L={payload['Ls']}."
    )
    lines.append("")
    lines.append("| n | L | method | params | Med ε | seed std | Med f | R | verdict |")
    lines.append("|--:|--:|:-------|------:|------:|---------:|------:|--:|:--------|")
    # stable order
    keys = sorted(payload["table"], key=lambda k: (payload["table"][k]["n"], payload["table"][k]["L"]))
    for k in keys:
        cell = payload["table"][k]
        n, L = cell["n"], cell["L"]
        lines.append(
            f"| {n} | {L} | fourier_qaoa | {cell['n_params_qaoa']} | {fmt(cell['med_eps_qaoa'])} | "
            f"{fmt(cell['std_eps_qaoa'])} | {fmt(cell['med_f_qaoa'],3)} |  |  |"
        )
        lines.append(
            f"| {n} | {L} | power_p | {cell['n_params_snap']} | {fmt(cell['med_eps_snap'])} | "
            f"{fmt(cell['std_eps_snap'])} | {fmt(cell['med_f_snap'],3)} | {fmt(cell['R'],3)} | {cell['verdict']} |"
        )
    lines.append("")
    lines.append(
        r"ε = (⟨H⟩ − E_0)/n on holdout, median over instances, then median over seeds. "
        "R = Med ε_SNAP / Med ε_QAOA. BEAT if R≤0.98, TIE if R≤1.02, else UNDER. "
        "Lower ε is better."
    )
    lines.append("")
    lines.append("Per-seed holdout Med ε:")
    lines.append("")
    for key in sorted(payload.get("per_method") or {}):
        ag = payload["per_method"][key]
        vals = ", ".join(fmt(v) for v in ag["per_seed_med_eps"])
        lines.append(f"- `{key}` seeds {ag['seeds']}: {vals}")
    lines.append("")
    lines.append("## Kill checks (L=16)")
    lines.append("")
    kd = payload["kills"]
    lines.append(f"- Within one seed-std on both n: {yesno(kd['within_one_seed_std_both_n'])}")
    lines.append(f"- Advantage shrinks n6→n9: {yesno(kd['advantage_shrinks'])} ({kd['shrink_detail']})")
    lines.append(f"- Both methods f>0.99 on both n: {yesno(kd['saturated_both_n'])}")
    lines.append(f"- Fired: {kd['fired'] or 'none'}")
    lines.append("")
    l16 = sorted(
        (v for v in payload["table"].values() if int(v["L"]) == 16),
        key=lambda v: int(v["n"]),
    )
    if l16:
        bits = []
        for v in l16:
            bits.append(
                f"n{v['n']} R={fmt(v['R'],3)} (ε {fmt(v['med_eps_snap'])} vs {fmt(v['med_eps_qaoa'])}, "
                f"gap {fmt(v['adv_eps'])}, seed std {fmt(max(v['std_eps_snap'], v['std_eps_qaoa']))})"
            )
        lines.append("L=16 margin: " + "; ".join(bits) + ".")
        lines.append("")
    lines.append("## Self-check")
    lines.append("")
    sc = payload["self_check"]
    lines.append(
        f"Fock twin vs `Nov3rdAnsatz` max |Δp| Z={sc['circuit_max_abs_pZ']:.2e}, "
        f"X={sc['circuit_max_abs_pX']:.2e}. "
        f"QAOA-point vs ideal qubit mixer max |Δp|={sc['qaoa_point_max_abs_p']:.2e}. "
        f"X-histogram vs Pauli ⟨X⟩ max |Δ|={sc['x_hist_vs_pauli']:.2e}."
    )
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    for note in payload.get("notes") or []:
        lines.append(f"- {note}")
    lines.append("")
    lines.append(f"Wall clock {payload['wall_s']:.1f}s. Branch `{payload.get('branch','')}`.")
    lines.append("")
    return "\n".join(lines)


def run_grid(bundles, lam, sims, args, ed_cache) -> list[dict]:
    """Fourier QAOA first. power_p polishes the dual of that same seed.

    The linear Hamming shell is the QAOA mixer, so a cold start can lose to
    QAOA just by missing that point. Warm φ_j=−β j, c2=c3=0, then L-BFGS.
    The round-−1 checkpoint is the warm dual, so best-by-hold never ships a
    polish that lost to it.
    """
    rows = []
    fourier_theta = {}
    for n in args.ns:
        b = bundles[n]
        e0_tr = torch.as_tensor(ed_cache[(n, "train")], dtype=torch.float64)
        e0_ho = torch.as_tensor(ed_cache[(n, "hold")], dtype=torch.float64)
        for L in args.Ls:
            rounds = args.rounds_L2 if L <= 2 else args.rounds_L16
            max_iter = args.max_iter_L2 if L <= 2 else args.max_iter_L16
            for seed in args.seeds:
                log(f"train fourier_qaoa n={n} L={L} seed={seed} λ={lam}")
                rec = train_one(
                    method="fourier_qaoa",
                    n=n,
                    L=L,
                    seed=seed,
                    sim=sims[n],
                    Hc_tr=b["Hc_tr"],
                    Hc_ho=b["Hc_ho"],
                    e0_tr=e0_tr,
                    e0_ho=e0_ho,
                    lam=lam,
                    rounds=rounds,
                    max_iter=max_iter,
                    lr=args.lr,
                )
                rows.append(rec)
                fourier_theta[(n, L, seed)] = torch.tensor(rec["theta"], dtype=torch.float64)
                log(
                    f"  done hold ε={rec['hold']['med_eps']:.4f} "
                    f"f={rec['hold']['med_f']:.3f} wall={rec['wall_s']:.1f}s"
                )
            for seed in args.seeds:
                theta0 = power_theta_from_fourier(fourier_theta[(n, L, seed)], L)
                log(f"train power_p n={n} L={L} seed={seed} λ={lam} warm=fourier dual")
                rec = train_one(
                    method="power_p",
                    n=n,
                    L=L,
                    seed=seed,
                    sim=sims[n],
                    Hc_tr=b["Hc_tr"],
                    Hc_ho=b["Hc_ho"],
                    e0_tr=e0_tr,
                    e0_ho=e0_ho,
                    lam=lam,
                    rounds=rounds,
                    max_iter=max_iter,
                    lr=args.lr,
                    theta0=theta0,
                )
                rows.append(rec)
                log(
                    f"  done hold ε={rec['hold']['med_eps']:.4f} "
                    f"f={rec['hold']['med_f']:.3f} wall={rec['wall_s']:.1f}s "
                    f"best_round={rec['best_round']}"
                )
    return rows


def ed_split(insts, n, lam) -> list[dict]:
    rows = []
    for i, ins in enumerate(insts):
        rows.append(ed_observables(ins, lam, product=True))
        if (i + 1) % 8 == 0 or i + 1 == len(insts):
            log(f"  ED n={n} λ={lam} {i+1}/{len(insts)}")
    return attach_n(rows, n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default=str(S.OUT / "quantum_h_explore" / "smoke_A"))
    ap.add_argument("--ns", type=str, default="6,9")
    ap.add_argument("--Ls", type=str, default="2,16")
    ap.add_argument("--seeds", type=str, default="0,1,2")
    ap.add_argument("--lambdas", type=str, default=",".join(str(x) for x in LAMBDAS_DEFAULT))
    ap.add_argument("--B-scan", type=int, default=10)
    ap.add_argument("--B-train", type=int, default=32)
    ap.add_argument("--B-hold", type=int, default=48)
    ap.add_argument("--rounds-L2", type=int, default=18)
    ap.add_argument("--rounds-L16", type=int, default=24)
    ap.add_argument("--max-iter-L2", type=int, default=8)
    ap.add_argument("--max-iter-L16", type=int, default=8)
    ap.add_argument("--lr", type=float, default=0.35)
    ap.add_argument("--skip-self-check", action="store_true")
    args = ap.parse_args()
    args.ns = tuple(int(x) for x in args.ns.split(",") if x.strip())
    args.Ls = tuple(int(x) for x in args.Ls.split(",") if x.strip())
    args.seeds = tuple(int(x) for x in args.seeds.split(",") if x.strip())
    lambdas = tuple(float(x) for x in args.lambdas.split(",") if x.strip())

    torch.set_num_threads(8)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t_all = time.time()
    log("self-check" if not args.skip_self_check else "self-check skipped")
    check = {"ok": True, "skipped": True} if args.skip_self_check else self_check()
    if not args.skip_self_check:
        log("self-check", json.dumps(check))

    bundles = {}
    for n in args.ns:
        log(f"load n={n} train={args.B_train} hold={args.B_hold}")
        train = load_train_instances(n, args.B_train, TRAIN_PACKS / f"train_n{n}.npz")
        hold = load_scale_holdout(n, args.B_hold, SCALE1000_BASE)
        tr_seeds = {ins.seed for ins in train}
        ho_seeds = {ins.seed for ins in hold}
        inter = tr_seeds & ho_seeds
        if inter:
            raise RuntimeError(f"n={n} train/holdout seed overlap {list(inter)[:5]}")
        ms = [ins.m for ins in train + hold]
        degs = np.stack([ins.degree for ins in train + hold], 0)
        log(
            f"  m min/med/max={min(ms)}/{np.median(ms):.0f}/{max(ms)} "
            f"degree min/max={int(degs.min())}/{int(degs.max())} "
            f"frac J+={np.mean([J for ins in train+hold for *_,J in ins.clauses]) :.3f}"
        )
        bundles[n] = {
            "train": train,
            "hold": hold,
            "Hc_tr": stack_diags(train, "cpu", torch.float64),
            "Hc_ho": stack_diags(hold, "cpu", torch.float64),
        }

    scans = {}
    for n in args.ns:
        scans[n] = scan_lambdas(bundles[n]["train"][: args.B_scan], n, lambdas)
    choice = choose_lambda(scans, lambdas)
    lam = float(choice["lambda"])
    log("choice", json.dumps(choice))

    # Quantumness on the full train+holdout at the chosen λ (not only the scan slice).
    ed_full = {}
    ed_cache = {}
    quantum_at_choice = {}
    for n in args.ns:
        log(f"full ED n={n} λ={lam}")
        tr_rows = ed_split(bundles[n]["train"], n, lam)
        ho_rows = ed_split(bundles[n]["hold"], n, lam)
        both = summarize_ed(tr_rows + ho_rows)
        ho_sum = summarize_ed(ho_rows)
        quantum_at_choice[str(n)] = {
            "all": both,
            "holdout": ho_sum,
            "pass": bool(both["quantum_pass"]),
        }
        ed_full[str(n)] = {"train": tr_rows, "hold": ho_rows, "summary_all": both, "summary_hold": ho_sum}
        ed_cache[(n, "train")] = [r["E0"] for r in tr_rows]
        ed_cache[(n, "hold")] = [r["E0"] for r in ho_rows]
        log(
            f"  full n={n} peak={both['med_peak']:.3f} prodΔ={both['med_prod_gap']:.3f} "
            f"pass={both['quantum_pass']}"
        )

    quantum_ok = all(quantum_at_choice[str(n)]["pass"] for n in args.ns)
    notes = []
    for n in args.ns:
        insts_all = bundles[n]["train"] + bundles[n]["hold"]
        ms = [ins.m for ins in insts_all]
        dmax = [int(ins.degree.max()) for ins in insts_all]
        notes.append(
            f"n={n}: extracted m median {float(np.median(ms)):.0f} "
            f"(min {min(ms)}, max {max(ms)}), degree ≤ {max(dmax)}. "
            "Cap is 3, so a few n=9 instances stop at m=8 instead of raising the degree."
        )
    notes.append(
        "power_p has 4L parameters and Fourier QAOA has 2L (q=L). "
        "power_p is polished from the trained Fourier dual, so the comparison asks "
        "whether c2 and c3 shells beat the mixer they contain. It is not a matched-parameter claim."
    )
    notes.append(
        "lambda_c is the grid point with the highest median cavity-0 entanglement. "
        "The smoke lambda is the grid point nearest 0.9 lambda_c that still passes the quantumness gate on both n when one exists."
    )
    notes.append(
        "Holdout checkpoint selection uses the holdout median residual (protocol best-by-hold). "
        "The same rule is applied to both methods."
    )
    if not quantum_ok:
        notes.append("Quantumness gate failed on at least one n at every preferred λ; no training.")

    rows: list[dict] = []
    if quantum_ok:
        sims = {n: FockSim(n, dtype=torch.float64) for n in args.ns}
        rows = run_grid(bundles, lam, sims, args, ed_cache)
        # saturation retry: if both methods are saturated on both n at L=16,
        # retrain once at the entropy-peak λ when that differs.
        by_key: dict = {}
        for r in rows:
            by_key.setdefault((r["n"], r["L"]), {})
        # placeholder filled below after aggregate — check raw first
        agg = {}
        for r in rows:
            agg.setdefault((r["n"], r["L"], r["method"]), []).append(r)
        methods_agg = {}
        for (n, L, method), rs in agg.items():
            methods_agg.setdefault((n, L), {})[method] = aggregate_method(rs)
        table_now = ratio_table(methods_agg)
        kills_now = kill_rules(table_now)
        lc_vals = [float(v) for v in choice["lambda_c_entropy"].values()]
        lc_target = float(np.median(lc_vals))
        # snap target to grid
        lc_grid = min(lambdas, key=lambda l: abs(l - lc_target))
        if kills_now["saturated_both_n"] and abs(lc_grid - lam) > 1e-9:
            notes.append(
                f"L=16 saturated at λ={lam}. Retrained once at entropy-peak grid λ={lc_grid}."
            )
            log(f"saturation retry at λ={lc_grid}")
            lam = float(lc_grid)
            choice["lambda_retried_from"] = choice["lambda"]
            choice["lambda"] = lam
            for n in args.ns:
                tr_rows = ed_split(bundles[n]["train"], n, lam)
                ho_rows = ed_split(bundles[n]["hold"], n, lam)
                both = summarize_ed(tr_rows + ho_rows)
                quantum_at_choice[str(n)] = {
                    "all": both,
                    "holdout": summarize_ed(ho_rows),
                    "pass": bool(both["quantum_pass"]),
                }
                ed_full[str(n)] = {
                    "train": tr_rows,
                    "hold": ho_rows,
                    "summary_all": both,
                    "summary_hold": summarize_ed(ho_rows),
                }
                ed_cache[(n, "train")] = [r["E0"] for r in tr_rows]
                ed_cache[(n, "hold")] = [r["E0"] for r in ho_rows]
            quantum_ok = all(quantum_at_choice[str(n)]["pass"] for n in args.ns)
            if quantum_ok:
                rows = run_grid(bundles, lam, sims, args, ed_cache)
            else:
                notes.append("Retry λ failed the quantumness gate; training at the retry λ was skipped.")
                rows = []

    table = {}
    per_method = {}
    if rows:
        grouped: dict = {}
        for r in rows:
            grouped.setdefault((r["n"], r["L"], r["method"]), []).append(r)
        methods_agg = {}
        for (n, L, method), rs in grouped.items():
            ag = aggregate_method(rs)
            methods_agg.setdefault((n, L), {})[method] = ag
            per_method[f"n{n}_L{L}_{method}"] = ag
        table = ratio_table(methods_agg)
    kills = kill_rules(table) if table else {
        "fired": ["no_train"],
        "within_one_seed_std_both_n": False,
        "advantage_shrinks": False,
        "shrink_detail": "",
        "saturated_both_n": False,
        "kill": True,
    }
    if not quantum_ok:
        kills = dict(kills)
        kills["fired"] = list(kills.get("fired") or []) + ["quantumness_gate"]
        kills["kill"] = True
    verdict = family_verdict(choice, table, kills, quantum_ok)

    # drop bulky per-round traces from the summary; keep per-run files
    run_dir = out / "runs"
    run_dir.mkdir(exist_ok=True)
    slim_rows = []
    for r in rows:
        tag = f"n{r['n']}_L{r['L']}_{r['method']}_seed{r['seed']}"
        (run_dir / f"{tag}.json").write_text(json.dumps(_jsonable(r), indent=2) + "\n")
        slim = {k: r[k] for k in r if k not in ("theta", "phi", "gamma", "history_tail")}
        slim["history_tail"] = r["history_tail"]
        slim_rows.append(slim)

    payload = {
        "family": "A",
        "name": "transverse-field Max-E3LIN2",
        "sign": "H = H_c - lambda * sum X",
        "verdict": verdict,
        "choice": choice,
        "quantum_at_choice": quantum_at_choice,
        "scans": {str(n): {"summary": sc["summary"]} for n, sc in scans.items()},
        "table": table,
        "per_method": per_method,
        "kills": kills,
        "self_check": check,
        "notes": notes,
        "B_scan": args.B_scan,
        "B_train": args.B_train,
        "B_hold": args.B_hold,
        "seeds": list(args.seeds),
        "Ls": list(args.Ls),
        "lambdas": list(lambdas),
        "runs": slim_rows,
        "wall_s": time.time() - t_all,
        "branch": "cursor/quantum-h-smoke-a-7099",
        "train_packs": str(TRAIN_PACKS),
        "scale_base": SCALE1000_BASE,
    }
    # instance seeds for audit
    payload["seeds_used"] = {
        str(n): {
            "train": [ins.seed for ins in bundles[n]["train"]],
            "hold": [ins.seed for ins in bundles[n]["hold"]],
            "m_train": [ins.m for ins in bundles[n]["train"]],
            "m_hold": [ins.m for ins in bundles[n]["hold"]],
        }
        for n in args.ns
    }
    (out / "summary.json").write_text(json.dumps(_jsonable(payload), indent=2) + "\n")
    (out / "ed_scan.json").write_text(
        json.dumps(_jsonable({"scans": scans, "choice": choice, "full": ed_full}), indent=2) + "\n"
    )
    # summary.json above includes scans summaries only; ed_scan has per-instance.
    md = render_md(payload)
    (out / "RESULTS.md").write_text(md)
    log("verdict", verdict, "wall", f"{payload['wall_s']:.1f}s")
    log("wrote", out)


if __name__ == "__main__":
    main()
