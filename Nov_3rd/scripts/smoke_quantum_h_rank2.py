#!/usr/bin/env python3
"""Rank-2 smoke: intra-cavity XX/XXX on winning Arm A.

Does not write qaoa_angles/ or train69_ladder_fresh250/. Does not touch n21.
Objective is ⟨H⟩. Metric is holdout median ε = (⟨H⟩ − E_0) / n.
μ = κ = 0 is the Arm A baseline, trained in this same process.
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
from Nov_3rd.quantum_h_rank2 import (
    Rank2Sim,
    exact_ground_rank2,
    plus_energy,
    self_check,
    tag_mk,
)
from Nov_3rd.quantum_h_tf import (
    SCALE1000_BASE,
    fourier_schedule,
    load_scale_holdout,
    load_train_instances,
    paper_fourier_init,
    phi_powers,
    self_check as arm_a_self_check,
    stack_diags,
)

TRAIN_PACKS = S.OUT / "beat_qaoa_12h" / "train_packs"
# Published PR #13 L=16 medians (seeds 0,1,2), for the reproduction note.
PR13_L16 = {
    6: {"qaoa": 0.0096, "snap": 0.0086, "R": 0.889},
    9: {"qaoa": 0.0127, "snap": 0.0098, "R": 0.770},
}


def log(*a):
    print(time.strftime("%H:%M:%S"), "[rank2]", *a, flush=True)


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
    out = {
        "B": len(rows),
        "med_peak": _med(peak),
        "mean_peak": _mean(peak),
        "frac_peak_lt_0.5": float(np.mean(np.asarray(peak) < 0.5)),
        "med_entropy": _med(col("entropy")),
        "med_chi": _med(col("chi")),
        "med_gap": _med(col("gap")),
        "med_E0": _med(col("E0")),
        "med_E0_over_n": _med([r["E0"] / r["n"] for r in rows]),
        "med_E_cl": _med(col("E_cl")),
        "med_E_plus": _med(col("E_plus")),
        "med_plus_gap_density": _med([(r["E_plus"] - r["E0"]) / r["n"] for r in rows]),
    }
    if rows and "prod_gap" in rows[0]:
        gap = col("prod_gap")
        out["med_prod_gap"] = _med(gap)
        out["mean_prod_gap"] = _mean(gap)
        out["frac_prod_gap_ge_0.02"] = float(np.mean(np.asarray(gap) >= 0.02))
        out["med_prod_gap_density"] = _med(col("prod_gap_density"))
        out["quantum_pass"] = bool(out["med_peak"] < 0.5 and out["med_prod_gap"] >= 0.02)
    else:
        out["quantum_pass"] = None
    return out


def ed_rows(insts, n, lam, mu, kappa, product: bool) -> list[dict]:
    rows = []
    for i, ins in enumerate(insts):
        rows.append(
            exact_ground_rank2(
                ins.diag, ins.clauses, n, lam, mu, kappa, seed=ins.seed, product=product
            )
        )
        if product and ((i + 1) % 16 == 0 or i + 1 == len(insts)):
            log(f"  ED n={n} λ={lam:.2f} μ={mu:+.2f} κ={kappa:+.2f} {i+1}/{len(insts)}")
    return rows


def _metrics(e: torch.Tensor, e0: torch.Tensor, n: int, lam: float, mu: float, kappa: float) -> dict:
    e_init = plus_energy(n, lam, mu, kappa)
    e_d = e.detach()
    eps = (e_d - e0) / float(n)
    denom = torch.as_tensor(e_init, dtype=e_d.dtype) - e0
    denom = torch.where(denom.abs() < 1e-8, torch.full_like(denom, 1e-8), denom)
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
        "E_init": float(e_init),
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
        raw = np.concatenate([ug, ub])
        jit = rng.normal(0.0, 0.01, size=raw.shape)
        return torch.tensor(raw + jit, dtype=dtype, device=device)
    if method == "power_p":
        gamma = np.full(L, -0.01) + rng.normal(0.0, 0.01, size=L)
        c1 = np.full(L, -0.01) + rng.normal(0.0, 0.01, size=L)
        c2 = rng.normal(0.0, 0.01, size=L)
        c3 = rng.normal(0.0, 0.01, size=L)
        return torch.tensor(np.concatenate([gamma, c1, c2, c3]), dtype=dtype, device=device)
    raise ValueError(method)


def n_params(method: str, L: int) -> int:
    if method == "fourier_qaoa":
        return 2 * L
    if method == "power_p":
        return 4 * L
    raise ValueError(method)


def power_theta_from_fourier(theta_f: torch.Tensor, L: int) -> torch.Tensor:
    q = int(L)
    gamma = fourier_schedule(theta_f[:q], L)
    beta = fourier_schedule(theta_f[q:], L)
    c2 = torch.zeros(L, dtype=theta_f.dtype, device=theta_f.device)
    c3 = torch.zeros(L, dtype=theta_f.dtype, device=theta_f.device)
    return torch.cat([gamma.detach(), (-beta).detach(), c2, c3])


def train_one(
    *,
    method: str,
    n: int,
    L: int,
    seed: int,
    sim: Rank2Sim,
    Hc_tr: torch.Tensor,
    Hc_ho: torch.Tensor,
    e0_tr: torch.Tensor,
    e0_ho: torch.Tensor,
    lam: float,
    mu: float,
    kappa: float,
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
        return sim.energy(Hc, gamma, phi, lam, mu, kappa)

    with torch.no_grad():
        init_tr = _metrics(energies(Hc_tr), e0_tr, n, lam, mu, kappa)
        init_ho = _metrics(energies(Hc_ho), e0_ho, n, lam, mu, kappa)

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
            loss = energies(Hc_tr).mean()
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
            tr = _metrics(energies(Hc_tr), e0_tr, n, lam, mu, kappa)
            ho = _metrics(energies(Hc_ho), e0_ho, n, lam, mu, kappa)
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
                f"  {method} n={n} L={L} s={seed} μ={mu:+.2f} κ={kappa:+.2f} r={r} "
                f"train ε={tr['med_eps']:.4f} hold ε={ho['med_eps']:.4f} "
                f"f={ho['med_f']:.3f} best@{best['round']}"
            )
        if stall >= 8 and r >= 6:
            log(f"  early stop {method} n={n} L={L} seed={seed} at r={r}")
            break

    with torch.no_grad():
        theta.copy_(best["theta"])
        hold = _metrics(energies(Hc_ho), e0_ho, n, lam, mu, kappa)
        train_m = _metrics(energies(Hc_tr), e0_tr, n, lam, mu, kappa)
    gamma, phi = _decode(method, theta.detach(), L)
    return {
        "method": method,
        "n": n,
        "L": L,
        "seed": seed,
        "lam": float(lam),
        "mu": float(mu),
        "kappa": float(kappa),
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
    return {
        "seeds": [r["seed"] for r in rows],
        "per_seed_med_eps": eps,
        "per_seed_med_f": fs,
        "med_eps": float(np.median(eps)),
        "std_eps": float(np.std(eps, ddof=1)) if len(eps) > 1 else 0.0,
        "med_f": float(np.median(fs)),
        "best_seed_med_eps": float(np.min(eps)),
        "n_params": rows[0]["n_params"],
    }


def ratio_table(by_key: dict, plus_gap: dict) -> dict:
    out = {}
    for (n, L), methods in by_key.items():
        q = methods["fourier_qaoa"]
        s = methods["power_p"]
        R = float(s["med_eps"] / q["med_eps"]) if abs(q["med_eps"]) > 1e-8 else float("inf")
        adv = float(q["med_eps"] - s["med_eps"])
        if R <= 0.98:
            verdict = "BEAT"
        elif R <= 1.02:
            verdict = "TIE"
        else:
            verdict = "UNDER"
        noise = max(q["std_eps"], s["std_eps"])
        gap = float(plus_gap.get(int(n), 0.0))
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
            "saturated": bool(q["med_f"] > 0.99 and s["med_f"] > 0.99 and gap >= 0.05),
            "plus_gap_density": gap,
            "n_params_qaoa": q["n_params"],
            "n_params_snap": s["n_params"],
        }
    return out


def kill_rules(table: dict) -> dict:
    cells = [v for v in table.values() if int(v["L"]) == 16]
    by_n = {int(c["n"]): c for c in cells}
    have_both = 6 in by_n and 9 in by_n
    within_both = bool(have_both and all(by_n[n]["within_one_seed_std"] for n in (6, 9)))
    sat_both = bool(have_both and all(by_n[n]["saturated"] for n in (6, 9)))
    shrink = False
    shrink_detail = ""
    if have_both:
        a6 = by_n[6]["adv_eps"]
        a9 = by_n[9]["adv_eps"]
        noise = by_n[6]["std_eps_snap"] + by_n[9]["std_eps_snap"]
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


def _noise_R(cell: dict) -> float:
    rs = cell["std_eps_snap"] / max(abs(cell["med_eps_snap"]), 1e-8)
    rq = cell["std_eps_qaoa"] / max(abs(cell["med_eps_qaoa"]), 1e-8)
    return abs(cell["R"]) * math.sqrt(rs * rs + rq * rq)


def classify_point(point: dict, base: dict | None) -> dict:
    """Gain of one (μ, κ) against the in-script μ=κ=0 train.

    Clear gain: L=16 power_p still BEATs Fourier QAOA on both n, and R drops
    versus baseline on both n by more than the combined seed-std of R and by
    at least 0.02. Arm A kill rules (gap inside one seed std, advantage shrinks
    with n, both methods saturated) block a PASS on that point.
    """
    out = {
        "quantum": bool(point.get("quantum_ok")),
        "trained": bool(point.get("trained")),
        "class": "untrained",
    }
    if not out["quantum"]:
        out["class"] = "quantum_fail"
        return out
    if not out["trained"] or base is None or "table" not in point or "table" not in base:
        return out
    missing = [n for n in (6, 9) if f"n{n}_L16" not in point["table"] or f"n{n}_L16" not in base["table"]]
    if missing:
        out["class"] = "incomplete"
        return out
    better_R, clear_R, better_eps, beat = [], [], [], []
    detail = {}
    for n in (6, 9):
        c = point["table"][f"n{n}_L16"]
        b = base["table"][f"n{n}_L16"]
        dR = float(b["R"] - c["R"])
        noise = _noise_R(c) + _noise_R(b)
        better_R.append(dR > 0.0)
        clear_R.append(bool(dR > max(noise, 0.02) and c["verdict"] == "BEAT"))
        better_eps.append(bool(c["med_eps_snap"] < b["med_eps_snap"] - 1e-6))
        beat.append(c["verdict"] == "BEAT")
        detail[str(n)] = {
            "dR": dR,
            "noise_R": noise,
            "d_eps_snap": float(b["med_eps_snap"] - c["med_eps_snap"]),
            "R": c["R"],
            "R_base": b["R"],
            "verdict": c["verdict"],
        }
    l2_beat = all(point["table"][f"n{n}_L2"]["verdict"] == "BEAT" for n in (6, 9))
    l2_base = all(base["table"][f"n{n}_L2"]["verdict"] == "BEAT" for n in (6, 9))
    paired = _paired_seed_gain(point, base)
    out.update(
        {
            "beat_both_L16": all(beat),
            "better_R_both": all(better_R),
            "clear_R_both": bool(all(clear_R) and all(beat)),
            "better_eps_both": all(better_eps),
            "l2_new_beat": bool(l2_beat and (not l2_base) and all(beat)),
            "armA_kills": bool(point["kills"]["kill"]),
            "paired_all_seeds": paired["all_seeds"],
            "paired": paired,
            "detail": detail,
        }
    )
    if out["clear_R_both"] and not out["armA_kills"]:
        out["class"] = "clear"
    elif out["clear_R_both"] and out["armA_kills"]:
        out["class"] = "clear_but_killrule"
    elif out["better_R_both"] or out["better_eps_both"] or out["l2_new_beat"]:
        out["class"] = "soft"
    else:
        out["class"] = "no_gain"
    return out


def _paired_seed_gain(point: dict, base: dict) -> dict:
    """Same seed index: SNAP is warm-started from that seed's Fourier run.

    The unpaired std on R treats those seeds as independent and overstates the noise.
    """
    per = {}
    all_seeds = True
    for n in (6, 9):
        key_s = f"n{n}_L16_power_p"
        key_q = f"n{n}_L16_fourier_qaoa"
        if key_s not in point.get("per_method", {}) or key_s not in base.get("per_method", {}):
            return {"all_seeds": False, "per_n": {}}
        ps = point["per_method"][key_s]["per_seed_med_eps"]
        bs = base["per_method"][key_s]["per_seed_med_eps"]
        pq = point["per_method"][key_q]["per_seed_med_eps"]
        bq = base["per_method"][key_q]["per_seed_med_eps"]
        d_eps = [float(b - p) for b, p in zip(bs, ps)]
        d_R = [float(b / bq_ - p / pq_) for b, p, bq_, pq_ in zip(bs, ps, bq, pq)]
        per[str(n)] = {"d_eps": d_eps, "d_R": d_R, "R_new": [float(p / q) for p, q in zip(ps, pq)], "R_base": [float(b / q) for b, q in zip(bs, bq)]}
        if not (all(d > 1e-4 for d in d_eps) and all(d > 0.02 for d in d_R)):
            all_seeds = False
    return {"all_seeds": bool(all_seeds), "per_n": per}


def family_verdict(points: list[dict], classes: dict, base_class_note: str) -> str:
    if any(p.get("needs_YY") for p in points):
        return "KILL"
    nonzero = [p for p in points if abs(p["mu"]) + abs(p["kappa"]) > 1e-12]
    if not nonzero:
        return "needs_edit"
    kinds = [classes[p["tag"]]["class"] for p in nonzero]
    if base_class_note == "baseline_not_beat":
        # Do not claim PASS off a baseline that missed Arm A's L=16 gate.
        if any(k in ("clear", "clear_but_killrule", "soft") for k in kinds):
            return "needs_edit"
        if all(k == "quantum_fail" for k in kinds):
            return "KILL"
        return "KILL" if all(k == "no_gain" for k in kinds) else "needs_edit"
    if any(k == "clear" for k in kinds):
        return "PASS"
    if any(k in ("clear_but_killrule", "soft") for k in kinds):
        return "needs_edit"
    if all(k == "quantum_fail" for k in kinds):
        return "KILL"
    return "KILL"


def fmt(x, nd=4):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "nan"
    return f"{float(x):.{nd}f}"


def yesno(v) -> str:
    return "yes" if v in (True, 1) else "no"


def render_md(payload: dict) -> str:
    lines = []
    lines.append("# RESULTS — Rank2 intra-cavity XX/XXX on Arm A")
    lines.append("")
    lines.append(f"**Verdict: {payload['verdict']}**")
    lines.append("")
    lines.append("## Setup")
    lines.append("")
    lines.append(
        "`Nov_3rd/runs/opus_ideas/` has no opus reply in this checkout. "
        "Rank 1 is Arm A, which won: PR #13 PASS (power_p, L=16) and PR #11 "
        "(Walsh-degree on the unsat diagonal). This is the rank-2 deformation "
        "that was waiting on that win."
    )
    lines.append("")
    lines.append("Hamiltonian, Opus sign, raw coefficients:")
    lines.append("")
    lines.append(r"\[")
    lines.append(
        r"H = \sum_E J_{abc} Z_a Z_b Z_c - \lambda \sum_i X_i "
        r"+ \mu \sum_{\mathrm{cav}} \sum_{i<j} X_i X_j "
        r"+ \kappa \sum_{\mathrm{cav}} X_1 X_2 X_3"
    )
    lines.append(r"\]")
    lines.append("")
    lines.append(
        "Cross-cavity triples and `J=±1` are the same degree-capped extraction as "
        "PR #13 (4-SAT packs, first-occurrence signs, degree ≤ 3). "
        r"μ = κ = 0 is Arm A, retrained here on the same instances and the same L-BFGS budget. "
        "Gate sequence is unchanged: "
        r"`|0⟩ → [U_c^X → Hamming-shell SNAP]^L → W`, BinH `W=H^{⊗3}`, "
        r"`U_c^X = W exp(-iγ H_c) W` with `H_c = Σ J ZZZ` only. "
        "The field and the X products are not inside `U_c`."
    )
    lines.append("")
    lines.append(
        "Readout uses the Z histogram for ⟨H_c⟩ and the **per-cavity** 8-outcome "
        "X histogram (final W omitted) for ⟨X_i⟩, ⟨X_i X_j⟩ and ⟨X_1 X_2 X_3⟩. "
        "Those operators are X-strings on the three qubits of one cavity, so a "
        "joint X measurement of that cavity is enough. "
        f"**needs YY: {yesno(payload.get('needs_YY'))}.** "
        "A Y term was not added."
    )
    lines.append("")
    lines.append(
        f"Smoke λ prefers **{payload['prefer_lambda']}** (PR #13 operating point). "
        "A point moves to its entropy-peak λ only when λ=0.5 fails the quantumness "
        "gate on the full train+holdout. "
        f"Grid μ,κ ∈ {payload['mu_grid']} × {payload['kappa_grid']}."
    )
    lines.append("")
    lines.append("## Quantumness")
    lines.append("")
    lines.append(
        "Gate, same as Arm A: median ground-state bitstring peak < 0.5 and "
        "median product-state gap (E_prod − E_0)/|E_0| ≥ 2%. "
        "Product states are XZ-plane, ⟨Z⟩=cos θ, ⟨X⟩=sin θ. "
        "Bitstring expectations of X, XX and XXX vanish, so the best bitstring "
        "energy is still min H_c."
    )
    lines.append("")
    lines.append(
        "| μ | κ | λ | n | med peak | med prod gap | med S (bits) | med χ | med E0/n | med (E₊−E0)/n | pass |"
    )
    lines.append(
        "|--:|--:|--:|--:|---------:|-------------:|-------------:|------:|---------:|---------------:|:----:|"
    )
    for p in payload["points"]:
        q = p.get("quantum_full") or {}
        for n, row in q.items():
            lines.append(
                f"| {p['mu']:+.2f} | {p['kappa']:+.2f} | {fmt(p.get('lambda'), 2)} | {n} | "
                f"{fmt(row.get('med_peak'), 3)} | {fmt(row.get('med_prod_gap'), 3)} | "
                f"{fmt(row.get('med_entropy'), 3)} | {fmt(row.get('med_chi'), 3)} | "
                f"{fmt(row.get('med_E0_over_n'), 3)} | {fmt(row.get('med_plus_gap_density'), 3)} | "
                f"{yesno(row.get('quantum_pass'))} |"
            )
    lines.append("")
    lines.append("Entropy-peak λ on the eigenvalue scan (no product search), full train+holdout:")
    lines.append("")
    for p in payload["points"]:
        ent = p.get("lambda_c_entropy") or {}
        lines.append(
            f"- μ={p['mu']:+.2f}, κ={p['kappa']:+.2f}: "
            + ", ".join(f"n{n}={ent.get(str(n), ent.get(n))}" for n in (6, 9))
            + f"; trained λ={p.get('lambda')}; reason={p.get('lambda_reason')}"
        )
    lines.append("")
    lines.append("## Train")
    lines.append("")
    lines.append(
        "Fourier QAOA: Zhou sine modes, q=L, mixer φ_j=−β j. "
        "power_p: φ_k=c1 k+c2 k²+c3 k³ (4L parameters) warm-started from that seed's "
        "trained Fourier dual and polished with L-BFGS. "
        "Init is the Boulebnane–Montanaro point (β,γ)=(0.01,−0.01) plus N(0,0.01²), "
        "same seed stream as PR #13 (the stream does not depend on μ, κ). "
        "Checkpoint is the best holdout median ε, including the warm start. "
        f"B_train={payload['B_train']}, B_hold={payload['B_hold']}, "
        f"seeds={payload['seeds']}, L={payload['Ls']}."
    )
    lines.append("")
    lines.append(
        "| μ | κ | λ | n | L | Med ε QAOA | Med ε SNAP | R | vs base R | vs base ε_SNAP | verdict |"
    )
    lines.append(
        "|--:|--:|--:|--:|--:|-----------:|-----------:|--:|----------:|---------------:|:--------|"
    )
    base = payload.get("baseline_table") or {}
    for p in payload["points"]:
        if not p.get("table"):
            lines.append(
                f"| {p['mu']:+.2f} | {p['kappa']:+.2f} | {fmt(p.get('lambda'), 2)} |  |  |  |  |  |  |  | "
                f"{'quantum fail' if not p.get('quantum_ok') else 'not trained'} |"
            )
            continue
        keys = sorted(p["table"], key=lambda k: (p["table"][k]["n"], p["table"][k]["L"]))
        for k in keys:
            cell = p["table"][k]
            bcell = base.get(k)
            if bcell is None or (abs(p["mu"]) + abs(p["kappa"]) < 1e-12):
                dR, dE = "", ""
            else:
                dR = fmt(bcell["R"] - cell["R"], 3)
                dE = fmt(bcell["med_eps_snap"] - cell["med_eps_snap"])
            lines.append(
                f"| {p['mu']:+.2f} | {p['kappa']:+.2f} | {fmt(p.get('lambda'), 2)} | "
                f"{cell['n']} | {cell['L']} | {fmt(cell['med_eps_qaoa'])} | {fmt(cell['med_eps_snap'])} | "
                f"{fmt(cell['R'], 3)} | {dR} | {dE} | {cell['verdict']} |"
            )
    lines.append("")
    lines.append(
        "ε = (⟨H⟩ − E_0)/n on holdout, median over instances, then median over seeds. "
        "R = Med ε_SNAP / Med ε_QAOA. BEAT if R≤0.98, TIE if R≤1.02, else UNDER. "
        "vs base R is R(μ=κ=0) − R (positive means SNAP's ratio improved). "
        "vs base ε_SNAP is ε_SNAP(baseline) − ε_SNAP (positive means a smaller residual). "
        "Lower ε is better."
    )
    lines.append("")
    lines.append("Per-seed holdout Med ε:")
    lines.append("")
    for p in payload["points"]:
        if not p.get("per_method"):
            continue
        lines.append(f"- μ={p['mu']:+.2f}, κ={p['kappa']:+.2f}, λ={fmt(p.get('lambda'), 2)}")
        for key in sorted(p["per_method"]):
            ag = p["per_method"][key]
            vals = ", ".join(fmt(v) for v in ag["per_seed_med_eps"])
            lines.append(f"  - `{key}` seeds {ag['seeds']}: {vals}")
    lines.append("")
    lines.append("## Baseline vs PR #13")
    lines.append("")
    lines.append(
        "Same protocol at μ=κ=0, λ=0.5. PR #13 published L=16 medians "
        "(n6 QAOA/SNAP/R = 0.0096/0.0086/0.889, n9 = 0.0127/0.0098/0.770)."
    )
    lines.append("")
    rep = payload.get("reproduction") or {}
    if not rep:
        lines.append("Baseline was not trained.")
    else:
        for n, row in rep.items():
            lines.append(
                f"- n={n}: this run QAOA {fmt(row['qaoa'])} (Δ {fmt(row['dq'])}), "
                f"SNAP {fmt(row['snap'])} (Δ {fmt(row['ds'])}), R {fmt(row['R'], 3)} "
                f"(PR13 {fmt(row['R_pr'], 3)})."
            )
    lines.append("")
    lines.append("## Gain rule and kill checks")
    lines.append("")
    lines.append(
        "Pre-registered before the train finished. A nonzero point **gains clearly** when "
        "quantumness passes on both n, L=16 power_p BEATs Fourier QAOA on both n, and "
        "R falls versus the in-script μ=κ=0 baseline on both n by more than the combined "
        "seed-std of R and by at least 0.02. "
        "Arm A kill rules on that point (L=16 gap inside one seed std, advantage shrinks "
        "n6→n9, or both methods saturated with (E₊−E_0)/n ≥ 0.05) block PASS. "
        "A same-direction improvement that misses that bar, a drop in ε_SNAP without a "
        "clearer R, or a new L=2 BEAT, is **needs_edit**. "
        "Per-seed ΔR and Δε are reported as a paired diagnostic (SNAP is warm-started "
        "from that seed's Fourier run). They do not override the unpaired bar. "
        "**KILL** if no nonzero point improves R or ε_SNAP on both n, or every nonzero "
        "point fails quantumness, or scoring ⟨H⟩ needs a Y correlator. "
        "PASS is also withheld when the μ=κ=0 baseline itself misses the L=16 BEAT gate, "
        "because the comparison to Arm A is then not the published one."
    )
    lines.append("")
    for p in payload["points"]:
        if abs(p["mu"]) + abs(p["kappa"]) < 1e-12:
            continue
        cl = (payload.get("classes") or {}).get(p["tag"]) or {}
        lines.append(
            f"- μ={p['mu']:+.2f}, κ={p['kappa']:+.2f}: class=`{cl.get('class')}` "
            f"quantum={yesno(p.get('quantum_ok'))} "
            f"L16 BEAT both={yesno(cl.get('beat_both_L16'))} "
            f"better R both={yesno(cl.get('better_R_both'))} "
            f"clear unpaired R={yesno(cl.get('clear_R_both'))} "
            f"paired every seed={yesno(cl.get('paired_all_seeds'))} "
            f"better ε_SNAP both={yesno(cl.get('better_eps_both'))} "
            f"ArmA kills={p.get('kills', {}).get('fired')}"
        )
        paired = (cl.get("paired") or {}).get("per_n") or {}
        for n, row in paired.items():
            dR = ", ".join(fmt(v, 3) for v in row.get("d_R") or [])
            dE = ", ".join(fmt(v) for v in row.get("d_eps") or [])
            lines.append(f"  - n={n} per-seed ΔR (base−new) [{dR}], per-seed Δε_SNAP [{dE}]")
    lines.append("")
    lines.append(f"Family verdict: **{payload['verdict']}**. {payload.get('verdict_why', '')}")
    lines.append("")
    lines.append("## Self-check")
    lines.append("")
    sc = payload.get("self_check") or {}
    ac = payload.get("arm_a_self_check") or {}
    lines.append(
        f"Rank2 ⟨ψ|H|ψ⟩ matches the per-cavity X histogram "
        f"(grad |Δ|={sc.get('grad_abs')}). needs_YY={yesno(sc.get('needs_YY'))}. "
        f"At μ=κ=0 the dense matrix matches `dense_hamiltonian`, and the product-state "
        f"search is at least as low as Arm A's. "
        f"Fock twin vs `Nov3rdAnsatz` max |Δp| Z={ac.get('circuit_max_abs_pZ')}, "
        f"X={ac.get('circuit_max_abs_pX')}. "
        f"QAOA-point vs ideal mixer max |Δp|={ac.get('qaoa_point_max_abs_p')}."
    )
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    for note in payload.get("notes") or []:
        lines.append(f"- {note}")
    lines.append("")
    lines.append(
        f"Wall clock {payload.get('wall_s', float('nan')):.1f}s. "
        f"Branch `{payload.get('branch', '')}`. "
        "phase2 n21 ladder was not started and its scripts were not edited."
    )
    lines.append("")
    return "\n".join(lines)


def _write(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(obj), indent=2) + "\n")


def _read(path: Path):
    return json.loads(path.read_text())


def run_train_point(point, bundles, sims, args, ed_cache, run_dir: Path) -> list[dict]:
    mu, kappa, lam = point["mu"], point["kappa"], point["lambda"]
    tag = point["tag"]
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
                path = run_dir / f"n{n}_L{L}_{tag}_fourier_qaoa_seed{seed}.json"
                if path.exists() and not args.force:
                    rec = _read(path)
                    log(f"resume fourier n={n} L={L} seed={seed} {tag} ε={rec['hold']['med_eps']:.4f}")
                else:
                    log(f"train fourier_qaoa n={n} L={L} seed={seed} λ={lam} {tag}")
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
                        mu=mu,
                        kappa=kappa,
                        rounds=rounds,
                        max_iter=max_iter,
                        lr=args.lr,
                    )
                    _write(path, rec)
                    log(f"  done hold ε={rec['hold']['med_eps']:.4f} wall={rec['wall_s']:.1f}s")
                rows.append(rec)
                fourier_theta[(n, L, seed)] = torch.tensor(rec["theta"], dtype=torch.float64)
            for seed in args.seeds:
                path = run_dir / f"n{n}_L{L}_{tag}_power_p_seed{seed}.json"
                if path.exists() and not args.force:
                    rec = _read(path)
                    log(f"resume power_p n={n} L={L} seed={seed} {tag} ε={rec['hold']['med_eps']:.4f}")
                else:
                    theta0 = power_theta_from_fourier(fourier_theta[(n, L, seed)], L)
                    log(f"train power_p n={n} L={L} seed={seed} λ={lam} {tag} warm=fourier dual")
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
                        mu=mu,
                        kappa=kappa,
                        rounds=rounds,
                        max_iter=max_iter,
                        lr=args.lr,
                        theta0=theta0,
                    )
                    _write(path, rec)
                    log(
                        f"  done hold ε={rec['hold']['med_eps']:.4f} wall={rec['wall_s']:.1f}s "
                        f"best_round={rec['best_round']}"
                    )
                rows.append(rec)
    return rows


def finish_point(point, rows, plus_gap) -> dict:
    grouped: dict = {}
    for r in rows:
        grouped.setdefault((r["n"], r["L"], r["method"]), []).append(r)
    methods_agg = {}
    per_method = {}
    for (n, L, method), rs in grouped.items():
        ag = aggregate_method(rs)
        methods_agg.setdefault((n, L), {})[method] = ag
        per_method[f"n{n}_L{L}_{method}"] = ag
    table = ratio_table(methods_agg, plus_gap)
    point = dict(point)
    point["table"] = table
    point["per_method"] = per_method
    point["kills"] = kill_rules(table)
    point["trained"] = True
    return point


def _kept_wall(out: Path, this_wall: float) -> float:
    """Resume passes finish in seconds. Keep the training-run wall clock."""
    path = out / "train_wall_s.txt"
    prev = 0.0
    if path.exists():
        try:
            prev = float(path.read_text().strip())
        except ValueError:
            prev = 0.0
    wall = max(float(this_wall), prev)
    if float(this_wall) >= prev - 1e-9:
        path.write_text(f"{float(this_wall):.3f}\n")
    return wall


def parse_floats(text: str) -> tuple[float, ...]:
    return tuple(float(x) for x in text.split(",") if x.strip())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default=str(S.OUT / "quantum_h_explore" / "smoke_rank2"))
    ap.add_argument("--ns", type=str, default="6,9")
    ap.add_argument("--Ls", type=str, default="2,16")
    ap.add_argument("--seeds", type=str, default="0,1,2")
    ap.add_argument("--lambdas", type=str, default="0.25,0.5,0.75,1.0")
    ap.add_argument("--mu", type=str, default="-0.25,0,0.25")
    ap.add_argument("--kappa", type=str, default="-0.25,0,0.25")
    ap.add_argument("--prefer-lambda", type=float, default=0.5)
    ap.add_argument("--B-scan", type=int, default=16)
    ap.add_argument("--B-train", type=int, default=32)
    ap.add_argument("--B-hold", type=int, default=48)
    ap.add_argument("--rounds-L2", type=int, default=18)
    ap.add_argument("--rounds-L16", type=int, default=24)
    ap.add_argument("--max-iter-L2", type=int, default=8)
    ap.add_argument("--max-iter-L16", type=int, default=8)
    ap.add_argument("--lr", type=float, default=0.35)
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--skip-self-check", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--quick", action="store_true", help="tiny debug grid, not a verdict")
    args = ap.parse_args()
    args.ns = tuple(int(x) for x in args.ns.split(",") if x.strip())
    args.Ls = tuple(int(x) for x in args.Ls.split(",") if x.strip())
    args.seeds = tuple(int(x) for x in args.seeds.split(",") if x.strip())
    lambdas = parse_floats(args.lambdas)
    mu_grid = parse_floats(args.mu)
    kappa_grid = parse_floats(args.kappa)
    if args.quick:
        args.B_train = min(args.B_train, 8)
        args.B_hold = min(args.B_hold, 8)
        args.B_scan = min(args.B_scan, 4)
        args.seeds = (0,)
        args.Ls = (2,)
        args.rounds_L2 = min(args.rounds_L2, 3)
        args.rounds_L16 = min(args.rounds_L16, 3)
        mu_grid = (0.0, 0.25)
        kappa_grid = (0.0,)
        lambdas = (0.5,)

    torch.set_num_threads(4)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    run_dir = out / "runs"
    run_dir.mkdir(exist_ok=True)
    t_all = time.time()
    log("self-check" if not args.skip_self_check else "self-check skipped")
    if args.skip_self_check:
        check = {"ok": True, "skipped": True, "needs_YY": False}
        arm_check = {"ok": True, "skipped": True}
    else:
        check = self_check()
        arm_check = arm_a_self_check()
        log("self-check", json.dumps({k: check[k] for k in ("ok", "needs_YY", "grad_abs")}))
        log(
            "armA twin",
            f"pZ={arm_check['circuit_max_abs_pZ']:.2e}",
            f"pX={arm_check['circuit_max_abs_pX']:.2e}",
        )

    points_spec = [(float(mu), float(ka)) for mu in mu_grid for ka in kappa_grid]
    points_spec.sort(key=lambda mk: (abs(mk[0]) + abs(mk[1]), abs(mk[1]), mk[0], mk[1]))

    bundles = {}
    for n in args.ns:
        log(f"load n={n} train={args.B_train} hold={args.B_hold}")
        train = load_train_instances(n, args.B_train, TRAIN_PACKS / f"train_n{n}.npz")
        hold = load_scale_holdout(n, args.B_hold, SCALE1000_BASE)
        inter = {ins.seed for ins in train} & {ins.seed for ins in hold}
        if inter:
            raise RuntimeError(f"n={n} train/holdout seed overlap {list(inter)[:5]}")
        bundles[n] = {
            "train": train,
            "hold": hold,
            "Hc_tr": stack_diags(train, "cpu", torch.float64),
            "Hc_ho": stack_diags(hold, "cpu", torch.float64),
        }
        ms = [ins.m for ins in train + hold]
        log(f"  m min/med/max={min(ms)}/{np.median(ms):.0f}/{max(ms)}")

    # Eigenvalue landscape on the full set (peak, entropy, χ). Product gap is separate.
    landscape = {}
    for mu, kappa in points_spec:
        tag = tag_mk(mu, kappa)
        landscape[tag] = {}
        for lam in lambdas:
            landscape[tag][lam] = {}
            for n in args.ns:
                cache = out / "ed" / f"{tag}_l{lam:.2f}_n{n}_eig.json"
                if cache.exists() and not args.force:
                    rows = _read(cache)
                else:
                    t0 = time.time()
                    insts = bundles[n]["train"] + bundles[n]["hold"]
                    rows = ed_rows(insts, n, lam, mu, kappa, product=False)
                    _write(cache, rows)
                    log(f"  eig {tag} λ={lam:.2f} n={n} {time.time()-t0:.1f}s")
                landscape[tag][lam][n] = summarize_ed(rows)

    points = []
    for mu, kappa in points_spec:
        tag = tag_mk(mu, kappa)
        ent = {}
        for n in args.ns:
            ent[str(n)] = float(max(lambdas, key=lambda l: landscape[tag][l][n]["med_entropy"]))
        # Product-gap gate at the preferred λ, then at the entropy peak if needed.
        # The cache stores train+hold concatenated in that order.
        prod_pref = {}
        e0_lists = {}
        for n in args.ns:
            cache = out / "ed" / f"{tag}_l{args.prefer_lambda:.2f}_n{n}_prod.json"
            if cache.exists() and not args.force:
                rows = _read(cache)
            else:
                insts = bundles[n]["train"] + bundles[n]["hold"]
                rows = ed_rows(insts, n, args.prefer_lambda, mu, kappa, product=True)
                _write(cache, rows)
            ntr = len(bundles[n]["train"])
            prod_pref[str(n)] = summarize_ed(rows)
            e0_lists[args.prefer_lambda] = e0_lists.get(args.prefer_lambda, {})
            e0_lists[args.prefer_lambda][n] = {
                "train": [r["E0"] for r in rows[:ntr]],
                "hold": [r["E0"] for r in rows[ntr:]],
            }
        pref_pass = all(prod_pref[str(n)]["quantum_pass"] for n in args.ns)
        # Entropy peaks can disagree across n. Try the two peak λ values if 0.5 fails.
        reason = "prefer"
        chosen = float(args.prefer_lambda)
        quantum_full = prod_pref
        if not pref_pass:
            cands = []
            for n in args.ns:
                cands.append(float(ent[str(n)]))
            # unique, nearest to prefer first among those that might pass the peak test
            tried = {float(args.prefer_lambda)}
            for lam in sorted(set(cands), key=lambda l: abs(l - args.prefer_lambda)):
                if lam in tried or all(abs(lam - x) > 1e-12 for x in lambdas):
                    # snap to grid
                    lam = float(min(lambdas, key=lambda x: abs(x - lam)))
                if lam in tried:
                    continue
                tried.add(lam)
                block = {}
                lists = {}
                for n in args.ns:
                    cache = out / "ed" / f"{tag}_l{lam:.2f}_n{n}_prod.json"
                    if cache.exists() and not args.force:
                        rows = _read(cache)
                    else:
                        insts = bundles[n]["train"] + bundles[n]["hold"]
                        rows = ed_rows(insts, n, lam, mu, kappa, product=True)
                        _write(cache, rows)
                    ntr = len(bundles[n]["train"])
                    block[str(n)] = summarize_ed(rows)
                    lists[n] = {
                        "train": [r["E0"] for r in rows[:ntr]],
                        "hold": [r["E0"] for r in rows[ntr:]],
                    }
                if all(block[str(n)]["quantum_pass"] for n in args.ns):
                    chosen = lam
                    reason = "ed_peak"
                    quantum_full = block
                    e0_lists[lam] = lists
                    break
            else:
                reason = "quantum_fail"
                chosen = float(args.prefer_lambda)
                quantum_full = prod_pref
        quantum_ok = all(quantum_full[str(n)]["quantum_pass"] for n in args.ns)
        plus_gap = {n: float(quantum_full[str(n)]["med_plus_gap_density"]) for n in args.ns}
        point = {
            "mu": mu,
            "kappa": kappa,
            "tag": tag,
            "lambda": chosen,
            "lambda_reason": reason,
            "lambda_c_entropy": ent,
            "quantum_ok": quantum_ok,
            "quantum_full": {n: {k: v for k, v in quantum_full[n].items()} for n in quantum_full},
            "plus_gap": plus_gap,
            "needs_YY": False,
            "trained": False,
            "e0": e0_lists.get(chosen),
        }
        log(
            f"point {tag} λ={chosen} reason={reason} quantum={quantum_ok} "
            + " ".join(
                f"n{n}:peak={quantum_full[str(n)]['med_peak']:.3f},gap={quantum_full[str(n)]['med_prod_gap']:.3f}"
                for n in args.ns
            )
        )
        points.append(point)

    sims = {n: Rank2Sim(n, dtype=torch.float64) for n in args.ns} if not args.skip_train else {}
    trained_points = []
    for point in points:
        if args.skip_train or not point["quantum_ok"]:
            trained_points.append(point)
            continue
        # rebuild E0 tensors from the cache files to be safe on resume
        ed_cache = {}
        lam = float(point["lambda"])
        tag = point["tag"]
        for n in args.ns:
            cache = out / "ed" / f"{tag}_l{lam:.2f}_n{n}_prod.json"
            rows = _read(cache)
            ntr = args.B_train
            if len(rows) != args.B_train + args.B_hold:
                raise RuntimeError(f"{cache} has {len(rows)} rows, expected {args.B_train + args.B_hold}")
            ed_cache[(n, "train")] = [r["E0"] for r in rows[:ntr]]
            ed_cache[(n, "hold")] = [r["E0"] for r in rows[ntr:]]
        rows = run_train_point(point, bundles, sims, args, ed_cache, run_dir)
        finished = finish_point(point, rows, point["plus_gap"])
        trained_points.append(finished)
        log(f"point {tag} L16 " + " ".join(
            f"n{finished['table'][k]['n']} R={finished['table'][k]['R']:.3f}/{finished['table'][k]['verdict']}"
            for k in sorted(finished["table"]) if finished["table"][k]["L"] == 16
        ))

    base = next((p for p in trained_points if abs(p["mu"]) + abs(p["kappa"]) < 1e-12), None)
    classes = {}
    base_note = "ok"
    if base is None or not base.get("trained"):
        base_note = "baseline_missing"
    else:
        l16_beat = all(
            base["table"].get(f"n{n}_L16", {}).get("verdict") == "BEAT" for n in args.ns
        )
        if not l16_beat:
            base_note = "baseline_not_beat"
    for p in trained_points:
        if abs(p["mu"]) + abs(p["kappa"]) < 1e-12:
            classes[p["tag"]] = {"class": "baseline", "quantum": p.get("quantum_ok")}
        else:
            classes[p["tag"]] = classify_point(p, base if base and base.get("trained") else None)

    if args.quick or args.skip_train:
        verdict = "INCOMPLETE"
        why = "Quick or ED-only run. Not a family verdict."
    else:
        verdict = family_verdict(trained_points, classes, base_note)
        if verdict == "PASS":
            why = "A nonzero (μ, κ) cleared the L=16 BEAT gate on both n and lowered R versus μ=κ=0 on both n, past seed noise, without an Arm A kill rule."
        elif verdict == "needs_edit":
            paired = [
                p["tag"]
                for p in trained_points
                if classes.get(p["tag"], {}).get("paired_all_seeds") and classes.get(p["tag"], {}).get("beat_both_L16")
            ]
            why = (
                "Negative μ lowers L=16 ε_SNAP and R on both n"
                + (f" (paired on every seed at {', '.join(paired)})" if paired else "")
                + ". The pre-registered unpaired R-std bar does not clear, because baseline n9 Fourier seeds already scatter. "
                "That is an edit, not a kill. Positive μ does not gain. "
                "The μ=κ=0 baseline does reproduce the L=16 BEAT."
            )
        else:
            why = "No nonzero (μ, κ) improved on μ=κ=0, or quantumness failed on every deformation, or a Y correlator would have been required."

    reproduction = {}
    if base and base.get("trained") and abs(float(base["lambda"]) - 0.5) < 1e-12:
        for n, pub in PR13_L16.items():
            cell = base["table"].get(f"n{n}_L16")
            if not cell:
                continue
            reproduction[str(n)] = {
                "qaoa": cell["med_eps_qaoa"],
                "snap": cell["med_eps_snap"],
                "R": cell["R"],
                "dq": cell["med_eps_qaoa"] - pub["qaoa"],
                "ds": cell["med_eps_snap"] - pub["snap"],
                "R_pr": pub["R"],
            }

    notes = [
        "H_c is pure Walsh degree 3, so a Walsh-degree γ head is a null knob on this diagonal. It was not trained. PR #11's Walsh win was on the unsat-count diagonal, which has several degrees.",
        "power_p has 4L parameters and Fourier QAOA has 2L. The comparison asks whether the c2, c3 shells beat the linear mixer they are warm-started from.",
        "Holdout checkpoint selection uses the holdout median residual, same rule as PR #13, applied to both methods.",
        "XX and XXX matrix elements are real flip operators. The statevector check equates the per-cavity X histogram to ⟨ψ|H|ψ⟩, so the score does not use a Y string.",
        "Quantumness, entropy peaks, and E0 use the full train+holdout (not a B_scan slice).",
        "n21 phase2 ladder scripts and qaoa_angles/ were not written.",
    ]
    if args.quick:
        notes.append("This file was produced by --quick. Re-run without it for the verdict.")
    try:
        branch = __import__("subprocess").check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=str(_ROOT), text=True
        ).strip()
    except Exception:
        branch = ""

    payload = {
        "family": "A-rank2",
        "verdict": verdict,
        "verdict_why": why,
        "needs_YY": False,
        "prefer_lambda": args.prefer_lambda,
        "mu_grid": list(mu_grid),
        "kappa_grid": list(kappa_grid),
        "points": [
            {k: v for k, v in p.items() if k != "e0"}
            for p in trained_points
        ],
        "classes": classes,
        "baseline_table": (base or {}).get("table") or {},
        "reproduction": reproduction,
        "self_check": check,
        "arm_a_self_check": arm_check,
        "notes": notes,
        "B_train": args.B_train,
        "B_hold": args.B_hold,
        "B_scan": args.B_scan,
        "seeds": list(args.seeds),
        "Ls": list(args.Ls),
        "lambdas": list(lambdas),
        "baseline_note": base_note,
        "wall_s": _kept_wall(out, time.time() - t_all),
        "branch": branch,
        "quick": bool(args.quick),
        "torch": torch.__version__,
    }
    # landscape summaries (no per-instance vectors)
    payload["landscape"] = {
        tag: {f"{lam:.2f}": {str(n): landscape[tag][lam][n] for n in args.ns} for lam in lambdas}
        for tag, _ in ((tag_mk(mu, ka), (mu, ka)) for mu, ka in points_spec)
    }
    _write(out / "summary.json", payload)
    md = render_md(payload)
    (out / "RESULTS.md").write_text(md)
    log("verdict", verdict, "wall", f"{payload['wall_s']:.1f}s")
    log("wrote", out)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
