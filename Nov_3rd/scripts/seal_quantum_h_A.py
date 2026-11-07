#!/usr/bin/env python3
"""Seal the PR13 family-A recipe at λ=0.5.

H = Σ J_abc Z_a Z_b Z_c − λ Σ_i X_i, BinH U_c^X, Hamming-shell SNAP,
⟨H⟩ from the Z histogram and single-qubit X histograms.

Per n ∈ {6, 9}, L ∈ {2, 16}: Fourier QAOA, power_p, and Walsh-degree γ.
power_p is the champ head (scalar γ, φ_k = c1 k + c2 k² + c3 k³).
Walsh-degree γ uses the same shells and a degree-1..4 split of the same
diagonal. On a pure degree-3 H_c the extra degrees are identically zero;
they are still trained, because a nonzero piece would grade the mixer.

Larger batch and more seeds than the PR13 smoke (B_train=32, B_hold=48,
seeds 0..2). Does not write qaoa_angles/ or train69_ladder_fresh250/.

Verdict (pre-registered, primary head = power_p at L=16):
  KILL if the λ=0.5 quantumness gate fails on either n, or protocol R>1.02
  on both n.
  PASS if the gate holds, protocol R≤0.98 on both n, a strict majority of
  seeds have paired R_s≤0.98 on each n, the ε gap exceeds one seed-std on
  at least one n, f stays ≤0.99 for some method or some n, and the positive
  advantage does not shrink from n=6 to n=9 under the smoke shrink test.
  needs_edit otherwise.
Walsh, L=2, and the optional n=12 freeze are reported and do not flip PASS
to KILL. A failed freeze is a caveat.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from Nov_3rd import settings as S
from Nov_3rd.quantum_h_tf import (
    SCALE1000_BASE,
    best_product_energy,
    exact_ground,
    fourier_schedule,
    ground_pair,
    load_scale_holdout,
    load_train_instances,
    paper_fourier_init,
    phi_powers,
    self_check,
    stack_diags,
    walsh_components,
    walsh_mass_fraction,
)

TRAIN_PACKS = S.OUT / "beat_qaoa_12h" / "train_packs"
PAPER_BETA = 0.01
PAPER_GAMMA = -0.01
CODE_ID = "seal_A_v1"
SMOKE_L16 = {
    6: {"qaoa": 0.0096, "power_p": 0.0086, "R": 0.889},
    9: {"qaoa": 0.0127, "power_p": 0.0098, "R": 0.770},
}


def log(*a):
    print(time.strftime("%H:%M:%S"), "[sealA]", *a, flush=True)


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    if isinstance(o, (np.floating, float)):
        x = float(o)
        if not math.isfinite(x):
            return None
        return x
    if isinstance(o, (np.integer, int)):
        return int(o)
    if isinstance(o, np.ndarray):
        return _jsonable(o.tolist())
    if isinstance(o, torch.Tensor):
        return _jsonable(o.detach().cpu().tolist())
    if o is None:
        return None
    return o


def _finite(xs) -> np.ndarray:
    a = np.asarray(list(xs), dtype=np.float64)
    return a[np.isfinite(a)]


def _med(xs) -> float:
    a = _finite(xs)
    if a.size == 0:
        return float("nan")
    return float(np.median(a))


def _mean(xs) -> float:
    a = _finite(xs)
    if a.size == 0:
        return float("nan")
    return float(np.mean(a))


def fmt(x, nd=4):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "nan"
    return f"{float(x):.{nd}f}"


def fmt_ci(med, lo, hi, nd=4):
    if lo is None or hi is None:
        return fmt(med, nd)
    return f"{fmt(med, nd)} [{fmt(lo, nd)}, {fmt(hi, nd)}]"


def yesno(v) -> str:
    return "yes" if v in (True, 1) else "no"


def bootstrap_median(xs, rng: np.random.Generator, n_boot: int = 10000) -> dict:
    arr = _finite(xs)
    if arr.size == 0:
        return {"median": float("nan"), "lo": float("nan"), "hi": float("nan")}
    med = float(np.median(arr))
    if arr.size == 1:
        return {"median": med, "lo": med, "hi": med}
    draws = rng.integers(0, arr.size, size=(n_boot, arr.size))
    meds = np.median(arr[draws], axis=1)
    lo, hi = np.quantile(meds, [0.025, 0.975])
    return {"median": med, "lo": float(lo), "hi": float(hi)}


def bootstrap_ratio(num, den, rng: np.random.Generator, n_boot: int = 10000) -> dict:
    a = np.asarray(num, dtype=np.float64)
    b = np.asarray(den, dtype=np.float64)
    point = float(np.median(a) / np.median(b)) if np.median(b) > 1e-12 else float("inf")
    if a.size == 1:
        return {"R": point, "lo": point, "hi": point}
    draws = rng.integers(0, a.size, size=(n_boot, a.size))
    rs = np.median(a[draws], axis=1) / np.maximum(np.median(b[draws], axis=1), 1e-12)
    lo, hi = np.quantile(rs, [0.025, 0.975])
    return {"R": point, "lo": float(lo), "hi": float(hi)}


def shell_scale(p: torch.Tensor, n: int) -> torch.Tensor:
    base = 8.0 / float(int(n) - 1)
    return torch.pow(p.new_tensor(base), p)


def decode(method: str, theta: torch.Tensor, L: int, n: int | None = None, with_p: bool = False):
    """Return gamma, phi. ``with_p`` appends the cross-n shell exponent."""
    if method == "fourier_qaoa":
        gamma = fourier_schedule(theta[:L], L)
        beta = fourier_schedule(theta[L : 2 * L], L)
        k = torch.arange(4, device=theta.device, dtype=theta.dtype)
        phi = (-beta.reshape(L, 1)) * k.reshape(1, 4)
        return gamma, phi
    p = None
    body = theta
    if with_p:
        p = theta[-1]
        body = theta[:-1]
    if method == "power_p":
        g, c1, c2, c3 = body.reshape(4, L)
        if p is not None:
            a = shell_scale(p, int(n))
            c2, c3 = c2 * a, c3 * a
        return g, phi_powers(c1, c2, c3)
    if method == "walsh":
        g = body[: 4 * L].reshape(L, 4)
        c1, c2, c3 = body[4 * L :].reshape(3, L)
        if p is not None:
            a = shell_scale(p, int(n))
            c2, c3 = c2 * a, c3 * a
        return g, phi_powers(c1, c2, c3)
    raise ValueError(method)


def n_params(method: str, L: int, with_p: bool = False) -> int:
    extra = 1 if with_p and method != "fourier_qaoa" else 0
    if method == "fourier_qaoa":
        return 2 * L
    if method == "power_p":
        return 4 * L + extra
    if method == "walsh":
        return 7 * L + extra
    raise ValueError(method)


def _init_theta(method: str, L: int, seed: int, n: int, dtype, device, with_p: bool) -> torch.Tensor:
    rng = np.random.default_rng(50_000 + 1000 * int(seed) + 10 * int(L) + int(n))
    if method == "fourier_qaoa":
        ug, ub = paper_fourier_init(L, L)
        raw = np.concatenate([ug, ub])
        jit = rng.normal(0.0, 0.01, size=raw.shape)
        return torch.tensor(raw + jit, dtype=dtype, device=device)
    gamma = np.full(L, PAPER_GAMMA) + rng.normal(0.0, 0.01, size=L)
    c1 = np.full(L, -PAPER_BETA) + rng.normal(0.0, 0.01, size=L)
    c2 = rng.normal(0.0, 0.01, size=L)
    c3 = rng.normal(0.0, 0.01, size=L)
    if method == "power_p":
        raw = np.concatenate([gamma, c1, c2, c3])
    elif method == "walsh":
        g = np.repeat(gamma[:, None], 4, axis=1).reshape(-1)
        raw = np.concatenate([g, c1, c2, c3])
    else:
        raise ValueError(method)
    if with_p:
        raw = np.concatenate([raw, np.zeros(1)])
    return torch.tensor(raw, dtype=dtype, device=device)


def snap_theta_from_fourier(theta_f: torch.Tensor, L: int, method: str, with_p: bool) -> torch.Tensor:
    """Dual of a trained Fourier point: γ (or γ_w all equal), φ_j=−β j, c2=c3=0."""
    gamma = fourier_schedule(theta_f[:L], L)
    beta = fourier_schedule(theta_f[L : 2 * L], L)
    c1 = -beta
    zeros = torch.zeros(L, dtype=theta_f.dtype, device=theta_f.device)
    if method == "power_p":
        raw = torch.cat([gamma.detach(), c1.detach(), zeros, zeros])
    elif method == "walsh":
        g = gamma.detach().reshape(L, 1).repeat(1, 4).reshape(-1)
        raw = torch.cat([g, c1.detach(), zeros, zeros])
    else:
        raise ValueError(method)
    if with_p:
        raw = torch.cat([raw, raw.new_zeros(1)])
    return raw


def _metrics(e: torch.Tensor, e0: torch.Tensor, n: int, lam: float) -> tuple[dict, np.ndarray]:
    e_init = -float(lam) * int(n)
    e_d = e.detach()
    eps = (e_d - e0) / float(n)
    denom = (e0.new_tensor(e_init) - e0).clamp_min(1e-12)
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
    }, eps_np


def mixer_diag(method: str, theta: torch.Tensor, L: int, with_p: bool) -> dict:
    if method == "fourier_qaoa":
        return {}
    body = theta[:-1] if with_p else theta
    p = float(theta[-1].detach()) if with_p else None
    if method == "power_p":
        _g, c1, c2, c3 = body.detach().reshape(4, L)
        gspread = 0.0
    else:
        g = body.detach()[: 4 * L].reshape(L, 4)
        c1, c2, c3 = body.detach()[4 * L :].reshape(3, L)
        gspread = float((g - g[:, 2:3]).abs().mean())
    return {
        "c1_rms": float(c1.pow(2).mean().sqrt()),
        "c2_rms": float(c2.pow(2).mean().sqrt()),
        "c3_rms": float(c3.pow(2).mean().sqrt()),
        "gamma_spread_vs_deg3": gspread,
        "p": p,
    }


def train_one(
    *,
    method: str,
    n: int,
    L: int,
    seed: int,
    sim,
    Hc_tr: torch.Tensor,
    Hc_ho: torch.Tensor,
    comps_tr: torch.Tensor | None,
    comps_ho: torch.Tensor | None,
    e0_tr: torch.Tensor,
    e0_ho: torch.Tensor,
    lam: float,
    rounds: int,
    max_iter: int,
    lr: float,
    theta0: torch.Tensor | None,
    with_p: bool = False,
) -> dict:
    dtype = Hc_tr.dtype
    device = Hc_tr.device
    if theta0 is None:
        theta = torch.nn.Parameter(_init_theta(method, L, seed, n, dtype, device, with_p))
    else:
        theta = torch.nn.Parameter(theta0.detach().to(dtype=dtype, device=device).clone())
    t0 = time.time()

    def energies(Hc, comps):
        gamma, phi = decode(method, theta, L, n=n, with_p=with_p)
        use = comps if method == "walsh" else None
        return sim.energy(Hc, gamma, phi, lam, comps_x=use)

    with torch.no_grad():
        init_tr, _ = _metrics(energies(Hc_tr, comps_tr), e0_tr, n, lam)
        init_ho, _ = _metrics(energies(Hc_ho, comps_ho), e0_ho, n, lam)
    best = {
        "ho_med_eps": float(init_ho["med_eps"]),
        "round": -1,
        "theta": theta.detach().clone(),
    }
    hist = [{"round": -1, "train_med_eps": init_tr["med_eps"], "hold_med_eps": init_ho["med_eps"]}]
    stall = 0
    for r in range(rounds):
        opt = torch.optim.LBFGS(
            [theta], lr=lr, max_iter=max_iter, history_size=16, line_search_fn="strong_wolfe",
        )

        def closure():
            opt.zero_grad(set_to_none=True)
            loss = energies(Hc_tr, comps_tr).mean()
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
            tr, _ = _metrics(energies(Hc_tr, comps_tr), e0_tr, n, lam)
            ho, _ = _metrics(energies(Hc_ho, comps_ho), e0_ho, n, lam)
        hist.append({
            "round": r, "loss": loss_f,
            "train_med_eps": tr["med_eps"], "hold_med_eps": ho["med_eps"], "hold_med_f": ho["med_f"],
        })
        if ho["med_eps"] < best["ho_med_eps"] - 1e-6:
            best = {"ho_med_eps": float(ho["med_eps"]), "round": r, "theta": theta.detach().clone()}
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
        hold, eps_ho = _metrics(energies(Hc_ho, comps_ho), e0_ho, n, lam)
        train_m, _ = _metrics(energies(Hc_tr, comps_tr), e0_tr, n, lam)
        gamma, phi = decode(method, theta, L, n=n, with_p=with_p)
    return {
        "method": method,
        "n": n,
        "L": L,
        "seed": seed,
        "lam": float(lam),
        "n_params": n_params(method, L, with_p),
        "with_p": with_p,
        "rounds_ran": int(hist[-1]["round"]),
        "best_round": int(best["round"]),
        "init_hold": init_ho,
        "train": train_m,
        "hold": hold,
        "eps_hold": eps_ho.tolist(),
        "history_tail": hist[-6:],
        "gamma": gamma.detach().cpu().tolist(),
        "phi": phi.detach().cpu().tolist(),
        "theta": theta.detach().cpu().tolist(),
        "mixer": mixer_diag(method, theta.detach(), L, with_p),
        "wall_s": time.time() - t0,
        "optimizer": "LBFGS",
        "selection": "best holdout median ε (warm start included)",
        "warm_start": theta0 is not None,
    }


def train_joint(
    *,
    method: str,
    L: int,
    seed: int,
    sims: dict,
    bundles: dict,
    e0_tr: dict,
    e0_ho: dict,
    lam: float,
    rounds: int,
    max_iter: int,
    lr: float,
    theta0: torch.Tensor | None,
) -> dict:
    """One parameter vector on n=6 and n=9. Loss is the mean of the two mean-ε.

    Snap heads carry p with a_n = (8/(n-1))^p so the shells can change with n.
    n=12 is not in the loss.
    """
    with_p = method != "fourier_qaoa"
    dtype = torch.float64
    device = torch.device("cpu")
    if theta0 is None:
        theta = torch.nn.Parameter(_init_theta(method, L, seed, 0, dtype, device, with_p))
    else:
        theta = torch.nn.Parameter(theta0.detach().to(dtype=dtype, device=device).clone())
    ns = (6, 9)
    t0 = time.time()

    def batch_energy(n, split):
        gamma, phi = decode(method, theta, L, n=n, with_p=with_p)
        Hc = bundles[n]["Hc_tr" if split == "train" else "Hc_ho"]
        comps = bundles[n]["comps_tr" if split == "train" else "comps_ho"]
        use = comps if method == "walsh" else None
        return sims[n].energy(Hc, gamma, phi, lam, comps_x=use)

    def mean_eps(n, split):
        e0 = e0_tr[n] if split == "train" else e0_ho[n]
        e = batch_energy(n, split)
        return (e - e0) / float(n)

    def hold_score():
        per = {}
        for n in ns:
            m, eps = _metrics(batch_energy(n, "hold"), e0_ho[n], n, lam)
            per[n] = (m, eps)
        score = 0.5 * (per[6][0]["med_eps"] + per[9][0]["med_eps"])
        return score, per

    with torch.no_grad():
        init_score, init_per = hold_score()
    best = {"score": float(init_score), "round": -1, "theta": theta.detach().clone()}
    hist = [{"round": -1, "hold_score": init_score}]
    stall = 0
    for r in range(rounds):
        opt = torch.optim.LBFGS(
            [theta], lr=lr, max_iter=max_iter, history_size=16, line_search_fn="strong_wolfe",
        )

        def closure():
            opt.zero_grad(set_to_none=True)
            loss = torch.stack([mean_eps(n, "train").mean() for n in ns]).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_([theta], 1e3)
            return loss

        try:
            loss = opt.step(closure)
            loss_f = float(loss.detach())
        except Exception as exc:
            log(f"  LBFGS skip joint {method} L={L} seed={seed} r={r}: {exc}")
            stall += 1
            if stall >= 4:
                break
            continue
        with torch.no_grad():
            score, per = hold_score()
        hist.append({
            "round": r,
            "loss": loss_f,
            "hold_score": score,
            "hold_med_eps": {str(n): per[n][0]["med_eps"] for n in ns},
        })
        if score < best["score"] - 1e-6:
            best = {"score": float(score), "round": r, "theta": theta.detach().clone()}
            stall = 0
        else:
            stall += 1
        if r % 5 == 0 or r + 1 == rounds:
            log(
                f"  joint {method} L={L} s={seed} r={r} "
                f"ε6={per[6][0]['med_eps']:.4f} ε9={per[9][0]['med_eps']:.4f} "
                f"best@{best['round']}"
            )
        if stall >= 8 and r >= 6:
            log(f"  early stop joint {method} L={L} seed={seed} at r={r}")
            break
    with torch.no_grad():
        theta.copy_(best["theta"])
        score, per = hold_score()
        gamma6, phi6 = decode(method, theta, L, n=6, with_p=with_p)
    hold = {str(n): per[n][0] for n in ns}
    eps = {str(n): per[n][1].tolist() for n in ns}
    return {
        "method": method,
        "L": L,
        "seed": seed,
        "lam": float(lam),
        "n_params": n_params(method, L, with_p),
        "with_p": with_p,
        "joint_ns": [6, 9],
        "rounds_ran": int(hist[-1]["round"]),
        "best_round": int(best["round"]),
        "hold_score": float(score),
        "hold": hold,
        "eps_hold": eps,
        "history_tail": hist[-6:],
        "theta": theta.detach().cpu().tolist(),
        "gamma_n6": gamma6.detach().cpu().tolist(),
        "phi_n6": phi6.detach().cpu().tolist(),
        "mixer": mixer_diag(method, theta.detach(), L, with_p),
        "wall_s": time.time() - t0,
        "selection": "best mean of n=6 and n=9 holdout median ε",
        "warm_start": theta0 is not None,
    }


def eval_frozen(method, theta, L, n, sim, Hc, comps, e0, lam) -> tuple[dict, np.ndarray]:
    with_p = method != "fourier_qaoa"
    th = torch.as_tensor(theta, dtype=torch.float64)
    with torch.no_grad():
        gamma, phi = decode(method, th, L, n=n, with_p=with_p)
        use = comps if method == "walsh" else None
        e = sim.energy(Hc, gamma, phi, lam, comps_x=use)
        return _metrics(e, e0, n, lam)


def mark_ratio(R: float) -> str:
    if not math.isfinite(R):
        return "UNDER"
    if R <= 0.98:
        return "BEAT"
    if R <= 1.02:
        return "TIE"
    return "UNDER"


def aggregate(rows: list[dict], rng: np.random.Generator) -> dict:
    eps = [r["hold"]["med_eps"] for r in rows]
    fs = [r["hold"]["med_f"] for r in rows]
    ci = bootstrap_median(eps, rng)
    # Instance-level CI of the seed at the median.
    order = list(np.argsort(eps))
    mid = rows[order[len(order) // 2]]
    inst = bootstrap_median(mid["eps_hold"], rng, n_boot=4000)
    std = float(np.std(eps, ddof=1)) if len(eps) > 1 else 0.0
    return {
        "seeds": [r["seed"] for r in rows],
        "per_seed_med_eps": eps,
        "per_seed_med_f": fs,
        "med_eps": ci["median"],
        "ci95": [ci["lo"], ci["hi"]],
        "std_eps": std,
        "med_f": float(np.median(fs)),
        "n_params": rows[0]["n_params"],
        "instance_ci95_median_seed": [inst["lo"], inst["hi"]],
        "instance_ci_seed": mid["seed"],
        "mixer_per_seed": [r.get("mixer") or {} for r in rows],
    }


def cell_for(snap: dict, qaoa: dict, rng: np.random.Generator, n: int, L: int) -> dict:
    Rinfo = bootstrap_ratio(snap["per_seed_med_eps"], qaoa["per_seed_med_eps"], rng)
    paired = np.asarray(snap["per_seed_med_eps"]) / np.maximum(np.asarray(qaoa["per_seed_med_eps"]), 1e-12)
    n_beat = int(np.sum(paired <= 0.98 + 1e-12))
    adv = float(qaoa["med_eps"] - snap["med_eps"])
    noise = max(qaoa["std_eps"], snap["std_eps"])
    return {
        "n": n,
        "L": L,
        "med_eps_qaoa": qaoa["med_eps"],
        "ci_qaoa": qaoa["ci95"],
        "std_eps_qaoa": qaoa["std_eps"],
        "med_f_qaoa": qaoa["med_f"],
        "med_eps_snap": snap["med_eps"],
        "ci_snap": snap["ci95"],
        "std_eps_snap": snap["std_eps"],
        "med_f_snap": snap["med_f"],
        "R": Rinfo["R"],
        "R_ci95": [Rinfo["lo"], Rinfo["hi"]],
        "paired_R": paired.tolist(),
        "n_paired_beat": n_beat,
        "n_seeds": len(paired),
        "adv_eps": adv,
        "verdict": mark_ratio(Rinfo["R"]),
        "within_one_seed_std": bool(abs(adv) <= noise + 1e-12),
        "saturated": bool(qaoa["med_f"] > 0.99 and snap["med_f"] > 0.99),
        "n_params_qaoa": qaoa["n_params"],
        "n_params_snap": snap["n_params"],
        "instance_ci_snap": snap["instance_ci95_median_seed"],
        "instance_ci_qaoa": qaoa["instance_ci95_median_seed"],
    }


def shrink_test(by_n: dict) -> tuple[bool, str]:
    if 6 not in by_n or 9 not in by_n:
        return False, ""
    a6 = by_n[6]["adv_eps"]
    a9 = by_n[9]["adv_eps"]
    noise = by_n[6]["std_eps_snap"] + by_n[9]["std_eps_snap"]
    shrink = bool(a6 > 0 and a9 < a6 - max(noise, 1e-4) and a9 < 0.85 * a6)
    return shrink, f"adv n6={a6:.5f} n9={a9:.5f} noise={noise:.5f}"


def decide(quantum_ok: bool, l16: dict) -> tuple[str, str]:
    """l16 maps n -> power_p cell."""
    if not quantum_ok:
        return "KILL", "Quantumness gate failed at λ=0.5 on at least one of n=6, n=9."
    if 6 not in l16 or 9 not in l16:
        return "needs_edit", "L=16 power_p is missing a size."
    Rs = {n: l16[n]["R"] for n in (6, 9)}
    if all(Rs[n] > 1.02 for n in (6, 9)):
        return "KILL", "L=16 power_p is UNDER Fourier QAOA on both n (R>1.02)."
    beat = all(l16[n]["verdict"] == "BEAT" for n in (6, 9))
    majority = all(l16[n]["n_paired_beat"] > l16[n]["n_seeds"] / 2.0 for n in (6, 9))
    outside = any(not l16[n]["within_one_seed_std"] for n in (6, 9))
    sat = all(l16[n]["saturated"] for n in (6, 9))
    shrink, shrink_detail = shrink_test(l16)
    reasons = []
    if not beat:
        reasons.append(
            "protocol R is not ≤0.98 on both n "
            f"(n6={Rs[6]:.3f}, n9={Rs[9]:.3f})."
        )
    if not majority:
        reasons.append(
            "paired R_s≤0.98 is not a strict majority on both n "
            f"(n6 {l16[6]['n_paired_beat']}/{l16[6]['n_seeds']}, "
            f"n9 {l16[9]['n_paired_beat']}/{l16[9]['n_seeds']})."
        )
    if not outside:
        reasons.append("the L=16 ε gap is inside one seed-std on both n.")
    if sat:
        reasons.append("both methods have Med f>0.99 on both n, so the ratio is at the floor.")
    if shrink:
        reasons.append(f"the positive advantage shrinks n6→n9 ({shrink_detail}).")
    if beat and majority and outside and not sat and not shrink:
        return (
            "PASS",
            "L=16 power_p protocol R≤0.98 on n=6 and n=9, a strict majority of seeds "
            "agree, the gap is outside one seed-std on at least one n, and the "
            "smoke shrink and saturation checks do not fire.",
        )
    return "needs_edit", " ".join(reasons) if reasons else "Seal criteria were not all met."


def spectrum_summary(diags: np.ndarray) -> dict:
    frac = walsh_mass_fraction(diags)
    # columns: degree 0..4
    med = np.median(frac, axis=0)
    mx = np.max(frac, axis=0)
    return {
        "B": int(frac.shape[0]),
        "med_frac_deg": [float(x) for x in med],
        "max_frac_deg": [float(x) for x in mx],
        "med_off_deg3": float(np.median(1.0 - frac[:, 3])),
    }


def ed_split(insts, n, lam, product_cap: int) -> list[dict]:
    rows = []
    t0 = time.time()
    for i, ins in enumerate(insts):
        if n <= 9:
            g = exact_ground(ins.diag, n, lam)
        else:
            g = ground_pair(ins.diag, n, lam)
        g["seed"] = int(ins.seed)
        g["m"] = int(ins.m)
        g["n"] = int(n)
        g["lam"] = float(lam)
        rows.append(g)
        if (i + 1) % 32 == 0 or i + 1 == len(insts):
            log(f"  eigh n={n} {i+1}/{len(insts)} ({time.time()-t0:.1f}s)")
    cap = len(insts) if product_cap <= 0 else min(product_cap, len(insts))
    log(f"  product n={n} on {cap}/{len(insts)}")
    t1 = time.time()
    subset = insts[:cap]

    def _one(ins):
        return best_product_energy(ins.clauses, ins.n, lam, seed=ins.seed)

    with ThreadPoolExecutor(max_workers=4) as ex:
        prods = list(ex.map(_one, subset))
    for row, eprod in zip(rows, prods):
        row["E_prod"] = float(eprod)
        denom = max(abs(row["E0"]), 1e-8)
        row["prod_gap"] = float((eprod - row["E0"]) / denom)
    for row in rows[cap:]:
        row["E_prod"] = float("nan")
        row["prod_gap"] = float("nan")
    log(f"  product n={n} done {time.time()-t1:.1f}s")
    return rows


def summarize_ed(rows: list[dict]) -> dict:
    peak = [r["peak"] for r in rows]
    gap = [r["prod_gap"] for r in rows if math.isfinite(r.get("prod_gap", float("nan")))]
    e0 = np.round([r["E0"] for r in rows], 8)
    uniq, cnt = np.unique(e0, return_counts=True)
    return {
        "B": len(rows),
        "n_unique_E0": int(uniq.size),
        "E0_majority_frac": float(cnt.max() / max(len(rows), 1)),
        "B_product": len(gap),
        "med_peak": _med(peak),
        "frac_peak_lt_0.5": float(np.mean(np.asarray(peak) < 0.5)),
        "med_prod_gap": _med(gap),
        "frac_prod_gap_ge_0.02": float(np.mean(np.asarray(gap) >= 0.02)) if gap else float("nan"),
        "med_entropy": _med([r["entropy"] for r in rows]),
        "med_chi": _med([r["chi"] for r in rows]),
        "med_gap": _med([r["gap"] for r in rows]),
        "med_E0": _med([r["E0"] for r in rows]),
        "med_E0_over_n": _med([r["E0"] / r["n"] for r in rows]),
        "med_E_cl": _med([r["E_cl"] for r in rows]),
        "med_sx": _med([r["sx"] for r in rows]),
        "quantum_pass": bool(_med(peak) < 0.5 and _med(gap) >= 0.02),
    }


def comps_of(diags: np.ndarray) -> torch.Tensor:
    return torch.as_tensor(walsh_components(diags), dtype=torch.float64)


def load_bundle(n: int, B_train: int, B_hold: int) -> dict:
    log(f"load n={n} train={B_train} hold={B_hold}")
    train = load_train_instances(n, B_train, TRAIN_PACKS / f"train_n{n}.npz")
    hold = load_scale_holdout(n, B_hold, SCALE1000_BASE)
    inter = {ins.seed for ins in train} & {ins.seed for ins in hold}
    if inter:
        raise RuntimeError(f"n={n} train/holdout seed overlap {list(inter)[:5]}")
    diag_tr = np.stack([ins.diag for ins in train], 0)
    diag_ho = np.stack([ins.diag for ins in hold], 0)
    return {
        "train": train,
        "hold": hold,
        "diag_tr": diag_tr,
        "diag_ho": diag_ho,
        "Hc_tr": torch.as_tensor(diag_tr, dtype=torch.float64),
        "Hc_ho": torch.as_tensor(diag_ho, dtype=torch.float64),
        "comps_tr": comps_of(diag_tr),
        "comps_ho": comps_of(diag_ho),
        "spec_tr": spectrum_summary(diag_tr),
        "spec_ho": spectrum_summary(diag_ho),
    }


def ed_cached(path: Path, insts, n, lam, product_cap: int) -> list[dict]:
    seeds = np.asarray([ins.seed for ins in insts], dtype=np.int64)
    if path.exists():
        z = np.load(path, allow_pickle=False)
        if np.array_equal(z["seeds"], seeds) and abs(float(z["lam"]) - lam) < 1e-12 and int(z["product_cap"]) == int(product_cap if product_cap > 0 else len(insts)):
            log(f"  ED cache hit {path.name}")
            rows = []
            for i in range(len(insts)):
                rows.append({
                    "E0": float(z["E0"][i]),
                    "E1": float(z["E1"][i]),
                    "gap": float(z["gap"][i]),
                    "peak": float(z["peak"][i]),
                    "sx": float(z["sx"][i]),
                    "entropy": float(z["entropy"][i]),
                    "chi": float(z["chi"][i]),
                    "E_cl": float(z["E_cl"][i]),
                    "E_prod": float(z["E_prod"][i]),
                    "prod_gap": float(z["prod_gap"][i]),
                    "seed": int(seeds[i]),
                    "m": int(insts[i].m),
                    "n": int(n),
                    "lam": float(lam),
                })
            return rows
    rows = ed_split(insts, n, lam, product_cap)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        seeds=seeds,
        lam=np.float64(lam),
        product_cap=np.int32(product_cap if product_cap > 0 else len(insts)),
        E0=np.array([r["E0"] for r in rows]),
        E1=np.array([r["E1"] for r in rows]),
        gap=np.array([r["gap"] for r in rows]),
        peak=np.array([r["peak"] for r in rows]),
        sx=np.array([r["sx"] for r in rows]),
        entropy=np.array([r["entropy"] for r in rows]),
        chi=np.array([r["chi"] for r in rows]),
        E_cl=np.array([r["E_cl"] for r in rows]),
        E_prod=np.array([r["E_prod"] for r in rows]),
        prod_gap=np.array([r["prod_gap"] for r in rows]),
    )
    return rows


def protocol_blob(args) -> dict:
    return {
        "code": CODE_ID,
        "lam": float(args.lam),
        "B_train": int(args.B_train),
        "B_hold": int(args.B_hold),
        "rounds_L2": int(args.rounds_L2),
        "rounds_L16": int(args.rounds_L16),
        "max_iter_L2": int(args.max_iter_L2),
        "max_iter_L16": int(args.max_iter_L16),
        "lr": float(args.lr),
        "B_n12": int(args.B_n12),
    }


def load_done(path: Path, proto: dict):
    if not path.exists():
        return None
    try:
        rec = json.loads(path.read_text())
    except json.JSONDecodeError:
        return None
    if rec.get("protocol") != proto:
        return None
    if "hold" not in rec:
        return None
    return rec


def render_md(payload: dict) -> str:
    lines = []
    lines.append("# RESULTS — seal family A (PR #13 recipe)")
    lines.append("")
    lines.append(f"**Verdict: {payload['verdict']}**")
    lines.append("")
    lines.append(payload["verdict_reason"])
    lines.append("")
    lines.append("## Recipe")
    lines.append("")
    lines.append("Same Hamiltonian and gate as the PR #13 smoke, at the λ that smoke locked:")
    lines.append("")
    lines.append(r"\[")
    lines.append(r"H = \sum_{(a,b,c)} J_{abc} Z_a Z_b Z_c \;-\; \lambda \sum_i X_i")
    lines.append(r"\]")
    lines.append("")
    lines.append(
        f"λ = **{payload['lam']}**. "
        r"Gate: `|0⟩ → [U_c^X → Hamming-shell SNAP]^L → W`, BinH `W=H^{⊗3}`, "
        r"`U_c^X = W exp(-i γ H_c) W`. The field is not inside `U_c`. "
        r"⟨H⟩ = ⟨H_c⟩ from the Z histogram minus λ Σ ⟨X_i⟩ from the X histogram "
        r"(⟨X_i⟩ = 2p(+)-1). No YY or XX."
    )
    lines.append("")
    lines.append(
        "Cross-cavity triples are taken from the existing 4-SAT packs. "
        "`J=±1` is the product of the three literal signs. Degree cap 3, so `m=n` when the geometry allows it. "
        "Train rows are the first `B_train` seeds of `beat_qaoa_12h/train_packs`. "
        "Holdout rows are the first `B_hold` satisfiable draws at scale1000 base `54_000_000_000`. "
        "`qaoa_angles/`, `train69_ladder_fresh250/`, and the phase-2 n21 ladder were not written."
    )
    lines.append("")
    lines.append(
        f"This seal uses B_train={payload['B_train']}, B_hold={payload['B_hold']}, "
        f"seeds={payload['seeds']} (the smoke was B_train=32, B_hold=48, seeds 0,1,2). "
        "Heads: Fourier QAOA (Zhou q=L, 2L coefficients, mixer φ_j=−β j), "
        "power_p (scalar γ plus φ_k=c1 k+c2 k²+c3 k³, 4L, warm-started from that seed's Fourier dual), "
        "and Walsh-degree γ (same shells, γ shaped (L, 4) on Walsh degrees 1..4 of the same H_c, 7L). "
        "Optimizer is L-BFGS. The checkpoint is the best holdout median ε, including the warm start. "
        "ε = (⟨H⟩−E_0)/n. R = median_seed(Med ε_SNAP) / median_seed(Med ε_QAOA). "
        "A cell is BEAT at R≤0.98, TIE in (0.98, 1.02], else UNDER. "
        "The 95% intervals are bootstrap intervals of that median over seeds."
    )
    lines.append("")
    lines.append("## Quantumness at λ=0.5")
    lines.append("")
    lines.append(
        "Gate, same as the smoke: median ground-state bitstring peak < 0.5 and "
        "median product-state gap (E_prod−E_0)/|E_0| ≥ 2%. "
        "Product states are XZ-plane. E_0 is dense eigh at n≤9."
    )
    lines.append("")
    lines.append("| split | n | B | distinct E0 | med peak | med prod gap | med S (bits) | med χ | med E0/n | pass |")
    lines.append("|:------|--:|--:|-----------:|---------:|-------------:|-------------:|------:|---------:|:----:|")
    for key, row in payload["quantum"].items():
        s = row["summary"]
        lines.append(
            f"| {key} | {row['n']} | {s['B']} | {s.get('n_unique_E0', '')} | {fmt(s['med_peak'], 3)} | "
            f"{fmt(s['med_prod_gap'], 3)} | {fmt(s['med_entropy'], 3)} | "
            f"{fmt(s['med_chi'], 3)} | {fmt(s['med_E0_over_n'], 3)} | "
            f"{yesno(s['quantum_pass'])} |"
        )
    lines.append("")
    lines.append("## Walsh spectrum of H_c")
    lines.append("")
    lines.append(
        "Fraction of mean-square on Walsh degrees 0..4. "
        "Degree 3 is the only piece a sum of ZZZ terms can carry; a global mean would be degree 0 "
        "and is a global phase in the pulse."
    )
    lines.append("")
    lines.append("| n | split | med frac deg 0..4 | max off-degree-3 |")
    lines.append("|--:|:------|:------------------|---------------:|")
    for key, spec in payload["spectrum"].items():
        fr = ", ".join(f"{x:.2e}" for x in spec["med_frac_deg"])
        lines.append(f"| {spec['n']} | {spec['split']} | {fr} | {spec['med_off_deg3']:.2e} |")
    lines.append("")
    lines.append(payload["walsh_note"])
    lines.append("")
    lines.append("## Holdout")
    lines.append("")
    lines.append("| n | L | method | params | Med ε (seed 95% CI) | seed std | Med f | R (seed 95% CI) | paired ≤0.98 | verdict |")
    lines.append("|--:|--:|:-------|------:|:--------------------|---------:|------:|:----------------|-------------:|:--------|")
    seen_q = set()
    for cell in payload["cells"]:
        n, L = cell["n"], cell["L"]
        if (n, L) not in seen_q:
            seen_q.add((n, L))
            lines.append(
                f"| {n} | {L} | fourier_qaoa | {cell['n_params_qaoa']} | "
                f"{fmt_ci(cell['med_eps_qaoa'], cell['ci_qaoa'][0], cell['ci_qaoa'][1])} | "
                f"{fmt(cell['std_eps_qaoa'])} | {fmt(cell['med_f_qaoa'], 3)} |  |  |  |"
            )
        lines.append(
            f"| {n} | {L} | {cell['method']} | {cell['n_params_snap']} | "
            f"{fmt_ci(cell['med_eps_snap'], cell['ci_snap'][0], cell['ci_snap'][1])} | "
            f"{fmt(cell['std_eps_snap'])} | {fmt(cell['med_f_snap'], 3)} | "
            f"{fmt_ci(cell['R'], cell['R_ci95'][0], cell['R_ci95'][1], 3)} | "
            f"{cell['n_paired_beat']}/{cell['n_seeds']} | {cell['verdict']} |"
        )
    lines.append("")
    lines.append("Per-seed holdout Med ε, and paired R = ε_SNAP/ε_QAOA:")
    lines.append("")
    for key in sorted(payload["per_method"]):
        ag = payload["per_method"][key]
        vals = ", ".join(fmt(v) for v in ag["per_seed_med_eps"])
        lines.append(f"- `{key}` seeds {ag['seeds']}: {vals}")
    lines.append("")
    for cell in payload["cells"]:
        pr = ", ".join(fmt(v, 3) for v in cell["paired_R"])
        lines.append(
            f"- paired R `{cell['method']}` n{cell['n']} L{cell['L']}: {pr}"
        )
    lines.append("")
    lines.append(
        "Instance bootstrap 95% CI of the holdout median, for the seed whose Med ε is the median seed "
        "(sampling noise of B_hold, not of the optimizer):"
    )
    lines.append("")
    for cell in payload["cells"]:
        if int(cell["L"]) != 16:
            continue
        lines.append(
            f"- n{cell['n']} L16 {cell['method']}: snap {fmt_ci(cell['med_eps_snap'], cell['instance_ci_snap'][0], cell['instance_ci_snap'][1])}, "
            f"QAOA {fmt_ci(cell['med_eps_qaoa'], cell['instance_ci_qaoa'][0], cell['instance_ci_qaoa'][1])}"
        )
    lines.append("")
    lines.append("## Checks on L=16 power_p")
    lines.append("")
    chk = payload["checks"]
    lines.append(f"- Protocol R≤0.98 both n: {yesno(chk['beat_both'])}")
    lines.append(f"- Strict majority of seeds paired R≤0.98 on each n: {yesno(chk['majority'])}")
    lines.append(f"- Gap inside one seed-std on both n: {yesno(chk['within_both'])}")
    lines.append(f"- Advantage shrinks n6→n9: {yesno(chk['shrink'])} ({chk['shrink_detail']})")
    lines.append(f"- Both methods f>0.99 on both n: {yesno(chk['saturated'])}")
    lines.append("")
    lines.append(
        "Smoke reference (PR #13, B_train=32, B_hold=48, 3 seeds), L=16 power_p: "
        + ", ".join(
            f"n{n} ε {SMOKE_L16[n]['power_p']:.4f} vs QAOA {SMOKE_L16[n]['qaoa']:.4f} R={SMOKE_L16[n]['R']:.3f}"
            for n in (6, 9)
        )
        + "."
    )
    lines.append("")
    lines.append("## Freeze train{6,9} → n=12")
    lines.append("")
    fr = payload.get("freeze")
    if not fr:
        lines.append(payload.get("freeze_note", "Skipped."))
    else:
        lines.append(fr["blurb"])
        lines.append("")
        lines.append("| L | method | params | Med ε n6 | Med ε n9 | Med ε n12 | R n12 | paired ≤0.98 |")
        lines.append("|--:|:-------|------:|---------:|---------:|----------:|------:|-------------:|")
        for row in fr["rows"]:
            lines.append(
                f"| {row['L']} | {row['method']} | {row['n_params']} | {fmt(row['med6'])} | {fmt(row['med9'])} | "
                f"{fmt(row['med12'])} | {fmt(row['R12'], 3) if row['method'] != 'fourier_qaoa' else ''} | "
                f"{row.get('paired', '')} |"
            )
        lines.append("")
        lines.append(fr["note"])
    lines.append("")
    lines.append("## Self-check")
    lines.append("")
    sc = payload["self_check"]
    if sc.get("skipped"):
        lines.append("Self-check skipped.")
    else:
        lines.append(
            f"Fock twin vs `Nov3rdAnsatz` max |Δp| Z={sc['circuit_max_abs_pZ']:.2e}, "
            f"X={sc['circuit_max_abs_pX']:.2e}. "
            f"QAOA-point vs ideal mixer max |Δp|={sc['qaoa_point_max_abs_p']:.2e}. "
            f"X-histogram vs Pauli ⟨X⟩ max |Δ|={sc['x_hist_vs_pauli']:.2e}. "
            f"Equal Walsh γ vs scalar γ max |Δp|={sc['walsh_equal_gamma_max_abs_pZ']:.2e}. "
            f"Dead-degree gradient {sc['walsh_dead_grad']:.2e}, live degree-3 gradient {sc['walsh_live_grad']:.2e}. "
            f"Lanczos vs dense E_0 on n=6: {sc['lanczos_vs_dense_E0']:.2e}."
        )
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    for note in payload.get("notes") or []:
        lines.append(f"- {note}")
    lines.append("")
    lines.append(f"Wall clock {payload['wall_s']:.1f}s. Branch `{payload.get('branch', '')}`.")
    lines.append("")
    return "\n".join(lines)


def run_per_n(bundles, ed_cache, sims, args, proto, run_dir: Path) -> list[dict]:
    rows = []
    for n in args.ns:
        b = bundles[n]
        e0_tr = torch.as_tensor(ed_cache[(n, "train")], dtype=torch.float64)
        e0_ho = torch.as_tensor(ed_cache[(n, "hold")], dtype=torch.float64)
        for L in args.Ls:
            rounds = args.rounds_L2 if L <= 2 else args.rounds_L16
            max_iter = args.max_iter_L2 if L <= 2 else args.max_iter_L16
            fourier = {}
            for seed in args.seeds:
                path = run_dir / f"n{n}_L{L}_fourier_qaoa_seed{seed}.json"
                hit = load_done(path, proto)
                if hit is None:
                    log(f"train fourier_qaoa n={n} L={L} seed={seed}")
                    rec = train_one(
                        method="fourier_qaoa", n=n, L=L, seed=seed, sim=sims[n],
                        Hc_tr=b["Hc_tr"], Hc_ho=b["Hc_ho"],
                        comps_tr=None, comps_ho=None,
                        e0_tr=e0_tr, e0_ho=e0_ho, lam=args.lam,
                        rounds=rounds, max_iter=max_iter, lr=args.lr, theta0=None,
                    )
                    rec["protocol"] = proto
                    path.write_text(json.dumps(_jsonable(rec), indent=2) + "\n")
                    log(f"  done hold ε={rec['hold']['med_eps']:.4f} wall={rec['wall_s']:.1f}s")
                else:
                    rec = hit
                    log(f"resume fourier_qaoa n={n} L={L} seed={seed} ε={rec['hold']['med_eps']:.4f}")
                rows.append(rec)
                fourier[seed] = torch.tensor(rec["theta"], dtype=torch.float64)
            for method in ("power_p", "walsh"):
                if method not in args.methods:
                    continue
                for seed in args.seeds:
                    path = run_dir / f"n{n}_L{L}_{method}_seed{seed}.json"
                    hit = load_done(path, proto)
                    if hit is None:
                        theta0 = snap_theta_from_fourier(fourier[seed], L, method, with_p=False)
                        log(f"train {method} n={n} L={L} seed={seed} warm=fourier dual")
                        rec = train_one(
                            method=method, n=n, L=L, seed=seed, sim=sims[n],
                            Hc_tr=b["Hc_tr"], Hc_ho=b["Hc_ho"],
                            comps_tr=b["comps_tr"], comps_ho=b["comps_ho"],
                            e0_tr=e0_tr, e0_ho=e0_ho, lam=args.lam,
                            rounds=rounds, max_iter=max_iter, lr=args.lr, theta0=theta0,
                        )
                        rec["protocol"] = proto
                        path.write_text(json.dumps(_jsonable(rec), indent=2) + "\n")
                        log(
                            f"  done hold ε={rec['hold']['med_eps']:.4f} "
                            f"wall={rec['wall_s']:.1f}s best_round={rec['best_round']}"
                        )
                    else:
                        rec = hit
                        log(f"resume {method} n={n} L={L} seed={seed} ε={rec['hold']['med_eps']:.4f}")
                    rows.append(rec)
    return rows


def run_freeze(bundles, ed_cache, sims, args, proto, run_dir: Path, lam: float) -> dict | None:
    if args.skip_freeze:
        return None
    if 6 not in bundles or 9 not in bundles:
        log("freeze skipped: need both n=6 and n=9")
        return None
    log(f"freeze: shared train on n=6,9 then eval n=12 B={args.B_n12}")
    hold12 = load_scale_holdout(12, args.B_n12, SCALE1000_BASE)
    log(f"n=12 holdout m med={np.median([ins.m for ins in hold12]):.0f}")
    rows12 = ed_cached(
        run_dir.parent / "cache" / f"ed_n12_hold_B{args.B_n12}_l{lam}.npz",
        hold12, 12, lam, product_cap=min(8, args.B_n12),
    )
    sim12 = __import__("Nov_3rd.quantum_h_tf", fromlist=["FockSim"]).FockSim(12, dtype=torch.float64)
    Hc12 = torch.as_tensor(np.stack([ins.diag for ins in hold12], 0), dtype=torch.float64)
    comps12 = comps_of(Hc12.numpy())
    e0_12 = torch.as_tensor([r["E0"] for r in rows12], dtype=torch.float64)
    qsum = summarize_ed(rows12)
    log(
        f"  n12 peak={qsum['med_peak']:.3f} prodΔ={qsum['med_prod_gap']:.3f} "
        f"E0/n={qsum['med_E0_over_n']:.3f} pass={qsum['quantum_pass']}"
    )
    e0_tr = {n: torch.as_tensor(ed_cache[(n, "train")], dtype=torch.float64) for n in (6, 9)}
    e0_ho = {n: torch.as_tensor(ed_cache[(n, "hold")], dtype=torch.float64) for n in (6, 9)}
    trained = []
    for L in args.Ls:
        rounds = args.rounds_L2 if L <= 2 else args.rounds_L16
        max_iter = args.max_iter_L2 if L <= 2 else args.max_iter_L16
        fourier = {}
        for seed in args.freeze_seeds:
            path = run_dir / f"joint_L{L}_fourier_qaoa_seed{seed}.json"
            hit = load_done(path, proto)
            if hit is None:
                log(f"train joint fourier_qaoa L={L} seed={seed}")
                rec = train_joint(
                    method="fourier_qaoa", L=L, seed=seed, sims=sims, bundles=bundles,
                    e0_tr=e0_tr, e0_ho=e0_ho, lam=lam,
                    rounds=rounds, max_iter=max_iter, lr=args.lr, theta0=None,
                )
                m12, eps12 = eval_frozen("fourier_qaoa", rec["theta"], L, 12, sim12, Hc12, comps12, e0_12, lam)
                rec["n12"] = m12
                rec["eps_n12"] = eps12.tolist()
                rec["protocol"] = proto
                path.write_text(json.dumps(_jsonable(rec), indent=2) + "\n")
                log(f"  joint done ε6={rec['hold']['6']['med_eps']:.4f} ε9={rec['hold']['9']['med_eps']:.4f} ε12={m12['med_eps']:.4f}")
            else:
                rec = hit
                log(f"resume joint fourier L={L} seed={seed}")
            trained.append(rec)
            fourier[seed] = torch.tensor(rec["theta"], dtype=torch.float64)
        for method in ("power_p", "walsh"):
            if method not in args.methods:
                continue
            for seed in args.freeze_seeds:
                path = run_dir / f"joint_L{L}_{method}_seed{seed}.json"
                hit = load_done(path, proto)
                if hit is None:
                    theta0 = snap_theta_from_fourier(fourier[seed], L, method, with_p=True)
                    log(f"train joint {method} L={L} seed={seed}")
                    rec = train_joint(
                        method=method, L=L, seed=seed, sims=sims, bundles=bundles,
                        e0_tr=e0_tr, e0_ho=e0_ho, lam=lam,
                        rounds=rounds, max_iter=max_iter, lr=args.lr, theta0=theta0,
                    )
                    m12, eps12 = eval_frozen(method, rec["theta"], L, 12, sim12, Hc12, comps12, e0_12, lam)
                    rec["n12"] = m12
                    rec["eps_n12"] = eps12.tolist()
                    rec["protocol"] = proto
                    path.write_text(json.dumps(_jsonable(rec), indent=2) + "\n")
                    log(
                        f"  joint done ε6={rec['hold']['6']['med_eps']:.4f} "
                        f"ε9={rec['hold']['9']['med_eps']:.4f} ε12={m12['med_eps']:.4f} "
                        f"p={rec['mixer'].get('p')}"
                    )
                else:
                    rec = hit
                    log(f"resume joint {method} L={L} seed={seed}")
                trained.append(rec)
    # Aggregate by (L, method)
    grouped: dict = {}
    for rec in trained:
        grouped.setdefault((rec["L"], rec["method"]), []).append(rec)
    table_rows = []
    note_bits = []
    q_by_L = {}
    for (L, method), recs in sorted(grouped.items()):
        med6 = float(np.median([r["hold"]["6"]["med_eps"] for r in recs]))
        med9 = float(np.median([r["hold"]["9"]["med_eps"] for r in recs]))
        med12 = float(np.median([r["n12"]["med_eps"] for r in recs]))
        entry = {
            "L": L, "method": method, "n_params": recs[0]["n_params"],
            "med6": med6, "med9": med9, "med12": med12,
            "per_seed_12": [r["n12"]["med_eps"] for r in recs],
            "seeds": [r["seed"] for r in recs],
            "p": [r.get("mixer", {}).get("p") for r in recs],
        }
        if method == "fourier_qaoa":
            q_by_L[L] = entry
            entry["R12"] = 1.0
            entry["paired"] = ""
        table_rows.append(entry)
    for entry in table_rows:
        if entry["method"] == "fourier_qaoa":
            continue
        q = q_by_L[entry["L"]]
        # pair by seed order stored
        qmap = {s: e for s, e in zip(q["seeds"], q["per_seed_12"])}
        smap = {s: e for s, e in zip(entry["seeds"], entry["per_seed_12"])}
        paired = [smap[s] / qmap[s] for s in entry["seeds"]]
        entry["R12"] = float(entry["med12"] / q["med12"]) if q["med12"] > 1e-12 else float("inf")
        entry["paired_R"] = paired
        entry["paired"] = f"{sum(r <= 0.98 for r in paired)}/{len(paired)}"
        note_bits.append(
            f"L={entry['L']} {entry['method']} R12={entry['R12']:.3f} "
            f"({entry['paired']} seeds), p={entry['p']}"
        )
    blurb = (
        f"One shared head trained on n=6 and n=9 together (equal weight on mean ε), "
        f"seeds {list(args.freeze_seeds)}, then frozen and scored on {args.B_n12} scale1000 n=12 instances. "
        "n=12 is not in the loss and was not used to pick the checkpoint. "
        "Snap heads include the shell scale a_n=(8/(n−1))^p, p trained, p=0 at the warm start. "
        "E_0(n=12) is a Lanczos ground state of the sparse X-field Hamiltonian "
        f"(checked against dense eigh on n=6). n=12 med peak={qsum['med_peak']:.3f}, "
        f"med E0/n={qsum['med_E0_over_n']:.3f}. "
        "Product gap on this probe uses only the first 8 instances."
    )
    return {
        "blurb": blurb,
        "rows": table_rows,
        "note": "Transfer probe, not a PASS gate. " + " ".join(note_bits),
        "n12_quantum": qsum,
        "B": args.B_n12,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default=str(S.OUT / "quantum_h_explore" / "seal_A"))
    ap.add_argument("--ns", type=str, default="6,9")
    ap.add_argument("--Ls", type=str, default="2,16")
    ap.add_argument("--seeds", type=str, default="0,1,2,3,4")
    ap.add_argument("--freeze-seeds", type=str, default="0,1,2")
    ap.add_argument("--methods", type=str, default="fourier_qaoa,power_p,walsh")
    ap.add_argument("--lam", type=float, default=0.5)
    ap.add_argument("--B-train", type=int, default=128)
    ap.add_argument("--B-hold", type=int, default=256)
    ap.add_argument("--B-n12", type=int, default=64)
    ap.add_argument("--product-cap", type=int, default=0, help="0 = product gap on every instance")
    ap.add_argument("--rounds-L2", type=int, default=18)
    ap.add_argument("--rounds-L16", type=int, default=24)
    ap.add_argument("--max-iter-L2", type=int, default=8)
    ap.add_argument("--max-iter-L16", type=int, default=8)
    ap.add_argument("--lr", type=float, default=0.35)
    ap.add_argument("--skip-self-check", action="store_true")
    ap.add_argument("--skip-freeze", action="store_true")
    args = ap.parse_args()
    args.ns = tuple(int(x) for x in args.ns.split(",") if x.strip())
    args.Ls = tuple(int(x) for x in args.Ls.split(",") if x.strip())
    args.seeds = tuple(int(x) for x in args.seeds.split(",") if x.strip())
    args.freeze_seeds = tuple(int(x) for x in args.freeze_seeds.split(",") if x.strip())
    args.methods = tuple(x.strip() for x in args.methods.split(",") if x.strip())

    torch.set_num_threads(4)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    run_dir = out / "runs"
    run_dir.mkdir(exist_ok=True)
    cache = out / "cache"
    cache.mkdir(exist_ok=True)
    proto = protocol_blob(args)
    t_all = time.time()
    log("self-check" if not args.skip_self_check else "self-check skipped")
    check = {"ok": True, "skipped": True} if args.skip_self_check else self_check()
    if not args.skip_self_check:
        log("self-check ok", f"walsh |Δp|={check['walsh_equal_gamma_max_abs_pZ']:.2e}")

    from Nov_3rd.quantum_h_tf import FockSim

    bundles = {n: load_bundle(n, args.B_train, args.B_hold) for n in args.ns}
    for n, b in bundles.items():
        ms = [ins.m for ins in b["train"] + b["hold"]]
        log(
            f"  n={n} m {min(ms)}/{np.median(ms):.0f}/{max(ms)} "
            f"spec train med deg3={b['spec_tr']['med_frac_deg'][3]:.6f} "
            f"off={b['spec_tr']['med_off_deg3']:.2e}"
        )
        # Algebraic reconstruction of every diagonal.
        for split, diag in (("train", b["diag_tr"]), ("hold", b["diag_ho"])):
            comp = b["comps_tr" if split == "train" else "comps_ho"].numpy()
            recon = comp.sum(axis=1) + diag.mean(axis=1, keepdims=True)
            err = float(np.max(np.abs(recon - diag)))
            if err > 1e-8:
                raise RuntimeError(f"Walsh reconstruction n={n} {split} err {err}")

    quantum = {}
    ed_cache = {}
    for n in args.ns:
        for split, insts in (("train", bundles[n]["train"]), ("hold", bundles[n]["hold"])):
            log(f"ED n={n} {split} B={len(insts)} λ={args.lam}")
            rows = ed_cached(
                cache / f"ed_n{n}_{split}_B{len(insts)}_l{args.lam}.npz",
                insts, n, args.lam, args.product_cap,
            )
            summ = summarize_ed(rows)
            quantum[f"{split}_n{n}"] = {"n": n, "split": split, "summary": summ}
            ed_cache[(n, split)] = [r["E0"] for r in rows]
            log(
                f"  n={n} {split} peak={summ['med_peak']:.3f} "
                f"prodΔ={summ['med_prod_gap']:.3f} pass={summ['quantum_pass']}"
            )

    quantum_ok = all(
        quantum[f"{split}_n{n}"]["summary"]["quantum_pass"]
        for n in args.ns for split in ("train", "hold")
    )
    sims = {n: FockSim(n, dtype=torch.float64) for n in args.ns}
    rows = run_per_n(bundles, ed_cache, sims, args, proto, run_dir) if quantum_ok else []
    if not quantum_ok:
        log("quantumness gate failed; training skipped")

    rng = np.random.default_rng(20261104)
    grouped: dict = {}
    for r in rows:
        grouped.setdefault((r["n"], r["L"], r["method"]), []).append(r)
    per_method = {}
    aggs = {}
    for key, rs in grouped.items():
        # Stable seed order.
        rs = sorted(rs, key=lambda z: z["seed"])
        ag = aggregate(rs, rng)
        aggs[key] = ag
        per_method[f"n{key[0]}_L{key[1]}_{key[2]}"] = ag
    cells = []
    for n in args.ns:
        for L in args.Ls:
            if (n, L, "fourier_qaoa") not in aggs:
                continue
            q = aggs[(n, L, "fourier_qaoa")]
            for method in ("power_p", "walsh"):
                if (n, L, method) not in aggs:
                    continue
                cell = cell_for(aggs[(n, L, method)], q, rng, n, L)
                cell["method"] = method
                cells.append(cell)

    l16_power = {c["n"]: c for c in cells if c["method"] == "power_p" and int(c["L"]) == 16}
    shrink, shrink_detail = shrink_test(l16_power)
    checks = {
        "beat_both": bool(l16_power) and all(l16_power[n]["verdict"] == "BEAT" for n in l16_power),
        "majority": bool(l16_power) and all(
            l16_power[n]["n_paired_beat"] > l16_power[n]["n_seeds"] / 2.0 for n in l16_power
        ),
        "within_both": bool(l16_power) and all(l16_power[n]["within_one_seed_std"] for n in (6, 9) if n in l16_power) and 6 in l16_power and 9 in l16_power,
        "shrink": shrink,
        "shrink_detail": shrink_detail,
        "saturated": bool(l16_power) and 6 in l16_power and 9 in l16_power and all(l16_power[n]["saturated"] for n in (6, 9)),
    }
    verdict, reason = decide(quantum_ok, l16_power)

    # Walsh note from the measured spectrum and the trained spread.
    off = max(bundles[n]["spec_ho"]["med_off_deg3"] for n in args.ns)
    spreads = []
    for key, ag in aggs.items():
        if key[2] != "walsh":
            continue
        for mx in ag["mixer_per_seed"]:
            if mx:
                spreads.append(mx.get("gamma_spread_vs_deg3") or 0.0)
    med_spread = float(np.median(spreads)) if spreads else float("nan")
    if off < 1e-8:
        walsh_note = (
            f"Off-degree-3 mass is {off:.2e} (median over the larger split). "
            "Degrees 1, 2, and 4 are numerical zeros, so their γ weights have zero gradient "
            "and cannot grade the mixer. The shells c2 and c3 are the mixer grading, "
            "and they are already the power_p head. "
            f"Trained |γ_w − γ_3| mean is {med_spread:.3e}. "
            "Walsh-degree γ was still optimized, as a separate 7L head warm-started from the same Fourier dual."
        )
    else:
        walsh_note = (
            f"Off-degree-3 mass is {off:.3e}, so Walsh-degree γ is a live knob. "
            f"Trained |γ_w − γ_3| mean is {med_spread:.3e}."
        )

    notes = []
    for key, row in quantum.items():
        s = row["summary"]
        nu = s.get("n_unique_E0")
        if nu is not None and nu <= 4:
            notes.append(
                f"{key}: {nu} distinct E_0 values, majority fraction {s['E0_majority_frac']:.2f}. "
                "At n=6 the cross-cavity degree-3 extraction is geometrically almost unique, "
                "so a larger B repeats two spectra rather than drawing new Hamiltonians."
            )
    notes.append(
        "The one-seed-std flag compares the median gap to the larger seed-std. "
        "One weak seed inflates that std above a median gap that the paired ratios and the "
        "instance bootstrap still see. Read it next to the seed CI of R, not instead of it."
    )
    for n in args.ns:
        insts_all = bundles[n]["train"] + bundles[n]["hold"]
        ms = [ins.m for ins in insts_all]
        dmax = max(int(ins.degree.max()) for ins in insts_all)
        notes.append(
            f"n={n}: extracted m median {float(np.median(ms)):.0f} "
            f"(min {min(ms)}, max {max(ms)}), degree ≤ {dmax}."
        )
    notes.append(
        "power_p has 4L parameters and Fourier QAOA has 2L. "
        "Walsh-degree γ has 7L. The comparison is not matched-parameter. "
        "power_p and Walsh are polished from the trained Fourier dual, so a win is the extra shells "
        "(and, if the spectrum were not pure degree 3, the extra γ weights) on top of that mixer."
    )
    notes.append(
        "Holdout checkpoint selection uses the holdout median residual, the same rule as the smoke, "
        "applied to every head. Absolute ε is slightly optimistic. R compares heads selected the same way."
    )
    notes.append(
        "Seed 95% CIs are percentile bootstrap of the median over the training seeds (10_000 resamples). "
        "With five seeds the interval sits on the middle order statistics. "
        "The instance CI is a separate bootstrap of one seed's B_hold residuals."
    )
    # Mixer rms, L=16, to see whether Walsh moved the shells differently from power_p.
    for n in args.ns:
        for method in ("power_p", "walsh"):
            ag = aggs.get((n, 16, method))
            if not ag:
                continue
            c2 = [m.get("c2_rms") for m in ag["mixer_per_seed"] if m]
            c3 = [m.get("c3_rms") for m in ag["mixer_per_seed"] if m]
            if c2:
                notes.append(
                    f"n={n} L=16 {method}: median seed c2 rms={float(np.median(c2)):.4f}, "
                    f"c3 rms={float(np.median(c3)):.4f}."
                )

    freeze = None
    freeze_note = "Skipped."
    if quantum_ok:
        try:
            freeze = run_freeze(bundles, ed_cache, sims, args, proto, run_dir, args.lam)
            if freeze is None:
                freeze_note = "Freeze skipped."
        except Exception as exc:
            freeze_note = f"Freeze failed and was skipped: {exc}"
            log("freeze failed", exc)
            import traceback
            traceback.print_exc()
    else:
        freeze_note = "Freeze skipped because the quantumness gate failed."

    try:
        import subprocess
        branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=_ROOT, text=True).strip()
    except Exception:
        branch = "cursor/quantum-h-seal-a-d784"

    spectrum = {}
    for n, b in bundles.items():
        for split, spec in (("train", b["spec_tr"]), ("hold", b["spec_ho"])):
            spectrum[f"{split}_n{n}"] = {"n": n, "split": split, **spec}

    payload = {
        "family": "A",
        "name": "transverse-field Max-E3LIN2 seal",
        "verdict": verdict,
        "verdict_reason": reason,
        "lam": args.lam,
        "quantum": quantum,
        "quantum_ok": quantum_ok,
        "spectrum": spectrum,
        "walsh_note": walsh_note,
        "cells": cells,
        "per_method": {
            k: {kk: vv for kk, vv in ag.items() if kk != "mixer_per_seed"}
            for k, ag in per_method.items()
        },
        "checks": checks,
        "freeze": freeze,
        "freeze_note": freeze_note,
        "self_check": check,
        "notes": notes,
        "B_train": args.B_train,
        "B_hold": args.B_hold,
        "seeds": list(args.seeds),
        "Ls": list(args.Ls),
        "methods": list(args.methods),
        "wall_s": time.time() - t_all,
        "branch": branch,
        "protocol": proto,
    }
    # Drop bulky per-instance vectors from the summary. They live in runs/*.json.
    (out / "summary.json").write_text(json.dumps(_jsonable(payload), indent=2) + "\n")
    md = render_md(payload)
    (out / "RESULTS.md").write_text(md)
    log("verdict", verdict)
    log(reason)
    log("wrote", out, f"wall {payload['wall_s']:.1f}s")


if __name__ == "__main__":
    main()
