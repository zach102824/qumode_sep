#!/usr/bin/env python3
"""Smoke: power-law long-range Ising + driver, shallow Walsh SNAP.

    H = Σ_{i<j} (J_ij / |i-j|^α) Z_i Z_j − λ Σ_i X_i
        + μ Σ_c Σ_{i<j ∈ c} X_i X_j

α=1.5, J_ij∈{±1} on the open cavity chain. λ∈{0.5, 1.0}, μ∈{0, −0.5}.
Train n∈{6, 9}. Freeze n=12. L∈{2, 4}. Seeds 0, 1, 2.

Arms: Walsh SNAP (degree-2 γ, Hamming shell) and Fourier QAOA.
HVA (β^X, β^XX) is added only when μ≠0. Readout is Z-hist + X-hist.

n=21 is not read or written. settings.py is not modified.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from Nov_3rd import settings as S
from Nov_3rd.powerlaw_tfim import (
    ALPHA,
    KILL_R,
    PASS_R_DRIVER,
    PASS_R_PURE,
    PROD_GAP_MIN,
    PEAK_LT,
    TRANSFER_SLACK,
    cost_diag,
    couplings,
    cross_cavity_weight,
    decide,
    expectation,
    init_method,
    instance_eps,
    instance_seed,
    parameter_list,
    prepare,
    quantum_observables,
    quantum_summary,
    ratio,
    self_check,
    shell_nonlinearity,
    zz_diag_from_K,
)

PROTOCOL = "powerlaw_tfim_breadth_v1"
RUN = S.OUT / "quantum_h_explore" / "powerlaw_tfim_breadth"


def log(*a):
    print(time.strftime("%H:%M:%S"), "[powerlaw]", *a, flush=True)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(path)


def _fmt(x, nd=4):
    if x is None or not np.isfinite(x):
        return "inf"
    return f"{float(x):.{nd}f}"


def build_diag_pool(n: int, count: int, cache: Path) -> tuple[np.ndarray, np.ndarray]:
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.exists():
        z = np.load(cache)
        if int(z["n"]) == int(n) and int(z["count"]) == int(count) and float(z["alpha"]) == float(ALPHA):
            log(f"diag cache hit n{n} B={count}")
            return np.asarray(z["diag"], dtype=np.float64), np.asarray(z["seeds"], dtype=np.int64)
    log(f"building diag n{n} B={count}")
    t0 = time.time()
    diags = np.empty((count, 1 << n), dtype=np.float64)
    seeds = np.empty(count, dtype=np.int64)
    for i in range(count):
        _J, K = couplings(n, i)
        diags[i] = zz_diag_from_K(K)
        seeds[i] = instance_seed(n, i)
    np.savez_compressed(cache, diag=diags, seeds=seeds, n=n, count=count, alpha=ALPHA)
    log(f"diag n{n} B={count} in {time.time()-t0:.1f}s  std={diags.std():.3f}")
    return diags, seeds


def build_quantum(n: int, diag: np.ndarray, seeds: np.ndarray, lam: float, mu: float, cache: Path) -> dict:
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.exists():
        z = np.load(cache)
        same = (
            int(z["n"]) == int(n)
            and abs(float(z["lam"]) - float(lam)) < 1e-15
            and abs(float(z["mu"]) - float(mu)) < 1e-15
            and np.array_equal(z["seeds"], seeds)
        )
        if same:
            log(f"quantum cache hit n{n} λ={lam} μ={mu} B={len(seeds)}")
            return {k: np.asarray(z[k]) for k in ("e0", "peak", "e_prod", "prod_gap", "prod_gap_density")}
    log(f"quantum n{n} λ={lam:.2f} μ={mu:.2f} B={len(seeds)}")
    t0 = time.time()
    e0 = np.empty(len(seeds))
    peak = np.empty(len(seeds))
    e_prod = np.empty(len(seeds))
    gap = np.empty(len(seeds))
    dens = np.empty(len(seeds))
    for i in range(len(seeds)):
        _J, K = couplings(n, i)
        obs = quantum_observables(K, diag[i], n, lam, mu, seed=int(seeds[i]))
        e0[i] = obs["E0"]
        peak[i] = obs["peak"]
        e_prod[i] = obs["E_prod"]
        gap[i] = obs["prod_gap"]
        dens[i] = obs["prod_gap_density"]
        if (i + 1) == len(seeds) or (i + 1) % 8 == 0:
            log(f"  n{n} {i+1}/{len(seeds)} E0={e0[i]:.3f} peak={peak[i]:.3f} gap={gap[i]:.3f}")
    np.savez_compressed(
        cache,
        e0=e0,
        peak=peak,
        e_prod=e_prod,
        prod_gap=gap,
        prod_gap_density=dens,
        seeds=seeds,
        n=n,
        lam=lam,
        mu=mu,
        alpha=ALPHA,
    )
    log(f"quantum n{n} λ={lam:.2f} μ={mu:.2f} done in {time.time()-t0:.1f}s")
    return {"e0": e0, "peak": peak, "e_prod": e_prod, "prod_gap": gap, "prod_gap_density": dens}


def angles_to_lists(angles: dict) -> dict:
    out = {}
    for k, v in angles.items():
        if torch.is_tensor(v):
            out[k] = v.detach().cpu().to(torch.float64).numpy().tolist()
        else:
            out[k] = v
    return out


def lists_to_angles(raw: dict) -> dict:
    return {k: torch.tensor(v, dtype=torch.float64) for k, v in raw.items() if k != "method"}


def train_loss(method, angles, diag, e0, idx, n, lam, mu) -> torch.Tensor:
    d = diag.index_select(0, idx)
    e0b = e0.index_select(0, idx)
    psi = prepare(method, cost_diag(d, n), angles)
    h = expectation(psi, d, n, lam, mu)
    return ((h - e0b) / float(n)).mean()


def eval_split(method, angles, diag, e0, n, lam, mu, chunk: int) -> dict:
    eps = instance_eps(method, angles, diag, e0, n, lam, mu, chunk=chunk).detach().cpu().numpy()
    return {
        "n": int(n),
        "B": int(eps.size),
        "med_eps": float(np.median(eps)),
        "mean_eps": float(np.mean(eps)),
        "eps": [float(x) for x in eps.tolist()],
    }


def math_finite(x: float) -> bool:
    return x == x and abs(x) != float("inf")


def train_one(method, L, seed, lam, mu, pools, *, rounds: int, batch: int) -> dict:
    torch.manual_seed(int(seed) + 1000 * int(L) + int(round(100 * lam)) + int(round(10 * abs(mu))))
    rng = np.random.default_rng(int(seed) + 17 * int(L))
    angles = init_method(method, L, dtype=torch.float64)
    params = parameter_list(method, angles)
    tr6, tr9 = pools["train6"], pools["train9"]
    ho9 = pools["hold9"]
    n6 = int(tr6["diag"].shape[0])
    n9 = int(tr9["diag"].shape[0])
    b6 = min(batch // 2, n6)
    b9 = min(batch - b6, n9)

    def hold_med() -> float:
        ev = eval_split(method, angles, ho9["diag"], ho9["e0"], 9, lam, mu, chunk=16)
        return float(ev["med_eps"])

    t0 = time.time()
    init_med = hold_med()
    best = {
        "med9": init_med,
        "round": -1,
        "angles": {k: v.detach().cpu().clone() for k, v in angles.items()},
    }
    hist = [{"round": -1, "hold_med9": init_med, "loss": None}]
    log(f"L{L} {method} s{seed} λ={lam:.2f} μ={mu:.2f} init n9-hold Medε={init_med:.5f}")

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
                train_loss(method, angles, tr6["diag"], tr6["e0"], i6, 6, lam, mu)
                + train_loss(method, angles, tr9["diag"], tr9["e0"], i9, 9, lam, mu)
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
        "L": int(L),
        "seed": int(seed),
        "lam": float(lam),
        "mu": float(mu),
        "init_hold_med9": init_med,
        "best_round": int(best["round"]),
        "best_hold_med9": float(best["med9"]),
        "wall_s": time.time() - t0,
        "hist": hist,
        "angles": best["angles"],
        "n_params": int(sum(p.numel() for p in params)),
    }


def seed_path(run: Path, L, method, seed, lam, mu) -> Path:
    tag = f"L{L}_{method}_s{seed}_l{lam:.2f}_m{mu:.2f}.json"
    return run / tag


def budget_ok(payload: dict, rounds: int, train_b: int, hold_b: int, freeze_b: int) -> bool:
    b = payload.get("budget") or {}
    return (
        payload.get("protocol") == PROTOCOL
        and int(b.get("rounds", -1)) == int(rounds)
        and int(b.get("train_b", -1)) == int(train_b)
        and int(b.get("hold_b", -1)) == int(hold_b)
        and int(b.get("freeze_b", -1)) == int(freeze_b)
        and "eval" in payload
        and "angles" in payload
    )


def attach_eval(trained: dict, pools, lam, mu) -> dict:
    angles = trained["angles"]
    ev = {}
    for name, n, chunk in (
        ("6", 6, 16),
        ("9", 9, 16),
        ("12", 12, 4),
    ):
        pool = pools[{"6": "hold6", "9": "hold9", "12": "freeze12"}[name]]
        ev[name] = eval_split(trained["method"], angles, pool["diag"], pool["e0"], n, lam, mu, chunk=chunk)
        if float(np.min(ev[name]["eps"])) < -1e-3:
            raise RuntimeError(
                f"negative excess {min(ev[name]['eps'])} at n{n} {trained['method']} "
                "(E0 is above a prepared state)"
            )
        log(
            f"  eval n{n} {trained['method']} Medε={ev[name]['med_eps']:.5f}"
        )
    trained = dict(trained)
    trained["angles"] = angles_to_lists(angles)
    trained["eval"] = ev
    return trained


def diagnostics(method: str, angles: dict) -> dict:
    if method == "snap":
        return {
            "shell_rms": shell_nonlinearity(np.asarray(angles["phi"], dtype=np.float64)),
            "mean_abs_gamma2": float(np.mean(np.abs(angles["gamma2"]))),
        }
    if method == "hva":
        return {"mean_abs_beta_xx": float(np.mean(np.abs(angles["beta_xx"])))}
    return {}


def aggregate(payloads: list[dict], quantum_by_key: dict) -> dict:
    """Median-over-seeds of each seed's median-over-instances excess."""
    groups: dict[tuple, list] = {}
    for p in payloads:
        key = (float(p["lam"]), float(p["mu"]), int(p["L"]), p["method"])
        groups.setdefault(key, []).append(p)
    cells = []
    rows = []
    lams = sorted({k[0] for k in groups})
    mus = sorted({k[1] for k in groups})
    depths = sorted({k[2] for k in groups})
    for lam in lams:
        for mu in mus:
            q12 = quantum_by_key[(lam, mu, 12)]
            q6 = quantum_by_key[(lam, mu, 6)]
            q9 = quantum_by_key[(lam, mu, 9)]
            for L in depths:
                methods = []
                if (lam, mu, L, "fourier") in groups:
                    methods.append("fourier")
                if (lam, mu, L, "hva") in groups:
                    methods.append("hva")
                if (lam, mu, L, "snap") in groups:
                    methods.append("snap")
                if "snap" not in methods or "fourier" not in methods:
                    continue
                med = {}
                per_seed = {}
                seed_ids = {}
                diag = {}
                for method in methods:
                    runs = sorted(groups[(lam, mu, L, method)], key=lambda p: int(p["seed"]))
                    seed_ids[method] = [int(p["seed"]) for p in runs]
                    med[method] = {
                        n: float(np.median([p["eval"][n]["med_eps"] for p in runs]))
                        for n in ("6", "9", "12")
                    }
                    diag[method] = [diagnostics(method, p["angles"]) for p in runs]
                snap_seeds = {int(p["seed"]): p for p in groups[(lam, mu, L, "snap")]}
                ratios = {}
                for base in ("fourier", "hva"):
                    if base not in med:
                        continue
                    ratios[base] = {
                        n: ratio(med["snap"][n], med[base][n]) for n in ("6", "9", "12")
                    }
                    base_seeds = {int(p["seed"]): p for p in groups[(lam, mu, L, base)]}
                    paired = []
                    for s, sp in sorted(snap_seeds.items()):
                        if s not in base_seeds:
                            continue
                        paired.append(ratio(sp["eval"]["12"]["med_eps"], base_seeds[s]["eval"]["12"]["med_eps"]))
                    per_seed[base] = paired
                cells.append(
                    {
                        "lam": lam,
                        "mu": mu,
                        "L": L,
                        "quantum_pass": bool(q12["pass"]),
                        "quantum12": q12,
                        "quantum6": q6,
                        "quantum9": q9,
                        "ratios": ratios,
                        "per_seed": per_seed,
                        "med_eps": med,
                        "seeds": seed_ids,
                        "diagnostics": diag,
                    }
                )
                rows.append(cells[-1])
    decision = decide(cells)
    return {"cells": rows, "decision": decision}


def _md_quantum(q: dict) -> str:
    flag = "pass" if q["pass"] else "fail"
    return (
        f"{q['peak_med']:.3f} / {q['gap_med']:.3f} ({flag}, "
        f"peak<0.5 on {q['frac_peak_below']:.0%}, gap≥{PROD_GAP_MIN:.2f} on {q['frac_gap_ok']:.0%})"
    )


def render_results(summary: dict) -> str:
    d = summary["decision"]
    cfg = summary["config"]
    lines = []
    lines.append("# Power-law long-range Ising — breadth smoke")
    lines.append("")
    reason = d["reason"]
    if reason:
        reason = reason[:1].upper() + reason[1:]
    lines.append(f"**Verdict: {d['verdict']}.** {reason}.")
    lines.append("")
    lines.append(
        "This is a 3-seed smoke at paper depth L∈{2, 4}. "
        "It is not the 5-seed shallow seal used for the unsat SAFE point."
    )
    lines.append("")
    lines.append("## Hamiltonian")
    lines.append("")
    lines.append("Second non-CSP family. The diagonal is a spin glass, not an unsat count.")
    lines.append("")
    lines.append("$$")
    lines.append(r"H = \sum_{i<j} \frac{J_{ij}}{|i-j|^{1.5}} Z_i Z_j - \lambda \sum_i X_i + \mu \sum_c \sum_{i<j\in c} X_i X_j")
    lines.append("$$")
    lines.append("")
    lines.append(
        r"\(J_{ij}\in\{\pm 1\}\) is drawn once per instance on the open chain. "
        r"Qubit \(i\) sits at site \(i\), qubit 0 is the MSB, cavity \(c\) owns \([3c, 3c+3)\). "
        "Bonds that cross a cavity boundary stay in the sum. "
        + r"Cross-cavity share of \(\sum|K_{ij}|\) is "
        + ", ".join(f"n{n}={summary['cross_weight'][str(n)]:.3f}" for n in (6, 9, 12))
        + "."
    )
    lines.append("")
    lines.append(
        r"The circuit phases \(U_C(\gamma)=\exp(-i\gamma H_{ZZ}/n)\). "
        r"Dividing by \(n\) only rescales \(\gamma\); \(\langle H\rangle\) and \(E_0\) use the formula above. "
        r"\(H_{ZZ}\) is pure Walsh degree 2, so Walsh SNAP trains \(\gamma_2\) only. "
        r"The Hamming shell \(\varphi_k\) is free. Linear \(\varphi_k=-\beta k\) is the Fourier X mixer."
    )
    lines.append("")
    lines.append(
        r"Readout is the Z histogram for every ZZ and the X histogram for \(\langle X\rangle\) and intra-cavity \(\langle XX\rangle\). No Y."
    )
    lines.append("")
    lines.append("## Gate")
    lines.append("")
    lines.append(
        r"Med \(\varepsilon\) is the median over seeds of each seed's median-over-instances excess \(\langle H\rangle-E_0\). "
        r"\(R = \mathrm{Med}\varepsilon_{\mathrm{SNAP}} / \mathrm{Med}\varepsilon_{\mathrm{baseline}}\)."
    )
    lines.append("")
    lines.append(
        f"- Quantumness on the frozen n=12 pool: median bitstring peak < {PEAK_LT} and "
        + rf"median product gap \((E_{{\mathrm{{prod}}}}-E_0)/|E_0|\) ≥ {PROD_GAP_MIN}. Otherwise that (λ, μ) is needs_edit."
    )
    lines.append(
        f"- Driver PASS (μ=−0.5): some baseline in {{Fourier, HVA}} has n12 R ≤ {PASS_R_DRIVER:.2f}, "
        f"n6 and n9 R ≤ {KILL_R:.2f}, and n12 R ≤ max(n6, n9) + {TRANSFER_SLACK:.2f} (transfer held). "
        "Either baseline is enough."
    )
    lines.append(
        f"- Pure-family PASS (μ=0): the same rule against Fourier at n12 R ≤ {PASS_R_PURE:.2f}."
    )
    lines.append(
        f"- KILL: at L=4, every (λ, μ) has Fourier R > {KILL_R:.2f} at both n6 and n9. "
        "A driver or pure PASS outranks KILL."
    )
    lines.append("- n6 and n9 ratios are holdout. n12 is the freeze. n=21 was not loaded.")
    lines.append("")
    if d["driver_passes"]:
        lines.append("Driver cells that pass: " + "; ".join(d["driver_passes"]) + ".")
        lines.append("")
    if d["pure_passes"]:
        lines.append("Pure-family cells that pass: " + "; ".join(d["pure_passes"]) + ".")
        lines.append("")
    seal = d["seal_warranted"]
    why = seal["why"]
    if why:
        why = why[:1].upper() + why[1:]
    lines.append(
        f"**5-seed seal:** {'warranted' if seal['warranted'] else 'not warranted'}. {why}"
    )
    lines.append("")
    lines.append("## Setup")
    lines.append("")
    lines.append("| knob | value |")
    lines.append("|---|---|")
    lines.append(f"| α | {cfg['alpha']} |")
    lines.append(f"| λ, μ | {cfg['lambdas']} × {cfg['mus']} |")
    lines.append(f"| L | {cfg['depths']} |")
    lines.append(f"| seeds | {cfg['seeds']} |")
    lines.append(
        f"| pools | train B={cfg['train_b']} (n6 and n9, indices 0..), "
        f"holdout B={cfg['hold_b']}, freeze n12 B={cfg['freeze_b']} |"
    )
    lines.append(
        f"| optimizer | L-BFGS {cfg['rounds']}×{cfg['lbfgs_iter']}, lr {cfg['lbfgs_lr']}, "
        f"batch {cfg['batch']} (half n6 / half n9), grad clip 5 |"
    )
    lines.append("| init | β=β^X=0.01, β^XX=0, γ=γ_2=−0.01, φ_k=−β k |")
    lines.append("| checkpoint | lowest n9-holdout Med ε |")
    lines.append(f"| instance seed | {cfg['seed_formula']} |")
    lines.append("")
    lines.append("## Quantumness")
    lines.append("")
    lines.append("Median bitstring peak and median product gap. The pass bit is the n=12 row.")
    lines.append("")
    lines.append("| λ | μ | n6 hold | n9 hold | n12 freeze |")
    lines.append("|---:|---:|---|---|---|")
    seen = set()
    for c in summary["cells"]:
        key = (c["lam"], c["mu"])
        if key in seen:
            continue
        seen.add(key)
        lines.append(
            f"| {c['lam']:.2f} | {c['mu']:.2f} | {_md_quantum(c['quantum6'])} | "
            f"{_md_quantum(c['quantum9'])} | {_md_quantum(c['quantum12'])} |"
        )
    lines.append("")
    lines.append("## Ratios")
    lines.append("")
    lines.append("R < 1 means Walsh SNAP has the lower excess energy.")
    lines.append("")
    for L in cfg["depths"]:
        lines.append(f"### L={L}")
        lines.append("")
        lines.append("| λ | μ | n | Medε Fourier | Medε HVA | Medε SNAP | R Fourier | R HVA |")
        lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|")
        for c in summary["cells"]:
            if int(c["L"]) != int(L):
                continue
            for n in ("6", "9", "12"):
                role = {"6": "6 hold", "9": "9 hold", "12": "12 freeze"}[n]
                hva_eps = c["med_eps"].get("hva", {}).get(n)
                rf = c["ratios"].get("fourier", {}).get(n)
                rh = c["ratios"].get("hva", {}).get(n) if "hva" in c["ratios"] else None
                lines.append(
                    f"| {c['lam']:.2f} | {c['mu']:.2f} | {role} | {_fmt(c['med_eps']['fourier'][n])} | "
                    f"{_fmt(hva_eps) if hva_eps is not None else '—'} | {_fmt(c['med_eps']['snap'][n])} | "
                    f"{_fmt(rf)} | {_fmt(rh) if rh is not None else '—'} |"
                )
        lines.append("")
        lines.append("| λ | μ | cell | seeds n12 R Fourier | seeds n12 R HVA | shell RMS | mean \\|β^XX\\| |")
        lines.append("|---:|---:|---|---|---|---:|---:|")
        for c in summary["cells"]:
            if int(c["L"]) != int(L):
                continue
            call = c["call"]["pass"]
            label = "PASS " + ",".join(c["call"]["passed_by"]) if call else c["call"]["reason"]
            sf = ", ".join(_fmt(r, 3) for r in c["per_seed"].get("fourier", []))
            sh = ", ".join(_fmt(r, 3) for r in c["per_seed"].get("hva", [])) or "—"
            shells = [d.get("shell_rms") for d in c["diagnostics"].get("snap", []) if "shell_rms" in d]
            bxx = [d.get("mean_abs_beta_xx") for d in c["diagnostics"].get("hva", []) if "mean_abs_beta_xx" in d]
            shell_s = _fmt(float(np.median(shells))) if shells else "—"
            bxx_s = _fmt(float(np.median(bxx))) if bxx else "—"
            lines.append(
                f"| {c['lam']:.2f} | {c['mu']:.2f} | {label} | {sf} | {sh} | {shell_s} | {bxx_s} |"
            )
        lines.append("")
    if summary.get("l16"):
        lines.append("## L=16 light probe")
        lines.append("")
        lines.append(summary["l16"]["note"])
        lines.append("")
        p = summary["l16"]
        lines.append("| n | Medε Fourier | Medε HVA | Medε SNAP | R Fourier | R HVA |")
        lines.append("|---:|---:|---:|---:|---:|---:|")
        for n in ("6", "9", "12"):
            lines.append(
                f"| {n} | {_fmt(p['med_eps']['fourier'][n])} | "
                f"{_fmt(p['med_eps'].get('hva', {}).get(n)) if 'hva' in p['med_eps'] else '—'} | "
                f"{_fmt(p['med_eps']['snap'][n])} | {_fmt(p['ratios']['fourier'][n])} | "
                f"{_fmt(p['ratios'].get('hva', {}).get(n)) if 'hva' in p['ratios'] else '—'} |"
            )
        lines.append("")
        lines.append("The probe is one seed and a shorter L-BFGS budget. It is not part of the verdict.")
        lines.append("")
    lines.append("## Reading")
    lines.append("")
    lines.append(summary["reading"])
    lines.append("")
    lines.append(
        "Per-seed JSON is `L{L}_{method}_s{seed}_l{λ}_m{μ}.json` in this folder. "
        "Angles are the restored n9-holdout checkpoint. "
        "`qaoa_angles/`, `train69_ladder_fresh250/`, `settings.py`, and the phase-2 n=21 ladder were not written."
    )
    lines.append("")
    return "\n".join(lines)


def reading_for(summary_decision: dict, cells: list[dict]) -> str:
    verdict = summary_decision["verdict"]
    reason = summary_decision["reason"]
    reason = reason[:1].upper() + reason[1:] if reason else reason
    bits = [reason + "."]
    best_f = None
    best_h = None
    for c in cells:
        rf = c["ratios"].get("fourier", {}).get("12")
        rh = c["ratios"].get("hva", {}).get("12")
        if rf is not None and np.isfinite(rf) and (best_f is None or rf < best_f[0]):
            best_f = (rf, c)
        if rh is not None and np.isfinite(rh) and (best_h is None or rh < best_h[0]):
            best_h = (rh, c)
    if best_f is not None:
        c = best_f[1]
        rf6 = c["ratios"]["fourier"]["6"]
        rf9 = c["ratios"]["fourier"]["9"]
        bits.append(
            f"Best frozen n12 Fourier ratio is {_fmt(best_f[0])} at λ={c['lam']:.2f} μ={c['mu']:.2f} L={c['L']} "
            f"(SNAP Medε {_fmt(c['med_eps']['snap']['12'])}, Fourier {_fmt(c['med_eps']['fourier']['12'])}). "
            f"The same cell is {_fmt(rf6)} at n6 and {_fmt(rf9)} at n9."
        )
    if best_h is not None:
        c = best_h[1]
        bits.append(
            f"Best n12 HVA ratio is {_fmt(best_h[0])} at λ={c['lam']:.2f} μ={c['mu']:.2f} L={c['L']} "
            f"(HVA Medε {_fmt(c['med_eps']['hva']['12'])})."
        )
    q_fail = sorted({(c["lam"], c["mu"]) for c in cells if not c["quantum_pass"]})
    q_ok = sorted({(c["lam"], c["mu"]) for c in cells if c["quantum_pass"]})
    if q_ok:
        bits.append(
            "Quantum n12 ground states (peak < 0.5 and product gap ≥ 0.03): "
            + ", ".join(f"λ={lam:.2f} μ={mu:.2f}" for lam, mu in q_ok)
            + "."
        )
    if q_fail:
        bits.append(
            "Product-gap failures, so those cells cannot PASS: "
            + ", ".join(f"λ={lam:.2f} μ={mu:.2f}" for lam, mu in q_fail)
            + ". A spread-out bitstring peak is not enough when a product state already reaches the ground energy."
        )
    if verdict == "PASS":
        bits.append(
            "Shallow Walsh SNAP is ahead of the baseline named in the passing cell, "
            "on a ground state that is not a product state and not a bitstring."
        )
    elif verdict == "KILL":
        bits.append(
            "At L=4 the training sizes do not beat Fourier anywhere on the grid. "
            "This family does not earn a seal."
        )
    else:
        bits.append(
            "H_ZZ is pure Walsh degree 2, so the SNAP cost has one live γ per layer. "
            "Any lead over Fourier has to come from the Hamming shell."
        )
        shells = []
        bxxs = []
        for c in cells:
            if abs(float(c["mu"])) < 1e-15:
                continue
            shells.extend(d.get("shell_rms") for d in c["diagnostics"].get("snap", []) if d.get("shell_rms") is not None)
            bxxs.extend(
                d.get("mean_abs_beta_xx")
                for d in c["diagnostics"].get("hva", [])
                if d.get("mean_abs_beta_xx") is not None
            )
        if shells or bxxs:
            extra = []
            if shells:
                extra.append(f"median shell RMS off φ_k=a+bk is {_fmt(float(np.median(shells)))}")
            if bxxs:
                extra.append(f"median mean |β^XX| is {_fmt(float(np.median(bxxs)))}")
            bits.append(
                "On the μ=−0.5 cells the shell does leave the X mixer ("
                + "; ".join(extra)
                + "), and SNAP still does not clear 0.90 against Fourier or against HVA."
            )
        ratio_only = []
        for c in cells:
            if c["quantum_pass"] or abs(float(c["mu"])) > 1e-15:
                continue
            rf = c["ratios"]["fourier"]["12"]
            if np.isfinite(rf) and rf <= PASS_R_PURE:
                ratio_only.append(
                    f"λ={c['lam']:.2f} μ=0 L={c['L']} would be a pure-family ratio ({_fmt(rf)}) and fails the product gap"
                )
        if ratio_only:
            bits.append("Ratio without quantumness: " + "; ".join(ratio_only) + ".")
        if best_f is not None:
            seeds = best_f[1]["per_seed"].get("fourier") or []
            finite = [float(s) for s in seeds if np.isfinite(s)]
            if finite and min(finite) > PASS_R_DRIVER:
                bits.append(
                    f"Every seed at that best Fourier cell is above {PASS_R_DRIVER:.2f} "
                    f"(min {_fmt(min(finite), 3)}). Two more seeds do not reopen a 0.90 driver PASS."
                )
    return " ".join(bits)


def methods_for(mu: float) -> tuple[str, ...]:
    if abs(float(mu)) < 1e-15:
        return ("fourier", "snap")
    return ("fourier", "hva", "snap")


def run_grid(run: Path, cfg: dict) -> list[dict]:
    cache = run / "cache"
    count = {
        6: cfg["train_b"] + cfg["hold_b"],
        9: cfg["train_b"] + cfg["hold_b"],
        12: cfg["freeze_b"],
    }
    diags = {}
    seeds = {}
    for n, b in count.items():
        diags[n], seeds[n] = build_diag_pool(n, b, cache / f"diag_n{n}_B{b}.npz")
    payloads = []
    for lam in cfg["lambdas"]:
        for mu in cfg["mus"]:
            q = {}
            for n in (6, 9, 12):
                q[n] = build_quantum(
                    n,
                    diags[n],
                    seeds[n],
                    lam,
                    mu,
                    cache / f"q_n{n}_B{count[n]}_l{lam:.2f}_m{mu:.2f}.npz",
                )
            pools = {}
            tb, hb, fb = cfg["train_b"], cfg["hold_b"], cfg["freeze_b"]
            for name, n, sl in (
                ("train6", 6, slice(0, tb)),
                ("hold6", 6, slice(tb, tb + hb)),
                ("train9", 9, slice(0, tb)),
                ("hold9", 9, slice(tb, tb + hb)),
                ("freeze12", 12, slice(0, fb)),
            ):
                pools[name] = {
                    "diag": torch.tensor(diags[n][sl], dtype=torch.float64),
                    "e0": torch.tensor(q[n]["e0"][sl], dtype=torch.float64),
                }
            q_hold = {
                6: quantum_summary(q[6]["peak"][tb : tb + hb], q[6]["prod_gap"][tb : tb + hb]),
                9: quantum_summary(q[9]["peak"][tb : tb + hb], q[9]["prod_gap"][tb : tb + hb]),
                12: quantum_summary(q[12]["peak"][:fb], q[12]["prod_gap"][:fb]),
            }
            (run / "cache").mkdir(parents=True, exist_ok=True)
            write_json(
                run / "cache" / f"quantum_summary_l{lam:.2f}_m{mu:.2f}.json",
                {"lam": lam, "mu": mu, "n6": q_hold[6], "n9": q_hold[9], "n12": q_hold[12]},
            )
            for L in cfg["depths"]:
                for seed in cfg["seeds"]:
                    for method in methods_for(mu):
                        path = seed_path(run, L, method, seed, lam, mu)
                        if path.exists():
                            old = json.loads(path.read_text())
                            if budget_ok(old, cfg["rounds"], tb, hb, fb):
                                log(f"resume {path.name}")
                                payloads.append(old)
                                continue
                        trained = train_one(
                            method, L, seed, lam, mu, pools, rounds=cfg["rounds"], batch=cfg["batch"]
                        )
                        trained = attach_eval(trained, pools, lam, mu)
                        trained["protocol"] = PROTOCOL
                        trained["budget"] = {
                            "rounds": cfg["rounds"],
                            "train_b": tb,
                            "hold_b": hb,
                            "freeze_b": fb,
                            "batch": cfg["batch"],
                            "lbfgs_iter": cfg["lbfgs_iter"],
                            "lbfgs_lr": cfg["lbfgs_lr"],
                            "alpha": ALPHA,
                        }
                        trained["diagnostics"] = diagnostics(method, trained["angles"])
                        # Drop per-instance eps from the long hist-bearing file? Keep them;
                        # B≤64 and they make the median auditable.
                        write_json(path, trained)
                        payloads.append(trained)
                        log(
                            f"wrote {path.name} wall={trained['wall_s']:.0f}s "
                            f"n12 Medε={trained['eval']['12']['med_eps']:.4f}"
                        )
    # quantum summaries are reloaded by the caller from disk via payloads + cache
    return payloads


def quantum_from_disk(run: Path, cfg: dict) -> dict:
    out = {}
    for lam in cfg["lambdas"]:
        for mu in cfg["mus"]:
            path = run / "cache" / f"quantum_summary_l{lam:.2f}_m{mu:.2f}.json"
            raw = json.loads(path.read_text())
            out[(lam, mu, 6)] = raw["n6"]
            out[(lam, mu, 9)] = raw["n9"]
            out[(lam, mu, 12)] = raw["n12"]
    return out


def load_payloads(run: Path) -> list[dict]:
    rows = []
    for path in sorted(run.glob("L*_*.json")):
        if path.name.startswith("L16_"):
            continue
        rows.append(json.loads(path.read_text()))
    return rows


def run_l16(run: Path, cfg: dict, decision: dict, cells: list[dict]) -> dict | None:
    """One seed, fewer rounds, on the best μ≠0 cell. Not part of the gate."""
    cands = [c for c in cells if abs(float(c["mu"])) > 1e-15 and int(c["L"]) in (2, 4)]
    if not cands:
        return None
    def score(c):
        rs = [c["ratios"]["fourier"]["12"]]
        if "hva" in c["ratios"]:
            rs.append(c["ratios"]["hva"]["12"])
        return min(rs)
    best = min(cands, key=score)
    lam, mu, Lref = float(best["lam"]), float(best["mu"]), int(best["L"])
    rounds = max(8, cfg["rounds"] // 2)
    note = (
        f"Light probe at the best driver cell of the smoke (λ={lam:.2f}, μ={mu:.2f}, "
        f"reference L={Lref}). Seed 0 only, {rounds} L-BFGS rounds, same pools. "
        "Not in the verdict."
    )
    log(note)
    # Reuse pools by calling the same builders (cache hits) and training L=16.
    probe_cfg = dict(cfg)
    probe_cfg["depths"] = (16,)
    probe_cfg["seeds"] = (0,)
    probe_cfg["lambdas"] = (lam,)
    probe_cfg["mus"] = (mu,)
    probe_cfg["rounds"] = rounds
    # Isolate probe JSON names under the same run dir. train_one writes via seed_path
    # which would collide with a later full L=16. The gate loader skips L16_*.
    payloads = []
    cache = run / "cache"
    count = {6: cfg["train_b"] + cfg["hold_b"], 9: cfg["train_b"] + cfg["hold_b"], 12: cfg["freeze_b"]}
    diags, seeds = {}, {}
    for n, b in count.items():
        diags[n], seeds[n] = build_diag_pool(n, b, cache / f"diag_n{n}_B{b}.npz")
    q = {}
    for n in (6, 9, 12):
        q[n] = build_quantum(n, diags[n], seeds[n], lam, mu, cache / f"q_n{n}_B{count[n]}_l{lam:.2f}_m{mu:.2f}.npz")
    tb, hb, fb = cfg["train_b"], cfg["hold_b"], cfg["freeze_b"]
    pools = {}
    for name, n, sl in (
        ("train6", 6, slice(0, tb)),
        ("hold6", 6, slice(tb, tb + hb)),
        ("train9", 9, slice(0, tb)),
        ("hold9", 9, slice(tb, tb + hb)),
        ("freeze12", 12, slice(0, fb)),
    ):
        pools[name] = {
            "diag": torch.tensor(diags[n][sl], dtype=torch.float64),
            "e0": torch.tensor(q[n]["e0"][sl], dtype=torch.float64),
        }
    for method in methods_for(mu):
        path = seed_path(run, 16, method, 0, lam, mu)
        if path.exists():
            old = json.loads(path.read_text())
            if budget_ok(old, rounds, tb, hb, fb):
                payloads.append(old)
                continue
        trained = train_one(method, 16, 0, lam, mu, pools, rounds=rounds, batch=cfg["batch"])
        trained = attach_eval(trained, pools, lam, mu)
        trained["protocol"] = PROTOCOL
        trained["probe"] = True
        trained["budget"] = {
            "rounds": rounds,
            "train_b": tb,
            "hold_b": hb,
            "freeze_b": fb,
            "batch": cfg["batch"],
            "lbfgs_iter": cfg["lbfgs_iter"],
            "lbfgs_lr": cfg["lbfgs_lr"],
            "alpha": ALPHA,
        }
        write_json(path, trained)
        payloads.append(trained)
    med = {}
    for method in methods_for(mu):
        p = next(x for x in payloads if x["method"] == method)
        med[method] = {n: float(p["eval"][n]["med_eps"]) for n in ("6", "9", "12")}
    ratios = {"fourier": {n: ratio(med["snap"][n], med["fourier"][n]) for n in ("6", "9", "12")}}
    if "hva" in med:
        ratios["hva"] = {n: ratio(med["snap"][n], med["hva"][n]) for n in ("6", "9", "12")}
    return {"note": note, "lam": lam, "mu": mu, "rounds": rounds, "med_eps": med, "ratios": ratios, "reference_L": Lref}


def cross_weights(cfg: dict, run: Path) -> dict:
    out = {}
    for n, b in ((6, cfg["train_b"]), (9, cfg["train_b"]), (12, cfg["freeze_b"])):
        _J, K = couplings(n, 0)
        out[str(n)] = cross_cavity_weight(K)
        # touch b so a zero-size pool cannot hide; the weight is geometry, not the draw
        _ = b
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--l16", choices=("auto", "on", "off"), default="auto")
    ap.add_argument("--rounds", type=int, default=None)
    args = ap.parse_args()

    torch.set_num_threads(4)
    log("self-check")
    self_check()
    log("self-check passed")

    cfg = {
        "alpha": ALPHA,
        "lambdas": (0.5, 1.0),
        "mus": (0.0, -0.5),
        "depths": (2, 4),
        "seeds": (0, 1, 2),
        "train_b": 64,
        "hold_b": 32,
        "freeze_b": 64,
        "rounds": S.LBFGS_ROUNDS,
        "batch": 32,
        "lbfgs_iter": S.LBFGS_ITER,
        "lbfgs_lr": S.LBFGS_LR,
        "seed_formula": "1500003017 + n * 100003 + index",
    }
    run = RUN
    if args.pilot:
        cfg.update(
            {
                "lambdas": (0.5,),
                "mus": (-0.5,),
                "depths": (2,),
                "seeds": (0,),
                "train_b": 8,
                "hold_b": 4,
                "freeze_b": 4,
                "rounds": 3,
                "batch": 8,
            }
        )
        run = RUN / "pilot"
        args.l16 = "off"
    if args.rounds is not None:
        cfg["rounds"] = int(args.rounds)
    run.mkdir(parents=True, exist_ok=True)
    log(f"run dir {run}")
    log(
        f"grid λ={cfg['lambdas']} μ={cfg['mus']} L={cfg['depths']} seeds={cfg['seeds']} "
        f"rounds={cfg['rounds']} train={cfg['train_b']} hold={cfg['hold_b']} freeze={cfg['freeze_b']}"
    )
    t0 = time.time()
    try:
        run_grid(run, cfg)
    except Exception:
        traceback.print_exc()
        raise
    payloads = [p for p in load_payloads(run) if p.get("protocol") == PROTOCOL]
    # load_payloads already drops L16. Filter to this budget's depths.
    payloads = [
        p
        for p in payloads
        if int(p["L"]) in set(cfg["depths"])
        and float(p["lam"]) in set(cfg["lambdas"])
        and float(p["mu"]) in set(map(float, cfg["mus"]))
        and int(p["seed"]) in set(cfg["seeds"])
    ]
    qmap = quantum_from_disk(run, cfg)
    summary_core = aggregate(payloads, qmap)
    # Stitch the per-cell call onto the rows for the markdown.
    call_by = {
        (float(c["lam"]), float(c["mu"]), int(c["L"])): c["call"]
        for c in summary_core["decision"]["cells"]
    }
    for row in summary_core["cells"]:
        row["call"] = call_by[(float(row["lam"]), float(row["mu"]), int(row["L"]))]
    l16 = None
    if args.l16 == "on" or (
        args.l16 == "auto" and summary_core["decision"]["verdict"] in ("PASS", "needs_edit")
    ):
        # Light probe only when the smoke is complete and not an obvious KILL.
        if summary_core["decision"]["verdict"] != "KILL":
            l16 = run_l16(run, cfg, summary_core["decision"], summary_core["cells"])
    summary = {
        "protocol": PROTOCOL,
        "config": {k: (list(v) if isinstance(v, tuple) else v) for k, v in cfg.items()},
        "cross_weight": cross_weights(cfg, run),
        "decision": summary_core["decision"],
        "cells": summary_core["cells"],
        "wall_s": time.time() - t0,
        "l16": l16,
    }
    summary["reading"] = reading_for(summary["decision"], summary["cells"])
    # decision.cells duplicates rows; drop the bulky copy from the json decision
    decision_public = {k: v for k, v in summary["decision"].items() if k != "cells"}
    summary["decision"] = decision_public
    write_json(run / "summary.json", summary)
    text = render_results(summary)
    (run / "RESULTS.md").write_text(text)
    log(f"verdict {summary['decision']['verdict']} in {summary['wall_s']:.0f}s")
    log(f"wrote {run / 'RESULTS.md'}")


if __name__ == "__main__":
    main()
