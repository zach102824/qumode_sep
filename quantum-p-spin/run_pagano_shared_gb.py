"""Milestone 3: shared-γ + per-layer β (shared_gb) on Pagano AF XX TFIM.

Same H as run_pagano_af_qaoa (AF XX + B Y).
Ansatz p=L layers: one shared γ, β_ℓ per layer.
  |ψ⟩ = ∏_{ℓ=1}^L exp(-i β_ℓ H_B/J0) exp(-i γ H_A/J0) |+y⟩^⊗N
Compare vs fully free (β_ℓ, γ_ℓ) p=L QAOA at small L.
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
from numpy.linalg import eigh
from scipy.optimize import minimize

from run_pagano_af_qaoa import (
    build_J, build_HA, build_HB, plus_y_state, half_cut_entropy,
    eta_score, energy, eigh_pack, apply_qaoa_layers,
)

PKG = Path(__file__).resolve().parent
OUT_DEFAULT = PKG / "runs" / "compare" / "pagano_shared_gb"


def optimize_shared_gb(packs_A, packs_B, H, J0, psi0, Egs, Emax, L, seed=0, n_starts=8):
    """Params: [γ, β_0, ..., β_{L-1}] — L+1 params."""
    rng = np.random.default_rng(seed)

    def fun(x):
        gamma = float(x[0])
        betas = x[1:]
        gammas = np.full(L, gamma)
        psi = apply_qaoa_layers(psi0, packs_A, packs_B, J0, betas, gammas)
        return energy(psi, H)

    bounds = [(0.0, 2 * np.pi)] * (L + 1)
    best = None
    starts = []
    # heuristic: γ small, β ~ π/3 / L scale
    for s in range(n_starts):
        g0 = float(rng.uniform(0.05, 0.8))
        b0 = rng.uniform(0.2, 1.5, size=L)
        starts.append(np.concatenate([[g0], b0]))
    # also zeros-ish
    starts.append(np.full(L + 1, 0.3))

    for x0 in starts:
        res = minimize(
            fun, x0=x0, method="L-BFGS-B", bounds=bounds,
            options={"maxiter": 120, "ftol": 1e-14},
        )
        if best is None or res.fun < best[0]:
            best = (float(res.fun), res.x.copy())

    gamma = float(best[1][0])
    betas = best[1][1:].tolist()
    psi = apply_qaoa_layers(psi0, packs_A, packs_B, J0, betas, [gamma] * L)
    E = energy(psi, H)
    return {
        "gamma": gamma, "betas": betas,
        "E": E, "eta": eta_score(E, Egs, Emax),
        "n_params": L + 1,
    }


def optimize_free_qaoa(packs_A, packs_B, H, J0, psi0, Egs, Emax, L, seed=0, n_starts=6):
    """2L free angles."""
    rng = np.random.default_rng(seed + 7)

    def fun(x):
        betas = x[:L]
        gammas = x[L:]
        psi = apply_qaoa_layers(psi0, packs_A, packs_B, J0, betas, gammas)
        return energy(psi, H)

    bounds = [(0.0, 2 * np.pi)] * (2 * L)
    best = None
    for _ in range(n_starts):
        x0 = rng.uniform(0.05, 1.2, size=2 * L)
        res = minimize(
            fun, x0=x0, method="L-BFGS-B", bounds=bounds,
            options={"maxiter": 150, "ftol": 1e-14},
        )
        if best is None or res.fun < best[0]:
            best = (float(res.fun), res.x.copy())

    betas = best[1][:L].tolist()
    gammas = best[1][L:].tolist()
    psi = apply_qaoa_layers(psi0, packs_A, packs_B, J0, betas, gammas)
    E = energy(psi, H)
    return {
        "betas": betas, "gammas": gammas,
        "E": E, "eta": eta_score(E, Egs, Emax),
        "n_params": 2 * L,
    }


def run_cell(n, B_over_J0, L, alpha, J0, seed):
    t0 = time.time()
    B = float(B_over_J0) * float(J0)
    J = build_J(n, alpha, J0)
    HA = build_HA(n, J)
    HB = build_HB(n, B)
    H = HA.astype(np.complex128) + HB
    evals, evecs = eigh(H)
    Egs, Emax = float(evals[0].real), float(evals[-1].real)
    gs = evecs[:, 0]
    S = half_cut_entropy(gs, n)
    psi0 = plus_y_state(n)
    packs_A = eigh_pack(HA.astype(np.complex128))
    packs_B = eigh_pack(HB)
    E_init = energy(psi0, H)

    sg = optimize_shared_gb(packs_A, packs_B, H, J0, psi0, Egs, Emax, L, seed=seed)
    fr = optimize_free_qaoa(packs_A, packs_B, H, J0, psi0, Egs, Emax, L, seed=seed)
    return {
        "n": n, "L": L, "B_over_J0": B_over_J0, "alpha": alpha,
        "Egs": Egs, "Emax": Emax, "S_half": S,
        "E_init": E_init, "eta_init": eta_score(E_init, Egs, Emax),
        "shared_gb_eta": sg["eta"], "shared_gb_E": sg["E"],
        "shared_gb_gamma": sg["gamma"], "shared_gb_betas": sg["betas"],
        "free_eta": fr["eta"], "free_E": fr["E"],
        "free_betas": fr["betas"], "free_gammas": fr["gammas"],
        "seconds": time.time() - t0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ns", default="6,9,12")
    ap.add_argument("--Ls", default="1,2,4")
    ap.add_argument("--Bs", default="-0.3")
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--J0", type=float, default=1.0)
    ap.add_argument("--out", default=str(OUT_DEFAULT))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    ns = [int(x) for x in args.ns.split(",") if x.strip()]
    Ls = [int(x) for x in args.Ls.split(",") if x.strip()]
    Bs = [float(x) for x in args.Bs.split(",") if x.strip()]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log = out / "run.log"

    def say(msg):
        line = f"{time.strftime('%H:%M:%S')} [sgb] {msg}"
        print(line, flush=True)
        with log.open("a") as f:
            f.write(line + "\n")

    rows = []
    say(f"start ns={ns} Ls={Ls} Bs={Bs}")
    for n in ns:
        for B in Bs:
            for L in Ls:
                say(f"n={n} L={L} B/J0={B:g}")
                r = run_cell(n, B, L, args.alpha, args.J0, args.seed)
                rows.append(r)
                say(
                    f"  shared_gb η={r['shared_gb_eta']:.4f}  free η={r['free_eta']:.4f} "
                    f"init={r['eta_init']:.4f} γ={r['shared_gb_gamma']:.3f} ({r['seconds']:.1f}s)"
                )
                # incremental
                _write(rows, out)
    _write(rows, out)
    say(f"done → {out}")


def _write(rows, out):
    fields = [
        "n", "L", "B_over_J0", "alpha", "Egs", "Emax", "S_half",
        "eta_init", "shared_gb_eta", "free_eta", "shared_gb_E", "free_E",
        "shared_gb_gamma", "seconds",
    ]
    with (out / "summary.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fields})
    (out / "summary.json").write_text(json.dumps(rows, indent=2, default=str))

    lines = [
        "# Pagano AF XX — shared_gb vs free QAOA",
        "",
        "H = Σ J_ij XX + B Σ Y, J=J0/|i-j|, α=1, B/J0 from grid.",
        "shared_gb: one γ + β_ℓ per layer (L+1 params).",
        "free: (β_ℓ, γ_ℓ) fully free (2L params).",
        "",
        "| n | L | B/J0 | η_shared_gb | η_free | η_init | γ* |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| {r['n']} | {r['L']} | {r['B_over_J0']:g} | {r['shared_gb_eta']:.4f} | "
            f"{r['free_eta']:.4f} | {r['eta_init']:.4f} | {r['shared_gb_gamma']:.4f} |"
        )
    lines += ["", f"Generated {time.strftime('%Y-%m-%d %H:%M:%S %Z')}", ""]
    (out / "RESULTS.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
