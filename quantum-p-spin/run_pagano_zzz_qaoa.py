"""Milestone 2: Pagano-like QAOA with 3-body ZZZ cost (ZZ→ZZZ).

H = H_A + H_B = Σ_{i<j<k} J_ijk Z_i Z_j Z_k + B Σ_i X_i
J_ijk = c / r_ijk^α > 0, r_ijk = max(|i-j|,|i-k|,|j-k|) open 1D,
c chosen so mean of nearest triples (r=2) equals J0=1.
Init |ψ0⟩ = |+⟩_x^{⊗N} = GS of H_B (B<0).
QAOA same form as Pagano with H_B/J0, H_A/J0.
"""
from __future__ import annotations

import argparse
import json
import time
from itertools import combinations
from pathlib import Path

import numpy as np
from numpy.linalg import eigh
from scipy.optimize import minimize

from run_pagano_af_qaoa import (
    eta_score, energy, eigh_pack, apply_U, apply_qaoa_layers,
    half_cut_entropy, write_results,
)

PKG = Path(__file__).resolve().parent
OUT_DEFAULT = PKG / "runs" / "compare" / "pagano_zzz_qaoa"


def plus_x_state(n: int) -> np.ndarray:
    s = np.array([1.0, 1.0], dtype=np.complex128) / np.sqrt(2.0)
    psi = np.array([1.0], dtype=np.complex128)
    for _ in range(n):
        psi = np.kron(psi, s)
    return psi


def build_HB_X(n: int, B: float) -> np.ndarray:
    dim = 1 << n
    H = np.zeros((dim, dim), dtype=np.float64)
    idx = np.arange(dim)
    for i in range(n):
        mi = 1 << (n - 1 - i)
        partner = idx ^ mi
        H[idx, partner] += float(B)
    return H  # real symmetric


def triple_r(i, j, k):
    return max(abs(i - j), abs(i - k), abs(j - k))


def build_J3(n: int, alpha: float = 1.0, J0: float = 1.0):
    triples = np.asarray(list(combinations(range(n), 3)), dtype=np.int64)
    w = np.array([triple_r(int(i), int(j), int(k)) ** (-alpha) for i, j, k in triples], dtype=np.float64)
    # nearest triples have r=2 (e.g. 0,1,2)
    nn_mask = np.array([triple_r(int(i), int(j), int(k)) == 2 for i, j, k in triples])
    avg_nn = float(w[nn_mask].mean()) if nn_mask.any() else float(w.mean())
    J3 = w * (float(J0) / avg_nn)
    return triples, J3, float(avg_nn * (J0 / avg_nn))


def build_HA_ZZZ(n: int, triples, J3) -> np.ndarray:
    dim = 1 << n
    diag = np.zeros(dim, dtype=np.float64)
    x = np.arange(dim, dtype=np.int64)
    for (i, j, k), jv in zip(triples, J3):
        def z(b):
            return np.where(((x >> (n - 1 - b)) & 1) == 0, 1.0, -1.0)
        diag += float(jv) * z(int(i)) * z(int(j)) * z(int(k))
    return np.diag(diag)


def optimize_p1(packs_A, packs_B, H, J0, psi0, Egs, Emax, gs, grid=31, seed=0):
    def E_of(beta, gamma):
        psi = apply_qaoa_layers(psi0, packs_A, packs_B, J0, [beta], [gamma])
        return energy(psi, H)

    best_E, best_bg = np.inf, (0.5, 0.3)
    betas = np.linspace(0.0, np.pi, grid)
    gammas = np.linspace(0.0, np.pi, grid)
    landscape = np.empty((grid, grid))
    for ib, b in enumerate(betas):
        for ig, g in enumerate(gammas):
            E = E_of(b, g)
            landscape[ib, ig] = E
            if E < best_E:
                best_E, best_bg = E, (float(b), float(g))

    starts = [best_bg]
    flat = landscape.ravel()
    k = min(12, flat.size)
    for t in np.argpartition(flat, k - 1)[:k]:
        ib, ig = divmod(int(t), grid)
        starts.append((float(betas[ib]), float(gammas[ig])))
    rng = np.random.default_rng(seed)
    for _ in range(6):
        starts.append((float(rng.uniform(0, np.pi)), float(rng.uniform(0, np.pi))))

    opt_E, opt_bg = best_E, best_bg
    for x0 in starts:
        res = minimize(
            lambda x: E_of(float(x[0]), float(x[1])),
            x0=np.asarray(x0), method="L-BFGS-B",
            bounds=[(0.0, 2 * np.pi), (0.0, 2 * np.pi)],
            options={"maxiter": 60, "ftol": 1e-14},
        )
        if res.fun < opt_E:
            opt_E = float(res.fun)
            opt_bg = (float(res.x[0]) % (2 * np.pi), float(res.x[1]) % (2 * np.pi))

    psi = apply_qaoa_layers(psi0, packs_A, packs_B, J0, [opt_bg[0]], [opt_bg[1]])
    E_sv = energy(psi, H)
    E_init = energy(psi0, H)
    return {
        "beta": opt_bg[0], "gamma": opt_bg[1],
        "E_sv": E_sv, "E_init": E_init,
        "eta_sv": eta_score(E_sv, Egs, Emax),
        "eta_init": eta_score(E_init, Egs, Emax),
        "F_gs": float(abs(np.vdot(gs, psi)) ** 2),
    }


def run_one(n, B_over_J0, alpha, J0, seed, grid):
    t0 = time.time()
    B = float(B_over_J0) * float(J0)
    triples, J3, avg_nn = build_J3(n, alpha, J0)
    HA = build_HA_ZZZ(n, triples, J3)
    HB = build_HB_X(n, B)
    H = HA.astype(np.complex128) + HB.astype(np.complex128)
    evals, evecs = eigh(H)
    Egs, Emax = float(evals[0].real), float(evals[-1].real)
    gs = evecs[:, 0]
    k = int(np.argmax(np.abs(gs)))
    gs = gs * (np.abs(gs[k]) / (gs[k] + 1e-30))
    S = half_cut_entropy(gs, n)
    psi0 = plus_x_state(n)
    packs_A = eigh_pack(HA.astype(np.complex128))
    packs_B = eigh_pack(HB.astype(np.complex128))
    ov = float(abs(np.vdot(packs_B[1][:, 0], psi0)) ** 2)
    p1 = optimize_p1(packs_A, packs_B, H, J0, psi0, Egs, Emax, gs, grid=grid, seed=seed)
    return {
        "model": "AF_ZZZ",
        "n": n, "p": 1, "alpha": alpha, "J0": J0,
        "B_over_J0": B_over_J0, "B": B,
        "Egs": Egs, "Emax": Emax, "S_half": S,
        "avg_NN": avg_nn, "ov_psi0_HB_gs": ov,
        "beta": p1["beta"], "gamma": p1["gamma"],
        "E_sv": p1["E_sv"], "E_init": p1["E_init"],
        "eta_sv": p1["eta_sv"], "eta_init": p1["eta_init"],
        "F_gs": p1["F_gs"],
        "seconds": time.time() - t0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ns", default="6,9,12")
    ap.add_argument("--Bs", default="-0.1,-0.3,-0.8")
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--J0", type=float, default=1.0)
    ap.add_argument("--grid", type=int, default=31)
    ap.add_argument("--out", default=str(OUT_DEFAULT))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    ns = [int(x) for x in args.ns.split(",") if x.strip()]
    Bs = [float(x) for x in args.Bs.split(",") if x.strip()]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log = out / "run.log"

    def say(msg):
        line = f"{time.strftime('%H:%M:%S')} [zzz] {msg}"
        print(line, flush=True)
        with log.open("a") as f:
            f.write(line + "\n")

    rows = []
    say(f"start ns={ns} Bs={Bs} alpha={args.alpha}")
    for n in ns:
        for B in Bs:
            say(f"n={n} B/J0={B:g}")
            r = run_one(n, B, args.alpha, args.J0, args.seed, args.grid)
            rows.append(r)
            say(
                f"  η={r['eta_sv']:.4f} (init {r['eta_init']:.4f}) S={r['S_half']:.4f} "
                f"β={r['beta']:.3f} γ={r['gamma']:.3f} ({r['seconds']:.1f}s)"
            )
            write_results(rows, out, title="Pagano-like ZZZ QAOA (ZZ→ZZZ)")
    write_results(rows, out, title="Pagano-like ZZZ QAOA (ZZ→ZZZ)")
    say(f"done → {out}")


if __name__ == "__main__":
    main()
