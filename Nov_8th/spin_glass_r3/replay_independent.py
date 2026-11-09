"""Independent replay of selected states. Imports nothing from the r1/r2/r3 runners or Nov_3rd.

Built from definitions: Pauli strings via scipy.sparse.kron, H = sum_{i<j} J_ij Z_i Z_j + 0.1 sum_i X_i,
layer = exp(-i [a Hc + b Khat_ZZ + c Mhat_ZZ]) then exp(-i [c0 sum X + c1 sum_cav X_a X_b + c2 sum_cav X_a X_b X_c]),
evolved with expm_multiply from |+>^N; ground space from eigsh.
"""
from __future__ import annotations

import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh, expm_multiply

HERE = Path(__file__).resolve().parent
PREREG = json.loads((HERE / "PREREG.json").read_text())
I2 = sp.identity(2, format="csr")
X2 = sp.csr_matrix(np.array([[0.0, 1.0], [1.0, 0.0]]))
Z2 = sp.csr_matrix(np.array([[1.0, 0.0], [0.0, -1.0]]))


def pauli(N, ops):
    """ops: dict site -> 2x2. Site 0 is the least significant bit (rightmost kron factor)."""
    out = sp.identity(1, format="csr")
    for site in range(N - 1, -1, -1):
        out = sp.kron(out, ops.get(site, I2), format="csr")
    return out


def couplings(N, k):
    g = np.random.default_rng(np.random.SeedSequence([20261115, N, k]))
    J = np.zeros((N, N))
    for i in range(N):
        for j in range(i + 1, N):
            J[i, j] = J[j, i] = g.normal()
    return J


def offdiag_rms(C):
    iu = np.triu_indices(C.shape[0], 1)
    return float(np.sqrt(np.mean(C[iu] ** 2)))


def features(J):
    N = J.shape[0]
    K = J @ J
    K = 0.5 * (K + K.T)
    np.fill_diagonal(K, 0.0)
    m = N // 3
    cav = [[a, a + m, a + 2 * m] for a in range(m)]
    M = np.zeros_like(J)
    for p in range(m):
        for q in range(p + 1, m):
            mean = np.mean([J[i, j] for i in cav[p] for j in cav[q]])
            for i in cav[p]:
                for j in cav[q]:
                    M[i, j] = M[j, i] = mean
    rj = offdiag_rms(J)
    return K * (rj / offdiag_rms(K)), M * (rj / offdiag_rms(M)), cav


def zz_sum(C, N):
    H = sp.csr_matrix((1 << N, 1 << N))
    for i in range(N):
        for j in range(i + 1, N):
            if C[i, j] != 0.0:
                H = H + C[i, j] * pauli(N, {i: Z2, j: Z2})
    return H


def build(N, k):
    J = couplings(N, k)
    Kh, Mh, cav = features(J)
    Xs = [pauli(N, {i: X2}) for i in range(N)]
    SX = sum(Xs[1:], Xs[0])
    XX = sp.csr_matrix((1 << N, 1 << N))
    XXX = sp.csr_matrix((1 << N, 1 << N))
    for a, b, c in cav:
        XX = XX + pauli(N, {a: X2, b: X2}) + pauli(N, {a: X2, c: X2}) + pauli(N, {b: X2, c: X2})
        XXX = XXX + pauli(N, {a: X2, b: X2, c: X2})
    Hc, HK, HM = zz_sum(J, N), zz_sum(Kh, N), zz_sum(Mh, N)
    return {"N": N, "J": J, "Hc": Hc, "HK": HK, "HM": HM, "SX": SX, "XX": XX, "XXX": XXX,
            "H": (Hc + 0.1 * SX).tocsr(), "sJ": float(np.sqrt(np.mean(np.sum(J * J, axis=1))))}


def state(ops, fam, x, L):
    N = ops["N"]
    x = np.asarray(x, float)
    psi = np.full(1 << N, 2.0 ** (-N / 2), dtype=complex)
    qaoa = fam.startswith("qaoa")
    head = {"nodata": 1, "K": 2, "KM": 3}.get(fam, 0)
    for l in range(L):
        if qaoa:
            D = x[l] * ops["Hc"]
            Bm = x[L + l] * ops["SX"]
        else:
            b = x[1] if head > 1 else 0.0
            c = x[2] if head > 2 else 0.0
            D = x[0] * ops["Hc"] + b * ops["HK"] + c * ops["HM"]
            c0, c1, c2 = x[head + 3 * l: head + 3 * l + 3]
            Bm = c0 * ops["SX"] + c1 * ops["XX"] + c2 * ops["XXX"]
        psi = expm_multiply(-1j * D.tocsc(), psi)
        psi = expm_multiply(-1j * Bm.tocsc(), psi)
    return psi


def ground(ops):
    H = ops["H"]
    kk = 8
    while True:
        w, V = eigsh(H, k=kk, which="SA", tol=1e-12)
        o = np.argsort(w); w, V = w[o], V[:, o]
        if w[-1] - w[0] > 1e-4 or kk >= 32:
            return w, V
        kk *= 2


def replay(key):
    N, k = key
    p = HERE / "eval" / f"n{N}_k{k}.json"
    if not p.exists():
        return None
    ev = json.loads(p.read_text())
    ops = build(N, k)
    w, V = ground(ops)
    E0 = float(w[0])
    dim = 1 << N
    flip = np.arange(dim) ^ (dim - 1)
    wins = {}
    for name, tol in (("1e-08", 1e-8), ("1e-06", 1e-6), ("1e-04", 1e-4)):
        wins[name] = V[:, w - E0 <= tol]
    sub6 = wins["1e-06"]
    par = [float(np.vdot(sub6[:, i], sub6[flip, i]).real) for i in range(sub6.shape[1])]
    rows = []
    for c in ev["cells"]:
        psi = state(ops, c["family"], c["x_final"], c["L"])
        Hpsi = ops["H"] @ psi
        E = float(np.vdot(psi, Hpsi).real)
        Fs = {t: float(np.sum(np.abs(W.conj().T @ psi) ** 2)) for t, W in wins.items()}
        spin_flip_exp = float(np.vdot(psi, psi[flip]).real)
        rows.append({"r": c["r"], "family": c["family"], "L": c["L"], "E_replay": E,
                     "dE": E - c["E"], "dE_eval": E - c["E_check"], "F_sub_replay": Fs["1e-06"],
                     "dF_sub": Fs["1e-06"] - c["F_sub"], "F_sub_1e-8": Fs["1e-08"], "F_sub_1e-4": Fs["1e-04"],
                     "norm_err": float(abs(np.linalg.norm(psi) - 1.0)), "spin_flip_parity": spin_flip_exp})
    res = {"N": N, "k": k, "E0_replay": E0, "dE0": E0 - ev["E0"], "n_sub_replay": int(sub6.shape[1]),
           "n_sub_main": ev["n_sub"], "lowest_levels": [float(v) for v in w[:6]],
           "window_spin_parity": par, "gs_residual": float(np.linalg.norm(ops["H"] @ V[:, 0] - E0 * V[:, 0])),
           "cells": rows}
    (HERE / "replay").mkdir(exist_ok=True)
    (HERE / "replay" / f"n{N}_k{k}.json").write_text(json.dumps(res) + "\n")
    return key


def main():
    (HERE / "replay").mkdir(exist_ok=True)
    ks = PREREG["k_primary"] + PREREG["k_secondary"]
    keys = [(N, k) for N in (12, 9, 6) for k in ks]
    with Pool(int(PREREG["workers"])) as p:
        done = [x for x in p.map(replay, keys, chunksize=1) if x]
    print("replayed", len(done))


if __name__ == "__main__":
    main()
