"""Independent replay. Imports none of the round runners.

Rebuilds J from the seed, the maximin triple partition, K-hat and M-hat,
and the cavity XX and XXX mixers as sparse Pauli strings. Evolves |+> with
expm_multiply. Ground space from eigsh.
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
PACKED = {"nodata_mm", "K_mm", "KM_mm"}


def pauli(N, ops):
    out = sp.identity(1, format="csr")
    for site in range(N - 1, -1, -1):
        out = sp.kron(out, ops.get(site, I2), format="csr")
    return out


def couplings(N, k):
    g = np.random.default_rng(np.random.SeedSequence([20261115, N, k]))
    J = np.zeros((N, N))
    for i in range(N):
        for j in range(i + 1, N):
            J[i, j] = J[j, i] = float(g.normal())
    return J


def maximin_packing(J):
    """Independent copy of the maximin rule. Not the ferromagnetic-sum search."""
    J = np.asarray(J, dtype=np.float64)
    N = int(J.shape[0])
    best, best_score = None, -1.0e300

    def rec(rem, acc):
        nonlocal best, best_score
        if not rem:
            part = tuple(sorted(tuple(sorted(t)) for t in acc))
            score = min(-float(J[t[0], t[1]] + J[t[0], t[2]] + J[t[1], t[2]]) for t in part)
            if score > best_score + 1e-12 or (abs(score - best_score) <= 1e-12 and (best is None or part < best)):
                best, best_score = part, score
            return
        s0, rest = rem[0], rem[1:]
        for i in range(len(rest)):
            for j in range(i + 1, len(rest)):
                rec([rest[k] for k in range(len(rest)) if k != i and k != j], acc + [(s0, rest[i], rest[j])])

    rec(list(range(N)), [])
    return [list(t) for t in best]


def interleaved(N):
    return [[a, a + N // 3, a + 2 * N // 3] for a in range(N // 3)]


def offdiag_rms(C):
    iu = np.triu_indices(C.shape[0], 1)
    return float(np.sqrt(np.mean(C[iu] ** 2)))


def features(J, packing):
    K = 0.5 * ((J @ J) + (J @ J).T)
    np.fill_diagonal(K, 0.0)
    M = np.zeros_like(J)
    m = len(packing)
    for p in range(m):
        for q in range(p + 1, m):
            mean = float(np.mean([J[i, j] for i in packing[p] for j in packing[q]]))
            for i in packing[p]:
                for j in packing[q]:
                    M[i, j] = M[j, i] = mean
    rj = offdiag_rms(J)
    return K * (rj / offdiag_rms(K)), M * (rj / offdiag_rms(M))


def zz_sum(C, N):
    H = sp.csr_matrix((1 << N, 1 << N))
    for i in range(N):
        for j in range(i + 1, N):
            if C[i, j] != 0.0:
                H = H + C[i, j] * pauli(N, {i: Z2, j: Z2})
    return H


def build(N, k, packing):
    J = couplings(N, k)
    Kh, Mh = features(J, packing)
    Xs = [pauli(N, {i: X2}) for i in range(N)]
    SX = sum(Xs[1:], Xs[0])
    XX = sp.csr_matrix((1 << N, 1 << N))
    XXX = sp.csr_matrix((1 << N, 1 << N))
    for a, b, c in packing:
        XX = XX + pauli(N, {a: X2, b: X2}) + pauli(N, {a: X2, c: X2}) + pauli(N, {b: X2, c: X2})
        XXX = XXX + pauli(N, {a: X2, b: X2, c: X2})
    Hc = zz_sum(J, N)
    return {"Hc": Hc, "HK": zz_sum(Kh, N), "HM": zz_sum(Mh, N), "SX": SX, "XX": XX, "XXX": XXX,
            "H": (Hc + 0.1 * SX).tocsr(), "N": N}


def layer(ops, diag, c0, c1, c2, psi):
    Bm = c0 * ops["SX"] + c1 * ops["XX"] + c2 * ops["XXX"]
    return expm_multiply(-1j * Bm.tocsc(), expm_multiply(-1j * diag.tocsc(), psi))


def state(ops, fam, x, L):
    x = np.asarray(x, float)
    N = ops["N"]
    psi = np.full(1 << N, 2.0 ** (-N / 2), dtype=complex)
    if fam == "qaoa":
        for ell in range(L):
            psi = layer(ops, x[ell] * ops["Hc"], x[L + ell], 0.0, 0.0, psi)
        return psi
    head = {"nodata": 1, "nodata_mm": 1, "K": 2, "K_mm": 2, "KM": 3, "KM_mm": 3}[fam]
    b = float(x[1]) if head > 1 else 0.0
    c = float(x[2]) if head > 2 else 0.0
    for ell in range(L):
        D = x[0] * ops["Hc"] + b * ops["HK"] + c * ops["HM"]
        c0, c1, c2 = x[head + 3 * ell: head + 3 * ell + 3]
        psi = layer(ops, D, c0, c1, c2, psi)
    return psi


def ground(ops):
    kk = 8
    while True:
        w, V = eigsh(ops["H"], k=kk, which="SA", tol=1e-12)
        o = np.argsort(w)
        w, V = w[o], V[:, o]
        if w[-1] - w[0] > 1e-4 or kk >= 32:
            return w, V
        kk *= 2


def replay(key):
    N, k = key
    p = HERE / "eval" / f"n{N}_k{k}.json"
    if not p.exists():
        return None
    ev = json.loads(p.read_text())
    J = couplings(N, k)
    part = maximin_packing(J)
    if part != ev["packing_maximin"]:
        raise SystemExit(f"independent packing disagrees N={N} k={k}")
    ops = {True: build(N, k, part), False: build(N, k, interleaved(N))}
    w, V = ground(ops[False])
    E0 = float(w[0])
    wins = {}
    for name, tol in (("1e-08", 1e-8), ("1e-06", 1e-6), ("1e-04", 1e-4)):
        wins[name] = V[:, w - E0 <= tol]
    rows = []
    keep = set(PREREG["replay_families"])
    for c in ev["cells"]:
        if c["family"] not in keep:
            continue
        if c["family"] in ("KM_gl", "KM_pack", "KM_fid", "KM_fe"):
            raise SystemExit("forbidden family is not replayed")
        psi = state(ops[c["family"] in PACKED], c["family"], c["x_final"], c["L"])
        E = float(np.vdot(psi, ops[False]["H"] @ psi).real)
        Fs = {t: float(np.sum(np.abs(W.conj().T @ psi) ** 2)) for t, W in wins.items()}
        rows.append({"r": c["r"], "family": c["family"], "L": c["L"], "E_replay": E,
                     "dE_rel": (E - c["E"]) / max(1.0, abs(E)), "dF_sub": Fs["1e-06"] - c["F_sub"],
                     "F_sub_replay": Fs["1e-06"], "F_sub_1e-8": Fs["1e-08"], "F_sub_1e-4": Fs["1e-04"],
                     "norm_err": float(abs(np.linalg.norm(psi) - 1.0))})
    res = {"N": N, "k": k, "E0_replay": E0, "dE0": E0 - ev["E0"], "n_sub_replay": int(wins["1e-06"].shape[1]),
           "n_sub_main": ev["n_sub"], "gs_residual": float(np.linalg.norm(ops[False]["H"] @ V[:, 0] - E0 * V[:, 0])),
           "cells": rows}
    (HERE / "replay").mkdir(exist_ok=True)
    (HERE / "replay" / f"n{N}_k{k}.json").write_text(json.dumps(res) + "\n")
    return key


def main():
    keys = [(N, k) for N in (12, 9, 6) for k in PREREG["k_confirm"]]
    with Pool(int(PREREG["workers"])) as pool:
        done = [x for x in pool.map(replay, keys, chunksize=1) if x]
    print("replayed", len(done))


if __name__ == "__main__":
    main()
