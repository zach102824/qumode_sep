"""Staggered Baxter–Wu cost on the triangular torus.

    H_C = - sum_▲ ZZZ + λ sum_▼ ZZZ
    H   = H_C - Gamma * sum_i X_i

λ = 1 is the staggered texture. λ = -1 is the uniform-sign (both ferro) control.
Qubit q is bit q. The ground state is the lowest eigenvector of the full 2^N
operator. No sublattice is projected out.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.sparse.linalg import LinearOperator, eigsh

from geometry import build_triangles


def cost_diagonal(geom: dict, lam: float = 1.0) -> np.ndarray:
    n = geom["n"]
    hc = np.zeros(1 << n, dtype=np.float64)
    idx = np.arange(1 << n, dtype=np.uint32)

    def add(tris, coeff):
        for tri in tris:
            bits = np.zeros(1 << n, dtype=np.uint8)
            for v in tri:
                bits ^= ((idx >> np.uint32(v)) & np.uint32(1)).astype(np.uint8)
            hc[:] += float(coeff) * np.where(bits == 0, 1.0, -1.0)

    add(geom["up"], -1.0)
    add(geom["down"], float(lam))
    return hc


def kron_zzz(n: int, tri) -> np.ndarray:
    """Independent diagonal: qubit q is bit q."""
    Z = np.array([[1.0, 0.0], [0.0, -1.0]])
    I = np.eye(2)
    acc = np.array([[1.0]])
    wanted = set(int(v) for v in tri)
    for q in range(n - 1, -1, -1):
        acc = np.kron(acc, Z if q in wanted else I)
    return np.diag(acc).real.copy()


def explicit_cost(geom: dict, lam: float = 1.0) -> np.ndarray:
    n = geom["n"]
    hc = np.zeros(1 << n, dtype=np.float64)
    for tri in geom["up"]:
        hc += -1.0 * kron_zzz(n, tri)
    for tri in geom["down"]:
        hc += float(lam) * kron_zzz(n, tri)
    return hc


def gamma_period(hc: np.ndarray) -> dict:
    """If H_C takes values in δℤ, γ → γ+2π/δ is a global phase on the cost unitary."""
    scaled = np.rint(hc * 2.0)
    if float(np.max(np.abs(hc * 2.0 - scaled))) > 1e-6:
        raise RuntimeError("cost diagonal is not in (1/2)Z")
    scaled = scaled.astype(np.int64)
    base = int(scaled.min())
    g = 0
    for v in np.unique(scaled):
        g = math.gcd(g, int(v) - base)
    delta = g / 2.0
    return {
        "delta": float(delta),
        "gcd_half_integers": int(g),
        "T_gamma": float(2.0 * math.pi / delta),
        "beta_period": float(math.pi),
    }


def apply_H(v, hc, gamma, n):
    out = hc * v
    idx = np.arange(v.shape[0], dtype=np.int64)
    for q in range(n):
        out = out - float(gamma) * v[idx ^ (1 << q)]
    return out


def ground_dense(hc, gamma, n):
    dim = 1 << n
    H = np.diag(hc.astype(np.float64))
    idx = np.arange(dim)
    for q in range(n):
        partner = idx ^ (1 << q)
        upper = partner > idx
        ii, jj = idx[upper], partner[upper]
        H[ii, jj] -= float(gamma)
        H[jj, ii] -= float(gamma)
    evals, evecs = np.linalg.eigh(H)
    gs = np.real(evecs[:, 0]).copy()
    if gs.sum() < 0:
        gs *= -1.0
    gs /= np.linalg.norm(gs)
    return {
        "E0": float(evals[0]),
        "E1": float(evals[1]),
        "Emax": float(evals[-1]),
        "gap": float(evals[1] - evals[0]),
        "evals": evals.real.copy(),
        "gs": gs,
        "residual": float(np.linalg.norm(H @ gs - evals[0] * gs)),
        "method": "dense_eigh",
    }


def ground_lanczos(hc, gamma, n, k=2):
    dim = 1 << n

    def matvec(v):
        return apply_H(np.asarray(v, dtype=np.float64), hc, gamma, n)

    op = LinearOperator((dim, dim), matvec=matvec, dtype=np.float64)
    evals, evecs = eigsh(op, k=k, which="SA", tol=1e-10, maxiter=max(4000, 10 * n))
    order = np.argsort(evals.real)
    e0 = float(evals.real[order[0]])
    e1 = float(evals.real[order[1]])
    gs = np.real(evecs[:, order[0]]).astype(np.float64)
    gs /= np.linalg.norm(gs)
    if gs.sum() < 0:
        gs *= -1.0
    hi = eigsh(op, k=1, which="LA", tol=1e-8, maxiter=max(3000, 8 * n), return_eigenvectors=False)
    return {
        "E0": e0,
        "E1": e1,
        "Emax": float(hi[0]),
        "gap": e1 - e0,
        "gs": gs,
        "residual": float(np.linalg.norm(matvec(gs) - e0 * gs)),
        "method": "eigsh",
    }


def ground_state(hc, gamma, n):
    if n <= 12:
        return ground_dense(hc, gamma, n)
    return ground_lanczos(hc, gamma, n)


def classical_manifold(hc: np.ndarray) -> dict:
    emin = float(hc.min())
    idx = np.flatnonzero(np.isclose(hc, emin))
    return {"Emin": emin, "Emax_diag": float(hc.max()), "degeneracy": int(idx.size), "indices": idx}


def manifold_weight(psi, indices) -> float:
    p = np.abs(np.asarray(psi)) ** 2
    return float(p[indices].sum())
