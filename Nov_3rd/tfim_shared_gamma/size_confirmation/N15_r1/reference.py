"""Matrix-free N=15 reference. No dense 32768-by-32768 matrix."""

from __future__ import annotations

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import LinearOperator, eigsh

import harness as h


def apply_h(target: h.Target, psi: np.ndarray) -> np.ndarray:
    return target.hc * psi + target.B * h.apply_sum_x(psi, target.n)


def sparse_h(target: h.Target) -> csr_matrix:
    """Independent Hamiltonian: diagonal ZZ plus one X per qubit."""
    n = target.n
    dim = 1 << n
    idx = np.arange(dim, dtype=np.int64)
    rows = [idx]
    cols = [idx]
    data = [target.hc.real.copy()]
    for q in range(n):
        dest = idx ^ (1 << q)
        rows.append(idx)
        cols.append(dest)
        data.append(np.full(dim, target.B, dtype=np.float64))
    return csr_matrix(
        (np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
        shape=(dim, dim),
    )


def ground_state(target: h.Target, k: int = 6, seed: int = 15) -> dict:
    dim = target.dim
    op = LinearOperator(
        (dim, dim),
        matvec=lambda v: apply_h(target, np.asarray(v, dtype=np.complex128)).real,
        dtype=np.float64,
    )
    rng = np.random.default_rng(seed)
    v0 = rng.normal(size=dim)
    v0 /= np.linalg.norm(v0)
    evals, evecs = eigsh(op, k=k, which="SA", v0=v0, tol=1e-12, maxiter=2000)
    order = np.argsort(evals)
    evals = evals[order]
    evecs = evecs[:, order]
    gs = evecs[:, 0].astype(np.complex128)
    gs /= np.linalg.norm(gs)
    residual = float(np.linalg.norm(apply_h(target, gs) - evals[0] * gs))
    parity = float(np.vdot(gs, gs[::-1]).real)
    excited = []
    for i in range(1, evals.size):
        v = evecs[:, i].astype(np.complex128)
        v /= np.linalg.norm(v)
        excited.append({
            "gap": float(evals[i] - evals[0]),
            "parity": float(np.vdot(v, v[::-1]).real),
            "residual": float(np.linalg.norm(apply_h(target, v) - evals[i] * v)),
        })
    same = next(row["gap"] for row in excited if row["parity"] > 0.5)
    return {
        "E0": float(evals[0]),
        "evals": [float(v) for v in evals],
        "gs": gs,
        "residual": residual,
        "parity": parity,
        "full_gap": float(evals[1] - evals[0]),
        "excited": excited,
        "same_parity_gap": same,
        "method": "scipy.sparse.linalg.eigsh LinearOperator, which=SA, tol=1e-12",
        "k": k,
    }
