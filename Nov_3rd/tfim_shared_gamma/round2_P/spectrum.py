"""Full-space spectrum of the frozen target. No sector substitution for fidelity.

The accessible-sector gap is reported separately. A two-eigenvalue sparse
gap is not the full gap; Round 1 saw ~0.43 where the full gap is ~0.0016.
"""

from __future__ import annotations

import numpy as np

from target import z_table


def dense_hamiltonian(hc: np.ndarray, B: float, n: int) -> np.ndarray:
    dim = hc.shape[0]
    H = np.diag(np.asarray(hc, dtype=np.float64))
    x = np.arange(dim)
    for q in range(n):
        H[x, x ^ (1 << q)] = float(B)
    return H


def parity_expectation(psi: np.ndarray, n: int) -> float:
    # prod X sends x to x with all n bits flipped.
    return float(np.vdot(psi, psi[::-1]).real) if n == int(np.round(np.log2(psi.shape[0]))) else float("nan")


def reflection_indices(n: int) -> np.ndarray:
    dim = 1 << n
    x = np.arange(dim, dtype=np.uint32)
    y = np.zeros(dim, dtype=np.uint32)
    for src in range(n):
        bit = (x >> np.uint32(src)) & np.uint32(1)
        y |= bit << np.uint32(n - 1 - src)
    return y.astype(np.int64)


def reflection_expectation(psi: np.ndarray, ridx: np.ndarray) -> float:
    return float(np.vdot(psi, psi[ridx]).real)


def half_chain_entropy(psi: np.ndarray, n: int) -> float:
    nA = n // 2
    # Little-endian: qubits 0..nA-1 are the fastest bits.
    mat = np.asarray(psi, dtype=np.complex128).reshape((1 << (n - nA), 1 << nA))
    s = np.linalg.svd(mat, compute_uv=False)
    p = np.clip(np.abs(s) ** 2, 0.0, None)
    p = p[p > 1e-16]
    p = p / p.sum()
    return float(-np.sum(p * np.log(p)))


def canonicalize(psi: np.ndarray) -> np.ndarray:
    v = np.asarray(psi, dtype=np.complex128).copy()
    k = int(np.argmax(np.abs(v)))
    if abs(v[k]) > 0.0:
        v *= np.exp(-1j * np.angle(v[k]))
    if float(np.sum(v.real)) < 0.0:
        v = -v
    v.imag[np.abs(v.imag) < 1e-14] = 0.0
    return v


def state_hash(psi: np.ndarray) -> str:
    import hashlib
    a = np.ascontiguousarray(canonicalize(psi))
    return hashlib.sha256(a.tobytes()).hexdigest()


def analyze(hc: np.ndarray, B: float, n: int) -> dict:
    H = dense_hamiltonian(hc, B, n)
    evals, evecs = np.linalg.eigh(H)
    ridx = reflection_indices(n)
    rows = []
    for k in range(min(8, evals.size)):
        psi = canonicalize(evecs[:, k])
        rows.append({
            "k": k,
            "E": float(evals[k]),
            "parity": parity_expectation(psi, n),
            "reflection": reflection_expectation(psi, ridx),
            "residual": float(np.linalg.norm(H @ psi - evals[k] * psi)),
        })
    gs = canonicalize(evecs[:, 0])
    # Sector membership of the whole spectrum, using the raw eigenvectors.
    p_exp = np.empty(evals.size)
    r_exp = np.empty(evals.size)
    # Only the lowest dozen need exact expectations for the gap; computing all
    # 4096 reflections is cheap (a gather).
    flipped = evecs[::-1, :]
    p_exp = np.sum(evecs.conj() * flipped, axis=0).real
    reflected = evecs[ridx, :]
    r_exp = np.sum(evecs.conj() * reflected, axis=0).real
    in_sector = (p_exp > 0.5) & (r_exp > 0.5)
    sector_evals = evals[in_sector]
    if sector_evals.size == 0:
        sector_gap = float("nan")
        sector_E0 = float("nan")
    else:
        sector_E0 = float(sector_evals[0])
        sector_gap = float(sector_evals[1] - sector_evals[0]) if sector_evals.size > 1 else float("nan")
    return {
        "E0": float(evals[0]),
        "Emax": float(evals[-1]),
        "full_gap": float(evals[1] - evals[0]),
        "sector_E0": sector_E0,
        "sector_gap": sector_gap,
        "n_sector": int(np.count_nonzero(in_sector)),
        "gs_in_sector": bool(in_sector[0]),
        "entropy_nats": half_chain_entropy(gs, n),
        "gs_residual": float(np.linalg.norm(H @ gs - evals[0] * gs)),
        "gs_parity": parity_expectation(gs, n),
        "gs_reflection": reflection_expectation(gs, ridx),
        "gs_min_real": float(np.min(gs.real)),
        "gs_imag_max": float(np.max(np.abs(gs.imag))),
        "lowest8": rows,
        "evals": evals,
        "gs": gs,
        "ridx": ridx,
        "note": (
            "full_gap is E1-E0 of the unrestricted Hamiltonian. "
            "sector_gap is the gap inside P=+1 and R=+1. "
            "Fidelity always uses the full-space ground state."
        ),
    }
