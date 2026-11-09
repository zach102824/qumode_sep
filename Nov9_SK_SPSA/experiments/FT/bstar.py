"""Per-instance transverse field that maximizes ground-state entanglement.

This is classical exact diagonalization. It does not call SPSA and it does not
replace the archived b=1 and b=1.5 pilot.

The cut is fixed before any instance is seen: subsystem A is sites
0 .. floor(N/2)-1 in the LSB site order (site i is bit i). For odd N the
parts differ by one spin. The cut is not chosen to maximize the entropy.
"""
from __future__ import annotations

import numpy as np

from Nov9_SK_SPSA.experiments.FT.instance import dense_H, make_J, pair_diag, spin_parity, z_table

GAP_NONDEGENERATE = 1e-8
ENTROPY_TIE = 1e-4


def balanced_cut_size(N: int) -> int:
    if N < 2:
        raise ValueError("need at least two spins")
    return int(N) // 2


def von_neumann_bits(psi: np.ndarray, n_a: int) -> float:
    """Entropy, in bits, of the lowest n_a bits. No renormalization."""
    amp = np.asarray(psi, dtype=np.complex128).reshape(-1)
    n = int(round(np.log2(amp.size)))
    if amp.size != 1 << n or not 1 <= n_a < n:
        raise ValueError("psi must be a full computational state and the cut must be proper")
    block = amp.reshape(1 << (n - n_a), 1 << n_a)
    rho = block.conj().T @ block
    ev = np.clip(np.linalg.eigvalsh(rho).real, 0.0, None)
    ev = ev[ev > 1e-15]
    return float(-np.sum(ev * np.log2(ev)))


def spectrum(J: np.ndarray, b: float):
    zz = pair_diag(J, z_table(J.shape[0]))
    evals, evecs = np.linalg.eigh(dense_H(zz, b))
    gs = np.real(evecs[:, 0]).astype(np.float64)
    gs /= np.linalg.norm(gs)
    if float(gs.sum()) < 0.0:
        gs = -gs
    return evals, gs


def gs_checks(evals: np.ndarray, gs: np.ndarray) -> dict:
    gap = float(evals[1] - evals[0])
    parity = spin_parity(gs)
    return {
        "E0": float(evals[0]),
        "E1": float(evals[1]),
        "gap": gap,
        "nondegenerate": bool(gap > GAP_NONDEGENERATE),
        "parity": parity,
        "even": bool(parity > 0.5),
        "gs_min": float(gs.min()),
        "positive": bool(gs.min() > 0.0),
        "eligible": bool(gap > GAP_NONDEGENERATE and parity > 0.5 and gs.min() > 0.0),
    }


def entropy_at(J: np.ndarray, b: float) -> dict:
    if not b > 0.0:
        raise ValueError("B* search stays at b>0 so the Perron vector is the even positive ground state")
    evals, gs = spectrum(J, b)
    n_a = balanced_cut_size(J.shape[0])
    checks = gs_checks(evals, gs)
    checks.update({
        "b": float(b),
        "n_a": n_a,
        "n_b": int(J.shape[0]) - n_a,
        "entropy_bits": von_neumann_bits(gs, n_a),
        "cut": f"sites 0..{n_a - 1} versus the rest",
    })
    return checks


def maximize_entropy(J: np.ndarray, b_lo: float = 0.05, b_hi: float = 4.0, n_grid: int = 40) -> dict:
    """Grid plus two parabolic refinements. Returns the eligible maximizer and the plateau."""
    if b_lo <= 0.0 or b_hi <= b_lo:
        raise ValueError("scan interval must lie in b>0")
    grid = np.linspace(b_lo, b_hi, int(n_grid))
    rows = [entropy_at(J, float(b)) for b in grid]
    if rows[0]["entropy_bits"] >= max(r["entropy_bits"] for r in rows) - ENTROPY_TIE:
        raise RuntimeError("entropy maximum is at the lower edge; widen b_lo")
    if rows[-1]["entropy_bits"] >= max(r["entropy_bits"] for r in rows) - ENTROPY_TIE:
        raise RuntimeError("entropy maximum is at the upper edge; widen b_hi")
    b = float(max(rows, key=lambda r: r["entropy_bits"])["b"])
    for _ in range(2):
        below = [r for r in rows if r["b"] < b - 1e-12]
        above = [r for r in rows if r["b"] > b + 1e-12]
        if not below or not above:
            break
        left = max(below, key=lambda r: r["b"])
        right = min(above, key=lambda r: r["b"])
        mid = min(rows, key=lambda r: abs(r["b"] - b))
        xs = np.array([left["b"], mid["b"], right["b"]], dtype=np.float64)
        ys = np.array([left["entropy_bits"], mid["entropy_bits"], right["entropy_bits"]], dtype=np.float64)
        coef = np.polyfit(xs, ys, 2)
        if coef[0] >= 0.0:
            break
        vertex = float(-coef[1] / (2.0 * coef[0]))
        if not left["b"] < vertex < right["b"]:
            break
        if any(abs(r["b"] - vertex) < 1e-8 for r in rows):
            b = vertex
            break
        rows.append(entropy_at(J, vertex))
        b = vertex
    peak = max(r["entropy_bits"] for r in rows)
    plateau = [r for r in rows if peak - r["entropy_bits"] <= ENTROPY_TIE]
    eligible = [r for r in plateau if r["eligible"]]
    if not eligible:
        raise RuntimeError("entropy plateau has no positive even nondegenerate ground state")
    chosen = max(eligible, key=lambda r: r["entropy_bits"])
    return {
        "Bstar": float(chosen["b"]),
        "entropy_bits": float(chosen["entropy_bits"]),
        "checks": chosen,
        "plateau_b": [float(r["b"]) for r in plateau],
        "n_evaluated": len(rows),
        "b_lo": float(b_lo),
        "b_hi": float(b_hi),
    }
