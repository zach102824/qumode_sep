"""Tied polynomial f(Hc) = Hc + theta (FK + Q4) / rK.

Hc^2 = c0 I + 2 FK + 2 Q4, with c0 = sum_edges J^2.
Q4 sums each unordered pair of disjoint edges once. Each 4-site support
accumulates the three pairings. rK = RMS(FK) / RMS(J) over all pair slots.
"""
from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path

import numpy as np

R2 = Path(__file__).resolve().parents[1] / "round2_H"
if str(R2) not in sys.path:
    sys.path.insert(0, str(R2))

from target import fk_matrix, pair_diagonal, upper_rms, z_table  # noqa: E402

# Frozen before any optimization. Permutes the 4-body coefficients.
SHUFFLE_SEED = 31108


def edge_c0(J: np.ndarray) -> float:
    n = J.shape[0]
    iu = np.triu_indices(n, 1)
    return float(np.sum(J[iu] ** 2))


def quadruples(n: int) -> list[tuple[int, int, int, int]]:
    return list(combinations(range(n), 4))


def q4_coefficients(J: np.ndarray) -> tuple[list[tuple[int, int, int, int]], np.ndarray]:
    """Coefficient of Z_a Z_b Z_c Z_d is the sum of the three edge pairings."""
    quads = quadruples(J.shape[0])
    coef = np.empty(len(quads), dtype=np.float64)
    for t, (a, b, c, d) in enumerate(quads):
        coef[t] = (
            J[a, b] * J[c, d]
            + J[a, c] * J[b, d]
            + J[a, d] * J[b, c]
        )
    return quads, coef


def q4_diagonal(z: np.ndarray, quads, coef: np.ndarray) -> np.ndarray:
    diag = np.zeros(z.shape[0], dtype=np.float64)
    for (a, b, c, d), w in zip(quads, coef):
        if w == 0.0:
            continue
        diag += w * z[:, a] * z[:, b] * z[:, c] * z[:, d]
    return diag


def shuffle_coefficients(coef: np.ndarray, seed: int = SHUFFLE_SEED) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.Generator(np.random.PCG64(seed))
    perm = rng.permutation(coef.size)
    return coef[perm].copy(), perm


def build_features(J: np.ndarray) -> dict:
    n = J.shape[0]
    z = z_table(n)
    hc = pair_diagonal(J, z)
    K = fk_matrix(J)
    fk = pair_diagonal(K, z)
    c0 = edge_c0(J)
    quads, coef = q4_coefficients(J)
    q4 = q4_diagonal(z, quads, coef)
    identity = hc ** 2 - (c0 + 2.0 * fk + 2.0 * q4)
    scale = max(1.0, float(np.max(np.abs(hc ** 2))))
    rK = upper_rms(K) / upper_rms(J)
    # RMS of the 495 (or C(N,4)) four-body slots, zeros included.
    rms_q4 = float(np.sqrt(np.mean(coef * coef)))
    rms_J = upper_rms(J)
    q4_hat = q4 * (rms_J / rms_q4)
    shuf_coef, perm = shuffle_coefficients(coef)
    q4_shuf = q4_diagonal(z, quads, shuf_coef)
    return {
        "n": n,
        "hc": hc,
        "K": K,
        "fk": fk,
        "q4": q4,
        "c0": c0,
        "identity_max_abs": float(np.max(np.abs(identity))),
        "identity_max_scaled": float(np.max(np.abs(identity)) / scale),
        "rK": float(rK),
        "rms_J": float(rms_J),
        "rms_FK": float(upper_rms(K)),
        "rms_Q4": rms_q4,
        "n_pairs": n * (n - 1) // 2,
        "n_fourbody": len(quads),
        "feature_F": (fk + q4) / rK,
        "feature_A": fk / rK,
        "feature_shuffle": (fk + q4_shuf) / rK,
        "q4_hat": q4_hat,
        "shuffle_perm": perm,
        "shuffle_seed": SHUFFLE_SEED,
        "q4_coef": coef,
    }


def pair_walsh(diag: np.ndarray, z: np.ndarray, i: int, j: int) -> float:
    """Coefficient of Z_i Z_j in a diagonal operator."""
    return float(np.mean(diag * z[:, i] * z[:, j]))
