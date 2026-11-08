"""Round-2 family P features.

G = Hc + theta_K * F_K + theta_P * F_P

F_K is the full off-diagonal of J@J, every pair, not truncated.
F_P = sum_{a<b} Jbar_ab P_a P_b
P_a = Z_{3a} Z_{3a+1} Z_{3a+2}
Jbar_ab = (sum of the original J_ij with i in a, j in b) / 9

Normalization, frozen for this round: match RMS in the full Pauli
coefficient space of that weight, zeros included.
  two-body space size = C(N, 2)
  six-body space size = C(N, 6)
RMS(c) = sqrt(mean of c_alpha^2 over that whole space).
Hc's upper triangle is the two-body reference. Scales are stored.
A nonzero-only RMS is recorded as a diagnostic and is not the training scale.

S_a = sum_{i in cavity a} Z_i is the two-body control. Shuffled Jbar is a
fixed permutation, seed 20261107, chosen before any score is computed.
N=6 has one cavity pair, so that shuffle is the identity.
"""

from __future__ import annotations

from math import comb
from typing import Dict, List, Sequence, Tuple

import numpy as np

from target import rms_list, upper_values, z_table

SHUFFLE_SEED = 20261107


def pair_K(J: np.ndarray) -> tuple[np.ndarray, float]:
    M = J @ J
    dropped = float(np.trace(M))
    K = np.array(M, dtype=np.float64, copy=True)
    np.fill_diagonal(K, 0.0)
    K = 0.5 * (K + K.T)
    np.fill_diagonal(K, 0.0)
    return K, dropped


def cavity_pairs(n: int) -> List[Tuple[int, int]]:
    m = n // 3
    return [(a, b) for a in range(m) for b in range(a + 1, m)]


def jbar_matrix(J: np.ndarray) -> np.ndarray:
    n = J.shape[0]
    m = n // 3
    out = np.zeros((m, m), dtype=np.float64)
    for a, b in cavity_pairs(n):
        s = 0.0
        for i in range(3 * a, 3 * a + 3):
            for j in range(3 * b, 3 * b + 3):
                s += float(J[i, j])
        out[a, b] = out[b, a] = s / 9.0
    return out


def shuffled_jbar(Jbar: np.ndarray, n: int, seed: int = SHUFFLE_SEED) -> tuple[np.ndarray, List[int]]:
    pairs = cavity_pairs(n)
    m = Jbar.shape[0]
    values = [float(Jbar[a, b]) for a, b in pairs]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(pairs)).tolist()
    out = np.zeros_like(Jbar)
    for new_i, old_i in enumerate(perm):
        a, b = pairs[new_i]
        out[a, b] = out[b, a] = values[old_i]
    # N=6: one pair, permutation is trivial.
    return out, [int(i) for i in perm]


def diag_from_pairs(coeff: np.ndarray, z: np.ndarray) -> np.ndarray:
    n = coeff.shape[0]
    out = np.zeros(z.shape[1], dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            if coeff[i, j] != 0.0:
                out += coeff[i, j] * z[i] * z[j]
    return out


def six_body_diag(n: int, weight_by_pair: Dict[Tuple[int, int], float], z: np.ndarray) -> np.ndarray:
    """Diagonal of sum_{a<b} w_ab P_a P_b. P_a is the product of the cavity's three Z's."""
    out = np.zeros(z.shape[1], dtype=np.float64)
    p = []
    m = n // 3
    for a in range(m):
        p.append(z[3 * a] * z[3 * a + 1] * z[3 * a + 2])
    for (a, b), w in weight_by_pair.items():
        if w != 0.0:
            out += float(w) * p[a] * p[b]
    return out


def sab_pair_coeff(n: int, Jbar: np.ndarray) -> np.ndarray:
    """Two-body coefficients of sum_{a<b} Jbar_ab S_a S_b.

    S_a S_b = sum_{i in a, j in b} Z_i Z_j, so every bond between the blocks
    gets the same coefficient Jbar_ab. Intra-cavity bonds stay zero.
    """
    coeff = np.zeros((n, n), dtype=np.float64)
    for a, b in cavity_pairs(n):
        w = float(Jbar[a, b])
        for i in range(3 * a, 3 * a + 3):
            for j in range(3 * b, 3 * b + 3):
                coeff[i, j] = coeff[j, i] = w
    return coeff


def _scale_to_rms(raw_values_including_zeros: np.ndarray, rms_hc: float) -> float:
    rms = rms_list(raw_values_including_zeros)
    if rms <= 0.0:
        raise RuntimeError("feature RMS vanished")
    return float(rms_hc / rms)


def six_coeff_vector(n: int, weight_by_pair: Dict[Tuple[int, int], float]) -> np.ndarray:
    """All C(N,6) coefficients, zeros included. Nonzero only for exact cavity-pair supports."""
    from itertools import combinations

    index = {frozenset(range(3 * a, 3 * a + 3)) | frozenset(range(3 * b, 3 * b + 3)): w
             for (a, b), w in weight_by_pair.items()}
    vals = np.zeros(comb(n, 6), dtype=np.float64)
    for t, subset in enumerate(combinations(range(n), 6)):
        vals[t] = float(index.get(frozenset(subset), 0.0))
    return vals


def build_features(J: np.ndarray) -> dict:
    n = J.shape[0]
    z = z_table(n)
    from target import hc_diagonal

    hc = hc_diagonal(J, z)
    j_vals = upper_values(J)
    rms_hc = rms_list(j_vals)
    K, dropped = pair_K(J)
    k_vals = upper_values(K)
    scale_k = _scale_to_rms(k_vals, rms_hc)
    fk = diag_from_pairs(K * scale_k, z)

    Jbar = jbar_matrix(J)
    weights = {(a, b): float(Jbar[a, b]) for a, b in cavity_pairs(n)}
    raw_six = six_coeff_vector(n, weights)
    scale_p = _scale_to_rms(raw_six, rms_hc)
    fp = six_body_diag(n, {k: v * scale_p for k, v in weights.items()}, z)

    Jbar_s, perm = shuffled_jbar(Jbar, n)
    weights_s = {(a, b): float(Jbar_s[a, b]) for a, b in cavity_pairs(n)}
    raw_six_s = six_coeff_vector(n, weights_s)
    scale_ps = _scale_to_rms(raw_six_s, rms_hc)
    fp_shuf = six_body_diag(n, {k: v * scale_ps for k, v in weights_s.items()}, z)

    sab = sab_pair_coeff(n, Jbar)
    sab_vals = upper_values(sab)
    scale_s = _scale_to_rms(sab_vals, rms_hc)
    fs = diag_from_pairs(sab * scale_s, z)

    nonzero_p = np.array([w for w in weights.values()], dtype=np.float64)
    scale_p_nonzero_only = _scale_to_rms(nonzero_p, rms_hc)

    return {
        "n": n,
        "hc": hc,
        "fk": fk,
        "fp": fp,
        "fp_shuf": fp_shuf,
        "fs": fs,
        "z": z,
        "K": K,
        "Jbar": Jbar,
        "Jbar_shuf": Jbar_s,
        "shuffle_seed": SHUFFLE_SEED,
        "shuffle_perm_old_index_for_each_new_pair": perm,
        "shuffle_is_identity": bool(np.allclose(Jbar, Jbar_s)),
        "rms_Hc": rms_hc,
        "scale_K": scale_k,
        "scale_P": scale_p,
        "scale_P_shuffle": scale_ps,
        "scale_S": scale_s,
        "scale_P_if_nonzero_supports_only": scale_p_nonzero_only,
        "dropped_J2_identity": dropped,
        "n_two_body": int(j_vals.size),
        "n_six_body": int(raw_six.size),
        "n_nonzero_six": int(np.count_nonzero(raw_six)),
        "normalization": (
            "RMS over the full weight-w Pauli coefficient space, zeros included. "
            "Two-body reference is the Hc upper triangle. "
            "scale_P_if_nonzero_supports_only is diagnostic and is not used in G."
        ),
    }


def regression(y: np.ndarray, columns: Sequence[np.ndarray], weights: np.ndarray | None = None) -> dict:
    if weights is None:
        A = np.column_stack(list(columns))
        coef, _, rank, _ = np.linalg.lstsq(A, y, rcond=None)
        resid = y - A @ coef
        ss_res = float(np.dot(resid, resid))
        centered = y - y.mean()
        ss_tot = float(np.dot(centered, centered))
        r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-30 else float("nan")
        ratio = float(np.sqrt(ss_res / y.size) / np.std(y)) if np.std(y) > 1e-30 else float("nan")
        return {"R2": r2, "residual_rms_ratio": ratio, "rank": int(rank), "n": int(y.size)}
    w = np.asarray(weights, dtype=np.float64)
    sw = np.sqrt(np.maximum(w, 0.0))
    A = np.column_stack(list(columns))
    coef, _, rank, _ = np.linalg.lstsq(A * sw[:, None], y * sw, rcond=None)
    resid = y - A @ coef
    ss_res = float(np.sum(w * resid ** 2))
    ybar = float(np.sum(w * y) / np.sum(w))
    ss_tot = float(np.sum(w * (y - ybar) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-30 else float("nan")
    ratio = float(np.sqrt(ss_res / np.sum(w)) / np.sqrt(ss_tot / np.sum(w))) if ss_tot > 1e-30 else float("nan")
    return {"R2": r2, "residual_rms_ratio": ratio, "rank": int(rank), "n": int(y.size)}
