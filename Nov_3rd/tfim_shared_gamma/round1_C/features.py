"""Coupling features for the shared generator.

G = Hc + theta_K F_K + theta_C F_star

F_K is built from K = offdiag(J @ J), i.e. K_jk = sum_i J_ij J_ik for j != k.
The diagonal of J@J is dropped because Z_j^2 = I would only add a constant.

F_star = sum_i sum_{j<k<l, all distinct from i and each other}
            J_ij J_ik J_il  Z_i Z_j Z_k Z_l

The same 4-qubit Pauli mask is produced once for every choice of the
distinguished index. Those contributions are summed, not overwritten.

Normalization (explicit): both features are rescaled so their Pauli
Hilbert-Schmidt norms equal ||Hc||_HS = sqrt(sum_{i<j} J_ij^2). Theta is
therefore a dimensionless weight relative to Hc. This is a convention, not a
Schrieffer-Wolff effective Hamiltonian.

Truncation rule (primary, hardware pair-gate class):
    keep a 4-body mask iff its qubits occupy at most 2 cavities.
Cavities are contiguous blocks of 3 qubits. Any pair of cavities is allowed,
adjacent or not, because a two-cavity diagonal is one 64-phase gate under the
compile model. That pair-gate cost equivalence is an assumption, not a measured
hardware result. N=6 has only 2 cavities, so the rule keeps every mask.
N=12 drops masks that touch 3 or 4 cavities. The unrestricted star is retained
only as a diagnostic.
"""

from __future__ import annotations

from itertools import combinations
from typing import Dict, Iterable, List, Tuple

import numpy as np

from hamiltonian import z_eigenvalues


def cavity_of(q: int) -> int:
    return q // 3


def pair_K(J: np.ndarray) -> Tuple[np.ndarray, float]:
    """Return symmetric K with zero diagonal, and the dropped identity constant.

    The dropped constant is sum_j (J@J)_jj, which is the coefficient of I that
    would appear if diagonal entries were kept and reduced with Z^2 = I.
    It does not enter F_K.
    """
    M = J @ J
    dropped = float(np.trace(M))
    K = np.array(M, dtype=np.float64, copy=True)
    np.fill_diagonal(K, 0.0)
    K = 0.5 * (K + K.T)
    np.fill_diagonal(K, 0.0)
    return K, dropped


def hs_norm_upper(K: np.ndarray) -> float:
    n = K.shape[0]
    s = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            s += float(K[i, j]) ** 2
    return float(np.sqrt(s))


def _mask_of(qubits: Iterable[int]) -> int:
    m = 0
    for q in qubits:
        m |= 1 << int(q)
    return m


def _popcount_int(m: int) -> int:
    return int(m.bit_count())


def star_coefficients(J: np.ndarray, max_cavities: int | None) -> Dict[int, float]:
    """Combined raw coefficients of Z-masks.

    max_cavities=None keeps every mask (unrestricted).
    max_cavities=2 applies the primary truncation.
    A contribution is kept or dropped after the four indices are known, and
    duplicate masks sum their weights.
    """
    n = J.shape[0]
    acc: Dict[int, float] = {}
    centers_hit: Dict[int, int] = {}
    for i in range(n):
        others = [t for t in range(n) if t != i]
        for comb in combinations(others, 3):
            j, k, l = comb  # combinations emits sorted triples
            weight = float(J[i, j] * J[i, k] * J[i, l])
            if weight == 0.0:
                continue
            qubits = (i, j, k, l)
            # Distinctness is required for a genuine 4-body monomial.
            if len(set(qubits)) != 4:
                raise RuntimeError("star term repeated an index; Z^2 reduction would be required")
            cavs = {cavity_of(q) for q in qubits}
            if max_cavities is not None and len(cavs) > max_cavities:
                continue
            mask = _mask_of(qubits)
            acc[mask] = acc.get(mask, 0.0) + weight
            centers_hit[mask] = centers_hit.get(mask, 0) + 1
    # Stash center counts on the function for diagnostics via attribute on return
    star_coefficients.last_center_counts = centers_hit  # type: ignore[attr-defined]
    return acc


def diag_from_pairs(K: np.ndarray, z: np.ndarray) -> np.ndarray:
    n = K.shape[0]
    dim = z.shape[1]
    out = np.zeros(dim, dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            if K[i, j] != 0.0:
                out += K[i, j] * z[i] * z[j]
    return out


def diag_from_masks(coeffs: Dict[int, float], z: np.ndarray) -> np.ndarray:
    """Diagonal of sum_m c_m prod_{q in m} Z_q."""
    n = z.shape[0]
    dim = z.shape[1]
    out = np.zeros(dim, dtype=np.float64)
    for mask, c in coeffs.items():
        if c == 0.0:
            continue
        sign = np.ones(dim, dtype=np.float64)
        bit = mask
        q = 0
        while bit:
            if bit & 1:
                sign *= z[q]
            bit >>= 1
            q += 1
        if q > n:
            raise RuntimeError("mask overflow")
        out += float(c) * sign
    return out


def hs_from_coeffs(coeffs: Dict[int, float]) -> float:
    return float(np.sqrt(sum(float(c) ** 2 for c in coeffs.values())))


def build_features(J: np.ndarray, max_cavities: int | None = 2) -> Dict:
    """Normalized F_K and F_star diagonals plus accounting numbers.

    F_*_diag is the diagonal of the normalized feature (same HS norm as Hc).
    """
    n = J.shape[0]
    z = z_eigenvalues(n)
    from hamiltonian import hc_diag_from_J, hs_norm_pairs

    hc = hc_diag_from_J(J, z)
    hs_hc = hs_norm_pairs(J)
    K, dropped_const = pair_K(J)
    fk_raw = diag_from_pairs(K, z)
    hs_k = hs_norm_upper(K)
    if hs_k <= 0.0 or hs_hc <= 0.0:
        raise RuntimeError("feature or Hc norm vanished")
    scale_k = hs_hc / hs_k  # multiply raw coefficients by this
    fk = fk_raw * scale_k

    raw_coeffs = star_coefficients(J, max_cavities=max_cavities)
    center_counts = dict(getattr(star_coefficients, "last_center_counts", {}))
    hs_star = hs_from_coeffs(raw_coeffs)
    if hs_star <= 0.0:
        raise RuntimeError("star feature vanished after truncation")
    scale_star = hs_hc / hs_star
    norm_coeffs = {m: float(c) * scale_star for m, c in raw_coeffs.items()}
    fs = diag_from_masks(norm_coeffs, z)

    # Unrestricted star, for the norm-retained fraction and the N=6 identity check.
    raw_full = star_coefficients(J, max_cavities=None)
    hs_full = hs_from_coeffs(raw_full)
    # Projection of the full star onto the kept masks (same raw weights).
    hs_kept = hs_star
    n_cav_hist = {1: 0, 2: 0, 3: 0, 4: 0}
    for mask in raw_full:
        qs = [q for q in range(n) if mask & (1 << q)]
        nc = len({cavity_of(q) for q in qs})
        n_cav_hist[nc] = n_cav_hist.get(nc, 0) + 1
    max_centers = max(center_counts.values()) if center_counts else 0
    odd_masks = [m for m in raw_coeffs if _popcount_int(m) % 2 == 1]
    bad_weight = [m for m in raw_coeffs if _popcount_int(m) != 4]

    # Pair-gate support: cavity pairs touched by Hc (all distance classes) and by stars.
    def pairs_from_two_body(mat: np.ndarray) -> List[Tuple[int, int]]:
        found = set()
        for i in range(n):
            for j in range(i + 1, n):
                if abs(mat[i, j]) > 0.0:
                    a, b = cavity_of(i), cavity_of(j)
                    if a > b:
                        a, b = b, a
                    found.add((a, b))
        return sorted(found)

    star_pairs = set()
    star_intra = 0
    for mask in raw_coeffs:
        qs = [q for q in range(n) if mask & (1 << q)]
        cavs = sorted({cavity_of(q) for q in qs})
        if len(cavs) == 1:
            star_intra += 1
        elif len(cavs) == 2:
            star_pairs.add((cavs[0], cavs[1]))
    hc_pairs = pairs_from_two_body(J)
    k_pairs = pairs_from_two_body(K)

    return {
        "n": n,
        "max_cavities": max_cavities,
        "hc_diag": hc,
        "K": K,
        "fk_diag": fk,
        "fs_diag": fs,
        "z": z,
        "hs_Hc": hs_hc,
        "hs_K_raw": hs_k,
        "hs_star_raw": hs_star,
        "hs_star_full_raw": hs_full,
        "scale_K": scale_k,
        "scale_star": scale_star,
        "dropped_J2_identity_constant": dropped_const,
        "n_star_masks": len(raw_coeffs),
        "n_star_masks_full": len(raw_full),
        "fraction_hs_norm_retained": float(hs_kept / hs_full) if hs_full else float("nan"),
        "max_centers_per_mask": int(max_centers),
        "center_count_histogram": _hist(center_counts.values()),
        "cavity_touch_histogram_full": n_cav_hist,
        "n_odd_masks": len(odd_masks),
        "n_non_weight4_masks": len(bad_weight),
        "hc_cavity_pairs": hc_pairs,
        "K_cavity_pairs": k_pairs,
        "star_cavity_pairs": sorted(star_pairs),
        "n_star_intra_cavity_masks": star_intra,
        "raw_star_coeffs": {str(m): float(c) for m, c in raw_coeffs.items()},
        "intra_hc_hs_fraction": _intra_fraction(J),
    }


def _hist(values) -> Dict[str, int]:
    h: Dict[str, int] = {}
    for v in values:
        h[str(int(v))] = h.get(str(int(v)), 0) + 1
    return h


def _intra_fraction(J: np.ndarray) -> float:
    n = J.shape[0]
    intra = 0.0
    total = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            w = float(J[i, j]) ** 2
            total += w
            if cavity_of(i) == cavity_of(j):
                intra += w
    return float(intra / total) if total else float("nan")


def orthogonality_report(hc: np.ndarray, fk: np.ndarray, fs: np.ndarray, z: np.ndarray) -> Dict:
    """Quantify that F_star is not a constant or a two-body operator.

    Full computational space: Pauli weights are orthogonal, so inner products of
    a pure weight-4 operator with I, with Hc, with F_K, and with every ZZ must
    vanish. Subsets of bitstrings can still make the sampled diagonals collinear.
    Those subset regressions are the 'relevant state' tests.
    """
    def ip(a, b):
        return float(np.dot(a, b))

    n = z.shape[0]
    dim = z.shape[1]
    ones = np.ones(dim, dtype=np.float64)
    # All two-body diagonals.
    cols = [ones]
    for i in range(n):
        for j in range(i + 1, n):
            cols.append(z[i] * z[j])
    A = np.column_stack(cols)
    coef, residuals, rank, sv = np.linalg.lstsq(A, fs, rcond=None)
    resid = fs - A @ coef
    ss_res = float(np.dot(resid, resid))
    ss_tot = float(np.dot(fs - fs.mean(), fs - fs.mean()))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    return {
        "inner_fs_ones": ip(fs, ones),
        "inner_fs_hc": ip(fs, hc),
        "inner_fs_fk": ip(fs, fk),
        "inner_fk_hc": ip(fk, hc),
        "inner_fk_ones": ip(fk, ones),
        "full_two_body_regression_R2": r2,
        "full_two_body_residual_rel": float(np.sqrt(ss_res) / np.linalg.norm(fs)),
        "two_body_design_rank": int(rank),
        "n_two_body_columns_including_identity": int(A.shape[1]),
        "fs_l2": float(np.linalg.norm(fs)),
        "note": (
            "R2 near 0 and a unit residual fraction mean F_star is orthogonal to "
            "every constant plus two-body Z string on the full computational basis. "
            "That is an operator identity, not a statement about subsets."
        ),
    }


def weighted_regression(y: np.ndarray, columns: List[np.ndarray], w: np.ndarray) -> Dict:
    """Least squares with weights w, R^2 against the weighted mean of y."""
    w = np.asarray(w, dtype=np.float64)
    sw = np.sqrt(np.maximum(w, 0.0))
    A = np.column_stack(columns)
    keep = [c for c in range(A.shape[1]) if np.linalg.norm(A[:, c] * sw) > 1e-12]
    if not keep or y.size < 2 or float(np.sum(w)) <= 0.0:
        return {"R2": float("nan"), "residual_rms_ratio": float("nan"), "n": int(y.size), "rank": 0}
    A_use = A[:, keep]
    coef, _, rank, _ = np.linalg.lstsq(A_use * sw[:, None], y * sw, rcond=None)
    pred = A_use @ coef
    resid = y - pred
    ss_res = float(np.sum(w * resid ** 2))
    ybar = float(np.sum(w * y) / np.sum(w))
    ss_tot = float(np.sum(w * (y - ybar) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-30 else float("nan")
    rms_res = float(np.sqrt(ss_res / np.sum(w)))
    rms_y = float(np.sqrt(ss_tot / np.sum(w))) if ss_tot > 0 else float("nan")
    ratio = float(rms_res / rms_y) if rms_y and rms_y > 1e-30 else float("nan")
    return {
        "R2": r2,
        "residual_rms_ratio": ratio,
        "n": int(y.size),
        "rank": int(rank),
        "fs_std": float(rms_y),
        "pred_std": float(np.sqrt(np.sum(w * (pred - ybar) ** 2) / np.sum(w))),
    }


def _regression(fs: np.ndarray, columns: List[np.ndarray]) -> Dict:
    A = np.column_stack(columns)
    # Drop numerically empty columns.
    keep = []
    for c in range(A.shape[1]):
        if np.linalg.norm(A[:, c]) > 1e-12:
            keep.append(c)
    if len(keep) < A.shape[1]:
        A_use = A[:, keep]
    else:
        A_use = A
    if A_use.shape[0] < 2 or A_use.shape[1] == 0:
        return {"R2": float("nan"), "residual_rms_ratio": float("nan"), "n": int(A.shape[0]), "rank": 0}
    coef, _, rank, _ = np.linalg.lstsq(A_use, fs, rcond=None)
    pred = A_use @ coef
    resid = fs - pred
    ss_res = float(np.dot(resid, resid))
    centered = fs - fs.mean()
    ss_tot = float(np.dot(centered, centered))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-30 else float("nan")
    rms_fs = float(np.sqrt(np.mean(centered ** 2))) if fs.size else float("nan")
    rms_res = float(np.sqrt(np.mean(resid ** 2))) if fs.size else float("nan")
    ratio = float(rms_res / rms_fs) if rms_fs and rms_fs > 1e-30 else float("nan")
    # Singular values of column-normalized design including fs, when the fit is vs cost.
    sv = np.linalg.svd(A_use, compute_uv=False)
    return {
        "R2": r2,
        "residual_rms_ratio": ratio,
        "n": int(fs.shape[0]),
        "rank": int(rank),
        "design_singular_values": [float(s) for s in sv],
        "fs_std": float(np.std(fs)),
        "pred_std": float(np.std(pred)),
    }


def _idx_to_bits(idx: int, n: int) -> List[int]:
    return [int((idx >> (n - 1 - q)) & 1) for q in range(n)]


def _bits_to_idx(bits: List[int]) -> int:
    idx = 0
    for b in bits:
        idx = (idx << 1) | int(b)
    return idx


def domain_wall_indices(n: int) -> Dict[str, List[int]]:
    """Classical AF catalog. Bit 0 is qubit 0 (left)."""
    neel_a = [q % 2 for q in range(n)]
    neel_b = [1 - (q % 2) for q in range(n)]

    def expand(base: List[int], flips: Iterable[int]) -> int:
        bits = list(base)
        for q in flips:
            bits[q] = 1 - bits[q]
        return _bits_to_idx(bits)

    neel = [expand(neel_a, []), expand(neel_b, [])]
    singles = []
    pairs = []
    blocks = []
    segments = []
    for base in (neel_a, neel_b):
        for q in range(n):
            singles.append(expand(base, [q]))
        for a, b in combinations(range(n), 2):
            pairs.append(expand(base, [a, b]))
        for cav in range(n // 3):
            blocks.append(expand(base, [3 * cav, 3 * cav + 1, 3 * cav + 2]))
        for a in range(n):
            for b in range(a, n):
                segments.append(expand(base, range(a, b + 1)))
    def uniq(xs):
        return sorted(set(int(i) for i in xs))
    return {
        "neel": uniq(neel),
        "single_flips": uniq(singles),
        "pair_flips": uniq(pairs),
        "cavity_block_flips": uniq(blocks),
        "contiguous_segments": uniq(segments),
    }


def relevant_state_report(hc: np.ndarray, fk: np.ndarray, fs: np.ndarray, gs: np.ndarray | None) -> Dict:
    n = int(round(np.log2(hc.shape[0])))
    catalog = domain_wall_indices(n)
    sets = {
        "neel": catalog["neel"],
        "single_flips": catalog["single_flips"],
        "excitations_single_pair_block": sorted(set(catalog["single_flips"]) | set(catalog["pair_flips"]) | set(catalog["cavity_block_flips"])),
        "contiguous_domain_walls": catalog["contiguous_segments"],
    }
    # Low classical window: 64 lowest Hc states, always, plus everyone within a small gap.
    order = np.argsort(hc)
    low = order[: min(64, hc.shape[0])].tolist()
    sets["lowest_64_classical"] = [int(i) for i in low]
    out = {}
    for name, idxs in sets.items():
        idx = np.array(idxs, dtype=np.int64)
        fs_s = fs[idx]
        hc_s = hc[idx]
        fk_s = fk[idx]
        ones = np.ones(idx.shape[0], dtype=np.float64)
        out[name] = {
            "vs_constant_plus_Hc": _regression(fs_s, [ones, hc_s]),
            "vs_constant_plus_Hc_plus_FK": _regression(fs_s, [ones, hc_s, fk_s]),
            "hc_std": float(np.std(hc_s)),
            "fk_std": float(np.std(fk_s)),
            "fs_std": float(np.std(fs_s)),
            "n": int(idx.shape[0]),
            "fs_values_if_small": [float(v) for v in fs_s] if idx.shape[0] <= 4 else None,
            "hc_values_if_small": [float(v) for v in hc_s] if idx.shape[0] <= 4 else None,
        }
    # Full space, four columns only (not every ZZ): singular values after column normalization.
    ones = np.ones_like(hc)
    M = np.column_stack([ones, hc, fk, fs])
    Mn = M / np.maximum(np.linalg.norm(M, axis=0), 1e-30)
    sv = np.linalg.svd(Mn, compute_uv=False)
    out["full_space_column_normalized_singular_values"] = [float(s) for s in sv]
    out["full_space_min_over_max_sv"] = float(sv[-1] / sv[0]) if sv[0] > 0 else float("nan")
    if gs is not None:
        p = np.abs(gs) ** 2
        p = np.asarray(p / p.sum(), dtype=np.float64)
        ones = np.ones_like(hc)
        out["gs_probability_weighted"] = {
            "vs_constant_plus_Hc": weighted_regression(fs, [ones, hc], p),
            "vs_constant_plus_Hc_plus_FK": weighted_regression(fs, [ones, hc, fk], p),
            "disclosed": (
                "This regression uses the exact ground state as a diagnostic weight. "
                "It is not used to choose features, thetas, or the training objective."
            ),
        }
        support = np.flatnonzero(p > 1e-4)
        out["gs_support_prob_gt_1e-4"] = {
            "vs_constant_plus_Hc": _regression(fs[support], [np.ones(support.size), hc[support]]),
            "vs_constant_plus_Hc_plus_FK": _regression(fs[support], [np.ones(support.size), hc[support], fk[support]]),
            "n": int(support.size),
            "prob_mass": float(p[support].sum()),
        }
    return out
