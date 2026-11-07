"""Layer-center resampling and Chebyshev schedules. Frozen in protocol.json."""

from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline


def layer_centers(L: int) -> np.ndarray:
    ell = np.arange(1, L + 1, dtype=np.float64)
    return (ell - 0.5) / float(L)


def unwrap_period(values, period: float) -> np.ndarray:
    out = np.asarray(values, dtype=np.float64).copy()
    if len(out) < 2 or period is None:
        return out
    for i in range(1, len(out)):
        shift = np.round((out[i - 1] - out[i]) / period) * period
        out[i] += shift
    return out


def resample_shells(shells, L_dst: int, method: str = "auto") -> np.ndarray:
    """Resample a (L_src, 3) shell schedule onto L_dst layer centers.

    Interior: linear, or a natural cubic when the source has at least 3 layers
    and method is not 'linear'. Outside the source knots: constant extrapolation.
    """
    src = np.asarray(shells, dtype=np.float64)
    if src.ndim != 2 or src.shape[1] != 3:
        raise ValueError(src.shape)
    L_src = src.shape[0]
    x = layer_centers(L_src)
    xq = layer_centers(L_dst)
    xq_clip = np.clip(xq, x[0], x[-1])
    use_cubic = method == "cubic" or (method == "auto" and L_src >= 3)
    out = np.zeros((L_dst, 3), dtype=np.float64)
    for k in range(3):
        y = unwrap_period(src[:, k], np.pi)
        if L_src == 1 or not use_cubic:
            out[:, k] = np.interp(xq_clip, x, y)
        else:
            out[:, k] = CubicSpline(x, y, bc_type="natural")(xq_clip)
    return out


def chebyshev_basis(L: int, degree: int) -> np.ndarray:
    """Rows are T_0(z)..T_D(z) at z=2*x_l-1. Shape (L, D+1)."""
    z = 2.0 * layer_centers(L) - 1.0
    cols = [np.ones_like(z), z]
    for _ in range(2, degree + 1):
        cols.append(2.0 * z * cols[-1] - cols[-2])
    return np.column_stack(cols[: degree + 1])


def shells_from_chebyshev(coeff, L: int) -> np.ndarray:
    """coeff shape (3, D+1) -> shells (L, 3)."""
    coeff = np.asarray(coeff, dtype=np.float64)
    degree = coeff.shape[1] - 1
    basis = chebyshev_basis(L, degree)
    return basis @ coeff.T


def pack_full(gamma, theta_k, theta_m, shells) -> np.ndarray:
    shells = np.asarray(shells, dtype=np.float64)
    return np.concatenate([[gamma, theta_k, theta_m], shells.reshape(-1)])


def unpack_full(x, L: int):
    x = np.asarray(x, dtype=np.float64)
    return float(x[0]), float(x[1]), float(x[2]), x[3:].reshape(L, 3).copy()
