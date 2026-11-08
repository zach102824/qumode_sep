"""Layer-center resampling. Frozen with the round-5 preregistration."""

from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline


def centers(L: int) -> np.ndarray:
    return (np.arange(1, L + 1, dtype=np.float64) - 0.5) / float(L)


def unwrap_pi(values) -> np.ndarray:
    out = np.asarray(values, dtype=np.float64).copy()
    for i in range(1, len(out)):
        out[i] += np.round((out[i - 1] - out[i]) / np.pi) * np.pi
    return out


def resample_shells(shells, L_dst: int, method: str = "auto") -> np.ndarray:
    src = np.asarray(shells, dtype=np.float64)
    L_src = src.shape[0]
    x = centers(L_src)
    xq = np.clip(centers(L_dst), x[0], x[-1])
    cubic = method == "cubic" or (method == "auto" and L_src >= 3)
    out = np.zeros((L_dst, 3))
    for k in range(3):
        y = unwrap_pi(src[:, k])
        if L_src == 1 or not cubic:
            out[:, k] = np.interp(xq, x, y)
        else:
            out[:, k] = CubicSpline(x, y, bc_type="natural")(xq)
    return out


def pack(gamma, theta_k, theta_m, shells) -> np.ndarray:
    return np.concatenate([[gamma, theta_k, theta_m], np.asarray(shells, dtype=np.float64).reshape(-1)])
