"""Oscillator displacement, strict-logical D8, and shell phases.

Production true-D is the unitary exponential of the truncated generator.
It is not an 8-by-8 crop of a larger exponential, and states are not renormalized.
D8 is a diagnostic gate and is never a displacement.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import expm

from common import ALPHA_MAX, CODE_LEVELS


def annihilation(nf: int) -> np.ndarray:
    a = np.zeros((nf, nf), dtype=np.float64)
    idx = np.arange(1, nf, dtype=np.float64)
    a[idx.astype(int) - 1, idx.astype(int)] = np.sqrt(idx)
    return a


def generator(nf: int, strict8: bool = False) -> np.ndarray:
    a = annihilation(nf)
    G = a.T - a
    if strict8:
        masked = np.zeros_like(G)
        masked[:CODE_LEVELS, :CODE_LEVELS] = G[:CODE_LEVELS, :CODE_LEVELS]
        return masked
    return G


def displacement_matrix(alpha: float, nf: int, strict8: bool = False) -> np.ndarray:
    G = generator(nf, strict8=strict8)
    return expm(float(alpha) * G)


def hadamard3() -> np.ndarray:
    H1 = np.array([[1.0, 1.0], [1.0, -1.0]], dtype=np.float64) / np.sqrt(2.0)
    W = H1.copy()
    W = np.kron(W, H1)
    W = np.kron(W, H1)
    return W


def W_binH(nf: int) -> np.ndarray:
    W = np.eye(nf, dtype=np.float64)
    W[:CODE_LEVELS, :CODE_LEVELS] = hadamard3()
    return W


def gray_bits(f: int) -> int:
    return int(f) ^ (int(f) >> 1)


def ungray(g: int) -> int:
    g = int(g)
    s = g >> 1
    while s:
        g ^= s
        s >>= 1
    return g


def logical_bits_of_level(f: int, encoding: str) -> int:
    if f >= CODE_LEVELS:
        raise ValueError("logical bits are only defined for f < 8")
    if encoding == "binary":
        return int(f)
    if encoding == "gray":
        return gray_bits(f)
    raise ValueError(encoding)


def popcount_level(f: int, encoding: str) -> int:
    bits = logical_bits_of_level(f, encoding)
    return int(bits & 1) + int((bits >> 1) & 1) + int((bits >> 2) & 1)


def shell_phase_of_pop(pop: int, s1: float, s2: float, s3: float) -> float:
    p = int(pop)
    return float(s1) * p + float(s2) * (p * (p - 1) / 2.0) + (float(s3) if p == 3 else 0.0)


def snap_diagonal(nf: int, s1: float, s2: float, s3: float, encoding: str) -> np.ndarray:
    """Bare shell phases. Levels >= 8 get phase 0 (approved simulation D2)."""
    ph = np.zeros(nf, dtype=np.float64)
    for f in range(CODE_LEVELS):
        ph[f] = shell_phase_of_pop(popcount_level(f, encoding), s1, s2, s3)
    return np.exp(-1j * ph)


def walsh_shell_phase(pop: int, c1: float, c2: float, c3: float) -> float:
    t1 = (3.0, 1.0, -1.0, -3.0)[pop]
    t2 = (3.0, -1.0, -1.0, 3.0)[pop]
    t3 = (1.0, -1.0, 1.0, -1.0)[pop]
    return c1 * t1 + c2 * t2 + c3 * t3


def s_to_walsh(s1: float, s2: float, s3: float) -> tuple[float, float, float, float]:
    """Linear map taking polynomial shell phases to Walsh phases plus a constant.

    The 3x3 block is invertible. The constant is a global phase.
    """
    c1 = -0.5 * s1 - 0.5 * s2 - 0.125 * s3
    c2 = 0.25 * s2 + 0.125 * s3
    c3 = -0.125 * s3
    lam = 1.5 * s1 + 0.75 * s2 + 0.125 * s3
    return c1, c2, c3, lam


def flip_level(f: int, bit: int, encoding: str) -> int:
    if encoding == "binary":
        return int(f) ^ (1 << bit)
    g = gray_bits(f) ^ (1 << bit)
    return ungray(g)


def flip_all_table(encoding: str) -> np.ndarray:
    out = np.zeros(CODE_LEVELS, dtype=np.int64)
    for f in range(CODE_LEVELS):
        if encoding == "binary":
            out[f] = f ^ 7
        else:
            out[f] = ungray(gray_bits(f) ^ 7)
    return out


def u_from_alpha(alpha: float) -> float:
    x = float(alpha) / ALPHA_MAX
    x = min(1.0 - 1e-12, max(-1.0 + 1e-12, x))
    return float(np.arctanh(x))


def alpha_from_u(u: float) -> float:
    return float(ALPHA_MAX * np.tanh(u))
