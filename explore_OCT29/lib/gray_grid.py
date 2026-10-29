"""Gray-code product grid for fixed K=3 bits/cavity (multi-cavity)."""
from __future__ import annotations
import numpy as np
from .packing import K_BITS, NU, n_cavities


def gray(i: int) -> int:
    return i ^ (i >> 1)


def ungray(g: int) -> int:
    g = int(g)
    s = g >> 1
    while s:
        g ^= s
        s >>= 1
    return g


def gray_table(nu: int = NU) -> np.ndarray:
    return np.array([gray(i) for i in range(nu)], dtype=np.int64)


def build_grid_to_x(n: int) -> np.ndarray:
    """Map Fock tuple (f0,...,f_{C-1}) -> bitstring index x (var0=MSB).

    Cavity c owns vars x_{3c}, x_{3c+1}, x_{3c+2}; Gray bits j=0..2 inside cavity.
    Returns array of shape (NU,)*C with int64 x values.
    """
    C = n_cavities(n)
    shape = (NU,) * C
    out = np.zeros(shape, dtype=np.int64)
    gtab = gray_table(NU)
    for flat in range(NU ** C):
        idx = np.unravel_index(flat, shape)
        bits = []
        for c in range(C):
            g = int(gtab[idx[c]])
            for j in range(K_BITS):
                bits.append((g >> j) & 1)
        x = 0
        for v, b in enumerate(bits):
            x |= int(b) << (n - 1 - v)
        out[idx] = x
    return out


def cost_phase_on_grid(E: np.ndarray, n: int, grid_to_x: np.ndarray | None = None) -> np.ndarray:
    """Map energy vector E[x] onto product Gray grid."""
    if grid_to_x is None:
        grid_to_x = build_grid_to_x(n)
    return E[grid_to_x].astype(np.float32)
