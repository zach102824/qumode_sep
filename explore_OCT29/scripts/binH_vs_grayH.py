"""BinH vs Gray-H helpers for explore_OCT29.

BinH: W = H^{⊗3} on Fock 0..7 (binary bit order), identity pad above.
Gray-H (Arm C / probe6): Π_gray · H^{⊗3} — see probe6_spectral_mixer.build_W_H.

Restored locally: this file was imported by gadget_Uc_* but never committed.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from explore_OCT29.lib.packing import K_BITS, NU, n_cavities


def build_W_BinH(nf: int = 8, dtype=torch.complex64) -> torch.Tensor:
    """W = H^{⊗3} on levels 0..7; identity on 8..nf-1. Self-adjoint."""
    H1 = torch.tensor([[1.0, 1.0], [1.0, -1.0]], dtype=torch.float64) / math.sqrt(2.0)
    H3 = H1
    for _ in range(2):
        H3 = torch.kron(H3, H1)
    W = torch.eye(nf, dtype=torch.float64)
    W[:8, :8] = H3
    return W.to(dtype=dtype)


def build_grid_to_x_binary(n: int) -> np.ndarray:
    """Map Fock tuple (f0,..f_{C-1}) → bitstring index x (var0=MSB).

    Cavity c owns vars x_{3c..3c+2}; bit j of Fock f is (f >> j) & 1 (binary, not Gray).
    """
    C = n_cavities(n)
    shape = (NU,) * C
    out = np.zeros(shape, dtype=np.int64)
    for flat in range(NU ** C):
        idx = np.unravel_index(flat, shape)
        bits = []
        for c in range(C):
            f = int(idx[c])
            for j in range(K_BITS):
                bits.append((f >> j) & 1)
        x = 0
        for v, b in enumerate(bits):
            x |= int(b) << (n - 1 - v)
        out[idx] = x
    return out
