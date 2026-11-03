"""QAOA angle initialization for Nov_3rd training.

Default init follows Boulebnane–Montanaro arXiv:2208.06909
§Parameter optimization: all β=0.01, all γ=−0.01, single start
(nstart=1 — one train is enough; no multi-start ramps / paper-p10 / random).

Angle convention (qaoa_ref):
  U_C = exp(-i γ Cn),  U_B = exp(-i β X/2),  Cn = E·n/m
"""
from __future__ import annotations

import numpy as np

# arXiv:2208.06909 §Parameter optimization
QAOA_INIT_BETA = 0.01
QAOA_INIT_GAMMA = -0.01
QAOA_NSTART = 1

ANGLE_CONVENTION = (
    "qaoa_ref: U_C=exp(-i γ Cn), U_B=exp(-i β X/2), Cn=E*n/m"
)
INIT_NOTES = (
    "arXiv:2208.06909 Boulebnane–Montanaro §Parameter optimization: "
    "β_ℓ=0.01, γ_ℓ=-0.01 for each layer; nstart=1"
)


def paper_start(L: int) -> np.ndarray:
    """Return length-2L vector [β_0..β_{L-1}, γ_0..γ_{L-1}] = paper init."""
    L = int(L)
    if L < 1:
        raise ValueError(f"L must be >=1, got {L}")
    betas = np.full(L, QAOA_INIT_BETA, dtype=np.float64)
    gammas = np.full(L, QAOA_INIT_GAMMA, dtype=np.float64)
    return np.concatenate([betas, gammas])


def make_starts(
    L: int,
    nstart: int = QAOA_NSTART,
    rng: np.random.Generator | None = None,
) -> list[np.ndarray]:
    """Paper-only starts. nstart is locked to 1 (extra starts ignored).

    For L=1 returns ``[array([0.01, -0.01])]``.
    """
    del rng  # unused; kept for call-site compatibility
    if int(nstart) != QAOA_NSTART:
        # One train from paper init is enough — do not multi-start.
        nstart = QAOA_NSTART
    return [paper_start(L)]
