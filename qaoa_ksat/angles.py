"""Published average-instance QAOA angles from the paper's data repo."""

from __future__ import annotations

import numpy as np

from .ksat import THRESHOLD_R
from .paper_data.optimal_angles import optimal_angles


def paper_angles(k: int, p: int, r: float | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Return (betas, gammas) of length p for random k-SAT at ratio r (threshold default)."""
    if r is None:
        r = float(THRESHOLD_R[int(k)])
    key = (int(k), float(r))
    if key not in optimal_angles:
        raise KeyError(f"no published angles for k={k}, r={r}")
    by_p = optimal_angles[key]
    if int(p) not in by_p:
        available = sorted(int(x) for x in by_p)
        raise KeyError(f"no published angles for k={k} p={p}; have {available}")
    block = by_p[int(p)]
    betas = np.asarray(block["betas"], dtype=float).reshape(-1)
    gammas = np.asarray(block["gammas"], dtype=float).reshape(-1)
    if betas.size != int(p) or gammas.size != int(p):
        raise ValueError(f"angle length mismatch for k={k} p={p}")
    return betas, gammas


def available_p(k: int, r: float | None = None) -> list[int]:
    if r is None:
        r = float(THRESHOLD_R[int(k)])
    return sorted(int(x) for x in optimal_angles[(int(k), float(r))])
