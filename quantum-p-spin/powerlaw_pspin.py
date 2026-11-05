"""Power-law p=3 TFIM (ring diam) for Nov_3rd QAOA sweeps.

H = -s Σ_{i<j<k} J_ijk Z_i Z_j Z_k - Γ Σ_i X_i

Couplings (disordered power-law on a ring):
  r_ijk = diam{i,j,k} := max(d(i,j), d(i,k), d(j,k))
  d(a,b) = min(|a-b|, n-|a-b|)          # circular distance
  w_ijk  = r_ijk^{-α}                    # α=0 → all-to-all (w=1)
  J_ijk  = ξ_ijk * w_ijk / rms(ξ⊙w)     # ξ~N(0,1); unit RMS so α=0 ≈ SK
  s      = sqrt(3)/n                     # same intensive SK scale as pspin.scale_p3

α=0 recovers the Gaussian all-to-all p=3 glass (up to finite-sample RMS renorm).
Larger α → more local triples dominate.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pspin import (
    PSpinInstance,
    all_triples,
    ground_state,
    hc_diagonal,
    scale_p3,
)


def ring_dist(a: int, b: int, n: int) -> int:
    d = abs(int(a) - int(b))
    return min(d, n - d)


def triple_diam(i: int, j: int, k: int, n: int) -> int:
    return max(ring_dist(i, j, n), ring_dist(i, k, n), ring_dist(j, k, n))


def powerlaw_weights(n: int, alpha: float, triples: np.ndarray) -> np.ndarray:
    """w_ijk = r_ijk^{-α}; α=0 → ones."""
    if abs(float(alpha)) < 1e-15:
        return np.ones(len(triples), dtype=np.float64)
    w = np.empty(len(triples), dtype=np.float64)
    for t, (i, j, k) in enumerate(triples):
        r = triple_diam(int(i), int(j), int(k), n)
        w[t] = float(r) ** (-float(alpha))
    return w


def make_powerlaw_instance(
    n: int,
    Gamma: float,
    seed: int,
    alpha: float,
) -> PSpinInstance:
    """One disordered power-law p=3 TFIM instance (ξ~N(0,1) × r^{-α}, unit RMS)."""
    rng = np.random.default_rng(seed)
    triples = all_triples(n)
    w = powerlaw_weights(n, alpha, triples)
    xi = rng.normal(0.0, 1.0, size=len(triples))
    raw = xi * w
    rms = float(np.sqrt(np.mean(raw * raw)))
    if rms < 1e-15:
        raise RuntimeError(f"degenerate weights n={n} alpha={alpha}")
    J = raw / rms
    s = scale_p3(n)
    Hc = hc_diagonal(n, triples, J, s)
    E0, gs = ground_state(Hc, Gamma, n, seed=seed + 17)
    k = int(np.argmax(np.abs(gs)))
    phase = np.abs(gs[k]) / (gs[k] + 1e-30)
    gs = gs * phase
    return PSpinInstance(
        n=n,
        Gamma=float(Gamma),
        seed=int(seed),
        J=J.astype(np.float64),
        triples=triples,
        scale=s,
        Hc_diag=Hc.astype(np.float64),
        E0=E0,
        gs=gs.astype(np.complex128),
    )


def formula_doc() -> str:
    return (
        "H = -s Σ_{i<j<k} J_ijk Z_i Z_j Z_k - Γ Σ_i X_i,  "
        "s=√3/n,  r_ijk=diam{i,j,k} on ring,  "
        "J_ijk = ξ_ijk r_ijk^{-α} / rms(ξ⊙r^{-α}),  ξ~N(0,1).  "
        "α=0 ⇒ all-to-all Gaussian p=3 (SK baseline)."
    )
