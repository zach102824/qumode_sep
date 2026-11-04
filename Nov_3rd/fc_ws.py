"""Footprint-class Walsh × commutator-matched shell (FC-WS).

Opus Idea 1 (joint U_c^X + Hamming-shell SNAP). Graded H_c, labeled as such.

Classes C ∈ {(1), (2), (1,1), (3), (2,1), (1,1,1)} group Walsh monomials by how
many bits of the monomial sit in each cavity they touch (b=3).

U_c^X layer ℓ:
    D_ℓ = exp(-i γ_ℓ Σ_C (1+δ_C) Σ_{S∈C} ŵ_S Z_S)
with degree ≥4 residual kept at weight 1 (no δ) and the degree-0 Walsh mode
omitted (global phase). δ=0 recovers scalar γ·H_c up to that global phase.
If the base schedule is already degree-graded γ_{ℓ,w}, the same formula uses
γ_{ℓ,degree(C)} in place of γ_ℓ, so δ=0 recovers Walsh-degree γ_w exactly.

Anticommutation table O_{r,t}: fraction of weight-t X-strings on one cavity
that anticommute with a fixed r-bit Z-piece (odd overlap), t=1,2,3:
    r=1: (1/3, 2/3, 1)
    r=2: (2/3, 2/3, 0)
    r=3: (1,   0,   1)

Shell orders are the radial Walsh / Krawtchouk coefficients of the champ
phases φ_k (k=0..3). With M[t, k] = K_t(k; 3) and M^2 = 2^3 I,
    μ = φ @ M,    φ = μ @ M / 8.
(The forward sum is Σ_k K_k(t) φ_k, which is φ @ M, not φ @ M^T.
M^2 = 8 I supplies the inverse.)
FC-WS rescales μ_{ℓ,t} ← μ_{ℓ,t}^champ · (1 + κ Õ_t) for t=1,2,3 (μ_0 fixed)
and maps back. Õ is one global coupling of the class boosts to the orders
that anticommute with them:

    Õ_t = Σ_C w_C δ_C Σ_{r∈pieces(C)} O_{r,t}  /  Σ_C w_C |pieces(C)|

w_C = Σ_{S∈C} |ŵ_S| is the instance's Walsh mass of class C (from H_c).
κ=0 leaves φ at the champ shell. δ_C = δ_{degree(C)} and κ=0 is Walsh-degree
grading with a shared per-degree factor (the γ_w special case inside this
7-parameter family when the base γ_ℓ is scalar).
"""
from __future__ import annotations

from math import comb

import torch

# Canonical class order. Index 5 is (1,1,1), optionally tied to index 2 (1,1).
CLASSES: tuple[tuple[int, ...], ...] = (
    (1,),
    (2,),
    (1, 1),
    (3,),
    (2, 1),
    (1, 1, 1),
)
CLASS_NAME = ("(1)", "(2)", "(1,1)", "(3)", "(2,1)", "(1,1,1)")
CLASS_DEGREE = (1, 2, 2, 3, 3, 3)  # total bits = degree
N_CLASS = 6
# Index into a (4,) degree-graded γ for degrees 1..3. Degree 4 is the residual.
CLASS_DEG_INDEX = (0, 1, 1, 2, 2, 2)

# O[r-1, t-1] for r,t in {1,2,3}
O_TABLE = (
    (1.0 / 3.0, 2.0 / 3.0, 1.0),
    (2.0 / 3.0, 2.0 / 3.0, 0.0),
    (1.0, 0.0, 1.0),
)


def krawtchouk(t: int, k: int, n: int = 3) -> int:
    """K_t(k; n) = Σ_j (-1)^j C(k,j) C(n-k, t-j)."""
    s = 0
    for j in range(t + 1):
        if j <= k and (t - j) <= (n - k):
            s += ((-1) ** j) * comb(k, j) * comb(n - k, t - j)
    return s


def krawtchouk_matrix(n: int = 3) -> torch.Tensor:
    """M[t, k] = K_t(k; n), shape (n+1, n+1), float64."""
    m = [[float(krawtchouk(t, k, n)) for k in range(n + 1)] for t in range(n + 1)]
    return torch.tensor(m, dtype=torch.float64)


# n=3 shell: rows t=0..3, cols k=0..3. M @ M = 8 I (involution up to 2^n).
KRAW_M = krawtchouk_matrix(3)


def _order_piece_table() -> tuple[torch.Tensor, torch.Tensor]:
    """S_O[c, t] = Σ_{r∈pieces(C)} O_{r,t}, n_pieces[c] = |pieces(C)|."""
    rows = []
    npieces = []
    for pieces in CLASSES:
        acc = [0.0, 0.0, 0.0]
        for r in pieces:
            for t in range(3):
                acc[t] += O_TABLE[r - 1][t]
        rows.append(acc)
        npieces.append(float(len(pieces)))
    return (
        torch.tensor(rows, dtype=torch.float64),
        torch.tensor(npieces, dtype=torch.float64),
    )


S_O, N_PIECES = _order_piece_table()


def effective_delta(delta: torch.Tensor, pool_111: bool) -> torch.Tensor:
    """Map a (6,) or pooled (5,) δ to the six class multipliers.

    Pooling ties δ_{(1,1,1)} = δ_{(1,1)} (Opus failure mode: (1,1,1) is absent
    at n=6 and rare at n=9). Classes stay separate in the circuit; only the
    parameter is shared.
    """
    if delta.numel() == 5:
        return torch.stack([delta[0], delta[1], delta[2], delta[3], delta[4], delta[2]])
    if delta.numel() != 6:
        raise ValueError(f"delta numel {delta.numel()} not in {{5,6}}")
    if not pool_111:
        return delta
    return torch.stack([delta[0], delta[1], delta[2], delta[3], delta[4], delta[2]])


def o_tilde(w: torch.Tensor, delta_eff: torch.Tensor) -> torch.Tensor:
    """Õ_t for t=1,2,3. w and delta_eff are (6,). Returns (3,).

    Zero Walsh mass → Õ = 0 (no shell correction on that instance).
    """
    w64 = w.to(dtype=torch.float64).reshape(6)
    d64 = delta_eff.to(dtype=torch.float64).reshape(6)
    so = S_O.to(device=w64.device)
    npce = N_PIECES.to(device=w64.device)
    numer = (w64 * d64) @ so  # (3,)
    denom = (w64 * npce).sum()
    # Keep a grad path even when denom vanishes.
    return numer / denom.clamp(min=1e-12)


def mu_from_phi(phi: torch.Tensor) -> torch.Tensor:
    """φ (..., 4) → μ (..., 4). μ = φ @ M, M[t,k]=K_t(k;3), so μ_t = Σ_k K_k(t) φ_k."""
    M = KRAW_M.to(device=phi.device)
    return phi.to(torch.float64) @ M


def phi_from_mu(mu: torch.Tensor) -> torch.Tensor:
    """μ (..., 4) → φ (..., 4). φ = μ @ M / 8, since M^2 = 8 I and φ_k = 2^{-3} Σ_t K_t(k) μ_t."""
    M = KRAW_M.to(device=mu.device)
    return (mu.to(torch.float64) @ M) / 8.0


def rescale_shell_phi(phi: torch.Tensor, o_t: torch.Tensor, kappa: torch.Tensor) -> torch.Tensor:
    """μ_t ← μ_t (1+κ Õ_t) for t≥1, then inverse Krawtchouk. phi: (L, 4)."""
    mu = mu_from_phi(phi)
    ot = o_t.to(dtype=torch.float64, device=mu.device).reshape(3)
    kap = kappa.to(dtype=torch.float64, device=mu.device).reshape(())
    scale_tail = 1.0 + kap * ot
    ones = torch.ones(1, dtype=torch.float64, device=mu.device)
    scale = torch.cat([ones, scale_tail], dim=0)  # (4,) t=0..3
    phi2 = phi_from_mu(mu * scale)
    return phi2.to(dtype=phi.dtype)


def footprint_class_ids(n: int, b: int = 3, device=None) -> torch.Tensor:
    """Walsh-index → class id, shape (2^n,).

    -1 degree 0 (dropped), -2 degree≥4 or any non-shell footprint,
    0..5 the six classes. Var v is bit (n-1-v) of the integer index
    (var0 = MSB), cavity c owns vars 3c..3c+2.
    """
    if n % b != 0:
        raise ValueError(f"n={n} not divisible by b={b}")
    n_cav = n // b
    N = 1 << n
    dev = device
    idx = torch.arange(N, device=dev, dtype=torch.int64)
    counts = []
    for c in range(n_cav):
        cnt = torch.zeros(N, dtype=torch.int64, device=dev)
        for j in range(b):
            v = b * c + j
            bit = n - 1 - v
            cnt = cnt + ((idx >> bit) & 1)
        counts.append(cnt)
    counts_t = torch.stack(counts, dim=1)  # (N, C)
    deg = counts_t.sum(dim=1)
    sorted_c, _ = torch.sort(counts_t, dim=1, descending=True)
    a = sorted_c[:, 0]
    b1 = sorted_c[:, 1] if n_cav >= 2 else torch.zeros(N, dtype=torch.int64, device=dev)
    c3 = sorted_c[:, 2] if n_cav >= 3 else torch.zeros(N, dtype=torch.int64, device=dev)
    ids = torch.full((N,), -1, dtype=torch.int64, device=dev)
    ids = torch.where((a == 1) & (b1 == 0), torch.zeros_like(ids), ids)
    ids = torch.where((a == 2) & (b1 == 0), torch.ones_like(ids), ids)
    ids = torch.where((a == 1) & (b1 == 1) & (c3 == 0), torch.full_like(ids, 2), ids)
    ids = torch.where((a == 3) & (b1 == 0), torch.full_like(ids, 3), ids)
    ids = torch.where((a == 2) & (b1 == 1) & (c3 == 0), torch.full_like(ids, 4), ids)
    ids = torch.where((a == 1) & (b1 == 1) & (c3 == 1) & (deg == 3), torch.full_like(ids, 5), ids)
    unmatched = (deg > 0) & (ids < 0)
    ids = torch.where(unmatched, torch.full_like(ids, -2), ids)
    return ids


def anticommute_fraction(r: int, t: int, nbits: int = 3) -> float:
    """Fraction of t-subsets with odd overlap against a fixed r-subset."""
    from itertools import combinations

    bits = range(nbits)
    R = set(range(r))
    total = 0
    odd = 0
    for T in combinations(bits, t):
        total += 1
        if len(R.intersection(T)) % 2 == 1:
            odd += 1
    return odd / total if total else float("nan")
