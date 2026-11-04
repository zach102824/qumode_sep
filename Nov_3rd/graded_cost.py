"""Graded H_c cost pieces for U_c^X (Walsh-degree split).

E is the unsat-clause count on bitstrings (var0 = MSB), same layout as
GadgetAnsatz.cost_Cn / sat_gen.fast_energies. Degree-w Walsh components
are obtained by FWHT → popcount mask → iFWHT. The w=0 term is a global
phase and is dropped.

Label: graded H_c. Not a projector gadget; equal γ_w recovers γ·E.
"""
from __future__ import annotations

import torch

from explore_OCT29.lib.sat_gen import clause_count

WALSH_DEGREES = (1, 2, 3, 4)


def fwht(x: torch.Tensor) -> torch.Tensor:
    """Unnormalized Walsh–Hadamard transform along the last axis.

    Self-inverse up to a factor of N = x.shape[-1] (must be a power of 2).
    Bit 0 of the integer index is the LSB, matching χ_S(x)=(-1)^{popcount(S&x)}
    on the same layout as E[x] (var0 = MSB of x).
    """
    n_last = int(x.shape[-1])
    if n_last <= 0 or (n_last & (n_last - 1)) != 0:
        raise ValueError(f"last dim {n_last} is not a power of 2")
    y = x.contiguous().clone()
    orig = y.shape
    h = 1
    while h < n_last:
        y = y.reshape(*orig[:-1], n_last // (2 * h), 2, h)
        a = y[..., 0, :].clone()
        b = y[..., 1, :].clone()
        y[..., 0, :] = a + b
        y[..., 1, :] = a - b
        y = y.reshape(*orig)
        h *= 2
    return y


def popcount_arange(n: int, device=None) -> torch.Tensor:
    """popcount(0..2^n-1) via bit-smear; no Python loop over 2^n."""
    N = 1 << n
    idx = torch.arange(N, device=device)
    bits = (idx.unsqueeze(-1) >> torch.arange(n, device=device)) & 1
    return bits.sum(dim=-1)


def walsh_degree_E(E: torch.Tensor, n: int, degrees=WALSH_DEGREES) -> torch.Tensor:
    """Split E into degree-w Walsh components.

    E: (2^n,) or (B, 2^n) unsat counts. Returns (..., W, 2^n) with W=len(degrees).
    Reconstruction: E ≈ mean(E) + sum_w E_w  (w=0 is the constant).
    """
    if E.shape[-1] != (1 << n):
        raise ValueError(f"E last dim {E.shape[-1]} != 2^{n}")
    work = E.to(dtype=torch.float64)
    N = work.shape[-1]
    hat = fwht(work)
    pc = popcount_arange(n, device=work.device)
    parts = []
    for w in degrees:
        mask = (pc == int(w)).to(dtype=hat.dtype)
        parts.append(fwht(hat * mask) / float(N))
    stacked = torch.stack(parts, dim=-2)  # (..., W, 2^n)
    return stacked.to(dtype=torch.float32 if E.dtype == torch.float32 else E.dtype)


def cn_parts_from_E(E: torch.Tensor, n: int, grid_to_x: torch.Tensor, degrees=WALSH_DEGREES) -> torch.Tensor:
    """Map Walsh components onto the Fock product grid with cost_Cn scale n/m.

    Does NOT subtract amax — the circuit amax-centers the γ-weighted sum.
    Returns (W, *grid) for a single instance, or (B, W, *grid) if E is batched.
    """
    m = float(clause_count(n))
    scale = float(n) / m
    Ew = walsh_degree_E(E, n, degrees=degrees)
    g = grid_to_x.long().reshape(-1)
    grid_shape = tuple(grid_to_x.shape)
    if Ew.dim() == 2:
        parts = Ew[:, g] * scale  # (W, 2^n_grid)
        return parts.reshape(Ew.shape[0], *grid_shape)
    # batched (B, W, 2^n)
    parts = Ew[:, :, g] * scale
    return parts.reshape(Ew.shape[0], Ew.shape[1], *grid_shape)


def equal_gamma_rel_err(E: torch.Tensor, n: int, grid_to_x: torch.Tensor, gamma: float = 0.37) -> float:
    """Relative RMS of equal-γ_w arg vs γ·cost_Cn (amax-centered)."""
    m = float(clause_count(n))
    scale = float(n) / m
    E = E.to(dtype=torch.float64).reshape(-1)
    E_grid = E[grid_to_x.long().reshape(-1)].reshape(grid_to_x.shape)
    cn = scale * E_grid
    cn = cn - cn.amax()
    parts = cn_parts_from_E(E.to(dtype=torch.float32), n, grid_to_x).to(dtype=torch.float64)
    arg = float(gamma) * parts.sum(dim=0)
    arg = arg - arg.amax()
    ref = float(gamma) * cn
    num = (arg - ref).pow(2).mean().sqrt()
    den = ref.pow(2).mean().sqrt().clamp(min=1e-12)
    return float((num / den).item())
