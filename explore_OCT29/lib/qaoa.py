"""Ideal qubit QAOA reference for 4-SAT TEB packs.

  |+>^{⊗n} → ∏_ℓ [ U_B(β_ℓ) U_C(γ_ℓ) ]
  U_C = exp(-i γ Cn),  Cn = E · (n/m),  m = round(9.93 n)
  U_B = ⊗_q exp(-i β X_q / 2)

State layout: flat length-2^n, var0 = MSB.
"""
from __future__ import annotations

import math
from typing import Optional, Sequence

import numpy as np
import torch

from .sat_gen import clause_count

__all__ = ["clause_count", "qaoa_psucc", "inv_p_from_psucc", "summarize_inv"]


def _apply_x_mixer(psi: torch.Tensor, beta: torch.Tensor, n: int) -> torch.Tensor:
    """Apply ⊗ exp(-i β X/2) to (B, 2^n) state."""
    c = torch.cos(beta / 2).to(psi.dtype)
    s = (-1j * torch.sin(beta / 2)).to(psi.dtype)
    Bsz = psi.shape[0]
    for q in range(n):
        v = psi.reshape(Bsz, 1 << q, 2, 1 << (n - 1 - q))
        a0, a1 = v[:, :, 0, :], v[:, :, 1, :]
        psi = torch.stack([c * a0 + s * a1, s * a0 + c * a1], dim=2).reshape(Bsz, -1)
    return psi


def qaoa_psucc(
    betas: torch.Tensor,
    gammas: torch.Tensor,
    E: torch.Tensor,
    n: int,
    m: Optional[int] = None,
) -> torch.Tensor:
    """Batched success probability. E: (B, 2^n) unsat counts; returns (B,)."""
    if E.ndim == 1:
        E = E.reshape(1, -1)
    Bsz, dim = E.shape
    if dim != (1 << int(n)):
        raise ValueError(f"E width {dim} != 2^{n}")
    if m is None:
        m = clause_count(n)
    device = E.device
    Cn = E.to(dtype=torch.float32) * (float(n) / float(m))
    betas = betas.to(device=device, dtype=torch.float32).reshape(-1)
    gammas = gammas.to(device=device, dtype=torch.float32).reshape(-1)
    if betas.numel() != gammas.numel():
        raise ValueError(f"L mismatch betas={betas.numel()} gammas={gammas.numel()}")
    psi = torch.full((Bsz, dim), 1.0 / math.sqrt(dim), dtype=torch.complex64, device=device)
    for ell in range(int(gammas.numel())):
        psi = psi * torch.exp(-1j * gammas[ell].to(torch.complex64) * Cn.to(torch.complex64))
        psi = _apply_x_mixer(psi, betas[ell], int(n))
    probs = psi.real.square() + psi.imag.square()
    sat = (E == 0).to(dtype=probs.dtype)
    return (probs * sat).sum(dim=-1)


def inv_p_from_psucc(ps) -> np.ndarray:
    ps = np.asarray(ps, dtype=np.float64).reshape(-1)
    with np.errstate(divide="ignore"):
        return np.where(ps > 0, 1.0 / np.maximum(ps, 1e-300), np.inf)


def summarize_inv(invs: Sequence[float]) -> dict:
    a = np.asarray(invs, dtype=np.float64).reshape(-1)
    fin = a[np.isfinite(a)]
    if fin.size == 0:
        nan = float("nan")
        return {
            "median_inv_p": nan,
            "B90_inv_p": nan,
            "mean_inv_p": nan,
            "frac_inv_le_2": nan,
            "B": int(a.size),
            "n_finite": 0,
        }
    return {
        "median_inv_p": float(np.median(a)),
        "B90_inv_p": float(np.quantile(a, 0.90)),
        "mean_inv_p": float(np.mean(fin)),
        "frac_inv_le_2": float(np.mean(a <= 2.0)),
        "B": int(a.size),
        "n_finite": int(fin.size),
    }
