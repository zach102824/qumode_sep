"""Ideal qubit QAOA for 4-SAT (Boulebnane–Montanaro half-angle convention).

Angle convention (Nov_3rd/qaoa_init.ANGLE_CONVENTION):
  U_C = exp(-i γ Cn),  U_B = exp(-i β X/2),  Cn = E·n/m
  m = round(9.93 n)
  start |+⟩^{⊗n}; each layer is cost then mixer.
  State index: var0 = MSB (same layout as sat_gen.fast_energies).

This module was missing from the tree. The circuit above is the one
blockmix.fidelity_vs_qaoa and Nov_3rd T6 compare against.
"""
from __future__ import annotations

import math

import numpy as np
import torch

CLAUSE_RATIO = 9.93


def clause_count(n: int) -> int:
    return int(round(CLAUSE_RATIO * float(n)))


def inv_p_from_psucc(ps) -> np.ndarray:
    arr = np.asarray(ps, dtype=np.float64).reshape(-1)
    return 1.0 / np.maximum(arr, 1e-12)


def summarize_inv(inv) -> dict:
    raw = np.asarray(inv, dtype=np.float64).reshape(-1)
    finite = np.isfinite(raw)
    v = raw[finite]
    if v.size == 0:
        med = b90 = mean = frac = float("nan")
    else:
        med = float(np.median(v))
        b90 = float(np.quantile(v, 0.90))
        mean = float(np.mean(v))
        frac = float(np.mean(v <= 2.0))
    return {
        "median_inv_p": med,
        "B90_inv_p": b90,
        "mean_inv_p": mean,
        "frac_inv_le_2": frac,
        "B": int(raw.size),
        "n_finite": int(finite.sum()),
    }


def _apply_mixer_half(psi: torch.Tensor, beta: torch.Tensor, n: int) -> torch.Tensor:
    """Apply ⊗_j exp(-i (β/2) X_j). psi is (B, 2^n), var0 = MSB."""
    theta = (beta.to(dtype=psi.real.dtype) * 0.5)
    c = torch.cos(theta).to(dtype=psi.dtype)
    s = (-1j * torch.sin(theta)).to(dtype=psi.dtype)
    B = psi.shape[0]
    t = psi.reshape(B, *([2] * int(n)))
    for q in range(int(n)):
        t = t.movedim(q + 1, 1)
        t0 = t[:, 0]
        t1 = t[:, 1]
        t = torch.stack((c * t0 + s * t1, s * t0 + c * t1), dim=1)
        t = t.movedim(1, q + 1)
    return t.reshape(B, -1)


def qaoa_psucc(
    betas: torch.Tensor,
    gammas: torch.Tensor,
    E: torch.Tensor,
    n: int,
    m: int | None = None,
) -> torch.Tensor:
    """Success probability (mass on E==0). E is (B, 2^n) or (2^n,). Returns (B,)."""
    if E.ndim == 1:
        E = E.unsqueeze(0)
    if E.ndim != 2:
        raise ValueError(f"E must be (B, 2^n), got {tuple(E.shape)}")
    Bsz, dim = E.shape
    n = int(n)
    if dim != (1 << n):
        raise ValueError(f"E width {dim} != 2^{n}")
    if m is None:
        m = clause_count(n)
    device = E.device
    betas = betas.to(device=device, dtype=torch.float32).reshape(-1)
    gammas = gammas.to(device=device, dtype=torch.float32).reshape(-1)
    if betas.numel() != gammas.numel():
        raise ValueError(f"L mismatch betas={betas.numel()} gammas={gammas.numel()}")
    Cn = E.to(dtype=torch.float32) * (float(n) / float(m))
    psi = torch.full(
        (Bsz, dim),
        1.0 / math.sqrt(dim),
        dtype=torch.complex64,
        device=device,
    )
    for ell in range(int(betas.numel())):
        g = gammas[ell].to(dtype=torch.complex64)
        psi = psi * torch.exp(-1j * g * Cn.to(dtype=torch.complex64))
        psi = _apply_mixer_half(psi, betas[ell], n)
    probs = psi.real.square() + psi.imag.square()
    sat = (E == 0).to(dtype=probs.dtype)
    return (probs * sat).sum(dim=-1)
