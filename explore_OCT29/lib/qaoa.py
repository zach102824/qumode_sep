"""Ideal qubit QAOA used by explore_OCT29 / Nov_3rd.

Convention (qaoa_ref / Nov_3rd.qaoa_init):
  |+>^n,  U_C = exp(-i γ Cn),  U_B = exp(-i β X/2),  Cn = E * n / m
  flat index, var0 = MSB (matches sat_gen.fast_energies).
"""
from __future__ import annotations

import math

import numpy as np
import torch

from .sat_gen import clause_count

__all__ = [
    "clause_count",
    "grow_angles",
    "inv_p_from_psucc",
    "linear_ramp_angles",
    "paper_k4_p10_mapped",
    "qaoa_psucc",
    "summarize_inv",
]


def inv_p_from_psucc(ps, eps: float = 1e-12) -> np.ndarray:
    p = np.asarray(ps, dtype=np.float64).reshape(-1)
    return 1.0 / np.maximum(p, eps)


def summarize_inv(inv) -> dict:
    arr = np.asarray(inv, dtype=np.float64).reshape(-1)
    finite = np.isfinite(arr)
    x = arr[finite]
    if x.size == 0:
        med = b90 = mean = frac = float("nan")
    else:
        med = float(np.median(x))
        b90 = float(np.quantile(x, 0.9))
        mean = float(np.mean(x))
        frac = float(np.mean(x <= 2.0))
    return {
        "median_inv_p": med,
        "B90_inv_p": b90,
        "mean_inv_p": mean,
        "frac_inv_le_2": frac,
        "B": int(arr.size),
        "n_finite": int(finite.sum()),
    }


def _apply_x_mixer(psi: torch.Tensor, beta: torch.Tensor, n: int) -> torch.Tensor:
    """⊗ exp(-i β X/2). psi: (B, 2^n), qubit 0 = MSB."""
    c = torch.cos(beta * 0.5).to(dtype=torch.complex64)
    s = (-1j * torch.sin(beta * 0.5)).to(dtype=torch.complex64)
    B = psi.shape[0]
    t = psi
    for q in range(n):
        x = t.reshape(B, 1 << q, 2, 1 << (n - 1 - q))
        t0 = x[:, :, 0, :]
        t1 = x[:, :, 1, :]
        y0 = c * t0 + s * t1
        y1 = s * t0 + c * t1
        t = torch.stack((y0, y1), dim=2).reshape(B, 1 << n)
    return t


def qaoa_psucc(
    betas: torch.Tensor,
    gammas: torch.Tensor,
    E: torch.Tensor,
    n: int,
    m: int | None = None,
) -> torch.Tensor:
    """Success probability on satisfying assignments. E: (B, 2^n) or (2^n,)."""
    if E.ndim == 1:
        E = E.reshape(1, -1)
    if E.ndim != 2:
        raise ValueError(f"E must be (B, 2^n), got {tuple(E.shape)}")
    B, dim = E.shape
    n = int(n)
    if dim != (1 << n):
        raise ValueError(f"E width {dim} != 2^{n}")
    if m is None:
        m = clause_count(n)
    device = E.device
    Cn = E.to(dtype=torch.float32, device=device) * (float(n) / float(m))
    betas = betas.to(device=device, dtype=torch.float32).reshape(-1)
    gammas = gammas.to(device=device, dtype=torch.float32).reshape(-1)
    if betas.numel() != gammas.numel():
        raise ValueError(f"L mismatch betas={betas.numel()} gammas={gammas.numel()}")
    psi = torch.full(
        (B, dim),
        1.0 / math.sqrt(dim),
        dtype=torch.complex64,
        device=device,
    )
    for ell in range(int(betas.numel())):
        g = gammas[ell].to(dtype=torch.complex64)
        psi = psi * torch.exp(-1j * g * Cn.to(dtype=torch.complex64))
        psi = _apply_x_mixer(psi, betas[ell], n)
    probs = psi.real.square() + psi.imag.square()
    sat = (E == 0).to(dtype=probs.dtype)
    return (probs * sat).sum(dim=-1)


def linear_ramp_angles(L: int, scale: float = 1.0):
    """qaoa_ref ramp: β from 0.8→0.1, γ from 0.1→0.8, times scale."""
    L = int(L)
    b = np.linspace(0.8, 0.1, L) * float(scale)
    g = np.linspace(0.1, 0.8, L) * float(scale)
    return b.astype(np.float64), g.astype(np.float64)


def grow_angles(betas, gammas, L: int):
    """Resample a published angle schedule onto L layers."""
    b = np.asarray(betas, dtype=np.float64).reshape(-1)
    g = np.asarray(gammas, dtype=np.float64).reshape(-1)
    p = int(b.size)
    L = int(L)
    if L == p:
        return b.copy(), g.copy()
    src = np.linspace(0.0, 1.0, p)
    dst = np.linspace(0.0, 1.0, L)
    return np.interp(dst, src, b), np.interp(dst, src, g)


def paper_k4_p10_mapped():
    """Published k=4 p=10 angles in qaoa_ref units: b=β, g=-γ/2 (Cn=E when m=n)."""
    from qaoa_ksat.angles import paper_angles

    betas, gammas = paper_angles(4, 10, 9.93)
    return np.asarray(betas, dtype=np.float64), -np.asarray(gammas, dtype=np.float64) / 2.0
