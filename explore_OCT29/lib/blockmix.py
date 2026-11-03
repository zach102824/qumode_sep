"""Bit-frame block-mixer SNAP (ideal W) for arbitrary bits-per-cavity b.

Logical circuit (Opus gate-0):
  |+>^{⊗n} → ∏_ℓ [ U_C(γ_ℓ)  M(θ_ℓ)^{⊗C} ]
  M(θ) = H^{⊗b} diag(e^{-i θ}) H^{⊗b}
  U_C: ψ *= exp(-i γ Cn), Cn = E · (n/m), m = round(9.93 n)

QAOA point: θ_B = -β · popcount(B)  ⇒  M = ⊗ exp(-i β X / 2)  (global phase free).

State layout: flat length-2^n, var0 = MSB (matches explore_OCT29.lib.qaoa).
Cavity c owns bits [c*b, (c+1)*b) in MSB-first order.
"""
from __future__ import annotations

import math
from typing import Optional, Sequence, Tuple

import numpy as np
import torch

from .qaoa import clause_count, inv_p_from_psucc, qaoa_psucc, summarize_inv


def popcount_levels(b: int, device=None, dtype=torch.float32) -> torch.Tensor:
    """popcount(0..2^b-1) as float tensor."""
    dim = 1 << int(b)
    pc = torch.tensor([bin(i).count("1") for i in range(dim)], device=device, dtype=torch.long)
    return pc.to(dtype=dtype)


def qaoa_theta_from_beta(betas: torch.Tensor, b: int) -> torch.Tensor:
    """θ_{ℓ,B} = -β_ℓ · pc(B). Shape (L, 2^b)."""
    betas = betas.to(dtype=torch.float32).reshape(-1)
    pc = popcount_levels(b, device=betas.device, dtype=torch.float32)  # (2^b,)
    return (-betas).reshape(-1, 1) * pc.reshape(1, -1)


def shell_phi_from_beta(betas: torch.Tensor, b: int) -> torch.Tensor:
    """Arm-C style φ_pc = -β · pc for pc=0..b; shape (L, b+1)."""
    betas = betas.to(dtype=torch.float32).reshape(-1)
    pcs = torch.arange(b + 1, dtype=torch.float32, device=betas.device)
    return (-betas).reshape(-1, 1) * pcs.reshape(1, -1)


def expand_head_theta(
    mode: str,
    vec: torch.Tensor,
    b: int,
    *,
    few_M: int = 3,
) -> torch.Tensor:
    """Map a head parameter vector → length-2^b phases θ_B.

    Modes (generalized from Nov snap_heads for arbitrary b):
      x_pop  — θ = β · pc            vec (1,)
      shell  — θ = φ_{pc}            vec (b+1,)
      gauss  — θ = α + β pc + δ pc²  vec (3,)
      free   — θ_B free              vec (2^b,)
      bits   — θ = Σ_j α_j bit_j     vec (b,)
    """
    dim = 1 << int(b)
    device = vec.device
    vec = vec.to(dtype=torch.float32).reshape(-1)
    pc = popcount_levels(b, device=device, dtype=torch.float32)
    ks = torch.arange(dim, device=device, dtype=torch.long)

    if mode == "x_pop":
        return vec.reshape(-1)[0] * pc
    if mode == "shell":
        ph = vec.reshape(b + 1)
        return ph[pc.long()]
    if mode == "gauss":
        a, bb, d = vec.reshape(3)
        return a + bb * pc + d * pc * pc
    if mode == "free":
        return vec.reshape(dim)
    if mode == "bits":
        a = vec.reshape(b)
        th = torch.zeros(dim, dtype=torch.float32, device=device)
        # bit j of level k: LSB of k is bit 0 (Fock-style)
        for j in range(b):
            th = th + a[j] * ((ks >> j) & 1).to(torch.float32)
        return th
    if mode == "few_m":
        th = torch.zeros(dim, dtype=torch.float32, device=device)
        m = min(int(few_M), dim)
        th[:m] = vec.reshape(-1)[:m]
        return th
    raise ValueError(f"unknown mode {mode}")


def head_dim(mode: str, b: int, few_M: int = 3) -> int:
    if mode == "x_pop":
        return 1
    if mode == "shell":
        return b + 1
    if mode == "gauss":
        return 3
    if mode == "free":
        return 1 << b
    if mode == "bits":
        return b
    if mode == "few_m":
        return int(few_M)
    raise ValueError(mode)


def _apply_H_on_cavity(psi: torch.Tensor, n: int, b: int, cav: int) -> torch.Tensor:
    """Apply H^{⊗b} on cavity `cav` (MSB-first bit blocks). psi: (B, 2^n)."""
    Bsz, dim = psi.shape
    assert dim == (1 << n)
    C = n // b
    assert 0 <= cav < C
    # bits owned: start = cav*b .. cav*b+b-1 in MSB-first axis order
    # reshape to (B, 2^{cav*b}, 2^b, 2^{n-(cav+1)*b})
    left = cav * b
    right = n - (cav + 1) * b
    x = psi.reshape(Bsz, 1 << left, 1 << b, 1 << right)
    # Walsh–Hadamard on axis 2 via iterative single-bit H
    for j in range(b):
        # bit j within the cavity block (j=0 = MSB of the b-block)
        pre = j
        post = b - 1 - j
        y = x.reshape(Bsz, 1 << left, 1 << pre, 2, 1 << post, 1 << right)
        t0 = y[:, :, :, 0, :, :]
        t1 = y[:, :, :, 1, :, :]
        inv = 1.0 / math.sqrt(2.0)
        y0 = (t0 + t1) * inv
        y1 = (t0 - t1) * inv
        x = torch.stack((y0, y1), dim=3).reshape(Bsz, 1 << left, 1 << b, 1 << right)
    return x.reshape(Bsz, dim)


def _apply_diag_phase_cavity(
    psi: torch.Tensor, theta: torch.Tensor, n: int, b: int, cav: int
) -> torch.Tensor:
    """Multiply by e^{-i θ[level]} on cavity levels. theta: (2^b,)."""
    Bsz, dim = psi.shape
    left = cav * b
    right = n - (cav + 1) * b
    x = psi.reshape(Bsz, 1 << left, 1 << b, 1 << right)
    phase = torch.exp(-1j * theta.to(torch.complex64)).reshape(1, 1, -1, 1)
    x = x * phase
    return x.reshape(Bsz, dim)


def apply_block_mixer(
    psi: torch.Tensor, theta: torch.Tensor, n: int, b: int
) -> torch.Tensor:
    """Apply M(θ)^{⊗C} with M = H^{⊗b} diag(e^{-iθ}) H^{⊗b}. theta: (2^b,)."""
    if n % b != 0:
        raise ValueError(f"n={n} not divisible by b={b}")
    C = n // b
    theta = theta.to(dtype=torch.float32).reshape(1 << b)
    for c in range(C):
        psi = _apply_H_on_cavity(psi, n, b, c)
        psi = _apply_diag_phase_cavity(psi, theta, n, b, c)
        psi = _apply_H_on_cavity(psi, n, b, c)
    return psi


def blockmix_psucc(
    gammas: torch.Tensor,
    thetas: torch.Tensor,
    E: torch.Tensor,
    n: int,
    b: int,
    m: Optional[int] = None,
) -> torch.Tensor:
    """Batched success probability. E: (B, 2^n); thetas: (L, 2^b); returns (B,)."""
    if E.ndim != 2:
        raise ValueError(f"E must be (B, 2^n), got {tuple(E.shape)}")
    Bsz, dim = E.shape
    if dim != (1 << int(n)):
        raise ValueError(f"E width {dim} != 2^{n}")
    if n % b != 0:
        raise ValueError(f"n={n} not divisible by b={b}")
    if m is None:
        m = clause_count(n)
    Cn = E.to(dtype=torch.float32) * (float(n) / float(m))
    device = E.device
    gammas = gammas.to(device=device, dtype=torch.float32).reshape(-1)
    thetas = thetas.to(device=device, dtype=torch.float32)
    if thetas.ndim == 1:
        thetas = thetas.reshape(1, -1)
    if thetas.shape[0] != gammas.numel():
        raise ValueError(f"L mismatch gammas={gammas.numel()} thetas={thetas.shape}")
    if thetas.shape[1] != (1 << b):
        raise ValueError(f"theta width {thetas.shape[1]} != 2^{b}")

    psi = torch.full(
        (Bsz, dim),
        1.0 / math.sqrt(dim),
        dtype=torch.complex64,
        device=device,
    )
    for ell in range(int(gammas.numel())):
        g = gammas[ell].to(torch.complex64)
        psi = psi * torch.exp(-1j * g * Cn.to(torch.complex64))
        psi = apply_block_mixer(psi, thetas[ell], int(n), int(b))
    probs = psi.real.square() + psi.imag.square()
    sat = (E == 0).to(dtype=probs.dtype)
    return (probs * sat).sum(dim=-1)


def dual_qaoa_psucc(
    betas: torch.Tensor,
    gammas: torch.Tensor,
    E: torch.Tensor,
    n: int,
    b: int,
    m: Optional[int] = None,
) -> torch.Tensor:
    """QAOA-warm dual: θ = -β · pc on packing b."""
    thetas = qaoa_theta_from_beta(betas, b)
    return blockmix_psucc(gammas, thetas, E, n, b, m=m)


@torch.no_grad()
def fidelity_vs_qaoa(
    betas: torch.Tensor,
    gammas: torch.Tensor,
    E: torch.Tensor,
    n: int,
    b: int,
) -> Tuple[float, float]:
    """Compare blockmix dual-QAOA p_succ to ideal qubit QAOA. Returns (max|Δp|, max|rel|)."""
    p_q = qaoa_psucc(betas, gammas, E, n)
    p_b = dual_qaoa_psucc(betas, gammas, E, n, b)
    dp = (p_b - p_q).abs()
    rel = dp / p_q.clamp(min=1e-12)
    return float(dp.max().cpu()), float(rel.max().cpu())
