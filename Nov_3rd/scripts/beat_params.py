"""Shared shell polynomial params for beat_qaoa_12h (training-only; gates locked).

φ_k = c1·k + c2·k² + c3·k³   (k=0..3)
QAOA dual warm: c1=-β, c2=c3=0.
"""
from __future__ import annotations

import torch

from Nov_3rd import settings as S


def k_powers(device=None, dtype=torch.float32) -> torch.Tensor:
    k = torch.arange(S.B_BITS + 1, dtype=dtype, device=device)  # 0..3
    return torch.stack([k, k ** 2, k ** 3], dim=0)  # (3, 4)


def phi_from_c(c1: torch.Tensor, c2: torch.Tensor, c3: torch.Tensor) -> torch.Tensor:
    """c*: (L,) → phi (L, 4)."""
    L = int(c1.numel())
    pows = k_powers(device=c1.device, dtype=c1.dtype)  # (3,4)
    C = torch.stack([c1.reshape(L), c2.reshape(L), c3.reshape(L)], dim=1)  # (L,3)
    return C @ pows  # (L,4)


def warm_from_qaoa(betas: torch.Tensor, gammas: torch.Tensor):
    """Return gamma, c1, c2, c3 tensors (not Parameters)."""
    L = int(betas.numel())
    gamma = gammas.reshape(L).float().contiguous()
    c1 = (-betas.reshape(L)).float().contiguous()
    c2 = torch.zeros(L, dtype=torch.float32)
    c3 = torch.zeros(L, dtype=torch.float32)
    return gamma, c1, c2, c3


def scale_c23(c2: torch.Tensor, c3: torch.Tensor, n: int, kappa: torch.Tensor | float):
    """Rule-scaled dual-breaking: c_{2,3}^{(n)} = κ · 2/(n-1) · c_{2,3}."""
    a = float(kappa) if not torch.is_tensor(kappa) else kappa
    s = a * (2.0 / float(n - 1))
    return c2 * s, c3 * s


def project_ball(c2: torch.Tensor, c3: torch.Tensor, rho: float):
    """Project stacked (c2,c3) onto ℓ2 ball of radius rho (per-layer or global).

    Uses global ‖·‖_2 over all layers' c2,c3.
    """
    with torch.no_grad():
        v = torch.cat([c2.reshape(-1), c3.reshape(-1)])
        nrm = float(v.norm().item())
        if nrm > rho and nrm > 0:
            scale = rho / nrm
            c2.mul_(scale)
            c3.mul_(scale)
