"""Fourier / QAOA-dual constraint on Nov_3rd Hamming-shell SNAP.

Same cavity stack as circuit_snap, but φ_k = -β · k (k=0..3) so only
(β, γ) per layer — 2L free params vs SNAP's 5L (γ + free φ_0..3).
"""
from __future__ import annotations

import torch

from circuit_snap import B_SHELL, fidelity, forward_state, interpolate_angles, make_ansatz

__all__ = [
    "B_SHELL",
    "make_ansatz",
    "forward_state",
    "fidelity",
    "interpolate_angles",
    "shell_phi_from_betas",
    "init_fourier_angles",
    "fidelity_fourier",
    "n_params",
]


def shell_phi_from_betas(betas: torch.Tensor) -> torch.Tensor:
    """φ_k = -β · k for k=0..3 → shape (L, 4)."""
    L = int(betas.numel())
    pcs = torch.arange(B_SHELL, dtype=betas.dtype, device=betas.device)
    return (-betas.reshape(L, 1)) * pcs.reshape(1, -1)


def init_fourier_angles(L: int, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """Paper-ish warm: β=0.01, γ=0.01 (+ tiny seed jitter)."""
    torch.manual_seed(seed)
    beta = torch.full((L,), 0.01, dtype=torch.float32) + 0.001 * torch.randn(L)
    gamma = torch.full((L,), 0.01, dtype=torch.float32) + 0.001 * torch.randn(L)
    return beta, gamma


def fidelity_fourier(sim, Hc_diag, gs, gamma, beta) -> torch.Tensor:
    phi = shell_phi_from_betas(beta)
    return fidelity(sim, Hc_diag, gs, gamma, phi)


def n_params(method: str, L: int) -> int:
    if method == "snap":
        return 5 * int(L)  # γ + φ_0..3
    if method in ("fourier", "qaoa"):
        return 2 * int(L)  # β, γ
    raise ValueError(method)
