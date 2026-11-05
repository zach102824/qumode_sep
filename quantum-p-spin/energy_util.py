"""Energy ⟨ψ|H|ψ⟩ for Nov_3rd QAOA-line ansatz on quantum p-spin (differentiable)."""
from __future__ import annotations

import torch

from circuit_snap import forward_state


def energy_from_state(
    psi: torch.Tensor,
    Hc_diag: torch.Tensor,
    Gamma: float,
    n: int,
) -> torch.Tensor:
    """⟨ψ|H|ψ⟩ with H = diag(Hc) - Γ ∑_i X_i. psi: (2**n,) complex."""
    psi = psi.to(dtype=torch.complex64)
    Hc = Hc_diag.to(device=psi.device, dtype=torch.float32)
    # ⟨Hc⟩
    p = (psi.real.square() + psi.imag.square())
    e_c = torch.dot(p, Hc)
    # ⟨Hx⟩ = -Γ ∑_i ⟨X_i⟩; X_i flips bit i (var0=MSB)
    e_x = torch.zeros((), dtype=torch.float32, device=psi.device)
    dim = psi.shape[0]
    idx = torch.arange(dim, device=psi.device)
    for i in range(n):
        shift = n - 1 - i
        partner = idx ^ (1 << shift)
        # ⟨ψ|X|ψ⟩ = 2 Re ∑_x ψ*_x ψ_{x⊕i}  (each pair counted once via full sum)
        ov = torch.dot(psi.conj(), psi[partner])
        e_x = e_x - float(Gamma) * ov.real
    return e_c + e_x


def energy(
    sim,
    Hc_diag: torch.Tensor,
    Gamma: float,
    n: int,
    gamma: torch.Tensor,
    phi: torch.Tensor,
) -> torch.Tensor:
    """⟨ψ(γ,φ)|H|ψ⟩ differentiable w.r.t. gamma/phi."""
    psi = forward_state(sim, Hc_diag, gamma, phi)
    return energy_from_state(psi, Hc_diag, Gamma, n)


def fidelity_and_energy(
    sim,
    Hc_diag: torch.Tensor,
    gs: torch.Tensor,
    Gamma: float,
    n: int,
    gamma: torch.Tensor,
    phi: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    psi = forward_state(sim, Hc_diag, gamma, phi)
    ov = torch.dot(gs.conj().to(psi.dtype), psi)
    F = ov.real.square() + ov.imag.square()
    E = energy_from_state(psi, Hc_diag, Gamma, n)
    return F, E
