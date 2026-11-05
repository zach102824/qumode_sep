"""Nov_3rd cavity Hamming-shell SNAP motif for quantum p-spin (read-only reuse).

Motif (locked Nov_3rd):
  |0⟩ → [U_c^X → shell SNAP]^L → final W → Fock
  U_c^X = W D W,  D = exp(-i γ H_c) on Fock/product grid
  SNAP: φ_k for k=0..3 via popcount (4 params/cavity/layer; shared cavities)

Here H_c is the Z-diagonal p-spin cost (raw eigenvalues; no SAT n/m norm).
Fidelity is |⟨GS|ψ⟩|^2 with ψ mapped to Z bitstring order (var0=MSB).
"""
from __future__ import annotations

import torch

from Nov_3rd.ansatz import Nov3rdAnsatz, make_sim
from Nov_3rd import settings as S
from explore_OCT29.lib.packing import NU


B_SHELL = S.B_BITS + 1  # 4


def make_ansatz(n: int, L: int, device) -> Nov3rdAnsatz:
    return make_sim(n, L, device)


def hc_on_grid(sim: Nov3rdAnsatz, Hc_diag: torch.Tensor) -> torch.Tensor:
    """H_c eigenvalues on Fock product grid (NU,)*C."""
    Hc = Hc_diag.to(device=sim.lam.device, dtype=sim.rdtype)
    return Hc[sim.grid_to_x]


def forward_state(
    sim: Nov3rdAnsatz,
    Hc_diag: torch.Tensor,
    gamma: torch.Tensor,
    phi: torch.Tensor,
) -> torch.Tensor:
    """Return Z-basis statevector (2**n,), complex, aligned with var0=MSB index.

    gamma: (L,), phi: (L, 4) shell phases.
    """
    L = sim.L
    gamma = gamma.to(dtype=sim.rdtype, device=sim.lam.device).reshape(L)
    phi = phi.to(dtype=sim.rdtype, device=sim.lam.device)
    if tuple(phi.shape) != (L, B_SHELL):
        raise ValueError(f"phi shape {tuple(phi.shape)} != {(L, B_SHELL)}")

    Cn = hc_on_grid(sim, Hc_diag)  # raw H_c on grid; no SAT rescaling
    # optional shift for numerical phase gauge (doesn't change physics up to global)
    Cn = Cn - Cn.mean()

    psi = sim.vacuum()
    sl = tuple(slice(0, NU) for _ in range(sim.C))

    for ell in range(L):
        # U_c^X = W D W
        psi = sim._apply_all(psi, sim.Wmat)
        phase = torch.exp(-1j * (gamma[ell] * Cn).to(sim.cdtype))
        psi_new = psi.clone()
        psi_new[sl] = psi[sl] * phase
        psi = psi_new
        psi = sim._apply_all(psi, sim.Wmat)
        # bare Hamming-shell SNAP (shared across cavities)
        th = phi[ell][sim.popcount]
        psi = sim._apply_all(psi, sim._snap_theta(th))

    # final W → Fock (= Z bitstring packing)
    psi = sim._apply_all(psi, sim.Wmat)

    # map Fock product → bitstring amplitudes
    n = sim.cfg.n
    dim = 1 << n
    amps_grid = psi[sl].reshape(-1)
    x_flat = sim.grid_to_x.reshape(-1)
    out = torch.zeros(dim, dtype=sim.cdtype, device=sim.lam.device)
    out[x_flat] = amps_grid
    # renormalize computational subspace (nf==NU so should already be ~1)
    nrm = torch.linalg.vector_norm(out)
    out = out / nrm.clamp(min=1e-12)
    return out


def fidelity(
    sim: Nov3rdAnsatz,
    Hc_diag: torch.Tensor,
    gs: torch.Tensor,
    gamma: torch.Tensor,
    phi: torch.Tensor,
) -> torch.Tensor:
    """|⟨GS|ψ⟩|^2 (scalar tensor, differentiable w.r.t. gamma/phi)."""
    psi = forward_state(sim, Hc_diag, gamma, phi)
    ov = torch.dot(gs.conj().to(psi.dtype), psi)
    return (ov.real.square() + ov.imag.square())


def init_angles(L: int, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """Small QAOA-like warm: γ=0.01, φ_k=-0.01·k."""
    g = torch.full((L,), 0.01, dtype=torch.float32)
    pcs = torch.arange(B_SHELL, dtype=torch.float32)
    phi = (-0.01 * pcs).unsqueeze(0).expand(L, -1).contiguous().clone()
    # tiny seed jitter so seeds differ
    torch.manual_seed(seed)
    g = g + 0.001 * torch.randn(L)
    phi = phi + 0.001 * torch.randn(L, B_SHELL)
    return g, phi


def interpolate_angles(
    gamma6: torch.Tensor,
    phi6: torch.Tensor,
    gamma9: torch.Tensor,
    phi9: torch.Tensor,
    n: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Linear interp of angles in 1/n between n=6 and n=9 → target n."""
    x6, x9, xn = 1.0 / 6.0, 1.0 / 9.0, 1.0 / float(n)
    t = (xn - x6) / (x9 - x6)  # 0 at n=6, 1 at n=9
    g = (1 - t) * gamma6 + t * gamma9
    p = (1 - t) * phi6 + t * phi9
    return g, p
