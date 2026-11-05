"""Ansatz unitarity / structure tests for Nov_3rd QAOA-line on p-spin."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

PKG = Path(__file__).resolve().parents[1]
ROOT = PKG.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(PKG))

from pspin import make_instance, hc_torch, gs_torch, apply_H  # noqa: E402
from circuit_snap import make_ansatz, forward_state, B_SHELL  # noqa: E402
from circuit_fourier import shell_phi_from_betas  # noqa: E402
from energy_util import energy, energy_from_state  # noqa: E402


def test_W_unitary():
    device = torch.device("cpu")
    for n in (6, 9, 12):
        sim = make_ansatz(n, 2, device)
        W = sim.Wmat.detach().cpu().to(torch.complex128)
        I = torch.eye(W.shape[0], dtype=torch.complex128)
        err = torch.linalg.norm(W.conj().T @ W - I).item()
        assert err < 1e-5, f"W not unitary n={n} err={err}"
    print("PASS W_unitary")


def test_layer_norm_preserving():
    """U_c^X and SNAP each preserve ‖ψ‖ on the Fock product grid."""
    device = torch.device("cpu")
    n, L = 6, 2
    sim = make_ansatz(n, L, device)
    inst = make_instance(n, 0.5, seed=0)
    Hc = hc_torch(inst, device)
    from circuit_snap import hc_on_grid
    from explore_OCT29.lib.packing import NU

    Cn = hc_on_grid(sim, Hc)
    Cn = Cn - Cn.mean()
    psi = sim.vacuum()
    sl = tuple(slice(0, NU) for _ in range(sim.C))
    n0 = float(torch.linalg.vector_norm(psi).cpu())

    # one U_c^X
    psi = sim._apply_all(psi, sim.Wmat)
    phase = torch.exp(-1j * (0.3 * Cn).to(sim.cdtype))
    psi_new = psi.clone()
    psi_new[sl] = psi[sl] * phase
    psi = psi_new
    psi = sim._apply_all(psi, sim.Wmat)
    n1 = float(torch.linalg.vector_norm(psi).cpu())
    assert abs(n1 - n0) < 1e-5, (n0, n1)

    # SNAP
    beta = torch.tensor([0.2], dtype=torch.float32)
    phi = shell_phi_from_betas(beta.expand(L))[0]
    th = phi[sim.popcount]
    psi = sim._apply_all(psi, sim._snap_theta(th))
    n2 = float(torch.linalg.vector_norm(psi).cpu())
    assert abs(n2 - n0) < 1e-5, (n0, n2)
    print("PASS layer_norm_preserving")


def test_full_circuit_norm():
    device = torch.device("cpu")
    for n, L in [(6, 4), (9, 2)]:
        sim = make_ansatz(n, L, device)
        inst = make_instance(n, 0.5, seed=0)
        Hc = hc_torch(inst, device)
        gamma = torch.full((L,), 0.3)
        beta = torch.full((L,), 0.2)
        phi = shell_phi_from_betas(beta)
        psi = forward_state(sim, Hc, gamma, phi)
        nrm = float(torch.linalg.vector_norm(psi).cpu())
        assert abs(nrm - 1.0) < 1e-5, nrm
    print("PASS full_circuit_norm")


def test_zero_angles_near_vacuum_map():
    """γ=β=0: Uc^X=SNAP=I (W²=I); final W maps vacuum → |+⟩^⊗n (equal amp)."""
    device = torch.device("cpu")
    n, L = 6, 3
    sim = make_ansatz(n, L, device)
    inst = make_instance(n, 0.5, seed=0)
    Hc = hc_torch(inst, device)
    gamma = torch.zeros(L)
    beta = torch.zeros(L)
    phi = shell_phi_from_betas(beta)
    psi = forward_state(sim, Hc, gamma, phi)
    # W|0⟩ = |+⟩^n → uniform |amp|² = 1/2^n
    probs = (psi.real.square() + psi.imag.square()).detach().cpu().numpy()
    assert abs(probs.sum() - 1.0) < 1e-5
    assert abs(probs.max() - probs.min()) < 1e-5, (probs.max(), probs.min())
    assert abs(float(probs[0]) - 1.0 / (1 << n)) < 1e-5
    # also independent of L
    sim2 = make_ansatz(n, 1, device)
    psi2 = forward_state(sim2, Hc, torch.zeros(1), shell_phi_from_betas(torch.zeros(1)))
    diff = float(torch.linalg.vector_norm(psi - psi2).cpu())
    assert diff < 1e-5, diff
    print(f"PASS zero_angles uniform p0={float(probs[0]):.8f} L-indep ok")


def test_phi_qaoa_line_nesting():
    """φ_k = -β k structure."""
    beta = torch.tensor([0.5, -0.3, 1.2], dtype=torch.float32)
    phi = shell_phi_from_betas(beta)
    assert phi.shape == (3, B_SHELL)
    for ell, b in enumerate(beta.tolist()):
        for k in range(B_SHELL):
            assert abs(phi[ell, k].item() + b * k) < 1e-6
    print("PASS phi_qaoa_line_nesting")


def test_shared_gamma_matches_free():
    """Expanding shared γ to all layers matches free QAOA with γ_ℓ=γ."""
    device = torch.device("cpu")
    n, L = 6, 4
    sim = make_ansatz(n, L, device)
    inst = make_instance(n, 0.5, seed=1)
    Hc = hc_torch(inst, device)
    gs = gs_torch(inst, device)
    g_shared = 0.17
    beta = torch.tensor([0.1, -0.05, 0.2, 0.08], dtype=torch.float32)
    gamma_free = torch.full((L,), g_shared)
    phi = shell_phi_from_betas(beta)
    psi1 = forward_state(sim, Hc, gamma_free, phi)
    psi2 = forward_state(sim, Hc, torch.full((L,), g_shared), phi)
    diff = float(torch.linalg.vector_norm(psi1 - psi2).cpu())
    assert diff < 1e-6, diff
    # energy/fid agree
    from circuit_snap import fidelity
    F1 = float(fidelity(sim, Hc, gs, gamma_free, phi))
    F2 = float(fidelity(sim, Hc, gs, torch.full((L,), g_shared), phi))
    assert abs(F1 - F2) < 1e-7
    print("PASS shared_gamma_matches_free")


def test_energy_matches_apply_H():
    device = torch.device("cpu")
    n, L = 6, 2
    sim = make_ansatz(n, L, device)
    inst = make_instance(n, 0.5, seed=0)
    Hc = hc_torch(inst, device)
    gamma = torch.full((L,), 0.01)
    beta = torch.full((L,), 0.01)
    phi = shell_phi_from_betas(beta)
    psi = forward_state(sim, Hc, gamma, phi)
    E_t = float(energy_from_state(psi, Hc, 0.5, n).cpu())
    psi_np = psi.detach().cpu().numpy().astype(np.complex128)
    Hv = apply_H(psi_np, inst.Hc_diag, 0.5, n)
    E_np = float(np.vdot(psi_np, Hv).real)
    assert abs(E_t - E_np) < 1e-4, (E_t, E_np)
    # GS Rayleigh
    gs = gs_torch(inst, device)
    E_gs = float(energy_from_state(gs, Hc, 0.5, n).cpu())
    assert abs(E_gs - inst.E0) < 1e-4, (E_gs, inst.E0)
    print(f"PASS energy_matches_apply_H E={E_t:.6f}")


def test_autodiff_grad_energy():
    """Finite-diff vs autodiff on ⟨H⟩ w.r.t. β,γ."""
    device = torch.device("cpu")
    n, L = 6, 2
    sim = make_ansatz(n, L, device)
    inst = make_instance(n, 0.5, seed=0)
    Hc = hc_torch(inst, device)
    gamma = torch.nn.Parameter(torch.full((L,), 0.01))
    beta = torch.nn.Parameter(torch.full((L,), 0.01))
    phi = shell_phi_from_betas(beta)
    E = energy(sim, Hc, 0.5, n, gamma, phi)
    E.backward()
    g_auto = gamma.grad.detach().clone()
    b_auto = beta.grad.detach().clone()

    eps = 1e-3
    # finite diff gamma[0]
    def E_at(g, b):
        with torch.no_grad():
            return float(energy(sim, Hc, 0.5, n, g, shell_phi_from_betas(b)).cpu())

    g_p = gamma.detach().clone(); g_p[0] += eps
    g_m = gamma.detach().clone(); g_m[0] -= eps
    fd_g0 = (E_at(g_p, beta.detach()) - E_at(g_m, beta.detach())) / (2 * eps)
    assert abs(fd_g0 - float(g_auto[0])) < 5e-2, (fd_g0, float(g_auto[0]))

    b_p = beta.detach().clone(); b_p[0] += eps
    b_m = beta.detach().clone(); b_m[0] -= eps
    fd_b0 = (E_at(gamma.detach(), b_p) - E_at(gamma.detach(), b_m)) / (2 * eps)
    assert abs(fd_b0 - float(b_auto[0])) < 5e-2, (fd_b0, float(b_auto[0]))
    print(f"PASS autodiff_grad_energy fd_g0={fd_g0:.5f} auto={float(g_auto[0]):.5f}")


def main():
    test_W_unitary()
    test_layer_norm_preserving()
    test_full_circuit_norm()
    test_zero_angles_near_vacuum_map()
    test_phi_qaoa_line_nesting()
    test_shared_gamma_matches_free()
    test_energy_matches_apply_H()
    test_autodiff_grad_energy()
    print("ALL ANSATZ TESTS PASSED")


if __name__ == "__main__":
    main()
