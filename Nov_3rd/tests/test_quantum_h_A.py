"""Arm A gates: sign of H, Z+X readout, locked SNAP forward, Fourier roundtrip."""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from Nov_3rd.ansatz import make_sim
from Nov_3rd.quantum_h_A import (
    PAPER_BETA,
    PAPER_GAMMA,
    SnapSimulator,
    a_of_n,
    analyze,
    angles_from_modes,
    energy_from_z_state,
    generate_instances,
    ground_pair,
    paper_modes,
    pauli_x_sum,
    synthesize,
    phi_from_shell,
    plus_energy,
    qaoa_energy,
    sum_x,
)


class T:
    def __init__(self):
        self.fails = []

    def check(self, name, ok, detail=""):
        print(("PASS " if ok else "FAIL ") + name + (f" {detail}" if detail and not ok else ""))
        if not ok:
            self.fails.append(name)


def test_sign_and_bitstring(t: T):
    n = 3
    xsum = pauli_x_sum(n)
    # One bit flip between 0 and the MSB neighbor carries −λ, not +λ.
    H = np.diag(np.zeros(1 << n)) - 0.4 * xsum
    t.check("stoquastic_offdiag", abs(H[0, 1 << (n - 1)] + 0.4) < 1e-12)
    E = np.array([0, 1, 1, 2, 1, 2, 2, 3], dtype=np.float64)
    e0, psi = ground_pair(E, xsum, 0.0)
    t.check("lambda0_energy", abs(e0 - 0.0) < 1e-9)
    t.check("lambda0_support", float(np.abs(psi[E == 0]).sum() ** 2) > 0.99 or abs(e0) < 1e-8)
    # At λ=0 the diagonal GS energy is the min unsat. Support can mix degeneracy.
    t.check("lambda0_in_kernel", float((np.abs(psi) ** 2)[E == 0].sum()) > 1.0 - 1e-6)
    e_pos, psi_pos = ground_pair(E, xsum, 0.8)
    t.check("field_lowers", e_pos < -1e-6)
    # ⟨X⟩ of the stoquastic GS is positive (field −λX wants X=+1).
    xn = float(np.real(np.vdot(psi_pos, xsum @ psi_pos)))
    t.check("gs_X_positive", xn > 0.0, f"Xn={xn}")


def test_readout_matches_matrix(t: T):
    torch.manual_seed(0)
    n = 3
    E = torch.tensor([[0, 1, 1, 2, 1, 2, 2, 4]], dtype=torch.float32)
    psi = torch.randn(1, 8, dtype=torch.complex64)
    psi = psi / psi.norm()
    lam = 0.7
    got = float(energy_from_z_state(psi, E, lam))
    xsum = pauli_x_sum(n)
    v = psi.detach().numpy().reshape(-1).astype(np.complex128)
    H = np.diag(E.numpy().reshape(-1).astype(np.float64)) - lam * xsum
    exact = float(np.real(np.vdot(v, H @ v)))
    t.check("z_x_hist_equals_matrix", abs(got - exact) < 1e-5, f"{got} vs {exact}")
    # |+> has ⟨X_i⟩=1 and uniform Z.
    plus = torch.full((1, 8), 1.0 / math.sqrt(8), dtype=torch.complex64)
    t.check("plus_X", abs(float(sum_x(plus)) - 3.0) < 1e-5)
    t.check("plus_energy", abs(float(plus_energy(E, lam, n)) - float(energy_from_z_state(plus, E, lam))) < 1e-5)


def test_fourier_roundtrip(t: T):
    L = 16
    A, B = paper_modes(L)
    betas, gammas = angles_from_modes(A.double(), B.double(), L)
    t.check("paper_beta", torch.allclose(betas, torch.full((L,), PAPER_BETA, dtype=betas.dtype), atol=1e-6))
    t.check("paper_gamma", torch.allclose(gammas, torch.full((L,), PAPER_GAMMA, dtype=gammas.dtype), atol=1e-6))
    sched = torch.tensor([0.2, -0.1, 0.05, 0.4], dtype=torch.float64)
    back = synthesize(analyze(sched, "dst"), 4, "dst")
    t.check("dst_inverse", torch.allclose(back, sched, atol=1e-8))


def test_snap_matches_locked_forward(t: T):
    n, L = 6, 2
    pack = generate_instances(n, 2)
    E = torch.tensor(pack["E"][0], dtype=torch.float32)
    gamma = torch.tensor([0.31, -0.17], dtype=torch.float32)
    phi = torch.tensor(
        [[0.0, -0.2, -0.5, -0.9], [0.05, 0.15, 0.4, 0.7]],
        dtype=torch.float32,
    )
    ref = make_sim(n, L, torch.device("cpu"))
    out = ref.forward_nov3rd(E, gamma, phi)
    snap = SnapSimulator(n, L, "cpu")
    psi = snap.forward_fock(E.unsqueeze(0), gamma, phi)
    got = (psi.real.square() + psi.imag.square())[0]
    ref_p = out["probs"]
    t.check("snap_probs", torch.allclose(got, ref_p, atol=1e-5), f"max {float((got-ref_p).abs().max())}")
    # Graded γ_w = γ on every degree matches the scalar pulse up to a global phase.
    gw = gamma.reshape(L, 1).expand(L, 4).contiguous()
    psi_g = snap.forward_fock(E.unsqueeze(0), gw, phi)
    # Global phase per shot: compare probabilities, which drop the phase.
    got_g = (psi_g.real.square() + psi_g.imag.square())[0]
    t.check("walsh_equals_scalar_probs", torch.allclose(got_g, ref_p, atol=1e-4), f"max {float((got_g-ref_p).abs().max())}")


def test_dual_energy(t: T):
    """Hamming-shell φ_k=−β k should track qubit QAOA on ⟨H⟩, not only on p_succ."""
    n, L, lam = 6, 2, 0.75
    pack = generate_instances(n, 3)
    E = torch.tensor(pack["E"], dtype=torch.float32)
    betas = torch.tensor([0.15, -0.22], dtype=torch.float32)
    gammas = torch.tensor([-0.18, 0.07], dtype=torch.float32)
    q = qaoa_energy(betas, gammas, E, lam, n)
    c1 = -betas
    c2 = torch.zeros(L)
    c3 = torch.zeros(L)
    phi = phi_from_shell(c1, c2, c3)
    snap = SnapSimulator(n, L, "cpu")
    s = snap.energy(E, gammas, phi, lam)
    gap = float((q - s).abs().max())
    t.check("dual_energy", gap < 1e-3, f"max |Δ|={gap} q={q.detach().tolist()} s={s.detach().tolist()}")


def test_power_p_scale(t: T):
    p = torch.tensor(0.0)
    t.check("a9", abs(float(a_of_n(9, p)) - 1.0) < 1e-6)
    p2 = torch.tensor(1.0)
    t.check("a6", abs(float(a_of_n(6, p2)) - (8.0 / 5.0)) < 1e-5)


def main():
    t = T()
    test_sign_and_bitstring(t)
    test_readout_matches_matrix(t)
    test_fourier_roundtrip(t)
    test_snap_matches_locked_forward(t)
    test_dual_energy(t)
    test_power_p_scale(t)
    print(f"\n{len(t.fails)} failed")
    return 1 if t.fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
