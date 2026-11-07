"""Independent checks of the family-D algebra. Failures abort the round."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.linalg import expm

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ansatz import (  # noqa: E402
    G1,
    G2,
    G3,
    Model,
    energy_and_grad,
    final_state,
    logical_evolve_from_physical_input,
    mixer_unitary,
    physical_evolve,
    qaoa_forward,
    shell_forward,
    shell_phi_levels,
)
from common import build_J, dense_H, half_chain_entropy_bits, plus_state  # noqa: E402


def check(name: str, cond: bool, detail: str = "") -> None:
    if not cond:
        raise AssertionError(f"FAIL {name}: {detail}")
    print(f"PASS {name}")


def fd_grad(fun, x, eps=1e-7):
    g = np.zeros_like(x, dtype=np.float64)
    for i in range(x.size):
        step = np.zeros_like(x)
        step[i] = eps
        g[i] = (fun(x + step) - fun(x - step)) / (2 * eps)
    return g


def main() -> dict:
    report = {}
    # --- Pauli shell identity on one cavity ---
    I2 = np.eye(2, dtype=np.complex128)
    H1 = np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2.0)
    W3 = np.kron(np.kron(H1, H1), H1)
    Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)
    ZZ_pairs = []
    # eigenvalue of sum Z and sum ZZ and ZZZ on computational basis, local index slow-to-fast
    for k in range(8):
        bits = [(k >> (2 - b)) & 1 for b in range(3)]
        zs = [1 - 2 * b for b in bits]
        n1 = sum(bits)
        s2_formula = ((3 - 2 * n1) ** 2 - 3) / 2.0
        s2_direct = zs[0] * zs[1] + zs[0] * zs[2] + zs[1] * zs[2]
        check(f"s2 matches pairs n={n1} state={k}", abs(s2_formula - s2_direct) < 1e-12, f"{s2_formula} vs {s2_direct}")
        zzz = zs[0] * zs[1] * zs[2]
        check(f"ZZZ = (-1)^n state={k}", zzz == (-1) ** n1)

    rng = np.random.default_rng(7)
    for trial in range(6):
        c = rng.normal(size=3)
        U = mixer_unitary(*c)
        phi = shell_phi_levels(*c)
        # Diagonal SNAP in the Z basis, then conjugate by W.
        S = np.zeros((8, 8), dtype=np.complex128)
        for basis in range(8):
            bits = [(basis >> (2 - b)) & 1 for b in range(3)]
            S[basis, basis] = np.exp(-1j * phi[sum(bits)])
        M_from_snap = W3 @ S @ W3
        check(f"M = W S W trial {trial}", np.allclose(U, M_from_snap, atol=1e-10))
        # Commuting factorization.
        U_fact = expm(-1j * c[0] * G1) @ expm(-1j * c[1] * G2) @ expm(-1j * c[2] * G3)
        check(f"commuting exp factorization trial {trial}", np.allclose(U, U_fact, atol=1e-10))

    comm = [np.linalg.norm(G1 @ G2 - G2 @ G1), np.linalg.norm(G1 @ G3 - G3 @ G1), np.linalg.norm(G2 @ G3 - G3 @ G2)]
    check("G1,G2,G3 commute", max(comm) < 1e-12, str(comm))
    report["generator_commutator_norms"] = comm

    # c2 -> c2+pi/2 is a global phase on one cavity (eigenvalues 3 and -1).
    U_c2 = mixer_unitary(0.2, 0.3 + 0.5 * np.pi, -0.1)
    U_c2_0 = mixer_unitary(0.2, 0.3, -0.1)
    ratio = U_c2 @ U_c2_0.conj().T
    check("c2 pi/2 shift is global phase", np.allclose(ratio, ratio[0, 0] * np.eye(8), atol=1e-10))

    # --- N=6 model vs a fixed field (not the entropy-selected field) ---
    n = 6
    J = build_J(n)
    B = -0.45
    model = Model(n, J, B)
    check("NN coupling is J0", abs(J[0, 1] - 1.0) < 1e-12)
    check("open chain, not periodic", abs(J[0, n - 1] - (n - 1) ** (-1.1)) < 1e-12)
    H = dense_H(model.hc, B, n)
    off = H - np.diag(np.diag(H))
    check("stoquastic off-diagonals <= 0", float(off.max()) <= 1e-12)
    psi = plus_state(n)
    e_plus = model.energy(psi)
    check("energy of |+> is B*N", abs(e_plus - B * n) < 1e-10, f"{e_plus} vs {B*n}")
    check("apply_H matches dense", np.allclose(model.apply_H(psi), H @ psi, atol=1e-10))
    # Random state energy matches quadratic form.
    v = rng.normal(size=model.dim) + 1j * rng.normal(size=model.dim)
    v = v / np.linalg.norm(v)
    e1 = model.energy(v)
    e2 = float(np.real(np.vdot(v, H @ v)))
    check("local energy matches dense quadratic form", abs(e1 - e2) < 1e-9, f"{e1} vs {e2}")

    # c2=c3=0 shell equals QAOA with a repeated gamma.
    g = 0.37
    betas = np.array([0.21, -0.11, 0.55])
    shell = np.column_stack([betas, np.zeros(3), np.zeros(3)])
    a = shell_forward(model, g, shell)
    b = qaoa_forward(model, np.full(3, g), betas)
    check("c2=c3=0 is shared-gamma QAOA", abs(np.vdot(a, b)) > 1 - 1e-10)

    # Zero shell does not identity-pad a shared gamma.
    s1 = shell_forward(model, 0.4, np.zeros((1, 3)))
    s2 = shell_forward(model, 0.4, np.zeros((2, 3)))
    s1_double = shell_forward(model, 0.8, np.zeros((1, 3)))
    check("zero shell L=2 is not L=1 at the same gamma", abs(np.vdot(s1, s2)) < 0.999)
    check("zero shell L=2 merges to a single cost of angle 2 gamma", abs(np.vdot(s2, s1_double)) > 1 - 1e-10)

    # Gamma is not pi-periodic.
    b0 = np.array([0.4])
    e_g = model.energy(qaoa_forward(model, np.array([0.35]), b0))
    e_gpi = model.energy(qaoa_forward(model, np.array([0.35 + np.pi]), b0))
    check("weighted gamma is not pi-periodic", abs(e_g - e_gpi) > 1e-3, f"delta={e_g-e_gpi}")
    report["gamma_pi_shift_energy_delta"] = float(e_g - e_gpi)

    # beta period pi/2 on the reachable ray.
    g3 = np.array([0.22, -0.15, 0.4])
    b3 = np.array([0.18, 0.33, -0.27])
    st = qaoa_forward(model, g3, b3)
    st2 = qaoa_forward(model, g3, b3 + 0.5 * np.pi)
    check("beta + pi/2 preserves the QAOA ray", abs(np.vdot(st, st2)) > 1 - 1e-10)
    shell_a = np.array([[0.2, -0.15, 0.1], [0.05, 0.2, -0.3], [-0.1, 0.0, 0.25]])
    st_s = shell_forward(model, 0.31, shell_a)
    for col, label in ((0, "c1"), (1, "c2"), (2, "c3")):
        shifted = shell_a.copy()
        shifted[1, col] += 0.5 * np.pi
        st_sh = shell_forward(model, 0.31, shifted)
        ov = abs(np.vdot(st_s, st_sh))
        check(f"{label} + pi/2 on one layer preserves the ray", ov > 1 - 1e-10, f"overlap={ov}")

    # Time reversal conjugates the state.
    st_tr = qaoa_forward(model, -g3, -b3)
    check("time reversal conjugates QAOA state", np.allclose(st_tr, st.conj(), atol=1e-10))
    sh_tr = shell_forward(model, -0.31, -shell_a)
    check("time reversal conjugates shell state", np.allclose(sh_tr, st_s.conj(), atol=1e-10))

    # Full physical vs logical unitary at N=6 for L=1,2,3.
    basis_overlaps = []
    for L in (1, 2, 3):
        gamma = float(rng.normal())
        shell = rng.normal(scale=0.4, size=(L, 3))
        worst = 1.0
        for basis in range(model.dim):
            psi0 = np.zeros(model.dim, dtype=np.complex128)
            psi0[basis] = 1.0
            phys = physical_evolve(model, gamma, shell, psi0)
            logi = logical_evolve_from_physical_input(model, gamma, shell, psi0)
            worst = min(worst, abs(np.vdot(phys, logi)))
        basis_overlaps.append(worst)
        check(f"physical vs logical full basis L={L}", worst > 1 - 1e-9, f"min overlap {worst}")
    report["physical_logical_min_overlap"] = basis_overlaps

    # Vacuum physical circuit matches logical |+> preparation.
    gamma = 0.27
    shell = np.array([[0.15, -0.2, 0.05], [0.4, 0.1, -0.3]])
    vac = np.zeros(model.dim, dtype=np.complex128)
    vac[0] = 1.0
    phys_vac = physical_evolve(model, gamma, shell, vac)
    log_vac = shell_forward(model, gamma, shell)
    check("vacuum physical readout equals logical |+> circuit", abs(np.vdot(phys_vac, log_vac)) > 1 - 1e-10)

    # Analytic gradients vs central differences, all methods, and physical FD.
    methods = ("qaoa", "shared_beta", "nov3_D", "drop_c2", "drop_c3")
    grad_err = {}
    for method in methods:
        L = 2
        from ansatz import n_params

        x = rng.normal(scale=0.25, size=n_params(method, L))

        def fun(z, method=method, L=L):
            return energy_and_grad(model, method, L, z)[0]

        E, g = energy_and_grad(model, method, L, x)
        g_fd = fd_grad(fun, x)
        err = float(np.max(np.abs(g - g_fd)))
        grad_err[method] = err
        check(f"analytic grad vs FD {method}", err < 5e-7, f"max abs {err}")
        # Forward-only state energy matches grad energy.
        e_state = model.energy(final_state(model, method, L, x))
        check(f"state energy matches grad energy {method}", abs(e_state - E) < 1e-10, f"{e_state} vs {E}")

    # Physical objective gradient vs logical analytic gradient.
    L = 2
    x = rng.normal(scale=0.3, size=1 + 3 * L)

    def phys_energy(z):
        gamma = float(z[0])
        shell = z[1:].reshape(L, 3)
        psi = physical_evolve(model, gamma, shell, vac)
        return model.energy(psi)

    _, g_log = energy_and_grad(model, "nov3_D", L, x)
    g_phys = fd_grad(phys_energy, x, eps=1e-7)
    err = float(np.max(np.abs(g_log - g_phys)))
    report["physical_vs_logical_grad_max_abs"] = err
    report["logical_grad_fd_max_abs"] = grad_err
    check("physical FD grad matches logical analytic grad", err < 5e-6, f"max abs {err}")

    # Norm preservation.
    st = final_state(model, "nov3_D", 3, rng.normal(scale=0.5, size=1 + 9))
    check("shell circuit preserves norm", abs(np.linalg.norm(st) - 1) < 1e-10)

    # Half-chain entropy of |+> is 0, and a Bell-like cut is positive. Smoke only.
    ent_plus = half_chain_entropy_bits(plus_state(6), 6)
    check("plus entropy is 0", abs(ent_plus) < 1e-10, str(ent_plus))

    # Z^2 = I is already inside s2. Record a counterexample note to the false
    # freedom "zero shell identity-pads".
    report["zero_shell_identity_pad"] = False
    report["zero_shell_merges_adjacent_costs"] = True
    print("ALL ALGEBRA CHECKS PASSED")
    return report


if __name__ == "__main__":
    main()
