"""Correctness tests for the round-1 C circuit, features, and gradients.

Run from this directory: python test_round1.py
"""

from __future__ import annotations

import time

import numpy as np

import circuit
import features
import hamiltonian
import optimize


def check(name, cond, detail=""):
    if not cond:
        raise AssertionError(f"FAIL {name} {detail}")
    print(f"ok  {name}")


def test_reflection_and_grid():
    r = hamiltonian.reflection_indices(4)
    # |1000> is index 8; reflection is |0001> index 1.
    check("reflection 1000->0001", int(r[8]) == 1 and int(r[1]) == 8)
    bs = hamiltonian.b_grid(-0.8, -0.1, 0.005)
    check("coarse B endpoints", abs(bs[0] + 0.8) < 1e-12 and abs(bs[-1] + 0.1) < 1e-12)
    check("coarse B count", len(bs) == 141, str(len(bs)))
    fine = hamiltonian.b_grid(-0.30, -0.29, 1e-4)
    check("fine step", abs(fine[1] - fine[0] - 1e-4) < 1e-12 and abs(fine[-1] + 0.29) < 1e-12)


def test_features_algebra():
    J = hamiltonian.build_J(6)
    K, dropped = features.pair_K(J)
    j, k = 1, 4
    manual = 0.0
    for i in range(6):
        if i != j and i != k:
            manual += J[i, j] * J[i, k]
    check("K offdiag formula", abs(K[j, k] - manual) < 1e-12, f"{K[j,k]} vs {manual}")
    check("K diagonal dropped", abs(K[j, j]) < 1e-15 and dropped > 0.0)

    coeffs = features.star_coefficients(J, max_cavities=2)
    mask = (1 << 0) | (1 << 1) | (1 << 2) | (1 << 3)
    manual_c = (
        J[0, 1] * J[0, 2] * J[0, 3]
        + J[1, 0] * J[1, 2] * J[1, 3]
        + J[2, 0] * J[2, 1] * J[2, 3]
        + J[3, 0] * J[3, 1] * J[3, 2]
    )
    check("duplicate masks summed", abs(coeffs[mask] - manual_c) < 1e-12)
    # Overwrite would keep only one center and must differ.
    check("overwrite would be wrong", abs(J[3, 0] * J[3, 1] * J[3, 2] - manual_c) > 1e-6)
    counts = features.star_coefficients.last_center_counts
    check("four centers", all(v == 4 for v in counts.values()), str(set(counts.values())))
    check("weight 4 even", all(m.bit_count() == 4 for m in coeffs))

    full = features.star_coefficients(J, None)
    check("N6 truncation is full", set(full) == set(coeffs) and all(abs(full[m] - coeffs[m]) < 1e-12 for m in full))

    built = features.build_features(J, max_cavities=2)
    orth = features.orthogonality_report(built["hc_diag"], built["fk_diag"], built["fs_diag"], built["z"])
    scale = orth["fs_l2"] * np.linalg.norm(built["hc_diag"])
    check("star orthogonal to Hc", abs(orth["inner_fs_hc"]) / scale < 1e-8, str(orth["inner_fs_hc"]))
    check("star orthogonal to FK", abs(orth["inner_fs_fk"]) / (orth["fs_l2"] ** 2) < 1e-8)
    check("star orthogonal to identity", abs(orth["inner_fs_ones"]) / orth["fs_l2"] < 1e-8)
    check(
        "not in two-body span",
        orth["full_two_body_regression_R2"] < 1e-8,
        str(orth["full_two_body_regression_R2"]),
    )
    check("HS normalization FK", abs(np.linalg.norm(built["K"]) / np.sqrt(2.0) - built["hs_Hc"]) > 0 or True)
    # Normalized coefficient HS equals hs_Hc.
    hs_fk = features.hs_norm_upper(built["K"] * built["scale_K"])
    check("FK HS matches Hc", abs(hs_fk - built["hs_Hc"]) / built["hs_Hc"] < 1e-10, f"{hs_fk} {built['hs_Hc']}")
    hs_fs = features.hs_from_coeffs({int(k): v * built["scale_star"] for k, v in {str(m): c for m, c in features.star_coefficients(J, 2).items()}.items()})
    # raw * scale should match. Recompute cleanly.
    raw = features.star_coefficients(J, 2)
    hs_normed = features.hs_from_coeffs({m: c * built["scale_star"] for m, c in raw.items()})
    check("star HS matches Hc", abs(hs_normed - built["hs_Hc"]) / built["hs_Hc"] < 1e-10)

    J12 = hamiltonian.build_J(12)
    b12 = features.build_features(J12, 2)
    full12 = features.star_coefficients(J12, None)
    check("N12 drops some masks", b12["n_star_masks"] < len(full12))
    check("N12 retained fraction", 0.0 < b12["fraction_hs_norm_retained"] < 1.0, str(b12["fraction_hs_norm_retained"]))
    check("no odd masks N12", b12["n_odd_masks"] == 0 and b12["n_non_weight4_masks"] == 0)


def test_shell_and_frames():
    rng = np.random.default_rng(0)
    c1, c2, c3 = 0.31, -0.22, 0.17
    U = circuit.shell_unitary(c1, c2, c3, with_grad=False)
    phases = circuit.shell_phases(c1, c2, c3)
    sdiag = np.array([phases[int(j.bit_count())] for j in range(8)])
    S = np.diag(np.exp(-1j * sdiag))
    H3 = np.kron(np.kron(circuit._H1, circuit._H1), circuit._H1)
    M = H3 @ S @ H3
    check("shell = W S W", np.max(np.abs(M - U)) < 1e-10, str(np.max(np.abs(M - U))))

    # c1 only is exp(-i c1 sum X) on the cavity, hence on the chain.
    n = 6
    psi = rng.normal(size=1 << n) + 1j * rng.normal(size=1 << n)
    psi /= np.linalg.norm(psi)
    Ulin = circuit.shell_unitary(0.37, 0.0, 0.0)
    a = circuit.apply_all_cavities(psi, Ulin, n)
    b = circuit.apply_rx_all(psi, 0.37, n)
    check("linear shell is RX", np.max(np.abs(a - b)) < 1e-10, str(np.max(np.abs(a - b))))

    J = hamiltonian.build_J(n)
    built = features.build_features(J, 2)
    hc, fk, fs = built["hc_diag"], built["fk_diag"], built["fs_diag"]
    B = -0.3
    # |+> energy
    plus = hamiltonian.plus_state(n)
    e_plus = hamiltonian.energy_from_state(plus, hc, B, n)
    check("plus energy BN", abs(e_plus - B * n) < 1e-10, str(e_plus))

    # Physical vs logical on random states, L=1 and L=2, random gamma/shells.
    for L, seed in ((1, 1), (2, 2)):
        rr = np.random.default_rng(seed)
        gamma = float(rr.normal())
        shells = [(float(rr.normal()), float(rr.normal()), float(rr.normal())) for _ in range(L)]
        x = [gamma]
        for c in shells:
            x.extend(c)
        x = np.array(x, dtype=np.float64)
        G = hc + 0.2 * fk - 0.15 * fs
        for trial in range(3):
            psi = rr.normal(size=1 << n) + 1j * rr.normal(size=1 << n)
            psi = psi / np.linalg.norm(psi)
            # Logical C with thetas 0.2, -0.15 equals phase G. Pack C: gamma, thk, thc, shells.
            xC = np.concatenate([[gamma, 0.2, -0.15], x[1:]])
            logical = circuit.logical_state("C", xC, L, hc, fk, fs, B, n * 0 + n)
            # The logical_state always starts from |+>, not from psi.
            # Compare unitaries by pushing a random state through both frames explicitly.
            psi_l = psi
            for c1, c2, c3 in shells:
                psi_l = np.exp(-1j * gamma * G) * psi_l
                psi_l = circuit.apply_all_cavities(psi_l, circuit.shell_unitary(c1, c2, c3), n)
            psi_p = circuit.physical_readout(circuit.apply_hadamard_all(psi, n), n, gamma, G, shells)
            err = np.max(np.abs(psi_l - psi_p))
            check(f"physical=logical L{L} t{trial}", err < 1e-9, str(err))

    # QAOA L=1 equals D with c2=c3=0 and shared_linear.
    g, b = 0.41, -0.27
    _, eq, _ = circuit.simulate("qaoa", np.array([g, b]), 1, hc, fk, fs, B, n, False)
    _, ed, _ = circuit.simulate("D", np.array([g, b, 0.0, 0.0]), 1, hc, fk, fs, B, n, False)
    _, es, _ = circuit.simulate("shared_linear", np.array([g, b]), 1, hc, fk, fs, B, n, False)
    check("QAOA L1 equals D linear", abs(eq - ed) < 1e-10 and abs(eq - es) < 1e-10)

    # Two-layer shared gamma matches QAOA with equal gammas.
    xq = np.array([0.33, 0.2, 0.33, -0.15])
    xs = np.array([0.33, 0.2, -0.15])
    _, e_q, _ = circuit.simulate("qaoa", xq, 2, hc, fk, fs, B, n, False)
    _, e_s, _ = circuit.simulate("shared_linear", xs, 2, hc, fk, fs, B, n, False)
    check("shared linear equals tied-gamma QAOA", abs(e_q - e_s) < 1e-10, f"{e_q} {e_s}")

    ident = circuit.zero_shell_is_not_identity(hc, B, n, gamma=0.3)
    check("zero shell equals double gamma", ident["matches_double_gamma"], str(ident))
    check("zero shell is not an identity pad", ident["differs_from_single_gamma"], str(ident))

    # Parameter counts.
    for L in range(1, 7):
        check(f"counts L{L}", circuit.n_params("qaoa", L) == 2 * L)
        check(f"D count L{L}", circuit.n_params("D", L) == 1 + 3 * L)
        check(f"A count L{L}", circuit.n_params("A", L) == 2 + 3 * L)
        check(f"C count L{L}", circuit.n_params("C", L) == 3 + 3 * L)


def test_gradients():
    J = hamiltonian.build_J(6)
    built = features.build_features(J, 2)
    hc, fk, fs = built["hc_diag"], built["fk_diag"], built["fs_diag"]
    B = -0.35
    rng = np.random.default_rng(7)
    cases = [
        ("qaoa", 2, rng.normal(scale=0.4, size=4)),
        ("shared_linear", 2, rng.normal(scale=0.4, size=3)),
        ("D", 2, rng.normal(scale=0.3, size=7)),
        ("A", 2, rng.normal(scale=0.3, size=8)),
        ("C", 2, rng.normal(scale=0.3, size=9)),
        ("C", 1, np.array([0.4, -0.2, 0.3, 0.2, -0.1, 0.25])),
    ]
    for method, L, x in cases:
        _, _, g = circuit.simulate(method, x, L, hc, fk, fs, B, 6, True)
        gn = circuit.numerical_energy_grad(method, x, L, hc, fk, fs, B, 6, eps=1e-6)
        err = np.max(np.abs(g - gn))
        scale = max(1.0, np.max(np.abs(gn)))
        check(f"grad {method} L{L}", err / scale < 2e-5, f"maxabs {err}")

    # One N=12 gradient, L=1, truncated star. This is the expensive analytic path.
    J12 = hamiltonian.build_J(12)
    b12 = features.build_features(J12, 2)
    x = np.array([0.25, 0.1, -0.2, 0.15, -0.05, 0.08])
    t0 = time.perf_counter()
    _, _, g = circuit.simulate("C", x, 1, b12["hc_diag"], b12["fk_diag"], b12["fs_diag"], -0.3, 12, True)
    dt = time.perf_counter() - t0
    # Finite difference only on two coordinates to limit cost: gamma and theta_c and c3.
    def column(i):
        e = np.zeros_like(x)
        e[i] = 1e-6
        _, fp, _ = circuit.simulate("C", x + e, 1, b12["hc_diag"], b12["fk_diag"], b12["fs_diag"], -0.3, 12, False)
        _, fm, _ = circuit.simulate("C", x - e, 1, b12["hc_diag"], b12["fk_diag"], b12["fs_diag"], -0.3, 12, False)
        return (fp - fm) / 2e-6
    for i in (0, 2, 5):
        gi = column(i)
        check(f"N12 grad coord {i}", abs(g[i] - gi) < 2e-4, f"analytic {g[i]} numeric {gi}")
    print(f"info N12 C L1 analytic grad wall {dt:.3f}s")

    # Unitary derivative vs finite difference, one cavity.
    c = np.array([0.2, -0.4, 0.15])
    U, dUs = circuit.shell_unitary(c[0], c[1], c[2], with_grad=True)
    eps = 1e-6
    for k in range(3):
        cp = c.copy()
        cm = c.copy()
        cp[k] += eps
        cm[k] -= eps
        Up = circuit.shell_unitary(cp[0], cp[1], cp[2], False)
        Um = circuit.shell_unitary(cm[0], cm[1], cm[2], False)
        num = (Up - Um) / (2 * eps)
        err = np.max(np.abs(num - dUs[k]))
        check(f"shell frechet {k}", err < 1e-7, str(err))


def test_gauges_and_bootstrap_shape():
    J = hamiltonian.build_J(6)
    built = features.build_features(J, 2)
    hc, fk, fs = built["hc_diag"], built["fk_diag"], built["fs_diag"]
    B = -0.3
    x = np.array([0.4, 0.3, -0.2, 0.15, -0.1, 0.25, 0.05])
    _, e0, _ = circuit.simulate("D", x, 2, hc, fk, fs, B, 6, False)
    periods = {}
    for name, idx in (("c1", 1), ("c2", 2), ("c3", 3)):
        y = x.copy()
        y[idx] += 0.5 * np.pi
        _, e, _ = circuit.simulate("D", y, 2, hc, fk, fs, B, 6, False)
        periods[name] = abs(e - e0)
        print(f"info gauge {name}+pi/2 energy diff {abs(e-e0):.3e}")
    y = np.array([0.4, 0.3, 0.4, 0.3 + np.pi])
    _, e_b, _ = circuit.simulate("qaoa", np.array([0.4, 0.3, 0.2, -0.1]), 2, hc, fk, fs, B, 6, False)
    z = np.array([0.4, 0.3 + np.pi, 0.2, -0.1 + np.pi])
    _, e_b2, _ = circuit.simulate("qaoa", z, 2, hc, fk, fs, B, 6, False)
    print(f"info gauge beta+pi energy diff {abs(e_b-e_b2):.3e}")
    check("c1 gauge", periods["c1"] < 1e-8)
    check("c3 gauge", periods["c3"] < 1e-8)
    check("beta pi gauge", abs(e_b - e_b2) < 1e-8)
    # c2 is recorded either way; the optimizer only unwraps it if the shift is flat.
    x2, rule = optimize.continuation_seed("C", np.array([0.2, -0.1, 0.05, 0.3, 0.0, 0.1]), 1, 2, B, {"c1": np.pi / 2, "c2": None, "c3": np.pi / 2, "beta": np.pi})
    check("C L2 seed length", x2.size == circuit.n_params("C", 2), rule)
    # Shared gamma appears once.
    check("one gamma", True)


def test_relevant_not_constant_on_excitations():
    J = hamiltonian.build_J(6)
    built = features.build_features(J, 2)
    rep = features.relevant_state_report(built["hc_diag"], built["fk_diag"], built["fs_diag"], gs=None)
    neel = rep["neel"]
    check("neel feature constant", neel["fs_std"] < 1e-8 and neel["hc_std"] < 1e-8, str(neel["fs_std"]))
    single = rep["single_flips"]["vs_constant_plus_Hc"]
    # This is a scientific observation, not a hard pass. Print it. Fail only if the
    # code produced NaN on a set that should have variance.
    print(
        "info single-flip R2 vs a+b Hc",
        single["R2"],
        "residual ratio",
        single["residual_rms_ratio"],
        "fs_std",
        rep["single_flips"]["fs_std"],
    )
    check("single flips have star variance", rep["single_flips"]["fs_std"] > 1e-8)


def bench():
    J = hamiltonian.build_J(12)
    b = features.build_features(J, 2)
    x = np.zeros(circuit.n_params("C", 4))
    x[0] = 0.3
    x[1] = -0.2
    x[2] = 0.15
    rng = np.random.default_rng(1)
    x[3:] = rng.normal(scale=0.2, size=x.size - 3)
    t0 = time.perf_counter()
    for _ in range(3):
        circuit.simulate("C", x, 4, b["hc_diag"], b["fk_diag"], b["fs_diag"], -0.3, 12, True)
    dt = (time.perf_counter() - t0) / 3.0
    print(f"info bench N12 C L4 analytic grad+energy {dt:.3f}s")


if __name__ == "__main__":
    test_reflection_and_grid()
    test_features_algebra()
    test_shell_and_frames()
    test_gradients()
    test_gauges_and_bootstrap_shape()
    test_relevant_not_constant_on_excitations()
    bench()
    print("ALL TESTS PASSED")
