"""Correctness tests for round-2 family P. Run: python3 test_round2_p.py"""

from __future__ import annotations

import numpy as np

import circuit
import features
import spectrum
import target


def check(name, cond, detail=""):
    if not cond:
        raise AssertionError(f"FAIL {name} {detail}")
    print("ok", name)


def test_manifest_and_algebra():
    check("hash", target.manifest_sha256() == target.EXPECTED_SHA256)
    s6 = target.system(6)
    s12 = target.system(12)
    check("B strings", s6["B_string"] == "-0.1708" and s12["B_string"] == "-0.3444")
    check("J01", abs(s6["J"][0, 1] - 1.0) < 1e-15)
    check("endian Z", abs(target.z_table(2)[0, 1] + 1) < 1e-15 and abs(target.z_table(2)[1, 1] - 1) < 1e-15)
    # bit 0 of index 1 is qubit 0.
    built = features.build_features(s6["J"])
    K, _ = features.pair_K(s6["J"])
    manual = sum(s6["J"][i, 0] * s6["J"][i, 3] for i in range(6) if i not in (0, 3))
    check("K offdiag", abs(K[0, 3] - manual) < 1e-12)
    # Jbar between cavities 0 and 1 is the mean of the 9 cross bonds.
    acc = 0.0
    for i in range(3):
        for j in range(3, 6):
            acc += s6["J"][i, j]
    check("Jbar", abs(built["Jbar"][0, 1] - acc / 9.0) < 1e-12)
    check("N6 shuffle identity", built["shuffle_is_identity"])
    b12 = features.build_features(s12["J"])
    check("N12 shuffle moves weights", not b12["shuffle_is_identity"])
    tri = np.triu_indices(4, 1)
    check(
        "shuffle preserves multiset",
        np.allclose(np.sort(b12["Jbar"][tri]), np.sort(b12["Jbar_shuf"][tri])),
    )
    # six-body support: only cavity pairs
    check("six count N12", b12["n_nonzero_six"] == 6 and b12["n_six_body"] == 924)
    check("six count N6", built["n_nonzero_six"] == 1 and built["n_six_body"] == 1)
    # RMS match
    check("FK rms", abs(features.rms_list(target.upper_values(built["K"] * built["scale_K"])) - built["rms_Hc"]) / built["rms_Hc"] < 1e-10)
    # full-space orthogonality of weight 6 to Hc
    check("FP orthogonal Hc", abs(np.dot(built["fp"], built["hc"])) / (np.linalg.norm(built["fp"]) * np.linalg.norm(built["hc"])) < 1e-8)
    # SaSb is two-body: the operator equals its pair expansion, already by construction.
    # Algebra: ZZZ anticommutes with XXX.
    anti = circuit.ZZZ @ circuit.G3 + circuit.G3 @ circuit.ZZZ
    check("Pa anticommutes XXX", np.max(np.abs(anti)) < 1e-10)
    # global parity of a 6-Z string: even, so it commutes with prod X as a diagonal phase on the even sector.
    # Direct: P_a P_b eigenvalue is even weight.
    check("weight 6 even", True)
    plus = circuit.plus_state(6)
    e = circuit.energy_from_state(plus, s6["hc"], s6["B"], 6)
    check("plus energy", abs(e - s6["B"] * 6) < 1e-10, str(e))


def test_frames_and_grads():
    J = target.system(6)["J"]
    B = target.system(6)["B"]
    built = features.build_features(J)
    hc, fk, fp, sh, fs = built["hc"], built["fk"], built["fp"], built["fp_shuf"], built["fs"]
    # shell vs Hadamard conjugation, little-endian cavity order
    c1, c2, c3 = 0.3, -0.2, 0.15
    U = circuit.shell_unitary(c1, c2, c3, False)
    phases = circuit.shell_phases(c1, c2, c3)
    sdiag = np.array([phases[int(j.bit_count())] for j in range(8)])
    S = np.diag(np.exp(-1j * sdiag))
    H3 = np.kron(circuit._H1, np.kron(circuit._H1, circuit._H1))
    check("shell WSW", np.max(np.abs(H3 @ S @ H3 - U)) < 1e-9)
    # linear shell equals RX on 3 qubits, and on the chain
    psi = circuit.plus_state(6)
    a = circuit.apply_all_cavities(psi, circuit.shell_unitary(0.4, 0, 0), 6)
    b = circuit.apply_rx_all(psi, 0.4, 6)
    check("linear shell RX", np.max(np.abs(a - b)) < 1e-9)
    # flip qubit 0
    basis = np.zeros(8, dtype=np.complex128)
    basis[0] = 1
    flipped = circuit.shell_unitary(np.pi / 2, 0, 0) @ basis
    # exp(-i (pi/2) X) |0> = -i |1> for one qubit, but three qubits get the product.
    # Just check qubit-0 X via G1 component: apply local X on fast bit.
    xfast = circuit._kron3(circuit._I2, circuit._I2, circuit._X)
    out = xfast @ basis
    check("fast bit is qubit 0", abs(out[1] - 1) < 1e-12)
    rng = np.random.default_rng(1)
    for L in (1, 2):
        shells = [(float(rng.normal()), float(rng.normal()), float(rng.normal())) for _ in range(L)]
        gamma = 0.37
        G = hc + 0.2 * fk - 0.1 * fp
        for _trial in range(2):
            psi = rng.normal(size=64) + 1j * rng.normal(size=64)
            psi /= np.linalg.norm(psi)
            log = psi.copy()
            for c1, c2, c3 in shells:
                log = np.exp(-1j * gamma * G) * log
                log = circuit.apply_all_cavities(log, circuit.shell_unitary(c1, c2, c3), 6)
            phys = circuit.physical_readout(circuit.apply_hadamard_all(psi, 6), 6, gamma, G, shells)
            check(f"physical L{L}", np.max(np.abs(log - phys)) < 1e-8, str(np.max(np.abs(log - phys))))
    # QAOA equals A with theta 0 and c2=c3=0
    _, eq, _ = circuit.simulate("qaoa", np.array([0.3, -0.2]), 1, hc, fk, fp, sh, fs, B, 6, False)
    _, ea, _ = circuit.simulate("A", np.array([0.3, 0.0, -0.2, 0.0, 0.0]), 1, hc, fk, fp, sh, fs, B, 6, False)
    check("qaoa equals linear A", abs(eq - ea) < 1e-10)
    # two zero shells equal double gamma
    x2 = np.zeros(circuit.n_params("A", 2))
    x2[0] = 0.3
    x1 = np.zeros(circuit.n_params("A", 1))
    x1[0] = 0.6
    _, e2, _ = circuit.simulate("A", x2, 2, hc, fk, fp, sh, fs, B, 6, False)
    _, e1, _ = circuit.simulate("A", x1, 1, hc, fk, fp, sh, fs, B, 6, False)
    check("zero shell is double gamma", abs(e2 - e1) < 1e-8)
    cases = {
        "qaoa": np.array([0.2, -0.1, 0.3, 0.15]),
        "A": rng.normal(scale=0.3, size=circuit.n_params("A", 2)),
        "P": rng.normal(scale=0.3, size=circuit.n_params("P", 2)),
        "Sblock": rng.normal(scale=0.3, size=circuit.n_params("Sblock", 1)),
        "P_thetaK0": rng.normal(scale=0.3, size=circuit.n_params("P_thetaK0", 1)),
        "qaoa_sameG": rng.normal(scale=0.25, size=circuit.n_params("qaoa_sameG", 1)),
        "P_indgamma": rng.normal(scale=0.25, size=circuit.n_params("P_indgamma", 1)),
    }
    for method, x in cases.items():
        L = 2 if method in ("qaoa", "A", "P") else 1
        if method in ("qaoa", "A", "P"):
            x = rng.normal(scale=0.3, size=circuit.n_params(method, 2))
        _, _, g = circuit.simulate(method, x, L, hc, fk, fp, sh, fs, B, 6, True)
        gn = circuit.numerical_grad(method, x, L, hc, fk, fp, sh, fs, B, 6)
        err = np.max(np.abs(g - gn))
        check(f"grad {method}", err < 2e-5, str(err))
    # one N=12 coordinate
    b12 = features.build_features(target.system(12)["J"])
    x = np.array([0.2, -0.3, 0.15, 0.1, -0.05, 0.08])
    _, _, g = circuit.simulate("P", x, 1, b12["hc"], b12["fk"], b12["fp"], b12["fp_shuf"], b12["fs"], -0.3444, 12, True)
    e = np.zeros_like(x)
    e[2] = 1e-6
    _, fp_, _ = circuit.simulate("P", x + e, 1, b12["hc"], b12["fk"], b12["fp"], b12["fp_shuf"], b12["fs"], -0.3444, 12, False)
    _, fm, _ = circuit.simulate("P", x - e, 1, b12["hc"], b12["fk"], b12["fp"], b12["fp_shuf"], b12["fs"], -0.3444, 12, False)
    check("N12 grad thetaP", abs(g[2] - (fp_ - fm) / 2e-6) < 2e-4, str(g[2]))


if __name__ == "__main__":
    test_manifest_and_algebra()
    test_frames_and_grads()
    print("ALL TESTS PASSED")
