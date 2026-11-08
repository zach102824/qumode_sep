"""Stride-4 replay and mapping checks. python3 test_replay.py"""

from __future__ import annotations

import numpy as np

import sim

STRIDE4 = [[0, 4, 8], [1, 5, 9], [2, 6, 10], [3, 7, 11]]


def check(name, cond, detail=""):
    if not cond:
        raise AssertionError(f"FAIL {name} {detail}")
    print("ok", name)


def test_replay():
    check("manifest", sim.sha256_file(sim.MANIFEST) == sim.EXPECTED)
    sys = sim.parse_system(STRIDE4)
    sub = sim.parse_system([[0, 2, 4], [6, 8, 10], [1, 3, 5], [7, 9, 11]])
    check("stride rM differs from sublattice", abs(sys["rM"] - sub["rM"]) > 1e-6, f"{sys['rM']} {sub['rM']}")
    # M is the block-mean projection of inter pairs, not all of Hc.
    inter = np.array(sys["J"], copy=True)
    for cav in STRIDE4:
        for a in cav:
            for b in cav:
                inter[a, b] = 0.0
    proj = sim.block_mean_matrix(inter, STRIDE4)
    check("M equals block projection", np.max(np.abs(sys["M"] - proj)) < 1e-12)
    check("M is not all of Hc", np.max(np.abs(sys["M"] - sys["J"])) > 1e-3)
    stored = np.load(sim.ROOT / "source" / "control_stride4.npz")
    x = stored["x"]
    psi_s = stored["psi"]
    psi_l = sim.logical_state(sys, 6, x)
    psi_p = sim.physical_state(sys, 6, x)
    check("logical vs packed", abs(np.vdot(psi_l, psi_p)) >= 1 - 1e-12, str(abs(np.vdot(psi_l, psi_p))))
    check("source overlap", abs(np.vdot(psi_l, psi_s)) >= 1 - 1e-12, str(abs(np.vdot(psi_l, psi_s))))
    spec = sim.spectrum(sys)
    E, var = sim.energy_var(psi_l, sys)
    F = sim.fidelity(spec["gs"], psi_l)
    check("energy 1e-10", abs((E - spec["E0"]) - 0.2923991307066025) < 1e-10, str(E - spec["E0"]))
    check("fidelity 1e-10", abs(F - 0.8579519623570816) < 1e-10, str(F))
    check("variance recorded", abs(var - 1.24736172045138) < 1e-8, str(var))
    rev = sim.parse_system(STRIDE4, bit_reverse=True)
    psi_r = sim.logical_state(rev, 6, x)
    # The shell is symmetric in the three cavity bits, so listing them backwards is the same circuit.
    check("within-cavity reversal is a shell symmetry", abs(np.vdot(psi_r, psi_s)) >= 1 - 1e-12, str(abs(np.vdot(psi_r, psi_s))))
    psi_inv = sim.physical_state(sys, 6, x, invert_map=True)
    check("inverted basis map changes the state", abs(np.vdot(psi_inv, psi_s)) < 0.999, str(abs(np.vdot(psi_inv, psi_s))))
    # pi shift is a global phase
    y = x.copy()
    y[3] += np.pi
    psi_pi = sim.logical_state(sys, 6, y)
    phase = np.vdot(psi_l, psi_pi)
    err = np.max(np.abs(psi_l - psi_pi * np.exp(-1j * np.angle(phase))))
    check("pi shell shift", err < 1e-8, str(err))
    # conjugation
    y = x.copy()
    y[0] *= -1
    y[3:] *= -1
    psi_c = sim.logical_state(sys, 6, y)
    check("conjugate overlap", abs(np.vdot(psi_l, np.conj(psi_c))) > 1 - 1e-8)
    check("conjugate is not the same ray", abs(np.vdot(psi_l, psi_c)) < 0.999)
    # joint relabel
    rng = np.random.default_rng(1)
    perm = rng.permutation(12)
    new, new_of_old = sim.permute_joint(sys, perm)
    psi_new = np.empty_like(psi_l)
    psi_new[new_of_old] = psi_l
    e_old, _ = sim.energy_var(psi_l, sys)
    e_new, _ = sim.energy_var(psi_new, new)
    check("relabel energy", abs(e_old - e_new) < 1e-8, f"{e_old} {e_new}")
    gs_new = np.empty_like(spec["gs"])
    gs_new[new_of_old] = spec["gs"]
    check("relabel fidelity", abs(sim.fidelity(spec["gs"], psi_l) - sim.fidelity(gs_new, psi_new)) < 1e-8)
    # gradient
    rng = np.random.default_rng(2)
    xx = rng.normal(scale=0.2, size=9)
    _, _, g, _ = sim.logical_grad(sys, 2, xx)
    gn = np.zeros_like(xx)
    eps = 1e-6
    for i in range(xx.size):
        e = np.zeros_like(xx)
        e[i] = eps
        ep = sim.logical_grad(sys, 2, xx + e)[1]
        em = sim.logical_grad(sys, 2, xx - e)[1]
        gn[i] = (ep - em) / (2 * eps)
    check("gradient", np.max(np.abs(g - gn)) < 2e-5, str(np.max(np.abs(g - gn))))
    print("E0", spec["E0"], "gap", spec["full_gap"], "rM", sys["rM"], "rK", sys["rK"])


if __name__ == "__main__":
    test_replay()
    print("ALL TESTS PASSED")
