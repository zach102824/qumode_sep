"""Interpolation, shell period, conjugation, and Chebyshev gradient checks."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "round3_R"))
import circuit
import model

import schedules


def check(name, cond, detail=""):
    if not cond:
        raise AssertionError(f"FAIL {name} {detail}")
    print("ok", name)


def test_endpoints():
    src = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    out = schedules.resample_shells(src, 4, method="linear")
    # centers 0.125, 0.375, 0.625, 0.875 against source knots 0.25, 0.75
    expect = [0.0, 0.25, 0.75, 1.0]
    unwrapped = schedules.unwrap_period([0.0, np.pi + 0.2], np.pi)
    check("pi unwrap", np.allclose(unwrapped, [0.0, 0.2]), str(unwrapped))
    check("constant ends and linear interior", np.allclose(out[:, 0], expect), str(out[:, 0]))
    # A query outside a 3-point cubic must stay at the endpoint, not the polynomial.
    src3 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    out3 = schedules.resample_shells(src3, 6, method="cubic")
    check("cubic outside is endpoint", abs(out3[0, 0] - src3[0, 0]) < 1e-12, str(out3[0, 0]))
    check("cubic outside right endpoint", abs(out3[-1, 0] - src3[-1, 0]) < 1e-12)


def test_pi_and_conjugation():
    sys6 = model.system(6)
    trips, _ = model.packing("sublattice", 6)
    idx = model.precompute(trips, 6)
    fm, _, _ = model.block_mean_diag(sys6["J"], trips, sys6["z"])
    rng = np.random.default_rng(4)
    x = rng.normal(scale=0.3, size=circuit.n_params("M", 2))
    psi, E, _, _ = circuit.simulate("M", x, 2, sys6["hc"], sys6["fk"], fm, sys6["B"], 6, idx, False)
    for slot in (3, 4, 5):  # first layer c1, c2, c3
        y = x.copy()
        y[slot] += np.pi
        psi2, E2, _, _ = circuit.simulate("M", y, 2, sys6["hc"], sys6["fk"], fm, sys6["B"], 6, idx, False)
        phase = np.vdot(psi, psi2)
        aligned = psi2 * np.exp(-1j * np.angle(phase))
        err = np.max(np.abs(psi - aligned))
        check(f"pi shift slot {slot}", abs(E - E2) < 1e-8 and err < 1e-8, f"E {E-E2} state {err}")
    y = x.copy()
    y[0] = -x[0]
    shells = y[3:].reshape(2, 3)
    y[3:] = -shells.reshape(-1)
    psi3, E3, _, _ = circuit.simulate("M", y, 2, sys6["hc"], sys6["fk"], fm, sys6["B"], 6, idx, False)
    check("conjugation energy", abs(E - E3) < 1e-8)
    direct = abs(np.vdot(psi, psi3))
    conj = abs(np.vdot(psi, np.conj(psi3)))
    check("conjugation matches conj state", conj > 1 - 1e-8, str(conj))
    check("conjugation is not automatic state identity", direct < 0.999, str(direct))


def test_chebyshev_grad():
    sys6 = model.system(6)
    trips, _ = model.packing("sublattice", 6)
    idx = model.precompute(trips, 6)
    fm, _, _ = model.block_mean_diag(sys6["J"], trips, sys6["z"])
    L, D = 3, 1
    basis = schedules.chebyshev_basis(L, D)
    coeff = np.array([0.2, -0.1, 0.05, 0.3, 0.1, -0.2, 0.4, 0.2, 0.0])
    # layout gamma, thk, thm, then a[k, d]

    def full_from(c):
        a = c[3:].reshape(3, D + 1)
        shells = basis @ a.T
        return schedules.pack_full(c[0], c[1], c[2], shells)

    def energy(c):
        _, E, _, _ = circuit.simulate("M", full_from(c), L, sys6["hc"], sys6["fk"], fm, sys6["B"], 6, idx, False)
        return E

    def analytic(c):
        x = full_from(c)
        _, E, g, _ = circuit.simulate("M", x, L, sys6["hc"], sys6["fk"], fm, sys6["B"], 6, idx, True)
        out = np.zeros_like(c)
        out[:3] = g[:3]
        gs = g[3:].reshape(L, 3)
        for k in range(3):
            out[3 + k * (D + 1): 3 + (k + 1) * (D + 1)] = basis.T @ gs[:, k]
        return E, out

    _, g = analytic(coeff)
    eps = 1e-6
    gn = np.zeros_like(coeff)
    for i in range(coeff.size):
        e = np.zeros_like(coeff)
        e[i] = eps
        gn[i] = (energy(coeff + e) - energy(coeff - e)) / (2 * eps)
    check("chebyshev grad", np.max(np.abs(g - gn)) < 2e-5, str(np.max(np.abs(g - gn))))


if __name__ == "__main__":
    test_endpoints()
    test_pi_and_conjugation()
    test_chebyshev_grad()
    print("ALL TESTS PASSED")
