"""Unit tests for the paper-faithful random k-SAT QAOA simulator."""

from __future__ import annotations

import numpy as np
import pytest

from qaoa_ksat.angles import paper_angles
from qaoa_ksat.circuit import apply_mixer, plus_state, qaoa_state, success_probability
from qaoa_ksat.ksat import (
    Clause,
    energy_of_index,
    energy_table,
    n_satisfying,
    random_ksat,
)


def test_energy_table_matches_bruteforce():
    rng = np.random.default_rng(0)
    inst = random_ksat(6, 3, r=4.0, rng=rng, seed=0)
    energy = energy_table(inst)
    for idx in range(1 << inst.n):
        assert int(energy[idx]) == energy_of_index(inst, idx)


def test_tautology_clause_never_unsatisfied():
    # (x0 ∨ ¬x0) is always true.
    clause = Clause(variables=np.array([0, 0], dtype=np.int32), signs=np.array([1, 0], dtype=np.int8))
    assert clause.unsat_constraints() is None
    from qaoa_ksat.ksat import KSATInstance

    inst = KSATInstance(n=3, k=2, r=1.0, clauses=(clause,))
    energy = energy_table(inst)
    assert np.all(energy == 0)


def test_single_positive_unit_clause():
    clause = Clause(variables=np.array([0], dtype=np.int32), signs=np.array([1], dtype=np.int8))
    from qaoa_ksat.ksat import KSATInstance

    inst = KSATInstance(n=3, k=1, r=1.0, clauses=(clause,))
    energy = energy_table(inst)
    # Unsat iff x0 = 0: even indices.
    for idx in range(8):
        assert int(energy[idx]) == (0 if (idx & 1) else 1)


def test_gamma_zero_is_uniform_over_sat():
    rng = np.random.default_rng(1)
    inst = random_ksat(8, 8, rng=rng)
    energy = energy_table(inst)
    psi = qaoa_state([0.3], [0.0], energy, n=8)
    psucc = success_probability(psi, energy)
    assert psucc == pytest.approx(n_satisfying(energy) / 256.0, rel=0, abs=1e-12)


def test_beta_zero_does_not_mix():
    rng = np.random.default_rng(2)
    inst = random_ksat(6, 3, r=4.0, rng=rng)
    energy = energy_table(inst)
    psi = qaoa_state([0.0], [1.2], energy, n=6)
    # Cost is diagonal, so Born probabilities stay uniform.
    probs = np.abs(psi) ** 2
    assert probs == pytest.approx(np.full(64, 1 / 64), abs=1e-12)


def test_mixer_pi_is_global_bitflip():
    n = 4
    rng = np.random.default_rng(3)
    psi0 = rng.normal(size=1 << n) + 1j * rng.normal(size=1 << n)
    psi0 = psi0 / np.linalg.norm(psi0)
    psi1 = apply_mixer(psi0, np.pi, n)
    # Rx(π) = -i X, so all bits flip and a global (-i)^n phase appears.
    flipped = np.empty_like(psi0)
    mask = (1 << n) - 1
    for idx in range(1 << n):
        flipped[idx ^ mask] = psi0[idx]
    phase = (-1j) ** n
    assert psi1 == pytest.approx(phase * flipped, abs=1e-10)


def test_plus_state_normalized():
    psi = plus_state(5)
    assert psi.shape == (32,)
    assert abs(np.linalg.norm(psi) - 1.0) < 1e-12


def test_random_assignment_mean_matches_poisson_formula():
    # Unconditioned on SAT: E[p_succ] = exp(-r n / 2^k) for uniform measurement.
    n, k, r = 8, 8, 176.54
    rng = np.random.default_rng(4)
    ps = []
    for _ in range(80):
        inst = random_ksat(n, k, r, rng=rng)
        energy = energy_table(inst)
        ps.append(n_satisfying(energy) / (1 << n))
    expected = float(np.exp(-r * n / (2**k)))
    assert float(np.mean(ps)) == pytest.approx(expected, rel=0.25)


def test_paper_angles_8sat_p1_and_p14():
    b, g = paper_angles(8, 1)
    assert b.shape == (1,) and g.shape == (1,)
    assert b[0] == pytest.approx(0.5433996420760803)
    b14, g14 = paper_angles(8, 14)
    assert b14.shape == (14,) and g14.shape == (14,)


def test_mixer_matches_kronecker_rx():
    i2 = np.eye(2)
    x = np.array([[0.0, 1.0], [1.0, 0.0]])

    def rx(b: float) -> np.ndarray:
        return np.cos(b / 2) * i2 - 1j * np.sin(b / 2) * x

    rng = np.random.default_rng(0)
    for n in range(1, 6):
        u = rx(0.7)
        for _ in range(n - 1):
            u = np.kron(u, rx(0.7))
        psi = rng.normal(size=1 << n) + 1j * rng.normal(size=1 << n)
        psi = psi / np.linalg.norm(psi)
        got = apply_mixer(psi, 0.7, n)
        assert got == pytest.approx(u @ psi, abs=1e-10)


def test_p1_beats_random_on_sat_instances():
    rng = np.random.default_rng(5)
    betas, gammas = paper_angles(8, 1)
    n_sat = 0
    n_beat = 0
    for _ in range(40):
        inst = random_ksat(8, 8, rng=rng)
        energy = energy_table(inst)
        if int(energy.min()) != 0:
            continue
        n_sat += 1
        psi = qaoa_state(betas, gammas, energy, n=8)
        psucc = success_probability(psi, energy)
        rand = n_satisfying(energy) / 256.0
        if psucc > rand:
            n_beat += 1
    assert n_sat >= 5
    assert n_beat == n_sat
