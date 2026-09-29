"""Tests for the 8-qubit hardware-efficient ansatz (noiseless/hea.py)."""

from __future__ import annotations

import numpy as np
import pytest

from noiseless.hea import (
    CZ_PAIRS_1IDX,
    CZ_SIGNS,
    HEASimulator,
    hea_layers_for_params,
    hea_state,
    lattice_edges_1idx,
    n_hea_parameters,
    optimize_hea_trial,
    random_hea_params,
)
from noiseless.qaoa import index_from_bitstring


@pytest.mark.parametrize("L,n", [(1, 16), (2, 24), (3, 32)])
def test_param_count(L, n):
    assert n_hea_parameters(L) == 8 * (L + 1) == n
    assert hea_layers_for_params(n) == L
    assert random_hea_params(L, np.random.default_rng(0)).shape == (n,)


def test_cz_edges_are_lattice_edges():
    expected = {(1, 2), (2, 3), (3, 4), (8, 7), (7, 6), (6, 5), (1, 8), (2, 7), (3, 6), (4, 5)}
    expected = {(min(a, b), max(a, b)) for a, b in expected}
    got = [(min(a, b), max(a, b)) for a, b in CZ_PAIRS_1IDX]
    assert len(got) == 10
    assert len(set(got)) == 10
    assert set(got) == expected == lattice_edges_1idx()


def test_cz_signs_match_explicit_pairs():
    for idx in range(256):
        bits = format(idx, "08b")
        n_ones = sum(1 for a, b in CZ_PAIRS_1IDX if bits[a - 1] == "1" and bits[b - 1] == "1")
        assert CZ_SIGNS[idx] == (-1) ** n_ones


@pytest.mark.parametrize("L", [1, 2, 3])
def test_normalized(L):
    rng = np.random.default_rng(L)
    for _ in range(5):
        x = rng.uniform(-4, 4, n_hea_parameters(L))
        psi = hea_state(x)
        assert psi.shape == (256,)
        assert np.isclose(np.vdot(psi, psi).real, 1.0, atol=1e-12)


@pytest.mark.parametrize("L", [1, 2, 3])
def test_zero_angles_give_all_zero(L):
    psi = hea_state(np.zeros(n_hea_parameters(L)))
    expected = np.zeros(256)
    expected[0] = 1.0
    assert np.allclose(psi, expected)


@pytest.mark.parametrize("q", range(8))
def test_pi_flips_single_qubit(q):
    # θ=π on qubit q (0-indexed = q_{q+1}) in the final RY layer flips it: |0..1_q..0⟩.
    L = 2
    x = np.zeros(n_hea_parameters(L)).reshape(L + 1, 8)
    x[L, q] = np.pi
    psi = hea_state(x.reshape(-1))
    bits = ["0"] * 8
    bits[q] = "1"
    idx = index_from_bitstring("".join(bits))
    assert np.isclose(abs(psi[idx]) ** 2, 1.0)
    # In the first layer too (CZ on a single-excitation state is trivial).
    x = np.zeros(n_hea_parameters(L)).reshape(L + 1, 8)
    x[0, q] = np.pi
    psi = hea_state(x.reshape(-1))
    assert np.isclose(abs(psi[idx]) ** 2, 1.0)


def test_cz_phase_on_edge():
    # RY(π/2) on q1,q2 in layer 1 -> |++>; CZ(1,2) gives relative phase on |11>.
    x = np.zeros(16).reshape(2, 8)
    x[0, 0] = x[0, 1] = np.pi / 2
    psi = hea_state(x.reshape(-1))
    assert np.isclose(psi[index_from_bitstring("11000000")], -0.5)
    assert np.isclose(psi[index_from_bitstring("00000000")], 0.5)


def test_short_hea_spsa():
    rng = np.random.default_rng(1)
    energies = rng.normal(size=256)
    gs = format(int(np.argmin(energies)), "08b")
    sim = HEASimulator(energies=energies, ground_bitstring=gs)
    res = optimize_hea_trial(sim, 1, maxiter=5, rng=np.random.default_rng(2))
    assert res.x.shape == (16,)
    assert 0.0 <= res.p_gs <= 1.0
    assert res.ground_bitstring == gs
