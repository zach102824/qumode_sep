"""General encoding spec (variable→slot permutation, cavity codeword maps) and ansatz symmetry."""

from __future__ import annotations

from math import factorial

import numpy as np
import pytest

from noiseless.circuit_local_ecd import random_parameters
from noiseless.encoding import (
    EncodingSpec,
    SWAP_DE_AB,
    bits_from_bitstring,
    bitstring_from_bits,
    canonical_perm,
    denm_from_bits,
    bits_from_denm,
    denm_from_flat,
    distinct_assignments,
    energy_tensor_for_spec,
    energy_tensor_from_terms,
    load_four_sat_npz,
    logical_energies_from_terms,
)
from noiseless.spsa_gibbs import NoiselessSimulator, ground_flat_from_bitstring, grow_trial
from noiseless.tests.test_spsa_adam import _HAMS, needs_hams
from noiseless.unitaries import build_fixed_u

_U = build_fixed_u("jp")


def _sim(inst, enc, L=2):
    return NoiselessSimulator(
        u_fixed=_U, energy_tensor=inst["energy_tensor"], n_layers=L,
        ground_bitstring=inst["ground_bitstring"],
        ground_flat_index=ground_flat_from_bitstring(inst["ground_bitstring"], enc), encoding=enc)


def test_spec_validation_and_names():
    assert EncodingSpec().label() == "perm=01234567"
    with pytest.raises(ValueError):
        EncodingSpec(perm=(0, 0, 1, 2, 3, 4, 5, 6))
    with pytest.raises(NotImplementedError):
        EncodingSpec(cavity_a=(0, 1, 2, 3, 4, 5, 6, 9))
    assert EncodingSpec().slot_of_variable()["Z3"] == "A2"


@pytest.mark.parametrize("perm", [tuple(range(8)), (3, 7, 0, 5, 1, 6, 2, 4)])
@pytest.mark.parametrize("name", ["binary", "gray"])
def test_bits_roundtrip(perm, name):
    base = EncodingSpec.from_name(name)
    spec = EncodingSpec(perm, base.cavity_a, base.cavity_b)
    seen = set()
    for idx in range(256):
        denm = denm_from_flat(idx)
        x = bits_from_denm(*denm, spec)
        assert denm_from_bits(x, spec) == denm
        seen.add(bitstring_from_bits(x))
    assert len(seen) == 256


@needs_hams
@pytest.mark.parametrize("name", ["binary", "gray"])
def test_identity_spec_reproduces_legacy_exactly(name):
    leg = load_four_sat_npz(_HAMS[0], encoding=name)
    new = load_four_sat_npz(_HAMS[0], encoding=EncodingSpec.from_name(name))
    assert np.array_equal(leg["energy_tensor"], new["energy_tensor"])
    assert leg["ground_flat_index"] == new["ground_flat_index"]
    assert leg["ground_bitstring"] == new["ground_bitstring"]
    le = logical_energies_from_terms(leg["terms"], leg["identity"])
    assert np.array_equal(energy_tensor_for_spec(le, EncodingSpec.from_name(name)), leg["energy_tensor"])
    # full optimisation is bit-for-bit identical
    runs = []
    for enc in (name, EncodingSpec.from_name(name)):
        inst = load_four_sat_npz(_HAMS[0], encoding=enc)
        runs.append(grow_trial(_sim(inst, enc, 1), final_layers=3, total_steps=30,
                               rng=np.random.default_rng(1), optimizer="spsa_adam"))
    assert np.array_equal(runs[0].x, runs[1].x) and runs[0].p_gs == runs[1].p_gs
    assert runs[0].most_likely_bitstring == runs[1].most_likely_bitstring


@needs_hams
def test_permutation_same_ground_energy_permuted_bits():
    perm = (3, 7, 0, 5, 1, 6, 2, 4)
    spec = EncodingSpec(perm)
    leg = load_four_sat_npz(_HAMS[1])
    new = load_four_sat_npz(_HAMS[1], encoding=spec)
    assert new["ground_energy"] == leg["ground_energy"]
    assert new["ground_bitstring"] == leg["ground_bitstring"]  # logical GS is encoding-free
    assert np.array_equal(np.sort(new["energies_flat"]), np.sort(leg["energies_flat"]))
    # physical GS bits are the logical GS bits placed on their slots
    x = bits_from_bitstring(leg["ground_bitstring"])
    phys = np.zeros(8, dtype=int)
    phys[list(perm)] = x
    d, e, na, nb = new["ground_denm"]
    assert (d, e, na, nb) == (phys[0], phys[1], int("".join(map(str, phys[2:5])), 2),
                              int("".join(map(str, phys[5:])), 2))
    # fast path == slow path
    le = logical_energies_from_terms(leg["terms"], leg["identity"])
    assert np.array_equal(energy_tensor_for_spec(le, spec), new["energy_tensor"])
    assert np.array_equal(energy_tensor_from_terms(leg["terms"], leg["identity"], encoding=spec),
                          new["energy_tensor"])


def _swap_params(x):
    y = np.asarray(x, dtype=float).reshape(-1, 8).copy()
    y[:, [0, 1, 2, 3, 4, 5, 6, 7]] = y[:, [1, 0, 3, 2, 5, 4, 7, 6]]
    return y.reshape(-1)


@needs_hams
def test_swap_dA_eB_is_an_ansatz_symmetry_but_partial_swaps_are_not():
    perm = (3, 7, 0, 5, 1, 6, 2, 4)
    spec = EncodingSpec(perm)
    sw = spec.swapped()
    x = random_parameters(3, np.random.default_rng(0))
    xs = _swap_params(x)
    s1 = _sim(load_four_sat_npz(_HAMS[0], encoding=spec), spec, 3)
    s2 = _sim(load_four_sat_npz(_HAMS[0], encoding=sw), sw, 3)
    for eta in (0.5, 3.0):
        assert s1.cost(x, eta) == pytest.approx(s2.cost(xs, eta), rel=1e-12, abs=1e-12)
    e1, e2 = s1.evaluate(x), s2.evaluate(xs)
    assert e1["p_gs"] == pytest.approx(e2["p_gs"], abs=1e-12)
    assert e1["most_likely_bitstring"] == e2["most_likely_bitstring"]
    # swapping only the transmons (d↔e) or only the cavities (A↔B) is NOT a symmetry
    for slot_map in ((1, 0, 2, 3, 4, 5, 6, 7), (0, 1, 5, 6, 7, 2, 3, 4)):
        other = EncodingSpec(tuple(slot_map[j] for j in perm))
        s3 = _sim(load_four_sat_npz(_HAMS[0], encoding=other), other, 3)
        best = min(abs(s3.cost(y, 3.0) - s1.cost(x, 3.0)) for y in (x, xs))
        assert best > 1e-6


def test_distinct_assignment_classes():
    reps = distinct_assignments()
    assert len(reps) == factorial(8) // 2 == 20160
    assert reps[0] == tuple(range(8))
    assert len(set(reps)) == len(reps)
    assert all(canonical_perm(q) == q for q in reps[:500])
    assert canonical_perm(tuple(SWAP_DE_AB)) == tuple(range(8))
