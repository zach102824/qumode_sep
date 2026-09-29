"""Tests for Gray cavity encoding and layer growth (grow_trial)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from noiseless.circuit_local_ecd import (
    apply_circuit_np,
    born_probs_np,
    n_parameters,
    qobj_to_np,
    random_parameters,
)
from noiseless.encoding import (
    NFOCK,
    bitstring_from_bits,
    bits_from_denm,
    denm_from_bits,
    denm_from_flat,
    energy_tensor_from_terms,
    gray_decode,
    gray_encode,
    list_four_sat_npz,
    load_four_sat_npz,
    z_terms_from_npz,
)
from noiseless.spsa_gibbs import (
    NoiselessSimulator,
    ground_flat_from_bitstring,
    grow_trial,
    split_steps,
    transparent_layer_params,
)
from noiseless.unitaries import build_fixed_u

_REPO = Path(__file__).resolve().parents[2]
_HAM_DIR = _REPO / "Hamiltonians" / "four_sat"
_HAMS = list_four_sat_npz(_HAM_DIR)
needs_hams = pytest.mark.skipif(not _HAMS, reason="no four_sat Hamiltonians")


def test_gray_code_basics():
    assert [gray_encode(n) for n in range(8)] == [0, 1, 3, 2, 6, 7, 5, 4]
    for n in range(64):
        assert gray_decode(gray_encode(n)) == n
    # adjacent Fock numbers differ in exactly one Gray bit
    for n in range(NFOCK - 1):
        assert bin(gray_encode(n) ^ gray_encode(n + 1)).count("1") == 1


def test_gray_roundtrip_all_states():
    seen = set()
    for d in range(2):
        for e in range(2):
            for na in range(NFOCK):
                for nb in range(NFOCK):
                    bits = bits_from_denm(d, e, na, nb, "gray")
                    assert denm_from_bits(bits, "gray") == (d, e, na, nb)
                    seen.add(bitstring_from_bits(bits))
                    # transmon bits unchanged
                    assert bits[0] == d and bits[1] == e
    assert len(seen) == 256


def test_default_encoding_is_binary_and_unchanged():
    assert bitstring_from_bits(bits_from_denm(1, 0, 6, 0)) == "10110000"
    assert bitstring_from_bits(bits_from_denm(0, 1, 0, 5)) == "01000101"
    assert bitstring_from_bits(bits_from_denm(1, 0, 6, 0, "binary")) == "10110000"
    # gray: 6 -> 5 (101), 5 -> 7 (111)
    assert bitstring_from_bits(bits_from_denm(1, 0, 6, 0, "gray")) == "10101000"
    assert bitstring_from_bits(bits_from_denm(0, 1, 0, 5, "gray")) == "01000111"
    with pytest.raises(ValueError):
        bits_from_denm(0, 0, 0, 0, "unary")


@needs_hams
@pytest.mark.parametrize("path", _HAMS[:5], ids=lambda p: p.name)
def test_gray_energy_tensor_is_permutation(path):
    terms, meta = z_terms_from_npz(path)
    eb = energy_tensor_from_terms(terms, meta["identity"])
    eb2 = energy_tensor_from_terms(terms, meta["identity"], encoding="binary")
    eg = energy_tensor_from_terms(terms, meta["identity"], encoding="gray")
    assert np.array_equal(eb, eb2)
    np.testing.assert_array_equal(np.sort(eb.reshape(-1)), np.sort(eg.reshape(-1)))
    # explicit permutation: E_gray[d,e,a,b] = E_bin[d,e,gray(a),gray(b)]
    for a in range(NFOCK):
        for b in range(NFOCK):
            assert np.array_equal(eg[:, :, a, b], eb[:, :, gray_encode(a), gray_encode(b)])
    ib = load_four_sat_npz(path)
    ig = load_four_sat_npz(path, encoding="gray")
    assert ib["ground_bitstring"] == ig["ground_bitstring"]  # same logical GS
    assert ib["ground_energy"] == pytest.approx(ig["ground_energy"], abs=0)
    assert ib["n_ground"] == ig["n_ground"]
    gi = ground_flat_from_bitstring(ig["ground_bitstring"], "gray")
    assert gi == ig["ground_flat_index"]
    assert ig["energies_flat"][gi] == ib["ground_energy"]
    assert ground_flat_from_bitstring(ib["ground_bitstring"]) == ib["ground_flat_index"]


@needs_hams
def test_default_load_unchanged():
    path = _HAMS[0]
    a = load_four_sat_npz(path)
    b = load_four_sat_npz(path, encoding="binary")
    assert np.array_equal(a["energy_tensor"], b["energy_tensor"])
    assert a["ground_bitstring"] == b["ground_bitstring"]
    d, e, na, nb = denm_from_flat(a["ground_flat_index"])
    assert bitstring_from_bits(bits_from_denm(d, e, na, nb)) == a["ground_bitstring"]


@pytest.mark.parametrize("u_name", ["jp", "cz_nm", "identity"])
@pytest.mark.parametrize("L", [1, 2, 3])
def test_transparent_layer_preserves_probabilities(u_name, L):
    u = qobj_to_np(build_fixed_u(u_name))
    rng = np.random.default_rng(123 + L)
    for _ in range(3):
        x = random_parameters(L, rng)
        p0 = born_probs_np(apply_circuit_np(x, L, u))
        x1 = np.concatenate([x, transparent_layer_params()])
        p1 = born_probs_np(apply_circuit_np(x1, L + 1, u))
        assert np.max(np.abs(p0 - p1)) < 1e-12


def test_split_steps():
    assert split_steps(200, 1) == [200]
    assert split_steps(200, 2) == [100, 100]
    assert split_steps(200, 3) == [67, 67, 66]
    assert split_steps(200, 4) == [50, 50, 50, 50]
    assert sum(split_steps(201, 4)) == 201


def _grow_sim(encoding="binary"):
    path = _HAMS[0]
    inst = load_four_sat_npz(path, encoding=encoding)
    return NoiselessSimulator(
        u_fixed=build_fixed_u("jp"),
        energy_tensor=inst["energy_tensor"],
        n_layers=1,
        ground_bitstring=inst["ground_bitstring"],
        ground_flat_index=ground_flat_from_bitstring(inst["ground_bitstring"], encoding),
        encoding=encoding,
    )


@needs_hams
@pytest.mark.parametrize("encoding", ["binary", "gray"])
def test_grow_trial_stages_and_param_counts(encoding):
    sim = _grow_sim(encoding)
    res = grow_trial(sim, final_layers=3, total_steps=12, rng=np.random.default_rng(7))
    assert [s["n_layers"] for s in res.stages] == [1, 2, 3]
    assert [s["n_params"] for s in res.stages] == [n_parameters(L) for L in (1, 2, 3)]
    assert [s["steps"] for s in res.stages] == [4, 4, 4]
    assert res.x.size == n_parameters(3)
    assert sim.n_layers == 3
    assert res.nit == 12
    # insertion is transparent: p(GS) right after insert equals previous stage's final p(GS)
    for prev, cur in zip(res.stages[:-1], res.stages[1:]):
        assert abs(cur["p_gs_after_insert"] - prev["p_gs"]) < 1e-12
    assert res.p_gs == pytest.approx(res.stages[-1]["p_gs"])
    ev = sim.evaluate(res.x)
    assert ev["p_gs"] == pytest.approx(res.p_gs, abs=1e-12)


@needs_hams
def test_grow_trial_reproducible():
    r1 = grow_trial(_grow_sim(), final_layers=2, total_steps=10, rng=np.random.default_rng(99))
    r2 = grow_trial(_grow_sim(), final_layers=2, total_steps=10, rng=np.random.default_rng(99))
    assert np.array_equal(r1.x, r2.x)
    assert r1.p_gs == r2.p_gs
    assert r1.stages == r2.stages


@needs_hams
def test_runner_worker_encoding_and_grow(tmp_path):
    from noiseless.run_u_sweep import _worker

    base = dict(
        ham_path=str(_HAMS[0]),
        ham_file=_HAMS[0].name,
        u_name="jp",
        n_layers=2,
        trial=0,
        seed=5,
        steps=6,
        spsa_a=None,
    )
    r_bin = _worker(dict(base))
    r_bin2 = _worker(dict(base, encoding="binary"))
    assert r_bin["ok"] and r_bin["x"] == r_bin2["x"] and r_bin["encoding"] == "binary"
    r_g = _worker(dict(base, encoding="gray", grow=True))
    assert r_g["ok"], r_g.get("traceback")
    assert r_g["encoding"] == "gray" and r_g["grow"] is True
    assert [s["steps"] for s in r_g["stages"]] == [3, 3]
    assert r_g["ground_bitstring"] == r_bin["ground_bitstring"]
    r_g2 = _worker(dict(base, encoding="gray", grow=True))
    assert r_g2["x"] == r_g["x"]


@needs_hams
def test_grow_trial_steps_per_stage():
    sim = _grow_sim()
    res = grow_trial(
        sim, final_layers=3, total_steps=999, rng=np.random.default_rng(3), steps_per_stage=5
    )
    assert [s["steps"] for s in res.stages] == [5, 5, 5]
    assert res.nit == 15
    assert [s["n_params"] for s in res.stages] == [8, 16, 24]
    from noiseless.run_u_sweep import _worker

    r = _worker(
        dict(
            ham_path=str(_HAMS[0]), ham_file=_HAMS[0].name, u_name="jp", n_layers=2, trial=0,
            seed=5, steps=200, spsa_a=None, grow=True, grow_steps_per_stage=4,
        )
    )
    assert r["ok"], r.get("traceback")
    assert [s["steps"] for s in r["stages"]] == [4, 4] and r["nit"] == 8
