"""Tests for the k-bit-per-cavity ECD generalization (noiseless/ecd_kbit.py, run_ecd_kbit.py)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from noiseless.circuit_local_ecd import apply_circuit_np, qobj_to_np, random_parameters
from noiseless.ecd_kbit import (
    KbitEcdSimulator,
    KbitLayout,
    apply_circuit,
    born,
    logical_energies_from_npz,
)
from noiseless.encoding import bits_from_denm, list_four_sat_npz, load_four_sat_npz
from noiseless.spsa_gibbs import NoiselessSimulator, ground_flat_from_bitstring, grow_trial
from noiseless.unitaries import build_fixed_u

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "Hamiltonians"))
_HAMS8 = list_four_sat_npz(_REPO / "Hamiltonians" / "four_sat")
_HAMS16 = sorted((_REPO / "Hamiltonians" / "four_sat_scaling" / "n16").glob("four_sat_n16_*.npz"))
_FLEET = sorted((_REPO / "noiseless" / "results").glob("grow_tune_F_lr_sched_2*Z.json"))
needs8 = pytest.mark.skipif(not _HAMS8, reason="no n=8 four_sat Hamiltonians")
needs16 = pytest.mark.skipif(not _HAMS16, reason="no n=16 scaling instances")


def _legacy_sim(idx=0, L=1):
    inst = load_four_sat_npz(_HAMS8[idx])
    return NoiselessSimulator(u_fixed=build_fixed_u("jp"), energy_tensor=inst["energy_tensor"], n_layers=L,
                              ground_bitstring=inst["ground_bitstring"],
                              ground_flat_index=ground_flat_from_bitstring(inst["ground_bitstring"]))


def _k3_sim(idx=0, L=1, method="legacy"):
    E, meta = logical_energies_from_npz(_HAMS8[idx], 8)
    gs = load_four_sat_npz(_HAMS8[idx])["ground_bitstring"]
    return KbitEcdSimulator(KbitLayout(3, 8), E, L, gs, method=method)


def test_circuit_legacy_bitwise_and_fast_close_n8():
    u = qobj_to_np(build_fixed_u("jp"))
    for s in range(3):
        x = random_parameters(3, np.random.default_rng(s))
        old = apply_circuit_np(x, 3, u)
        assert np.array_equal(old, apply_circuit(x, 3, 8, method="legacy"))
        assert np.abs(old - apply_circuit(x, 3, 8, method="fast")).max() < 1e-12


def test_fast_matches_legacy_large_nf():
    x = 2.0 * random_parameters(2, np.random.default_rng(5))
    a = apply_circuit(x, 2, 140, method="legacy")
    b = apply_circuit(x, 2, 140, method="fast")
    assert np.abs(a - b).max() < 1e-11
    assert abs(np.linalg.norm(b) - 1) < 1e-12


@needs8
def test_k3_energy_and_ground_match_legacy():
    for i in range(3):
        leg = _legacy_sim(i)
        new = _k3_sim(i)
        assert np.array_equal(leg.energies_flat, new.energies_flat)  # no leaked levels at nf = 2^k
        assert new.ground_flat_index == leg.ground_flat_index
        assert not (~new.valid).any()


@needs8
def test_k3_grow_trial_bit_for_bit_vs_legacy_sim():
    kw = dict(final_layers=3, optimizer="spsa_adam", steps_schedule=[10, 10, 10], lr_schedule=[0.5, 0.2, 0.05])
    a = grow_trial(_legacy_sim(1), rng=np.random.default_rng(11), **kw)
    b = grow_trial(_k3_sim(1), rng=np.random.default_rng(11), **kw)
    assert np.array_equal(a.x, b.x) and a.p_gs == b.p_gs and a.energy_mean == b.energy_mean
    assert a.most_likely_bitstring == b.most_likely_bitstring


@needs8
@pytest.mark.skipif(not _FLEET, reason="grow_tune_F_lr_sched fleet json not present (gitignored)")
def test_k3_reproduces_stored_tuned_fleet_records():
    """Same seed -> identical p(GS)/<E> as the stored n=8 tuned fleet (jp L=4, lr 0.5/0.2/0.05/0.02)."""
    recs = json.loads(_FLEET[-1].read_text())["records"]
    by = {(r["ham_file"], r["trial"]): r for r in recs}
    for hf, t in (("four_sat_000.npz", 0), ("four_sat_000.npz", 3), ("four_sat_007.npz", 1)):
        r = by[(hf, t)]
        idx = [p.name for p in _HAMS8].index(hf)
        sim = _k3_sim(idx)
        g = grow_trial(sim, final_layers=4, rng=np.random.default_rng(r["seed"]), optimizer="spsa_adam",
                       total_steps=800, lr_schedule=[0.5, 0.2, 0.05, 0.02])
        assert g.p_gs == r["p_gs"] and g.energy_mean == r["energy_mean"] and g.nfev == r["nfev"] == 1604
        assert np.array_equal(g.x, np.asarray(r["x"]))


@needs16
def test_n16_bit_mapping_matches_clause_energies():
    from four_sat_scaling import load_instance, unsat_counts

    for path in _HAMS16[:3]:
        E, meta = logical_energies_from_npz(path, 16)
        d = load_instance(path)
        cnt = unsat_counts(d["clauses"], d["polarities"], 16).astype(float)
        assert np.allclose(E, cnt, atol=1e-9)
        lay = KbitLayout(7, 160)
        Ef = lay.energy_flat(E)
        rng = np.random.default_rng(0)
        for _ in range(200):
            dd, ee, na, nb = int(rng.integers(2)), int(rng.integers(2)), int(rng.integers(128)), int(rng.integers(128))
            bits = [dd, ee] + [(na >> (6 - j)) & 1 for j in range(7)] + [(nb >> (6 - j)) & 1 for j in range(7)]
            flat = ((dd * 2 + ee) * 160 + na) * 160 + nb
            assert Ef[flat] == cnt[int("".join(map(str, bits)), 2)]
            assert lay.logical_of_flat(flat) == int("".join(map(str, bits)), 2)
        # n = 8 convention: k = 3 bit map equals encoding.bits_from_denm
        for dd, ee, na, nb in ((1, 0, 5, 2), (0, 1, 7, 0), (1, 1, 3, 6)):
            b8 = bits_from_denm(dd, ee, na, nb)
            assert KbitLayout(3, 8).logical_of_flat(((dd * 2 + ee) * 8 + na) * 8 + nb) == int("".join(map(str, b8)), 2)
        # leaked levels: E_max + 1, and exactly 4*(160^2 - 128^2) of them
        sim = KbitEcdSimulator(lay, E, 1, meta["ground_bitstring"])
        assert np.all(Ef[~sim.valid] == E.max() + 1) and (~sim.valid).sum() == 4 * (160**2 - 128**2)
        assert Ef[sim.ground_flat_index] == 0.0 and E.min() == 0.0


@needs16
def test_n16_truncation_convergence():
    """Probabilities over the encoded block converge with nf for |β| up to ~12 (Fock ~ 36 per layer)."""
    rng = np.random.default_rng(3)
    x = random_parameters(4, rng)
    for ell in range(4):  # scale betas up to |β| ~ 12
        x[8 * ell: 8 * ell + 4] *= 4.0
    p = {nf: born(apply_circuit(x, 4, nf)).reshape(4, nf, nf) for nf in (160, 192, 256)}
    ref = p[256][:, :128, :128]
    for nf in (160, 192):
        assert np.abs(p[nf][:, :128, :128] - ref).max() < 1e-8


@needs16
def test_runner_worker_smoke(tmp_path):
    from noiseless import run_ecd_kbit as R

    job = dict(n=16, L=2, inst=0, trial=0, path=str(_HAMS16[0]), nf=136, nf_check=144,
               seed=R.trial_seed(20260917, 0, 2, 0), steps_schedule=[2, 3], lr_schedule=[0.5, 0.05],
               kick=0.05, c=0.15)
    r = R.worker(job)
    assert r["nfev"] == 2 * 5 + 2 and r["n_params"] == 16 and len(r["abs_betas"]) == 4
    assert 0 <= r["leakage"] <= 1 and r["nf_check"] == 144
    assert [s["adam_lr"] for s in r["stages"]] == [0.5, 0.05]
    assert r["seed"] == 20260917 + 11_000 + 20
