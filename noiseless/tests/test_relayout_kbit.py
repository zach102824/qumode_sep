"""Tests for the k-bit relayout (noiseless/relayout_kbit.py, run_relayout_kbit.py)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from noiseless.ecd_kbit import KbitLayout, logical_energies_from_npz
from noiseless.encoding import (
    EncodingSpec,
    corner_spec_for_bitstring,
    energy_tensor_for_spec,
    list_four_sat_npz,
    load_four_sat_npz,
    logical_energies_from_terms,
    polish_bitstring,
)
from noiseless.relayout_kbit import XorKbitSim, cavity_masks_for, default_nf, polish_logical, relayout_trial_kbit
from noiseless.run_relayout_kbit import ham_paths, lr_schedule_for, trial_seed
from noiseless.spsa_gibbs import ground_flat_from_bitstring, relayout_trial
from noiseless.unitaries import build_fixed_u

_REPO = Path(__file__).resolve().parents[2]
_HAMS8 = list_four_sat_npz(_REPO / "Hamiltonians" / "four_sat")


def _clause_energies(path, n):
    d = np.load(path)
    idx = np.arange(1 << n, dtype=np.int64)
    e = np.zeros(1 << n)
    for vs, ps in zip(d["clauses"], d["polarities"]):
        viol = np.ones(1 << n, dtype=bool)
        for v, p in zip(vs, ps):
            bit = (idx >> (n - 1 - int(v))) & 1
            viol &= bit == (0 if p > 0 else 1)
        e += viol
    return e


@pytest.mark.skipif(not _HAMS8, reason="no n=8 set")
def test_k3_legacy_matches_relayout_trial_bitwise():
    inst = load_four_sat_npz(_HAMS8[2])
    E = logical_energies_from_terms(inst["terms"], inst["identity"])
    gs = inst["ground_bitstring"]
    old = relayout_trial(build_fixed_u("jp"), E, gs, final_layers=4, rng=np.random.default_rng(7),
                         relayout_rounds=2, grow=True, optimizer="spsa_adam", steps_per_stage=4,
                         lr_schedule=[0.5, 0.2, 0.05, 0.02], relayout_steps=6, relayout_lr=0.1,
                         relayout_init="small", relayout_return="best", relayout_guess="best",
                         relayout_fixed_stop=False, relayout_target="xor_vacuum", relayout_layers=4)
    new = relayout_trial_kbit(E, gs, k=3, nf=8, final_layers=4, rng=np.random.default_rng(7), relayout_rounds=2,
                              relayout_steps=6, relayout_lr=0.1, steps_per_stage=4, method="legacy")
    assert [r["p_gs"] for r in old.rounds] == [r["p_gs"] for r in new["rounds"]]
    assert [r["fun_common_eta"] for r in old.rounds] == [r["fun_common_eta"] for r in new["rounds"]]
    assert [r["next_guess"] for r in old.rounds] == [r["next_guess"] for r in new["rounds"]]
    assert old.p_gs == new["p_gs"] and old.nfev == new["nfev"]
    assert np.array_equal(old.x, np.asarray(new["x"]))


@pytest.mark.skipif(not _HAMS8, reason="no n=8 set")
def test_k3_xor_energies_match_corner_spec():
    inst = load_four_sat_npz(_HAMS8[0])
    E = logical_energies_from_terms(inst["terms"], inst["identity"])
    rng = np.random.default_rng(1)
    for g in rng.integers(0, 256, size=6):
        bs = format(int(g), "08b")
        spec = corner_spec_for_bitstring(bs, EncodingSpec())
        xa, xb = cavity_masks_for(int(g), 3)
        sim = XorKbitSim(KbitLayout(3, 8), E, 1, inst["ground_bitstring"], method="legacy", xa=xa, xb=xb)
        assert np.array_equal(sim.energies_flat, energy_tensor_for_spec(E, spec).reshape(-1))
        assert sim.ground_flat_index == ground_flat_from_bitstring(inst["ground_bitstring"], spec)


@pytest.mark.parametrize("n", [10, 12])
def test_mask_energy_reorder_n10_n12(n):
    paths = ham_paths(n, "scaling")
    assert len(paths) == 20
    k = (n - 2) // 2
    nf = default_nf(k)
    E, meta = logical_energies_from_npz(paths[3], n)
    assert np.array_equal(E, _clause_energies(paths[3], n))  # Z expansion == clause count
    gs = meta["ground_bitstring"]
    rng = np.random.default_rng(n)
    for g in [int(gs, 2)] + list(rng.integers(0, 1 << n, size=4)):
        xa, xb = cavity_masks_for(int(g), k)
        sim = XorKbitSim(KbitLayout(k, nf), E, 1, gs, xa=xa, xb=xb)
        ef = sim.energies_flat.reshape(2, 2, nf, nf)
        d, e = (int(g) >> (2 * k + 1)) & 1, (int(g) >> (2 * k)) & 1
        assert ef[d, e, 0, 0] == E[int(g)]  # guess sits at Fock (0, 0), transmons unchanged
        for _ in range(50):
            dd, ee = rng.integers(0, 2, size=2)
            na, nb = rng.integers(0, 1 << k, size=2)
            lg = (int(dd) << (2 * k + 1)) | (int(ee) << (2 * k)) | ((int(na) ^ xa) << k) | (int(nb) ^ xb)
            assert ef[dd, ee, na, nb] == E[lg]
        assert np.all(ef[:, :, 1 << k:, :] == E.max() + 1) and np.all(ef[:, :, :, 1 << k:] == E.max() + 1)
        # decode of a basis state concentrated on the GS's physical location
        gl = int(gs, 2)
        gd, ge = (gl >> (2 * k + 1)) & 1, (gl >> (2 * k)) & 1
        ga, gb = ((gl >> k) & ((1 << k) - 1)) ^ xa, (gl & ((1 << k) - 1)) ^ xb
        assert sim.ground_flat_index == ((gd * 2 + ge) * nf + ga) * nf + gb
        assert sim.energies_flat[sim.ground_flat_index] == E.min() == 0


def test_polish_matches_n8_and_general():
    rng = np.random.default_rng(0)
    E8 = rng.integers(0, 5, size=256).astype(float)
    for v in range(0, 256, 7):
        assert polish_logical(v, E8, 8) == int(polish_bitstring(format(v, "08b"), E8, 1), 2)
    n = 10
    E = rng.integers(0, 6, size=1 << n).astype(float)
    for v in rng.integers(0, 1 << n, size=20):
        cands = [int(v)] + [int(v) ^ (1 << j) for j in range(n)]
        assert polish_logical(int(v), E, n) == min(cands, key=lambda u: (E[u], u))


def test_lr_schedule_and_seeds():
    assert lr_schedule_for(4) == [0.5, 0.2, 0.05, 0.02]
    s6 = lr_schedule_for(6)
    assert len(s6) == 6 and abs(s6[0] - 0.5) < 1e-12 and abs(s6[-1] - 0.02) < 1e-12
    assert all(a >= b for a, b in zip(s6, s6[1:]))
    # run_u_sweep seed with --seed-layers 4, U = jp (index 11)
    assert trial_seed(20260917, 3, 7) == 20260917 + 300_000 + 11_000 + 40 + 7


def test_small_n10_trial_runs_and_leakage_small():
    paths = ham_paths(10, "scaling")
    E, meta = logical_energies_from_npz(paths[0], 10)
    rec = relayout_trial_kbit(E, meta["ground_bitstring"], k=4, nf=default_nf(4), final_layers=2,
                              rng=np.random.default_rng(0), relayout_rounds=2, relayout_steps=10,
                              steps_per_stage=5, lr_schedule=[0.5, 0.05])
    assert len(rec["rounds"]) == 3
    assert rec["nfev"] == 2 * 11 + 2 * 21
    assert 0.0 <= rec["p_gs"] <= 1.0 and rec["leakage"] < 1e-3


def test_explore_exploit_smoke_n8():
    """Explore-exploit protocol: runs, counts evals, pool guess has minimal energy among candidates."""
    import numpy as np
    from noiseless.relayout_kbit import explore_exploit_trial_kbit
    from noiseless.run_relayout_kbit import energies_for, ham_paths

    E, gs = energies_for(str(ham_paths(8, "scaling")[0]), 8, "scaling")
    rec = explore_exploit_trial_kbit(E, gs, k=3, nf=24, final_layers=2, rng=np.random.default_rng(1),
                                     explore_rounds=2, topk=2, relayout_rounds=1, relayout_steps=5,
                                     steps_per_stage=3, lr_schedule=[0.5, 0.2])
    assert [q["phase"] for q in rec["rounds"]] == ["explore", "explore", "exploit"]
    assert rec["nfev"] == 2 * 2 * (2 * 3 + 1) + (2 * 5 + 1)
    assert rec["n_lookups"] == 3 * 2 * 9
    es = [q["next_guess_energy"] for q in rec["rounds"]]
    assert all(b <= a for a, b in zip(es, es[1:]))
    ex = rec["rounds"][2]
    g = int(rec["rounds"][1]["next_guess"], 2)
    assert (ex["xa"], ex["xb"]) == ((g >> 3) & 7, g & 7)
