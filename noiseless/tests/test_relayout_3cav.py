"""Tests for the multi-cavity (3 cavity / 3 transmon) device, noiseless/relayout_3cav.py."""
from __future__ import annotations

import numpy as np
import pytest

from noiseless import spsa_gibbs as sg
from noiseless.ecd_kbit import KbitLayout, gray_decode_arr
from noiseless.relayout_3cav import (MultiCavLayout, XorMultiCavSim, explore_exploit_trial_3cav,
                                     make_param_helpers, param_layout)
from noiseless.circuit_local_ecd import random_parameters as rp2
from noiseless.relayout_kbit import XorKbitSim, explore_exploit_trial_kbit
from noiseless.run_relayout_kbit import energies_for, ham_paths


def _rand_E(n, seed=0):
    r = np.random.default_rng(seed)
    E = r.integers(0, 40, size=1 << n).astype(float)
    gs = int(r.integers(1 << n))
    E[gs] = -5.0
    return E, format(gs, f"0{n}b")


# --- (1) reduces to the 2-cavity code when C holds 0 bits -------------------------------------------
def test_param_helpers_match_2pair():
    h = make_param_helpers(2)
    a = h["random_parameters"](3, np.random.default_rng(5))
    b = rp2(3, np.random.default_rng(5))
    assert np.array_equal(a, b)
    assert np.array_equal(h["transparent_layer_params"](), sg.transparent_layer_params())
    assert np.array_equal(h["small_beta_parameters"](2, np.random.default_rng(1)),
                          sg.small_beta_parameters(2, np.random.default_rng(1)))


@pytest.mark.parametrize("code", ["gray", "binary"])
def test_layout_reduces_to_2cav(code):
    k, nf = 3, 24
    a = MultiCavLayout((k, k, 0), (nf, nf, 1), encoding=code)
    b = KbitLayout(k, nf, encoding=code)
    assert a.P == 2 and a.edges == [(0, 1)] and a.dims == b.dims
    assert np.array_equal(a.flat_of_logical(), b.flat_of_logical())
    assert np.array_equal(a.valid_mask(), b.valid_mask())
    from noiseless.classical_layout import slot_order
    assert a.slots() == slot_order(k)


@pytest.mark.parametrize("mask", [0, 0b101100])
def test_state_and_eval_reduce_to_2cav(mask):
    k, nf, n, L = 3, 24, 8, 3
    E, gs = _rand_E(n, 1)
    x = rp2(L, np.random.default_rng(9))
    a = XorMultiCavSim(MultiCavLayout((k, k, 0), (nf, nf, 1)), E, L, gs, xa=mask)
    b = XorKbitSim(KbitLayout(k, nf, encoding="gray"), E, L, gs, xa=mask >> k, xb=mask & 7)
    assert np.array_equal(a.energies_flat, b.energies_flat)
    assert a.ground_flat_index == b.ground_flat_index
    pa, pb = a.probs_from_x(x), b.probs_from_x(x)
    assert np.max(np.abs(pa - pb)) < 1e-13
    ea, eb = a.evaluate(x), b.evaluate(x)
    for kk in ("most_likely_bitstring", "success"):
        assert ea[kk] == eb[kk]
    for kk in ("p_gs", "energy_mean", "leakage"):
        assert abs(ea[kk] - eb[kk]) < 1e-12


def test_full_trial_reduces_to_2cav():
    """Whole explore+SA+relabel trial through the device hook == the original 2-cavity trial."""
    k, nf, n = 3, 24, 8
    E, gs = _rand_E(n, 2)
    kw = dict(final_layers=2, explore_rounds=2, explore_mask="sa", sa_budget=40, sa_seed=3, sa_mode="fixed_low",
              explore_layers=2, explore_steps=4, explore_eta_scale=128.0, raw_k=6, polish_t=2,
              relayout_rounds=1, relayout_steps=5, relayout_lr=0.05, steps_per_stage=4, lr_schedule=[0.5, 0.05],
              code="gray")
    a = explore_exploit_trial_kbit(E, gs, k=k, nf=nf, rng=np.random.default_rng(11), **kw)
    b = explore_exploit_trial_3cav(E, gs, ks=(k, k, 0), nfs=(nf, nf, 1), rng=np.random.default_rng(11), **kw)
    for kk in ("nfev", "n_lookups", "n_lookups_sa", "most_likely_bitstring", "success", "selected_round"):
        assert a[kk] == b[kk], kk
    assert abs(a["p_gs"] - b["p_gs"]) < 1e-9
    assert np.allclose(a["x"], b["x"], atol=1e-9)
    assert [r["next_guess"] for r in a["rounds"]] == [r["next_guess"] for r in b["rounds"]]
    # helpers restored after the context
    assert len(sg.transparent_layer_params()) == 8


# --- (2) energy diagonal mapping, n = 16, ks 5,4,4 -------------------------------------------------
def test_energy_diagonal_all_2_16_strings():
    p = ham_paths(16, "wmaxsat")[0]
    E, gs = energies_for(str(p), 16, "wmaxsat")
    lay = MultiCavLayout((5, 4, 4), (48, 32, 32), encoding="gray")
    assert lay.n_qubits == 16 and lay.dims == (2, 2, 2, 48, 32, 32) and lay.edges == [(0, 1), (1, 2)]
    idx = np.arange(1 << 16)
    flat = lay.flat_of_logical()
    assert len(np.unique(flat)) == 1 << 16  # injective
    # independent decode: bits (d, e, f | A 5 | B 4 | C 4), Gray code per cavity
    d, e, f = (idx >> 15) & 1, (idx >> 14) & 1, (idx >> 13) & 1
    nA, nB, nC = gray_decode_arr((idx >> 8) & 31), gray_decode_arr((idx >> 4) & 15), gray_decode_arr(idx & 15)
    ref = ((((d * 2 + e) * 2 + f) * 48 + nA) * 32 + nB) * 32 + nC
    assert np.array_equal(flat, ref)
    ef = lay.energy_flat(E)
    assert np.array_equal(ef[flat], E)
    valid = lay.valid_mask()
    assert valid.sum() == 1 << 16 and np.array_equal(np.flatnonzero(valid), np.sort(flat))
    assert np.all(ef[~valid] == E.max() + 1)
    assert all(lay.logical_of_flat(int(fl)) == int(i) for i, fl in zip(idx[::97], flat[::97]))
    # XOR relabel on all three cavity codes: the guess sits at Fock (0, 0, 0)
    g = int(gs, 2)
    sim = XorMultiCavSim(lay, E, 1, gs, xa=g & ((1 << 13) - 1))
    co = np.unravel_index(sim.ground_flat_index, lay.dims)
    assert co[3:] == (0, 0, 0) and co[:3] == ((g >> 15) & 1, (g >> 14) & 1, (g >> 13) & 1)
    assert np.array_equal(sim.energies_flat[flat], E[idx ^ (g & 8191)])


def test_3cav_circuit_unitary_and_transparent_layer():
    lay = MultiCavLayout((2, 2, 2), (8, 6, 6), ring=True)
    h = make_param_helpers(3)
    E, gs = _rand_E(9, 4)
    sim = XorMultiCavSim(lay, E, 2, gs)
    x = h["random_parameters"](2, np.random.default_rng(0))
    from noiseless.relayout_3cav import apply_circuit_multi
    psi = apply_circuit_multi(x, 2, lay)
    assert abs(np.vdot(psi, psi) - 1) < 1e-10
    p2 = sim.probs_from_x(x)
    sim.n_layers = 3
    p3 = sim.probs_from_x(np.concatenate([x, h["transparent_layer_params"]()]))
    assert np.max(np.abs(p2 - p3)) < 1e-12
    with param_layout(3):
        assert len(sg.transparent_layer_params()) == 12
