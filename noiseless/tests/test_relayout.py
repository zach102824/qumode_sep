"""Polish + corner re-encoding, and the grow-then-relayout-at-L* protocol."""

from __future__ import annotations

import json

import numpy as np
import pytest

from noiseless.encoding import (
    EncodingSpec,
    bits_from_bitstring,
    corner_spec_for_bitstring,
    denm_from_bits,
    load_four_sat_npz,
    logical_energies_from_terms,
    polish_bitstring,
)
from noiseless.circuit_local_ecd import n_parameters
from noiseless.spsa_gibbs import (
    grow_trial,
    relayout_trial,
)
from noiseless.tests.test_spsa_adam import _HAMS, _sim, needs_hams
from noiseless.unitaries import build_fixed_u

_U = build_fixed_u("jp")


def test_corner_spec_puts_bitstring_at_fock_zero():
    bits = "11101110"
    spec = corner_spec_for_bitstring(bits)
    d, e, na, nb = denm_from_bits(bits_from_bitstring(bits), spec)
    x = bits_from_bitstring(bits)
    assert (d, e) == (int(x[0]), int(x[1]))
    assert (na, nb) == (0, 0)
    assert corner_spec_for_bitstring(bits, spec) == spec  # idempotent
    # Hamming-1 cavity LSB neighbour lands on Fock 1, not an interior level
    flip = bits[:4] + ("0" if bits[4] == "1" else "1") + bits[5:]
    _, _, na1, _ = denm_from_bits(bits_from_bitstring(flip), spec)
    assert na1 in (1, 2, 4)


@needs_hams
def test_polish_recovers_unique_gs_from_hamming1():
    inst = load_four_sat_npz(_HAMS[0])
    le = logical_energies_from_terms(inst["terms"], inst["identity"])
    gs = inst["ground_bitstring"]
    assert polish_bitstring(gs, le, radius=1) == gs
    for k in range(8):
        nbr = list(gs)
        nbr[k] = "1" if nbr[k] == "0" else "0"
        assert polish_bitstring("".join(nbr), le, radius=1) == gs
    assert polish_bitstring(gs[::-1], le, radius=0) == gs[::-1]


@needs_hams
def test_relayout_round0_matches_grow_trial_bit_for_bit():
    inst = load_four_sat_npz(_HAMS[0])
    le = logical_energies_from_terms(inst["terms"], inst["identity"])
    g = grow_trial(
        _sim(2, 0),
        final_layers=2,
        total_steps=12,
        rng=np.random.default_rng(0),
        optimizer="spsa_adam",
    )
    r = relayout_trial(
        _U,
        le,
        inst["ground_bitstring"],
        final_layers=2,
        rng=np.random.default_rng(0),
        relayout_rounds=0,
        optimizer="spsa_adam",
        total_steps=12,
    )
    assert np.array_equal(g.x, r.x)
    assert g.p_gs == r.p_gs
    assert g.most_likely_bitstring == r.most_likely_bitstring
    assert r.rounds is not None and len(r.rounds) == 1
    assert r.rounds[0]["encoding"] == EncodingSpec().label()
    assert r.stages is not None and [s["n_layers"] for s in r.stages] == [1, 2]


@needs_hams
def test_relayout_extra_rounds_stay_at_final_layers():
    inst = load_four_sat_npz(_HAMS[0])
    le = logical_energies_from_terms(inst["terms"], inst["identity"])
    r = relayout_trial(
        _U,
        le,
        inst["ground_bitstring"],
        final_layers=2,
        rng=np.random.default_rng(1),
        relayout_rounds=2,
        optimizer="spsa_adam",
        steps_per_stage=3,
        lr_schedule=[0.5, 0.2],
    )
    assert r.rounds is not None and len(r.rounds) >= 1
    assert r.rounds[0]["stages"] is not None
    assert [s["n_layers"] for s in r.rounds[0]["stages"]] == [1, 2]
    extra = r.rounds[1:]
    if extra:
        assert all(rnd["stages"] is None for rnd in extra)
        assert all(rnd["n_layers"] == 2 for rnd in extra)
        assert all(rnd["encoding"] != r.rounds[0]["encoding"] for rnd in extra)
        # fresh-init extra round uses the first-stage lr, not 0.2
        assert extra[0]["adam_lr"] == pytest.approx(0.5)
        assert r.relayout_stop in ("fixed_point", "max_rounds")
    # top-level stages still describe the L=1→2 growth
    assert r.stages is not None and r.stages[0]["n_layers"] == 1
    assert r.x.size == n_parameters(2)


@needs_hams
def test_relayout_cli_records_rounds(tmp_path):
    from noiseless.run_u_sweep import main

    rc = main(
        [
            "--preset", "legacy",
            "--u-names", "jp",
            "--layers", "2",
            "--optimizer", "spsa_adam",
            "--grow",
            "--grow-steps-per-stage", "2",
            "--grow-lr-schedule", "0.5,0.2",
            "--relayout",
            "--relayout-rounds", "1",
            "--max-h", "1",
            "--trials", "1",
            "--workers", "1",
            "--outdir", str(tmp_path),
            "--tag", "rl",
        ]
    )
    assert rc == 0
    d = json.loads(next(tmp_path.glob("rl_2*Z.json")).read_text())
    assert d["args"]["relayout"] is True
    rec = d["records"][0]
    assert rec["ok"] and rec["relayout"] is True
    assert rec["rounds"] and rec["rounds"][0]["stages"]
    assert rec["stages"][0]["n_layers"] == 1
    assert rec["encoding_base"] == "binary"
    # --no-relayout keeps a plain grow record
    rc = main(
        [
            "--preset", "legacy",
            "--u-names", "jp",
            "--layers", "2",
            "--optimizer", "spsa_adam",
            "--grow",
            "--grow-steps-per-stage", "2",
            "--max-h", "1",
            "--trials", "1",
            "--workers", "1",
            "--outdir", str(tmp_path),
            "--tag", "nrl",
        ]
    )
    assert rc == 0
    d2 = json.loads(next(tmp_path.glob("nrl_2*Z.json")).read_text())
    assert d2["args"]["relayout"] is False
    assert d2["records"][0]["rounds"] is None
    assert d2["records"][0]["nfev"] == 2 * 5


# --------------------------------------------------------------------------- 2026-10-08 additions


def test_rule_layout_best_and_bad_place_bitstring_as_specified():
    from itertools import product

    from noiseless.encoding import gs_location_binary, rule_layout_for_bitstring

    rng = np.random.default_rng(0)
    for _ in range(12):
        g = "".join(str(int(b)) for b in rng.integers(0, 2, size=8))
        best = rule_layout_for_bitstring(g, "best")
        assert best.cavity_a == tuple(range(8)) and best.cavity_b == tuple(range(8))  # binary
        _, _, na, nb = gs_location_binary(g, best.perm)
        assert na in (0, 7) and nb in (0, 7)  # every 8-bit string has a tier-0/0 layout
        assert rule_layout_for_bitstring(g, "best") == best  # deterministic
        if 2 <= g.count("1") <= 6:
            bad = rule_layout_for_bitstring(g, "bad")
            _, _, na, nb = gs_location_binary(g, bad.perm)
            assert 2 <= na <= 5 and 2 <= nb <= 5
    with pytest.raises(ValueError):
        rule_layout_for_bitstring("11111111", "bad")
    del product


def test_small_beta_parameters_same_rng_stream_and_small_beta():
    from noiseless.circuit_local_ecd import random_parameters
    from noiseless.spsa_gibbs import small_beta_parameters

    r1, r2 = np.random.default_rng(7), np.random.default_rng(7)
    xs = small_beta_parameters(4, r1)
    xr = random_parameters(4, r2)
    assert r1.random() == r2.random()  # same number of draws
    for ell in range(4):
        b = 8 * ell
        mags = np.abs(xs[b : b + 2] + 1j * xs[b + 2 : b + 4])
        assert np.all(mags <= 0.1 + 1e-12)
        assert np.allclose(xs[b + 4 : b + 8], xr[b + 4 : b + 8])


@needs_hams
@pytest.mark.parametrize("init", ["random", "small", "warm", "grow"])
@pytest.mark.parametrize("target", ["xor_vacuum", "rule"])
def test_relayout_options_run_and_round0_unchanged(init, target):
    inst = load_four_sat_npz(_HAMS[0])
    le = logical_energies_from_terms(inst["terms"], inst["identity"])
    kw = dict(final_layers=2, grow=True, optimizer="spsa_adam", steps_per_stage=3,
              relayout_rounds=2, relayout_steps=3)
    base = relayout_trial(_U, le, inst["ground_bitstring"], rng=np.random.default_rng(11), **kw)
    res = relayout_trial(_U, le, inst["ground_bitstring"], rng=np.random.default_rng(11),
                         relayout_init=init, relayout_target=target, relayout_return="best", **kw)
    assert res.rounds[0]["p_gs"] == base.rounds[0]["p_gs"]  # round 0 independent of options
    assert sum(r["selected"] for r in res.rounds) == 1
    sel = next(r for r in res.rounds if r["selected"])
    assert sel["fun_common_eta"] == min(r["fun_common_eta"] for r in res.rounds)
    assert res.p_gs == sel["p_gs"]
    assert res.nfev == sum(r["nfev"] for r in res.rounds)
    if target == "rule" and len(res.rounds) > 1:
        assert "A=" not in res.rounds[1]["encoding"]  # permutation-only, binary code
    if init != "random" and len(res.rounds) > 1:
        assert res.rounds[1]["init"] == init


@needs_hams
def test_relayout_defaults_unchanged_last_round_returned():
    inst = load_four_sat_npz(_HAMS[0])
    le = logical_energies_from_terms(inst["terms"], inst["identity"])
    res = relayout_trial(_U, le, inst["ground_bitstring"], rng=np.random.default_rng(2),
                         final_layers=2, grow=True, optimizer="spsa_adam", steps_per_stage=3,
                         relayout_rounds=2, relayout_steps=3)
    assert res.rounds[-1]["selected"] and res.p_gs == res.rounds[-1]["p_gs"]


def test_layout_spec_names():
    from noiseless.encoding import gs_location_binary
    from noiseless.layouts import layout_spec

    g = "11111100"
    assert layout_spec("identity", g) == EncodingSpec()
    _, _, na, nb = gs_location_binary(g, layout_spec("rule_best", g).perm)
    assert (na, nb) == (7, 7)
    _, _, na, nb = gs_location_binary(g, layout_spec("rule_bad", g).perm)
    assert 2 <= na <= 5 and 2 <= nb <= 5


@needs_hams
def test_relayout_grow_init_regrows_from_L1():
    inst = load_four_sat_npz(_HAMS[0])
    le = logical_energies_from_terms(inst["terms"], inst["identity"])
    res = relayout_trial(_U, le, inst["ground_bitstring"], rng=np.random.default_rng(4),
                         final_layers=3, grow=True, optimizer="spsa_adam", steps_per_stage=3,
                         relayout_rounds=1, relayout_steps=2, relayout_init="grow",
                         relayout_target="rule", lr_schedule=[0.5, 0.2, 0.05])
    if len(res.rounds) > 1:
        r1 = res.rounds[1]
        assert [st["n_layers"] for st in r1["stages"]] == [1, 2, 3]
        assert r1["nfev"] == 3 * (2 * 2 + 1)
        assert [st["adam_lr"] for st in r1["stages"]] == [0.5, 0.2, 0.05]
