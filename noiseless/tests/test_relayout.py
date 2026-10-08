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
