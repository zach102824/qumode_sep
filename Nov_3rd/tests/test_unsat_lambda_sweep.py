"""Verdict tests for the unsat Walsh λ sweep at μ=−1.00. No training."""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from Nov_3rd.seal_unsat import mark_of
from Nov_3rd.unsat_lambda_sweep import (
    ANCHOR_LAM,
    FIXED_MU,
    LAM_GRID,
    PUBLISHED_ANCHOR_R,
    cache_filename,
    decide_lambda_sweep,
    file_tag,
    gap_tail,
    lam_label,
    mu_ckpt_tag,
)
from Nov_3rd.unsat_rank2_xx import cache_filename as cache_filename_xx


def _cell(R, seeds=None):
    if seeds is None:
        seeds = [R, R, R]
    return {"R": float(R), "seed_Rs": [float(x) for x in seeds], "mark": mark_of(float(R))}


def _row(R12, seeds=None, peak=0.008, gap=0.028, r6=0.09, r9=0.08, r15=0.12, peak15=0.002):
    return {
        "walsh": {
            6: _cell(r6),
            9: _cell(r9),
            12: _cell(R12, seeds),
            15: _cell(r15),
        },
        "quantum": {
            6: {"peak_med": peak, "prod_gap_med": gap},
            9: {"peak_med": peak, "prod_gap_med": gap},
            12: {"peak_med": peak, "prod_gap_med": gap},
            15: {"peak_med": peak15, "prod_gap_med": gap},
        },
    }


def _grid(overrides=None, anchor_seeds=None):
    overrides = overrides or {}
    rows = {}
    anchor_seeds = anchor_seeds or [0.07706783164675858, 0.2072829709604493, 0.09675474400205866]
    for lam in LAM_GRID:
        if abs(lam - ANCHOR_LAM) < 1e-12:
            rows[lam_label(lam)] = _row(PUBLISHED_ANCHOR_R, seeds=anchor_seeds)
        else:
            spec = overrides.get(lam, {})
            extra = {k: spec[k] for k in spec if k not in ("R", "seeds")}
            rows[lam_label(lam)] = _row(spec.get("R", 0.20), seeds=spec.get("seeds"), **extra)
    return rows


def test_tags_match_mu_neg_filenames():
    assert FIXED_MU == -1.00
    assert file_tag(0.75) == "0p75"
    assert file_tag(1.25) == "1p25"
    assert file_tag(1.00) == "1p00"
    assert lam_label(1.25) == "1.25"
    assert mu_ckpt_tag(-1.00) == "mu-1p00"
    assert mu_ckpt_tag(-0.50) == "mu-0p50"
    name = cache_filename(12, "transfer", 64, 0.75, -1.00)
    assert name == "unsat_xx_n12_transfer_B64_l0.75_m-1.00.npz"
    assert name == cache_filename_xx(12, "transfer", 64, 0.75, -1.00)
    assert "n21" not in name
    assert 21 not in LAM_GRID
    assert 1.25 in LAM_GRID


def test_gap_tail_fraction():
    tail = gap_tail([0.019, 0.028, 0.030, 0.010], [0.001, 0.002, 0.001, 0.004])
    assert tail["n_inst"] == 4
    assert tail["n_gap_below"] == 2
    assert abs(tail["frac_gap_below"] - 0.5) < 1e-12
    assert abs(tail["gap_min"] - 0.010) < 1e-12
    assert abs(tail["peak_max"] - 0.004) < 1e-12


def test_anchor_stays_when_it_is_the_lowest_safe_R():
    out = decide_lambda_sweep(_grid({0.65: {"R": 0.140}, 1.25: {"R": 0.160}}))
    assert out["verdict"] == "PASS", out
    assert out["operating_lam"] == ANCHOR_LAM
    assert out["still_optimal"] is True
    assert out["split"] is False


def test_lower_safe_lambda_passes_without_a_002_drop():
    # A drop of 0.007 is below the old 0.02 switch bar. The gap-safe minimum still wins.
    out = decide_lambda_sweep(_grid({
        0.90: {"R": 0.090, "seeds": [0.090, 0.088, 0.092], "r15": 0.11, "gap": 0.025},
    }))
    assert out["verdict"] == "PASS", out
    assert abs(out["operating_lam"] - 0.90) < 1e-12
    assert out.get("still_optimal") is not True
    assert out["split"] is False


def test_two_safe_picks_lower_R():
    out = decide_lambda_sweep(_grid({
        0.90: {"R": 0.080},
        0.50: {"R": 0.070},
    }))
    assert out["verdict"] == "PASS", out
    assert abs(out["operating_lam"] - 0.50) < 1e-12


def test_thin_gap_reports_unconstrained_and_safe():
    out = decide_lambda_sweep(_grid({
        0.50: {"R": 0.040, "gap": 0.010, "seeds": [0.040, 0.041, 0.039]},
        1.25: {"R": 0.080, "gap": 0.022},
    }))
    assert out["verdict"] == "PASS", out
    assert out["split"] is True
    assert abs(out["unconstrained_lam"] - 0.50) < 1e-12
    assert abs(out["safe_lam"] - 1.25) < 1e-12
    assert abs(out["operating_lam"] - 1.25) < 1e-12
    assert "Unconstrained" in out["reason"]
    assert "Safe winner" in out["reason"]


def test_thin_gap_keeps_anchor_when_it_is_the_safe_minimum():
    out = decide_lambda_sweep(_grid({
        0.50: {"R": 0.040, "gap": 0.011},
    }))
    assert out["verdict"] == "PASS", out
    assert out["operating_lam"] == ANCHOR_LAM
    assert out["still_optimal"] is True
    assert out["split"] is True
    assert abs(out["unconstrained_lam"] - 0.50) < 1e-12
    assert abs(out["safe_lam"] - ANCHOR_LAM) < 1e-12


def test_no_safe_cell_needs_edit():
    rows = _grid({lam: {"R": 0.20, "gap": 0.01} for lam in LAM_GRID if abs(lam - ANCHOR_LAM) > 1e-12})
    rows[lam_label(ANCHOR_LAM)]["quantum"][12]["prod_gap_med"] = 0.015
    rows[lam_label(ANCHOR_LAM)]["quantum"][6]["prod_gap_med"] = 0.015
    rows[lam_label(ANCHOR_LAM)]["quantum"][9]["prod_gap_med"] = 0.015
    rows[lam_label(ANCHOR_LAM)]["quantum"][15]["prod_gap_med"] = 0.015
    out = decide_lambda_sweep(rows)
    assert out["verdict"] == "needs_edit", out
    assert out["safe_lam"] is None
    assert out["operating_lam"] is None
    assert abs(out["unconstrained_lam"] - ANCHOR_LAM) < 1e-12


def test_bitstring_safe_winner_kills():
    out = decide_lambda_sweep(_grid({0.50: {"R": 0.040, "peak": 0.80, "gap": 0.04}}))
    assert out["verdict"] == "KILL", out
    assert abs(out["operating_lam"] - 0.50) < 1e-12
    assert "bitstring" in out["reason"]


def test_missing_lambda_needs_edit():
    rows = _grid()
    del rows[lam_label(1.25)]
    out = decide_lambda_sweep(rows)
    assert out["verdict"] == "needs_edit", out
    assert "1.25" in out["reason"]


def test_drifted_anchor_needs_edit():
    rows = _grid()
    rows[lam_label(ANCHOR_LAM)]["walsh"][12] = _cell(0.40, [0.40, 0.40, 0.40])
    out = decide_lambda_sweep(rows)
    assert out["verdict"] == "needs_edit", out
    assert "PR #28" in out["reason"]


def test_anchor_bitstring_kills_when_it_is_the_safe_minimum():
    rows = _grid()
    rows[lam_label(ANCHOR_LAM)]["quantum"][12]["peak_med"] = 0.72
    out = decide_lambda_sweep(rows)
    assert out["verdict"] == "KILL", out
    assert out["operating_lam"] == ANCHOR_LAM


def test_safe_winner_with_n15_flip_needs_edit():
    out = decide_lambda_sweep(_grid({0.90: {"R": 0.070, "r15": 1.15, "gap": 0.03}}))
    assert out["verdict"] == "needs_edit", out
    assert abs(out["operating_lam"] - 0.90) < 1e-12


def test_safe_winner_needs_two_seed_beats():
    out = decide_lambda_sweep(_grid({
        0.65: {"R": 0.070, "seeds": [0.05, 1.20, 1.30], "gap": 0.03},
    }))
    assert out["verdict"] == "needs_edit", out
    assert abs(out["operating_lam"] - 0.65) < 1e-12
    assert "seeds" in out["reason"]
