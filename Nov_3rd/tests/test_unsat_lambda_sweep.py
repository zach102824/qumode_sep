"""Verdict tests for the unsat Walsh λ sweep. No training."""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from Nov_3rd.seal_unsat import mark_of
from Nov_3rd.unsat_lambda_sweep import (
    ANCHOR_LAM,
    LAM_GRID,
    PUBLISHED_ANCHOR_R,
    cache_filename,
    decide_lambda_sweep,
    file_tag,
    lam_label,
)
from Nov_3rd.unsat_rank2_xx import cache_filename as cache_filename_xx


def _cell(R, seeds=None):
    if seeds is None:
        seeds = [R, R, R]
    return {"R": float(R), "seed_Rs": [float(x) for x in seeds], "mark": mark_of(float(R))}


def _row(R12, seeds=None, peak=0.008, gap=0.065, r6=0.12, r9=0.12, r15=0.18, peak15=0.002):
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
    anchor_seeds = anchor_seeds or [0.1543244187015855, 0.2565897061852136, 0.13134464683799887]
    for lam in LAM_GRID:
        if abs(lam - ANCHOR_LAM) < 1e-12:
            rows[lam_label(lam)] = _row(PUBLISHED_ANCHOR_R, seeds=anchor_seeds)
        else:
            spec = overrides.get(lam, {})
            extra = {k: spec[k] for k in spec if k not in ("R", "seeds")}
            rows[lam_label(lam)] = _row(spec.get("R", 0.20), seeds=spec.get("seeds"), **extra)
    return rows


def test_tags_match_pr26_filenames():
    assert file_tag(0.75) == "0p75"
    assert file_tag(0.50) == "0p50"
    assert file_tag(1.00) == "1p00"
    assert file_tag(0.65) == "0p65"
    assert lam_label(0.75) == "0.75"
    name = cache_filename(12, "transfer", 64, 0.75, -0.50)
    assert name == "unsat_xx_n12_transfer_B64_l0.75_m-0.50.npz"
    assert name == cache_filename_xx(12, "transfer", 64, 0.75, -0.50)
    assert "n21" not in name
    assert 21 not in LAM_GRID


def test_anchor_stays_when_nothing_is_clearly_lower():
    out = decide_lambda_sweep(_grid({0.65: {"R": 0.140}, 0.90: {"R": 0.160}}))
    assert out["verdict"] == "PASS", out
    assert out["operating_lam"] == ANCHOR_LAM
    assert out["still_optimal"] is True


def test_clear_lower_lambda_passes():
    out = decide_lambda_sweep(_grid({
        0.90: {"R": 0.120, "seeds": [0.120, 0.118, 0.122], "r15": 0.16},
    }))
    assert out["verdict"] == "PASS", out
    assert abs(out["operating_lam"] - 0.90) < 1e-12
    assert out.get("still_optimal") is not True


def test_two_clear_picks_lower_R():
    out = decide_lambda_sweep(_grid({
        0.90: {"R": 0.125},
        0.50: {"R": 0.110},
    }))
    assert out["verdict"] == "PASS", out
    assert abs(out["operating_lam"] - 0.50) < 1e-12


def test_seed_median_must_clear_the_same_bar():
    # Ratio drops by 0.03, but the median per-seed ratio stays near the anchor.
    out = decide_lambda_sweep(_grid({
        1.00: {"R": 0.120, "seeds": [0.05, 0.149, 0.40]},
    }))
    assert out["verdict"] == "PASS", out
    assert out["operating_lam"] == ANCHOR_LAM
    assert out["still_optimal"] is True


def test_low_R_bitstring_does_not_replace_anchor():
    out = decide_lambda_sweep(_grid({0.50: {"R": 0.080, "peak": 0.80}}))
    assert out["verdict"] == "PASS", out
    assert out["operating_lam"] == ANCHOR_LAM
    assert "bitstring" in out["reason"]


def test_missing_lambda_needs_edit():
    rows = _grid()
    del rows[lam_label(1.00)]
    out = decide_lambda_sweep(rows)
    assert out["verdict"] == "needs_edit", out


def test_drifted_anchor_needs_edit():
    rows = _grid()
    rows[lam_label(ANCHOR_LAM)]["walsh"][12] = _cell(0.40, [0.40, 0.40, 0.40])
    out = decide_lambda_sweep(rows)
    assert out["verdict"] == "needs_edit", out
    assert "PR #26" in out["reason"]


def test_anchor_bitstring_kills():
    rows = _grid()
    rows[lam_label(ANCHOR_LAM)]["quantum"][12]["peak_med"] = 0.72
    out = decide_lambda_sweep(rows)
    assert out["verdict"] == "KILL", out


def test_clear_winner_with_n15_flip_needs_edit():
    out = decide_lambda_sweep(_grid({0.90: {"R": 0.110, "r15": 1.15}}))
    assert out["verdict"] == "needs_edit", out
    assert abs(out["operating_lam"] - 0.90) < 1e-12
