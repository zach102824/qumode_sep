"""Verdict tests for the unsat Walsh μ sweep. No training."""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from Nov_3rd.seal_unsat import mark_of
from Nov_3rd.unsat_mu_sweep import (
    ANCHOR_MU,
    MU_GRID,
    PUBLISHED_ANCHOR_R,
    cache_tag,
    decide_mu_sweep,
    file_tag,
    mu_label,
)


def _cell(R, seeds=None):
    if seeds is None:
        seeds = [R, R, R]
    return {"R": float(R), "seed_Rs": [float(x) for x in seeds], "mark": mark_of(float(R))}


def _row(R12, seeds=None, peak=0.04, gap=0.11, r6=0.25, r9=0.28, r15=0.36, peak15=0.01):
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
    for mu in MU_GRID:
        if abs(mu - ANCHOR_MU) < 1e-12:
            rows[mu_label(mu)] = _row(
                PUBLISHED_ANCHOR_R,
                seeds=anchor_seeds or [0.33011169158114184, 0.5263177552725697, 0.32848156792745614],
            )
        else:
            spec = overrides.get(mu, {})
            rows[mu_label(mu)] = _row(spec.get("R", 0.40), seeds=spec.get("seeds"), **{
                k: spec[k] for k in spec if k not in ("R", "seeds")
            })
    return rows


def test_tags_match_pr23_filenames():
    assert file_tag(-0.25) == "-0p25"
    assert cache_tag(-0.25) == "m-0.25"
    assert file_tag(0.0) == "0p00"
    assert cache_tag(-0.50) == "m-0.50"
    assert cache_tag(-0.05) == "m-0.05"
    assert mu_label(-0.25) == "-0.25"
    assert mu_label(0.0) == "+0.00"


def test_anchor_stays_when_nothing_is_clearly_lower():
    out = decide_mu_sweep(_grid({-0.35: {"R": 0.320}, -0.15: {"R": 0.345}}))
    assert out["verdict"] == "PASS", out
    assert out["operating_mu"] == ANCHOR_MU
    assert out["still_optimal"] is True


def test_clear_lower_mu_passes():
    out = decide_mu_sweep(_grid({-0.35: {"R": 0.300, "seeds": [0.300, 0.298, 0.305], "r15": 0.34}}))
    assert out["verdict"] == "PASS", out
    assert abs(out["operating_mu"] + 0.35) < 1e-12
    assert out.get("still_optimal") is not True


def test_two_clear_picks_lower_R():
    out = decide_mu_sweep(_grid({
        -0.35: {"R": 0.305},
        -0.15: {"R": 0.290},
    }))
    assert out["verdict"] == "PASS", out
    assert abs(out["operating_mu"] + 0.15) < 1e-12


def test_low_R_bitstring_does_not_replace_anchor():
    out = decide_mu_sweep(_grid({-0.50: {"R": 0.180, "peak": 0.80}}))
    assert out["verdict"] == "PASS", out
    assert out["operating_mu"] == ANCHOR_MU
    assert "bitstring" in out["reason"]


def test_missing_mu_needs_edit():
    rows = _grid()
    del rows[mu_label(0.0)]
    out = decide_mu_sweep(rows)
    assert out["verdict"] == "needs_edit", out


def test_drifted_anchor_needs_edit():
    rows = _grid()
    rows[mu_label(ANCHOR_MU)]["walsh"][12] = _cell(0.50, [0.50, 0.50, 0.50])
    out = decide_mu_sweep(rows)
    assert out["verdict"] == "needs_edit", out
    assert "PR #23" in out["reason"]


def test_anchor_bitstring_kills():
    rows = _grid()
    rows[mu_label(ANCHOR_MU)]["quantum"][12]["peak_med"] = 0.72
    out = decide_mu_sweep(rows)
    assert out["verdict"] == "KILL", out


def test_clear_winner_with_n15_flip_needs_edit():
    out = decide_mu_sweep(_grid({-0.35: {"R": 0.300, "r15": 1.15}}))
    assert out["verdict"] == "needs_edit", out
    assert abs(out["operating_mu"] + 0.35) < 1e-12
