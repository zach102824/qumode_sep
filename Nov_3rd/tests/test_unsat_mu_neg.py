"""Verdict tests for the more-negative unsat Walsh μ sweep. No training."""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from Nov_3rd.seal_unsat import mark_of
from Nov_3rd.unsat_mu_neg import (
    ANCHOR_MU,
    EDGE_MU,
    MU_GRID,
    PUBLISHED_ANCHOR_R,
    cache_tag,
    decide_mu_neg,
    file_tag,
    mu_label,
)


def _cell(R, seeds=None):
    if seeds is None:
        seeds = [R, R, R]
    return {"R": float(R), "seed_Rs": [float(x) for x in seeds], "mark": mark_of(float(R))}


def _row(R12, seeds=None, peak=0.004, gap=0.06, r6=0.12, r9=0.12, r15=0.18, peak15=0.001):
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


def _grid(overrides=None):
    overrides = overrides or {}
    rows = {}
    for mu in MU_GRID:
        if abs(mu - ANCHOR_MU) < 1e-12:
            rows[mu_label(mu)] = _row(PUBLISHED_ANCHOR_R)
        else:
            spec = overrides.get(mu, {})
            rows[mu_label(mu)] = _row(
                spec.get("R", 0.20),
                seeds=spec.get("seeds"),
                **{k: spec[k] for k in spec if k not in ("R", "seeds")},
            )
    return rows


def test_tags_match_checkpoint_names():
    assert file_tag(-1.0) == "-1p00"
    assert cache_tag(-1.0) == "m-1.00"
    assert file_tag(-0.85) == "-0p85"
    assert cache_tag(-0.85) == "m-0.85"
    assert file_tag(-0.75) == "-0p75"
    assert file_tag(-0.65) == "-0p65"
    assert file_tag(-0.50) == "-0p50"
    assert cache_tag(-0.50) == "m-0.50"
    assert mu_label(-1.0) == "-1.00"
    assert mu_label(-0.50) == "-0.50"


def test_anchor_stays_when_nothing_is_clearly_lower():
    out = decide_mu_neg(_grid({-0.65: {"R": 0.140}, -1.0: {"R": 0.145}}))
    assert out["verdict"] == "PASS", out
    assert out["operating_mu"] == ANCHOR_MU
    assert out["still_optimal"] is True
    assert out.get("further_edge") is not True
    assert "μ=−0.50" in out["reason"]
    assert "PR #26" in out["reason"]


def test_clear_lower_mu_passes():
    out = decide_mu_neg(_grid({-0.75: {"R": 0.120, "seeds": [0.118, 0.121, 0.119], "r15": 0.16}}))
    assert out["verdict"] == "PASS", out
    assert abs(out["operating_mu"] + 0.75) < 1e-12
    assert out.get("still_optimal") is not True
    assert out.get("further_edge") is not True


def test_two_clear_picks_lower_R():
    out = decide_mu_neg(_grid({
        -0.85: {"R": 0.110},
        -0.65: {"R": 0.100},
    }))
    assert out["verdict"] == "PASS", out
    assert abs(out["operating_mu"] + 0.65) < 1e-12


def test_edge_win_is_called_further_edge():
    out = decide_mu_neg(_grid({-1.0: {"R": 0.090, "gap": 0.04, "peak": 0.002}}))
    assert out["verdict"] == "PASS", out
    assert abs(out["operating_mu"] - EDGE_MU) < 1e-12
    assert out["further_edge"] is True
    assert "further edge" in out["reason"]


def test_low_R_product_gap_does_not_replace_anchor():
    out = decide_mu_neg(_grid({-1.0: {"R": 0.080, "gap": 0.01}}))
    assert out["verdict"] == "PASS", out
    assert out["operating_mu"] == ANCHOR_MU
    assert out["quantumness_break"] is True
    assert "product gap" in out["reason"]
    assert "Quantumness break" in out["reason"]


def test_low_R_bitstring_does_not_replace_anchor():
    out = decide_mu_neg(_grid({-0.85: {"R": 0.080, "peak": 0.80}}))
    assert out["verdict"] == "PASS", out
    assert out["operating_mu"] == ANCHOR_MU
    assert out["quantumness_break"] is True
    assert "bitstring" in out["reason"]


def test_missing_mu_needs_edit():
    rows = _grid()
    del rows[mu_label(-0.85)]
    out = decide_mu_neg(rows)
    assert out["verdict"] == "needs_edit", out


def test_drifted_anchor_needs_edit():
    rows = _grid()
    rows[mu_label(ANCHOR_MU)]["walsh"][12] = _cell(0.40, [0.40, 0.40, 0.40])
    out = decide_mu_neg(rows)
    assert out["verdict"] == "needs_edit", out
    assert "PR #26" in out["reason"]


def test_anchor_bitstring_kills():
    rows = _grid()
    rows[mu_label(ANCHOR_MU)]["quantum"][12]["peak_med"] = 0.72
    out = decide_mu_neg(rows)
    assert out["verdict"] == "KILL", out


def test_clear_winner_with_n15_flip_needs_edit():
    out = decide_mu_neg(_grid({-0.75: {"R": 0.110, "r15": 1.15}}))
    assert out["verdict"] == "needs_edit", out
    assert abs(out["operating_mu"] + 0.75) < 1e-12
