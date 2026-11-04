"""Verdict tests for the quantumness-safe unsat Walsh μ grid. No training."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from Nov_3rd.seal_unsat import mark_of
from Nov_3rd.unsat_mu_safe import (
    ANCHOR_MU,
    FRAC_MIN,
    GAP_FLOOR,
    MED_GAP_MIN,
    MU_GRID,
    PUBLISHED_ANCHOR_R,
    RESUME_R,
    cache_tag,
    decide_mu_safe,
    file_tag,
    gap_stats,
    mu_label,
)


def _cell(R, seeds=None):
    if seeds is None:
        seeds = [R, R, R]
    return {"R": float(R), "seed_Rs": [float(x) for x in seeds], "mark": mark_of(float(R))}


def _row(R, gap=0.06, frac=1.0, peak=0.003, seeds=None, n_inst=64, r6=0.11, r9=0.11, r15=0.16):
    n_ge = int(round(float(frac) * n_inst))
    # Keep the stored fraction exact rather than the rounded count's ratio.
    return {
        "walsh": {
            6: _cell(r6),
            9: _cell(r9),
            12: _cell(R, seeds),
            15: _cell(r15),
        },
        "quantum": {
            6: {"peak_med": peak, "prod_gap_med": gap},
            9: {"peak_med": peak, "prod_gap_med": gap},
            12: {
                "peak_med": peak,
                "prod_gap_med": gap,
                "frac_gap_ge": float(frac),
                "n_inst": int(n_inst),
                "n_gap_ge": n_ge,
                "gap_min": float(gap) / 2.0,
            },
            15: {"peak_med": 0.001, "prod_gap_med": gap},
        },
    }


def _grid(overrides=None):
    overrides = overrides or {}
    rows = {}
    for mu in MU_GRID:
        if abs(mu - ANCHOR_MU) < 1e-12:
            spec = {"R": PUBLISHED_ANCHOR_R, "gap": 0.065, "frac": 1.0}
        elif any(abs(mu - known) < 1e-12 for known in RESUME_R):
            spec = {"R": RESUME_R[mu], "gap": 0.028, "frac": 0.984375}
        else:
            spec = {"R": 0.20, "gap": 0.04, "frac": 1.0}
        spec.update(overrides.get(mu, {}))
        rows[mu_label(mu)] = _row(
            spec["R"],
            gap=spec.get("gap", 0.06),
            frac=spec.get("frac", 1.0),
            peak=spec.get("peak", 0.003),
            seeds=spec.get("seeds"),
            n_inst=spec.get("n_inst", 64),
        )
    return rows


def test_tags_match_prepare_names():
    assert file_tag(-0.95) == "-0p95"
    assert cache_tag(-0.95) == "m-0.95"
    assert file_tag(-0.90) == "-0p90"
    assert cache_tag(-0.90) == "m-0.90"
    assert file_tag(-0.80) == "-0p80"
    assert cache_tag(-0.80) == "m-0.80"
    assert file_tag(-0.60) == "-0p60"
    assert cache_tag(-0.60) == "m-0.60"
    assert file_tag(-1.0) == "-1p00"
    assert cache_tag(-1.0) == "m-1.00"
    assert mu_label(-0.60) == "-0.60"
    assert 21 not in MU_GRID


def test_gap_floor_counts_equality_and_the_95_percent_count():
    ge = np.array([GAP_FLOOR] * 61 + [0.01] * 3)
    below = np.array([GAP_FLOOR] * 60 + [0.019] * 4)
    assert gap_stats(ge)["frac_gap_ge"] + 1e-12 >= FRAC_MIN
    assert gap_stats(below)["frac_gap_ge"] < FRAC_MIN
    assert gap_stats(ge)["n_gap_ge"] == 61
    assert gap_stats(below)["n_gap_below"] == 4


def test_safe_is_constrained_min_and_edge_is_unconstrained():
    out = decide_mu_safe(_grid({
        -1.00: {"R": RESUME_R[-1.00], "gap": 0.028, "frac": 63 / 64},
        -0.60: {"R": 0.135, "gap": 0.053, "frac": 1.0},
        -0.50: {"R": PUBLISHED_ANCHOR_R, "gap": 0.065, "frac": 1.0},
    }))
    assert out["verdict"] == "PASS", out
    assert abs(out["safe_mu"] + 0.60) < 1e-12
    assert abs(out["operating_mu"] + 0.60) < 1e-12
    assert abs(out["edge_mu"] + 1.00) < 1e-12
    assert out["same_champ"] is False
    assert "SAFE" in out["reason"]
    assert "EDGE" in out["reason"]
    assert "Operate at μ=-0.60" in out["reason"]


def test_same_champ_when_the_minimum_clears_the_bar():
    out = decide_mu_safe(_grid({
        -1.00: {"R": RESUME_R[-1.00], "gap": 0.055, "frac": 1.0},
    }))
    assert out["verdict"] == "PASS", out
    assert abs(out["safe_mu"] + 1.00) < 1e-12
    assert abs(out["edge_mu"] + 1.00) < 1e-12
    assert out["same_champ"] is True
    assert "same champ" in out["reason"]


def test_resumed_anchor_must_match():
    rows = _grid()
    rows[mu_label(ANCHOR_MU)]["walsh"][12] = _cell(0.40)
    out = decide_mu_safe(rows)
    assert out["verdict"] == "needs_edit", out
    assert "resumed checkpoint" in out["reason"]


def test_median_gap_just_under_the_bar_is_not_safe():
    out = decide_mu_safe(_grid({
        -0.60: {"R": 0.120, "gap": MED_GAP_MIN - 0.001, "frac": 1.0},
    }))
    assert out["verdict"] == "PASS", out
    assert abs(out["safe_mu"] - ANCHOR_MU) < 1e-12
    assert abs(out["edge_mu"] + 1.00) < 1e-12


def test_median_gap_at_the_bar_is_safe():
    out = decide_mu_safe(_grid({
        -0.60: {"R": 0.120, "gap": MED_GAP_MIN, "frac": 1.0, "seeds": [0.12, 0.12, 0.12]},
    }))
    assert out["verdict"] == "PASS", out
    assert abs(out["safe_mu"] + 0.60) < 1e-12


def test_sixty_of_sixty_four_fails_the_fraction_bar():
    out = decide_mu_safe(_grid({
        -0.60: {"R": 0.120, "gap": 0.06, "frac": 60 / 64, "n_inst": 64},
    }))
    assert abs(out["safe_mu"] - ANCHOR_MU) < 1e-12
    assert any(abs(item["mu"] + 0.60) < 1e-12 for item in out["infeasible"])


def test_sixty_one_of_sixty_four_passes():
    out = decide_mu_safe(_grid({
        -0.60: {"R": 0.120, "gap": 0.06, "frac": 61 / 64, "n_inst": 64},
    }))
    assert abs(out["safe_mu"] + 0.60) < 1e-12


def test_missing_mu_needs_edit():
    rows = _grid()
    del rows[mu_label(-0.90)]
    out = decide_mu_safe(rows)
    assert out["verdict"] == "needs_edit", out
    assert "-0.90" in out["reason"]


def test_safe_loss_to_fourier_kills():
    overrides = {mu: {"R": 1.20, "gap": 0.06, "frac": 1.0} for mu in MU_GRID}
    overrides[ANCHOR_MU] = {"R": PUBLISHED_ANCHOR_R, "gap": 0.02, "frac": 0.5}
    # Every thick-gap point loses, and the anchor itself misses the gap bar,
    # so the only feasible points are the ones we set to R=1.20.
    for mu in RESUME_R:
        if abs(mu - ANCHOR_MU) > 1e-12:
            overrides[mu] = {"R": RESUME_R[mu], "gap": 0.02, "frac": 0.5}
    out = decide_mu_safe(_grid(overrides))
    assert out["verdict"] == "KILL", out
    assert "Fourier" in out["reason"]


def test_tie_on_R_prefers_the_thicker_gap():
    out = decide_mu_safe(_grid({
        -0.60: {"R": 0.120, "gap": 0.052, "frac": 1.0},
        -0.80: {"R": 0.120, "gap": 0.070, "frac": 1.0},
    }))
    assert abs(out["safe_mu"] + 0.80) < 1e-12


def _run_all():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print("ok", fn.__name__)
    print(f"{len(fns)} passed")


if __name__ == "__main__":
    _run_all()
