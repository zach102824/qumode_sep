"""Verdict tests for the 2D unsat Walsh λ×μ grid. No training."""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from Nov_3rd.seal_unsat import mark_of
from Nov_3rd.unsat_lambda_mu_grid import (
    CELLS,
    FRAC_MIN,
    GAP_FLOOR,
    LAM_GRID,
    MED_GAP_MIN,
    MU_GRID,
    PUBLISHED_R,
    RESUME_CELLS,
    cell_key,
    decide_lambda_mu_grid,
    file_stem,
    gap_stats,
    is_resume,
)
import numpy as np


def _cell(R, seeds=None):
    if seeds is None:
        seeds = [R, R, R]
    return {"R": float(R), "seed_Rs": [float(x) for x in seeds], "mark": mark_of(float(R))}


def _row(R, gap=0.06, frac=1.0, peak=0.003, seeds=None, n_inst=64, r6=0.11, r9=0.11, r15=0.16, n_ge=None):
    if n_ge is None:
        n_ge = int(round(float(frac) * n_inst))
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
                "n_gap_ge": int(n_ge),
                "gap_min": float(gap) / 2.0,
            },
            15: {"peak_med": 0.001, "prod_gap_med": gap},
        },
    }


# Published gaps on the resumed cells. New cells sit above both champs.
_KNOWN_GAP = {
    (0.75, -0.90): 0.032,
    (0.75, -0.75): 0.041,
    (0.75, -0.60): 0.053,
    (0.75, -0.50): 0.065,
    (0.90, -0.50): 0.050325499554510114,
    (1.00, -0.50): 0.04280196992916723,
}


def _grid(overrides=None):
    overrides = overrides or {}
    rows = {}
    for lam, mu in CELLS:
        if (lam, mu) in PUBLISHED_R:
            spec = {"R": PUBLISHED_R[(lam, mu)], "gap": _KNOWN_GAP[(lam, mu)], "frac": 1.0}
        else:
            spec = {"R": 0.20, "gap": 0.04, "frac": 1.0}
        spec.update(overrides.get((lam, mu), {}))
        rows[cell_key(lam, mu)] = _row(
            spec["R"],
            gap=spec.get("gap", 0.06),
            frac=spec.get("frac", 1.0),
            peak=spec.get("peak", 0.003),
            seeds=spec.get("seeds"),
            n_inst=spec.get("n_inst", 64),
            r6=spec.get("r6", 0.11),
            r9=spec.get("r9", 0.11),
            r15=spec.get("r15", 0.16),
            n_ge=spec.get("n_ge"),
        )
    return rows


def test_grid_is_the_requested_product():
    assert MU_GRID == (-0.90, -0.75, -0.60, -0.50)
    assert LAM_GRID == (0.75, 0.90, 1.00)
    assert len(CELLS) == 12
    assert len(RESUME_CELLS) == 6
    assert all(is_resume(lam, mu) for lam, mu in RESUME_CELLS)
    assert sum(1 for lam, mu in CELLS if not is_resume(lam, mu)) == 6
    assert (1.00, -1.00) not in CELLS
    assert 21 not in LAM_GRID and 21 not in MU_GRID
    assert file_stem(0.75, -0.90) == "l0p75_m-0p90"
    assert file_stem(1.00, -0.50) == "l1p00_m-0p50"
    stems = [file_stem(lam, mu) for lam, mu in CELLS]
    assert len(stems) == len(set(stems))


def test_published_story_names_safe_and_edge():
    v = decide_lambda_mu_grid(_grid())
    assert v["verdict"] == "PASS"
    assert v["safe_lam"] == 0.75 and v["safe_mu"] == -0.50
    assert abs(v["safe_R"] - PUBLISHED_R[(0.75, -0.50)]) < 1e-12
    assert v["edge_lam"] == 0.75 and v["edge_mu"] == -0.90
    assert abs(v["edge_R"] - PUBLISHED_R[(0.75, -0.90)]) < 1e-12
    assert v["operating_lam"] == 0.75 and v["operating_mu"] == -0.50
    assert v["same_champ"] is False
    assert "SAFE" in v["reason"] and "EDGE" in v["reason"]


def test_lower_safe_r_replaces_the_anchor():
    v = decide_lambda_mu_grid(_grid({(1.00, -0.60): {"R": 0.080, "gap": 0.055, "frac": 1.0}}))
    assert v["verdict"] == "PASS"
    assert v["same_champ"] is True
    assert v["safe_lam"] == 1.00 and v["safe_mu"] == -0.60
    assert v["edge_lam"] == 1.00 and v["edge_mu"] == -0.60


def test_thin_low_r_is_edge_only():
    v = decide_lambda_mu_grid(_grid({(0.90, -0.90): {"R": 0.040, "gap": 0.030, "frac": 1.0}}))
    assert v["verdict"] == "PASS"
    assert v["edge_lam"] == 0.90 and v["edge_mu"] == -0.90
    assert v["safe_lam"] == 0.75 and v["safe_mu"] == -0.50


def test_instance_bar_blocks_edge():
    v = decide_lambda_mu_grid(
        _grid({(1.00, -0.90): {"R": 0.010, "gap": 0.040, "frac": 0.90, "n_ge": 58}})
    )
    assert v["verdict"] == "PASS"
    assert v["edge_lam"] == 0.75 and v["edge_mu"] == -0.90
    assert not any(
        abs(c["lam"] - 1.00) < 1e-12 and abs(c["mu"] + 0.90) < 1e-12 for c in v["edge_pool"]
    )


def test_sixty_of_sixty_four_fails_and_sixty_one_passes():
    fail = decide_lambda_mu_grid(
        _grid({(1.00, -0.90): {"R": 0.010, "gap": 0.080, "frac": 60 / 64, "n_ge": 60}})
    )
    assert fail["edge_mu"] == -0.90 and fail["edge_lam"] == 0.75
    ok = decide_lambda_mu_grid(
        _grid({(1.00, -0.90): {"R": 0.010, "gap": 0.040, "frac": 61 / 64, "n_ge": 61}})
    )
    assert ok["verdict"] == "PASS"
    assert ok["edge_lam"] == 1.00 and ok["edge_mu"] == -0.90
    assert ok["safe_lam"] == 0.75


def test_median_exactly_at_the_bar_is_safe():
    v = decide_lambda_mu_grid(_grid({(0.90, -0.90): {"R": 0.040, "gap": MED_GAP_MIN, "frac": FRAC_MIN, "n_inst": 100, "n_ge": 95}}))
    assert v["safe_lam"] == 0.90 and v["safe_mu"] == -0.90


def test_tie_prefers_the_thicker_gap():
    v = decide_lambda_mu_grid(
        _grid(
            {
                (0.90, -0.60): {"R": 0.15015595428732964, "gap": 0.080, "frac": 1.0},
            }
        )
    )
    assert v["verdict"] == "PASS"
    assert v["safe_lam"] == 0.90 and v["safe_mu"] == -0.60


def test_missing_cell_needs_edit():
    rows = _grid()
    del rows[cell_key(1.00, -0.75)]
    v = decide_lambda_mu_grid(rows)
    assert v["verdict"] == "needs_edit"
    assert v["safe_lam"] is None


def test_anchor_drift_needs_edit():
    v = decide_lambda_mu_grid(_grid({(0.75, -0.50): {"R": 0.20, "gap": 0.065, "frac": 1.0}}))
    assert v["verdict"] == "needs_edit"
    assert "0.1502" in v["reason"] or "0.1501" in v["reason"]


def test_no_safe_cell_needs_edit_and_keeps_edge():
    overrides = {pair: {"gap": 0.03, "frac": 1.0} for pair in CELLS}
    v = decide_lambda_mu_grid(_grid(overrides))
    assert v["verdict"] == "needs_edit"
    assert v["safe_lam"] is None
    assert v["edge_lam"] == 0.75 and v["edge_mu"] == -0.90


def test_safe_loss_kills():
    # Published ratios stay put and miss the gap bar. One new cell clears it and loses to Fourier.
    overrides = {}
    for pair, published in PUBLISHED_R.items():
        overrides[pair] = {"R": published, "gap": 0.01, "frac": 0.5, "n_ge": 32}
    for pair in CELLS:
        if pair in PUBLISHED_R or pair == (1.00, -0.60):
            continue
        overrides[pair] = {"R": 0.20, "gap": 0.01, "frac": 0.5, "n_ge": 32}
    overrides[(1.00, -0.60)] = {"R": 1.05, "gap": 0.08, "frac": 1.0, "seeds": [1.05, 1.05, 1.05]}
    v = decide_lambda_mu_grid(_grid(overrides))
    assert v["verdict"] == "KILL"
    assert v["safe_lam"] == 1.00 and v["safe_mu"] == -0.60


def test_bitstring_is_ineligible():
    v = decide_lambda_mu_grid(
        _grid({(1.00, -0.90): {"R": 0.010, "gap": 0.080, "frac": 1.0, "peak": 0.80}})
    )
    assert v["verdict"] == "PASS"
    assert v["safe_lam"] == 0.75 and v["edge_lam"] == 0.75
    assert v["edge_mu"] == -0.90


def test_seed_gate_and_n15_flip_need_edit():
    seeds = decide_lambda_mu_grid(
        _grid({(0.75, -0.50): {"seeds": [0.10, 1.05, 1.05]}})
    )
    assert seeds["verdict"] == "needs_edit"
    assert "seeds" in seeds["reason"]
    flip = decide_lambda_mu_grid(_grid({(0.75, -0.50): {"r15": 1.10}}))
    assert flip["verdict"] == "needs_edit"
    assert "n=15" in flip["reason"]


def test_gap_stats_counts_the_floor():
    stats = gap_stats([0.02, 0.019, 0.05, GAP_FLOOR])
    assert stats["n_inst"] == 4
    assert stats["n_gap_ge"] == 3
    assert stats["gap_min"] == 0.019
    assert abs(stats["frac_gap_ge"] - 0.75) < 1e-12
