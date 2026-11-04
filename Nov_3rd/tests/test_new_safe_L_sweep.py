"""Labels for the NEW_SAFE depth sweep. No training."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_new_safe_L_sweep.py"
_SPEC = importlib.util.spec_from_file_location("new_safe_L_sweep", _PATH)
mod = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(mod)


def test_new_safe_coefficients():
    assert mod.LAM == 0.75
    assert mod.MU == 0.0
    assert mod.LAMBDA_X == -0.75
    assert mod.MU_XX == 0.0
    assert 21 not in mod.DEPTH_SEEDS


def test_depth_call_boundaries():
    assert mod.depth_call(0.62) == "win"
    assert mod.depth_call(0.85) == "win"
    assert mod.depth_call(0.851) == "soft"
    assert mod.depth_call(0.979) == "soft"
    assert mod.depth_call(0.98) == "tie"
    assert mod.depth_call(1.02) == "tie"
    assert mod.depth_call(1.021) == "loss"
    assert mod.depth_call(float("nan")) == "undefined"


def test_pass_rule_matches_task():
    # The gate is L=2 R and Λ, both inclusive at the printed bars.
    assert mod.BAR_R == 0.85
    assert mod.BAR_LAMBDA == 0.80
    summary = {
        "verdict": "PASS",
        "L2_R_n12": 0.62,
        "lambda_n12": 0.748,
        "gs": {
            "6": {"lambda_med": 0.7612},
            "9": {"lambda_med": 0.7555},
            "12": {"lambda_med": 0.7477},
        },
        "depths": [
            {
                "L": 2,
                "by_n": {
                    "6": {"r": 0.614, "med_snap": 0.22, "med_fourier": 0.36},
                    "9": {"r": 0.631, "med_snap": 0.30, "med_fourier": 0.47},
                    "12": {
                        "r": 0.6201,
                        "call": "win",
                        "seeds_r_le_085": 5,
                        "n_seeds": 5,
                        "med_snap": 0.4264,
                        "med_fourier": 0.6876,
                        "paired_snap_below": 640,
                        "paired_n": 640,
                    },
                },
            },
            {
                "L": 8,
                "by_n": {
                    "12": {
                        "r": 1.923,
                        "call": "loss",
                        "seeds_r_le_085": 0,
                        "n_seeds": 3,
                        "med_snap": 0.226,
                        "med_fourier": 0.117,
                        "paired_snap_below": 0,
                        "paired_n": 384,
                    }
                },
            },
        ],
    }
    note = mod._paper_note(summary)
    assert "R=0.620" in note
    assert "not a deep win" in note
    assert "0.748" in note
    assert "L=8 R=1.923" in note


def test_median_ratio_uses_numpy_median():
    rows = []
    for seed, snap, four in ((0, 0.40, 0.80), (1, 0.50, 0.70), (2, 0.60, 0.90)):
        for method, med in (("snap", snap), ("fourier", four)):
            rows.append(
                {
                    "method": method,
                    "seed": seed,
                    "best_round": 0,
                    "wall_s": 1.0,
                    "eval": {"12": {"med_eps": med, "eps": [med]}},
                }
            )
    block = mod._block(rows, "12")
    assert block["med_snap"] == 0.50
    assert block["med_fourier"] == 0.80
    assert np.isclose(block["r"], 0.625)
    assert block["call"] == "win"
    assert block["paired_snap_below"] == 3
