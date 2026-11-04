"""Honesty gate and B=2 shell for the NEW_SAFE depth sweep. No training."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_new_safe_B2_L_sweep.py"
_SPEC = importlib.util.spec_from_file_location("new_safe_B2_L_sweep", _PATH)
mod = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(mod)


def test_new_safe_coefficients():
    assert mod.LAM == 0.75
    assert mod.MU == 0.0
    assert mod.LAMBDA_X == -0.75
    assert mod.MU_XX == 0.0
    assert mod.DEPTHS == (2, 4, 8, 16)
    assert mod.SEEDS == (0, 1, 2, 3, 4)
    assert 21 not in mod.DEPTHS
    assert mod.METHODS == ("fourier", "B2")


def test_refuses_n21():
    try:
        mod.load_pack(21)
    except RuntimeError as exc:
        assert "21" in str(exc)
    else:
        raise AssertionError("load_pack(21) should refuse")


def test_b2_contains_paper_shell():
    k = np.arange(4, dtype=np.float64)
    paper = -0.01 * k
    recon = (paper @ mod.B2_BASIS.T) @ mod.B2_BASIS
    err = float(np.max(np.abs(recon - paper)))
    assert err < 1e-15
    assert mod.B2_RECON_ERROR < 1e-15
    # Two orthonormal rows, both inside span{k, k^2}.
    assert mod.B2_BASIS.shape == (2, 4)
    gram = mod.B2_BASIS @ mod.B2_BASIS.T
    assert np.allclose(gram, np.eye(2), atol=1e-12)


def test_honesty_gate():
    assert mod.honesty_verdict(0.62, 1.9, 2.2) == "PASS"
    assert mod.honesty_verdict(0.85, 1.0, 1.0) == "PASS"
    # L=2 softens.
    assert mod.honesty_verdict(0.851, 1.9, 2.2) == "needs_edit"
    # B=2 wins deep at either depth.
    assert mod.honesty_verdict(0.62, 0.80, 2.2) == "needs_edit"
    assert mod.honesty_verdict(0.62, 1.9, 0.99) == "needs_edit"
    assert mod.honesty_verdict(0.62, 0.999, 1.5) == "needs_edit"
    reasons = mod.honesty_reasons(0.90, 0.70, 1.4)
    assert any("softens" in r for r in reasons)
    assert any("L=8" in r for r in reasons)
    assert not any("L=16" in r and "wins deep" in r for r in reasons)


def test_depth_call_boundaries():
    assert mod.depth_call(0.62) == "win"
    assert mod.depth_call(0.85) == "win"
    assert mod.depth_call(0.851) == "soft"
    assert mod.depth_call(0.979) == "soft"
    assert mod.depth_call(0.98) == "tie"
    assert mod.depth_call(1.0) == "tie"
    assert mod.depth_call(1.02) == "tie"
    assert mod.depth_call(1.021) == "loss"
    assert mod.depth_call(float("nan")) == "undefined"


def test_median_of_five_seed_medians():
    rows = []
    snaps = (0.40, 0.50, 0.60, 0.70, 0.80)
    fours = (0.80, 0.70, 0.90, 1.00, 1.10)
    for seed, snap, four in zip(range(5), snaps, fours):
        for method, med in (("B2", snap), ("fourier", four)):
            rows.append(
                {
                    "method": method,
                    "seed": seed,
                    "best_round": 0,
                    "wall_s": 1.0,
                    "eval": {"12": {"med_eps": med, "eps": [med, med + 0.1]}},
                }
            )
    block = mod._block(rows, "12")
    assert block["med_b2"] == 0.60
    assert block["med_fourier"] == 0.90
    assert np.isclose(block["r"], 0.60 / 0.90)
    assert block["call"] == "win"
    assert block["n_seeds"] == 5
    assert block["paired_b2_below"] == 10


def test_parse_jobs_rejects_n21_depth():
    try:
        mod._parse_jobs("21", "0", "B2")
    except SystemExit:
        return
    raise AssertionError("depth 21 should be rejected")
