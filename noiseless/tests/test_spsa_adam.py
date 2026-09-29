"""Tests for optimizer='spsa_adam' (SPSA gradient estimate + Adam update)."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest

from noiseless.encoding import list_four_sat_npz, load_four_sat_npz
from noiseless.spsa_gibbs import (
    ADAM_LR,
    NoiselessSimulator,
    ground_flat_from_bitstring,
    grow_trial,
    optimize_trial,
    run_spsa,
    run_spsa_adam,
)
from noiseless.unitaries import build_fixed_u

_REPO = Path(__file__).resolve().parents[2]
_HAMS = list_four_sat_npz(_REPO / "Hamiltonians" / "four_sat")
needs_hams = pytest.mark.skipif(not _HAMS, reason="no four_sat Hamiltonians")


def _sim(L: int, idx: int = 0) -> NoiselessSimulator:
    inst = load_four_sat_npz(_HAMS[idx])
    return NoiselessSimulator(
        u_fixed=build_fixed_u("jp"),
        energy_tensor=inst["energy_tensor"],
        n_layers=L,
        ground_bitstring=inst["ground_bitstring"],
        ground_flat_index=ground_flat_from_bitstring(inst["ground_bitstring"]),
    )


def _quad(x: np.ndarray) -> float:
    return float(np.sum((x - 0.3) ** 2))


def _reference_spsa_pre_adam(fun, x0, *, maxiter, rng, a=0.2, c=0.15, A=10.0,
                             alpha=0.602, gamma=0.101, on_before_step=None):
    """Verbatim copy of run_spsa as of commit 2c59b35 (before spsa_adam)."""
    x = np.asarray(x0, dtype=float).copy()
    nfev = 0
    for k in range(1, int(maxiter) + 1):
        if on_before_step is not None:
            on_before_step(k, x)
        ak = a / (k + A) ** alpha
        ck = c / k**gamma
        delta = rng.choice([-1.0, 1.0], size=x.size)
        yp = float(fun(x + ck * delta))
        ym = float(fun(x - ck * delta))
        nfev += 2
        ghat = (yp - ym) / (2.0 * ck) * delta
        x = x - ak * ghat
    return x, float(fun(x)), nfev + 1


def test_run_spsa_unchanged_vs_reference_loop():
    x0 = np.linspace(-1, 1, 16)
    r1 = run_spsa(_quad, x0, maxiter=50, rng=np.random.default_rng(3), a=0.1)
    r2 = _reference_spsa_pre_adam(_quad, x0, maxiter=50, rng=np.random.default_rng(3), a=0.1)
    assert np.array_equal(r1[0], r2[0]) and r1[1] == r2[1] and r1[2] == r2[2]


def test_run_spsa_adam_eval_count_and_rng_stream():
    x0 = np.linspace(-1, 1, 16)
    calls = {"n": 0}

    def f(x):
        calls["n"] += 1
        return _quad(x)

    rng_a, rng_s = np.random.default_rng(5), np.random.default_rng(5)
    x, fx, nfev = run_spsa_adam(f, x0, maxiter=37, rng=rng_a)
    assert nfev == calls["n"] == 2 * 37 + 1
    _, _, nfev_s = run_spsa(_quad, x0, maxiter=37, rng=rng_s)
    assert nfev == nfev_s
    # both consumed the rng identically (same Rademacher draws per step)
    assert rng_a.random() == rng_s.random()
    assert fx < _quad(x0)  # descends on a quadratic


def test_run_spsa_adam_same_perturbations_as_spsa():
    x0 = np.linspace(-1, 1, 12)
    pts_a: list[np.ndarray] = []
    pts_s: list[np.ndarray] = []
    run_spsa_adam(lambda x: pts_a.append(x.copy()) or _quad(x), x0, maxiter=10,
                  rng=np.random.default_rng(9))
    run_spsa(lambda x: pts_s.append(x.copy()) or _quad(x), x0, maxiter=10,
             rng=np.random.default_rng(9))
    # step 1 evaluates x0 ± c1·δ1 in both
    assert np.array_equal(pts_a[0], pts_s[0]) and np.array_equal(pts_a[1], pts_s[1])
    # every step's direction δ_k and c_k agree: (x+ - x-) identical
    for k in range(10):
        da = pts_a[2 * k] - pts_a[2 * k + 1]
        ds = pts_s[2 * k] - pts_s[2 * k + 1]
        np.testing.assert_allclose(da, ds, rtol=0, atol=1e-12)


def test_run_spsa_adam_first_step_is_lr_sign():
    # With bias correction, step 1 moves every coordinate by lr·sign(g) (up to eps).
    x0 = np.linspace(-1, 1, 8)
    x1, _, _ = run_spsa_adam(_quad, x0, maxiter=1, rng=np.random.default_rng(1), lr=0.05)
    np.testing.assert_allclose(np.abs(x1 - x0), 0.05, rtol=1e-6)


@needs_hams
def test_optimize_trial_spsa_adam_runs_and_deterministic():
    r1 = optimize_trial(_sim(2), maxiter=15, rng=np.random.default_rng(42), optimizer="spsa_adam")
    r2 = optimize_trial(_sim(2), maxiter=15, rng=np.random.default_rng(42), optimizer="spsa_adam")
    assert r1.optimizer == "spsa_adam"
    assert np.array_equal(r1.x, r2.x) and r1.fun == r2.fun and r1.p_gs == r2.p_gs
    assert r1.nfev == 2 * 15 + 1
    rs = optimize_trial(_sim(2), maxiter=15, rng=np.random.default_rng(42))
    assert rs.nfev == r1.nfev and rs.optimizer == "spsa"
    assert not np.array_equal(rs.x, r1.x)
    r3 = optimize_trial(_sim(2), maxiter=15, rng=np.random.default_rng(42), optimizer="spsa_adam",
                        adam_lr=0.1)
    assert not np.array_equal(r3.x, r1.x)
    with pytest.raises(ValueError):
        optimize_trial(_sim(2), maxiter=2, optimizer="adam")


@needs_hams
def test_spsa_adam_eta_refresh_identical_schedule():
    s_a, s_s = _sim(2), _sim(2)
    optimize_trial(s_a, maxiter=23, rng=np.random.default_rng(1), optimizer="spsa_adam")
    optimize_trial(s_s, maxiter=23, rng=np.random.default_rng(1))
    assert [h["step"] for h in s_a.eta_ctrl.history] == [h["step"] for h in s_s.eta_ctrl.history]
    assert [h["step"] for h in s_a.eta_ctrl.history] == [1, 6, 11, 16, 21]
    # first refresh is from the same x0, so identical η
    assert s_a.eta_ctrl.history[0]["eta"] == s_s.eta_ctrl.history[0]["eta"]


@needs_hams
def test_default_spsa_bit_for_bit_golden():
    """Hashes recorded from commit 2c59b35 (pre-spsa_adam) on four_sat_000, jp."""
    r = optimize_trial(_sim(2), maxiter=20, rng=np.random.default_rng(20260917))
    assert r.nfev == 41 and r.fun == 7.390778594072803
    assert hashlib.sha256(r.x.tobytes()).hexdigest() == (
        "c7d10cfb6b0fb9a467e60266068334a409bde5a80bfa93c3621ea658368d90bf"
    )
    g = grow_trial(_sim(1), final_layers=2, total_steps=999, steps_per_stage=6,
                   rng=np.random.default_rng(11))
    assert g.nfev == 26 and g.fun == 4.087035548507408
    assert hashlib.sha256(g.x.tobytes()).hexdigest() == (
        "605302198ce02077bd4029cedd3f0c9ca9efc5c47d6e91c20c564c6b823e50ab"
    )


@needs_hams
def test_grow_trial_spsa_adam():
    g1 = grow_trial(_sim(1), final_layers=3, total_steps=0, steps_per_stage=5,
                    rng=np.random.default_rng(8), optimizer="spsa_adam")
    g2 = grow_trial(_sim(1), final_layers=3, total_steps=0, steps_per_stage=5,
                    rng=np.random.default_rng(8), optimizer="spsa_adam")
    assert g1.optimizer == "spsa_adam"
    assert np.array_equal(g1.x, g2.x) and g1.stages == g2.stages
    assert [s["nfev"] for s in g1.stages] == [11, 11, 11] and g1.nfev == 33
    gs = grow_trial(_sim(1), final_layers=3, total_steps=0, steps_per_stage=5,
                    rng=np.random.default_rng(8))
    assert gs.nfev == g1.nfev
    for prev, cur in zip(g1.stages[:-1], g1.stages[1:]):
        assert abs(cur["p_gs_after_insert"] - prev["p_gs"]) < 1e-12


@needs_hams
def test_runner_worker_and_cli_spsa_adam(tmp_path):
    from noiseless.run_u_sweep import _worker, main

    base = dict(ham_path=str(_HAMS[0]), ham_file=_HAMS[0].name, u_name="jp", n_layers=2,
                trial=0, seed=5, steps=6, spsa_a=None)
    r = _worker(dict(base, optimizer="spsa_adam", adam_lr=0.02))
    assert r["ok"], r.get("traceback")
    assert r["optimizer"] == "spsa_adam" and r["adam_lr"] == 0.02 and r["nfev"] == 13
    r_def = _worker(dict(base, optimizer="spsa_adam"))
    assert r_def["adam_lr"] == ADAM_LR and r_def["x"] != r["x"]
    r_s = _worker(dict(base))
    assert r_s["optimizer"] == "spsa" and r_s["adam_lr"] is None and r_s["nfev"] == 13
    rg = _worker(dict(base, optimizer="spsa_adam", grow=True, grow_steps_per_stage=3))
    assert rg["ok"], rg.get("traceback")
    assert rg["nfev"] == 14 and [s["steps"] for s in rg["stages"]] == [3, 3]
    rc = main(["--u-names", "jp", "--layers", "2", "--trials", "1", "--steps", "3",
               "--max-h", "1", "--workers", "1", "--optimizer", "spsa_adam",
               "--adam-lr", "0.1", "--outdir", str(tmp_path), "--tag", "t"])
    assert rc == 0
    summ = list(tmp_path.glob("t_*_summary.json"))
    assert summ
    import json
    d = json.loads(summ[0].read_text())
    assert d["optimizer"] == "spsa_adam" and d["args"]["adam_lr"] == 0.1
