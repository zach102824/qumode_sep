"""Tests for BFGS η modes (fixed / restart) and BFGS layer growth."""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from noiseless.spsa_gibbs import grow_trial, optimize_trial
from noiseless.tests.test_spsa_adam import _HAMS, _sim, needs_hams


def _h(x) -> str:
    return hashlib.sha256(np.asarray(x).tobytes()).hexdigest()


@needs_hams
def test_legacy_bfgs_default_unchanged():
    """Default eta_mode='callback' reproduces the pre-change BFGS numerics (recorded at f1bbfd0)."""
    r = optimize_trial(_sim(4, 0), maxiter=15, rng=np.random.default_rng(5), optimizer="bfgs")
    assert (_h(r.x), r.p_gs, r.nfev, r.nit, r.opt_status) == (
        "145aeaefc1e88b8964d34bcec1ebe496dc203df203bbee99fb68bf4dd8b1f292",
        0.21054655810248737, 529, 15, 1)
    assert r.opt_info is None
    g = grow_trial(_sim(1, 0), final_layers=3, total_steps=30, rng=np.random.default_rng(123),
                   optimizer="bfgs")
    assert (_h(g.x), g.p_gs, g.nfev, g.nit) == (
        "1d87b141230a7fc5cc8ce1c4ab628a4d64736e37452472bf64ba9dfb04f2c53e",
        0.33902756697050956, 735, 30)


@needs_hams
def test_bfgs_fixed_eta_single_refresh():
    sim = _sim(2)
    r = optimize_trial(sim, maxiter=10, rng=np.random.default_rng(1), optimizer="bfgs",
                       bfgs_eta_mode="fixed")
    assert len(sim.eta_ctrl.history) == 1
    assert r.opt_info["n_restarts"] == 0 and len(r.opt_info["runs"]) == 1
    assert r.nit == r.opt_info["runs"][0]["nit"] <= 10
    assert r.nfev == r.opt_info["runs"][0]["nfev"] + 1  # + final eval
    assert r.eta == r.opt_info["etas"][0]


@needs_hams
def test_bfgs_restart_eta_bookkeeping():
    sim = _sim(2)
    r = optimize_trial(sim, maxiter=200, rng=np.random.default_rng(2), optimizer="bfgs",
                       bfgs_eta_mode="restart")
    info = r.opt_info
    runs = info["runs"]
    assert info["termination"] in ("eta_converged", "x_stationary", "maxiter", "max_restarts")
    assert r.nit == sum(u["nit"] for u in runs) <= 200
    assert r.nfev == sum(u["nfev"] for u in runs) + 1
    # every BFGS run used the η refreshed just before it; one refresh after each run
    assert [u["eta"] for u in runs] == info["etas"][: len(runs)]
    assert len(sim.eta_ctrl.history) == len(info["etas"])
    if info["termination"] == "eta_converged":
        assert abs(info["etas"][-1] - info["etas"][-2]) <= 1e-2 * info["etas"][-2]
    # deterministic
    r2 = optimize_trial(_sim(2), maxiter=200, rng=np.random.default_rng(2), optimizer="bfgs",
                        bfgs_eta_mode="restart")
    assert np.array_equal(r.x, r2.x) and r.nfev == r2.nfev


@needs_hams
def test_grow_bfgs_restart_stages():
    g = grow_trial(_sim(1), final_layers=3, rng=np.random.default_rng(3), optimizer="bfgs",
                   steps_per_stage=25, bfgs_eta_mode="restart")
    st = g.stages
    assert [s["n_layers"] for s in st] == [1, 2, 3]
    assert all(0 <= s["nit"] <= 25 for s in st)
    assert g.nit == sum(s["nit"] for s in st)
    assert g.nfev == sum(s["nfev"] for s in st)
    for s in st:
        assert s["wall_s"] > 0 and s["termination"] in ("eta_converged", "x_stationary", "maxiter", "max_restarts")
        assert s["etas"] and s["opt_message"].startswith(s["termination"])
    # warm start: appending the transparent layer leaves p(GS) unchanged
    for prev, cur in zip(st, st[1:]):
        assert cur["p_gs_after_insert"] == pytest.approx(prev["p_gs"], abs=1e-10)


@needs_hams
def test_bfgs_eta_mode_validation(tmp_path):
    with pytest.raises(ValueError):
        optimize_trial(_sim(1), maxiter=2, rng=np.random.default_rng(0), optimizer="bfgs",
                       bfgs_eta_mode="nope")
    from noiseless.run_u_sweep import main

    with pytest.raises(SystemExit):
        main(["--u-names", "jp", "--layers", "2", "--optimizer", "spsa", "--bfgs-eta-mode", "fixed",
              "--outdir", str(tmp_path)])
    rc = main(["--u-names", "jp", "--layers", "2", "--trials", "1", "--steps", "4", "--max-h", "1",
               "--workers", "1", "--optimizer", "bfgs", "--grow", "--bfgs-eta-mode", "restart",
               "--outdir", str(tmp_path), "--tag", "t"])
    assert rc == 0
