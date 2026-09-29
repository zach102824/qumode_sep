"""Tests for per-stage growth schedules (steps / Adam lr / η scale / SPSA c) and start depth."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from noiseless.spsa_gibbs import gibbs_objective, grow_trial
from noiseless.tests.test_spsa_adam import _HAMS, _sim, needs_hams


def _h(x) -> str:
    return hashlib.sha256(np.asarray(x).tobytes()).hexdigest()


# Recorded with commit 28f9ec6 (before per-stage schedules) on four_sat_000/001, jp.
GOLDEN = {
    ("spsa_adam", 0): ("3c419d6e15ae02d534a6dd0e92ee39355a0c59ecb9a4452ed693877e064bf9f2", 9.330941951309149e-06),
    ("spsa_adam", 1): ("077fa8b0d5d2ad4534c67a6ab4dd2ba1dd10303bbd4901b77cc77a132b319e24", 0.01267296861422559),
    ("spsa", 0): ("10ab809b58fe7726db054913cafa786d98f31e48aa9332983cc9bb9b013e9f8a", 0.16386425122193143),
    ("spsa", 1): ("11fbcf9429070ce1afcb1798e039ef1c0c215fb5667495afcee9d379991e6cbf", 0.12363084975670936),
}


def _g0(opt, **kw):
    return grow_trial(_sim(1, 0), final_layers=3, total_steps=30,
                      rng=np.random.default_rng(123), optimizer=opt, **kw)


def _g1(opt, **kw):
    return grow_trial(_sim(1, 1), final_layers=4, total_steps=40, rng=np.random.default_rng(7),
                      optimizer=opt, steps_per_stage=10, start_layers=2, kick_sigma=0.2, **kw)


@needs_hams
@pytest.mark.parametrize("opt", ["spsa_adam", "spsa"])
def test_defaults_bit_for_bit_vs_28f9ec6(opt):
    for i, g in enumerate((_g0(opt), _g1(opt))):
        assert g.nfev == 63
        assert (_h(g.x), g.p_gs) == GOLDEN[(opt, i)]


@needs_hams
@pytest.mark.parametrize("opt", ["spsa_adam", "spsa"])
def test_explicit_default_schedules_identical(opt):
    """Schedules equal to the defaults reproduce the legacy run exactly."""
    g = _g0(opt, steps_schedule=[10, 10, 10], lr_schedule=[0.05] * 3,
            eta_scale_schedule=[1.0] * 3, c_schedule=[0.15] * 3)
    assert (_h(g.x), g.p_gs) == GOLDEN[(opt, 0)]
    g = _g1(opt, steps_schedule=[10, 10, 10], eta_scale_schedule=[1, 1, 1])
    assert (_h(g.x), g.p_gs) == GOLDEN[(opt, 1)]


@needs_hams
def test_steps_schedule_counts_and_eval_budget():
    g = grow_trial(_sim(1), final_layers=4, rng=np.random.default_rng(3),
                   optimizer="spsa_adam", steps_schedule=[2, 3, 5, 6])
    assert [s["steps"] for s in g.stages] == [2, 3, 5, 6]
    assert [s["nfev"] for s in g.stages] == [5, 7, 11, 13]
    assert g.nfev == 2 * 16 + 4 and g.nit == 16
    # start 2 → 3 stages; same step total gives 2*16 + 3 evals
    g2 = grow_trial(_sim(1), final_layers=4, rng=np.random.default_rng(3), start_layers=2,
                    optimizer="spsa_adam", steps_schedule=[5, 5, 6])
    assert [s["n_layers"] for s in g2.stages] == [2, 3, 4] and g2.nfev == 35


@needs_hams
def test_schedule_length_mismatch_raises():
    with pytest.raises(ValueError):
        grow_trial(_sim(1), final_layers=3, rng=np.random.default_rng(0), steps_schedule=[1, 2])
    with pytest.raises(ValueError):
        grow_trial(_sim(1), final_layers=3, rng=np.random.default_rng(0), total_steps=6,
                   lr_schedule=[0.1] * 4)


@needs_hams
def test_lr_and_c_schedules_applied_per_stage():
    g = _g0("spsa_adam", lr_schedule=[0.05, 0.05, 0.2], c_schedule=[0.15, 0.15, 0.05])
    assert [s["adam_lr"] for s in g.stages] == [0.05, 0.05, 0.2]
    assert [s["spsa_c"] for s in g.stages] == [0.15, 0.15, 0.05]
    ref = _g0("spsa_adam")
    # first two stages identical to the default run, last differs
    assert g.stages[:2] == ref.stages[:2]
    assert _h(g.x) != _h(ref.x)


@needs_hams
def test_eta_scale_multiplies_controller_eta_and_restores():
    sim = _sim(1)
    captured = []
    orig = sim.refresh_eta

    def spy(x, step):
        v = orig(x, step)
        captured.append((sim.n_layers, v, sim.eta_ctrl.eta))
        return v

    sim.refresh_eta = spy
    grow_trial(sim, final_layers=3, total_steps=30, rng=np.random.default_rng(1),
               optimizer="spsa_adam", eta_scale_schedule=[0.5, 1.0, 2.0])
    scale = {1: 0.5, 2: 1.0, 3: 2.0}
    assert captured
    for L, used, ctrl in captured:
        assert used == pytest.approx(scale[L] * ctrl, rel=0, abs=1e-15)
    assert sim.eta_scale == 1.0


def test_eta_is_inverse_temperature():
    """Small η → cost ≈ η·<E> (hot, mean energy); large η → cost ≈ -log p(GS) + η·Emin (cold)."""
    e = np.array([0.0, 1.0, 2.0])
    p = np.array([0.2, 0.5, 0.3])
    eta = 1e-6
    assert gibbs_objective(p, e, eta) == pytest.approx(eta * float(p @ e), rel=1e-4)
    assert gibbs_objective(p, e, 60.0) == pytest.approx(-np.log(0.2), rel=1e-6)


@needs_hams
def test_runner_cli_schedules(tmp_path):
    from noiseless.run_u_sweep import _worker, main

    base = dict(ham_path=str(_HAMS[0]), ham_file=_HAMS[0].name, u_name="jp", n_layers=3,
                trial=0, seed=5, steps=6, spsa_a=None, optimizer="spsa_adam", grow=True)
    r = _worker(dict(base, grow_steps_schedule=[1, 2, 3], grow_lr_schedule=[0.1, 0.1, 0.03],
                     grow_eta_scale=[0.5, 0.7, 1.0], grow_c_schedule=[0.15, 0.1, 0.05]))
    assert r["ok"], r.get("traceback")
    assert [s["steps"] for s in r["stages"]] == [1, 2, 3] and r["nfev"] == 15
    assert [s["eta_scale"] for s in r["stages"]] == [0.5, 0.7, 1.0]
    # default job == explicit-default schedules
    r0 = _worker(dict(base))
    r1 = _worker(dict(base, grow_steps_schedule=[2, 2, 2], grow_lr_schedule=[0.05] * 3,
                      grow_eta_scale=[1.0] * 3, grow_c_schedule=[0.15] * 3))
    assert r0["x"] == r1["x"] and r0["p_gs"] == r1["p_gs"]
    rc = main(["--u-names", "jp", "--layers", "3", "--trials", "1", "--steps", "6", "--max-h", "1",
               "--workers", "1", "--optimizer", "spsa_adam", "--grow", "--grow-start", "2",
               "--grow-steps-schedule", "2,4", "--grow-lr-schedule", "0.1,0.03",
               "--grow-eta-scale", "0.5,1", "--grow-c-schedule", "0.15,0.05",
               "--outdir", str(tmp_path), "--tag", "t"])
    assert rc == 0
    d = json.loads(next(tmp_path.glob("t_*_summary.json")).read_text())
    assert d["args"]["grow_steps_schedule"] == [2, 4]
    assert d["args"]["grow_eta_scale"] == [0.5, 1.0]
    with pytest.raises(SystemExit):
        main(["--u-names", "jp", "--layers", "4", "--grow", "--grow-steps-schedule", "1,2",
              "--outdir", str(tmp_path)])
    with pytest.raises(SystemExit):
        main(["--u-names", "jp", "--layers", "4", "--grow-c-schedule", "0.1,0.1,0.1,0.1",
              "--outdir", str(tmp_path)])
