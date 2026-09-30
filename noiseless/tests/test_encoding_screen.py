"""Encoding screen driver: identity reproduces run_u_sweep tuned default; resumable JSONL."""

from __future__ import annotations

import json

import numpy as np

from noiseless.tests.test_spsa_adam import _HAMS, needs_hams


@needs_hams
def test_identity_class_reproduces_tuned_default_trial():
    from noiseless.run_encoding_screen import init_seed, run_trial
    from noiseless.run_u_sweep import _worker

    seed = init_seed(20260917, 0, 1)
    r = run_trial({"ham_path": str(_HAMS[0]), "ham_file": _HAMS[0].name, "ham_index": 0,
                   "class_idx": 0, "perm": list(range(8)), "init": 1, "seed": seed})
    w = _worker({"ham_path": str(_HAMS[0]), "ham_file": _HAMS[0].name, "u_name": "jp",
                 "n_layers": 4, "trial": 1, "seed": seed, "steps": 800, "spsa_a": None,
                 "optimizer": "spsa_adam", "grow": True,
                 "grow_lr_schedule": [0.5, 0.2, 0.05, 0.02]})
    assert w["ok"], w.get("traceback")
    assert r["p_gs"] == w["p_gs"] and r["nfev"] == w["nfev"] == 1604
    assert r["most_likely_bitstring"] == w["most_likely_bitstring"]


def test_select_assignments():
    from noiseless.run_encoding_screen import select_assignments

    s = select_assignments("smoke", 9, 20260917)
    assert s[0] == 0 and len(set(s)) == 10 and all(0 < v < 20160 for v in s[1:])
    assert s == select_assignments("smoke", 9, 20260917)
    assert select_assignments("range:3:6", 0, 0) == [3, 4, 5]
    assert select_assignments("1,7", 0, 0) == [1, 7]


@needs_hams
def test_screen_resume(tmp_path, monkeypatch):
    import noiseless.run_encoding_screen as scr

    calls = []

    def fake(job):  # short trials: patch the growth budget via a tiny stand-in
        calls.append((job["class_idx"], job["init"]))
        return {**job, "p_gs": 0.1 * job["init"], "success": True, "energy_mean": 1.0,
                "fun": 0.0, "nfev": 0, "most_likely_bitstring": "", "ground_bitstring": "",
                "ground_energy": 0.0, "stage_p_gs": [], "wall_s": 0.01}

    monkeypatch.setattr(scr, "run_trial", fake)
    out = tmp_path / "s.jsonl"
    argv = ["--assignments", "0,5", "--inits", "2", "--workers", "1", "--out", str(out)]
    assert scr.main(argv) == 0 and len(calls) == 4
    # truncate to 3 lines (+ a broken partial line) and resume: only the missing trial reruns
    lines = out.read_text().splitlines()
    out.write_text("\n".join(lines[:3]) + "\n{\"partial")
    calls.clear()
    assert scr.main(argv) == 0 and len(calls) == 1
    summ = json.loads((tmp_path / "s_summary.json").read_text())
    assert summ["n_assignments"] == 2 and summ["n_trials"] == 4
    assert summ["identity"]["class_idx"] == 0
    assert np.isclose(summ["identity"]["mean_p_gs"], 0.05)
