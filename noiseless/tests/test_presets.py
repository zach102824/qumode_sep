"""run_u_sweep presets: tuned growth + Adam is the default; --preset legacy restores old defaults."""

from __future__ import annotations

import json

from noiseless.tests.test_spsa_adam import needs_hams


def _run(tmp_path, tag, *argv):
    from noiseless.run_u_sweep import main

    rc = main([*argv, "--max-h", "1", "--trials", "1", "--workers", "1",
               "--outdir", str(tmp_path), "--tag", tag])
    assert rc == 0
    d = json.loads(next(tmp_path.glob(f"{tag}_2*Z.json")).read_text())
    return d["args"], d["records"]


@needs_hams
def test_tuned_default_equals_explicit_legacy_flags(tmp_path):
    a, r = _run(tmp_path, "tdef")
    assert a["preset"] == "tuned" and a["u_names"] == "jp" and a["layers"] == "4"
    assert a["optimizer"] == "spsa_adam" and a["grow"] is True
    assert a["grow_lr_schedule"] == [0.5, 0.2, 0.05, 0.02] and a["grow_steps_per_stage"] == 200
    assert len(r) == 1 and r[0]["nfev"] == 1604
    assert [s["adam_lr"] for s in r[0]["stages"]] == [0.5, 0.2, 0.05, 0.02]
    # the same run spelled out with the legacy preset (how the tuned fleet was launched)
    _, r2 = _run(tmp_path, "texp", "--preset", "legacy", "--u-names", "jp", "--layers", "4",
                 "--optimizer", "spsa_adam", "--grow", "--steps", "800",
                 "--grow-lr-schedule", "0.5,0.2,0.05,0.02")
    assert r2[0]["x"] == r[0]["x"] and r2[0]["p_gs"] == r[0]["p_gs"] and r2[0]["seed"] == r[0]["seed"]


@needs_hams
def test_legacy_preset_and_overrides(tmp_path):
    a, r = _run(tmp_path, "leg", "--preset", "legacy", "--u-names", "jp", "--layers", "2",
                "--steps", "3")
    assert a["optimizer"] == "spsa" and a["grow"] is False and a["grow_lr_schedule"] is None
    assert r[0]["nfev"] == 7 and r[0]["optimizer"] == "spsa"
    # explicit flags win over the tuned preset
    a, r = _run(tmp_path, "nog", "--no-grow", "--steps", "3")
    assert a["grow"] is False and a["optimizer"] == "spsa_adam" and r[0]["nfev"] == 7
    a, _ = _run(tmp_path, "alr", "--adam-lr", "0.1", "--grow-steps-per-stage", "2")
    assert a["grow"] is True and a["grow_lr_schedule"] is None and a["adam_lr"] == 0.1
    # --steps given → total budget split (no 200/stage override)
    a, r = _run(tmp_path, "stp", "--steps", "8")
    assert a["grow_steps_per_stage"] is None and r[0]["nfev"] == 4 * 5
    # smoke keeps its legacy meaning
    from noiseless.run_u_sweep import main
    assert main(["--smoke", "--workers", "1", "--outdir", str(tmp_path)]) == 0
    d = json.loads(next(tmp_path.glob("smoke_2*Z.json")).read_text())
    assert d["args"]["preset"] == "legacy" and d["args"]["optimizer"] == "spsa"
