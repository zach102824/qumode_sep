"""N=15 mixer-freeze: split, stitch, 4-D ledger, no test peek."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np

from Nov3_INIT.alloc import iterations_equal4, n_params, run_equal4, x0_center4, x0_qaoa_center4
from Nov3_INIT.n15mix import TEST, TRAIN, build_freeze, run_mix_equal4, sigmas_free, split_free, stitch, x0_free
from Nov9_SK_SPSA.experiments.FT.instance import packing

ROOT = Path(__file__).resolve().parent
PREREG = ROOT / "PREREG_N15MIX.md"
OUT = ROOT / "results" / "n15mix" / "TESTS.json"
R: list[dict] = []


def check(name: str, err: float, tol: float) -> None:
    ok = err <= tol
    R.append({"name": name, "err": float(err), "tol": float(tol), "pass": bool(ok)})
    print(f"{name:48s} {err:.3e}  tol {tol}  {'PASS' if ok else 'FAIL'}")


def main() -> None:
    text = PREREG.read_text()
    ids = list(range(10888, 10904))
    check("T_seeds", 0.0 if list(TRAIN) + list(TEST) == ids else 1.0, 0.0)
    check("T_split", 0.0 if (
        list(TRAIN) == list(range(10888, 10896))
        and list(TEST) == list(range(10896, 10904))
        and "do not peek test before freeze" in text.lower()
    ) else 1.0, 0.0)
    forbidden = (
        set(range(10872, 10888)) | set(range(10856, 10872)) | set(range(10840, 10856))
        | set(range(10824, 10840)) | set(range(10808, 10824)) | set(range(10792, 10808))
        | set(range(10776, 10792)) | set(range(10760, 10776)) | set(range(10744, 10760))
        | set(range(10728, 10744)) | set(range(10712, 10728)) | set(range(10696, 10712))
        | set(range(10664, 10880)) & set(range(10664, 10680))
        | set(range(10624, 10632)) | set(range(10560, 10568)) | set(range(10568, 10584))
        | set(range(10552, 10560)) | set(range(10144, 10152))
        | set(range(11000, 11032)) | set(range(20000, 21000))
        | set(range(10904, 10920))
    )
    check("T_not_reserved", 0.0 if not (set(ids) & forbidden) else 1.0, 0.0)
    check("T_pack15", 0.0 if packing(15) == [[a, a + 5, a + 10] for a in range(5)] else 1.0, 0.0)
    check("T_T122", abs(iterations_equal4(1024) - 122), 0.0)
    check("T_nparam", abs(n_params("Nov3rd", 1) - 6) + abs(n_params("QAOA", 1) - 2), 0.0)
    check("T_primary_mixer", 0.0 if "MIXER_FREEZE_NEGATIVE_CI" in text and "two-template" in text.lower() else 1.0, 0.0)
    check("T_not_closed_copies", 0.0 if "two-template / better-of-two" in text.lower() and "unsigned-mean copy" in text.lower() else 1.0, 0.0)
    check("T_no_l3", 0.0 if "q=4096" in text else 1.0, 0.0)

    x6 = x0_center4("Nov3rd", 1, 0, None)
    xf = split_free(x6)
    xs = stitch(xf, 0.11, -0.22)
    check("T_center_slice", float(np.max(np.abs(xf - np.array([0.25, -0.5, 0.0, 0.0])))), 0.0)
    check("T_stitch_roundtrip", float(np.max(np.abs(xs - np.array([0.25, 0.11, -0.22, -0.5, 0.0, 0.0])))), 0.0)
    check("T_sig_free", float(np.max(np.abs(sigmas_free() - np.array([0.10, 0.15, 0.15, 0.15])))), 0.0)
    check("T_x0_free_s0", float(np.max(np.abs(x0_free(0, None) - np.array([0.25, -0.5, 0.0, 0.0])))), 0.0)

    ncall = {"n": 0}

    def dummy_full(x, ncall=ncall):
        ncall["n"] += 1
        x = np.asarray(x, float)
        return float(np.sum(np.square(x))), np.asarray(x, float).copy()

    ncall["n"] = 0
    init = [None if s == 0 else np.random.default_rng(20100 + s) for s in range(4)]
    spsa = [np.random.default_rng(20200 + s) for s in range(4)]
    out = run_mix_equal4(dummy_full, 0.12, -0.08, 1024, init, spsa)
    q = out["queries"]
    check("T_mix_ledger_Q", abs(q["total"] - 1024) + abs(ncall["n"] - 1024), 0.0)
    check("T_mix_n_iter", abs(out["n_iter"] - 122), 0.0)
    check("T_mix_cats", abs(q["init"] - 4) + abs(q["calibration"] - 32) + abs(q["iteration"] - 976) + abs(q["candidate"] - 12), 0.0)
    check("T_mix_locked_mixer", abs(out["x"][1] - 0.12) + abs(out["x"][2] + 0.08), 1e-12)

    init6 = [None if s == 0 else np.random.default_rng(20300 + s) for s in range(4)]
    spsa6 = [np.random.default_rng(20400 + s) for s in range(4)]
    ncall["n"] = 0
    out6 = run_equal4(dummy_full, "Nov3rd", 1, 1024, init6, spsa6, x0_fn=x0_center4)
    check("T_full_ledger_Q", abs(out6["queries"]["total"] - 1024), 0.0)
    xq = x0_qaoa_center4(1, 0, None)
    check("T_qaoa_center_L1", float(np.max(np.abs(xq - np.array([0.25, 0.25])))), 0.0)

    rng = np.random.default_rng(2)
    rows = []
    for s in TRAIN:
        xn_s = rng.normal(size=6)
        xn_s[1] = 0.4 + 0.01 * rng.normal()
        xn_s[2] = -0.05 + 0.01 * rng.normal()
        rows.append({"seed": int(s), "arm": "nov3_center4", "ok": True, "split": "train", "x": xn_s.tolist()})
    fr = build_freeze(rows, "deadbeef")
    check("T_freeze_mixer_only", 0.0 if (
        abs(fr["theta_K_mean"] - float(np.mean([r["x"][1] for r in rows]))) < 1e-12
        and fr["frozen_names"] == ["theta_K", "theta_M"]
        and fr["primary_gate"] == "MIXER_FREEZE_NEGATIVE_CI"
        and fr["not_two_template_copy"]
    ) else 1.0, 0.0)
    leaked = False
    try:
        bad = list(rows) + [{"seed": 10896, "arm": "nov3_center4", "ok": True, "split": "test", "x": [0.0] * 6}]
        build_freeze(bad, "deadbeef")
        leaked = True
    except ValueError:
        leaked = False
    check("T_no_test_peek", 0.0 if not leaked else 1.0, 0.0)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    allpass = all(r["pass"] for r in R)
    OUT.write_text(json.dumps({"n": len(R), "pass": allpass, "rows": R, "prereg_sha256": hashlib.sha256(PREREG.read_bytes()).hexdigest()}, indent=2) + "\n")
    print(f"{'ALL PASS' if allpass else 'FAIL'} n={len(R)}")
    if not allpass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
