"""N=9 transfer: split, v4 L=1, exclusive seeds, no test peek."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np

from Nov3_INIT.alloc import iterations_equal4, n_params, run_equal4, x0_center4, x0_qaoa_center4
from Nov3_INIT.n9xfer import PR116_N15_TRAIN_STD, PR118_N12_TRAIN_STD, TEST, TRAIN, build_freeze
from Nov9_SK_SPSA.experiments.FT.instance import packing

ROOT = Path(__file__).resolve().parent
PREREG = ROOT / "PREREG_N9XFER.md"
OUT = ROOT / "results" / "n9xfer" / "TESTS.json"
R: list[dict] = []


def check(name: str, err: float, tol: float) -> None:
    ok = err <= tol
    R.append({"name": name, "err": float(err), "tol": float(tol), "pass": bool(ok)})
    print(f"{name:48s} {err:.3e}  tol {tol}  {'PASS' if ok else 'FAIL'}")


def main() -> None:
    text = PREREG.read_text()
    ids = list(range(10696, 10712))
    check("T_seeds", 0.0 if list(TRAIN) + list(TEST) == ids else 1.0, 0.0)
    check("T_split", 0.0 if (
        list(TRAIN) == list(range(10696, 10704))
        and list(TEST) == list(range(10704, 10712))
        and "do not peek test before freeze" in text.lower()
    ) else 1.0, 0.0)
    forbidden = (
        set(range(10432, 10440)) | set(range(10552, 10560)) | set(range(10600, 10616))
        | set(range(10648, 10664)) | set(range(10680, 10696)) | set(range(10536, 10552))
        | set(range(10584, 10600)) | set(range(10504, 10520)) | set(range(10632, 10648))
        | set(range(10520, 10536)) | set(range(10568, 10584)) | set(range(10480, 10488))
        | set(range(10456, 10464)) | set(range(10376, 10384)) | set(range(10368, 10376))
        | set(range(10616, 10624)) | set(range(10624, 10632)) | set(range(10560, 10568))
        | set(range(10664, 10680)) | set(range(10144, 10152))
        | set(range(11000, 11032)) | set(range(20000, 21000))
    )
    check("T_not_reserved", 0.0 if not (set(ids) & forbidden) else 1.0, 0.0)
    check("T_pack9", 0.0 if packing(9) == [[a, a + 3, a + 6] for a in range(3)] else 1.0, 0.0)
    check("T_T122", abs(iterations_equal4(1024) - 122), 0.0)
    check("T_nparam", abs(n_params("Nov3rd", 1) - 6) + abs(n_params("QAOA", 1) - 2), 0.0)
    check("T_primary_mean", 0.0 if "shared-mean" in text and "Do **not** switch the primary" in text else 1.0, 0.0)
    check("T_pr116_stds", 0.0 if PR116_N15_TRAIN_STD == {
        "theta_K": 0.0511, "theta_M": 0.0233, "s1": 0.8058, "beta": 0.4015,
    } else 1.0, 0.0)
    check("T_pr118_stds", 0.0 if PR118_N12_TRAIN_STD == {
        "theta_K": 0.0573, "theta_M": 0.0152, "s1": 0.8265, "beta": 0.5523,
    } else 1.0, 0.0)
    check("T_no_l3", 0.0 if "q=4096" in text else 1.0, 0.0)
    check("T_xfer_charge", 0.0 if "32800" in text and "=32" in text.replace(" ", "") else 1.0, 0.0)

    ncall = {"n": 0}

    def dummy(x, ncall=ncall):
        ncall["n"] += 1
        return float(np.sum(np.square(x))), np.asarray(x, float).copy()

    init = [None if s == 0 else np.random.default_rng(9100 + s) for s in range(4)]
    spsa = [np.random.default_rng(9200 + s) for s in range(4)]
    ncall["n"] = 0
    out = run_equal4(dummy, "Nov3rd", 1, 1024, init, spsa, x0_fn=x0_center4)
    q = out["queries"]
    check("T_ledger_Q", abs(q["total"] - 1024) + abs(ncall["n"] - 1024), 0.0)
    check("T_n_iter", abs(out["n_iter"] - 122), 0.0)
    check("T_cats", abs(q["init"] - 4) + abs(q["calibration"] - 32) + abs(q["iteration"] - 976) + abs(q["candidate"] - 12), 0.0)
    xn = x0_center4("Nov3rd", 1, 0, None)
    check("T_nov3_center_L1", float(np.max(np.abs(xn - np.array([0.25, 0.0, 0.0, -0.5, 0.0, 0.0])))), 0.0)
    xq = x0_qaoa_center4(1, 0, None)
    check("T_qaoa_center_L1", float(np.max(np.abs(xq - np.array([0.25, 0.25])))), 0.0)

    rng = np.random.default_rng(0)
    rows = []
    for s in TRAIN:
        rows.append({"seed": int(s), "arm": "nov3_center4", "ok": True, "split": "train", "x": rng.normal(size=6).tolist()})
        rows.append({"seed": int(s), "arm": "qaoa_center4", "ok": True, "split": "train", "x": rng.normal(size=2).tolist()})
    fr = build_freeze(rows, "deadbeef")
    check("T_freeze_n", abs(fr["nov3"]["n"] - 8) + abs(fr["qaoa"]["n"] - 8), 0.0)
    check("T_freeze_dim", abs(fr["nov3"]["dim"] - 6) + abs(fr["qaoa"]["dim"] - 2), 0.0)
    check("T_freeze_primary_mean", 0.0 if fr["primary_aggregator"] == "mean" else 1.0, 0.0)
    check("T_freeze_median_also", 0.0 if fr["also_frozen"] == "median" else 1.0, 0.0)
    leaked = False
    try:
        bad = list(rows) + [{"seed": 10704, "arm": "nov3_center4", "ok": True, "split": "test", "x": [0.0] * 6}]
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
