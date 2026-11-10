"""N=15 two-template: split, clusters, s1-only align, no test peek."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np

from Nov3_INIT.alloc import iterations_equal4, n_params, run_equal4, x0_center4, x0_qaoa_center4
from Nov3_INIT.n15tmpl import TEST, TRAIN, align_s1, build_freeze, pack_cluster
from Nov9_SK_SPSA.experiments.FT.instance import packing

ROOT = Path(__file__).resolve().parent
PREREG = ROOT / "PREREG_N15TMPL.md"
OUT = ROOT / "results" / "n15tmpl" / "TESTS.json"
R: list[dict] = []


def check(name: str, err: float, tol: float) -> None:
    ok = err <= tol
    R.append({"name": name, "err": float(err), "tol": float(tol), "pass": bool(ok)})
    print(f"{name:48s} {err:.3e}  tol {tol}  {'PASS' if ok else 'FAIL'}")


def main() -> None:
    text = PREREG.read_text()
    ids = list(range(10856, 10872))
    check("T_seeds", 0.0 if list(TRAIN) + list(TEST) == ids else 1.0, 0.0)
    check("T_split", 0.0 if (
        list(TRAIN) == list(range(10856, 10864))
        and list(TEST) == list(range(10864, 10872))
        and "do not peek test before freeze" in text.lower()
    ) else 1.0, 0.0)
    forbidden = (
        set(range(10712, 10728)) | set(range(10728, 10744)) | set(range(10744, 10760))
        | set(range(10760, 10776)) | set(range(10776, 10792)) | set(range(10792, 10808)) | set(range(10808, 10824)) | set(range(10824, 10840)) | set(range(10840, 10856)) | set(range(10624, 10632))
        | set(range(10560, 10568)) | set(range(10696, 10712)) | set(range(10664, 10680))
        | set(range(10680, 10696)) | set(range(10648, 10664)) | set(range(10600, 10616))
        | set(range(10616, 10624)) | set(range(10432, 10440)) | set(range(10552, 10560))
        | set(range(10536, 10552)) | set(range(10584, 10600)) | set(range(10504, 10520))
        | set(range(10520, 10536)) | set(range(10568, 10584)) | set(range(10480, 10488))
        | set(range(10456, 10464)) | set(range(10376, 10384)) | set(range(10368, 10376))
        | set(range(10632, 10648)) | set(range(10144, 10152))
        | set(range(11000, 11032)) | set(range(20000, 21000))
    )
    check("T_not_reserved", 0.0 if not (set(ids) & forbidden) else 1.0, 0.0)
    check("T_pack15", 0.0 if packing(15) == [[a, a + 5, a + 10] for a in range(5)] else 1.0, 0.0)
    check("T_T122", abs(iterations_equal4(1024) - 122), 0.0)
    check("T_nparam", abs(n_params("Nov3rd", 1) - 6) + abs(n_params("QAOA", 1) - 2), 0.0)
    check("T_primary_two_template", 0.0 if "TWO_TEMPLATE_TRANSFER_NEGATIVE_CI" in text and "better-of-two" in text else 1.0, 0.0)
    check("T_not_unsigned_copy", 0.0 if "unsigned-mean copy" in text.lower() and "sign-aligned-only copy" in text.lower() else 1.0, 0.0)
    check("T_no_l3", 0.0 if "q=4096" in text else 1.0, 0.0)

    ncall = {"n": 0}

    def dummy(x, ncall=ncall):
        ncall["n"] += 1
        return float(np.sum(np.square(x))), np.asarray(x, float).copy()

    init = [None if s == 0 else np.random.default_rng(10100 + s) for s in range(4)]
    spsa = [np.random.default_rng(10200 + s) for s in range(4)]
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

    y, flipped = align_s1(np.array([0.1, 0.2, 0.3, -0.9, 0.4, 0.5]))
    check("T_align_flips_s1_only", float(np.max(np.abs(y - np.array([0.1, 0.2, 0.3, 0.9, 0.4, 0.5])))) + (0.0 if flipped else 1.0), 0.0)
    y2, flipped2 = align_s1(np.array([0.1, 0.2, 0.3, 0.9, 0.4, 0.5]))
    check("T_align_pos_unchanged", float(np.max(np.abs(y2 - np.array([0.1, 0.2, 0.3, 0.9, 0.4, 0.5])))) + (0.0 if not flipped2 else 1.0), 0.0)

    empty = pack_cluster([10856], np.array([[0.0, 0.0, 0.0, -1.0, 0.0, 0.0]]), ("gamma", "theta_K", "theta_M", "s1", "s2", "s3"), 3, "plus")
    check("T_empty_cluster_no_mean", 0.0 if empty.get("empty") and "mean" not in empty else 1.0, 0.0)

    rng = np.random.default_rng(0)
    rows = []
    for i, s in enumerate(TRAIN):
        xn_s = rng.normal(size=6)
        xn_s[3] = 0.8 if i < 4 else -0.8
        xq_s = rng.normal(size=2)
        xq_s[1] = 0.4 if i < 5 else -0.4
        rows.append({"seed": int(s), "arm": "nov3_center4", "ok": True, "split": "train", "x": xn_s.tolist()})
        rows.append({"seed": int(s), "arm": "qaoa_center4", "ok": True, "split": "train", "x": xq_s.tolist()})
    fr = build_freeze(rows, "deadbeef")
    check("T_freeze_clusters", 0.0 if (
        not fr["nov3_cluster_plus"]["empty"] and fr["nov3_cluster_plus"]["n"] == 4
        and not fr["nov3_cluster_minus"]["empty"] and fr["nov3_cluster_minus"]["n"] == 4
        and fr["nov3_two_template"]["two_template"]
        and fr["qaoa_cluster_plus"]["n"] == 5
        and fr["qaoa_cluster_minus"]["n"] == 3
    ) else 1.0, 0.0)
    check("T_catalog_7", abs(fr["n_templates"] - 7) + abs(fr["planned_transfer_queries"] - 56), 0.0)
    check("T_primary_locked", 0.0 if fr["primary_gate"] == "TWO_TEMPLATE_TRANSFER_NEGATIVE_CI" and fr["do_not_switch_primary_after_test"] else 1.0, 0.0)
    leaked = False
    try:
        bad = list(rows) + [{"seed": 10864, "arm": "nov3_center4", "ok": True, "split": "test", "x": [0.0] * 6}]
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
