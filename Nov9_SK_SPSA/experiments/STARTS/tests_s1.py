"""S1/S2 splits. S4 reuse audit. TQA is start 0. Dummy Q inclusive."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["NUMBA_NUM_THREADS"] = "1"

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "N3COND"))
import audit_s4 as A  # noqa: E402
import nov3 as N  # noqa: E402
import spsa as SP  # noqa: E402
import starts as U  # noqa: E402

OUT = HERE.parents[1] / "results" / "STARTS"
R = []


def check(name, err, tol):
    ok = err <= tol
    R.append({"name": name, "err": float(err), "tol": float(tol), "pass": bool(ok)})
    print(f"{name:48s} {err:.3e}  tol {tol}  {'PASS' if ok else 'FAIL'}")


def main():
    s1n = U.starts_for_n("Nov3", 10048, 15, 3, 1)
    s1q = U.starts_for_n("QAOA", 10048, 15, 3, 1)
    check("T_S1_len", abs(len(s1n) - 1) + abs(len(s1q) - 1), 0.0)
    check("T_S1_is_tqa_nov3", float(np.max(np.abs(s1n[0] - N.tqa_nov3(3)))), 0.0)
    check("T_S1_is_tqa_qaoa", float(np.max(np.abs(s1q[0] - N.tqa_qaoa(3)))), 0.0)
    s2 = U.starts_for_n("Nov3", 10048, 15, 3, 2)
    s4 = U.starts_for_n("Nov3", 10048, 15, 3, 4)
    check("T_S2_prefix_S4", max(float(np.max(np.abs(s2[i] - s4[i]))) for i in range(2)), 0.0)
    check("T_S1_is_S4_start0", float(np.max(np.abs(s1n[0] - s4[0]))), 0.0)
    hist = N.starts_for("Nov3", 10048, 15, 3)
    check("T_S4_still_C0", max(float(np.max(np.abs(s4[i] - hist[i]))) for i in range(4)), 0.0)
    per1, k1 = SP.split_cap(3072, 1)
    per2, k2 = SP.split_cap(3072, 2)
    per4, k4 = SP.split_cap(3072, 4)
    check("T_S1_K", abs(per1 - 3072) + abs(k1 - 1531), 0.0)
    check("T_S2_K", abs(per2 - 1536) + abs(k2 - 763), 0.0)
    check("T_S4_K", abs(per4 - 768) + abs(k4 - 379), 0.0)
    check("T_equal_Q", abs(1 * per1 - 3072) + abs(2 * per2 - 3072) + abs(4 * per4 - 3072), 0.0)
    ncall = {"n": 0}

    def dummy(x, ncall=ncall):
        ncall["n"] += 1
        return float(np.sum(np.square(x)))

    sig = np.full(12, 0.1)
    rng = lambda s: np.random.default_rng(s + 3)
    for nst, kexp, ncal, ncand in ((1, 1531, 8, 2), (2, 763, 16, 4)):
        ncall["n"] = 0
        starts = [np.zeros(12) + 0.02 * i for i in range(nst)]
        out = SP.spsa_run(dummy, starts, sig, rng, 3072)
        check(f"T_queries_{nst}", abs(out["queries"] - 3072) + abs(ncall["n"] - 3072), 0.0)
        check(f"T_Krun_{nst}", abs(out["K_per_start"] - kexp), 0.0)
        check(f"T_cal_{nst}", abs(out["query_cat"]["calibration"] - ncal), 0.0)
        check(f"T_cand_{nst}", abs(out["query_cat"]["candidate"] - ncand), 0.0)
        check(f"T_live_{nst}", 0.0 if not any(m["gain_frozen"] for m in out["start_meta"]) else 1.0, 0.0)
    rows = A.audit_all()
    check("T_S4_reuse_n", abs(len(rows) - 16), 0.0)
    check("T_S4_reuse_ok", 0.0 if A.all_ok(rows) else 1.0, 0.0)
    pr = json.loads((HERE / "PREREG_STARTS_S1.json").read_text())
    check("T_prereg_seeds", 0.0 if pr["seeds"]["ids"] == list(range(10048, 10056)) else 1.0, 0.0)
    check("T_prereg_read_dev", 0.0 if pr["seeds"]["role"] == "read_dev" else 1.0, 0.0)
    check("T_prereg_gate", 0.0 if pr["gate"]["cell"].startswith("N15 L3 Nov3") else 1.0, 0.0)
    check("T_no_gr", 0.0 if pr["not_a_gr_adoption"] else 1.0, 0.0)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "S4_REUSE_AUDIT.json").write_text(json.dumps({"n": len(rows), "all_ok": A.all_ok(rows), "rows": rows}, indent=1))
    allpass = all(r["pass"] for r in R)
    (OUT / "TESTS_S1.json").write_text(json.dumps({"n": len(R), "pass": allpass, "rows": R}))
    print(f"{'ALL PASS' if allpass else 'FAIL'} n={len(R)}")
    if not allpass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
