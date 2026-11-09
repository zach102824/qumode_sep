"""Frozen PR83/84/85 S1-only at L=1,2 Q=qL. No recipe edits. Not continuation."""
from __future__ import annotations

import hashlib
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
import nov3 as N  # noqa: E402
import spsa as SP  # noqa: E402
import starts as U  # noqa: E402

OUT = HERE.parents[1] / "results" / "STARTS"
PR = json.loads((HERE / "PREREG_STARTS_S1SH.json").read_text())
FILES = {
    "starts.py": HERE / "starts.py",
    "N3COND/spsa.py": HERE.parent / "N3COND" / "spsa.py",
    "N3COND/nov3.py": HERE.parent / "N3COND" / "nov3.py",
}
KEXP = {1024: 507, 2048: 1019}
R = []


def check(name, err, tol):
    ok = err <= tol
    R.append({"name": name, "err": float(err), "tol": float(tol), "pass": bool(ok)})
    print(f"{name:48s} {err:.3e}  tol {tol}  {'PASS' if ok else 'FAIL'}")


def main():
    frozen = PR["frozen_from_PR83_84_85"]["file_sha256"]
    for rel, path in FILES.items():
        got = hashlib.sha256(path.read_bytes()).hexdigest()
        check(f"T_sha_{rel}", 0.0 if got == frozen[rel] else 1.0, 0.0)
    ids = PR["seeds"]["ids"]
    check("T_seeds", 0.0 if ids == list(range(10080, 10088)) else 1.0, 0.0)
    check("T_not_prior", 0.0 if 10079 not in ids and 10026 not in ids and 10018 not in ids else 1.0, 0.0)
    check("T_batch", 0.0 if PR["batch"]["N"] == [15] and PR["batch"]["L"] == [1, 2]
          and PR["batch"]["arms"] == ["S1"] and PR["batch"]["Q"] == {"1": 1024, "2": 2048} else 1.0, 0.0)
    s1 = U.starts_for_n("Nov3", 10080, 15, 1, 1)
    s2 = U.starts_for_n("Nov3", 10080, 15, 2, 1)
    check("T_S1_L1_nparam", abs(len(s1) - 1) + abs(s1[0].size - 6), 0.0)
    check("T_S1_L2_nparam", abs(len(s2) - 1) + abs(s2[0].size - 9), 0.0)
    check("T_S1_L1_tqa", float(np.max(np.abs(s1[0] - N.tqa_nov3(1)))), 0.0)
    check("T_S1_L2_tqa", float(np.max(np.abs(s2[0] - N.tqa_nov3(2)))), 0.0)
    q1 = U.starts_for_n("QAOA", 10080, 15, 1, 1)
    check("T_QAOA_L1_nparam", abs(q1[0].size - 2), 0.0)
    check("T_cold_not_cont", 0.0 if PR["continuation_closed"] and "cold" in PR["allocation"] else 1.0, 0.0)
    check("T_not_r10", 0.0 if PR["not_local_r10"] else 1.0, 0.0)
    for Q, kexp in KEXP.items():
        per, k = SP.split_cap(Q, 1)
        check(f"T_K_{Q}", abs(per - Q) + abs(k - kexp), 0.0)
    ncall = {"n": 0}

    def dummy(x, ncall=ncall):
        ncall["n"] += 1
        return float(np.sum(np.square(x)))

    rng = lambda s: np.random.default_rng(s + 9)
    for Q, kexp in KEXP.items():
        ncall["n"] = 0
        out = SP.spsa_run(dummy, [np.zeros(6)], np.full(6, 0.1), rng, Q)
        check(f"T_Qrun_{Q}", abs(out["queries"] - Q) + abs(ncall["n"] - Q), 0.0)
        check(f"T_Krun_{Q}", abs(out["K_per_start"] - kexp), 0.0)
        check(f"T_cal_{Q}", abs(out["query_cat"]["calibration"] - 8), 0.0)
        check(f"T_cand_{Q}", abs(out["query_cat"]["candidate"] - 2), 0.0)
        check(f"T_live_{Q}", 0.0 if not any(m["gain_frozen"] for m in out["start_meta"]) else 1.0, 0.0)
    check("T_no_gr", 0.0 if PR["not_a_gr_adoption"] and PR["not_a_s1_retune"] else 1.0, 0.0)
    OUT.mkdir(parents=True, exist_ok=True)
    allpass = all(r["pass"] for r in R)
    (OUT / "TESTS_S1SH.json").write_text(json.dumps({"n": len(R), "pass": allpass, "rows": R}))
    print(f"{'ALL PASS' if allpass else 'FAIL'} n={len(R)}")
    if not allpass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
