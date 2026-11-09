"""Frozen PR83/84 S1/S4 at L=6, Q=6144. No recipe edits."""
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
PR = json.loads((HERE / "PREREG_STARTS_S1L6.json").read_text())
FILES = {
    "starts.py": HERE / "starts.py",
    "N3COND/spsa.py": HERE.parent / "N3COND" / "spsa.py",
    "N3COND/nov3.py": HERE.parent / "N3COND" / "nov3.py",
}
R = []


def check(name, err, tol):
    ok = err <= tol
    R.append({"name": name, "err": float(err), "tol": float(tol), "pass": bool(ok)})
    print(f"{name:48s} {err:.3e}  tol {tol}  {'PASS' if ok else 'FAIL'}")


def main():
    frozen = PR["frozen_from_PR83_84"]["file_sha256"]
    for rel, path in FILES.items():
        got = hashlib.sha256(path.read_bytes()).hexdigest()
        check(f"T_sha_{rel}", 0.0 if got == frozen[rel] else 1.0, 0.0)
    ids = PR["seeds"]["ids"]
    check("T_seeds", 0.0 if ids == list(range(10064, 10072)) else 1.0, 0.0)
    check("T_not_prior", 0.0 if 10063 not in ids and 10048 not in ids else 1.0, 0.0)
    check("T_batch", 0.0 if PR["batch"] == {"N": [15], "L": [6], "q": 1024, "Q": 6144,
                                            "circuits": ["Nov3", "QAOA"], "arms": ["S1", "S4"]} else 1.0, 0.0)
    s1 = U.starts_for_n("Nov3", 10064, 15, 6, 1)
    s4 = U.starts_for_n("Nov3", 10064, 15, 6, 4)
    hist = N.starts_for("Nov3", 10064, 15, 6)
    check("T_S1_len_nparam", abs(len(s1) - 1) + abs(s1[0].size - 21), 0.0)
    check("T_S1_tqa", float(np.max(np.abs(s1[0] - N.tqa_nov3(6)))), 0.0)
    check("T_S4_C0", max(float(np.max(np.abs(s4[i] - hist[i]))) for i in range(4)), 0.0)
    check("T_S1_is_S4_0", float(np.max(np.abs(s1[0] - s4[0]))), 0.0)
    q1 = U.starts_for_n("QAOA", 10064, 15, 6, 1)
    check("T_QAOA_S1_nparam", abs(q1[0].size - 12), 0.0)
    per1, k1 = SP.split_cap(6144, 1)
    per4, k4 = SP.split_cap(6144, 4)
    check("T_S1_K", abs(per1 - 6144) + abs(k1 - 3067), 0.0)
    check("T_S4_K", abs(per4 - 1536) + abs(k4 - 763), 0.0)
    check("T_Q_qL", abs(1024 * 6 - 6144), 0.0)
    ncall = {"n": 0}

    def dummy(x, ncall=ncall):
        ncall["n"] += 1
        return float(np.sum(np.square(x)))

    rng = lambda s: np.random.default_rng(s + 9)
    for nst, dim, kexp, ncal, ncand in ((1, 21, 3067, 8, 2), (4, 21, 763, 32, 8)):
        ncall["n"] = 0
        starts = [np.zeros(dim) + 0.01 * i for i in range(nst)]
        out = SP.spsa_run(dummy, starts, np.full(dim, 0.1), rng, 6144)
        check(f"T_Q_{nst}", abs(out["queries"] - 6144) + abs(ncall["n"] - 6144), 0.0)
        check(f"T_Krun_{nst}", abs(out["K_per_start"] - kexp), 0.0)
        check(f"T_cal_{nst}", abs(out["query_cat"]["calibration"] - ncal), 0.0)
        check(f"T_cand_{nst}", abs(out["query_cat"]["candidate"] - ncand), 0.0)
        check(f"T_live_{nst}", 0.0 if not any(m["gain_frozen"] for m in out["start_meta"]) else 1.0, 0.0)
    check("T_gate_cell", 0.0 if PR["gate"]["cell"].startswith("N15 L6 Nov3") else 1.0, 0.0)
    check("T_no_gr", 0.0 if PR["not_a_gr_adoption"] and PR["not_a_s1_retune"] else 1.0, 0.0)
    OUT.mkdir(parents=True, exist_ok=True)
    allpass = all(r["pass"] for r in R)
    (OUT / "TESTS_S1L6.json").write_text(json.dumps({"n": len(R), "pass": allpass, "rows": R}))
    print(f"{'ALL PASS' if allpass else 'FAIL'} n={len(R)}")
    if not allpass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
