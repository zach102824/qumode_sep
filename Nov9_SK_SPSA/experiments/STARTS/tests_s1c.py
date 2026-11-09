"""Frozen PR83 recipe. Fresh seeds. No fit changes."""
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
PR = json.loads((HERE / "PREREG_STARTS_S1C.json").read_text())
R = []


def check(name, err, tol):
    ok = err <= tol
    R.append({"name": name, "err": float(err), "tol": float(tol), "pass": bool(ok)})
    print(f"{name:48s} {err:.3e}  tol {tol}  {'PASS' if ok else 'FAIL'}")


def sha(rel):
    return hashlib.sha256((HERE.parents[2] / rel if False else {
        "starts.py": HERE / "starts.py",
        "N3COND/spsa.py": HERE.parent / "N3COND" / "spsa.py",
        "N3COND/nov3.py": HERE.parent / "N3COND" / "nov3.py",
    }[rel]).read_bytes()).hexdigest()


def main():
    frozen = PR["frozen_from_PR83"]["file_sha256"]
    check("T_sha_starts", 0.0 if sha("starts.py") == frozen["starts.py"] else 1.0, 0.0)
    check("T_sha_spsa", 0.0 if sha("N3COND/spsa.py") == frozen["N3COND/spsa.py"] else 1.0, 0.0)
    check("T_sha_nov3", 0.0 if sha("N3COND/nov3.py") == frozen["N3COND/nov3.py"] else 1.0, 0.0)
    ids = PR["seeds"]["ids"]
    check("T_seeds", 0.0 if ids == list(range(10056, 10064)) else 1.0, 0.0)
    check("T_not_10048", 0.0 if 10048 not in ids and 10055 not in ids else 1.0, 0.0)
    check("T_not_final", 0.0 if 20000 not in ids and 11000 not in ids else 1.0, 0.0)
    check("T_N_both", 0.0 if PR["batch"]["N"] == [12, 15] else 1.0, 0.0)
    check("T_no_S2", 0.0 if PR["batch"]["arms"] == ["S1", "S4"] else 1.0, 0.0)
    check("T_gate_both", 0.0 if "N12 AND at N15" in PR["gate"]["pass"] else 1.0, 0.0)
    s1 = U.starts_for_n("Nov3", 10056, 12, 3, 1)
    s4 = U.starts_for_n("Nov3", 10056, 12, 3, 4)
    hist = N.starts_for("Nov3", 10056, 12, 3)
    check("T_S1_tqa", float(np.max(np.abs(s1[0] - N.tqa_nov3(3)))), 0.0)
    check("T_S4_C0", max(float(np.max(np.abs(s4[i] - hist[i]))) for i in range(4)), 0.0)
    check("T_S1_is_S4_0", float(np.max(np.abs(s1[0] - s4[0]))), 0.0)
    per1, k1 = SP.split_cap(3072, 1)
    per4, k4 = SP.split_cap(3072, 4)
    check("T_S1_K_frozen", abs(k1 - 1531) + abs(per1 - 3072), 0.0)
    check("T_S4_K_frozen", abs(k4 - 379) + abs(per4 - 768), 0.0)
    ncall = {"n": 0}

    def dummy(x, ncall=ncall):
        ncall["n"] += 1
        return float(np.sum(np.square(x)))

    sig = np.full(12, 0.1)
    rng = lambda s: np.random.default_rng(s + 5)
    for nst, kexp, ncal, ncand in ((1, 1531, 8, 2), (4, 379, 32, 8)):
        ncall["n"] = 0
        starts = [np.zeros(12) + 0.01 * i for i in range(nst)]
        out = SP.spsa_run(dummy, starts, sig, rng, 3072)
        check(f"T_Q_{nst}", abs(out["queries"] - 3072) + abs(ncall["n"] - 3072), 0.0)
        check(f"T_K_{nst}", abs(out["K_per_start"] - kexp), 0.0)
        check(f"T_cal_{nst}", abs(out["query_cat"]["calibration"] - ncal), 0.0)
        check(f"T_cand_{nst}", abs(out["query_cat"]["candidate"] - ncand), 0.0)
        check(f"T_live_{nst}", 0.0 if not any(m["gain_frozen"] for m in out["start_meta"]) else 1.0, 0.0)
    OUT.mkdir(parents=True, exist_ok=True)
    allpass = all(r["pass"] for r in R)
    (OUT / "TESTS_S1C.json").write_text(json.dumps({"n": len(R), "pass": allpass, "rows": R}))
    print(f"{'ALL PASS' if allpass else 'FAIL'} n={len(R)}")
    if not allpass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
