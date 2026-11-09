"""Same FT v4 alloc.py. N=15 L∈{2,3,4} Q=1024·L. Characterization, not adoption."""
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
sys.path.insert(0, str(HERE.parent / "LEXACT"))
sys.path.insert(0, str(HERE.parent / "LE"))
import alloc as A  # noqa: E402
import core as C  # noqa: E402
import kernel_m2 as K2  # noqa: E402
import sim as S  # noqa: E402
import wrap as W  # noqa: E402

OUT = HERE.parents[1] / "results" / "FTREP"
PR = json.loads((HERE / "PREREG_FTREP_D.json").read_text())
ALLOC_SHA = "de595de592f33a1d5419dd52f3cbd366d0f3b328ffec9dfe4c13231a55c2a1a1"
R = []


def check(name, err, tol):
    ok = err <= tol
    R.append({"name": name, "err": float(err), "tol": float(tol), "pass": bool(ok)})
    print(f"{name:48s} {err:.3e}  tol {tol}  {'PASS' if ok else 'FAIL'}")


def main():
    got = hashlib.sha256((HERE / "alloc.py").read_bytes()).hexdigest()
    check("T_alloc_sha", 0.0 if got == ALLOC_SHA == PR["ft_source"]["alloc_py_sha256"] else 1.0, 0.0)
    ids = PR["seeds"]["ids"]
    check("T_seeds", 0.0 if ids == list(range(10264, 10272)) else 1.0, 0.0)
    check("T_not_prior", 0.0 if 10263 not in ids and 10247 not in ids and 10151 not in ids else 1.0, 0.0)
    check("T_not_r19", 0.0 if 10184 not in ids and 10232 not in ids else 1.0, 0.0)
    check("T_batch", 0.0 if PR["batch"]["N"] == [15] and PR["batch"]["L"] == [2, 3, 4]
          and PR["batch"]["q"] == 1024
          and PR["batch"]["Q_of_L"] == {"2": 2048, "3": 3072, "4": 4096} else 1.0, 0.0)
    check("T_char", 0.0 if PR["characterization_not_adoption"] and PR["report"]["no_adoption_gate"] else 1.0, 0.0)
    check("T_iter_L2", abs(A.iterations_equal4(2048) - 250), 0.0)
    check("T_iter_L3", abs(A.iterations_equal4(3072) - 378), 0.0)
    check("T_iter_L4", abs(A.iterations_equal4(4096) - 506), 0.0)
    xn4 = A.x0_center4("Nov3rd", 4, 0, None)
    want = np.array([0.25, 0.0, 0.0] + [-0.5, 0.0, 0.0] * 4)
    check("T_nov3_center_L4", float(np.max(np.abs(xn4 - want))), 0.0)
    xq4 = A.x0_qaoa_center4(4, 0, None)
    check("T_qaoa_center_L4", float(np.max(np.abs(xq4 - 0.25))), 0.0)
    ncall = {"n": 0}

    def dummy(x, ncall=ncall):
        ncall["n"] += 1
        return float(np.sum(np.square(x))), None

    rngs_i, rngs_s = W.rng_lists(10264, 15, 0.5, 4, "Nov3rd")
    ncall["n"] = 0
    out = A.run_equal4(dummy, "Nov3rd", 4, 4096, rngs_i, rngs_s, x0_fn=W.draw_fn("nov3_center4"))
    check("T_ledger_Q4", abs(out["queries"]["total"] - 4096) + abs(ncall["n"] - 4096), 0.0)
    check("T_n_iter_L4", abs(out["n_iter"] - 506), 0.0)
    q = out["queries"]
    check("T_cats_L4", abs(q["init"] - 4) + abs(q["calibration"] - 32)
          + abs(q["iteration"] - 4048) + abs(q["candidate"] - 12), 0.0)
    try:
        K2.set_threads(1)
        K2.warmup()
    except ValueError:
        pass
    J = C.make_J(10264, 6)
    for L in (2, 3, 4):
        simn = S.make_sim("Nov3", J, 0.5, L, engine="m2")
        simq = S.make_sim("QAOA", J, 0.5, L, engine="m2")
        xn = A.x0_center4("Nov3rd", L, 0, None)
        xq = W.to_le("QAOA", A.x0_qaoa_center4(L, 0, None))
        check(f"T_nparam_L{L}", abs(simn.nparam() - (3 + 3 * L)) + abs(simq.nparam() - 2 * L), 0.0)
        check(f"T_center_identity_L{L}", abs(simn.energy(xn) - simq.energy(xq)), 1e-8)
    check("T_no_gr", 0.0 if PR["not_a_gr_adoption"] and PR["no_candidacy_revival"] else 1.0, 0.0)
    OUT.mkdir(parents=True, exist_ok=True)
    allpass = all(r["pass"] for r in R)
    (OUT / "TESTS_FTREP_D.json").write_text(json.dumps({"n": len(R), "pass": allpass, "rows": R}))
    print(f"{'ALL PASS' if allpass else 'FAIL'} n={len(R)}")
    if not allpass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
