"""Same FT v4 alloc.py. N∈{12,15} L=3 Q=3072. No schedule tweak."""
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
PR = json.loads((HERE / "PREREG_FTREP_L3.json").read_text())
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
    check("T_seeds", 0.0 if ids == list(range(10192, 10200)) else 1.0, 0.0)
    check("T_not_prior", 0.0 if 10191 not in ids and 10175 not in ids and 10159 not in ids else 1.0, 0.0)
    check("T_not_holdout", 0.0 if 10144 not in ids and 10151 not in ids else 1.0, 0.0)
    check("T_not_r19", 0.0 if 10184 not in ids and 10191 not in ids else 1.0, 0.0)
    check("T_not_train", 0.0 if 10018 not in ids and 10025 not in ids else 1.0, 0.0)
    check("T_batch", 0.0 if PR["batch"]["N"] == [12, 15] and PR["batch"]["L"] == [3]
          and PR["batch"]["Q"] == 3072 and PR["batch"]["q"] == 1024
          and PR["batch"]["arms"] == ["nov3_center4", "qaoa_center4"] else 1.0, 0.0)
    check("T_no_n18", 0.0 if PR["no_n18_l3"] and PR["not_a_schedule_tweak"] else 1.0, 0.0)
    check("T_iter_3072", abs(A.iterations_equal4(3072) - 378), 0.0)
    xn = A.x0_center4("Nov3rd", 3, 0, None)
    want = np.array([0.25, 0.0, 0.0, -0.5, 0.0, 0.0, -0.5, 0.0, 0.0, -0.5, 0.0, 0.0])
    check("T_nov3_center_L3", float(np.max(np.abs(xn - want))), 0.0)
    xq = A.x0_qaoa_center4(3, 0, None)
    check("T_qaoa_center_L3", float(np.max(np.abs(xq - 0.25))), 0.0)
    xq_le = W.to_le("QAOA", xq)
    check("T_qaoa_map_center", float(np.max(np.abs(xq_le - 0.25))), 0.0)
    ncall = {"n": 0}

    def dummy(x, ncall=ncall):
        ncall["n"] += 1
        return float(np.sum(np.square(x))), None

    rngs_i, rngs_s = W.rng_lists(10192, 12, 0.5, 3, "Nov3rd")
    ncall["n"] = 0
    out = A.run_equal4(dummy, "Nov3rd", 3, 3072, rngs_i, rngs_s, x0_fn=W.draw_fn("nov3_center4"))
    check("T_ledger_Q", abs(out["queries"]["total"] - 3072) + abs(ncall["n"] - 3072), 0.0)
    check("T_n_iter", abs(out["n_iter"] - 378), 0.0)
    q = out["queries"]
    check("T_cats", abs(q["init"] - 4) + abs(q["calibration"] - 32)
          + abs(q["iteration"] - 3024) + abs(q["candidate"] - 12), 0.0)
    try:
        K2.set_threads(1)
        K2.warmup()
    except ValueError:
        pass
    J = C.make_J(10192, 6)
    simn = S.make_sim("Nov3", J, 0.5, 3, engine="m2")
    simq = S.make_sim("QAOA", J, 0.5, 3, engine="m2")
    check("T_nparam", abs(simn.nparam() - 12) + abs(simq.nparam() - 6), 0.0)
    en = simn.energy(xn)
    eq = simq.energy(xq_le)
    check("T_center_identity_L3", abs(en - eq), 1e-8)
    check("T_gates", 0.0 if "L3-15" in PR["gate"] and "L3-12" in PR["gate"] else 1.0, 0.0)
    check("T_no_gr", 0.0 if PR["not_a_gr_adoption"] else 1.0, 0.0)
    OUT.mkdir(parents=True, exist_ok=True)
    allpass = all(r["pass"] for r in R)
    (OUT / "TESTS_FTREP_L3.json").write_text(json.dumps({"n": len(R), "pass": allpass, "rows": R}))
    print(f"{'ALL PASS' if allpass else 'FAIL'} n={len(R)}")
    if not allpass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
