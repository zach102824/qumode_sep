"""L3S8 freeze: v4 C4 and spread8 unmodified; 8-start split documented; seeds exclusive."""
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
sys.path.insert(0, str(HERE.parent / "FTREP"))
sys.path.insert(0, str(HERE.parent / "N3COND"))
sys.path.insert(0, str(HERE.parent / "LEXACT"))
sys.path.insert(0, str(HERE.parent / "LE"))
import alloc as A  # noqa: E402
import core as C  # noqa: E402
import kernel_m2 as K2  # noqa: E402
import s8wrap as S8  # noqa: E402
import sim as S  # noqa: E402
import wrap as W  # noqa: E402

OUT = HERE.parents[1] / "results" / "L3S8"
PR = json.loads((HERE / "PREREG_L3S8.json").read_text())
ALLOC_SHA = "de595de592f33a1d5419dd52f3cbd366d0f3b328ffec9dfe4c13231a55c2a1a1"
R = []


def check(name, err, tol):
    ok = err <= tol
    R.append({"name": name, "err": float(err), "tol": float(tol), "pass": bool(ok)})
    print(f"{name:48s} {err:.3e}  tol {tol}  {'PASS' if ok else 'FAIL'}")


def main():
    got = hashlib.sha256((HERE.parent / "FTREP" / "alloc.py").read_bytes()).hexdigest()
    check("T_alloc_sha", 0.0 if got == ALLOC_SHA == PR["ft_source"]["alloc_py_sha256"] else 1.0, 0.0)
    ids = PR["seeds"]["ids"]
    check("T_seeds", 0.0 if ids == list(range(10400, 10408)) else 1.0, 0.0)
    check("T_not_le10391", 0.0 if min(ids) > 10391 else 1.0, 0.0)
    check("T_not_s8sf", 0.0 if 10368 not in ids and 10375 not in ids else 1.0, 0.0)
    check("T_not_r25", 0.0 if not any(s in ids for s in range(10344, 10360)) else 1.0, 0.0)
    check("T_not_holdout", 0.0 if 10144 not in ids and 10151 not in ids else 1.0, 0.0)
    check("T_not_val_final", 0.0 if 11000 not in ids and 20000 not in ids else 1.0, 0.0)
    check("T_batch", 0.0 if PR["batch"] == {
        "N": [15], "L": [3], "q": 1024, "Q": 3072, "p_QAOA": 3,
        "circuits": ["Nov3", "QAOA"], "arms": ["C4", "S8", "QAOA"],
    } else 1.0, 0.0)
    check("T_no_twostage", 0.0 if PR["not_a_two_stage_freeze_order"] and PR["not_pair_phase"] else 1.0, 0.0)
    check("T_gate_written", 0.0 if PR["gate"]["written_before_fits"] and PR["written_before_fits"] else 1.0, 0.0)
    check("T_no_gr", 0.0 if PR["not_a_gr_adoption"] and PR["no_candidacy_revival"] else 1.0, 0.0)
    check("T_T_c4", abs(A.iterations_equal4(3072) - 378) + abs(PR["split_documented_before_fits"]["C4"]["T"] - 378), 0.0)
    check("T_T_s8", abs(A.iterations_spread8(3072) - 186) + abs(PR["split_documented_before_fits"]["S8"]["T"] - 186), 0.0)
    check("T_pack15", abs(len(C.packing(15)) - 5), 0.0)
    xn0 = A.x0_spread8("Nov3rd", 3, 0, None)
    xc4 = A.x0_center4("Nov3rd", 3, 0, None)
    check("T_s8_start0_center", float(np.max(np.abs(xn0 - xc4))), 0.0)
    xq0 = A.x0_spread8("QAOA", 3, 0, None)
    check("T_qaoa_start0_center", float(np.max(np.abs(xq0 - 0.25))), 0.0)
    ncall = {"n": 0}

    def dummy(x, ncall=ncall):
        ncall["n"] += 1
        xx = np.asarray(x, float).copy()
        return float(np.sum(np.square(xx))), xx

    rngs_i, rngs_s = W.rng_lists(10400, 15, 0.5, 3, "Nov3rd")
    ncall["n"] = 0
    outc = A.run_equal4(dummy, "Nov3rd", 3, 3072, rngs_i, rngs_s, x0_fn=W.draw_fn("nov3_center4"))
    check("T_C4_Q", abs(outc["queries"]["total"] - 3072) + abs(ncall["n"] - 3072), 0.0)
    check("T_C4_T", abs(outc["n_iter"] - 378), 0.0)
    qc = outc["queries"]
    check("T_C4_cats", abs(qc["init"] - 4) + abs(qc["calibration"] - 32)
          + abs(qc["iteration"] - 3024) + abs(qc["candidate"] - 12), 0.0)

    rngs_i8, rngs_s8 = S8.rng_lists_spread8(10400, 15, 0.5, 3, S8.METHOD_ID_S8_NOV3)
    ncall["n"] = 0
    outs = A.run_spread8(dummy, "Nov3rd", 3, 3072, rngs_i8, rngs_s8)
    check("T_S8_Q", abs(outs["queries"]["total"] - 3072) + abs(ncall["n"] - 3072), 0.0)
    check("T_S8_T", abs(outs["n_iter"] - 186), 0.0)
    qs = outs["queries"]
    check("T_S8_cats", abs(qs["init"] - 8) + abs(qs["calibration"] - 64)
          + abs(qs["iteration"] - 2976) + abs(qs["candidate"] - 24), 0.0)
    check("T_S8_nstarts", abs(len(outs["tails"]) - 8), 0.0)

    rngs_iq, rngs_sq = S8.rng_lists_spread8(10400, 15, 0.5, 3, S8.METHOD_ID_S8_QAOA)
    ncall["n"] = 0
    outq = A.run_spread8(dummy, "QAOA", 3, 3072, rngs_iq, rngs_sq)
    check("T_QAOA_Q", abs(outq["queries"]["total"] - 3072) + abs(ncall["n"] - 3072), 0.0)
    check("T_QAOA_T", abs(outq["n_iter"] - 186), 0.0)
    qq = outq["queries"]
    check("T_QAOA_cats", abs(qq["init"] - 8) + abs(qq["calibration"] - 64)
          + abs(qq["iteration"] - 2976) + abs(qq["candidate"] - 24), 0.0)
    check("T_ids", abs(S8.METHOD_ID_S8_NOV3 - 90) + abs(S8.METHOD_ID_S8_QAOA - 91), 0.0)
    try:
        K2.set_threads(1)
        K2.warmup()
    except ValueError:
        pass
    J = C.make_J(10400, 15)
    simn = S.make_sim("Nov3", J, 0.5, 3, engine="m2")
    simq = S.make_sim("QAOA", J, 0.5, 3, engine="m2")
    xn = A.x0_center4("Nov3rd", 3, 0, None)
    xq = W.to_le("QAOA", A.x0_qaoa_center4(3, 0, None))
    check("T_nparam_L3", abs(simn.nparam() - 12) + abs(simq.nparam() - 6), 0.0)
    check("T_center_identity_L3", abs(simn.energy(xn) - simq.energy(xq)), 1e-8)
    OUT.mkdir(parents=True, exist_ok=True)
    allpass = all(r["pass"] for r in R)
    (OUT / "TESTS_L3S8.json").write_text(json.dumps({"n": len(R), "pass": allpass, "rows": R}))
    print(f"{'ALL PASS' if allpass else 'FAIL'} n={len(R)}")
    if not allpass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
