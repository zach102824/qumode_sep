"""L3TIE freeze: tied SNAP dim 6; JOINT/QAOA unmodified v4; seeds exclusive."""
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
import sim as S  # noqa: E402
import tie as T  # noqa: E402
import wrap as W  # noqa: E402

OUT = HERE.parents[1] / "results" / "L3TIE"
PR = json.loads((HERE / "PREREG_L3TIE.json").read_text())
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
    check("T_seeds", 0.0 if ids == list(range(10552, 10560)) else 1.0, 0.0)
    held = PR["seeds"]["held_unused_same_band"]
    check("T_held", 0.0 if held == list(range(10560, 10568)) else 1.0, 0.0)
    check("T_not_r31", 0.0 if not any(s in ids for s in range(10536, 10552)) else 1.0, 0.0)
    check("T_not_r30", 0.0 if not any(s in ids for s in range(10504, 10520)) else 1.0, 0.0)
    check("T_not_ft", 0.0 if not any(s in ids for s in list(range(10520, 10536)) + list(range(10568, 10584))) else 1.0, 0.0)
    check("T_not_n21", 0.0 if not any(s in ids for s in range(10432, 10440)) else 1.0, 0.0)
    check("T_not_l3s8", 0.0 if not any(s in ids for s in range(10400, 10408)) else 1.0, 0.0)
    check("T_not_holdout", 0.0 if 10144 not in ids and 10151 not in ids else 1.0, 0.0)
    check("T_not_val_final", 0.0 if 11000 not in ids and 20000 not in ids else 1.0, 0.0)
    check("T_batch", 0.0 if PR["batch"] == {
        "N": [15], "L": [3], "q": 1024, "Q": 3072, "p_QAOA": 3,
        "circuits": ["Nov3", "QAOA"], "arms": ["JOINT", "TIE", "QAOA"],
    } else 1.0, 0.0)
    check("T_closed_levers", 0.0 if (
        PR["not_a_start_count_variant"] and PR["not_a_two_stage_freeze_order"]
        and PR["not_a_shell_unshare"] and PR["not_pair_phase"]
        and PR["not_per_cavity_snap"] and PR["not_paper_vm"]
        and PR["not_a_center4_l2_copy"] and PR["not_a_depth_curve"]
    ) else 1.0, 0.0)
    check("T_gate_written", 0.0 if PR["gate"]["written_before_fits"] and PR["written_before_fits"] else 1.0, 0.0)
    check("T_no_gr", 0.0 if PR["not_a_gr_adoption"] and PR["no_candidacy_revival"] else 1.0, 0.0)
    check("T_T378", abs(A.iterations_equal4(3072) - 378)
          + abs(PR["split_documented_before_fits"]["TIE"]["T"] - 378)
          + abs(PR["split_documented_before_fits"]["JOINT"]["T"] - 378), 0.0)
    check("T_pack15", abs(len(C.packing(15)) - 5), 0.0)
    check("T_ids", abs(T.METHOD_ID_TIE - 94), 0.0)

    x1 = A.x0_center4("Nov3rd", 1, 0, None)
    x3 = A.x0_center4("Nov3rd", 3, 0, None)
    xt = T.expand_tie(x1, 3)
    check("T_tie_expand_center", float(np.max(np.abs(xt - x3))), 0.0)
    check("T_tie_dim", abs(x1.size - 6) + abs(xt.size - 12), 0.0)

    ncall = {"n": 0}

    def dummy(x, ncall=ncall):
        ncall["n"] += 1
        xx = np.asarray(x, float).copy()
        return float(np.sum(np.square(xx))), xx

    rngs_i, rngs_s = W.rng_lists(10552, 15, 0.5, 3, "Nov3rd")
    ncall["n"] = 0
    outj = A.run_equal4(dummy, "Nov3rd", 3, 3072, rngs_i, rngs_s, x0_fn=W.draw_fn("nov3_center4"))
    check("T_JOINT_Q", abs(outj["queries"]["total"] - 3072) + abs(ncall["n"] - 3072), 0.0)
    check("T_JOINT_T", abs(outj["n_iter"] - 378), 0.0)
    qj = outj["queries"]
    check("T_JOINT_cats", abs(qj["init"] - 4) + abs(qj["calibration"] - 32)
          + abs(qj["iteration"] - 3024) + abs(qj["candidate"] - 12), 0.0)

    rngs_it, rngs_st = T.rng_lists_tie(10552, 15, 0.5, 3)

    def dummy_tie(x, ncall=ncall):
        ncall["n"] += 1
        xx = T.expand_tie(x, 3)
        return float(np.sum(np.square(xx))), xx

    ncall["n"] = 0
    outt = A.run_equal4(dummy_tie, "Nov3rd", 1, 3072, rngs_it, rngs_st, x0_fn=T.draw_tie)
    check("T_TIE_Q", abs(outt["queries"]["total"] - 3072) + abs(ncall["n"] - 3072), 0.0)
    check("T_TIE_T", abs(outt["n_iter"] - 378), 0.0)
    check("T_TIE_spsa_dim", abs(np.asarray(outt["x"]).size - 6), 0.0)
    check("T_TIE_aux_dim", abs(np.asarray(outt["aux"]).size - 12), 0.0)
    qt = outt["queries"]
    check("T_TIE_cats", abs(qt["init"] - 4) + abs(qt["calibration"] - 32)
          + abs(qt["iteration"] - 3024) + abs(qt["candidate"] - 12), 0.0)

    rngs_iq, rngs_sq = W.rng_lists(10552, 15, 0.5, 3, "QAOA")
    ncall["n"] = 0
    outq = A.run_equal4(dummy, "QAOA", 3, 3072, rngs_iq, rngs_sq, x0_fn=W.draw_fn("qaoa_center4"))
    check("T_QAOA_Q", abs(outq["queries"]["total"] - 3072) + abs(ncall["n"] - 3072), 0.0)
    check("T_QAOA_T", abs(outq["n_iter"] - 378), 0.0)
    qq = outq["queries"]
    check("T_QAOA_cats", abs(qq["init"] - 4) + abs(qq["calibration"] - 32)
          + abs(qq["iteration"] - 3024) + abs(qq["candidate"] - 12), 0.0)

    try:
        K2.set_threads(1)
        K2.warmup()
    except ValueError:
        pass
    J = C.make_J(10552, 15)
    simn = S.make_sim("Nov3", J, 0.5, 3, engine="m2")
    simq = S.make_sim("QAOA", J, 0.5, 3, engine="m2")
    xn = A.x0_center4("Nov3rd", 3, 0, None)
    xq = W.to_le("QAOA", A.x0_qaoa_center4(3, 0, None))
    check("T_nparam_L3", abs(simn.nparam() - 12) + abs(simq.nparam() - 6), 0.0)
    check("T_center_identity_L3", abs(simn.energy(xn) - simq.energy(xq)), 1e-8)
    check("T_tie_center_energy", abs(simn.energy(T.expand_tie(x1, 3)) - simn.energy(xn)), 1e-8)

    OUT.mkdir(parents=True, exist_ok=True)
    allpass = all(r["pass"] for r in R)
    (OUT / "TESTS_L3TIE.json").write_text(json.dumps({"n": len(R), "pass": allpass, "rows": R}))
    print(f"{'ALL PASS' if allpass else 'FAIL'} n={len(R)}")
    if not allpass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
