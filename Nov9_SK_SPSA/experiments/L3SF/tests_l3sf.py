"""L3SF freeze: v4 JOINT unmodified; shell-first split documented; seeds exclusive."""
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
import shellfirst as SF  # noqa: E402
import sim as S  # noqa: E402
import wrap as W  # noqa: E402

OUT = HERE.parents[1] / "results" / "L3SF"
PR = json.loads((HERE / "PREREG_L3SF.json").read_text())
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
    check("T_seeds", 0.0 if ids == list(range(10368, 10376)) else 1.0, 0.0)
    check("T_not_l3ts", 0.0 if 10360 not in ids and 10367 not in ids else 1.0, 0.0)
    check("T_not_d18", 0.0 if 10320 not in ids and 10327 not in ids else 1.0, 0.0)
    check("T_not_ft_pair", 0.0 if not any(s in ids for s in range(10328, 10336)) else 1.0, 0.0)
    check("T_not_r25", 0.0 if not any(s in ids for s in range(10344, 10360)) else 1.0, 0.0)
    check("T_not_reserved", 0.0 if 10376 not in ids and max(ids) <= 10375 else 1.0, 0.0)
    check("T_window", 0.0 if min(ids) >= 10360 and max(ids) <= 10375 else 1.0, 0.0)
    check("T_not_holdout", 0.0 if 10144 not in ids and 10151 not in ids else 1.0, 0.0)
    check("T_not_val_final", 0.0 if 11000 not in ids and 20000 not in ids else 1.0, 0.0)
    check("T_batch", 0.0 if PR["batch"] == {
        "N": [15], "L": [3], "q": 1024, "Q": 3072, "p_QAOA": 3,
        "circuits": ["Nov3", "QAOA"], "arms": ["JOINT", "SHELL_FIRST", "QAOA"],
    } else 1.0, 0.0)
    sp = PR["split_documented_before_fits"]
    check("T_split_Q", abs(sp["Q_total"] - 3072) + abs(sp["Q_shell_stage"] - 1536)
          + abs(sp["Q_uc_stage"] - 1536), 0.0)
    check("T_T_shell", abs(A.iterations_equal4(1536) - 186) + abs(int(sp["T_shell"]) - 186)
          + abs(SF.T_SHELL - 186), 0.0)
    check("T_T_uc", abs(SF.iterations_nblocks(1536, 2) - 378) + abs(int(sp["T_uc"]) - 378)
          + abs(SF.T_UC - 378), 0.0)
    check("T_not_l3ts_order", 0.0 if PR["not_l3ts_joint_then_shell"] and PR["not_pair_phase"] else 1.0, 0.0)
    check("T_gate_written", 0.0 if PR["gate"]["written_before_fits"] and PR["written_before_fits"] else 1.0, 0.0)
    check("T_no_gr", 0.0 if PR["not_a_gr_adoption"] and PR["no_candidacy_revival"] else 1.0, 0.0)
    sig_j = A.sigmas_of("Nov3rd", 3)
    sig_s = SF.shell_sigmas(3)
    sig_u = SF.uc_sigmas(3)
    check("T_shell_uc_zero", float(np.max(np.abs(sig_s[:3]))), 0.0)
    check("T_shell_rest", float(np.max(np.abs(sig_s[3:] - sig_j[3:]))), 0.0)
    check("T_uc_shell_zero", float(np.max(np.abs(sig_u[3:]))), 0.0)
    check("T_uc_rest", float(np.max(np.abs(sig_u[:3] - sig_j[:3]))), 0.0)
    check("T_ids", abs(SF.METHOD_ID_SHELL - 82) + abs(SF.METHOD_ID_UC - 83), 0.0)
    check("T_pack15", abs(len(C.packing(15)) - 5), 0.0)
    ncall = {"n": 0}

    def dummy(x, ncall=ncall):
        ncall["n"] += 1
        xx = np.asarray(x, float).copy()
        return float(np.sum(np.square(xx))), xx

    rngs_i, rngs_s = W.rng_lists(10368, 15, 0.5, 3, "Nov3rd")
    ncall["n"] = 0
    outj = A.run_equal4(dummy, "Nov3rd", 3, 3072, rngs_i, rngs_s, x0_fn=W.draw_fn("nov3_center4"))
    check("T_JOINT_Q", abs(outj["queries"]["total"] - 3072) + abs(ncall["n"] - 3072), 0.0)
    check("T_JOINT_T", abs(outj["n_iter"] - 378), 0.0)
    ncall["n"] = 0
    outt = SF.run_shellfirst(dummy, 3, 3072, 10368, 15, 0.5)
    check("T_SF_Q", abs(outt["queries"]["total"] - 3072) + abs(ncall["n"] - 3072), 0.0)
    check("T_SF_T", abs(outt["n_iter_shell"] - 186) + abs(outt["n_iter_uc"] - 378), 0.0)
    qt = outt["queries"]
    check("T_SF_cats", abs(qt["init"] - 6) + abs(qt["calibration"] - 48)
          + abs(qt["iteration"] - 3000) + abs(qt["candidate"] - 18), 0.0)
    q1, q2 = outt["queries_shell"], outt["queries_uc"]
    check("T_SF_stage_cats", abs(q1["total"] - 1536) + abs(q2["total"] - 1536)
          + abs(q1["init"] - 4) + abs(q2["init"] - 2)
          + abs(q1["iteration"] - 1488) + abs(q2["iteration"] - 1512), 0.0)
    x = np.asarray(outt["x"])
    check("T_uc_stage1_frozen_in_shells", float(np.max(np.abs(x[3:] - outt["shells_frozen"]))), 1e-15)
    check("T_start0_uc_if_sel", 0.0 if outt["shell_start_id"] != 0 else
          float(np.max(np.abs(outt["uc_frozen"] - np.array([0.25, 0.0, 0.0])))), 1e-15)
    rngs_iq, rngs_sq = W.rng_lists(10368, 15, 0.5, 3, "QAOA")
    ncall["n"] = 0
    outq = A.run_equal4(dummy, "QAOA", 3, 3072, rngs_iq, rngs_sq, x0_fn=W.draw_fn("qaoa_center4"))
    check("T_QAOA_Q", abs(outq["queries"]["total"] - 3072) + abs(ncall["n"] - 3072), 0.0)
    try:
        K2.set_threads(1)
        K2.warmup()
    except ValueError:
        pass
    J = C.make_J(10368, 15)
    simn = S.make_sim("Nov3", J, 0.5, 3, engine="m2")
    simq = S.make_sim("QAOA", J, 0.5, 3, engine="m2")
    xn = A.x0_center4("Nov3rd", 3, 0, None)
    xq = W.to_le("QAOA", A.x0_qaoa_center4(3, 0, None))
    check("T_nparam_L3", abs(simn.nparam() - 12) + abs(simq.nparam() - 6), 0.0)
    check("T_center_identity_L3", abs(simn.energy(xn) - simq.energy(xq)), 1e-8)
    OUT.mkdir(parents=True, exist_ok=True)
    allpass = all(r["pass"] for r in R)
    (OUT / "TESTS_L3SF.json").write_text(json.dumps({"n": len(R), "pass": allpass, "rows": R}))
    print(f"{'ALL PASS' if allpass else 'FAIL'} n={len(R)}")
    if not allpass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
