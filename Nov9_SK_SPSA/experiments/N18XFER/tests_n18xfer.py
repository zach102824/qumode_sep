"""N18XFER freeze: train/test split, v4 L=1, exclusive seeds, no test peek."""
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
import freeze as F  # noqa: E402
import kernel_m2 as K2  # noqa: E402
import sim as S  # noqa: E402
import wrap as W  # noqa: E402

OUT = HERE.parents[1] / "results" / "N18XFER"
PR = json.loads((HERE / "PREREG_N18XFER.json").read_text())
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
    check("T_seeds", 0.0 if ids == list(range(10648, 10664)) else 1.0, 0.0)
    check("T_split", 0.0 if (
        PR["split_documented_before_fits"]["train"] == list(range(10648, 10656))
        and PR["split_documented_before_fits"]["test"] == list(range(10656, 10664))
        and PR["split_documented_before_fits"]["do_not_peek_test_before_freeze"]
    ) else 1.0, 0.0)
    check("T_not_r31", 0.0 if not any(s in ids for s in range(10536, 10552)) else 1.0, 0.0)
    check("T_not_r32", 0.0 if not any(s in ids for s in range(10584, 10600)) else 1.0, 0.0)
    check("T_not_r30", 0.0 if not any(s in ids for s in range(10504, 10520)) else 1.0, 0.0)
    check("T_not_ft", 0.0 if not any(s in ids for s in (
        list(range(10520, 10536)) + list(range(10552, 10560))
        + list(range(10568, 10584)) + list(range(10616, 10632))
        + list(range(10664, 10680))
    )) else 1.0, 0.0)
    check("T_not_local_next", 0.0 if not any(s in ids for s in range(10632, 10648)) else 1.0, 0.0)
    check("T_not_pr116", 0.0 if not any(s in ids for s in range(10600, 10616)) else 1.0, 0.0)
    check("T_not_n21", 0.0 if not any(s in ids for s in range(10432, 10440)) else 1.0, 0.0)
    check("T_not_l3tie", 0.0 if not any(s in ids for s in range(10552, 10560)) else 1.0, 0.0)
    check("T_not_hole", 0.0 if not any(s in ids for s in range(10560, 10568)) else 1.0, 0.0)
    check("T_not_holdout", 0.0 if 10144 not in ids and 10151 not in ids else 1.0, 0.0)
    check("T_not_val_final", 0.0 if 11000 not in ids and 20000 not in ids else 1.0, 0.0)
    check("T_batch", 0.0 if PR["batch"] == {
        "N": [18], "L": [1], "q": 1024, "Q": 1024, "T": 122, "p_QAOA": 1,
        "arms": ["nov3_center4", "qaoa_center4"],
    } else 1.0, 0.0)
    check("T_primary_stays_mean", 0.0 if PR["freeze_from_train"]["primary_aggregator"] == "mean"
          and PR["freeze_from_train"]["aggregator_not_chosen_on_test"]
          and PR["freeze_from_train"]["not_switched_to_median_from_pr116_test"] else 1.0, 0.0)
    check("T_no_l2l3", 0.0 if PR["no_l2_or_deeper"] and PR["not_a_l3_variant"]
          and PR["not_layer_tied_snap"] and PR["not_an_architecture_change"] else 1.0, 0.0)
    check("T_gate_written", 0.0 if PR["report"]["written_before_fits"] and PR["written_before_fits"] else 1.0, 0.0)
    check("T_no_gr", 0.0 if PR["not_a_gr_adoption"] and PR["no_candidacy_revival"] else 1.0, 0.0)
    check("T_T122", abs(A.iterations_equal4(1024) - 122) + abs(PR["batch"]["T"] - 122), 0.0)
    check("T_pack18", abs(len(C.packing(18)) - 6), 0.0)
    check("T_xfer_charge", 0.0 if PR["stage_B"]["transfer_eval"]["queries_each"] == 1
          and PR["ledger"]["transfer_charge"].endswith("= 32") else 1.0, 0.0)
    check("T_primary_mean", 0.0 if "shared-mean" in PR["report"]["primary_label"] else 1.0, 0.0)

    ncall = {"n": 0}

    def dummy(x, ncall=ncall):
        ncall["n"] += 1
        return float(np.sum(np.square(x))), np.asarray(x, float).copy()

    rngs_i, rngs_s = W.rng_lists(10648, 18, 0.5, 1, "Nov3rd")
    ncall["n"] = 0
    out = A.run_equal4(dummy, "Nov3rd", 1, 1024, rngs_i, rngs_s, x0_fn=W.draw_fn("nov3_center4"))
    check("T_ledger_Q", abs(out["queries"]["total"] - 1024) + abs(ncall["n"] - 1024), 0.0)
    check("T_n_iter", abs(out["n_iter"] - 122), 0.0)
    q = out["queries"]
    check("T_cats", abs(q["init"] - 4) + abs(q["calibration"] - 32)
          + abs(q["iteration"] - 976) + abs(q["candidate"] - 12), 0.0)
    xn = A.x0_center4("Nov3rd", 1, 0, None)
    check("T_nov3_center_L1", float(np.max(np.abs(xn - np.array([0.25, 0.0, 0.0, -0.5, 0.0, 0.0])))), 0.0)
    xq = A.x0_qaoa_center4(1, 0, None)
    check("T_qaoa_center_L1", float(np.max(np.abs(xq - 0.25))), 0.0)

    train_rows = []
    rng = np.random.default_rng(0)
    for s in range(10648, 10656):
        xn_s = rng.normal(size=6)
        xq_s = rng.normal(size=2)
        train_rows.append({
            "seed": s, "arm": "nov3_center4", "status": "ok", "split": "train",
            "x": xn_s.tolist(), "x_ft": xn_s.tolist(),
        })
        train_rows.append({
            "seed": s, "arm": "qaoa_center4", "status": "ok", "split": "train",
            "x": xq_s.tolist(), "x_ft": xq_s.tolist(),
        })
    fr = F.build_freeze(train_rows, "deadbeef")
    check("T_freeze_n", abs(fr["nov3"]["n"] - 8) + abs(fr["qaoa"]["n"] - 8), 0.0)
    check("T_freeze_dim", abs(fr["nov3"]["dim"] - 6) + abs(fr["qaoa"]["dim"] - 2), 0.0)
    leaked = False
    try:
        bad = list(train_rows) + [{
            "seed": 10656, "arm": "nov3_center4", "status": "ok", "split": "test",
            "x": [0.0] * 6, "x_ft": [0.0] * 6,
        }]
        F.build_freeze(bad, "deadbeef")
        leaked = True
    except ValueError:
        leaked = False
    check("T_no_test_peek", 0.0 if not leaked else 1.0, 0.0)

    try:
        K2.set_threads(1)
        K2.warmup()
    except ValueError:
        pass
    J = C.make_J(10648, 18)
    simn = S.make_sim("Nov3", J, 0.5, 1, engine="m2")
    simq = S.make_sim("QAOA", J, 0.5, 1, engine="m2")
    check("T_nparam_L1", abs(simn.nparam() - 6) + abs(simq.nparam() - 2), 0.0)
    xq_le = W.to_le("QAOA", xq)
    check("T_center_identity_L1", abs(simn.energy(xn) - simq.energy(xq_le)), 1e-8)
    check("T_qaoa_p1_layout", float(np.max(np.abs(xq - xq_le))), 0.0)

    OUT.mkdir(parents=True, exist_ok=True)
    allpass = all(r["pass"] for r in R)
    (OUT / "TESTS_N18XFER.json").write_text(json.dumps({"n": len(R), "pass": allpass, "rows": R}))
    print(f"{'ALL PASS' if allpass else 'FAIL'} n={len(R)}")
    if not allpass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
