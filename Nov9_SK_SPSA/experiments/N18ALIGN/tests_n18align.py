"""N18ALIGN freeze: train/test split, s1-only gauge, exclusive seeds, no test peek."""
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

OUT = HERE.parents[1] / "results" / "N18ALIGN"
PR = json.loads((HERE / "PREREG_N18ALIGN.json").read_text())
ALLOC_SHA = "de595de592f33a1d5419dd52f3cbd366d0f3b328ffec9dfe4c13231a55c2a1a1"
R = []


def check(name, err, tol):
    ok = err <= tol
    R.append({"name": name, "err": float(err), "tol": float(tol), "pass": bool(ok)})
    print(f"{name:48s} {err:.3e}  tol {tol}  {'PASS' if ok else 'FAIL'}")


def rows_from_nov3(xs):
    out = []
    for i, x in enumerate(xs):
        seed = 10680 + i
        out.append({
            "seed": seed, "arm": "nov3_center4", "status": "ok", "split": "train",
            "x": list(x), "x_ft": list(x),
        })
        out.append({
            "seed": seed, "arm": "qaoa_center4", "status": "ok", "split": "train",
            "x": [0.1, 0.2], "x_ft": [0.1, 0.2],
        })
    return out


def main():
    got = hashlib.sha256((HERE.parent / "FTREP" / "alloc.py").read_bytes()).hexdigest()
    check("T_alloc_sha", 0.0 if got == ALLOC_SHA == PR["ft_source"]["alloc_py_sha256"] else 1.0, 0.0)
    ids = PR["seeds"]["ids"]
    check("T_seeds", 0.0 if ids == list(range(10680, 10696)) else 1.0, 0.0)
    check("T_split", 0.0 if (
        PR["split_documented_before_fits"]["train"] == list(range(10680, 10688))
        and PR["split_documented_before_fits"]["test"] == list(range(10688, 10696))
        and PR["split_documented_before_fits"]["do_not_peek_test_before_freeze"]
        and PR["split_documented_before_fits"]["do_not_reuse_pr119_seeds"]
    ) else 1.0, 0.0)
    check("T_not_pr119", 0.0 if not any(s in ids for s in range(10648, 10664)) else 1.0, 0.0)
    check("T_not_ft_next", 0.0 if not any(s in ids for s in range(10664, 10680)) else 1.0, 0.0)
    check("T_not_ft_after", 0.0 if not any(s in ids for s in range(10696, 10712)) else 1.0, 0.0)
    check("T_not_r33", 0.0 if not any(s in ids for s in range(10632, 10648)) else 1.0, 0.0)
    check("T_not_hold_10624", 0.0 if not any(s in ids for s in range(10624, 10632)) else 1.0, 0.0)
    check("T_not_r31", 0.0 if not any(s in ids for s in range(10536, 10552)) else 1.0, 0.0)
    check("T_not_r32", 0.0 if not any(s in ids for s in range(10584, 10600)) else 1.0, 0.0)
    check("T_not_r30", 0.0 if not any(s in ids for s in range(10504, 10520)) else 1.0, 0.0)
    check("T_not_ft", 0.0 if not any(s in ids for s in (
        list(range(10520, 10536)) + list(range(10552, 10560))
        + list(range(10568, 10584)) + list(range(10616, 10632))
    )) else 1.0, 0.0)
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
    check("T_primary_sign_aligned", 0.0 if (
        PR["freeze_from_train"]["primary_vector"] == "nov3_sign_aligned_mean"
        and PR["freeze_from_train"]["aggregator_not_chosen_on_test"]
        and PR["report"]["primary_label"].startswith("TRANSFER_MEAN_NEGATIVE_CI")
        and "sign-aligned-mean" in PR["report"]["primary_label"]
    ) else 1.0, 0.0)
    check("T_gauge_s1_only", 0.0 if (
        PR["freeze_from_train"]["sign_aligned"]["map"] == "flip_s1_only_if_s1_lt_0"
        and PR["freeze_from_train"]["sign_aligned"]["covariant_discrete_gauge_in_saved_recipe"] is False
        and PR["freeze_from_train"]["sign_aligned"]["theta_K_theta_M_unchanged"]
        and F.GAUGE["covariant_discrete_gauge_in_saved_recipe"] is False
    ) else 1.0, 0.0)
    check("T_no_l2l3", 0.0 if PR["no_l2_or_deeper"] and PR["not_a_l3_variant"]
          and PR["not_layer_tied_snap"] and PR["not_an_architecture_change"]
          and PR["not_an_unsigned_mean_copy_at_other_N"] else 1.0, 0.0)
    check("T_gate_written", 0.0 if PR["report"]["written_before_fits"] and PR["written_before_fits"] else 1.0, 0.0)
    check("T_no_gr", 0.0 if PR["not_a_gr_adoption"] and PR["no_candidacy_revival"] else 1.0, 0.0)
    check("T_T122", abs(A.iterations_equal4(1024) - 122) + abs(PR["batch"]["T"] - 122), 0.0)
    check("T_pack18", abs(len(C.packing(18)) - 6), 0.0)
    check("T_xfer_charge", 0.0 if PR["stage_B"]["transfer_eval"]["queries_each"] == 1
          and PR["ledger"]["transfer_charge"].endswith("= 32") else 1.0, 0.0)
    check("T_stageB_vectors", 0.0 if PR["stage_B"]["transfer_eval"]["vectors"] == [
        "nov3_unsigned_mean", "nov3_sign_aligned_mean",
        "nov3_larger_cluster_mean", "qaoa_unsigned_mean",
    ] else 1.0, 0.0)

    ncall = {"n": 0}

    def dummy(x, ncall=ncall):
        ncall["n"] += 1
        return float(np.sum(np.square(x))), np.asarray(x, float).copy()

    rngs_i, rngs_s = W.rng_lists(10680, 18, 0.5, 1, "Nov3rd")
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

    y, flipped = F.align_s1([0.1, -0.4, 0.02, -0.8, 0.05, -0.1])
    check("T_align_flips_s1_only", float(np.max(np.abs(y - np.array([0.1, -0.4, 0.02, 0.8, 0.05, -0.1]))))
          + (0.0 if flipped else 1.0), 0.0)
    y2, flipped2 = F.align_s1([0.1, -0.4, 0.02, 0.8, 0.05, -0.1])
    check("T_align_keeps_positive_s1", float(np.max(np.abs(y2 - np.array([0.1, -0.4, 0.02, 0.8, 0.05, -0.1]))))
          + (0.0 if not flipped2 else 1.0), 0.0)

    xs44 = []
    for i in range(4):
        xs44.append([0.0, -0.4, 0.0, -0.8, 0.0, 0.0])
    for i in range(4):
        xs44.append([0.0, -0.4, 0.0, 0.9, 0.0, 0.0])
    fr44 = F.build_freeze(rows_from_nov3(xs44), "deadbeef")
    check("T_tie_chooses_s1_gt_0", 0.0 if (
        fr44["nov3_larger_cluster"]["rule"] == "tie_s1_gt_0"
        and fr44["nov3_larger_cluster"]["chosen"] == "s1>0"
        and fr44["nov3_larger_cluster"]["n_pos"] == 4
        and fr44["nov3_larger_cluster"]["n_neg"] == 4
        and abs(fr44["nov3_larger_cluster"]["mean"][3] - 0.9) < 1e-12
    ) else 1.0, 0.0)
    check("T_align_mean_s1_abs", abs(fr44["nov3_sign_aligned"]["mean"][3] - 0.85), 1e-12)
    check("T_unsigned_mean_s1_near_zero", abs(fr44["nov3_unsigned"]["mean"][3] - 0.05), 1e-12)
    check("T_align_n_flipped", abs(fr44["nov3_sign_aligned"]["n_flipped"] - 4), 0.0)
    check("T_thetaK_untouched", abs(fr44["nov3_sign_aligned"]["mean"][1] + 0.4)
          + abs(fr44["nov3_unsigned"]["mean"][1] + 0.4), 1e-12)

    xs53 = [[0.0, -0.4, 0.0, -0.7, 0.0, 0.0] for _ in range(5)]
    xs53 += [[0.0, -0.4, 0.0, 0.9, 0.0, 0.0] for _ in range(3)]
    fr53 = F.build_freeze(rows_from_nov3(xs53), "deadbeef")
    check("T_larger_neg_cluster", 0.0 if (
        fr53["nov3_larger_cluster"]["chosen"] == "s1<0"
        and fr53["nov3_larger_cluster"]["n_cluster"] == 5
        and abs(fr53["nov3_larger_cluster"]["mean"][3] + 0.7) < 1e-12
    ) else 1.0, 0.0)

    leaked = False
    try:
        bad = rows_from_nov3(xs44) + [{
            "seed": 10688, "arm": "nov3_center4", "status": "ok", "split": "test",
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
    J = C.make_J(10680, 18)
    simn = S.make_sim("Nov3", J, 0.5, 1, engine="m2")
    simq = S.make_sim("QAOA", J, 0.5, 1, engine="m2")
    check("T_nparam_L1", abs(simn.nparam() - 6) + abs(simq.nparam() - 2), 0.0)
    xq_le = W.to_le("QAOA", xq)
    check("T_center_identity_L1", abs(simn.energy(xn) - simq.energy(xq_le)), 1e-8)
    check("T_qaoa_p1_layout", float(np.max(np.abs(xq - xq_le))), 0.0)

    OUT.mkdir(parents=True, exist_ok=True)
    allpass = all(r["pass"] for r in R)
    (OUT / "TESTS_N18ALIGN.json").write_text(json.dumps({"n": len(R), "pass": allpass, "rows": R}))
    print(f"{'ALL PASS' if allpass else 'FAIL'} n={len(R)}")
    if not allpass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
