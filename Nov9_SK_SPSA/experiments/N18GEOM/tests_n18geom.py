"""N18GEOM freeze: train/test split, θ_K/θ_M projection, exclusive seeds, no test peek."""
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

OUT = HERE.parents[1] / "results" / "N18GEOM"
PR = json.loads((HERE / "PREREG_N18GEOM.json").read_text())
ALLOC_SHA = "de595de592f33a1d5419dd52f3cbd366d0f3b328ffec9dfe4c13231a55c2a1a1"
R = []


def check(name, err, tol):
    ok = err <= tol
    R.append({"name": name, "err": float(err), "tol": float(tol), "pass": bool(ok)})
    print(f"{name:48s} {err:.3e}  tol {tol}  {'PASS' if ok else 'FAIL'}")


def rows_from(nov3s):
    out = []
    for i, xn in enumerate(nov3s):
        seed = 10872 + i
        out.append({
            "seed": seed, "arm": "nov3_center4", "status": "ok", "split": "train",
            "x": list(xn), "x_ft": list(xn),
        })
    return out


def main():
    got = hashlib.sha256((HERE.parent / "FTREP" / "alloc.py").read_bytes()).hexdigest()
    check("T_alloc_sha", 0.0 if got == ALLOC_SHA == PR["ft_source"]["alloc_py_sha256"] else 1.0, 0.0)
    ids = PR["seeds"]["ids"]
    check("T_seeds", 0.0 if ids == list(range(10872, 10888)) else 1.0, 0.0)
    check("T_split", 0.0 if (
        PR["split_documented_before_fits"]["train"] == list(range(10872, 10880))
        and PR["split_documented_before_fits"]["test"] == list(range(10880, 10888))
        and PR["split_documented_before_fits"]["do_not_peek_test_before_freeze"]
    ) else 1.0, 0.0)
    bands = [
        ("T_not_other_10888", range(10888, 10904)),
        ("T_not_ft_10856", range(10856, 10872)),
        ("T_not_pr129", range(10840, 10856)),
        ("T_not_r35", range(10824, 10840)),
        ("T_not_ft_pr127", range(10808, 10824)),
        ("T_not_pr126", range(10792, 10808)),
        ("T_not_ft_pr124", range(10776, 10792)),
        ("T_not_pr125", range(10760, 10776)),
        ("T_not_ft_pr122", range(10744, 10760)),
        ("T_not_pr123", range(10728, 10744)),
        ("T_not_r34", range(10712, 10728)),
        ("T_not_pr121", range(10680, 10696)),
        ("T_not_pr119", range(10648, 10664)),
        ("T_not_pr116", range(10600, 10616)),
        ("T_not_hold_10624", range(10624, 10632)),
        ("T_not_hole", range(10560, 10568)),
    ]
    for name, band in bands:
        check(name, 0.0 if not any(s in ids for s in band) else 1.0, 0.0)
    check("T_not_holdout", 0.0 if 10144 not in ids and 10151 not in ids else 1.0, 0.0)
    check("T_not_val_final", 0.0 if 11000 not in ids and 20000 not in ids else 1.0, 0.0)
    check("T_batch", 0.0 if PR["batch"] == {
        "N": [18], "L": [1], "q": 1024, "Q": 1024, "T": 122, "p_QAOA": 1,
        "arms": ["nov3_center4", "qaoa_center4", "nov3_geom_tk_tm"],
    } else 1.0, 0.0)
    check("T_primary_geom", 0.0 if (
        PR["freeze_from_train"]["frozen_coordinates"] == ["theta_K", "theta_M"]
        and PR["freeze_from_train"]["not_two_template"]
        and "GEOM_TRANSFER_NEGATIVE_CI" in PR["report"]["primary_label"]
        and PR["not_a_two_template_copy"]
        and PR["not_an_unsigned_mean_copy"]
        and PR["does_not_reopen_pr119"]
    ) else 1.0, 0.0)
    check("T_no_l2l3", 0.0 if PR["no_l2_or_deeper"] and PR["not_a_l3_variant"]
          and PR["not_an_architecture_change"] else 1.0, 0.0)
    check("T_gate_written", 0.0 if PR["report"]["written_before_fits"] and PR["written_before_fits"] else 1.0, 0.0)
    check("T_no_gr", 0.0 if PR["not_a_gr_adoption"] and PR["no_candidacy_revival"] else 1.0, 0.0)
    check("T_T122", abs(A.iterations_equal4(1024) - 122) + abs(PR["batch"]["T"] - 122), 0.0)
    check("T_pack18", abs(len(C.packing(18)) - 6), 0.0)
    check("T_charge_32k", 0.0 if "32768" in PR["ledger"]["independent_and_geom_charge"] else 1.0, 0.0)

    ncall = {"n": 0}

    def dummy(x, ncall=ncall):
        ncall["n"] += 1
        return float(np.sum(np.square(x))), np.asarray(x, float).copy()

    rngs_i, rngs_s = W.rng_lists(10872, 18, 0.5, 1, "Nov3rd")
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

    nov3s = []
    for i in range(8):
        nov3s.append([0.1 + 0.02 * i, -0.40, 0.01, -0.8 if i < 4 else 0.9, 0.02, 0.03])
    fr = F.build_freeze(rows_from(nov3s), "deadbeef")
    check("T_freeze_tk_tm", abs(fr["frozen_mean"]["theta_K"] + 0.40)
          + abs(fr["frozen_mean"]["theta_M"] - 0.01), 1e-12)
    check("T_freeze_idx", 0.0 if fr["frozen_indices"] == [1, 2] else 1.0, 0.0)
    y = F.project([0.2, 9.0, 8.0, 0.5, 0.1, 0.2], fr)
    check("T_project", abs(y[1] + 0.40) + abs(y[2] - 0.01) + abs(y[0] - 0.2), 1e-12)

    leaked = False
    try:
        bad = rows_from(nov3s) + [{
            "seed": 10880, "arm": "nov3_center4", "status": "ok", "split": "test",
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
    J = C.make_J(10872, 18)
    simn = S.make_sim("Nov3", J, 0.5, 1, engine="m2")
    simq = S.make_sim("QAOA", J, 0.5, 1, engine="m2")
    check("T_nparam_L1", abs(simn.nparam() - 6) + abs(simq.nparam() - 2), 0.0)
    xq_le = W.to_le("QAOA", xq)
    check("T_center_identity_L1", abs(simn.energy(xn) - simq.energy(xq_le)), 1e-8)
    check("T_qaoa_p1_layout", float(np.max(np.abs(xq - xq_le))), 0.0)

    OUT.mkdir(parents=True, exist_ok=True)
    allpass = all(r["pass"] for r in R)
    (OUT / "TESTS_N18GEOM.json").write_text(json.dumps({"n": len(R), "pass": allpass, "rows": R}))
    print(f"{'ALL PASS' if allpass else 'FAIL'} n={len(R)}")
    if not allpass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
