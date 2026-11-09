"""Same FT v4 alloc.py. Intentional replay of holdout 10144–10151. N15 L2 Q=2048."""
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
PR = json.loads((HERE / "PREREG_FTREP_H.json").read_text())
FT = json.loads((OUT / "FT_HOLDOUT_BD9EFFE3F.json").read_text())
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
    check("T_seeds_holdout", 0.0 if ids == list(range(10144, 10152)) else 1.0, 0.0)
    check("T_intentional", 0.0 if PR["intentional_holdout_replay"] and PR["not_a_collision"] else 1.0, 0.0)
    check("T_not_dev", 0.0 if 10018 not in ids and 10152 not in ids and 10168 not in ids else 1.0, 0.0)
    check("T_not_r19", 0.0 if 10184 not in ids and 10192 not in ids else 1.0, 0.0)
    check("T_batch", 0.0 if PR["batch"]["N_required"] == [15] and PR["batch"]["L"] == [2]
          and PR["batch"]["Q"] == 2048 and PR["batch"]["q"] == 1024 else 1.0, 0.0)
    check("T_iter_2048", abs(A.iterations_equal4(2048) - 250), 0.0)
    check("T_ft_table_commit", 0.0 if FT["source_commit"].startswith("bd9effe3f") else 1.0, 0.0)
    check("T_ft_table_seeds", 0.0 if [r["seed"] for r in FT["cells"]["N15"]] == ids else 1.0, 0.0)
    ncall = {"n": 0}

    def dummy(x, ncall=ncall):
        ncall["n"] += 1
        return float(np.sum(np.square(x))), None

    rngs_i, rngs_s = W.rng_lists(10144, 15, 0.5, 2, "Nov3rd")
    ncall["n"] = 0
    out = A.run_equal4(dummy, "Nov3rd", 2, 2048, rngs_i, rngs_s, x0_fn=W.draw_fn("nov3_center4"))
    check("T_ledger_Q", abs(out["queries"]["total"] - 2048) + abs(ncall["n"] - 2048), 0.0)
    check("T_n_iter", abs(out["n_iter"] - 250), 0.0)
    q = out["queries"]
    check("T_cats", abs(q["init"] - 4) + abs(q["calibration"] - 32)
          + abs(q["iteration"] - 2000) + abs(q["candidate"] - 12), 0.0)
    try:
        K2.set_threads(1)
        K2.warmup()
    except ValueError:
        pass
    xn = A.x0_center4("Nov3rd", 2, 0, None)
    xq = W.to_le("QAOA", A.x0_qaoa_center4(2, 0, None))
    J = C.make_J(10144, 6)
    simn = S.make_sim("Nov3", J, 0.5, 2, engine="m2")
    simq = S.make_sim("QAOA", J, 0.5, 2, engine="m2")
    check("T_center_identity", abs(simn.energy(xn) - simq.energy(xq)), 1e-8)
    check("T_gate_R", 0.0 if PR["gate"]["R"].startswith("N=15") else 1.0, 0.0)
    check("T_no_gr", 0.0 if PR["not_a_gr_adoption"] and PR["no_candidacy_revival"] else 1.0, 0.0)
    OUT.mkdir(parents=True, exist_ok=True)
    allpass = all(r["pass"] for r in R)
    (OUT / "TESTS_FTREP_H.json").write_text(json.dumps({"n": len(R), "pass": allpass, "rows": R}))
    print(f"{'ALL PASS' if allpass else 'FAIL'} n={len(R)}")
    if not allpass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
