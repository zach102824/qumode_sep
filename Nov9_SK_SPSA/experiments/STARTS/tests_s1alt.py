"""Frozen JOINT S1 vs ALT block schedule at L=2 Q=2048. No recipe edits."""
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
import alt_spsa as ALT  # noqa: E402
import nov3 as N  # noqa: E402
import spsa as SP  # noqa: E402
import starts as U  # noqa: E402

OUT = HERE.parents[1] / "results" / "STARTS"
PR = json.loads((HERE / "PREREG_STARTS_S1ALT.json").read_text())
FILES = {
    "starts.py": HERE / "starts.py",
    "N3COND/spsa.py": HERE.parent / "N3COND" / "spsa.py",
    "N3COND/nov3.py": HERE.parent / "N3COND" / "nov3.py",
}
R = []


def check(name, err, tol):
    ok = err <= tol
    R.append({"name": name, "err": float(err), "tol": float(tol), "pass": bool(ok)})
    print(f"{name:48s} {err:.3e}  tol {tol}  {'PASS' if ok else 'FAIL'}")


def main():
    frozen = PR["frozen_from_PR83_87"]["file_sha256"]
    for rel, path in FILES.items():
        got = hashlib.sha256(path.read_bytes()).hexdigest()
        check(f"T_sha_{rel}", 0.0 if got == frozen[rel] else 1.0, 0.0)
    ids = PR["seeds"]["ids"]
    check("T_seeds", 0.0 if ids == list(range(10088, 10096)) else 1.0, 0.0)
    check("T_not_prior", 0.0 if 10087 not in ids and 10080 not in ids else 1.0, 0.0)
    check("T_batch", 0.0 if PR["batch"] == {"N": [15], "L": [2], "q": 1024, "Q": 2048,
                                            "circuits": ["Nov3", "QAOA"],
                                            "arms": ["JOINT", "ALT"]} else 1.0, 0.0)
    s1 = U.starts_for_n("Nov3", 10088, 15, 2, 1)
    check("T_S1_tqa", float(np.max(np.abs(s1[0] - N.tqa_nov3(2)))) + abs(s1[0].size - 9), 0.0)
    ma, mb = ALT.block_masks("Nov3", 2)
    check("T_Nov3_A", float(np.max(np.abs(ma - np.array([1, 1, 1, 0, 0, 0, 0, 0, 0])))), 0.0)
    check("T_Nov3_B", float(np.max(np.abs(mb - np.array([0, 0, 0, 1, 1, 1, 1, 1, 1])))), 0.0)
    ga, gb = ALT.block_masks("QAOA", 2)
    check("T_QAOA_A", float(np.max(np.abs(ga - np.array([1, 1, 0, 0])))), 0.0)
    check("T_QAOA_B", float(np.max(np.abs(gb - np.array([0, 0, 1, 1])))), 0.0)
    per, k = SP.split_cap(2048, 1)
    check("T_K", abs(per - 2048) + abs(k - 1019), 0.0)
    ncall = {"n": 0}

    def dummy(x, ncall=ncall):
        ncall["n"] += 1
        return float(np.sum(np.square(x)))

    rng = lambda s: np.random.default_rng(s + 9)
    ncall["n"] = 0
    outj = SP.spsa_run(dummy, [np.zeros(9)], np.full(9, 0.1), rng, 2048)
    check("T_JOINT_Q", abs(outj["queries"] - 2048) + abs(ncall["n"] - 2048), 0.0)
    check("T_JOINT_K", abs(outj["K_per_start"] - 1019), 0.0)
    check("T_JOINT_live", 0.0 if not any(m["gain_frozen"] for m in outj["start_meta"]) else 1.0, 0.0)
    seen = []
    ncall["n"] = 0

    def rec(x, ncall=ncall, seen=seen):
        ncall["n"] += 1
        seen.append(np.asarray(x, float).copy())
        return float(np.sum(np.square(x)))

    rng = lambda s: np.random.default_rng(s + 9)
    outa = ALT.spsa_run_alt(rec, [np.zeros(9)], np.ones(9), rng, 2048, ma, mb)
    check("T_ALT_Q", abs(outa["queries"] - 2048) + abs(ncall["n"] - 2048), 0.0)
    check("T_ALT_K", abs(outa["K_per_start"] - 1019), 0.0)
    m = outa["start_meta"][0]
    check("T_ALT_steps", abs(m["n_block_A_steps"] - 510) + abs(m["n_block_B_steps"] - 509), 0.0)
    check("T_no_prefix", 0.0 if m["joint_prefix_K"] == 0 else 1.0, 0.0)
    check("T_ALT_live", 0.0 if not m["gain_frozen"] else 1.0, 0.0)
    check("T_ALT_cal", abs(outa["query_cat"]["calibration"] - 8), 0.0)
    check("T_ALT_cand", abs(outa["query_cat"]["candidate"] - 2), 0.0)
    p0, p1 = seen[8], seen[9]
    check("T_first_pert_B_zero", float(np.max(np.abs(p0[3:])) + np.max(np.abs(p1[3:]))), 0.0)
    check("T_first_pert_A_moved", 0.0 if np.max(np.abs(p0[:3])) > 0 else 1.0, 0.0)
    rngj = lambda s: np.random.default_rng(11)
    rnga = lambda s: np.random.default_rng(11)
    ncall["n"] = 0
    j = SP.spsa_run(dummy, [np.zeros(9) + 0.05], np.full(9, 0.1), rngj, 2048)
    ncall["n"] = 0
    a = ALT.spsa_run_alt(dummy, [np.zeros(9) + 0.05], np.full(9, 0.1), rnga, 2048, ma, mb)
    check("T_same_gain_cal", abs(j["start_meta"][0]["a_calibrated"] - a["start_meta"][0]["a_calibrated"]), 1e-12)
    check("T_gate_cell", 0.0 if PR["gate"]["cell"].startswith("N15 L2 Nov3") else 1.0, 0.0)
    check("T_no_gr", 0.0 if PR["not_a_gr_adoption"] and PR["not_a_s1_retune"] else 1.0, 0.0)
    OUT.mkdir(parents=True, exist_ok=True)
    allpass = all(r["pass"] for r in R)
    (OUT / "TESTS_S1ALT.json").write_text(json.dumps({"n": len(R), "pass": allpass, "rows": R}))
    print(f"{'ALL PASS' if allpass else 'FAIL'} n={len(R)}")
    if not allpass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
