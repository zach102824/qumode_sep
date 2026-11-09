"""Frozen PR88 S1JOINT vs S1ALT at L=2,3 Q=qL. No recipe edits."""
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
PR = json.loads((HERE / "PREREG_STARTS_S1ALT23.json").read_text())
FILES = {
    "starts.py": HERE / "starts.py",
    "N3COND/spsa.py": HERE.parent / "N3COND" / "spsa.py",
    "N3COND/nov3.py": HERE.parent / "N3COND" / "nov3.py",
    "alt_spsa.py": HERE / "alt_spsa.py",
}
KEXP = {2048: 1019, 3072: 1531}
AB = {1019: (510, 509), 1531: (766, 765)}
R = []


def check(name, err, tol):
    ok = err <= tol
    R.append({"name": name, "err": float(err), "tol": float(tol), "pass": bool(ok)})
    print(f"{name:48s} {err:.3e}  tol {tol}  {'PASS' if ok else 'FAIL'}")


def main():
    frozen = PR["frozen_from_PR83_88"]["file_sha256"]
    for rel, path in FILES.items():
        got = hashlib.sha256(path.read_bytes()).hexdigest()
        check(f"T_sha_{rel}", 0.0 if got == frozen[rel] else 1.0, 0.0)
    ids = PR["seeds"]["ids"]
    check("T_seeds", 0.0 if ids == list(range(10096, 10104)) else 1.0, 0.0)
    check("T_not_prior", 0.0 if 10095 not in ids and 10088 not in ids else 1.0, 0.0)
    check("T_batch", 0.0 if PR["batch"]["N"] == [15] and PR["batch"]["L"] == [2, 3]
          and PR["batch"]["arms"] == ["S1JOINT", "S1ALT"]
          and PR["batch"]["Q"] == {"2": 2048, "3": 3072} else 1.0, 0.0)
    check("T_S1_L2_tqa", float(np.max(np.abs(U.starts_for_n("Nov3", 10096, 15, 2, 1)[0] - N.tqa_nov3(2)))), 0.0)
    check("T_S1_L3_nparam", abs(U.starts_for_n("Nov3", 10096, 15, 3, 1)[0].size - 12), 0.0)
    ma, mb = ALT.block_masks("Nov3", 3)
    check("T_Nov3_L3_A", abs(float(ma[:3].sum()) - 3) + abs(float(ma[3:].sum())), 0.0)
    check("T_Nov3_L3_B", abs(float(mb[3:].sum()) - 9) + abs(float(mb[:3].sum())), 0.0)
    ga, gb = ALT.block_masks("QAOA", 3)
    check("T_QAOA_L3", abs(float(ga[:3].sum()) - 3) + abs(float(gb[3:].sum()) - 3), 0.0)
    ncall = {"n": 0}

    def dummy(x, ncall=ncall):
        ncall["n"] += 1
        return float(np.sum(np.square(x)))

    for Q, kexp in KEXP.items():
        per, k = SP.split_cap(Q, 1)
        check(f"T_K_{Q}", abs(per - Q) + abs(k - kexp), 0.0)
        na, nb = AB[kexp]
        dim = 9 if Q == 2048 else 12
        ma, mb = ALT.block_masks("Nov3", 2 if Q == 2048 else 3)
        rng = lambda s: np.random.default_rng(s + 9)
        ncall["n"] = 0
        outj = SP.spsa_run(dummy, [np.zeros(dim)], np.full(dim, 0.1), rng, Q)
        check(f"T_JOINT_Q_{Q}", abs(outj["queries"] - Q) + abs(ncall["n"] - Q), 0.0)
        ncall["n"] = 0
        rng = lambda s: np.random.default_rng(s + 9)
        outa = ALT.spsa_run_alt(dummy, [np.zeros(dim)], np.full(dim, 0.1), rng, Q, ma, mb)
        m = outa["start_meta"][0]
        check(f"T_ALT_Q_{Q}", abs(outa["queries"] - Q) + abs(ncall["n"] - Q), 0.0)
        check(f"T_ALT_steps_{Q}", abs(m["n_block_A_steps"] - na) + abs(m["n_block_B_steps"] - nb), 0.0)
        check(f"T_no_prefix_{Q}", 0.0 if m["joint_prefix_K"] == 0 else 1.0, 0.0)
        check(f"T_live_{Q}", 0.0 if not m["gain_frozen"] else 1.0, 0.0)
    check("T_no_gr", 0.0 if PR["not_a_gr_adoption"] and PR["not_an_alt_retune"] else 1.0, 0.0)
    check("T_gate_both_L", 0.0 if PR["gate"]["cells"] == ["N15 L2 Nov3", "N15 L3 Nov3"] else 1.0, 0.0)
    OUT.mkdir(parents=True, exist_ok=True)
    allpass = all(r["pass"] for r in R)
    (OUT / "TESTS_S1ALT23.json").write_text(json.dumps({"n": len(R), "pass": allpass, "rows": R}))
    print(f"{'ALL PASS' if allpass else 'FAIL'} n={len(R)}")
    if not allpass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
