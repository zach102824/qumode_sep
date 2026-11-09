"""Frozen S1JOINT / PR88 S1ALT11 / 1:3 S1ALT13 at L=2 Q=2048. alt_spsa.py unmodified."""
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
import alt13_spsa as A13  # noqa: E402
import alt_spsa as ALT  # noqa: E402
import nov3 as N  # noqa: E402
import spsa as SP  # noqa: E402
import starts as U  # noqa: E402

OUT = HERE.parents[1] / "results" / "STARTS"
PR = json.loads((HERE / "PREREG_STARTS_S1ALT13.json").read_text())
FILES = {
    "starts.py": HERE / "starts.py",
    "N3COND/spsa.py": HERE.parent / "N3COND" / "spsa.py",
    "N3COND/nov3.py": HERE.parent / "N3COND" / "nov3.py",
    "alt_spsa.py": HERE / "alt_spsa.py",
}
R = []


def check(name, err, tol):
    ok = err <= tol
    R.append({"name": name, "err": float(err), "tol": float(tol), "pass": bool(ok)})
    print(f"{name:48s} {err:.3e}  tol {tol}  {'PASS' if ok else 'FAIL'}")


def main():
    frozen = PR["frozen"]["file_sha256"]
    for rel, path in FILES.items():
        got = hashlib.sha256(path.read_bytes()).hexdigest()
        check(f"T_sha_{rel}", 0.0 if got == frozen[rel] else 1.0, 0.0)
    ids = PR["seeds"]["ids"]
    check("T_seeds", 0.0 if ids == list(range(10104, 10112)) else 1.0, 0.0)
    check("T_not_prior", 0.0 if 10103 not in ids and 10096 not in ids else 1.0, 0.0)
    check("T_batch", 0.0 if PR["batch"]["N"] == [15] and PR["batch"]["L"] == [2]
          and PR["batch"]["arms"] == ["S1JOINT", "S1ALT11", "S1ALT13"]
          and PR["batch"]["Q"] == 2048 else 1.0, 0.0)
    check("T_S1_tqa", float(np.max(np.abs(U.starts_for_n("Nov3", 10104, 15, 2, 1)[0] - N.tqa_nov3(2)))), 0.0)
    ma, mb = ALT.block_masks("Nov3", 2)
    check("T_Nov3_A", abs(float(ma[:3].sum()) - 3) + abs(float(ma[3:].sum())), 0.0)
    check("T_Nov3_B", abs(float(mb[3:].sum()) - 6) + abs(float(mb[:3].sum())), 0.0)
    ga, gb = ALT.block_masks("QAOA", 2)
    check("T_QAOA", abs(float(ga[:2].sum()) - 2) + abs(float(gb[2:].sum()) - 2), 0.0)
    seq = "".join(A13.block_for_k_13(k) for k in range(8))
    check("T_phase_start_A_then_BBB", 0.0 if seq == "ABBBABBB" else 1.0, 0.0)
    leftover = [A13.block_for_k_13(k) for k in range(1016, 1019)]
    check("T_leftover_continue_ABB", 0.0 if leftover == ["A", "B", "B"] else 1.0, 0.0)
    nA = sum(1 for k in range(1019) if A13.block_for_k_13(k) == "A")
    check("T_K1019_counts", abs(nA - 255) + abs((1019 - nA) - 764), 0.0)
    per, k = SP.split_cap(2048, 1)
    check("T_K_2048", abs(per - 2048) + abs(k - 1019), 0.0)
    ncall = {"n": 0}

    def dummy(x, ncall=ncall):
        ncall["n"] += 1
        return float(np.sum(np.square(x)))

    dim = 9
    ma, mb = ALT.block_masks("Nov3", 2)
    rng = lambda s: np.random.default_rng(s + 13)
    ncall["n"] = 0
    outj = SP.spsa_run(dummy, [np.zeros(dim)], np.full(dim, 0.1), rng, 2048)
    check("T_JOINT_Q", abs(outj["queries"] - 2048) + abs(ncall["n"] - 2048), 0.0)
    ncall["n"] = 0
    rng = lambda s: np.random.default_rng(s + 13)
    out11 = ALT.spsa_run_alt(dummy, [np.zeros(dim)], np.full(dim, 0.1), rng, 2048, ma, mb)
    m11 = out11["start_meta"][0]
    check("T_ALT11_Q", abs(out11["queries"] - 2048) + abs(ncall["n"] - 2048), 0.0)
    check("T_ALT11_steps", abs(m11["n_block_A_steps"] - 510) + abs(m11["n_block_B_steps"] - 509), 0.0)
    check("T_ALT11_no_prefix", 0.0 if m11["joint_prefix_K"] == 0 else 1.0, 0.0)
    ncall["n"] = 0
    rng = lambda s: np.random.default_rng(s + 13)
    out13 = A13.spsa_run_alt13(dummy, [np.zeros(dim)], np.full(dim, 0.1), rng, 2048, ma, mb)
    m13 = out13["start_meta"][0]
    check("T_ALT13_Q", abs(out13["queries"] - 2048) + abs(ncall["n"] - 2048), 0.0)
    check("T_ALT13_steps", abs(m13["n_block_A_steps"] - 255) + abs(m13["n_block_B_steps"] - 764), 0.0)
    check("T_ALT13_no_prefix", 0.0 if m13["joint_prefix_K"] == 0 else 1.0, 0.0)
    check("T_ALT13_leftover_policy", 0.0 if m13["leftover_policy"] == "continue_phase" and m13["leftover_blocks"] == ["A", "B", "B"] else 1.0, 0.0)
    check("T_live", 0.0 if (not m11["gain_frozen"] and not m13["gain_frozen"]) else 1.0, 0.0)
    check("T_no_gr", 0.0 if PR["not_a_gr_adoption"] and PR["not_an_alt11_retune"] else 1.0, 0.0)
    OUT.mkdir(parents=True, exist_ok=True)
    allpass = all(r["pass"] for r in R)
    (OUT / "TESTS_S1ALT13.json").write_text(json.dumps({"n": len(R), "pass": allpass, "rows": R}))
    print(f"{'ALL PASS' if allpass else 'FAIL'} n={len(R)}")
    if not allpass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
