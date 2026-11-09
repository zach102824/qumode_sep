"""Frozen S1JOINT L1→L warm-start. C1 cont.py and S1 recipes unmodified."""
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
sys.path.insert(0, str(HERE.parent / "STARTS"))
import nov3 as N  # noqa: E402
import s1w as W  # noqa: E402
import spsa as SP  # noqa: E402
import starts as ST  # noqa: E402

OUT = HERE.parents[1] / "results" / "DEPTH"
PR = json.loads((HERE / "PREREG_DEPTH_S1W.json").read_text())
FILES = {
    "starts.py": HERE.parent / "STARTS" / "starts.py",
    "N3COND/spsa.py": HERE.parent / "N3COND" / "spsa.py",
    "N3COND/nov3.py": HERE.parent / "N3COND" / "nov3.py",
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
    check("T_seeds", 0.0 if ids == list(range(10128, 10136)) else 1.0, 0.0)
    check("T_not_r15", 0.0 if 10127 not in ids and 10120 not in ids else 1.0, 0.0)
    check("T_not_le", 0.0 if 10111 not in ids and 10104 not in ids else 1.0, 0.0)
    check("T_batch", 0.0 if PR["batch"]["N"] == [12, 15] and PR["batch"]["L"] == [2, 3]
          and PR["batch"]["arms"] == ["COLD", "WARM", "QAOA"]
          and PR["batch"]["Q"] == {"2": 2048, "3": 3072} else 1.0, 0.0)
    check("T_Q_rule", 0.0 if PR["frozen"]["Q_rule"].startswith("Q=1024*L") else 1.0, 0.0)
    x1 = np.array([0.3, -0.1, 0.05, 0.4, -0.2, 0.1])
    for L in (2, 3):
        p = W.pad_l1(x1, L)
        check(f"T_pad_len_L{L}", abs(p.size - (3 + 3 * L)), 0.0)
        check(f"T_pad_prefix_L{L}", float(np.max(np.abs(p[:6] - x1))), 0.0)
        check(f"T_pad_extra_zero_L{L}", float(np.max(np.abs(p[6:]))), 0.0)
    check("T_S1_L1_tqa", float(np.max(np.abs(ST.starts_for_n("Nov3", 10128, 15, 1, 1)[0] - N.tqa_nov3(1)))), 0.0)
    check("T_S1_nstarts", abs(len(ST.starts_for_n("Nov3", 10128, 15, 2, 1)) - 1), 0.0)
    check("T_K_1024", abs(SP.split_cap(1024, 1)[1] - 507), 0.0)
    check("T_K_2048", abs(SP.split_cap(2048, 1)[1] - 1019), 0.0)
    check("T_K_3072", abs(SP.split_cap(3072, 1)[1] - 1531), 0.0)
    check("T_share_L2", abs(1024 + 1024 - 2048), 0.0)
    check("T_share_L3", abs(1024 + 2048 - 3072), 0.0)
    check("T_warm_mid", abs(W.WARM_MID - 240), 0.0)
    ncall = {"n": 0}

    def dummy(x, ncall=ncall):
        ncall["n"] += 1
        return float(np.sum(np.square(x)))

    rng = lambda s: np.random.default_rng(s + 28)
    ncall["n"] = 0
    out1 = SP.spsa_run(dummy, [np.zeros(6)], np.full(6, 0.1), rng, 1024)
    rem = 1024
    rng = lambda s: np.random.default_rng(s + 29)
    outL = SP.spsa_run(dummy, [W.pad_l1(np.zeros(6), 2)], np.full(9, 0.1), rng, rem)
    check("T_warm_Q_L2", abs(out1["queries"] + outL["queries"] - 2048) + abs(ncall["n"] - 2048), 0.0)
    check("T_warm_K", abs(out1["K_per_start"] - 507) + abs(outL["K_per_start"] - 507), 0.0)
    check("T_live", 0.0 if (not out1["start_meta"][0]["gain_frozen"] and not outL["start_meta"][0]["gain_frozen"]) else 1.0, 0.0)
    check("T_gate_adopt", 0.0 if PR["gate"]["adopt_cell"] == "N15 L2" else 1.0, 0.0)
    check("T_no_gr", 0.0 if PR["not_a_gr_adoption"] and PR["not_a_new_ansatz"] else 1.0, 0.0)
    OUT.mkdir(parents=True, exist_ok=True)
    allpass = all(r["pass"] for r in R)
    (OUT / "TESTS_S1W.json").write_text(json.dumps({"n": len(R), "pass": allpass, "rows": R}))
    print(f"{'ALL PASS' if allpass else 'FAIL'} n={len(R)}")
    if not allpass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
