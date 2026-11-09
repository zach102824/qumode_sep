"""S4 matches C0. S8 is equal-budget 8 starts. Q inclusive."""
from __future__ import annotations

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
import kernel_m2 as K2  # noqa: E402
import nov3 as N  # noqa: E402
import sim as S  # noqa: E402
import spsa as SP  # noqa: E402
import starts as U  # noqa: E402
import core as C  # noqa: E402

OUT = HERE.parents[1] / "results" / "STARTS"
R = []


def check(name, err, tol):
    ok = err <= tol
    R.append({"name": name, "err": float(err), "tol": float(tol), "pass": bool(ok)})
    print(f"{name:48s} {err:.3e}  tol {tol}  {'PASS' if ok else 'FAIL'}")


def main():
    K2.set_threads(1)
    try:
        K2.warmup()
    except ValueError:
        pass
    s4 = U.starts_for_n("Nov3", 10048, 15, 3, 4)
    hist = N.starts_for("Nov3", 10048, 15, 3)
    check("T_S4_matches_C0_nov3", max(float(np.max(np.abs(a - b))) for a, b in zip(s4, hist)), 0.0)
    s4q = U.starts_for_n("QAOA", 10048, 15, 3, 4)
    histq = N.starts_for("QAOA", 10048, 15, 3)
    check("T_S4_matches_C0_qaoa", max(float(np.max(np.abs(a - b))) for a, b in zip(s4q, histq)), 0.0)
    s8 = U.starts_for_n("Nov3", 10048, 15, 3, 8)
    check("T_S8_len", abs(len(s8) - 8), 0.0)
    check("T_S8_prefix_is_S4", max(float(np.max(np.abs(s8[i] - s4[i]))) for i in range(4)), 0.0)
    check("T_S8_nparam", abs(s8[7].size - 12), 0.0)
    per4, k4 = SP.split_cap(3072, 4)
    per8, k8 = SP.split_cap(3072, 8)
    check("T_S4_K", abs(per4 - 768) + abs(k4 - 379), 0.0)
    check("T_S8_K", abs(per8 - 384) + abs(k8 - 187), 0.0)
    check("T_equal_budget", abs(4 * per4 - 8 * per8) + abs(4 * per4 - 3072), 0.0)
    ncall = {"n": 0}

    def dummy(x, ncall=ncall):
        ncall["n"] += 1
        return float(np.sum(np.square(x)))

    sig = np.full(12, 0.1)
    rng = lambda s: np.random.default_rng(s + 11)
    for nst, cap, kexp, ncal_exp, ncand_exp in ((4, 3072, 379, 32, 8), (8, 3072, 187, 64, 16)):
        ncall["n"] = 0
        starts = [np.zeros(12) + 0.01 * i for i in range(nst)]
        out = SP.spsa_run(dummy, starts, sig, rng, cap)
        check(f"T_queries_{nst}", abs(out["queries"] - cap) + abs(ncall["n"] - cap), 0.0)
        check(f"T_K_{nst}", abs(out["K_per_start"] - kexp), 0.0)
        check(f"T_cal_{nst}", abs(out["query_cat"]["calibration"] - ncal_exp), 0.0)
        check(f"T_cand_{nst}", abs(out["query_cat"]["candidate"] - ncand_exp), 0.0)
        check(f"T_gain_live_{nst}", 0.0 if not any(m["gain_frozen"] for m in out["start_meta"]) else 1.0, 0.0)
    J = C.make_J(10048, 6)
    for circ in ("Nov3", "QAOA"):
        ref = S.make_sim(circ, J, 0.4, 3, engine="ref")
        m2 = S.make_sim(circ, J, 0.4, 3, engine="m2")
        x = U.starts_for_n(circ, 10048, 6, 3, 4)[0]
        check(f"T_m2_ref_{circ}", abs(ref.energy(x) - m2.energy(x)), 1e-10)
    pr = json.loads((HERE / "PREREG_STARTS_S8.json").read_text())
    ids = pr["seeds"]["ids"]
    check("T_prereg_seeds", 0.0 if ids == list(range(10048, 10056)) and 10010 not in ids else 1.0, 0.0)
    check("T_prereg_gate", 0.0 if pr["gate"]["cell"].startswith("N15 L3 Nov3") else 1.0, 0.0)
    check("T_no_gr", 0.0 if pr["not_a_gr_adoption"] and "Do not patch GR" in pr["gate"]["gr"] else 1.0, 0.0)
    OUT.mkdir(parents=True, exist_ok=True)
    allpass = all(r["pass"] for r in R)
    (OUT / "TESTS.json").write_text(json.dumps({"n": len(R), "pass": allpass, "rows": R}))
    print(f"{'ALL PASS' if allpass else 'FAIL'} n={len(R)}")
    if not allpass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
