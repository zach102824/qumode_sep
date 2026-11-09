"""Pad identity and charged split. QAOA extra layers with 0 angles are identity."""
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
sys.path.insert(0, str(HERE.parent / "LE"))
sys.path.insert(0, str(HERE.parent / "LEXACT"))
import cont as U  # noqa: E402
import core as C  # noqa: E402
import kernel_m2 as K2  # noqa: E402
import nov3 as N  # noqa: E402
import sim as S  # noqa: E402
import spsa as SP  # noqa: E402

OUT = HERE.parents[1] / "results" / "DEPTH"
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
    J = C.make_J(10036, 6)
    b = 0.4
    x1n = np.array([0.3, -0.1, 0.05, 0.4, -0.2, 0.1])
    x1q = np.array([0.25, -0.3])
    for L in (2, 3):
        pn = U.pad_l1("Nov3", x1n, L)
        pq = U.pad_l1("QAOA", x1q, L)
        check(f"T_nov3_len_L{L}", abs(pn.size - (3 + 3 * L)), 0.0)
        check(f"T_qaoa_len_L{L}", abs(pq.size - 2 * L), 0.0)
        check(f"T_nov3_prefix_L{L}", float(np.max(np.abs(pn[:6] - x1n))), 0.0)
        qn = S.make_sim("QAOA", J, b, L, engine="m2")
        q1 = S.make_sim("QAOA", J, b, 1, engine="m2")
        check(f"T_qaoa_pad_identity_E_L{L}", abs(qn.energy(pq) - q1.energy(x1q)), 1e-10)
        pref = q1.state(x1q)
        pst = qn.state(pq)
        check(f"T_qaoa_pad_identity_state_L{L}", float(np.max(np.abs(pref - pst))), 1e-12)
        refq = S.make_sim("QAOA", J, b, L, engine="ref")
        check(f"T_qaoa_m2_vs_ref_L{L}", abs(qn.energy(pq) - refq.energy(pq)), 1e-10)
        nn = S.make_sim("Nov3", J, b, L, engine="m2")
        nref = S.make_sim("Nov3", J, b, L, engine="ref")
        check(f"T_nov3_m2_vs_ref_L{L}", abs(nn.energy(pn) - nref.energy(pn)), 1e-10)
        starts = U.continuation_starts("Nov3", 10036, 6, L, x1n)
        check(f"T_cont_start0_is_pad_L{L}", float(np.max(np.abs(starts[0] - pn))), 0.0)
        idx = U.new_layer_idx("Nov3", L)
        hold = [i for i in range(pn.size) if i not in idx]
        dmax = max(float(np.max(np.abs(starts[s][hold] - pn[hold]))) for s in range(1, 4))
        check(f"T_cont_keeps_l1_L{L}", dmax, 0.0)
    per, K = SP.split_cap(1024)
    check("T_split_1024", abs(per - 256) + abs(K - 123), 0.0)
    per, K = SP.split_cap(2048)
    check("T_split_2048", abs(per - 512) + abs(K - 251), 0.0)
    check("T_Q_L2", abs(1024 + 1024 - 2048), 0.0)
    check("T_Q_L3", abs(1024 + 2048 - 3072), 0.0)
    pr = json.loads((HERE / "PREREG_DEPTH_C1.json").read_text())
    check("T_prereg_seeds", 0.0 if pr["seeds"]["ids"][0] == 10036 and 10010 not in pr["seeds"]["ids"] else 1.0, 0.0)
    check("T_prereg_gate_N15L3", 0.0 if pr["gate"]["cell"].startswith("N15 L3 Nov3") else 1.0, 0.0)
    ncall = {"n": 0}

    def dummy(x, ncall=ncall):
        ncall["n"] += 1
        return float(np.sum(np.square(x)))

    rng = lambda s: np.random.default_rng(s + 7)
    x0 = np.zeros(6)
    starts = [x0, x0 + 0.1, x0 + 0.2, x0 + 0.3]
    sig = np.full(6, 0.1)
    for cap in (1024, 2048, 3072):
        ncall["n"] = 0
        out = SP.spsa_run(dummy, starts, sig, rng, cap)
        check(f"T_spsa_queries_eq_cap_{cap}", abs(out["queries"] - cap) + abs(ncall["n"] - cap), 0.0)
        check(f"T_spsa_gain_not_frozen_{cap}", 0.0 if not any(m["gain_frozen"] for m in out["start_meta"]) else 1.0, 0.0)
    OUT.mkdir(parents=True, exist_ok=True)
    allpass = all(r["pass"] for r in R)
    (OUT / "TESTS.json").write_text(json.dumps({"n": len(R), "pass": allpass, "rows": R}))
    print(f"{'ALL PASS' if allpass else 'FAIL'} n={len(R)}")
    if not allpass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
