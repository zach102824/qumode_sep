"""M3: spawn-pool results match serial T=1 on N9/N12."""
from __future__ import annotations

import json
import os
import sys
from multiprocessing import get_context
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["NUMBA_NUM_THREADS"] = "1"

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "N3COND"))
import kernel_m2 as K2  # noqa: E402
import nov3 as N  # noqa: E402
import pool as P  # noqa: E402
import run_lexact as R1  # noqa: E402
import sim as S  # noqa: E402

OUT = HERE.parents[1] / "results" / "LEXACT"
R = []


def check(name, err, tol):
    ok = err <= tol
    R.append({"name": name, "err": float(err), "tol": float(tol), "pass": bool(ok)})
    print(f"{name:48s} {err:.3e}  tol {tol}  {'PASS' if ok else 'FAIL'}")


def main():
    os.environ["NUMBA_NUM_THREADS"] = "1"
    K2.set_threads(1)
    K2.warmup()
    J, meta, gs = R1.instance(9, 10010)
    b = meta["Bstar"]
    x = N.tqa_nov3(1)
    serial = S.make_sim("Nov3", J, b, 1, engine="m2")
    eref = serial.energy(x)
    pref = serial.state(x)
    dref = serial.diagnostics(x, gs)
    task = {"N": 9, "seed": 10010, "L": 1, "circuit": "Nov3", "x": x}
    ctx = get_context("spawn")
    with ctx.Pool(2, initializer=P.init_worker) as pool:
        out = pool.map(P.job_energy, [task, task])
    check("T_pool_E", abs(out[0]["E"] - eref), 1e-10)
    check("T_pool_F", abs(out[0]["F"] - dref["F"]), 1e-12)
    check("T_pool_repeat", abs(out[0]["E"] - out[1]["E"]), 1e-12)
    check("T_state0", abs(out[0]["state0"][0] - float(pref[0].real)) + abs(out[0]["state0"][1] - float(pref[0].imag)), 1e-12)
    check("T_worker_T1", abs(out[0]["threads"]["numba_threads"] - 1), 0.0)
    rec = R1.spsa_traced(serial.energy, N.starts_for("Nov3", 10010, 9, 1),
                         serial.sigma0(), lambda s: N.rng_for(10010, 9, 1, 40, s, 0), 128)
    stask = {"N": 9, "seed": 10010, "L": 1, "circuit": "Nov3", "Q": 128}
    with ctx.Pool(1, initializer=P.init_worker) as pool:
        sp = pool.apply(P.job_spsa, (stask,))
    tr = np.asarray(rec["trace"])
    tf = np.asarray(sp["trace"])
    check("T_spsa_trace", float(np.max(np.abs(tr - tf))), 1e-10)
    check("T_spsa_start", 0.0 if rec["start"] == sp["start"] and rec["cand"] == sp["cand"] else 1.0, 0.0)
    check("T_spsa_Q", abs(sp["queries"] - 128), 0.0)
    pr = json.loads((HERE / "PREREG_LEXACT_M3.json").read_text())
    check("T_prereg_K4", 0.0 if pr["hypothesis"]["K"] == 4 and pr["not_a_gr_adoption"] else 1.0, 0.0)
    OUT.mkdir(parents=True, exist_ok=True)
    allpass = all(r["pass"] for r in R)
    (OUT / "TESTS_M3.json").write_text(json.dumps({"n": len(R), "pass": allpass, "rows": R}))
    print(f"{'ALL PASS' if allpass else 'FAIL'} n={len(R)}")
    if not allpass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
