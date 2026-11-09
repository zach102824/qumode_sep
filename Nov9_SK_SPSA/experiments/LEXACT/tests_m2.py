"""M2 conformance vs reference and vs M1."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "N3COND"))
sys.path.insert(0, str(HERE.parent / "LE"))
import core as C  # noqa: E402
import kernel_m2 as K2  # noqa: E402
import nov3 as N  # noqa: E402
import sim as S  # noqa: E402

OUT = HERE.parents[1] / "results" / "LEXACT"
R = []


def check(name, err, tol):
    ok = err <= tol
    R.append({"name": name, "err": float(err), "tol": float(tol), "pass": bool(ok)})
    print(f"{name:48s} {err:.3e}  tol {tol}  {'PASS' if ok else 'FAIL'}")


def main():
    K2.warmup()
    check("T_threads_frozen", abs(K2.thread_meta()["numba_threads"] - 4), 0.0)
    rng = np.random.default_rng(0)
    for nq in (6, 9, 12):
        J = C.make_J(10010, nq)
        b = 0.4
        gs = np.ones(1 << nq)
        gs = gs / np.linalg.norm(gs)
        for L in range(1, 7):
            ref_n, m1_n, m2_n = N.Nov3(J, b, L), S.Nov3Fast(J, b, L), S.Nov3M2(J, b, L)
            ref_q, m1_q, m2_q = N.QAOA(J, b, L), S.QAOAFast(J, b, L), S.QAOAM2(J, b, L)
            for p in range(4):
                xn = rng.uniform(-0.5, 0.5, size=ref_n.nparam())
                xq = rng.uniform(-0.5, 0.5, size=ref_q.nparam())
                pn, p2 = ref_n.state(xn), m2_n.state(xn)
                check(f"T_nov3_state_N{nq}_L{L}_p{p}", float(np.max(np.abs(pn - p2))), 1e-12)
                check(f"T_nov3_vs_m1_N{nq}_L{L}_p{p}", float(np.max(np.abs(m1_n.state(xn) - p2))), 1e-12)
                check(f"T_nov3_E_N{nq}_L{L}_p{p}", abs(ref_n.energy(xn) - m2_n.energy(xn)), 1e-10)
                dn, d2 = ref_n.diagnostics(xn, gs), m2_n.diagnostics(xn, gs)
                check(f"T_nov3_F_N{nq}_L{L}_p{p}", abs(dn["F"] - d2["F"]), 1e-12)
                pq, q2 = ref_q.state(xq), m2_q.state(xq)
                check(f"T_qaoa_state_N{nq}_L{L}_p{p}", float(np.max(np.abs(pq - q2))), 1e-12)
                check(f"T_qaoa_E_N{nq}_L{L}_p{p}", abs(ref_q.energy(xq) - m2_q.energy(xq)), 1e-10)
                dq, dq2 = ref_q.diagnostics(xq, gs), m2_q.diagnostics(xq, gs)
                check(f"T_qaoa_F_N{nq}_L{L}_p{p}", abs(dq["F"] - dq2["F"]), 1e-12)
            x = N.tqa_nov3(L)
            gamma, theta, s = m2_n.unpack(x)
            xq_embed = np.concatenate([np.full(L, gamma), -0.5 * s[:, 0]])
            check(f"T_embed_E_N{nq}_L{L}", abs(m2_n.energy(x) - m2_q.energy(xq_embed)), 1e-10)
    pr = json.loads((HERE / "PREREG_LEXACT_M2.json").read_text())
    check("T_prereg_T4", 0.0 if pr["hypothesis"]["threads"] == 4 and pr["not_a_gr_adoption"] else 1.0, 0.0)
    OUT.mkdir(parents=True, exist_ok=True)
    allpass = all(r["pass"] for r in R)
    (OUT / "TESTS_M2.json").write_text(json.dumps({"n": len(R), "pass": allpass, "rows": R}))
    print(f"{'ALL PASS' if allpass else 'FAIL'} n={len(R)} threads={K2.thread_meta()}")
    if not allpass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
