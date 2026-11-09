"""Exact-kernel conformance against N3COND/nov3.py and LE/core.py QAOA."""
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
import kernel as K  # noqa: E402
import nov3 as N  # noqa: E402
import sim as S  # noqa: E402

OUT = HERE.parents[1] / "results" / "LEXACT"
R = []


def check(name, err, tol):
    ok = err <= tol
    R.append({"name": name, "err": float(err), "tol": float(tol), "pass": bool(ok)})
    print(f"{name:48s} {err:.3e}  tol {tol}  {'PASS' if ok else 'FAIL'}")


def main():
    K.warmup()
    rng = np.random.default_rng(0)
    for nq in (6, 9, 12):
        J = C.make_J(10010, nq)
        b = 0.4
        gs = np.ones(1 << nq)
        gs = gs / np.linalg.norm(gs)
        for L in range(1, 7):
            ref_n = N.Nov3(J, b, L)
            fast_n = S.Nov3Fast(J, b, L)
            ref_q = N.QAOA(J, b, L)
            fast_q = S.QAOAFast(J, b, L)
            for p in range(4):
                xn = rng.uniform(-0.5, 0.5, size=ref_n.nparam())
                xq = rng.uniform(-0.5, 0.5, size=ref_q.nparam())
                pn, pf = ref_n.state(xn), fast_n.state(xn)
                check(f"T_nov3_state_N{nq}_L{L}_p{p}", float(np.max(np.abs(pn - pf))), 1e-12)
                check(f"T_nov3_E_N{nq}_L{L}_p{p}", abs(ref_n.energy(xn) - fast_n.energy(xn)), 1e-10)
                dn, df = ref_n.diagnostics(xn, gs), fast_n.diagnostics(xn, gs)
                check(f"T_nov3_F_N{nq}_L{L}_p{p}", abs(dn["F"] - df["F"]), 1e-12)
                pq, qf = ref_q.state(xq), fast_q.state(xq)
                check(f"T_qaoa_state_N{nq}_L{L}_p{p}", float(np.max(np.abs(pq - qf))), 1e-12)
                check(f"T_qaoa_E_N{nq}_L{L}_p{p}", abs(ref_q.energy(xq) - fast_q.energy(xq)), 1e-10)
                dq, dfq = ref_q.diagnostics(xq, gs), fast_q.diagnostics(xq, gs)
                check(f"T_qaoa_F_N{nq}_L{L}_p{p}", abs(dq["F"] - dfq["F"]), 1e-12)
            # QAOA embedding of Nov3: theta=0, s2=s3=0
            x = N.tqa_nov3(L)
            e_n = fast_n.energy(x)
            xq = N.tqa_qaoa(L)
            # shared-gamma Nov3 TQA is not the same as independent-gamma QAOA TQA except the embedding identity:
            # theta=0 s2=s3=0 Nov3 with gamma_shared and s1_l = -2 beta_l equals QAOA with that shared gamma repeated.
            gamma, theta, s = fast_n.unpack(x)
            xq_embed = np.concatenate([np.full(L, gamma), -0.5 * s[:, 0]])
            check(f"T_embed_E_N{nq}_L{L}", abs(e_n - fast_q.energy(xq_embed)), 1e-10)
            psi_n, psi_r = fast_n.state(x), ref_n.state_plus_rewrite(x)
            # global phase from SNAP vs rewrite is already matched in the reference tests; here match energy
            check(f"T_plus_rewrite_E_N{nq}_L{L}", abs(fast_n.energy(x) - ref_n.energy_of_state(psi_r)), 1e-10)
    pr = json.loads((HERE / "PREREG_LEXACT_M1.json").read_text())
    check("T_prereg_not_gr", 0.0 if pr["not_a_gr_adoption"] else 1.0, 0.0)
    OUT.mkdir(parents=True, exist_ok=True)
    allpass = all(r["pass"] for r in R)
    (OUT / "TESTS.json").write_text(json.dumps({"n": len(R), "pass": allpass, "rows": R}))
    print(f"{'ALL PASS' if allpass else 'FAIL'} n={len(R)}")
    if not allpass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
