"""LEXACT M2 timings and short inclusive SPSA traces.

    OMP_NUM_THREADS=1 python Nov9_SK_SPSA/experiments/LEXACT/run_m2.py
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["NUMBA_NUM_THREADS"] = "4"

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "N3COND"))
import kernel as K  # noqa: E402
import kernel_m2 as K2  # noqa: E402
import nov3 as N  # noqa: E402
import run_lexact as R1  # noqa: E402
import sim as S  # noqa: E402

PREREG_PATH = HERE / "PREREG_LEXACT_M2.json"
PREREG = json.loads(PREREG_PATH.read_text())
OUT = HERE.parents[1] / "results" / "LEXACT"


def prereg_sha():
    return hashlib.sha256(PREREG_PATH.read_bytes()).hexdigest()


def rss_mb():
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return float(line.split()[1]) / 1024.0
    return float("nan")


def time_calls(fn, n, warmup=2):
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t0) / n


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    status = {"pid": os.getpid(), "stage": "LEXACT_M2", "started": time.time(),
              "prereg_sha256": prereg_sha(), "prior_n3cond": 772224}
    (OUT / "STATUS_M2.json").write_text(json.dumps(status))
    t0 = time.perf_counter()
    K.warmup()
    compile_m1 = time.perf_counter() - t0
    t0 = time.perf_counter()
    K2.warmup()
    compile_m2 = time.perf_counter() - t0
    print(f"[m2] pid {os.getpid()} prereg {prereg_sha()[:12]} {K2.thread_meta()} "
          f"compile m1 {compile_m1:.3f}s m2 {compile_m2:.3f}s rss {rss_mb():.1f}MB", flush=True)

    rec = {
        "pid": os.getpid(), "prereg_sha256": prereg_sha(),
        "compile_m1_s": compile_m1, "compile_m2_s": compile_m2,
        "threads": K2.thread_meta(), "rss_mb_start": rss_mb(),
        "prior_n3cond_ledger": 772224, "lexact_m1_queries_preserved": 768,
        "timings": [], "spsa": [],
    }
    cells = [(6, 1, 10010), (6, 6, 10010), (9, 1, 10010), (9, 6, 10010),
             (12, 1, 10010), (12, 6, 10010), (15, 1, 10032), (15, 3, 10032), (18, 1, 10032)]
    for nq, L, seed in cells:
        print(f"[m2] N{nq} L{L} seed {seed}", flush=True)
        t_prep = time.perf_counter()
        J, meta, gs = R1.instance(nq, seed)
        prep_s = time.perf_counter() - t_prep
        b = float(meta["Bstar"])
        rng = np.random.default_rng([seed, nq, L, 2])
        nrep = 8 if nq <= 12 else (4 if nq == 15 else 2)
        row = {"N": nq, "L": L, "seed": seed, "Bstar": b, "field_prep_s": prep_s,
               "source": meta.get("source", "built"), "dim": 1 << nq,
               "rss_mb": rss_mb(), "nrep": nrep, "circuits": {}}
        for circ in ("Nov3", "QAOA"):
            ref = S.make_sim(circ, J, b, L, engine="ref")
            m1 = S.make_sim(circ, J, b, L, engine="m1")
            m2 = S.make_sim(circ, J, b, L, engine="m2")
            x = rng.uniform(-0.4, 0.4, size=ref.nparam())
            t_ref = time_calls(lambda: ref.energy(x), nrep)
            t_m1 = time_calls(lambda: m1.energy(x), nrep)
            t_m2 = time_calls(lambda: m2.energy(x), nrep)
            eref, e1, e2 = ref.energy(x), m1.energy(x), m2.energy(x)
            pref, p2 = ref.state(x), m2.state(x)
            dref, d2 = ref.diagnostics(x, gs), m2.diagnostics(x, gs)
            row["circuits"][circ] = {
                "t_ref_s": t_ref, "t_m1_s": t_m1, "t_m2_s": t_m2,
                "speedup_vs_ref": t_ref / t_m2, "speedup_vs_m1": t_m1 / t_m2,
                "dE_ref": abs(eref - e2), "dE_m1": abs(e1 - e2),
                "dstate": float(np.max(np.abs(pref - p2))),
                "dF": abs(dref["F"] - d2["F"]),
                "E_ref": eref, "F_ref": dref["F"], "F_m2": d2["F"],
            }
            print(f"  {circ} ref {t_ref*1e3:.2f} m1 {t_m1*1e3:.2f} m2 {t_m2*1e3:.2f} ms "
                  f"vsM1 x{t_m1/t_m2:.2f} dE {abs(eref-e2):.2e} dstate {np.max(np.abs(pref-p2)):.2e}",
                  flush=True)
        rec["timings"].append(row)

    for job in PREREG["batch"]["spsa_short"]:
        nq, L, seed, Q = job["N"], job["L"], job["seed"], job["Q"]
        print(f"[m2] SPSA N{nq} L{L} Q={Q}", flush=True)
        J, meta, gs = R1.instance(nq, seed)
        b = float(meta["Bstar"])
        for circ in job["circuits"]:
            ref = S.make_sim(circ, J, b, L, engine="ref")
            m2 = S.make_sim(circ, J, b, L, engine="m2")
            starts = N.starts_for(circ, seed, nq, L)
            cid = N.circuit_id(circ)
            rngs = lambda s, seed=seed, nq=nq, L=L, cid=cid: N.rng_for(seed, nq, L, cid, s, 0)
            t0 = time.perf_counter()
            rref = R1.spsa_traced(ref.energy, starts, ref.sigma0(), rngs, Q)
            t_ref = time.perf_counter() - t0
            t0 = time.perf_counter()
            rfast = R1.spsa_traced(m2.energy, starts, m2.sigma0(), rngs, Q)
            t_fast = time.perf_counter() - t0
            tr = np.asarray(rref["trace"])
            tf = np.asarray(rfast["trace"])
            d = {
                "N": nq, "L": L, "seed": seed, "circuit": circ, "Q": Q,
                "queries": Q, "outside_cap": 0, "t_ref_s": t_ref, "t_m2_s": t_fast,
                "speedup": t_ref / t_fast,
                "max_abs_dE_trace": float(np.max(np.abs(tr - tf))),
                "E_sel_ref": rref["E_sel"], "E_sel_m2": rfast["E_sel"],
                "same_start": rref["start"] == rfast["start"],
                "same_cand": rref["cand"] == rfast["cand"],
                "F_ref": ref.diagnostics(rref["x"], gs)["F"],
                "F_m2": m2.diagnostics(rfast["x"], gs)["F"],
                "selection": "energy_only",
            }
            rec["spsa"].append(d)
            print(f"  {circ} trace max|dE| {d['max_abs_dE_trace']:.2e} x{d['speedup']:.2f}", flush=True)

    rec["rss_mb_end"] = rss_mb()
    rec["finished"] = time.time()
    rec["wall_s"] = rec["finished"] - status["started"]
    rec["new_queries"] = int(sum(s["Q"] for s in rec["spsa"]))
    rec["gr_note"] = "GR frozen recipe unchanged. ~2200h figure is a fixed-field probe, not B* completion."
    (OUT / "summary_M2.json").write_text(json.dumps(rec, indent=1))
    status.update({"finished": rec["finished"], "wall_s": rec["wall_s"], "new_queries": rec["new_queries"]})
    (OUT / "STATUS_M2.json").write_text(json.dumps(status))
    print("[m2] done", rec["wall_s"], "queries", rec["new_queries"], flush=True)


if __name__ == "__main__":
    main()
