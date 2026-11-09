"""Spawn workers for independent instances. BLAS=1, kernel T=1. No shared state."""
from __future__ import annotations

import os
import time


def init_worker():
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    os.environ["NUMBA_NUM_THREADS"] = "1"


_CACHE = None


def _engine():
    global _CACHE
    if _CACHE is None:
        init_worker()
        import kernel_m2 as K2
        import nov3 as N
        import run_lexact as R1
        import sim as S
        K2.set_threads(1)
        K2.warmup()
        _CACHE = (K2, N, R1, S)
    return _CACHE


def rss_mb():
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return float(line.split()[1]) / 1024.0
    return float("nan")


def job_energy(task):
    nq, seed, L, circ, x = task["N"], task["seed"], task["L"], task["circuit"], task["x"]
    K2, N, R1, S = _engine()
    J, meta, gs = R1.instance(nq, seed)
    sim = S.make_sim(circ, J, meta["Bstar"], L, engine="m2")
    t0 = time.perf_counter()
    e = float(sim.energy(x))
    d = sim.diagnostics(x, gs)
    psi = sim.state(x)
    return {
        "kind": "energy", "N": nq, "seed": seed, "L": L, "circuit": circ,
        "E": e, "F": d["F"], "state0": [float(psi[0].real), float(psi[0].imag)],
        "wall_s": time.perf_counter() - t0, "pid": os.getpid(), "rss_mb": rss_mb(),
        "threads": K2.thread_meta(),
    }


def job_spsa(task):
    nq, seed, L, circ, Q = task["N"], task["seed"], task["L"], task["circuit"], task["Q"]
    K2, N, R1, S = _engine()
    J, meta, gs = R1.instance(nq, seed)
    sim = S.make_sim(circ, J, meta["Bstar"], L, engine="m2")
    starts = N.starts_for(circ, seed, nq, L)
    cid = N.circuit_id(circ)
    rngs = lambda s, seed=seed, nq=nq, L=L, cid=cid: N.rng_for(seed, nq, L, cid, s, 0)
    t0 = time.perf_counter()
    out = R1.spsa_traced(sim.energy, starts, sim.sigma0(), rngs, Q)
    d = sim.diagnostics(out["x"], gs)
    return {
        "kind": "spsa", "N": nq, "seed": seed, "L": L, "circuit": circ, "Q": Q,
        "E_sel": out["E_sel"], "start": out["start"], "cand": out["cand"],
        "trace": out["trace"], "queries": out["queries"], "outside_cap": 0,
        "F": d["F"], "wall_s": time.perf_counter() - t0, "pid": os.getpid(),
        "rss_mb": rss_mb(), "threads": K2.thread_meta(),
        "query_cat": out["query_cat"],
    }
