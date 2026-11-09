"""LEXACT M3: K=4 instance-parallel N18 energy + Q=128 SPSA.

    OMP_NUM_THREADS=1 python Nov9_SK_SPSA/experiments/LEXACT/run_m3.py
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from multiprocessing import get_context
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
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

PREREG_PATH = HERE / "PREREG_LEXACT_M3.json"
PREREG = json.loads(PREREG_PATH.read_text())
OUT = HERE.parents[1] / "results" / "LEXACT"
K_WORKERS = 4


def prereg_sha():
    return hashlib.sha256(PREREG_PATH.read_bytes()).hexdigest()


def rss_mb():
    return P.rss_mb()


def pick_seeds():
    want = list(PREREG["seeds"]["exclusive_dev"])
    fb = list(PREREG["seeds"]["fallback_in_band"])
    out, notes = [], []
    for seed in want + fb:
        if len(out) == 4:
            break
        J, meta, gs = R1.instance(18, seed)
        ok = bool(meta.get("accepted_lexact") or meta.get("valid")) and meta.get("E1_parity") == "odd"
        notes.append({"seed": seed, "accepted": bool(ok), "Bstar": meta.get("Bstar"),
                      "prep_s": meta.get("field_prep_s"), "source": meta.get("source", "built")})
        if ok:
            out.append(seed)
        else:
            notes[-1]["skipped"] = True
    if len(out) != 4:
        raise RuntimeError(f"need 4 accepted N18 B* seeds, got {out}")
    return out, notes


def serial_energy(nq, seed, L, circ, x):
    J, meta, gs = R1.instance(nq, seed)
    sim = S.make_sim(circ, J, meta["Bstar"], L, engine="m2")
    ref = S.make_sim(circ, J, meta["Bstar"], L, engine="ref")
    t0 = time.perf_counter()
    e = float(sim.energy(x))
    d = sim.diagnostics(x, gs)
    psi = sim.state(x)
    return {"E": e, "F": d["F"], "state0": [float(psi[0].real), float(psi[0].imag)],
            "E_ref": float(ref.energy(x)), "F_ref": ref.diagnostics(x, gs)["F"],
            "wall_s": time.perf_counter() - t0}


def serial_spsa(nq, seed, L, circ, Q):
    J, meta, gs = R1.instance(nq, seed)
    sim = S.make_sim(circ, J, meta["Bstar"], L, engine="m2")
    starts = N.starts_for(circ, seed, nq, L)
    cid = N.circuit_id(circ)
    rngs = lambda s, seed=seed, nq=nq, L=L, cid=cid: N.rng_for(seed, nq, L, cid, s, 0)
    t0 = time.perf_counter()
    out = R1.spsa_traced(sim.energy, starts, sim.sigma0(), rngs, Q)
    d = sim.diagnostics(out["x"], gs)
    return {"E_sel": out["E_sel"], "start": out["start"], "cand": out["cand"],
            "trace": out["trace"], "queries": out["queries"], "F": d["F"],
            "wall_s": time.perf_counter() - t0}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    status = {"pid": os.getpid(), "stage": "LEXACT_M3", "started": time.time(),
              "prereg_sha256": prereg_sha()}
    (OUT / "STATUS_M3.json").write_text(json.dumps(status))
    K2.set_threads(1)
    t0 = time.perf_counter()
    K2.warmup()
    compile_s = time.perf_counter() - t0
    print(f"[m3] pid {os.getpid()} K={K_WORKERS} T=1 compile {compile_s:.3f}s rss {rss_mb():.1f}MB", flush=True)

    t_prep = time.perf_counter()
    seeds, seed_notes = pick_seeds()
    prep_s = time.perf_counter() - t_prep
    print(f"[m3] seeds {seeds} field_prep_wall {prep_s:.1f}s", flush=True)

    rng = np.random.default_rng(18)
    energy_tasks = []
    for seed in seeds:
        J, meta, _ = R1.instance(18, seed)
        sim = S.make_sim("Nov3", J, meta["Bstar"], 1, engine="m2")
        x = rng.uniform(-0.4, 0.4, size=sim.nparam())
        energy_tasks.append({"N": 18, "seed": seed, "L": 1, "circuit": "Nov3", "x": x})

    t0 = time.perf_counter()
    serial_e = [serial_energy(**{k: t[k] for k in ("N", "seed", "L", "circuit")}, x=t["x"]) for t in energy_tasks]
    serial_e_wall = time.perf_counter() - t0
    ctx = get_context("spawn")
    t0 = time.perf_counter()
    with ctx.Pool(K_WORKERS, initializer=P.init_worker) as pool:
        pool_e = pool.map(P.job_energy, energy_tasks)
    pool_e_wall = time.perf_counter() - t0

    spsa_tasks = [{"N": 18, "seed": s, "L": 1, "circuit": c, "Q": 128}
                  for s in seeds for c in ("Nov3", "QAOA")]
    t0 = time.perf_counter()
    serial_s = [serial_spsa(t["N"], t["seed"], t["L"], t["circuit"], t["Q"]) for t in spsa_tasks]
    serial_s_wall = time.perf_counter() - t0
    t0 = time.perf_counter()
    with ctx.Pool(K_WORKERS, initializer=P.init_worker) as pool:
        pool_s = list(pool.imap_unordered(P.job_spsa, spsa_tasks))
    pool_s_wall = time.perf_counter() - t0
    pool_s.sort(key=lambda r: (r["seed"], r["circuit"]))
    serial_s_sorted = sorted(zip(spsa_tasks, serial_s), key=lambda z: (z[0]["seed"], z[0]["circuit"]))
    serial_s = [z[1] for z in serial_s_sorted]

    e_gate, s_gate = [], []
    for a, b in zip(serial_e, pool_e):
        e_gate.append({"seed": energy_tasks[serial_e.index(a)]["seed"] if a in serial_e else None,
                       "dE": abs(a["E"] - b["E"]), "dF": abs(a["F"] - b["F"]),
                       "dstate0": abs(a["state0"][0] - b["state0"][0]) + abs(a["state0"][1] - b["state0"][1])})
    # zip energy_tasks order: pool.map preserves order
    e_gate = []
    for t, a, b in zip(energy_tasks, serial_e, pool_e):
        e_gate.append({"seed": t["seed"], "dE": abs(a["E"] - b["E"]), "dF": abs(a["F"] - b["F"]),
                       "dE_ref": abs(a["E_ref"] - b["E"]), "dF_ref": abs(a["F_ref"] - b["F"]),
                       "dstate0": abs(a["state0"][0] - b["state0"][0]) + abs(a["state0"][1] - b["state0"][1]),
                       "worker_pid": b["pid"], "worker_rss_mb": b["rss_mb"], "worker_threads": b["threads"]})
    for a, b in zip(serial_s, pool_s):
        tr, tf = np.asarray(a["trace"]), np.asarray(b["trace"])
        s_gate.append({
            "seed": b["seed"], "circuit": b["circuit"], "queries": b["queries"],
            "same_start": a["start"] == b["start"], "same_cand": a["cand"] == b["cand"],
            "max_abs_dE_trace": float(np.max(np.abs(tr - tf))),
            "E_sel_serial": a["E_sel"], "E_sel_pool": b["E_sel"],
            "F_serial": a["F"], "F_pool": b["F"], "worker_pid": b["pid"],
            "worker_rss_mb": b["rss_mb"],
        })
    n_inst = len(seeds)
    n_fits = len(spsa_tasks)
    rec = {
        "pid": os.getpid(), "prereg_sha256": prereg_sha(), "compile_s": compile_s,
        "K": K_WORKERS, "kernel_threads": 1, "blas": 1, "queue": "spawn Pool imap_unordered",
        "seeds": seeds, "seed_notes": seed_notes, "field_prep_wall_s": prep_s,
        "rss_mb_parent": rss_mb(),
        "energy": {
            "n": len(energy_tasks), "serial_wall_s": serial_e_wall, "pool_wall_s": pool_e_wall,
            "speedup": serial_e_wall / pool_e_wall, "gates": e_gate,
            "evals_per_hour_serial": 3600.0 * len(energy_tasks) / serial_e_wall,
            "evals_per_hour_pool": 3600.0 * len(energy_tasks) / pool_e_wall,
        },
        "spsa": {
            "n_instances": n_inst, "n_fits": n_fits, "Q": 128,
            "serial_wall_s": serial_s_wall, "pool_wall_s": pool_s_wall,
            "speedup": serial_s_wall / pool_s_wall,
            "instances_per_hour_serial": 3600.0 * n_inst / serial_s_wall,
            "instances_per_hour_pool": 3600.0 * n_inst / pool_s_wall,
            "fits_per_hour_serial": 3600.0 * n_fits / serial_s_wall,
            "fits_per_hour_pool": 3600.0 * n_fits / pool_s_wall,
            "queries_new": n_fits * 128,
            "outside_cap": 0,
            "gates": s_gate,
        },
        "conformance": {
            "energy_all_pass": all(g["dE"] <= 1e-10 and g["dF"] <= 1e-12 and g["dE_ref"] <= 1e-10 for g in e_gate),
            "spsa_all_pass": all(g["same_start"] and g["same_cand"] and g["max_abs_dE_trace"] <= 1e-10 for g in s_gate),
        },
        "prior_n3cond": 772224, "lexact_m1_m2_spsa": 1536,
        "gr_note": "GR frozen recipe unchanged.",
    }
    rec["wall_s"] = time.time() - status["started"]
    rec["finished"] = time.time()
    (OUT / "summary_M3.json").write_text(json.dumps(rec, indent=1))
    status.update({"finished": rec["finished"], "wall_s": rec["wall_s"],
                   "new_queries": rec["spsa"]["queries_new"]})
    (OUT / "STATUS_M3.json").write_text(json.dumps(status))
    print(json.dumps({"instances_per_hour_pool": rec["spsa"]["instances_per_hour_pool"],
                      "speedup_spsa": rec["spsa"]["speedup"],
                      "conformance": rec["conformance"],
                      "queries": rec["spsa"]["queries_new"]}, indent=1), flush=True)


if __name__ == "__main__":
    main()
