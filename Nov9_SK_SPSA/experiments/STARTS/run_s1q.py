"""STARTS S1Q: frozen S1-only Nov3 vs QAOA at N15 L=3, Q in {3072,6144,12288}.

    OMP_NUM_THREADS=1 python Nov9_SK_SPSA/experiments/STARTS/run_s1q.py
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
sys.path.insert(0, str(HERE.parent / "LEXACT"))
sys.path.insert(0, str(HERE.parent / "LE"))
sys.path.insert(0, str(HERE.parent / "IMAX"))
import bstar_sparse as BS  # noqa: E402
import core as C  # noqa: E402
import kernel_m2 as K2  # noqa: E402
import sim as S  # noqa: E402
import spsa as SP  # noqa: E402
import starts as U  # noqa: E402

PREREG_PATH = HERE / "PREREG_STARTS_S1Q.json"
PREREG = json.loads(PREREG_PATH.read_text())
OUT = HERE.parents[1] / "results" / "STARTS"
INST = OUT / "instances"
RUNS = OUT / "runs"
QS = [int(q) for q in PREREG["batch"]["Q"]]
LFIX = int(PREREG["batch"]["L"][0])
NFIX = int(PREREG["batch"]["N"][0])
KEXP = {3072: 1531, 6144: 3067, 12288: 6139}
HEAVY_S = 1800.0


def prereg_sha():
    return hashlib.sha256(PREREG_PATH.read_bytes()).hexdigest()


def run_path(nq, seed, L, circ, Q):
    return RUNS / f"N{nq}" / f"s{seed}_L{L}_{circ}_S1_Q{Q}.json"


def accept(meta):
    if not (meta.get("valid") and float(meta.get("gs_resid", 1)) < 1e-9):
        return False
    ge = meta.get("gap_even")
    go = meta.get("gap_odd")
    if ge is not None and not (float(ge) > 0):
        return False
    if go is None:
        return meta.get("E1_parity") == "odd"
    if float(go) <= 0:
        return False
    e1 = float(meta.get("gap_E1_E0", go))
    return bool(float(go) <= e1 + 1e-8)


def instance(nq, seed):
    INST.mkdir(parents=True, exist_ok=True)
    pj, pz = INST / f"N{nq}_s{seed}_Bstar.json", INST / f"N{nq}_s{seed}_Bstar.npz"
    if pj.exists() and pz.exists():
        meta = json.loads(pj.read_text())
        J = C.make_J(seed, nq)
        assert meta["J_sha256"] == C.J_sha(J)
        meta = dict(meta)
        meta["accepted"] = accept(meta)
        return J, meta, np.load(pz)["gs"]
    t0 = time.perf_counter()
    J, meta, gs = BS.build(nq, seed)
    meta = dict(meta)
    meta["accepted"] = accept(meta)
    meta["field_prep_s"] = time.perf_counter() - t0
    meta["gs_prep"] = "classical, not charged to Q"
    meta["band"] = "exclusive_unread_dev_10072_10079"
    slim = {k: v for k, v in meta.items() if k != "J_upper"}
    pj.write_text(json.dumps(slim))
    np.savez_compressed(pz, gs=gs)
    return J, meta, gs


def warmup_worker():
    try:
        K2.warmup()
    except ValueError:
        pass
    try:
        K2.set_threads(1)
    except ValueError:
        pass


def run_one(task):
    nq, seed, L, circ, Q = task
    path = run_path(nq, seed, L, circ, Q)
    if path.exists():
        return str(path), "skip"
    path.parent.mkdir(parents=True, exist_ok=True)
    warmup_worker()
    t0 = time.time()
    J, meta, gs = instance(nq, seed)
    rec = {
        "N": nq, "seed": seed, "L": L, "circuit": circ, "arm": "S1", "nstarts": 1,
        "q": None, "Q": Q, "band": "exclusive_unread_dev", "Bstar": meta["Bstar"],
        "E0": meta["E0"], "E1_parity": meta.get("E1_parity"),
        "prereg_sha256": prereg_sha(), "cap_inclusive": True,
        "outside_cap_queries": 0, "pid": os.getpid(),
        "instance_accepted": bool(meta.get("accepted")), "optimizer": "C0",
        "engine": "m2", "queries": 0, "query_cat": {}, "reused": False,
        "stage": "S1Q",
    }
    try:
        if not rec["instance_accepted"]:
            raise RuntimeError(
                f"B* rejected checks={meta.get('hard_checks')} E1={meta.get('E1_parity')}"
            )
        sim = S.make_sim(circ, J, meta["Bstar"], L, engine="m2")
        starts = U.starts_for_n(circ, seed, nq, L, 1)
        out = SP.spsa_run(sim.energy, starts, sim.sigma0(), U.rng_for_arm(circ, seed, nq, L), Q)
        rec["queries"] = out["queries"]
        rec["query_cat"] = out["query_cat"]
        if out["queries"] != Q:
            raise RuntimeError(f"S1 used {out['queries']} cap {Q}")
        if out["K_per_start"] != KEXP[Q]:
            raise RuntimeError(f"K {out['K_per_start']} expected {KEXP[Q]} at Q={Q}")
        if any(m["gain_frozen"] for m in out["start_meta"]):
            raise RuntimeError("gain was frozen")
        rec.update({
            "status": "ok", "E_sel": out["E_sel"], "x": out["x"].tolist(),
            "K_per_start": out["K_per_start"], "per_start": out["per_start"],
            "sel_start": out["start"], "sel_cand": out["cand"],
            "n_cands": len(out["all_cand_E"]), "all_cand_E": out["all_cand_E"],
            "start_meta": out["start_meta"],
        })
        d = sim.diagnostics(rec["x"], gs)
        rec["diag"] = d
        rec["E_err_over_N"] = (d["E"] - meta["E0"]) / nq
        rec["F"] = d["F"]
    except Exception as e:
        rec.update({"status": "failure", "error": repr(e)})
    rec["wall_s"] = time.time() - t0
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(rec))
    tmp.replace(path)
    return str(path), rec.get("status", "failure")


def write_status(status):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "STATUS_S1Q.json").write_text(json.dumps(status, indent=1))


def prebuild():
    rows = []
    for seed in PREREG["seeds"]["ids"]:
        print(f"[s1q] B* N{NFIX} seed {seed}", flush=True)
        J, meta, gs = instance(NFIX, seed)
        row = {
            "N": NFIX, "seed": seed, "accepted": bool(meta.get("accepted")),
            "Bstar": meta.get("Bstar"), "E0": meta.get("E0"),
            "E1_parity": meta.get("E1_parity"), "gap_even": meta.get("gap_even"),
            "gap_odd": meta.get("gap_odd"), "gs_resid": meta.get("gs_resid"),
            "valid": meta.get("valid"), "hard_checks": meta.get("hard_checks"),
            "field_prep_s": meta.get("field_prep_s"), "J_sha256": meta.get("J_sha256"),
        }
        rows.append(row)
        print(f"  accepted={row['accepted']} B*={row['Bstar']:.5f} E1={row['E1_parity']} "
              f"resid={row['gs_resid']:.2e} {row['field_prep_s']}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "INSTANCES_S1Q.json").write_text(json.dumps({"n": len(rows), "rows": rows}, indent=1))
    return rows


def pool_run(tasks, label, status):
    print(f"[s1q] {label} n={len(tasks)}", flush=True)
    t0 = time.time()
    ctx = get_context("spawn")
    with ctx.Pool(4) as pool:
        for i, (p, st) in enumerate(pool.imap_unordered(run_one, tasks, chunksize=1)):
            print(f"[{label} {i+1}/{len(tasks)}] {time.time()-t0:7.1f}s {st} {Path(p).name}", flush=True)
    wall = time.time() - t0
    status.setdefault("stage_wall_s", {})[label] = wall
    write_status(status)
    print(f"[s1q] {label} done {wall:.1f}s", flush=True)
    return wall


def main():
    warmup_worker()
    OUT.mkdir(parents=True, exist_ok=True)
    inst_rows = prebuild()
    accepted = {r["seed"] for r in inst_rows if r["accepted"]}
    skipped = []
    by_q = {Q: [] for Q in QS}
    for seed in PREREG["seeds"]["ids"]:
        for circ in PREREG["batch"]["circuits"]:
            for Q in QS:
                if seed not in accepted:
                    skipped.append((NFIX, seed, LFIX, circ, Q))
                    continue
                by_q[Q].append((NFIX, seed, LFIX, circ, Q))
    status = {
        "pid": os.getpid(), "stage": "STARTS_S1Q", "started": time.time(),
        "n_tasks": sum(len(v) for v in by_q.values()),
        "n_skipped_rejected_Bstar": len(skipped),
        "prereg_sha256": prereg_sha(), "nproc": 4, "Q": QS, "L": LFIX,
        "seeds": PREREG["seeds"]["ids"], "band": "exclusive_unread_dev",
        "frozen_no_fit_changes": True, "arm": "S1",
        "stage_wall_s": {}, "q12288": "pending",
    }
    write_status(status)
    print(f"[s1q] pid {os.getpid()} tasks {status['n_tasks']} skipped {len(skipped)} "
          f"prereg {prereg_sha()[:12]}", flush=True)
    wall_3072 = pool_run(by_q[3072], "Q3072", status)
    wall_6144 = pool_run(by_q[6144], "Q6144", status)
    status["checkpoint_after_6144"] = True
    est_12288 = wall_6144 * (12288 / 6144)
    status["estimated_q12288_s"] = est_12288
    write_status(status)
    print(f"[s1q] checkpoint 3072={wall_3072:.1f}s 6144={wall_6144:.1f}s "
          f"est12288={est_12288:.1f}s", flush=True)
    if est_12288 > HEAVY_S:
        status["q12288"] = "skipped_heavy"
        status["blocker"] = (
            f"estimated Q=12288 wall {est_12288:.1f}s > {HEAVY_S:.0f}s "
            f"from measured Q=6144 {wall_6144:.1f}s"
        )
        print("[s1q] STOP resource gate", status["blocker"], flush=True)
    else:
        pool_run(by_q[12288], "Q12288", status)
        status["q12288"] = "ran"
    status["finished"] = time.time()
    status["wall_s"] = status["finished"] - status["started"]
    write_status(status)
    print("[s1q] done", status["wall_s"], flush=True)


if __name__ == "__main__":
    main()
