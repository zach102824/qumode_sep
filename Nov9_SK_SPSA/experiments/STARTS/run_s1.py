"""STARTS S1/S2: concentrate Q on TQA. Reuse audited S4. Inclusive Q=3072.

    OMP_NUM_THREADS=1 python Nov9_SK_SPSA/experiments/STARTS/run_s1.py
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
import audit_s4 as A  # noqa: E402
import bstar_sparse as BS  # noqa: E402
import core as C  # noqa: E402
import kernel_m2 as K2  # noqa: E402
import sim as S  # noqa: E402
import spsa as SP  # noqa: E402
import starts as U  # noqa: E402

PREREG_PATH = HERE / "PREREG_STARTS_S1.json"
PREREG = json.loads(PREREG_PATH.read_text())
OUT = HERE.parents[1] / "results" / "STARTS"
INST = OUT / "instances"
RUNS = OUT / "runs"
NSTART = {"S1": 1, "S2": 2, "S4": 4}


def prereg_sha():
    return hashlib.sha256(PREREG_PATH.read_bytes()).hexdigest()


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
    meta["band"] = "read_dev_10048_10055"
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
    nq, seed, L, circ, arm = task
    Q = 3072
    nst = NSTART[arm]
    path = RUNS / f"N{nq}" / f"s{seed}_L{L}_{circ}_{arm}_q1024.json"
    if path.exists() and arm == "S4":
        return str(path), "reuse"
    if path.exists():
        return str(path), "skip"
    path.parent.mkdir(parents=True, exist_ok=True)
    warmup_worker()
    t0 = time.time()
    J, meta, gs = instance(nq, seed)
    rec = {
        "N": nq, "seed": seed, "L": L, "circuit": circ, "arm": arm, "nstarts": nst,
        "q": 1024, "Q": Q, "band": "read_dev", "Bstar": meta["Bstar"],
        "E0": meta["E0"], "E1_parity": meta.get("E1_parity"),
        "prereg_sha256": prereg_sha(), "cap_inclusive": True,
        "outside_cap_queries": 0, "pid": os.getpid(),
        "instance_accepted": bool(meta.get("accepted")), "optimizer": "C0",
        "engine": "m2", "queries": 0, "query_cat": {}, "reused": False,
    }
    try:
        if not rec["instance_accepted"]:
            raise RuntimeError(f"B* rejected {meta.get('hard_checks')}")
        sim = S.make_sim(circ, J, meta["Bstar"], L, engine="m2")
        starts = U.starts_for_n(circ, seed, nq, L, nst)
        out = SP.spsa_run(sim.energy, starts, sim.sigma0(), U.rng_for_arm(circ, seed, nq, L), Q)
        rec["queries"] = out["queries"]
        rec["query_cat"] = out["query_cat"]
        if out["queries"] != Q:
            raise RuntimeError(f"{arm} used {out['queries']} cap {Q}")
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


def main():
    warmup_worker()
    OUT.mkdir(parents=True, exist_ok=True)
    s4_rows = A.audit_all()
    (OUT / "S4_REUSE_AUDIT.json").write_text(json.dumps(
        {"n": len(s4_rows), "all_ok": A.all_ok(s4_rows), "rows": s4_rows}, indent=1))
    reuse_s4 = A.all_ok(s4_rows)
    print(f"[s1] S4 reuse {'OK' if reuse_s4 else 'FAIL — will refit and charge'}", flush=True)
    if not reuse_s4:
        for r in s4_rows:
            if not r["ok"]:
                print("  mismatch", r["path"], r["problems"], flush=True)
    arms = ["S1", "S2"]
    if not reuse_s4:
        arms = ["S4"] + arms
    tasks = []
    for seed in PREREG["seeds"]["ids"]:
        J, meta, gs = instance(15, seed)
        if not meta.get("accepted"):
            print(f"[s1] skip rejected B* seed {seed}", flush=True)
            continue
        for circ in PREREG["batch"]["circuits"]:
            for arm in arms:
                tasks.append((15, seed, 3, circ, arm))
    status = {
        "pid": os.getpid(), "stage": "STARTS_S1", "started": time.time(),
        "n_tasks": len(tasks), "s4_reused": reuse_s4, "arms": arms,
        "prereg_sha256": prereg_sha(), "nproc": 4, "seeds": PREREG["seeds"]["ids"],
        "band": "read_dev",
    }
    (OUT / "STATUS_S1.json").write_text(json.dumps(status, indent=1))
    print(f"[s1] pid {os.getpid()} tasks {len(tasks)} prereg {prereg_sha()[:12]}", flush=True)
    t0 = time.time()
    ctx = get_context("spawn")
    with ctx.Pool(4) as pool:
        for i, (p, st) in enumerate(pool.imap_unordered(run_one, tasks, chunksize=1)):
            print(f"[{i+1}/{len(tasks)}] {time.time()-t0:7.1f}s {st} {Path(p).name}", flush=True)
    status["finished"] = time.time()
    status["wall_s"] = status["finished"] - status["started"]
    (OUT / "STATUS_S1.json").write_text(json.dumps(status, indent=1))
    print("[s1] done", status["wall_s"], flush=True)


if __name__ == "__main__":
    main()
