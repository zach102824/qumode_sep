"""STARTS S1ALT13: JOINT vs PR88 1:1 vs 1:3 shell-heavy at N15 L=2 Q=2048.

    OMP_NUM_THREADS=1 python Nov9_SK_SPSA/experiments/STARTS/run_s1alt13.py
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
import alt13_spsa as A13  # noqa: E402
import alt_spsa as ALT  # noqa: E402
import bstar_sparse as BS  # noqa: E402
import core as C  # noqa: E402
import kernel_m2 as K2  # noqa: E402
import sim as S  # noqa: E402
import spsa as SP  # noqa: E402
import starts as U  # noqa: E402

PREREG_PATH = HERE / "PREREG_STARTS_S1ALT13.json"
PREREG = json.loads(PREREG_PATH.read_text())
OUT = HERE.parents[1] / "results" / "STARTS"
INST = OUT / "instances"
RUNS = OUT / "runs"
Q = int(PREREG["batch"]["Q"])
LFIX = int(PREREG["batch"]["L"][0])
NFIX = int(PREREG["batch"]["N"][0])
KEXP = 1019
AB = {"S1ALT11": (510, 509), "S1ALT13": (255, 764)}


def prereg_sha():
    return hashlib.sha256(PREREG_PATH.read_bytes()).hexdigest()


def run_path(nq, seed, L, circ, arm):
    return RUNS / f"N{nq}" / f"s{seed}_L{L}_{circ}_{arm}_Q{Q}.json"


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
    meta["band"] = "exclusive_unread_dev_10104_10111"
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
    path = run_path(nq, seed, L, circ, arm)
    if path.exists():
        return str(path), "skip"
    path.parent.mkdir(parents=True, exist_ok=True)
    warmup_worker()
    t0 = time.time()
    J, meta, gs = instance(nq, seed)
    rec = {
        "N": nq, "seed": seed, "L": L, "circuit": circ, "arm": arm, "nstarts": 1,
        "q": 1024, "Q": Q, "band": "exclusive_unread_dev", "Bstar": meta["Bstar"],
        "E0": meta["E0"], "E1_parity": meta.get("E1_parity"),
        "prereg_sha256": prereg_sha(), "cap_inclusive": True,
        "outside_cap_queries": 0, "pid": os.getpid(),
        "instance_accepted": bool(meta.get("accepted")),
        "optimizer": {"S1JOINT": "C0", "S1ALT11": "C0_ALT", "S1ALT13": "C0_ALT13"}[arm],
        "engine": "m2", "queries": 0, "query_cat": {}, "reused": False,
        "stage": "S1ALT13", "continuation": False,
    }
    try:
        if not rec["instance_accepted"]:
            raise RuntimeError(
                f"B* rejected checks={meta.get('hard_checks')} E1={meta.get('E1_parity')}"
            )
        sim = S.make_sim(circ, J, meta["Bstar"], L, engine="m2")
        starts = U.starts_for_n(circ, seed, nq, L, 1)
        rng = U.rng_for_arm(circ, seed, nq, L)
        if arm == "S1JOINT":
            out = SP.spsa_run(sim.energy, starts, sim.sigma0(), rng, Q)
        elif arm == "S1ALT11":
            ma, mb = ALT.block_masks(circ, L)
            out = ALT.spsa_run_alt(sim.energy, starts, sim.sigma0(), rng, Q, ma, mb)
        elif arm == "S1ALT13":
            ma, mb = ALT.block_masks(circ, L)
            out = A13.spsa_run_alt13(sim.energy, starts, sim.sigma0(), rng, Q, ma, mb)
        else:
            raise ValueError(arm)
        rec["queries"] = out["queries"]
        rec["query_cat"] = out["query_cat"]
        if out["queries"] != Q:
            raise RuntimeError(f"{arm} used {out['queries']} cap {Q}")
        if out["K_per_start"] != KEXP:
            raise RuntimeError(f"K {out['K_per_start']} expected {KEXP}")
        if any(m["gain_frozen"] for m in out["start_meta"]):
            raise RuntimeError("gain was frozen")
        if arm in AB:
            m0 = out["start_meta"][0]
            na, nb = AB[arm]
            if m0.get("joint_prefix_K", 1) != 0:
                raise RuntimeError("JOINT prefix was used")
            if m0.get("n_block_A_steps") != na or m0.get("n_block_B_steps") != nb:
                raise RuntimeError(f"block steps {m0.get('n_block_A_steps')} {m0.get('n_block_B_steps')}")
            if arm == "S1ALT13" and m0.get("leftover_blocks") != ["A", "B", "B"]:
                raise RuntimeError(f"leftover {m0.get('leftover_blocks')}")
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


def prebuild():
    rows = []
    for seed in PREREG["seeds"]["ids"]:
        print(f"[s1alt13] B* N{NFIX} seed {seed}", flush=True)
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
    (OUT / "INSTANCES_S1ALT13.json").write_text(json.dumps({"n": len(rows), "rows": rows}, indent=1))
    return rows


def main():
    warmup_worker()
    OUT.mkdir(parents=True, exist_ok=True)
    inst_rows = prebuild()
    accepted = {r["seed"] for r in inst_rows if r["accepted"]}
    tasks, skipped = [], []
    for seed in PREREG["seeds"]["ids"]:
        for circ in PREREG["batch"]["circuits"]:
            for arm in PREREG["batch"]["arms"]:
                if seed not in accepted:
                    skipped.append((NFIX, seed, LFIX, circ, arm))
                    continue
                tasks.append((NFIX, seed, LFIX, circ, arm))
    status = {
        "pid": os.getpid(), "stage": "STARTS_S1ALT13", "started": time.time(),
        "n_tasks": len(tasks), "n_skipped_rejected_Bstar": len(skipped),
        "prereg_sha256": prereg_sha(), "nproc": 4, "Q": Q, "L": LFIX,
        "seeds": PREREG["seeds"]["ids"], "band": "exclusive_unread_dev",
        "frozen_no_fit_changes": True, "arms": ["S1JOINT", "S1ALT11", "S1ALT13"],
    }
    (OUT / "STATUS_S1ALT13.json").write_text(json.dumps(status, indent=1))
    print(f"[s1alt13] pid {os.getpid()} tasks {len(tasks)} skipped {len(skipped)} "
          f"Q={Q} prereg {prereg_sha()[:12]}", flush=True)
    t0 = time.time()
    ctx = get_context("spawn")
    with ctx.Pool(4) as pool:
        for i, (p, st) in enumerate(pool.imap_unordered(run_one, tasks, chunksize=1)):
            print(f"[{i+1}/{len(tasks)}] {time.time()-t0:7.1f}s {st} {Path(p).name}", flush=True)
    status["finished"] = time.time()
    status["wall_s"] = status["finished"] - status["started"]
    (OUT / "STATUS_S1ALT13.json").write_text(json.dumps(status, indent=1))
    print("[s1alt13] done", status["wall_s"], flush=True)


if __name__ == "__main__":
    main()
