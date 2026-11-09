"""STARTS S1ALT23: stacked S1JOINT vs S1ALT at N15 L=2,3 Q=qL.

    OMP_NUM_THREADS=1 python Nov9_SK_SPSA/experiments/STARTS/run_s1alt23.py
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
import alt_spsa as ALT  # noqa: E402
import bstar_sparse as BS  # noqa: E402
import core as C  # noqa: E402
import kernel_m2 as K2  # noqa: E402
import sim as S  # noqa: E402
import spsa as SP  # noqa: E402
import starts as U  # noqa: E402

PREREG_PATH = HERE / "PREREG_STARTS_S1ALT23.json"
PREREG = json.loads(PREREG_PATH.read_text())
OUT = HERE.parents[1] / "results" / "STARTS"
INST = OUT / "instances"
RUNS = OUT / "runs"
LS = [int(L) for L in PREREG["batch"]["L"]]
NFIX = int(PREREG["batch"]["N"][0])
Q_OF = {2: 2048, 3: 3072}
KEXP = {2048: 1019, 3072: 1531}
AB = {1019: (510, 509), 1531: (766, 765)}


def prereg_sha():
    return hashlib.sha256(PREREG_PATH.read_bytes()).hexdigest()


def run_path(nq, seed, L, circ, arm, Q):
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
    meta["band"] = "exclusive_unread_dev_10096_10103"
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
    Q = Q_OF[L]
    path = run_path(nq, seed, L, circ, arm, Q)
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
        "optimizer": "C0" if arm == "S1JOINT" else "C0_ALT",
        "engine": "m2", "queries": 0, "query_cat": {}, "reused": False,
        "stage": "S1ALT23", "continuation": False,
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
        elif arm == "S1ALT":
            ma, mb = ALT.block_masks(circ, L)
            out = ALT.spsa_run_alt(sim.energy, starts, sim.sigma0(), rng, Q, ma, mb)
        else:
            raise ValueError(arm)
        rec["queries"] = out["queries"]
        rec["query_cat"] = out["query_cat"]
        if out["queries"] != Q:
            raise RuntimeError(f"{arm} used {out['queries']} cap {Q}")
        if out["K_per_start"] != KEXP[Q]:
            raise RuntimeError(f"K {out['K_per_start']} expected {KEXP[Q]}")
        if any(m["gain_frozen"] for m in out["start_meta"]):
            raise RuntimeError("gain was frozen")
        if arm == "S1ALT":
            m0 = out["start_meta"][0]
            na, nb = AB[KEXP[Q]]
            if m0.get("joint_prefix_K", 1) != 0:
                raise RuntimeError("JOINT prefix was used")
            if m0.get("n_block_A_steps") != na or m0.get("n_block_B_steps") != nb:
                raise RuntimeError(f"block steps {m0.get('n_block_A_steps')} {m0.get('n_block_B_steps')}")
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
        print(f"[s1alt23] B* N{NFIX} seed {seed}", flush=True)
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
    (OUT / "INSTANCES_S1ALT23.json").write_text(json.dumps({"n": len(rows), "rows": rows}, indent=1))
    return rows


def main():
    warmup_worker()
    OUT.mkdir(parents=True, exist_ok=True)
    inst_rows = prebuild()
    accepted = {r["seed"] for r in inst_rows if r["accepted"]}
    tasks, skipped = [], []
    for seed in PREREG["seeds"]["ids"]:
        for L in LS:
            for circ in PREREG["batch"]["circuits"]:
                for arm in PREREG["batch"]["arms"]:
                    if seed not in accepted:
                        skipped.append((NFIX, seed, L, circ, arm))
                        continue
                    tasks.append((NFIX, seed, L, circ, arm))
    status = {
        "pid": os.getpid(), "stage": "STARTS_S1ALT23", "started": time.time(),
        "n_tasks": len(tasks), "n_skipped_rejected_Bstar": len(skipped),
        "prereg_sha256": prereg_sha(), "nproc": 4, "L": LS, "Q_of_L": Q_OF,
        "seeds": PREREG["seeds"]["ids"], "band": "exclusive_unread_dev",
        "frozen_no_fit_changes": True, "arms": ["S1JOINT", "S1ALT"],
    }
    (OUT / "STATUS_S1ALT23.json").write_text(json.dumps(status, indent=1))
    print(f"[s1alt23] pid {os.getpid()} tasks {len(tasks)} skipped {len(skipped)} "
          f"prereg {prereg_sha()[:12]}", flush=True)
    t0 = time.time()
    ctx = get_context("spawn")
    with ctx.Pool(4) as pool:
        for i, (p, st) in enumerate(pool.imap_unordered(run_one, tasks, chunksize=1)):
            print(f"[{i+1}/{len(tasks)}] {time.time()-t0:7.1f}s {st} {Path(p).name}", flush=True)
    status["finished"] = time.time()
    status["wall_s"] = status["finished"] - status["started"]
    (OUT / "STATUS_S1ALT23.json").write_text(json.dumps(status, indent=1))
    print("[s1alt23] done", status["wall_s"], flush=True)


if __name__ == "__main__":
    main()
