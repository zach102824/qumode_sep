"""B3 runner. In-cap shrink versus matched scalar. Does not write C0/C1 files.

    OMP_NUM_THREADS=1 python Nov9_SK_SPSA/experiments/N3COND/run_b3.py --workers 4
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import nov3 as N  # noqa: E402
import run_n3cond as R1  # noqa: E402
import spsa as S  # noqa: E402

PREREG_PATH = HERE / "PREREG_N3COND_B3.json"
PREREG = json.loads(PREREG_PATH.read_text())
OUT = HERE.parents[1] / "results" / "N3COND"
RUNS = OUT / "runs"


def prereg_sha():
    return hashlib.sha256(PREREG_PATH.read_bytes()).hexdigest()


def run_one(task):
    Nqubits, seed, L, circuit, arm, q = task
    if arm not in ("M0", "M2"):
        raise ValueError(arm)
    path = RUNS / f"N{Nqubits}" / f"s{seed}_L{L}_{circuit}_{arm}_q{q}.json"
    legacy = RUNS / f"N{Nqubits}" / f"s{seed}_L{L}_{circuit}_C0_q{q}.json"
    if path.resolve() == legacy.resolve():
        raise RuntimeError("refusing to write a historical name")
    if path.exists():
        return str(path), "skip"
    path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    J, meta, gs = R1.instance(Nqubits, seed)
    Q = q * L
    sim = N.make_sim(circuit, J, meta["Bstar"], L)
    starts = N.starts_for(circuit, seed, Nqubits, L)
    cid = N.circuit_id(circuit)
    rec = {
        "N": Nqubits, "seed": seed, "L": L, "circuit": circuit, "conditioner": arm, "q": q, "Q": Q,
        "band": "read_dev", "seeds_are_fresh_holdout": False,
        "Bstar": meta["Bstar"], "circuit_id": cid, "prereg_sha256": prereg_sha(),
        "cap_inclusive_of_calibration": True, "outside_cap_queries": 0,
        "hypothesis": PREREG["hypothesis"]["id"], "pid": os.getpid(),
        "diag_recompute_not_charged": 1, "instance_accepted": meta["accepted_n3cond"],
    }
    try:
        if not meta["accepted_n3cond"]:
            raise RuntimeError("instance failed acceptance")
        out = S.spsa_run_axial(
            sim.energy, starts, sim.sigma0(),
            lambda s: N.rng_for(seed, Nqubits, L, cid, s, 0),
            Q, shrink=(arm == "M2"),
        )
        d = sim.diagnostics(out["x"], gs)
        rec.update({
            "status": "ok", "E_sel": out["E_sel"], "x": out["x"].tolist(), "queries": out["queries"],
            "query_cat": out["query_cat"], "K_per_start": out["K_per_start"], "per_start": out["per_start"],
            "calibration_queries_per_start": out["calibration_queries_per_start"],
            "outside_cap_queries": out["outside_cap_queries"],
            "sel_start": out["start"], "sel_cand": out["cand"], "all_cand_E": out["all_cand_E"],
            "start_meta": out["start_meta"], "diag": d,
        })
        if out["queries"] != Q or out["query_cat"]["calibration"] + out["query_cat"]["perturb"] + out["query_cat"]["candidate"] != Q:
            raise RuntimeError("query ledger does not add up")
    except Exception as e:
        d0 = sim.diagnostics(starts[0], gs)
        rec.update({"status": "failure", "error": repr(e), "diag": d0, "queries": rec.get("queries", 0)})
    F = rec["diag"]["F"]
    rec["F_plus"] = meta["F_plus"]
    rec["eta"] = (F - meta["F_plus"]) / (1.0 - meta["F_plus"])
    rec["E0"] = meta["E0"]
    rec["E_plus"] = meta["E_plus"]
    rec["closure_E"] = (meta["E_plus"] - rec["diag"]["E"]) / (meta["E_plus"] - meta["E0"])
    rec["resid_rel"] = (rec["diag"]["E"] - meta["E0"]) / abs(meta["E0"])
    rec["wall_s"] = time.time() - t0
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(rec))
    tmp.replace(path)
    return str(path), rec["status"]


def main():
    assert os.environ["OMP_NUM_THREADS"] == "1"
    seeds = PREREG["seeds"]["ids"]
    tasks = []
    for n in PREREG["batch"]["N"]:
        for seed in seeds:
            for L in PREREG["batch"]["L"]:
                for circuit in PREREG["batch"]["circuits"]:
                    for arm in PREREG["batch"]["arms"]:
                        for q in PREREG["batch"]["q"]:
                            tasks.append((n, seed, L, circuit, arm, q))
    status = {"pid": os.getpid(), "stage": "B3", "started": time.time(), "n_tasks": len(tasks),
              "prereg_sha256": prereg_sha(), "cap_note": "axial calibration is inside Q; outside_cap_queries=0"}
    (OUT / "STATUS_B3.json").write_text(json.dumps(status))
    print(f"[b3] pid {os.getpid()} tasks {len(tasks)} prereg {prereg_sha()[:12]}", flush=True)
    t0 = time.time()
    with Pool(4) as P:
        for i, (p, st) in enumerate(P.imap_unordered(run_one, tasks, chunksize=1)):
            if st != "skip":
                print(f"[{i+1}/{len(tasks)}] {time.time()-t0:7.1f}s {st} {Path(p).name}", flush=True)
    status["finished"] = time.time()
    status["wall_s"] = status["finished"] - status["started"]
    (OUT / "STATUS_B3.json").write_text(json.dumps(status))
    print("[b3] done", status["wall_s"], flush=True)


if __name__ == "__main__":
    main()
