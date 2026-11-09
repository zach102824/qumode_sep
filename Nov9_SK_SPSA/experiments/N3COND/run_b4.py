"""B4 runner. C0-gain shrink versus sham. Does not rewrite C0, C1, M0, or M2.

    OMP_NUM_THREADS=1 python Nov9_SK_SPSA/experiments/N3COND/run_b4.py
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

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import nov3 as N  # noqa: E402
import run_n3cond as R1  # noqa: E402
import spsa as S  # noqa: E402

PREREG_PATH = HERE / "PREREG_N3COND_B4.json"
PREREG = json.loads(PREREG_PATH.read_text())
OUT = HERE.parents[1] / "results" / "N3COND"
RUNS = OUT / "runs"


def prereg_sha():
    return hashlib.sha256(PREREG_PATH.read_bytes()).hexdigest()


def run_one(task):
    Nqubits, seed, circuit, arm = task
    if arm not in ("S0", "S2"):
        raise ValueError(arm)
    path = RUNS / f"N{Nqubits}" / f"s{seed}_L3_{circuit}_{arm}_q1024.json"
    if path.name.find("_C0_") >= 0 or path.name.find("_M0_") >= 0:
        raise RuntimeError("refusing to write a historical name")
    if path.exists():
        return str(path), "skip"
    path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    J, meta, gs = R1.instance(Nqubits, seed)
    Q = 3072
    sim = N.make_sim(circuit, J, meta["Bstar"], 3)
    starts = N.starts_for(circuit, seed, Nqubits, 3)
    cid = N.circuit_id(circuit)
    rec = {
        "N": Nqubits, "seed": seed, "L": 3, "circuit": circuit, "conditioner": arm, "q": 1024, "Q": Q,
        "band": "read_dev", "seeds_are_fresh_holdout": False, "Bstar": meta["Bstar"],
        "prereg_sha256": prereg_sha(), "hypothesis": PREREG["hypothesis"]["id"],
        "cap_inclusive_of_calibration": True, "outside_cap_queries": 0,
        "pid": os.getpid(), "diag_recompute_not_charged": 1,
        "instance_accepted": meta["accepted_n3cond"],
    }
    try:
        if not meta["accepted_n3cond"]:
            raise RuntimeError("instance failed acceptance")
        out = S.spsa_run_c0gain(
            sim.energy, starts, sim.sigma0(),
            lambda s: N.rng_for(seed, Nqubits, 3, cid, s, 0),
            Q, shrink=(arm == "S2"),
        )
        d = sim.diagnostics(out["x"], gs)
        if any(m["axial_used_for_gain"] or m["a_source"] != "simultaneous_c0" for m in out["start_meta"]):
            raise RuntimeError("gain was not the C0 simultaneous gain")
        rec.update({
            "status": "ok", "E_sel": out["E_sel"], "x": out["x"].tolist(), "queries": out["queries"],
            "query_cat": out["query_cat"], "K_per_start": out["K_per_start"], "per_start": out["per_start"],
            "axial_queries_per_start": out["axial_queries_per_start"],
            "simultaneous_queries_per_start": out["simultaneous_queries_per_start"],
            "outside_cap_queries": 0, "sel_start": out["start"], "sel_cand": out["cand"],
            "all_cand_E": out["all_cand_E"], "start_meta": out["start_meta"], "diag": d,
        })
        parts = out["query_cat"]["calibration"] + out["query_cat"]["perturb"] + out["query_cat"]["candidate"]
        if out["queries"] != Q or parts != Q:
            raise RuntimeError("query ledger does not add up")
    except Exception as e:
        d0 = sim.diagnostics(starts[0], gs)
        rec.update({"status": "failure", "error": repr(e), "diag": d0, "queries": rec.get("queries", 0)})
    F = rec["diag"]["F"]
    rec["F_plus"] = meta["F_plus"]
    rec["E0"] = meta["E0"]
    rec["E_plus"] = meta["E_plus"]
    rec["wall_s"] = time.time() - t0
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(rec))
    tmp.replace(path)
    return str(path), rec["status"]


def main():
    assert os.environ["OMP_NUM_THREADS"] == "1"
    tasks = []
    for n in PREREG["batch"]["N"]:
        for seed in PREREG["seeds"]["ids"]:
            for circuit in PREREG["batch"]["circuits"]:
                for arm in PREREG["batch"]["new_arms"]:
                    tasks.append((n, seed, circuit, arm))
    status = {"pid": os.getpid(), "stage": "B4", "started": time.time(), "n_tasks": len(tasks),
              "prereg_sha256": prereg_sha(),
              "cap_note": "axial and simultaneous calibration are inside Q; C0 files are not re-run"}
    (OUT / "STATUS_B4.json").write_text(json.dumps(status))
    print(f"[b4] pid {os.getpid()} tasks {len(tasks)} prereg {prereg_sha()[:12]}", flush=True)
    t0 = time.time()
    with Pool(4) as P:
        for i, (p, st) in enumerate(P.imap_unordered(run_one, tasks, chunksize=1)):
            if st != "skip":
                print(f"[{i+1}/{len(tasks)}] {time.time()-t0:7.1f}s {st} {Path(p).name}", flush=True)
    status["finished"] = time.time()
    status["wall_s"] = status["finished"] - status["started"]
    (OUT / "STATUS_B4.json").write_text(json.dumps(status))
    print("[b4] done", status["wall_s"], flush=True)


if __name__ == "__main__":
    main()
