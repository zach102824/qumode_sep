"""L1XFER: N=15 L=1 concentration/transfer. Stages: train, freeze, test.

    OMP_NUM_THREADS=1 python Nov9_SK_SPSA/experiments/L1XFER/run_l1xfer.py train
    OMP_NUM_THREADS=1 python Nov9_SK_SPSA/experiments/L1XFER/run_l1xfer.py freeze
    OMP_NUM_THREADS=1 python Nov9_SK_SPSA/experiments/L1XFER/run_l1xfer.py test
"""
from __future__ import annotations

import hashlib
import json
import os
import resource
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
sys.path.insert(0, str(HERE.parent / "FTREP"))
sys.path.insert(0, str(HERE.parent / "N3COND"))
sys.path.insert(0, str(HERE.parent / "LEXACT"))
sys.path.insert(0, str(HERE.parent / "LE"))
sys.path.insert(0, str(HERE.parent / "IMAX"))
import alloc as A  # noqa: E402
import bstar_sparse as BS  # noqa: E402
import core as C  # noqa: E402
import freeze as F  # noqa: E402
import kernel_m2 as K2  # noqa: E402
import sim as S  # noqa: E402
import wrap as W  # noqa: E402

PREREG_PATH = HERE / "PREREG_L1XFER.json"
PREREG = json.loads(PREREG_PATH.read_text())
OUT = HERE.parents[1] / "results" / "L1XFER"
INST = OUT / "instances"
RUNS = OUT / "runs"
VECTORS = OUT / "TRANSFER_VECTORS_L1XFER.json"
NFIX = 15
LFIX = 1
QFIX = 1024
TFIX = 122
BAND = "exclusive_unread_dev"
TRAIN = list(PREREG["split_documented_before_fits"]["train"])
TEST = list(PREREG["split_documented_before_fits"]["test"])
ARMS = ("nov3_center4", "qaoa_center4")
NPROC = 4


def prereg_sha():
    return hashlib.sha256(PREREG_PATH.read_bytes()).hexdigest()


def fit_path(seed, arm):
    circ = "QAOA" if arm == "qaoa_center4" else "Nov3"
    return RUNS / f"N{NFIX}" / f"s{seed}_L{LFIX}_{circ}_{arm}_Q{QFIX}.json"


def xfer_path(seed, arm, agg):
    circ = "QAOA" if arm == "qaoa_center4" else "Nov3"
    return RUNS / f"N{NFIX}" / f"s{seed}_L{LFIX}_{circ}_{arm}_xfer_{agg}_Q1.json"


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
    meta["band"] = "exclusive_unread_dev_10600_10615"
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


def build_one(seed):
    warmup_worker()
    print(f"[l1xfer] B* N{NFIX} seed {seed}", flush=True)
    J, meta, gs = instance(NFIX, seed)
    del gs, J
    row = {
        "N": NFIX, "seed": seed, "accepted": bool(meta.get("accepted")),
        "Bstar": meta.get("Bstar"), "E0": meta.get("E0"),
        "E1_parity": meta.get("E1_parity"), "gap_even": meta.get("gap_even"),
        "gap_odd": meta.get("gap_odd"), "gs_resid": meta.get("gs_resid"),
        "valid": meta.get("valid"), "hard_checks": meta.get("hard_checks"),
        "field_prep_s": meta.get("field_prep_s"), "J_sha256": meta.get("J_sha256"),
        "split": "train" if seed in TRAIN else "test",
    }
    print(f"  accepted={row['accepted']} B*={row['Bstar']:.5f} E1={row['E1_parity']} "
          f"resid={row['gs_resid']:.2e} {row['field_prep_s']}", flush=True)
    return row


def run_fit(task):
    seed, arm, split = task
    if split == "test" and not VECTORS.exists():
        raise RuntimeError("refusing test fit before transfer vectors are frozen")
    circ = "QAOA" if arm == "qaoa_center4" else "Nov3"
    method = "QAOA" if circ == "QAOA" else "Nov3rd"
    path = fit_path(seed, arm)
    if path.exists():
        return str(path), "skip"
    path.parent.mkdir(parents=True, exist_ok=True)
    warmup_worker()
    t0 = time.time()
    cpu0 = time.process_time()
    J, meta, gs = instance(NFIX, seed)
    rec = {
        "N": NFIX, "seed": seed, "L": LFIX, "circuit": circ, "arm": arm,
        "split": split, "q": 1024, "Q": QFIX, "band": BAND, "Bstar": meta["Bstar"],
        "E0": meta["E0"], "E1_parity": meta.get("E1_parity"),
        "prereg_sha256": prereg_sha(), "cap_inclusive": True,
        "outside_cap_queries": 0, "pid": os.getpid(),
        "instance_accepted": bool(meta.get("accepted")),
        "engine": "m2", "queries": 0, "query_cat": {},
        "reused": False, "stage": "L1XFER", "continuation": False,
        "ft_alloc_sha256": PREREG["ft_source"]["alloc_py_sha256"],
        "ft_head_commit": PREREG["ft_source"]["head_commit"],
        "optimizer": "FT_v4_center4", "nstarts": 4,
        "no_l2_or_deeper": True, "not_an_architecture_change": True,
        "role": "independent_fit",
    }
    try:
        if not rec["instance_accepted"]:
            raise RuntimeError(
                f"B* rejected checks={meta.get('hard_checks')} E1={meta.get('E1_parity')}"
            )
        sim = S.make_sim(circ, J, meta["Bstar"], LFIX, engine="m2")
        rng_init, rng_spsa = W.rng_lists(seed, NFIX, meta["Bstar"], LFIX, method)

        def propagate(x, _sim=sim, _method=method):
            xl = W.to_le(_method, x)
            return float(_sim.energy(xl)), xl

        out = A.run_equal4(propagate, method, LFIX, QFIX, rng_init, rng_spsa, x0_fn=W.draw_fn(arm))
        rec["queries"] = int(out["queries"]["total"])
        rec["query_cat"] = {
            "init": out["queries"]["init"],
            "calibration": out["queries"]["calibration"],
            "perturb": out["queries"]["iteration"],
            "candidate": out["queries"]["candidate"],
            "transfer": 0,
            "monitor": out["queries"]["monitoring"],
            "screen": out["queries"]["screen"],
        }
        if out["queries"]["total"] != QFIX:
            raise RuntimeError(f"used {out['queries']['total']} cap {QFIX}")
        if out["n_iter"] != TFIX:
            raise RuntimeError(f"n_iter {out['n_iter']} expected {TFIX}")
        qc = rec["query_cat"]
        if (qc["init"] != 4 or qc["calibration"] != 32
                or qc["perturb"] != 976 or qc["candidate"] != 12):
            raise RuntimeError(f"cats {qc}")
        if qc["monitor"] != 0 or qc["screen"] != 0:
            raise RuntimeError("unexpected screen/monitor")
        x_le = np.asarray(out["aux"], float)
        rec.update({
            "status": "ok", "E_sel": float(out["E"]), "x_ft": out["x"].tolist(),
            "x": x_le.tolist(), "K_per_start": int(out["n_iter"]),
            "per_start": QFIX // 4,
            "sel_start": int(out["start_id"]), "sel_cand": out["which"],
            "n_cands": 12, "start_tails": out["tails"],
        })
        d = sim.diagnostics(x_le, gs)
        rec["diag"] = d
        rec["E_err_over_N"] = (d["E"] - meta["E0"]) / NFIX
        rec["F"] = d["F"]
    except Exception as e:
        rec.update({"status": "failure", "error": repr(e)})
    rec["wall_s"] = time.time() - t0
    rec["cpu_s"] = time.process_time() - cpu0
    rec["peak_rss_kib"] = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(rec))
    tmp.replace(path)
    return str(path), rec.get("status", "failure")


def run_xfer(task):
    seed, arm, agg, x_le = task
    if not VECTORS.exists():
        raise RuntimeError("refusing transfer eval before freeze")
    if seed not in TEST:
        raise RuntimeError(f"transfer eval on non-test seed {seed}")
    circ = "QAOA" if arm == "qaoa_center4" else "Nov3"
    path = xfer_path(seed, arm, agg)
    if path.exists():
        return str(path), "skip"
    path.parent.mkdir(parents=True, exist_ok=True)
    warmup_worker()
    t0 = time.time()
    cpu0 = time.process_time()
    J, meta, gs = instance(NFIX, seed)
    x_le = np.asarray(x_le, dtype=np.float64)
    rec = {
        "N": NFIX, "seed": seed, "L": LFIX, "circuit": circ, "arm": arm,
        "split": "test", "aggregator": agg, "q": None, "Q": 1, "band": BAND,
        "Bstar": meta["Bstar"], "E0": meta["E0"], "E1_parity": meta.get("E1_parity"),
        "prereg_sha256": prereg_sha(), "cap_inclusive": True,
        "outside_cap_queries": 0, "pid": os.getpid(),
        "instance_accepted": bool(meta.get("accepted")),
        "engine": "m2", "queries": 0, "query_cat": {},
        "reused": False, "stage": "L1XFER", "continuation": False,
        "ft_alloc_sha256": PREREG["ft_source"]["alloc_py_sha256"],
        "ft_head_commit": PREREG["ft_source"]["head_commit"],
        "optimizer": "frozen_shared_vector", "nstarts": 0,
        "no_l2_or_deeper": True, "not_an_architecture_change": True,
        "role": "transfer_eval", "x": x_le.tolist(),
    }
    try:
        if not rec["instance_accepted"]:
            raise RuntimeError("B* rejected")
        sim = S.make_sim(circ, J, meta["Bstar"], LFIX, engine="m2")
        d = sim.diagnostics(x_le, gs)
        rec["queries"] = 1
        rec["query_cat"] = {
            "init": 0, "calibration": 0, "perturb": 0, "candidate": 0,
            "transfer": 1, "monitor": 0, "screen": 0,
        }
        rec.update({
            "status": "ok", "E_sel": float(d["E"]), "diag": d,
            "E_err_over_N": (d["E"] - meta["E0"]) / NFIX, "F": d["F"],
        })
    except Exception as e:
        rec.update({"status": "failure", "error": repr(e)})
    rec["wall_s"] = time.time() - t0
    rec["cpu_s"] = time.process_time() - cpu0
    rec["peak_rss_kib"] = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(rec))
    tmp.replace(path)
    return str(path), rec.get("status", "failure")


def load_train_ok():
    rows = []
    d = RUNS / f"N{NFIX}"
    if not d.exists():
        return rows
    for seed in TRAIN:
        for arm in ARMS:
            p = fit_path(seed, arm)
            if p.exists():
                rows.append(json.loads(p.read_text()))
    return rows


def stage_train():
    warmup_worker()
    OUT.mkdir(parents=True, exist_ok=True)
    if VECTORS.exists():
        raise SystemExit("freeze already written; will not refit train under a locked freeze")
    t_bstar = time.time()
    ctx = get_context("spawn")
    with ctx.Pool(NPROC) as pool:
        inst_rows = list(pool.imap(build_one, TRAIN + TEST, chunksize=1))
    inst_rows = sorted(inst_rows, key=lambda r: r["seed"])
    (OUT / "INSTANCES_L1XFER.json").write_text(json.dumps({"n": len(inst_rows), "rows": inst_rows}, indent=1))
    accepted = {r["seed"] for r in inst_rows if r["accepted"]}
    tasks, skipped = [], []
    for seed in TRAIN:
        for arm in ARMS:
            if seed not in accepted:
                skipped.append((seed, arm))
                continue
            tasks.append((seed, arm, "train"))
    status = {
        "pid": os.getpid(), "stage": "L1XFER_train", "started": time.time(),
        "bstar_wall_s": time.time() - t_bstar,
        "n_tasks": len(tasks), "n_skipped_rejected_Bstar": len(skipped),
        "prereg_sha256": prereg_sha(), "nproc": NPROC, "N": NFIX, "L": LFIX,
        "Q": QFIX, "T": TFIX, "train": TRAIN, "test": TEST, "band": BAND,
        "ft_alloc_sha256": PREREG["ft_source"]["alloc_py_sha256"],
        "peeked_test_fits": False, "peeked_transfer": False,
    }
    (OUT / "STATUS_L1XFER.json").write_text(json.dumps(status, indent=1))
    print(f"[l1xfer] train tasks {len(tasks)} skipped {len(skipped)} "
          f"B* wall {status['bstar_wall_s']:.1f}s", flush=True)
    t0 = time.time()
    with ctx.Pool(NPROC) as pool:
        for i, (p, st) in enumerate(pool.imap_unordered(run_fit, tasks, chunksize=1)):
            print(f"[train {i+1}/{len(tasks)}] {time.time()-t0:7.1f}s {st} {Path(p).name}", flush=True)
    status["train_finished"] = time.time()
    status["train_fit_wall_s"] = status["train_finished"] - t0
    (OUT / "STATUS_L1XFER.json").write_text(json.dumps(status, indent=1))
    print("[l1xfer] train done", status["train_fit_wall_s"], flush=True)


def stage_freeze():
    if VECTORS.exists():
        raise SystemExit(f"freeze already written: {VECTORS}")
    test_hits = list((RUNS / f"N{NFIX}").glob("s1060[8-9]_*")) + list((RUNS / f"N{NFIX}").glob("s1061[0-5]_*"))
    if test_hits:
        raise SystemExit(f"test artifacts exist before freeze: {test_hits[:3]}")
    rows = load_train_ok()
    freeze = F.build_freeze(rows, prereg_sha())
    VECTORS.write_text(json.dumps(freeze, indent=1))
    status_path = OUT / "STATUS_L1XFER.json"
    status = json.loads(status_path.read_text()) if status_path.exists() else {}
    status["freeze_written"] = True
    status["freeze_sha256"] = hashlib.sha256(VECTORS.read_bytes()).hexdigest()
    status_path.write_text(json.dumps(status, indent=1))
    print("[l1xfer] froze mean/median from train n=8; test untouched", flush=True)


def stage_test():
    if not VECTORS.exists():
        raise SystemExit("freeze missing; will not run test")
    freeze = json.loads(VECTORS.read_text())
    if freeze.get("prereg_sha256") != prereg_sha():
        raise SystemExit("freeze prereg hash mismatch")
    warmup_worker()
    inst = json.loads((OUT / "INSTANCES_L1XFER.json").read_text())
    accepted = {r["seed"] for r in inst["rows"] if r.get("accepted")}
    fit_tasks, skipped = [], []
    for seed in TEST:
        for arm in ARMS:
            if seed not in accepted:
                skipped.append((seed, arm))
                continue
            fit_tasks.append((seed, arm, "test"))
    xfer_tasks = []
    for seed in TEST:
        if seed not in accepted:
            continue
        xfer_tasks.append((seed, "nov3_center4", "mean", freeze["nov3"]["mean"]))
        xfer_tasks.append((seed, "nov3_center4", "median", freeze["nov3"]["median"]))
        xfer_tasks.append((seed, "qaoa_center4", "mean", freeze["qaoa"]["mean"]))
        xfer_tasks.append((seed, "qaoa_center4", "median", freeze["qaoa"]["median"]))
    status_path = OUT / "STATUS_L1XFER.json"
    status = json.loads(status_path.read_text()) if status_path.exists() else {}
    status.update({
        "stage": "L1XFER_test", "test_started": time.time(),
        "n_test_fit_tasks": len(fit_tasks), "n_xfer_tasks": len(xfer_tasks),
        "n_skipped_rejected_Bstar_test": len(skipped),
        "freeze_sha256": hashlib.sha256(VECTORS.read_bytes()).hexdigest(),
    })
    status_path.write_text(json.dumps(status, indent=1))
    print(f"[l1xfer] test fits {len(fit_tasks)} xfer {len(xfer_tasks)}", flush=True)
    ctx = get_context("spawn")
    t0 = time.time()
    with ctx.Pool(NPROC) as pool:
        for i, (p, st) in enumerate(pool.imap_unordered(run_fit, fit_tasks, chunksize=1)):
            print(f"[testfit {i+1}/{len(fit_tasks)}] {time.time()-t0:7.1f}s {st} {Path(p).name}", flush=True)
        for i, (p, st) in enumerate(pool.imap_unordered(run_xfer, xfer_tasks, chunksize=1)):
            print(f"[xfer {i+1}/{len(xfer_tasks)}] {time.time()-t0:7.1f}s {st} {Path(p).name}", flush=True)
    status["test_finished"] = time.time()
    status["test_wall_s"] = status["test_finished"] - t0
    status_path.write_text(json.dumps(status, indent=1))
    print("[l1xfer] test done", status["test_wall_s"], flush=True)


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in ("train", "freeze", "test"):
        raise SystemExit("usage: run_l1xfer.py {train|freeze|test}")
    stage = sys.argv[1]
    if stage == "train":
        stage_train()
    elif stage == "freeze":
        stage_freeze()
    else:
        stage_test()


if __name__ == "__main__":
    main()
