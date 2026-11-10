"""N18GEOM: N=18 L=1 SK-geometry freeze. Stages: train, freeze, test.

    OMP_NUM_THREADS=1 python Nov9_SK_SPSA/experiments/N18GEOM/run_n18geom.py train
    OMP_NUM_THREADS=1 python Nov9_SK_SPSA/experiments/N18GEOM/run_n18geom.py freeze
    OMP_NUM_THREADS=1 python Nov9_SK_SPSA/experiments/N18GEOM/run_n18geom.py test
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

PREREG_PATH = HERE / "PREREG_N18GEOM.json"
PREREG = json.loads(PREREG_PATH.read_text())
OUT = HERE.parents[1] / "results" / "N18GEOM"
INST = OUT / "instances"
RUNS = OUT / "runs"
VECTORS = OUT / "TRANSFER_VECTORS_N18GEOM.json"
NFIX = 18
LFIX = 1
QFIX = 1024
TFIX = 122
BAND = "exclusive_unread_dev"
TRAIN = list(PREREG["split_documented_before_fits"]["train"])
TEST = list(PREREG["split_documented_before_fits"]["test"])
NPROC = 4


def prereg_sha():
    return hashlib.sha256(PREREG_PATH.read_bytes()).hexdigest()


def fit_path(seed, arm):
    if arm == "qaoa_center4":
        circ, tag = "QAOA", "qaoa_center4"
    elif arm == "nov3_geom_tk_tm":
        circ, tag = "Nov3", "nov3_geom_tk_tm"
    else:
        circ, tag = "Nov3", "nov3_center4"
    return RUNS / f"N{NFIX}" / f"s{seed}_L{LFIX}_{circ}_{tag}_Q{QFIX}.json"


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
    meta["band"] = "exclusive_unread_dev_10872_10887"
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
    print(f"[n18geom] B* N{NFIX} seed {seed}", flush=True)
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


def make_propagate(sim, method, freeze=None):
    def propagate(x, _sim=sim, _method=method, _freeze=freeze):
        xl = W.to_le(_method, x)
        if _freeze is not None:
            xl = F.project(xl, _freeze)
        return float(_sim.energy(xl)), xl
    return propagate


def run_fit(task):
    seed, arm, split = task
    if split == "test" and not VECTORS.exists():
        raise RuntimeError("refusing test fit before geometry freeze")
    if arm == "qaoa_center4":
        circ, method, draw_arm = "QAOA", "QAOA", "qaoa_center4"
        freeze = None
        role = "independent_fit"
        optimizer = "FT_v4_center4"
    elif arm == "nov3_geom_tk_tm":
        circ, method, draw_arm = "Nov3", "Nov3rd", "nov3_center4"
        freeze = json.loads(VECTORS.read_text())
        role = "geom_spsa"
        optimizer = "FT_v4_center4_with_tk_tm_projection"
        if seed not in TEST:
            raise RuntimeError(f"geom SPSA on non-test seed {seed}")
    else:
        circ, method, draw_arm = "Nov3", "Nov3rd", "nov3_center4"
        freeze = None
        role = "independent_fit"
        optimizer = "FT_v4_center4"
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
        "reused": False, "stage": "N18GEOM", "continuation": False,
        "ft_alloc_sha256": PREREG["ft_source"]["alloc_py_sha256"],
        "ft_head_commit": PREREG["ft_source"]["head_commit"],
        "optimizer": optimizer, "nstarts": 4,
        "no_l2_or_deeper": True, "not_an_architecture_change": True,
        "not_a_two_template_copy": True, "role": role,
    }
    try:
        if not rec["instance_accepted"]:
            raise RuntimeError(
                f"B* rejected checks={meta.get('hard_checks')} E1={meta.get('E1_parity')}"
            )
        sim = S.make_sim(circ, J, meta["Bstar"], LFIX, engine="m2")
        rng_init, rng_spsa = W.rng_lists(seed, NFIX, meta["Bstar"], LFIX, method)
        propagate = make_propagate(sim, method, freeze)
        out = A.run_equal4(propagate, method, LFIX, QFIX, rng_init, rng_spsa, x0_fn=W.draw_fn(draw_arm))
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
        if freeze is not None:
            x_le = F.project(x_le, freeze)
            for i, name in zip(F.FROZEN_IDX, F.FROZEN_NAMES):
                if abs(x_le[i] - float(freeze["frozen_mean"][name])) > 1e-12:
                    raise RuntimeError(f"geom failed to freeze {name}")
        rec.update({
            "status": "ok", "E_sel": float(out["E"]), "x_ft": x_le.tolist(),
            "x": x_le.tolist(), "K_per_start": int(out["n_iter"]),
            "per_start": QFIX // 4,
            "sel_start": int(out["start_id"]), "sel_cand": out["which"],
            "n_cands": 12, "start_tails": out["tails"],
        })
        if freeze is not None:
            rec["frozen_mean"] = freeze["frozen_mean"]
            rec["frozen_indices"] = list(F.FROZEN_IDX)
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


def load_train_ok():
    rows = []
    for seed in TRAIN:
        p = fit_path(seed, "nov3_center4")
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
    (OUT / "INSTANCES_N18GEOM.json").write_text(json.dumps({"n": len(inst_rows), "rows": inst_rows}, indent=1))
    accepted = {r["seed"] for r in inst_rows if r["accepted"]}
    tasks, skipped = [], []
    for seed in TRAIN:
        if seed not in accepted:
            skipped.append((seed, "nov3_center4"))
            continue
        tasks.append((seed, "nov3_center4", "train"))
    status = {
        "pid": os.getpid(), "stage": "N18GEOM_train", "started": time.time(),
        "bstar_wall_s": time.time() - t_bstar,
        "n_tasks": len(tasks), "n_skipped_rejected_Bstar": len(skipped),
        "prereg_sha256": prereg_sha(), "nproc": NPROC, "N": NFIX, "L": LFIX,
        "Q": QFIX, "T": TFIX, "train": TRAIN, "test": TEST, "band": BAND,
        "ft_alloc_sha256": PREREG["ft_source"]["alloc_py_sha256"],
        "peeked_test_fits": False,
    }
    (OUT / "STATUS_N18GEOM.json").write_text(json.dumps(status, indent=1))
    print(f"[n18geom] train tasks {len(tasks)} skipped {len(skipped)} "
          f"B* wall {status['bstar_wall_s']:.1f}s", flush=True)
    t0 = time.time()
    with ctx.Pool(NPROC) as pool:
        for i, (p, st) in enumerate(pool.imap_unordered(run_fit, tasks, chunksize=1)):
            print(f"[train {i+1}/{len(tasks)}] {time.time()-t0:7.1f}s {st} {Path(p).name}", flush=True)
    status["train_finished"] = time.time()
    status["train_fit_wall_s"] = status["train_finished"] - t0
    (OUT / "STATUS_N18GEOM.json").write_text(json.dumps(status, indent=1))
    print("[n18geom] train done", status["train_fit_wall_s"], flush=True)


def stage_freeze():
    if VECTORS.exists():
        raise SystemExit(f"freeze already written: {VECTORS}")
    rundir = RUNS / f"N{NFIX}"
    test_hits = []
    if rundir.exists():
        for p in rundir.iterdir():
            if any(p.name.startswith(f"s{seed}_") for seed in TEST):
                test_hits.append(p)
    if test_hits:
        raise SystemExit(f"test artifacts exist before freeze: {test_hits[:3]}")
    rows = load_train_ok()
    freeze = F.build_freeze(rows, prereg_sha())
    VECTORS.write_text(json.dumps(freeze, indent=1))
    status_path = OUT / "STATUS_N18GEOM.json"
    status = json.loads(status_path.read_text()) if status_path.exists() else {}
    status["freeze_written"] = True
    status["freeze_sha256"] = hashlib.sha256(VECTORS.read_bytes()).hexdigest()
    status["frozen_mean"] = freeze["frozen_mean"]
    status_path.write_text(json.dumps(status, indent=1))
    print("[n18geom] froze θ_K, θ_M from train n=8; test untouched", freeze["frozen_mean"], flush=True)


def stage_test():
    if not VECTORS.exists():
        raise SystemExit("freeze missing; will not run test")
    freeze = json.loads(VECTORS.read_text())
    if freeze.get("prereg_sha256") != prereg_sha():
        raise SystemExit("freeze prereg hash mismatch")
    warmup_worker()
    inst = json.loads((OUT / "INSTANCES_N18GEOM.json").read_text())
    accepted = {r["seed"] for r in inst["rows"] if r.get("accepted")}
    tasks, skipped = [], []
    for seed in TEST:
        if seed not in accepted:
            skipped.append(seed)
            continue
        tasks.append((seed, "nov3_center4", "test"))
        tasks.append((seed, "qaoa_center4", "test"))
        tasks.append((seed, "nov3_geom_tk_tm", "test"))
    status_path = OUT / "STATUS_N18GEOM.json"
    status = json.loads(status_path.read_text()) if status_path.exists() else {}
    status.update({
        "stage": "N18GEOM_test", "test_started": time.time(),
        "n_test_fit_tasks": len(tasks),
        "n_skipped_rejected_Bstar_test": len(skipped),
        "freeze_sha256": hashlib.sha256(VECTORS.read_bytes()).hexdigest(),
    })
    status_path.write_text(json.dumps(status, indent=1))
    print(f"[n18geom] test fits {len(tasks)}", flush=True)
    ctx = get_context("spawn")
    t0 = time.time()
    with ctx.Pool(NPROC) as pool:
        for i, (p, st) in enumerate(pool.imap_unordered(run_fit, tasks, chunksize=1)):
            print(f"[test {i+1}/{len(tasks)}] {time.time()-t0:7.1f}s {st} {Path(p).name}", flush=True)
    status["test_finished"] = time.time()
    status["test_wall_s"] = status["test_finished"] - t0
    status_path.write_text(json.dumps(status, indent=1))
    print("[n18geom] test done", status["test_wall_s"], flush=True)


def main():
    if len(sys.argv) != 2 or sys.argv[1] not in ("train", "freeze", "test"):
        raise SystemExit("usage: run_n18geom.py {train|freeze|test}")
    stage = sys.argv[1]
    if stage == "train":
        stage_train()
    elif stage == "freeze":
        stage_freeze()
    else:
        stage_test()


if __name__ == "__main__":
    main()
