"""N21L1: N=21 L=1 Q=1024 v4 center-4. Max 2 workers, BLAS1, serial fallback.

    OMP_NUM_THREADS=1 python Nov9_SK_SPSA/experiments/N21L1/run_n21l1.py
"""
from __future__ import annotations

import hashlib
import json
import os
import resource
import sys
import time
import traceback
from concurrent.futures.process import BrokenProcessPool
from multiprocessing import get_context
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["NUMBA_NUM_THREADS"] = "1"
os.environ["MALLOC_ARENA_MAX"] = "2"

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "FTREP"))
sys.path.insert(0, str(HERE.parent / "N3COND"))
sys.path.insert(0, str(HERE.parent / "LEXACT"))
sys.path.insert(0, str(HERE.parent / "LE"))
sys.path.insert(0, str(HERE.parent / "IMAX"))
import alloc as A  # noqa: E402
import bstar_sparse as BS  # noqa: E402
import core as C  # noqa: E402
import kernel_m2 as K2  # noqa: E402
import sim as S  # noqa: E402
import wrap as W  # noqa: E402

PREREG_PATH = HERE / "PREREG_N21L1.json"
PREREG = json.loads(PREREG_PATH.read_text())
OUT = HERE.parents[1] / "results" / "N21L1"
INST = OUT / "instances"
RUNS = OUT / "runs"
NFIX = 21
LFIX = 1
QFIX = 1024
TFIX = 122
BAND = "exclusive_unread_dev"
NPROC_MAX = 2


def prereg_sha():
    return hashlib.sha256(PREREG_PATH.read_bytes()).hexdigest()


def run_path(seed, circ, arm):
    return RUNS / f"N{NFIX}" / f"s{seed}_L{LFIX}_{circ}_{arm}_Q{QFIX}.json"


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
    meta["band"] = "exclusive_unread_dev_10432_10439"
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


def rss_kib():
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def build_one(seed):
    warmup_worker()
    print(f"[n21l1] B* N{NFIX} seed {seed} rss_kib={rss_kib()}", flush=True)
    J, meta, gs = instance(NFIX, seed)
    row = {
        "N": NFIX, "seed": seed, "accepted": bool(meta.get("accepted")),
        "Bstar": meta.get("Bstar"), "E0": meta.get("E0"),
        "E1_parity": meta.get("E1_parity"), "gap_even": meta.get("gap_even"),
        "gap_odd": meta.get("gap_odd"), "gs_resid": meta.get("gs_resid"),
        "valid": meta.get("valid"), "hard_checks": meta.get("hard_checks"),
        "field_prep_s": meta.get("field_prep_s"), "J_sha256": meta.get("J_sha256"),
        "peak_rss_kib": rss_kib(),
    }
    print(f"  accepted={row['accepted']} B*={row['Bstar']:.5f} E1={row['E1_parity']} "
          f"resid={row['gs_resid']:.2e} wall={row['field_prep_s']:.1f}s rss={row['peak_rss_kib']}",
          flush=True)
    return row


def run_one(task):
    seed, arm = task
    circ = "QAOA" if arm == "qaoa_center4" else "Nov3"
    method = "QAOA" if circ == "QAOA" else "Nov3rd"
    path = run_path(seed, circ, arm)
    if path.exists():
        return str(path), "skip"
    path.parent.mkdir(parents=True, exist_ok=True)
    warmup_worker()
    t0 = time.time()
    cpu0 = time.process_time()
    J, meta, gs = instance(NFIX, seed)
    rec = {
        "N": NFIX, "seed": seed, "L": LFIX, "circuit": circ, "arm": arm, "nstarts": 4,
        "q": 1024, "Q": QFIX, "band": BAND, "Bstar": meta["Bstar"],
        "E0": meta["E0"], "E1_parity": meta.get("E1_parity"),
        "prereg_sha256": prereg_sha(), "cap_inclusive": True,
        "outside_cap_queries": 0, "pid": os.getpid(),
        "instance_accepted": bool(meta.get("accepted")),
        "optimizer": "FT_v4_center4", "engine": "m2", "queries": 0, "query_cat": {},
        "reused": False, "stage": "N21L1", "continuation": False,
        "ft_alloc_sha256": PREREG["ft_source"]["alloc_py_sha256"],
        "ft_head_commit": PREREG["ft_source"]["head_commit"],
        "no_l2_or_deeper": True,
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
            "monitor": out["queries"]["monitoring"],
            "screen": out["queries"]["screen"],
        }
        if out["queries"]["total"] != QFIX:
            raise RuntimeError(f"used {out['queries']['total']} cap {QFIX}")
        if out["n_iter"] != TFIX:
            raise RuntimeError(f"n_iter {out['n_iter']} expected {TFIX}")
        if out["queries"]["monitoring"] != 0 or out["queries"]["screen"] != 0:
            raise RuntimeError("unexpected screen/monitor")
        x_le = np.asarray(out["aux"], float)
        rec.update({
            "status": "ok", "E_sel": float(out["E"]), "x_ft": out["x"].tolist(),
            "x": x_le.tolist(), "K_per_start": int(out["n_iter"]), "per_start": QFIX // 4,
            "sel_start": int(out["start_id"]), "sel_cand": out["which"],
            "n_cands": 12, "start_tails": out["tails"],
        })
        d = sim.diagnostics(x_le, gs)
        rec["diag"] = d
        rec["E_err_over_N"] = (d["E"] - meta["E0"]) / NFIX
        rec["F"] = d["F"]
    except Exception as e:
        rec.update({"status": "failure", "error": repr(e), "traceback": traceback.format_exc()})
    rec["wall_s"] = time.time() - t0
    rec["cpu_s"] = time.process_time() - cpu0
    rec["peak_rss_kib"] = rss_kib()
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(rec))
    tmp.replace(path)
    return str(path), rec.get("status", "failure")


def map_with_fallback(fn, items, nproc, label):
    """Try nproc workers; on OOM/worker death retry remaining items serially."""
    used = int(nproc)
    if nproc > 1:
        try:
            ctx = get_context("spawn")
            out = []
            with ctx.Pool(nproc, initializer=warmup_worker, maxtasksperchild=1) as pool:
                for i, r in enumerate(pool.imap(fn, items, chunksize=1)):
                    out.append(r)
                    print(f"[{label} {i+1}/{len(items)}] nproc={nproc} {r if not isinstance(r, tuple) else r[-1]}",
                          flush=True)
            return out, used, None
        except (MemoryError, OSError, BrokenProcessPool) as e:
            print(f"[{label}] nproc={nproc} failed ({type(e).__name__}: {e}); fallback serial",
                  flush=True)
            used = 1
            err = f"{type(e).__name__}: {e}"
        except Exception as e:
            print(f"[{label}] nproc={nproc} failed ({type(e).__name__}: {e}); fallback serial",
                  flush=True)
            used = 1
            err = f"{type(e).__name__}: {e}"
        else:
            err = None
    else:
        err = None
    out = []
    for i, t in enumerate(items):
        r = fn(t)
        out.append(r)
        print(f"[{label} {i+1}/{len(items)}] nproc=1 {r if not isinstance(r, tuple) else r[-1]}",
              flush=True)
    return out, used, err


def main():
    warmup_worker()
    OUT.mkdir(parents=True, exist_ok=True)
    seeds = list(PREREG["seeds"]["ids"])
    t_bstar = time.time()
    inst_rows, nproc_bstar, bstar_fallback = map_with_fallback(
        build_one, seeds, NPROC_MAX, "bstar"
    )
    inst_rows = sorted(inst_rows, key=lambda r: r["seed"])
    (OUT / "INSTANCES_N21L1.json").write_text(json.dumps({"n": len(inst_rows), "rows": inst_rows}, indent=1))
    accepted = {r["seed"] for r in inst_rows if r["accepted"]}
    tasks, skipped = [], []
    for seed in seeds:
        for arm in PREREG["batch"]["arms"]:
            if seed not in accepted:
                skipped.append((seed, arm))
                continue
            tasks.append((seed, arm))
    status = {
        "pid": os.getpid(), "stage": "N21L1", "started": time.time(),
        "bstar_wall_s": time.time() - t_bstar,
        "n_tasks": len(tasks), "n_skipped_rejected_Bstar": len(skipped),
        "prereg_sha256": prereg_sha(), "nproc_max": NPROC_MAX,
        "nproc_bstar": nproc_bstar, "bstar_fallback": bstar_fallback,
        "N": NFIX, "L": LFIX, "Q": QFIX, "T": TFIX, "seeds": seeds, "band": BAND,
        "ft_alloc_sha256": PREREG["ft_source"]["alloc_py_sha256"],
        "arms": list(PREREG["batch"]["arms"]),
        "no_l2_or_deeper": True,
    }
    (OUT / "STATUS_N21L1.json").write_text(json.dumps(status, indent=1))
    print(f"[n21l1] pid {os.getpid()} tasks {len(tasks)} skipped {len(skipped)} "
          f"B* wall {status['bstar_wall_s']:.1f}s nproc_bstar={nproc_bstar} "
          f"prereg {prereg_sha()[:12]}", flush=True)
    t0 = time.time()
    fit_rows, nproc_fit, fit_fallback = map_with_fallback(
        run_one, tasks, NPROC_MAX, "fit"
    )
    status["finished"] = time.time()
    status["fit_wall_s"] = status["finished"] - t0
    status["wall_s"] = status["finished"] - status["started"]
    status["nproc_fit"] = nproc_fit
    status["fit_fallback"] = fit_fallback
    (OUT / "STATUS_N21L1.json").write_text(json.dumps(status, indent=1))
    print("[n21l1] done", status["wall_s"], "fits", status["fit_wall_s"],
          "nproc_fit", nproc_fit, flush=True)
    ok = sum(1 for _, st in fit_rows if st == "ok")
    if ok == 0 and tasks:
        raise SystemExit("N=21 produced zero ok fits after ≤2-worker attempt")


if __name__ == "__main__":
    main()
