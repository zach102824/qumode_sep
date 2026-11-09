"""FTREP D18: N=18 center-4 depth curve. L=2,3 required; L=4 after they finish.

    OMP_NUM_THREADS=1 python Nov9_SK_SPSA/experiments/FTREP/run_ftrep_d18.py
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

PREREG_PATH = HERE / "PREREG_FTREP_D18.json"
PREREG = json.loads(PREREG_PATH.read_text())
OUT = HERE.parents[1] / "results" / "FTREP"
INST = OUT / "instances"
RUNS = OUT / "runs"
NFIX = 18
Q_OF = {int(k): int(v) for k, v in PREREG["batch"]["Q_of_L"].items()}
T_OF = {int(k): int(v) for k, v in PREREG["batch"]["T_of_L"].items()}
BAND = "exclusive_unread_dev"


def prereg_sha():
    return hashlib.sha256(PREREG_PATH.read_bytes()).hexdigest()


def run_path(seed, ell, circ, arm, Q):
    return RUNS / f"N{NFIX}" / f"s{seed}_L{ell}_{circ}_{arm}_Q{Q}.json"


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
    meta["band"] = "exclusive_unread_dev_10320_10327"
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
    print(f"[ftrep-d18] B* N{NFIX} seed {seed}", flush=True)
    J, meta, gs = instance(NFIX, seed)
    row = {
        "N": NFIX, "seed": seed, "accepted": bool(meta.get("accepted")),
        "Bstar": meta.get("Bstar"), "E0": meta.get("E0"),
        "E1_parity": meta.get("E1_parity"), "gap_even": meta.get("gap_even"),
        "gap_odd": meta.get("gap_odd"), "gs_resid": meta.get("gs_resid"),
        "valid": meta.get("valid"), "hard_checks": meta.get("hard_checks"),
        "field_prep_s": meta.get("field_prep_s"), "J_sha256": meta.get("J_sha256"),
    }
    print(f"  accepted={row['accepted']} B*={row['Bstar']:.5f} E1={row['E1_parity']} "
          f"resid={row['gs_resid']:.2e} {row['field_prep_s']}", flush=True)
    return row


def run_one(task):
    seed, ell, arm = task
    Q = Q_OF[ell]
    T = T_OF[ell]
    circ = "QAOA" if arm == "qaoa_center4" else "Nov3"
    method = "QAOA" if circ == "QAOA" else "Nov3rd"
    path = run_path(seed, ell, circ, arm, Q)
    if path.exists():
        return str(path), "skip"
    path.parent.mkdir(parents=True, exist_ok=True)
    warmup_worker()
    t0 = time.time()
    cpu0 = time.process_time()
    J, meta, gs = instance(NFIX, seed)
    rec = {
        "N": NFIX, "seed": seed, "L": ell, "circuit": circ, "arm": arm, "nstarts": 4,
        "q": 1024, "Q": Q, "band": BAND, "Bstar": meta["Bstar"],
        "E0": meta["E0"], "E1_parity": meta.get("E1_parity"),
        "prereg_sha256": prereg_sha(), "cap_inclusive": True,
        "outside_cap_queries": 0, "pid": os.getpid(),
        "instance_accepted": bool(meta.get("accepted")),
        "optimizer": "FT_v4_center4", "engine": "m2", "queries": 0, "query_cat": {},
        "reused": False, "stage": "FTREP_D18", "continuation": False,
        "ft_alloc_sha256": PREREG["ft_source"]["alloc_py_sha256"],
        "ft_head_commit": PREREG["ft_source"]["head_commit"],
        "characterization_not_adoption": True,
    }
    try:
        if not rec["instance_accepted"]:
            raise RuntimeError(
                f"B* rejected checks={meta.get('hard_checks')} E1={meta.get('E1_parity')}"
            )
        sim = S.make_sim(circ, J, meta["Bstar"], ell, engine="m2")
        rng_init, rng_spsa = W.rng_lists(seed, NFIX, meta["Bstar"], ell, method)

        def propagate(x, _sim=sim, _method=method):
            xl = W.to_le(_method, x)
            return float(_sim.energy(xl)), xl

        out = A.run_equal4(propagate, method, ell, Q, rng_init, rng_spsa, x0_fn=W.draw_fn(arm))
        rec["queries"] = int(out["queries"]["total"])
        rec["query_cat"] = {
            "init": out["queries"]["init"],
            "calibration": out["queries"]["calibration"],
            "perturb": out["queries"]["iteration"],
            "candidate": out["queries"]["candidate"],
            "monitor": out["queries"]["monitoring"],
            "screen": out["queries"]["screen"],
        }
        if out["queries"]["total"] != Q:
            raise RuntimeError(f"used {out['queries']['total']} cap {Q}")
        if out["n_iter"] != T:
            raise RuntimeError(f"n_iter {out['n_iter']} expected {T}")
        if out["queries"]["monitoring"] != 0 or out["queries"]["screen"] != 0:
            raise RuntimeError("unexpected screen/monitor")
        x_le = np.asarray(out["aux"], float)
        rec.update({
            "status": "ok", "E_sel": float(out["E"]), "x_ft": out["x"].tolist(),
            "x": x_le.tolist(), "K_per_start": int(out["n_iter"]), "per_start": Q // 4,
            "sel_start": int(out["start_id"]), "sel_cand": out["which"],
            "n_cands": 16, "start_tails": out["tails"],
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


def run_block(label, tasks, ctx):
    print(f"[ftrep-d18] block {label} tasks {len(tasks)}", flush=True)
    t0 = time.time()
    with ctx.Pool(4) as pool:
        for i, (p, st) in enumerate(pool.imap_unordered(run_one, tasks, chunksize=1)):
            print(f"[{label} {i+1}/{len(tasks)}] {time.time()-t0:7.1f}s {st} {Path(p).name}", flush=True)
    return time.time() - t0


def main():
    warmup_worker()
    OUT.mkdir(parents=True, exist_ok=True)
    seeds = list(PREREG["seeds"]["ids"])
    t_bstar = time.time()
    ctx = get_context("spawn")
    with ctx.Pool(4) as pool:
        inst_rows = list(pool.imap(build_one, seeds, chunksize=1))
    inst_rows = sorted(inst_rows, key=lambda r: r["seed"])
    (OUT / "INSTANCES_FTREP_D18.json").write_text(json.dumps({"n": len(inst_rows), "rows": inst_rows}, indent=1))
    accepted = {r["seed"] for r in inst_rows if r["accepted"]}
    def tasks_for(ells):
        tasks, skipped = [], []
        for seed in seeds:
            for ell in ells:
                for arm in PREREG["batch"]["arms"]:
                    if seed not in accepted:
                        skipped.append((seed, ell, arm))
                        continue
                    tasks.append((seed, ell, arm))
        return tasks, skipped
    req, skip_req = tasks_for(PREREG["batch"]["L_required"])
    opt, skip_opt = tasks_for(PREREG["batch"]["L_optional"])
    status = {
        "pid": os.getpid(), "stage": "FTREP_D18", "started": time.time(),
        "bstar_wall_s": time.time() - t_bstar,
        "n_required_tasks": len(req), "n_optional_tasks": len(opt),
        "n_skipped_rejected_Bstar": len(skip_req) + len(skip_opt),
        "prereg_sha256": prereg_sha(), "nproc": 4, "N": NFIX,
        "L_required": PREREG["batch"]["L_required"],
        "L_optional": PREREG["batch"]["L_optional"],
        "Q_of_L": Q_OF, "seeds": seeds, "band": BAND,
        "ft_alloc_sha256": PREREG["ft_source"]["alloc_py_sha256"],
        "characterization_not_adoption": True,
        "l4_attempted": False, "l4_finished": False,
        "arms": ["nov3_center4", "qaoa_center4"],
    }
    (OUT / "STATUS_FTREP_D18.json").write_text(json.dumps(status, indent=1))
    print(f"[ftrep-d18] pid {os.getpid()} required {len(req)} optional {len(opt)} "
          f"B* wall {status['bstar_wall_s']:.1f}s", flush=True)
    status["required_fit_wall_s"] = run_block("L2L3", req, ctx)
    status["required_finished"] = time.time()
    (OUT / "STATUS_FTREP_D18.json").write_text(json.dumps(status, indent=1))
    print("[ftrep-d18] required L=2,3 done; starting L=4", flush=True)
    status["l4_attempted"] = True
    status["optional_fit_wall_s"] = run_block("L4", opt, ctx)
    status["l4_finished"] = True
    status["finished"] = time.time()
    status["fit_wall_s"] = status["required_fit_wall_s"] + status["optional_fit_wall_s"]
    status["wall_s"] = status["finished"] - status["started"]
    (OUT / "STATUS_FTREP_D18.json").write_text(json.dumps(status, indent=1))
    print("[ftrep-d18] done", status["wall_s"], "fits", status["fit_wall_s"], flush=True)


if __name__ == "__main__":
    main()
