"""L3SF: N=15 L=3 JOINT vs SHELL_FIRST vs QAOA p=3, Q=3072.

    OMP_NUM_THREADS=1 python Nov9_SK_SPSA/experiments/L3SF/run_l3sf.py
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
import kernel_m2 as K2  # noqa: E402
import shellfirst as SF  # noqa: E402
import sim as S  # noqa: E402
import wrap as W  # noqa: E402

PREREG_PATH = HERE / "PREREG_L3SF.json"
PREREG = json.loads(PREREG_PATH.read_text())
OUT = HERE.parents[1] / "results" / "L3SF"
INST = OUT / "instances"
RUNS = OUT / "runs"
NFIX = 15
ELL = 3
QFIX = 3072
T_JOINT = 378
BAND = "exclusive_unread_dev"


def prereg_sha():
    return hashlib.sha256(PREREG_PATH.read_bytes()).hexdigest()


def run_path(seed, arm):
    circ = "QAOA" if arm == "QAOA" else "Nov3"
    return RUNS / f"N{NFIX}" / f"s{seed}_L{ELL}_{circ}_{arm}_Q{QFIX}.json"


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
    meta["band"] = "exclusive_unread_dev_10368_10375"
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
    print(f"[l3sf] B* N{NFIX} seed {seed}", flush=True)
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
    seed, arm = task
    circ = "QAOA" if arm == "QAOA" else "Nov3"
    method = "QAOA" if circ == "QAOA" else "Nov3rd"
    path = run_path(seed, arm)
    if path.exists():
        return str(path), "skip"
    path.parent.mkdir(parents=True, exist_ok=True)
    warmup_worker()
    t0 = time.time()
    cpu0 = time.process_time()
    J, meta, gs = instance(NFIX, seed)
    rec = {
        "N": NFIX, "seed": seed, "L": ELL, "circuit": circ, "arm": arm, "nstarts": 4,
        "q": 1024, "Q": QFIX, "band": BAND, "Bstar": meta["Bstar"],
        "E0": meta["E0"], "E1_parity": meta.get("E1_parity"),
        "prereg_sha256": prereg_sha(), "cap_inclusive": True,
        "outside_cap_queries": 0, "pid": os.getpid(),
        "instance_accepted": bool(meta.get("accepted")),
        "engine": "m2", "queries": 0, "query_cat": {},
        "reused": False, "stage": "L3SF", "continuation": False,
        "ft_alloc_sha256": PREREG["ft_source"]["alloc_py_sha256"],
        "ft_head_commit": PREREG["ft_source"]["head_commit"],
        "no_pair_phase": True,
        "not_l3ts_joint_then_shell": True,
    }
    try:
        if not rec["instance_accepted"]:
            raise RuntimeError(
                f"B* rejected checks={meta.get('hard_checks')} E1={meta.get('E1_parity')}"
            )
        sim = S.make_sim(circ, J, meta["Bstar"], ELL, engine="m2")
        rng_init, rng_spsa = W.rng_lists(seed, NFIX, meta["Bstar"], ELL, method)

        def propagate(x, _sim=sim, _method=method):
            xl = W.to_le(_method, x)
            return float(_sim.energy(xl)), xl

        if arm == "SHELL_FIRST":
            rec["optimizer"] = "FT_v4_shell_first"
            rec["n_uc_blocks"] = 2
            out = SF.run_shellfirst(propagate, ELL, QFIX, seed, NFIX, meta["Bstar"])
            rec["queries"] = int(out["queries"]["total"])
            rec["query_cat"] = {
                "init": out["queries"]["init"],
                "calibration": out["queries"]["calibration"],
                "perturb": out["queries"]["iteration"],
                "candidate": out["queries"]["candidate"],
                "monitor": out["queries"]["monitoring"],
                "screen": out["queries"]["screen"],
            }
            rec["query_cat_shell"] = {
                "init": out["queries_shell"]["init"],
                "calibration": out["queries_shell"]["calibration"],
                "perturb": out["queries_shell"]["iteration"],
                "candidate": out["queries_shell"]["candidate"],
            }
            rec["query_cat_uc"] = {
                "init": out["queries_uc"]["init"],
                "calibration": out["queries_uc"]["calibration"],
                "perturb": out["queries_uc"]["iteration"],
                "candidate": out["queries_uc"]["candidate"],
            }
            if out["queries"]["total"] != QFIX:
                raise RuntimeError(f"used {out['queries']['total']} cap {QFIX}")
            if out["n_iter_shell"] != SF.T_SHELL or out["n_iter_uc"] != SF.T_UC:
                raise RuntimeError(f"T shell {out['n_iter_shell']} uc {out['n_iter_uc']}")
            x_le = np.asarray(out["aux"], float)
            rec.update({
                "status": "ok", "E_sel": float(out["E"]), "E_shell_stage": float(out["E_shell"]),
                "x_ft": out["x"].tolist(), "x": x_le.tolist(),
                "x_shell": out["x_shell"].tolist(),
                "uc_frozen": out["uc_frozen"].tolist(),
                "shells_frozen": out["shells_frozen"].tolist(),
                "K_per_start": int(out["n_iter_shell"]),
                "K_uc": int(out["n_iter_uc"]),
                "per_start": QFIX // 2 // 4,
                "sel_start": int(out["start_id"]),
                "sel_cand": out["which"],
                "shell_start_id": int(out["shell_start_id"]),
                "shell_which": out["shell_which"],
                "n_cands": 18, "start_tails_shell": out["tails_shell"],
                "start_tails_uc": out["tails_uc"],
            })
            if not np.allclose(x_le[3:], out["shells_frozen"]):
                raise RuntimeError("selected x moved frozen shells")
        else:
            rec["optimizer"] = "FT_v4_center4"
            out = A.run_equal4(
                propagate, method, ELL, QFIX, rng_init, rng_spsa, x0_fn=W.draw_fn(
                    "qaoa_center4" if arm == "QAOA" else "nov3_center4"
                )
            )
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
            if out["n_iter"] != T_JOINT:
                raise RuntimeError(f"n_iter {out['n_iter']} expected {T_JOINT}")
            x_le = np.asarray(out["aux"], float)
            rec.update({
                "status": "ok", "E_sel": float(out["E"]), "x_ft": out["x"].tolist(),
                "x": x_le.tolist(), "K_per_start": int(out["n_iter"]), "per_start": QFIX // 4,
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


def main():
    warmup_worker()
    OUT.mkdir(parents=True, exist_ok=True)
    seeds = list(PREREG["seeds"]["ids"])
    t_bstar = time.time()
    ctx = get_context("spawn")
    with ctx.Pool(4) as pool:
        inst_rows = list(pool.imap(build_one, seeds, chunksize=1))
    inst_rows = sorted(inst_rows, key=lambda r: r["seed"])
    (OUT / "INSTANCES_L3SF.json").write_text(json.dumps({"n": len(inst_rows), "rows": inst_rows}, indent=1))
    accepted = {r["seed"] for r in inst_rows if r["accepted"]}
    tasks, skipped = [], []
    for seed in seeds:
        for arm in PREREG["batch"]["arms"]:
            if seed not in accepted:
                skipped.append((seed, arm))
                continue
            tasks.append((seed, arm))
    status = {
        "pid": os.getpid(), "stage": "L3SF", "started": time.time(),
        "bstar_wall_s": time.time() - t_bstar,
        "n_tasks": len(tasks), "n_skipped_rejected_Bstar": len(skipped),
        "prereg_sha256": prereg_sha(), "nproc": 4, "N": NFIX, "L": ELL,
        "Q": QFIX, "seeds": seeds, "band": BAND,
        "ft_alloc_sha256": PREREG["ft_source"]["alloc_py_sha256"],
        "split": PREREG["split_documented_before_fits"],
        "arms": list(PREREG["batch"]["arms"]),
        "no_pair_phase": True,
        "not_l3ts_joint_then_shell": True,
    }
    (OUT / "STATUS_L3SF.json").write_text(json.dumps(status, indent=1))
    print(f"[l3sf] pid {os.getpid()} tasks {len(tasks)} skipped {len(skipped)} "
          f"B* wall {status['bstar_wall_s']:.1f}s prereg {prereg_sha()[:12]}", flush=True)
    t0 = time.time()
    with ctx.Pool(4) as pool:
        for i, (p, st) in enumerate(pool.imap_unordered(run_one, tasks, chunksize=1)):
            print(f"[{i+1}/{len(tasks)}] {time.time()-t0:7.1f}s {st} {Path(p).name}", flush=True)
    status["finished"] = time.time()
    status["fit_wall_s"] = status["finished"] - t0
    status["wall_s"] = status["finished"] - status["started"]
    (OUT / "STATUS_L3SF.json").write_text(json.dumps(status, indent=1))
    print("[l3sf] done", status["wall_s"], "fits", status["fit_wall_s"], flush=True)


if __name__ == "__main__":
    main()
