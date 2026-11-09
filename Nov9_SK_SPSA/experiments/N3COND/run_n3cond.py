"""N3COND driver. Scale block first, fits second. BLAS threads = 1.

    OMP_NUM_THREADS=1 python Nov9_SK_SPSA/experiments/N3COND/run_n3cond.py --stage scale --N 9
    OMP_NUM_THREADS=1 python Nov9_SK_SPSA/experiments/N3COND/run_n3cond.py --stage fit --N 9 --workers 4
"""
from __future__ import annotations

import argparse
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
sys.path.insert(0, str(HERE.parent / "LE"))
sys.path.insert(0, str(HERE.parent / "IMAX"))
import bstar as BS  # noqa: E402
import core as C  # noqa: E402
import nov3 as N  # noqa: E402
import spsa as S  # noqa: E402

PREREG_PATH = HERE / "PREREG_N3COND_B1.json"
PREREG = json.loads(PREREG_PATH.read_text())
OUT = HERE.parents[1] / "results" / "N3COND"
INST = OUT / "instances"
RUNS = OUT / "runs"
DEV = PREREG["seeds"]["dev_scale"]
HOLD = PREREG["seeds"]["holdout"]
NPROBE = 1  # one axial probe per coordinate per cold start; repeats are identical at exact energy


def prereg_sha():
    return hashlib.sha256(PREREG_PATH.read_bytes()).hexdigest()


def band_of(seed):
    if seed in DEV:
        return "dev"
    if seed in HOLD:
        return "holdout"
    raise ValueError(f"seed {seed} is outside the preregistered split")


def accept(meta):
    return bool(meta["valid"] and meta["E1_parity"] == "odd" and meta["gap_even"] is not None
                and meta["gap_even"] > 0 and meta["gs_resid"] < 1e-9)


def instance(Nqubits, seed):
    INST.mkdir(parents=True, exist_ok=True)
    pj, pz = INST / f"N{Nqubits}_s{seed}_Bstar.json", INST / f"N{Nqubits}_s{seed}_Bstar.npz"
    J = C.make_J(seed, Nqubits)
    if pj.exists():
        meta = json.loads(pj.read_text())
        assert meta["J_sha256"] == C.J_sha(J)
        gs = np.load(pz)["gs"]
        return J, meta, gs
    J, meta, gs = BS.build(Nqubits, seed)
    meta = dict(meta)
    meta["accepted_n3cond"] = accept(meta)
    meta["gs_prep"] = "classical exact diagonalization, not charged to Q"
    np.savez_compressed(pz, gs=gs)
    pj.write_text(json.dumps(meta))
    return J, meta, gs


def scale_path(Nqubits):
    return OUT / f"SCALE_N{Nqubits}.json"


def run_scale(Nqubits, L_list):
    if any(s in HOLD for s in DEV):
        raise RuntimeError("dev/holdout overlap")
    cells = {}
    t0 = time.time()
    for L in L_list:
        for circuit in ("Nov3", "QAOA"):
            rows = []
            queries = 0
            fam_index = None
            for seed in DEV:
                J, meta, _ = instance(Nqubits, seed)
                if not meta["accepted_n3cond"]:
                    raise RuntimeError(f"dev seed {seed} N{Nqubits} failed B* acceptance; scale not frozen")
                sim = N.make_sim(circuit, J, meta["Bstar"], L)
                if circuit == "Nov3" and C.rms(C.upper(C.feature_M(J, C.packing(Nqubits)))) <= 0:
                    raise RuntimeError("M feature is zero; refusing to drop it")
                fam_index = sim.families()
                sig = sim.sigma0()
                cid = N.circuit_id(circuit)
                starts = N.starts_for(circuit, seed, Nqubits, L)

                def probe(x0, sim=sim, sig=sig):
                    nonlocal queries

                    def f(x):
                        nonlocal queries
                        queries += 1
                        return sim.energy(x)

                    return S.axial_probe(f, x0, sig)

                for x0 in starts:
                    rows.append(probe(x0))
            block = np.vstack(rows)
            lam, a0, g, gref = S.freeze_lambda(block)
            expect = len(DEV) * 4 * sim.nparam() * 2
            if queries != expect:
                raise RuntimeError(f"scale queries {queries} != {expect}")
            fam = {}
            for name, ix in fam_index.items():
                fam[name] = float(np.median(g[ix]))
            key = f"N{Nqubits}_L{L}_{circuit}"
            cells[key] = {
                "N": Nqubits, "L": L, "circuit": circuit,
                "lambda": lam.tolist(), "a0": a0, "g_median": g.tolist(), "g_ref": gref,
                "family_median_gabs": fam, "queries": queries,
                "lambda_min": float(lam.min()), "lambda_max": float(lam.max()),
                "n_clipped": int(np.sum((lam <= S.LAM_LO + 1e-15) | (lam >= S.LAM_HI - 1e-15))),
                "probe": "axial",
            }
            print(f"[scale] {key} queries {queries} a0 {a0:.6g} lambda [{lam.min():.3f},{lam.max():.3f}] "
                  f"clipped {cells[key]['n_clipped']}", flush=True)
    payload = {
        "prereg_sha256": prereg_sha(), "N": Nqubits, "dev_seeds": DEV, "holdout_not_used": HOLD,
        "probes_per_start": NPROBE, "lambda_clip": [S.LAM_LO, S.LAM_HI],
        "cells": cells, "wall_s": time.time() - t0, "pid": os.getpid(),
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    sha = hashlib.sha256(raw.encode()).hexdigest()
    OUT.mkdir(parents=True, exist_ok=True)
    path = scale_path(Nqubits)
    path.write_text(json.dumps({"sha256": sha, "body": payload}))
    print(f"[scale] wrote {path} sha {sha[:12]}", flush=True)
    return path


def load_scale(Nqubits):
    path = scale_path(Nqubits)
    data = json.loads(path.read_text())
    body = data["body"]
    raw = json.dumps(body, sort_keys=True, separators=(",", ":"))
    sha = hashlib.sha256(raw.encode()).hexdigest()
    if sha != data["sha256"]:
        raise RuntimeError("scale file hash mismatch")
    if body["prereg_sha256"] != prereg_sha():
        raise RuntimeError("scale file was frozen under a different prereg")
    if body["dev_seeds"] != DEV or body["holdout_not_used"] != HOLD:
        raise RuntimeError("scale file seed split does not match the prereg")
    if any(s in body["dev_seeds"] for s in HOLD):
        raise RuntimeError("holdout leaked into the scale file")
    body["sha256"] = sha
    return body


def run_one(task):
    Nqubits, seed, L, circuit, cond, q = task
    path = RUNS / f"N{Nqubits}" / f"s{seed}_L{L}_{circuit}_{cond}_q{q}.json"
    if path.exists():
        return str(path), "skip"
    path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    scale = load_scale(Nqubits)
    J, meta, gs = instance(Nqubits, seed)
    Q = q * L
    cell = scale["cells"][f"N{Nqubits}_L{L}_{circuit}"]
    sim = N.make_sim(circuit, J, meta["Bstar"], L)
    starts = N.starts_for(circuit, seed, Nqubits, L)
    cid = N.circuit_id(circuit)
    sig = sim.sigma0()
    a_override = None
    if cond == "C1":
        sig = sig * np.asarray(cell["lambda"], float)
        a_override = cell["a0"]
    elif cond != "C0":
        raise ValueError(cond)
    rec = {
        "N": Nqubits, "seed": seed, "L": L, "circuit": circuit, "conditioner": cond, "q": q, "Q": Q,
        "band": band_of(seed), "Bstar": meta["Bstar"], "circuit_id": cid,
        "method_id": N.method_id(circuit, cond), "prereg_sha256": prereg_sha(),
        "scale_sha256": scale["sha256"], "instance_accepted": meta["accepted_n3cond"],
        "pid": os.getpid(), "diag_recompute_not_charged": 1,
    }
    try:
        if not meta["accepted_n3cond"]:
            raise RuntimeError("instance failed acceptance")
        out = S.spsa_run(sim.energy, starts, sig, lambda s: N.rng_for(seed, Nqubits, L, cid, s, 0), Q, a_override=a_override)
        d = sim.diagnostics(out["x"], gs)
        rec.update({"status": "ok", "E_sel": out["E_sel"], "x": out["x"].tolist(), "queries": out["queries"],
                    "query_cat": out["query_cat"], "K_per_start": out["K_per_start"], "per_start": out["per_start"],
                    "sel_start": out["start"], "sel_cand": out["cand"], "all_cand_E": out["all_cand_E"],
                    "start_meta": out["start_meta"], "diag": d})
        if cond == "C1":
            rec["a0"] = cell["a0"]
            rec["lambda_min"] = cell["lambda_min"]
            rec["lambda_max"] = cell["lambda_max"]
    except Exception as e:
        d0 = sim.diagnostics(starts[0], gs)
        rec.update({"status": "failure", "error": repr(e), "diag": d0, "queries": rec.get("queries", 0)})
    F = rec["diag"]["F"]
    rec["F_plus"] = meta["F_plus"]
    rec["eta"] = (F - meta["F_plus"]) / (1.0 - meta["F_plus"])
    rec["E0"] = meta["E0"]
    rec["E1"] = meta["E1"]
    rec["E_plus"] = meta["E_plus"]
    rec["E1_parity"] = meta["E1_parity"]
    rec["gap_even"] = meta["gap_even"]
    rec["gap_E1_E0"] = meta["gap_E1_E0"]
    rec["closure_E"] = (meta["E_plus"] - rec["diag"]["E"]) / (meta["E_plus"] - meta["E0"])
    rec["resid_rel"] = (rec["diag"]["E"] - meta["E0"]) / abs(meta["E0"])
    rec["wall_s"] = time.time() - t0
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(rec))
    tmp.replace(path)
    return str(path), rec["status"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("scale", "fit"), required=True)
    ap.add_argument("--N", type=int, nargs="+", required=True)
    ap.add_argument("--L", type=int, nargs="+", default=None)
    ap.add_argument("--q", type=int, nargs="+", default=None)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    allowed = set(PREREG["batch_B1"]["N"]) | set(PREREG["batch_B2_precommitted"]["N"])
    L_default = PREREG["batch_B1"]["L"]
    q_default = PREREG["batch_B1"]["q"]
    Ls = a.L if a.L is not None else L_default
    qs = a.q if a.q is not None else q_default
    assert all(n in allowed for n in a.N)
    assert all(L in L_default for L in Ls)
    assert all(q in q_default for q in qs)
    assert os.environ["OMP_NUM_THREADS"] == "1"
    seeds = DEV + HOLD
    print(f"[n3cond] pid {os.getpid()} stage {a.stage} prereg {prereg_sha()[:12]}", flush=True)
    for n in a.N:
        for seed in seeds:
            _, meta, _ = instance(n, seed)
            print(f"[inst] N{n} s{seed} {band_of(seed)} B*={meta['Bstar']:.6f} S_A={meta['S_A']:.4f} "
                  f"gap={meta['gap_E1_E0']:.5f} even={meta['gap_even']:.4f} E1={meta['E1_parity']} "
                  f"F+={meta['F_plus']:.4f} accept={meta['accepted_n3cond']}", flush=True)
    if a.stage == "scale":
        for n in a.N:
            run_scale(n, Ls)
        return
    tasks = []
    for n in a.N:
        sc = load_scale(n)
        have = set(sc["cells"])
        for L in Ls:
            for circuit in ("Nov3", "QAOA"):
                if f"N{n}_L{L}_{circuit}" not in have:
                    raise RuntimeError(f"missing scale cell N{n} L{L} {circuit}")
        for seed in seeds:
            meta = json.loads((INST / f"N{n}_s{seed}_Bstar.json").read_text())
            if not meta["accepted_n3cond"]:
                print(f"[skip] N{n} s{seed} not accepted", flush=True)
                continue
            for L in Ls:
                for circuit in ("Nov3", "QAOA"):
                    for cond in ("C0", "C1"):
                        for q in qs:
                            tasks.append((n, seed, L, circuit, cond, q))
    tasks.sort(key=lambda t: (t[0], -t[5], -t[2], t[3], t[4], t[1]))
    status = {"pid": os.getpid(), "stage": "fit", "started": time.time(), "n_tasks": len(tasks),
              "prereg_sha256": prereg_sha(), "tasks_head": [list(t) for t in tasks[:8]]}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "STATUS.json").write_text(json.dumps(status))
    print(f"[fit] tasks {len(tasks)} workers {a.workers}", flush=True)
    t0 = time.time()
    with Pool(a.workers) as P:
        for i, (p, st) in enumerate(P.imap_unordered(run_one, tasks, chunksize=1)):
            if st != "skip":
                print(f"[{i+1}/{len(tasks)}] {time.time()-t0:7.1f}s {st} {Path(p).name}", flush=True)
    status["finished"] = time.time()
    status["wall_s"] = status["finished"] - status["started"]
    (OUT / "STATUS.json").write_text(json.dumps(status))
    print("[fit] done", status["wall_s"], flush=True)


if __name__ == "__main__":
    main()
