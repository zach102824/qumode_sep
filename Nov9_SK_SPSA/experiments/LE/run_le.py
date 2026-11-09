"""Experiment A (N9-LE) milestone-1 driver. Dev seeds only. Checkpoints one JSON per run.

python Nov9_SK_SPSA/experiments/LE/run_le.py --N 6 9 --workers 4
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

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import core as C  # noqa: E402

PREREG_PATH = HERE / "PREREG_LE_M1.json"
PREREG = json.loads(PREREG_PATH.read_text())
OUT = HERE.parents[1] / "results" / "LE"
INST = OUT / "instances"
RUNS = OUT / "runs"


def prereg_sha():
    return hashlib.sha256(PREREG_PATH.read_bytes()).hexdigest()


def instance(N, seed, b):
    INST.mkdir(parents=True, exist_ok=True)
    stem = INST / f"N{N}_s{seed}_b{b}"
    J = C.make_J(seed, N)
    if (stem.with_suffix(".json")).exists():
        meta = json.loads(stem.with_suffix(".json").read_text())
        gs = np.load(stem.with_suffix(".npz"))["gs"]
        assert meta["J_sha256"] == C.J_sha(J)
        return J, meta, gs
    g = C.exact_gs(J, b)
    meta = {"N": N, "seed": seed, "b": b, "J_sha256": C.J_sha(J), "J_upper": C.upper(J).tolist(),
            "E0": g["E0"], "E1": g["E1"], "E0_odd": g["E0_odd"], "Delta_odd": g["E0_odd"] - g["E0"],
            "gs_resid": g["resid"], "F_plus": C.plus_fidelity(g["gs"]),
            "E_plus": float(-b * N)}
    np.savez_compressed(stem.with_suffix(".npz"), gs=g["gs"])
    stem.with_suffix(".json").write_text(json.dumps(meta))
    return J, meta, g["gs"]


def arm_spec(arm):
    return PREREG["arms"][arm]


def build(arm, J, b, L, nf):
    sp = arm_spec(arm)
    if sp["kind"] == "qaoa":
        return C.QAOA(J, b, L)
    return C.Nov9(J, b, L, nf=nf, ext=sp.get("ext", "Uz"), strict=sp.get("strict", False),
                  s3off=sp.get("s3off", False))


def rng(seed, N, b, L, mid, start, stream):
    return np.random.default_rng(np.random.SeedSequence([seed, N, int(round(10 * b)), L, mid, start, stream]))


def starts_for(arm, sim, seed, N, b, L):
    sp = arm_spec(arm)
    mid = sp["method_id"]
    out = []
    if sp["kind"] == "qaoa":
        dt = PREREG["init"]["qaoa_tqa_dt"]
        t = (np.arange(L) + 0.5) / L
        out.append(np.concatenate([t * dt, -(1 - t) * dt]))
        for s in range(1, 4):
            r = rng(seed, N, b, L, mid, s, 2)
            out.append(r.uniform(-0.6, 0.6, size=2 * L))
        return out
    pat = np.array([(-1.0) ** ell for ell in range(L)]) if sp["pattern"] == "alt" else np.ones(L)
    ns = 2 if sp.get("s3off") else 3
    a0 = PREREG["init"]["alpha0"]
    x = np.concatenate([[PREREG["init"]["gamma0"]], [0.0, 0.0], np.arctanh(a0 * pat / C.ALPHA_MAX), np.zeros(ns * L)])
    out.append(x)
    for s in range(1, 4):
        r = rng(seed, N, b, L, mid, s, 2)
        g = r.uniform(-0.5, 0.5)
        th = r.uniform(-0.5, 0.5, size=2)
        amp = r.uniform(0.02, 0.08, size=L) * (1.0 if r.random() < 0.5 else -1.0)
        ss = r.uniform(-0.3, 0.3, size=ns * L)
        out.append(np.concatenate([[g], th, np.arctanh(amp * pat / C.ALPHA_MAX), ss]))
    return out


def run_one(task):
    N, seed, b, L, arm, cap = task
    path = RUNS / f"N{N}" / f"s{seed}_b{b}_L{L}_{arm}_cap{cap}.json"
    if path.exists():
        return str(path), "skip"
    path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    J, meta, gs = instance(N, seed, b)
    nf = PREREG["nf_fit"][str(N)]
    sim = build(arm, J, b, L, nf)
    sp = arm_spec(arm)
    starts = starts_for(arm, sim, seed, N, b, L)
    rec = {"N": N, "seed": seed, "b": b, "L": L, "arm": arm, "cap": cap, "method_id": sp["method_id"],
           "label": sp["label"], "prereg_sha256": prereg_sha(), "nf_fit": None if sp["kind"] == "qaoa" else sim.nf,
           "pid": os.getpid()}
    try:
        r = C.spsa_run(sim.energy, starts, sim.sigma(), lambda s: rng(seed, N, b, L, sp["method_id"], s, 0), cap)
        d = sim.diagnostics(r["x"], gs)
        rec.update({"status": "ok", "E_sel": r["E_sel"], "x": r["x"].tolist(), "queries": r["queries"],
                    "query_cat": r["query_cat"], "K_per_start": r["K_per_start"], "sel_start": r["start"],
                    "sel_cand": r["cand"], "all_cand_E": r["all_cand_E"], "diag": d})
        if sp["kind"] != "qaoa":
            gamma, theta, alpha, s = sim.unpack(r["x"])
            rec["params"] = {"gamma": float(gamma), "theta": theta.tolist(), "alpha": alpha.tolist(), "s": s.tolist()}
            if not sp.get("strict") and cap == max(PREREG["caps"]):
                conv = {}
                for nf2 in PREREG["nf_ladder"]:
                    s2 = C.Nov9(J, b, L, nf=nf2, ext=sp.get("ext", "Uz"), s3off=sp.get("s3off", False))
                    conv[str(nf2)] = s2.diagnostics(r["x"], gs)
                rec["cutoff"] = {k: {kk: v[kk] for kk in ("E", "F", "p_leak", "norm")} for k, v in conv.items()}
                if sp.get("ext", "Uz") == "Uz":
                    s0 = C.Nov9(J, b, L, nf=nf, ext="U0", s3off=sp.get("s3off", False))
                    rec["cross_U0_same_params"] = s0.diagnostics(r["x"], gs)
    except Exception as e:  # logged failure, scored as x0 (protocol)
        d0 = sim.diagnostics(starts[0], gs)
        rec.update({"status": "failure", "error": repr(e), "diag": d0})
    F = rec["diag"]["F"]
    rec["F_plus"] = meta["F_plus"]
    rec["eta"] = (F - meta["F_plus"]) / (1 - meta["F_plus"])
    rec["E0"] = meta["E0"]
    rec["resid_rel"] = (rec["diag"]["E"] - meta["E0"]) / abs(meta["E0"])
    rec["Delta_odd"] = meta["Delta_odd"]
    rec["wall_s"] = time.time() - t0
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(rec))
    tmp.replace(path)
    return str(path), rec["status"]


def tasks(Ns):
    out = []
    for N in Ns:
        for seed in PREREG["seeds"]:
            for b in PREREG["fields"]:
                for L in PREREG["L"]:
                    for arm, sp in PREREG["arms"].items():
                        if L < sp.get("min_L", 1):
                            continue
                        for cap in PREREG["caps"]:
                            out.append((N, seed, b, L, arm, cap))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, nargs="+", required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    for N in a.N:
        assert str(N) in PREREG["nf_fit"], f"N={N} not preregistered"
    for N in a.N:
        for seed in PREREG["seeds"]:
            for b in PREREG["fields"]:
                instance(N, seed, b)
    T = tasks(a.N)
    T.sort(key=lambda t: (t[0], -t[5], t[3]))
    if a.limit:
        T = T[: a.limit]
    print(f"[run_le] pid {os.getpid()} tasks {len(T)} prereg {prereg_sha()[:12]}", flush=True)
    t0 = time.time()
    with Pool(a.workers) as P:
        for i, (p, st) in enumerate(P.imap_unordered(run_one, T, chunksize=1)):
            if st != "skip" or i % 50 == 0:
                print(f"[{i+1}/{len(T)}] {time.time()-t0:7.1f}s {st} {Path(p).name}", flush=True)
    print("[run_le] done", time.time() - t0)


if __name__ == "__main__":
    main()
