"""IMAX milestone-1 driver (B* fields, dev seeds only). Checkpoints one JSON per run.

OMP_NUM_THREADS=1 python Nov9_SK_SPSA/experiments/IMAX/run_imax.py --N 6 9 --workers 4
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
sys.path.insert(0, str(HERE.parent / "LE"))
sys.path.insert(0, str(HERE))
import core as C  # noqa: E402
import bstar as BS  # noqa: E402

PREREG_PATH = HERE / "PREREG_IMAX_M1.json"
PREREG = json.loads(PREREG_PATH.read_text())
OUT = HERE.parents[1] / "results" / "IMAX"
INST = OUT / "instances"
RUNS = OUT / "runs"
TOL, NTOL = 1e-8, 1e-10


def prereg_sha():
    return hashlib.sha256(PREREG_PATH.read_bytes()).hexdigest()


def instance(N, seed):
    INST.mkdir(parents=True, exist_ok=True)
    pj, pz = INST / f"N{N}_s{seed}_Bstar.json", INST / f"N{N}_s{seed}_Bstar.npz"
    J = C.make_J(seed, N)
    if pj.exists():
        meta = json.loads(pj.read_text())
        assert meta["J_sha256"] == C.J_sha(J)
        return J, meta, np.load(pz)["gs"]
    J, meta, gs = BS.build(N, seed)
    np.savez_compressed(pz, gs=gs)
    pj.write_text(json.dumps(meta))
    return J, meta, gs


def build(sp, J, b, L, nf):
    if sp["kind"] == "qaoa":
        return C.QAOA(J, b, L)
    return C.Nov9(J, b, L, nf=nf, ext=sp.get("ext", "Uz"), strict=sp.get("strict", False), axis=sp["axis"])


def rng(seed, N, L, mid, start, stream):
    return np.random.default_rng(np.random.SeedSequence([seed, N, 0, L, mid, start, stream]))


def starts_for(sp, seed, N, L):
    mid = sp["method_id"]
    out = []
    if sp["kind"] == "qaoa":
        t = (np.arange(L) + 0.5) / L
        out.append(np.concatenate([t * 0.75, -(1 - t) * 0.75]))
        for s in range(1, 4):
            out.append(rng(seed, N, L, mid, s, 2).uniform(-0.6, 0.6, size=2 * L))
        return out
    pat = np.array([(-1.0) ** ell for ell in range(L)])
    out.append(np.concatenate([[0.25], [0.0, 0.0], np.arctanh(0.06 * pat / C.ALPHA_MAX), np.zeros(3 * L)]))
    for s in range(1, 4):
        # init stream keyed on the RE/IM-shared id so IM and RE start from identical points
        r = rng(seed, N, L, 30, s, 2)
        g = r.uniform(-0.5, 0.5)
        th = r.uniform(-0.5, 0.5, size=2)
        amp = r.uniform(0.02, 0.08, size=L) * (1.0 if r.random() < 0.5 else -1.0)
        ss = r.uniform(-0.3, 0.3, size=3 * L)
        out.append(np.concatenate([[g], th, np.arctanh(amp * pat / C.ALPHA_MAX), ss]))
    return out


def run_one(task):
    N, seed, L, arm, cap = task
    path = RUNS / f"N{N}" / f"s{seed}_L{L}_{arm}_cap{cap}.json"
    if path.exists():
        return str(path), "skip"
    path.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    J, meta, gs = instance(N, seed)
    b = meta["Bstar"]
    sp = PREREG["arms"][arm]
    nf = PREREG["nf_fit"]
    sim = build(sp, J, b, L, nf)
    starts = starts_for(sp, seed, N, L)
    rec = {"N": N, "seed": seed, "Bstar": b, "L": L, "arm": arm, "cap": cap, "method_id": sp["method_id"],
           "label": sp["label"], "prereg_sha256": prereg_sha(), "instance_valid": meta["valid"],
           "nf_fit": None if sp["kind"] == "qaoa" else sim.nf, "pid": os.getpid()}
    try:
        # SPSA perturbation stream keyed per arm (independent noise per method)
        r = C.spsa_run(sim.energy, starts, sim.sigma(), lambda s: rng(seed, N, L, sp["method_id"], s, 0), cap)
        d = sim.diagnostics(r["x"], gs)
        rec.update({"status": "ok", "E_sel": r["E_sel"], "x": r["x"].tolist(), "queries": r["queries"],
                    "query_cat": r["query_cat"], "K_per_start": r["K_per_start"], "sel_start": r["start"],
                    "sel_cand": r["cand"], "all_cand_E": r["all_cand_E"], "diag": d})
        if sp["kind"] != "qaoa":
            gamma, theta, alpha, s = sim.unpack(r["x"])
            rec["params"] = {"gamma": float(gamma), "theta": theta.tolist(), "alpha": alpha.tolist(), "s": s.tolist()}
            if not sp.get("strict"):
                ladder = PREREG["nf_ladder"] if cap == max(PREREG["caps"]) else [nf, 48]
                conv = {}
                for nf2 in ladder:
                    s2 = sim if nf2 == nf else C.Nov9(J, b, L, nf=nf2, ext=sp.get("ext", "Uz"), axis=sp["axis"])
                    conv[str(nf2)] = s2.diagnostics(r["x"], gs)
                rec["cutoff"] = {k: {kk: v[kk] for kk in ("E", "F", "p_leak", "norm")} for k, v in conv.items()}
                a, z = conv[str(nf)], conv["48"]
                rec["cutoff_certified"] = bool(max(abs(a["E"] - z["E"]), abs(a["F"] - z["F"]), abs(a["p_leak"] - z["p_leak"])) <= TOL
                                              and abs(a["norm"] - 1) <= NTOL and abs(z["norm"] - 1) <= NTOL)
                if sp.get("ext", "Uz") == "Uz" and cap == max(PREREG["caps"]):
                    s0 = C.Nov9(J, b, L, nf=nf, ext="U0", axis=sp["axis"])
                    rec["cross_U0_same_params"] = s0.diagnostics(r["x"], gs)
    except Exception as e:  # logged failure, scored at start 0 (protocol)
        d0 = sim.diagnostics(starts[0], gs)
        rec.update({"status": "failure", "error": repr(e), "diag": d0})
    F = rec["diag"]["F"]
    rec["F_plus"] = meta["F_plus"]
    rec["eta"] = (F - meta["F_plus"]) / (1 - meta["F_plus"])
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, nargs="+", required=True)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    allowed = set(PREREG["N_stage1"]) | set(PREREG["N_stage2"])
    assert all(N in allowed for N in a.N), "N not preregistered"
    for N in a.N:
        for seed in PREREG["seeds"]:
            _, m, _ = instance(N, seed)
            print(f"[inst] N{N} s{seed} B*={m['Bstar']:.6f} S_A={m['S_A']:.4f} gap={m['gap_E1_E0']:.5f} "
                  f"even_gap={m['gap_even']:.4f} F+={m['F_plus']:.4f} valid={m['valid']}", flush=True)
    T = [(N, seed, L, arm, cap) for N in a.N for seed in PREREG["seeds"] for L in PREREG["L"]
         for arm in PREREG["arms"] for cap in PREREG["caps"]]
    T = [t for t in T if json.loads((INST / f"N{t[0]}_s{t[1]}_Bstar.json").read_text())["valid"]]
    T.sort(key=lambda t: (t[0], -t[4], -t[2]))
    print(f"[run_imax] pid {os.getpid()} tasks {len(T)} prereg {prereg_sha()[:12]}", flush=True)
    t0 = time.time()
    with Pool(a.workers) as P:
        for i, (p, st) in enumerate(P.imap_unordered(run_one, T, chunksize=1)):
            if st != "skip":
                print(f"[{i+1}/{len(T)}] {time.time()-t0:7.1f}s {st} {Path(p).name}", flush=True)
    print("[run_imax] done", time.time() - t0, flush=True)


if __name__ == "__main__":
    main()
