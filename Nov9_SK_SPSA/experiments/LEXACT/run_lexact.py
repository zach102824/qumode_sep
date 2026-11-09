"""LEXACT M1: conformance timings and short inclusive SPSA traces.

    OMP_NUM_THREADS=1 python Nov9_SK_SPSA/experiments/LEXACT/run_lexact.py
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "N3COND"))
sys.path.insert(0, str(HERE.parent / "LE"))
sys.path.insert(0, str(HERE.parent / "IMAX"))
import bstar_sparse as BS  # noqa: E402
import core as C  # noqa: E402
import kernel as K  # noqa: E402
import nov3 as N  # noqa: E402
import sim as S  # noqa: E402
import spsa as SP  # noqa: E402

PREREG_PATH = HERE / "PREREG_LEXACT_M1.json"
PREREG = json.loads(PREREG_PATH.read_text())
OUT = HERE.parents[1] / "results" / "LEXACT"
INST = OUT / "instances"
N3INST = HERE.parents[1] / "results" / "N3COND" / "instances"


def prereg_sha():
    return hashlib.sha256(PREREG_PATH.read_bytes()).hexdigest()


def accept(meta):
    return bool(meta["valid"] and meta.get("hard_checks", {}).get("gs_flip_even", False)
                and meta["gs_resid"] < 1e-9)


def instance(nq, seed):
    INST.mkdir(parents=True, exist_ok=True)
    cached = [
        N3INST / f"N{nq}_s{seed}_Bstar.json",
        INST / f"N{nq}_s{seed}_Bstar.json",
    ]
    for pj in cached:
        pz = pj.with_suffix(".npz")
        if pj.exists() and pz.exists():
            meta = json.loads(pj.read_text())
            J = C.make_J(seed, nq)
            assert meta["J_sha256"] == C.J_sha(J)
            gs = np.load(pz)["gs"]
            meta = dict(meta)
            meta["accepted_lexact"] = accept(meta) if "valid" in meta else bool(meta.get("accepted_n3cond"))
            meta["source"] = str(pj)
            return J, meta, gs
    t0 = time.perf_counter()
    J, meta, gs = BS.build(nq, seed)
    meta = dict(meta)
    meta["accepted_lexact"] = accept(meta)
    meta["accepted_n3cond"] = meta["accepted_lexact"]
    meta["field_prep_s"] = time.perf_counter() - t0
    meta["gs_prep"] = meta.get("gs_prep", "classical, not charged to Q")
    pj = INST / f"N{nq}_s{seed}_Bstar.json"
    slim = {k: v for k, v in meta.items() if k != "J_upper"}
    pj.write_text(json.dumps(slim))
    np.savez_compressed(pj.with_suffix(".npz"), gs=gs)
    return J, meta, gs


class TraceBudget(SP.Budget):
    def __init__(self, f, cap):
        super().__init__(f, cap)
        self.trace = []

    def __call__(self, x, cat):
        e = super().__call__(x, cat)
        self.trace.append(float(e))
        return e


def spsa_traced(f, starts, sig, rng_for_start, cap):
    per, K = SP.split_cap(cap, len(starts))
    B = TraceBudget(f, cap)
    allc, meta = [], []
    for si, x0 in enumerate(starts):
        cands, m = SP.spsa_start(B, np.asarray(x0, float), sig, rng_for_start(si), K)
        for e, name, xx in cands:
            allc.append((e, si, name, xx))
        meta.append(m)
    e, si, name, xx = min(allc, key=lambda t: t[0])
    if B.n != cap:
        raise RuntimeError(f"used {B.n} cap {cap}")
    return {"E_sel": e, "start": si, "cand": name, "x": xx.tolist(), "queries": B.n,
            "query_cat": dict(B.cat), "K_per_start": K, "trace": B.trace,
            "outside_cap_queries": 0, "start_meta": meta}


def time_calls(fn, n, warmup=2):
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t0) / n


def rss_mb():
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return float(line.split()[1]) / 1024.0
    return float("nan")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    status = {"pid": os.getpid(), "stage": "LEXACT_M1", "started": time.time(),
              "prereg_sha256": prereg_sha(), "prior_ledger_preserved": 772224}
    (OUT / "STATUS.json").write_text(json.dumps(status))
    print(f"[lexact] pid {os.getpid()} prereg {prereg_sha()[:12]} rss {rss_mb():.1f}MB", flush=True)

    t_comp0 = time.perf_counter()
    K.warmup()
    compile_s = time.perf_counter() - t_comp0
    print(f"[lexact] compile/cache {compile_s:.3f}s", flush=True)

    collisions = []
    for s in range(10032, 10048):
        for folder in (N3INST, HERE.parents[1] / "results" / "LE" / "instances",
                       HERE.parents[1] / "results" / "IMAX" / "instances"):
            if list(folder.glob(f"*s{s}*")):
                collisions.append(s)
    rec = {
        "pid": os.getpid(), "prereg_sha256": prereg_sha(), "compile_s": compile_s,
        "seed_collisions_10032_10047": collisions,
        "prior_n3cond_ledger": 772224, "rss_mb_start": rss_mb(),
        "timings": [], "spsa": [], "conformance_spot": [],
    }

    cells = [(6, 1, 10010), (6, 6, 10010), (9, 1, 10010), (9, 6, 10010),
             (12, 1, 10010), (12, 6, 10010), (15, 1, 10032), (15, 3, 10032), (18, 1, 10032)]
    for nq, L, seed in cells:
        print(f"[lexact] instance N{nq} seed {seed}", flush=True)
        t_prep = time.perf_counter()
        J, meta, gs = instance(nq, seed)
        prep_s = time.perf_counter() - t_prep
        if not meta.get("accepted_lexact", meta.get("accepted_n3cond")):
            raise RuntimeError(f"instance N{nq} s{seed} failed B* checks {meta.get('hard_checks')}")
        b = float(meta["Bstar"])
        rng = np.random.default_rng([seed, nq, L])
        nrep = 8 if nq <= 12 else (4 if nq == 15 else 2)
        row = {"N": nq, "L": L, "seed": seed, "Bstar": b, "field_prep_s": prep_s,
               "source": meta.get("source", "built"), "accepted": True, "dim": 1 << nq,
               "rss_mb": rss_mb(), "nrep": nrep, "circuits": {}}
        for circ in ("Nov3", "QAOA"):
            ref = S.make_sim(circ, J, b, L, fast=False)
            fast = S.make_sim(circ, J, b, L, fast=True)
            x = rng.uniform(-0.4, 0.4, size=ref.nparam())
            t_ref = time_calls(lambda: ref.energy(x), nrep)
            t_fast = time_calls(lambda: fast.energy(x), nrep)
            eref, efast = ref.energy(x), fast.energy(x)
            pref, pfast = ref.state(x), fast.state(x)
            dref, dfast = ref.diagnostics(x, gs), fast.diagnostics(x, gs)
            row["circuits"][circ] = {
                "t_ref_s": t_ref, "t_fast_s": t_fast, "speedup": t_ref / t_fast,
                "dE": abs(eref - efast),
                "dstate": float(np.max(np.abs(pref - pfast))),
                "dF": abs(dref["F"] - dfast["F"]),
                "E_ref": eref, "F_ref": dref["F"], "F_fast": dfast["F"],
            }
            print(f"  {circ} N{nq} L{L} ref {t_ref*1e3:.2f}ms fast {t_fast*1e3:.2f}ms "
                  f"x{t_ref/t_fast:.2f} dE {abs(eref-efast):.2e}", flush=True)
        rec["timings"].append(row)

    spsa_jobs = PREREG["batch"]["spsa_short"]
    for job in spsa_jobs:
        nq, L, seed, Q = job["N"], job["L"], job["seed"], job["Q"]
        print(f"[lexact] SPSA N{nq} L{L} Q={Q}", flush=True)
        J, meta, gs = instance(nq, seed)
        b = float(meta["Bstar"])
        for circ in job["circuits"]:
            ref = S.make_sim(circ, J, b, L, fast=False)
            fast = S.make_sim(circ, J, b, L, fast=True)
            starts = N.starts_for(circ, seed, nq, L)
            cid = N.circuit_id(circ)
            rngs = lambda s, seed=seed, nq=nq, L=L, cid=cid: N.rng_for(seed, nq, L, cid, s, 0)
            t0 = time.perf_counter()
            rref = spsa_traced(ref.energy, starts, ref.sigma0(), rngs, Q)
            t_ref = time.perf_counter() - t0
            t0 = time.perf_counter()
            rfast = spsa_traced(fast.energy, starts, fast.sigma0(), rngs, Q)
            t_fast = time.perf_counter() - t0
            tr = np.asarray(rref["trace"])
            tf = np.asarray(rfast["trace"])
            d = {
                "N": nq, "L": L, "seed": seed, "circuit": circ, "Q": Q,
                "band": "read_dev" if seed < 10032 else "exclusive_dev_10032",
                "queries": Q, "outside_cap": 0, "t_ref_s": t_ref, "t_fast_s": t_fast,
                "speedup": t_ref / t_fast, "max_abs_dE_trace": float(np.max(np.abs(tr - tf))),
                "E_sel_ref": rref["E_sel"], "E_sel_fast": rfast["E_sel"],
                "same_start": rref["start"] == rfast["start"],
                "same_cand": rref["cand"] == rfast["cand"],
                "F_ref": ref.diagnostics(rref["x"], gs)["F"],
                "F_fast": fast.diagnostics(rfast["x"], gs)["F"],
                "selection": "energy_only",
            }
            rec["spsa"].append(d)
            print(f"  {circ} trace max|dE| {d['max_abs_dE_trace']:.2e} x{d['speedup']:.2f} "
                  f"E {rref['E_sel']:.6f}", flush=True)

    rec["rss_mb_end"] = rss_mb()
    rec["finished"] = time.time()
    rec["wall_s"] = rec["finished"] - status["started"]
    rec["new_queries"] = int(sum(s["Q"] for s in rec["spsa"]))
    rec["gr_2200h_note"] = ("GR full-grid ~2200h on 3 workers is a rough fixed-field probe, "
                            "not a B* completion estimate and not reproduced here.")
    (OUT / "summary_M1.json").write_text(json.dumps(rec, indent=1))
    status.update({"finished": rec["finished"], "wall_s": rec["wall_s"], "new_queries": rec["new_queries"]})
    (OUT / "STATUS.json").write_text(json.dumps(status))
    print("[lexact] done", rec["wall_s"], "queries", rec["new_queries"], flush=True)


if __name__ == "__main__":
    main()
