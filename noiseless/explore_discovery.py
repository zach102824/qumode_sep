#!/usr/bin/env python3
"""Discovery-only scan for the explore phase of explore_exploit_trial_kbit.

Runs E growth restarts (mask 0 then random masks) and records, after each restart, whether the min-energy
pool candidate is the GS. Cheap way to tune E, s, L_explore, K, fix-up per n before full runs.
Output: one JSON line per config to noiseless/results/relayout_runs/discovery/<tag>.json (gitignored dir).
"""
from __future__ import annotations
import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse, json, sys, time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))
from noiseless.run_relayout_kbit import ham_paths, energies_for, trial_seed, lr_schedule_for
from noiseless.relayout_kbit import default_nf, polish_logical, XorKbitSim
from noiseless.ecd_kbit import KbitLayout
from noiseless.spsa_gibbs import grow_trial

OUT = _REPO / "noiseless" / "results" / "relayout_runs" / "discovery"


def descend(v, E, n):
    while True:
        w = polish_logical(v, E, n, 1)
        if w == v:
            return v
        v = w


def bit_perm_map(perm, n):
    """Pm[v'] = original logical index whose bit perm[i] is bit i of v' (MSB-first positions)."""
    idx = np.arange(1 << n, dtype=np.int64)
    out = np.zeros_like(idx)
    for i in range(n):
        b = (idx >> (n - 1 - i)) & 1
        out |= b << (n - 1 - int(perm[i]))
    return out


def job(a):
    n, L, s, E_runs, Kmax, mask, inst, trial, nf_extra, enc, start, lrs, etas, cc = a
    k = (n - 2) // 2
    nf = (1 << k) + nf_extra if nf_extra is not None else default_nf(k)
    lay = KbitLayout(k, nf, enc)
    E, gs = energies_for(str(ham_paths(n, "scaling")[inst]), n, "scaling")
    g = int(gs, 2)
    rng = np.random.default_rng(trial_seed(20260917, inst, trial))
    f0 = lay.flat_of_logical()
    Ks = [K for K in (1, 2, 4, 8, 16) if K <= Kmax]
    pools = {K: set() for K in Ks}
    poolsD = {K: set() for K in Ks}
    hit = {K: [] for K in Ks}
    hitD = {K: [] for K in Ks}
    pgs, leak = [], []
    for m in range(E_runs):
        Pm = None
        if m == 0 or mask == "zero":
            xa = xb = 0
        else:
            xa, xb = int(rng.integers(1 << k)), int(rng.integers(1 << k))
            if mask == "perm":
                Pm = bit_perm_map(rng.permutation(n), n)
        if Pm is None:
            Ps, gs_s = None, gs
            sim = XorKbitSim(lay, E, L, gs, xa=xa, xb=xb)
        else:
            Es = E[Pm]
            gs_s = format(int(np.where(Pm == g)[0][0]), f"0{n}b")
            sim = XorKbitSim(lay, Es, L, gs_s, xa=xa, xb=xb)
        sched = lr_schedule_for(L)[L - (L - start + 1):] if start > 1 else lr_schedule_for(L)
        sched = [lrs * v for v in sched]
        res = grow_trial(sim, final_layers=L, rng=rng, start_layers=start, c=cc, eta_scale_schedule=[etas] * len(sched), A=10.0, optimizer="spsa_adam",
                         steps_per_stage=s, lr_schedule=sched)
        probs = sim.probs_from_x(res.x)
        pl = probs[f0]
        msk = (xa << k) | xb
        order = np.argsort(-pl, kind="stable")[:Kmax]
        pgs.append(float(res.p_gs))
        leak.append(float(1 - pl.sum()))
        for K in Ks:
            for j in order[:K]:
                v = int(j) ^ msk
                if Pm is not None:
                    v = int(Pm[v])
                pools[K].add(polish_logical(v, E, n))
                poolsD[K].add(descend(v, E, n))
            hit[K].append(min(pools[K], key=lambda v: (E[v], v)) == g)
            hitD[K].append(min(poolsD[K], key=lambda v: (E[v], v)) == g)
    return dict(inst=inst, hit={K: hit[K] for K in Ks}, hitD={K: hitD[K] for K in Ks}, pgs=pgs, leak=leak)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=12)
    p.add_argument("--L", type=int, default=4)
    p.add_argument("--s", type=int, default=50)
    p.add_argument("--E", type=int, default=12)
    p.add_argument("--K", type=int, default=8)
    p.add_argument("--mask", default="random")
    p.add_argument("--trials", type=int, default=5)
    p.add_argument("--nf-extra", type=int, default=None)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--encoding", default="binary")
    p.add_argument("--start-layers", type=int, default=1)
    p.add_argument("--lr-scale", type=float, default=1.0)
    p.add_argument("--eta-scale", type=float, default=1.0)
    p.add_argument("--c", type=float, default=0.15)
    a = p.parse_args()
    jobs = [(a.n, a.L, a.s, a.E, a.K, a.mask, i, t, a.nf_extra, a.encoding, a.start_layers, a.lr_scale, a.eta_scale, a.c) for i in range(len(ham_paths(a.n, "scaling"))) for t in range(a.trials)]
    t0 = time.time()
    with ProcessPoolExecutor(a.workers) as ex:
        res = list(ex.map(job, jobs))
    tag = f"n{a.n:02d}_L{a.L}s{a.s}_E{a.E}K{a.K}_{a.mask}" + (f"_nfx{a.nf_extra}" if a.nf_extra is not None else "") \
        + ("" if a.encoding == "binary" else f"_{a.encoding}") + ("" if a.start_layers == 1 else f"_st{a.start_layers}") \
        + ("" if a.lr_scale == 1.0 else f"_lrx{a.lr_scale:g}") \
        + ("" if a.eta_scale == 1.0 else f"_etax{a.eta_scale:g}") + ("" if a.c == 0.15 else f"_c{a.c:g}")
    ev = (a.L - a.start_layers + 1) * (2 * a.s + 1)
    out = {"tag": tag, "evals_per_run": ev, "trials": len(res), "wall_s": time.time() - t0,
           "hit": {K: np.mean([r["hit"][K] for r in res], axis=0).round(3).tolist() for K in res[0]["hit"]},
           "hit_descent": {K: np.mean([r["hitD"][K] for r in res], axis=0).round(3).tolist() for K in res[0]["hitD"]},
           "run0_pgs": float(np.mean([r["pgs"][0] for r in res])), "mean_leak": float(np.mean([r["leak"] for r in res])),
           "per_h_final_hitK1": {}}
    for r in res:
        out["per_h_final_hitK1"].setdefault(r["inst"], []).append(r["hit"][1][-1])
    out["per_h_final_hitK1"] = {h: float(np.mean(v)) for h, v in out["per_h_final_hitK1"].items()}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{tag}.json").write_text(json.dumps(out))
    print(tag, "evals/run", ev, "wall", round(out["wall_s"]), flush=True)
    for K in out["hit"]:
        print(f"  K={K:<2} hit by run: {out['hit'][K]}")
        print(f"  K={K:<2} descent:     {out['hit_descent'][K]}")
    print("  run0 p(GS)", round(out["run0_pgs"], 3), "mean leak", round(out["mean_leak"], 4), flush=True)


if __name__ == "__main__":
    main()
