#!/usr/bin/env python3
"""Raw explore logger: E growth runs per (H, trial); stores the top-16 most probable logical code states of every
run (in the logical frame, after undoing XOR mask and bit permutation) plus GS rank / p(GS) / leakage, so that any
(K, fix-up radius, E, descent) policy can be evaluated offline with explore_raw_analyze.py.
Output: noiseless/results/relayout_runs/raw/<tag>.jsonl (gitignored dir), resumable per (inst, trial)."""
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
from noiseless.relayout_kbit import XorKbitSim, bit_perm_map, growth_lr_schedule
from noiseless.ecd_kbit import KbitLayout
from noiseless.spsa_gibbs import grow_trial

OUT = _REPO / "noiseless" / "results" / "relayout_runs" / "raw"
TOPN = 128


def job(a):
    cfg, inst, trial = a
    n, L, s = cfg["n"], cfg["L"], cfg["s"]
    k = (n - 2) // 2
    nf = (1 << k) + cfg["nf_extra"]
    lay = KbitLayout(k, nf)
    E, gs = energies_for(str(ham_paths(n, "scaling")[inst]), n, "scaling")
    g = int(gs, 2)
    rng = np.random.default_rng(trial_seed(20260917, inst, trial))
    f0 = lay.flat_of_logical()
    sched = [cfg["lr_scale"] * v for v in (lr_schedule_for(L) if L == 4 else growth_lr_schedule(L))]
    st = cfg["start"]
    sched = sched[st - 1:]
    runs = []
    for m in range(cfg["E"]):
        Pm = None
        if m == 0 or cfg["mask"] == "zero":
            xa = xb = 0
        else:
            xa, xb = int(rng.integers(1 << k)), int(rng.integers(1 << k))
            if cfg["mask"] == "perm":
                Pm = bit_perm_map(rng.permutation(n), n)
        if Pm is None:
            sim = XorKbitSim(lay, E, L, gs, xa=xa, xb=xb)
        else:
            gs_s = format(int(np.flatnonzero(Pm == g)[0]), f"0{n}b")
            sim = XorKbitSim(lay, E[Pm], L, gs_s, xa=xa, xb=xb)
        res = grow_trial(sim, final_layers=L, rng=rng, start_layers=st, c=cfg["c"], A=10.0, optimizer="spsa_adam",
                         steps_per_stage=s, lr_schedule=sched, kick_sigma=cfg["kick"],
                         eta_scale_schedule=[cfg["xeta"]] * len(sched), init_beta_max=cfg["xbeta"])
        probs = sim.probs_from_x(res.x)
        pl = probs[f0]
        msk = (xa << k) | xb
        order = np.argsort(-pl, kind="stable")[:TOPN]
        raw = []
        for j in order:
            v = int(j) ^ msk
            if Pm is not None:
                v = int(Pm[v])
            raw.append(v)
        # rank of GS in the probability order (logical frame)
        gl_phys = g
        if Pm is not None:
            gl_phys = int(np.flatnonzero(Pm == g)[0])
        pg = float(pl[gl_phys ^ msk])
        rank = int((pl > pg).sum())
        runs.append(dict(raw=raw, pgs=pg, rank=rank, leak=float(1 - pl.sum()), nfev=int(res.nfev)))
    return dict(inst=inst, trial=trial, gs=g, runs=runs)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=14)
    p.add_argument("--L", type=int, default=4)
    p.add_argument("--s", type=int, default=50)
    p.add_argument("--E", type=int, default=12)
    p.add_argument("--mask", default="perm")
    p.add_argument("--xeta", type=float, default=16.0)
    p.add_argument("--lr-scale", type=float, default=1.0)
    p.add_argument("--xbeta", type=float, default=None, help="max |beta| of the L=1 random init (default 3)")
    p.add_argument("--kick", type=float, default=0.05)
    p.add_argument("--c", type=float, default=0.15)
    p.add_argument("--start", type=int, default=1)
    p.add_argument("--nf-extra", type=int, default=16)
    p.add_argument("--trials", type=int, default=5)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--tag", default=None)
    a = p.parse_args()
    cfg = dict(n=a.n, L=a.L, s=a.s, E=a.E, mask=a.mask, xeta=a.xeta, lr_scale=a.lr_scale, xbeta=a.xbeta, kick=a.kick,
               c=a.c, start=a.start, nf_extra=a.nf_extra)
    tag = a.tag or (f"n{a.n:02d}_L{a.L}s{a.s}_E{a.E}_{a.mask}_x{a.xeta:g}" + (f"_lrx{a.lr_scale:g}" if a.lr_scale != 1 else "")
                    + (f"_b{a.xbeta:g}" if a.xbeta is not None else "") + (f"_kick{a.kick:g}" if a.kick != 0.05 else "")
                    + (f"_c{a.c:g}" if a.c != 0.15 else "") + (f"_st{a.start}" if a.start != 1 else "")
                    + (f"_nfx{a.nf_extra}" if a.nf_extra != 16 else ""))
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{tag}.jsonl"
    done = set()
    if path.exists():
        for line in path.read_text().splitlines():
            if line.strip():
                r = json.loads(line); done.add((r["inst"], r["trial"]))
    jobs = [(cfg, i, t) for i in range(len(ham_paths(a.n, "scaling"))) for t in range(a.trials) if (i, t) not in done]
    print(f"[{time.strftime('%F %T')}] {tag}: {len(jobs)} jobs ({len(done)} done)", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(a.workers) as ex, open(path, "a") as fh:
        for r in ex.map(job, jobs):
            fh.write(json.dumps(r) + "\n"); fh.flush()
    print(f"[{time.strftime('%T')}] {tag} done wall={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
