"""Scaling study on planted 4-SAT family F1 (Hamiltonians/four_sat_scaling/).

Arms
  ry0          RY-only product ansatz (n params); Gibbs cost exact for n <= 20,
               sampled (N_s samples/eval, CRN) for n >= 24
  ry0_exact    RY-only with exact enumerated Gibbs cost (forced)
  ry0_sampled  RY-only with sampled Gibbs cost (forced)
  ry0_split    RY-only with EXACT Gibbs cost via the hi/lo product factorization
               (numba, uint8 spectrum; used for n = 24, 28; validated vs ry0 at n = 16, 20)
  hea1, hea2   HEA L=1,2 (2 n / 3 n params) on the 2 x n/2 snake lattice, statevector
  classical    WalkSAT + SA, 25 runs/instance at budgets B0 x {1,10,100,1000}, B0=401

Same SPSA/Gibbs conventions as noiseless/run_ecd_vs_qaoa.py (200 steps, 25 trials,
seed 20260917). Seed per trial:
  seed0 + 1_000_000*inst + 100_000*arm_code + 1_000*n + trial,
arm_code: ry0* -> 0 (all RY-only variants share seeds so x0 / SPSA Δ are identical),
hea1 -> 1, hea2 -> 2, classical -> 3 (one job per instance, trial index 0).

Checkpointing: every finished job is appended as one JSON line to
  noiseless/results/scaling_runs/<tag>/<arm>_n<nn>.jsonl
and skipped on restart. ``--summarize`` aggregates all jsonl files of a tag into
noiseless/results/scaling_<tag>_summary.json.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "Hamiltonians"))

from four_sat_scaling import instance_paths, load_instance  # noqa: E402
from noiseless.scaling_ansatz import (  # noqa: E402
    DEFAULT_N_SAMPLES,
    HEAnSimulator,
    ProductRYSimulator,
    n_params,
    optimize_trial,
    spectrum,
)

ARM_CODE = {"ry0": 0, "ry0_exact": 0, "ry0_sampled": 0, "ry0_split": 0, "hea1": 1, "hea2": 2, "classical": 3}
EXACT_MAX_N = 20
RUN_ROOT = _REPO / "noiseless" / "results" / "scaling_runs"


@lru_cache(maxsize=4)
def _inst(path: str):
    return load_instance(path)


@lru_cache(maxsize=1)
def _split_spec(path: str) -> np.ndarray:
    from noiseless.scaling_ansatz import split_spectrum

    d = _inst(path)
    return split_spectrum(d["clauses"], d["polarities"], d["n"])


@lru_cache(maxsize=2)
def _spec(path: str) -> np.ndarray:
    d = _inst(path)
    return spectrum(d["clauses"], d["polarities"], d["n"])


def trial_seed(seed0: int, inst: int, arm: str, n: int, trial: int) -> int:
    return int(seed0) + 1_000_000 * int(inst) + 100_000 * ARM_CODE[arm] + 1_000 * int(n) + int(trial)


def worker(job: dict) -> dict:
    t0 = time.perf_counter()
    arm, path, n = job["arm"], job["path"], int(job["n"])
    d = _inst(path)
    gs = d["ground_bitstring"]
    rec = {k: job[k] for k in ("arm", "n", "inst", "trial", "seed")}
    rec["file"] = d["file"]
    rec["m"] = int(len(d["clauses"]))
    if arm == "classical":
        from noiseless.classical_baselines import run_baselines

        rec.update(run_baselines(d["clauses"], d["polarities"], n, job["seed"], trials=job["trials"]))
        rec["wall_s"] = time.perf_counter() - t0
        return rec
    rng = np.random.default_rng(job["seed"])
    if arm == "ry0_split":
        from noiseless.scaling_ansatz import ProductRYSplitSimulator

        sim = ProductRYSplitSimulator(d["clauses"], d["polarities"], gs, n, E_split=_split_spec(path))
        npar = n
        srng = None
        rec["cost_mode"] = "exact_split"
        rec["n_samples"] = None
    elif arm.startswith("ry0"):
        mode = {"ry0_exact": "exact", "ry0_sampled": "sampled"}.get(arm, "exact" if n <= EXACT_MAX_N else "sampled")
        sim = ProductRYSimulator(
            d["clauses"], d["polarities"], gs, n, cost_mode=mode, n_samples=job["n_samples"],
            energies=_spec(path) if mode == "exact" else None,
        )
        npar = n
        srng = np.random.default_rng([job["seed"], 7])
        rec["cost_mode"] = mode
        rec["n_samples"] = job["n_samples"] if mode == "sampled" else None
    else:
        L = int(arm[-1])
        sim = HEAnSimulator(_spec(path), gs, n, L)
        npar = n_params(n, L)
        srng = None
    out = optimize_trial(sim, npar, maxiter=job["steps"], rng=rng, sample_rng=srng)
    rec.update(
        n_params=npar, success=out.success, p_gs=out.p_gs, energy_mean=out.energy_mean,
        most_likely_bitstring=out.most_likely_bitstring, ground_bitstring=gs, fun=out.fun,
        eta=out.eta, nfev=out.nfev, x=out.x.tolist(), wall_s=time.perf_counter() - t0,
    )
    return rec


def _ckpt_path(tag: str, arm: str, n: int) -> Path:
    return RUN_ROOT / tag / f"{arm}_n{n:02d}.jsonl"


def _done(tag: str, arm: str, n: int) -> set[tuple[int, int]]:
    p = _ckpt_path(tag, arm, n)
    if not p.exists():
        return set()
    out = set()
    for line in p.read_text().splitlines():
        if line.strip():
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            out.add((int(r["inst"]), int(r["trial"])))
    return out


def build_jobs(a) -> list[dict]:
    jobs = []
    for n in [int(x) for x in a.n.split(",") if x.strip()]:
        paths = instance_paths(n)
        insts = list(range(len(paths)))[: a.max_h] if a.instances is None else [int(i) for i in a.instances.split(",")]
        for arm in [s.strip() for s in a.arms.split(",") if s.strip()]:
            done = _done(a.tag, arm, n)
            for i in insts:
                trials = [0] if arm == "classical" else list(range(a.trial_offset, a.trial_offset + a.trials))
                for t in trials:
                    if (i, t) in done:
                        continue
                    jobs.append(dict(arm=arm, n=n, inst=i, trial=t, path=str(paths[i]), steps=a.steps,
                                     trials=a.trials, n_samples=a.n_samples,
                                     seed=trial_seed(a.seed, i, arm, n, t)))
    # heavy jobs first for load balance
    cost = {"hea2": 3, "hea1": 2}
    jobs.sort(key=lambda j: -(cost.get(j["arm"], 1) * (1 << min(j["n"], 24))))
    return jobs


def summarize(tag: str) -> dict:
    per = {}
    for p in sorted((RUN_ROOT / tag).glob("*.jsonl")):
        recs = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
        if not recs:
            continue
        arm, n = recs[0]["arm"], int(recs[0]["n"])
        key = f"{arm}_n{n:02d}"
        if arm == "classical":
            s = {"arm": arm, "n": n, "instances": len(recs), "budgets": recs[0]["budgets"]}
            for alg in ("walksat", "sa"):
                s[alg] = {b: float(np.mean([r[f"{alg}_success"][str(b)] for r in recs])) for b in recs[0]["budgets"]}
            s["wall_s_mean_per_instance"] = float(np.mean([r["wall_s"] for r in recs]))
        else:
            pg = np.array([r["p_gs"] for r in recs])
            by_inst = {}
            for r in recs:
                by_inst.setdefault(r["inst"], []).append(r)
            s = {
                "arm": arm, "n": n, "n_params": recs[0]["n_params"], "cost_mode": recs[0].get("cost_mode"),
                "instances": len(by_inst), "trials_total": len(recs),
                "success_rate": float(np.mean([r["success"] for r in recs])),
                "mean_p_gs": float(pg.mean()), "median_p_gs": float(np.median(pg)),
                "mean_log10_p_gs": float(np.mean(np.log10(np.maximum(pg, 1e-300)))),
                "mean_energy": float(np.mean([r["energy_mean"] for r in recs])),
                "per_instance_mean_p_gs": {int(i): float(np.mean([r["p_gs"] for r in v])) for i, v in sorted(by_inst.items())},
                "per_instance_success": {int(i): float(np.mean([r["success"] for r in v])) for i, v in sorted(by_inst.items())},
                "wall_s_mean_per_trial": float(np.mean([r["wall_s"] for r in recs])),
            }
        per[key] = s
    out = {"tag": tag, "cells": per}
    path = _REPO / "noiseless" / "results" / f"scaling_{tag}_summary.json"
    path.write_text(json.dumps(out, indent=1) + "\n")
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arms", default="ry0,classical")
    p.add_argument("--n", default="8,12,16,20,24,28")
    p.add_argument("--max-h", type=int, default=20)
    p.add_argument("--instances", default=None, help="comma list of instance indices (overrides --max-h)")
    p.add_argument("--trials", type=int, default=25)
    p.add_argument("--trial-offset", type=int, default=0)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--n-samples", type=int, default=DEFAULT_N_SAMPLES)
    p.add_argument("--seed", type=int, default=20260917)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--tag", default="main")
    p.add_argument("--summarize", action="store_true")
    a = p.parse_args(argv)
    if a.summarize:
        s = summarize(a.tag)
        print(json.dumps({k: {kk: vv for kk, vv in v.items() if not kk.startswith("per_instance")} for k, v in s["cells"].items()}, indent=1))
        return 0
    jobs = build_jobs(a)
    print(f"[{time.strftime('%H:%M:%S')}] {len(jobs)} jobs, tag={a.tag}", flush=True)
    (RUN_ROOT / a.tag).mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(worker, j): j for j in jobs}
        for k, f in enumerate(as_completed(futs), 1):
            j = futs[f]
            try:
                r = f.result()
            except Exception as e:  # keep going; failed jobs are re-run on restart
                print(f"FAILED {j['arm']} n={j['n']} inst={j['inst']} t={j['trial']}: {e!r}", flush=True)
                continue
            with open(_ckpt_path(a.tag, r["arm"], r["n"]), "a") as fh:
                fh.write(json.dumps(r) + "\n")
            if k % 25 == 0 or k == len(jobs):
                print(f"[{time.strftime('%H:%M:%S')}] {k}/{len(jobs)} done, elapsed {time.time() - t0:.0f}s "
                      f"(last {r['arm']} n={r['n']} {r['wall_s']:.1f}s)", flush=True)
    summarize(a.tag)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
