#!/usr/bin/env python3
"""Evaluate fixed-angle random k-SAT QAOA (paper protocol).

Only satisfiable instances are kept, matching Sec. III B. Running time on an
instance is 1 / p_succ (expected shots to a satisfying assignment).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from qaoa_ksat.angles import available_p, paper_angles
from qaoa_ksat.circuit import qaoa_state, success_probability
from qaoa_ksat.ksat import THRESHOLD_R, energy_table, is_satisfiable, n_satisfying, random_ksat

# Table II of the paper: log2(median TTS) ≈ intercept + slope * n, fit on n=12..20.
PAPER_LOG2_MEDIAN_TTS = {
    14: {"intercept": -1.064, "slope": 0.326, "error": 0.005},
    60: {"intercept": -2.842, "slope": 0.302, "error": 0.007},
}


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def random_assignment_psucc(energy: np.ndarray) -> float:
    dim = max(int(energy.size), 1)
    return float(n_satisfying(energy) / dim)


def eval_one(job: dict) -> dict:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    t0 = time.perf_counter()
    n = int(job["n"])
    k = int(job["k"])
    r = float(job["r"])
    p = int(job["p"])
    seed = int(job["seed"])
    rng = np.random.default_rng(seed)
    tries = 0
    while True:
        tries += 1
        inst = random_ksat(n, k, r, rng=rng, seed=None)
        energy = energy_table(inst)
        if is_satisfiable(energy):
            break
        if tries > int(job.get("max_unsat_tries", 2000)):
            return {
                "ok": False,
                "error": f"no SAT instance after {tries} draws",
                "n": n,
                "p": p,
                "seed": seed,
                "wall_s": time.perf_counter() - t0,
            }
    betas = np.asarray(job["betas"], dtype=float)
    gammas = np.asarray(job["gammas"], dtype=float)
    psi = qaoa_state(betas, gammas, energy, n)
    psucc = success_probability(psi, energy)
    rand_p = random_assignment_psucc(energy)
    n_sat = n_satisfying(energy)
    tts = float("inf") if psucc <= 0.0 else 1.0 / psucc
    return {
        "ok": True,
        "error": None,
        "n": n,
        "k": k,
        "r": r,
        "p": p,
        "seed": seed,
        "trial": int(job["trial"]),
        "m": int(inst.m),
        "n_sat": int(n_sat),
        "e_min": int(energy.min()),
        "e_mean": float(energy.mean()),
        "psucc": psucc,
        "tts": tts,
        "rand_psucc": rand_p,
        "sat_tries": tries,
        "wall_s": time.perf_counter() - t0,
    }


def _summarize(rows: list[dict]) -> dict:
    ok = [x for x in rows if x.get("ok")]
    if not ok:
        return {"n_ok": 0}
    ps = np.array([x["psucc"] for x in ok], dtype=float)
    tts = np.array([x["tts"] for x in ok], dtype=float)
    rand = np.array([x["rand_psucc"] for x in ok], dtype=float)
    finite_tts = tts[np.isfinite(tts)]
    return {
        "n_ok": len(ok),
        "n_fail": len(rows) - len(ok),
        "mean_psucc": float(ps.mean()),
        "median_psucc": float(np.median(ps)),
        "std_psucc": float(ps.std(ddof=1)) if len(ps) > 1 else 0.0,
        "mean_tts": float(finite_tts.mean()) if finite_tts.size else None,
        "median_tts": float(np.median(finite_tts)) if finite_tts.size else None,
        "mean_rand_psucc": float(rand.mean()),
        "median_rand_psucc": float(np.median(rand)),
        "mean_n_sat": float(np.mean([x["n_sat"] for x in ok])),
        "mean_m": float(np.mean([x["m"] for x in ok])),
        "mean_wall_s": float(np.mean([x["wall_s"] for x in ok])),
        "qaoa_over_random": float(ps.mean() / rand.mean()) if rand.mean() > 0 else None,
    }


def paper_prediction(p: int, n: int) -> dict | None:
    fit = PAPER_LOG2_MEDIAN_TTS.get(int(p))
    if fit is None:
        return None
    log2_med = fit["intercept"] + fit["slope"] * int(n)
    med_tts = 2.0 ** log2_med
    return {
        "log2_median_tts": log2_med,
        "median_tts": med_tts,
        "implied_median_psucc": 1.0 / med_tts,
        "slope_error": fit["error"],
        "fit_range": "n=12..20",
        "extrapolated": not (12 <= int(n) <= 20),
    }


def run_grid(
    *,
    n: int,
    p_list: list[int],
    n_instances: int,
    k: int = 8,
    r: float | None = None,
    seed: int = 0,
    workers: int = 1,
    out_dir: Path | None = None,
) -> dict:
    r = float(THRESHOLD_R[k] if r is None else r)
    out_dir = Path(out_dir) if out_dir is not None else Path(__file__).resolve().parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    all_rows: list[dict] = []
    summaries = []
    for p in p_list:
        betas, gammas = paper_angles(k, p, r)
        jobs = []
        for trial in range(n_instances):
            jobs.append(
                {
                    "n": n,
                    "k": k,
                    "r": r,
                    "p": p,
                    "trial": trial,
                    "seed": int(seed) + 1_000_003 * int(p) + trial,
                    "betas": betas.tolist(),
                    "gammas": gammas.tolist(),
                }
            )
        rows: list[dict] = []
        t0 = time.perf_counter()
        if workers <= 1:
            for job in jobs:
                rows.append(eval_one(job))
        else:
            with ProcessPoolExecutor(max_workers=int(workers)) as ex:
                futs = [ex.submit(eval_one, job) for job in jobs]
                for fut in as_completed(futs):
                    rows.append(fut.result())
        rows.sort(key=lambda x: int(x.get("trial", 0)))
        summ = _summarize(rows)
        summ.update(
            {
                "n": n,
                "k": k,
                "r": r,
                "p": p,
                "n_instances": n_instances,
                "seed": seed,
                "elapsed_s": time.perf_counter() - t0,
                "paper": paper_prediction(p, n),
            }
        )
        summaries.append(summ)
        all_rows.extend(rows)
        pred = summ["paper"]
        pred_s = ""
        if pred is not None:
            extra = " [extrapolated]" if pred["extrapolated"] else ""
            pred_s = (
                f"  paper median TTS={pred['median_tts']:.3f} "
                f"(implied p_succ={pred['implied_median_psucc']:.4g}){extra}"
            )
        print(
            f"n={n} p={p}: mean p_succ={summ.get('mean_psucc')}  "
            f"median p_succ={summ.get('median_psucc')}  "
            f"median TTS={summ.get('median_tts')}  "
            f"random mean p_succ={summ.get('mean_rand_psucc')}  "
            f"ok={summ.get('n_ok')}/{n_instances}  "
            f"wall={summ.get('elapsed_s'):.1f}s"
            f"{pred_s}",
            flush=True,
        )
    payload = {
        "created_utc": _now(),
        "n": n,
        "k": k,
        "r": r,
        "p_list": p_list,
        "n_instances": n_instances,
        "seed": seed,
        "summaries": summaries,
        "rows": all_rows,
    }
    out_path = out_dir / f"eval_n{n}_k{k}_{stamp}.json"
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"wrote {out_path}", flush=True)
    return payload


def main() -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--p", type=str, default="1,2,5,10,14")
    ap.add_argument("--n-instances", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=max(1, os.cpu_count() or 1))
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()
    p_list = []
    for tok in args.p.split(","):
        tok = tok.strip()
        if not tok:
            continue
        p_list.append(int(tok))
    if not p_list:
        p_list = available_p(args.k)
    run_grid(
        n=args.n,
        p_list=p_list,
        n_instances=args.n_instances,
        k=args.k,
        seed=args.seed,
        workers=args.workers,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()
