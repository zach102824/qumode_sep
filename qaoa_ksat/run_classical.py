#!/usr/bin/env python3
"""Run WalkSAT / WalkSATlm / SATLike on saved 8-SAT Hamiltonians."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from qaoa_ksat.classical import run_solver
from qaoa_ksat.ksat import clauses_as_arrays, instance_from_dict

HAM_DIR = Path(__file__).resolve().parent / "hamiltonians"
OUT_DIR = Path(__file__).resolve().parent / "results"

# Table II: log2(median TTS) = a + b n, n=12..20.
PAPER_CLASSICAL = {
    "walksatlm": {"intercept": -0.309, "slope": 0.325, "error": 0.007},
    "walksat_qaoa": {"intercept": -3.232, "slope": 0.295, "error": 0.011},
    "schoning": {"intercept": -2.657, "slope": 0.649, "error": 0.006},
}


def paper_median_tts(name: str, n: int) -> float | None:
    fit = PAPER_CLASSICAL.get(name)
    if fit is None:
        return None
    return float(2.0 ** (fit["intercept"] + fit["slope"] * int(n)))


def _job(spec: dict) -> dict:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    t0 = time.perf_counter()
    d = json.loads(Path(spec["path"]).read_text())
    inst = instance_from_dict(d)
    cv, cs = clauses_as_arrays(inst)
    flips, lookups = run_solver(spec["solver"], cv, cs, inst.n, int(spec["run_seed"]), int(spec["cap"]))
    return {
        "ok": flips >= 0,
        "n": inst.n,
        "m": inst.m,
        "path": spec["path"],
        "eval_p": d.get("eval_p"),
        "eval_seed": d.get("eval_seed"),
        "solver": spec["solver"],
        "run_seed": spec["run_seed"],
        "flips": int(flips),
        "lookups": int(lookups),
        "wall_s": time.perf_counter() - t0,
    }


def list_instances(n: int, p: int | None, limit: int | None) -> list[Path]:
    paths = sorted((HAM_DIR / f"n{n:02d}").glob("seed_*.json"))
    if p is not None:
        kept = []
        for path in paths:
            d = json.loads(path.read_text())
            if int(d.get("eval_p", -1)) == int(p):
                kept.append(path)
        paths = kept
    if limit is not None:
        paths = paths[: int(limit)]
    return paths


def summarize(rows: list[dict], cap: int) -> dict:
    ok = [r for r in rows if r.get("ok")]
    flips = np.array([r["flips"] for r in ok], dtype=float) if ok else np.array([])
    lookups = np.array([r["lookups"] for r in ok], dtype=float) if ok else np.array([])
    return {
        "n_ok": len(ok),
        "n_fail": len(rows) - len(ok),
        "cap": cap,
        "median_flips": None if flips.size == 0 else float(np.median(flips)),
        "mean_flips": None if flips.size == 0 else float(flips.mean()),
        "p90_flips": None if flips.size == 0 else float(np.percentile(flips, 90)),
        "median_lookups": None if lookups.size == 0 else float(np.median(lookups)),
        "mean_lookups": None if lookups.size == 0 else float(lookups.mean()),
    }


def main() -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n", type=str, default="8,12,16,20")
    ap.add_argument("--p-filter", type=int, default=14, help="Only instances from this QAOA p eval (None=all)")
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--solvers", type=str, default="walksat_simple,walksatlm,satlike")
    ap.add_argument("--runs", type=int, default=5, help="Independent random starts per instance")
    ap.add_argument("--cap", type=int, default=200_000)
    ap.add_argument("--workers", type=int, default=max(1, os.cpu_count() or 1))
    ap.add_argument("--all-p", action="store_true", help="Ignore --p-filter")
    args = ap.parse_args()
    ns = [int(x) for x in args.n.split(",") if x.strip()]
    solvers = [s.strip() for s in args.solvers.split(",") if s.strip()]
    p_filter = None if args.all_p else args.p_filter

    all_rows = []
    summaries = []
    for n in ns:
        paths = list_instances(n, p_filter, args.limit)
        if not paths:
            print(f"n={n}: no saved instances", flush=True)
            continue
        for solver in solvers:
            jobs = []
            for i, path in enumerate(paths):
                for r in range(args.runs):
                    jobs.append(
                        {
                            "path": str(path),
                            "solver": solver,
                            "run_seed": 20261016 + 100003 * n + 1009 * i + r,
                            "cap": args.cap,
                        }
                    )
            t0 = time.perf_counter()
            print(
                f"start n={n} {solver} instances={len(paths)} jobs={len(jobs)}",
                flush=True,
            )
            rows = []
            if args.workers <= 1:
                for job in jobs:
                    rows.append(_job(job))
            else:
                with ProcessPoolExecutor(max_workers=int(args.workers)) as ex:
                    futs = [ex.submit(_job, job) for job in jobs]
                    for fut in as_completed(futs):
                        rows.append(fut.result())
            summ = summarize(rows, args.cap)
            paper = paper_median_tts("walksatlm" if solver == "walksatlm" else "", n)
            if solver == "walksatlm":
                paper = paper_median_tts("walksatlm", n)
            else:
                paper = None
            summ.update(
                {
                    "n": n,
                    "solver": solver,
                    "n_instances": len(paths),
                    "runs_per_instance": args.runs,
                    "p_filter": p_filter,
                    "elapsed_s": time.perf_counter() - t0,
                    "paper_walksatlm_median_tts": paper_median_tts("walksatlm", n),
                }
            )
            summaries.append(summ)
            all_rows.extend(rows)
            print(
                f"n={n} {solver}: ok={summ['n_ok']}/{len(jobs)}  "
                f"median flips={summ['median_flips']}  "
                f"median lookups={summ['median_lookups']}  "
                f"paper WalkSATlm Table II={summ['paper_walksatlm_median_tts']:.3f}  "
                f"wall={summ['elapsed_s']:.1f}s",
                flush=True,
            )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / "classical_on_qaoa_sat.json"
    out.write_text(json.dumps({"summaries": summaries, "rows": all_rows}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
