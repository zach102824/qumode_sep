#!/usr/bin/env python3
"""CLI: sweep fixed U × L* × Hamiltonians with SPSA Gibbs (noiseless).

Examples:
  python -m noiseless.run_u_sweep --smoke
  python -m noiseless.run_u_sweep --ham-dir Hamiltonians/four_sat --u-names all \\
      --layers 2,3,4 --trials 5 --steps 200 --workers 2 --tag fleet1
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# Allow `python noiseless/run_u_sweep.py` from repo root
_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from noiseless.circuit_local_ecd import n_parameters
from noiseless.encoding import list_four_sat_npz, load_four_sat_npz
from noiseless.spsa_gibbs import (
    NoiselessSimulator,
    ground_flat_from_bitstring,
    optimize_trial,
    scale_spsa_a,
)
from noiseless.unitaries import U_NAMES, build_fixed_u


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _worker(job: dict) -> dict:
    """One (H, U, L*, trial) SPSA run — process-safe."""
    t0 = time.perf_counter()
    try:
        inst = load_four_sat_npz(job["ham_path"])
        u = build_fixed_u(job["u_name"])
        sim = NoiselessSimulator(
            u_fixed=u,
            energy_tensor=inst["energy_tensor"],
            n_layers=int(job["n_layers"]),
            ground_bitstring=inst["ground_bitstring"],
            ground_flat_index=ground_flat_from_bitstring(inst["ground_bitstring"]),
        )
        rng = np.random.default_rng(int(job["seed"]))
        a = job.get("spsa_a")
        if a is None:
            a = scale_spsa_a(n_parameters(int(job["n_layers"])))
        result = optimize_trial(
            sim,
            maxiter=int(job["steps"]),
            rng=rng,
            a=float(a),
            c=float(job.get("spsa_c", 0.15)),
            A=float(job.get("spsa_A", 10.0)),
        )
        return {
            "ok": True,
            "ham_file": job["ham_file"],
            "u_name": job["u_name"],
            "n_layers": int(job["n_layers"]),
            "trial": int(job["trial"]),
            "seed": int(job["seed"]),
            "success": bool(result.success),
            "p_gs": float(result.p_gs),
            "most_likely_bitstring": result.most_likely_bitstring,
            "ground_bitstring": result.ground_bitstring,
            "fun": float(result.fun),
            "eta": float(result.eta),
            "energy_mean": float(result.energy_mean),
            "nfev": int(result.nfev),
            "nit": int(result.nit),
            "spsa_a": float(a),
            "wall_s": float(time.perf_counter() - t0),
            "x": result.x.tolist(),
            "error": None,
        }
    except Exception as exc:  # noqa: BLE001 — fleet must continue
        return {
            "ok": False,
            "ham_file": job.get("ham_file"),
            "u_name": job.get("u_name"),
            "n_layers": job.get("n_layers"),
            "trial": job.get("trial"),
            "seed": job.get("seed"),
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "wall_s": float(time.perf_counter() - t0),
        }


def _aggregate(records: list[dict]) -> dict:
    by_key: dict[tuple, list[dict]] = {}
    for r in records:
        if not r.get("ok"):
            continue
        key = (r["u_name"], int(r["n_layers"]), r["ham_file"])
        by_key.setdefault(key, []).append(r)

    cells = []
    for (u_name, n_layers, ham_file), trials in sorted(by_key.items()):
        n = len(trials)
        n_succ = sum(1 for t in trials if t["success"])
        pgs = [float(t["p_gs"]) for t in trials]
        cells.append(
            {
                "u_name": u_name,
                "n_layers": n_layers,
                "ham_file": ham_file,
                "n_trials": n,
                "n_success": n_succ,
                "success_rate": n_succ / n if n else 0.0,
                "mean_p_gs": float(np.mean(pgs)) if pgs else 0.0,
                "max_p_gs": float(np.max(pgs)) if pgs else 0.0,
                "median_p_gs": float(np.median(pgs)) if pgs else 0.0,
            }
        )

    # Aggregate across Hamiltonians per (U, L*)
    by_ul: dict[tuple, list[dict]] = {}
    for c in cells:
        by_ul.setdefault((c["u_name"], c["n_layers"]), []).append(c)
    ranking = []
    for (u_name, n_layers), group in by_ul.items():
        weights = [g["n_trials"] for g in group]
        tot = sum(weights)
        succ = sum(g["n_success"] for g in group)
        mean_p = float(
            np.average([g["mean_p_gs"] for g in group], weights=weights)
        ) if tot else 0.0
        ranking.append(
            {
                "u_name": u_name,
                "n_layers": n_layers,
                "n_hamiltonians": len(group),
                "n_trials": tot,
                "n_success": succ,
                "success_rate": succ / tot if tot else 0.0,
                "mean_p_gs": mean_p,
            }
        )
    ranking.sort(key=lambda r: (r["success_rate"], r["mean_p_gs"]), reverse=True)
    return {"cells": cells, "ranking": ranking}


def build_jobs(args: argparse.Namespace) -> list[dict]:
    ham_dir = Path(args.ham_dir)
    paths = list_four_sat_npz(ham_dir)
    if args.max_h is not None:
        paths = paths[: int(args.max_h)]
    if not paths:
        raise SystemExit(f"No four_sat_*.npz under {ham_dir}")

    # Verify 8-qubit
    for p in paths:
        inst = load_four_sat_npz(p)
        if inst["num_spins"] != 8:
            raise SystemExit(f"{p.name} has num_spins={inst['num_spins']}, need 8")
        if inst["n_ground"] != 1:
            raise SystemExit(f"{p.name} has n_ground={inst['n_ground']}, need unique GS")

    if args.u_names.strip().lower() == "all":
        u_names = list(U_NAMES)
    else:
        u_names = [s.strip() for s in args.u_names.split(",") if s.strip()]
    layers = [int(x) for x in args.layers.split(",") if x.strip()]

    jobs = []
    seed0 = int(args.seed)
    for hi, path in enumerate(paths):
        for u_name in u_names:
            for L in layers:
                for t in range(int(args.trials)):
                    seed = seed0 + 100_000 * hi + 1_000 * U_NAMES.index(u_name) + 10 * L + t
                    jobs.append(
                        {
                            "ham_path": str(path.resolve()),
                            "ham_file": path.name,
                            "u_name": u_name,
                            "n_layers": L,
                            "trial": t,
                            "seed": seed,
                            "steps": int(args.steps),
                            "spsa_a": args.spsa_a,
                            "spsa_c": float(args.spsa_c),
                            "spsa_A": float(args.spsa_A),
                        }
                    )
    return jobs


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ham-dir", type=str, default=str(_REPO / "Hamiltonians" / "four_sat"))
    p.add_argument("--u-names", type=str, default="all")
    p.add_argument("--layers", type=str, default="2,3,4")
    p.add_argument("--trials", type=int, default=5)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--workers", type=int, default=max(1, min(4, os.cpu_count() or 1)))
    p.add_argument("--seed", type=int, default=20260917)
    p.add_argument("--max-h", type=int, default=None)
    p.add_argument("--spsa-a", type=float, default=None)
    p.add_argument("--spsa-c", type=float, default=0.15)
    p.add_argument("--spsa-A", type=float, default=10.0)
    p.add_argument("--outdir", type=str, default=str(_REPO / "noiseless" / "results"))
    p.add_argument("--tag", type=str, default="run")
    p.add_argument("--smoke", action="store_true", help="1 H, 1 U, L=2, 8 steps, 1 trial")
    p.add_argument("--synthetic", action="store_true", help="Use synthetic diagonal H (no NPZ)")
    args = p.parse_args(argv)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.smoke:
        args.u_names = "identity"
        args.layers = "2"
        args.trials = 1
        args.steps = 8
        args.max_h = 1
        args.tag = args.tag if args.tag != "run" else "smoke"

    if args.synthetic or (args.smoke and not list_four_sat_npz(Path(args.ham_dir))):
        # Synthetic unique GS at bitstring 01011010 → denm encoding
        from noiseless.encoding import DIMS, bits_from_bitstring, denm_from_bits, flat_index

        gs = "01011010"
        d, e, n_a, n_b = denm_from_bits(bits_from_bitstring(gs))
        gidx = flat_index(d, e, n_a, n_b)
        tensor = np.ones(DIMS, dtype=float)
        tensor[d, e, n_a, n_b] = 0.0
        syn_path = outdir / "synthetic_h.npz"
        # Fake NPZ-compatible via inline job path override
        jobs = [
            {
                "ham_path": "__synthetic__",
                "ham_file": "synthetic.npz",
                "u_name": "identity",
                "n_layers": 2,
                "trial": 0,
                "seed": int(args.seed),
                "steps": int(args.steps),
                "spsa_a": args.spsa_a,
                "spsa_c": float(args.spsa_c),
                "spsa_A": float(args.spsa_A),
                "_synthetic": {
                    "energy_tensor": tensor,
                    "ground_bitstring": gs,
                    "ground_flat_index": gidx,
                },
            }
        ]

        def _worker_syn(job: dict) -> dict:
            if job.get("ham_path") == "__synthetic__":
                t0 = time.perf_counter()
                syn = job["_synthetic"]
                u = build_fixed_u(job["u_name"])
                sim = NoiselessSimulator(
                    u_fixed=u,
                    energy_tensor=syn["energy_tensor"],
                    n_layers=int(job["n_layers"]),
                    ground_bitstring=syn["ground_bitstring"],
                    ground_flat_index=int(syn["ground_flat_index"]),
                )
                rng = np.random.default_rng(int(job["seed"]))
                a = job.get("spsa_a") or scale_spsa_a(n_parameters(int(job["n_layers"])))
                result = optimize_trial(sim, maxiter=int(job["steps"]), rng=rng, a=float(a))
                return {
                    "ok": True,
                    "ham_file": "synthetic.npz",
                    "u_name": job["u_name"],
                    "n_layers": int(job["n_layers"]),
                    "trial": 0,
                    "seed": int(job["seed"]),
                    "success": bool(result.success),
                    "p_gs": float(result.p_gs),
                    "most_likely_bitstring": result.most_likely_bitstring,
                    "ground_bitstring": result.ground_bitstring,
                    "fun": float(result.fun),
                    "eta": float(result.eta),
                    "energy_mean": float(result.energy_mean),
                    "nfev": int(result.nfev),
                    "nit": int(result.nit),
                    "spsa_a": float(a),
                    "wall_s": float(time.perf_counter() - t0),
                    "x": result.x.tolist(),
                    "error": None,
                }
            return _worker(job)

        print(f"[{_now_iso()}] synthetic smoke: {len(jobs)} job(s)", flush=True)
        records = [_worker_syn(j) for j in jobs]
    else:
        jobs = build_jobs(args)
        print(
            f"[{_now_iso()}] starting {len(jobs)} jobs "
            f"(workers={args.workers}, steps={args.steps}, tag={args.tag})",
            flush=True,
        )
        records: list[dict] = []
        if args.workers <= 1 or len(jobs) == 1:
            for i, job in enumerate(jobs):
                rec = _worker(job)
                records.append(rec)
                status = "OK" if rec.get("ok") else "FAIL"
                print(
                    f"  [{i+1}/{len(jobs)}] {status} "
                    f"{rec.get('u_name')} L={rec.get('n_layers')} "
                    f"{rec.get('ham_file')} t={rec.get('trial')} "
                    f"succ={rec.get('success')} p_gs={rec.get('p_gs')} "
                    f"wall={rec.get('wall_s'):.1f}s",
                    flush=True,
                )
        else:
            with ProcessPoolExecutor(max_workers=int(args.workers)) as ex:
                futs = {ex.submit(_worker, job): job for job in jobs}
                done = 0
                for fut in as_completed(futs):
                    rec = fut.result()
                    records.append(rec)
                    done += 1
                    status = "OK" if rec.get("ok") else "FAIL"
                    print(
                        f"  [{done}/{len(jobs)}] {status} "
                        f"{rec.get('u_name')} L={rec.get('n_layers')} "
                        f"{rec.get('ham_file')} t={rec.get('trial')} "
                        f"succ={rec.get('success')} p_gs={rec.get('p_gs')} "
                        f"wall={rec.get('wall_s'):.1f}s",
                        flush=True,
                    )

    agg = _aggregate(records)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = outdir / f"{args.tag}_{stamp}.json"
    payload = {
        "created_utc": _now_iso(),
        "tag": args.tag,
        "args": {
            "ham_dir": str(args.ham_dir),
            "u_names": args.u_names,
            "layers": args.layers,
            "trials": args.trials,
            "steps": args.steps,
            "workers": args.workers,
            "seed": args.seed,
            "max_h": args.max_h,
        },
        "n_jobs": len(records),
        "n_ok": sum(1 for r in records if r.get("ok")),
        "n_fail": sum(1 for r in records if not r.get("ok")),
        "aggregate": agg,
        "records": records,
    }
    # Drop large x vectors from ranking-friendly copy? Keep them for now.
    out_path.write_text(json.dumps(payload, indent=2))
    summary_path = outdir / f"{args.tag}_{stamp}_summary.json"
    summary_path.write_text(
        json.dumps(
            {
                "created_utc": payload["created_utc"],
                "tag": args.tag,
                "args": payload["args"],
                "n_jobs": payload["n_jobs"],
                "n_ok": payload["n_ok"],
                "n_fail": payload["n_fail"],
                "ranking": agg["ranking"],
                "cells": agg["cells"],
            },
            indent=2,
        )
    )
    print(f"[{_now_iso()}] wrote {out_path}", flush=True)
    print(f"[{_now_iso()}] summary {summary_path}", flush=True)
    if agg["ranking"]:
        best = agg["ranking"][0]
        print(
            f"BEST: U={best['u_name']} L*={best['n_layers']} "
            f"success_rate={best['success_rate']:.3f} mean_p_gs={best['mean_p_gs']:.4f}",
            flush=True,
        )
    return 0 if payload["n_fail"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
