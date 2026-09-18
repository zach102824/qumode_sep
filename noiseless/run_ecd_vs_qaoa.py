#!/usr/bin/env python3
"""Three-arm bake-off: ECD(bs_pi4) vs QAOA-full vs QAOA-NN at matched param counts.

Param tiers: 16 / 24 / 32
  ECD:       L* ∈ {2, 3, 4}  (8 params/layer)
  QAOA:      p  ∈ {8,12,16}  (2 params/layer)

Tag default: fleet_ecd_vs_qaoa
"""

from __future__ import annotations

import os as _os

_os.environ.setdefault("OMP_NUM_THREADS", "1")
_os.environ.setdefault("MKL_NUM_THREADS", "1")
_os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
_os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

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

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from noiseless.circuit_local_ecd import n_parameters
from noiseless.encoding import (
    list_four_sat_npz,
    load_four_sat_npz,
    z_terms_from_npz,
)
from noiseless.qaoa import (
    QAOASimulator,
    diagonal_spectrum_from_terms,
    n_qaoa_parameters,
    optimize_qaoa_trial,
    truncate_terms_nn_ring,
)
from noiseless.spsa_gibbs import (
    NoiselessSimulator,
    ground_flat_from_bitstring,
    optimize_trial,
    scale_spsa_a,
)
from noiseless.unitaries import build_fixed_u

ARMS = ("ecd", "qaoa_full", "qaoa_nn")
ECD_LAYERS = (2, 3, 4)
QAOA_P = (8, 12, 16)
PARAM_TIERS = (16, 24, 32)


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _depth_for_arm(arm: str, n_params: int) -> int:
    if arm == "ecd":
        return {16: 2, 24: 3, 32: 4}[int(n_params)]
    return {16: 8, 24: 12, 32: 16}[int(n_params)]


def _worker(job: dict) -> dict:
    t0 = time.perf_counter()
    arm = job["arm"]
    try:
        ham_path = Path(job["ham_path"])
        inst = load_four_sat_npz(ham_path)
        gs = inst["ground_bitstring"]
        terms, meta = z_terms_from_npz(ham_path)
        identity = float(meta["identity"])
        full_e = diagonal_spectrum_from_terms(terms, identity)
        nn_terms, trunc = truncate_terms_nn_ring(terms)
        nn_e = diagonal_spectrum_from_terms(nn_terms, identity)
        rng = np.random.default_rng(int(job["seed"]))
        n_params = int(job["n_params"])
        depth = int(job["depth"])
        a = job.get("spsa_a")
        if a is None:
            a = scale_spsa_a(n_params)
        a = float(a)

        if arm == "ecd":
            assert n_parameters(depth) == n_params
            u = build_fixed_u("bs_pi4")
            sim = NoiselessSimulator(
                u_fixed=u,
                energy_tensor=inst["energy_tensor"],
                n_layers=depth,
                ground_bitstring=gs,
                ground_flat_index=ground_flat_from_bitstring(gs),
            )
            result = optimize_trial(
                sim,
                maxiter=int(job["steps"]),
                rng=rng,
                a=a,
                c=float(job.get("spsa_c", 0.15)),
                A=float(job.get("spsa_A", 10.0)),
            )
            return {
                "ok": True,
                "arm": arm,
                "ham_file": job["ham_file"],
                "n_params": n_params,
                "depth": depth,
                "n_layers": depth,
                "p_layers": None,
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
                "spsa_a": a,
                "wall_s": float(time.perf_counter() - t0),
                "nn_trunc": trunc,
                "x": result.x.tolist(),
                "error": None,
            }

        if arm == "qaoa_full":
            cost_e = full_e
            eval_e = full_e
        elif arm == "qaoa_nn":
            cost_e = nn_e
            eval_e = full_e
        else:
            raise ValueError(f"unknown arm {arm!r}")

        assert n_qaoa_parameters(depth) == n_params
        sim = QAOASimulator(
            cost_energies=cost_e,
            ground_bitstring=gs,
            eval_energies=eval_e,
        )
        result = optimize_qaoa_trial(
            sim,
            depth,
            maxiter=int(job["steps"]),
            rng=rng,
            a=a,
            c=float(job.get("spsa_c", 0.15)),
            A=float(job.get("spsa_A", 10.0)),
        )
        return {
            "ok": True,
            "arm": arm,
            "ham_file": job["ham_file"],
            "n_params": n_params,
            "depth": depth,
            "n_layers": None,
            "p_layers": depth,
            "trial": int(job["trial"]),
            "seed": int(job["seed"]),
            "success": bool(result.success),
            "p_gs": float(result.p_gs),
            "most_likely_bitstring": result.most_likely_bitstring,
            "ground_bitstring": result.ground_bitstring,
            "fun": float(result.fun),
            "eta": float(result.eta),
            "energy_mean": float(result.energy_mean),
            "energy_mean_cost": float(result.energy_mean_cost),
            "nfev": int(result.nfev),
            "nit": int(result.nit),
            "spsa_a": a,
            "wall_s": float(time.perf_counter() - t0),
            "nn_trunc": trunc,
            "x": result.x.tolist(),
            "error": None,
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "ok": False,
            "arm": job.get("arm"),
            "ham_file": job.get("ham_file"),
            "n_params": job.get("n_params"),
            "depth": job.get("depth"),
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
        key = (r["arm"], int(r["n_params"]), r["ham_file"])
        by_key.setdefault(key, []).append(r)
    cells = []
    for (arm, n_params, ham_file), trials in sorted(by_key.items()):
        n = len(trials)
        n_succ = sum(1 for t in trials if t["success"])
        pgs = [float(t["p_gs"]) for t in trials]
        cells.append(
            {
                "arm": arm,
                "n_params": n_params,
                "depth": int(trials[0]["depth"]),
                "ham_file": ham_file,
                "n_trials": n,
                "n_success": n_succ,
                "success_rate": n_succ / n if n else 0.0,
                "mean_p_gs": float(np.mean(pgs)) if pgs else 0.0,
                "max_p_gs": float(np.max(pgs)) if pgs else 0.0,
                "median_p_gs": float(np.median(pgs)) if pgs else 0.0,
            }
        )
    by_ap: dict[tuple, list[dict]] = {}
    for c in cells:
        by_ap.setdefault((c["arm"], c["n_params"]), []).append(c)
    ranking = []
    for (arm, n_params), group in by_ap.items():
        weights = [g["n_trials"] for g in group]
        tot = sum(weights)
        succ = sum(g["n_success"] for g in group)
        mean_p = (
            float(np.average([g["mean_p_gs"] for g in group], weights=weights)) if tot else 0.0
        )
        ranking.append(
            {
                "arm": arm,
                "n_params": n_params,
                "depth": _depth_for_arm(arm, n_params),
                "n_hamiltonians": len(group),
                "n_trials": tot,
                "n_success": succ,
                "success_rate": succ / tot if tot else 0.0,
                "mean_p_gs": mean_p,
            }
        )
    ranking.sort(key=lambda r: (r["n_params"], -r["success_rate"], -r["mean_p_gs"]))
    by_tier = {}
    for tier in PARAM_TIERS:
        rows = [r for r in ranking if r["n_params"] == tier]
        rows.sort(key=lambda r: (r["success_rate"], r["mean_p_gs"]), reverse=True)
        by_tier[str(tier)] = rows
        if rows:
            by_tier[f"{tier}_winner"] = rows[0]["arm"]
    return {"cells": cells, "ranking": ranking, "by_tier": by_tier}


def _nn_trunc_summary(records: list[dict]) -> dict:
    truncs = [r["nn_trunc"] for r in records if r.get("ok") and r.get("nn_trunc")]
    # one per ham (from any arm); dedupe by collecting unique ham via first-seen
    # Prefer averaging across unique ham files from qaoa_nn trials trial==0
    per_ham: dict[str, dict] = {}
    for r in records:
        if not r.get("ok") or not r.get("nn_trunc"):
            continue
        hf = r["ham_file"]
        if hf not in per_ham:
            per_ham[hf] = r["nn_trunc"]
    if not per_ham:
        return {}
    vals = list(per_ham.values())
    keys = [
        "n_terms_full",
        "n_terms_kept",
        "n_terms_dropped",
        "n_weight1",
        "n_weight2_nn",
        "n_weight2_non_nn",
        "n_weight_ge3",
        "frac_terms_dropped",
        "frac_coeff_l1_dropped",
    ]
    out = {"n_hamiltonians": len(vals), "per_hamiltonian": per_ham}
    for k in keys:
        arr = [float(v[k]) for v in vals]
        out[f"mean_{k}"] = float(np.mean(arr))
        out[f"min_{k}"] = float(np.min(arr))
        out[f"max_{k}"] = float(np.max(arr))
    return out


def _bestofn(records: list[dict], best_of_n: int = 25, top_k: int = 3) -> dict:
    by: dict[tuple, list[dict]] = {}
    for r in records:
        if not r.get("ok"):
            continue
        key = (r["arm"], int(r["n_params"]), r["ham_file"])
        by.setdefault(key, []).append(r)
    per_ham: dict[str, list[dict]] = {arm: [] for arm in ARMS}
    tier_summary: dict[str, dict] = {}
    for (arm, n_params, ham_file), trials in sorted(by.items()):
        pgs = sorted((float(t["p_gs"]) for t in trials), reverse=True)
        n_succ = sum(1 for t in trials if t["success"])
        best = pgs[0] if pgs else 0.0
        top = pgs[:top_k]
        row = {
            "arm": arm,
            "n_params": n_params,
            "ham_file": ham_file,
            "n_trials": len(trials),
            "mean_p_gs": float(np.mean([float(t["p_gs"]) for t in trials])) if trials else 0.0,
            "mean_success": n_succ / len(trials) if trials else 0.0,
            "n_success": n_succ,
            f"best_of_{best_of_n}_p_gs": best,
            f"best_of_{best_of_n}_trial_success": bool(
                any(t["success"] and float(t["p_gs"]) == best for t in trials)
            )
            if trials
            else False,
            "any_trial_success": n_succ > 0,
            f"mean_top{top_k}_p_gs": float(np.mean(top)) if top else 0.0,
            f"top{top_k}_p_gs": top,
        }
        per_ham.setdefault(arm, []).append(row)
        tk = f"{arm}_{n_params}"
        tier_summary.setdefault(tk, []).append(row)
    aggregates = []
    for tk, rows in sorted(tier_summary.items()):
        arm, n_params_s = tk.rsplit("_", 1)
        n_params = int(n_params_s)
        aggregates.append(
            {
                "arm": arm,
                "n_params": n_params,
                "n_hamiltonians": len(rows),
                "mean_p_gs": float(np.mean([r["mean_p_gs"] for r in rows])),
                f"mean_best_of_{best_of_n}_p_gs": float(
                    np.mean([r[f"best_of_{best_of_n}_p_gs"] for r in rows])
                ),
                f"mean_top{top_k}_p_gs": float(
                    np.mean([r[f"mean_top{top_k}_p_gs"] for r in rows])
                ),
                "mean_success": float(np.mean([r["mean_success"] for r in rows])),
            }
        )
    aggregates.sort(key=lambda r: (r["n_params"], -r["mean_success"], -r["mean_p_gs"]))
    return {
        "best_of_N": best_of_n,
        "top_k": top_k,
        "per_hamiltonian": per_ham,
        "aggregates": aggregates,
    }


def build_jobs(args: argparse.Namespace) -> list[dict]:
    paths = list_four_sat_npz(Path(args.ham_dir))
    if args.max_h is not None:
        paths = paths[: int(args.max_h)]
    if not paths:
        raise SystemExit(f"No four_sat_*.npz under {args.ham_dir}")
    for p in paths:
        inst = load_four_sat_npz(p)
        if inst["num_spins"] != 8:
            raise SystemExit(f"{p.name} has num_spins={inst['num_spins']}, need 8")
        if inst["n_ground"] != 1:
            raise SystemExit(f"{p.name} has n_ground={inst['n_ground']}, need unique GS")

    arms = [s.strip() for s in args.arms.split(",") if s.strip()]
    for a in arms:
        if a not in ARMS:
            raise SystemExit(f"unknown arm {a!r}; choose from {ARMS}")
    tiers = [int(x) for x in args.param_tiers.split(",") if x.strip()]
    jobs = []
    seed0 = int(args.seed)
    arm_code = {a: i for i, a in enumerate(ARMS)}
    for hi, path in enumerate(paths):
        for arm in arms:
            for n_params in tiers:
                depth = _depth_for_arm(arm, n_params)
                for t in range(int(args.trials)):
                    seed = (
                        seed0
                        + 1_000_000 * hi
                        + 100_000 * arm_code[arm]
                        + 1_000 * n_params
                        + t
                    )
                    jobs.append(
                        {
                            "arm": arm,
                            "ham_path": str(path.resolve()),
                            "ham_file": path.name,
                            "n_params": n_params,
                            "depth": depth,
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
    p.add_argument("--arms", type=str, default="ecd,qaoa_full,qaoa_nn")
    p.add_argument("--param-tiers", type=str, default="16,24,32")
    p.add_argument("--trials", type=int, default=25)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--seed", type=int, default=20260918)
    p.add_argument("--max-h", type=int, default=None)
    p.add_argument("--spsa-a", type=float, default=None)
    p.add_argument("--spsa-c", type=float, default=0.15)
    p.add_argument("--spsa-A", type=float, default=10.0)
    p.add_argument("--outdir", type=str, default=str(_REPO / "noiseless" / "results"))
    p.add_argument("--tag", type=str, default="fleet_ecd_vs_qaoa")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args(argv)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if args.smoke:
        args.arms = "ecd,qaoa_full,qaoa_nn"
        args.param_tiers = "16"
        args.trials = 1
        args.steps = 8
        args.max_h = 1
        if args.tag == "fleet_ecd_vs_qaoa":
            args.tag = "smoke_ecd_vs_qaoa"

    jobs = build_jobs(args)
    print(
        f"[{_now()}] starting {len(jobs)} jobs workers={args.workers} "
        f"steps={args.steps} tag={args.tag} arms={args.arms} tiers={args.param_tiers}",
        flush=True,
    )
    records: list[dict] = []
    if args.workers <= 1 or len(jobs) == 1:
        for i, job in enumerate(jobs):
            rec = _worker(job)
            records.append(rec)
            print(
                f"  [{i+1}/{len(jobs)}] {'OK' if rec.get('ok') else 'FAIL'} "
                f"{rec.get('arm')} P={rec.get('n_params')} {rec.get('ham_file')} "
                f"succ={rec.get('success')} p_gs={rec.get('p_gs')} wall={rec.get('wall_s'):.2f}s",
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
                if done % 50 == 0 or done == len(jobs) or not rec.get("ok"):
                    print(
                        f"  [{done}/{len(jobs)}] {'OK' if rec.get('ok') else 'FAIL'} "
                        f"{rec.get('arm')} P={rec.get('n_params')} {rec.get('ham_file')} "
                        f"succ={rec.get('success')} p_gs={rec.get('p_gs')} "
                        f"wall={rec.get('wall_s'):.2f}s",
                        flush=True,
                    )

    agg = _aggregate(records)
    nn_stats = _nn_trunc_summary(records)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = outdir / f"{args.tag}_{stamp}.json"
    summary_path = outdir / f"{args.tag}_{stamp}_summary.json"
    bestofn_path = outdir / f"{args.tag}_{stamp}_bestofn.json"
    payload = {
        "created_utc": _now(),
        "tag": args.tag,
        "args": {
            "ham_dir": str(args.ham_dir),
            "arms": args.arms,
            "param_tiers": args.param_tiers,
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
        "nn_truncation": nn_stats,
        "records": records,
    }
    # Drop bulky x vectors from disk full dump? Keep for parity with other fleets.
    out_path.write_text(json.dumps(payload, indent=2))
    summary = {
        "created_utc": payload["created_utc"],
        "tag": args.tag,
        "args": payload["args"],
        "n_jobs": payload["n_jobs"],
        "n_ok": payload["n_ok"],
        "n_fail": payload["n_fail"],
        "ranking": agg["ranking"],
        "by_tier": agg["by_tier"],
        "cells": agg["cells"],
        "nn_truncation": {
            k: v
            for k, v in nn_stats.items()
            if k != "per_hamiltonian"
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    bon = _bestofn(records, best_of_n=int(args.trials), top_k=3)
    bestofn_payload = {
        "created_utc": _now(),
        "tag": args.tag,
        "source_full": out_path.name,
        "source_summary": summary_path.name,
        "settings": {
            "arms": [s.strip() for s in args.arms.split(",") if s.strip()],
            "param_tiers": [int(x) for x in args.param_tiers.split(",") if x.strip()],
            "trials_per_H": int(args.trials),
            "steps": int(args.steps),
            "n_hamiltonians": len({r["ham_file"] for r in records if r.get("ok")}),
            "n_jobs": len(records),
            "best_of_N": int(args.trials),
            "top_k": 3,
        },
        **bon,
    }
    bestofn_path.write_text(json.dumps(bestofn_payload, indent=2))

    print(f"[{_now()}] wrote {out_path}", flush=True)
    print(f"[{_now()}] summary {summary_path}", flush=True)
    print(f"[{_now()}] bestofn {bestofn_path}", flush=True)
    for tier in PARAM_TIERS:
        rows = agg["by_tier"].get(str(tier), [])
        if not rows:
            continue
        print(f"--- tier {tier} params ---", flush=True)
        for r in rows:
            print(
                f"  {r['arm']:10s} depth={r['depth']:2d} "
                f"succ={r['success_rate']:.3f} mean_p={r['mean_p_gs']:.4f} N={r['n_trials']}",
                flush=True,
            )
        print(f"  WINNER: {agg['by_tier'].get(f'{tier}_winner')}", flush=True)
    if nn_stats:
        print(
            f"NN trunc: mean frac terms dropped={nn_stats.get('mean_frac_terms_dropped', float('nan')):.3f} "
            f"mean frac |coeff| dropped={nn_stats.get('mean_frac_coeff_l1_dropped', float('nan')):.3f}",
            flush=True,
        )
    return 0 if payload["n_fail"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
