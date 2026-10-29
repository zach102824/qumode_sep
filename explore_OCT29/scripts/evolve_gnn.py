#!/usr/bin/env python3
"""M-GNN — nonstop GNN evolve; disk keeps best only under runs/gnn_best/.

Dual-GPU (chaodao 2×5090): each round trains TWO candidates in parallel
  — one on cuda:0, one on cuda:1 — then best-only promote.
Single GPU / CPU: still runs two candidates (sequentially if same device).

Loop: mutate/random arch → short_train on n6&9 → VAL score → overwrite best or discard.
Stop: touch runs/STOP_EVOLVE  or  --max_trials N
CSV: runs/gnn_evolve_trials.csv (append-only)
"""
from __future__ import annotations
import argparse
import csv
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch

_PKG = Path(__file__).resolve().parents[1]
_ROOT = _PKG.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from explore_OCT29.lib.device import dual_devices, parse_gpus_arg, resolve_device  # noqa: E402
from explore_OCT29.lib.gnn import (  # noqa: E402
    BASELINE_ARCH,
    GNNArch,
    mutate_arch,
    random_arch,
)
from explore_OCT29.lib.train_loop import (  # noqa: E402
    combined_score,
    load_gnn_best,
    save_gnn_best,
    short_train,
    should_promote,
)

STOP_NAME = "STOP_EVOLVE"
CSV_NAME = "gnn_evolve_trials.csv"
LOCK_NAME = "gnn_best.lock"


def _acquire_lock(lock_path: Path, timeout: float = 60.0) -> int:
    t0 = time.time()
    while True:
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, f"{os.getpid()}\n".encode())
            return fd
        except FileExistsError:
            if time.time() - t0 > timeout:
                raise TimeoutError(f"lock {lock_path} held too long")
            time.sleep(0.5)


def _release_lock(fd: int, lock_path: Path):
    try:
        os.close(fd)
    finally:
        try:
            lock_path.unlink()
        except FileNotFoundError:
            pass


def append_csv(path: Path, row: dict, fieldnames: list):
    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        if new:
            w.writeheader()
        w.writerow(row)


def _run_one(cand: GNNArch, device, steps, L, max_val, seed, data_dir=None):
    metrics, params, info = short_train(
        cand,
        steps=steps,
        L=L,
        max_val=max_val,
        seed=seed,
        device=device,
        data_dir=data_dir,
    )
    sc = combined_score(metrics)
    return cand, metrics, params, info, sc, str(device)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max_trials", type=int, default=0, help="0 = forever (counts candidate pairs×2 as trials)")
    ap.add_argument("--steps", type=int, default=15, help="short_train steps per candidate")
    ap.add_argument("--max_val", type=int, default=12)
    ap.add_argument("--L", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eps_mutate", type=float, default=0.7, help="P(mutate best) else random")
    ap.add_argument("--gpu", type=str, default="", help="optional CUDA_VISIBLE_DEVICES (e.g. 0,1)")
    ap.add_argument(
        "--gpus",
        type=str,
        default="0,1",
        help="two device indices/specs for parallel candidates (default 0,1)",
    )
    ap.add_argument("--n_parallel", type=int, default=2, help="candidates per round (default 2)")
    ap.add_argument("--min_improve", type=float, default=0.0, help="require S improve by this")
    ap.add_argument("--runs", type=str, default=None)
    ap.add_argument("--sequential", action="store_true", help="force sequential (debug)")
    args = ap.parse_args(argv)

    if args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    runs = Path(args.runs) if args.runs else _PKG / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    stop_path = runs / STOP_NAME
    csv_path = runs / CSV_NAME
    lock_path = runs / LOCK_NAME

    gpu_list = parse_gpus_arg(args.gpus)
    devices = dual_devices(gpu_list if gpu_list else None)
    # stretch/shrink to n_parallel
    while len(devices) < args.n_parallel:
        devices.append(devices[-1])
    devices = devices[: args.n_parallel]

    print(
        f"evolve_gnn backend=torch cuda_available={torch.cuda.is_available()} "
        f"cuda_count={torch.cuda.device_count() if torch.cuda.is_available() else 0} "
        f"devices={[str(d) for d in devices]}",
        flush=True,
    )

    parent_arch, _ = load_gnn_best(runs)
    if parent_arch is None:
        parent_arch = BASELINE_ARCH
        print("No gnn_best yet — parent = baseline", parent_arch.to_dict())
    else:
        print("Resumed parent from gnn_best:", parent_arch.to_dict())

    best_score = float("inf")
    arch_json = runs / "gnn_best" / "arch.json"
    if arch_json.exists():
        meta = json.load(open(arch_json))
        mets = {int(k): v for k, v in meta.get("metrics", {}).items()}
        if mets:
            best_score = combined_score(mets)
            print(f"Current best S_comb={best_score:.4f}")

    rng = np.random.default_rng(args.seed)
    fields = [
        "trial_id", "wall_s", "keep", "S_comb", "best_S", "arch_json",
        "n6_med", "n6_b90", "n9_med", "n9_b90", "seed", "parent_S", "device",
    ]
    trial = 0
    while True:
        if stop_path.exists():
            print("STOP file seen — exiting evolve loop")
            break
        if args.max_trials and trial >= args.max_trials:
            print("max_trials reached — exit")
            break

        # propose n_parallel candidates
        cands = []
        modes = []
        for _ in range(args.n_parallel):
            if rng.random() < args.eps_mutate:
                cands.append(mutate_arch(parent_arch, rng, n_flips=int(rng.integers(1, 3))))
                modes.append("mutate")
            else:
                cands.append(random_arch(rng))
                modes.append("random")

        print(
            f"\n=== round starting trials {trial+1}..{trial+args.n_parallel} "
            f"on {[str(d) for d in devices]} ===",
            flush=True,
        )
        for i, (c, m) in enumerate(zip(cands, modes)):
            print(f"  cand[{i}] ({m}) {c.to_dict()} -> {devices[i]}", flush=True)

        t0 = time.time()
        results = []

        def _job(i):
            tid = trial + i + 1
            return tid, _run_one(
                cands[i],
                devices[i],
                args.steps,
                args.L,
                args.max_val,
                args.seed + tid,
            )

        if args.sequential or args.n_parallel == 1:
            for i in range(args.n_parallel):
                if stop_path.exists():
                    break
                try:
                    tid, payload = _job(i)
                    results.append((tid, i, payload, None))
                except Exception as e:
                    results.append((trial + i + 1, i, None, e))
        else:
            with ThreadPoolExecutor(max_workers=args.n_parallel) as ex:
                futs = {ex.submit(_job, i): i for i in range(args.n_parallel)}
                for fut in as_completed(futs):
                    i = futs[fut]
                    try:
                        tid, payload = fut.result()
                        results.append((tid, i, payload, None))
                    except Exception as e:
                        results.append((trial + i + 1, i, None, e))

        # process in trial_id order
        results.sort(key=lambda x: x[0])
        for tid, ci, payload, err in results:
            trial = max(trial, tid)
            if err is not None or payload is None:
                print(f"trial {tid} failed:", err)
                append_csv(csv_path, {
                    "trial_id": tid, "wall_s": f"{time.time()-t0:.1f}", "keep": "error",
                    "S_comb": "", "best_S": f"{best_score:.6f}",
                    "arch_json": json.dumps(cands[ci].to_dict()),
                    "n6_med": "", "n6_b90": "", "n9_med": "", "n9_b90": "",
                    "seed": args.seed + tid, "parent_S": f"{best_score:.6f}",
                    "device": str(devices[ci]),
                }, fields)
                continue
            cand, metrics, params, info, sc, dev_s = payload
            keep = should_promote(sc, best_score, args.min_improve)
            row = {
                "trial_id": tid,
                "wall_s": f"{time.time()-t0:.1f}",
                "keep": "keep" if keep else "discard",
                "S_comb": f"{sc:.6f}",
                "best_S": f"{best_score:.6f}",
                "arch_json": json.dumps(cand.to_dict()),
                "n6_med": f"{metrics.get(6, {}).get('median_inv_p', '')}",
                "n6_b90": f"{metrics.get(6, {}).get('B90_inv_p', '')}",
                "n9_med": f"{metrics.get(9, {}).get('median_inv_p', '')}",
                "n9_b90": f"{metrics.get(9, {}).get('B90_inv_p', '')}",
                "seed": args.seed + tid,
                "parent_S": f"{best_score:.6f}",
                "device": dev_s,
            }
            append_csv(csv_path, row, fields)
            if keep:
                fd = _acquire_lock(lock_path)
                try:
                    # re-check under lock
                    if should_promote(sc, best_score, args.min_improve):
                        save_gnn_best(cand, params, metrics, runs_dir=runs)
                        best_score = sc
                        parent_arch = cand
                        print(f"PROMOTED trial {tid} S_comb={sc:.4f} on {dev_s} → runs/gnn_best/")
                    else:
                        print(f"discard trial {tid} (lost race) S_comb={sc:.4f}")
                        row["keep"] = "discard_race"
                finally:
                    _release_lock(fd, lock_path)
            else:
                print(f"discard trial {tid} S_comb={sc:.4f} (best={best_score:.4f}) on {dev_s}")


if __name__ == "__main__":
    main()
