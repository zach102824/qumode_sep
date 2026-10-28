#!/usr/bin/env python3
"""M-GNN — nonstop GNN evolve; disk keeps best only under runs/gnn_best/.

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
from pathlib import Path

import numpy as np

_PKG = Path(__file__).resolve().parents[1]
_ROOT = _PKG.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from qtimes_cost.lib.gnn import (  # noqa: E402
    BASELINE_ARCH,
    GNNArch,
    mutate_arch,
    random_arch,
)
from qtimes_cost.lib.train_loop import (  # noqa: E402
    combined_score,
    load_gnn_best,
    save_gnn_best,
    short_train,
)

STOP_NAME = "STOP_EVOLVE"
CSV_NAME = "gnn_evolve_trials.csv"
LOCK_NAME = "gnn_best.lock"


def _acquire_lock(lock_path: Path, timeout: float = 60.0) -> int:
    """Simple exclusive file lock via O_EXCL; returns fd."""
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


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max_trials", type=int, default=0, help="0 = forever")
    ap.add_argument("--steps", type=int, default=15, help="short_train steps per trial")
    ap.add_argument("--max_val", type=int, default=12)
    ap.add_argument("--L", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eps_mutate", type=float, default=0.7, help="P(mutate best) else random")
    ap.add_argument("--gpu", type=str, default="", help="optional CUDA_VISIBLE_DEVICES hint")
    ap.add_argument("--min_improve", type=float, default=0.0, help="require S improve by this")
    ap.add_argument("--runs", type=str, default=None)
    args = ap.parse_args(argv)

    if args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    runs = Path(args.runs) if args.runs else _PKG / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    stop_path = runs / STOP_NAME
    csv_path = runs / CSV_NAME
    lock_path = runs / LOCK_NAME

    parent_arch, _ = load_gnn_best(runs)
    if parent_arch is None:
        parent_arch = BASELINE_ARCH
        print("No gnn_best yet — parent = baseline", parent_arch.to_dict())
    else:
        print("Resumed parent from gnn_best:", parent_arch.to_dict())

    best_score = float("inf")
    # try to recover score from arch.json metrics
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
        "n6_med", "n6_b90", "n9_med", "n9_b90", "seed", "parent_S",
    ]
    trial = 0
    while True:
        if stop_path.exists():
            print("STOP file seen — exiting evolve loop")
            break
        if args.max_trials and trial >= args.max_trials:
            print("max_trials reached — exit")
            break
        trial += 1
        if rng.random() < args.eps_mutate:
            cand = mutate_arch(parent_arch, rng, n_flips=int(rng.integers(1, 3)))
            mode = "mutate"
        else:
            cand = random_arch(rng)
            mode = "random"
        print(f"\n=== trial {trial} ({mode}) {cand.to_dict()} ===", flush=True)
        t0 = time.time()
        try:
            metrics, params, info = short_train(
                cand,
                steps=args.steps,
                L=args.L,
                max_val=args.max_val,
                seed=args.seed + trial,
            )
            sc = combined_score(metrics)
        except Exception as e:
            print("trial failed:", e)
            append_csv(csv_path, {
                "trial_id": trial, "wall_s": f"{time.time()-t0:.1f}", "keep": "error",
                "S_comb": "", "best_S": f"{best_score:.6f}", "arch_json": json.dumps(cand.to_dict()),
                "n6_med": "", "n6_b90": "", "n9_med": "", "n9_b90": "",
                "seed": args.seed + trial, "parent_S": f"{best_score:.6f}",
            }, fields)
            continue

        keep = sc + args.min_improve < best_score
        row = {
            "trial_id": trial,
            "wall_s": f"{time.time()-t0:.1f}",
            "keep": "keep" if keep else "discard",
            "S_comb": f"{sc:.6f}",
            "best_S": f"{best_score:.6f}",
            "arch_json": json.dumps(cand.to_dict()),
            "n6_med": f"{metrics.get(6,{}).get('median_inv_p','')}",
            "n6_b90": f"{metrics.get(6,{}).get('B90_inv_p','')}",
            "n9_med": f"{metrics.get(9,{}).get('median_inv_p','')}",
            "n9_b90": f"{metrics.get(9,{}).get('B90_inv_p','')}",
            "seed": args.seed + trial,
            "parent_S": f"{best_score:.6f}",
        }
        append_csv(csv_path, row, fields)

        if keep:
            fd = _acquire_lock(lock_path)
            try:
                save_gnn_best(cand, params, metrics, runs_dir=runs)
                best_score = sc
                parent_arch = cand
                print(f"PROMOTED trial {trial} S_comb={sc:.4f} → runs/gnn_best/")
            finally:
                _release_lock(fd, lock_path)
        else:
            print(f"discard trial {trial} S_comb={sc:.4f} (best={best_score:.4f})")
            # discard candidate weights (never saved)


if __name__ == "__main__":
    main()
