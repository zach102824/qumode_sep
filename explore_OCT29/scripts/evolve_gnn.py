#!/usr/bin/env python3
"""Nonstop GNN evolve — step 0 shared mixer. Promote on zero-shot n12.

Skeleton frozen: shared SNAP+D + γ_ℓ, Φ=H_cost. Evolve arch/heads/train only.
No OpenEvolve. No head_q / per-cavity mixer mutations.

Loop: mutate/random arch → short_train on n6&9 → score val@6&9 + zero-shot@12
      → promote only if n12 S beats champion → runs/gnn_best/
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
    eval_pack_metrics,
    load_gnn_best,
    load_pack,
    pick_transfer_state,
    save_gnn_best,
    short_train,
    should_promote,
    transfer_score,
    FREEDOM_STEP,
    SHARED_MIXER,
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


def _eval_n12(arch, states_by_n, L, max_n12, seed, device, data_dir, nf=None):
    """Zero-shot n12 with train-n weights (prefer n9). Returns metrics or None if no pack."""
    try:
        load_pack(12, data_dir=data_dir)
    except FileNotFoundError:
        return None
    state = pick_transfer_state(states_by_n)
    return eval_pack_metrics(
        12, arch, state, split="val", L=L, max_count=max_n12,
        seed=seed, data_dir=data_dir, device=device, nf=nf,
    )


def _run_one(cand: GNNArch, device, steps, L, max_val, max_n12, seed, data_dir=None, nf=None):
    metrics, params, info = short_train(
        cand,
        steps=steps,
        L=L,
        max_val=max_val,
        seed=seed,
        device=device,
        data_dir=data_dir,
        nf=nf,
    )
    sc_train = combined_score(metrics)
    m12 = _eval_n12(cand, params, L, max_n12, seed, device, data_dir, nf=nf)
    if m12 is not None:
        metrics[12] = m12
        sc_promote = transfer_score(m12)
    else:
        # fallback: promote on train-n only if n12 pack missing (should not happen in prod)
        sc_promote = sc_train
    return cand, metrics, params, info, sc_promote, sc_train, str(device)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max_trials", type=int, default=0, help="0 = forever")
    ap.add_argument("--steps", type=int, default=15, help="short_train steps per candidate")
    ap.add_argument("--max_val", type=int, default=12)
    ap.add_argument("--max_n12", type=int, default=8, help="zero-shot n12 val count for promote gate")
    ap.add_argument("--L", type=int, default=5)
    ap.add_argument("--nf", type=int, default=None, help="Fock truncation (default packing.default_nf)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eps_mutate", type=float, default=0.7, help="P(mutate best) else random")
    ap.add_argument("--gpu", type=str, default="", help="optional CUDA_VISIBLE_DEVICES")
    ap.add_argument(
        "--gpus",
        type=str,
        default="",
        help="device specs for parallel candidates (default empty → CPU)",
    )
    ap.add_argument("--n_parallel", type=int, default=1, help="candidates per round (default 1 for CPU)")
    ap.add_argument("--min_improve", type=float, default=0.0, help="require n12 S improve by this")
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

    # clear stale STOP from prior runs if user wants forever — do NOT auto-delete;
    # only warn if present at start
    if stop_path.exists():
        print(f"WARNING: {stop_path} exists — loop will exit immediately. Remove it to run.", flush=True)

    gpu_list = parse_gpus_arg(args.gpus) if args.gpus.strip() else []
    if gpu_list:
        devices = dual_devices(gpu_list)
    else:
        devices = [resolve_device("cpu")]
    while len(devices) < args.n_parallel:
        devices.append(devices[-1])
    devices = devices[: args.n_parallel]

    print(
        f"evolve_gnn step0 shared_mixer={SHARED_MIXER} freedom_step={FREEDOM_STEP} "
        f"backend=torch cuda_available={torch.cuda.is_available()} "
        f"devices={[str(d) for d in devices]} L={args.L} promote=n12_zero_shot",
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
        if 12 in mets:
            best_score = transfer_score(mets[12])
            print(f"Current best n12 S={best_score:.4f} "
                  f"Med={mets[12].get('median_inv_p')} B90={mets[12].get('B90_inv_p')}")
        elif mets:
            best_score = combined_score(mets)
            print(f"Current best (no n12 yet) S_comb={best_score:.4f} — will replace once n12 scored")

    rng = np.random.default_rng(args.seed)
    fields = [
        "trial_id", "wall_s", "keep", "S_n12", "S_train", "best_S_n12", "arch_json",
        "n6_med", "n6_b90", "n9_med", "n9_b90", "n12_med", "n12_b90",
        "seed", "parent_S", "device", "freedom_step",
    ]
    trial = 0
    while True:
        if stop_path.exists():
            print("STOP file seen — exiting evolve loop")
            break
        if args.max_trials and trial >= args.max_trials:
            print("max_trials reached — exit")
            break

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
                args.max_n12,
                args.seed + tid,
                nf=args.nf,
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

        results.sort(key=lambda x: x[0])
        for tid, ci, payload, err in results:
            trial = max(trial, tid)
            if err is not None or payload is None:
                print(f"trial {tid} failed:", err)
                append_csv(csv_path, {
                    "trial_id": tid, "wall_s": f"{time.time()-t0:.1f}", "keep": "error",
                    "S_n12": "", "S_train": "", "best_S_n12": f"{best_score:.6f}",
                    "arch_json": json.dumps(cands[ci].to_dict()),
                    "n6_med": "", "n6_b90": "", "n9_med": "", "n9_b90": "",
                    "n12_med": "", "n12_b90": "",
                    "seed": args.seed + tid, "parent_S": f"{best_score:.6f}",
                    "device": str(devices[ci]), "freedom_step": FREEDOM_STEP,
                }, fields)
                continue
            cand, metrics, params, info, sc_promote, sc_train, dev_s = payload
            keep = should_promote(sc_promote, best_score, args.min_improve)
            row = {
                "trial_id": tid,
                "wall_s": f"{time.time()-t0:.1f}",
                "keep": "keep" if keep else "discard",
                "S_n12": f"{sc_promote:.6f}",
                "S_train": f"{sc_train:.6f}",
                "best_S_n12": f"{best_score:.6f}",
                "arch_json": json.dumps(cand.to_dict()),
                "n6_med": f"{metrics.get(6, {}).get('median_inv_p', '')}",
                "n6_b90": f"{metrics.get(6, {}).get('B90_inv_p', '')}",
                "n9_med": f"{metrics.get(9, {}).get('median_inv_p', '')}",
                "n9_b90": f"{metrics.get(9, {}).get('B90_inv_p', '')}",
                "n12_med": f"{metrics.get(12, {}).get('median_inv_p', '')}",
                "n12_b90": f"{metrics.get(12, {}).get('B90_inv_p', '')}",
                "seed": args.seed + tid,
                "parent_S": f"{best_score:.6f}",
                "device": dev_s,
                "freedom_step": FREEDOM_STEP,
            }
            append_csv(csv_path, row, fields)
            msg = (
                f"trial {tid} S_n12={sc_promote:.4f} S_train={sc_train:.4f} "
                f"n6={metrics.get(6, {}).get('median_inv_p', float('nan')):.2f}/"
                f"{metrics.get(6, {}).get('B90_inv_p', float('nan')):.2f} "
                f"n9={metrics.get(9, {}).get('median_inv_p', float('nan')):.2f}/"
                f"{metrics.get(9, {}).get('B90_inv_p', float('nan')):.2f} "
                f"n12={metrics.get(12, {}).get('median_inv_p', float('nan')):.2f}/"
                f"{metrics.get(12, {}).get('B90_inv_p', float('nan')):.2f}"
            )
            if keep:
                fd = _acquire_lock(lock_path)
                try:
                    if should_promote(sc_promote, best_score, args.min_improve):
                        save_gnn_best(
                            cand, params, metrics, runs_dir=runs,
                            extra_meta={"promote_metric": "n12_zero_shot", "S_n12": sc_promote},
                        )
                        best_score = sc_promote
                        parent_arch = cand
                        print(f"PROMOTED {msg} → runs/gnn_best/")
                    else:
                        print(f"discard (lost race) {msg}")
                finally:
                    _release_lock(fd, lock_path)
            else:
                print(f"discard {msg} (best_n12={best_score:.4f}) on {dev_s}")


if __name__ == "__main__":
    main()
