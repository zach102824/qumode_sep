#!/usr/bin/env python3
"""Train Arm C shell at L=2 on n9+n12 (probe6 protocol), eval scale1000 B=1000.

Fresh init (no L=16 warm-start). Writes to runs/armC_L2_scale1000/ only —
does not touch runs/armC_B1000 or probe6 L=16 / L=1 params.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

_PKG = Path(__file__).resolve().parents[1]
_ROOT = _PKG.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from explore_OCT29.lib.drivers import fit_b  # noqa: E402
from explore_OCT29.lib.spectral_arm import build_W_H  # noqa: E402
from explore_OCT29.scripts.eval_armC_B1000 import (  # noqa: E402
    _S,
    _jsonable,
    eval_n,
    fit_windows,
)
from explore_OCT29.scripts.probe6_spectral_mixer import train_arm  # noqa: E402

OUT = _PKG / "runs" / "armC_L2_scale1000"
DATA = _PKG / "data" / "scale1000"


def log(*a):
    print(time.strftime("%H:%M:%S"), "[armC_L2]", *a, flush=True)


def write_results_md(path: Path, meta: dict, ladder: dict, fits: dict, train_info: dict):
    lines = [
        "# Arm C L=2 — B=1000 scaling eval",
        "",
        f"_Generated {time.strftime('%Y-%m-%d %H:%M:%S %z')} (Asia/Shanghai)_",
        "",
        f"**Motif:** probe6 Arm C free shells — prep=`W_H`, layer `COST → W_H·SNAP(φ_pc)·W_H†`, L={meta['L']}",
        f"**Train:** n∈{{9,12}} sealed train, steps={meta['steps']}, max_train={meta['max_train']}, "
        f"max_val={meta['max_val']}, seed={meta['seed']}, lr={meta['lr']}, gamma0/beta0=None (fresh L=2)",
        f"**Params:** `{meta['params']}`",
        f"**Pack:** `{meta['data_dir']}` split=`{meta['split']}`",
        f"**B:** {meta['B']} per n (requested)",
        "",
        "## Train (val @ best step)",
        "",
        f"- best_step={train_info.get('best_step')}  Scomb={train_info.get('score')}",
        f"- train_wall_s={train_info.get('wall_s'):.1f}",
        "",
        "| n | Med | B90 | B |",
        "|---|-----|-----|---|",
    ]
    for n in (9, 12):
        v = (train_info.get("val") or {}).get(n) or (train_info.get("val") or {}).get(str(n))
        if not v:
            continue
        lines.append(
            f"| {n} | {v['median_inv_p']:.3f} | {v['B90_inv_p']:.3f} | {v.get('B', '?')} |"
        )
    lines += [
        "",
        "## Ladder (scale1000)",
        "",
        "| n | Med | B90 | leak | L×Med | frac_inv≤2 | B | wall_min |",
        "|---|-----|-----|------|-------|------------|---|----------|",
    ]
    for n in sorted(int(k) for k in ladder.keys()):
        m = ladder[n] if n in ladder else ladder[str(n)]
        lines.append(
            f"| {n} | {m['median_inv_p']:.3f} | {m['B90_inv_p']:.3f} | "
            f"{m['mean_leak']:.4f} | {m['L_times_Med']:.1f} | "
            f"{m['frac_inv_le_2']:.3f} | {m['B']} | {m['wall_s']/60:.1f} |"
        )
    lines += ["", "## b fits", ""]
    for name, f in fits.items():
        bm = f["b_Med"]["b"]
        bb = f["b_B90"]["b"]
        lines.append(
            f"- **{name}** (ns={f['ns']}): b_Med≈**{bm:.4f}**, b_B90≈**{bb:.4f}**"
        )
    lines += [
        "",
        "## Notes",
        "",
        "- Fresh L=2 train; does not reuse L=16 champ reshape.",
        "- Out dir separate from `runs/armC_B1000` (L=16).",
        "",
    ]
    path.write_text("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ns", type=str, default="9,12,15,18")
    ap.add_argument("--B", type=int, default=1000)
    ap.add_argument("--L", type=int, default=2)
    ap.add_argument("--nf", type=int, default=16)
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--max_train", type=int, default=32)
    ap.add_argument("--max_val", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--data_dir", type=str, default=str(DATA))
    ap.add_argument("--split", type=str, default="scale1000")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--progress_every", type=int, default=25)
    ap.add_argument("--skip_train", action="store_true",
                    help="reuse existing arm_C_shell_params.pt in OUT")
    args = ap.parse_args()

    if args.L != 2:
        log(f"WARNING: L={args.L} (script intended for L=2)")

    os.environ.setdefault("OMP_NUM_THREADS", str(args.threads))
    os.environ.setdefault("MKL_NUM_THREADS", str(args.threads))
    torch.set_num_threads(args.threads)

    OUT.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir)
    device = torch.device(args.device)
    W = build_W_H(args.nf)
    params_path = OUT / "arm_C_shell_params.pt"
    train_log_path = OUT / "train.json"

    train_info = {}
    if args.skip_train and params_path.exists():
        log(f"skip_train: loading {params_path}")
        st = torch.load(params_path, map_location="cpu", weights_only=False)
        if train_log_path.exists():
            train_info = json.loads(train_log_path.read_text())
    else:
        log(
            f"=== train Arm C shell L={args.L} on n9+n12 "
            f"steps={args.steps} seed={args.seed} gamma0=beta0=None ==="
        )
        t0 = time.time()
        best = train_arm(
            "shell", W, args.L, args.nf, device,
            steps=args.steps, max_train=args.max_train, max_val=args.max_val,
            seed=args.seed, gamma0=None, beta0=None, lr=args.lr,
        )
        train_wall = time.time() - t0
        st = best["state"]
        torch.save(st, params_path)
        # make val JSON-serializable
        val_j = {}
        for n, m in (best.get("val") or {}).items():
            val_j[str(n)] = {k: (float(v) if isinstance(v, (float, np.floating)) else
                                 int(v) if isinstance(v, (int, np.integer)) else v)
                             for k, v in m.items() if k != "inv_p_all"}
        train_info = {
            "best_step": int(best["step"]),
            "score": float(best["score"]),
            "val": val_j,
            "wall_s": float(train_wall),
            "L": args.L,
            "mode": "shell",
            "steps": args.steps,
            "seed": args.seed,
            "lr": args.lr,
            "max_train": args.max_train,
            "max_val": args.max_val,
            "gamma0": None,
            "beta0": None,
        }
        train_log_path.write_text(json.dumps(_jsonable(train_info), indent=2))
        log(
            f"train DONE step={best['step']} Scomb={best['score']:.3f} "
            f"wall={train_wall/60:.1f}min → {params_path}"
        )
        for n in (9, 12):
            v = best["val"][n]
            log(f"  val n{n}: Med={v['median_inv_p']:.3f} B90={v['B90_inv_p']:.3f}")

    gamma = st["gamma"].detach().cpu().to(dtype=torch.float32).reshape(-1)
    phi = st["phi"].detach().cpu().to(dtype=torch.float32)
    L = int(gamma.numel())
    if L != args.L or phi.shape != (L, 4):
        raise SystemExit(f"bad state: L={L} phi={tuple(phi.shape)} expected L={args.L}")

    ns = [int(x) for x in args.ns.split(",") if x.strip()]
    meta = {
        "L": L,
        "nf": args.nf,
        "B": args.B,
        "params": str(params_path),
        "data_dir": str(data_dir),
        "split": args.split,
        "device": str(device),
        "threads": args.threads,
        "steps": args.steps,
        "max_train": args.max_train,
        "max_val": args.max_val,
        "seed": args.seed,
        "lr": args.lr,
        "started": time.strftime("%Y-%m-%d %H:%M:%S %z"),
    }

    # normalize train_info val keys to int for MD
    if train_info.get("val"):
        train_info["val"] = {
            (int(k) if str(k).isdigit() else k): v
            for k, v in train_info["val"].items()
        }

    ladder = {}
    results_path = OUT / "results.json"
    if results_path.exists():
        try:
            prev = json.loads(results_path.read_text())
            for k, v in (prev.get("ladder") or {}).items():
                ladder[int(k)] = v
            log(f"resumed partial ladder ns={sorted(ladder)}")
        except Exception as e:
            log("resume skip", e)

    for n in ns:
        if n in ladder and ladder[n].get("B", 0) >= args.B:
            log(f"skip n={n} already done B={ladder[n]['B']}")
            continue
        log(f"=== eval n={n} B<={args.B} ===")
        m = eval_n(
            n, W, gamma, phi, L, args.nf, device, data_dir,
            split=args.split, max_count=args.B,
            progress_every=args.progress_every,
        )
        ladder[n] = m
        log(
            f"DONE n={n}: Med={m['median_inv_p']:.3f} B90={m['B90_inv_p']:.3f} "
            f"leak={m['mean_leak']:.4f} L×Med={m['L_times_Med']:.1f} "
            f"frac≤2={m['frac_inv_le_2']:.3f} B={m['B']} wall={m['wall_s']/60:.1f}min"
        )
        fits = fit_windows(ladder)
        blob = {
            "meta": meta,
            "train": _jsonable(train_info),
            "ladder": {str(k): v for k, v in sorted(ladder.items())},
            "fits": _jsonable(fits),
            "updated": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        }
        # incremental: keep inv_p_all in memory but strip for mid JSON size
        mid = _jsonable(blob)
        for n_s, mm in mid["ladder"].items():
            invs = mm.pop("inv_p_all", None)
            if invs is not None:
                np.save(OUT / f"inv_p_n{n_s}.npy", np.asarray(invs, dtype=np.float64))
        results_path.write_text(json.dumps(mid, indent=2))
        write_results_md(OUT / "RESULTS.md", meta, ladder, fits, train_info)
        log(f"wrote {results_path}")

    fits = fit_windows(ladder)
    meta["finished"] = time.strftime("%Y-%m-%d %H:%M:%S %z")
    blob = {
        "meta": meta,
        "train": _jsonable(train_info),
        "ladder": {str(k): v for k, v in sorted(ladder.items())},
        "fits": _jsonable(fits),
        "updated": meta["finished"],
    }
    slim = _jsonable(blob)
    for n_s, m in slim["ladder"].items():
        invs = m.pop("inv_p_all", None)
        if invs is not None:
            np.save(OUT / f"inv_p_n{n_s}.npy", np.asarray(invs, dtype=np.float64))
    results_path.write_text(json.dumps(slim, indent=2))
    write_results_md(OUT / "RESULTS.md", meta, ladder, fits, train_info)
    log("ALL DONE")
    for name, f in fits.items():
        log(f"fit {name}: b_Med={f['b_Med']['b']:.4f} b_B90={f['b_B90']['b']:.4f}")


if __name__ == "__main__":
    main()
