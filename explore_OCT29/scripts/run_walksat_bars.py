#!/usr/bin/env python3
"""M1 — classical WalkSAT bars on pack test seeds → baselines/walksat_n{n}.json.

Uses pure-Python SKC WalkSAT (lib/walksat.py). On chaodao you can later compile
lib/walksat_cls.c and swap the runner; schema stays the same.
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

_PKG = Path(__file__).resolve().parents[1]
_ROOT = _PKG.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from explore_OCT29.lib.walksat import aggregate_bars, walksat_instance  # noqa: E402


def load_test_clauses(pack_path: Path):
    z = np.load(pack_path)
    n = int(z["n"])
    m = int(z["m"])
    seeds = z["test_seeds"]
    cls_v = z["test_cls_v"]
    cls_s = z["test_cls_s"]
    return n, m, seeds, cls_v, cls_s


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ns", type=str, default="6,9,12,15,18,21")
    ap.add_argument("--data", type=str, default=None)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--R", type=int, default=50, help="runs per instance")
    ap.add_argument("--p", type=float, default=0.5, help="WalkSAT noise param")
    ap.add_argument("--max_inst", type=int, default=None, help="cap test instances")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args(argv)

    data_dir = Path(args.data) if args.data else _PKG / "data"
    out_dir = Path(args.out) if args.out else _PKG / "baselines"
    out_dir.mkdir(parents=True, exist_ok=True)

    for n_s in args.ns.split(","):
        n_s = n_s.strip()
        if not n_s:
            continue
        n = int(n_s)
        pack = data_dir / f"data_n{n}.npz"
        out = out_dir / f"walksat_n{n}.json"
        if not pack.exists():
            print(f"[skip] missing {pack}")
            continue
        if out.exists() and not args.overwrite:
            print(f"[skip] {out} exists")
            continue
        n_, m, seeds, cls_v, cls_s = load_test_clauses(pack)
        assert n_ == n
        B = len(seeds)
        if args.max_inst is not None:
            B = min(B, args.max_inst)
        print(f"n={n}: WalkSAT p={args.p} R={args.R} on {B} test insts...", flush=True)
        t0 = time.time()
        per_med_flips = []
        per_med_evals = []
        rows = []
        for i in range(B):
            cl = [
                (cls_v[i, c].tolist(), cls_s[i, c].tolist())
                for c in range(m)
            ]
            res = walksat_instance(n, cl, R=args.R, p=args.p, seed=int(seeds[i]))
            per_med_flips.append(res["flips"]["median"])
            per_med_evals.append(res["evals"]["median"])
            rows.append({"seed": int(seeds[i]), "flips": res["flips"], "evals": res["evals"]})
            if (i + 1) % 10 == 0 or i + 1 == B:
                print(f"  {i+1}/{B}  wall={time.time()-t0:.1f}s", flush=True)
        summary = {
            "n": n,
            "m": m,
            "algo": "walksat_skc_python",
            "p": args.p,
            "R": args.R,
            "n_inst": B,
            "flips_across_inst": aggregate_bars(per_med_flips),
            "evals_across_inst": aggregate_bars(per_med_evals),
            "note": "Across-inst stats use per-instance median over R runs. "
                    "Compile lib/walksat_cls.c for faster UBCSAT-like binary later.",
            "instances": rows,
            "wall_s": time.time() - t0,
        }
        with open(out, "w") as f:
            json.dump(summary, f, indent=2)
        print(
            f"Saved {out}  Med_flips={summary['flips_across_inst']['median']:.3f} "
            f"B90={summary['flips_across_inst']['B90']:.3f}"
        )


if __name__ == "__main__":
    main()
