#!/usr/bin/env python3
"""Scale-eval — ansatz+GNN on eval n using gnn_best (zero-shot via shared heads).

For n without weights_n{n}.pt, loads train-n weights (prefer n9) into an n-sized
FactorGNN — shapes match because mixer heads are independent of C.
Requires data/data_n{n}.npz with dense E for the split.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

_PKG = Path(__file__).resolve().parents[1]
_ROOT = _PKG.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from explore_OCT29.lib.gnn import BASELINE_ARCH  # noqa: E402
from explore_OCT29.lib.train_loop import (  # noqa: E402
    eval_pack_metrics,
    load_gnn_best,
    load_pack,
    pick_transfer_state,
)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ns", type=str, default="6,9,12")
    ap.add_argument("--split", type=str, default="test")
    ap.add_argument("--L", type=int, default=5)
    ap.add_argument("--max_count", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--device", type=str, default="cpu", help="cpu / cuda:0")
    ap.add_argument("--nf", type=int, default=None)
    args = ap.parse_args(argv)

    arch, params_by_n = load_gnn_best()
    if arch is None:
        arch = BASELINE_ARCH
        params_by_n = {}
        print("No gnn_best — evaluating random-init baseline arch")

    transfer_state = pick_transfer_state(params_by_n)
    results = {}
    for n_s in args.ns.split(","):
        n = int(n_s.strip())
        try:
            pack = load_pack(n)
        except FileNotFoundError as e:
            print("skip", e)
            continue
        if f"{args.split}_E" not in pack:
            print(f"skip n={n}: no dense E for split={args.split} (generate with store_E)")
            continue
        p = params_by_n.get(n)
        tag = "native"
        if p is None:
            p = transfer_state
            tag = "zero-shot(from train-n)" if p is not None else "fresh-init"
            print(f"n={n}: no weights_n{n}.pt — {tag}")
        met = eval_pack_metrics(
            n, arch, p, split=args.split, L=args.L,
            max_count=args.max_count, seed=args.seed,
            device=args.device or None, nf=args.nf,
        )
        results[n] = met
        print(f"n={n}: Med={met['median_inv_p']:.3f} B90={met['B90_inv_p']:.3f} "
              f"leak={met['mean_leak']:.4f} wall={met['wall_s']:.1f}s [{tag}]")

    out = Path(args.out) if args.out else _PKG / "runs" / "eval_scale_last.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({str(k): v for k, v in results.items()}, f, indent=2)
    print("Wrote", out)


if __name__ == "__main__":
    main()
