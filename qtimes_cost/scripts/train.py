#!/usr/bin/env python3
"""M2 thin entry — short train on n=6&9 with current/baseline GNN arch."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

_PKG = Path(__file__).resolve().parents[1]
_ROOT = _PKG.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from qtimes_cost.lib.gnn import BASELINE_ARCH, GNNArch  # noqa: E402
from qtimes_cost.lib.train_loop import load_gnn_best, save_gnn_best, short_train  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--L", type=int, default=5)
    ap.add_argument("--max_val", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save_best", action="store_true", help="write runs/gnn_best if improved/empty")
    ap.add_argument("--out", type=str, default=None, help="optional JSON metrics path")
    args = ap.parse_args(argv)

    arch, _ = load_gnn_best()
    if arch is None:
        arch = BASELINE_ARCH
        print("Using baseline arch", arch.to_dict())
    else:
        print("Using gnn_best arch", arch.to_dict())

    metrics, params, info = short_train(
        arch, steps=args.steps, L=args.L, max_val=args.max_val, seed=args.seed
    )
    payload = {"info": info, "metrics": {str(k): v for k, v in metrics.items()}}
    out = Path(args.out) if args.out else _PKG / "runs" / "train_last.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)
    print("Wrote", out)
    if args.save_best:
        save_gnn_best(arch, params, metrics)
        print("Saved runs/gnn_best/")


if __name__ == "__main__":
    main()
