#!/usr/bin/env python3
"""M2 scale-eval — run ansatz+GNN on eval n using snapshotted gnn_best.

For n with dense E in the pack. Large-n chunking is a stub note for now.
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
from explore_OCT29.lib.train_loop import eval_pack_metrics, load_gnn_best, load_pack  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ns", type=str, default="6,9,12")
    ap.add_argument("--split", type=str, default="test")
    ap.add_argument("--L", type=int, default=5)
    ap.add_argument("--max_count", type=int, default=32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default=None)
    ap.add_argument("--device", type=str, default="", help="cuda:0 / cpu")
    args = ap.parse_args(argv)

    arch, params_by_n = load_gnn_best()
    if arch is None:
        arch = BASELINE_ARCH
        params_by_n = {}
        print("No gnn_best — evaluating random-init baseline arch")

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
        # Prefer n-specific weights if present, else None (fresh init of arch)
        p = params_by_n.get(n)
        if p is None and params_by_n:
            # fallback: use closest train-n weights keys only for readout init — skip
            print(f"n={n}: no weights_n{n}.npz — fresh GNN init with best arch")
        met = eval_pack_metrics(
            n, arch, p, split=args.split, L=args.L,
            max_count=args.max_count, seed=args.seed,
            device=args.device or None,
        )
        results[n] = met
        print(f"n={n}: Med={met['median_inv_p']:.3f} B90={met['B90_inv_p']:.3f} "
              f"leak={met['mean_leak']:.4f} wall={met['wall_s']:.1f}s")

    out = Path(args.out) if args.out else _PKG / "runs" / "eval_scale_last.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump({str(k): v for k, v in results.items()}, f, indent=2)
    print("Wrote", out)


if __name__ == "__main__":
    main()
