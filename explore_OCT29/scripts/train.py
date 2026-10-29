#!/usr/bin/env python3
"""Short train on n=6&9 with current/baseline GNN arch (step 0 shared mixer)."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

_PKG = Path(__file__).resolve().parents[1]
_ROOT = _PKG.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from explore_OCT29.lib.device import resolve_device  # noqa: E402
from explore_OCT29.lib.gnn import BASELINE_ARCH, GNNArch  # noqa: E402
from explore_OCT29.lib.train_loop import (  # noqa: E402
    eval_pack_metrics,
    load_gnn_best,
    load_pack,
    pick_transfer_state,
    save_gnn_best,
    short_train,
    transfer_score,
)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--L", type=int, default=5, help="depth; FactorGNN path often 5–16")
    ap.add_argument("--max_val", type=int, default=20)
    ap.add_argument("--max_n12", type=int, default=8, help="also score zero-shot n12 if pack exists")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="cpu", help="cpu / cuda:0 (default cpu for box)")
    ap.add_argument("--gpu", type=str, default="", help="optional CUDA_VISIBLE_DEVICES")
    ap.add_argument("--nf", type=int, default=None)
    ap.add_argument("--save_best", action="store_true", help="write runs/gnn_best")
    ap.add_argument("--out", type=str, default=None, help="optional JSON metrics path")
    args = ap.parse_args(argv)

    if args.gpu:
        import os
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    device = resolve_device(args.device or None)
    print(f"train device={device} L={args.L} freedom_step=0 shared_mixer=True")

    arch, _ = load_gnn_best()
    if arch is None:
        arch = BASELINE_ARCH
        print("Using baseline arch", arch.to_dict())
    else:
        print("Using gnn_best arch", arch.to_dict())

    metrics, params, info = short_train(
        arch, steps=args.steps, L=args.L, max_val=args.max_val,
        seed=args.seed, device=device, nf=args.nf,
    )

    # optional zero-shot n12 for champion seed
    try:
        load_pack(12)
        state = pick_transfer_state(params)
        m12 = eval_pack_metrics(
            12, arch, state, split="val", L=args.L, max_count=args.max_n12,
            seed=args.seed, device=device, nf=args.nf,
        )
        metrics[12] = m12
        print(f"n12 zero-shot: Med={m12['median_inv_p']:.3f} B90={m12['B90_inv_p']:.3f} "
              f"S={transfer_score(m12):.4f}")
    except FileNotFoundError:
        print("n12 pack missing — train metrics only (generate with gen_teb_packs)")

    payload = {"info": info, "metrics": {str(k): v for k, v in metrics.items()}}
    out = Path(args.out) if args.out else _PKG / "runs" / "train_last.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)
    print("Wrote", out)
    if args.save_best:
        extra = {}
        if 12 in metrics:
            extra = {"promote_metric": "n12_zero_shot", "S_n12": transfer_score(metrics[12])}
        save_gnn_best(arch, params, metrics, extra_meta=extra)
        print("Saved runs/gnn_best/")


if __name__ == "__main__":
    main()
