#!/usr/bin/env python3
"""Eval a saved qtimes_cost checkpoint at one or more n.

n=6 and n=9 load weights_n{n}.pt directly. A larger n (for example 12) copies
the n=9 message-passing body, q, and gamma, and repeats the last cavity's
α0 / displacement / SNAP head. Learned α0 is kept. No training.
"""
from __future__ import annotations

import argparse
import json
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import torch

from qtimes_cost.lib.ansatz import AnsatzConfig, MultiCavityAnsatz
from qtimes_cost.lib.cloud_worker import eval_nets, transfer_gnn_n
from qtimes_cost.lib.gnn import FactorGNN, GNNArch
from qtimes_cost.lib.train_loop import PKG, _split_arrays, load_pack


def _load_arch(ckpt: str) -> tuple[GNNArch, int]:
    meta = json.load(open(os.path.join(ckpt, "best.json")))
    arch = GNNArch.from_dict(meta["arch"])
    L = int(meta["cfg"]["L"])
    return arch, L, meta


def eval_ckpt(ckpt: str, ns: list[int], splits: list[str], max_count: int | None, parent_n: int = 9) -> dict:
    torch.set_num_threads(1)
    arch, L, meta = _load_arch(ckpt)
    out = {
        "ckpt": ckpt,
        "L": L,
        "cfg": meta.get("cfg"),
        "parent_n": parent_n,
        "transfer": "exact weights_n{n} when present; else body+q+gamma from parent_n, last cavity repeated",
        "by_n": {},
    }
    parent_path = os.path.join(ckpt, f"weights_n{parent_n}.pt")
    parent_state = torch.load(parent_path, map_location="cpu", weights_only=True) if os.path.exists(parent_path) else None
    for n in ns:
        pack = load_pack(n)
        weight_path = os.path.join(ckpt, f"weights_n{n}.pt")
        gnn = FactorGNN(arch, n=n, m=int(pack["m"]), L=L, seed=0)
        if os.path.exists(weight_path):
            state = torch.load(weight_path, map_location="cpu", weights_only=True)
            gnn.load_state_dict(state)
            how = "exact"
        elif parent_state is not None:
            src_C = transfer_gnn_n(gnn, parent_state)
            how = f"transfer_from_n{parent_n}_C{src_C}_to_C{gnn.C}_repeat_last_cavity"
        else:
            raise FileNotFoundError(f"no weights for n={n} and no parent {parent_path}")
        gnn.eval()
        sim = MultiCavityAnsatz(AnsatzConfig(n=n, L=L))
        out["by_n"][str(n)] = {"load": how, "splits": {}}
        for split in splits:
            arrays = {n: _split_arrays(pack, split, max_count=max_count)}
            # eval_nets expects a dict of nets keyed by n
            metrics = eval_nets({n: gnn}, {n: sim}, arrays, max_count or len(arrays[n][0]))
            out["by_n"][str(n)]["splits"][split] = {
                k: (float(v) if hasattr(v, "real") or isinstance(v, (int, float)) else v)
                for k, v in metrics[n].items()
            }
            m = metrics[n]
            print(
                f"n={n} {split} B={m['B']} Med={m['median_inv_p']:.3f} "
                f"B90={m['B90_inv_p']:.3f} leak={m['mean_leak']:.4f} ({how})",
                flush=True,
            )
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", type=str, default=str(PKG / "runs" / "cloud_day1" / "BEST_round1"))
    ap.add_argument("--ns", type=str, default="6,9,12")
    ap.add_argument("--splits", type=str, default="val,test")
    ap.add_argument("--max-count", type=int, default=0, help="0 = full split")
    ap.add_argument("--parent-n", type=int, default=9)
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args(argv)
    ns = [int(x) for x in args.ns.split(",") if x.strip()]
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]
    max_count = None if int(args.max_count) <= 0 else int(args.max_count)
    payload = eval_ckpt(args.ckpt, ns, splits, max_count, parent_n=int(args.parent_n))
    out = args.out or str(PKG / "runs" / "cloud_day1" / "n12_scale.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)
    print("Wrote", out, flush=True)


if __name__ == "__main__":
    main()
