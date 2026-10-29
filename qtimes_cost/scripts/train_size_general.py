#!/usr/bin/env python3
"""Train one SizeGeneralGNN on n=6 and n=9, then eval n=12 with the same module.

Cavity heads are a shared MLP over per-cavity variable pools. No per-n module
and no cavity-repeat. Does not write runs/cloud_day1/BEST.
"""
from __future__ import annotations

import argparse
import json
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import time

import numpy as np
import torch

from qtimes_cost.lib.ansatz import AnsatzConfig, MultiCavityAnsatz
from qtimes_cost.lib.gnn import GNNArch, SizeGeneralGNN
from qtimes_cost.lib.train_loop import PKG, _instance_loss, _split_arrays, load_pack


def _copy_body(gnn: SizeGeneralGNN, factor_state: dict) -> list[str]:
    merged = {k: v.detach().clone() for k, v in gnn.state_dict().items()}
    copied = []
    for key, dst in list(merged.items()):
        if key not in factor_state:
            continue
        if key.startswith("cav_") or key.startswith("head_alpha") or key.startswith("head_A") or key.startswith("head_TH"):
            continue
        parent = factor_state[key].detach().to(dtype=dst.dtype)
        if tuple(parent.shape) != tuple(dst.shape):
            continue
        merged[key] = parent
        copied.append(key)
    gnn.load_state_dict(merged)
    return copied


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--steps", type=int, default=240)
    ap.add_argument("--L", type=int, default=12)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--leakpen", type=float, default=0.3)
    ap.add_argument("--body-from", type=str, default=str(PKG / "runs" / "cloud_day1" / "BEST_round1" / "weights_n9.pt"))
    ap.add_argument("--max-val", type=int, default=40)
    ap.add_argument("--max-test", type=int, default=100)
    ap.add_argument("--out", type=str, default=str(PKG / "runs" / "cloud_day1" / "size_general"))
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args(argv)
    torch.set_num_threads(1)
    arch = GNNArch(H=128, rounds=4, agg="mean", layernorm=True, residual=True, degree_features=True, lr=args.lr)
    gnn = SizeGeneralGNN(arch, L=args.L, seed=0)
    copied = []
    resume_path = os.path.join(args.out, "weights.pt")
    if args.resume and os.path.exists(resume_path):
        gnn.load_state_dict(torch.load(resume_path, map_location="cpu", weights_only=True))
        copied = ["resume"]
        print(f"resumed {resume_path}", flush=True)
    elif args.body_from and os.path.exists(args.body_from):
        copied = _copy_body(gnn, torch.load(args.body_from, map_location="cpu", weights_only=True))
        print(f"body tensors copied: {len(copied)}", flush=True)

    packs = {n: load_pack(n) for n in (6, 9, 12)}
    sims = {n: MultiCavityAnsatz(AnsatzConfig(n=n, L=args.L)) for n in packs}
    train = {n: _split_arrays(packs[n], "train", max_count=64) for n in (6, 9)}
    opt = torch.optim.Adam(gnn.parameters(), lr=args.lr, weight_decay=1e-4)
    rng = np.random.default_rng(0)

    def eval_n(n, split, count):
        arrays = {n: _split_arrays(packs[n], split, max_count=count)}
        # eval_nets calls forward_graph(cv, cs) without n. SizeGeneral needs n.
        # Evaluate directly.
        gnn.eval()
        _seeds, cls_v, cls_s, _nsol, E = arrays[n]
        inv, leak = [], []
        with torch.inference_mode():
            for b in range(len(_seeds)):
                heads = gnn.forward_graph(
                    torch.as_tensor(cls_v[b]),
                    torch.as_tensor(cls_s[b], dtype=torch.float32),
                    n=n,
                )
                out = sims[n].forward(
                    torch.as_tensor(E[b], dtype=torch.float32),
                    heads["alpha0"], heads["A"], heads["TH"], heads["q_logits"],
                    gamma=heads["gamma"],
                )
                inv.append(float(out["inv_p"]))
                leak.append(float(out["p_leak"]))
        gnn.train()
        inv_a = np.asarray(inv, float)
        med = float(np.median(inv_a))
        b90 = float(np.quantile(inv_a, 0.9))
        return {
            "B": int(len(inv_a)),
            "median_inv_p": med,
            "B90_inv_p": b90,
            "mean_leak": float(np.mean(leak)),
            "S": float((np.log(max(med, 1e-12)) + np.log(max(b90, 1e-12))) / 2),
        }

    best = None
    t0 = time.time()
    for step in range(args.steps):
        opt.zero_grad(set_to_none=True)
        losses = []
        for n in (6, 9):
            seeds, cls_v, cls_s, _nsol, E = train[n]
            idx = rng.choice(len(seeds), size=min(2, len(seeds)), replace=False)
            for b in idx:
                b = int(b)
                heads = gnn.forward_graph(
                    torch.as_tensor(cls_v[b]),
                    torch.as_tensor(cls_s[b], dtype=torch.float32),
                    n=n,
                )
                out = sims[n].forward(
                    torch.as_tensor(E[b], dtype=torch.float32),
                    heads["alpha0"], heads["A"], heads["TH"], heads["q_logits"],
                    gamma=heads["gamma"],
                )
                losses.append(_instance_loss(out, leakpen=args.leakpen))
        torch.stack(losses).mean().backward()
        torch.nn.utils.clip_grad_norm_(gnn.parameters(), 5.0)
        opt.step()
        if step % 40 == 0 or step == args.steps - 1:
            m12 = eval_n(12, "val", 8)
            mark = ""
            if best is None or m12["S"] < best["S"]:
                best = {**m12, "step": step, "state": {k: v.detach().cpu().clone() for k, v in gnn.state_dict().items()}}
                mark = " *"
            print(
                f"step {step} t={time.time()-t0:.0f}s n12 val8 Med={m12['median_inv_p']:.2f} "
                f"B90={m12['B90_inv_p']:.2f} leak={m12['mean_leak']:.4f}{mark}",
                flush=True,
            )

    gnn.load_state_dict(best["state"])
    os.makedirs(args.out, exist_ok=True)
    torch.save(best["state"], os.path.join(args.out, "weights.pt"))
    report = {
        "L": args.L,
        "leakpen": args.leakpen,
        "steps": args.steps,
        "body_from": args.body_from,
        "body_tensors_copied": copied,
        "best_step": best["step"],
        "note": "one SizeGeneralGNN; L=16 BEST directory was not written",
        "by_n": {},
    }
    for n in (6, 9, 12):
        report["by_n"][str(n)] = {}
        for split, count in (("val", args.max_val), ("test", args.max_test)):
            m = eval_n(n, split, count)
            report["by_n"][str(n)][split] = m
            print(
                f"n={n} {split} B={m['B']} Med={m['median_inv_p']:.3f} "
                f"B90={m['B90_inv_p']:.3f} leak={m['mean_leak']:.4f}",
                flush=True,
            )
    # drop state from json
    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump(report, f, indent=2)
    print("Wrote", args.out, flush=True)


if __name__ == "__main__":
    main()
