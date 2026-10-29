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
    ap.add_argument("--n12-train", type=int, default=0, help="if >0, include this many n=12 train formulas from step 0")
    ap.add_argument("--n12-batch", type=int, default=1)
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
    train_ns = [6, 9] + ([12] if args.n12_train > 0 else [])
    train = {}
    for n in train_ns:
        cap = args.n12_train if n == 12 else 64
        train[n] = _split_arrays(packs[n], "train", max_count=cap)
    batch_for = {6: 2, 9: 2, 12: max(1, args.n12_batch)}
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
        # Per-n mean, then mix. n=12 is in the batch from step 0 when requested.
        mix_w = {6: 0.35, 9: 0.40, 12: 0.25}
        per_n = []
        for n in train_ns:
            seeds, cls_v, cls_s, _nsol, E = train[n]
            idx = rng.choice(len(seeds), size=min(batch_for[n], len(seeds)), replace=False)
            losses = []
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
            per_n.append(mix_w[n] * torch.stack(losses).mean())
        torch.stack(per_n).sum().backward()
        torch.nn.utils.clip_grad_norm_(gnn.parameters(), 5.0)
        opt.step()
        if step % 40 == 0 or step == args.steps - 1:
            parts = {n: eval_n(n, "val", 8) for n in (6, 9, 12)}
            score = 0.25 * parts[6]["S"] + 0.35 * parts[9]["S"] + 0.40 * parts[12]["S"]
            mark = ""
            if best is None or score < best["score"]:
                best = {
                    "score": score,
                    "step": step,
                    "parts": parts,
                    "state": {k: v.detach().cpu().clone() for k, v in gnn.state_dict().items()},
                }
                mark = " *"
            print(
                f"step {step} t={time.time()-t0:.0f}s "
                + " ".join(
                    f"n{n}={parts[n]['median_inv_p']:.1f}/{parts[n]['B90_inv_p']:.1f}"
                    for n in (6, 9, 12)
                )
                + f" S={score:.3f}{mark}",
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
        "n12_train": int(args.n12_train),
        "n12_batch": int(args.n12_batch) if args.n12_train > 0 else 0,
        "note": "one SizeGeneralGNN; L=16 BEST is written only if n=6 and n=9 test both improve",
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
    l16 = {"6": 11.0707426071167, "9": 8.794253826141357}
    t6 = report["by_n"]["6"]["test"]["median_inv_p"]
    t9 = report["by_n"]["9"]["test"]["median_inv_p"]
    beats = bool(t6 < l16["6"] and t9 < l16["9"])
    report["l16_test"] = l16
    report["beats_l16_test_both"] = beats
    report["best_overwritten"] = False
    print(
        f"n6/n9 test {t6:.2f}/{t9:.2f} vs L16 {l16['6']:.2f}/{l16['9']:.2f}; "
        f"both_improve={beats}. L16 BEST weights left in place.",
        flush=True,
    )
    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump(report, f, indent=2)
    print("Wrote", args.out, flush=True)


if __name__ == "__main__":
    main()
