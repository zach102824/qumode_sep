#!/usr/bin/env python3
"""Short joint fine-tune that adds n=12. No cavity-repeat.

FactorGNN fixes C = n/3 in the readout Linear layers when the module is
built. The forward does not grow C from the graph, so a size-generalizing
head is not already there. This run builds a C=4 network, copies only the
n-independent body (and q, gamma) from the L=12 n=9 BEST, leaves α0 / SNAP /
displacement heads at their own init, and trains on n=6, n=9, and a few n=12
instances together.

  python -m qtimes_cost.scripts.finetune_n12 --steps 100
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
from qtimes_cost.lib.cloud_worker import copy_shared_body, eval_nets
from qtimes_cost.lib.gnn import FactorGNN, GNNArch
from qtimes_cost.lib.train_loop import PKG, _instance_loss, _split_arrays, load_pack


def _load_best(ckpt: str):
    meta = json.load(open(os.path.join(ckpt, "best.json")))
    arch = GNNArch.from_dict(meta["arch"])
    L = int(meta["cfg"]["L"])
    return arch, L, meta


def _build(arch, n, m, L, state=None, body_from=None):
    gnn = FactorGNN(arch, n=n, m=m, L=L, seed=100 + n)
    how = "fresh"
    if state is not None:
        gnn.load_state_dict(state)
        how = "exact"
    elif body_from is not None:
        copied = copy_shared_body(gnn, body_from)
        how = f"body_q_gamma_from_n9 ({len(copied)} tensors); cavity heads fresh"
    sim = MultiCavityAnsatz(AnsatzConfig(n=n, L=L))
    return gnn, sim, how


def _eval_split(gnn, sim, pack, n, split, max_count):
    arrays = {n: _split_arrays(pack, split, max_count=max_count)}
    metrics = eval_nets({n: gnn}, {n: sim}, arrays, max_count or len(arrays[n][0]))
    return {k: float(v) if isinstance(v, (int, float, np.floating)) else v for k, v in metrics[n].items()}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", type=str, default=str(PKG / "runs" / "cloud_day1" / "BEST_round1"))
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--lr12", type=float, default=1e-4)
    ap.add_argument("--lr69", type=float, default=2e-5)
    ap.add_argument("--leakpen", type=float, default=0.1)
    ap.add_argument("--n12-train", type=int, default=32)
    ap.add_argument("--batch12", type=int, default=2)
    ap.add_argument("--max-val", type=int, default=40)
    ap.add_argument("--max-test", type=int, default=100)
    ap.add_argument("--out", type=str, default=str(PKG / "runs" / "cloud_day1" / "n12_finetune"))
    ap.add_argument("--resume", action="store_true", help="continue from weights already in --out")
    ap.add_argument("--freeze-69", action="store_true", help="train only the n=12 net; n=6 and n=9 stay fixed")
    args = ap.parse_args(argv)
    torch.set_num_threads(1)
    arch, L, meta = _load_best(args.ckpt)
    if L != 12:
        raise SystemExit(f"expected L=12 parent, got L={L} from {args.ckpt}")

    packs = {n: load_pack(n) for n in (6, 9, 12)}
    state9 = torch.load(os.path.join(args.ckpt, "weights_n9.pt"), map_location="cpu", weights_only=True)
    nets, sims, how = {}, {}, {}
    for n in (6, 9):
        state = torch.load(os.path.join(args.ckpt, f"weights_n{n}.pt"), map_location="cpu", weights_only=True)
        nets[n], sims[n], how[n] = _build(arch, n, int(packs[n]["m"]), L, state=state)
    nets[12], sims[12], how[12] = _build(arch, 12, int(packs[12]["m"]), L, body_from=state9)
    if args.resume:
        for n in (6, 9, 12):
            path = os.path.join(args.out, f"weights_n{n}.pt")
            if os.path.exists(path):
                nets[n].load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
                how[n] = f"resume {path}"
    print("load", how, flush=True)

    train = {n: _split_arrays(packs[n], "train", max_count=128 if n < 12 else args.n12_train) for n in nets}
    val_small = {n: _split_arrays(packs[n], "val", max_count=8) for n in nets}
    opt12 = torch.optim.Adam(nets[12].parameters(), lr=args.lr12, weight_decay=1e-4)
    opt69 = torch.optim.Adam(
        [p for n in (6, 9) for p in nets[n].parameters()], lr=args.lr69, weight_decay=1e-4,
    )
    rng = np.random.default_rng(0)
    best = None
    t0 = time.time()

    def score12():
        gnn, sim = nets[12], sims[12]
        gnn.eval()
        metrics = eval_nets(
            {12: gnn}, {12: sim},
            {12: val_small[12]}, 8,
        )[12]
        gnn.train()
        return float(metrics["median_inv_p"]), float(metrics["B90_inv_p"]), float(metrics["mean_leak"]), float(metrics["S"])

    for step in range(args.steps):
        opt12.zero_grad(set_to_none=True)
        opt69.zero_grad(set_to_none=True)
        losses = []
        plan = [(12, args.batch12), (6, 1), (9, 1)]
        for n, bs in plan:
            gnn, sim = nets[n], sims[n]
            gnn.train()
            seeds, cls_v, cls_s, _nsol, E = train[n]
            idx = rng.choice(len(seeds), size=min(bs, len(seeds)), replace=False)
            for b in idx:
                b = int(b)
                heads = gnn.forward_graph(
                    torch.as_tensor(cls_v[b]),
                    torch.as_tensor(cls_s[b], dtype=torch.float32),
                )
                out = sim.forward(
                    torch.as_tensor(E[b], dtype=torch.float32),
                    heads["alpha0"], heads["A"], heads["TH"], heads["q_logits"],
                    gamma=heads["gamma"],
                )
                w = args.leakpen if n == 12 else min(args.leakpen, 0.1)
                losses.append(_instance_loss(out, leakpen=w))
        loss = torch.stack(losses).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(nets[12].parameters(), 5.0)
        if not args.freeze_69:
            torch.nn.utils.clip_grad_norm_([p for n in (6, 9) for p in nets[n].parameters()], 5.0)
        opt12.step()
        if not args.freeze_69:
            opt69.step()
        if step % 20 == 0 or step == args.steps - 1:
            med, b90, leak, S = score12()
            mark = ""
            if best is None or S < best["S"]:
                best = {
                    "step": step, "S": S, "med": med, "b90": b90, "leak": leak,
                    "state": {k: v.detach().cpu().clone() for k, v in nets[12].state_dict().items()},
                }
                mark = " *"
            print(
                f"step {step} t={time.time()-t0:.0f}s n12 val8 Med={med:.2f} B90={b90:.2f} leak={leak:.4f}{mark}",
                flush=True,
            )

    nets[12].load_state_dict(best["state"])
    out_dir = args.out
    os.makedirs(out_dir, exist_ok=True)
    torch.save(best["state"], os.path.join(out_dir, "weights_n12.pt"))
    for n in (6, 9):
        torch.save(
            {k: v.detach().cpu() for k, v in nets[n].state_dict().items()},
            os.path.join(out_dir, f"weights_n{n}.pt"),
        )
    report = {"parent": args.ckpt, "L": L, "how": how, "best_search": {k: v for k, v in best.items() if k != "state"}, "by_n": {}}
    for n in (6, 9, 12):
        report["by_n"][str(n)] = {}
        for split, count in (("val", args.max_val), ("test", args.max_test)):
            m = _eval_split(nets[n], sims[n], packs[n], n, split, count)
            report["by_n"][str(n)][split] = m
            print(
                f"n={n} {split} B={m['B']} Med={m['median_inv_p']:.3f} "
                f"B90={m['B90_inv_p']:.3f} leak={m['mean_leak']:.4f}",
                flush=True,
            )
    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(report, f, indent=2)
    print("Wrote", out_dir, flush=True)


if __name__ == "__main__":
    main()
