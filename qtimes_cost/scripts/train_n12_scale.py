#!/usr/bin/env python3
"""n>=12 scale training for SizeGeneralGNN. Does not write BEST/.

More n=12 formulas, an L sweep warm-started from the Pareto checkpoint,
and checkpoint choice by median(1/p) / WalkSAT flips on the same val formulas.
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
from qtimes_cost.lib.packing import NU
from qtimes_cost.lib.train_loop import PKG, _instance_loss, _split_arrays, load_pack
from qtimes_cost.lib.walksat import walksat_instance


def _grow_rows(src: torch.Tensor, dst_len: int) -> torch.Tensor:
    src_len = src.shape[0]
    dst = src.new_zeros((dst_len,) + tuple(src.shape[1:]))
    n = min(src_len, dst_len)
    dst[:n] = src[:n]
    if dst_len > src_len:
        dst[src_len:] = src[-1]
    return dst


def _split_cav_out(t: torch.Tensor, L: int):
    """Leading axis is 2 + 2L + 8L."""
    a = t[:2]
    A = t[2:2 + 2 * L].reshape(L, 2, *t.shape[1:])
    TH = t[2 + 2 * L:].reshape(L, NU, *t.shape[1:])
    return a, A, TH


def _pack_cav_out(a, A, TH):
    return torch.cat([a, A.reshape(-1, *a.shape[1:]), TH.reshape(-1, *a.shape[1:])], dim=0)


def transplant_L(src: dict, src_L: int, gnn: SizeGeneralGNN) -> None:
    """Copy a SizeGeneral checkpoint across circuit depth. New layers repeat the last."""
    dst_L = gnn.L
    merged = {k: v.detach().clone() for k, v in gnn.state_dict().items()}
    for key, dst in list(merged.items()):
        if key not in src or key.startswith("snap_"):
            continue
        parent = src[key].detach().to(dtype=dst.dtype)
        if tuple(parent.shape) == tuple(dst.shape):
            merged[key] = parent
            continue
        if key in ("head_q.weight", "head_q.bias"):
            merged[key] = _grow_rows(parent, dst_L)
            continue
        if key in ("cav_out.weight", "cav_out.bias"):
            a, A, TH = _split_cav_out(parent, src_L)
            A2 = _grow_rows(A, dst_L)
            TH2 = _grow_rows(TH, dst_L)
            merged[key] = _pack_cav_out(a, A2, TH2)
    gnn.load_state_dict(merged, strict=False)


def ws_medians(pack, n: int, count: int, R: int = 8) -> np.ndarray:
    seeds, cls_v, cls_s, _nsol, _E = _split_arrays(pack, "val", max_count=count)
    m = cls_v.shape[1]
    out = []
    for i in range(len(seeds)):
        cl = [(cls_v[i, c].tolist(), cls_s[i, c].tolist()) for c in range(m)]
        res = walksat_instance(n, cl, R=R, p=0.5, seed=int(seeds[i]))
        out.append(max(res["flips"]["median"], 1.0))
    return np.asarray(out, float)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--L", type=int, required=True)
    ap.add_argument("--src", type=str, default=str(PKG / "runs" / "cloud_day1" / "BEST_pareto" / "weights.pt"))
    ap.add_argument("--src-L", type=int, default=16)
    ap.add_argument("--steps", type=int, default=180)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--leakpen", type=float, default=0.2)
    ap.add_argument("--n12-train", type=int, default=256)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--out", type=str, required=True)
    args = ap.parse_args(argv)
    torch.set_num_threads(1)
    arch = GNNArch(H=128, rounds=4, agg="mean", layernorm=True, residual=True, degree_features=True, lr=args.lr)
    gnn = SizeGeneralGNN(arch, L=args.L, seed=0)
    src = torch.load(args.src, map_location="cpu", weights_only=True)
    transplant_L(src, args.src_L, gnn)
    pack = load_pack(12)
    sim = MultiCavityAnsatz(AnsatzConfig(n=12, L=args.L))
    train = _split_arrays(pack, "train", max_count=args.n12_train)
    val = _split_arrays(pack, "val", max_count=24)
    print("WalkSAT on 24 val formulas...", flush=True)
    ws = ws_medians(pack, 12, 24, R=8)
    opt = torch.optim.Adam(gnn.parameters(), lr=args.lr, weight_decay=1e-4)
    rng = np.random.default_rng(args.L)

    def eval_val():
        gnn.eval()
        seeds, cls_v, cls_s, _nsol, E = val
        inv, leak = [], []
        with torch.inference_mode():
            for b in range(len(seeds)):
                h = gnn.forward_graph(torch.as_tensor(cls_v[b]), torch.as_tensor(cls_s[b], dtype=torch.float32), n=12)
                o = sim.forward(torch.as_tensor(E[b], dtype=torch.float32), h["alpha0"], h["A"], h["TH"], h["q_logits"], gamma=h["gamma"])
                inv.append(float(o["inv_p"]))
                leak.append(float(o["p_leak"]))
        gnn.train()
        inv_a = np.asarray(inv, float)
        ratio = inv_a / ws[:len(inv_a)]
        return {
            "median_inv_p": float(np.median(inv_a)),
            "B90_inv_p": float(np.quantile(inv_a, 0.9)),
            "mean_leak": float(np.mean(leak)),
            "median_over_ws": float(np.median(ratio)),
        }

    best_raw = None
    best_ws = None
    stall_raw = 0
    t0 = time.time()
    seeds, cls_v, cls_s, _nsol, E = train
    for step in range(args.steps):
        opt.zero_grad(set_to_none=True)
        idx = rng.choice(len(seeds), size=min(args.batch, len(seeds)), replace=False)
        losses = []
        for b in idx:
            b = int(b)
            h = gnn.forward_graph(torch.as_tensor(cls_v[b]), torch.as_tensor(cls_s[b], dtype=torch.float32), n=12)
            o = sim.forward(torch.as_tensor(E[b], dtype=torch.float32), h["alpha0"], h["A"], h["TH"], h["q_logits"], gamma=h["gamma"])
            losses.append(_instance_loss(o, leakpen=args.leakpen))
        torch.stack(losses).mean().backward()
        torch.nn.utils.clip_grad_norm_(gnn.parameters(), 5.0)
        opt.step()
        if step % 20 == 0 or step == args.steps - 1:
            m = eval_val()
            state = {k: v.detach().cpu().clone() for k, v in gnn.state_dict().items()}
            mark = ""
            if best_raw is None or m["median_inv_p"] < best_raw["median_inv_p"]:
                best_raw = {**m, "step": step, "state": state}
                stall_raw = 0
                mark += " raw*"
            else:
                stall_raw += 1
            if best_ws is None or m["median_over_ws"] < best_ws["median_over_ws"]:
                best_ws = {**m, "step": step, "state": state}
                mark += " ws*"
            print(
                f"L={args.L} step {step} t={time.time()-t0:.0f}s "
                f"med={m['median_inv_p']:.1f} B90={m['B90_inv_p']:.1f} "
                f"med/ws={m['median_over_ws']:.2f} leak={m['mean_leak']:.3f}{mark}",
                flush=True,
            )
            if stall_raw >= args.patience and step > 80:
                print(f"early stop L={args.L} at step {step}", flush=True)
                break

    os.makedirs(args.out, exist_ok=True)

    def full_eval(state, tag):
        gnn.load_state_dict(state)
        gnn.eval()
        test = _split_arrays(pack, "test", max_count=100)
        val40 = _split_arrays(pack, "val", max_count=40)
        blocks = {}
        for name, split in (("val", val40), ("test", test)):
            seeds_s, cv, cs, _n, Es = split
            inv, leak = [], []
            with torch.inference_mode():
                for b in range(len(seeds_s)):
                    h = gnn.forward_graph(torch.as_tensor(cv[b]), torch.as_tensor(cs[b], dtype=torch.float32), n=12)
                    o = sim.forward(torch.as_tensor(Es[b], dtype=torch.float32), h["alpha0"], h["A"], h["TH"], h["q_logits"], gamma=h["gamma"])
                    inv.append(float(o["inv_p"]))
                    leak.append(float(o["p_leak"]))
            inv_a = np.asarray(inv, float)
            blocks[name] = {
                "B": int(len(inv_a)),
                "median_inv_p": float(np.median(inv_a)),
                "B90_inv_p": float(np.quantile(inv_a, 0.9)),
                "mean_leak": float(np.mean(leak)),
            }
            print(
                f"L={args.L} {tag} {name} Med={blocks[name]['median_inv_p']:.2f} "
                f"B90={blocks[name]['B90_inv_p']:.2f} leak={blocks[name]['mean_leak']:.4f}",
                flush=True,
            )
        path = os.path.join(args.out, f"weights_{tag}.pt")
        torch.save(state, path)
        return blocks

    report = {
        "L": args.L,
        "n12_train": args.n12_train,
        "leakpen": args.leakpen,
        "lr": args.lr,
        "raw": {"step": best_raw["step"], "val24": {k: best_raw[k] for k in ("median_inv_p", "B90_inv_p", "median_over_ws", "mean_leak")}, "splits": full_eval(best_raw["state"], "raw")},
        "ws": {"step": best_ws["step"], "val24": {k: best_ws[k] for k in ("median_inv_p", "B90_inv_p", "median_over_ws", "mean_leak")}, "splits": full_eval(best_ws["state"], "ws")},
        "best_untouched": True,
    }
    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump(report, f, indent=2)
    print("Wrote", args.out, flush=True)


if __name__ == "__main__":
    main()
