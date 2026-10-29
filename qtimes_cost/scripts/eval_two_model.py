#!/usr/bin/env python3
"""Two-model path: L=16 BEST for n<=9, SizeGeneral checkpoint for n>=12.

Does not write BEST/.
"""
from __future__ import annotations

import json
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")

import torch

from qtimes_cost.lib.ansatz import AnsatzConfig, MultiCavityAnsatz
from qtimes_cost.lib.gnn import FactorGNN, GNNArch, SizeGeneralGNN
from qtimes_cost.lib.train_loop import PKG, _split_arrays, load_pack
import numpy as np


def _eval_factor(ckpt, n, splits, max_count):
    meta = json.load(open(os.path.join(ckpt, "best.json")))
    arch = GNNArch.from_dict(meta["arch"])
    L = int(meta["cfg"]["L"])
    pack = load_pack(n)
    gnn = FactorGNN(arch, n=n, m=int(pack["m"]), L=L, seed=0)
    gnn.load_state_dict(torch.load(os.path.join(ckpt, f"weights_n{n}.pt"), map_location="cpu", weights_only=True))
    gnn.eval()
    sim = MultiCavityAnsatz(AnsatzConfig(n=n, L=L))
    return L, _splits(gnn, sim, pack, n, splits, max_count, size_general=False)


def _eval_size(weights, L, n, splits, max_count):
    arch = GNNArch(H=128, rounds=4, agg="mean", layernorm=True, residual=True, degree_features=True)
    gnn = SizeGeneralGNN(arch, L=L, seed=0)
    gnn.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True), strict=False)
    gnn.eval()
    pack = load_pack(n)
    sim = MultiCavityAnsatz(AnsatzConfig(n=n, L=L))
    return _splits(gnn, sim, pack, n, splits, max_count, size_general=True)


def _splits(gnn, sim, pack, n, splits, max_count, size_general):
    out = {}
    for split in splits:
        seeds, cls_v, cls_s, _nsol, E = _split_arrays(pack, split, max_count=max_count)
        inv, leak = [], []
        with torch.inference_mode():
            for b in range(len(seeds)):
                cv = torch.as_tensor(cls_v[b])
                cs = torch.as_tensor(cls_s[b], dtype=torch.float32)
                if size_general:
                    heads = gnn.forward_graph(cv, cs, n=n)
                else:
                    heads = gnn.forward_graph(cv, cs)
                o = sim.forward(
                    torch.as_tensor(E[b], dtype=torch.float32),
                    heads["alpha0"], heads["A"], heads["TH"], heads["q_logits"],
                    gamma=heads["gamma"],
                )
                inv.append(float(o["inv_p"]))
                leak.append(float(o["p_leak"]))
        inv_a = np.asarray(inv, float)
        med = float(np.median(inv_a))
        b90 = float(np.quantile(inv_a, 0.9))
        out[split] = {
            "B": int(len(inv_a)),
            "median_inv_p": med,
            "B90_inv_p": b90,
            "mean_leak": float(np.mean(leak)),
        }
        print(f"n={n} {split} Med={med:.3f} B90={b90:.3f} leak={out[split]['mean_leak']:.4f}", flush=True)
    return out


def main():
    torch.set_num_threads(1)
    small = os.path.join(str(PKG), "runs", "cloud_day1", "BEST")
    scale_w = os.path.join(str(PKG), "runs", "cloud_day1", "BEST_pareto", "weights_n12_L24.pt")
    L6, m6 = _eval_factor(small, 6, ("val", "test"), None)
    L9, m9 = _eval_factor(small, 9, ("val", "test"), None)
    m12 = _eval_size(scale_w, 24, 12, ("val", "test"), None)
    payload = {
        "rule": "n<=9 uses L=16 FactorGNN BEST; n>=12 uses SizeGeneral weights in BEST_pareto",
        "best_untouched": True,
        "n6": {"model": "L16_BEST", "L": L6, "splits": m6},
        "n9": {"model": "L16_BEST", "L": L9, "splits": m9},
        "n12": {"model": "SizeGeneral_L24_scale", "L": 24, "splits": m12},
    }
    out = os.path.join(str(PKG), "runs", "cloud_day1", "two_model.json")
    with open(out, "w") as f:
        json.dump(payload, f, indent=2)
    print("Wrote", out)


if __name__ == "__main__":
    main()
