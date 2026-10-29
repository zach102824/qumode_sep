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
from qtimes_cost.lib.gnn import FactorGNN, GNNArch, SizeGeneralGNN
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


def distill_cavity_head(gnn: SizeGeneralGNN, factor_ckpt: str, pack9: dict, steps: int = 40) -> float:
    """Fit only the shared cavity MLP so n=9 outputs match the L=16 FactorGNN head.

    The factor head reads a global pool. This matches its α0, displacement, and
    SNAP on n=9 formulas. It does not copy those tensors across cavities.
    """
    meta_path = os.path.join(os.path.dirname(factor_ckpt), "best.json")
    meta = json.load(open(meta_path))
    arch = GNNArch.from_dict(meta["arch"])
    factor = FactorGNN(arch, n=9, m=int(pack9["m"]), L=int(meta["cfg"]["L"]), seed=0)
    factor.load_state_dict(torch.load(factor_ckpt, map_location="cpu", weights_only=True))
    factor.eval()
    for p in factor.parameters():
        p.requires_grad_(False)
    for name, p in gnn.named_parameters():
        p.requires_grad_(name.startswith("cav_"))
    opt = torch.optim.Adam([p for p in gnn.parameters() if p.requires_grad], lr=1e-3)
    seeds, cls_v, cls_s, _nsol, _E = _split_arrays(pack9, "train", max_count=32)
    rng = np.random.default_rng(2)
    last = 0.0
    for _step in range(steps):
        opt.zero_grad(set_to_none=True)
        idx = rng.choice(len(seeds), size=4, replace=False)
        loss = None
        for b in idx:
            b = int(b)
            cv = torch.as_tensor(cls_v[b])
            cs = torch.as_tensor(cls_s[b], dtype=torch.float32)
            hs = gnn.forward_graph(cv, cs, n=9)
            with torch.no_grad():
                hf = factor.forward_graph(cv, cs)
            term = (
                (hs["alpha0"] - hf["alpha0"]).pow(2).mean()
                + (hs["A"] - hf["A"]).pow(2).mean()
                + (hs["TH"] - hf["TH"]).pow(2).mean()
            )
            loss = term if loss is None else loss + term
        loss = loss / 4
        loss.backward()
        opt.step()
        last = float(loss.detach())
    for p in gnn.parameters():
        p.requires_grad_(True)
    return last


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
    ap.add_argument("--n12-train", type=int, default=0, help="if >0, include this many n=12 train formulas")
    ap.add_argument("--n12-batch", type=int, default=1)
    ap.add_argument("--phase1", type=int, default=0, help="steps on n=6+n=9 only before n=12 is ramped in")
    ap.add_argument("--phase2", type=int, default=0, help="steps that ramp the n=12 loss weight")
    ap.add_argument("--leakpen12", type=float, default=0.0, help="n=12 leak penalty; 0 uses --leakpen")
    ap.add_argument("--n6-w", type=float, default=0.0, help="if >0 with --n9-w, override the mix")
    ap.add_argument("--n9-w", type=float, default=0.0)
    ap.add_argument("--n12-w", type=float, default=0.0)
    ap.add_argument("--n9-batch", type=int, default=2)
    ap.add_argument("--n9-train", type=int, default=64)
    ap.add_argument("--val-n9", type=int, default=16, help="n=9 val count used to pick the checkpoint")
    ap.add_argument("--cap-n6", type=float, default=0.0, help="keep checkpoints with n6 val median at or below this")
    ap.add_argument("--cap-n12", type=float, default=0.0, help="keep checkpoints with n12 val median at or below this")
    ap.add_argument("--n9-snap", action="store_true", help="train a zero-init SNAP residual used only at n=9")
    ap.add_argument("--snap-only", action="store_true", help="with --n9-snap, freeze every other parameter")
    ap.add_argument("--patience", type=int, default=0, help="stop after this many n=9 val checks without a new best")
    args = ap.parse_args(argv)
    torch.set_num_threads(1)
    arch = GNNArch(H=128, rounds=4, agg="mean", layernorm=True, residual=True, degree_features=True, lr=args.lr)
    gnn = SizeGeneralGNN(arch, L=args.L, seed=0)
    if args.n9_snap:
        gnn.enable_n9_snap()
    copied = []
    resume_path = os.path.join(args.out, "weights.pt")
    if args.resume and os.path.exists(resume_path):
        gnn.load_state_dict(torch.load(resume_path, map_location="cpu", weights_only=True), strict=False)
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
        if n == 12:
            cap = args.n12_train
        elif n == 9:
            cap = args.n9_train
        else:
            cap = 64
        train[n] = _split_arrays(packs[n], "train", max_count=cap)
    batch_for = {6: 2, 9: max(1, args.n9_batch), 12: max(1, args.n12_batch)}
    if args.snap_only:
        if not args.n9_snap:
            raise SystemExit("--snap-only requires --n9-snap")
        for name, p in gnn.named_parameters():
            p.requires_grad_(name.startswith("snap_n9"))
    opt = torch.optim.Adam([p for p in gnn.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.0 if args.snap_only else 1e-4)
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

    leak_for = {6: args.leakpen, 9: args.leakpen, 12: args.leakpen12 or args.leakpen}
    total_steps = (args.phase1 + args.phase2) if args.phase1 > 0 else args.steps
    anchor = None
    best = None
    stall = 0
    t0 = time.time()

    def mix_at(step):
        if args.n9_w > 0:
            w = {6: args.n6_w, 9: args.n9_w, 12: args.n12_w if 12 in train_ns else 0.0}
        elif args.phase1 <= 0:
            w = {6: 0.35, 9: 0.40, 12: 0.25 if 12 in train_ns else 0.0}
        elif step < args.phase1 or 12 not in train_ns:
            w = {6: 0.45, 9: 0.55, 12: 0.0}
        else:
            ramp = min(1.0, (step - args.phase1 + 1) / max(args.phase2, 1))
            w = {6: 0.42, 9: 0.48, 12: 0.05 + 0.20 * ramp}
        z = sum(w[n] for n in train_ns)
        return {n: w[n] / z for n in train_ns}

    if args.cap_n6 > 0 or args.cap_n12 > 0:
        parts0 = {n: eval_n(n, "val", {6: 16, 9: max(16, args.val_n9), 12: 12}[n]) for n in (6, 9, 12)}
        best = {
            "score": parts0[9]["median_inv_p"],
            "step": -1,
            "parts": parts0,
            "state": {k: v.detach().cpu().clone() for k, v in gnn.state_dict().items()},
        }
        print(
            "step -1 loaded "
            + " ".join(f"n{n}={parts0[n]['median_inv_p']:.1f}" for n in (6, 9, 12)),
            flush=True,
        )

    for step in range(total_steps):
        opt.zero_grad(set_to_none=True)
        mix_w = mix_at(step)
        per_n = []
        for n in train_ns:
            if mix_w[n] <= 0:
                continue
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
                losses.append(_instance_loss(out, leakpen=leak_for[n]))
            per_n.append(mix_w[n] * torch.stack(losses).mean())
        torch.stack(per_n).sum().backward()
        torch.nn.utils.clip_grad_norm_(gnn.parameters(), 5.0)
        opt.step()
        if step % 20 == 0 or step == total_steps - 1:
            counts = {6: 16, 9: max(16, args.val_n9), 12: 12}
            parts = {n: eval_n(n, "val", counts[n]) for n in (6, 9, 12)}
            caps_ok = (
                (args.cap_n6 <= 0 or parts[6]["median_inv_p"] <= args.cap_n6)
                and (args.cap_n12 <= 0 or parts[12]["median_inv_p"] <= args.cap_n12)
            )
            if args.phase1 > 0 and step < args.phase1:
                score = 0.45 * parts[6]["S"] + 0.55 * parts[9]["S"]
            else:
                score = 0.40 * parts[6]["S"] + 0.40 * parts[9]["S"] + 0.20 * parts[12]["S"]
            state = {k: v.detach().cpu().clone() for k, v in gnn.state_dict().items()}
            mark = ""
            if args.cap_n6 > 0 or args.cap_n12 > 0:
                better_n9 = best is None or parts[9]["median_inv_p"] < best["parts"][9]["median_inv_p"]
                if caps_ok and better_n9:
                    best = {"score": parts[9]["median_inv_p"], "step": step, "parts": parts, "state": state}
                    stall = 0
                    mark = " *"
                elif args.patience > 0:
                    stall += 1
            else:
                if args.phase1 > 0 and step < args.phase1 and (anchor is None or score < anchor["score"]):
                    anchor = {"score": score, "step": step, "parts": parts, "state": state}
                cap6 = 22.0 if anchor is None else max(18.0, 1.4 * anchor["parts"][6]["median_inv_p"])
                cap9 = 28.0 if anchor is None else max(22.0, 1.4 * anchor["parts"][9]["median_inv_p"])
                held = parts[6]["median_inv_p"] <= cap6 and parts[9]["median_inv_p"] <= cap9
                if held and (best is None or parts[12]["median_inv_p"] < best["parts"][12]["median_inv_p"]):
                    best = {"score": score, "step": step, "parts": parts, "state": state}
                    mark = " *"
                elif best is None:
                    best = {"score": score, "step": step, "parts": parts, "state": state}
                    mark = " *"
            print(
                f"step {step} t={time.time()-t0:.0f}s w12={mix_w.get(12, 0):.2f} "
                + " ".join(
                    f"n{n}={parts[n]['median_inv_p']:.1f}/{parts[n]['B90_inv_p']:.1f}"
                    for n in (6, 9, 12)
                )
                + f" S={score:.3f} cap={int(caps_ok)}{mark}",
                flush=True,
            )
            if args.patience > 0 and stall >= args.patience:
                print(f"early stop at step {step}: n9 val flat for {stall} checks", flush=True)
                break

    gnn.load_state_dict(best["state"])
    os.makedirs(args.out, exist_ok=True)
    torch.save(best["state"], os.path.join(args.out, "weights.pt"))
    report = {
        "L": args.L,
        "leakpen": args.leakpen,
        "steps": total_steps,
        "phase1": int(args.phase1),
        "phase2": int(args.phase2),
        "leakpen12": float(leak_for[12]),
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
