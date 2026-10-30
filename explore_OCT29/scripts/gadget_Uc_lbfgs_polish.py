#!/usr/bin/env python3
"""Opus #5 — multi-seed LBFGS polish on current projector-Uc champ (L32).

Honesty: still Uc=exp(-iγ Hc). Warm from unsat_params.pt (tail plain).
Soft pool {9,12,15} (n15=val). Pure LBFGS rounds with seed-varied minibatches
(+ optional tiny init noise).

Promote if n18 Med OR B90 ≤ 98% of champ AND n9/12 Med ≤ 105% of champ.
Writes runs/gadget_Uc/lbfgs_polish/
Stop: touch runs/STOP_LBFGS
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

_PKG = Path(__file__).resolve().parents[1]
_ROOT = _PKG.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from explore_OCT29.lib.drivers import fit_b  # noqa: E402
from explore_OCT29.scripts.gadget_Uc_explore import (  # noqa: E402
    _S,
    _jsonable,
    build_W_BinH,
    eval_arm,
    make_sim,
    scale1000_eval,
)
from explore_OCT29.scripts.gadget_Uc_soft_n15 import load_train_E  # noqa: E402

OUT = _PKG / "runs" / "gadget_Uc" / "lbfgs_polish"
CHAMP = _PKG / "runs" / "gadget_Uc" / "unsat_params.pt"
STOP = _PKG / "runs" / "STOP_LBFGS"
CHAMP_SCALE = {
    9: {"Med": 1.086, "B90": 1.799},
    12: {"Med": 1.192, "B90": 2.829},
    15: {"Med": 1.411, "B90": 5.520},
    18: {"Med": 1.752, "B90": 11.813},
}


def log(*a):
    print(time.strftime("%H:%M:%S"), "[lbfgs]", *a, flush=True)


def stopped() -> bool:
    return STOP.exists()


def plain_loss(invs: torch.Tensor, leaks: torch.Tensor, leakpen=0.1) -> torch.Tensor:
    x = invs.clamp(min=1.0)
    return torch.log(x).mean() + leakpen * leaks.mean()


def polish_seed(
    W, L, nf, device, seed, gamma0, phi0, *,
    noise=1e-3, lbfgs_rounds=8, closure_batch=6, max_train=48, max_val=16,
):
    """One multi-round LBFGS polish with seed-specific minibatch + tiny noise."""
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    g = gamma0.detach().clone()
    p = phi0.detach().clone()
    if noise > 0:
        g = g + noise * torch.randn_like(g)
        p = p + noise * torch.randn_like(p)
    gamma = torch.nn.Parameter(g)
    phi = torch.nn.Parameter(p)
    params = [gamma, phi]

    train_ns = (9, 12, 15)
    packs, sources = {}, {}
    for n in train_ns:
        E, src = load_train_E(n, max_train, device)
        packs[n] = E
        sources[n] = src
    sims = {n: make_sim(n, W, L, nf, device) for n in train_ns}

    best = {"score": float("inf"), "state": None, "val": None, "round": -1}

    def score_vals(vals):
        return 0.25 * vals[9]["S"] + 0.35 * vals[12]["S"] + 0.40 * vals[15]["S"]

    for rnd in range(lbfgs_rounds):
        if stopped():
            log("STOP_LBFGS"); break
        opt = torch.optim.LBFGS(
            params, lr=0.15, max_iter=6, history_size=12, line_search_fn="strong_wolfe"
        )

        def closure():
            opt.zero_grad()
            il, ll = [], []
            for n, Eall in packs.items():
                B = min(closure_batch, len(Eall))
                idxs = rng.choice(len(Eall), size=B, replace=False)
                for j in idxs:
                    o = sims[n].forward_gadget(Eall[int(j)], gamma, phi, "unsat")
                    il.append(o["inv_p"]); ll.append(o["p_leak"])
            lt = plain_loss(torch.stack(il), torch.stack(ll))
            lt.backward()
            return lt

        try:
            loss = opt.step(closure)
            loss_v = float(loss.detach()) if torch.is_tensor(loss) else float(loss)
        except Exception as e:
            log(f"seed{seed} round{rnd} LBFGS skip", e)
            loss_v = float("nan")

        vals = {
            n: eval_arm(
                n, W, gamma.detach(), phi.detach(), L, nf, device, "unsat",
                split="val", max_count=max_val,
            )
            for n in train_ns
        }
        sc = score_vals(vals)
        if sc < best["score"]:
            best = {
                "score": sc,
                "round": rnd,
                "val": vals,
                "state": {
                    "gamma": gamma.detach().cpu().clone(),
                    "phi": phi.detach().cpu().clone(),
                    "L": L,
                    "mode": "shell",
                    "cost_mode": "unsat",
                    "W": "BinH",
                    "train_ns": list(train_ns),
                    "sources": sources,
                    "loss_kind": "lbfgs_polish",
                    "seed": seed,
                    "noise": noise,
                },
            }
        bits = " ".join(
            f"n{n}:{vals[n]['median_inv_p']:.2f}/{vals[n]['B90_inv_p']:.2f}"
            for n in train_ns
        )
        log(f"seed{seed} rnd{rnd}: loss={loss_v:.4f} S={sc:.3f} {bits}")

    return best


def promote(sc):
    m18 = sc[18]
    b90_ok = m18["B90_inv_p"] <= CHAMP_SCALE[18]["B90"] * 0.98
    med_ok = m18["median_inv_p"] <= CHAMP_SCALE[18]["Med"] * 0.98
    improve = bool(b90_ok or med_ok)
    small_ok = (
        sc[9]["median_inv_p"] <= CHAMP_SCALE[9]["Med"] * 1.05
        and sc[12]["median_inv_p"] <= CHAMP_SCALE[12]["Med"] * 1.05
    )
    # n15 soft: allow slight drift ≤1.05
    n15_ok = sc[15]["median_inv_p"] <= CHAMP_SCALE[15]["Med"] * 1.05
    keep = bool(improve and small_ok and n15_ok)
    reasons = [
        f"n18 Med {m18['median_inv_p']:.3f} vs {CHAMP_SCALE[18]['Med']} (≤98%={CHAMP_SCALE[18]['Med']*0.98:.3f}) ok={med_ok}",
        f"n18 B90 {m18['B90_inv_p']:.3f} vs {CHAMP_SCALE[18]['B90']} (≤98%={CHAMP_SCALE[18]['B90']*0.98:.3f}) ok={b90_ok}",
        f"n9/12 Med ok={small_ok}; n15 ok={n15_ok}",
    ]
    return keep, reasons


def write_md(results):
    lines = [
        "# Multi-seed LBFGS polish — L32 projector champ",
        "",
        f"_Generated {time.strftime('%Y-%m-%d %H:%M:%S')} +0800 (Asia/Shanghai)_",
        "",
        "Honesty: still `Uc=exp(-iγ Hc)`. No blind stencil.",
        "",
        "## Champ baseline (tail plain)",
        "",
        "| n | Med | B90 |",
        "|---|-----|-----|",
    ]
    for n in (9, 12, 15, 18):
        lines.append(f"| {n} | {CHAMP_SCALE[n]['Med']} | {CHAMP_SCALE[n]['B90']} |")
    lines += [
        "",
        "## Seeds (val S + scale1000 if evaluated)",
        "",
        "| seed | val S | n9 | n12 | n15 | n18 | keep |",
        "|------|-------|----|-----|-----|-----|------|",
    ]
    for name, block in results.get("seeds", {}).items():
        sc = block.get("scale1000") or {}
        def cell(n):
            m = sc.get(n) or sc.get(str(n))
            if not m:
                return "—"
            return f"{m['median_inv_p']:.3f}/{m['B90_inv_p']:.3f}"
        lines.append(
            f"| {name} | {block.get('score', float('nan')):.3f} | "
            f"{cell(9)} | {cell(12)} | {cell(15)} | {cell(18)} | {block.get('keep')} |"
        )
    lines += ["", f"Winner: **{results.get('winner')}**", ""]
    if results.get("b_fits"):
        lines.append(
            f"b_Med≈{results['b_fits'].get('Med',{}).get('b')}, "
            f"b_B90≈{results['b_fits'].get('B90',{}).get('b')}"
        )
    (OUT / "RESULTS.md").write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=int, default=32)
    ap.add_argument("--nf", type=int, default=8)
    ap.add_argument("--seeds", default="0,1,2,3,4,5,6,7")
    ap.add_argument("--noise", type=float, default=1e-3)
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--closure_batch", type=int, default=6)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--scale_B", type=int, default=1000)
    ap.add_argument("--top_k_scale", type=int, default=3)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    W = build_W_BinH(nf=args.nf)
    st0 = torch.load(CHAMP, map_location="cpu", weights_only=False)
    g0, p0 = st0["gamma"].float(), st0["phi"].float()
    assert int(st0.get("L", g0.numel())) == args.L

    seed_ids = [int(x) for x in args.seeds.split(",") if x.strip()]
    results = {
        "champ": CHAMP_SCALE,
        "seeds": {},
        "honesty": "Uc=exp(-iγ Hc) unsat projector",
        "noise": args.noise,
        "rounds": args.rounds,
    }

    ranked = []
    for sid in seed_ids:
        if stopped():
            break
        log(f"=== seed {sid} ===")
        best = polish_seed(
            W, args.L, args.nf, device, sid, g0, p0,
            noise=args.noise, lbfgs_rounds=args.rounds,
            closure_batch=args.closure_batch,
        )
        st = best["state"]
        torch.save(st, OUT / f"seed{sid}_params.pt")
        results["seeds"][f"seed{sid}"] = {
            "score": best["score"],
            "round": best["round"],
            "val": best["val"],
        }
        ranked.append((best["score"], sid))
        (OUT / "results.json").write_text(json.dumps(_jsonable(results), indent=2))
        write_md(results)

    ranked.sort()
    top = [sid for _, sid in ranked[: args.top_k_scale]]
    log(f"top seeds for scale1000: {top}")

    winner = None
    best_key = None  # (b90, med) lower better
    for sid in top:
        if stopped():
            break
        st = torch.load(OUT / f"seed{sid}_params.pt", map_location="cpu", weights_only=False)
        sc = scale1000_eval(
            W, st["gamma"], st["phi"], args.L, args.nf, device, "unsat",
            ns=[9, 12, 15, 18], max_count=args.scale_B,
        )
        sc_i = {int(k): v for k, v in sc.items()}
        keep, reasons = promote(sc_i)
        results["seeds"][f"seed{sid}"].update({
            "scale1000": sc_i,
            "keep": keep,
            "reasons": reasons,
        })
        log(f"seed{sid} keep={keep}", reasons)
        (OUT / "results.json").write_text(json.dumps(_jsonable(results), indent=2))
        write_md(results)
        if keep:
            key = (sc_i[18]["B90_inv_p"], sc_i[18]["median_inv_p"])
            if best_key is None or key < best_key:
                best_key = key
                winner = f"seed{sid}"

    results["winner"] = winner
    if winner:
        import shutil
        shutil.copy(OUT / f"{winner}_params.pt", OUT / "promoted_params.pt")
        # also replace champ
        bak = _PKG / "runs" / "gadget_Uc" / "unsat_params_tail_plain_backup.pt"
        if not bak.exists():
            shutil.copy(CHAMP, bak)
        shutil.copy(OUT / f"{winner}_params.pt", CHAMP)
        sc = results["seeds"][winner]["scale1000"]
        ns = [9, 12, 15, 18]
        results["b_fits"] = {
            "Med": fit_b(ns, [sc[n]["median_inv_p"] for n in ns]),
            "B90": fit_b(ns, [sc[n]["B90_inv_p"] for n in ns]),
        }
        log(f"PROMOTE {winner}")
    else:
        log("no promote")
    (OUT / "results.json").write_text(json.dumps(_jsonable(results), indent=2))
    write_md(results)


if __name__ == "__main__":
    main()
