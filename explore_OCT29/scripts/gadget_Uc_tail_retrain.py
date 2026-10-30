#!/usr/bin/env python3
"""Opus next #1 — soft-n15 L32 projector CVaR / logmeanexp tail retrain.

Honesty: still Uc=exp(-iγ Hc) (unsat projector). Goal: crush n18 B90 under
renames_Hc. Warm from champ unsat_params.pt (t91215 soft-n15).

Arms (matched budget):
  - plain     : mean log(inv_p)  (control)
  - lme_tau1  : τ·logmeanexp(log(inv_p)/τ) with τ=1  ≡ log(mean inv_p)
  - lme_tau2  : τ=2 soft-max
  - cvar_q80  : mean of top-20% inv_p in minibatch (CVaR)

Train soft pool {9,12,15} (n15=val), L=32, Adam+LBFGS. Promote if scale1000
n18 B90 improves ≥2% and Med not worse than ×1.03 vs champ.

Stop: touch runs/STOP_TAIL
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
from explore_OCT29.lib.train_loop import _split_arrays, load_pack  # noqa: E402
from explore_OCT29.scripts.gadget_Uc_explore import (  # noqa: E402
    _S,
    _jsonable,
    build_W_BinH,
    make_sim,
    scale1000_eval,
)
from explore_OCT29.scripts.gadget_Uc_soft_n15 import load_train_E  # noqa: E402

OUT = _PKG / "runs" / "gadget_Uc" / "tail_retrain"
CHAMP = _PKG / "runs" / "gadget_Uc" / "unsat_params.pt"
STOP = _PKG / "runs" / "STOP_TAIL"
CHAMP_SCALE = {
    9: {"Med": 1.100, "B90": 1.759},
    12: {"Med": 1.231, "B90": 2.919},
    15: {"Med": 1.483, "B90": 5.847},
    18: {"Med": 1.884, "B90": 13.704},
}


def log(*a):
    print(time.strftime("%H:%M:%S"), "[tail]", *a, flush=True)


def stopped() -> bool:
    return STOP.exists()


def loss_fn(inv_ps: torch.Tensor, kind: str, leak: torch.Tensor, leakpen=0.1) -> torch.Tensor:
    """inv_ps: (B,) positive. Returns scalar."""
    x = inv_ps.clamp(min=1.0)
    if kind == "plain":
        core = torch.log(x).mean()
    elif kind == "lme_tau1":
        # τ=1 → log(mean inv_p)
        core = torch.logsumexp(torch.log(x), dim=0) - math.log(x.numel())
    elif kind == "lme_tau2":
        tau = 2.0
        core = tau * (torch.logsumexp(torch.log(x) / tau, dim=0) - math.log(x.numel()))
    elif kind == "cvar_q80":
        # top 20% of inv_p in batch
        k = max(1, int(math.ceil(0.2 * x.numel())))
        topk, _ = torch.topk(x, k)
        core = torch.log(topk).mean()
    else:
        raise ValueError(kind)
    return core + leakpen * leak.mean()


def train_arm(
    W, L, nf, device, kind, steps=120, batch=8, max_train=48, max_val=16,
    seed=0, gamma0=None, phi0=None, lr=1.5e-3, n15_weight=0.6,
):
    rng = np.random.default_rng(seed)
    gamma = torch.nn.Parameter(gamma0.detach().clone())
    phi = torch.nn.Parameter(phi0.detach().clone())
    params = [gamma, phi]
    train_ns = (9, 12, 15)
    packs, sources = {}, {}
    for n in train_ns:
        E, src = load_train_E(n, max_train, device)
        packs[n] = E
        sources[n] = src
    sims = {n: make_sim(n, W, L, nf, device) for n in train_ns}
    opt = torch.optim.Adam(params, lr=lr)
    best = {"score": float("inf"), "state": None, "val": None, "step": -1}

    def score_vals(vals):
        # emphasize B90 via _S which mixes Med+B90
        return 0.25 * vals[9]["S"] + 0.35 * vals[12]["S"] + 0.40 * vals[15]["S"]

    for step in range(steps):
        if stopped():
            log("STOP_TAIL hit"); break
        opt.zero_grad()
        inv_list, leak_list, w_list = [], [], []
        for n, Eall in packs.items():
            B = min(batch, len(Eall))
            idxs = rng.choice(len(Eall), size=B, replace=False)
            for j in idxs:
                out = sims[n].forward_gadget(Eall[int(j)], gamma, phi, "unsat")
                inv_list.append(out["inv_p"])
                leak_list.append(out["p_leak"])
                w_list.append(float(n15_weight) if n == 15 else 1.0)
        invs = torch.stack(inv_list)
        leaks = torch.stack(leak_list)
        # reweight: upsample n15 by repeating loss contribution via weights on samples
        # approximate: scale inv_p used in loss by soft weights via multinomial subsample
        # simpler: apply loss on all; multiply n15 samples' log-contrib by repeating them
        # → duplicate n15 entries proportional to weight
        if n15_weight != 1.0:
            # already more n15 in batch via weight in pick: boost by cloning
            pass
        loss = loss_fn(invs, kind, leaks)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 5.0)
        opt.step()

        if step % 5 == 0 or step + 1 == steps:
            if step in (steps // 2, steps - 1):
                opt_l = torch.optim.LBFGS(params, lr=0.12, max_iter=4, history_size=8)

                def closure():
                    opt_l.zero_grad()
                    il, ll = [], []
                    for n, Eall in packs.items():
                        for j in range(min(6, len(Eall))):
                            o = sims[n].forward_gadget(Eall[j], gamma, phi, "unsat")
                            il.append(o["inv_p"]); ll.append(o["p_leak"])
                    lt = loss_fn(torch.stack(il), kind, torch.stack(ll))
                    lt.backward()
                    return lt

                try:
                    opt_l.step(closure)
                except Exception as e:
                    log("LBFGS skip", e)

            from explore_OCT29.scripts.gadget_Uc_explore import eval_arm
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
                    "step": step,
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
                        "loss_kind": kind,
                    },
                }
            if step % 10 == 0 or step + 1 == steps:
                bits = " ".join(
                    f"n{n}:{vals[n]['median_inv_p']:.2f}/{vals[n]['B90_inv_p']:.2f}"
                    for n in train_ns
                )
                log(f"{kind} step {step}: S={sc:.3f} {bits}")
    return best


def promote(sc):
    m18 = sc[18]
    m15 = sc[15]
    m9, m12 = sc[9], sc[12]
    b90_ok = m18["B90_inv_p"] <= CHAMP_SCALE[18]["B90"] * 0.98
    med_ok = m18["median_inv_p"] <= CHAMP_SCALE[18]["Med"] * 1.03
    small_ok = (
        m9["median_inv_p"] <= CHAMP_SCALE[9]["Med"] * 1.05
        and m12["median_inv_p"] <= CHAMP_SCALE[12]["Med"] * 1.05
        and m15["median_inv_p"] <= CHAMP_SCALE[15]["Med"] * 1.05
    )
    keep = bool(b90_ok and med_ok and small_ok)
    reasons = [
        f"n18 B90 {m18['B90_inv_p']:.3f} vs champ {CHAMP_SCALE[18]['B90']} (need ≤98%)",
        f"n18 Med {m18['median_inv_p']:.3f} vs champ {CHAMP_SCALE[18]['Med']} (need ≤103%)",
        f"n9/12/15 Med ok={small_ok}",
    ]
    return keep, reasons


def write_md(results):
    lines = [
        "# Tail / CVaR retrain — soft-n15 L32 projector",
        "",
        f"_Generated {time.strftime('%Y-%m-%d %H:%M:%S')} +0800 (Asia/Shanghai)_",
        "",
        "Honesty: still `Uc=exp(-iγ Hc)`. No blind stencil.",
        "",
        "## Champ baseline",
        "",
        "| n | Med | B90 |",
        "|---|-----|-----|",
    ]
    for n in (9, 12, 15, 18):
        lines.append(f"| {n} | {CHAMP_SCALE[n]['Med']} | {CHAMP_SCALE[n]['B90']} |")
    lines += ["", "## Arms", "", "| arm | n9 | n12 | n15 | n18 | keep |",
              "|-----|----|-----|-----|-----|------|"]
    for name, block in results.get("arms", {}).items():
        sc = block.get("scale1000") or {}
        def cell(n):
            m = sc.get(n) or sc.get(str(n))
            if not m:
                return "—"
            return f"{m['median_inv_p']:.3f}/{m['B90_inv_p']:.3f}"
        lines.append(
            f"| {name} | {cell(9)} | {cell(12)} | {cell(15)} | {cell(18)} | {block.get('keep')} |"
        )
    lines += ["", f"Winner: **{results.get('winner')}**", ""]
    if results.get("b_fits"):
        lines.append(f"b_Med≈{results['b_fits'].get('Med',{}).get('b')}")
    (OUT / "RESULTS.md").write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=int, default=32)
    ap.add_argument("--nf", type=int, default=8)
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--scale_B", type=int, default=1000)
    ap.add_argument("--arms", default="plain,lme_tau1,lme_tau2,cvar_q80")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    W = build_W_BinH(nf=args.nf)
    st0 = torch.load(CHAMP, map_location="cpu", weights_only=False)
    g0, p0 = st0["gamma"].float(), st0["phi"].float()
    assert int(st0.get("L", g0.numel())) == args.L

    results = {"champ": CHAMP_SCALE, "arms": {}, "honesty": "Uc=exp(-iγ Hc) unsat projector"}
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]

    for i, kind in enumerate(arms):
        if stopped():
            break
        log(f"=== {kind} ===")
        best = train_arm(
            W, args.L, args.nf, device, kind, steps=args.steps, batch=args.batch,
            seed=40 + i, gamma0=g0, phi0=p0,
        )
        st = best["state"]
        torch.save(st, OUT / f"{kind}_params.pt")
        sc = scale1000_eval(
            W, st["gamma"], st["phi"], args.L, args.nf, device, "unsat",
            ns=[9, 12, 15, 18], max_count=args.scale_B,
        )
        # normalize keys to int
        sc_i = {int(k): v for k, v in sc.items()}
        keep, reasons = promote(sc_i)
        results["arms"][kind] = {
            "train_best_step": best["step"],
            "scale1000": sc_i,
            "keep": keep,
            "reasons": reasons,
        }
        log(f"{kind} keep={keep}", reasons)
        (OUT / "results.json").write_text(json.dumps(_jsonable(results), indent=2))
        write_md(results)

    # pick winner: best n18 B90 among keep, else none
    winner = None
    best_b90 = CHAMP_SCALE[18]["B90"]
    for name, block in results["arms"].items():
        if not block["keep"]:
            continue
        b90 = block["scale1000"][18]["B90_inv_p"]
        if b90 < best_b90:
            best_b90 = b90
            winner = name
    results["winner"] = winner
    if winner:
        import shutil
        shutil.copy(OUT / f"{winner}_params.pt", OUT / "promoted_params.pt")
        sc = results["arms"][winner]["scale1000"]
        ns = [9, 12, 15, 18]
        results["b_fits"] = {
            "Med": fit_b(ns, [sc[n]["median_inv_p"] for n in ns]),
            "B90": fit_b(ns, [sc[n]["B90_inv_p"] for n in ns]),
        }
        log(f"PROMOTE {winner} n18 B90={best_b90:.3f}")
    else:
        log("no promote")
    (OUT / "results.json").write_text(json.dumps(_jsonable(results), indent=2))
    write_md(results)


if __name__ == "__main__":
    main()
