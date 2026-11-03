#!/usr/bin/env python3
"""Idea C: warm-start from pareto champ; freeze γ,c1; polish c2,c3 with med_align in delta-ball around champ (NOT QAOA-origin ball)."""
from __future__ import annotations
import argparse, json, math, os, sys, time
from pathlib import Path
import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from Nov_3rd import settings as S
from Nov_3rd.ansatz import make_sim
from Nov_3rd.data_io import load_qaoa, log
from beat_params import phi_from_c
from train_beat_train69 import (
    EVAL_NS, OUT_ROOT, TRAIN_NS, batch_loss, eval_state, holdout_score,
    load_train_n, precompute_qaoa_inv, snapshot_params, write_eval,
)

CHAMP = OUT_ROOT / "champion" / "params_L2_train69_pareto_a080_s1.pt"


def project_delta_ball(c2, c3, c2_0, c3_0, rho: float):
    """Project (c2,c3) onto ℓ2 ball of radius rho centered at (c2_0,c3_0)."""
    with torch.no_grad():
        d2 = c2 - c2_0
        d3 = c3 - c3_0
        v = torch.cat([d2.reshape(-1), d3.reshape(-1)])
        nrm = float(v.norm().item())
        if nrm > rho and nrm > 0:
            scale = rho / nrm
            c2.copy_(c2_0 + d2 * scale)
            c3.copy_(c3_0 + d3 * scale)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--init", type=str, default=str(CHAMP))
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--rho", type=float, default=0.03)
    ap.add_argument("--adam_steps", type=int, default=200)
    ap.add_argument("--adam_lr", type=float, default=3e-4)
    ap.add_argument("--seeds", type=str, default="0,1,2,3,4,5")
    ap.add_argument("--soft_q", type=float, default=0.5)
    ap.add_argument("--w6", type=float, default=1.0)
    ap.add_argument("--w9", type=float, default=2.0)
    ap.add_argument("--batch_per_n", type=int, default=48)
    ap.add_argument("--holdout_frac", type=float, default=0.25)
    args = ap.parse_args()

    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    torch.set_num_threads(args.threads)
    device = torch.device("cpu")
    L = 2
    idea = "trust69"
    weights = {6: args.w6, 9: args.w9}
    out_dir = OUT_ROOT / "trust69" / f"med_align_delta_rho{args.rho:g}_q{args.soft_q:g}"
    out_dir.mkdir(parents=True, exist_ok=True)

    init = torch.load(args.init, map_location="cpu", weights_only=False)
    betas, gammas = load_qaoa(L)
    bundles = {}
    for n in TRAIN_NS:
        E = load_train_n(n).to(device)
        inv_q = precompute_qaoa_inv(betas, gammas, E.cpu(), n, device)
        sim = make_sim(n, L, device)
        bundles[n] = (E, inv_q.to(device), sim)
    qmed_map = {n: float(bundles[n][1].detach().median().cpu()) for n in bundles}
    log(f"warm init={args.init} rho={args.rho} qmed={qmed_map}")

    best_all = None
    for seed in [int(x) for x in args.seeds.split(",") if x.strip()]:
        torch.manual_seed(seed); np.random.seed(seed)
        rng = np.random.default_rng(seed)
        gamma = init["gamma"].to(device).detach().clone().requires_grad_(False)
        c1 = init["c1"].to(device).detach().clone().requires_grad_(False)
        # perturb c2,c3 inside ball
        c2_0 = init["c2"].to(device).detach().clone()
        c3_0 = init["c3"].to(device).detach().clone()
        c2 = c2_0.clone()
        c3 = c3_0.clone()
        if seed > 0:
            with torch.no_grad():
                noise = torch.randn(c2.numel() + c3.numel())
                noise = noise / (noise.norm() + 1e-12) * (args.rho * float(rng.uniform(0.2, 0.9)))
                c2 = c2 + noise[: c2.numel()].reshape_as(c2).to(device)
                c3 = c3 + noise[c2.numel() :].reshape_as(c3).to(device)
                project_delta_ball(c2, c3, c2_0, c3_0, args.rho)
        c2 = c2.detach().requires_grad_(True)
        c3 = c3.detach().requires_grad_(True)
        p = None
        trainables = [c2, c3]
        n9_B = len(bundles[9][0])
        perm = rng.permutation(n9_B)
        n_hold = max(1, int(round(args.holdout_frac * n9_B)))
        hold_idx, pool = perm[:n_hold], perm[n_hold:]
        best = {"score": float("inf"), "snap": None, "step": -1}
        opt = torch.optim.Adam(trainables, lr=args.adam_lr)
        for step in range(args.adam_steps):
            idx_map = {
                6: rng.choice(len(bundles[6][0]), size=min(args.batch_per_n, len(bundles[6][0])), replace=False),
                9: rng.choice(pool, size=min(args.batch_per_n, len(pool)), replace=False),
            }
            opt.zero_grad()
            loss = batch_loss(
                idea, bundles, gamma, c1, c2, c3, p, idx_map, 0.0, weights,
                soft_med=True, loss_mode="med_align", qmed_map=qmed_map, soft_q=args.soft_q,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainables, 5.0)
            opt.step()
            project_delta_ball(c2, c3, c2_0, c3_0, args.rho)
            if step % 25 == 0 or step == args.adam_steps - 1:
                sc = holdout_score(
                    idea, bundles[9], gamma, c1, c2, c3, p, hold_idx,
                    soft_med=True, loss_mode="med_align", qmed9=qmed_map[9], soft_q=args.soft_q,
                )
                log(f"warm s{seed} adam{step} loss={float(loss):.5f} hold={sc:.5f}")
                if sc < best["score"]:
                    best = {"score": sc, "snap": snapshot_params(gamma, c1, c2, c3, p), "step": step}
        snap = best["snap"] or snapshot_params(gamma, c1, c2, c3, p)
        state = {
            "idea": idea, "L": L, "seed": seed,
            "gamma": snap["gamma"], "c1": snap["c1"], "c2": snap["c2"], "c3": snap["c3"],
            "p": None, "rho": args.rho, "loss_mode": "med_align", "soft_q": args.soft_q,
            "init": args.init, "warm": True, "holdout_best": best["score"],
            "recipe": S.RECIPE_NAME, "motif": S.MOTIF, "snap_head": S.SNAP_HEAD,
            "lock": "train69_only",
        }
        pt = out_dir / f"params_L{L}_seed{seed}.pt"
        torch.save(state, pt)
        rows = eval_state(state, device, EVAL_NS)
        ab, path = write_eval(out_dir, idea, L, seed, rows, {"warm_from": args.init, "hold": best["score"]})
        r15 = rows[15]["R"]
        if best_all is None or r15 < best_all[0]:
            best_all = (r15, ab, str(path), {n: rows[n]["R"] for n in EVAL_NS})
        log(f"warm s{seed} ALL={ab} R={best_all[-1] if ab else {n: rows[n]['R'] for n in EVAL_NS}}")
    log(f"warm DONE best_n15={best_all}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
