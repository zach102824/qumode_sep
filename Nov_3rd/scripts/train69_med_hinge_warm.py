#!/usr/bin/env python3
"""Idea D: delta-ball warm with hinge med_align — relu(log R_med - log(0.98))."""
from __future__ import annotations
import argparse, os, sys, time
from pathlib import Path
import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT)); sys.path.insert(0, str(Path(__file__).resolve().parent))
from Nov_3rd import settings as S
from Nov_3rd.ansatz import make_sim
from Nov_3rd.data_io import load_qaoa, log
from train_beat_train69 import (
    EVAL_NS, OUT_ROOT, TRAIN_NS, angles_for_n, eval_state, instance_inv_ps,
    load_train_n, precompute_qaoa_inv, soft_quantile, snapshot_params, write_eval,
)

CHAMP = OUT_ROOT / "champion" / "params_L2_train69_pareto_a080_s1.pt"
BEAT_LOG = float(np.log(0.98))


def project_delta_ball(c2, c3, c2_0, c3_0, rho: float):
    with torch.no_grad():
        d2, d3 = c2 - c2_0, c3 - c3_0
        v = torch.cat([d2.reshape(-1), d3.reshape(-1)])
        nrm = float(v.norm().item())
        if nrm > rho and nrm > 0:
            s = rho / nrm
            c2.copy_(c2_0 + d2 * s); c3.copy_(c3_0 + d3 * s)


def hinge_loss(bundles, gamma, c1, c2, c3, idx_map, qmed_map, weights, soft_q=0.5):
    terms = []; wsum = 0.0
    for n, (E, inv_q, sim) in bundles.items():
        w = float(weights.get(n, 1.0))
        if w <= 0: continue
        g, phi = angles_for_n("trust69", gamma, c1, c2, c3, None, n)
        invs = instance_inv_ps(sim, E, g, phi, idx_map[n])
        sm = soft_quantile(invs, q=soft_q)
        qmed = torch.tensor(qmed_map[n], device=sm.device, dtype=sm.dtype)
        logR = torch.log((sm / qmed.clamp(min=1e-12)).clamp(min=1e-12))
        terms.append(w * torch.relu(logR - BEAT_LOG))
        wsum += w
    return torch.stack(terms).sum() / max(wsum, 1e-12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rho", type=float, default=0.02)
    ap.add_argument("--adam_steps", type=int, default=300)
    ap.add_argument("--seeds", type=str, default="0,1,2,3,4,5")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--soft_q", type=float, default=0.5)
    args = ap.parse_args()
    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    torch.set_num_threads(args.threads)
    device = torch.device("cpu"); L = 2
    init = torch.load(CHAMP, map_location="cpu", weights_only=False)
    betas, gammas = load_qaoa(L)
    bundles = {}
    for n in TRAIN_NS:
        E = load_train_n(n).to(device)
        inv_q = precompute_qaoa_inv(betas, gammas, E.cpu(), n, device)
        bundles[n] = (E, inv_q.to(device), make_sim(n, L, device))
    qmed_map = {n: float(bundles[n][1].median().cpu()) for n in bundles}
    out = OUT_ROOT / "trust69" / f"med_hinge_delta_rho{args.rho:g}"
    out.mkdir(parents=True, exist_ok=True)
    weights = {6: 1.0, 9: 2.0}
    best_all = None
    for seed in [int(x) for x in args.seeds.split(",") if x.strip()]:
        torch.manual_seed(seed); np.random.seed(seed); rng = np.random.default_rng(seed)
        gamma = init["gamma"].to(device); c1 = init["c1"].to(device)
        c2_0 = init["c2"].to(device).detach().clone(); c3_0 = init["c3"].to(device).detach().clone()
        c2, c3 = c2_0.clone(), c3_0.clone()
        if seed > 0:
            with torch.no_grad():
                noise = torch.randn(c2.numel() + c3.numel())
                noise = noise / (noise.norm() + 1e-12) * (args.rho * float(rng.uniform(0.2, 0.9)))
                c2 = c2 + noise[:c2.numel()].reshape_as(c2)
                c3 = c3 + noise[c2.numel():].reshape_as(c3)
                project_delta_ball(c2, c3, c2_0, c3_0, args.rho)
        c2 = c2.detach().requires_grad_(True); c3 = c3.detach().requires_grad_(True)
        opt = torch.optim.Adam([c2, c3], lr=3e-4)
        for step in range(args.adam_steps):
            idx_map = {n: rng.choice(len(bundles[n][0]), size=min(48, len(bundles[n][0])), replace=False) for n in TRAIN_NS}
            opt.zero_grad()
            loss = hinge_loss(bundles, gamma, c1, c2, c3, idx_map, qmed_map, weights, args.soft_q)
            loss.backward(); torch.nn.utils.clip_grad_norm_([c2, c3], 5.0); opt.step()
            project_delta_ball(c2, c3, c2_0, c3_0, args.rho)
            if step % 50 == 0:
                log(f"hinge s{seed} adam{step} loss={float(loss.detach()):.5f}")
        snap = snapshot_params(gamma, c1, c2, c3, None)
        state = {"idea": "trust69", "L": L, "seed": seed, **{k: snap[k] for k in ("gamma","c1","c2","c3")},
                 "p": None, "rho": args.rho, "loss": "med_hinge", "recipe": S.RECIPE_NAME,
                 "motif": S.MOTIF, "snap_head": S.SNAP_HEAD, "lock": "train69_only"}
        torch.save(state, out / f"params_L{L}_seed{seed}.pt")
        rows = eval_state(state, device, EVAL_NS)
        ab, path = write_eval(out, "trust69", L, seed, rows, {"hinge": True})
        r15 = rows[15]["R"]; knee = max(rows[12]["R"], r15)
        log(f"hinge s{seed} ALL={ab} knee={knee:.4f} R={{n: rows[n]['R'] for n in EVAL_NS}}")
        if best_all is None or knee < best_all[0]:
            best_all = (knee, ab, str(path), {n: rows[n]["R"] for n in EVAL_NS})
    log(f"hinge DONE best={best_all}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
