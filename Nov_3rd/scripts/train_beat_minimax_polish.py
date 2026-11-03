#!/usr/bin/env python3
"""Idea A: minimax CORE-RATIO polish warm-started from Idea1 checkpoint."""
from __future__ import annotations

import argparse, json, os, sys, time
from pathlib import Path
import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from Nov_3rd import settings as S
from Nov_3rd.ansatz import make_sim
from Nov_3rd.data_io import load_qaoa, log
from Nov_3rd.eval_scale import eval_n
from explore_OCT29.lib.qaoa import qaoa_psucc, inv_p_from_psucc
from beat_params import phi_from_c
from train_beat_idea import load_train_n, precompute_qaoa_inv, eval_state

OUT = S.OUT / "beat_qaoa_12h" / "round_2" / "ideaA_minimax"


def soft_max_loss(terms, tau: float):
    # terms: list of scalar log-ratio means per n
    t = torch.stack(terms)
    return tau * torch.logsumexp(t / tau, dim=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--warm", type=str, required=True)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--adam_steps", type=int, default=200)
    ap.add_argument("--lbfgs_rounds", type=int, default=20)
    ap.add_argument("--tau", type=float, default=0.01)
    ap.add_argument("--batch_per_n", type=int, default=32)
    ap.add_argument("--n15_max", type=int, default=1000)
    args = ap.parse_args()

    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    os.environ["MKL_NUM_THREADS"] = str(args.threads)
    torch.set_num_threads(args.threads)
    device = torch.device("cpu")
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    warm = torch.load(args.warm, map_location="cpu", weights_only=False)
    L = args.L
    betas, gammas_q = load_qaoa(L)
    gamma = torch.nn.Parameter(warm["gamma"].to(device).float().clone())
    c1 = torch.nn.Parameter(warm["c1"].to(device).float().clone())
    c2 = torch.nn.Parameter(warm["c2"].to(device).float().clone())
    c3 = torch.nn.Parameter(warm["c3"].to(device).float().clone())
    trainables = [gamma, c1, c2, c3]

    train_ns = [6, 9, 12, 15]
    bundles = {}
    for n in train_ns:
        max_c = args.n15_max if n == 15 else 250
        E = load_train_n(n, max_count=max_c).to(device)
        log(f"minimax L{L} s{args.seed}: n{n} B={len(E)}")
        inv_q = precompute_qaoa_inv(betas, gammas_q, E.cpu(), n, device).to(device)
        bundles[n] = (E, inv_q, make_sim(n, L, device))

    def loss_fn(idx_map, lam):
        terms = []
        for n in train_ns:
            E, inv_q, sim = bundles[n]
            phi = phi_from_c(c1, c2, c3)
            ls = []
            for j in idx_map[n]:
                out = sim.forward_nov3rd(E[int(j)], gamma, phi)
                ratio = out["inv_p"].clamp(min=1e-12) / inv_q[int(j)].clamp(min=1e-12)
                ls.append(torch.log(ratio.clamp(min=1e-12)))
            terms.append(torch.stack(ls).mean())
        loss = soft_max_loss(terms, args.tau)
        if lam > 0:
            loss = loss + lam * (c2.pow(2).mean() + c3.pow(2).mean())
        return loss

    t0 = time.time()
    # Adam: λ from small → 0
    opt = torch.optim.Adam(trainables, lr=3e-4)
    for step in range(args.adam_steps):
        frac = step / max(args.adam_steps - 1, 1)
        lam = 1e-2 * (1 - frac)  # λ0/10 → 0  (Idea1 used 1e-1→1e-3)
        idx_map = {
            n: rng.choice(len(bundles[n][0]), size=min(args.batch_per_n, len(bundles[n][0])), replace=False)
            for n in train_ns
        }
        opt.zero_grad()
        loss = loss_fn(idx_map, lam)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainables, 5.0)
        opt.step()
        if step % 25 == 0 or step == args.adam_steps - 1:
            log(f"minimax L{L} s{args.seed} adam {step}/{args.adam_steps} loss={float(loss.detach()):.5f} lam={lam:.2e}")

    # LBFGS λ=0
    for r in range(args.lbfgs_rounds):
        opt_l = torch.optim.LBFGS(trainables, lr=0.25, max_iter=8, history_size=16, line_search_fn="strong_wolfe")
        idx_map = {
            n: rng.choice(len(bundles[n][0]), size=min(args.batch_per_n, len(bundles[n][0])), replace=False)
            for n in train_ns
        }
        def closure():
            opt_l.zero_grad()
            lt = loss_fn(idx_map, 0.0)
            lt.backward()
            torch.nn.utils.clip_grad_norm_(trainables, 5.0)
            return lt
        try:
            loss = opt_l.step(closure)
            loss_f = float(loss.detach()) if torch.is_tensor(loss) else float(loss)
        except Exception as e:
            log(f"minimax LBFGS skip: {e}"); continue
        log(f"minimax L{L} s{args.seed} lbfgs r{r}/{args.lbfgs_rounds} loss={loss_f:.5f}")

    OUT.mkdir(parents=True, exist_ok=True)
    state = {
        "idea": "minimax_core_ratio",
        "L": L,
        "seed": args.seed,
        "train_ns": train_ns,
        "gamma": gamma.detach().cpu(),
        "c1": c1.detach().cpu(),
        "c2": c2.detach().cpu(),
        "c3": c3.detach().cpu(),
        "kappa": None,
        "tau": args.tau,
        "warm": args.warm,
        "wall_s": time.time() - t0,
        "recipe": S.RECIPE_NAME,
        "motif": S.MOTIF,
        "snap_head": S.SNAP_HEAD,
    }
    pt = OUT / f"params_L{L}_seed{args.seed}.pt"
    torch.save(state, pt)
    rows = eval_state(state, device, [6, 9, 12, 15])
    payload = {
        "idea": "minimax_core_ratio",
        "L": L,
        "seed": args.seed,
        "tau": args.tau,
        "wall_s": state["wall_s"],
        "per_n": {str(k): v for k, v in rows.items()},
        "finished": time.strftime("%Y-%m-%d %H:%M:%S %z"),
    }
    (OUT / f"eval_L{L}_seed{args.seed}.json").write_text(json.dumps(payload, indent=2) + "\n")
    lines = [f"# minimax CORE-RATIO L={L} seed={args.seed}", "",
             "| n | QAOA Med | Med | R | verdict |", "|---|----------|-----|---|---------|"]
    for n in [6, 9, 12, 15]:
        r = rows[n]
        lines.append(f"| {n} | {r['QAOA_Med']:.4f} | {r['Med']:.4f} | {r['R']:.4f} | {r['verdict']} |")
    (OUT / f"eval_L{L}_seed{args.seed}.md").write_text("\n".join(lines) + "\n")
    log(f"minimax done → {pt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
