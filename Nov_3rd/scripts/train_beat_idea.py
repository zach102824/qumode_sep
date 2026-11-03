#!/usr/bin/env python3
"""Round-1 beat-QAOA trainers (training-only; Nov_3rd gate sequence locked).

Ideas:
  1 core_ratio     — train {6,9,12,15}, ratio-to-QAOA loss, free γ,c*
  2 rule_scaled    — train {9,12}, κ·2/(n-1) on c2,c3
  3 trust_residual — train {9,12,15}, freeze γ,c1; ball on c2,c3
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
_SCR = Path(__file__).resolve().parent
if str(_SCR) not in sys.path:
    sys.path.insert(0, str(_SCR))

from Nov_3rd import settings as S
from Nov_3rd.ansatz import make_sim
from Nov_3rd.data_io import load_qaoa, load_qaoa_scale_ref, log
from Nov_3rd.eval_scale import eval_n
from explore_OCT29.lib.qaoa import qaoa_psucc, inv_p_from_psucc

from beat_params import phi_from_c, project_ball, scale_c23, warm_from_qaoa

OUT_ROOT = S.OUT / "beat_qaoa_12h" / "round_1"
TRAIN_PACKS = S.OUT / "beat_qaoa_12h" / "train_packs"


def load_train_n(n: int, max_count: int | None = None) -> torch.Tensor:
    path = TRAIN_PACKS / f"train_n{n}.npz"
    d = np.load(path, allow_pickle=True)
    E = np.asarray(d["train_E"])
    if max_count is not None:
        E = E[: int(max_count)]
    return torch.as_tensor(E, dtype=torch.float32)


def precompute_qaoa_inv(betas, gammas, E: torch.Tensor, n: int, device) -> torch.Tensor:
    """Return (B,) inv_p under fair QAOA for ratio loss."""
    invs = []
    B = E.shape[0]
    chunk = 64 if n <= 9 else (16 if n <= 12 else 4)
    for s0 in range(0, B, chunk):
        Eb = E[s0 : s0 + chunk].to(device)
        ps = qaoa_psucc(betas.to(device), gammas.to(device), Eb, n)
        invs.append(inv_p_from_psucc(ps.detach().cpu().numpy()))
    return torch.as_tensor(np.concatenate(invs), dtype=torch.float32)


def make_params(idea: str, betas, gammas, device, kappa0: float = 1.0):
    g0, c10, c20, c30 = warm_from_qaoa(betas, gammas)
    g0, c10, c20, c30 = [t.to(device) for t in (g0, c10, c20, c30)]
    if idea == "trust_residual":
        gamma = g0.detach().clone()  # frozen
        c1 = c10.detach().clone()    # frozen
        c2 = torch.nn.Parameter(c20.clone())
        c3 = torch.nn.Parameter(c30.clone())
        kappa = None
        trainables = [c2, c3]
    elif idea == "rule_scaled":
        gamma = torch.nn.Parameter(g0.clone())
        c1 = torch.nn.Parameter(c10.clone())
        c2 = torch.nn.Parameter(c20.clone())
        c3 = torch.nn.Parameter(c30.clone())
        kappa = torch.nn.Parameter(torch.tensor(float(kappa0), device=device))
        trainables = [gamma, c1, c2, c3, kappa]
    else:  # core_ratio
        gamma = torch.nn.Parameter(g0.clone())
        c1 = torch.nn.Parameter(c10.clone())
        c2 = torch.nn.Parameter(c20.clone())
        c3 = torch.nn.Parameter(c30.clone())
        kappa = None
        trainables = [gamma, c1, c2, c3]
    return gamma, c1, c2, c3, kappa, trainables


def angles_for_n(idea, gamma, c1, c2, c3, kappa, n: int):
    if idea == "rule_scaled":
        c2n, c3n = scale_c23(c2, c3, n, kappa)
        return gamma, phi_from_c(c1, c2n, c3n)
    return gamma, phi_from_c(c1, c2, c3)


def batch_loss(idea, sims, bundles, gamma, c1, c2, c3, kappa, idx_map, lam_c: float):
    """Mean over n of mean_i log(Eθ/EQ) + lam * (c2²+c3²)."""
    terms = []
    for n, (E, inv_q, sim) in bundles.items():
        idx = idx_map[n]
        g, phi = angles_for_n(idea, gamma, c1, c2, c3, kappa, n)
        ls = []
        for j in idx:
            out = sim.forward_nov3rd(E[int(j)], g, phi)
            ratio = out["inv_p"].clamp(min=1e-12) / inv_q[int(j)].to(out["inv_p"].device).clamp(min=1e-12)
            ls.append(torch.log(ratio.clamp(min=1e-12)))
        terms.append(torch.stack(ls).mean())
    loss = torch.stack(terms).mean()
    # regularize dual-breaking (use unscaled c2,c3)
    if lam_c > 0:
        loss = loss + lam_c * (c2.pow(2).mean() + c3.pow(2).mean())
    return loss


def train_one(
    *,
    idea: str,
    L: int,
    seed: int,
    train_ns: list[int],
    device,
    adam_steps: int,
    adam_lr: float,
    lbfgs_rounds: int,
    lbfgs_iter: int,
    batch_per_n: int,
    lam0: float,
    lam1: float,
    rho: float | None,
    kappa0: float,
    out_dir: Path,
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    betas, gammas = load_qaoa(L)
    gamma, c1, c2, c3, kappa, trainables = make_params(idea, betas, gammas, device, kappa0=kappa0)

    bundles = {}
    for n in train_ns:
        E = load_train_n(n).to(device)
        log(f"{idea} L{L} seed{seed}: load train n{n} B={len(E)}")
        inv_q = precompute_qaoa_inv(betas, gammas, E.cpu(), n, device)
        sim = make_sim(n, L, device)
        bundles[n] = (E, inv_q.to(device), sim)

    hist = []
    t0 = time.time()

    # Adam
    opt = torch.optim.Adam(trainables, lr=adam_lr)
    for step in range(adam_steps):
        # λ anneal
        frac = step / max(adam_steps - 1, 1)
        lam = lam0 * (1 - frac) + lam1 * frac
        idx_map = {
            n: rng.choice(len(bundles[n][0]), size=min(batch_per_n, len(bundles[n][0])), replace=False)
            for n in train_ns
        }
        opt.zero_grad()
        loss = batch_loss(idea, None, bundles, gamma, c1, c2, c3, kappa, idx_map, lam)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainables, 5.0)
        opt.step()
        if idea == "trust_residual" and rho is not None:
            project_ball(c2, c3, rho)
        hist.append({"phase": "adam", "step": step, "loss": float(loss.detach()), "lam": lam})
        if step % 50 == 0 or step == adam_steps - 1:
            log(f"{idea} L{L} s{seed} adam {step}/{adam_steps} loss={hist[-1]['loss']:.5f} lam={lam:.2e}")

    # LBFGS
    for r in range(lbfgs_rounds):
        opt_l = torch.optim.LBFGS(
            trainables, lr=0.25, max_iter=lbfgs_iter, history_size=16, line_search_fn="strong_wolfe"
        )
        idx_map = {
            n: rng.choice(len(bundles[n][0]), size=min(batch_per_n, len(bundles[n][0])), replace=False)
            for n in train_ns
        }
        lam = lam1

        def closure():
            opt_l.zero_grad()
            lt = batch_loss(idea, None, bundles, gamma, c1, c2, c3, kappa, idx_map, lam)
            lt.backward()
            torch.nn.utils.clip_grad_norm_(trainables, 5.0)
            return lt

        try:
            loss = opt_l.step(closure)
            loss_f = float(loss.detach()) if torch.is_tensor(loss) else float(loss)
        except Exception as e:
            log(f"{idea} L{L} s{seed} LBFGS r{r} skip: {e}")
            continue
        if idea == "trust_residual" and rho is not None:
            project_ball(c2, c3, rho)
        hist.append({"phase": "lbfgs", "round": r, "loss": loss_f, "lam": lam})
        log(f"{idea} L{L} s{seed} lbfgs r{r}/{lbfgs_rounds} loss={loss_f:.5f}")

    state = {
        "idea": idea,
        "L": L,
        "seed": seed,
        "train_ns": train_ns,
        "gamma": gamma.detach().cpu(),
        "c1": c1.detach().cpu(),
        "c2": c2.detach().cpu(),
        "c3": c3.detach().cpu(),
        "kappa": None if kappa is None else float(kappa.detach().cpu()),
        "rho": rho,
        "betas_qaoa": betas.cpu(),
        "gammas_qaoa": gammas.cpu(),
        "wall_s": time.time() - t0,
        "hist_tail": hist[-20:],
        "recipe": S.RECIPE_NAME,
        "motif": S.MOTIF,
        "snap_head": S.SNAP_HEAD,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    pt = out_dir / f"params_L{L}_seed{seed}.pt"
    torch.save(state, pt)
    meta = {
        "idea": idea,
        "L": L,
        "seed": seed,
        "train_ns": train_ns,
        "kappa": state["kappa"],
        "rho": rho,
        "wall_s": state["wall_s"],
        "final_loss": hist[-1]["loss"] if hist else None,
        "c2_rms": float(c2.detach().cpu().pow(2).mean().sqrt()),
        "c3_rms": float(c3.detach().cpu().pow(2).mean().sqrt()),
        "params": str(pt),
    }
    (out_dir / f"train_L{L}_seed{seed}.json").write_text(json.dumps(meta, indent=2) + "\n")
    log(f"{idea} L{L} s{seed} saved {pt} wall={state['wall_s']:.0f}s")
    return state, meta


@torch.no_grad()
def eval_state(state, device, ns, max_count=1000):
    idea = state["idea"]
    L = int(state["L"])
    gamma = state["gamma"].to(device)
    c1 = state["c1"].to(device)
    c2 = state["c2"].to(device)
    c3 = state["c3"].to(device)
    kappa = state.get("kappa")
    if kappa is not None:
        kappa = torch.tensor(float(kappa), device=device)
    qref = load_qaoa_scale_ref(L)
    rows = {}
    for n in ns:
        g, phi = angles_for_n(idea, gamma, c1, c2, c3, kappa, n)
        m = eval_n(n, L, g, phi, device, max_count=max_count)
        q = qref.get(n) or {}
        qmed = float(q.get("Med", float("nan")))
        med = float(m["median_inv_p"])
        R = med / qmed if qmed == qmed and qmed > 0 else float("nan")
        if R != R:
            v = "NO_REF"
        elif R <= 0.98:
            v = "BEAT"
        elif R <= 1.02:
            v = "TIE"
        else:
            v = "UNDER"
        rows[n] = {
            "Med": med,
            "B90": float(m["B90_inv_p"]),
            "QAOA_Med": qmed,
            "R": R,
            "verdict": v,
            "mean_leak": float(m["mean_leak"]),
            "B": int(m["B"]),
        }
        log(f"eval {idea} L{L} n{n}: Med={med:.4f} Q={qmed:.4f} R={R:.4f} {v}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--idea", required=True, choices=["core_ratio", "rule_scaled", "trust_residual"])
    ap.add_argument("--L", type=int, required=True, choices=[1, 2, 16])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--adam_steps", type=int, default=None)
    ap.add_argument("--adam_lr", type=float, default=None)
    ap.add_argument("--lbfgs_rounds", type=int, default=None)
    ap.add_argument("--lbfgs_iter", type=int, default=8)
    ap.add_argument("--batch_per_n", type=int, default=32)
    ap.add_argument("--rho", type=float, default=0.1)
    ap.add_argument("--kappa0", type=float, default=1.0)
    ap.add_argument("--eval_only", type=str, default="", help="path to params .pt")
    ap.add_argument("--skip_eval", action="store_true")
    args = ap.parse_args()

    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    os.environ["MKL_NUM_THREADS"] = str(args.threads)
    torch.set_num_threads(args.threads)
    device = torch.device("cpu")

    idea = args.idea
    if idea == "core_ratio":
        train_ns = [6, 9, 12, 15]
        adam_steps = 600 if args.adam_steps is None else args.adam_steps
        adam_lr = 1e-3 if args.adam_lr is None else args.adam_lr
        lbfgs_rounds = 12 if args.lbfgs_rounds is None else args.lbfgs_rounds
        lam0, lam1 = 1e-1, 1e-3
        rho = None
        out_dir = OUT_ROOT / "idea1_core_ratio"
    elif idea == "rule_scaled":
        train_ns = [9, 12]
        adam_steps = 800 if args.adam_steps is None else args.adam_steps
        adam_lr = 1e-3 if args.adam_lr is None else args.adam_lr
        lbfgs_rounds = 8 if args.lbfgs_rounds is None else args.lbfgs_rounds
        lam0, lam1 = 1e-2, 1e-3
        rho = None
        out_dir = OUT_ROOT / "idea2_rule_scaled"
    else:
        train_ns = [9, 12, 15]
        adam_steps = 600 if args.adam_steps is None else args.adam_steps
        adam_lr = 5e-4 if args.adam_lr is None else args.adam_lr
        lbfgs_rounds = 8 if args.lbfgs_rounds is None else args.lbfgs_rounds
        lam0, lam1 = 0.0, 0.0
        rho = float(args.rho)
        out_dir = OUT_ROOT / "idea3_trust_residual"

    if args.eval_only:
        state = torch.load(args.eval_only, map_location="cpu", weights_only=False)
        rows = eval_state(state, device, [6, 9, 12, 15])
        payload = {"idea": idea, "L": args.L, "seed": state.get("seed"), "per_n": {str(k): v for k, v in rows.items()}}
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"eval_L{args.L}_seed{state.get('seed', 'x')}.json"
        path.write_text(json.dumps(payload, indent=2) + "\n")
        log(f"wrote {path}")
        return 0

    state, meta = train_one(
        idea=idea,
        L=args.L,
        seed=args.seed,
        train_ns=train_ns,
        device=device,
        adam_steps=adam_steps,
        adam_lr=adam_lr,
        lbfgs_rounds=lbfgs_rounds,
        lbfgs_iter=args.lbfgs_iter,
        batch_per_n=args.batch_per_n,
        lam0=lam0,
        lam1=lam1,
        rho=rho,
        kappa0=args.kappa0,
        out_dir=out_dir,
    )
    if not args.skip_eval:
        rows = eval_state(state, device, [6, 9, 12, 15])
        payload = {
            "idea": idea,
            "L": args.L,
            "seed": args.seed,
            "train": meta,
            "per_n": {str(k): v for k, v in rows.items()},
            "finished": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        }
        path = out_dir / f"eval_L{args.L}_seed{args.seed}.json"
        path.write_text(json.dumps(payload, indent=2) + "\n")
        # md snippet
        lines = [
            f"# {idea} L={args.L} seed={args.seed}",
            "",
            "| n | QAOA Med | Med | R | verdict |",
            "|---|----------|-----|---|---------|",
        ]
        for n in [6, 9, 12, 15]:
            r = rows[n]
            lines.append(f"| {n} | {r['QAOA_Med']:.4f} | {r['Med']:.4f} | {r['R']:.4f} | {r['verdict']} |")
        lines.append("")
        (out_dir / f"eval_L{args.L}_seed{args.seed}.md").write_text("\n".join(lines) + "\n")
        log(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
