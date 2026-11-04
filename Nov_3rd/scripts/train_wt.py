#!/usr/bin/env python3
"""Train and eval WT: shared Walsh-degree tilt + power_p SNAP.

Train {6, 9} only. τ_1 ≡ 1. τ = 1 is the power_p circuit (champ, when warm-started
from a power_p checkpoint). CORE eval is scale1000, default B=1000.
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
from Nov_3rd.wt import (
    P_HI,
    P_LO,
    TAU_HI,
    TAU_LO,
    max_tau_dev,
    snap_angles,
    tau_full,
    tilt_collapsed,
)
from beat_params import warm_from_qaoa
from explore_OCT29.lib.qaoa import inv_p_from_psucc, qaoa_psucc

TRAIN_NS = [6, 9]
EVAL_NS = [6, 9, 12, 15]
IDEA = "wt"


def load_train_n(packs: Path, n: int, max_count: int | None) -> torch.Tensor:
    d = np.load(packs / f"train_n{n}.npz", allow_pickle=True)
    E = np.asarray(d["train_E"])
    if max_count is not None:
        E = E[: int(max_count)]
    return torch.as_tensor(E, dtype=torch.float32)


def precompute_qaoa_inv(betas, gammas, E: torch.Tensor, n: int, device) -> torch.Tensor:
    invs = []
    B = E.shape[0]
    chunk = 64 if n <= 9 else (16 if n <= 12 else 4)
    for s0 in range(0, B, chunk):
        Eb = E[s0 : s0 + chunk].to(device)
        ps = qaoa_psucc(betas.to(device), gammas.to(device), Eb, n)
        invs.append(inv_p_from_psucc(ps.detach().cpu().numpy()))
    return torch.as_tensor(np.concatenate(invs), dtype=torch.float32)


def _loss(bundles, gamma, c1, c2, c3, p, tau_free, idx_map, lam_c, weights, lam_p):
    """power_p log-ratio loss. The c2/c3 penalty uses a(n), same as the champ trainer."""
    from Nov_3rd.wt import a_of_n

    terms = []
    wsum = 0.0
    for n, (E, inv_q, sim) in bundles.items():
        gw, phi = snap_angles(gamma, c1, c2, c3, p, tau_free, n)
        logs = []
        for j in idx_map[n]:
            out = sim.forward_nov3rd(E[int(j)], gw, phi)
            ratio = out["inv_p"].clamp(min=1e-12) / inv_q[int(j)].to(out["inv_p"].device).clamp(min=1e-12)
            logs.append(torch.log(ratio.clamp(min=1e-12)))
        w = float(weights.get(n, 1.0))
        terms.append(w * torch.stack(logs).mean())
        wsum += w
        if lam_c > 0:
            a_n = a_of_n(n, p)
            terms.append(lam_c * w * ((c2 * a_n).pow(2).mean() + (c3 * a_n).pow(2).mean()))
            wsum += w
    loss = torch.stack(terms).sum() / max(wsum, 1e-12)
    if lam_p > 0:
        loss = loss + lam_p * p.pow(2)
    return loss


@torch.no_grad()
def holdout_score(bundle_n9, gamma, c1, c2, c3, p, tau_free, hold_idx) -> float:
    E, inv_q, sim = bundle_n9
    gw, phi = snap_angles(gamma, c1, c2, c3, p, tau_free, 9)
    logs = []
    for j in hold_idx:
        out = sim.forward_nov3rd(E[int(j)], gw, phi)
        ratio = out["inv_p"].clamp(min=1e-12) / inv_q[int(j)].to(out["inv_p"].device).clamp(min=1e-12)
        logs.append(torch.log(ratio.clamp(min=1e-12)))
    return float(torch.stack(logs).median().detach())


def snapshot(gamma, c1, c2, c3, p, tau_free):
    return {
        "gamma": gamma.detach().cpu().clone(),
        "c1": c1.detach().cpu().clone(),
        "c2": c2.detach().cpu().clone(),
        "c3": c3.detach().cpu().clone(),
        "p": float(p.detach().cpu()),
        "tau_free": tau_free.detach().cpu().clone(),
    }


def _clamp_(p, tau_free):
    with torch.no_grad():
        p.clamp_(P_LO, P_HI)
        tau_free.clamp_(TAU_LO, TAU_HI)


@torch.no_grad()
def eval_state(state, device, *, eval_B: int, qaoa_dir: Path | None, ns=None) -> dict:
    ns = list(ns or EVAL_NS)
    L = int(state["L"])
    gamma = state["gamma"].to(device)
    c1 = state["c1"].to(device)
    c2 = state["c2"].to(device)
    c3 = state["c3"].to(device)
    p = torch.tensor(float(state["p"]), device=device)
    tau_free = state["tau_free"].to(device)
    qref = load_qaoa_scale_ref(L, qaoa_dir=qaoa_dir)
    rows = {}
    for n in ns:
        gw, phi = snap_angles(gamma, c1, c2, c3, p, tau_free, n)
        # scalar twin at the same φ, only meaningful as a diagnostic when τ≈1
        m = eval_n(n, L, gw, phi, device, max_count=eval_B)
        q = qref.get(n) or {}
        qmed = float(q.get("Med", float("nan")))
        med = float(m["median_inv_p"])
        rows[n] = {
            "Med": med,
            "B90": float(m["B90_inv_p"]),
            "QAOA_Med": qmed,
            "R": (med / qmed) if qmed == qmed and qmed > 0 else float("nan"),
            "mean_leak": float(m["mean_leak"]),
            "B": int(m["B"]),
            "wall_s": float(m["wall_s"]),
        }
        log(f"eval WT L{L} n{n} B={m['B']}: Med={med:.4f} Q={qmed:.4f} ({m['wall_s']:.1f}s)")
    return rows


def train_one(args, device):
    seed = int(args.seed)
    L = int(args.L)
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    packs = Path(args.packs)
    qaoa_dir = Path(args.qaoa_dir) if args.qaoa_dir else None

    if args.warm_champ:
        champ = torch.load(args.warm_champ, map_location="cpu", weights_only=False)
        if int(champ["L"]) != L:
            raise SystemExit(f"warm champ L={champ['L']} != {L}")
        gamma = torch.nn.Parameter(champ["gamma"].to(device).float().clone())
        c1 = torch.nn.Parameter(champ["c1"].to(device).float().clone())
        c2 = torch.nn.Parameter(champ["c2"].to(device).float().clone())
        c3 = torch.nn.Parameter(champ["c3"].to(device).float().clone())
        p = torch.nn.Parameter(torch.tensor(float(champ["p"]), device=device))
        betas = champ["betas_qaoa"].float()
        gammas = champ["gammas_qaoa"].float()
        how = f"champ:{args.warm_champ}"
        log(f"WT L{L} s{seed}: warm champ p={float(p.detach()):.4f} from {args.warm_champ}")
    else:
        betas, gammas = load_qaoa(L, qaoa_dir=qaoa_dir)
        g0, c10, c20, c30 = [t.to(device) for t in warm_from_qaoa(betas, gammas)]
        gamma = torch.nn.Parameter(g0.clone())
        c1 = torch.nn.Parameter(c10.clone())
        c2 = torch.nn.Parameter(torch.zeros_like(c20) + 1e-4)
        c3 = torch.nn.Parameter(torch.zeros_like(c30) + 1e-4)
        p = torch.nn.Parameter(torch.tensor(float(args.p0), device=device))
        how = "qaoa_warm"
    tau_free = torch.nn.Parameter(torch.ones(3, device=device))
    trainables = [gamma, c1, c2, c3, p, tau_free]

    bundles = {}
    for n in TRAIN_NS:
        E = load_train_n(packs, n, args.max_train).to(device)
        log(f"WT L{L} s{seed}: train n{n} B={len(E)}")
        inv_q = precompute_qaoa_inv(betas, gammas, E.cpu(), n, device)
        bundles[n] = (E, inv_q.to(device), make_sim(n, L, device))

    n9_B = len(bundles[9][0])
    perm = rng.permutation(n9_B)
    n_hold = max(1, int(round(args.holdout_frac * n9_B)))
    hold_idx = perm[:n_hold]
    train_idx9 = perm[n_hold:]
    weights = {6: float(args.w6), 9: float(args.w9)}
    log(f"WT L{L} s{seed}: n9 holdout={n_hold} train_pool={len(train_idx9)}")

    hist = []
    t0 = time.time()
    best = {"score": float("inf"), "snap": snapshot(gamma, c1, c2, c3, p, tau_free), "step": "init"}

    def maybe_update(tag: str):
        sc = holdout_score(bundles[9], gamma, c1, c2, c3, p, tau_free, hold_idx)
        hist.append({"phase": "holdout", "tag": tag, "hold_med_logR": sc, "tau_dev": max_tau_dev(tau_free)})
        if sc < best["score"]:
            best["score"] = sc
            best["snap"] = snapshot(gamma, c1, c2, c3, p, tau_free)
            best["step"] = tag
            log(f"WT L{L} s{seed} NEW BEST holdout={sc:.5f} τdev={max_tau_dev(tau_free):.4f} @ {tag}")

    # Score the warm point first so a tilt that does not help restores τ=1.
    maybe_update("warm")

    opt = torch.optim.Adam(trainables, lr=args.adam_lr)
    for step in range(args.adam_steps):
        frac = step / max(args.adam_steps - 1, 1)
        lam = args.lam0 * (1 - frac) + args.lam1 * frac
        idx_map = {}
        for n in TRAIN_NS:
            pool = train_idx9 if n == 9 else len(bundles[n][0])
            if n == 9:
                take = min(args.batch_per_n, len(train_idx9))
                idx_map[n] = rng.choice(train_idx9, size=take, replace=False)
            else:
                idx_map[n] = rng.choice(pool, size=min(args.batch_per_n, pool), replace=False)
        opt.zero_grad()
        loss = _loss(bundles, gamma, c1, c2, c3, p, tau_free, idx_map, lam, weights, args.lam_p)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainables, 5.0)
        opt.step()
        _clamp_(p, tau_free)
        hist.append({"phase": "adam", "step": step, "loss": float(loss.detach()), "lam": lam})
        if step % 25 == 0 or step == args.adam_steps - 1:
            log(
                f"WT L{L} s{seed} adam {step}/{args.adam_steps} loss={hist[-1]['loss']:.5f} "
                f"lam={lam:.2e} p={float(p.detach()):.3f} τdev={max_tau_dev(tau_free):.4f} "
                f"τ={tau_full(tau_free).detach().cpu().tolist()}"
            )
            maybe_update(f"adam{step}")

    for r in range(args.lbfgs_rounds):
        opt_l = torch.optim.LBFGS(
            trainables, lr=0.25, max_iter=args.lbfgs_iter, history_size=16, line_search_fn="strong_wolfe"
        )
        idx_map = {}
        for n in TRAIN_NS:
            if n == 9:
                take = min(args.batch_per_n, len(train_idx9))
                idx_map[n] = rng.choice(train_idx9, size=take, replace=False)
            else:
                idx_map[n] = rng.choice(len(bundles[n][0]), size=min(args.batch_per_n, len(bundles[n][0])), replace=False)
        lam = args.lam1

        def closure():
            opt_l.zero_grad()
            lt = _loss(bundles, gamma, c1, c2, c3, p, tau_free, idx_map, lam, weights, args.lam_p)
            lt.backward()
            torch.nn.utils.clip_grad_norm_(trainables, 5.0)
            return lt

        try:
            loss = opt_l.step(closure)
            loss_f = float(loss.detach()) if torch.is_tensor(loss) else float(loss)
        except Exception as e:
            log(f"WT L{L} s{seed} LBFGS r{r} skip: {e}")
            continue
        _clamp_(p, tau_free)
        hist.append({"phase": "lbfgs", "round": r, "loss": loss_f, "lam": lam})
        log(f"WT L{L} s{seed} lbfgs {r}/{args.lbfgs_rounds} loss={loss_f:.5f} τdev={max_tau_dev(tau_free):.4f}")
        maybe_update(f"lbfgs{r}")

    snap = best["snap"]
    with torch.no_grad():
        gamma.data.copy_(snap["gamma"].to(device))
        c1.data.copy_(snap["c1"].to(device))
        c2.data.copy_(snap["c2"].to(device))
        c3.data.copy_(snap["c3"].to(device))
        p.data.fill_(snap["p"])
        tau_free.data.copy_(snap["tau_free"].to(device))

    state = {
        "idea": IDEA,
        "label": "shared_walsh_degree_tilt_power_p",
        "L": L,
        "seed": seed,
        "train_ns": list(TRAIN_NS),
        "gamma": gamma.detach().cpu(),
        "c1": c1.detach().cpu(),
        "c2": c2.detach().cpu(),
        "c3": c3.detach().cpu(),
        "p": float(p.detach().cpu()),
        "tau_free": tau_free.detach().cpu(),
        "tau": tau_full(tau_free).detach().cpu(),
        "tau_dev": max_tau_dev(tau_free),
        "collapsed": tilt_collapsed(tau_free),
        "warm": how,
        "weights": weights,
        "holdout_frac": args.holdout_frac,
        "holdout_best": best["score"],
        "holdout_best_step": best["step"],
        "betas_qaoa": betas.cpu() if torch.is_tensor(betas) else betas,
        "gammas_qaoa": gammas.cpu() if torch.is_tensor(gammas) else gammas,
        "wall_s": time.time() - t0,
        "hist_tail": hist[-40:],
        "packs": str(packs),
        "max_train": args.max_train,
        "recipe": S.RECIPE_NAME,
        "lock": "train69_only",
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pt = out_dir / "params.pt"
    torch.save(state, pt)
    meta = {
        "idea": IDEA,
        "L": L,
        "seed": seed,
        "p": state["p"],
        "tau": state["tau"].tolist(),
        "tau_dev": state["tau_dev"],
        "collapsed": state["collapsed"],
        "holdout_best": state["holdout_best"],
        "holdout_best_step": state["holdout_best_step"],
        "warm": how,
        "wall_s": state["wall_s"],
        "params": str(pt),
    }
    (out_dir / "train.json").write_text(json.dumps(meta, indent=2) + "\n")
    log(
        f"WT L{L} s{seed} saved {pt} wall={state['wall_s']:.0f}s "
        f"τ={meta['tau']} dev={state['tau_dev']:.4f} collapsed={state['collapsed']}"
    )
    return state


def write_eval(out_dir: Path, state, rows: dict) -> None:
    payload = {
        "idea": IDEA,
        "L": int(state["L"]),
        "seed": int(state["seed"]),
        "tau": [float(x) for x in state["tau"].reshape(-1)],
        "tau_dev": float(state["tau_dev"]),
        "collapsed": bool(state["collapsed"]),
        "p": float(state["p"]),
        "per_n": {str(k): v for k, v in rows.items()},
        "finished": time.strftime("%Y-%m-%d %H:%M:%S %z"),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "eval.json").write_text(json.dumps(payload, indent=2) + "\n")
    lines = [
        f"# WT L={state['L']} seed={state['seed']}",
        "",
        f"τ = {payload['tau']} (dev {payload['tau_dev']:.4f}, collapsed={payload['collapsed']})",
        "",
        "| n | Med | B90 | QAOA Med | R | B |",
        "|---|-----|-----|----------|---|---|",
    ]
    for n in EVAL_NS:
        if n not in rows:
            continue
        r = rows[n]
        lines.append(
            f"| {n} | {r['Med']:.4f} | {r['B90']:.4f} | {r['QAOA_Med']:.4f} | {r['R']:.4f} | {r['B']} |"
        )
    (out_dir / "eval.md").write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser(description="Train/eval WT shared Walsh tilt + power_p")
    ap.add_argument("--L", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--packs", type=str, required=True)
    ap.add_argument("--out-dir", type=str, required=True)
    ap.add_argument("--qaoa-dir", type=str, default="")
    ap.add_argument("--warm-champ", type=str, default="", help="power_p params.pt; τ=1 starts there")
    ap.add_argument("--p0", type=float, default=0.0)
    ap.add_argument("--adam-steps", type=int, default=600)
    ap.add_argument("--adam-lr", type=float, default=1e-3)
    ap.add_argument("--lbfgs-rounds", type=int, default=12)
    ap.add_argument("--lbfgs-iter", type=int, default=8)
    ap.add_argument("--batch-per-n", type=int, default=32)
    ap.add_argument("--max-train", type=int, default=None)
    ap.add_argument("--lam0", type=float, default=1e-1)
    ap.add_argument("--lam1", type=float, default=1e-3)
    ap.add_argument("--lam-p", type=float, default=1e-2)
    ap.add_argument("--holdout-frac", type=float, default=0.2)
    ap.add_argument("--w6", type=float, default=1.0)
    ap.add_argument("--w9", type=float, default=2.0)
    ap.add_argument("--eval-B", type=int, default=1000)
    ap.add_argument("--eval-ns", type=str, default="6,9,12,15")
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--skip-eval", action="store_true")
    args = ap.parse_args()

    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    os.environ["MKL_NUM_THREADS"] = str(args.threads)
    torch.set_num_threads(args.threads)
    device = torch.device("cpu")
    out_dir = Path(args.out_dir)
    qaoa_dir = Path(args.qaoa_dir) if args.qaoa_dir else None

    if not args.skip_train:
        train_one(args, device)
    elif not (out_dir / "params.pt").exists():
        raise SystemExit(f"--skip-train but no {out_dir / 'params.pt'}")
    if args.skip_eval:
        return 0
    state = torch.load(out_dir / "params.pt", map_location="cpu", weights_only=False)
    ns = [int(x) for x in args.eval_ns.split(",") if x.strip()]
    rows = eval_state(state, device, eval_B=args.eval_B, qaoa_dir=qaoa_dir, ns=ns)
    write_eval(out_dir, state, rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
