#!/usr/bin/env python3
"""Train{6,9}-only beat-QAOA (Zach t2563u).

Ideas (gates locked — Hamming-shell + Nov1st X-sandwich):
  ratio69_w9  — Idea1: free γ,c1-c3 unscaled; w6=1,w9=2; early-stop on held-out n9
  power_p     — Idea2: a_n=(8/(n-1))^p; free γ,c1-c3,p; penalize scaled c2,c3
  trust69     — Idea3: freeze γ,c1; ball on c2,c3; soft-median optional
  ratio69     — baseline free shared, equal weights (no early-stop)
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

from beat_params import phi_from_c, project_ball, warm_from_qaoa

OUT_ROOT = S.OUT / "beat_qaoa_12h" / "train69"
TRAIN_PACKS = S.OUT / "beat_qaoa_12h" / "train_packs"
TRAIN_NS = [6, 9]
EVAL_NS = [6, 9, 12, 15]
IDEAS = ("ratio69_w9", "power_p", "trust69", "ratio69")


def load_train_n(n: int, max_count: int | None = None) -> torch.Tensor:
    path = TRAIN_PACKS / f"train_n{n}.npz"
    d = np.load(path, allow_pickle=True)
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


def make_params(idea: str, betas, gammas, device, p0: float = 0.0, rho: float | None = None, freeze_p: bool = False):
    g0, c10, c20, c30 = [t.to(device) for t in warm_from_qaoa(betas, gammas)]
    if idea == "trust69":
        gamma = g0.detach().clone()
        c1 = c10.detach().clone()
        c2 = torch.nn.Parameter(torch.zeros_like(c20) + 1e-4)
        c3 = torch.nn.Parameter(torch.zeros_like(c30) + 1e-4)
        p = None
        trainables = [c2, c3]
    elif idea == "power_p":
        gamma = torch.nn.Parameter(g0.clone())
        c1 = torch.nn.Parameter(c10.clone())
        c2 = torch.nn.Parameter(torch.zeros_like(c20) + 1e-4)
        c3 = torch.nn.Parameter(torch.zeros_like(c30) + 1e-4)
        if freeze_p:
            p = torch.tensor(float(p0), device=device)
            trainables = [gamma, c1, c2, c3]
        else:
            p = torch.nn.Parameter(torch.tensor(float(p0), device=device))
            trainables = [gamma, c1, c2, c3, p]
    else:  # ratio69 / ratio69_w9
        gamma = torch.nn.Parameter(g0.clone())
        c1 = torch.nn.Parameter(c10.clone())
        c2 = torch.nn.Parameter(c20.clone())
        c3 = torch.nn.Parameter(c30.clone())
        p = None
        trainables = [gamma, c1, c2, c3]
    return gamma, c1, c2, c3, p, trainables


def a_of_n(idea: str, n: int, p) -> torch.Tensor | float:
    """Strength multiplier for c2,c3 at size n."""
    if idea != "power_p":
        return 1.0
    # a_n = (8/(n-1))^p  so a9=1
    base = 8.0 / float(n - 1)
    if torch.is_tensor(p):
        # clip p in forward via tanh map? use raw + clamp in train step
        return torch.pow(torch.tensor(base, device=p.device, dtype=p.dtype), p)
    return float(base) ** float(p)


def angles_for_n(idea, gamma, c1, c2, c3, p, n: int):
    a = a_of_n(idea, n, p)
    if torch.is_tensor(a):
        c2n, c3n = c2 * a, c3 * a
    else:
        c2n, c3n = c2 * float(a), c3 * float(a)
    return gamma, phi_from_c(c1, c2n, c3n)


def soft_median(vals: torch.Tensor, temp: float = 0.05) -> torch.Tensor:
    """Differentiable soft-median via sorted weights (approx)."""
    v = vals.reshape(-1)
    if v.numel() == 1:
        return v[0]
    # soft quantile at 0.5 using temperature on ranks
    order = torch.argsort(v)
    vs = v[order]
    # triangular / logistic weights around median index
    mid = 0.5 * (v.numel() - 1)
    idx = torch.arange(v.numel(), device=v.device, dtype=v.dtype)
    w = torch.softmax(-((idx - mid) / max(temp * v.numel(), 1e-3)).abs(), dim=0)
    return (w * vs).sum()


def instance_log_ratios(sim, E, inv_q, g, phi, idx) -> torch.Tensor:
    ls = []
    for j in idx:
        out = sim.forward_nov3rd(E[int(j)], g, phi)
        ratio = out["inv_p"].clamp(min=1e-12) / inv_q[int(j)].to(out["inv_p"].device).clamp(min=1e-12)
        ls.append(torch.log(ratio.clamp(min=1e-12)))
    return torch.stack(ls)


def batch_loss(
    idea,
    bundles,
    gamma,
    c1,
    c2,
    c3,
    p,
    idx_map,
    lam_c: float,
    weights: dict[int, float],
    *,
    soft_med: bool = False,
    lam_p: float = 0.0,
):
    terms = []
    wsum = 0.0
    for n, (E, inv_q, sim) in bundles.items():
        idx = idx_map[n]
        g, phi = angles_for_n(idea, gamma, c1, c2, c3, p, n)
        logs = instance_log_ratios(sim, E, inv_q, g, phi, idx)
        tn = soft_median(logs) if soft_med else logs.mean()
        w = float(weights.get(n, 1.0))
        terms.append(w * tn)
        wsum += w
        # penalize scaled c2,c3 for power_p (avoid gauge)
        if idea == "power_p" and lam_c > 0:
            a = a_of_n(idea, n, p)
            if torch.is_tensor(a):
                terms.append(lam_c * w * ((c2 * a).pow(2).mean() + (c3 * a).pow(2).mean()))
            else:
                terms.append(lam_c * w * ((c2 * a).pow(2).mean() + (c3 * a).pow(2).mean()))
            wsum += w  # keep scale roughly
    loss = torch.stack(terms).sum() / max(wsum, 1e-12)
    if idea != "power_p" and lam_c > 0:
        loss = loss + lam_c * (c2.pow(2).mean() + c3.pow(2).mean())
    if idea == "power_p" and lam_p > 0 and p is not None:
        loss = loss + lam_p * p.pow(2)
    return loss


@torch.no_grad()
def holdout_score(idea, bundle_n9, gamma, c1, c2, c3, p, hold_idx, soft_med: bool = False) -> float:
    E, inv_q, sim = bundle_n9
    g, phi = angles_for_n(idea, gamma, c1, c2, c3, p, 9)
    logs = instance_log_ratios(sim, E, inv_q, g, phi, hold_idx)
    if soft_med:
        return float(soft_median(logs).detach())
    # soft-median-ish: true median of log-ratios
    return float(logs.median().detach())


def snapshot_params(gamma, c1, c2, c3, p):
    return {
        "gamma": gamma.detach().cpu().clone(),
        "c1": c1.detach().cpu().clone(),
        "c2": c2.detach().cpu().clone(),
        "c3": c3.detach().cpu().clone(),
        "p": None if p is None else float(p.detach().cpu()),
    }


def load_snapshot(snap, device, idea, betas, gammas):
    gamma = snap["gamma"].to(device)
    c1 = snap["c1"].to(device)
    c2 = snap["c2"].to(device)
    c3 = snap["c3"].to(device)
    p = None if snap["p"] is None else torch.tensor(float(snap["p"]), device=device)
    return gamma, c1, c2, c3, p


def train_one(
    *,
    idea: str,
    L: int,
    seed: int,
    device,
    adam_steps: int,
    adam_lr: float,
    lbfgs_rounds: int,
    lbfgs_iter: int,
    batch_per_n: int,
    lam0: float,
    lam1: float,
    lam_p: float,
    rho: float | None,
    p0: float,
    freeze_p: bool = False,
    max_train: int | None,
    holdout_frac: float,
    soft_med: bool,
    weights: dict[int, float],
    out_dir: Path,
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    betas, gammas = load_qaoa(L)
    gamma, c1, c2, c3, p, trainables = make_params(idea, betas, gammas, device, p0=p0, freeze_p=freeze_p)

    bundles = {}
    for n in TRAIN_NS:
        E = load_train_n(n, max_count=max_train).to(device)
        log(f"{idea} L{L} seed{seed}: load train n{n} B={len(E)}")
        inv_q = precompute_qaoa_inv(betas, gammas, E.cpu(), n, device)
        sim = make_sim(n, L, device)
        bundles[n] = (E, inv_q.to(device), sim)

    # hold out ~20% of n9 for early-stop / checkpoint
    n9_B = len(bundles[9][0])
    perm = rng.permutation(n9_B)
    n_hold = max(1, int(round(holdout_frac * n9_B))) if idea in ("ratio69_w9", "power_p", "trust69") else 0
    hold_idx = perm[:n_hold] if n_hold else np.array([], dtype=int)
    train_idx9_pool = perm[n_hold:] if n_hold else perm
    log(f"{idea} L{L} s{seed}: n9 holdout={n_hold} train_pool={len(train_idx9_pool)}")

    hist = []
    t0 = time.time()
    best = {"score": float("inf"), "snap": snapshot_params(gamma, c1, c2, c3, p), "step": -1}

    def maybe_update_best(step_tag: str):
        if n_hold == 0:
            return
        sc = holdout_score(idea, bundles[9], gamma, c1, c2, c3, p, hold_idx, soft_med=soft_med)
        hist.append({"phase": "holdout", "tag": step_tag, "hold_med_logR": sc})
        if sc < best["score"]:
            best["score"] = sc
            best["snap"] = snapshot_params(gamma, c1, c2, c3, p)
            best["step"] = step_tag
            log(f"{idea} L{L} s{seed} NEW BEST holdout={sc:.5f} @ {step_tag}")

    opt = torch.optim.Adam(trainables, lr=adam_lr)
    for step in range(adam_steps):
        frac = step / max(adam_steps - 1, 1)
        lam = lam0 * (1 - frac) + lam1 * frac
        idx_map = {}
        for n in TRAIN_NS:
            if n == 9 and len(train_idx9_pool):
                take = min(batch_per_n, len(train_idx9_pool))
                idx_map[n] = rng.choice(train_idx9_pool, size=take, replace=False)
            else:
                idx_map[n] = rng.choice(len(bundles[n][0]), size=min(batch_per_n, len(bundles[n][0])), replace=False)
        opt.zero_grad()
        loss = batch_loss(
            idea, bundles, gamma, c1, c2, c3, p, idx_map, lam, weights, soft_med=soft_med, lam_p=lam_p
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainables, 5.0)
        opt.step()
        if idea == "power_p" and p is not None:
            with torch.no_grad():
                p.clamp_(-1.0, 2.0)
        if idea == "trust69" and rho is not None:
            project_ball(c2, c3, rho)
        hist.append({"phase": "adam", "step": step, "loss": float(loss.detach()), "lam": lam})
        if step % 25 == 0 or step == adam_steps - 1:
            extra = f" p={float(p):.3f}" if p is not None else ""
            log(f"{idea} L{L} s{seed} adam {step}/{adam_steps} loss={hist[-1]['loss']:.5f} lam={lam:.2e}{extra}")
            maybe_update_best(f"adam{step}")

    for r in range(lbfgs_rounds):
        opt_l = torch.optim.LBFGS(
            trainables, lr=0.25, max_iter=lbfgs_iter, history_size=16, line_search_fn="strong_wolfe"
        )
        idx_map = {}
        for n in TRAIN_NS:
            if n == 9 and len(train_idx9_pool):
                take = min(batch_per_n, len(train_idx9_pool))
                idx_map[n] = rng.choice(train_idx9_pool, size=take, replace=False)
            else:
                idx_map[n] = rng.choice(len(bundles[n][0]), size=min(batch_per_n, len(bundles[n][0])), replace=False)
        lam = lam1

        def closure():
            opt_l.zero_grad()
            lt = batch_loss(
                idea, bundles, gamma, c1, c2, c3, p, idx_map, lam, weights, soft_med=soft_med, lam_p=lam_p
            )
            lt.backward()
            torch.nn.utils.clip_grad_norm_(trainables, 5.0)
            return lt

        try:
            loss = opt_l.step(closure)
            loss_f = float(loss.detach()) if torch.is_tensor(loss) else float(loss)
        except Exception as e:
            log(f"{idea} L{L} s{seed} LBFGS r{r} skip: {e}")
            continue
        if idea == "power_p" and p is not None:
            with torch.no_grad():
                p.clamp_(-1.0, 2.0)
        if idea == "trust69" and rho is not None:
            project_ball(c2, c3, rho)
        hist.append({"phase": "lbfgs", "round": r, "loss": loss_f, "lam": lam})
        log(f"{idea} L{L} s{seed} lbfgs r{r}/{lbfgs_rounds} loss={loss_f:.5f}")
        maybe_update_best(f"lbfgs{r}")

    # restore best holdout checkpoint if we tracked one
    if n_hold and best["score"] < float("inf"):
        log(f"{idea} L{L} s{seed}: restoring best holdout={best['score']:.5f} @ {best['step']}")
        snap = best["snap"]
        with torch.no_grad():
            gamma.copy_(snap["gamma"].to(device)) if isinstance(gamma, torch.nn.Parameter) else None
            # always write into tensors used for save
            g_f = snap["gamma"].to(device)
            c1_f = snap["c1"].to(device)
            c2_f = snap["c2"].to(device)
            c3_f = snap["c3"].to(device)
            p_f = snap["p"]
            if isinstance(gamma, torch.nn.Parameter):
                gamma.data.copy_(g_f)
            else:
                gamma = g_f
            if isinstance(c1, torch.nn.Parameter):
                c1.data.copy_(c1_f)
            else:
                c1 = c1_f
            c2.data.copy_(c2_f)
            c3.data.copy_(c3_f)
            if p is not None and p_f is not None:
                p.data.fill_(p_f)
    else:
        g_f, c1_f, c2_f, c3_f = gamma, c1, c2, c3
        p_f = None if p is None else float(p.detach().cpu())

    # re-read final values from parameters
    g_out = gamma.detach().cpu() if torch.is_tensor(gamma) else g_f.detach().cpu()
    c1_out = c1.detach().cpu()
    c2_out = c2.detach().cpu()
    c3_out = c3.detach().cpu()
    p_out = None if p is None else float(p.detach().cpu())

    state = {
        "idea": idea,
        "L": L,
        "seed": seed,
        "train_ns": list(TRAIN_NS),
        "gamma": g_out,
        "c1": c1_out,
        "c2": c2_out,
        "c3": c3_out,
        "p": p_out,
        "rho": rho,
        "weights": weights,
        "holdout_frac": holdout_frac,
        "holdout_best": best["score"] if n_hold else None,
        "holdout_best_step": best["step"] if n_hold else None,
        "betas_qaoa": betas.cpu(),
        "gammas_qaoa": gammas.cpu(),
        "wall_s": time.time() - t0,
        "hist_tail": hist[-30:],
        "recipe": S.RECIPE_NAME,
        "motif": S.MOTIF,
        "snap_head": S.SNAP_HEAD,
        "lock": "train69_only",
        "soft_med": soft_med,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    pt = out_dir / f"params_L{L}_seed{seed}.pt"
    torch.save(state, pt)
    meta = {
        "idea": idea,
        "L": L,
        "seed": seed,
        "train_ns": list(TRAIN_NS),
        "p": p_out,
        "rho": rho,
        "weights": weights,
        "wall_s": state["wall_s"],
        "final_loss": hist[-1]["loss"] if hist and "loss" in hist[-1] else None,
        "holdout_best": state["holdout_best"],
        "holdout_best_step": state["holdout_best_step"],
        "c2_rms": float(c2_out.pow(2).mean().sqrt()),
        "c3_rms": float(c3_out.pow(2).mean().sqrt()),
        "params": str(pt),
    }
    (out_dir / f"train_L{L}_seed{seed}.json").write_text(json.dumps(meta, indent=2) + "\n")
    log(f"{idea} L{L} s{seed} saved {pt} wall={state['wall_s']:.0f}s")
    return state, meta


@torch.no_grad()
def eval_state(state, device, ns=EVAL_NS, max_count=1000):
    idea = state["idea"]
    L = int(state["L"])
    gamma = state["gamma"].to(device)
    c1 = state["c1"].to(device)
    c2 = state["c2"].to(device)
    c3 = state["c3"].to(device)
    p = state.get("p")
    if p is not None:
        p = torch.tensor(float(p), device=device)
    qref = load_qaoa_scale_ref(L)
    rows = {}
    for n in ns:
        g, phi = angles_for_n(idea, gamma, c1, c2, c3, p, n)
        m = eval_n(n, L, g, phi, device, max_count=max_count)
        q = qref.get(n) or {}
        qmed = float(q.get("Med", float("nan")))
        qb90 = float(q.get("B90", float("nan"))) if "B90" in q else float("nan")
        med = float(m["median_inv_p"])
        b90 = float(m["B90_inv_p"])
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
            "B90": b90,
            "QAOA_Med": qmed,
            "QAOA_B90": qb90,
            "R": R,
            "verdict": v,
            "mean_leak": float(m["mean_leak"]),
            "B": int(m["B"]),
        }
        log(f"eval {idea} L{L} n{n}: Med={med:.4f} Q={qmed:.4f} R={R:.4f} {v}")
    return rows


def write_eval(out_dir: Path, idea: str, L: int, seed, rows: dict, extra: dict | None = None):
    payload = {
        "idea": idea,
        "L": L,
        "seed": seed,
        "train_ns": list(TRAIN_NS),
        "per_n": {str(k): v for k, v in rows.items()},
        "finished": time.strftime("%Y-%m-%d %H:%M:%S %z"),
    }
    if extra:
        payload.update(extra)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"eval_L{L}_seed{seed}.json"
    path.write_text(json.dumps(payload, indent=2) + "\n")
    lines = [
        f"# {idea} L={L} seed={seed} train{{6,9}}",
        "",
        "| n | QAOA Med | Med | R | verdict |",
        "|---|----------|-----|---|---------|",
    ]
    for n in EVAL_NS:
        r = rows[n]
        lines.append(f"| {n} | {r['QAOA_Med']:.4f} | {r['Med']:.4f} | {r['R']:.4f} | {r['verdict']} |")
    all_beat = all(rows[n]["verdict"] == "BEAT" for n in EVAL_NS)
    lines += ["", f"**ALL_CORE_BEAT** = {all_beat}", ""]
    (out_dir / f"eval_L{L}_seed{seed}.md").write_text("\n".join(lines) + "\n")
    log(f"wrote {path} ALL_CORE_BEAT={all_beat}")
    return all_beat, path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--idea", required=True, choices=list(IDEAS))
    ap.add_argument("--L", type=int, required=True, choices=[2, 16])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--adam_steps", type=int, default=None)
    ap.add_argument("--adam_lr", type=float, default=None)
    ap.add_argument("--lbfgs_rounds", type=int, default=None)
    ap.add_argument("--lbfgs_iter", type=int, default=8)
    ap.add_argument("--batch_per_n", type=int, default=32)
    ap.add_argument("--rho", type=float, default=0.1)
    ap.add_argument("--p0", type=float, default=0.0)
    ap.add_argument("--freeze_p", action="store_true", help="keep p fixed at --p0 (power_p)")
    ap.add_argument("--max_train", type=int, default=None)
    ap.add_argument("--lam0", type=float, default=None)
    ap.add_argument("--lam1", type=float, default=None)
    ap.add_argument("--lam_p", type=float, default=1e-2)
    ap.add_argument("--holdout_frac", type=float, default=0.2)
    ap.add_argument("--soft_med", action="store_true")
    ap.add_argument("--w6", type=float, default=None)
    ap.add_argument("--w9", type=float, default=None)
    ap.add_argument("--eval_only", type=str, default="")
    ap.add_argument("--skip_eval", action="store_true")
    ap.add_argument("--tag", type=str, default="", help="subdir tag under idea/")
    args = ap.parse_args()

    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    os.environ["MKL_NUM_THREADS"] = str(args.threads)
    torch.set_num_threads(args.threads)
    device = torch.device("cpu")

    idea = args.idea
    if idea in ("ratio69_w9", "power_p"):
        adam_steps = 600 if args.adam_steps is None else args.adam_steps
        adam_lr = 1e-3 if args.adam_lr is None else args.adam_lr
        lbfgs_rounds = 12 if args.lbfgs_rounds is None else args.lbfgs_rounds
        lam0 = 1e-1 if args.lam0 is None else args.lam0
        lam1 = 1e-3 if args.lam1 is None else args.lam1
        soft_med = bool(args.soft_med)
        w6 = 1.0 if args.w6 is None else args.w6
        w9 = 2.0 if args.w9 is None else args.w9
        rho = None
    elif idea == "trust69":
        adam_steps = 600 if args.adam_steps is None else args.adam_steps
        adam_lr = 5e-4 if args.adam_lr is None else args.adam_lr
        lbfgs_rounds = 8 if args.lbfgs_rounds is None else args.lbfgs_rounds
        lam0 = 0.0 if args.lam0 is None else args.lam0
        lam1 = 0.0 if args.lam1 is None else args.lam1
        soft_med = True if args.soft_med or True else False  # default soft-med for idea3
        soft_med = True
        w6 = 1.0 if args.w6 is None else args.w6
        w9 = 2.0 if args.w9 is None else args.w9
        rho = float(args.rho)
    else:  # ratio69
        adam_steps = 600 if args.adam_steps is None else args.adam_steps
        adam_lr = 1e-3 if args.adam_lr is None else args.adam_lr
        lbfgs_rounds = 12 if args.lbfgs_rounds is None else args.lbfgs_rounds
        lam0 = 1e-1 if args.lam0 is None else args.lam0
        lam1 = 1e-3 if args.lam1 is None else args.lam1
        soft_med = bool(args.soft_med)
        w6 = 1.0 if args.w6 is None else args.w6
        w9 = 1.0 if args.w9 is None else args.w9
        rho = None

    weights = {6: float(w6), 9: float(w9)}
    out_dir = OUT_ROOT / idea
    if args.tag:
        out_dir = out_dir / args.tag

    if args.eval_only:
        state = torch.load(args.eval_only, map_location="cpu", weights_only=False)
        rows = eval_state(state, device, EVAL_NS)
        write_eval(out_dir, state.get("idea", idea), int(state.get("L", args.L)), state.get("seed", "x"), rows, {"from": args.eval_only})
        return 0

    state, meta = train_one(
        idea=idea,
        L=args.L,
        seed=args.seed,
        device=device,
        adam_steps=adam_steps,
        adam_lr=adam_lr,
        lbfgs_rounds=lbfgs_rounds,
        lbfgs_iter=args.lbfgs_iter,
        batch_per_n=args.batch_per_n,
        lam0=lam0,
        lam1=lam1,
        lam_p=args.lam_p,
        rho=rho,
        p0=args.p0,
        freeze_p=bool(args.freeze_p),
        max_train=args.max_train,
        holdout_frac=args.holdout_frac if idea != "ratio69" else 0.0,
        soft_med=soft_med,
        weights=weights,
        out_dir=out_dir,
    )
    if not args.skip_eval:
        rows = eval_state(state, device, EVAL_NS)
        write_eval(out_dir, idea, args.L, args.seed, rows, {"train": meta})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
