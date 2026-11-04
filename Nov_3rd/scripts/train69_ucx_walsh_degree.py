#!/usr/bin/env python3
"""Train{6,9} Walsh-degree-graded γ (Opus Idea 1) — graded H_c, not non-Hc.

U_c^X = W · exp(-i Σ_{w=1..4} γ_{ℓ,w} E_w) · W
Warm: γ_{ℓ,w} = γ_QAOA[ℓ] for all w (recovers scalar champ).

Decisive: L=2, train{6,9}, CORE {6,9,12,15}, BEAT R≤0.98.
Default smoke uses --max_train/--eval_B=100 for fast relative ranking.
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
from Nov_3rd.data_io import load_qaoa, load_qaoa_scale_ref, load_split_E, log
from beat_params import phi_from_c, warm_from_qaoa
from explore_OCT29.lib.qaoa import qaoa_psucc, inv_p_from_psucc

OUT_ROOT = S.OUT / "beat_qaoa_12h" / "train69" / "ucx_bit_a"
TRAIN_PACKS = S.OUT / "beat_qaoa_12h" / "train_packs"
TRAIN_NS = [6, 9]
EVAL_NS = [6, 9, 12, 15]
IDEA = "walsh_degree"  # graded H_c


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


def soft_quantile(vals: torch.Tensor, q: float = 0.5, temp: float = 0.05) -> torch.Tensor:
    v = vals.reshape(-1)
    if v.numel() == 1:
        return v[0]
    order = torch.argsort(v)
    vs = v[order]
    mid = float(q) * (v.numel() - 1)
    idx = torch.arange(v.numel(), device=v.device, dtype=v.dtype)
    w = torch.softmax(-((idx - mid) / max(temp * v.numel(), 1e-3)).abs(), dim=0)
    return (w * vs).sum()


def soft_median(vals: torch.Tensor, temp: float = 0.05) -> torch.Tensor:
    return soft_quantile(vals, q=0.5, temp=temp)


def a_of_n(use_power_p: bool, n: int, p) -> torch.Tensor | float:
    if not use_power_p:
        return 1.0
    base = 8.0 / float(n - 1)
    if torch.is_tensor(p):
        return torch.pow(torch.tensor(base, device=p.device, dtype=p.dtype), p)
    return float(base) ** float(p)


def angles_for_n(gamma_w, c1, c2, c3, p, n: int, use_power_p: bool):
    a = a_of_n(use_power_p, n, p)
    if torch.is_tensor(a):
        c2n, c3n = c2 * a, c3 * a
    else:
        c2n, c3n = c2 * float(a), c3 * float(a)
    return gamma_w, phi_from_c(c1, c2n, c3n)


def instance_log_ratios(sim, E, inv_q, gw, phi, idx) -> torch.Tensor:
    ls = []
    for j in idx:
        out = sim.forward_nov3rd(E[int(j)], gw, phi)
        ratio = out["inv_p"].clamp(min=1e-12) / inv_q[int(j)].to(out["inv_p"].device).clamp(min=1e-12)
        ls.append(torch.log(ratio.clamp(min=1e-12)))
    return torch.stack(ls)


def instance_inv_ps(sim, E, gw, phi, idx) -> torch.Tensor:
    invs = []
    for j in idx:
        out = sim.forward_nov3rd(E[int(j)], gw, phi)
        invs.append(out["inv_p"].clamp(min=1e-12))
    return torch.stack(invs)


def batch_loss(
    bundles,
    gamma_w,
    c1,
    c2,
    c3,
    p,
    idx_map,
    lam_c: float,
    weights: dict[int, float],
    *,
    use_power_p: bool,
    soft_med: bool = False,
    lam_p: float = 0.0,
    loss_mode: str = "log_ratio",
    qmed_map: dict | None = None,
    soft_q: float = 0.5,
):
    terms = []
    wsum = 0.0
    for n, (E, inv_q, sim) in bundles.items():
        idx = idx_map[n]
        gw, phi = angles_for_n(gamma_w, c1, c2, c3, p, n, use_power_p)
        w = float(weights.get(n, 1.0))
        if loss_mode == "med_align":
            invs = instance_inv_ps(sim, E, gw, phi, idx)
            sm = soft_quantile(invs, q=soft_q) if soft_med else invs.median()
            if qmed_map is not None and n in qmed_map:
                qmed = qmed_map[n]
            else:
                qmed = inv_q.detach().median()
            if not torch.is_tensor(qmed):
                qmed = torch.tensor(float(qmed), device=sm.device, dtype=sm.dtype)
            else:
                qmed = qmed.to(device=sm.device, dtype=sm.dtype)
            tn = torch.log((sm / qmed.clamp(min=1e-12)).clamp(min=1e-12))
        else:
            logs = instance_log_ratios(sim, E, inv_q, gw, phi, idx)
            tn = soft_median(logs) if soft_med else logs.mean()
        terms.append(w * tn)
        wsum += w
        if use_power_p and lam_c > 0:
            a = a_of_n(True, n, p)
            terms.append(lam_c * w * ((c2 * a).pow(2).mean() + (c3 * a).pow(2).mean()))
            wsum += w
    loss = torch.stack(terms).sum() / max(wsum, 1e-12)
    if (not use_power_p) and lam_c > 0:
        loss = loss + lam_c * (c2.pow(2).mean() + c3.pow(2).mean())
    if use_power_p and lam_p > 0 and p is not None:
        loss = loss + lam_p * p.pow(2)
    return loss


@torch.no_grad()
def holdout_score(
    bundle_n9, gamma_w, c1, c2, c3, p, hold_idx,
    *, use_power_p, soft_med=False, loss_mode="log_ratio", qmed9=None, soft_q=0.5,
) -> float:
    E, inv_q, sim = bundle_n9
    gw, phi = angles_for_n(gamma_w, c1, c2, c3, p, 9, use_power_p)
    if loss_mode == "med_align":
        invs = instance_inv_ps(sim, E, gw, phi, hold_idx)
        sm = soft_quantile(invs, q=soft_q) if soft_med else invs.median()
        qmed = inv_q.median() if qmed9 is None else qmed9
        if not torch.is_tensor(qmed):
            qmed = torch.tensor(float(qmed), device=sm.device, dtype=sm.dtype)
        return float(torch.log((sm / qmed.clamp(min=1e-12)).clamp(min=1e-12)).detach())
    logs = instance_log_ratios(sim, E, inv_q, gw, phi, hold_idx)
    if soft_med:
        return float(soft_median(logs).detach())
    return float(logs.median().detach())


def snapshot_params(gamma_w, c1, c2, c3, p):
    return {
        "gamma_w": gamma_w.detach().cpu().clone(),
        "c1": c1.detach().cpu().clone(),
        "c2": c2.detach().cpu().clone(),
        "c3": c3.detach().cpu().clone(),
        "p": None if p is None else float(p.detach().cpu()),
    }


def gamma_spread(gamma_w: torch.Tensor) -> dict:
    """Diagnostics: do γ_w drift apart from equal?"""
    gw = gamma_w.detach().cpu().float()
    # per layer std across w, and mean |γ_w - mean_w|
    layer_std = gw.std(dim=1).tolist()
    mean_w = gw.mean(dim=1, keepdim=True)
    mean_abs_dev = (gw - mean_w).abs().mean().item()
    return {
        "layer_std": layer_std,
        "mean_abs_dev": float(mean_abs_dev),
        "gamma_w": gw.tolist(),
    }


@torch.no_grad()
def eval_n_B(n: int, L: int, gamma_w, phi, device, *, max_count: int) -> dict:
    E, key = load_split_E(n, "scale1000", max_count, device, data_dir=S.SCALE1000_DIR)
    sim = make_sim(n, L, device)
    invs, leaks = [], []
    t0 = time.time()
    B = len(E)
    for i in range(B):
        out = sim.forward_nov3rd(E[i], gamma_w, phi)
        invs.append(float(out["inv_p"].cpu()))
        leaks.append(float(out["p_leak"].cpu()))
    invs = np.asarray(invs, dtype=np.float64)
    return {
        "n": n,
        "split": key,
        "B": B,
        "median_inv_p": float(np.median(invs)),
        "B90_inv_p": float(np.quantile(invs, 0.90)),
        "mean_leak": float(np.mean(leaks)),
        "L": L,
        "wall_s": time.time() - t0,
    }


@torch.no_grad()
def eval_qaoa_med_B(n: int, L: int, betas, gammas, device, *, max_count: int) -> float:
    """QAOA Med on the same first-B scale1000 slice (fair R at B<1000)."""
    E, _ = load_split_E(n, "scale1000", max_count, device, data_dir=S.SCALE1000_DIR)
    inv = precompute_qaoa_inv(betas, gammas, E.cpu(), n, device)
    return float(inv.median().cpu())


def train_one(args, device):
    seed = args.seed
    L = args.L
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    betas, gammas = load_qaoa(L)
    g0, c10, c20, c30 = [t.to(device) for t in warm_from_qaoa(betas, gammas)]
    # warm γ_w = γ for all w
    gamma_w = torch.nn.Parameter(g0.reshape(L, 1).expand(L, 4).contiguous().clone())
    c1 = torch.nn.Parameter(c10.clone())
    if args.power_p:
        c2 = torch.nn.Parameter(torch.zeros_like(c20) + 1e-4)
        c3 = torch.nn.Parameter(torch.zeros_like(c30) + 1e-4)
        p = torch.nn.Parameter(torch.tensor(float(args.p0), device=device))
        trainables = [gamma_w, c1, c2, c3, p]
    else:
        c2 = torch.nn.Parameter(c20.clone())
        c3 = torch.nn.Parameter(c30.clone())
        p = None
        trainables = [gamma_w, c1, c2, c3]

    use_power_p = bool(args.power_p)
    bundles = {}
    for n in TRAIN_NS:
        E = load_train_n(n, max_count=args.max_train).to(device)
        log(f"{IDEA} L{L} seed{seed}: load train n{n} B={len(E)}")
        inv_q = precompute_qaoa_inv(betas, gammas, E.cpu(), n, device)
        sim = make_sim(n, L, device)
        bundles[n] = (E, inv_q.to(device), sim)

    qmed_map = {n: float(bundles[n][1].detach().median().cpu()) for n in bundles}
    log(f"{IDEA} L{L} s{seed}: train QAOA Meds " + ", ".join(f"n{n}={qmed_map[n]:.4f}" for n in sorted(qmed_map)))

    n9_B = len(bundles[9][0])
    perm = rng.permutation(n9_B)
    n_hold = max(1, int(round(args.holdout_frac * n9_B)))
    hold_idx = perm[:n_hold]
    train_idx9_pool = perm[n_hold:]
    log(f"{IDEA} L{L} s{seed}: n9 holdout={n_hold} train_pool={len(train_idx9_pool)}")

    weights = {6: float(args.w6), 9: float(args.w9)}
    hist = []
    t0 = time.time()
    best = {"score": float("inf"), "snap": snapshot_params(gamma_w, c1, c2, c3, p), "step": -1}

    def maybe_update_best(step_tag: str):
        sc = holdout_score(
            bundles[9], gamma_w, c1, c2, c3, p, hold_idx,
            use_power_p=use_power_p, soft_med=args.soft_med,
            loss_mode=args.loss_mode, qmed9=qmed_map.get(9), soft_q=args.soft_q,
        )
        hist.append({"phase": "holdout", "tag": step_tag, "hold_med_logR": sc})
        if sc < best["score"]:
            best["score"] = sc
            best["snap"] = snapshot_params(gamma_w, c1, c2, c3, p)
            best["step"] = step_tag
            log(f"{IDEA} L{L} s{seed} NEW BEST holdout={sc:.5f} @ {step_tag}")

    opt = torch.optim.Adam(trainables, lr=args.adam_lr)
    for step in range(args.adam_steps):
        frac = step / max(args.adam_steps - 1, 1)
        lam = args.lam0 * (1 - frac) + args.lam1 * frac
        idx_map = {}
        for n in TRAIN_NS:
            if n == 9 and len(train_idx9_pool):
                take = min(args.batch_per_n, len(train_idx9_pool))
                idx_map[n] = rng.choice(train_idx9_pool, size=take, replace=False)
            else:
                idx_map[n] = rng.choice(
                    len(bundles[n][0]), size=min(args.batch_per_n, len(bundles[n][0])), replace=False
                )
        opt.zero_grad()
        loss = batch_loss(
            bundles, gamma_w, c1, c2, c3, p, idx_map, lam, weights,
            use_power_p=use_power_p, soft_med=args.soft_med, lam_p=args.lam_p,
            loss_mode=args.loss_mode, qmed_map=qmed_map, soft_q=args.soft_q,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainables, 5.0)
        opt.step()
        if use_power_p and p is not None:
            with torch.no_grad():
                p.clamp_(-1.0, 2.0)
        hist.append({"phase": "adam", "step": step, "loss": float(loss.detach()), "lam": lam})
        if step % 25 == 0 or step == args.adam_steps - 1:
            extra = f" p={float(p):.3f}" if p is not None else ""
            sp = gamma_spread(gamma_w)
            log(
                f"{IDEA} L{L} s{seed} adam {step}/{args.adam_steps} "
                f"loss={hist[-1]['loss']:.5f} lam={lam:.2e} "
                f"γ_mad={sp['mean_abs_dev']:.4f}{extra}"
            )
            maybe_update_best(f"adam{step}")

    for r in range(args.lbfgs_rounds):
        opt_l = torch.optim.LBFGS(
            trainables, lr=0.25, max_iter=args.lbfgs_iter, history_size=16, line_search_fn="strong_wolfe"
        )
        idx_map = {}
        for n in TRAIN_NS:
            if n == 9 and len(train_idx9_pool):
                take = min(args.batch_per_n, len(train_idx9_pool))
                idx_map[n] = rng.choice(train_idx9_pool, size=take, replace=False)
            else:
                idx_map[n] = rng.choice(
                    len(bundles[n][0]), size=min(args.batch_per_n, len(bundles[n][0])), replace=False
                )
        lam = args.lam1

        def closure():
            opt_l.zero_grad()
            lt = batch_loss(
                bundles, gamma_w, c1, c2, c3, p, idx_map, lam, weights,
                use_power_p=use_power_p, soft_med=args.soft_med, lam_p=args.lam_p,
                loss_mode=args.loss_mode, qmed_map=qmed_map, soft_q=args.soft_q,
            )
            lt.backward()
            torch.nn.utils.clip_grad_norm_(trainables, 5.0)
            return lt

        try:
            loss = opt_l.step(closure)
            loss_f = float(loss.detach()) if torch.is_tensor(loss) else float(loss)
        except Exception as e:
            log(f"{IDEA} L{L} s{seed} LBFGS r{r} skip: {e}")
            continue
        if use_power_p and p is not None:
            with torch.no_grad():
                p.clamp_(-1.0, 2.0)
        hist.append({"phase": "lbfgs", "round": r, "loss": loss_f, "lam": lam})
        log(f"{IDEA} L{L} s{seed} lbfgs r{r}/{args.lbfgs_rounds} loss={loss_f:.5f}")
        maybe_update_best(f"lbfgs{r}")

    if best["score"] < float("inf"):
        log(f"{IDEA} L{L} s{seed}: restoring best holdout={best['score']:.5f} @ {best['step']}")
        snap = best["snap"]
        with torch.no_grad():
            gamma_w.data.copy_(snap["gamma_w"].to(device))
            c1.data.copy_(snap["c1"].to(device))
            c2.data.copy_(snap["c2"].to(device))
            c3.data.copy_(snap["c3"].to(device))
            if p is not None and snap["p"] is not None:
                p.data.fill_(snap["p"])

    spread = gamma_spread(gamma_w)
    state = {
        "idea": IDEA,
        "label": "graded_Hc_walsh_degree",
        "L": L,
        "seed": seed,
        "train_ns": list(TRAIN_NS),
        "gamma_w": gamma_w.detach().cpu(),
        "c1": c1.detach().cpu(),
        "c2": c2.detach().cpu(),
        "c3": c3.detach().cpu(),
        "p": None if p is None else float(p.detach().cpu()),
        "weights": weights,
        "holdout_frac": args.holdout_frac,
        "holdout_best": best["score"],
        "holdout_best_step": best["step"],
        "betas_qaoa": betas.cpu(),
        "gammas_qaoa": gammas.cpu(),
        "wall_s": time.time() - t0,
        "hist_tail": hist[-40:],
        "recipe": S.RECIPE_NAME,
        "motif": S.MOTIF,
        "snap_head": S.SNAP_HEAD,
        "lock": "train69_only",
        "soft_med": args.soft_med,
        "loss_mode": args.loss_mode,
        "soft_q": args.soft_q,
        "max_train": args.max_train,
        "eval_B": args.eval_B,
        "power_p": use_power_p,
        "gamma_spread": spread,
    }
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"B{args.eval_B}"
    pt = out_dir / f"params_L{L}_seed{seed}_{tag}.pt"
    torch.save(state, pt)
    meta = {
        "idea": IDEA,
        "label": "graded_Hc_walsh_degree",
        "L": L,
        "seed": seed,
        "p": state["p"],
        "wall_s": state["wall_s"],
        "holdout_best": state["holdout_best"],
        "holdout_best_step": state["holdout_best_step"],
        "gamma_spread": spread,
        "max_train": args.max_train,
        "eval_B": args.eval_B,
        "params": str(pt),
    }
    (out_dir / f"train_L{L}_seed{seed}_{tag}.json").write_text(json.dumps(meta, indent=2) + "\n")
    log(f"{IDEA} L{L} s{seed} saved {pt} wall={state['wall_s']:.0f}s γ_mad={spread['mean_abs_dev']:.4f}")
    return state, meta, pt


@torch.no_grad()
def eval_state(state, device, *, eval_B: int, use_scale_ref: bool = False):
    L = int(state["L"])
    gamma_w = state["gamma_w"].to(device)
    c1 = state["c1"].to(device)
    c2 = state["c2"].to(device)
    c3 = state["c3"].to(device)
    p = state.get("p")
    if p is not None:
        p = torch.tensor(float(p), device=device)
    use_power_p = bool(state.get("power_p", False))
    betas = state["betas_qaoa"]
    gammas = state["gammas_qaoa"]
    qref = load_qaoa_scale_ref(L) if use_scale_ref and eval_B >= 1000 else {}
    rows = {}
    for n in EVAL_NS:
        gw, phi = angles_for_n(gamma_w, c1, c2, c3, p, n, use_power_p)
        m = eval_n_B(n, L, gw, phi, device, max_count=eval_B)
        if eval_B >= 1000 and n in qref and qref[n]:
            qmed = float(qref[n].get("Med", float("nan")))
            qb90 = float(qref[n].get("B90", float("nan"))) if "B90" in qref[n] else float("nan")
        else:
            qmed = eval_qaoa_med_B(n, L, betas, gammas, device, max_count=eval_B)
            qb90 = float("nan")
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
            "eval_wall_s": float(m["wall_s"]),
        }
        log(f"eval {IDEA} L{L} n{n} B={m['B']}: Med={med:.4f} Q={qmed:.4f} R={R:.4f} {v} ({m['wall_s']:.1f}s)")
    return rows


def write_eval(out_dir: Path, L: int, seed, rows: dict, extra: dict | None = None, tag: str = ""):
    payload = {
        "idea": IDEA,
        "label": "graded_Hc_walsh_degree",
        "L": L,
        "seed": seed,
        "train_ns": list(TRAIN_NS),
        "per_n": {str(k): v for k, v in rows.items()},
        "finished": time.strftime("%Y-%m-%d %H:%M:%S %z"),
    }
    if extra:
        payload.update(extra)
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{tag}" if tag else ""
    path = out_dir / f"eval_L{L}_seed{seed}{suffix}.json"
    path.write_text(json.dumps(payload, indent=2) + "\n")
    lines = [
        f"# {IDEA} (graded H_c) L={L} seed={seed} train{{6,9}} {tag}",
        "",
        "| n | QAOA Med | Med | R | verdict | B |",
        "|---|----------|-----|---|---------|---|",
    ]
    for n in EVAL_NS:
        r = rows[n]
        lines.append(
            f"| {n} | {r['QAOA_Med']:.4f} | {r['Med']:.4f} | {r['R']:.4f} | {r['verdict']} | {r['B']} |"
        )
    all_beat = all(rows[n]["verdict"] == "BEAT" for n in EVAL_NS)
    knee = max(rows[n]["R"] for n in EVAL_NS)
    lines += ["", f"**ALL_CORE_BEAT** = {all_beat}", f"**knee R (max)** = {knee:.4f}", ""]
    (out_dir / f"eval_L{L}_seed{seed}{suffix}.md").write_text("\n".join(lines) + "\n")
    log(f"wrote {path} ALL_CORE_BEAT={all_beat} knee={knee:.4f}")
    return all_beat, knee, path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=int, default=2, choices=[2, 16])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--adam_steps", type=int, default=400)
    ap.add_argument("--adam_lr", type=float, default=1e-3)
    ap.add_argument("--lbfgs_rounds", type=int, default=8)
    ap.add_argument("--lbfgs_iter", type=int, default=8)
    ap.add_argument("--batch_per_n", type=int, default=32)
    ap.add_argument("--max_train", type=int, default=100, help="train pack truncate (smoke default 100)")
    ap.add_argument("--eval_B", type=int, default=100, help="CORE eval sample count (smoke default 100)")
    ap.add_argument("--lam0", type=float, default=1e-1)
    ap.add_argument("--lam1", type=float, default=1e-3)
    ap.add_argument("--lam_p", type=float, default=1e-2)
    ap.add_argument("--holdout_frac", type=float, default=0.2)
    ap.add_argument("--soft_med", action="store_true")
    ap.add_argument("--loss", dest="loss_mode", default="log_ratio", choices=["log_ratio", "med_align"])
    ap.add_argument("--soft_q", type=float, default=0.5)
    ap.add_argument("--w6", type=float, default=1.0)
    ap.add_argument("--w9", type=float, default=2.0)
    ap.add_argument("--power_p", action="store_true")
    ap.add_argument("--p0", type=float, default=0.0)
    ap.add_argument("--eval_only", type=str, default="")
    ap.add_argument("--skip_eval", action="store_true")
    ap.add_argument("--tag", type=str, default="walsh_degree")
    ap.add_argument("--out_dir", type=str, default="")
    args = ap.parse_args()

    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    os.environ["MKL_NUM_THREADS"] = str(args.threads)
    torch.set_num_threads(args.threads)
    device = torch.device("cpu")

    out_dir = Path(args.out_dir) if args.out_dir else (OUT_ROOT / args.tag)
    args.out_dir = str(out_dir)

    if args.eval_only:
        state = torch.load(args.eval_only, map_location="cpu", weights_only=False)
        rows = eval_state(state, device, eval_B=args.eval_B)
        write_eval(out_dir, int(state.get("L", args.L)), state.get("seed", "x"), rows,
                   {"from": args.eval_only}, tag=f"B{args.eval_B}")
        return 0

    state, meta, pt = train_one(args, device)
    if not args.skip_eval:
        rows = eval_state(state, device, eval_B=args.eval_B)
        write_eval(out_dir, args.L, args.seed, rows, {"train": meta, "params": str(pt)},
                   tag=f"B{args.eval_B}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
