#!/usr/bin/env python3
"""Idea1 — Walsh-degree-graded γ (graded H_c). L=2, train{6,9}, B=100 smoke.

U_c^X = W exp(-i Σ_{w=1..4} γ_{ℓ,w} E_w) W
E_w = degree-w Walsh component of unsat E (FWHT → popcount mask → iFWHT).
Warm: all γ_w = γ_champ (or QAOA γ). Shell φ_k = c1 k + c2 k² + c3 k³.

Eval CORE {6,9,12,15} on B=100 (same-sample QAOA Med). BEAT iff R≤0.98.
This is the L=2 capacity near-miss hunt (knee ≈0.9858 on B=1000).
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
from Nov_3rd.graded_cost import WALSH_DEGREES, cn_parts_from_E
from explore_OCT29.lib.qaoa import inv_p_from_psucc, qaoa_psucc
from explore_OCT29.lib.sat_gen import clause_count, instances, pack_clauses

from beat_params import phi_from_c, warm_from_qaoa

OUT_ROOT = S.OUT / "beat_qaoa_12h" / "train69" / "ucx_bit_a"
TRAIN_PACKS = S.OUT / "beat_qaoa_12h" / "train_packs"
TRAIN_NS = [6, 9]
EVAL_NS = [6, 9, 12, 15]
CHAMP_L2 = (
    S.PKG / "runs" / "beat_qaoa_12h" / "train69" / "champion" / "params_L2_train69_pareto_a080_s1.pt"
)
ARM = "graded_Hc_walsh_degree"
SMOKE_B = 100
K_W = len(WALSH_DEGREES)


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return float(o)
    if isinstance(o, (np.integer, int)):
        return int(o)
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, torch.Tensor):
        return o.detach().cpu().tolist()
    if o is None:
        return None
    return o


def ensure_eval_pack(n: int, B: int) -> Path:
    """B-sample CORE pack (SCB=54e9, same generator as scale1000 regen)."""
    out_dir = S.PKG / "data" / "scale1000"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"data_n{n}.npz"
    if path.exists():
        d = np.load(path, allow_pickle=True)
        key = "scale1000_E" if "scale1000_E" in d.files else "test_E"
        if key in d.files and len(d[key]) >= B:
            return path
    log(f"generating scale1000 n{n} B={B} (SCB=54e9)")
    t0 = time.time()
    m = clause_count(n)
    ins = instances(n, B, 54_000_000_000, compute_E=True)
    seeds = np.array([i["seed"] for i in ins], np.int64)
    nsol = np.array([i["nsol"] for i in ins], np.int32)
    cv, cs = pack_clauses([i["cl"] for i in ins], m, n)
    E = np.stack([i["E"] for i in ins]).astype(np.uint8)
    sp = {}
    for p in ("scale1000", "test"):
        sp.update(
            {
                f"{p}_seeds": seeds,
                f"{p}_nsol": nsol,
                f"{p}_cls_v": cv,
                f"{p}_cls_s": cs,
                f"{p}_E": E,
            }
        )
    np.savez_compressed(path, n=np.int32(n), m=np.int32(m), store_E=np.array([1], np.int32), **sp)
    log(f"wrote {path} B={B} wall={time.time()-t0:.0f}s")
    return path


def load_train_n(n: int, max_count: int) -> torch.Tensor:
    path = TRAIN_PACKS / f"train_n{n}.npz"
    d = np.load(path, allow_pickle=True)
    E = np.asarray(d["train_E"])[: int(max_count)]
    return torch.as_tensor(E, dtype=torch.float32)


def load_eval_E(n: int, max_count: int, device) -> torch.Tensor:
    ensure_eval_pack(n, max_count)
    from Nov_3rd.data_io import load_split_E

    E, _ = load_split_E(n, "scale1000", max_count, device, data_dir=S.SCALE1000_DIR)
    return E


def precompute_qaoa_inv(betas, gammas, E: torch.Tensor, n: int, device) -> torch.Tensor:
    invs = []
    B = E.shape[0]
    chunk = 64 if n <= 9 else (16 if n <= 12 else 4)
    for s0 in range(0, B, chunk):
        Eb = E[s0 : s0 + chunk].to(device)
        ps = qaoa_psucc(betas.to(device), gammas.to(device), Eb, n)
        invs.append(inv_p_from_psucc(ps.detach().cpu().numpy()))
    return torch.as_tensor(np.concatenate(invs), dtype=torch.float32)


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


def soft_median(vals: torch.Tensor, temp: float = 0.05) -> torch.Tensor:
    v = vals.reshape(-1)
    if v.numel() == 1:
        return v[0]
    order = torch.argsort(v)
    vs = v[order]
    mid = 0.5 * (v.numel() - 1)
    idx = torch.arange(v.numel(), device=v.device, dtype=v.dtype)
    w = torch.softmax(-((idx - mid) / max(temp * v.numel(), 1e-3)).abs(), dim=0)
    return (w * vs).sum()


def precompute_parts(sim, E: torch.Tensor) -> torch.Tensor:
    n = sim.cfg.n
    return cn_parts_from_E(E.to(sim.lam.device), n, sim.grid_to_x)


def instance_log_ratios(sim, E, parts, inv_q, gw, phi, idx) -> torch.Tensor:
    ls = []
    for j in idx:
        out = sim.forward_nov3rd_graded(E[int(j)], gw, phi, parts[int(j)])
        ratio = out["inv_p"].clamp(min=1e-12) / inv_q[int(j)].to(out["inv_p"].device).clamp(min=1e-12)
        ls.append(torch.log(ratio.clamp(min=1e-12)))
    return torch.stack(ls)


def batch_loss(bundles, gamma_w, c1, c2, c3, p, idx_map, lam_c, weights, use_power_p, lam_p, soft_med):
    terms = []
    wsum = 0.0
    for n, (E, parts, inv_q, sim) in bundles.items():
        idx = idx_map[n]
        gw, phi = angles_for_n(gamma_w, c1, c2, c3, p, n, use_power_p)
        w = float(weights.get(n, 1.0))
        logs = instance_log_ratios(sim, E, parts, inv_q, gw, phi, idx)
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
def holdout_score(bundle_n9, gamma_w, c1, c2, c3, p, hold_idx, use_power_p, soft_med) -> float:
    E, parts, inv_q, sim = bundle_n9
    gw, phi = angles_for_n(gamma_w, c1, c2, c3, p, 9, use_power_p)
    logs = instance_log_ratios(sim, E, parts, inv_q, gw, phi, hold_idx)
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


def load_champ_or_qaoa(L: int, device, warm: str):
    betas, gammas = load_qaoa(L)
    g0, c10, c20, c30 = [t.to(device) for t in warm_from_qaoa(betas, gammas)]
    src = "qaoa"
    if warm == "champ" and CHAMP_L2.exists() and L == 2:
        st = torch.load(CHAMP_L2, map_location="cpu", weights_only=False)
        if isinstance(st, dict) and "gamma" in st:
            g0 = st["gamma"].to(device).float().reshape(L)
            c10 = st["c1"].to(device).float().reshape(L)
            c20 = st["c2"].to(device).float().reshape(L)
            c30 = st["c3"].to(device).float().reshape(L)
            src = str(CHAMP_L2)
            log(f"warm from champ {CHAMP_L2.name}")
        else:
            log("champ pt missing gamma; falling back to QAOA")
    elif warm == "champ":
        log(f"champ warm requested but {CHAMP_L2} missing or L!={2}; QAOA warm")
    gamma_w = g0.reshape(L, 1).expand(L, K_W).contiguous().clone()
    return betas, gammas, gamma_w, c10, c20, c30, src


def make_params(L, device, warm: str, p0: float, use_power_p: bool, freeze_p: bool):
    betas, gammas, gamma_w0, c10, c20, c30, src = load_champ_or_qaoa(L, device, warm)
    gamma_w = torch.nn.Parameter(gamma_w0)
    c1 = torch.nn.Parameter(c10.clone())
    c2 = torch.nn.Parameter(c20.clone() if c20.abs().sum() > 0 else torch.zeros_like(c20) + 1e-4)
    c3 = torch.nn.Parameter(c30.clone() if c30.abs().sum() > 0 else torch.zeros_like(c30) + 1e-4)
    if use_power_p:
        if freeze_p:
            p = torch.tensor(float(p0), device=device)
            trainables = [gamma_w, c1, c2, c3]
        else:
            p = torch.nn.Parameter(torch.tensor(float(p0), device=device))
            trainables = [gamma_w, c1, c2, c3, p]
    else:
        p = None
        trainables = [gamma_w, c1, c2, c3]
    return betas, gammas, gamma_w, c1, c2, c3, p, trainables, src


def gamma_spread(gamma_w: torch.Tensor) -> dict:
    g = gamma_w.detach().cpu()
    per_layer = []
    for ell in range(g.shape[0]):
        row = g[ell]
        per_layer.append(
            {
                "mean": float(row.mean()),
                "std": float(row.std(unbiased=False)),
                "max_abs_dev": float((row - row.mean()).abs().max()),
                "values": [float(x) for x in row],
            }
        )
    flat_std = float((g - g.mean(dim=1, keepdim=True)).pow(2).mean().sqrt())
    return {"per_layer": per_layer, "rms_dev_from_equal": flat_std}


def train_one(args, device, out_dir: Path):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    L = int(args.L)

    betas, gammas, gamma_w, c1, c2, c3, p, trainables, warm_src = make_params(
        L, device, args.warm, args.p0, args.power_p, args.freeze_p
    )

    bundles = {}
    for n in TRAIN_NS:
        E = load_train_n(n, args.max_train).to(device)
        log(f"{ARM} L{L} seed{args.seed}: load train n{n} B={len(E)}")
        inv_q = precompute_qaoa_inv(betas, gammas, E.cpu(), n, device).to(device)
        sim = make_sim(n, L, device)
        parts = precompute_parts(sim, E)
        bundles[n] = (E, parts, inv_q, sim)

    n9_B = len(bundles[9][0])
    perm = rng.permutation(n9_B)
    n_hold = max(1, int(round(args.holdout_frac * n9_B)))
    hold_idx = perm[:n_hold]
    train_idx9_pool = perm[n_hold:]
    log(f"{ARM} L{L} s{args.seed}: n9 holdout={n_hold} train_pool={len(train_idx9_pool)} warm={warm_src}")

    weights = {6: float(args.w6), 9: float(args.w9)}
    hist = []
    t0 = time.time()
    best = {"score": float("inf"), "snap": snapshot_params(gamma_w, c1, c2, c3, p), "step": -1}

    def maybe_update_best(tag: str):
        sc = holdout_score(
            bundles[9], gamma_w, c1, c2, c3, p, hold_idx, args.power_p, args.soft_med
        )
        hist.append({"phase": "holdout", "tag": tag, "hold_med_logR": sc})
        if sc < best["score"]:
            best["score"] = sc
            best["snap"] = snapshot_params(gamma_w, c1, c2, c3, p)
            best["step"] = tag
            log(f"{ARM} L{L} s{args.seed} NEW BEST holdout={sc:.5f} @ {tag}")

    opt = torch.optim.Adam(trainables, lr=args.adam_lr)
    for step in range(args.adam_steps):
        frac = step / max(args.adam_steps - 1, 1)
        lam = args.lam0 * (1 - frac) + args.lam1 * frac
        idx_map = {}
        for n in TRAIN_NS:
            pool = train_idx9_pool if n == 9 else np.arange(len(bundles[n][0]))
            take = min(args.batch_per_n, len(pool))
            idx_map[n] = rng.choice(pool, size=take, replace=False)
        opt.zero_grad()
        loss = batch_loss(
            bundles, gamma_w, c1, c2, c3, p, idx_map, lam, weights,
            args.power_p, args.lam_p, args.soft_med,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainables, 5.0)
        opt.step()
        if args.power_p and p is not None and isinstance(p, torch.nn.Parameter):
            with torch.no_grad():
                p.clamp_(-1.0, 2.0)
        hist.append({"phase": "adam", "step": step, "loss": float(loss.detach()), "lam": lam})
        if step % 25 == 0 or step == args.adam_steps - 1:
            extra = f" p={float(p):.3f}" if p is not None else ""
            spr = gamma_spread(gamma_w)["rms_dev_from_equal"]
            log(
                f"{ARM} L{L} s{args.seed} adam {step}/{args.adam_steps} "
                f"loss={hist[-1]['loss']:.5f} lam={lam:.2e} γ_rms_dev={spr:.4f}{extra}"
            )
            maybe_update_best(f"adam{step}")

    for r in range(args.lbfgs_rounds):
        opt_l = torch.optim.LBFGS(
            trainables, lr=0.25, max_iter=args.lbfgs_iter, history_size=16, line_search_fn="strong_wolfe"
        )
        idx_map = {}
        for n in TRAIN_NS:
            pool = train_idx9_pool if n == 9 else np.arange(len(bundles[n][0]))
            take = min(args.batch_per_n, len(pool))
            idx_map[n] = rng.choice(pool, size=take, replace=False)
        lam = args.lam1

        def closure():
            opt_l.zero_grad()
            lt = batch_loss(
                bundles, gamma_w, c1, c2, c3, p, idx_map, lam, weights,
                args.power_p, args.lam_p, args.soft_med,
            )
            lt.backward()
            torch.nn.utils.clip_grad_norm_(trainables, 5.0)
            return lt

        try:
            loss = opt_l.step(closure)
            loss_f = float(loss.detach()) if torch.is_tensor(loss) else float(loss)
        except Exception as e:
            log(f"{ARM} L{L} s{args.seed} LBFGS r{r} skip: {e}")
            continue
        if args.power_p and p is not None and isinstance(p, torch.nn.Parameter):
            with torch.no_grad():
                p.clamp_(-1.0, 2.0)
        hist.append({"phase": "lbfgs", "round": r, "loss": loss_f, "lam": lam})
        log(f"{ARM} L{L} s{args.seed} lbfgs r{r}/{args.lbfgs_rounds} loss={loss_f:.5f}")
        maybe_update_best(f"lbfgs{r}")

    if best["score"] < float("inf"):
        log(f"{ARM} L{L} s{args.seed}: restoring best holdout={best['score']:.5f} @ {best['step']}")
        snap = best["snap"]
        with torch.no_grad():
            gamma_w.data.copy_(snap["gamma_w"].to(device))
            c1.data.copy_(snap["c1"].to(device))
            c2.data.copy_(snap["c2"].to(device))
            c3.data.copy_(snap["c3"].to(device))
            if p is not None and snap["p"] is not None and isinstance(p, torch.nn.Parameter):
                p.data.fill_(snap["p"])

    state = {
        "arm": ARM,
        "label": "graded H_c",
        "idea": "walsh_degree",
        "L": L,
        "seed": args.seed,
        "train_ns": list(TRAIN_NS),
        "max_train": args.max_train,
        "gamma_w": gamma_w.detach().cpu(),
        "c1": c1.detach().cpu(),
        "c2": c2.detach().cpu(),
        "c3": c3.detach().cpu(),
        "p": None if p is None else float(p.detach().cpu()),
        "use_power_p": bool(args.power_p),
        "warm": args.warm,
        "warm_src": warm_src,
        "weights": weights,
        "holdout_best": best["score"],
        "holdout_best_step": best["step"],
        "gamma_spread": gamma_spread(gamma_w),
        "betas_qaoa": betas.cpu(),
        "gammas_qaoa": gammas.cpu(),
        "wall_s": time.time() - t0,
        "hist_tail": hist[-30:],
        "recipe": S.RECIPE_NAME,
        "motif": S.MOTIF,
        "snap_head": S.SNAP_HEAD,
        "lock": "train69_only_graded_Hc_walsh",
        "B_eval": args.max_eval,
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    pt = out_dir / f"params_L{L}_seed{args.seed}.pt"
    torch.save(state, pt)
    meta = {
        "arm": ARM,
        "label": "graded H_c",
        "idea": "walsh_degree",
        "L": L,
        "seed": args.seed,
        "p": state["p"],
        "warm": args.warm,
        "warm_src": warm_src,
        "wall_s": state["wall_s"],
        "holdout_best": state["holdout_best"],
        "holdout_best_step": state["holdout_best_step"],
        "gamma_spread": state["gamma_spread"],
        "c2_rms": float(c2.detach().cpu().pow(2).mean().sqrt()),
        "c3_rms": float(c3.detach().cpu().pow(2).mean().sqrt()),
        "params": str(pt),
        "B_train": args.max_train,
        "B_eval": args.max_eval,
    }
    (out_dir / f"train_L{L}_seed{args.seed}.json").write_text(json.dumps(_jsonable(meta), indent=2) + "\n")
    log(f"{ARM} L{L} s{args.seed} saved {pt} wall={state['wall_s']:.0f}s")
    return state, meta


@torch.no_grad()
def eval_state(state, device, ns=EVAL_NS, max_count=SMOKE_B):
    L = int(state["L"])
    gamma_w = state["gamma_w"].to(device)
    c1 = state["c1"].to(device)
    c2 = state["c2"].to(device)
    c3 = state["c3"].to(device)
    p = state.get("p")
    if p is not None:
        p = torch.tensor(float(p), device=device)
    use_power_p = bool(state.get("use_power_p", False))
    betas = state["betas_qaoa"]
    gammas = state["gammas_qaoa"]
    qref1000 = load_qaoa_scale_ref(L)
    rows = {}
    for n in ns:
        E = load_eval_E(n, max_count, device)
        inv_q = precompute_qaoa_inv(betas, gammas, E.cpu(), n, device)
        qmed = float(inv_q.median())
        gw, phi = angles_for_n(gamma_w, c1, c2, c3, p, n, use_power_p)
        sim = make_sim(n, L, device)
        parts = precompute_parts(sim, E)
        invs, leaks = [], []
        t0 = time.time()
        B = len(E)
        for i in range(B):
            out = sim.forward_nov3rd_graded(E[i], gw, phi, parts[i])
            invs.append(float(out["inv_p"].cpu()))
            leaks.append(float(out["p_leak"].cpu()))
        invs = np.asarray(invs, dtype=np.float64)
        med = float(np.median(invs))
        R = med / qmed if qmed > 0 else float("nan")
        q1000 = float((qref1000.get(n) or {}).get("Med", float("nan")))
        R_vs_pub = med / q1000 if q1000 == q1000 and q1000 > 0 else float("nan")
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
            "B90": float(np.quantile(invs, 0.90)),
            "QAOA_Med_B": qmed,
            "QAOA_Med_pub1000": q1000,
            "R": R,
            "R_vs_pub1000": R_vs_pub,
            "verdict": v,
            "mean_leak": float(np.mean(leaks)),
            "B": B,
            "wall_s": time.time() - t0,
        }
        log(
            f"eval {ARM} L{L} n{n} B={B}: Med={med:.4f} Q_B={qmed:.4f} "
            f"R={R:.4f} {v} (R_vs_pub1000={R_vs_pub:.4f})"
        )
    return rows


@torch.no_grad()
def eval_plain_baseline(L: int, device, ns=EVAL_NS, max_count=SMOKE_B, warm="champ"):
    """Plain single-γ forward on the same B-sample (champ or QAOA)."""
    betas, gammas, gamma_w, c1, c2, c3, src = load_champ_or_qaoa(L, device, warm)
    gamma = gamma_w[:, 0].contiguous()
    phi = phi_from_c(c1, c2, c3)
    rows = {}
    for n in ns:
        E = load_eval_E(n, max_count, device)
        inv_q = precompute_qaoa_inv(betas, gammas, E.cpu(), n, device)
        qmed = float(inv_q.median())
        sim = make_sim(n, L, device)
        invs = []
        for i in range(len(E)):
            out = sim.forward_nov3rd(E[i], gamma, phi)
            invs.append(float(out["inv_p"].cpu()))
        med = float(np.median(invs))
        R = med / qmed if qmed > 0 else float("nan")
        v = "BEAT" if R <= 0.98 else ("TIE" if R <= 1.02 else "UNDER")
        rows[n] = {"Med": med, "QAOA_Med_B": qmed, "R": R, "verdict": v, "B": len(E), "warm_src": src}
        log(f"baseline plain-γ L{L} n{n} B={len(E)}: Med={med:.4f} Q={qmed:.4f} R={R:.4f} {v}")
    return rows, src


def write_eval(out_dir: Path, L: int, seed, rows: dict, extra: dict | None = None):
    payload = {
        "arm": ARM,
        "label": "graded H_c",
        "idea": "walsh_degree",
        "L": L,
        "seed": seed,
        "train_ns": list(TRAIN_NS),
        "B": SMOKE_B,
        "per_n": {str(k): v for k, v in rows.items()},
        "finished": time.strftime("%Y-%m-%d %H:%M:%S %z"),
    }
    if extra:
        payload.update(extra)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"eval_L{L}_seed{seed}.json"
    path.write_text(json.dumps(_jsonable(payload), indent=2) + "\n")
    lines = [
        f"# graded H_c Walsh-degree L={L} seed={seed} train{{6,9}} B={rows[EVAL_NS[0]]['B']}",
        "",
        "| n | QAOA Med (B) | Med | R | R vs pub1000 | verdict |",
        "|---|--------------|-----|---|---------------|---------|",
    ]
    for n in EVAL_NS:
        r = rows[n]
        lines.append(
            f"| {n} | {r['QAOA_Med_B']:.4f} | {r['Med']:.4f} | {r['R']:.4f} | "
            f"{r['R_vs_pub1000']:.4f} | {r['verdict']} |"
        )
    all_beat = all(rows[n]["verdict"] == "BEAT" for n in EVAL_NS)
    knee = max(rows[n]["R"] for n in (12, 15))
    lines += ["", f"**ALL_CORE_BEAT** = {all_beat}  (B-sample QAOA Med)", f"**knee max(R12,R15)** = {knee:.4f}", ""]
    (out_dir / f"eval_L{L}_seed{seed}.md").write_text("\n".join(lines) + "\n")
    log(f"wrote {path} ALL_CORE_BEAT={all_beat} knee={knee:.4f}")
    return all_beat, knee, path


def update_results_md(out_dir: Path):
    """Rewrite RESULTS.md from any eval_L2_seed*.json present."""
    evals = sorted(out_dir.glob("eval_L2_seed*.json"))
    lines = [
        "# Idea1 — Walsh-degree-graded γ (graded H_c)",
        "",
        "Arm label: **graded H_c**. Not a non-H_c / projector gadget.",
        "",
        "U_c^X = W exp(−i Σ_{w=1..4} γ_{ℓ,w} E_w) W, E_w = degree-w Walsh component of unsat E (FWHT).",
        "Equal γ_w recovers γ·E. Train {6,9} only. Motif locked: Hamming-shell SNAP, W=H^{⊗3} BinH.",
        "",
        f"**Smoke lock:** B={SMOKE_B} train + CORE eval (not full scale1000 B=1000).",
        "R = Med / Med_QAOA on the **same B samples**. BEAT iff R≤0.98 on all CORE n∈{6,9,12,15}.",
        "Published L=2 B=1000 knee was ≈0.9858; rank vs same-sample plain-γ baseline first.",
        "B=1000 confirm is optional follow-up only if B=100 clearly beats the knee.",
        "",
    ]
    base_p = out_dir / "baseline_plain_L2.json"
    if base_p.exists():
        b = json.loads(base_p.read_text())
        lines += [
            "## Same-sample plain-γ baseline",
            "",
            f"Warm: `{b.get('warm_src', '')}`",
            "",
            "| n | QAOA Med (B) | Med | R | verdict |",
            "|---|--------------|-----|---|---------|",
        ]
        for n in EVAL_NS:
            r = b["per_n"][str(n)]
            lines.append(f"| {n} | {r['QAOA_Med_B']:.4f} | {r['Med']:.4f} | {r['R']:.4f} | {r['verdict']} |")
        bk = max(b["per_n"][str(n)]["R"] for n in (12, 15))
        lines += ["", f"Baseline knee max(R12,R15) = **{bk:.4f}**", ""]

    if not evals:
        lines += ["## Idea1 seeds", "", "_no evals yet_", ""]
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "RESULTS.md").write_text("\n".join(lines) + "\n")
        return out_dir / "RESULTS.md"

    lines += [
        "## Idea1 seeds (L=2)",
        "",
        "| seed | n6 R | n9 R | n12 R | n15 R | knee | ALL_CORE | γ rms-dev | holdout |",
        "|------|------|------|-------|-------|------|----------|-----------|---------|",
    ]
    best_knee = None
    best_seed = None
    any_beat = False
    for ep in evals:
        d = json.loads(ep.read_text())
        seed = d["seed"]
        pn = d["per_n"]
        rs = [pn[str(n)]["R"] for n in EVAL_NS]
        knee = max(pn["12"]["R"], pn["15"]["R"])
        beat = all(pn[str(n)]["verdict"] == "BEAT" for n in EVAL_NS)
        any_beat = any_beat or beat
        tr = out_dir / f"train_L2_seed{seed}.json"
        spr = ""
        hold = ""
        if tr.exists():
            t = json.loads(tr.read_text())
            spr = f"{t.get('gamma_spread', {}).get('rms_dev_from_equal', float('nan')):.4f}"
            hb = t.get("holdout_best")
            hold = f"{hb:.4f}" if hb is not None else ""
        lines.append(
            f"| {seed} | {rs[0]:.4f} | {rs[1]:.4f} | {rs[2]:.4f} | {rs[3]:.4f} | "
            f"{knee:.4f} | {beat} | {spr} | {hold} |"
        )
        if best_knee is None or knee < best_knee:
            best_knee = knee
            best_seed = seed

    lines += ["", "## Per-seed CORE tables", ""]
    for ep in evals:
        md = ep.with_suffix(".md")
        if md.exists():
            lines += [md.read_text().strip(), ""]

    kill = True
    keep = False
    if best_knee is not None:
        # Kill if n12/n15 no better than ~0.986; pass if ALL_CORE R≤0.98
        kill = (not any_beat) and (best_knee > 0.986)
        keep = any_beat or (best_knee <= 0.982)
    lines += [
        "## Verdict",
        "",
        f"- Best seed: **{best_seed}** knee={best_knee:.4f}" if best_knee is not None else "- no seeds",
        f"- ALL_CORE (any seed): **{any_beat}**",
        f"- Kill (Opus: n12/15 stuck ~0.986, or γ_w collapse): **{kill}**",
        f"- Keep: **{keep}**",
        "- B=1000 confirm: only if ALL_CORE on B=100 or knee clearly < baseline.",
        "",
    ]
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "RESULTS.md"
    text = "\n".join(lines) + "\n"
    path.write_text(text)
    # also publish at ucx_bit_a/RESULTS.md (tag subdir is the run folder)
    (OUT_ROOT / "RESULTS.md").write_text(text)
    log(f"wrote {path}")
    return path


def parse_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=int, default=2, choices=[2, 16])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--adam_steps", type=int, default=300)
    ap.add_argument("--adam_lr", type=float, default=1e-3)
    ap.add_argument("--lbfgs_rounds", type=int, default=8)
    ap.add_argument("--lbfgs_iter", type=int, default=8)
    ap.add_argument("--batch_per_n", type=int, default=32)
    ap.add_argument("--max_train", type=int, default=SMOKE_B)
    ap.add_argument("--max_eval", type=int, default=SMOKE_B)
    ap.add_argument("--lam0", type=float, default=1e-1)
    ap.add_argument("--lam1", type=float, default=1e-3)
    ap.add_argument("--lam_p", type=float, default=1e-2)
    ap.add_argument("--holdout_frac", type=float, default=0.2)
    ap.add_argument("--soft_med", action="store_true")
    ap.add_argument("--w6", type=float, default=1.0)
    ap.add_argument("--w9", type=float, default=2.0)
    ap.add_argument("--warm", choices=["qaoa", "champ"], default="champ")
    ap.add_argument("--power_p", action="store_true")
    ap.add_argument("--p0", type=float, default=0.0)
    ap.add_argument("--freeze_p", action="store_true")
    ap.add_argument("--eval_only", type=str, default="")
    ap.add_argument("--baseline_only", action="store_true")
    ap.add_argument("--skip_eval", action="store_true")
    ap.add_argument("--tag", type=str, default="idea1_walsh")
    return ap.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    os.environ["MKL_NUM_THREADS"] = str(args.threads)
    torch.set_num_threads(args.threads)
    device = torch.device("cpu")
    out_dir = OUT_ROOT / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.baseline_only:
        rows, src = eval_plain_baseline(args.L, device, EVAL_NS, args.max_eval, warm=args.warm)
        payload = {
            "arm": "plain_gamma_baseline",
            "L": args.L,
            "B": args.max_eval,
            "warm": args.warm,
            "warm_src": src,
            "per_n": {str(k): v for k, v in rows.items()},
        }
        (out_dir / f"baseline_plain_L{args.L}.json").write_text(json.dumps(_jsonable(payload), indent=2) + "\n")
        update_results_md(out_dir)
        return 0

    if args.eval_only:
        state = torch.load(args.eval_only, map_location="cpu", weights_only=False)
        rows = eval_state(state, device, EVAL_NS, args.max_eval)
        write_eval(out_dir, int(state.get("L", args.L)), state.get("seed", "x"), rows, {"from": args.eval_only})
        update_results_md(out_dir)
        return 0

    state, meta = train_one(args, device, out_dir)
    if not args.skip_eval:
        rows = eval_state(state, device, EVAL_NS, args.max_eval)
        write_eval(out_dir, args.L, args.seed, rows, {"train": meta, "gamma_spread": state["gamma_spread"]})
    update_results_md(out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
