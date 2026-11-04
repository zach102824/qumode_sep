#!/usr/bin/env python3
"""Opus Idea 2 — cavity-locality-graded γ (honest graded H_c).

U_c^X = W · exp(-i [γ_{ℓ,in} E_in + γ_{ℓ,x} E_x]) · W
  E_in = Walsh strings inside one 3-bit cavity (+ DC)
  E_x  = Walsh strings touching ≥2 cavities
Warm γ_in = γ_x = γ_QAOA. Do NOT split by exact span.

Motif locked: Nov_3rd Hamming-shell SNAP + X-sandwich BinH.
Train{6,9} only; CORE eval {6,9,12,15}; default B=100 smoke; L=2 primary.
BEAT iff R = Med / Med_QAOA ≤ 0.98.
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

from beat_params import phi_from_c, warm_from_qaoa
from explore_OCT29.lib.qaoa import inv_p_from_psucc, qaoa_psucc
from Nov_3rd import settings as S
from Nov_3rd.ansatz import make_sim
from Nov_3rd.cavity_grade import split_E_np, walsh_energy_fracs
from Nov_3rd.data_io import load_qaoa, load_qaoa_scale_ref, load_split_E, log

OUT_ROOT = S.OUT / "beat_qaoa_12h" / "train69" / "ucx_bit_a" / "idea2_cavity"
TRAIN_PACKS = S.OUT / "beat_qaoa_12h" / "train_packs"
TRAIN_NS = [6, 9]
EVAL_NS = [6, 9, 12, 15]
LABEL = "graded_H_c"


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    if isinstance(o, (np.floating, float)):
        return float(o)
    if isinstance(o, (np.integer, int)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, torch.Tensor):
        return o.detach().cpu().tolist()
    if o is None:
        return None
    return o


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


def split_bundle_E(E: torch.Tensor, n: int) -> tuple[torch.Tensor, torch.Tensor, dict]:
    Ein, Ex = split_E_np(E.detach().cpu().numpy(), n)
    stats = walsh_energy_fracs(E.detach().cpu().numpy(), n)
    log(
        f"n{n} B={len(E)} E_x_frac={stats['frac_x']:.4f} E_in_frac={stats['frac_in']:.4f} "
        f"span={stats['frac_span']}"
    )
    return (
        torch.as_tensor(Ein, dtype=torch.float32, device=E.device),
        torch.as_tensor(Ex, dtype=torch.float32, device=E.device),
        stats,
    )


def make_params(betas, gammas, device, freeze_snap: bool = False):
    g0, c10, c20, c30 = [t.to(device) for t in warm_from_qaoa(betas, gammas)]
    gamma_in = torch.nn.Parameter(g0.clone())
    gamma_x = torch.nn.Parameter(g0.clone())  # warm γ_in = γ_x = γ
    if freeze_snap:
        c1, c2, c3 = c10.detach().clone(), c20.detach().clone(), c30.detach().clone()
        trainables = [gamma_in, gamma_x]
    else:
        c1 = torch.nn.Parameter(c10.clone())
        c2 = torch.nn.Parameter(c20.clone())
        c3 = torch.nn.Parameter(c30.clone())
        trainables = [gamma_in, gamma_x, c1, c2, c3]
    return gamma_in, gamma_x, c1, c2, c3, trainables


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


def instance_outputs(sim, E, Ein, Ex, inv_q, gin, gx, phi, idx, want_log: bool):
    out_inv, out_log = [], []
    for j in idx:
        o = sim.forward_nov3rd_cavity(E[int(j)], gin, gx, phi, Ein[int(j)], Ex[int(j)])
        inv = o["inv_p"].clamp(min=1e-12)
        out_inv.append(inv)
        if want_log:
            ratio = inv / inv_q[int(j)].to(inv.device).clamp(min=1e-12)
            out_log.append(torch.log(ratio.clamp(min=1e-12)))
    invs = torch.stack(out_inv)
    logs = torch.stack(out_log) if want_log else None
    return invs, logs


def batch_loss(bundles, gin, gx, c1, c2, c3, idx_map, lam_c, weights, *,
               loss_mode="med_align", qmed_map=None, soft_q=0.5, soft_med=True):
    terms, wsum = [], 0.0
    phi = phi_from_c(c1, c2, c3)
    for n, pack in bundles.items():
        E, Ein, Ex, inv_q, sim = pack["E"], pack["Ein"], pack["Ex"], pack["inv_q"], pack["sim"]
        idx = idx_map[n]
        w = float(weights.get(n, 1.0))
        invs, logs = instance_outputs(
            sim, E, Ein, Ex, inv_q, gin, gx, phi, idx, want_log=(loss_mode != "med_align")
        )
        if loss_mode == "med_align":
            sm = soft_quantile(invs, q=soft_q) if soft_med else invs.median()
            qmed = qmed_map[n] if qmed_map is not None else inv_q.detach().median()
            if not torch.is_tensor(qmed):
                qmed = torch.tensor(float(qmed), device=sm.device, dtype=sm.dtype)
            else:
                qmed = qmed.to(device=sm.device, dtype=sm.dtype)
            tn = torch.log((sm / qmed.clamp(min=1e-12)).clamp(min=1e-12))
        else:
            tn = soft_quantile(logs, q=0.5) if soft_med else logs.mean()
        terms.append(w * tn)
        wsum += w
    loss = torch.stack(terms).sum() / max(wsum, 1e-12)
    if lam_c > 0:
        loss = loss + lam_c * (c2.pow(2).mean() + c3.pow(2).mean())
    return loss


@torch.no_grad()
def holdout_score(pack, gin, gx, c1, c2, c3, hold_idx, *, loss_mode, qmed9, soft_q, soft_med):
    phi = phi_from_c(c1, c2, c3)
    invs, logs = instance_outputs(
        pack["sim"], pack["E"], pack["Ein"], pack["Ex"], pack["inv_q"],
        gin, gx, phi, hold_idx, want_log=(loss_mode != "med_align"),
    )
    if loss_mode == "med_align":
        sm = soft_quantile(invs, q=soft_q) if soft_med else invs.median()
        qmed = pack["inv_q"].median() if qmed9 is None else qmed9
        if not torch.is_tensor(qmed):
            qmed = torch.tensor(float(qmed), device=sm.device, dtype=sm.dtype)
        return float(torch.log((sm / qmed.clamp(min=1e-12)).clamp(min=1e-12)).detach())
    v = soft_quantile(logs, q=0.5) if soft_med else logs.median()
    return float(v.detach())


def snapshot(gin, gx, c1, c2, c3):
    return {
        "gamma_in": gin.detach().cpu().clone(),
        "gamma_x": gx.detach().cpu().clone(),
        "c1": c1.detach().cpu().clone(),
        "c2": c2.detach().cpu().clone(),
        "c3": c3.detach().cpu().clone(),
    }


def gamma_drift(gin, gx) -> dict:
    a = gin.detach().cpu().float().reshape(-1)
    b = gx.detach().cpu().float().reshape(-1)
    diff = (a - b).abs()
    scale = a.abs().mean().clamp(min=1e-8)
    return {
        "gamma_in": a.tolist(),
        "gamma_x": b.tolist(),
        "abs_diff": diff.tolist(),
        "rel_diff_mean": float(diff.mean() / scale),
        "equal_warm_unused": bool(float(diff.mean() / scale) < 0.02),
    }


def train_one(*, L, seed, device, adam_steps, adam_lr, lbfgs_rounds, lbfgs_iter,
              batch_per_n, lam0, lam1, max_train, holdout_frac, weights, out_dir,
              loss_mode, soft_q, freeze_snap):
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    betas, gammas = load_qaoa(L)
    gin, gx, c1, c2, c3, trainables = make_params(betas, gammas, device, freeze_snap=freeze_snap)

    bundles, frac_logs = {}, {}
    for n in TRAIN_NS:
        E = load_train_n(n, max_count=max_train).to(device)
        log(f"idea2_cavity L{L} seed{seed}: load train n{n} B={len(E)}")
        Ein, Ex, stats = split_bundle_E(E, n)
        frac_logs[n] = stats
        inv_q = precompute_qaoa_inv(betas, gammas, E.cpu(), n, device).to(device)
        sim = make_sim(n, L, device)
        bundles[n] = {"E": E, "Ein": Ein, "Ex": Ex, "inv_q": inv_q, "sim": sim}

    qmed_map = {n: float(bundles[n]["inv_q"].detach().median().cpu()) for n in bundles}
    log("train QAOA Meds " + ", ".join(f"n{n}={qmed_map[n]:.4f}" for n in sorted(qmed_map)))

    n9_B = len(bundles[9]["E"])
    perm = rng.permutation(n9_B)
    n_hold = max(1, int(round(holdout_frac * n9_B))) if holdout_frac > 0 else 0
    hold_idx = perm[:n_hold] if n_hold else np.array([], dtype=int)
    train_idx9 = perm[n_hold:] if n_hold else perm
    log(f"n9 holdout={n_hold} train_pool={len(train_idx9)}")

    hist = []
    t0 = time.time()
    best = {"score": float("inf"), "snap": snapshot(gin, gx, c1, c2, c3), "step": -1}

    def maybe_best(tag: str):
        if n_hold == 0:
            return
        sc = holdout_score(
            bundles[9], gin, gx, c1, c2, c3, hold_idx,
            loss_mode=loss_mode, qmed9=qmed_map.get(9), soft_q=soft_q, soft_med=True,
        )
        hist.append({"phase": "holdout", "tag": tag, "hold_med_logR": sc})
        if sc < best["score"]:
            best["score"] = sc
            best["snap"] = snapshot(gin, gx, c1, c2, c3)
            best["step"] = tag
            log(f"NEW BEST holdout={sc:.5f} @ {tag}")

    opt = torch.optim.Adam(trainables, lr=adam_lr)
    for step in range(adam_steps):
        frac = step / max(adam_steps - 1, 1)
        lam = lam0 * (1 - frac) + lam1 * frac
        idx_map = {}
        for n in TRAIN_NS:
            pool = train_idx9 if n == 9 else np.arange(len(bundles[n]["E"]))
            take = min(batch_per_n, len(pool))
            idx_map[n] = rng.choice(pool, size=take, replace=False)
        opt.zero_grad()
        loss = batch_loss(
            bundles, gin, gx, c1, c2, c3, idx_map, lam, weights,
            loss_mode=loss_mode, qmed_map=qmed_map, soft_q=soft_q, soft_med=True,
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainables, 5.0)
        opt.step()
        hist.append({"phase": "adam", "step": step, "loss": float(loss.detach()), "lam": lam})
        if step % 25 == 0 or step == adam_steps - 1:
            d = gamma_drift(gin, gx)
            log(
                f"adam {step}/{adam_steps} loss={hist[-1]['loss']:.5f} "
                f"dγ_rel={d['rel_diff_mean']:.3f} lam={lam:.2e}"
            )
            maybe_best(f"adam{step}")

    for r in range(lbfgs_rounds):
        opt_l = torch.optim.LBFGS(
            trainables, lr=0.25, max_iter=lbfgs_iter, history_size=16, line_search_fn="strong_wolfe"
        )
        idx_map = {}
        for n in TRAIN_NS:
            pool = train_idx9 if n == 9 else np.arange(len(bundles[n]["E"]))
            take = min(batch_per_n, len(pool))
            idx_map[n] = rng.choice(pool, size=take, replace=False)
        lam = lam1

        def closure():
            opt_l.zero_grad()
            lt = batch_loss(
                bundles, gin, gx, c1, c2, c3, idx_map, lam, weights,
                loss_mode=loss_mode, qmed_map=qmed_map, soft_q=soft_q, soft_med=True,
            )
            lt.backward()
            torch.nn.utils.clip_grad_norm_(trainables, 5.0)
            return lt

        try:
            loss = opt_l.step(closure)
            loss_f = float(loss.detach()) if torch.is_tensor(loss) else float(loss)
        except Exception as e:
            log(f"LBFGS r{r} skip: {e}")
            continue
        hist.append({"phase": "lbfgs", "round": r, "loss": loss_f, "lam": lam})
        log(f"lbfgs r{r}/{lbfgs_rounds} loss={loss_f:.5f}")
        maybe_best(f"lbfgs{r}")

    if n_hold and best["score"] < float("inf"):
        log(f"restoring best holdout={best['score']:.5f} @ {best['step']}")
        snap = best["snap"]
        with torch.no_grad():
            gin.data.copy_(snap["gamma_in"].to(device))
            gx.data.copy_(snap["gamma_x"].to(device))
            if isinstance(c1, torch.nn.Parameter):
                c1.data.copy_(snap["c1"].to(device))
                c2.data.copy_(snap["c2"].to(device))
                c3.data.copy_(snap["c3"].to(device))
            else:
                c1 = snap["c1"].to(device)
                c2 = snap["c2"].to(device)
                c3 = snap["c3"].to(device)

    drift = gamma_drift(gin, gx)
    state = {
        "idea": "idea2_cavity",
        "label": LABEL,
        "formula": "U_c^X = W exp(-i[γ_in E_in + γ_x E_x]) W",
        "L": L,
        "seed": seed,
        "train_ns": list(TRAIN_NS),
        "max_train": max_train,
        "gamma_in": gin.detach().cpu(),
        "gamma_x": gx.detach().cpu(),
        "c1": c1.detach().cpu() if torch.is_tensor(c1) else c1,
        "c2": c2.detach().cpu() if torch.is_tensor(c2) else c2,
        "c3": c3.detach().cpu() if torch.is_tensor(c3) else c3,
        "gamma": gin.detach().cpu(),  # alias for accidental loaders
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
        "W_frame": S.W_FRAME,
        "lock": "train69_only",
        "loss_mode": loss_mode,
        "soft_q": soft_q,
        "freeze_snap": freeze_snap,
        "gamma_drift": drift,
        "train_E_x_frac": {str(k): v for k, v in frac_logs.items()},
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    pt = out_dir / f"params_L{L}_seed{seed}.pt"
    torch.save(state, pt)
    meta = {
        "idea": "idea2_cavity",
        "label": LABEL,
        "L": L,
        "seed": seed,
        "train_ns": list(TRAIN_NS),
        "max_train": max_train,
        "wall_s": state["wall_s"],
        "final_loss": hist[-1]["loss"] if hist and "loss" in hist[-1] else None,
        "holdout_best": state["holdout_best"],
        "holdout_best_step": state["holdout_best_step"],
        "gamma_drift": drift,
        "c2_rms": float(c2.detach().cpu().pow(2).mean().sqrt()) if torch.is_tensor(c2) else None,
        "c3_rms": float(c3.detach().cpu().pow(2).mean().sqrt()) if torch.is_tensor(c3) else None,
        "params": str(pt),
        "train_E_x_frac": {str(k): v["frac_x"] for k, v in frac_logs.items()},
    }
    (out_dir / f"train_L{L}_seed{seed}.json").write_text(json.dumps(_jsonable(meta), indent=2) + "\n")
    log(f"saved {pt} wall={state['wall_s']:.0f}s drift={drift['rel_diff_mean']:.3f}")
    return state, meta


@torch.no_grad()
def eval_state(state, device, ns=EVAL_NS, max_count=100):
    L = int(state["L"])
    gin = state["gamma_in"].to(device)
    gx = state["gamma_x"].to(device)
    c1 = state["c1"].to(device)
    c2 = state["c2"].to(device)
    c3 = state["c3"].to(device)
    phi = phi_from_c(c1, c2, c3)
    betas = state["betas_qaoa"]
    gammas = state["gammas_qaoa"]
    qref = load_qaoa_scale_ref(L)
    rows, frac_eval = {}, {}
    for n in ns:
        E, key = load_split_E(n, "scale1000", max_count, device, data_dir=S.SCALE1000_DIR)
        Ein, Ex, stats = split_bundle_E(E, n)
        frac_eval[n] = stats
        sim = make_sim(n, L, device)
        invs, leaks = [], []
        t0 = time.time()
        B = len(E)
        every = 25 if n >= 15 else 50
        for i in range(B):
            out = sim.forward_nov3rd_cavity(E[i], gin, gx, phi, Ein[i], Ex[i])
            invs.append(float(out["inv_p"].cpu()))
            leaks.append(float(out["p_leak"].cpu()))
            if (i + 1) % every == 0:
                log(f"eval n{n} {i+1}/{B} Med≈{np.median(invs):.3f} ({(i+1)/(time.time()-t0+1e-9):.2f}/s)")
        invs_a = np.asarray(invs, dtype=np.float64)
        med = float(np.median(invs_a))
        b90 = float(np.quantile(invs_a, 0.90))
        q = qref.get(n) or {}
        qmed_off = float(q.get("Med", float("nan")))
        qb90_off = float(q.get("B90", float("nan"))) if "B90" in q else float("nan")
        inv_q = precompute_qaoa_inv(betas, gammas, E.cpu(), n, device).numpy()
        qmed_same = float(np.median(inv_q))
        qb90_same = float(np.quantile(inv_q, 0.90))
        R_off = med / qmed_off if qmed_off == qmed_off and qmed_off > 0 else float("nan")
        R = med / qmed_same if qmed_same > 0 else float("nan")

        def verdict(Rval):
            if Rval != Rval:
                return "NO_REF"
            if Rval <= 0.98:
                return "BEAT"
            if Rval <= 1.02:
                return "TIE"
            return "UNDER"

        rows[n] = {
            "Med": med,
            "B90": b90,
            "QAOA_Med_sameB": qmed_same,
            "QAOA_B90_sameB": qb90_same,
            "QAOA_Med": qmed_off,
            "QAOA_B90": qb90_off,
            "R": R,
            "R_officialB1000": R_off,
            "verdict": verdict(R),
            "verdict_official": verdict(R_off),
            "mean_leak": float(np.mean(leaks)),
            "B": int(B),
            "split": key,
            "E_x_frac": stats["frac_x"],
            "E_in_frac": stats["frac_in"],
            "frac_span": stats["frac_span"],
            "wall_s": time.time() - t0,
        }
        log(
            f"eval idea2_cavity L{L} n{n}: Med={med:.4f} Q_same={qmed_same:.4f} "
            f"R={R:.4f} {rows[n]['verdict']} E_x={stats['frac_x']:.3f}"
        )
    return rows, frac_eval


def write_eval(out_dir: Path, L: int, seed, rows: dict, extra: dict | None = None):
    payload = {
        "idea": "idea2_cavity",
        "label": LABEL,
        "L": L,
        "seed": seed,
        "train_ns": list(TRAIN_NS),
        "eval_ns": list(EVAL_NS),
        "per_n": {str(k): v for k, v in rows.items()},
        "finished": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "beat_rule": "R = Med/Med_QAOA ≤ 0.98 on same-B CORE instances",
        "n_gamma_classes": 2,
        "span_split": False,
    }
    if extra:
        payload.update(extra)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"eval_L{L}_seed{seed}.json"
    path.write_text(json.dumps(_jsonable(payload), indent=2) + "\n")
    lines = [
        f"# idea2_cavity (graded H_c) L={L} seed={seed} train{{6,9}}",
        "",
        "U_c^X = W · exp(−i [γ_{ℓ,in} E_in + γ_{ℓ,x} E_x]) · W",
        "",
        "E_in = Walsh strings inside one 3-bit cavity; E_x = strings touching ≥2 cavities.",
        "Warm γ_in = γ_x = γ. Two classes only (no exact-span split). Motif: Hamming-shell + BinH X-sandwich.",
        "",
        "| n | QAOA Med (same B) | Med | R | verdict | E_x frac | B |",
        "|---|-------------------|-----|---|---------|----------|---|",
    ]
    for n in EVAL_NS:
        r = rows[n]
        lines.append(
            f"| {n} | {r['QAOA_Med_sameB']:.4f} | {r['Med']:.4f} | {r['R']:.4f} | "
            f"{r['verdict']} | {r['E_x_frac']:.3f} | {r['B']} |"
        )
    all_beat = all(rows[n]["verdict"] == "BEAT" for n in EVAL_NS)
    Rs = [rows[n]["R"] for n in EVAL_NS]
    worse = list(Rs) == sorted(Rs)  # n6≤n9≤n12≤n15 steadily worse
    lines += [
        "",
        f"**ALL_CORE_BEAT** (same-B R≤0.98) = {all_beat}",
        "",
        f"Kill (R grows with n: n15>n12>n9)? **{bool(rows[15]['R'] > rows[12]['R'] > rows[9]['R'])}** "
        f"(R = {[round(rows[n]['R'], 4) for n in EVAL_NS]})",
        "",
    ]
    if extra and extra.get("train"):
        d = extra["train"].get("gamma_drift") or {}
        lines += [
            "## γ drift",
            "",
            f"- γ_in = {d.get('gamma_in')}",
            f"- γ_x  = {d.get('gamma_x')}",
            f"- rel |γ_in−γ_x| / |γ_in| = {d.get('rel_diff_mean')}",
            f"- unused extra freedom (rel<0.02)? **{d.get('equal_warm_unused')}**",
            "",
        ]
    (out_dir / f"eval_L{L}_seed{seed}.md").write_text("\n".join(lines) + "\n")
    log(f"wrote {path} ALL_CORE_BEAT={all_beat} worse_with_n={worse}")
    return all_beat, path, lines


def write_results_md(out_dir: Path, L: int, seed, rows, meta, all_beat):
    d = (meta or {}).get("gamma_drift") or {}
    lines = [
        "# Opus Idea 2 — cavity-locality-graded γ (graded H_c)",
        "",
        f"_L={L} seed={seed} train{{6,9}} B_train={meta.get('max_train') if meta else '?'} "
        f"CORE B={rows[6]['B'] if 6 in rows else '?'} smoke_",
        "",
        "**Formula.** $U_c^X = W\\cdot\\exp(-i[\\gamma_{\\ell,\\mathrm{in}} E_{\\mathrm{in}} + \\gamma_{\\ell,x} E_x])\\cdot W$",
        "",
        "- $E_{\\mathrm{in}}$: Walsh / X-strings inside one 3-bit cavity (plus DC).",
        "- $E_x$: strings that touch ≥2 cavities. **Not** split by exact span 1/2/3/4.",
        "- Warm start: $\\gamma_{\\mathrm{in}}=\\gamma_x=\\gamma_{\\mathrm{QAOA}}$, Hamming-shell SNAP $\\varphi_k=-\\beta\\cdot k$.",
        "- Motif locked: Nov_3rd Hamming-shell + X-sandwich BinH. Label: **graded H_c** (not non-$H_c$).",
        "- Gate: **BEAT** iff $R=\\mathrm{Med}/\\mathrm{Med}_{\\mathrm{QAOA}}\\le 0.98$ on CORE $\\{6,9,12,15\\}$.",
        "",
        "## Med / R (same-B smoke)",
        "",
        "| n | QAOA Med | Med | R | verdict | E_x frac |",
        "|---|----------|-----|---|---------|----------|",
    ]
    for n in EVAL_NS:
        r = rows[n]
        lines.append(
            f"| {n} | {r['QAOA_Med_sameB']:.4f} | {r['Med']:.4f} | **{r['R']:.4f}** | "
            f"{r['verdict']} | {r['E_x_frac']:.3f} |"
        )
    lines += [
        "",
        f"**ALL_CORE_BEAT** = {all_beat}",
        "",
        "## γ",
        "",
        f"| layer | γ_in | γ_x | |γ_in−γ_x| |",
        f"|-------|------|-----|-----------|",
    ]
    gin = d.get("gamma_in") or []
    gx = d.get("gamma_x") or []
    dif = d.get("abs_diff") or []
    for ell, (a, b, c) in enumerate(zip(gin, gx, dif)):
        lines.append(f"| {ell} | {a:.4f} | {b:.4f} | {c:.4f} |")
    lines += [
        "",
        f"rel drift = {d.get('rel_diff_mean')}; unused (γ_in≈γ_x)? **{d.get('equal_warm_unused')}**",
        "",
        "## Kill checks",
        "",
        f"- Extra freedom unused (γ collapse): {d.get('equal_warm_unused')}",
        f"- R steadily worse with n (n15>n12>n9): "
        f"{bool(rows[15]['R'] > rows[12]['R'] > rows[9]['R'])}",
        f"- L=2 near-miss bar (~0.986): n12 R={rows[12]['R']:.4f}, n15 R={rows[15]['R']:.4f}",
        "",
    ]
    path = out_dir / f"RESULTS_L{L}_seed{seed}.md"
    path.write_text("\n".join(lines) + "\n")
    log(f"wrote {path}")


def ensure_scale1000(ns, B: int):
    """Generate missing CORE scale1000 packs (gitignored) with SCB=54e9, at least B instances."""
    from explore_OCT29.lib.sat_gen import clause_count, instances, pack_clauses

    S.SCALE1000_DIR.mkdir(parents=True, exist_ok=True)
    scb = 54_000_000_000
    for n in ns:
        path = S.SCALE1000_DIR / f"data_n{n}.npz"
        if path.exists():
            d = np.load(path, allow_pickle=True)
            key = "scale1000_E" if "scale1000_E" in d.files else ("test_E" if "test_E" in d.files else None)
            got = 0 if key is None else len(d[key])
            if got >= B:
                log(f"scale1000 n{n} exists B={got}")
                continue
        need = max(B, 100)
        log(f"regen scale1000 n{n} B={need} (missing or short)")
        t0 = time.time()
        m = clause_count(n)
        ins = instances(n, need, scb, compute_E=True)
        seeds = np.array([i["seed"] for i in ins], np.int64)
        nsol = np.array([i["nsol"] for i in ins], np.int32)
        cv, cs = pack_clauses([i["cl"] for i in ins], m, n)
        E = np.stack([i["E"] for i in ins]).astype(np.uint8)
        sp = {}
        for p in ("scale1000", "test"):
            sp.update({f"{p}_seeds": seeds, f"{p}_nsol": nsol, f"{p}_cls_v": cv, f"{p}_cls_s": cs, f"{p}_E": E})
        np.savez_compressed(path, n=np.int32(n), m=np.int32(m), store_E=np.array([1], np.int32), **sp)
        log(f"wrote {path} B={need} wall={time.time()-t0:.0f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=int, default=2, choices=[2, 16])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--B", type=int, default=100, help="smoke B for train+CORE eval")
    ap.add_argument("--adam_steps", type=int, default=250)
    ap.add_argument("--adam_lr", type=float, default=1e-3)
    ap.add_argument("--lbfgs_rounds", type=int, default=6)
    ap.add_argument("--lbfgs_iter", type=int, default=8)
    ap.add_argument("--batch_per_n", type=int, default=16)
    ap.add_argument("--lam0", type=float, default=1e-1)
    ap.add_argument("--lam1", type=float, default=1e-3)
    ap.add_argument("--holdout_frac", type=float, default=0.2)
    ap.add_argument("--loss", dest="loss_mode", default="med_align", choices=["log_ratio", "med_align"])
    ap.add_argument("--soft_q", type=float, default=0.5)
    ap.add_argument("--w6", type=float, default=1.0)
    ap.add_argument("--w9", type=float, default=2.0)
    ap.add_argument("--freeze_snap", action="store_true")
    ap.add_argument("--eval_only", type=str, default="")
    ap.add_argument("--skip_eval", action="store_true")
    ap.add_argument("--warm_only", action="store_true", help="eval QAOA-warm γ_in=γ_x without training")
    ap.add_argument("--tag", type=str, default="")
    args = ap.parse_args()

    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    os.environ["MKL_NUM_THREADS"] = str(args.threads)
    torch.set_num_threads(args.threads)
    device = torch.device("cpu")

    out_dir = OUT_ROOT
    if args.tag:
        out_dir = out_dir / args.tag
    if args.freeze_snap:
        out_dir = out_dir / "freeze_snap"
    if args.warm_only:
        out_dir = out_dir / "warm_only"

    ensure_scale1000(EVAL_NS, args.B)
    weights = {6: float(args.w6), 9: float(args.w9)}

    if args.eval_only:
        state = torch.load(args.eval_only, map_location="cpu", weights_only=False)
        rows, _ = eval_state(state, device, EVAL_NS, max_count=args.B)
        all_beat, _, _ = write_eval(out_dir, int(state.get("L", args.L)), state.get("seed", "x"), rows, {"from": args.eval_only})
        write_results_md(out_dir, int(state.get("L", args.L)), state.get("seed", "x"), rows, {}, all_beat)
        return 0

    if args.warm_only:
        betas, gammas = load_qaoa(args.L)
        g0, c10, c20, c30 = warm_from_qaoa(betas, gammas)
        state = {
            "idea": "idea2_cavity",
            "label": LABEL,
            "L": args.L,
            "seed": args.seed,
            "gamma_in": g0,
            "gamma_x": g0.clone(),
            "c1": c10,
            "c2": c20,
            "c3": c30,
            "betas_qaoa": betas,
            "gammas_qaoa": gammas,
        }
        rows, _ = eval_state(state, device, EVAL_NS, max_count=args.B)
        all_beat, _, _ = write_eval(out_dir, args.L, args.seed, rows, {"warm_only": True})
        write_results_md(out_dir, args.L, args.seed, rows, {"gamma_drift": gamma_drift(g0, g0), "max_train": 0}, all_beat)
        return 0

    state, meta = train_one(
        L=args.L,
        seed=args.seed,
        device=device,
        adam_steps=args.adam_steps,
        adam_lr=args.adam_lr,
        lbfgs_rounds=args.lbfgs_rounds,
        lbfgs_iter=args.lbfgs_iter,
        batch_per_n=args.batch_per_n,
        lam0=args.lam0,
        lam1=args.lam1,
        max_train=args.B,
        holdout_frac=args.holdout_frac,
        weights=weights,
        out_dir=out_dir,
        loss_mode=args.loss_mode,
        soft_q=float(args.soft_q),
        freeze_snap=bool(args.freeze_snap),
    )
    if not args.skip_eval:
        rows, _ = eval_state(state, device, EVAL_NS, max_count=args.B)
        all_beat, _, _ = write_eval(out_dir, args.L, args.seed, rows, {"train": meta})
        write_results_md(out_dir, args.L, args.seed, rows, meta, all_beat)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
