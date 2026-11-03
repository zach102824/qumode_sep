#!/usr/bin/env python3
"""Dense multi-start trust ball on train{6,9} only (gates locked: Hamming-shell + Nov1st X-sandwich).

Idea3 variant (trust69, dense):
  - γ, c1 frozen at QAOA-L warm; only c2,c3 move, projected onto ‖(c2,c3)‖₂ ≤ ρ.
  - n_starts starts filling the ball (start 0 = QAOA dual point), each trained with
    full-batch Adam over ALL train{6,9} instances (dense, no mini-batch).
  - Rank starts by a train-only score (max_n∈{6,9} median log-ratio vs QAOA);
    top-k (deduped) go to the locked scale1000 CORE eval {6,9,12,15}.
  - `selected` = rank-1 by train score (honest pick); `oracle` = best-by-eval (reported, not claimed).

Training uses a batched re-implementation of Nov3rdAnsatz.forward_nov3rd, checked against the
locked per-instance forward at startup. Eval always uses the locked eval_n path.
"""
from __future__ import annotations

import argparse
import json
import math
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
from Nov_3rd.data_io import load_qaoa, log
from explore_OCT29.lib.packing import NU

from beat_params import phi_from_c, project_ball, warm_from_qaoa
from train_beat_train69 import (
    EVAL_NS,
    OUT_ROOT,
    TRAIN_NS,
    eval_state,
    load_train_n,
    precompute_qaoa_inv,
    soft_median,
)


class BatchedForward:
    """Batched (over instances) Nov_3rd forward for fixed E set; returns log inv_p (B,)."""

    def __init__(self, sim, E: torch.Tensor):
        self.sim = sim
        self.C = sim.C
        self.nf = sim.nf
        self.sl = (slice(None),) + tuple(slice(0, NU) for _ in range(self.C))
        Cn = torch.stack([sim.cost_Cn(E[i]) for i in range(len(E))])
        self.Cn = Cn.to(sim.cdtype)
        Eg = torch.stack([E[i].to(sim.lam.device)[sim.grid_to_x] for i in range(len(E))])
        self.mask = (Eg == 0).to(sim.rdtype)
        self.B = len(E)

    def _apply_all(self, psi: torch.Tensor, M: torch.Tensor) -> torch.Tensor:
        for c in range(self.C):
            ax = c + 1
            psi = torch.tensordot(M, psi, dims=([1], [ax])).movedim(0, ax)
        return psi

    def log_inv_p(self, gamma: torch.Tensor, phi: torch.Tensor, idx=None) -> torch.Tensor:
        sim = self.sim
        Cn = self.Cn if idx is None else self.Cn[idx]
        mask = self.mask if idx is None else self.mask[idx]
        B = Cn.shape[0]
        psi = torch.zeros((B,) + (self.nf,) * self.C, dtype=sim.cdtype, device=sim.lam.device)
        psi[(slice(None),) + (0,) * self.C] = 1.0 + 0.0j
        gamma = gamma.to(dtype=sim.rdtype).reshape(-1)
        phi = phi.to(dtype=sim.rdtype)
        for ell in range(int(gamma.numel())):
            psi = self._apply_all(psi, sim.Wmat)
            phase = torch.exp(-1j * gamma[ell].to(sim.cdtype) * Cn)
            psi_new = psi.clone()
            psi_new[self.sl] = psi[self.sl] * phase
            psi = psi_new
            psi = self._apply_all(psi, sim.Wmat)
            th = phi[ell][sim.popcount]
            psi = self._apply_all(psi, sim._snap_theta(th))
        psi = self._apply_all(psi, sim.Wmat)
        probs = psi.real.square() + psi.imag.square()
        succ = (probs[self.sl] * mask).reshape(B, -1).sum(dim=1).clamp(min=0.0)
        return -torch.log(succ.clamp(min=1e-12))


def check_batched(sim, bf: BatchedForward, E, gamma, phi, k: int = 4) -> float:
    with torch.no_grad():
        ref = torch.stack([sim.forward_nov3rd(E[i], gamma, phi)["inv_p"] for i in range(k)])
        got = torch.exp(bf.log_inv_p(gamma, phi, idx=torch.arange(k)))
    return float(((got - ref).abs() / ref).max())


def ball_starts(n_starts: int, dim: int, rho: float, rng: np.random.Generator) -> np.ndarray:
    """Start 0 = centre; rest uniform in the ℓ2 ball of radius rho (dim = 2L)."""
    out = [np.zeros(dim)]
    while len(out) < n_starts:
        v = rng.standard_normal(dim)
        v /= np.linalg.norm(v) + 1e-12
        r = rho * rng.uniform() ** (1.0 / dim)
        out.append(v * r)
    return np.stack(out[:n_starts])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=int, default=2, choices=[2, 16])
    ap.add_argument("--rho", type=float, required=True)
    ap.add_argument("--n_starts", type=int, default=64)
    ap.add_argument("--adam_steps", type=int, default=300)
    ap.add_argument("--adam_lr", type=float, default=None, help="default rho/20")
    ap.add_argument("--topk_eval", type=int, default=12)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--w6", type=float, default=1.0)
    ap.add_argument("--w9", type=float, default=2.0)
    ap.add_argument("--loss", choices=["softmed", "mean"], default="softmed")
    ap.add_argument("--dedupe_tol", type=float, default=None, help="default rho/50")
    ap.add_argument("--tag", type=str, default="")
    args = ap.parse_args()

    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    os.environ["MKL_NUM_THREADS"] = str(args.threads)
    torch.set_num_threads(args.threads)
    device = torch.device("cpu")
    rho = float(args.rho)
    lr = rho / 20.0 if args.adam_lr is None else float(args.adam_lr)
    dedupe_tol = rho / 50.0 if args.dedupe_tol is None else float(args.dedupe_tol)
    L = int(args.L)
    weights = {6: float(args.w6), 9: float(args.w9)}
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    out_dir = OUT_ROOT / "trust69_dense" / (args.tag or f"rho{rho:g}_ns{args.n_starts}")
    out_dir.mkdir(parents=True, exist_ok=True)

    betas, gammas = load_qaoa(L)
    g0, c1, _, _ = warm_from_qaoa(betas, gammas)
    g0, c1 = g0.to(device), c1.to(device)

    bundles = {}
    for n in TRAIN_NS:
        E = load_train_n(n).to(device)
        inv_q = precompute_qaoa_inv(betas, gammas, E.cpu(), n, device).to(device)
        sim = make_sim(n, L, device)
        bf = BatchedForward(sim, E)
        phi_chk = phi_from_c(c1, torch.full_like(c1, 0.3 * rho), torch.full_like(c1, -0.2 * rho))
        err = check_batched(sim, bf, E, g0, phi_chk)
        log(f"dense L{L} n{n}: B={len(E)} batched-vs-locked max rel err={err:.2e}")
        if err > 1e-3:
            raise RuntimeError(f"batched forward mismatch n{n}: {err}")
        bundles[n] = (bf, torch.log(inv_q.clamp(min=1e-12)))

    def per_n_logs(c2, c3):
        phi = phi_from_c(c1, c2, c3)
        return {n: bf.log_inv_p(g0, phi) - logq for n, (bf, logq) in bundles.items()}

    def loss_of(logs):
        tot, ws = 0.0, 0.0
        for n, lr_n in logs.items():
            w = weights.get(n, 1.0)
            if w == 0:
                continue
            t = soft_median(lr_n) if args.loss == "softmed" else lr_n.mean()
            tot = tot + w * t
            ws += w
        return tot / max(ws, 1e-12)

    @torch.no_grad()
    def train_score(c2, c3):
        logs = per_n_logs(c2, c3)
        med = {n: float(v.median()) for n, v in logs.items()}
        return max(med.values()), med

    starts = ball_starts(args.n_starts, 2 * L, rho, rng)
    t0 = time.time()
    results = []
    for si, x0 in enumerate(starts):
        c2 = torch.nn.Parameter(torch.as_tensor(x0[:L], dtype=torch.float32).clone())
        c3 = torch.nn.Parameter(torch.as_tensor(x0[L:], dtype=torch.float32).clone())
        opt = torch.optim.Adam([c2, c3], lr=lr)
        best = (float("inf"), None, None, -1)
        for step in range(args.adam_steps):
            opt.zero_grad()
            loss = loss_of(per_n_logs(c2, c3))
            loss.backward()
            opt.step()
            project_ball(c2, c3, rho)
            if step % 25 == 0 or step == args.adam_steps - 1:
                sc, _ = train_score(c2, c3)
                if sc < best[0]:
                    best = (sc, c2.detach().clone(), c3.detach().clone(), step)
        sc, med = train_score(best[1], best[2])
        nrm = float(torch.cat([best[1], best[2]]).norm())
        results.append(
            {
                "start": si,
                "x0": x0.tolist(),
                "c2": best[1].tolist(),
                "c3": best[2].tolist(),
                "norm": nrm,
                "train_score_maxlogR": sc,
                "train_medR": {str(n): math.exp(v) for n, v in med.items()},
                "best_step": best[3],
            }
        )
        log(
            f"dense rho={rho:g} start {si+1}/{len(starts)} score={sc:+.5f} "
            f"R6={math.exp(med[6]):.4f} R9={math.exp(med[9]):.4f} |c|={nrm:.4f} "
            f"({time.time()-t0:.0f}s)"
        )

    results.sort(key=lambda r: r["train_score_maxlogR"])
    picked = []
    for r in results:
        v = np.array(r["c2"] + r["c3"])
        if all(np.linalg.norm(v - np.array(p["c2"] + p["c3"])) > dedupe_tol for p in picked):
            picked.append(r)
        if len(picked) >= args.topk_eval:
            break

    evals = []
    for rank, r in enumerate(picked):
        state = {
            "idea": "trust69",
            "L": L,
            "gamma": g0.cpu(),
            "c1": c1.cpu(),
            "c2": torch.tensor(r["c2"]),
            "c3": torch.tensor(r["c3"]),
            "p": None,
        }
        rows = eval_state(state, device, EVAL_NS)
        all_beat = all(rows[n]["verdict"] == "BEAT" for n in EVAL_NS)
        evals.append({"rank": rank, **r, "core": {str(n): rows[n] for n in EVAL_NS}, "ALL_CORE_BEAT": all_beat})
        log(f"dense rho={rho:g} rank{rank} start{r['start']}: " + " ".join(
            f"n{n}:R={rows[n]['R']:.4f}" for n in EVAL_NS) + f" ALL_CORE_BEAT={all_beat}")
        torch.save(
            {**state, "rho": rho, "train_ns": list(TRAIN_NS), "weights": weights, "lock": "train69_only",
             "recipe": S.RECIPE_NAME, "motif": S.MOTIF, "snap_head": S.SNAP_HEAD,
             "betas_qaoa": betas.cpu(), "gammas_qaoa": gammas.cpu()},
            out_dir / f"params_L{L}_rank{rank}.pt",
        )

    selected = evals[0]
    oracle = min(evals, key=lambda e: max(e["core"][str(n)]["R"] for n in EVAL_NS))
    best_n15 = min(evals, key=lambda e: e["core"]["15"]["R"])
    summary = {
        "script": "train69_dense_trust_ball.py",
        "args": vars(args),
        "adam_lr": lr,
        "L": L,
        "rho": rho,
        "train_ns": list(TRAIN_NS),
        "eval_ns": list(EVAL_NS),
        "weights": weights,
        "wall_s": time.time() - t0,
        "selected_rank": 0,
        "selected_ALL_CORE_BEAT": selected["ALL_CORE_BEAT"],
        "selected_core": selected["core"],
        "any_ALL_CORE_BEAT_in_topk": any(e["ALL_CORE_BEAT"] for e in evals),
        "oracle_rank": oracle["rank"],
        "best_n15_R_in_topk": best_n15["core"]["15"]["R"],
        "best_n15_rank": best_n15["rank"],
        "evals": evals,
        "all_starts": results,
        "finished": time.strftime("%Y-%m-%d %H:%M:%S %z"),
    }
    (out_dir / f"dense_L{L}.json").write_text(json.dumps(summary, indent=2) + "\n")

    lines = [
        f"# trust69_dense L={L} ρ={rho:g} — train{{6,9}} only, {args.n_starts} starts × {args.adam_steps} full-batch Adam",
        "",
        f"Loss={args.loss} w6={weights[6]:g} w9={weights[9]:g} lr={lr:g}. Rank by train max_n median log R; "
        f"top-{len(evals)} deduped → scale1000 CORE.",
        "",
        "| rank | start | train R6 | train R9 | R6 | R9 | R12 | R15 | ALL_CORE_BEAT |",
        "|------|-------|----------|----------|----|----|-----|-----|---------------|",
    ]
    for e in evals:
        c = e["core"]
        lines.append(
            f"| {e['rank']} | {e['start']} | {e['train_medR']['6']:.4f} | {e['train_medR']['9']:.4f} | "
            + " | ".join(f"{c[str(n)]['R']:.4f}" for n in EVAL_NS)
            + f" | {e['ALL_CORE_BEAT']} |"
        )
    lines += [
        "",
        f"**Selected (rank 0, train-only pick) ALL_CORE_BEAT** = {selected['ALL_CORE_BEAT']}",
        f"Any ALL_CORE_BEAT in top-{len(evals)}: {summary['any_ALL_CORE_BEAT_in_topk']}",
        f"Best n15 R in top-{len(evals)}: {best_n15['core']['15']['R']:.4f} (rank {best_n15['rank']}; eval-selected, not a claim)",
        "",
    ]
    (out_dir / f"dense_L{L}.md").write_text("\n".join(lines) + "\n")
    log(f"wrote {out_dir} selected ALL_CORE_BEAT={selected['ALL_CORE_BEAT']} best_n15_R={best_n15['core']['15']['R']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
