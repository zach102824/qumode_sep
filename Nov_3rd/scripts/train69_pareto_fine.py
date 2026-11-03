#!/usr/bin/env python3
"""Fine α-grid polish around the L=2 pareto knee (train{6,9} n9-median only).

Convex blend of a trust69 n9med checkpoint and a ratio69 endpoint, then Adam
on c2,c3 inside an ℓ2 δ-ball. Never scores n12/n15 in the loss. CORE eval
is {6,9,12,15} after polish. Hamming-shell SNAP, W=H⊗3, L=2.
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
sys.path[:0] = [str(_ROOT), str(Path(__file__).resolve().parent)]

from Nov_3rd import settings as S
from Nov_3rd.ansatz import make_sim
from Nov_3rd.data_io import load_qaoa, load_qaoa_scale_ref, log
from Nov_3rd.eval_scale import eval_n
from explore_OCT29.lib.qaoa import inv_p_from_psucc, qaoa_psucc
from beat_params import phi_from_c

BASE = S.OUT / "beat_qaoa_12h" / "train69"
TRAIN_PACKS = S.OUT / "beat_qaoa_12h" / "train_packs"
EVAL_NS = [6, 9, 12, 15]
RATIO = {
    "s0": BASE / "alpha_ray/params_L2_ratio69_s0_s0.pt",
    "s1": BASE / "alpha_ray/params_L2_ratio69_s0_s1.pt",
}


def load_E(n: int) -> torch.Tensor:
    d = np.load(TRAIN_PACKS / f"train_n{n}.npz", allow_pickle=True)
    return torch.as_tensor(np.asarray(d["train_E"]), dtype=torch.float32)


def load_pt(path: Path, device) -> dict:
    d = torch.load(path, map_location=device, weights_only=False)
    return {k: d[k].to(device).float() for k in ("gamma", "c1", "c2", "c3")}


def qaoa_inv(betas, gammas, E, n, device) -> torch.Tensor:
    invs = []
    for s0 in range(0, E.shape[0], 64):
        ps = qaoa_psucc(betas.to(device), gammas.to(device), E[s0 : s0 + 64].to(device), n)
        invs.append(inv_p_from_psucc(ps.detach().cpu().numpy()))
    return torch.as_tensor(np.concatenate(invs), dtype=torch.float32)


def project_delta(c2, c3, c20, c30, rho: float) -> None:
    with torch.no_grad():
        d2 = c2 - c20
        d3 = c3 - c30
        nrm = torch.sqrt((d2 ** 2).sum() + (d3 ** 2).sum()).clamp(min=1e-12)
        if float(nrm) > rho:
            scale = rho / nrm
            c2.copy_(c20 + d2 * scale)
            c3.copy_(c30 + d3 * scale)


def blend(A, B, alpha: float, device):
    a = float(alpha)
    out = {}
    for k in ("gamma", "c1", "c2", "c3"):
        out[k] = ((1.0 - a) * A[k] + a * B[k]).detach().to(device)
    return out


def n9_median_loss(sim9, E9, inv9, gamma, c1, c2, c3, idx, device):
    phi = phi_from_c(c1, c2, c3)
    logs = []
    for j in idx:
        out = sim9.forward_nov3rd(E9[int(j)], gamma, phi)
        ratio = out["inv_p"].clamp(min=1e-12) / inv9[int(j)].to(device).clamp(min=1e-12)
        logs.append(torch.log(ratio.clamp(min=1e-12)).reshape(-1))
    return torch.cat(logs).median()


def core_eval(gamma, c1, c2, c3, device, qref):
    phi = phi_from_c(c1, c2, c3)
    rows = {}
    for n in EVAL_NS:
        m = eval_n(n, 2, gamma, phi, device, max_count=1000)
        qmed = float((qref.get(n) or {}).get("Med", float("nan")))
        med = float(m["median_inv_p"])
        R = med / qmed
        v = "BEAT" if R <= 0.98 else ("TIE" if R <= 1.02 else "UNDER")
        rows[n] = {
            "Med": med,
            "R": R,
            "QAOA_Med": qmed,
            "B90": float(m["B90_inv_p"]),
            "verdict": v,
        }
    return rows


def write_hit(out: Path, payload: dict, gamma, c1, c2, c3) -> None:
    flag = out / "ALL_CORE_BEAT.flag"
    champ = out / "champion"
    champ.mkdir(parents=True, exist_ok=True)
    tag = f"a{payload['alpha']:.2f}_s{payload['seed']}"
    torch.save(
        {
            "gamma": gamma.detach().cpu(),
            "c1": c1.detach().cpu(),
            "c2": c2.detach().cpu(),
            "c3": c3.detach().cpu(),
            "idea": "pareto_fine",
            "L": 2,
            "payload": payload,
        },
        champ / f"params_{tag}.pt",
    )
    flag.write_text(json.dumps(payload, indent=2) + "\n")
    # top-level flag the caller asked for
    top = BASE / "ALL_CORE_BEAT.flag"
    top.write_text(
        "pareto_fine ALL_CORE_BEAT\n" + json.dumps(payload, indent=2) + "\n"
    )
    log(f"ALL_CORE_BEAT {tag} n15={payload['per_n']['15']['R']:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trust", type=int, required=True, help="n9med_rho005 seed")
    ap.add_argument("--ratio", choices=("s0", "s1"), required=True)
    ap.add_argument("--alphas", type=str, default="0.65,0.70,0.72,0.75,0.78,0.80,0.82,0.85")
    ap.add_argument("--polish_seeds", type=int, default=6)
    ap.add_argument("--rho", type=float, default=0.025)
    ap.add_argument("--adam_steps", type=int, default=400)
    ap.add_argument("--adam_lr", type=float, default=3e-4)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--holdout_frac", type=float, default=0.3)
    ap.add_argument("--skip_raw", action="store_true")
    ap.add_argument("--raw_only", action="store_true")
    ap.add_argument("--tag", type=str, default="")
    args = ap.parse_args()

    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    os.environ["MKL_NUM_THREADS"] = str(args.threads)
    torch.set_num_threads(args.threads)
    device = torch.device("cpu")

    alphas = [float(x) for x in args.alphas.split(",") if x.strip()]
    pair = f"t{args.trust}_r{args.ratio}"
    out = BASE / "pareto_fine" / (f"{pair}_{args.tag}" if args.tag else pair)
    out.mkdir(parents=True, exist_ok=True)

    trust_path = BASE / f"trust69/n9med_rho005/params_L2_seed{args.trust}.pt"
    ratio_path = RATIO[args.ratio]
    A = load_pt(trust_path, device)
    B = load_pt(ratio_path, device)

    betas, gammas = load_qaoa(2)
    E9 = load_E(9)
    inv9 = qaoa_inv(betas, gammas, E9, 9, device)
    sim9 = make_sim(9, 2, device)
    rng_split = np.random.default_rng(0)
    n_hold = max(1, int(args.holdout_frac * len(E9)))
    hold = rng_split.choice(len(E9), size=n_hold, replace=False)
    pool = np.array([i for i in range(len(E9)) if i not in set(hold.tolist())])
    qref = load_qaoa_scale_ref(2)
    log(
        f"pareto_fine {pair} alphas={alphas} seeds={args.polish_seeds} "
        f"rho={args.rho} adam={args.adam_steps} pool={len(pool)} hold={len(hold)}"
    )

    best = None
    rows_out = []

    def note(rec, gamma, c1, c2, c3):
        nonlocal best
        rows_out.append(rec)
        r15 = rec["per_n"]["15"]["R"]
        if best is None or r15 < best["per_n"]["15"]["R"]:
            best = rec
            torch.save(
                {
                    "gamma": gamma.detach().cpu(),
                    "c1": c1.detach().cpu(),
                    "c2": c2.detach().cpu(),
                    "c3": c3.detach().cpu(),
                    "idea": "pareto_fine",
                    "L": 2,
                    "best": True,
                },
                out / "params_best.pt",
            )
        (out / "summary.json").write_text(
            json.dumps({"pair": pair, "best": best, "n_done": len(rows_out)}, indent=2) + "\n"
        )
        if rec["ALL_CORE_BEAT"]:
            write_hit(out, rec, gamma, c1, c2, c3)

    for alpha in alphas:
        G = blend(A, B, alpha, device)
        if not args.skip_raw:
            rows = core_eval(G["gamma"], G["c1"], G["c2"], G["c3"], device, qref)
            allb = all(rows[n]["verdict"] == "BEAT" for n in EVAL_NS)
            rec = {
                "kind": "raw",
                "pair": pair,
                "trust": args.trust,
                "ratio": args.ratio,
                "alpha": alpha,
                "seed": None,
                "rho": 0.0,
                "per_n": {str(k): v for k, v in rows.items()},
                "ALL_CORE_BEAT": allb,
            }
            (out / f"eval_raw_a{alpha:.2f}.json").write_text(json.dumps(rec, indent=2) + "\n")
            rs = " ".join(f"n{n}={rows[n]['R']:.4f}:{rows[n]['verdict']}" for n in EVAL_NS)
            log(f"raw a={alpha:.2f} {rs} ALL={allb}")
            note(rec, G["gamma"], G["c1"], G["c2"], G["c3"])
            if allb:
                log(f"RAW ALL_CORE_BEAT alpha={alpha}")

    if args.raw_only:
        log(f"pareto_fine {pair} raw DONE best15={None if best is None else best['per_n']['15']['R']}")
        return 0

    for alpha in alphas:
        G = blend(A, B, alpha, device)
        c20 = G["c2"].detach()
        c30 = G["c3"].detach()
        for ps in range(args.polish_seeds):
            torch.manual_seed(100_000 * args.trust + 1_000 * int(round(alpha * 100)) + ps + (0 if args.ratio == "s0" else 17))
            rng = np.random.default_rng(50_000 * args.trust + int(round(alpha * 1000)) * 10 + ps + (0 if args.ratio == "s0" else 3))
            c2 = torch.nn.Parameter(c20.clone())
            c3 = torch.nn.Parameter(c30.clone())
            if ps % 2 == 0:
                with torch.no_grad():
                    c2.add_(0.005 * torch.randn_like(c2))
                    c3.add_(0.005 * torch.randn_like(c3))
                init = "tight"
            else:
                with torch.no_grad():
                    d2 = torch.randn_like(c2)
                    d3 = torch.randn_like(c3)
                    nrm = torch.sqrt((d2 ** 2).sum() + (d3 ** 2).sum()).clamp(min=1e-12)
                    # uniform in the 4-D ball
                    rad = args.rho * float(torch.rand(1) ** (1.0 / 4.0))
                    c2.copy_(c20 + d2 * (rad / nrm))
                    c3.copy_(c30 + d3 * (rad / nrm))
                init = "ball"
            project_delta(c2, c3, c20, c30, args.rho)
            opt = torch.optim.Adam([c2, c3], lr=args.adam_lr)
            t0 = time.time()
            last = None
            for step in range(args.adam_steps):
                idx = rng.choice(pool, size=min(64, len(pool)), replace=False)
                loss = n9_median_loss(sim9, E9, inv9, G["gamma"], G["c1"], c2, c3, idx, device)
                opt.zero_grad()
                loss.backward()
                opt.step()
                project_delta(c2, c3, c20, c30, args.rho)
                last = float(loss.detach())
            dt = time.time() - t0
            rows = core_eval(G["gamma"], G["c1"], c2, c3, device, qref)
            allb = all(rows[n]["verdict"] == "BEAT" for n in EVAL_NS)
            rec = {
                "kind": "polish",
                "pair": pair,
                "trust": args.trust,
                "ratio": args.ratio,
                "alpha": alpha,
                "seed": ps,
                "init": init,
                "rho": args.rho,
                "adam_steps": args.adam_steps,
                "adam_lr": args.adam_lr,
                "final_loss": last,
                "wall_s": dt,
                "trust_pt": str(trust_path),
                "ratio_pt": str(ratio_path),
                "per_n": {str(k): v for k, v in rows.items()},
                "ALL_CORE_BEAT": allb,
            }
            (out / f"eval_L2_a{alpha:.2f}_s{ps}.json").write_text(json.dumps(rec, indent=2) + "\n")
            torch.save(
                {
                    "gamma": G["gamma"].detach().cpu(),
                    "c1": G["c1"].detach().cpu(),
                    "c2": c2.detach().cpu(),
                    "c3": c3.detach().cpu(),
                    "idea": "pareto_fine",
                    "L": 2,
                    "alpha": alpha,
                    "seed": ps,
                    "rho": args.rho,
                    "pair": pair,
                },
                out / f"params_L2_a{alpha:.2f}_s{ps}.pt",
            )
            rs = " ".join(f"n{n}={rows[n]['R']:.4f}:{rows[n]['verdict']}" for n in EVAL_NS)
            log(f"polish a={alpha:.2f} s{ps} {init} {dt:.1f}s loss={last:.5f} {rs} ALL={allb}")
            note(rec, G["gamma"], G["c1"], c2, c3)

    log(f"pareto_fine {pair} DONE best15={None if best is None else best['per_n']['15']['R']} ALL={None if best is None else best['ALL_CORE_BEAT']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
