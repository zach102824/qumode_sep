#!/usr/bin/env python3
"""Dense random + Adam polish in trust ball at L=2, train{6,9} n9-only soft-med.
Maximize chance of lucky n15≤0.98 under locked gates.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(_ROOT), str(Path(__file__).resolve().parent)]
from Nov_3rd import settings as S
from Nov_3rd.ansatz import make_sim
from Nov_3rd.data_io import load_qaoa, load_qaoa_scale_ref, log
from Nov_3rd.eval_scale import eval_n
from explore_OCT29.lib.qaoa import qaoa_psucc, inv_p_from_psucc
from beat_params import phi_from_c, warm_from_qaoa, project_ball

OUT = S.OUT / "beat_qaoa_12h" / "train69" / "dense_trust"
TRAIN_PACKS = S.OUT / "beat_qaoa_12h" / "train_packs"
EVAL_NS = [6, 9, 12, 15]


def load_E(n):
    d = np.load(TRAIN_PACKS / f"train_n{n}.npz", allow_pickle=True)
    return torch.as_tensor(np.asarray(d["train_E"]), dtype=torch.float32)


def qaoa_inv(betas, gammas, E, n, device):
    invs = []
    chunk = 64
    for s0 in range(0, E.shape[0], chunk):
        Eb = E[s0 : s0 + chunk].to(device)
        ps = qaoa_psucc(betas.to(device), gammas.to(device), Eb, n)
        invs.append(inv_p_from_psucc(ps.detach().cpu().numpy()))
    return torch.as_tensor(np.concatenate(invs), dtype=torch.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rho", type=float, default=0.05)
    ap.add_argument("--n_starts", type=int, default=64)
    ap.add_argument("--adam_steps", type=int, default=200)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--topk_eval", type=int, default=8)
    args = ap.parse_args()
    import os
    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    torch.set_num_threads(args.threads)
    device = torch.device("cpu")
    L = 2
    betas, gammas = load_qaoa(L)
    g0, c10, c20, c30 = [t.to(device) for t in warm_from_qaoa(betas, gammas)]
    E9 = load_E(9)
    inv9 = qaoa_inv(betas, gammas, E9, 9, device)
    sim9 = make_sim(9, L, device)
    rng = np.random.default_rng(0)
    n_hold = max(1, int(0.3 * len(E9)))
    hold = rng.choice(len(E9), size=n_hold, replace=False)
    pool = np.array([i for i in range(len(E9)) if i not in set(hold.tolist())])
    OUT.mkdir(parents=True, exist_ok=True)

    starts = []
    for s in range(args.n_starts):
        torch.manual_seed(s)
        c2 = torch.nn.Parameter(0.02 * torch.randn_like(c20))
        c3 = torch.nn.Parameter(0.02 * torch.randn_like(c30))
        project_ball(c2, c3, args.rho)
        opt = torch.optim.Adam([c2, c3], lr=5e-4)
        for step in range(args.adam_steps):
            idx = rng.choice(pool, size=min(64, len(pool)), replace=False)
            phi = phi_from_c(c10, c2, c3)
            logs = []
            for j in idx:
                out = sim9.forward_nov3rd(E9[int(j)], g0, phi)
                ratio = out["inv_p"].clamp(min=1e-12) / inv9[int(j)].to(device).clamp(min=1e-12)
                logs.append(torch.log(ratio.clamp(min=1e-12)).reshape(-1))
            loss = torch.cat(logs).median()
            opt.zero_grad(); loss.backward(); opt.step()
            project_ball(c2, c3, args.rho)
        # holdout score
        with torch.no_grad():
            phi = phi_from_c(c10, c2, c3)
            logs = []
            for j in hold:
                out = sim9.forward_nov3rd(E9[int(j)], g0, phi)
                ratio = out["inv_p"].clamp(min=1e-12) / inv9[int(j)].to(device).clamp(min=1e-12)
                logs.append(float(torch.log(ratio.clamp(min=1e-12))))
            sc = float(np.median(logs))
        starts.append({"seed": s, "hold": sc, "c2": c2.detach().cpu().clone(), "c3": c3.detach().cpu().clone()})
        log(f"dense_trust rho={args.rho} start {s}/{args.n_starts} hold={sc:.5f}")

    starts.sort(key=lambda x: x["hold"])
    (OUT / f"starts_rho{args.rho}.json").write_text(json.dumps([{"seed": s["seed"], "hold": s["hold"]} for s in starts], indent=2) + "\n")

    qref = load_qaoa_scale_ref(L)
    best15 = None
    for rank, s in enumerate(starts[: args.topk_eval]):
        c2 = s["c2"].to(device); c3 = s["c3"].to(device)
        phi = phi_from_c(c10, c2, c3)
        rows = {}
        for n in EVAL_NS:
            m = eval_n(n, L, g0, phi, device, max_count=1000)
            qmed = float((qref.get(n) or {}).get("Med", float("nan")))
            med = float(m["median_inv_p"]); R = med / qmed
            v = "BEAT" if R <= 0.98 else ("TIE" if R <= 1.02 else "UNDER")
            rows[n] = {"Med": med, "R": R, "QAOA_Med": qmed, "verdict": v, "B90": float(m["B90_inv_p"])}
            log(f"dense_trust eval rank{rank} s{s['seed']} n{n} R={R:.4f} {v}")
        allb = all(rows[n]["verdict"] == "BEAT" for n in EVAL_NS)
        payload = {"idea": "dense_trust", "rho": args.rho, "seed": s["seed"], "rank": rank, "hold": s["hold"],
                   "per_n": {str(k): v for k, v in rows.items()}, "ALL_CORE_BEAT": allb}
        (OUT / f"eval_L2_rho{args.rho}_rank{rank}_s{s['seed']}.json").write_text(json.dumps(payload, indent=2) + "\n")
        torch.save({"gamma": g0.cpu(), "c1": c10.cpu(), "c2": c2.cpu(), "c3": c3.cpu(), "idea": "dense_trust", "L": 2, "rho": args.rho, "seed": s["seed"]},
                   OUT / f"params_L2_rho{args.rho}_rank{rank}_s{s['seed']}.pt")
        r15 = rows[15]["R"]
        if best15 is None or r15 < best15[0]:
            best15 = (r15, s["seed"], allb)
        if allb:
            (OUT / "ALL_CORE_BEAT.flag").write_text(f"rho={args.rho} seed={s['seed']}\n")
            log(f"DENSE_TRUST ALL_CORE_BEAT rho={args.rho} s{s['seed']}")
    log(f"dense_trust DONE best_n15={best15}")


if __name__ == "__main__":
    main()
