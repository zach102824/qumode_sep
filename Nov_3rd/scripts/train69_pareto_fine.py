#!/usr/bin/env python3
"""Fine α grid around Pareto knee (~0.65–0.85) for L=2 train{6,9} ALL_CORE hunt."""
from __future__ import annotations
import argparse, json, sys
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
from beat_params import phi_from_c

OUT = S.OUT / "beat_qaoa_12h" / "train69" / "pareto_fine"
BASE = S.OUT / "beat_qaoa_12h" / "train69"
TRAIN_PACKS = S.OUT / "beat_qaoa_12h" / "train_packs"


def load_E(n):
    d = np.load(TRAIN_PACKS / f"train_n{n}.npz", allow_pickle=True)
    return torch.as_tensor(np.asarray(d["train_E"]), dtype=torch.float32)


def load_pt(path, device):
    d = torch.load(path, map_location=device, weights_only=False)
    return {k: d[k].to(device) for k in ("gamma", "c1", "c2", "c3")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--polish_seeds", type=int, default=6)
    ap.add_argument("--trust_seeds", type=str, default="7,5,2")
    ap.add_argument("--ratio_tags", type=str, default="s1,s0")
    ap.add_argument("--adam_steps", type=int, default=400)
    ap.add_argument("--delta", type=float, default=0.025)
    args = ap.parse_args()
    import os
    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    torch.set_num_threads(args.threads)
    device = torch.device("cpu")
    L = 2
    OUT.mkdir(parents=True, exist_ok=True)
    betas, gammas = load_qaoa(L)
    E9 = load_E(9)
    invs = []
    for s0 in range(0, len(E9), 64):
        ps = qaoa_psucc(betas, gammas, E9[s0:s0+64], 9)
        invs.append(inv_p_from_psucc(ps.detach().cpu().numpy()))
    inv9 = torch.as_tensor(np.concatenate(invs), dtype=torch.float32)
    sim9 = make_sim(9, L, device)
    rng = np.random.default_rng(2)
    hold = rng.choice(len(E9), size=max(1, int(0.3 * len(E9))), replace=False)
    pool = np.array([i for i in range(len(E9)) if i not in set(hold.tolist())])
    qref = load_qaoa_scale_ref(L)
    alphas = [0.65, 0.70, 0.72, 0.75, 0.78, 0.80, 0.82, 0.85]
    ratio_map = {
        "s0": BASE / "alpha_ray/params_L2_ratio69_s0_s0.pt",
        "s1": BASE / "alpha_ray/params_L2_ratio69_s0_s1.pt",
    }
    best = None
    for ts in [int(x) for x in args.trust_seeds.split(",")]:
        A = load_pt(BASE / f"trust69/n9med_rho005/params_L2_seed{ts}.pt", device)
        for rtag in args.ratio_tags.split(","):
            B = load_pt(ratio_map[rtag], device)
            for a in alphas:
                g0 = (1 - a) * A["gamma"] + a * B["gamma"]
                c10 = (1 - a) * A["c1"] + a * B["c1"]
                c20 = (1 - a) * A["c2"] + a * B["c2"]
                c30 = (1 - a) * A["c3"] + a * B["c3"]
                for ps in range(args.polish_seeds):
                    torch.manual_seed(3000 + 100 * ts + 17 * hash(rtag) % 1000 + int(100 * a) + ps)
                    gamma = g0.clone(); c1 = c10.clone()
                    c2 = torch.nn.Parameter(c20.clone() + 0.004 * torch.randn_like(c20))
                    c3 = torch.nn.Parameter(c30.clone() + 0.004 * torch.randn_like(c30))
                    with torch.no_grad():
                        d2 = c2 - c20; d3 = c3 - c30
                        nrm = torch.sqrt((d2**2).sum() + (d3**2).sum()).clamp(min=1e-12)
                        if float(nrm) > args.delta:
                            c2.copy_(c20 + d2 * (args.delta / nrm)); c3.copy_(c30 + d3 * (args.delta / nrm))
                    opt = torch.optim.Adam([c2, c3], lr=2.5e-4)
                    for step in range(args.adam_steps):
                        idx = rng.choice(pool, size=min(64, len(pool)), replace=False)
                        phi = phi_from_c(c1, c2, c3)
                        logs = []
                        for j in idx:
                            out = sim9.forward_nov3rd(E9[int(j)], gamma, phi)
                            ratio = out["inv_p"].clamp(min=1e-12) / inv9[int(j)].to(device).clamp(min=1e-12)
                            logs.append(torch.log(ratio.clamp(min=1e-12)).reshape(-1))
                        loss = torch.cat(logs).median()
                        opt.zero_grad(); loss.backward(); opt.step()
                        with torch.no_grad():
                            d2 = c2 - c20; d3 = c3 - c30
                            nrm = torch.sqrt((d2**2).sum() + (d3**2).sum()).clamp(min=1e-12)
                            if float(nrm) > args.delta:
                                c2.copy_(c20 + d2 * (args.delta / nrm)); c3.copy_(c30 + d3 * (args.delta / nrm))
                    phi = phi_from_c(c1, c2, c3)
                    rows = {}
                    for n in (6, 9, 12, 15):
                        m = eval_n(n, L, gamma, phi, device, max_count=1000)
                        qmed = float((qref.get(n) or {}).get("Med", float("nan")))
                        med = float(m["median_inv_p"]); R = med / qmed
                        v = "BEAT" if R <= 0.98 else ("TIE" if R <= 1.02 else "UNDER")
                        rows[n] = {"Med": med, "R": R, "QAOA_Med": qmed, "verdict": v}
                        log(f"pareto_fine t{ts}/{rtag} a={a:.2f} s{ps} n{n} R={R:.4f} {v}")
                    allb = all(rows[n]["verdict"] == "BEAT" for n in (6, 9, 12, 15))
                    tag = f"t{ts}_{rtag}_a{a:.2f}_s{ps}"
                    payload = {"idea": "pareto_fine", "trust_seed": ts, "ratio": rtag, "alpha": a, "seed": ps,
                               "per_n": {str(k): v for k, v in rows.items()}, "ALL_CORE_BEAT": allb}
                    (OUT / f"eval_L2_{tag}.json").write_text(json.dumps(payload, indent=2) + "\n")
                    torch.save({"gamma": gamma.cpu(), "c1": c1.cpu(), "c2": c2.detach().cpu(), "c3": c3.detach().cpu(),
                                "idea": "pareto_fine", "L": 2, "alpha": a, "seed": ps, "trust_seed": ts, "ratio": rtag},
                               OUT / f"params_L2_{tag}.pt")
                    r15 = rows[15]["R"]
                    if best is None or r15 < best[0]:
                        best = (r15, ts, rtag, a, ps, allb, {n: rows[n]["R"] for n in (6, 9, 12, 15)})
                    if allb:
                        (OUT / "ALL_CORE_BEAT.flag").write_text(f"{tag}\n")
                        log(f"PARETO_FINE ALL_CORE_BEAT {tag}")
    log(f"pareto_fine DONE best={best}")
    (OUT / "summary.json").write_text(json.dumps({"best": best}, indent=2) + "\n")


if __name__ == "__main__":
    main()
