#!/usr/bin/env python3
"""Round-3 Idea A: 1D/2D blend between n15-good (trust) and n6/12-good (ratio69) endpoints.

GD on {6,9} cannot sit on the Pareto knee because n15 is invisible.
A convex blend (and a tiny orthogonal δ in c2,c3) can, without scoring n12/n15 in the loss.
"""
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

OUT = S.OUT / "beat_qaoa_12h" / "train69" / "pareto_blend"
TRAIN_PACKS = S.OUT / "beat_qaoa_12h" / "train_packs"
EVAL_NS = [6, 9, 12, 15]
BASE = S.OUT / "beat_qaoa_12h" / "train69"


def load_E(n):
    d = np.load(TRAIN_PACKS / f"train_n{n}.npz", allow_pickle=True)
    return torch.as_tensor(np.asarray(d["train_E"]), dtype=torch.float32)


def load_pt(path: Path, device):
    d = torch.load(path, map_location=device, weights_only=False)
    return {k: d[k].to(device) if torch.is_tensor(d[k]) else d[k] for k in ("gamma", "c1", "c2", "c3")}


def hold_med_logR(sim9, E9, inv9, g, phi, hold, device):
    logs = []
    with torch.no_grad():
        for j in hold:
            out = sim9.forward_nov3rd(E9[int(j)], g, phi)
            ratio = out["inv_p"].clamp(min=1e-12) / inv9[int(j)].to(device).clamp(min=1e-12)
            logs.append(float(torch.log(ratio.clamp(min=1e-12))))
    return float(np.median(logs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--trust_pt", type=str, default=str(BASE / "trust69/n9med_rho005/params_L2_seed7.pt"))
    ap.add_argument("--ratio_pt", type=str, default=str(BASE / "alpha_ray/params_L2_ratio69_s0_s1.pt"))
    ap.add_argument("--polish_seeds", type=int, default=4)
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
    chunk = 64
    for s0 in range(0, E9.shape[0], chunk):
        Eb = E9[s0:s0+chunk]
        ps = qaoa_psucc(betas, gammas, Eb, 9)
        invs.append(inv_p_from_psucc(ps.detach().cpu().numpy()))
    inv9 = torch.as_tensor(np.concatenate(invs), dtype=torch.float32)
    sim9 = make_sim(9, L, device)
    rng = np.random.default_rng(0)
    n_hold = max(1, int(0.3 * len(E9)))
    hold = rng.choice(len(E9), size=n_hold, replace=False)

    A = load_pt(Path(args.trust_pt), device)  # n15-good
    B = load_pt(Path(args.ratio_pt), device)  # n6/12-good
    alphas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    grid = []
    for a in alphas:
        g = (1 - a) * A["gamma"] + a * B["gamma"]
        c1 = (1 - a) * A["c1"] + a * B["c1"]
        c2 = (1 - a) * A["c2"] + a * B["c2"]
        c3 = (1 - a) * A["c3"] + a * B["c3"]
        phi = phi_from_c(c1, c2, c3)
        # n6 kill-guard proxy: also score n6 hold-free mean logR on first 64 of train pack
        sc9 = hold_med_logR(sim9, E9, inv9, g, phi, hold, device)
        # penalize if n6 much better than n9 (ratio69 overfit signature)
        E6 = load_E(6)
        inv6 = []
        for s0 in range(0, min(64, E6.shape[0]), 32):
            Eb = E6[s0:s0+32]
            ps = qaoa_psucc(betas, gammas, Eb, 6)
            inv6.append(inv_p_from_psucc(ps.detach().cpu().numpy()))
        inv6 = torch.as_tensor(np.concatenate(inv6), dtype=torch.float32)
        sim6 = make_sim(6, L, device)
        logs6 = []
        with torch.no_grad():
            for j in range(min(64, len(E6))):
                out = sim6.forward_nov3rd(E6[j], g, phi)
                ratio = out["inv_p"].clamp(min=1e-12) / inv6[j].clamp(min=1e-12)
                logs6.append(float(torch.log(ratio.clamp(min=1e-12))))
        sc6 = float(np.median(logs6))
        pen = max(0.0, (sc9 - sc6) - 0.02)  # if n6 >0.02 better than n9
        score = sc9 + 2.0 * pen
        grid.append({"alpha": a, "hold9": sc9, "n6_med_logR": sc6, "score": score,
                     "gamma": g.detach().cpu(), "c1": c1.detach().cpu(),
                     "c2": c2.detach().cpu(), "c3": c3.detach().cpu()})
        log(f"pareto_blend alpha={a:.2f} hold9={sc9:.5f} n6={sc6:.5f} score={score:.5f}")

    grid.sort(key=lambda x: x["score"])
    (OUT / "grid.json").write_text(json.dumps([{k: v for k, v in g.items() if k not in ("gamma","c1","c2","c3")} for g in grid], indent=2) + "\n")

    # polish top-3 alphas with tiny δ in c2,c3
    qref = load_qaoa_scale_ref(L)
    best15 = None
    for rank, g0 in enumerate(grid[:3]):
        for ps in range(args.polish_seeds):
            torch.manual_seed(1000 * rank + ps)
            gamma = g0["gamma"].to(device).clone()
            c1 = g0["c1"].to(device).clone()
            c2 = torch.nn.Parameter(g0["c2"].to(device).clone() + 0.005 * torch.randn_like(g0["c2"].to(device)))
            c3 = torch.nn.Parameter(g0["c3"].to(device).clone() + 0.005 * torch.randn_like(g0["c3"].to(device)))
            # project ‖δ‖ relative to g0 endpoint residual ball ~0.02
            with torch.no_grad():
                d2 = c2 - g0["c2"].to(device); d3 = c3 - g0["c3"].to(device)
                nrm = torch.sqrt((d2**2).sum() + (d3**2).sum()).clamp(min=1e-12)
                if float(nrm) > 0.02:
                    c2.copy_(g0["c2"].to(device) + d2 * (0.02 / nrm))
                    c3.copy_(g0["c3"].to(device) + d3 * (0.02 / nrm))
            opt = torch.optim.Adam([c2, c3], lr=3e-4)
            pool = np.array([i for i in range(len(E9)) if i not in set(hold.tolist())])
            for step in range(300):
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
                    d2 = c2 - g0["c2"].to(device); d3 = c3 - g0["c3"].to(device)
                    nrm = torch.sqrt((d2**2).sum() + (d3**2).sum()).clamp(min=1e-12)
                    if float(nrm) > 0.02:
                        c2.copy_(g0["c2"].to(device) + d2 * (0.02 / nrm))
                        c3.copy_(g0["c3"].to(device) + d3 * (0.02 / nrm))
            phi = phi_from_c(c1, c2, c3)
            rows = {}
            for n in EVAL_NS:
                m = eval_n(n, L, gamma, phi, device, max_count=1000)
                qmed = float((qref.get(n) or {}).get("Med", float("nan")))
                med = float(m["median_inv_p"]); R = med / qmed
                v = "BEAT" if R <= 0.98 else ("TIE" if R <= 1.02 else "UNDER")
                rows[n] = {"Med": med, "R": R, "QAOA_Med": qmed, "verdict": v}
                log(f"pareto_blend polish a={g0['alpha']:.2f} s{ps} n{n} R={R:.4f} {v}")
            allb = all(rows[n]["verdict"] == "BEAT" for n in EVAL_NS)
            payload = {"idea": "pareto_blend", "alpha": g0["alpha"], "seed": ps, "rank": rank,
                       "per_n": {str(k): v for k, v in rows.items()}, "ALL_CORE_BEAT": allb}
            (OUT / f"eval_L2_a{g0['alpha']:.2f}_s{ps}.json").write_text(json.dumps(payload, indent=2) + "\n")
            torch.save({"gamma": gamma.cpu(), "c1": c1.cpu(), "c2": c2.detach().cpu(), "c3": c3.detach().cpu(),
                        "idea": "pareto_blend", "L": 2, "alpha": g0["alpha"], "seed": ps},
                       OUT / f"params_L2_a{g0['alpha']:.2f}_s{ps}.pt")
            r15 = rows[15]["R"]
            if best15 is None or r15 < best15[0]:
                best15 = (r15, g0["alpha"], ps, allb, {n: rows[n]["R"] for n in EVAL_NS})
            if allb:
                (OUT / "ALL_CORE_BEAT.flag").write_text(f"alpha={g0['alpha']} seed={ps}\n")
                log(f"PARETO_BLEND ALL_CORE_BEAT alpha={g0['alpha']} s{ps}")
    log(f"pareto_blend DONE best15={best15}")
    (OUT / "summary.json").write_text(json.dumps({"best15": best15}, indent=2) + "\n")


if __name__ == "__main__":
    main()
