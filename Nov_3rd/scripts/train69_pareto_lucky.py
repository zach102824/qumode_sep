#!/usr/bin/env python3
"""Lucky lottery around known Pareto knee a~0.80: many short polishes, prioritize transfer."""
from __future__ import annotations
import json, sys
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

OUT = S.OUT / "beat_qaoa_12h" / "train69" / "pareto_lucky"
BASE = S.OUT / "beat_qaoa_12h" / "train69"
TRAIN_PACKS = S.OUT / "beat_qaoa_12h" / "train_packs"
OUT.mkdir(parents=True, exist_ok=True)

def load_E(n):
    d = np.load(TRAIN_PACKS / f"train_n{n}.npz", allow_pickle=True)
    return torch.as_tensor(np.asarray(d["train_E"]), dtype=torch.float32)

def load_pt(path, device):
    d = torch.load(path, map_location=device, weights_only=False)
    return {k: d[k].to(device) for k in ("gamma", "c1", "c2", "c3")}

def main():
    import os
    os.environ["OMP_NUM_THREADS"] = "1"
    torch.set_num_threads(1)
    device = torch.device("cpu")
    L = 2
    betas, gammas = load_qaoa(L)
    E9 = load_E(9)
    invs = []
    for s0 in range(0, len(E9), 64):
        ps = qaoa_psucc(betas, gammas, E9[s0:s0+64], 9)
        invs.append(inv_p_from_psucc(ps.detach().cpu().numpy()))
    inv9 = torch.as_tensor(np.concatenate(invs), dtype=torch.float32)
    sim9 = make_sim(9, L, device)
    rng = np.random.default_rng(7)
    hold = rng.choice(len(E9), size=max(1, int(0.3 * len(E9))), replace=False)
    pool = np.array([i for i in range(len(E9)) if i not in set(hold.tolist())])
    qref = load_qaoa_scale_ref(L)
    A = load_pt(BASE / "trust69/n9med_rho005/params_L2_seed7.pt", device)
    B = load_pt(BASE / "alpha_ray/params_L2_ratio69_s0_s1.pt", device)
    alphas = [0.76, 0.78, 0.79, 0.80, 0.81, 0.82, 0.83, 0.84]
    best = None
    trial = 0
    for a in alphas:
        g0 = (1 - a) * A["gamma"] + a * B["gamma"]
        c10 = (1 - a) * A["c1"] + a * B["c1"]
        c20 = (1 - a) * A["c2"] + a * B["c2"]
        c30 = (1 - a) * A["c3"] + a * B["c3"]
        # first: eval pure blend
        for mode, steps, noise, delta in [("pure", 0, 0.0, 0.0), ("short", 80, 0.003, 0.015), ("med", 150, 0.005, 0.02)]:
            nrep = 1 if mode == "pure" else 8
            for ps in range(nrep):
                torch.manual_seed(9000 + trial); trial += 1
                gamma = g0.clone(); c1 = c10.clone()
                if mode == "pure":
                    c2 = c20.clone(); c3 = c30.clone()
                else:
                    c2 = torch.nn.Parameter(c20.clone() + noise * torch.randn_like(c20))
                    c3 = torch.nn.Parameter(c30.clone() + noise * torch.randn_like(c30))
                    with torch.no_grad():
                        d2 = c2 - c20; d3 = c3 - c30
                        nrm = torch.sqrt((d2**2).sum() + (d3**2).sum()).clamp(min=1e-12)
                        if float(nrm) > delta:
                            c2.copy_(c20 + d2 * (delta / nrm)); c3.copy_(c30 + d3 * (delta / nrm))
                    opt = torch.optim.Adam([c2, c3], lr=2e-4)
                    for step in range(steps):
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
                            if float(nrm) > delta:
                                c2.copy_(c20 + d2 * (delta / nrm)); c3.copy_(c30 + d3 * (delta / nrm))
                    c2 = c2.detach(); c3 = c3.detach()
                phi = phi_from_c(c1, c2, c3)
                rows = {}
                for n in (6, 9, 12, 15):
                    m = eval_n(n, L, gamma, phi, device, max_count=1000)
                    qmed = float((qref.get(n) or {}).get("Med", float("nan")))
                    med = float(m["median_inv_p"]); R = med / qmed
                    v = "BEAT" if R <= 0.98 else ("TIE" if R <= 1.02 else "UNDER")
                    rows[n] = {"Med": med, "R": R, "QAOA_Med": qmed, "verdict": v}
                allb = all(rows[n]["verdict"] == "BEAT" for n in (6, 9, 12, 15))
                tag = f"a{a:.2f}_{mode}_s{ps}"
                log(f"pareto_lucky {tag} n6={rows[6]['R']:.4f} n9={rows[9]['R']:.4f} n12={rows[12]['R']:.4f} n15={rows[15]['R']:.4f} ALL={allb}")
                payload = {"idea": "pareto_lucky", "alpha": a, "mode": mode, "seed": ps,
                           "per_n": {str(k): v for k, v in rows.items()}, "ALL_CORE_BEAT": allb}
                (OUT / f"eval_L2_{tag}.json").write_text(json.dumps(payload, indent=2) + "\n")
                torch.save({"gamma": gamma.cpu(), "c1": c1.cpu(), "c2": c2.cpu() if torch.is_tensor(c2) else c2,
                            "c3": c3.cpu() if torch.is_tensor(c3) else c3, "idea": "pareto_lucky", "L": 2, "alpha": a},
                           OUT / f"params_L2_{tag}.pt")
                r15 = rows[15]["R"]
                if best is None or r15 < best[0]:
                    best = (r15, a, mode, ps, allb, {n: rows[n]["R"] for n in (6, 9, 12, 15)})
                if allb:
                    (OUT / "ALL_CORE_BEAT.flag").write_text(f"{tag}\n")
                    log(f"PARETO_LUCKY ALL_CORE_BEAT {tag}")
    log(f"pareto_lucky DONE best={best}")
    (OUT / "summary.json").write_text(json.dumps({"best": best}, indent=2) + "\n")

if __name__ == "__main__":
    main()
