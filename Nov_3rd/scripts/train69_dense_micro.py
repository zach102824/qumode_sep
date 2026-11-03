#!/usr/bin/env python3
"""Round-3 Idea B: micro-ball densify around known n15-best trust checkpoints."""
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
from beat_params import phi_from_c, project_ball

OUT = S.OUT / "beat_qaoa_12h" / "train69" / "dense_micro"
TRAIN_PACKS = S.OUT / "beat_qaoa_12h" / "train_packs"
EVAL_NS = [6, 9, 12, 15]
BASE = S.OUT / "beat_qaoa_12h" / "train69"


def load_E(n):
    d = np.load(TRAIN_PACKS / f"train_n{n}.npz", allow_pickle=True)
    return torch.as_tensor(np.asarray(d["train_E"]), dtype=torch.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rho_micro", type=float, default=0.015)
    ap.add_argument("--n_starts", type=int, default=32)
    ap.add_argument("--adam_steps", type=int, default=400)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--topk_eval", type=int, default=8)
    ap.add_argument("--centers", type=str, default="7,5,2")  # n9med_rho005 best n15 seeds
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
    for s0 in range(0, E9.shape[0], 64):
        Eb = E9[s0:s0+64]
        ps = qaoa_psucc(betas, gammas, Eb, 9)
        invs.append(inv_p_from_psucc(ps.detach().cpu().numpy()))
    inv9 = torch.as_tensor(np.concatenate(invs), dtype=torch.float32)
    sim9 = make_sim(9, L, device)
    rng = np.random.default_rng(1)
    hold = rng.choice(len(E9), size=max(1, int(0.3 * len(E9))), replace=False)
    pool = np.array([i for i in range(len(E9)) if i not in set(hold.tolist())])

    centers = []
    for s in [int(x) for x in args.centers.split(",")]:
        pt = BASE / f"trust69/n9med_rho005/params_L2_seed{s}.pt"
        d = torch.load(pt, map_location=device, weights_only=False)
        centers.append((s, d["c1"].to(device), d["c2"].to(device), d["c3"].to(device), d["gamma"].to(device)))

    starts = []
    k = 0
    for cen_s, c10, c20, c30, g0 in centers:
        for local in range(args.n_starts // max(1, len(centers))):
            torch.manual_seed(10000 * cen_s + local)
            c2 = torch.nn.Parameter(c20.clone() + 0.01 * torch.randn_like(c20))
            c3 = torch.nn.Parameter(c30.clone() + 0.01 * torch.randn_like(c30))
            # keep within micro ball of center
            with torch.no_grad():
                d2 = c2 - c20; d3 = c3 - c30
                nrm = torch.sqrt((d2**2).sum() + (d3**2).sum()).clamp(min=1e-12)
                if float(nrm) > args.rho_micro:
                    c2.copy_(c20 + d2 * (args.rho_micro / nrm))
                    c3.copy_(c30 + d3 * (args.rho_micro / nrm))
            opt = torch.optim.Adam([c2, c3], lr=3e-4)
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
                with torch.no_grad():
                    d2 = c2 - c20; d3 = c3 - c30
                    nrm = torch.sqrt((d2**2).sum() + (d3**2).sum()).clamp(min=1e-12)
                    if float(nrm) > args.rho_micro:
                        c2.copy_(c20 + d2 * (args.rho_micro / nrm))
                        c3.copy_(c30 + d3 * (args.rho_micro / nrm))
            with torch.no_grad():
                phi = phi_from_c(c10, c2, c3)
                logs = []
                for j in hold:
                    out = sim9.forward_nov3rd(E9[int(j)], g0, phi)
                    ratio = out["inv_p"].clamp(min=1e-12) / inv9[int(j)].to(device).clamp(min=1e-12)
                    logs.append(float(torch.log(ratio.clamp(min=1e-12))))
                sc = float(np.median(logs))
            starts.append({"k": k, "cen": cen_s, "hold": sc, "g": g0.detach().cpu(), "c1": c10.detach().cpu(),
                           "c2": c2.detach().cpu(), "c3": c3.detach().cpu()})
            log(f"dense_micro cen{cen_s} k{k} hold={sc:.5f}")
            k += 1

    starts.sort(key=lambda x: x["hold"])
    qref = load_qaoa_scale_ref(L)
    best15 = None
    for rank, s in enumerate(starts[: args.topk_eval]):
        g0 = s["g"].to(device); c1 = s["c1"].to(device); c2 = s["c2"].to(device); c3 = s["c3"].to(device)
        phi = phi_from_c(c1, c2, c3)
        rows = {}
        for n in EVAL_NS:
            m = eval_n(n, L, g0, phi, device, max_count=1000)
            qmed = float((qref.get(n) or {}).get("Med", float("nan")))
            med = float(m["median_inv_p"]); R = med / qmed
            v = "BEAT" if R <= 0.98 else ("TIE" if R <= 1.02 else "UNDER")
            rows[n] = {"Med": med, "R": R, "QAOA_Med": qmed, "verdict": v}
            log(f"dense_micro eval rank{rank} cen{s['cen']} n{n} R={R:.4f} {v}")
        allb = all(rows[n]["verdict"] == "BEAT" for n in EVAL_NS)
        payload = {"idea": "dense_micro", "cen": s["cen"], "k": s["k"], "hold": s["hold"],
                   "per_n": {str(k): v for k, v in rows.items()}, "ALL_CORE_BEAT": allb}
        (OUT / f"eval_L2_cen{s['cen']}_k{s['k']}.json").write_text(json.dumps(payload, indent=2) + "\n")
        torch.save({"gamma": g0.cpu(), "c1": c1.cpu(), "c2": c2.cpu(), "c3": c3.cpu(), "idea": "dense_micro", "L": 2},
                   OUT / f"params_L2_cen{s['cen']}_k{s['k']}.pt")
        r15 = rows[15]["R"]
        if best15 is None or r15 < best15[0]:
            best15 = (r15, s["cen"], s["k"], allb)
        if allb:
            (OUT / "ALL_CORE_BEAT.flag").write_text(f"cen={s['cen']} k={s['k']}\n")
            log(f"DENSE_MICRO ALL_CORE_BEAT cen={s['cen']} k={s['k']}")
    log(f"dense_micro DONE best15={best15}")


if __name__ == "__main__":
    main()
