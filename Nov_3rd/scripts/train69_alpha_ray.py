#!/usr/bin/env python3
"""Grok R2 Idea3 alpha_ray: 1D rays QAOA→known L=2/L=16 dirs; polish tiny δ.
Train{6,9} lock. Never scores α on n12/n15.
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
from beat_params import phi_from_c, warm_from_qaoa

OUT = S.OUT / "beat_qaoa_12h" / "train69" / "alpha_ray"
TRAIN_PACKS = S.OUT / "beat_qaoa_12h" / "train_packs"
EVAL_NS = [6, 9, 12, 15]
ALPHAS = [0.0, 0.15, 0.3, 0.45, 0.6, 0.8, 1.0, 1.2]


def load_E(n, max_count=None):
    d = np.load(TRAIN_PACKS / f"train_n{n}.npz", allow_pickle=True)
    E = np.asarray(d["train_E"])
    if max_count is not None:
        E = E[: int(max_count)]
    return torch.as_tensor(E, dtype=torch.float32)


def qaoa_inv(betas, gammas, E, n, device):
    invs = []
    B = E.shape[0]
    chunk = 64 if n <= 9 else 16
    for s0 in range(0, B, chunk):
        Eb = E[s0 : s0 + chunk].to(device)
        ps = qaoa_psucc(betas.to(device), gammas.to(device), Eb, n)
        invs.append(inv_p_from_psucc(ps.detach().cpu().numpy()))
    return torch.as_tensor(np.concatenate(invs), dtype=torch.float32)


def pack_state(gamma, c1, c2, c3):
    return {k: v.detach().cpu().clone() for k, v in dict(gamma=gamma, c1=c1, c2=c2, c3=c3).items()}


def load_pt(path, device):
    st = torch.load(path, map_location="cpu", weights_only=False)
    return {k: st[k].to(device) for k in ("gamma", "c1", "c2", "c3")}


def mean_pool_L16_to_L2(st16, target_rms=0.08):
    # mean-pool layers 0-7 and 8-15
    out = {}
    for k in ("gamma", "c1", "c2", "c3"):
        t = st16[k]
        a = t[:8].mean(dim=0, keepdim=True)
        b = t[8:].mean(dim=0, keepdim=True)
        out[k] = torch.cat([a, b], dim=0)
    # rescale c2,c3 rms
    c2, c3 = out["c2"], out["c3"]
    rms = float(torch.sqrt((c2.pow(2).mean() + c3.pow(2).mean()) / 2).clamp(min=1e-12))
    scale = target_rms / rms
    out["c2"] = c2 * scale
    out["c3"] = c3 * scale
    return out


def ends_layers_L16(st16, idxs=(0, 15), target_rms=0.08):
    out = {}
    for k in ("gamma", "c1", "c2", "c3"):
        out[k] = torch.stack([st16[k][i] for i in idxs], dim=0)
    c2, c3 = out["c2"], out["c3"]
    rms = float(torch.sqrt((c2.pow(2).mean() + c3.pow(2).mean()) / 2).clamp(min=1e-12))
    scale = target_rms / max(rms, 1e-12)
    out["c2"] = c2 * scale
    out["c3"] = c3 * scale
    return out


@torch.no_grad()
def med_logR(sim, E, inv_q, gamma, phi, idx):
    # true median of log ratios on idx
    from beat_params import phi_from_c  # noqa
    # reuse instance path via sim
    logs = []
    for j in idx:
        out = sim.forward_nov3rd(E[int(j)], gamma, phi)
        ratio = out["inv_p"].clamp(min=1e-12) / inv_q[int(j)].to(out["inv_p"].device).clamp(min=1e-12)
        logs.append(torch.log(ratio.clamp(min=1e-12)).reshape(-1))
    return float(torch.cat(logs).median())


def angles(gamma, c1, c2, c3):
    return gamma, phi_from_c(c1, c2, c3)


def lerp(q, tgt, a):
    return {k: (1 - a) * q[k] + a * tgt[k] for k in q}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--polish_seeds", type=int, default=4)
    args = ap.parse_args()
    os_env = __import__("os")
    os_env.environ["OMP_NUM_THREADS"] = str(args.threads)
    torch.set_num_threads(args.threads)
    device = torch.device("cpu")
    L = 2
    betas, gammas = load_qaoa(L)
    q0 = warm_from_qaoa(betas, gammas)
    q = {k: v.to(device) for k, v in zip(("gamma", "c1", "c2", "c3"), q0)}

    # targets
    base = Path("Nov_3rd/runs/beat_qaoa_12h/train69")
    rays = {}
    rays["ratio69_s0"] = load_pt(base / "ratio69_w9/params_L2_seed0.pt", device)
    # trust rho 0.05 / 0.1 if present
    for tag, path in [
        ("trust_rho005", base / "trust69/rho0.05/params_L2_seed0.pt"),
        ("trust_rho01", base / "trust69/params_L2_seed0.pt"),
        ("n9med_s0", base / "trust69/n9med_rho005/params_L2_seed0.pt"),
    ]:
        if path.exists():
            rays[tag] = load_pt(path, device)
    p16 = base / "power_p/params_L16_seed0.pt"
    if p16.exists():
        st16 = load_pt(p16, device)
        rays["L16_power_meanpool"] = mean_pool_L16_to_L2(st16)
        rays["L16_power_ends"] = ends_layers_L16(st16)

    E6 = load_E(6); E9 = load_E(9)
    inv6 = qaoa_inv(betas, gammas, E6, 6, device)
    inv9 = qaoa_inv(betas, gammas, E9, 9, device)
    sim6 = make_sim(6, L, device)
    sim9 = make_sim(9, L, device)
    rng = np.random.default_rng(0)
    n9B = len(E9)
    n_hold = max(1, int(0.3 * n9B))
    hold = rng.choice(n9B, size=n_hold, replace=False)
    train_pool = np.array([i for i in range(n9B) if i not in set(hold.tolist())])

    OUT.mkdir(parents=True, exist_ok=True)
    grid = []
    for name, tgt in rays.items():
        best = None
        for a in ALPHAS:
            th = lerp(q, tgt, a)
            g, phi = angles(th["gamma"], th["c1"], th["c2"], th["c3"])
            sc9 = med_logR(sim9, E9, inv9, g, phi, hold)
            # n6 penalty if n6 much better than n9
            sc6 = med_logR(sim6, E6, inv6, g, phi, rng.choice(len(E6), size=min(64, len(E6)), replace=False))
            pen = 0.0
            if sc6 < sc9 - 0.02:
                pen = 0.05
            score = sc9 + pen
            rec = {"ray": name, "alpha": a, "hold_n9_med_logR": sc9, "n6_med_logR": sc6, "score": score}
            grid.append(rec)
            if best is None or score < best["score"]:
                best = {**rec, "theta": pack_state(**th)}
            log(f"alpha_ray {name} a={a:.2f} n9={sc9:.5f} n6={sc6:.5f} score={score:.5f}")
        # kill: no a in (0,1) beats a=0
        a0 = [g for g in grid if g["ray"] == name and g["alpha"] == 0.0][0]["hold_n9_med_logR"]
        interior = [g for g in grid if g["ray"] == name and 0 < g["alpha"] < 1]
        if not any(g["hold_n9_med_logR"] < a0 for g in interior):
            log(f"alpha_ray {name}: DEAD ray (no interior α beats α=0)")
            best["dead"] = True
        else:
            best["dead"] = False
        # save best α*
        torch.save({**best["theta"], "idea": "alpha_ray", "L": 2, "ray": name, "alpha": best["alpha"]}, OUT / f"params_L2_{name}_a.pt")
        # polish δ on c2,c3
        if best["dead"]:
            continue
        for seed in range(args.polish_seeds):
            torch.manual_seed(seed)
            th = {k: best["theta"][k].to(device).clone() for k in best["theta"]}
            # free δ as additive on c2,c3 with ball 0.02
            d2 = torch.nn.Parameter(torch.zeros_like(th["c2"]))
            d3 = torch.nn.Parameter(torch.zeros_like(th["c3"]))
            opt = torch.optim.Adam([d2, d3], lr=3e-4)
            for step in range(300):
                idx = rng.choice(train_pool, size=min(64, len(train_pool)), replace=False)
                c2 = th["c2"] + d2
                c3 = th["c3"] + d3
                # project ball on δ
                with torch.no_grad():
                    rms = torch.sqrt(d2.pow(2).sum() + d3.pow(2).sum()).clamp(min=1e-12)
                    if float(rms) > 0.02:
                        d2.mul_(0.02 / rms); d3.mul_(0.02 / rms)
                g, phi = angles(th["gamma"], th["c1"], c2, c3)
                # softish: mean logR on batch (fast)
                logs = []
                for j in idx:
                    out = sim9.forward_nov3rd(E9[int(j)], g, phi)
                    ratio = out["inv_p"].clamp(min=1e-12) / inv9[int(j)].to(device).clamp(min=1e-12)
                    logs.append(torch.log(ratio.clamp(min=1e-12)).reshape(-1))
                loss = torch.cat(logs).median()
                opt.zero_grad(); loss.backward(); opt.step()
            with torch.no_grad():
                rms = torch.sqrt(d2.pow(2).sum() + d3.pow(2).sum()).clamp(min=1e-12)
                if float(rms) > 0.02:
                    d2.mul_(0.02 / rms); d3.mul_(0.02 / rms)
            th_f = dict(gamma=th["gamma"], c1=th["c1"], c2=th["c2"]+d2.detach(), c3=th["c3"]+d3.detach())
            # eval CORE
            qref = load_qaoa_scale_ref(L)
            rows = {}
            g, phi = angles(th_f["gamma"], th_f["c1"], th_f["c2"], th_f["c3"])
            for n in EVAL_NS:
                m = eval_n(n, L, g, phi, device, max_count=1000)
                qmed = float((qref.get(n) or {}).get("Med", float("nan")))
                med = float(m["median_inv_p"])
                R = med / qmed if qmed == qmed and qmed > 0 else float("nan")
                v = "BEAT" if R <= 0.98 else ("TIE" if R <= 1.02 else "UNDER")
                rows[n] = {"Med": med, "R": R, "QAOA_Med": qmed, "verdict": v, "B90": float(m["B90_inv_p"])}
                log(f"alpha_ray polish {name} s{seed} n{n} R={R:.4f} {v}")
            allb = all(rows[n]["verdict"] == "BEAT" for n in EVAL_NS)
            payload = {"idea": "alpha_ray", "ray": name, "alpha": best["alpha"], "seed": seed, "L": 2,
                       "per_n": {str(k): v for k, v in rows.items()}, "ALL_CORE_BEAT": allb}
            (OUT / f"eval_L2_{name}_s{seed}.json").write_text(json.dumps(payload, indent=2) + "\n")
            torch.save({**{k: v.detach().cpu() for k, v in th_f.items()}, "idea": "alpha_ray", "L": 2, "ray": name, "seed": seed},
                       OUT / f"params_L2_{name}_s{seed}.pt")
            md = [f"# alpha_ray {name} α*={best['alpha']:.2f} seed={seed}", "",
                  "| n | QAOA Med | Med | R | verdict |", "|---|----------|-----|---|---------|"]
            for n in EVAL_NS:
                r = rows[n]
                md.append(f"| {n} | {r['QAOA_Med']:.4f} | {r['Med']:.4f} | {r['R']:.4f} | {r['verdict']} |")
            md += ["", f"**ALL_CORE_BEAT** = {allb}", ""]
            (OUT / f"eval_L2_{name}_s{seed}.md").write_text("\n".join(md) + "\n")
            if allb:
                log(f"ALPHA_RAY ALL_CORE_BEAT {name} s{seed}")
                (OUT / "ALL_CORE_BEAT.flag").write_text(f"{name} s{seed}\n")

    (OUT / "grid.json").write_text(json.dumps([{k: v for k, v in g.items() if k != "theta"} for g in grid], indent=2) + "\n")
    log("alpha_ray DONE")


if __name__ == "__main__":
    main()
