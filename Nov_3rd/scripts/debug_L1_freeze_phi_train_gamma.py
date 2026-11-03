#!/usr/bin/env python3
"""Debug probe A: freeze φ at QAOA-warm shell; LBFGS-train only γ on n6×1000.

Does NOT modify settings.py locks. Writes params + scale1000 eval under runs/debug_*.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from Nov_3rd import settings as S
from Nov_3rd.ansatz import make_sim
from Nov_3rd.data_io import (
    load_qaoa,
    load_qaoa_scale_ref,
    load_split_E,
    load_train_E,
    log,
    shell_phi_from_betas,
)
from Nov_3rd.eval_scale import eval_ladder
from Nov_3rd.train_bfgs import _loss, eval_med_b90, median_inv

TIE_HI = 1.02
BEAT_HI = 0.98


def verdict(R):
    if R is None:
        return "NO_QAOA"
    if R <= BEAT_HI:
        return "BEAT"
    if R <= TIE_HI:
        return "TIE"
    return "UNDER"


def train_gamma_only(L, device, E_train, betas0, gammas0, seed, E_val=None):
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    gamma = torch.nn.Parameter(gammas0.detach().clone().float())
    phi = shell_phi_from_betas(betas0).float()  # frozen tensor, not Parameter
    sim = make_sim(6, L, device)
    N = len(E_train)
    warm_med = median_inv(sim, gamma, phi, E_train, max_count=min(64, N))
    log(f"probeA seed{seed} warm Med≈{warm_med:.4f} (φ frozen)")

    best = {"score": float("inf"), "round": -1, "warm_med": warm_med, "train_med": None, "state": None}
    for r in range(S.LBFGS_ROUNDS):
        opt = torch.optim.LBFGS(
            [gamma],
            lr=S.LBFGS_LR,
            max_iter=S.LBFGS_ITER,
            history_size=16,
            line_search_fn="strong_wolfe",
        )
        idx = rng.choice(N, size=min(S.LBFGS_BATCH, N), replace=False)

        def closure():
            opt.zero_grad()
            ls = [_loss(sim.forward_nov3rd(E_train[int(j)], gamma, phi)) for j in idx]
            lt = torch.stack(ls).mean()
            lt.backward()
            torch.nn.utils.clip_grad_norm_([gamma], 5.0)
            return lt

        try:
            loss = opt.step(closure)
            loss_f = float(loss.detach()) if torch.is_tensor(loss) else float(loss)
        except Exception as e:
            log(f"probeA seed{seed} r{r} skip: {e}")
            continue

        tr_med = median_inv(sim, gamma, phi, E_train, max_count=min(128, N))
        val = eval_med_b90(sim, gamma.detach(), phi, E_val) if E_val is not None else None
        sc = float(val["S"]) if val else tr_med
        if sc < best["score"]:
            best = {
                "score": sc,
                "round": r,
                "warm_med": warm_med,
                "train_med": tr_med,
                "val": val,
                "state": {
                    "gamma": gamma.detach().cpu().clone(),
                    "phi": phi.detach().cpu().clone(),
                    "L": L,
                    "probe": "freeze_phi_train_gamma",
                    "seed": seed,
                    "phi_frozen": True,
                },
            }
        if r % 5 == 0 or r + 1 == S.LBFGS_ROUNDS:
            vbit = f" valMed={val['median_inv_p']:.3f}" if val else ""
            log(
                f"probeA seed{seed} r{r}/{S.LBFGS_ROUNDS}: loss≈{loss_f:.4f} "
                f"trainMed≈{tr_med:.3f}{vbit} bestS={best['score']:.3f}@r{best['round']} "
                f"γ={gamma.detach().tolist()}"
            )
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ns", type=int, nargs="+", default=[6, 9, 12, 15])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = ap.parse_args()

    L = 1
    device = torch.device("cpu")
    S.OUT.mkdir(parents=True, exist_ok=True)
    betas0, gammas0 = load_qaoa(L)
    E_train = load_train_E(device)
    E_val, _ = load_split_E(6, "val", S.MAX_VAL, device, data_dir=S.SEALED_DATA)
    qref = load_qaoa_scale_ref(L)
    n6q_path = S.OUT / "debug_qaoa_n6_L1.json"
    if n6q_path.exists():
        n6q = json.loads(n6q_path.read_text())
        qref[6] = {"Med": n6q["median_inv_p"], "B90": n6q["B90_inv_p"]}

    best_all = None
    seed_rows = []
    for seed in args.seeds:
        best = train_gamma_only(L, device, E_train, betas0, gammas0, seed, E_val=E_val)
        seed_rows.append(
            {
                "seed": seed,
                "score": best["score"],
                "round": best["round"],
                "train_med": best["train_med"],
                "warm_med": best["warm_med"],
                "gamma": best["state"]["gamma"].tolist(),
            }
        )
        torch.save(best["state"], S.OUT / f"debug_probeA_freeze_phi_seed{seed}_L1.pt")
        if best_all is None or best["score"] < best_all["score"]:
            best_all = {**best, "seed": seed}

    torch.save(best_all["state"], S.OUT / "debug_probeA_freeze_phi_best_L1.pt")
    st = best_all["state"]
    log("=== probeA scale1000 eval ===")
    rows = eval_ladder(L, st["gamma"], st["phi"], device, ns=tuple(args.ns))

    per = {}
    for n, m in sorted(rows.items()):
        qmed = qref.get(n, {}).get("Med")
        R = None if not qmed else float(m["median_inv_p"] / qmed)
        per[n] = {
            "Med": m["median_inv_p"],
            "B90": m["B90_inv_p"],
            "QAOA_Med": qmed,
            "R": R,
            "verdict": verdict(R),
        }
        rbit = "NA" if R is None else f"{R:.3f}×"
        log(f"probeA n{n}: Med={m['median_inv_p']:.3f} R={rbit} {per[n]['verdict']}")

    under = any(p["verdict"] == "UNDER" for p in per.values())
    summary = (
        "Probe A (freeze φ, train γ on n6): "
        + ("still UNDER on some n — try probe B train_ns=(9,12)." if under else "all R≤1.02 — φ drift was the damage; keep φ frozen.")
    )
    blob = {
        "probe": "freeze_phi_train_gamma",
        "L": 1,
        "seeds": seed_rows,
        "best_seed": best_all["seed"],
        "best_gamma": st["gamma"].tolist(),
        "phi_frozen_warm": st["phi"].tolist(),
        "eval": {str(k): v for k, v in per.items()},
        "under_any": under,
        "summary": summary,
        "asia_shanghai": time.strftime("%Y-%m-%d %H:%M %z"),
    }
    (S.OUT / "debug_probeA_freeze_phi_L1.json").write_text(json.dumps(blob, indent=2) + "\n")

    # append to main debug md
    md = S.OUT / "debug_L1_warm_vs_polished.md"
    lines = [
        "",
        "## Probe A — freeze φ, train γ only (n6×1000 LBFGS)",
        "",
        f"_Asia/Shanghai {time.strftime('%Y-%m-%d %H:%M')}_",
        "",
        f"Best seed={best_all['seed']}; γ={st['gamma'].tolist()} (warm was {gammas0.tolist()})",
        f"φ frozen at warm shell.",
        "",
        "| n | QAOA Med | probeA Med | R | verdict |",
        "|---|----------|------------|---|---------|",
    ]
    for n, p in sorted(per.items()):
        qm = "—" if p["QAOA_Med"] is None else f"{p['QAOA_Med']:.3f}"
        r = "—" if p["R"] is None else f"{p['R']:.3f}×"
        lines.append(f"| {n} | {qm} | {p['Med']:.3f} | {r} | {p['verdict']} |")
    lines += ["", summary, ""]
    with md.open("a") as f:
        f.write("\n".join(lines) + "\n")

    # also update json blob probe field
    main_json = S.OUT / "debug_L1_warm_vs_polished.json"
    if main_json.exists():
        main = json.loads(main_json.read_text())
        main["probe"] = blob
        main_json.write_text(json.dumps(main, indent=2) + "\n")

    log(summary)
    print(json.dumps({"under_any": under, "best_seed": best_all["seed"], "summary": summary}))


if __name__ == "__main__":
    main()
