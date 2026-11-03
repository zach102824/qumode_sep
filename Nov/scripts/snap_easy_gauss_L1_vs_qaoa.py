#!/usr/bin/env python3
"""Train 3p gauss Nov1st-final at L=1 and compare to QAOA L=1 on scale1000.

Fresh L=1 train (not a reshape of L=32 angles). Warm-start uses truncated
L32 gauss champ layer-0 + shell-fit gauss as alternatives; CVaR q=0.8 matches
the BEAT arm gp_cvar08_from_gauss_s0 protocol.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch

_NOV = Path(__file__).resolve().parents[1]
_ROOT = _NOV.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from explore_OCT29.scripts.binH_vs_grayH import build_W_BinH  # noqa: E402
from explore_OCT29.scripts.gadget_Uc_explore import _jsonable  # noqa: E402
from Nov.scripts.angle_explore import scale_record, train_one  # noqa: E402
from Nov.scripts.snap_easy_explore import load_shell_warm, warm_snap_for  # noqa: E402

OUT = _NOV / "runs" / "snap_easy" / "gauss_L1_vs_qaoa"
GAUSS_CHAMP = (
    _NOV
    / "runs"
    / "snap_easy"
    / "snap_gauss_perL"
    / "params_easy_snap_gauss_perL_L32_s0_L32_seed0.pt"
)
QAOA_L1 = {
    9: {"Med": 29.279, "B90": 63.429},
    12: {"Med": 90.033, "B90": 216.627},
    15: {"Med": 289.892, "B90": 750.514},
    18: {"Med": 920.513, "B90": 2585.026},
}


def log(*a):
    print(time.strftime("%H:%M:%S"), "[gauss-L1]", *a, flush=True)


def fit_b(ns, meds):
    # log Med = log a + n log b  →  b = exp(slope)
    xs = torch.tensor([float(n) for n in ns], dtype=torch.float64)
    ys = torch.tensor([math.log(m) for m in meds], dtype=torch.float64)
    x = xs - xs.mean()
    y = ys - ys.mean()
    slope = float((x * y).sum() / (x * x).sum())
    return math.exp(slope)


def write_results(tag: str, sc: dict, meta: dict):
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for n in (9, 12, 15, 18):
        m = sc[n]
        q = QAOA_L1[n]
        med, b90 = m["median_inv_p"], m["B90_inv_p"]
        rows.append({
            "n": n,
            "gauss_Med": med,
            "gauss_B90": b90,
            "gauss_pgs": 1.0 / med,
            "qaoa_Med": q["Med"],
            "qaoa_B90": q["B90"],
            "qaoa_pgs": 1.0 / q["Med"],
            "gauss_over_qaoa_Med": med / q["Med"],
            "gauss_over_qaoa_B90": b90 / q["B90"],
        })
    payload = {
        "tag": tag,
        "L": 1,
        "snap_mode": "gauss",
        "params_per_L": 3,
        "motif": "nov1st_final",
        "pack": "scale1000",
        "B": 1000,
        "qaoa_source": "explore_OCT29/baselines/scale1000/qaoa_L1_RESULTS.md",
        "meta": meta,
        "rows": rows,
        "written_at": time.strftime("%Y-%m-%d %H:%M:%S Asia/Shanghai"),
    }
    (OUT / "results.json").write_text(json.dumps(_jsonable(payload), indent=2) + "\n")

    g_meds = [r["gauss_Med"] for r in rows]
    q_meds = [r["qaoa_Med"] for r in rows]
    ns = [r["n"] for r in rows]
    b_g = fit_b(ns, g_meds)
    b_q = fit_b(ns, q_meds)

    lines = [
        "# 3p gauss SNAP L=1 vs QAOA L=1 — scale1000 B=1000",
        "",
        f"_Generated {time.strftime('%Y-%m-%d %H:%M:%S')} Asia/Shanghai_",
        "",
        f"**Ansatz (ours):** Nov1st-final, `snap_mode=gauss` (3 params/L: α+β·pc+δ·pc²), L=1, per-layer",
        f"**Train:** fresh L=1 (not reshape of L=32). tag=`{tag}`",
        f"**Protocol:** train n∈{{6,9}} max_train=200; loss=cvar q=0.8; Adam+LBFGS; warm={meta.get('warm')}",
        f"**QAOA L=1:** ideal qubit QAOA from `qaoa_L1_RESULTS.md` (Med n9/12/15/18 = 29.279/90.033/289.892/920.513)",
        "",
        "## Ladder (1/p) — 3p gauss L=1",
        "",
        "| n | Med | B90 | p(gs)≈1/Med |",
        "|---|-----|-----|-------------|",
    ]
    for r in rows:
        lines.append(
            f"| {r['n']} | {r['gauss_Med']:.3f} | {r['gauss_B90']:.3f} | {r['gauss_pgs']:.6f} |"
        )
    lines += [
        "",
        "## Side-by-side vs QAOA L=1",
        "",
        "| n | gauss Med | gauss B90 | gauss p(gs) | QAOA Med | QAOA B90 | QAOA p(gs) | gauss÷QAOA Med | gauss÷QAOA B90 |",
        "|---|-----------|-----------|-------------|----------|----------|------------|----------------|----------------|",
    ]
    for r in rows:
        lines.append(
            f"| {r['n']} | {r['gauss_Med']:.3f} | {r['gauss_B90']:.3f} | {r['gauss_pgs']:.6f} | "
            f"{r['qaoa_Med']:.3f} | {r['qaoa_B90']:.3f} | {r['qaoa_pgs']:.6f} | "
            f"**{r['gauss_over_qaoa_Med']:.3f}×** | {r['gauss_over_qaoa_B90']:.3f}× |"
        )
    lines += [
        "",
        "## Med-only (for chat)",
        "",
        "| n | QAOA L=1 | 3p gauss L=1 | gauss÷QAOA |",
        "|---|----------|--------------|------------|",
    ]
    for r in rows:
        lines.append(
            f"| {r['n']} | {r['qaoa_Med']:.1f} (p≈{r['qaoa_pgs']:.4f}) | "
            f"{r['gauss_Med']:.1f} (p≈{r['gauss_pgs']:.4f}) | "
            f"**{r['gauss_over_qaoa_Med']:.2f}×** |"
        )
    lines += [
        "",
        "## b fits (Med ~ a·b^n on n=9..18)",
        "",
        f"- **3p gauss L=1** b_Med≈**{b_g:.4f}**",
        f"- **QAOA L=1** b_Med≈**{b_q:.4f}**",
        "",
        "## Notes",
        "",
        "- This is the SNAP-easy BEAT ansatz family (`gp_cvar08_from_gauss_s0` at L=32) retrained fresh at L=1.",
        "- Arm C / Nov progressive shell are intentionally omitted; comparison is vs QAOA L=1 only.",
        f"- Champ path: `{meta.get('path','')}`",
        "",
    ]
    (OUT / "RESULTS.md").write_text("\n".join(lines) + "\n")
    log("wrote", OUT / "RESULTS.md")
    return payload


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nf", type=int, default=8)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--max_train", type=int, default=200)
    ap.add_argument("--max_val", type=int, default=40)
    ap.add_argument("--scale_B", type=int, default=1000)
    ap.add_argument("--seeds", default="0,10,22")
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--lr", type=float, default=2e-3)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    W = build_W_BinH(nf=args.nf)
    L = 1
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]

    st_champ = torch.load(GAUSS_CHAMP, map_location="cpu", weights_only=False)
    g_champ = st_champ["gamma"].float()[:L].clone()
    s_champ = st_champ["snap"].float()[:L].clone()
    g_shell, phi_shell = load_shell_warm(L)

    # Two warm sources × seeds; keep best train-val score
    warms = [
        ("gauss_trunc", g_champ, s_champ),
        ("shell_fit", g_shell if g_shell is not None else g_champ,
         warm_snap_for("gauss", False, L, 3, phi_shell) if phi_shell is not None else s_champ),
    ]

    best_overall = None
    all_rows = []
    for warm_name, gamma0, snap0 in warms:
        if snap0 is None:
            snap0 = s_champ
        for seed in seeds:
            tag = f"gauss_L1_cvar08_{warm_name}_s{seed}"
            log(f"=== train {tag} ===")
            best = train_one(
                W, L, args.nf, device,
                train_ns=(6, 9), max_train=args.max_train, max_val=args.max_val,
                seed=seed, lr=args.lr, steps=args.steps,
                gamma0=gamma0.clone(), snap0=snap0.clone(),
                snap_mode="gauss", snap_shared=False, few_M=3,
                loss_kind="cvar", leakpen=0.1, cvar_q=0.8,
                lbfgs_mid=True, lbfgs_final=True, lbfgs_rounds=4, lbfgs_iter=10,
                arm_tag=tag, out_dir=OUT / "ckpts",
            )
            v9 = (best.get("val") or {}).get(9) or {}
            log(f"  → train-val n9 Med={float(v9.get('median_inv_p', 1e9)):.3f}")
            row = {
                "tag": tag, "warm": warm_name, "seed": seed,
                "score": best.get("score"), "path": best.get("path"),
                "n9_Med": float(v9.get("median_inv_p", 1e9)),
                "state": best["state"],
            }
            all_rows.append(row)
            if best_overall is None or best["score"] < best_overall["score"]:
                best_overall = row

    log(f"champ = {best_overall['tag']} score={best_overall['score']:.4f} n9={best_overall['n9_Med']:.3f}")
    log("=== FULL scale1000 B=1000 ===")
    sc, _ = scale_record(
        W, best_overall["state"], args.nf, device, [9, 12, 15, 18], args.scale_B,
        OUT, best_overall["tag"] + "_full",
    )
    for n in (9, 12, 15, 18):
        m = sc[n]
        log(f"  n{n}: Med={m['median_inv_p']:.3f} B90={m['B90_inv_p']:.3f}")

    meta = {
        "warm": best_overall["warm"],
        "seed": best_overall["seed"],
        "path": best_overall["path"],
        "score": best_overall["score"],
        "steps": args.steps,
        "lr": args.lr,
        "loss": "cvar",
        "cvar_q": 0.8,
        "max_train": args.max_train,
        "candidates": [
            {k: v for k, v in r.items() if k != "state"} for r in all_rows
        ],
    }
    payload = write_results(best_overall["tag"], sc, meta)
    (OUT / "summary.json").write_text(json.dumps(_jsonable({
        "champ": {k: v for k, v in best_overall.items() if k != "state"},
        "scale": {str(k): v for k, v in sc.items()},
        "payload_tag": payload["tag"],
    }), indent=2) + "\n")
    log("DONE →", OUT)


if __name__ == "__main__":
    main()
