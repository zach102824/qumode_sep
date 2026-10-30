#!/usr/bin/env python3
"""Cheap follow-up on CVaR@L44 champ: LBFGS polish + γ+φ jitter confirm.

PAPER_TABLE already has CVaR row; this refreshes it after any promote.

Promote (LBFGS only): n18 B90 ≤ 0.98×9.145 (≥2% improve) AND Med ≤1.05× champ on n9–18.
Jitter is robustness confirm (σ∈{0,0.01,0.02,0.05}), not a promote path.

Honesty: Uc=exp(-iγ Hc). No L48+. No blind.
Writes runs/gadget_Uc/renames_Hc/cvar_polish_jitter/
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

_PKG = Path(__file__).resolve().parents[1]
_ROOT = _PKG.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from explore_OCT29.lib.drivers import fit_b  # noqa: E402
from explore_OCT29.scripts.gadget_Uc_explore import (  # noqa: E402
    _S,
    _jsonable,
    build_W_BinH,
    eval_arm,
    make_sim,
)
from explore_OCT29.scripts.gadget_Uc_lbfgs_polish import polish_seed  # noqa: E402
from explore_OCT29.lib.train_loop import _split_arrays, load_pack  # noqa: E402

OUT = _PKG / "runs" / "gadget_Uc" / "renames_Hc" / "cvar_polish_jitter"
CHAMP_PT = _PKG / "runs" / "gadget_Uc" / "unsat_params.pt"
SCALE = _PKG / "data" / "scale1000"
PAPER = _PKG / "runs" / "gadget_Uc" / "renames_Hc" / "PAPER_TABLE.md"
RH = _PKG / "runs" / "gadget_Uc" / "renames_Hc"

CHAMP = {
    9: {"Med": 1.045, "B90": 1.420},
    12: {"Med": 1.109, "B90": 2.163},
    15: {"Med": 1.228, "B90": 4.067},
    18: {"Med": 1.412, "B90": 9.145},
}
B90_THR = CHAMP[18]["B90"] * 0.98  # ≥2%
MED_MULT = 1.05

ARMC16 = {9: (1.441, 3.314), 12: (1.991, 6.252), 15: (2.921, 10.959), 18: (4.573, 27.810)}
ARMC44 = {9: (1.162, 1.754), 12: (1.353, 3.291), 15: (1.728, 6.399), 18: (2.160, 20.294)}
QAOA = {9: (2.242, 5.534), 12: (3.263, 10.868), 15: (5.099, 18.647), 18: (7.894, 34.070)}
WALK = {9: (12.5, 40.1, 51.0, 161.4), 12: (26.5, 65.9, 107.0, 264.6), 15: (43.8, 128.1, 176.0, 513.6), 18: (71.5, 203.6, 287.0, 815.4)}


def log(*a):
    print(time.strftime("%H:%M:%S"), "[cpj]", *a, flush=True)


@torch.no_grad()
def scale_eval(W, gamma, phi, L, nf, device, ns, B_by_n, tag=""):
    out = {}
    for n in ns:
        B = B_by_n.get(n, 1000)
        m = eval_arm(
            n, W, gamma, phi, L, nf, device, "unsat",
            split="scale1000", max_count=B, data_dir=SCALE,
            progress_every=100 if n >= 15 else 200,
        )
        out[n] = m
        log(f"{tag} n{n}: {m['median_inv_p']:.3f}/{m['B90_inv_p']:.3f}")
    return out


def gate_promote(sc):
    ok_b90 = sc[18]["B90_inv_p"] <= B90_THR
    ok_med = all(sc[n]["median_inv_p"] <= CHAMP[n]["Med"] * MED_MULT for n in (9, 12, 15, 18))
    return bool(ok_b90 and ok_med), {
        "ok_b90": ok_b90,
        "ok_med": ok_med,
        "thr_B90": B90_THR,
        "n18_B90": sc[18]["B90_inv_p"],
        "n18_Med": sc[18]["median_inv_p"],
    }


@torch.no_grad()
def joint_jitter(W, gamma, phi, L, nf, device, sigmas, ns, B, seed=0):
    """γ ← γ(1+σξ), φ ← φ+σξ; report Med/B90 vs σ."""
    rng = np.random.default_rng(seed)
    rows = {}
    g0 = gamma.detach().cpu().numpy().reshape(-1)
    p0 = phi.detach().cpu().numpy()
    for sig in sigmas:
        if sig == 0:
            g = torch.as_tensor(g0, dtype=torch.float32)
            p = torch.as_tensor(p0, dtype=torch.float32)
        else:
            g = torch.as_tensor(g0 * (1.0 + sig * rng.standard_normal(g0.shape)), dtype=torch.float32)
            p = torch.as_tensor(p0 + sig * rng.standard_normal(p0.shape), dtype=torch.float32)
        row = {}
        for n in ns:
            m = eval_arm(
                n, W, g, p, L, nf, device, "unsat",
                split="scale1000", max_count=B, data_dir=SCALE, progress_every=0,
            )
            row[n] = {"Med": m["median_inv_p"], "B90": m["B90_inv_p"]}
        rows[sig] = row
        log(
            f"jitter σ={sig}: "
            + " ".join(f"n{n} {row[n]['Med']:.3f}/{row[n]['B90']:.3f}" for n in ns)
        )
    return rows


def write_paper(champ_nums):
    ns = [9, 12, 15, 18]
    bM = fit_b(ns, [champ_nums[n][0] for n in ns])
    bB = fit_b(ns, [champ_nums[n][1] for n in ns])
    lines = [
        "# Paper mega-table — scale1000 B=1000 (CVaR@L44 champ)",
        "",
        f"_Generated {time.strftime('%Y-%m-%d %H:%M:%S %z')} (Asia/Shanghai)_",
        "",
        "**Metric:** quantum = Med/B90 of **1/p**; WalkSAT = flips/evals (different unit).",
        "",
        "Honesty: champ = **L44 CVaR_q80** (+ optional LBFGS polish if promoted) "
        "projector-Uc ≡ `exp(-iγ Hc)`. L48 STOP. No blind-cost.",
        "",
        "| n | **champ** | ArmC L16 | ArmC L44 pad | QAOA L16 | WalkSAT flips | WalkSAT evals |",
        "|---|-----------|----------|--------------|----------|---------------|---------------|",
    ]
    for n in ns:
        c = champ_nums[n]
        a16, a44, q, w = ARMC16[n], ARMC44[n], QAOA[n], WALK[n]
        lines.append(
            f"| {n} | **{c[0]:.3f}/{c[1]:.3f}** | {a16[0]:.3f}/{a16[1]:.3f} | "
            f"{a44[0]:.3f}/{a44[1]:.3f} | {q[0]:.3f}/{q[1]:.3f} | "
            f"{w[0]:.1f}/{w[1]:.1f} | {w[2]:.1f}/{w[3]:.1f} |"
        )
    lines += [
        "",
        "## b fits (9→18)",
        "",
        f"- champ b_Med≈**{bM['b']:.4f}**, b_B90≈**{bB['b']:.4f}**",
        f"- ArmC L16 b_Med≈**{fit_b(ns,[ARMC16[n][0] for n in ns])['b']:.4f}**",
        f"- ArmC L44 pad b_Med≈**{fit_b(ns,[ARMC44[n][0] for n in ns])['b']:.4f}** (n18 B=100)",
        f"- QAOA L16 b_Med≈**{fit_b(ns,[QAOA[n][0] for n in ns])['b']:.4f}**",
        f"- WalkSAT flips b_Med≈**{fit_b(ns,[WALK[n][0] for n in ns])['b']:.4f}**",
        "",
        "## Notes",
        "",
        "- Mixture Spearman>0.95 kill; freedom span_w miss 2% Med gate.",
        "- CVaR_q80 promoted over L44 pad (n18 B90 −5.2%).",
        "- ArmC L44 n18 B=100; else B=1000.",
        "",
    ]
    PAPER.write_text("\n".join(lines))
    (OUT / "PAPER_TABLE.md").write_text("\n".join(lines))
    return {"b_Med": bM["b"], "b_B90": bB["b"]}


def write_md(results, champ_nums, bfits):
    lines = [
        "# CVaR@L44 — LBFGS polish + joint jitter",
        "",
        f"_Generated {time.strftime('%Y-%m-%d %H:%M:%S %z')} (Asia/Shanghai)_",
        "",
        "Honesty: still `Uc=exp(-iγ Hc)`. No L48+. Blind banned.",
        "",
        f"Champ baseline: "
        + ", ".join(f"n{n} {CHAMP[n]['Med']:.3f}/{CHAMP[n]['B90']:.3f}" for n in (9, 12, 15, 18)),
        f"Promote gate: n18 B90 ≤ **{B90_THR:.3f}** (≥2%) AND Med ≤ {MED_MULT}×.",
        "",
        "## PAPER_TABLE",
        "",
        "Refreshed with current champ vs ArmC L16 / ArmC L44 / QAOA / WalkSAT "
        f"(b_Med≈{bfits['b_Med']:.4f}).",
        "",
        "## LBFGS polish",
        "",
        "| seed | val S | n9 | n12 | n15 | n18 | keep |",
        "|------|-------|----|-----|-----|-----|------|",
    ]
    for name, blk in (results.get("seeds") or {}).items():
        sc = blk.get("scale1000") or {}
        def cell(n):
            m = sc.get(n) or sc.get(str(n))
            return "—" if not m else f"{m['median_inv_p']:.3f}/{m['B90_inv_p']:.3f}"
        lines.append(
            f"| {name} | {blk.get('score', float('nan')):.3f} | "
            f"{cell(9)} | {cell(12)} | {cell(15)} | {cell(18)} | {blk.get('keep')} |"
        )
    lines += [
        "",
        f"Winner: **{results.get('winner')}** promote=**{results.get('promote')}**",
        f"gates={results.get('promote_gates')}",
        "",
        "## Joint γ+φ jitter (confirm)",
        "",
        "| σ | n9 | n12 | n15 |",
        "|---|----|-----|-----|",
    ]
    for sig, row in (results.get("jitter") or {}).items():
        cells = []
        for n in (9, 12, 15):
            r = row.get(n) or row.get(str(n))
            cells.append(f"{r['Med']:.3f}/{r['B90']:.3f}" if r else "—")
        lines.append(f"| {sig} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "## Final champ",
        "",
        "| n | Med | B90 |",
        "|---|-----|-----|",
    ]
    for n in (9, 12, 15, 18):
        lines.append(f"| {n} | {champ_nums[n][0]:.3f} | {champ_nums[n][1]:.3f} |")
    lines += [
        "",
        f"b_Med≈**{bfits['b_Med']:.4f}**, b_B90≈**{bfits['b_B90']:.4f}**",
        "",
    ]
    (OUT / "RESULTS.md").write_text("\n".join(lines))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--nf", type=int, default=8)
    ap.add_argument("--seeds", default="0,1,2,3")
    ap.add_argument("--rounds", type=int, default=6)
    ap.add_argument("--noise", type=float, default=5e-4)
    ap.add_argument("--scale_B", type=int, default=1000)
    ap.add_argument("--jitter_B", type=int, default=500)
    ap.add_argument("--skip_polish", action="store_true")
    ap.add_argument("--skip_jitter", action="store_true")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    os.environ.setdefault("OMP_NUM_THREADS", str(args.threads))
    torch.set_num_threads(args.threads)
    OUT.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    W = build_W_BinH(nf=args.nf)
    st0 = torch.load(CHAMP_PT, map_location="cpu", weights_only=False)
    L = int(st0["L"])
    assert L == 44, L
    g0, p0 = st0["gamma"].float(), st0["phi"].float()

    champ_nums = {n: (CHAMP[n]["Med"], CHAMP[n]["B90"]) for n in CHAMP}
    B_by_n = {9: args.scale_B, 12: args.scale_B, 15: args.scale_B, 18: args.scale_B}

    # Always refresh paper table with current champ first
    bfits = write_paper(champ_nums)
    log("PAPER_TABLE refreshed (pre-polish)")

    results = {
        "champ_baseline": CHAMP,
        "B90_thr": B90_THR,
        "honesty": "Uc=exp(-iγ Hc)",
        "started": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "seeds": {},
        "promote": False,
        "winner": None,
    }

    if not args.skip_polish:
        seed_list = [int(x) for x in args.seeds.split(",") if x.strip()]
        ranked = []
        for seed in seed_list:
            log(f"=== LBFGS polish seed{seed} ===")
            best = polish_seed(
                W, L, args.nf, device, seed, g0, p0,
                noise=args.noise, lbfgs_rounds=args.rounds,
                closure_batch=6, max_train=40, max_val=16,
            )
            torch.save(best["state"], OUT / f"seed{seed}_params.pt")
            results["seeds"][f"seed{seed}"] = {
                "score": best["score"],
                "round": best["round"],
            }
            ranked.append((best["score"], seed, best["state"]))
            (OUT / "results.json").write_text(json.dumps(_jsonable(results), indent=2))

        ranked.sort(key=lambda t: t[0])
        # evaluate top 2 on scale1000
        for score, seed, state in ranked[:2]:
            log(f"=== scale1000 seed{seed} (valS={score:.3f}) ===")
            sc = scale_eval(
                W, state["gamma"], state["phi"], L, args.nf, device,
                (9, 12, 15, 18), B_by_n, tag=f"seed{seed}",
            )
            keep, gates = gate_promote(sc)
            results["seeds"][f"seed{seed}"]["scale1000"] = {
                str(k): {
                    "median_inv_p": v["median_inv_p"],
                    "B90_inv_p": v["B90_inv_p"],
                    "S": v["S"],
                }
                for k, v in sc.items()
            }
            results["seeds"][f"seed{seed}"]["keep"] = keep
            results["seeds"][f"seed{seed}"]["gates"] = gates
            log(f"seed{seed} keep={keep} gates={gates}")
            if keep and not results["promote"]:
                results["promote"] = True
                results["winner"] = f"seed{seed}"
                results["promote_gates"] = gates
                # install champ
                bak = _PKG / "runs" / "gadget_Uc" / "unsat_params_pre_lbfgs_cvar_backup.pt"
                if not bak.exists():
                    import shutil
                    shutil.copy2(CHAMP_PT, bak)
                state = {
                    **{k: state[k] for k in state},
                    "tag": "L44_cvar_lbfgs",
                    "promoted_from": "L44_cvar_q80",
                    "scale1000": {
                        str(n): {"Med": sc[n]["median_inv_p"], "B90": sc[n]["B90_inv_p"]}
                        for n in sc
                    },
                }
                torch.save(state, CHAMP_PT)
                torch.save(state, OUT / "PROMOTED_params.pt")
                champ_nums = {n: (sc[n]["median_inv_p"], sc[n]["B90_inv_p"]) for n in sc}
                bfits = write_paper(champ_nums)
                log(f"PROMOTED seed{seed} → unsat_params.pt")
            (OUT / "results.json").write_text(json.dumps(_jsonable(results), indent=2))

        if not results["promote"]:
            results["winner"] = f"seed{ranked[0][1]}" if ranked else None
            results["promote_gates"] = None
            log("LBFGS no promote — keep CVaR champ")

    # jitter on *current* champ (post-promote if any)
    if not args.skip_jitter:
        st = torch.load(CHAMP_PT, map_location="cpu", weights_only=False)
        log("=== joint γ+φ jitter confirm ===")
        # cheaper: n9,12,15 only for jitter (n18 optional if time)
        jit = joint_jitter(
            W, st["gamma"].float(), st["phi"].float(), int(st["L"]), args.nf, device,
            sigmas=[0.0, 0.01, 0.02, 0.05], ns=(9, 12, 15), B=args.jitter_B, seed=0,
        )
        results["jitter"] = {
            str(sig): {str(n): v for n, v in row.items()} for sig, row in jit.items()
        }
        # verdict
        base9 = jit[0.0][9]["Med"]
        results["jitter_verdict"] = {
            "sigma_ok": [
                sig for sig in (0.01, 0.02, 0.05)
                if jit[sig][9]["Med"] <= base9 * 1.05
            ],
            "note": "σ≤2% expected mild; σ=5% may hurt",
        }
        log("jitter verdict", results["jitter_verdict"])

    results["finished"] = time.strftime("%Y-%m-%d %H:%M:%S %z")
    results["final_champ"] = {str(n): list(champ_nums[n]) for n in champ_nums}
    results["b_fits"] = bfits
    (OUT / "results.json").write_text(json.dumps(_jsonable(results), indent=2))
    write_md(results, champ_nums, bfits)

    # summary for renames_Hc
    summary = {
        "FINAL_TAG": "L44_cvar_lbfgs" if results["promote"] else "L44_cvar_q80",
        "FINAL": {str(n): list(champ_nums[n]) for n in champ_nums},
        "b_Med": bfits["b_Med"],
        "b_B90": bfits["b_B90"],
        "lbfgs_promote": results["promote"],
        "jitter": results.get("jitter_verdict"),
    }
    (RH / "wave_cvar_polish_summary.json").write_text(json.dumps(summary, indent=2))
    log("DONE", OUT / "RESULTS.md")


if __name__ == "__main__":
    main()
