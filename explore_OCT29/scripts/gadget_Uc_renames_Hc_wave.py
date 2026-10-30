#!/usr/bin/env python3
"""Restart wave under renames_Hc: (1) retrain L32 seed params (2) soft-n18
(3) SNAP φ phase noise. Honesty: still Uc=exp(-iγ Hc).

Writes runs/gadget_Uc/renames_Hc/{RESULTS.md,results.json,...}
"""
from __future__ import annotations

import argparse
import json
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
    scale1000_eval,
    train_arm,
)
from explore_OCT29.scripts.gadget_Uc_soft_n15 import load_train_E  # noqa: E402

OUT = _PKG / "runs" / "gadget_Uc" / "renames_Hc"
CHAMP = _PKG / "runs" / "gadget_Uc" / "unsat_params.pt"
LOCKED = {
    9: {"Med": 1.092, "B90": 1.776},
    12: {"Med": 1.197, "B90": 2.843},
    15: {"Med": 1.418, "B90": 5.505},
    18: {"Med": 1.752, "B90": 11.353},
}


def log(*a):
    print(time.strftime("%H:%M:%S"), "[rhc]", *a, flush=True)


def plain_loss(invs, leaks, leakpen=0.1):
    x = invs.clamp(min=1.0)
    return torch.log(x).mean() + leakpen * leaks.mean()


def soft_train(W, L, nf, device, train_ns, steps, max_train, seed, gamma0, phi0, weights=None):
    rng = np.random.default_rng(seed)
    gamma = torch.nn.Parameter(gamma0.detach().clone())
    phi = torch.nn.Parameter(phi0.detach().clone())
    packs, sources = {}, {}
    for n in train_ns:
        E, src = load_train_E(n, max_train, device)
        packs[n] = E
        sources[n] = src
    sims = {n: make_sim(n, W, L, nf, device) for n in train_ns}
    opt = torch.optim.Adam([gamma, phi], lr=2e-3)
    best = {"score": float("inf"), "state": None, "step": -1}
    wmap = weights or {}

    def score_vals(vals):
        # prefer larger n in soft pool
        keys = sorted(vals)
        if 18 in vals and 15 in vals:
            return 0.2 * vals[12]["S"] + 0.35 * vals[15]["S"] + 0.45 * vals[18]["S"]
        if 15 in vals:
            return 0.3 * vals[12]["S"] + 0.5 * vals[15]["S"] + 0.2 * vals[9]["S"]
        return 0.4 * vals[keys[0]]["S"] + 0.6 * vals[keys[-1]]["S"]

    for step in range(steps):
        opt.zero_grad()
        losses, ws = [], []
        for n, Eall in packs.items():
            idx = int(rng.integers(0, len(Eall)))
            out = sims[n].forward_gadget(Eall[idx], gamma, phi, "unsat")
            losses.append(plain_loss(out["inv_p"].unsqueeze(0), out["p_leak"].unsqueeze(0)))
            ws.append(float(wmap.get(n, 1.0)))
        wt = torch.tensor(ws, dtype=torch.float32)
        wt = wt / wt.sum()
        loss = (torch.stack(losses) * wt.to(losses[0].device)).sum()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([gamma, phi], 5.0)
        opt.step()
        if step % 5 == 0 or step + 1 == steps:
            if step in (steps // 2, steps - 1):
                opt_l = torch.optim.LBFGS([gamma, phi], lr=0.15, max_iter=5, history_size=10)

                def closure():
                    opt_l.zero_grad()
                    ls, wss = [], []
                    for n, Eall in packs.items():
                        for j in range(min(6, len(Eall))):
                            o = sims[n].forward_gadget(Eall[j], gamma, phi, "unsat")
                            ls.append(plain_loss(o["inv_p"].unsqueeze(0), o["p_leak"].unsqueeze(0)))
                            wss.append(float(wmap.get(n, 1.0)))
                    wtt = torch.tensor(wss, dtype=torch.float32, device=ls[0].device)
                    wtt = wtt / wtt.sum()
                    lt = (torch.stack(ls) * wtt).sum()
                    lt.backward()
                    return lt

                try:
                    opt_l.step(closure)
                except Exception as e:
                    log("LBFGS skip", e)
            vals = {
                n: eval_arm(n, W, gamma.detach(), phi.detach(), L, nf, device, "unsat",
                            split="val", max_count=12)
                for n in train_ns
            }
            sc = score_vals(vals)
            if sc < best["score"]:
                best = {
                    "score": sc,
                    "step": step,
                    "val": vals,
                    "state": {
                        "gamma": gamma.detach().cpu().clone(),
                        "phi": phi.detach().cpu().clone(),
                        "L": L,
                        "mode": "shell",
                        "cost_mode": "unsat",
                        "W": "BinH",
                        "train_ns": list(train_ns),
                        "sources": sources,
                    },
                }
            if step % 10 == 0 or step + 1 == steps:
                bits = " ".join(
                    f"n{n}:{vals[n]['median_inv_p']:.2f}/{vals[n]['B90_inv_p']:.2f}"
                    for n in train_ns
                )
                log(f"soft {train_ns} step {step}: S={sc:.3f} {bits}")
    return best


@torch.no_grad()
def snap_phase_noise(W, gamma, phi, L, nf, device, sigmas, ns, max_count, seed=0):
    """φ ← φ + σ ξ, ξ~N(0,1) iid; report Med/B90."""
    rng = np.random.default_rng(seed)
    out = {}
    g = gamma.detach().cpu().float()
    p0 = phi.detach().cpu().float()
    for sigma in sigmas:
        if sigma == 0:
            p = p0.clone()
        else:
            p = p0 + float(sigma) * torch.as_tensor(
                rng.standard_normal(size=tuple(p0.shape)), dtype=torch.float32
            )
        row = {}
        for n in ns:
            m = eval_arm(
                n, W, g, p, L, nf, device, "unsat",
                split="scale1000", max_count=max_count,
                data_dir=_PKG / "data" / "scale1000",
                progress_every=100 if n <= 12 else 50,
            )
            row[n] = {"Med": m["median_inv_p"], "B90": m["B90_inv_p"], "B": m["B"]}
            log(f"snapnoise σ={sigma} n{n}: {m['median_inv_p']:.3f}/{m['B90_inv_p']:.3f}")
        out[f"sigma_{sigma:.2f}"] = {"sigma": sigma, "scale1000": row}
    return out


def med_b90_table(sc):
    lines = []
    for n in (9, 12, 15, 18):
        m = sc.get(n) or sc.get(str(n))
        if not m:
            lines.append(f"| {n} | — | — |")
            continue
        med = m.get("median_inv_p", m.get("Med"))
        b90 = m.get("B90_inv_p", m.get("B90"))
        lines.append(f"| {n} | {med:.3f} | {b90:.3f} |")
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=int, default=32)
    ap.add_argument("--nf", type=int, default=8)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--retrain_steps", type=int, default=120)
    ap.add_argument("--soft18_steps", type=int, default=80)
    ap.add_argument("--scale_B", type=int, default=1000)
    ap.add_argument("--noise_B", type=int, default=1000)
    ap.add_argument("--skip_retrain", action="store_true")
    ap.add_argument("--skip_soft18", action="store_true")
    ap.add_argument("--skip_noise", action="store_true")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    W = build_W_BinH(nf=args.nf)
    L, nf = args.L, args.nf
    results = {
        "honesty": "Uc=exp(-iγ Hc); renames_Hc wave",
        "locked_champ": LOCKED,
        "note": "unsat_params.pt was missing from git; retrain produces working seed",
    }

    # --- 1) retrain L32 ---
    if args.skip_retrain and CHAMP.exists():
        st = torch.load(CHAMP, map_location="cpu", weights_only=False)
        g, p = st["gamma"].float(), st["phi"].float()
        log("loaded existing", CHAMP)
    else:
        log("=== retrain L32 on {9,12,15} soft ===")
        # warm from linspace if no champ
        if CHAMP.exists():
            st0 = torch.load(CHAMP, map_location="cpu", weights_only=False)
            g0, p0 = st0["gamma"].float(), st0["phi"].float()
            if g0.numel() != L:
                g0 = torch.linspace(0.3, 1.0, L)
                p0 = torch.zeros(L, 4)
        else:
            # stage-A: train {9,12} from scratch
            log("stage-A train_arm {9,12}")
            best0 = train_arm(
                "unsat", W, L, nf, device,
                steps=max(60, args.retrain_steps // 2),
                max_train=48, max_val=16, seed=7, train_ns=(9, 12),
            )
            g0, p0 = best0["state"]["gamma"].float(), best0["state"]["phi"].float()
            torch.save(best0["state"], OUT / "stageA_912_params.pt")
        best = soft_train(
            W, L, nf, device, (9, 12, 15),
            steps=args.retrain_steps, max_train=48, seed=11,
            gamma0=g0, phi0=p0, weights={9: 1.0, 12: 1.0, 15: 0.6},
        )
        st = best["state"]
        g, p = st["gamma"].float(), st["phi"].float()
        torch.save(st, OUT / "retrain_L32_params.pt")
        torch.save(st, CHAMP)  # seed working champ for downstream
        log("saved", CHAMP)

    log("=== scale1000 retrain seed ===")
    sc0 = scale1000_eval(W, g, p, L, nf, device, "unsat", ns=[9, 12, 15, 18], max_count=args.scale_B)
    results["retrain_L32"] = {"scale1000": sc0}
    (OUT / "results.json").write_text(json.dumps(_jsonable(results), indent=2))

    # --- 2) soft-n18 ---
    if not args.skip_soft18:
        log("=== soft-n18 train {9,12,15,18} ===")
        best18 = soft_train(
            W, L, nf, device, (9, 12, 15, 18),
            steps=args.soft18_steps, max_train=32, seed=18,
            gamma0=g, phi0=p,
            weights={9: 1.0, 12: 1.0, 15: 0.6, 18: 0.4},
        )
        st18 = best18["state"]
        torch.save(st18, OUT / "soft_n18_params.pt")
        sc18 = scale1000_eval(
            W, st18["gamma"], st18["phi"], L, nf, device, "unsat",
            ns=[9, 12, 15, 18], max_count=args.scale_B,
        )
        # promote vs retrain seed (and locked)
        def get(sc, n, k):
            m = sc.get(n) or sc.get(str(n)) or {}
            return m.get(k) or m.get("median_inv_p" if k == "Med" else "B90_inv_p")

        base = sc0
        thr_b90 = get(base, 18, "B90_inv_p") * 0.98
        ok_b90 = get(sc18, 18, "B90_inv_p") <= thr_b90
        ok_small = all(
            get(sc18, n, "median_inv_p") <= get(base, n, "median_inv_p") * 1.05
            for n in (9, 12, 15)
        )
        keep = bool(ok_b90 and ok_small)
        results["soft_n18"] = {
            "scale1000": sc18,
            "keep": keep,
            "reasons": [
                f"n18 B90 {get(sc18,18,'B90_inv_p'):.3f} vs thr {thr_b90:.3f} ({'ok' if ok_b90 else 'no'})",
                f"n9/12/15 Med ≤1.05× retrain ({'ok' if ok_small else 'no'})",
            ],
        }
        if keep:
            torch.save(st18, CHAMP)
            g, p = st18["gamma"].float(), st18["phi"].float()
            log("PROMOTE soft-n18 → unsat_params.pt")
        else:
            log("soft-n18 no promote")
        (OUT / "results.json").write_text(json.dumps(_jsonable(results), indent=2))

    # --- 3) SNAP phase noise ---
    if not args.skip_noise:
        log("=== SNAP φ phase noise ===")
        noise = snap_phase_noise(
            W, g, p, L, nf, device,
            sigmas=[0.0, 0.01, 0.02, 0.05, 0.10],
            ns=[9, 12],
            max_count=args.noise_B,
            seed=42,
        )
        results["snap_phase_noise"] = noise
        (OUT / "results.json").write_text(json.dumps(_jsonable(results), indent=2))

    # b fits
    def collect(sc):
        ns, meds, b90s = [], [], []
        for n in (9, 12, 15, 18):
            m = sc.get(n) or sc.get(str(n))
            if not m:
                continue
            ns.append(n)
            meds.append(m.get("median_inv_p", m.get("Med")))
            b90s.append(m.get("B90_inv_p", m.get("B90")))
        return ns, meds, b90s

    ns, meds, b90s = collect(sc0)
    results["retrain_b"] = {"b_Med": fit_b(ns, meds), "b_B90": fit_b(ns, b90s)}
    if "soft_n18" in results:
        ns2, m2, b2 = collect(results["soft_n18"]["scale1000"])
        results["soft18_b"] = {"b_Med": fit_b(ns2, m2), "b_B90": fit_b(ns2, b2)}

    # WalkSAT summary if present
    base_dir = _PKG / "baselines"
    ws = {}
    for n in (9, 12, 15, 18):
        pth = base_dir / f"walksat_n{n}.json"
        if pth.exists():
            d = json.loads(pth.read_text())
            ws[n] = {
                "flips": d.get("flips_across_inst"),
                "evals": d.get("evals_across_inst"),
                "n_inst": d.get("n_inst"),
                "R": d.get("R"),
                "split": d.get("split"),
            }
    results["walksat"] = ws

    (OUT / "results.json").write_text(json.dumps(_jsonable(results), indent=2))

    # RESULTS.md
    lines = [
        "# renames_Hc wave — SNAP φ noise + soft-n18 (+ retrain seed)",
        "",
        f"_Generated {time.strftime('%Y-%m-%d %H:%M:%S %z')} (Asia/Shanghai)_",
        "",
        "Honesty: still `Uc=exp(-iγ Hc)`. Blind cost banned.",
        "",
        "Opus-style pick: (1) SNAP φ noise (2) WalkSAT fair table (3) soft-n18 (4) L36/40 defer.",
        "Top-2 under renames_Hc: SNAP φ noise + soft-n18. WalkSAT filled in parallel.",
        "",
        "## Locked prior champ (LBFGS seed0) — reference",
        "",
        "| n | Med | B90 |",
        "|---|-----|-----|",
    ]
    for n in (9, 12, 15, 18):
        lines.append(f"| {n} | {LOCKED[n]['Med']:.3f} | {LOCKED[n]['B90']:.3f} |")
    lines += [
        "",
        "## Retrain L32 seed (params were missing from git)",
        "",
        "| n | Med | B90 |",
        "|---|-----|-----|",
    ]
    lines += med_b90_table(sc0)
    rb = results.get("retrain_b") or {}
    lines.append(""); lines.append(f"b_Med≈**{(rb.get('b_Med') or {}).get('b', float('nan')):.4f}**")

    if "soft_n18" in results:
        s = results["soft_n18"]
        lines += [
            "",
            f"## Soft-n18 (keep={s['keep']})",
            "",
            "| n | Med | B90 |",
            "|---|-----|-----|",
        ]
        lines += med_b90_table(s["scale1000"])
        lines.append("")
        for r in s["reasons"]:
            lines.append(f"- {r}")

    if "snap_phase_noise" in results:
        lines += [
            "",
            "## SNAP φ phase noise (freeze angles; φ ← φ+σξ)",
            "",
            "| σ | n9 Med/B90 | n12 Med/B90 | vs σ=0 n9 Med |",
            "|---|------------|-------------|----------------|",
        ]
        base_med = None
        for k, block in results["snap_phase_noise"].items():
            row = block["scale1000"]
            m9, m12 = row[9], row[12]
            if base_med is None:
                base_med = m9["Med"]
            ratio = m9["Med"] / base_med if base_med else float("nan")
            lines.append(
                f"| {block['sigma']:.2f} | {m9['Med']:.3f}/{m9['B90']:.3f} | "
                f"{m12['Med']:.3f}/{m12['B90']:.3f} | ×{ratio:.3f} |"
            )

    if ws:
        lines += [
            "",
            "## WalkSAT fair table (scale1000, R=20, max_inst=200)",
            "",
            "| n | flips Med/B90 | evals Med/B90 | champ 1/p Med/B90 |",
            "|---|---------------|---------------|-------------------|",
        ]
        for n in (9, 12, 15, 18):
            if n not in ws:
                continue
            f, e = ws[n]["flips"], ws[n]["evals"]
            c = LOCKED[n]
            lines.append(
                f"| {n} | {f['median']:.1f}/{f['B90']:.1f} | {e['median']:.1f}/{e['B90']:.1f} | "
                f"{c['Med']:.3f}/{c['B90']:.3f} |"
            )

    lines += [
        "",
        "## Notes",
        "",
        "- Retrain seed ≠ bit-identical to locked LBFGS seed0 (params never committed).",
        "- Soft-n18 promote gate vs **retrain** seed (not locked).",
        "- Still renames Hc; no blind stencil.",
        "",
    ]
    (OUT / "RESULTS.md").write_text("\n".join(lines) + "\n")
    log("wrote", OUT / "RESULTS.md")


if __name__ == "__main__":
    main()
