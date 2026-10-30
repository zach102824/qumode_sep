#!/usr/bin/env python3
"""Soft-include n18 FROM locked LBFGS seed0 champ (renames_Hc fix).

Prior soft-n18 on short retrain seed (~n18 Med 10–13) is INVALID vs locked.
Promote vs sealed locked Med/B90 only.
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
    _jsonable,
    build_W_BinH,
    eval_arm,
    make_sim,
    scale1000_eval,
)
from explore_OCT29.scripts.gadget_Uc_soft_n15 import load_train_E  # noqa: E402

OUT = _PKG / "runs" / "gadget_Uc" / "renames_Hc" / "soft_n18_from_champ"
CHAMP = _PKG / "runs" / "gadget_Uc" / "unsat_params.pt"
LOCKED = {
    9: {"Med": 1.092, "B90": 1.776},
    12: {"Med": 1.197, "B90": 2.843},
    15: {"Med": 1.418, "B90": 5.505},
    18: {"Med": 1.752, "B90": 11.353},
}
INVALID_PRIOR = {
    9: {"Med": 2.241, "B90": 5.200},
    12: {"Med": 3.555, "B90": 10.412},
    15: {"Med": 5.760, "B90": 18.361},
    18: {"Med": 9.802, "B90": 38.644},
}


def log(*a):
    print(time.strftime("%H:%M:%S"), "[soft18]", *a, flush=True)


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
    opt = torch.optim.Adam([gamma, phi], lr=1.5e-3)
    best = {"score": float("inf"), "state": None, "step": -1}
    wmap = weights or {}

    def score_vals(vals):
        # weight larger n but keep small-n honest
        return (
            0.15 * vals[9]["S"]
            + 0.20 * vals[12]["S"]
            + 0.25 * vals[15]["S"]
            + 0.40 * vals[18]["S"]
        )

    for step in range(steps):
        opt.zero_grad()
        losses, ws = [], []
        for n, Eall in packs.items():
            idx = int(rng.integers(0, len(Eall)))
            out = sims[n].forward_gadget(Eall[idx], gamma, phi, "unsat")
            losses.append(plain_loss(out["inv_p"].unsqueeze(0), out["p_leak"].unsqueeze(0)))
            ws.append(float(wmap.get(n, 1.0)))
        wt = torch.tensor(ws, dtype=torch.float32, device=losses[0].device)
        wt = wt / wt.sum()
        loss = (torch.stack(losses) * wt).sum()
        loss.backward()
        torch.nn.utils.clip_grad_norm_([gamma, phi], 5.0)
        opt.step()

        if step % 5 == 0 or step + 1 == steps:
            if step in (steps // 2, steps - 1):
                opt_l = torch.optim.LBFGS(
                    [gamma, phi], lr=0.12, max_iter=6, history_size=12, line_search_fn="strong_wolfe"
                )

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
                n: eval_arm(
                    n, W, gamma.detach(), phi.detach(), L, nf, device, "unsat",
                    split="val", max_count=12,
                )
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
                        "from": "locked_lbfgs_seed0",
                    },
                }
            if step % 10 == 0 or step + 1 == steps:
                bits = " ".join(
                    f"n{n}:{vals[n]['median_inv_p']:.2f}/{vals[n]['B90_inv_p']:.2f}"
                    for n in train_ns
                )
                log(f"step {step}: S={sc:.3f} {bits}")
    return best


def get_mb(sc, n):
    m = sc.get(n) or sc.get(str(n)) or {}
    return m.get("median_inv_p", m.get("Med")), m.get("B90_inv_p", m.get("B90"))


def promote(sc):
    """vs LOCKED sealed. Improve n18 Med OR B90 by ≥2%; n9/12/15 Med ≤105% locked."""
    reasons = []
    m18, b18 = get_mb(sc, 18)
    thr_med = LOCKED[18]["Med"] * 0.98
    thr_b90 = LOCKED[18]["B90"] * 0.98
    ok18 = (m18 is not None) and (m18 <= thr_med or b18 <= thr_b90)
    reasons.append(
        f"n18 Med {m18:.3f} (thr {thr_med:.3f}) B90 {b18:.3f} (thr {thr_b90:.3f}) → {'ok' if ok18 else 'no'}"
    )
    ok_small = True
    for n in (9, 12, 15):
        med, _ = get_mb(sc, n)
        lim = LOCKED[n]["Med"] * 1.05
        good = med is not None and med <= lim
        ok_small = ok_small and good
        reasons.append(f"n{n} Med {med:.3f} ≤ {lim:.3f} → {'ok' if good else 'no'}")
    keep = bool(ok18 and ok_small)
    return keep, reasons


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=int, default=32)
    ap.add_argument("--nf", type=int, default=8)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--scale_B", type=int, default=1000)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--n18_weight", type=float, default=0.5)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    W = build_W_BinH(nf=args.nf)
    st0 = torch.load(CHAMP, map_location="cpu", weights_only=False)
    g0, p0 = st0["gamma"].float(), st0["phi"].float()
    L = args.L
    assert g0.numel() == L, (g0.shape, L)

    results = {
        "honesty": "Uc=exp(-iγ Hc); soft-n18 FROM locked LBFGS seed0",
        "locked": LOCKED,
        "invalid_prior_soft_n18": INVALID_PRIOR,
        "invalid_note": (
            "Prior renames_Hc soft-n18 table (n18≈9.8/38.6) was on a bad short retrain "
            "seed, NOT locked champ — INVALID for promote decisions."
        ),
        "champ_source": str(CHAMP),
    }

    configs = [
        ("t9121518", (9, 12, 15, 18), {9: 1.0, 12: 1.0, 15: 0.6, 18: args.n18_weight}),
        ("t9121518_w1", (9, 12, 15, 18), {9: 1.0, 12: 1.0, 15: 0.7, 18: 1.0}),
    ]
    results["arms"] = {}

    for name, train_ns, weights in configs:
        log(f"=== {name} {train_ns} w={weights} ===")
        best = soft_train(
            W, L, args.nf, device, train_ns,
            steps=args.steps, max_train=48, seed=180 + hash(name) % 50,
            gamma0=g0, phi0=p0, weights=weights,
        )
        st = best["state"]
        torch.save(st, OUT / f"{name}_params.pt")
        sc = scale1000_eval(
            W, st["gamma"], st["phi"], L, args.nf, device, "unsat",
            ns=[9, 12, 15, 18], max_count=args.scale_B,
        )
        keep, reasons = promote(sc)
        ns = [9, 12, 15, 18]
        meds = [get_mb(sc, n)[0] for n in ns]
        b90s = [get_mb(sc, n)[1] for n in ns]
        results["arms"][name] = {
            "train_ns": list(train_ns),
            "weights": weights,
            "train_best_step": best["step"],
            "scale1000": sc,
            "keep": keep,
            "reasons": reasons,
            "b_Med": fit_b(ns, meds),
            "b_B90": fit_b(ns, b90s),
        }
        log(f"{name} keep={keep}", reasons)
        (OUT / "results.json").write_text(json.dumps(_jsonable(results), indent=2))

    # pick best keep by n18 B90 else best n18 Med among keeps; else none
    winner = None
    best_key = None
    for name, block in results["arms"].items():
        if not block["keep"]:
            continue
        m18, b18 = get_mb(block["scale1000"], 18)
        key = (b18, m18)
        if best_key is None or key < best_key:
            best_key = key
            winner = name
    results["winner"] = winner

    if winner:
        st = torch.load(OUT / f"{winner}_params.pt", map_location="cpu", weights_only=False)
        torch.save(st, OUT / "promoted_params.pt")
        # promote to champ with backup
        bak = _PKG / "runs" / "gadget_Uc" / "unsat_params_pre_soft18_backup.pt"
        if not bak.exists():
            import shutil
            shutil.copy(CHAMP, bak)
        torch.save(st, CHAMP)
        torch.save(st, _PKG / "runs" / "gadget_Uc" / "lbfgs_polish" / "soft18_promoted_params.pt")
        log("PROMOTED", winner, "→ unsat_params.pt")
    else:
        log("NO PROMOTE — locked champ unchanged")

    # RESULTS.md
    lines = [
        "# Soft-n18 FROM locked LBFGS seed0 (renames_Hc fix)",
        "",
        f"_Generated {time.strftime('%Y-%m-%d %H:%M:%S %z')} (Asia/Shanghai)_",
        "",
        "Honesty: still `Uc=exp(-iγ Hc)`. **No blind-cost.**",
        "",
        "## CRITICAL: prior soft-n18 INVALID",
        "",
        "Earlier `renames_Hc` soft-n18 table (n9–18 ≈ 2.24/5.20 … **9.80/38.64**) was trained from a",
        "**bad short retrain seed**, not locked LBFGS seed0. **Do not cite for promote.**",
        "",
        "## Locked champ (verified scale1000)",
        "",
        "| n | Med | B90 |",
        "|---|-----|-----|",
    ]
    for n in (9, 12, 15, 18):
        lines.append(f"| {n} | {LOCKED[n]['Med']:.3f} | {LOCKED[n]['B90']:.3f} |")
    lines += [
        "",
        "Gate: n18 Med≤98% locked **or** B90≤98% locked; AND n9/12/15 Med ≤105% locked.",
        "",
        "## Arms (scale1000 B=1000)",
        "",
        "| arm | n9 | n12 | n15 | n18 | keep |",
        "|-----|----|-----|-----|-----|------|",
    ]

    def cell(block, n):
        med, b90 = get_mb(block["scale1000"], n)
        return f"{med:.3f}/{b90:.3f}" if med is not None else "—"

    for name, block in results["arms"].items():
        lines.append(
            f"| {name} | {cell(block,9)} | {cell(block,12)} | {cell(block,15)} | "
            f"{cell(block,18)} | {block['keep']} |"
        )
        for r in block["reasons"]:
            lines.append(f"  - {r}")
    lines += [
        "",
        f"**Winner:** `{winner}`" if winner else "**Winner:** none (locked unchanged)",
        "",
        "Fair compare: zero-shot honesty = locked was trained without n18; soft-n18 uses n18 val",
        f"(B={16} typical). scale1000 remains holdout.",
        "",
    ]
    (OUT / "RESULTS.md").write_text("\n".join(lines) + "\n")
    (OUT / "results.json").write_text(json.dumps(_jsonable(results), indent=2))
    log("wrote", OUT / "RESULTS.md")


if __name__ == "__main__":
    main()
