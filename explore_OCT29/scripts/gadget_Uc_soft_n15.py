#!/usr/bin/env python3
"""Soft-include n15 in projector-Uc L32 train.

n15 pack has empty train_*; use val (40) as soft pool (scale1000 = holdout).
Arms: train{6,9,12,15} and train{9,12,15}. Promote if:
  n18 Med < 1.995*0.98  OR  (n15+n18 both improve vs champ and n9/12 Med not worse than ×1.05).

Also writes GRAPE library freeze recommendation into RESULTS.
Optional cheap follow-up: longer retrain L32 {6,9,12} with more steps (control).
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
from explore_OCT29.lib.train_loop import _split_arrays, load_pack  # noqa: E402
from explore_OCT29.scripts.gadget_Uc_explore import (  # noqa: E402
    ARM_C_B1000,
    _S,
    _jsonable,
    _loss,
    build_W_BinH,
    make_sim,
    scale1000_eval,
)

OUT = _PKG / "runs" / "gadget_Uc" / "soft_n15"
CHAMP = _PKG / "runs" / "gadget_Uc" / "unsat_params.pt"
BASE = {
    9: {"Med": 1.109, "B90": 1.828},
    12: {"Med": 1.258, "B90": 3.013},
    15: {"Med": 1.537, "B90": 5.767},
    18: {"Med": 1.995, "B90": 14.117},
}


def log(*a):
    print(time.strftime("%H:%M:%S"), "[soft15]", *a, flush=True)


def load_train_E(n, max_train, device):
    """For n15 use val if train empty."""
    pack = load_pack(n)
    _, _, _, _, E = _split_arrays(pack, "train", max_count=max_train)
    src = "train"
    if E is None or len(E) == 0:
        _, _, _, _, E = _split_arrays(pack, "val", max_count=max_train)
        src = "val"
    if E is None or len(E) == 0:
        raise RuntimeError(f"n={n}: no train/val E")
    log(f"pool n{n}: src={src} B={len(E)}")
    return torch.as_tensor(E, dtype=torch.float32, device=device), src


def train_multi(
    W, L, nf, device, train_ns, steps=100, max_train=32, max_val=12, seed=0,
    gamma0=None, phi0=None, lr=2e-3, n15_weight=0.5,
):
    rng = np.random.default_rng(seed)
    gamma = torch.nn.Parameter(
        gamma0.detach().clone() if gamma0 is not None else torch.linspace(0.3, 1.0, L)
    )
    phi = torch.nn.Parameter(
        phi0.detach().clone() if phi0 is not None else torch.zeros(L, 4)
    )
    params = [gamma, phi]
    packs = {}
    sources = {}
    for n in train_ns:
        E, src = load_train_E(n, max_train, device)
        packs[n] = E
        sources[n] = src
    sims = {n: make_sim(n, W, L, nf, device) for n in train_ns}
    opt = torch.optim.Adam(params, lr=lr)
    best = {"score": float("inf"), "state": None, "val": None, "step": -1}

    # score: prefer n12 and n15 if present else n12
    def score_vals(vals):
        if 15 in vals and 12 in vals:
            return 0.3 * vals[12]["S"] + 0.5 * vals[15]["S"] + 0.2 * vals.get(9, vals[12])["S"]
        if 12 in vals and 9 in vals:
            return 0.4 * vals[9]["S"] + 0.6 * vals[12]["S"]
        return list(vals.values())[0]["S"]

    for step in range(steps):
        opt.zero_grad()
        losses = []
        weights = []
        for n, Eall in packs.items():
            idx = int(rng.integers(0, len(Eall)))
            out = sims[n].forward_gadget(Eall[idx], gamma, phi, "unsat")
            losses.append(_loss(out))
            weights.append(float(n15_weight) if n == 15 else 1.0)
        w = torch.tensor(weights, dtype=torch.float32, device=losses[0].device)
        w = w / w.sum()
        loss = (torch.stack(losses) * w).sum()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 5.0)
        opt.step()

        if step % 5 == 0 or step + 1 == steps:
            if step in (steps // 2, steps - 1):
                opt_l = torch.optim.LBFGS(params, lr=0.15, max_iter=4, history_size=8)

                def closure():
                    opt_l.zero_grad()
                    ls, ws = [], []
                    for n, Eall in packs.items():
                        for j in range(min(6, len(Eall))):
                            o = sims[n].forward_gadget(Eall[j], gamma, phi, "unsat")
                            ls.append(_loss(o))
                            ws.append(float(n15_weight) if n == 15 else 1.0)
                    wt = torch.tensor(ws, dtype=torch.float32, device=ls[0].device)
                    wt = wt / wt.sum()
                    lt = (torch.stack(ls) * wt).sum()
                    lt.backward()
                    return lt

                try:
                    opt_l.step(closure)
                except Exception as e:
                    log("LBFGS skip", e)

            from explore_OCT29.scripts.gadget_Uc_explore import eval_arm
            vals = {}
            for n in train_ns:
                # for n15 val-trained: score on val is biased; still use for early stop
                vals[n] = eval_arm(
                    n, W, gamma.detach(), phi.detach(), L, nf, device, "unsat",
                    split="val", max_count=max_val,
                )
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
                log(f"train {train_ns} step {step}: S={sc:.3f} {bits}")
    return best


def promote(sc):
    """Return (keep, reasons, kind)."""
    reasons = []
    m18 = sc.get(18) or sc.get("str")
    m18 = sc.get(18) or sc.get("18")
    m15 = sc.get(15) or sc.get("15")
    m9 = sc.get(9) or sc.get("9")
    m12 = sc.get(12) or sc.get("12")
    thr18 = BASE[18]["Med"] * 0.98
    ok18 = m18 and m18["median_inv_p"] < thr18
    reasons.append(
        f"n18 {m18['median_inv_p']:.3f} {'<' if ok18 else '>='} {thr18:.3f} (zero-shot)"
    )
    joint = False
    if m15 and m18:
        better15 = m15["median_inv_p"] < BASE[15]["Med"]
        better18 = m18["median_inv_p"] < BASE[18]["Med"]
        ok9 = (not m9) or m9["median_inv_p"] <= BASE[9]["Med"] * 1.05
        ok12 = (not m12) or m12["median_inv_p"] <= BASE[12]["Med"] * 1.05
        joint = better15 and better18 and ok9 and ok12
        reasons.append(
            f"joint n15/n18 improve={better15}/{better18} n9/12 ok={ok9}/{ok12}"
        )
    keep = bool(ok18 or joint)
    kind = "n18_holdout" if ok18 else ("joint" if joint else "none")
    return keep, reasons, kind


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=int, default=32)
    ap.add_argument("--nf", type=int, default=8)
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--scale_B", type=int, default=1000)
    ap.add_argument("--n15_weight", type=float, default=0.5)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    W = build_W_BinH(nf=args.nf)
    st0 = torch.load(CHAMP, map_location="cpu", weights_only=False)
    g0, p0 = st0["gamma"].float(), st0["phi"].float()
    L = args.L

    results = {"baseline_L32": BASE, "arms": {}}
    configs = [
        ("t691215", (6, 9, 12, 15)),
        ("t91215", (9, 12, 15)),
    ]

    for name, train_ns in configs:
        log(f"=== {name} {train_ns} ===")
        best = train_multi(
            W, L, args.nf, device, train_ns, steps=args.steps, seed=15 + hash(name) % 20,
            gamma0=g0, phi0=p0, n15_weight=args.n15_weight,
        )
        st = best["state"]
        torch.save(st, OUT / f"{name}_params.pt")
        sc = scale1000_eval(
            W, st["gamma"], st["phi"], L, args.nf, device, "unsat",
            ns=[9, 12, 15, 18], max_count=args.scale_B,
        )
        keep, reasons, kind = promote(sc)
        results["arms"][name] = {
            "train_ns": list(train_ns),
            "train_best_step": best["step"],
            "scale1000": sc,
            "keep": keep,
            "kind": kind,
            "reasons": reasons,
            "sources": st.get("sources"),
        }
        log(f"{name} keep={keep} kind={kind}", reasons)
        (OUT / "results.json").write_text(json.dumps(_jsonable(results), indent=2))

    # pick best keep arm if any
    winner = None
    for name, block in results["arms"].items():
        if block["keep"]:
            winner = name
            break

    # cheap follow-up: if no promote, try higher n15_weight on t691215 briefly
    if winner is None:
        log("=== follow-up t691215 n15_weight=1.0 steps=60 ===")
        best = train_multi(
            W, L, args.nf, device, (6, 9, 12, 15), steps=60, seed=99,
            gamma0=g0, phi0=p0, n15_weight=1.0,
        )
        st = best["state"]
        torch.save(st, OUT / "t691215_w1_params.pt")
        sc = scale1000_eval(
            W, st["gamma"], st["phi"], L, args.nf, device, "unsat",
            ns=[9, 12, 15, 18], max_count=args.scale_B,
        )
        keep, reasons, kind = promote(sc)
        results["arms"]["t691215_w1"] = {
            "train_ns": [6, 9, 12, 15],
            "n15_weight": 1.0,
            "scale1000": sc,
            "keep": keep,
            "kind": kind,
            "reasons": reasons,
        }
        log(f"t691215_w1 keep={keep}", reasons)
        if keep:
            winner = "t691215_w1"
        (OUT / "results.json").write_text(json.dumps(_jsonable(results), indent=2))

    results["winner"] = winner

    # RESULTS.md
    lines = [
        "# Soft-include n15 on projector-Uc L32",
        "",
        f"_Generated {time.strftime('%Y-%m-%d %H:%M:%S %z')} (Asia/Shanghai)_",
        "",
        "n15 train_* empty → soft pool = **val** (40). Eval = scale1000 holdout (different seeds).",
        f"Gate: n18 Med < {BASE[18]['Med']*0.98:.3f} OR joint n15+n18 improve without n9/12 blow-up (>×1.05).",
        "",
        "## scale1000",
        "",
        "| arm | n9 | n12 | n15 | n18 | keep |",
        "|-----|----|-----|-----|-----|------|",
    ]

    def cell(block, n):
        m = (block.get("scale1000") or {}).get(n) or (block.get("scale1000") or {}).get(str(n))
        return f"{m['median_inv_p']:.3f}/{m['B90_inv_p']:.3f}" if m else "—"

    for name, block in results["arms"].items():
        lines.append(
            f"| {name} | {cell(block,9)} | {cell(block,12)} | {cell(block,15)} | "
            f"{cell(block,18)} | {block.get('keep')} |"
        )
    lines += [
        "",
        f"| L32 champ | {BASE[9]['Med']:.3f}/{BASE[9]['B90']:.3f} | "
        f"{BASE[12]['Med']:.3f}/{BASE[12]['B90']:.3f} | "
        f"{BASE[15]['Med']:.3f}/{BASE[15]['B90']:.3f} | "
        f"{BASE[18]['Med']:.3f}/{BASE[18]['B90']:.3f} | ref |",
        "",
    ]
    for name, block in results["arms"].items():
        lines.append(f"- **{name}**: keep={block.get('keep')} kind={block.get('kind')} — {'; '.join(block.get('reasons') or [])}")
    lines += ["", "## Verdict", ""]
    if winner:
        lines.append(f"**PROMOTE** `{winner}` → replace `unsat_params.pt`.")
    else:
        lines.append("**DISCARD** soft-n15 — keep L32 champ `unsat_params.pt`.")
    lines += [
        "",
        "## GRAPE library freeze recommendation",
        "",
        "With depth ladder stopped (L32) and span/soft-n15 exhausted or settled, **freeze the projector GRAPE library**:",
        "",
        "1. **26 local projectors** on 3-bit cavity (k=1,2,3) + two-cavity \(C_a\\otimes C_b\) for (1,3)/(2,2).",
        "2. Exact compile: \(U_Z(\\gamma)=\\mathrm{SNAP}(\\theta)\) on Fock support; BinH \(U_X=W^\\dagger\\mathrm{SNAP} W\).",
        "3. Instance only schedules which library ops fire (clause list) + shared \(\\gamma_\\ell,\\phi_{pc}\).",
        "4. Do **not** GRAPE densified `diag(E)` as a pulse target.",
        "5. Champ angles: L=32 shell SNAP on BinH; see `grape_library/LIBRARY.md`.",
        "",
        "Next explore (if any): hardware GRAPE fidelity for two-cavity tensors, not more classical depth.",
        "",
    ]
    (OUT / "RESULTS.md").write_text("\n".join(lines) + "\n")
    log("wrote", OUT / "RESULTS.md", "winner=", winner)


if __name__ == "__main__":
    main()
