#!/usr/bin/env python3
"""Projector-Uc probe: retrain BinH unsat on {6,9,12}; promote vs scale1000 baseline.

Baseline: n15 2.820/10.063, n18 4.329/24.342. Keep only if scale n15/n18 improve.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

_PKG = Path(__file__).resolve().parents[1]
_ROOT = _PKG.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from explore_OCT29.scripts.gadget_Uc_explore import (  # noqa: E402
    ARM_C_B1000,
    _jsonable,
    build_W_BinH,
    ladder_eval,
    scale1000_eval,
    train_arm,
)

OUT = _PKG / "runs" / "gadget_Uc" / "train6912"
UNSAT = _PKG / "runs" / "gadget_Uc" / "unsat_params.pt"
BASE = {
    9: {"Med": 1.445, "B90": 3.222},
    12: {"Med": 1.926, "B90": 6.148},
    15: {"Med": 2.820, "B90": 10.063},
    18: {"Med": 4.329, "B90": 24.342},
}


def log(*a):
    print(time.strftime("%H:%M:%S"), "[t6912]", *a, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--nf", type=int, default=8)
    ap.add_argument("--L", type=int, default=16)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--scale_ns", default="9,12,15,18")
    ap.add_argument("--scale_B", type=int, default=1000)
    ap.add_argument("--skip_n18", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    W = build_W_BinH(nf=args.nf)
    st0 = torch.load(UNSAT, map_location="cpu", weights_only=False)
    log("train unsat projector on {6,9,12} warm from unsat_params")
    # monkey: train_arm hardcodes train_ns — call with override by patching
    best = train_arm(
        "unsat", W, args.L, args.nf, device,
        steps=args.steps, max_train=32, max_val=12, seed=2,
        gamma0=st0["gamma"].float(), phi0=st0["phi"].float(),
        train_ns=(6, 9, 12),
    )
    # train_arm signature has train_ns — check if accepted
    st = best["state"]
    torch.save(st, OUT / "unsat_6912_params.pt")
    log("ladder")
    lad = ladder_eval(W, st["gamma"], st["phi"], args.L, args.nf, device, "unsat")
    scale_ns = [int(x) for x in args.scale_ns.split(",") if x.strip()]
    if args.skip_n18:
        scale_ns = [n for n in scale_ns if n != 18]
    log("scale1000", scale_ns)
    sc = scale1000_eval(
        W, st["gamma"], st["phi"], args.L, args.nf, device, "unsat",
        ns=scale_ns, max_count=args.scale_B,
    )
    # promote gate
    keep = False
    reasons = []
    for n in (15, 18):
        m = sc.get(n) or sc.get(str(n))
        if not m:
            reasons.append(f"missing scale n{n}")
            continue
        base = BASE[n]["Med"]
        if m["median_inv_p"] < base * 0.98:
            reasons.append(f"n{n} Med {m['median_inv_p']:.3f} < {base:.3f}*0.98")
        else:
            reasons.append(f"n{n} Med {m['median_inv_p']:.3f} NOT < {base:.3f}*0.98")
    m15 = sc.get(15) or sc.get("15")
    m18 = sc.get(18) or sc.get("18")
    if m15 and m18:
        keep = (m15["median_inv_p"] < BASE[15]["Med"] * 0.98) and (
            m18["median_inv_p"] < BASE[18]["Med"] * 0.98
        )
    elif m15:
        keep = m15["median_inv_p"] < BASE[15]["Med"] * 0.98

    results = {
        "train_ns": [6, 9, 12],
        "train_best_step": best["step"],
        "train_score": best["score"],
        "ladder": lad,
        "scale1000": sc,
        "baseline": BASE,
        "armC": ARM_C_B1000,
        "keep": keep,
        "reasons": reasons,
    }
    (OUT / "results.json").write_text(json.dumps(_jsonable(results), indent=2))

    lines = [
        "# Projector-Uc train {6,9,12} probe",
        "",
        f"_Generated {time.strftime('%Y-%m-%d %H:%M:%S %z')} (Asia/Shanghai)_",
        "",
        f"**keep={keep}** — " + "; ".join(reasons),
        "",
        "## scale1000 vs baseline projector / Arm C",
        "",
        "| n | train6912 | baseline proj | Arm C | vs base |",
        "|---|-----------|---------------|-------|---------|",
    ]
    for n in scale_ns:
        m = sc.get(n) or sc.get(str(n))
        if not m:
            continue
        b = BASE.get(n, {})
        a = ARM_C_B1000.get(n, {})
        rb = b.get("Med", float("nan")) / m["median_inv_p"]
        lines.append(
            f"| {n} | {m['median_inv_p']:.3f}/{m['B90_inv_p']:.3f} | "
            f"{b.get('Med', float('nan')):.3f}/{b.get('B90', float('nan')):.3f} | "
            f"{a.get('Med', float('nan')):.3f}/{a.get('B90', float('nan')):.3f} | ×{rb:.3f} |"
        )
    lines += [
        "",
        "## Verdict",
        "",
        ("PROMOTE — replace unsat_params.pt with unsat_6912_params.pt."
         if keep else
         "DISCARD — keep prior unsat_params.pt (train{9,12})."),
        "",
    ]
    (OUT / "RESULTS.md").write_text("\n".join(lines) + "\n")
    log("wrote", OUT / "RESULTS.md", "keep=", keep)


if __name__ == "__main__":
    main()
