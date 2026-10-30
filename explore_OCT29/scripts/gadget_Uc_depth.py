#!/usr/bin/env python3
"""Deploy-depth probe on projector-Uc champ (BinH).

1) Pad/interp L=16 angles → L∈{20,24}, scale1000 eval (no retrain)
2) Retrain free γ/φ at L=20 (and L=24 if time) on {6,9,12}
Promote: n15 Med < 2.754*0.98 and n18 Med < 4.194*0.98

Also fit b_Med/b_B90 for champ vs Arm C / QAOA L16.
Writes runs/gadget_Uc/depth/
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
    ARM_C_B1000,
    _jsonable,
    build_W_BinH,
    eval_arm,
    scale1000_eval,
    train_arm,
)

OUT = _PKG / "runs" / "gadget_Uc" / "depth"
CHAMP = _PKG / "runs" / "gadget_Uc" / "unsat_params.pt"
BASE = {
    9: {"Med": 1.406, "B90": 3.288},
    12: {"Med": 1.864, "B90": 6.106},
    15: {"Med": 2.754, "B90": 10.244},
    18: {"Med": 4.194, "B90": 24.302},
}
QAOA_B = {"b_Med_9_18": 1.1511, "b_B90_9_18": 1.2211}
ARMC_B = {"b_Med_9_18": 1.1369, "b_B90_9_18": 1.2604}


def log(*a):
    print(time.strftime("%H:%M:%S"), "[depth]", *a, flush=True)


def interp_angles(gamma: torch.Tensor, phi: torch.Tensor, L_new: int):
    """Linear interp schedule from L0 → L_new."""
    g = gamma.detach().cpu().numpy().reshape(-1)
    p = phi.detach().cpu().numpy()
    L0 = g.shape[0]
    x = np.linspace(0, 1, L0)
    xp = np.linspace(0, 1, L_new)
    g2 = np.interp(xp, x, g)
    p2 = np.stack([np.interp(xp, x, p[:, s]) for s in range(4)], axis=1)
    return torch.as_tensor(g2, dtype=torch.float32), torch.as_tensor(p2, dtype=torch.float32)


def fit_windows(meds: dict, b90s: dict):
    ns = [9, 12, 15, 18]
    out = {}
    windows = {
        "9_18": (9, 12, 15, 18),
        "9_15": (9, 12, 15),
        "12_18": (12, 15, 18),
        "9_12": (9, 12),
        "12_15": (12, 15),
        "15_18": (15, 18),
    }
    for name, wns in windows.items():
        m = [meds[n] for n in wns]
        b = [b90s[n] for n in wns]
        out[name] = {"Med": fit_b(list(wns), m), "B90": fit_b(list(wns), b)}
    return out


def gate(sc: dict) -> tuple[bool, list[str]]:
    reasons = []
    keep = True
    for n, thr_base in ((15, BASE[15]["Med"]), (18, BASE[18]["Med"])):
        m = sc.get(n) or sc.get(str(n))
        thr = thr_base * 0.98
        if not m:
            keep = False
            reasons.append(f"missing n{n}")
            continue
        ok = m["median_inv_p"] < thr
        reasons.append(
            f"n{n} Med {m['median_inv_p']:.3f} {'<' if ok else '>='} {thr:.3f}"
        )
        keep = keep and ok
    return keep, reasons


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nf", type=int, default=8)
    ap.add_argument("--steps", type=int, default=80)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--pad_Ls", default="20,24")
    ap.add_argument("--retrain_Ls", default="20")
    ap.add_argument("--scale_ns", default="9,12,15,18")
    ap.add_argument("--scale_B", type=int, default=1000)
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    W = build_W_BinH(nf=args.nf)
    st0 = torch.load(CHAMP, map_location="cpu", weights_only=False)
    g0, p0 = st0["gamma"].float(), st0["phi"].float()
    scale_ns = [int(x) for x in args.scale_ns.split(",") if x.strip()]

    # --- b fits for sealed champ ---
    meds = {n: BASE[n]["Med"] for n in (9, 12, 15, 18)}
    b90s = {n: BASE[n]["B90"] for n in (9, 12, 15, 18)}
    bfits = fit_windows(meds, b90s)
    log(
        "champ b_Med_9_18", f"{bfits['9_18']['Med']['b']:.4f}",
        "b_B90", f"{bfits['9_18']['B90']['b']:.4f}",
        "vs ArmC", ARMC_B, "vs QAOA", QAOA_B,
    )

    results = {
        "champ_BASE": BASE,
        "champ_b_fits": bfits,
        "refs": {"ArmC_B1000": ARMC_B, "QAOA_L16": QAOA_B, "ArmC_Med_B90": ARM_C_B1000},
        "pad": {},
        "retrain": {},
    }

    # --- pad/interp deploy ---
    pad_Ls = [int(x) for x in args.pad_Ls.split(",") if x.strip()]
    for L in pad_Ls:
        log(f"=== pad deploy L={L} ===")
        g, p = interp_angles(g0, p0, L)
        torch.save({"gamma": g, "phi": p, "L": L, "kind": "pad"}, OUT / f"pad_L{L}.pt")
        sc = scale1000_eval(W, g, p, L, args.nf, device, "unsat", ns=scale_ns, max_count=args.scale_B)
        keep, reasons = gate(sc)
        results["pad"][L] = {"scale1000": sc, "keep": keep, "reasons": reasons}
        log(f"pad L{L} keep={keep}", reasons)
        (OUT / "results.json").write_text(json.dumps(_jsonable(results), indent=2))

    # --- retrain at deeper L ---
    retrain_Ls = [int(x) for x in args.retrain_Ls.split(",") if x.strip()]
    for L in retrain_Ls:
        log(f"=== retrain L={L} train{{6,9,12}} ===")
        g_w, p_w = interp_angles(g0, p0, L)
        best = train_arm(
            "unsat", W, L, args.nf, device,
            steps=args.steps, max_train=32, max_val=12, seed=5 + L,
            gamma0=g_w, phi0=p_w, train_ns=(6, 9, 12),
        )
        st = best["state"]
        st["L"] = L
        st["kind"] = "retrain"
        torch.save(st, OUT / f"retrain_L{L}_params.pt")
        sc = scale1000_eval(
            W, st["gamma"], st["phi"], L, args.nf, device, "unsat",
            ns=scale_ns, max_count=args.scale_B,
        )
        keep, reasons = gate(sc)
        results["retrain"][L] = {
            "train_best_step": best["step"],
            "train_score": best["score"],
            "scale1000": sc,
            "keep": keep,
            "reasons": reasons,
        }
        log(f"retrain L{L} keep={keep}", reasons)
        (OUT / "results.json").write_text(json.dumps(_jsonable(results), indent=2))

    # --- RESULTS.md ---
    lines = [
        "# Deploy-depth probe — projector-Uc",
        "",
        f"_Generated {time.strftime('%Y-%m-%d %H:%M:%S %z')} (Asia/Shanghai)_",
        "",
        "## Champ scaling exponents (scale1000)",
        "",
        f"| window | b_Med | b_B90 | Arm C b_Med/B90 | QAOA L16 b_Med/B90 |",
        f"|--------|-------|-------|-----------------|---------------------|",
    ]
    for w in ("9_18", "9_15", "12_18", "15_18"):
        bm = bfits[w]["Med"]["b"]
        bb = bfits[w]["B90"]["b"]
        if w == "9_18":
            lines.append(
                f"| {w} | **{bm:.4f}** | **{bb:.4f}** | "
                f"{ARMC_B['b_Med_9_18']:.4f}/{ARMC_B['b_B90_9_18']:.4f} | "
                f"{QAOA_B['b_Med_9_18']:.4f}/{QAOA_B['b_B90_9_18']:.4f} |"
            )
        else:
            lines.append(f"| {w} | {bm:.4f} | {bb:.4f} | — | — |")
    lines += [
        "",
        f"Champ b_Med≈**{bfits['9_18']['Med']['b']:.4f}** slightly **better** (lower) than Arm C "
        f"{ARMC_B['b_Med_9_18']:.4f} and QAOA {QAOA_B['b_Med_9_18']:.4f}. "
        f"b_B90≈**{bfits['9_18']['B90']['b']:.4f}** vs Arm C {ARMC_B['b_B90_9_18']:.4f} / QAOA {QAOA_B['b_B90_9_18']:.4f}.",
        "",
        f"Promote gate: n15 Med < {BASE[15]['Med']*0.98:.3f}, n18 Med < {BASE[18]['Med']*0.98:.3f}.",
        "",
        "## Pad/interp deploy (no retrain)",
        "",
        "| L | n15 | n18 | keep |",
        "|---|-----|-----|------|",
    ]

    def cell(block, n):
        sc = block.get("scale1000") or {}
        m = sc.get(n) or sc.get(str(n))
        return f"{m['median_inv_p']:.3f}/{m['B90_inv_p']:.3f}" if m else "—"

    for L in pad_Ls:
        b = results["pad"].get(L) or results["pad"].get(str(L)) or {}
        lines.append(f"| {L} | {cell(b,15)} | {cell(b,18)} | {b.get('keep')} |")
    lines += ["", "## Retrain free γ/φ at deeper L", "",
              "| L | n15 | n18 | keep |", "|---|-----|-----|------|"]
    for L in retrain_Ls:
        b = results["retrain"].get(L) or results["retrain"].get(str(L)) or {}
        lines.append(f"| {L} | {cell(b,15)} | {cell(b,18)} | {b.get('keep')} |")

    any_keep = any(
        (results["pad"].get(L) or {}).get("keep") for L in pad_Ls
    ) or any(
        (results["retrain"].get(L) or {}).get("keep") for L in retrain_Ls
    )
    lines += ["", "## Verdict", ""]
    if any_keep:
        lines.append("PROMOTE deeper-L schedule — see keep=True row.")
        nxt = "Lock deeper L as champ; optional L=24 retrain if only L=20 promoted."
    else:
        lines.append("DISCARD depth probe — keep L=16 champ `unsat_params.pt`.")
        nxt = (
            "Next probe: **span-split γ_ℓ^(s)** on BinH projector-Uc (probe8-style, 3 γ per layer) "
            "warm from champ — different from discarded shared span_w. "
            "Or L=1 projector sanity (cost-only baseline) for documentation."
        )
    lines += ["", f"**Next (auto):** {nxt}", ""]
    results["next_probe"] = nxt
    results["any_keep"] = any_keep
    (OUT / "results.json").write_text(json.dumps(_jsonable(results), indent=2))
    (OUT / "RESULTS.md").write_text("\n".join(lines) + "\n")
    log("wrote", OUT / "RESULTS.md", "any_keep=", any_keep)


if __name__ == "__main__":
    main()
