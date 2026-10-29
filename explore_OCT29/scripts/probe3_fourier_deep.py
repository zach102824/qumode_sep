#!/usr/bin/env python3
"""Deepen Fourier on projector (q=3, deeper L, optional L_train=6).

Leaves probe5 alone. Writes explore_OCT29/runs/probe3_fourier_deep/.
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

from explore_OCT29.lib.drivers import CompiledW, fit_b  # noqa: E402
from explore_OCT29.scripts.probe3_fourier import (  # noqa: E402
    angles_from_modes,
    eval_at_L,
    init_modes_from_angles,
    train_modes,
    write_report as _unused,
)

OUT = _PKG / "runs" / "probe3_fourier_deep"
W_PATH = _PKG / "runs" / "driver_bakeoff" / "W.pt"
P2 = _PKG / "runs" / "probe2_train912" / "projector_912_params.pt"
P3 = _PKG / "runs" / "probe3_fourier" / "results.json"
P3_MODES = _PKG / "runs" / "probe3_fourier" / "fourier_modes.pt"


def log(*a):
    print(time.strftime("%H:%M:%S"), "[fourier_deep]", *a, flush=True)


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, torch.Tensor):
        return o.detach().cpu().tolist()
    return o


def sweep(W, u_g, u_b, Ls, ns, max_counts, nf, device):
    out = {}
    for n in ns:
        if not (_PKG / "data" / f"data_n{n}.npz").exists():
            continue
        out[n] = {}
        for L in Ls:
            m = eval_at_L(W, u_g, u_b, n, L, nf, max_counts.get(n, 4), device)
            out[n][L] = m
            log(f"n{n} L{L}: Med={m['median_inv_p']:.4g} B90={m['B90_inv_p']:.4g} S={m['S']:.3f}")
    return out


def fits_by_L(sweep_d, Ls):
    fb = {}
    for L in Ls:
        ns, meds, b90s = [], [], []
        for n in (9, 12, 15, 18):
            m = (sweep_d.get(n) or {}).get(L)
            if m and "median_inv_p" in m:
                ns.append(n); meds.append(m["median_inv_p"]); b90s.append(m["B90_inv_p"])
        fb[L] = {"Med": fit_b(ns, meds), "B90": fit_b(ns, b90s)}
    return fb


def mono_med(sweep_d, n, Ls):
    vals = [(L, (sweep_d.get(n) or {}).get(L, {}).get("median_inv_p")) for L in Ls]
    vals = [(L, v) for L, v in vals if v is not None]
    ok = all(vals[i + 1][1] <= vals[i][1] * 1.05 for i in range(len(vals) - 1)) if len(vals) > 1 else False
    return ok, vals


def run_config(tag, W, q, L_train, Ls, steps, max_train, max_val, max_counts, nf, seed, device, warm_g, warm_b):
    log(f"=== {tag}: q={q} L_train={L_train} deploy Ls={Ls} ===")
    best = train_modes(
        W, q, L_train, steps, max_train, max_val, nf, seed, device,
        warm_g=warm_g, warm_b=warm_b,
    )
    torch.save({
        "u_gamma": best["u_g"], "u_beta": best["u_b"], "q": q, "L_train": L_train,
        "tag": tag, "score": best["score"], "step": best["step"],
    }, OUT / f"modes_{tag}.pt")
    sw = sweep(W, best["u_g"].to(device), best["u_b"].to(device), Ls, (9, 12, 15, 18), max_counts, nf, device)
    fb = fits_by_L(sw, Ls)
    mono15, series15 = mono_med(sw, 15, Ls)
    mono18, series18 = mono_med(sw, 18, Ls)
    med15_L12 = (sw.get(15) or {}).get(12, {}).get("median_inv_p")
    med15_deep = (sw.get(15) or {}).get(Ls[-1], {}).get("median_inv_p")
    return {
        "tag": tag, "q": q, "L_train": L_train, "Ls": Ls,
        "u_gamma": best["u_g"].tolist(), "u_beta": best["u_b"].tolist(),
        "best_Scomb": best["score"], "best_step": best["step"],
        "sweep": sw, "fits_by_L": fb,
        "mono_n15": mono15, "series_n15": series15,
        "mono_n18": mono18, "series_n18": series18,
        "med15_L12": med15_L12, "med15_Lmax": med15_deep,
        "b_Med_L12": (fb.get(12) or {}).get("Med", {}).get("b"),
        "b_Med_Lmax": (fb.get(Ls[-1]) or {}).get("Med", {}).get("b"),
    }


def write_md(results, probe3_med15_L12, out_dir):
    lines = [
        "# Fourier deepen (q=3 / deeper L / L_train=6)\n\n",
        f"_Generated {time.strftime('%Y-%m-%d %H:%M:%S %z')} (Asia/Shanghai)_\n\n",
        f"Probe3 reference: q=2 L_train=4 deploy L12 → n15 Med≈**{probe3_med15_L12:.3g}**.\n\n",
    ]
    for block in results.get("runs", []):
        tag = block["tag"]
        lines.append(f"## {tag} (q={block['q']}, L_train={block['L_train']})\n\n")
        Ls = block["Ls"]
        lines.append("| n \\ L | " + " | ".join(str(L) for L in Ls) + " |\n")
        lines.append("|-------|" + "|".join(["------"] * len(Ls)) + "|\n")
        sw = block["sweep"]
        for n in (9, 12, 15, 18):
            cells = []
            for L in Ls:
                m = (sw.get(n) or {}).get(L) or {}
                if "median_inv_p" in m:
                    cells.append(f"{m['median_inv_p']:.3g}/{m['B90_inv_p']:.3g}")
                else:
                    cells.append("—")
            lines.append(f"| {n} | " + " | ".join(cells) + " |\n")
        lines.append("\n| L | b_Med | b_B90 |\n|---|-------|-------|\n")
        for L, f in block["fits_by_L"].items():
            lines.append(f"| {L} | {f['Med']['b']:.3f} | {f['B90']['b']:.3f} |\n")
        lines.append(
            f"\nMonotonic n15≈{block['mono_n15']} series={block['series_n15']}. "
            f"n15@L12={block.get('med15_L12')} n15@Lmax={block.get('med15_Lmax')}.\n\n"
        )
    lines.append("## Verdict\n\n")
    lines.append(results.get("verdict", "—") + "\n")
    (out_dir / "RESULTS.md").write_text("".join(lines))
    (out_dir / "results.json").write_text(json.dumps(_jsonable(results), indent=2) + "\n")
    log(f"wrote {out_dir/'RESULTS.md'}")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=80)
    ap.add_argument("--max_train", type=int, default=32)
    ap.add_argument("--max_val", type=int, default=12)
    ap.add_argument("--max_val15", type=int, default=6)
    ap.add_argument("--max_val18", type=int, default=4)
    ap.add_argument("--nf", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--skip_L6", action="store_true")
    ap.add_argument("--skip_L20", action="store_true")
    args = ap.parse_args(argv)
    torch.set_num_threads(max(1, int(os.environ.get("OMP_NUM_THREADS", "2"))))
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    t00 = time.time()

    W = CompiledW.load(W_PATH, device=device)
    warm_g = warm_b = None
    if P2.exists():
        blob = torch.load(P2, map_location="cpu", weights_only=False)
        warm_g, warm_b = blob["gamma"], blob["beta"]
        log("warm from probe2 free angles")
    # also try fit q=3 from probe3 q=2 modes if present
    if P3_MODES.exists() and warm_g is not None:
        # prefer free-angle warm (richer); modes init happens inside train via warm_g
        pass

    probe3_med15 = 9.67
    if P3.exists():
        js = json.loads(P3.read_text())
        try:
            probe3_med15 = float(js["sweep"]["15"]["12"]["median_inv_p"])
        except Exception:
            pass
    log(f"probe3 n15@L12 reference = {probe3_med15:.4g}")

    max_counts = {9: args.max_val, 12: args.max_val, 15: args.max_val15, 18: args.max_val18}
    Ls_main = [4, 6, 8, 12, 16]
    runs = []

    # 1) q=3, train L=4
    r1 = run_config(
        "q3_Ltrain4", W, q=3, L_train=4, Ls=Ls_main, steps=args.steps,
        max_train=args.max_train, max_val=args.max_val, max_counts=max_counts,
        nf=args.nf, seed=args.seed, device=device, warm_g=warm_g, warm_b=warm_b,
    )
    runs.append(r1)
    write_md({"runs": runs, "verdict": "(partial)"}, probe3_med15, OUT)

    # 2) q=3, train L=6 (INTERP-ish: deeper train, deploy deeper)
    if not args.skip_L6:
        # warm: sample probe2 angles... better: take q3 modes from r1, evaluate schedule at L=6 as init via lstsq on those angles
        g6, b6 = angles_from_modes(
            torch.as_tensor(r1["u_gamma"]), torch.as_tensor(r1["u_beta"]), 6
        )
        r2 = run_config(
            "q3_Ltrain6", W, q=3, L_train=6, Ls=Ls_main + ([20] if not args.skip_L20 else []),
            steps=args.steps, max_train=args.max_train, max_val=args.max_val,
            max_counts=max_counts, nf=args.nf, seed=args.seed + 7, device=device,
            warm_g=g6, warm_b=b6,
        )
        runs.append(r2)
        write_md({"runs": runs, "verdict": "(partial)"}, probe3_med15, OUT)

    # 3) deep deploy of best q=2 modes from probe3 (no retrain) at L=16,20
    q2_block = None
    if P3_MODES.exists():
        blob = torch.load(P3_MODES, map_location="cpu", weights_only=False)
        ug = blob["u_gamma"].to(device)
        ub = blob["u_beta"].to(device)
        Ls_q2 = [4, 6, 8, 12, 16] + ([] if args.skip_L20 else [20])
        log(f"=== q2_redeploy deep Ls={Ls_q2} (no retrain) ===")
        sw = sweep(W, ug, ub, Ls_q2, (9, 12, 15, 18), max_counts, args.nf, device)
        fb = fits_by_L(sw, Ls_q2)
        mono15, s15 = mono_med(sw, 15, Ls_q2)
        q2_block = {
            "tag": "q2_redeploy_deep", "q": 2, "L_train": int(blob.get("L_train", 4)),
            "Ls": Ls_q2, "u_gamma": ug.cpu().tolist(), "u_beta": ub.cpu().tolist(),
            "best_Scomb": None, "best_step": None,
            "sweep": sw, "fits_by_L": fb, "mono_n15": mono15, "series_n15": s15,
            "mono_n18": mono_med(sw, 18, Ls_q2)[0], "series_n18": mono_med(sw, 18, Ls_q2)[1],
            "med15_L12": (sw.get(15) or {}).get(12, {}).get("median_inv_p"),
            "med15_Lmax": (sw.get(15) or {}).get(Ls_q2[-1], {}).get("median_inv_p"),
            "b_Med_L12": (fb.get(12) or {}).get("Med", {}).get("b"),
            "b_Med_Lmax": (fb.get(Ls_q2[-1]) or {}).get("Med", {}).get("b"),
        }
        runs.append(q2_block)

    # pick best by n15 at deepest reasonable L (prefer L12 then L16)
    def score(block):
        m12 = block.get("med15_L12") or 1e9
        mmax = block.get("med15_Lmax") or 1e9
        return min(m12, mmax)

    best = min(runs, key=score)
    beat = (best.get("med15_L12") or 1e9) < probe3_med15 * 0.98 or (
        best.get("med15_Lmax") or 1e9
    ) < probe3_med15 * 0.98
    plateau = not beat and all(
        (b.get("med15_Lmax") or 1e9) > probe3_med15 * 0.9 for b in runs
    )
    verdict = (
        f"Best tag={best['tag']}: n15@L12={best.get('med15_L12')}, n15@Lmax={best.get('med15_Lmax')} "
        f"(probe3 L12 ref={probe3_med15:.3g}). mono_n15={best.get('mono_n15')}. "
    )
    if beat:
        verdict += "Deepen beat Probe3 n15 Med — keep Fourier deepen; probe5 continues; probe4 optional."
        nxt = "keep_fourier_probe5"
    elif plateau:
        verdict += "Deepen plateaus vs Probe3 — start probe 4 (K=4 span-split γ)."
        nxt = "probe4"
    else:
        verdict += "Mixed; prefer best deepen schedule + continue probe5; probe4 next if no further gain."
        nxt = "probe4_soon"

    results = {
        "runs": runs,
        "best_tag": best["tag"],
        "probe3_med15_L12_ref": probe3_med15,
        "verdict": verdict,
        "next": nxt,
        "wall_s": time.time() - t00,
        "finished": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    write_md(results, probe3_med15, OUT)
    log(verdict)
    log(f"DONE in {results['wall_s']:.0f}s → next={nxt}")
    # machine-readable next hint
    (OUT / "NEXT").write_text(nxt + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
