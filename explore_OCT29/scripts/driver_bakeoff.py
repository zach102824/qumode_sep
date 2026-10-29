#!/usr/bin/env python3
"""Opus probe 1 — driver bake-off (TWO arms; Ideal Gray-X dropped).

Arms
  1. projector : W·SNAP₀(β)·W†  (compiled flat prep W)
  2. snap_d    : seq_best motif COST→SNAP→D at L=4

Protocol
  Phase A: linear-ramp Δ scan at n9 (no training)
  Phase B: L-BFGS/Adam on params at n9
  Eval val packs n9/12/15; fit b from Med and B90

Usage (from repo root, venv active):
  OMP_NUM_THREADS=2 python -m explore_OCT29.scripts.driver_bakeoff
  python -m explore_OCT29.scripts.driver_bakeoff --skip_compile   # reuse W.pt
"""
from __future__ import annotations

import argparse
import json
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

from explore_OCT29.lib.drivers import (  # noqa: E402
    BAKE_RUNS,
    CompiledW,
    compile_flat_prep,
    eval_projector,
    eval_snap_d,
    fit_b,
    linear_ramp,
    scan_delta_projector,
    scan_delta_snap_d,
    train_projector_lbfgs,
    train_snap_d_lbfgs,
)
from explore_OCT29.lib.seq_ansatz import Motif  # noqa: E402


def log(*a):
    print(time.strftime("%H:%M:%S"), "[bakeoff]", *a, flush=True)


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if isinstance(obj, Path):
        return str(obj)
    return obj


def eval_ladder(arm_name, eval_fn, ns=(9, 12, 15), max_counts=None, **kw):
    max_counts = max_counts or {9: 24, 12: 16, 15: 8}
    out = {}
    for n in ns:
        mc = max_counts.get(n, 8)
        try:
            m = eval_fn(n=n, max_count=mc, **kw)
            out[n] = m
            log(
                f"{arm_name} n{n}: Med={m['median_inv_p']:.3g} B90={m['B90_inv_p']:.3g} "
                f"S={m['S']:.3f} leak={m.get('mean_leak', float('nan')):.3e} ({m['wall_s']:.1f}s B={m['B']})"
            )
        except Exception as e:
            log(f"{arm_name} n{n} FAILED: {e}")
            out[n] = {"error": str(e), "n": n}
    return out


def fit_from_ladder(ladder: dict) -> dict:
    ns, meds, b90s = [], [], []
    for n in sorted(k for k in ladder if isinstance(k, int) or str(k).isdigit()):
        n = int(n)
        m = ladder[n] if n in ladder else ladder[str(n)]
        if "median_inv_p" not in m:
            continue
        ns.append(n)
        meds.append(m["median_inv_p"])
        b90s.append(m["B90_inv_p"])
    return {"Med": fit_b(ns, meds), "B90": fit_b(ns, b90s)}


def write_report(results: dict, out_dir: Path):
    lines = []
    lines.append("# Driver bake-off (Opus probe 1) — two arms\n")
    lines.append(f"_Generated {time.strftime('%Y-%m-%d %H:%M:%S %z')} (Asia/Shanghai box clock)_\n")
    lines.append("Ideal Gray-X ceiling **dropped** (Zach steer). Arms: **projector** vs **snap_d**.\n")
    lines.append("\n## W compile\n")
    Wmeta = results.get("W", {})
    lines.append(
        f"- F={Wmeta.get('fidelity', '?')} leak={Wmeta.get('leak', '?')} "
        f"nf={Wmeta.get('nf')} pairs={Wmeta.get('n_pairs')}\n"
    )
    lines.append("\n## Results table (val Med / B90)\n\n")
    lines.append("| arm | phase | n9 | n12 | n15 | b_Med | b_B90 |\n")
    lines.append("|-----|-------|----|-----|-----|-------|-------|\n")
    for arm in ("projector", "snap_d"):
        for phase in ("ramp", "trained"):
            key = f"{arm}_{phase}"
            block = results.get(key)
            if not block:
                continue
            lad = block.get("ladder", {})
            fits = block.get("fits", {})

            def cell(n):
                m = lad.get(n) or lad.get(str(n)) or {}
                if "median_inv_p" not in m:
                    return "—"
                return f"{m['median_inv_p']:.3g}/{m['B90_inv_p']:.3g}"

            bM = fits.get("Med", {}).get("b", float("nan"))
            bB = fits.get("B90", {}).get("b", float("nan"))
            lines.append(
                f"| {arm} | {phase} | {cell(9)} | {cell(12)} | {cell(15)} | {bM:.3f} | {bB:.3f} |\n"
            )
    lines.append("\n## Winner\n\n")
    lines.append(results.get("verdict", "—") + "\n")
    lines.append("\n## Notes\n\n")
    lines.append("- Protocol: L=4; Phase A Δ ramp at n9; Phase B train at n9; eval n9/12/15.\n")
    lines.append("- Uniform guessing: b=2.0. OCT20 ideal-X reference (dropped arm): b≈1.25, n12 Med≈13.\n")
    lines.append("- No GNN.\n")
    (out_dir / "RESULTS.md").write_text("".join(lines))
    (out_dir / "results.json").write_text(json.dumps(_jsonable(results), indent=2) + "\n")
    log(f"wrote {out_dir / 'RESULTS.md'} and results.json")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default=str(BAKE_RUNS))
    ap.add_argument("--L", type=int, default=4)
    ap.add_argument("--nf", type=int, default=16)
    ap.add_argument("--n_pairs", type=int, default=3)
    ap.add_argument("--compile_steps", type=int, default=2500)
    ap.add_argument("--train_steps_proj", type=int, default=40)
    ap.add_argument("--train_steps_snap", type=int, default=80)
    ap.add_argument("--max_train", type=int, default=48)
    ap.add_argument("--max_val9", type=int, default=24)
    ap.add_argument("--max_val12", type=int, default=16)
    ap.add_argument("--max_val15", type=int, default=8)
    ap.add_argument("--skip_compile", action="store_true")
    ap.add_argument("--skip_snap", action="store_true")
    ap.add_argument("--skip_proj", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", type=str, default="cpu")
    args = ap.parse_args(argv)

    # leave cores for evolve_seq
    torch.set_num_threads(max(1, int(os.environ.get("OMP_NUM_THREADS", "2"))))

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    L = args.L
    nf = args.nf
    t00 = time.time()
    results = {
        "args": vars(args),
        "protocol": "Opus probe 1 two-arm (projector vs snap_d); Gray-X dropped",
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    max_counts = {9: args.max_val9, 12: args.max_val12, 15: args.max_val15}

    # ---- W compile ----
    W_path = out_dir / "W.pt"
    if args.skip_compile and W_path.exists():
        W = CompiledW.load(W_path, device=device)
        log(f"loaded W from {W_path}: F={W.fidelity:.6f} leak={W.leak:.3e}")
    else:
        log(f"compiling W nf={nf} pairs={args.n_pairs} steps={args.compile_steps}")
        W = compile_flat_prep(
            nf=nf,
            n_pairs=args.n_pairs,
            steps=args.compile_steps,
            seed=args.seed,
            device=device,
            log_fn=lambda *a: print(time.strftime("%H:%M:%S"), "[W]", *a, flush=True),
        )
        W.save(W_path)
        log(f"saved {W_path}")
    results["W"] = W.to_dict()
    (out_dir / "W_meta.json").write_text(json.dumps(W.to_dict(), indent=2) + "\n")

    if W.fidelity < 0.99 or W.leak > 5e-3:
        log(f"WARNING: W below target (F>=0.999, leak<1e-3); continuing with F={W.fidelity} leak={W.leak}")

    # =====================================================================
    # ARM 1: projector
    # =====================================================================
    if not args.skip_proj:
        log("=== ARM projector: Phase A Δ scan ===")
        scan = scan_delta_projector(
            W, L=L, n=9, max_count=min(16, args.max_val9), nf=nf, device=device,
            log_fn=lambda *a: print(time.strftime("%H:%M:%S"), "[scan-proj]", *a, flush=True),
        )
        best_D = float(scan["best"]["Delta"])
        log(f"best ramp Δ={best_D}")
        g, b = linear_ramp(L, best_D)
        gamma_r = torch.as_tensor(g)
        beta_r = torch.as_tensor(b)
        lad_r = eval_ladder(
            "projector/ramp",
            lambda n, max_count, **kw: eval_projector(
                n, W, gamma_r, beta_r, L=L, split="val", max_count=max_count, nf=nf, device=device
            ),
            max_counts=max_counts,
        )
        results["projector_ramp"] = {
            "Delta": best_D,
            "gamma": g.tolist(),
            "beta": b.tolist(),
            "scan": {"rows": [{k: v for k, v in r.items() if k != "psi"} for r in scan["rows"]],
                     "best_Delta": best_D},
            "ladder": lad_r,
            "fits": fit_from_ladder(lad_r),
        }
        write_report(results, out_dir)

        log("=== ARM projector: Phase B L-BFGS ===")
        gamma_t, beta_t, info = train_projector_lbfgs(
            W, L=L, n_train=9, max_train=args.max_train, max_val=args.max_val9,
            steps=args.train_steps_proj, Delta0=best_D, seed=args.seed, nf=nf, device=device,
            log_fn=lambda *a: print(time.strftime("%H:%M:%S"), "[proj]", *a, flush=True),
        )
        torch.save({"gamma": gamma_t, "beta": beta_t, "info": {k: v for k, v in info.items() if k not in ("gamma", "beta", "val")}},
                   out_dir / "projector_params.pt")
        lad_t = eval_ladder(
            "projector/trained",
            lambda n, max_count, **kw: eval_projector(
                n, W, gamma_t, beta_t, L=L, split="val", max_count=max_count, nf=nf, device=device
            ),
            max_counts=max_counts,
        )
        results["projector_trained"] = {
            "gamma": gamma_t.tolist(),
            "beta": beta_t.tolist(),
            "train_info": {
                "best_step": info.get("step"),
                "best_S_n9": info.get("loss"),
                "train_loss": info.get("train_loss"),
                "val_n9": info.get("val"),
            },
            "ladder": lad_t,
            "fits": fit_from_ladder(lad_t),
        }
        write_report(results, out_dir)

    # =====================================================================
    # ARM 2: snap_d (seq_best motif COST→SNAP→D, L=4 for fair compare)
    # =====================================================================
    if not args.skip_snap:
        motif = Motif(name="cost_snap_d_L4", prep=("D",), layer=("COST", "SNAP", "D"), L=L)
        motif.validate()
        log(f"=== ARM snap_d: motif {motif.name} Phase A Δ scan ===")
        scan_s = scan_delta_snap_d(
            motif, n=9, max_count=min(16, args.max_val9), nf=nf, device=device, seed=args.seed,
            log_fn=lambda *a: print(time.strftime("%H:%M:%S"), "[scan-snap]", *a, flush=True),
        )
        best_Ds = float(scan_s["best"]["Delta"])
        params_r = scan_s["best"]["params"]
        lad_sr = eval_ladder(
            "snap_d/ramp",
            lambda n, max_count, **kw: eval_snap_d(
                n, motif, params_r, split="val", max_count=max_count, nf=nf, device=device
            ),
            max_counts=max_counts,
        )
        results["snap_d_ramp"] = {
            "Delta": best_Ds,
            "motif": motif.to_dict(),
            "ladder": lad_sr,
            "fits": fit_from_ladder(lad_sr),
        }
        write_report(results, out_dir)

        log("=== ARM snap_d: Phase B Adam→LBFGS ===")
        params_t, info_s = train_snap_d_lbfgs(
            motif, n_train=9, max_train=args.max_train, max_val=args.max_val9,
            steps=args.train_steps_snap, Delta0=best_Ds, seed=args.seed, nf=nf, device=device,
            log_fn=lambda *a: print(time.strftime("%H:%M:%S"), "[snap_d]", *a, flush=True),
        )
        torch.save({"params": params_t, "motif": motif.to_dict()}, out_dir / "snap_d_params.pt")
        lad_st = eval_ladder(
            "snap_d/trained",
            lambda n, max_count, **kw: eval_snap_d(
                n, motif, params_t, split="val", max_count=max_count, nf=nf, device=device
            ),
            max_counts=max_counts,
        )
        results["snap_d_trained"] = {
            "motif": motif.to_dict(),
            "train_info": {
                "best_step": info_s.get("step"),
                "best_S_n9": info_s.get("score"),
                "val_n9": info_s.get("val"),
            },
            "ladder": lad_st,
            "fits": fit_from_ladder(lad_st),
        }
        write_report(results, out_dir)

    # ---- verdict ----
    def b_of(key):
        block = results.get(key) or {}
        return (block.get("fits") or {}).get("Med", {}).get("b", float("inf"))

    candidates = []
    for key in ("projector_trained", "snap_d_trained", "projector_ramp", "snap_d_ramp"):
        if key in results:
            fits = results[key]["fits"]
            lad = results[key]["ladder"]
            m12 = (lad.get(12) or lad.get("12") or {}).get("median_inv_p", float("inf"))
            candidates.append((key, fits["Med"]["b"], fits["B90"]["b"], m12))
    candidates.sort(key=lambda t: (t[1], t[3]))  # lower b_Med, then n12 Med
    if candidates:
        win = candidates[0]
        verdict = (
            f"Winner: **{win[0]}** with b_Med={win[1]:.3f}, b_B90={win[2]:.3f}, n12 Med={win[3]:.3g}. "
            f"Uniform b=2.0; want b≪2 for native projector to be interesting. "
        )
        if win[0].startswith("projector") and win[1] < 1.7:
            verdict += "Projector keeps b well below 2 → recommend probe 2 (train n9+n12) next."
        elif win[0].startswith("projector"):
            verdict += "Projector wins on this pack but b still high → try probe 5 (seq vocab with W atoms) or probe 2."
        else:
            verdict += "SNAP+D still ahead or projector weak → probe 5 seq vocab / richer mixer; do not revive GNN."
    else:
        verdict = "No completed arms."
    results["verdict"] = verdict
    results["wall_s"] = time.time() - t00
    results["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    write_report(results, out_dir)
    log(verdict)
    log(f"DONE in {results['wall_s']:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
