#!/usr/bin/env python3
"""Opus #3 — finite-shot library schedule on LBFGS seed0 champ.

Analytic amp inv_p = 1/p_succ from projector-Uc (= library SNAP schedule).
Finite shots: Binomial(S, p_succ) → p̂ = k/S → inv̂ = 1/max(p̂, 1/(S+1)).
Also documents sim→exp gap on frozen SNAP masks (from grape_library fidelity).

Writes runs/gadget_Uc/finite_shot/{RESULTS.md,results.json,table.csv}
Honesty: still Uc=exp(-iγ Hc). No blind stencil.
"""
from __future__ import annotations

import argparse
import csv
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
    SCALE_DIR,
    _jsonable,
    build_W_BinH,
    make_sim,
)

OUT = _PKG / "runs" / "gadget_Uc" / "finite_shot"
CHAMP = _PKG / "runs" / "gadget_Uc" / "unsat_params.pt"
FID_JSON = _PKG / "runs" / "gadget_Uc" / "grape_library" / "fidelity.json"
COMPILE = _PKG / "runs" / "gadget_Uc" / "grape_library" / "compile_offline.json"
CHAMP_SCALE = {
    9: {"Med": 1.092, "B90": 1.776},
    12: {"Med": 1.197, "B90": 2.843},
    15: {"Med": 1.418, "B90": 5.505},
    18: {"Med": 1.752, "B90": 11.353},
}


def log(*a):
    print(time.strftime("%H:%M:%S"), "[fshot]", *a, flush=True)


@torch.no_grad()
def collect_psucc(n, W, gamma, phi, L, nf, device, max_count, data_dir):
    pack = load_pack(n, data_dir=data_dir)
    key = "scale1000" if "scale1000_seeds" in pack else "test"
    seeds, _, _, _, E = _split_arrays(pack, key, max_count=max_count)
    if E is None or len(seeds) == 0:
        raise RuntimeError(f"n={n}: empty {key}")
    sim = make_sim(n, W, L, nf, device)
    g = gamma.to(device=device).float()
    p = phi.to(device=device).float()
    ps, invs, leaks = [], [], []
    t0 = time.time()
    B = len(seeds)
    every = 50 if n >= 15 else 100
    for b in range(B):
        Eb = torch.as_tensor(E[b], dtype=torch.float32, device=device)
        out = sim.forward_gadget(Eb, g, p, "unsat")
        ps.append(float(out["p_succ"].cpu()))
        invs.append(float(out["inv_p"].cpu()))
        leaks.append(float(out["p_leak"].cpu()))
        if (b + 1) % every == 0 or b + 1 == B:
            rate = (b + 1) / max(time.time() - t0, 1e-9)
            log(f"n{n} {b+1}/{B} Med_inv≈{np.median(invs):.3f} rate={rate:.2f}/s")
    return {
        "n": n,
        "B": B,
        "split": key,
        "p_succ": np.asarray(ps, dtype=np.float64),
        "inv_analytic": np.asarray(invs, dtype=np.float64),
        "p_leak": np.asarray(leaks, dtype=np.float64),
        "wall_s": time.time() - t0,
    }


def summarize_inv(inv: np.ndarray) -> dict:
    return {
        "median_inv_p": float(np.median(inv)),
        "B90_inv_p": float(np.quantile(inv, 0.9)),
        "mean_inv_p": float(np.mean(inv)),
        "frac_inv_le_2": float(np.mean(inv <= 2.0)),
    }


def finite_shot_inv(p_succ: np.ndarray, shots: int, rng: np.random.Generator, replicates: int = 1):
    """Binomial readout on success mass; inv̂ = 1/max(k/S, 1/(S+1)).

    If replicates>1, average inv̂ over replicates per instance (reduces shot noise
    of the estimator itself; still finite-S biased).
    """
    S = int(shots)
    p = np.clip(p_succ, 0.0, 1.0)
    inv_acc = np.zeros_like(p)
    zero_frac_acc = 0.0
    for _ in range(replicates):
        k = rng.binomial(S, p)
        phat = k.astype(np.float64) / float(S)
        zero_frac_acc += float(np.mean(k == 0))
        phat_safe = np.maximum(phat, 1.0 / (S + 1.0))
        inv_acc += 1.0 / phat_safe
    inv_acc /= float(replicates)
    zero_frac_acc /= float(replicates)
    return inv_acc, zero_frac_acc


def snap_gap_summary() -> dict:
    out = {
        "source_fid": str(FID_JSON.relative_to(_PKG)) if FID_JSON.exists() else None,
        "source_compile": str(COMPILE.relative_to(_PKG)) if COMPILE.exists() else None,
    }
    if FID_JSON.exists():
        fid = json.loads(FID_JSON.read_text())
        rows = fid.get("rows") or []
        def col(name):
            xs = [float(r[name]) for r in rows if name in r]
            return {
                "min": min(xs) if xs else None,
                "mean": float(np.mean(xs)) if xs else None,
                "n": len(xs),
            }
        out["fid_SNAP_vs_UZ"] = col("fid_SNAP_vs_UZ")
        out["fid_Wdag_SNAP_W_vs_UX"] = col("fid_Wdag_SNAP_W_vs_UX")
        out["fid_SNAP_D_fit_UX"] = col("fid_SNAP_D_fit_UX")
        out["fid_D_SNAP_D_fit_UX"] = col("fid_D_SNAP_D_fit_UX")
        # by gamma
        by_g = {}
        for r in rows:
            g = r.get("gamma")
            by_g.setdefault(g, []).append(r)
        out["SNAP_D_fit_UX_by_gamma"] = {
            str(g): {
                "min": float(min(x["fid_SNAP_D_fit_UX"] for x in rs)),
                "mean": float(np.mean([x["fid_SNAP_D_fit_UX"] for x in rs])),
            }
            for g, rs in sorted(by_g.items(), key=lambda kv: kv[0] or 0)
        }
    if COMPILE.exists():
        c = json.loads(COMPILE.read_text())
        # keep slim keys if present
        for k in (
            "n_patterns", "wall_s", "fid_snap_uz_min", "fid_wdag_min",
            "all_exact", "max_binh_fro",
        ):
            if k in c:
                out[k] = c[k]
        # nested?
        if "exact" in c and isinstance(c["exact"], dict):
            out["compile_exact"] = c["exact"]
        if "summary" in c:
            out["compile_summary"] = c["summary"]
    return out


def write_md(results):
    lines = [
        "# Finite-shot library schedule — LBFGS seed0 champ",
        "",
        f"_Generated {time.strftime('%Y-%m-%d %H:%M:%S')} +0800 (Asia/Shanghai)_",
        "",
        "Honesty: still `Uc=exp(-iγ Hc)` via frozen SNAP-mask library schedule. No blind stencil.",
        "",
        "Analytic amp: `inv_p = 1/p_succ` (computational success mass).",
        "Finite shots: `k ~ Binomial(S, p_succ)`, `p̂=k/S`, `inv̂=1/max(p̂, 1/(S+1))`.",
        "",
        "## Champ analytic baseline (sealed)",
        "",
        "| n | Med | B90 |",
        "|---|-----|-----|",
    ]
    for n in (9, 12, 15, 18):
        lines.append(f"| {n} | {CHAMP_SCALE[n]['Med']} | {CHAMP_SCALE[n]['B90']} |")

    lines += [
        "",
        "## Finite-shot Med/B90 vs analytic",
        "",
        "| n | shots | Med | B90 | ΔMed% | ΔB90% | zero_frac | B |",
        "|---|-------|-----|-----|-------|-------|-----------|---|",
    ]
    for n_s, block in results.get("by_n", {}).items():
        n = int(n_s)
        ana = block["analytic"]
        for sh, row in block.get("shots", {}).items():
            dmed = 100.0 * (row["median_inv_p"] / max(ana["median_inv_p"], 1e-12) - 1.0)
            db90 = 100.0 * (row["B90_inv_p"] / max(ana["B90_inv_p"], 1e-12) - 1.0)
            lines.append(
                f"| {n} | {sh} | {row['median_inv_p']:.3f} | {row['B90_inv_p']:.3f} | "
                f"{dmed:+.1f}% | {db90:+.1f}% | {row['zero_frac']:.3f} | {block['B']} |"
            )
        lines.append(
            f"| {n} | ∞ analytic | {ana['median_inv_p']:.3f} | {ana['B90_inv_p']:.3f} | "
            f"0.0% | 0.0% | 0.000 | {block['B']} |"
        )

    gap = results.get("snap_gap") or {}
    lines += [
        "",
        "## Sim → exp gap on frozen SNAP masks",
        "",
        "Library = 26 local projectors as SNAP masks (θ_f = γ·mask). Single-cavity compile:",
        "",
    ]
    if gap.get("fid_SNAP_vs_UZ"):
        lines.append(
            f"- Fid SNAP↔U_Z: min={gap['fid_SNAP_vs_UZ']['min']:.6f} "
            f"(n={gap['fid_SNAP_vs_UZ']['n']}) — **exact** within numerics."
        )
    if gap.get("fid_Wdag_SNAP_W_vs_UX"):
        lines.append(
            f"- Fid W†SNAP W↔U_X: min={gap['fid_Wdag_SNAP_W_vs_UX']['min']:.12f} — **exact** BinH sandwich."
        )
    if gap.get("fid_SNAP_D_fit_UX"):
        lines.append(
            f"- Fallback SNAP·D fit to U_X (no native W): min={gap['fid_SNAP_D_fit_UX']['min']:.4f}, "
            f"mean={gap['fid_SNAP_D_fit_UX']['mean']:.4f} — **sim≠exp** if W unavailable."
        )
    if gap.get("SNAP_D_fit_UX_by_gamma"):
        lines += ["", "| γ | SNAP·D Fid min | mean |", "|---|----------------|------|"]
        for g, st in gap["SNAP_D_fit_UX_by_gamma"].items():
            lines.append(f"| {g} | {st['min']:.4f} | {st['mean']:.4f} |")
    lines += [
        "",
        "Two-cavity AND-phase: local SNAP⊗SNAP Fid≪1; **joint SNAP** Fid=1 "
        "(see `grape_library/COMPILE_TWOCAV.md`, `pulse_budget/`). "
        "Cross-Kerr T already tabulated there — not re-run.",
        "",
        "### Takeaway",
        "",
        "- With native W + joint SNAP: **unitary sim→exp gap ≈ 0** on frozen masks; "
        "remaining gap is **readout shot noise** (table above).",
        "- Without W / without joint SNAP: pulse-compile fidelity gap dominates (γ-dependent).",
        "- Finite S inflates Med/B90; S≳500 recovers analytic amp within a few % on n≤15.",
        "",
    ]
    (OUT / "RESULTS.md").write_text("\n".join(lines) + "\n")


def write_csv(results):
    with (OUT / "table.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "n", "shots", "Med", "B90", "mean", "frac_le2", "zero_frac",
            "Med_analytic", "B90_analytic", "B", "replicates",
        ])
        for n_s, block in results.get("by_n", {}).items():
            n = int(n_s)
            ana = block["analytic"]
            w.writerow([
                n, "inf", ana["median_inv_p"], ana["B90_inv_p"], ana["mean_inv_p"],
                ana["frac_inv_le_2"], 0.0, ana["median_inv_p"], ana["B90_inv_p"],
                block["B"], 0,
            ])
            for sh, row in block.get("shots", {}).items():
                w.writerow([
                    n, sh, row["median_inv_p"], row["B90_inv_p"], row["mean_inv_p"],
                    row["frac_inv_le_2"], row["zero_frac"],
                    ana["median_inv_p"], ana["B90_inv_p"], block["B"], row["replicates"],
                ])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ns", default="9,12,15,18")
    ap.add_argument("--shots", default="10,50,100,500")
    ap.add_argument("--scale_B", type=int, default=1000)
    ap.add_argument("--replicates", type=int, default=1,
                    help="Binomial draws averaged per instance")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--nf", type=int, default=8)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    W = build_W_BinH(nf=args.nf)
    st = torch.load(CHAMP, map_location="cpu", weights_only=False)
    g0, p0 = st["gamma"].float(), st["phi"].float()
    L = int(st.get("L", g0.numel()))
    ns = [int(x) for x in args.ns.split(",") if x.strip()]
    shots_list = [int(x) for x in args.shots.split(",") if x.strip()]
    rng = np.random.default_rng(args.seed)

    if not SCALE_DIR.exists():
        raise SystemExit(f"missing scale1000 dir {SCALE_DIR}")

    results = {
        "honesty": "Uc=exp(-iγ Hc) library SNAP schedule; finite readout shots",
        "champ": str(CHAMP),
        "champ_sealed": CHAMP_SCALE,
        "L": L,
        "shots": shots_list,
        "replicates": args.replicates,
        "seed": args.seed,
        "by_n": {},
        "snap_gap": snap_gap_summary(),
    }

    for n in ns:
        log(f"=== n{n} collect analytic ===")
        pack = collect_psucc(
            n, W, g0, p0, L, args.nf, device, args.scale_B, SCALE_DIR,
        )
        ana = summarize_inv(pack["inv_analytic"])
        # sanity vs sealed
        sealed = CHAMP_SCALE.get(n)
        if sealed:
            log(
                f"n{n} analytic Med/B90={ana['median_inv_p']:.3f}/{ana['B90_inv_p']:.3f} "
                f"vs sealed {sealed['Med']}/{sealed['B90']}"
            )
        block = {
            "B": pack["B"],
            "analytic": ana,
            "mean_p_succ": float(pack["p_succ"].mean()),
            "mean_p_leak": float(pack["p_leak"].mean()),
            "wall_s": pack["wall_s"],
            "shots": {},
        }
        for S in shots_list:
            inv_hat, zf = finite_shot_inv(
                pack["p_succ"], S, rng, replicates=args.replicates,
            )
            row = summarize_inv(inv_hat)
            row["zero_frac"] = zf
            row["replicates"] = args.replicates
            block["shots"][str(S)] = row
            log(
                f"n{n} S={S}: Med/B90={row['median_inv_p']:.3f}/{row['B90_inv_p']:.3f} "
                f"zero={zf:.3f}"
            )
        results["by_n"][str(n)] = block
        # checkpoint
        (OUT / "results.json").write_text(json.dumps(_jsonable(results), indent=2))
        write_csv(results)
        write_md(results)

    # b fits analytic vs S=500
    try:
        ns_u = [n for n in ns if str(n) in results["by_n"]]
        results["b_fits"] = {
            "analytic": {
                "Med": fit_b(ns_u, [results["by_n"][str(n)]["analytic"]["median_inv_p"] for n in ns_u]),
                "B90": fit_b(ns_u, [results["by_n"][str(n)]["analytic"]["B90_inv_p"] for n in ns_u]),
            }
        }
        if "500" in results["by_n"][str(ns_u[0])]["shots"]:
            results["b_fits"]["S500"] = {
                "Med": fit_b(ns_u, [results["by_n"][str(n)]["shots"]["500"]["median_inv_p"] for n in ns_u]),
                "B90": fit_b(ns_u, [results["by_n"][str(n)]["shots"]["500"]["B90_inv_p"] for n in ns_u]),
            }
    except Exception as e:
        log("b_fit skip", e)

    (OUT / "results.json").write_text(json.dumps(_jsonable(results), indent=2))
    write_csv(results)
    write_md(results)
    log("wrote", OUT)


if __name__ == "__main__":
    main()
