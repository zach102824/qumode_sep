#!/usr/bin/env python3
"""SNAP-easiness restriction ladder (Nov1st-final motif locked).

Most-restricted → freer SNAP heads; train max_train=200 on n6+9; scale1000
vs sealed. Warm-start from best shell polish when the head is a projection.

Stop: touch Nov/runs/STOP_SNAP_EASY
Does NOT touch angle_explore polish PIDs.
"""
from __future__ import annotations

import argparse
import csv
import json
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
from Nov.scripts.angle_explore import (  # noqa: E402
    beats_sealed,
    promote_kill,
    scale_record,
    train_one,
)
from Nov.scripts.nov1st_final_train import SEALED_REF, smoke_overlap  # noqa: E402
from Nov.scripts.snap_heads import (  # noqa: E402
    SNAP_EASY_MODES,
    describe_easy_heads,
    fit_shell_to_mode,
    pad_snap,
    project_shared,
    snap_dim,
    snap_tag,
)

OUT = _NOV / "runs" / "snap_easy"
STOP = _NOV / "runs" / "STOP_SNAP_EASY"
SHELL_WARM = (
    _NOV
    / "runs"
    / "angle_explore"
    / "polish"
    / "shell"
    / "params_polish_shell_pL_adam600_lr5e3_lbfgs_L32_seed22.pt"
)

# Ladder: most restricted first. (mode, shared, few_M)
LADDER = [
    ("const", False, 3),
    ("const", True, 3),
    ("kerr", False, 3),
    ("kerr", True, 3),
    ("gauss", False, 3),
    ("gauss", True, 3),
    ("poly2_n", False, 3),
    ("poly2_n", True, 3),
    ("few_m", False, 2),
    ("few_m", False, 3),
    ("few_m", True, 3),
    ("gauss_plus1", False, 3),
    ("gauss_plus1", True, 3),
    ("poly3_pc", False, 3),
    ("poly3_n", False, 3),
    ("shell", False, 3),  # reference
]


def log(*a):
    print(time.strftime("%H:%M:%S"), "[snap-easy]", *a, flush=True)


def stopped() -> bool:
    return STOP.exists()


def load_shell_warm(L: int):
    if not SHELL_WARM.exists():
        log("no shell warm at", SHELL_WARM)
        return None, None
    st = torch.load(SHELL_WARM, map_location="cpu", weights_only=False)
    gamma = st["gamma"].float()
    phi = st.get("phi")
    if phi is None:
        phi = st["snap"].float()
    else:
        phi = phi.float()
    # pad/truncate gamma to L
    if gamma.numel() >= L:
        gamma_L = gamma[:L].clone()
        phi_L = phi[:L].clone()
    else:
        gamma_L = torch.zeros(L)
        gamma_L[: gamma.numel()] = gamma
        gamma_L[gamma.numel() :] = gamma[-1]
        phi_L = torch.zeros(L, 4)
        phi_L[: phi.shape[0]] = phi
        phi_L[phi.shape[0] :] = phi[-1]
    return gamma_L, phi_L


def warm_snap_for(mode: str, shared: bool, L: int, few_M: int, phi_L):
    if phi_L is None:
        return None
    try:
        fitted = fit_shell_to_mode(phi_L, mode, few_M=few_M)
    except Exception as e:
        log(f"fit_shell_to_mode({mode}) fail: {e}")
        return None
    if shared:
        return project_shared(fitted)
    return pad_snap(fitted, L, mode, shared=False, few_M=few_M)


def append_lb(path: Path, row: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists()
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()), extrasaction="ignore")
        if write_header:
            w.writeheader()
        w.writerow(row)


def write_results(path: Path, rows: list[dict], notes: str = ""):
    lines = [
        "# SNAP easiness — RESULTS",
        "",
        f"_Updated {time.strftime('%Y-%m-%d %H:%M Asia/Shanghai')}_",
        "",
        "Motif locked: Nov1st-final. Train n∈{6,9} max_train=200. Eval scale1000 B=1000.",
        "",
        f"Sealed: n9 {SEALED_REF[9]['Med']}/{SEALED_REF[9]['B90']}, "
        f"n12 {SEALED_REF[12]['Med']}/{SEALED_REF[12]['B90']}, "
        f"n15 {SEALED_REF[15]['Med']}/{SEALED_REF[15]['B90']}, "
        f"n18 {SEALED_REF[18]['Med']}/{SEALED_REF[18]['B90']}.",
        "",
        notes,
        "",
        "## Ladder (params/layer ↑)",
        "",
        "| tag | L | params | n9 Med/B90 | ×Med/×B90 | n12 | n15 | n18 | verdict |",
        "|-----|---|--------|------------|-----------|-----|-----|-----|---------|",
    ]
    for r in rows:
        def cell(n):
            m = r.get(f"n{n}_Med")
            b = r.get(f"n{n}_B90")
            xm = r.get(f"n{n}_Med_x")
            xb = r.get(f"n{n}_B90_x")
            if m is None:
                return "—"
            return f"{m:.3f}/{b:.3f} ({xm:.3f}×/{xb:.3f}×)"

        lines.append(
            f"| `{r.get('tag','')}` | {r.get('L')} | {r.get('params')} | "
            f"{cell(9)} | "
            f"{r.get('n9_Med_x','—')}/{r.get('n9_B90_x','—')} | "
            f"{cell(12)} | {cell(15)} | {cell(18)} | {r.get('verdict','')} |"
        )
    # pareto-ish
    survivors = [
        r for r in rows
        if r.get("n9_Med_x") is not None and r["n9_Med_x"] <= 1.10
    ]
    survivors.sort(key=lambda r: (r.get("params", 99), r.get("n9_Med_x", 99)))
    lines += ["", "## Most-restricted survivors (n9 Med ≤ 1.10× sealed)", ""]
    if not survivors:
        lines.append("_None yet._")
    else:
        for r in survivors:
            lines.append(
                f"- **{r['tag']}** params={r['params']} "
                f"n9 {r['n9_Med']:.3f}/{r['n9_B90']:.3f} "
                f"({r['n9_Med_x']:.3f}×/{r['n9_B90_x']:.3f}×)"
            )
    beat = any(r.get("beat_sealed") for r in rows)
    lines += [
        "",
        f"**BEAT_SEALED (restricted SNAP dual-beat all n):** {'YES' if beat else 'NO'}",
        "",
        "Stop file: `Nov/runs/STOP_SNAP_EASY`.",
    ]
    path.write_text("\n".join(lines) + "\n")


def run_ladder(W, nf, device, *, L_list, seeds, steps, max_train, max_val, lr, scale_B,
               out_dir: Path, do_scale: bool, modes_filter=None):
    out_dir.mkdir(parents=True, exist_ok=True)
    lb = out_dir / "leaderboard.csv"
    results_rows = []
    gamma_warm = {}
    phi_warm = {}
    for L in L_list:
        g, p = load_shell_warm(L)
        gamma_warm[L], phi_warm[L] = g, p

    for mode, shared, few_M in LADDER:
        if modes_filter and mode not in modes_filter:
            continue
        for L in L_list:
            if stopped():
                log("STOP"); return results_rows
            tag_base = snap_tag(mode, shared, few_M=few_M)
            params = snap_dim(mode, few_M=few_M)
            snap0 = warm_snap_for(mode, shared, L, few_M, phi_warm.get(L))
            gamma0 = gamma_warm.get(L)
            best_for_arm = None
            for seed in seeds:
                if stopped():
                    log("STOP"); return results_rows
                arm = f"easy_{tag_base}_L{L}_s{seed}"
                log(f"=== train {arm} params={params} warm={snap0 is not None} ===")
                best = train_one(
                    W, L, nf, device,
                    train_ns=(6, 9), max_train=max_train, max_val=max_val,
                    seed=seed, lr=lr, steps=steps,
                    gamma0=gamma0, snap0=snap0,
                    snap_mode=mode, snap_shared=shared, few_M=few_M,
                    loss_kind="log_inv", leakpen=0.1,
                    lbfgs_mid=True, lbfgs_final=True, lbfgs_rounds=2, lbfgs_iter=6,
                    arm_tag=arm, out_dir=out_dir / tag_base,
                )
                n9 = (best.get("val") or {}).get(9) or {}
                n9med = float(n9.get("median_inv_p", 1e9))
                verdict = promote_kill(n9med)
                log(f"  → n9 Med={n9med:.3f} {verdict}")
                append_lb(lb, {
                    "tag": arm, "mode": mode, "shared": int(shared), "few_M": few_M,
                    "L": L, "seed": seed, "params": params, "n9_Med": n9med,
                    "n9_B90": n9.get("B90_inv_p"), "n6_Med": (best.get("val") or {}).get(6, {}).get("median_inv_p"),
                    "verdict": verdict, "score": best.get("score"), "path": best.get("path", ""),
                })
                if verdict == "KILL":
                    log(f"  KILL seed {seed}; skip remaining seeds for this arm if all bad")
                    # still try other seeds once; if first is kill and Med>>10, skip rest
                    if n9med > 20:
                        break
                    continue
                if best_for_arm is None or best["score"] < best_for_arm["score"]:
                    best_for_arm = best
                    best_for_arm["arm_tag"] = arm
                    best_for_arm["params"] = params
                    best_for_arm["mode"] = mode
                    best_for_arm["shared"] = shared
                    best_for_arm["few_M"] = few_M
                    best_for_arm["verdict"] = verdict

            if best_for_arm is None:
                results_rows.append({
                    "tag": f"{tag_base}_L{L}", "L": L, "params": params,
                    "verdict": "KILL", "beat_sealed": False,
                })
                write_results(out_dir / "RESULTS.md", results_rows)
                continue

            row = {
                "tag": best_for_arm["arm_tag"],
                "L": L,
                "params": params,
                "mode": mode,
                "shared": shared,
                "verdict": best_for_arm["verdict"],
                "beat_sealed": False,
                "path": best_for_arm.get("path"),
            }
            # train-val n9 for quick ×
            v9 = (best_for_arm.get("val") or {}).get(9) or {}
            if v9:
                row["n9_Med"] = v9["median_inv_p"]
                row["n9_B90"] = v9["B90_inv_p"]
                row["n9_Med_x"] = v9["median_inv_p"] / SEALED_REF[9]["Med"]
                row["n9_B90_x"] = v9["B90_inv_p"] / SEALED_REF[9]["B90"]

            # Fast screen: n9+n12 at min(B,200). Full n15/18 deferred to finalists.
            do_this_scale = do_scale and best_for_arm["verdict"] in ("PROMOTE_CAND", "KEEP")
            if do_this_scale and row.get("n9_Med_x", 99) <= 2.5:
                screen_B = min(int(scale_B), 200)
                log(f"=== scale screen {row['tag']} n9/12 B={screen_B} ===")
                sc, _ = scale_record(
                    W, best_for_arm["state"], nf, device,
                    [9, 12], screen_B, out_dir, row["tag"] + f"_screenB{screen_B}",
                )
                row["beat_sealed"] = False
                for n in (9, 12):
                    m = sc.get(n)
                    if not m:
                        continue
                    ref = SEALED_REF[n]
                    row[f"n{n}_Med"] = m["median_inv_p"]
                    row[f"n{n}_B90"] = m["B90_inv_p"]
                    row[f"n{n}_Med_x"] = m["median_inv_p"] / ref["Med"]
                    row[f"n{n}_B90_x"] = m["B90_inv_p"] / ref["B90"]
                if row.get("n9_Med_x", 99) <= 1.15 and row.get("n12_Med_x", 99) <= 1.25:
                    row["finalist"] = True
                    log(f"  FINALIST {row['tag']} n9×={row.get('n9_Med_x'):.3f} n12×={row.get('n12_Med_x'):.3f}")

            results_rows.append(row)
            (out_dir / "summary.json").write_text(json.dumps(_jsonable({
                "rows": results_rows, "heads": describe_easy_heads(),
            }), indent=2))
            write_results(out_dir / "RESULTS.md", results_rows)

    # Full scale1000 on finalists / top-3 by n9 Med×
    if do_scale and not stopped():
        finals = [r for r in results_rows if r.get("finalist") and r.get("path")]
        ranked = sorted(
            [r for r in results_rows if r.get("n9_Med_x") is not None and r.get("path")],
            key=lambda r: r["n9_Med_x"],
        )[:3]
        seen, queue = set(), []
        for r in finals + ranked:
            if r["tag"] in seen:
                continue
            seen.add(r["tag"])
            queue.append(r)
        for r in queue:
            if stopped():
                break
            pt = Path(r["path"])
            if not pt.exists():
                continue
            st = torch.load(pt, map_location="cpu", weights_only=False)
            log(f"=== FULL scale1000 {r['tag']} B={scale_B} ===")
            sc, _ = scale_record(
                W, st, nf, device, [9, 12, 15, 18], scale_B, out_dir, r["tag"] + "_full",
            )
            beat, _ = beats_sealed(sc)
            r["beat_sealed"] = bool(beat)
            for n in (9, 12, 15, 18):
                m = sc.get(n)
                if not m:
                    continue
                ref = SEALED_REF[n]
                r[f"n{n}_Med"] = m["median_inv_p"]
                r[f"n{n}_B90"] = m["B90_inv_p"]
                r[f"n{n}_Med_x"] = m["median_inv_p"] / ref["Med"]
                r[f"n{n}_B90_x"] = m["B90_inv_p"] / ref["B90"]
            if beat:
                log("*** BEAT_SEALED restricted SNAP ***")
                (out_dir / "BEAT_SEALED").write_text(r["tag"] + "\n")
            write_results(out_dir / "RESULTS.md", results_rows)
        (out_dir / "summary.json").write_text(json.dumps(_jsonable({
            "rows": results_rows, "heads": describe_easy_heads(),
        }), indent=2))
    return results_rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nf", type=int, default=8)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--max_train", type=int, default=200)
    ap.add_argument("--max_val", type=int, default=40)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--lr", type=float, default=5e-3)
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--L", default="16,32")
    ap.add_argument("--scale_B", type=int, default=1000)
    ap.add_argument("--no_scale", action="store_true")
    ap.add_argument("--modes", default="", help="comma filter; empty=full ladder")
    ap.add_argument("--smoke-only", action="store_true")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    W = build_W_BinH(nf=args.nf)
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    L_list = [int(x) for x in args.L.split(",") if x.strip()]
    modes_filter = [x for x in args.modes.split(",") if x.strip()] or None

    sm = smoke_overlap(n=6, L=2, nf=args.nf, device=args.device, seed=0)
    (OUT / "smoke.json").write_text(json.dumps(_jsonable(sm), indent=2))
    if not sm["pass"]:
        raise SystemExit("smoke FAIL")
    if args.smoke_only:
        log("smoke-only done"); return

    log("START snap_easy ladder", "L=", L_list, "seeds=", seeds, "modes=", modes_filter or "ALL")
    rows = run_ladder(
        W, args.nf, device,
        L_list=L_list, seeds=seeds, steps=args.steps,
        max_train=args.max_train, max_val=args.max_val, lr=args.lr,
        scale_B=args.scale_B, out_dir=OUT, do_scale=not args.no_scale,
        modes_filter=modes_filter,
    )
    write_results(OUT / "RESULTS.md", rows, notes="Run complete." if not stopped() else "Stopped.")
    log("DONE →", OUT)


if __name__ == "__main__":
    main()
