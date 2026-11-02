#!/usr/bin/env python3
"""Close the tiny n9/n12 B90 gap on gauss SNAP (Nov1st-final).

Warm-starts from the near-beat gauss L32 s0 champ and/or shell polish fit.
Tries CVaR / longer Adam+LBFGS / more seeds / gauss_plus1 & poly3_pc residual.

Writes BEAT_SEALED.json only on strict Med+B90 dual-beat at all sealed n.
Honors Nov/runs/STOP_SNAP_EASY. Does NOT touch angle_explore polish PIDs.
"""
from __future__ import annotations

import argparse
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
from Nov.scripts.angle_explore import beats_sealed, scale_record, train_one  # noqa: E402
from Nov.scripts.nov1st_final_train import SEALED_REF  # noqa: E402
from Nov.scripts.snap_easy_explore import (  # noqa: E402
    STOP,
    load_shell_warm,
    warm_snap_for,
)
from Nov.scripts.snap_heads import fit_shell_to_mode, pad_snap  # noqa: E402

OUT = _NOV / "runs" / "snap_easy" / "gauss_polish"
GAUSS_CHAMP = (
    _NOV
    / "runs"
    / "snap_easy"
    / "snap_gauss_perL"
    / "params_easy_snap_gauss_perL_L32_s0_L32_seed0.pt"
)


def log(*a):
    print(time.strftime("%H:%M:%S"), "[gauss-polish]", *a, flush=True)


def stopped() -> bool:
    return STOP.exists()


def load_gauss_champ():
    st = torch.load(GAUSS_CHAMP, map_location="cpu", weights_only=False)
    return st["gamma"].float(), st["snap"].float(), st


def expand_gauss_to(mode: str, snap_g: torch.Tensor) -> torch.Tensor:
    """Pad/lift gauss (L,3) into gauss_plus1 / poly3_pc (L,4) with zero residual."""
    L = snap_g.shape[0]
    if mode == "gauss":
        return snap_g.clone()
    if mode in ("gauss_plus1", "poly3_pc"):
        out = torch.zeros(L, 4, dtype=torch.float32)
        out[:, :3] = snap_g
        return out
    raise ValueError(mode)


# (tag, mode, loss, cvar_q, steps, lr, lb_rounds, lb_iter, leakpen, warm_src)
ARMS = [
    ("cvar08_from_gauss", "gauss", "cvar", 0.8, 600, 2e-3, 4, 10, 0.1, "gauss"),
    ("cvar09_from_gauss", "gauss", "cvar", 0.9, 600, 2e-3, 4, 10, 0.05, "gauss"),
    ("adam800_from_gauss", "gauss", "log_inv", 0.8, 800, 2e-3, 4, 12, 0.1, "gauss"),
    ("adam600_lr5e3_from_gauss", "gauss", "log_inv", 0.8, 600, 5e-3, 3, 8, 0.1, "gauss"),
    ("cvar08_from_shell", "gauss", "cvar", 0.8, 600, 2e-3, 4, 10, 0.1, "shell"),
    ("gplus1_cvar_from_gauss", "gauss_plus1", "cvar", 0.8, 600, 2e-3, 4, 10, 0.1, "gauss"),
    ("poly3pc_cvar_from_gauss", "poly3_pc", "cvar", 0.8, 600, 2e-3, 4, 10, 0.1, "gauss"),
    ("gplus1_adam_from_shell", "gauss_plus1", "log_inv", 0.8, 600, 5e-3, 3, 8, 0.1, "shell"),
]


def write_beat_json(tag: str, sc: dict, path: Path):
    detail = {}
    for n, ref in SEALED_REF.items():
        m = sc.get(n) or sc.get(str(n))
        if not m:
            continue
        detail[str(n)] = {
            "Med": m["median_inv_p"],
            "B90": m["B90_inv_p"],
            "Med_x": m["median_inv_p"] / ref["Med"],
            "B90_x": m["B90_inv_p"] / ref["B90"],
            "sealed_Med": ref["Med"],
            "sealed_B90": ref["B90"],
        }
    payload = {
        "tag": tag,
        "beat_sealed": True,
        "criterion": "Med AND B90 <= sealed at all n in {9,12,15,18}",
        "detail": detail,
        "written_at": time.strftime("%Y-%m-%d %H:%M:%S Asia/Shanghai"),
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")
    # also plain marker for older watchers
    path.with_suffix("").write_text(tag + "\n")


def update_pareto(tag: str, sc: dict, beat: bool):
    lines = [
        "# SNAP-easy pareto — gauss polish",
        "",
        f"_Updated {time.strftime('%Y-%m-%d %H:%M Asia/Shanghai')}_",
        "",
        f"Arm: `{tag}`  beat={beat}",
        "",
        "Sealed: n9 1.092/1.776 · n12 1.197/2.843 · n15 1.418/5.505 · n18 1.752/11.353",
        "",
        "| n | Med / B90 | ×Med / ×B90 |",
        "|---|-----------|-------------|",
    ]
    for n in (9, 12, 15, 18):
        m = sc.get(n)
        if not m:
            continue
        ref = SEALED_REF[n]
        mx = m["median_inv_p"] / ref["Med"]
        bx = m["B90_inv_p"] / ref["B90"]
        lines.append(
            f"| {n} | {m['median_inv_p']:.3f} / {m['B90_inv_p']:.3f} | "
            f"{mx:.3f}× / {bx:.3f}× |"
        )
    lines += [
        "",
        f"**BEAT_SEALED:** {'YES' if beat else 'NO'}",
        "",
    ]
    (OUT.parent / "PARETO_gauss_polish.md").write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nf", type=int, default=8)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--L", type=int, default=32)
    ap.add_argument("--max_train", type=int, default=200)
    ap.add_argument("--max_val", type=int, default=40)
    ap.add_argument("--scale_B", type=int, default=1000)
    ap.add_argument("--seeds", default="0,10,22")
    ap.add_argument("--arms", default="", help="comma filter on arm tags")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    W = build_W_BinH(nf=args.nf)
    L = args.L
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    arm_filter = [x for x in args.arms.split(",") if x.strip()] or None

    g_champ, s_champ, _ = load_gauss_champ()
    g_shell, phi_shell = load_shell_warm(L)

    lb_path = OUT / "leaderboard.csv"
    results = []

    for arm_tag, mode, loss, cvar_q, steps, lr, lb_r, lb_i, leak, warm_src in ARMS:
        if arm_filter and arm_tag not in arm_filter:
            continue
        for seed in seeds:
            if stopped():
                log("STOP"); return
            # warm gamma/snap
            if warm_src == "gauss":
                gamma0 = g_champ[:L].clone()
                snap0 = expand_gauss_to(mode, s_champ[:L].clone())
            else:
                gamma0 = g_shell.clone() if g_shell is not None else g_champ[:L].clone()
                if mode == "gauss":
                    snap0 = warm_snap_for("gauss", False, L, 3, phi_shell)
                    if snap0 is None:
                        snap0 = s_champ[:L].clone()
                else:
                    # fit shell → mode, else lift from gauss champ
                    snap0 = warm_snap_for(mode, False, L, 3, phi_shell)
                    if snap0 is None:
                        snap0 = expand_gauss_to(mode, s_champ[:L].clone())

            full_tag = f"gp_{arm_tag}_s{seed}"
            log(f"=== {full_tag} mode={mode} loss={loss} q={cvar_q} steps={steps} ===")
            best = train_one(
                W, L, args.nf, device,
                train_ns=(6, 9), max_train=args.max_train, max_val=args.max_val,
                seed=seed, lr=lr, steps=steps,
                gamma0=gamma0, snap0=snap0,
                snap_mode=mode, snap_shared=False, few_M=3,
                loss_kind=loss, leakpen=leak, cvar_q=cvar_q,
                lbfgs_mid=True, lbfgs_final=True,
                lbfgs_rounds=lb_r, lbfgs_iter=lb_i,
                arm_tag=full_tag, out_dir=OUT / mode,
            )
            v9 = (best.get("val") or {}).get(9) or {}
            n9med = float(v9.get("median_inv_p", 1e9))
            n9b90 = float(v9.get("B90_inv_p", 1e9))
            log(f"  → train-val n9 {n9med:.4f}/{n9b90:.4f}")
            row = {
                "tag": full_tag, "mode": mode, "loss": loss, "cvar_q": cvar_q,
                "seed": seed, "steps": steps, "lr": lr, "warm": warm_src,
                "n9_Med": n9med, "n9_B90": n9b90,
                "n9_Med_x": n9med / SEALED_REF[9]["Med"],
                "n9_B90_x": n9b90 / SEALED_REF[9]["B90"],
                "path": best.get("path", ""),
                "score": best.get("score"),
            }
            results.append(row)
            # append csv
            import csv
            write_header = not lb_path.exists()
            with lb_path.open("a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(row.keys()))
                if write_header:
                    w.writeheader()
                w.writerow(row)

            # Only full-scale if train-val n9 looks competitive (Med≤1.05× and B90≤1.05×)
            if n9med / SEALED_REF[9]["Med"] <= 1.05 and n9b90 / SEALED_REF[9]["B90"] <= 1.08:
                if stopped():
                    log("STOP"); return
                st = best["state"]
                log(f"=== FULL scale1000 {full_tag} ===")
                sc, _ = scale_record(
                    W, st, args.nf, device, [9, 12, 15, 18], args.scale_B,
                    OUT, full_tag + "_full",
                )
                beat, detail = beats_sealed(sc)
                row["beat_sealed"] = bool(beat)
                for n in (9, 12, 15, 18):
                    m = sc.get(n)
                    if not m:
                        continue
                    ref = SEALED_REF[n]
                    row[f"n{n}_Med"] = m["median_inv_p"]
                    row[f"n{n}_B90"] = m["B90_inv_p"]
                    row[f"n{n}_Med_x"] = m["median_inv_p"] / ref["Med"]
                    row[f"n{n}_B90_x"] = m["B90_inv_p"] / ref["B90"]
                    log(
                        f"  n{n}: {m['median_inv_p']:.3f}/{m['B90_inv_p']:.3f} "
                        f"({row[f'n{n}_Med_x']:.3f}×/{row[f'n{n}_B90_x']:.3f}×)"
                    )
                update_pareto(full_tag, sc, beat)
                if beat:
                    log("*** BEAT_SEALED ***", full_tag)
                    write_beat_json(full_tag, sc, OUT.parent / "BEAT_SEALED.json")
                    (OUT / "BEAT_SEALED.json").write_text(
                        (OUT.parent / "BEAT_SEALED.json").read_text()
                    )
                    break  # seed loop
            if stopped():
                log("STOP"); return
        else:
            continue
        break  # broke from seed due to beat

    (OUT / "summary.json").write_text(json.dumps(_jsonable({"rows": results}), indent=2))
    log("DONE gauss polish →", OUT, "n_rows=", len(results))


if __name__ == "__main__":
    main()
