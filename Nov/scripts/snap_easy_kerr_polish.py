#!/usr/bin/env python3
"""Close Kerr / few_m M=2 B90 gaps (≤2 params/L) after gauss BEAT_SEALED.

Warm-starts from best existing kerr / few_m checkpoints and shell→mode fits.
Tries CVaR / longer Adam+LBFGS / more seeds.

Writes BEAT_SEALED_2PARAM.json only on strict Med+B90 dual-beat at all sealed n
(does NOT overwrite the gauss BEAT_SEALED.json).
Honors Nov/runs/STOP_SNAP_EASY. Does NOT touch angle_explore polish PIDs.
Also always finishes a clean full B=1000 table for the best pre-existing kerr ckpt.
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
from Nov.scripts.angle_explore import beats_sealed, scale_record, train_one  # noqa: E402
from Nov.scripts.nov1st_final_train import SEALED_REF  # noqa: E402
from Nov.scripts.snap_easy_explore import STOP, load_shell_warm, warm_snap_for  # noqa: E402

OUT = _NOV / "runs" / "snap_easy" / "kerr_polish"
EASY = _NOV / "runs" / "snap_easy"

KERR_CKPTS = [
    EASY / "snap_kerr_perL" / "params_easy_snap_kerr_perL_L32_s2_L32_seed2.pt",
    EASY / "snap_kerr_perL" / "params_easy_snap_kerr_perL_L32_s1_L32_seed1.pt",
    EASY / "snap_kerr_perL" / "params_easy_snap_kerr_perL_L32_s0_L32_seed0.pt",
]
FEWM2_CKPTS = [
    EASY / "snap_few_mM2_perL" / "params_easy_snap_few_mM2_perL_L32_s0_L32_seed0.pt",
    EASY / "snap_few_mM2_perL" / "params_easy_snap_few_mM2_perL_L32_s1_L32_seed1.pt",
]
# Prefer polished few_m M=2 (chain warm) when present — these beat explore baselines on n18 B90.
FEWM_POLISH_CKPTS = [
    # Prefer lowest known bottleneck first (cvar096 s22 ~1.264× > cvar098 s22 ~1.296×)
    OUT / "few_m" / "params_kp_fewm2_cvar096_n12_from_polish_s22_L32_seed22.pt",
    OUT / "few_m" / "params_kp_fewm2_cvar098_n12_from_polish_s22_L32_seed22.pt",
    OUT / "few_m" / "params_kp_fewm2_cvar098_n12_from_polish_s0_L32_seed0.pt",
    OUT / "few_m" / "params_kp_fewm2_cvar095_n12_from_polish_s10_L32_seed10.pt",
    OUT / "few_m" / "params_kp_fewm2_cvar095_n12_from_polish_s0_L32_seed0.pt",
    OUT / "few_m" / "params_kp_fewm2_cvar095_n12_from_polish_s2_L32_seed2.pt",
    OUT / "few_m" / "params_kp_fewm2_adam900_n12_from_best_s0_L32_seed0.pt",
    OUT / "few_m" / "params_kp_fewm2_adam900_n12_from_best_s33_L32_seed33.pt",
    OUT / "few_m" / "params_kp_fewm2_adam900_n12_from_best_s44_L32_seed44.pt",
    OUT / "few_m" / "params_kp_fewm2_cvar095_n12_from_best_s0_L32_seed0.pt",
    OUT / "few_m" / "params_kp_fewm2_cvar09_n12_from_best_s0_L32_seed0.pt",
]


def log(*a):
    print(time.strftime("%H:%M:%S"), "[kerr-polish]", *a, flush=True)


def stopped() -> bool:
    return STOP.exists()


def load_ckpt(path: Path):
    st = torch.load(path, map_location="cpu", weights_only=False)
    return st["gamma"].float(), st["snap"].float(), st


# (tag, mode, few_M, loss, cvar_q, steps, lr, lb_rounds, lb_iter, leakpen, warm, train_ns)
# warm: "kerr_best" | "few_m_best" | "shell" | "fresh"
# train_ns: include 12 to attack higher-n B90 blow-up (main Kerr failure mode)
ARMS = [
    # Chain from polished few_m (attacks residual n18 B90 gap; best ~1.301× s10)
    ("fewm2_cvar095_n12_from_polish", "few_m", 2, "cvar", 0.95, 900, 2e-3, 6, 14, 0.05, "few_m_polish", (6, 9, 12)),
    # Higher-CVaR / longer from best polish — push n18 B90 further
    ("fewm2_cvar098_n12_from_polish", "few_m", 2, "cvar", 0.98, 1000, 1.5e-3, 6, 14, 0.03, "few_m_polish", (6, 9, 12)),
    ("fewm2_cvar096_n12_from_polish", "few_m", 2, "cvar", 0.96, 1000, 1.5e-3, 6, 14, 0.04, "few_m_polish", (6, 9, 12)),
    ("fewm2_adam1500_n12_from_polish", "few_m", 2, "log_inv", 0.8, 1500, 1.5e-3, 6, 16, 0.08, "few_m_polish", (6, 9, 12)),
    # Longer / higher-q / n15+n18-proxy from best polish — attack residual n18 B90
    ("fewm2_cvar099_n12_from_polish", "few_m", 2, "cvar", 0.99, 1200, 1.2e-3, 6, 16, 0.02, "few_m_polish", (6, 9, 12)),
    ("fewm2_adam2000_n12_from_polish", "few_m", 2, "log_inv", 0.8, 2000, 1.2e-3, 7, 18, 0.06, "few_m_polish", (6, 9, 12)),
    ("fewm2_cvar098_n12n15_long", "few_m", 2, "cvar", 0.98, 1000, 1.0e-3, 6, 14, 0.03, "few_m_polish", (9, 12, 15)),
    ("fewm2_cvar09_n12_from_polish", "few_m", 2, "cvar", 0.9, 900, 2e-3, 6, 14, 0.05, "few_m_polish", (6, 9, 12)),
    ("fewm2_adam1200_n12_from_polish", "few_m", 2, "log_inv", 0.8, 1200, 2e-3, 6, 14, 0.1, "few_m_polish", (6, 9, 12)),
    ("fewm2_cvar098_n15_from_polish", "few_m", 2, "cvar", 0.98, 800, 1.2e-3, 5, 12, 0.03, "few_m_polish", (6, 9, 12, 15)),
    ("fewm2_cvar095_n15_from_polish", "few_m", 2, "cvar", 0.95, 700, 1.5e-3, 5, 12, 0.05, "few_m_polish", (6, 9, 12, 15)),
    ("fewm2_adam900_n15_from_polish", "few_m", 2, "log_inv", 0.8, 900, 2e-3, 5, 12, 0.1, "few_m_polish", (6, 9, 12, 15)),
    ("fewm2_cvar09_n15_from_polish", "few_m", 2, "cvar", 0.9, 700, 1.5e-3, 5, 12, 0.05, "few_m_polish", (6, 9, 12, 15)),
    # Warm from explore few_m best
    ("fewm2_adam900_n12_from_best", "few_m", 2, "log_inv", 0.8, 900, 3e-3, 5, 12, 0.1, "few_m_best", (6, 9, 12)),
    ("fewm2_cvar095_n12_from_best", "few_m", 2, "cvar", 0.95, 800, 2e-3, 6, 14, 0.05, "few_m_best", (6, 9, 12)),
    ("fewm2_cvar09_n12_from_best", "few_m", 2, "cvar", 0.9, 800, 2e-3, 5, 12, 0.05, "few_m_best", (6, 9, 12)),
    ("fewm2_adam800_from_best", "few_m", 2, "log_inv", 0.8, 800, 5e-3, 4, 10, 0.1, "few_m_best", (6, 9)),
    ("fewm2_cvar08_from_best", "few_m", 2, "cvar", 0.8, 700, 2e-3, 5, 12, 0.1, "few_m_best", (6, 9)),
    ("fewm2_cvar08_from_shell", "few_m", 2, "cvar", 0.8, 700, 2e-3, 5, 12, 0.1, "shell", (6, 9)),
    ("fewm2_cvar09_from_best", "few_m", 2, "cvar", 0.9, 700, 2e-3, 5, 12, 0.05, "few_m_best", (6, 9)),
    ("kerr_adam600_lr3e3_shell", "kerr", 3, "log_inv", 0.8, 600, 3e-3, 4, 10, 0.1, "shell", (6, 9)),
    # B90-focused: train with n12 + higher CVaR
    ("kerr_adam900_n12_from_best", "kerr", 3, "log_inv", 0.8, 900, 3e-3, 5, 12, 0.1, "kerr_best", (6, 9, 12)),
    ("kerr_cvar08_n12_from_best", "kerr", 3, "cvar", 0.8, 800, 2e-3, 5, 12, 0.1, "kerr_best", (6, 9, 12)),
    ("kerr_cvar095_n12_from_best", "kerr", 3, "cvar", 0.95, 800, 2e-3, 6, 14, 0.05, "kerr_best", (6, 9, 12)),
    ("kerr_cvar09_n12_from_best", "kerr", 3, "cvar", 0.9, 800, 2e-3, 5, 12, 0.05, "kerr_best", (6, 9, 12)),
    ("kerr_adam800_from_best", "kerr", 3, "log_inv", 0.8, 800, 5e-3, 4, 10, 0.1, "kerr_best", (6, 9)),
    ("kerr_cvar08_from_best", "kerr", 3, "cvar", 0.8, 700, 2e-3, 5, 12, 0.1, "kerr_best", (6, 9)),
    ("kerr_cvar08_from_shell", "kerr", 3, "cvar", 0.8, 700, 2e-3, 5, 12, 0.1, "shell", (6, 9)),
    ("kerr_cvar09_from_best", "kerr", 3, "cvar", 0.9, 700, 2e-3, 5, 12, 0.05, "kerr_best", (6, 9)),
]


def write_2param_beat(tag: str, sc: dict, path: Path):
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
        "params_per_L": 2,
        "beat_sealed": True,
        "criterion": "Med AND B90 <= sealed at all n in {9,12,15,18}",
        "detail": detail,
        "written_at": time.strftime("%Y-%m-%d %H:%M:%S Asia/Shanghai"),
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")
    path.with_suffix("").write_text(tag + "\n")


def update_pareto_2param(tag: str, mode: str, sc: dict, beat: bool):
    lines = [
        "# SNAP-easy pareto — ≤2-param polish (kerr / few_m M=2)",
        "",
        f"_Updated {time.strftime('%Y-%m-%d %H:%M Asia/Shanghai')}_",
        "",
        f"Arm: `{tag}`  mode={mode}  beat={beat}",
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
    lines += ["", f"**BEAT_SEALED (≤2 params/L):** {'YES' if beat else 'NO'}", ""]
    (OUT.parent / "PARETO_2param.md").write_text("\n".join(lines) + "\n")


def first_existing(paths):
    for p in paths:
        if p.exists():
            return p
    return None


def run_full_scale(W, st, nf, device, scale_B, tag: str):
    log(f"=== FULL scale1000 {tag} B={scale_B} ===")
    sc, _ = scale_record(
        W, st, nf, device, [9, 12, 15, 18], scale_B, OUT, tag + "_full",
    )
    # also mirror into parent snap_easy/ for RESULTS aggregator
    mirror = EASY / f"scale1000_{tag}_full.json"
    payload = {
        "tag": tag + "_full",
        "params": 2,
        "scale": {str(n): sc[n] for n in (9, 12, 15, 18) if n in sc},
        "rows": [],
    }
    for n in (9, 12, 15, 18):
        m = sc.get(n)
        if not m:
            continue
        ref = SEALED_REF[n]
        payload["rows"].append({
            "tag": tag + "_full", "n": n, "L": 32,
            "Med": m["median_inv_p"], "B90": m["B90_inv_p"],
            "Med_x": m["median_inv_p"] / ref["Med"],
            "B90_x": m["B90_inv_p"] / ref["B90"],
            "B": scale_B,
        })
        log(
            f"  n{n}: {m['median_inv_p']:.3f}/{m['B90_inv_p']:.3f} "
            f"({m['median_inv_p']/ref['Med']:.3f}×/{m['B90_inv_p']/ref['B90']:.3f}×)"
        )
    beat, _ = beats_sealed(sc)
    payload["beat_sealed"] = bool(beat)
    mirror.write_text(json.dumps(payload, indent=2) + "\n")
    # also write a kerr-specific clean table marker
    if tag.startswith("kerr_baseline"):
        (EASY / "KERR_FULL_B1000.json").write_text(json.dumps(payload, indent=2) + "\n")
    return sc, beat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nf", type=int, default=8)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--L", type=int, default=32)
    ap.add_argument("--max_train", type=int, default=200)
    ap.add_argument("--max_val", type=int, default=40)
    ap.add_argument("--scale_B", type=int, default=1000)
    ap.add_argument("--seeds", default="0,2,10,22")
    ap.add_argument("--arms", default="", help="comma filter on arm tags")
    ap.add_argument("--skip-baseline-scale", action="store_true")
    ap.add_argument("--retrain-existing", action="store_true",
                    help="retrain even if ckpt already exists")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    W = build_W_BinH(nf=args.nf)
    L = args.L
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    arm_filter = [x for x in args.arms.split(",") if x.strip()] or None

    g_shell, phi_shell = load_shell_warm(L)

    # --- baseline: finish clean full B=1000 on best existing kerr + few_m ---
    if not args.skip_baseline_scale and not stopped():
        for label, paths, mode, few_M in (
            ("kerr_baseline_s2", KERR_CKPTS, "kerr", 3),
            ("fewm2_baseline_s0", FEWM2_CKPTS[:1], "few_m", 2),
        ):
            pt = first_existing(paths)
            if pt is None:
                log("missing baseline", label)
                continue
            # skip if already have a clean full table for this tag in EASY
            if (EASY / f"scale1000_{label}_full.json").exists() and label.startswith("fewm2"):
                # few_m M2 already has easy_snap_few_mM2 full; still run kerr
                if "kerr" not in label:
                    log("skip existing", label)
                    continue
            if (EASY / "KERR_FULL_B1000.json").exists() and "kerr" in label:
                log("KERR_FULL_B1000.json already present; skip baseline kerr scale")
                continue
            gamma, snap, st = load_ckpt(pt)
            st = {"gamma": gamma[:L], "snap": snap[:L] if snap.dim() > 1 else snap,
                  "snap_mode": mode, "snap_shared": False, "few_M": few_M}
            # preserve other keys from file
            raw = torch.load(pt, map_location="cpu", weights_only=False)
            raw["gamma"] = gamma[:L]
            raw["snap"] = snap if snap.shape[0] == L or snap.dim() == 1 else snap[:L]
            sc, beat = run_full_scale(W, raw, args.nf, device, args.scale_B, label)
            update_pareto_2param(label, mode, sc, beat)
            if beat:
                log("*** BEAT_SEALED ≤2-param (baseline) ***", label)
                write_2param_beat(label, sc, EASY / "BEAT_SEALED_2PARAM.json")

    lb_path = OUT / "leaderboard.csv"
    results = []

    kerr_best_pt = first_existing(KERR_CKPTS)
    few_best_pt = first_existing(FEWM2_CKPTS)
    few_polish_pt = first_existing(FEWM_POLISH_CKPTS)
    g_kerr = s_kerr = None
    g_few = s_few = None
    g_few_p = s_few_p = None
    if kerr_best_pt:
        g_kerr, s_kerr, _ = load_ckpt(kerr_best_pt)
    if few_best_pt:
        g_few, s_few, _ = load_ckpt(few_best_pt)
    if few_polish_pt:
        g_few_p, s_few_p, _ = load_ckpt(few_polish_pt)
        log("few_m_polish warm from", few_polish_pt.name)
    else:
        log("few_m_polish warm missing; polish-chain arms will fall back to fresh")

    for arm_tag, mode, few_M, loss, cvar_q, steps, lr, lb_r, lb_i, leak, warm_src, train_ns in ARMS:
        if arm_filter and arm_tag not in arm_filter:
            continue
        for seed in seeds:
            if stopped():
                log("STOP"); return
            full_tag = f"kp_{arm_tag}_s{seed}"
            # Skip if this arm+seed already has a full B=1000 table
            if (EASY / f"scale1000_{full_tag}_full.json").exists() or (
                OUT / f"scale1000_{full_tag}_full.json"
            ).exists():
                log("skip done full", full_tag)
                continue
            # Skip if ckpt already trained (resume across watchdog restarts)
            ckpt_guess = OUT / mode / f"params_{full_tag}_L{L}_seed{seed}.pt"
            if ckpt_guess.exists() and not args.retrain_existing:
                # still try full-scale if train-val looked close but scale missing
                try:
                    gamma, snap, raw = load_ckpt(ckpt_guess)
                    raw = dict(raw)
                    raw["gamma"] = gamma[:L]
                    raw["snap"] = snap[:L] if snap.dim() > 1 else snap
                    raw["snap_mode"] = mode
                    raw["snap_shared"] = False
                    raw["few_M"] = few_M
                    meta = json.loads(
                        (OUT / mode / f"meta_{full_tag}_L{L}_seed{seed}.json").read_text()
                    ) if (OUT / mode / f"meta_{full_tag}_L{L}_seed{seed}.json").exists() else {}
                    v9 = ((meta.get("val") or {}).get("9") or (meta.get("val") or {}).get(9) or {})
                    n9med = float(v9.get("Med", v9.get("median_inv_p", 1e9)))
                    n9b90 = float(v9.get("B90", v9.get("B90_inv_p", 1e9)))
                    log(f"resume ckpt {full_tag} train-val n9 {n9med:.4f}/{n9b90:.4f}")
                    if (n9med / SEALED_REF[9]["Med"] <= 1.15
                            and n9b90 / SEALED_REF[9]["B90"] <= 1.45):
                        sc, beat = run_full_scale(
                            W, raw, args.nf, device, args.scale_B, full_tag,
                        )
                        update_pareto_2param(full_tag, mode, sc, beat)
                        if beat:
                            log("*** BEAT_SEALED ≤2-param ***", full_tag)
                            write_2param_beat(full_tag, sc, EASY / "BEAT_SEALED_2PARAM.json")
                            (OUT / "BEAT_SEALED_2PARAM.json").write_text(
                                (EASY / "BEAT_SEALED_2PARAM.json").read_text()
                            )
                            break
                    continue
                except Exception as e:
                    log("resume failed, retrain", full_tag, e)

            # Re-resolve polish warm each seed so newer full-evals (e.g. s10 1.301×) chain in
            if warm_src == "few_m_polish":
                fresh_pt = first_existing(FEWM_POLISH_CKPTS)
                if fresh_pt is not None:
                    g_few_p, s_few_p, _ = load_ckpt(fresh_pt)
                    log("few_m_polish warm →", fresh_pt.name)
            if warm_src == "kerr_best" and g_kerr is not None:
                gamma0 = g_kerr[:L].clone()
                snap0 = s_kerr[:L].clone() if s_kerr.dim() > 1 else s_kerr.clone()
            elif warm_src == "few_m_polish" and g_few_p is not None:
                gamma0 = g_few_p[:L].clone()
                snap0 = s_few_p[:L].clone() if s_few_p.dim() > 1 else s_few_p.clone()
            elif warm_src == "few_m_polish" and g_few is not None:
                # polish ckpt not ready yet — fall back to explore few_m best
                gamma0 = g_few[:L].clone()
                snap0 = s_few[:L].clone() if s_few.dim() > 1 else s_few.clone()
            elif warm_src == "few_m_best" and g_few is not None:
                gamma0 = g_few[:L].clone()
                snap0 = s_few[:L].clone() if s_few.dim() > 1 else s_few.clone()
            elif warm_src == "shell":
                gamma0 = g_shell.clone() if g_shell is not None else (
                    g_kerr[:L].clone() if g_kerr is not None else None
                )
                snap0 = warm_snap_for(mode, False, L, few_M, phi_shell)
            else:
                gamma0, snap0 = None, None

            # Cap per-n when training with n12/n15 so wall time stays sane
            mt = args.max_train
            if 15 in train_ns:
                mt = min(mt, 60)
            elif 12 in train_ns:
                mt = min(mt, 120)

            log(f"=== {full_tag} mode={mode} few_M={few_M} loss={loss} q={cvar_q} train_ns={train_ns} ===")
            best = train_one(
                W, L, args.nf, device,
                train_ns=train_ns, max_train=mt, max_val=args.max_val,
                seed=seed, lr=lr, steps=steps,
                gamma0=gamma0, snap0=snap0,
                snap_mode=mode, snap_shared=False, few_M=few_M,
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
                "tag": full_tag, "mode": mode, "few_M": few_M, "loss": loss,
                "cvar_q": cvar_q, "seed": seed, "steps": steps, "lr": lr,
                "warm": warm_src, "n9_Med": n9med, "n9_B90": n9b90,
                "n9_Med_x": n9med / SEALED_REF[9]["Med"],
                "n9_B90_x": n9b90 / SEALED_REF[9]["B90"],
                "path": best.get("path", ""), "score": best.get("score"),
            }
            results.append(row)
            write_header = not lb_path.exists()
            with lb_path.open("a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(row.keys()))
                if write_header:
                    w.writeheader()
                w.writerow(row)

            # Full-scale if train-val looks within striking distance of sealed
            # (kerr historically ~1.07 Med / 1.5 B90 — cast a wider net)
            # Wider net when n12 was in the train set (B90 is the real target)
            if 15 in train_ns:
                med_gate, b90_gate = 1.22, 1.60
            elif 12 in train_ns:
                med_gate, b90_gate = 1.18, 1.50
            else:
                med_gate, b90_gate = 1.12, 1.35
            if (n9med / SEALED_REF[9]["Med"] <= med_gate
                    and n9b90 / SEALED_REF[9]["B90"] <= b90_gate):
                if stopped():
                    log("STOP"); return
                sc, beat = run_full_scale(
                    W, best["state"], args.nf, device, args.scale_B, full_tag,
                )
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
                update_pareto_2param(full_tag, mode, sc, beat)
                if beat:
                    log("*** BEAT_SEALED ≤2-param ***", full_tag)
                    write_2param_beat(full_tag, sc, EASY / "BEAT_SEALED_2PARAM.json")
                    (OUT / "BEAT_SEALED_2PARAM.json").write_text(
                        (EASY / "BEAT_SEALED_2PARAM.json").read_text()
                    )
                    break  # seed loop
            if stopped():
                log("STOP"); return
        else:
            continue
        break  # broke from seed due to beat

    (OUT / "summary.json").write_text(json.dumps(_jsonable({"rows": results}), indent=2))
    log("DONE kerr/few_m polish →", OUT, "n_rows=", len(results))


if __name__ == "__main__":
    main()
