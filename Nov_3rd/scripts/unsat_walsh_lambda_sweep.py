#!/usr/bin/env python3
"""Sweep λ at the PR #28 point μ=−1.00. Same gate and L=16 budget.

    H = diag(unsat) − λ Σ_i X_i + μ Σ_cav Σ_{i<j} X_i X_j
    μ = −1.00
    λ ∈ {0.50, 0.65, 0.75, 0.90, 1.00, 1.25}

Train n=6 and n=9 only. Freeze n=12 and n=15. ⟨H⟩ is the Z histogram plus the
X histogram (single-X and intra-cavity XX). λ=0.75 reuses the PR #28 L=16
checkpoints. n=21 is not loaded.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import sys
import time
import traceback
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from Nov_3rd.unsat_lambda_sweep import (
    ANCHOR_LAM,
    FIXED_MU,
    LAM_GRID,
    PROD_GAP_MIN,
    PUBLISHED_ANCHOR_R,
    cache_filename,
    decide_lambda_sweep,
    file_tag,
    gap_tail,
    lam_label,
    mu_ckpt_tag,
)
from Nov_3rd.unsat_rank2_xx import batch_ground

SPECS = (
    (6, "train", 128),
    (6, "hold", 128),
    (9, "train", 128),
    (9, "hold", 128),
    (12, "transfer", 64),
    (15, "transfer", 32),
)


def log(*a):
    print(time.strftime("%H:%M:%S"), "[lamSweep]", *a, flush=True)


def _load_driver():
    path = Path(__file__).resolve().parent / "unsat_walsh_rank2_mu.py"
    spec = importlib.util.spec_from_file_location("unsat_walsh_rank2_mu", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    if isinstance(o, (np.floating, float)):
        return float(o)
    if isinstance(o, (np.integer, int)):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, torch.Tensor):
        return o.detach().cpu().tolist()
    if o is None or isinstance(o, str):
        return o
    return str(o)


def _paths_for(out: Path, L: int, method: str, seed: int, lam: float):
    stem = Path(out) / "runs" / f"lam{file_tag(lam)}_L{int(L)}_{method}_seed{int(seed)}"
    return Path(str(stem) + ".json"), Path(str(stem) + ".pt")


def _cache_path(out: Path, n: int, role: str, count: int, lam: float) -> Path:
    return out / "cache" / cache_filename(n, role, count, lam, FIXED_MU)


def _refuse_n21(*paths: Path):
    for path in paths:
        name = str(path)
        if "n21" in name or "n_21" in name or "/21/" in name:
            raise SystemExit(f"refusing n=21 path {path}")


def _copy_anchor(anchor: Path, out: Path):
    """PR #28 caches and L=16 checkpoints at λ=0.75, μ=−1.00. Not a retrain."""
    _refuse_n21(anchor, out)
    for n, role, count in SPECS:
        if int(n) == 21:
            raise SystemExit("refusing n=21")
        src = anchor / "cache" / cache_filename(n, role, count, ANCHOR_LAM, FIXED_MU)
        dst = _cache_path(out, n, role, count, ANCHOR_LAM)
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            continue
        if not src.exists():
            raise FileNotFoundError(src)
        shutil.copy2(src, dst)
        log(f"copied cache n={n} {role}")
    runs = out / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    ckpt = mu_ckpt_tag(FIXED_MU)
    for method in ("fourier_qaoa", "power_p", "walsh"):
        for seed in (0, 1, 2):
            for ext in (".json", ".pt"):
                src_name = f"{ckpt}_L16_{method}_seed{seed}{ext}"
                dst_js, dst_pt = _paths_for(out, 16, method, seed, ANCHOR_LAM)
                dst = dst_js if ext == ".json" else dst_pt
                if dst.exists():
                    continue
                src = anchor / "runs" / src_name
                if not src.exists():
                    raise FileNotFoundError(src)
                shutil.copy2(src, dst)
    for method in ("fourier_qaoa", "power_p", "walsh"):
        for seed in (0, 1, 2):
            js, pt = _paths_for(out, 16, method, seed, ANCHOR_LAM)
            rec = json.loads(js.read_text())
            if abs(float(rec["lam"]) - ANCHOR_LAM) > 1e-12 or abs(float(rec["mu"]) - FIXED_MU) > 1e-12:
                raise RuntimeError(f"anchor checkpoint drift {js.name}")
            if not pt.exists():
                raise RuntimeError(f"missing anchor weights {pt.name}")
            if "n21" in js.name:
                raise RuntimeError(f"n=21 checkpoint {js.name}")
    log("PR #28 λ=0.75 μ=−1.00 L=16 checkpoints are in place")


def _build_caches(anchor_out: Path, out: Path, lams, workers: int):
    """Reuse the PR #26 unsat diagonals. Only E0 changes with λ."""
    anchors = {}
    for n, role, count in SPECS:
        src = _cache_path(anchor_out, n, role, count, ANCHOR_LAM)
        z = np.load(src)
        anchors[(n, role)] = {k: z[k] for k in ("score", "pulse", "seeds", "m", "lam", "mu")}
        if int(anchors[(n, role)]["score"].shape[0]) != int(count):
            raise RuntimeError(f"anchor n={n} {role} B drift")
        if abs(float(anchors[(n, role)]["lam"]) - ANCHOR_LAM) > 1e-12:
            raise RuntimeError(f"anchor λ drift n={n} {role}")
        if abs(float(anchors[(n, role)]["mu"]) - FIXED_MU) > 1e-12:
            raise RuntimeError(f"anchor μ drift n={n} {role}")
    quantum = {}
    for lam in lams:
        for n, role, count in SPECS:
            if int(n) == 21:
                raise SystemExit("refusing n=21")
            dst = _cache_path(out, n, role, count, lam)
            if dst.exists():
                blob = np.load(dst)
                if abs(float(blob["lam"]) - float(lam)) > 1e-12 or abs(float(blob["mu"]) - FIXED_MU) > 1e-12:
                    raise RuntimeError(f"cache λ/μ drift {dst.name}")
                quantum[(lam_label(lam), n, role)] = {
                    "peak_med": float(np.median(blob["peak"])),
                    "entropy_med": float(np.median(blob["entropy"])),
                    "prod_gap_med": float(np.median(blob["prod_gap"])),
                    "residual_max": float(np.max(blob["residual"])),
                }
                continue
            base = anchors[(n, role)]
            t0 = time.time()
            log(f"ED n={n} {role} B={count} λ={lam:.2f} μ={FIXED_MU:+.2f}")
            gu = batch_ground(
                base["score"], n, float(lam), FIXED_MU, workers, seeds=base["seeds"]
            )
            dst.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                dst,
                score=base["score"],
                pulse=base["pulse"],
                e0=gu["E0"],
                peak=gu["peak"],
                entropy=gu["entropy"],
                residual=gu["residual"],
                e_prod=gu["E_prod"],
                prod_gap=gu["prod_gap"],
                seeds=base["seeds"],
                m=base["m"],
                lam=np.asarray(float(lam), dtype=np.float64),
                mu=np.asarray(float(FIXED_MU), dtype=np.float64),
            )
            quantum[(lam_label(lam), n, role)] = {
                "peak_med": float(np.median(gu["peak"])),
                "entropy_med": float(np.median(gu["entropy"])),
                "prod_gap_med": float(np.median(gu["prod_gap"])),
                "residual_max": float(np.max(gu["residual"])),
            }
            log(
                f"  n={n} {role} λ={lam:.2f} in {time.time() - t0:.1f}s "
                f"peak={quantum[(lam_label(lam), n, role)]['peak_med']:.3f} "
                f"gap={quantum[(lam_label(lam), n, role)]['prod_gap_med']:.3f}"
            )
    return quantum


def _row_from_cells(cells: dict, qdec: dict) -> dict:
    row = {"walsh": {}, "power_p": {}, "fourier": {}, "quantum": qdec}
    for method in ("walsh", "power_p", "fourier_qaoa"):
        key = "fourier" if method == "fourier_qaoa" else method
        for n in (6, 9, 12, 15):
            cell = cells.get((method, 16, n))
            if cell is None:
                continue
            packed = {
                "R": float(cell["R"]) if "R" in cell else None,
                "mark": cell.get("mark"),
                "med": float(cell["med"]),
                "seed_std": float(cell["seed_std"]),
                "per_seed": [float(x) for x in cell["per_seed"]],
                "seeds": [int(s) for s in cell["seeds"]],
                "n_params": int(cell["n_params"]),
                "role": cell["role"],
            }
            if "seed_Rs" in cell:
                packed["seed_Rs"] = [float(x) for x in cell["seed_Rs"]]
                packed["qaoa"] = float(cell["qaoa"])
            row[key][n] = packed
    return row


def _fmt(x, nd=3):
    if x is None:
        return "—"
    return f"{float(x):.{nd}f}"


def _at(block, n):
    if n in block:
        return block[n]
    return block[str(n)]


def _pct_gap(q: dict) -> str:
    if "frac_gap_below" not in q:
        return "—"
    frac = 100.0 * float(q["frac_gap_below"])
    return f"{frac:.1f} ({int(q['n_gap_below'])}/{int(q['n_inst'])})"


def render(payload: dict) -> str:
    v = payload["verdict"]
    lines = []
    lines.append("# RESULTS — unsat Walsh λ sweep at μ=−1.00, L=16 freeze n=12")
    lines.append("")
    lines.append(f"**Verdict: {v['verdict']}**")
    lines.append("")
    lines.append(v["reason"])
    lines.append("")
    lines.append("## Setup")
    lines.append("")
    lines.append(
        "Same gate as PR #26 and PR #28. The intra-cavity XX term is scored from the X histogram. "
        "It is not compiled into U_c. Angles for n=12 and n=15 were not in the loss. "
        "n=21 was not loaded."
    )
    lines.append("")
    lines.append("```text")
    lines.append("H = diag(unsat) − λ Σ_i X_i + μ Σ_cav Σ_{i<j} X_i X_j")
    lines.append("μ = −1.00")
    lines.append("λ ∈ {0.50, 0.65, 0.75, 0.90, 1.00, 1.25}")
    lines.append("U_c^X = W exp(−i γ C_n) W ,  W = H^{⊗3} ,  C_n = (n/m) unsat − max")
    lines.append("⟨H⟩ = Σ_z p_Z(z) unsat(z) − λ Σ_i ⟨X_i⟩ + μ Σ_cav Σ_{i<j} ⟨X_i X_j⟩")
    lines.append("ε = (⟨H⟩ − E_0) / n")
    lines.append("```")
    lines.append("")
    lines.append(
        f"Train B={payload['B_train']} per n from the PR #26 unsat diagonals "
        f"(those packs are base 51e9 / scale1000 base 54e9). "
        f"Holdout B={payload['B_hold']}. Freeze n=12 B={payload['B_transfer']}, "
        f"n=15 B={payload['B_n15']}. Seeds {payload['seeds']}. L=16 only. "
        f"L-BFGS rounds {payload['rounds_L16']}, max_iter {payload['max_iter_L16']}, "
        f"lr={payload['lr']}. Early stop is 8 rounds without a 10⁻⁶ improvement. "
        "That is the PR #26 L=16 budget. Fourier stores 2L Zhou coefficients. "
        "power_p stores 4L+1. Walsh stores 7L+1 and copies the scalar γ onto degrees 1–4 "
        "at the warm start. The loss is the mean of the per-n mean residuals on n=6 and n=9. "
        "The checkpoint minimizes the worst of the two holdout ratios against that seed's Fourier. "
        "λ=0.75 is the PR #28 checkpoint at μ=−1.00, not a retrain."
    )
    lines.append("")
    lines.append(
        "R = (median over seeds of median ε) / (the same number for Fourier QAOA on this λ). "
        "BEAT if R≤0.98, TIE if R≤1.02, else UNDER. "
        "The operating λ is the lowest frozen n=12 Walsh R among cells with median product gap "
        "≥ 0.02. If the lowest R on the grid misses that gap, both that unconstrained winner "
        "and the safe winner are named. Absolute residuals are not compared across λ."
    )
    lines.append("")
    lines.append("## Quantumness")
    lines.append("")
    lines.append(
        "E0 is Lanczos on the sparse Hamiltonian. A split is a bitstring when the median "
        "ground-state peak is at least 0.5. The product gap is (E_prod − E_0) / |E_0|. "
        "% gap<0.02 is the fraction of instances in that split under the median-gap gate."
    )
    lines.append("")
    lines.append(
        f"Worst |ΔE0| Lanczos vs dense over the grid = {payload['audit']['worst_e0']:.2e}. "
        f"Histogram vs ⟨ψ|H|ψ⟩ = {payload['audit']['worst_psi']:.2e}."
    )
    lines.append("")
    lines.append("| λ | n | role | peak Med | prod gap Med | % gap<0.02 |")
    lines.append("|--:|--:|:-----|---------:|-------------:|-----------:|")
    qrows = payload["quantum_rows"]
    for lam in payload["lams"]:
        block = qrows[lam_label(lam)]
        for n, role in ((6, "hold"), (9, "hold"), (12, "transfer"), (15, "transfer")):
            q = _at(block, n)
            lines.append(
                f"| {lam:.2f} | {n} | {role} | {_fmt(q['peak_med'])} | {_fmt(q['prod_gap_med'])} | {_pct_gap(q)} |"
            )
    lines.append("")
    lines.append("## Freeze table")
    lines.append("")
    lines.append(
        "n=6 and n=9 are the selection holdout. n=12 and n=15 are not. "
        "power_p is the scalar-γ control and does not choose λ."
    )
    lines.append("")
    lines.append("| λ | head | n=6 R | n=9 R | n=12 R | n=12 mark | n=15 R | n=12 Med ε |")
    lines.append("|--:|:-----|------:|------:|-------:|:----------|-------:|-----------:|")
    rows = payload["rows"]
    for lam in payload["lams"]:
        row = rows[lam_label(lam)]
        for head in ("walsh", "power_p"):
            if head not in row or 12 not in row[head] and "12" not in row[head]:
                continue
            if not row[head]:
                continue
            b6, b9, b12, b15 = (_at(row[head], n) for n in (6, 9, 12, 15))
            lines.append(
                f"| {lam:.2f} | {head} | {_fmt(b6['R'])} | {_fmt(b9['R'])} | "
                f"{_fmt(b12['R'])} | {b12['mark']} | {_fmt(b15['R'])} | {_fmt(b12['med'], 4)} |"
            )
    lines.append("")
    lines.append("Per-seed n=12 Walsh R, seeds [0, 1, 2]:")
    lines.append("")
    for lam in payload["lams"]:
        cell = _at(rows[lam_label(lam)]["walsh"], 12)
        seeds = ", ".join(_fmt(s) for s in cell["seed_Rs"])
        seed_med = float(np.median(cell["seed_Rs"]))
        lines.append(
            f"- λ={lam:.2f}: {seeds} (median of those R = {_fmt(seed_med)}); "
            f"ratio of medians = {_fmt(cell['R'])}"
        )
    lines.append("")
    lines.append("## Reading")
    lines.append("")
    seq = ", ".join(
        f"{_fmt(_at(rows[lam_label(lam)]['walsh'], 12)['R'])}" for lam in payload["lams"]
    )
    lines.append(
        "Frozen n=12 Walsh R is "
        + seq
        + " as λ runs "
        + ", ".join(f"{lam:.2f}" for lam in payload["lams"])
        + "."
    )
    gap_bits = []
    thin = []
    for lam in payload["lams"]:
        q = _at(qrows[lam_label(lam)], 12)
        gap_bits.append(
            f"{lam:.2f} gap {_fmt(q['prod_gap_med'])}, {int(q['n_gap_below'])}/{int(q['n_inst'])} under 0.02"
        )
        if float(q["prod_gap_med"]) + 1e-12 < PROD_GAP_MIN:
            thin.append(lam)
    lines.append("n=12 product gap by λ: " + "; ".join(gap_bits) + ".")
    if thin and not v.get("split"):
        names = ", ".join(f"{float(lam):.2f}" for lam in thin)
        lines.append(
            f"λ={names} misses the median-gap gate and does not undercut the gap-safe minimum, "
            "so the unconstrained winner and the safe winner are the same cell."
        )
    if v.get("split"):
        u_lam = v.get("unconstrained_lam")
        s_lam = v.get("safe_lam")
        if u_lam is not None and s_lam is not None:
            lines.append(
                f"Unconstrained n=12 winner λ={float(u_lam):.2f} misses the median-gap gate. "
                f"Safe winner λ={float(s_lam):.2f} is the lowest R with median gap ≥ 0.02."
            )
        elif u_lam is not None:
            lines.append(
                f"Unconstrained n=12 winner λ={float(u_lam):.2f} misses the median-gap gate. "
                "No safe winner on this grid."
            )
    elif v.get("operating_lam") is not None:
        lines.append(
            f"The lowest frozen n=12 Walsh R with median gap ≥ 0.02 is "
            f"λ={float(v['operating_lam']):.2f} "
            f"(R={_fmt(v.get('R12') or v.get('R_anchor'))})."
        )
    # power_p stays above Walsh when that is true at every λ.
    above = []
    for lam in payload["lams"]:
        row = rows[lam_label(lam)]
        if not row.get("power_p"):
            continue
        above.append(
            float(_at(row["power_p"], 12)["R"]) + 1e-12 >= float(_at(row["walsh"], 12)["R"])
        )
    rs = [float(_at(rows[lam_label(lam)]["walsh"], 12)["R"]) for lam in payload["lams"]]
    anchor_i = next((i for i, lam in enumerate(payload["lams"]) if abs(float(lam) - ANCHOR_LAM) < 1e-9), None)
    if anchor_i is not None and rs and rs[anchor_i] <= min(rs) + 1e-12:
        left_higher = anchor_i > 0 and rs[anchor_i - 1] > rs[anchor_i] + 1e-12
        right_higher = anchor_i + 1 < len(rs) and rs[anchor_i + 1] > rs[anchor_i] + 1e-12
        if left_higher and right_higher:
            lines.append(
                "λ=0.75 is the lowest frozen n=12 Walsh R on this grid, and both neighbors are higher."
            )
        else:
            lines.append("λ=0.75 is the lowest frozen n=12 Walsh R on this grid.")
    n15 = [(float(lam), float(_at(rows[lam_label(lam)]["walsh"], 15)["R"])) for lam in payload["lams"]]
    if n15:
        best15 = min(n15, key=lambda t: t[1])
        lines.append(f"Frozen n=15 Walsh R is lowest at λ={best15[0]:.2f} ({best15[1]:.3f}).")
    if above and all(above):
        p_best = min(
            payload["lams"],
            key=lambda lam: float(_at(rows[lam_label(lam)]["power_p"], 12)["R"]),
        )
        lines.append(
            "power_p n=12 R is at least the Walsh R at every λ. "
            f"power_p's lowest n=12 R is λ={float(p_best):.2f}."
        )
    elif above:
        lines.append(
            "power_p is below Walsh at some λ on n=12, so the λ ordering can depend on the head."
        )
    lines.append("")
    lines.append("## One-parameter cost pulse")
    lines.append("")
    lines.append("SNAP phase 0. XX is not in that pulse. The scan is the minimum median ε over γ ∈ [−2, 2].")
    lines.append("")
    lines.append("| λ | n | γ | Med ε | Walsh L=16 Med ε |")
    lines.append("|--:|--:|--:|------:|------------------:|")
    for lam in payload["lams"]:
        scan = payload["cost_scan"][lam_label(lam)]
        for n in (6, 9, 12, 15):
            sc = _at(scan, n)
            med = _at(rows[lam_label(lam)]["walsh"], n)["med"]
            lines.append(
                f"| {lam:.2f} | {n} | {float(sc['gamma']):.4f} | {_fmt(sc['med_eps'], 4)} | {_fmt(med, 4)} |"
            )
    lines.append("")
    lines.append("## Rule")
    lines.append("")
    lines.append(
        "Operating λ is the grid point with the lowest frozen n=12 L=16 Walsh ratio among cells "
        f"whose median product gap is at least {PROD_GAP_MIN:.2f}. The other gates stay: R≤0.98, "
        "at least two seeds beat their own Fourier, n=6/9/12 peak < 0.5, and n=15 R stays below "
        "1.10 with a non-bitstring ground state. If the lowest ratio on the whole grid misses the "
        "gap gate, the report names that unconstrained winner and the safe winner. The safe winner "
        "is the operating point when it clears the gates."
    )
    lines.append("")
    lines.append(
        "PASS if that safe cell clears the gates, including when it is still λ=0.75. "
        "needs_edit if the grid or the PR #28 anchor is incomplete, no cell keeps median gap ≥ 0.02, "
        "or the gap-safe minimum fails a transfer gate. KILL if that cell's ground state (n=6, 9, or 12) "
        "is a bitstring, Walsh there loses to Fourier, or the L=16 advantage reverses. "
        "power_p does not gate the call."
    )
    lines.append("")
    lines.append(
        f"Optimizer time summed over new fits {payload['opt_s']:.1f}s "
        f"(λ=0.75 resumed from PR #28, wall {payload['opt_s_resumed']:.1f}s). "
        f"Branch `{payload['branch']}`. "
        "`qaoa_angles/`, `train69_ladder_fresh250/`, `settings.py`, and the phase-2 n=21 ladder were not written."
    )
    lines.append("")
    return "\n".join(lines) + "\n"


def _namespace(workers: int):
    class A:
        pass

    args = A()
    args.B_train = 128
    args.B_hold = 128
    args.B_transfer = 64
    args.B_n15 = 32
    args.seeds = [0, 1, 2]
    args.Ls = [16]
    args.rounds_L2 = 18
    args.rounds_L16 = 48
    args.max_iter_L2 = 8
    args.max_iter_L16 = 8
    args.lr = 0.35
    args.workers = int(workers)
    args.no_resume = False
    return args


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out",
        type=str,
        default=str(_ROOT / "Nov_3rd" / "runs" / "quantum_h_explore" / "unsat_lambda_mu1"),
    )
    ap.add_argument("--anchor", type=str, required=True)
    ap.add_argument("--lams", type=str, default=",".join(f"{lam:.2f}" for lam in LAM_GRID))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--skip-power", action="store_true")
    ap.add_argument("--ed-only", action="store_true")
    args_cli = ap.parse_args()
    lams = tuple(float(s) for s in args_cli.lams.split(",") if s != "")
    for lam in lams:
        if lam < 0:
            raise SystemExit(f"refusing negative λ={lam}")
        if abs(lam - 21) < 1e-9:
            raise SystemExit("refusing n=21")
    out = Path(args_cli.out)
    anchor = Path(args_cli.anchor)
    _refuse_n21(out, anchor)
    out.mkdir(parents=True, exist_ok=True)
    log(f"out={out}")
    t0 = time.time()
    _copy_anchor(anchor, out)
    q_cache = _build_caches(out, out, lams, args_cli.workers)
    (out / "quantum_cache.json").write_text(json.dumps(_jsonable(q_cache), indent=2) + "\n")
    if args_cli.ed_only:
        log(f"ED only, {time.time() - t0:.1f}s")
        return

    drv = _load_driver()
    hold_lam = {"value": float(ANCHOR_LAM)}

    def _patched_paths(path, L, method, seed):
        return _paths_for(path, L, method, seed, hold_lam["value"])

    drv._paths = _patched_paths
    drv.LAM = float(ANCHOR_LAM)
    drv.MU = float(FIXED_MU)
    js, pt = drv._paths(out, 16, "walsh", 0)
    if js.name != "lam0p75_L16_walsh_seed0.json" or not js.exists() or not pt.exists():
        raise RuntimeError(f"λ=0.75 resume path drifted: {js}")
    log(f"resume path ok {js.name}")

    torch.set_num_threads(4)
    args = _namespace(args_cli.workers)
    rows = {}
    audits = []
    scans = {}
    opt_new = 0.0
    opt_resumed = 0.0
    q_rows = {}
    full = [round(lam, 2) for lam in lams] == [round(lam, 2) for lam in LAM_GRID]
    for lam in lams:
        hold_lam["value"] = float(lam)
        drv.LAM = float(lam)
        drv.MU = float(FIXED_MU)
        log(f"=== λ={lam:.2f} μ={FIXED_MU:+.2f} ===")
        prepared, _sims, quantum = drv.prepare(args, out)
        for key in prepared:
            if int(key[0]) == 21:
                raise RuntimeError("n=21 entered the sweep")
        audit = drv.audit(prepared)
        audits.append({"lam": float(lam), **audit})
        train_ns = [sp.n for sp in (prepared[(n, "train")]["split"] for n in (6, 9))]
        if train_ns != [6, 9]:
            raise RuntimeError(f"train split drifted {train_ns}")
        methods = ["fourier_qaoa", "walsh"] if args_cli.skip_power else ["fourier_qaoa", "power_p", "walsh"]
        if methods == ["fourier_qaoa", "power_p", "walsh"]:
            runs = drv.train(prepared, args, out)
        else:
            from Nov_3rd.quantum_h_transfer import walsh_theta_from_fourier

            train_splits = [prepared[(n, "train")]["split"] for n in (6, 9)]
            hold_splits = [prepared[(n, "hold")]["split"] for n in (6, 9)]
            report = [
                prepared[(6, "hold")]["split"],
                prepared[(9, "hold")]["split"],
                prepared[(12, "transfer")]["split"],
                prepared[(15, "transfer")]["split"],
            ]
            runs = []
            fourier = {}
            for seed in args.seeds:
                rec = drv._fit("fourier_qaoa", 16, seed, train_splits, hold_splits, report, args, None, None, out)
                fourier[seed] = rec
                runs.append(rec)
            for seed in args.seeds:
                ref = {n: fourier[seed]["eval"][n]["med_eps"] for n in (6, 9)}
                theta0 = walsh_theta_from_fourier(fourier[seed]["theta"], 16)
                rec = drv._fit("walsh", 16, seed, train_splits, hold_splits, report, args, theta0, ref, out)
                drv._assert_warm("walsh", rec, ref)
                runs.append(rec)
        for rec in runs:
            if abs(float(rec.get("lam", lam)) - float(lam)) > 1e-9:
                raise RuntimeError(f"fit λ drift {rec.get('method')} seed {rec.get('seed')}")
            if abs(float(rec.get("mu", 99)) - FIXED_MU) > 1e-9:
                raise RuntimeError(f"fit μ drift {rec.get('method')} seed {rec.get('seed')}")
            if abs(float(lam) - ANCHOR_LAM) < 1e-12:
                opt_resumed += float(rec["wall_s"])
            else:
                opt_new += float(rec["wall_s"])
        cells = drv.build_cells(runs) if not args_cli.skip_power else _cells_without_power(drv, runs)
        qdec = drv.quantum_for_decide(quantum)
        reported = [(n, role, count) for n, role, count in SPECS if role != "train"]
        for n, role, count in reported:
            blob = np.load(_cache_path(out, n, role, count, lam))
            tail = gap_tail(blob["prod_gap"], blob["peak"])
            if abs(tail["prod_gap_med"] - float(qdec[n]["prod_gap_med"])) > 1e-9:
                raise RuntimeError(
                    f"gap median drift λ={lam:.2f} n={n}: cache {tail['prod_gap_med']} vs {qdec[n]['prod_gap_med']}"
                )
            if abs(tail["peak_med"] - float(qdec[n]["peak_med"])) > 1e-9:
                raise RuntimeError(
                    f"peak median drift λ={lam:.2f} n={n}: cache {tail['peak_med']} vs {qdec[n]['peak_med']}"
                )
            qdec[n]["frac_gap_below"] = tail["frac_gap_below"]
            qdec[n]["n_gap_below"] = tail["n_gap_below"]
            qdec[n]["n_inst"] = tail["n_inst"]
            qdec[n]["gap_min"] = tail["gap_min"]
            qdec[n]["peak_max"] = tail["peak_max"]
        row = _row_from_cells(cells, qdec)
        rows[lam_label(lam)] = row
        q_rows[lam_label(lam)] = {str(n): qdec[n] for n in (6, 9, 12, 15)}
        scan = {}
        for n, role in ((6, "hold"), (9, "hold"), (12, "transfer"), (15, "transfer")):
            scan[str(n)] = drv.cost_scan(prepared[(n, role)]["split"].circuit, n)
            log(f"cost λ={lam:.2f} n={n} γ={scan[str(n)]['gamma']:.4f} ε={scan[str(n)]['med_eps']:.4f}")
        scans[lam_label(lam)] = scan
        w12 = row["walsh"][12]
        log(f"λ={lam:.2f} walsh n12 R={w12['R']:.3f} med ε={w12['med']:.4f}")
        partial = {
            "rows": rows,
            "quantum_rows": q_rows,
            "cost_scan": scans,
            "audits": audits,
            "opt_s": opt_new,
            "opt_s_resumed": opt_resumed,
        }
        (out / "partial.json").write_text(json.dumps(_jsonable(partial), indent=2) + "\n")

    if not full:
        log("partial grid, not writing a verdict")
        return
    verdict = decide_lambda_sweep(rows)
    log(f"VERDICT {verdict['verdict']}: {verdict['reason']}")
    worst_e0 = max(float(a["worst_e0"]) for a in audits)
    worst_psi = max(float(a["worst_psi"]) for a in audits)
    payload = {
        "verdict": verdict,
        "rows": rows,
        "quantum_rows": q_rows,
        "cost_scan": scans,
        "audit": {"worst_e0": worst_e0, "worst_psi": worst_psi, "per_lam": audits},
        "lams": [float(lam) for lam in lams],
        "mu": FIXED_MU,
        "B_train": args.B_train,
        "B_hold": args.B_hold,
        "B_transfer": args.B_transfer,
        "B_n15": args.B_n15,
        "seeds": args.seeds,
        "rounds_L16": args.rounds_L16,
        "max_iter_L16": args.max_iter_L16,
        "lr": args.lr,
        "opt_s": opt_new,
        "opt_s_resumed": opt_resumed,
        "wall_s": time.time() - t0,
        "branch": "cursor/unsat-walsh-lambda-mu1-01fb",
        "published_anchor_R": PUBLISHED_ANCHOR_R,
        "skip_power": bool(args_cli.skip_power),
    }
    text = render(payload)
    (out / "RESULTS.md").write_text(text)
    (out / "summary.json").write_text(json.dumps(_jsonable(payload), indent=2) + "\n")
    (out / "verdict.json").write_text(json.dumps(_jsonable(verdict), indent=2) + "\n")
    log(f"wrote {out / 'RESULTS.md'} verdict={verdict['verdict']} wall={payload['wall_s']:.1f}s")


def _cells_without_power(drv, runs):
    cells = {}
    by = {}
    for rec in runs:
        by.setdefault((rec["method"], int(rec["L"])), []).append(rec)
    for (method, L), group in by.items():
        group = sorted(group, key=lambda r: int(r["seed"]))
        for n in (6, 9, 12, 15):
            got = drv._agg(group, n, "med_eps")
            cells[(method, L, n)] = {
                "method": method,
                "L": int(L),
                "n": int(n),
                "role": group[0]["eval"][n]["role"],
                "med": got["med"],
                "seed_std": got["std"],
                "per_seed": got["per_seed"],
                "seeds": got["seeds"],
                "n_params": int(group[0]["n_params"]),
            }
    for (method, L) in list(by):
        if method != "walsh":
            continue
        for n in (6, 9, 12, 15):
            q = cells[("fourier_qaoa", L, n)]
            cell = cells[(method, L, n)]
            q_by_seed = {int(s): float(v) for s, v in zip(q["seeds"], q["per_seed"])}
            ref = float(q["med"])
            cell["qaoa"] = ref
            cell["R"] = float(cell["med"] / ref) if abs(ref) > 1e-12 else float("nan")
            cell["mark"] = drv.mark_of(cell["R"])
            cell["seed_Rs"] = [
                float(med / q_by_seed[int(seed)]) if abs(q_by_seed[int(seed)]) > 1e-12 else float("nan")
                for seed, med in zip(cell["seeds"], cell["per_seed"])
            ]
    return cells


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
