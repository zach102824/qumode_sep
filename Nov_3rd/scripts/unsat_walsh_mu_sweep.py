#!/usr/bin/env python3
"""Sweep μ on the unsat Walsh champ. Same gate, seeds, and L=16 budget as PR #23.

    H = diag(unsat) − 0.75 Σ_i X_i + μ Σ_cav Σ_{i<j} X_i X_j
    μ ∈ {−0.5, −0.35, −0.25, −0.15, −0.05, 0}

Train n=6 and n=9 only. Freeze n=12 and n=15. ⟨H⟩ is the Z histogram plus the
X histogram (single-X and intra-cavity XX). μ=−0.25 reuses the PR #23 L=16
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

from Nov_3rd.unsat_mu_sweep import (
    ANCHOR_MU,
    MU_GRID,
    PUBLISHED_ANCHOR_R,
    cache_tag,
    decide_mu_sweep,
    file_tag,
    mu_label,
)
from Nov_3rd.unsat_rank2_xx import LAM, batch_ground


def log(*a):
    print(time.strftime("%H:%M:%S"), "[muSweep]", *a, flush=True)


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


def _paths_for(drv, out: Path, L: int, method: str, seed: int):
    tag = file_tag(float(drv.MU))
    stem = Path(out) / "runs" / f"mu{tag}_L{int(L)}_{method}_seed{int(seed)}"
    return Path(str(stem) + ".json"), Path(str(stem) + ".pt")


SPECS = (
    (6, "train", 128),
    (6, "hold", 128),
    (9, "train", 128),
    (9, "hold", 128),
    (12, "transfer", 64),
    (15, "transfer", 32),
)


def _cache_path(out: Path, n: int, role: str, count: int, mu: float) -> Path:
    return out / "cache" / f"unsat_xx_n{n}_{role}_B{count}_l0.75_{cache_tag(mu)}.npz"


def _copy_anchor(anchor: Path, out: Path):
    """PR #23 caches and L=16 checkpoints. Filenames for μ=−0.25 are unchanged."""
    for n, role, count in SPECS:
        src = anchor / "cache" / f"unsat_xx_n{n}_{role}_B{count}_l0.75_m-0.25.npz"
        dst = _cache_path(out, n, role, count, ANCHOR_MU)
        if not src.exists():
            raise FileNotFoundError(src)
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            shutil.copy2(src, dst)
            log(f"copied cache n={n} {role}")
    runs = out / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    for method in ("fourier_qaoa", "power_p", "walsh"):
        for seed in (0, 1, 2):
            for ext in (".json", ".pt"):
                name = f"mu-0p25_L16_{method}_seed{seed}{ext}"
                src = anchor / "runs" / name
                dst = runs / name
                if not src.exists():
                    raise FileNotFoundError(src)
                if not dst.exists():
                    shutil.copy2(src, dst)
    log("PR #23 μ=−0.25 L=16 checkpoints are in place")


def _build_caches(anchor: Path, out: Path, mus, workers: int):
    """Reuse the PR #23 unsat diagonals. Only E0 changes with μ."""
    anchors = {}
    for n, role, count in SPECS:
        src = anchor / "cache" / f"unsat_xx_n{n}_{role}_B{count}_l0.75_m-0.25.npz"
        z = np.load(src)
        anchors[(n, role)] = {k: z[k] for k in ("score", "pulse", "seeds", "m", "lam")}
        if int(anchors[(n, role)]["score"].shape[0]) != int(count):
            raise RuntimeError(f"anchor n={n} {role} B drift")
    quantum = {}
    for mu in mus:
        for n, role, count in SPECS:
            dst = _cache_path(out, n, role, count, mu)
            if dst.exists():
                blob = np.load(dst)
                if abs(float(blob["mu"]) - float(mu)) > 1e-12:
                    raise RuntimeError(f"cache μ drift {dst.name}")
                quantum[(mu_label(mu), n, role)] = {
                    "peak_med": float(np.median(blob["peak"])),
                    "entropy_med": float(np.median(blob["entropy"])),
                    "prod_gap_med": float(np.median(blob["prod_gap"])),
                    "residual_max": float(np.max(blob["residual"])),
                }
                continue
            base = anchors[(n, role)]
            if abs(float(base["lam"]) - LAM) > 1e-12:
                raise RuntimeError("anchor λ drift")
            t0 = time.time()
            log(f"ED n={n} {role} B={count} μ={mu:+.2f}")
            gu = batch_ground(
                base["score"], n, LAM, float(mu), workers, seeds=base["seeds"]
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
                lam=np.asarray(LAM, dtype=np.float64),
                mu=np.asarray(float(mu), dtype=np.float64),
            )
            quantum[(mu_label(mu), n, role)] = {
                "peak_med": float(np.median(gu["peak"])),
                "entropy_med": float(np.median(gu["entropy"])),
                "prod_gap_med": float(np.median(gu["prod_gap"])),
                "residual_max": float(np.max(gu["residual"])),
            }
            log(
                f"  n={n} {role} μ={mu:+.2f} in {time.time() - t0:.1f}s "
                f"peak={quantum[(mu_label(mu), n, role)]['peak_med']:.3f} "
                f"gap={quantum[(mu_label(mu), n, role)]['prod_gap_med']:.3f}"
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
            if "published_R" in cell:
                packed["published_R"] = float(cell["published_R"])
            row[key][n] = packed
    return row


def _fmt(x, nd=3):
    if x is None:
        return "—"
    return f"{float(x):.{nd}f}"


def render(payload: dict) -> str:
    v = payload["verdict"]
    lines = []
    lines.append("# RESULTS — unsat Walsh μ sweep, L=16 freeze n=12")
    lines.append("")
    lines.append(f"**Verdict: {v['verdict']}**")
    lines.append("")
    lines.append(v["reason"])
    lines.append("")
    lines.append("## Setup")
    lines.append("")
    lines.append(
        "Same gate as PR #23. The intra-cavity XX term is scored from the X histogram. "
        "It is not compiled into U_c. Angles for n=12 and n=15 were not in the loss. "
        "n=21 was not loaded."
    )
    lines.append("")
    lines.append("```text")
    lines.append("H = diag(unsat) − 0.75 Σ_i X_i + μ Σ_cav Σ_{i<j} X_i X_j")
    lines.append("μ ∈ {−0.50, −0.35, −0.25, −0.15, −0.05, 0}")
    lines.append("U_c^X = W exp(−i γ C_n) W ,  W = H^{⊗3} ,  C_n = (n/m) unsat − max")
    lines.append("⟨H⟩ = Σ_z p_Z(z) unsat(z) − 0.75 Σ_i ⟨X_i⟩ + μ Σ_cav Σ_{i<j} ⟨X_i X_j⟩")
    lines.append("ε = (⟨H⟩ − E_0) / n")
    lines.append("```")
    lines.append("")
    lines.append(
        f"Train B={payload['B_train']} per n from the PR #23 unsat diagonals "
        f"(those packs are base 51e9 / scale1000 base 54e9). "
        f"Holdout B={payload['B_hold']}. Freeze n=12 B={payload['B_transfer']}, "
        f"n=15 B={payload['B_n15']}. Seeds {payload['seeds']}. L=16 only. "
        f"L-BFGS rounds {payload['rounds_L16']}, max_iter {payload['max_iter_L16']}, "
        f"lr={payload['lr']}. Early stop is 8 rounds without a 10⁻⁶ improvement. "
        "That is the PR #23 L=16 budget. Fourier stores 2L Zhou coefficients. "
        "power_p stores 4L+1. Walsh stores 7L+1 and copies the scalar γ onto degrees 1–4 "
        "at the warm start. The loss is the mean of the per-n mean residuals on n=6 and n=9. "
        "The checkpoint minimizes the worst of the two holdout ratios against that seed's Fourier. "
        "μ=−0.25 is the PR #23 checkpoint, not a retrain."
    )
    lines.append("")
    lines.append(
        "R = (median over seeds of median ε) / (the same number for Fourier QAOA on this μ). "
        "BEAT if R≤0.98, TIE if R≤1.02, else UNDER. "
        "A μ beats −0.25 clearly when both the ratio of medians and the median of the three "
        "per-seed ratios drop by at least 0.02. Absolute residuals are not compared across μ."
    )
    lines.append("")
    lines.append("## Quantumness")
    lines.append("")
    lines.append(
        "E0 is Lanczos on the sparse Hamiltonian. A split is a bitstring when the median "
        "ground-state peak is at least 0.5. The product gap is (E_prod − E_0) / |E_0|."
    )
    lines.append("")
    if payload.get("audit"):
        lines.append(
            f"Worst |ΔE0| Lanczos vs dense over the grid = {payload['audit']['worst_e0']:.2e}. "
            f"Histogram vs ⟨ψ|H|ψ⟩ = {payload['audit']['worst_psi']:.2e}."
        )
        lines.append("")
    lines.append("| μ | n | role | peak Med | prod gap Med |")
    lines.append("|--:|--:|:-----|---------:|-------------:|")
    role_of = {6: "hold", 9: "hold", 12: "transfer", 15: "transfer"}
    for mu in payload["mus"]:
        block = payload["quantum_rows"][mu_label(mu)]
        for n in (6, 9, 12, 15):
            q = block[str(n)]
            lines.append(
                f"| {mu:+.2f} | {n} | {role_of[n]} | {q['peak_med']:.3f} | {q['prod_gap_med']:.3f} |"
            )
    lines.append("")
    lines.append("## Freeze table")
    lines.append("")
    lines.append(
        "n=6 and n=9 are the selection holdout. n=12 and n=15 are not. "
        "power_p is the scalar-γ control and does not choose μ."
    )
    lines.append("")
    lines.append("| μ | head | n=6 R | n=9 R | n=12 R | n=12 mark | n=15 R | n=12 Med ε |")
    lines.append("|--:|:-----|------:|------:|-------:|:----------|-------:|-----------:|")
    for mu in payload["mus"]:
        row = payload["rows"][mu_label(mu)]
        for head in ("walsh", "power_p"):
            cells = row[head]
            if 12 not in cells:
                continue
            def r(n):
                cell = cells.get(n)
                if cell is None or cell.get("R") is None:
                    return "—"
                return f"{cell['R']:.3f}"
            mark = cells[12].get("mark") or ""
            lines.append(
                f"| {mu:+.2f} | {head} | {r(6)} | {r(9)} | {r(12)} | {mark} | {r(15)} | "
                f"{cells[12]['med']:.4f} |"
            )
    lines.append("")
    lines.append("Per-seed n=12 Walsh R, seeds [0, 1, 2]:")
    lines.append("")
    for mu in payload["mus"]:
        cell = payload["rows"][mu_label(mu)]["walsh"][12]
        rs = ", ".join(f"{x:.3f}" for x in cell["seed_Rs"])
        med = float(np.median(cell["seed_Rs"]))
        lines.append(
            f"- μ={mu:+.2f}: {rs} (median of those R = {med:.3f}); "
            f"ratio of medians = {cell['R']:.3f}"
        )
    lines.append("")
    lines.append("## One-parameter cost pulse")
    lines.append("")
    lines.append(
        "SNAP phase 0. XX is not in that pulse. The scan is the minimum median ε over γ ∈ [−2, 2]."
    )
    lines.append("")
    lines.append("| μ | n | γ | Med ε | Walsh L=16 Med ε |")
    lines.append("|--:|--:|--:|------:|------------------:|")
    for mu in payload["mus"]:
        for n in (6, 9, 12, 15):
            scan = payload["cost_scan"][mu_label(mu)][str(n)]
            w = payload["rows"][mu_label(mu)]["walsh"][n]["med"]
            lines.append(
                f"| {mu:+.2f} | {n} | {scan['gamma']:.4f} | {scan['med_eps']:.4f} | {w:.4f} |"
            )
    lines.append("")
    lines.append("## Rule")
    lines.append("")
    lines.append(
        "Operating μ is the quantum grid point with the lowest frozen n=12 L=16 Walsh ratio, "
        "provided that ratio is at least 0.02 below the in-run μ=−0.25 checkpoint on both "
        "the ratio of medians and the median per-seed ratio, R≤0.98, at least two seeds beat "
        "their own Fourier, n=12 peak < 0.5 and product gap ≥ 0.02, and n=15 R stays below 1.10 "
        "with a non-bitstring ground state. Otherwise μ=−0.25 stays."
    )
    lines.append("")
    lines.append(
        "PASS if that switch happens, or if the sweep shows −0.25 is still optimal. "
        "needs_edit if the grid or the PR #23 anchor is incomplete, or a clearly lower μ "
        "fails a transfer gate. KILL if the operating point's ground state (n=6, 9, or 12) "
        "is a bitstring, Walsh there loses to Fourier, or the L=16 advantage reverses. "
        "power_p does not gate the call."
    )
    lines.append("")
    lines.append(
        f"Optimizer time summed over new fits {payload['opt_s']:.1f}s "
        f"(μ=−0.25 resumed, wall {payload['opt_s_resumed']:.1f}s). "
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
        default=str(_ROOT / "Nov_3rd" / "runs" / "quantum_h_explore" / "unsat_mu_sweep"),
    )
    ap.add_argument(
        "--anchor",
        type=str,
        default=str(_ROOT / "Nov_3rd" / "runs" / "quantum_h_explore" / "unsat_rank2_mu"),
    )
    ap.add_argument("--mus", type=str, default=",".join(f"{m:.2f}" for m in MU_GRID))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--skip-power", action="store_true")
    ap.add_argument("--ed-only", action="store_true")
    args_cli = ap.parse_args()
    mus = tuple(float(s) for s in args_cli.mus.split(",") if s != "")
    if [round(m, 2) for m in mus] != [round(m, 2) for m in MU_GRID] and not args_cli.ed_only:
        # A partial grid can build caches. The verdict requires the full grid.
        log(f"partial grid {mus}")
    for m in mus:
        if abs(m - 21) < 1e-9:
            raise SystemExit("refusing n=21")
    out = Path(args_cli.out)
    anchor = Path(args_cli.anchor)
    out.mkdir(parents=True, exist_ok=True)
    log(f"out={out}")
    t0 = time.time()
    _copy_anchor(anchor, out)
    q_cache = _build_caches(anchor, out, mus, args_cli.workers)
    (out / "quantum_cache.json").write_text(json.dumps(_jsonable(q_cache), indent=2) + "\n")
    if args_cli.ed_only:
        log(f"ED only, {time.time() - t0:.1f}s")
        return

    drv = _load_driver()

    def _patched_paths(path, L, method, seed):
        return _paths_for(drv, path, L, method, seed)

    drv._paths = _patched_paths
    drv.MU = ANCHOR_MU
    js, pt = drv._paths(out, 16, "walsh", 0)
    if js.name != "mu-0p25_L16_walsh_seed0.json" or not js.exists() or not pt.exists():
        raise RuntimeError(f"μ=−0.25 resume path drifted: {js}")
    log(f"resume path ok {js.name}")

    torch.set_num_threads(4)
    args = _namespace(args_cli.workers)
    rows = {}
    audits = []
    scans = {}
    opt_new = 0.0
    opt_resumed = 0.0
    q_rows = {}
    for mu in mus:
        drv.MU = float(mu)
        log(f"=== μ={mu:+.2f} ===")
        prepared, _sims, quantum = drv.prepare(args, out)
        audit = drv.audit(prepared)
        audits.append({"mu": float(mu), **audit})
        # Confirm transfer n never enters the loss batches.
        train_ns = [sp.n for sp in (prepared[(n, "train")]["split"] for n in (6, 9))]
        if train_ns != [6, 9]:
            raise RuntimeError(f"train split drifted {train_ns}")
        runs = []
        methods = ["fourier_qaoa", "walsh"] if args_cli.skip_power else ["fourier_qaoa", "power_p", "walsh"]
        # Train through the PR #23 driver so the budget, warm start, and early stop match.
        # power_p is between Fourier and Walsh there; skipping it does not change Walsh's warm start.
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
            # Resumed PR #23 fits already paid their wall in that run. Count them apart.
            if abs(float(mu) - ANCHOR_MU) < 1e-12:
                opt_resumed += float(rec["wall_s"])
            else:
                opt_new += float(rec["wall_s"])
        if args_cli.skip_power:
            cells = _cells_without_power(drv, runs)
        else:
            cells = drv.build_cells(runs)
        qdec = drv.quantum_for_decide(quantum)
        row = _row_from_cells(cells, qdec)
        rows[mu_label(mu)] = row
        q_rows[mu_label(mu)] = {str(n): qdec[n] for n in (6, 9, 12, 15)}
        scan = {}
        for n, role in ((6, "hold"), (9, "hold"), (12, "transfer"), (15, "transfer")):
            scan[str(n)] = drv.cost_scan(prepared[(n, role)]["split"].circuit, n)
            log(f"cost μ={mu:+.2f} n={n} γ={scan[str(n)]['gamma']:.4f} ε={scan[str(n)]['med_eps']:.4f}")
        scans[mu_label(mu)] = scan
        w12 = row["walsh"][12]
        log(f"μ={mu:+.2f} walsh n12 R={w12['R']:.3f} med ε={w12['med']:.4f}")
        partial = {
            "rows": rows,
            "quantum_rows": q_rows,
            "cost_scan": scans,
            "audits": audits,
            "opt_s": opt_new,
            "opt_s_resumed": opt_resumed,
        }
        (out / "partial.json").write_text(json.dumps(_jsonable(partial), indent=2) + "\n")

    if [round(m, 2) for m in mus] != [round(m, 2) for m in MU_GRID]:
        log("partial grid, not writing a verdict")
        return
    verdict = decide_mu_sweep(rows)
    log(f"VERDICT {verdict['verdict']}: {verdict['reason']}")
    worst_e0 = max(float(a["worst_e0"]) for a in audits)
    worst_psi = max(float(a["worst_psi"]) for a in audits)
    payload = {
        "verdict": verdict,
        "rows": rows,
        "quantum_rows": q_rows,
        "cost_scan": scans,
        "audit": {"worst_e0": worst_e0, "worst_psi": worst_psi, "per_mu": audits},
        "mus": [float(m) for m in mus],
        "B_train": args.B_train,
        "B_hold": args.B_hold,
        "B_transfer": args.B_transfer,
        "B_n15": args.B_n15,
        "seeds": args.seeds,
        "rounds_L16": args.rounds_L16,
        "max_iter_L16": args.max_iter_L16,
        "lr": args.lr,
        "lam": LAM,
        "opt_s": opt_new,
        "opt_s_resumed": opt_resumed,
        "wall_s": time.time() - t0,
        "branch": "cursor/unsat-walsh-mu-sweep-e2e2",
        "published_anchor_R": PUBLISHED_ANCHOR_R,
        "skip_power": bool(args_cli.skip_power),
    }
    text = render(payload)
    (out / "RESULTS.md").write_text(text)
    (out / "summary.json").write_text(json.dumps(_jsonable(payload), indent=2) + "\n")
    (out / "verdict.json").write_text(json.dumps(_jsonable(verdict), indent=2) + "\n")
    log(f"wrote {out / 'RESULTS.md'} verdict={verdict['verdict']} wall={payload['wall_s']:.1f}s")


def _cells_without_power(drv, runs):
    """build_cells expects power_p. Aggregate Fourier and Walsh only."""
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
