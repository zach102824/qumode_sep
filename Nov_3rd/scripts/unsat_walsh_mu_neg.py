#!/usr/bin/env python3
"""Extend the unsat Walsh μ sweep below the PR #26 edge.

    H = diag(unsat) − 0.75 Σ_i X_i + μ Σ_cav Σ_{i<j} X_i X_j
    μ ∈ {−1.00, −0.85, −0.75, −0.65, −0.50}

Same gate, seeds, and L=16 budget as PR #26. Train n=6 and n=9 only.
Freeze n=12 and n=15. ⟨H⟩ is the Z histogram plus the X histogram.
μ=−0.50 reuses the PR #26 L=16 checkpoints. n=21 is not loaded.
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

from Nov_3rd.unsat_mu_neg import (
    ANCHOR_MU,
    MU_GRID,
    PUBLISHED_ANCHOR_R,
    cache_tag,
    decide_mu_neg,
    file_tag,
    mu_label,
)

BRANCH = "cursor/unsat-walsh-mu-neg-b37d"


def log(*a):
    print(time.strftime("%H:%M:%S"), "[muNeg]", *a, flush=True)


def _load_sweep():
    path = Path(__file__).resolve().parent / "unsat_walsh_mu_sweep.py"
    spec = importlib.util.spec_from_file_location("unsat_walsh_mu_sweep", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _same_grid(mus, grid) -> bool:
    if len(mus) != len(grid):
        return False
    return all(abs(float(a) - float(b)) < 5e-4 for a, b in zip(mus, grid))


def _copy_edge(sw, anchor: Path, out: Path):
    """PR #26 caches and L=16 checkpoints at μ=−0.50. Not a retrain."""
    for n, role, count in sw.SPECS:
        src = anchor / "cache" / f"unsat_xx_n{n}_{role}_B{count}_l0.75_{cache_tag(ANCHOR_MU)}.npz"
        dst = sw._cache_path(out, n, role, count, ANCHOR_MU)
        if not src.exists():
            raise FileNotFoundError(src)
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            shutil.copy2(src, dst)
            log(f"copied cache n={n} {role}")
    runs = out / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    tag = file_tag(ANCHOR_MU)
    for method in ("fourier_qaoa", "power_p", "walsh"):
        for seed in (0, 1, 2):
            for ext in (".json", ".pt"):
                name = f"mu{tag}_L16_{method}_seed{seed}{ext}"
                src = anchor / "runs" / name
                dst = runs / name
                if not src.exists():
                    raise FileNotFoundError(src)
                if not dst.exists():
                    shutil.copy2(src, dst)
    log("PR #26 μ=−0.50 L=16 checkpoints are in place")


def _gap_tails(out: Path) -> dict:
    """n=12 transfer product-gap tails. The gate uses the median; the tail is reported."""
    tails = {}
    for mu in MU_GRID:
        path = out / "cache" / f"unsat_xx_n12_transfer_B64_l0.75_{cache_tag(mu)}.npz"
        z = np.load(path)
        gap = np.asarray(z["prod_gap"], dtype=np.float64)
        peak = np.asarray(z["peak"], dtype=np.float64)
        tails[mu_label(mu)] = {
            "gap_min": float(gap.min()),
            "n_gap_below": int(np.sum(gap < 0.02)),
            "n_inst": int(gap.shape[0]),
            "peak_max": float(peak.max()),
        }
    return tails


def _at(block, n):
    if n in block:
        return block[n]
    return block[str(n)]


def render(payload: dict) -> str:
    v = payload["verdict"]
    lines = []
    lines.append("# RESULTS — unsat Walsh μ sweep below −0.50, L=16 freeze n=12")
    lines.append("")
    lines.append(f"**Verdict: {v['verdict']}**")
    lines.append("")
    lines.append(v["reason"])
    lines.append("")
    lines.append("## Setup")
    lines.append("")
    lines.append(
        "Same gate as PR #26. The intra-cavity XX term is scored from the X histogram. "
        "It is not compiled into U_c. Angles for n=12 and n=15 were not in the loss. "
        "n=21 was not loaded."
    )
    lines.append("")
    lines.append("```text")
    lines.append("H = diag(unsat) − 0.75 Σ_i X_i + μ Σ_cav Σ_{i<j} X_i X_j")
    lines.append("μ ∈ {−1.00, −0.85, −0.75, −0.65, −0.50}")
    lines.append("U_c^X = W exp(−i γ C_n) W ,  W = H^{⊗3} ,  C_n = (n/m) unsat − max")
    lines.append("⟨H⟩ = Σ_z p_Z(z) unsat(z) − 0.75 Σ_i ⟨X_i⟩ + μ Σ_cav Σ_{i<j} ⟨X_i X_j⟩")
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
        "μ=−0.50 is the PR #26 checkpoint, not a retrain."
    )
    lines.append("")
    lines.append(
        "R = (median over seeds of median ε) / (the same number for Fourier QAOA on this μ). "
        "BEAT if R≤0.98, TIE if R≤1.02, else UNDER. "
        "A μ beats −0.50 clearly when both the ratio of medians and the median of the three "
        "per-seed ratios drop by at least 0.02. Absolute residuals are not compared across μ."
    )
    lines.append("")
    lines.append("## Quantumness")
    lines.append("")
    lines.append(
        "E0 is Lanczos on the sparse Hamiltonian. A split is a bitstring when the median "
        "ground-state peak is at least 0.5. The product gap is (E_prod − E_0) / |E_0|. "
        "At μ=−0.50 the n=12 product gap on the PR #26 holdout was 0.065."
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
            if "12" not in cells and 12 not in cells:
                continue

            def r(n, cells=cells):
                try:
                    cell = _at(cells, n)
                except KeyError:
                    return "—"
                if cell.get("R") is None:
                    return "—"
                return f"{cell['R']:.3f}"

            mark = _at(cells, 12).get("mark") or ""
            lines.append(
                f"| {mu:+.2f} | {head} | {r(6)} | {r(9)} | {r(12)} | {mark} | {r(15)} | "
                f"{_at(cells, 12)['med']:.4f} |"
            )
    lines.append("")
    lines.append("Per-seed n=12 Walsh R, seeds [0, 1, 2]:")
    lines.append("")
    for mu in payload["mus"]:
        cell = _at(payload["rows"][mu_label(mu)]["walsh"], 12)
        rs = ", ".join(f"{x:.3f}" for x in cell["seed_Rs"])
        med = float(np.median(cell["seed_Rs"]))
        lines.append(
            f"- μ={mu:+.2f}: {rs} (median of those R = {med:.3f}); "
            f"ratio of medians = {cell['R']:.3f}"
        )
    lines.append("")
    curve = [float(_at(payload["rows"][mu_label(mu)]["walsh"], 12)["R"]) for mu in payload["mus"]]
    rising = all(curve[i] < curve[i + 1] - 1e-6 for i in range(len(curve) - 1))
    try:
        pp_vals = [float(_at(payload["rows"][mu_label(mu)]["power_p"], 12)["R"]) for mu in payload["mus"]]
        pp_above = all(p > w for p, w in zip(pp_vals, curve))
    except KeyError:
        pp_vals = None
        pp_above = False
    anchor_R = float(_at(payload["rows"][mu_label(ANCHOR_MU)]["walsh"], 12)["R"])
    clear_mus = [
        (float(mu), r)
        for mu, r in zip(payload["mus"], curve)
        if abs(float(mu) - ANCHOR_MU) > 1e-9 and anchor_R - r >= 0.02 - 1e-12
    ]
    lines.append("## Reading")
    lines.append("")
    listed = ", ".join(f"{r:.3f}" for r in curve)
    lines.append(
        f"Frozen n=12 Walsh R is {listed} as μ runs from {float(payload['mus'][0]):+.2f} "
        f"to {float(payload['mus'][-1]):+.2f}."
    )
    if rising:
        lines.append(
            "The ratio rises at every step toward μ=−0.50, so the minimum on this grid "
            "is the negative edge."
        )
    else:
        best_i = int(np.argmin(curve))
        lines.append(
            f"The lowest ratio on the grid is μ={float(payload['mus'][best_i]):+.2f} "
            f"at {curve[best_i]:.3f}."
        )
    if clear_mus:
        lines.append(
            "Grid points at least 0.02 below μ=−0.50 on the ratio of medians: "
            + ", ".join(f"{m:+.2f} at {r:.3f}" for m, r in clear_mus)
            + "."
        )
    else:
        lines.append("No grid point is 0.02 below μ=−0.50 on the ratio of medians.")
    if v.get("further_edge"):
        lines.append(
            "μ=−1.00 still wins and stays quantum, so this is a further edge. "
            "A μ below −1.00 was not trained."
        )
    if v.get("quantumness_break"):
        lines.append(
            "At least one more-negative μ fails the quantumness bar "
            "(bitstring peak ≥ 0.5, or n=12 product gap < 0.02) and does not replace μ=−0.50."
        )
    lines.append("")
    if pp_vals is not None and pp_above:
        lines.append(
            "power_p stays above Walsh at every μ. The μ move is not confined to the Walsh head."
        )
        pp_best = int(np.argmin(pp_vals))
        w_best = int(np.argmin(curve))
        if abs(float(payload["mus"][pp_best]) - float(payload["mus"][w_best])) > 1e-9:
            lines.append(
                f"power_p's lowest n=12 R on this grid is μ={float(payload['mus'][pp_best]):+.2f} "
                f"at {pp_vals[pp_best]:.3f}, so it does not pick the same point as Walsh."
            )
        lines.append("")
    gaps = []
    for mu in payload["mus"]:
        q = payload["quantum_rows"][mu_label(mu)]["12"]
        gaps.append(f"{float(mu):+.2f}:{q['prod_gap_med']:.3f}")
    lines.append("n=12 product gap by μ: " + ", ".join(gaps) + ".")
    tails = payload.get("gap_tails") or {}
    if tails:
        bits = []
        for mu in payload["mus"]:
            t = tails[mu_label(mu)]
            bits.append(
                f"{float(mu):+.2f} min {t['gap_min']:.3f} "
                f"({t['n_gap_below']}/{t['n_inst']} under 0.02, peak max {t['peak_max']:.3f})"
            )
        lines.append("n=12 product-gap minimum: " + "; ".join(bits) + ".")
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
            w = _at(payload["rows"][mu_label(mu)]["walsh"], n)["med"]
            lines.append(
                f"| {mu:+.2f} | {n} | {scan['gamma']:.4f} | {scan['med_eps']:.4f} | {w:.4f} |"
            )
    lines.append("")
    lines.append("## Rule")
    lines.append("")
    lines.append(
        "Operating μ is the quantum grid point with the lowest frozen n=12 L=16 Walsh ratio, "
        "provided that ratio is at least 0.02 below the in-run μ=−0.50 checkpoint on both "
        "the ratio of medians and the median per-seed ratio, R≤0.98, at least two seeds beat "
        "their own Fourier, n=12 peak < 0.5 and product gap ≥ 0.02, and n=15 R stays below 1.10 "
        "with a non-bitstring ground state. Otherwise μ=−0.50 stays."
    )
    lines.append("")
    lines.append(
        "PASS if a μ beats −0.50 under that rule, or if the sweep shows −0.50 is still optimal, "
        "including when a lower ratio is a quantumness break. "
        "needs_edit if the grid or the PR #26 anchor is incomplete, or a clearly lower μ "
        "fails a transfer gate. KILL if the operating point's ground state (n=6, 9, or 12) "
        "is a bitstring, Walsh there loses to Fourier, or the L=16 advantage reverses. "
        "power_p does not gate the call. If μ=−1.00 wins, say so: the grid is still on a further edge."
    )
    lines.append("")
    lines.append(
        f"Optimizer time summed over new fits {payload['opt_s']:.1f}s "
        f"(μ=−0.50 resumed, wall {payload['opt_s_resumed']:.1f}s). "
        f"Branch `{payload['branch']}`. "
        "`qaoa_angles/`, `train69_ladder_fresh250/`, `settings.py`, and the phase-2 n=21 ladder were not written."
    )
    lines.append("")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out",
        type=str,
        default=str(_ROOT / "Nov_3rd" / "runs" / "quantum_h_explore" / "unsat_mu_neg"),
    )
    ap.add_argument(
        "--anchor",
        type=str,
        default=str(_ROOT / "Nov_3rd" / "runs" / "quantum_h_explore" / "unsat_mu_sweep"),
    )
    ap.add_argument("--mus", type=str, default=",".join(f"{m:.2f}" for m in MU_GRID))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--skip-power", action="store_true")
    ap.add_argument("--ed-only", action="store_true")
    args_cli = ap.parse_args()
    mus = tuple(float(s) for s in args_cli.mus.split(",") if s != "")
    if any(abs(float(m) - 21.0) < 1e-6 for m in mus):
        raise SystemExit("refusing n=21")
    full = _same_grid(mus, MU_GRID)
    if not full and not args_cli.ed_only:
        log(f"partial grid {mus}")
    out = Path(args_cli.out)
    anchor = Path(args_cli.anchor)
    if "n21" in str(out) or "n21" in str(anchor):
        raise SystemExit("refusing an n=21 path")
    out.mkdir(parents=True, exist_ok=True)
    log(f"out={out}")
    t0 = time.time()
    sw = _load_sweep()
    _copy_edge(sw, anchor, out)
    q_cache = sw._build_caches(anchor, out, mus, args_cli.workers, diag_tag=cache_tag(ANCHOR_MU))
    (out / "quantum_cache.json").write_text(json.dumps(sw._jsonable(q_cache), indent=2) + "\n")
    if args_cli.ed_only:
        log(f"ED only, {time.time() - t0:.1f}s")
        return

    drv = sw._load_driver()

    def _patched_paths(path, L, method, seed):
        return sw._paths_for(drv, path, L, method, seed)

    drv._paths = _patched_paths
    drv.MU = ANCHOR_MU
    js, pt = drv._paths(out, 16, "walsh", 0)
    if js.name != "mu-0p50_L16_walsh_seed0.json" or not js.exists() or not pt.exists():
        raise RuntimeError(f"μ=−0.50 resume path drifted: {js}")
    log(f"resume path ok {js.name}")

    torch.set_num_threads(int(args_cli.threads))
    args = sw._namespace(args_cli.workers)
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
        if any(int(n) == 21 for n, _role in prepared):
            raise RuntimeError("n=21 entered prepare")
        audit = drv.audit(prepared)
        audits.append({"mu": float(mu), **audit})
        train_ns = [sp.n for sp in (prepared[(n, "train")]["split"] for n in (6, 9))]
        if train_ns != [6, 9]:
            raise RuntimeError(f"train split drifted {train_ns}")
        runs = []
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
            if abs(float(mu) - ANCHOR_MU) < 1e-12:
                opt_resumed += float(rec["wall_s"])
            else:
                opt_new += float(rec["wall_s"])
        if args_cli.skip_power:
            cells = sw._cells_without_power(drv, runs)
        else:
            cells = drv.build_cells(runs)
        qdec = drv.quantum_for_decide(quantum)
        row = sw._row_from_cells(cells, qdec)
        rows[mu_label(mu)] = row
        q_rows[mu_label(mu)] = {str(n): qdec[n] for n in (6, 9, 12, 15)}
        scan = {}
        for n, role in ((6, "hold"), (9, "hold"), (12, "transfer"), (15, "transfer")):
            scan[str(n)] = drv.cost_scan(prepared[(n, role)]["split"].circuit, n)
            log(f"cost μ={mu:+.2f} n={n} γ={scan[str(n)]['gamma']:.4f} ε={scan[str(n)]['med_eps']:.4f}")
        scans[mu_label(mu)] = scan
        w12 = row["walsh"][12]
        gap12 = qdec[12]["prod_gap_med"]
        peak12 = qdec[12]["peak_med"]
        log(
            f"μ={mu:+.2f} walsh n12 R={w12['R']:.3f} med ε={w12['med']:.4f} "
            f"peak={peak12:.3f} gap={gap12:.3f}"
        )
        partial = {
            "rows": rows,
            "quantum_rows": q_rows,
            "cost_scan": scans,
            "audits": audits,
            "opt_s": opt_new,
            "opt_s_resumed": opt_resumed,
            "mus": [float(m) for m in mus],
        }
        tag = file_tag(mus[0]) if not full else "full"
        (out / f"partial_{tag}.json").write_text(json.dumps(sw._jsonable(partial), indent=2) + "\n")

    if not full:
        log(f"partial grid, not writing a verdict. wall={time.time() - t0:.1f}s")
        return
    verdict = decide_mu_neg(rows)
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
        "lam": 0.75,
        "opt_s": opt_new,
        "opt_s_resumed": opt_resumed,
        "wall_s": time.time() - t0,
        "branch": BRANCH,
        "published_anchor_R": PUBLISHED_ANCHOR_R,
        "skip_power": bool(args_cli.skip_power),
        "gap_tails": _gap_tails(out),
    }
    text = render(payload)
    (out / "RESULTS.md").write_text(text)
    (out / "summary.json").write_text(json.dumps(sw._jsonable(payload), indent=2) + "\n")
    (out / "verdict.json").write_text(json.dumps(sw._jsonable(verdict), indent=2) + "\n")
    log(f"wrote {out / 'RESULTS.md'} verdict={verdict['verdict']} wall={payload['wall_s']:.1f}s")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
