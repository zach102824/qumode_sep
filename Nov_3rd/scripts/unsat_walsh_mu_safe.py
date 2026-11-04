#!/usr/bin/env python3
"""Fine μ grid for a quantumness-safe unsat Walsh freeze.

    H = diag(unsat) − 0.75 Σ_i X_i + μ Σ_cav Σ_{i<j} X_i X_j
    μ ∈ {−1.00, −0.95, −0.90, −0.85, −0.80, −0.75, −0.60, −0.50}

Same gate, seeds, and L=16 budget as PR #26. Train n=6 and n=9 only.
Freeze n=12 (and report n=15). λ=0.75. μ ∈ {−1.00, −0.85, −0.75, −0.50}
reuse the PR #28 checkpoints. n=21 is not loaded.

SAFE minimizes frozen n=12 Walsh R subject to median product gap ≥ 0.05 and
≥95% of n=12 instances with gap ≥ 0.02. EDGE is the unconstrained argmin of R.
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

from Nov_3rd.unsat_mu_safe import (
    ANCHOR_MU,
    MU_GRID,
    RESUME_MUS,
    cache_tag,
    decide_mu_safe,
    file_tag,
    gap_stats,
    mu_label,
)

BRANCH = "cursor/unsat-walsh-mu-safe-fa2b"
METHODS = ("fourier_qaoa", "power_p", "walsh")


def log(*a):
    print(time.strftime("%H:%M:%S"), "[muSafe]", *a, flush=True)


def _load_sweep():
    path = Path(__file__).resolve().parent / "unsat_walsh_mu_sweep.py"
    spec = importlib.util.spec_from_file_location("unsat_walsh_mu_sweep", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _is_resume(mu: float) -> bool:
    return any(abs(float(mu) - float(known)) < 1e-9 for known in RESUME_MUS)


def _copy_resumed(sw, resume: Path, out: Path, mus):
    """PR #26 / PR #28 caches and L=16 checkpoints. Not a retrain."""
    copied = 0
    for mu in mus:
        if not _is_resume(mu):
            continue
        for n, role, count in sw.SPECS:
            src = resume / "cache" / f"unsat_xx_n{n}_{role}_B{count}_l0.75_{cache_tag(mu)}.npz"
            dst = sw._cache_path(out, n, role, count, mu)
            if not src.exists():
                raise FileNotFoundError(src)
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists():
                shutil.copy2(src, dst)
                copied += 1
        runs = out / "runs"
        runs.mkdir(parents=True, exist_ok=True)
        tag = file_tag(mu)
        for method in METHODS:
            for seed in (0, 1, 2):
                for ext in (".json", ".pt"):
                    name = f"mu{tag}_L16_{method}_seed{seed}{ext}"
                    src = resume / "runs" / name
                    dst = runs / name
                    if not src.exists():
                        raise FileNotFoundError(src)
                    if not dst.exists():
                        shutil.copy2(src, dst)
                        copied += 1
    log(f"resumed checkpoints copied ({copied} new files)")


def _gap_tails(out: Path, mus) -> dict:
    tails = {}
    for mu in mus:
        path = out / "cache" / f"unsat_xx_n12_transfer_B64_l0.75_{cache_tag(mu)}.npz"
        gap = np.asarray(np.load(path)["prod_gap"], dtype=np.float64)
        peak = np.asarray(np.load(path)["peak"], dtype=np.float64)
        stats = gap_stats(gap)
        stats["peak_max"] = float(peak.max())
        stats["peak_med"] = float(np.median(peak))
        tails[mu_label(mu)] = stats
    return tails


def _attach_tails(qdec: dict, tails: dict) -> dict:
    q = {n: dict(qdec[n]) for n in qdec}
    q[12]["frac_gap_ge"] = float(tails["frac_gap_ge"])
    q[12]["n_gap_ge"] = int(tails["n_gap_ge"])
    q[12]["n_inst"] = int(tails["n_inst"])
    q[12]["n_gap_below"] = int(tails["n_gap_below"])
    q[12]["gap_min"] = float(tails["gap_min"])
    if abs(float(q[12]["prod_gap_med"]) - float(tails["gap_med"])) > 1e-9:
        raise RuntimeError(
            f"n=12 median gap drift {q[12]['prod_gap_med']} vs cache {tails['gap_med']}"
        )
    return q


def _at(block, n):
    if n in block:
        return block[n]
    return block[str(n)]


def _role(mu: float, verdict: dict) -> str:
    bits = []
    if verdict.get("safe_mu") is not None and abs(float(mu) - float(verdict["safe_mu"])) < 1e-9:
        bits.append("SAFE")
    if verdict.get("edge_mu") is not None and abs(float(mu) - float(verdict["edge_mu"])) < 1e-9:
        bits.append("EDGE")
    return "+".join(bits)


def render(payload: dict) -> str:
    v = payload["verdict"]
    lines = []
    lines.append("# RESULTS — quantumness-safe μ on unsat Walsh, L=16 freeze n=12")
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
        "n=21 was not loaded. λ=0.75 is fixed."
    )
    lines.append("")
    lines.append("```text")
    lines.append("H = diag(unsat) − 0.75 Σ_i X_i + μ Σ_cav Σ_{i<j} X_i X_j")
    lines.append("μ ∈ {−1.00, −0.95, −0.90, −0.85, −0.80, −0.75, −0.60, −0.50}")
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
        "μ ∈ {−1.00, −0.85, −0.75, −0.50} are the PR #28 checkpoints, not retrains. "
        "μ ∈ {−0.95, −0.90, −0.80, −0.60} are new fits on the same diagonals."
    )
    lines.append("")
    lines.append(
        "R = (median over seeds of median ε) / (the same number for Fourier QAOA on this μ). "
        "Lower R is the improvement direction from PR #26 and PR #28. "
        "BEAT if R≤0.98, TIE if R≤1.02, else UNDER. "
        "The SAFE champ is the argmin of frozen n=12 Walsh R among grid points with "
        "median product gap ≥ 0.05 and at least 95% of the 64 n=12 instances at gap ≥ 0.02. "
        "The EDGE champ is the unconstrained argmin of that R. The operating point is SAFE. "
        "A tie on R prefers the thicker median gap."
    )
    lines.append("")
    lines.append("## Recommendation")
    lines.append("")
    safe = v.get("safe_mu")
    edge = v.get("edge_mu")
    if safe is None:
        lines.append("No SAFE champ.")
    else:
        lines.append(f"SAFE champ: μ={float(safe):+.2f}, n=12 Walsh R={float(v['safe_R']):.3f}.")
    if edge is None:
        lines.append("No EDGE champ.")
    else:
        lines.append(f"EDGE champ: μ={float(edge):+.2f}, n=12 Walsh R={float(v['edge_R']):.3f}.")
    if v.get("operating_mu") is not None:
        lines.append(f"Operating μ: {float(v['operating_mu']):+.2f}.")
    lines.append("")
    lines.append("## Quantumness")
    lines.append("")
    lines.append(
        "E0 is Lanczos on the sparse Hamiltonian. A split is a bitstring when the median "
        "ground-state peak is at least 0.5. The product gap is (E_prod − E_0) / |E_0| "
        "on the frozen n=12 transfer set (B=64). The safe bar is the median of that set "
        "and the fraction of its instances, not the training set."
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
            q = block[str(n)] if str(n) in block else block[n]
            lines.append(
                f"| {float(mu):+.2f} | {n} | {role_of[n]} | {q['peak_med']:.3f} | {q['prod_gap_med']:.3f} |"
            )
    lines.append("")
    lines.append("## Freeze table")
    lines.append("")
    lines.append(
        "n=6 and n=9 are the selection holdout. n=12 and n=15 are not. "
        "power_p is the scalar-γ control and does not choose μ. "
        "The champ column marks SAFE, EDGE, or both."
    )
    lines.append("")
    lines.append(
        "| μ | champ | head | n=12 R | seed-med R | gap med | ≥0.02 | min gap | peak | n=6 R | n=9 R | n=15 R |"
    )
    lines.append(
        "|--:|:------|:-----|-------:|-----------:|--------:|------:|--------:|-----:|------:|------:|-------:|"
    )
    tails = payload.get("gap_tails") or {}
    for mu in payload["mus"]:
        row = payload["rows"][mu_label(mu)]
        tail = tails[mu_label(mu)]
        champ = _role(float(mu), v) or ""
        for head in ("walsh", "power_p"):
            cells = row.get(head) or {}
            if not cells:
                continue
            try:
                cell = _at(cells, 12)
            except KeyError:
                continue

            def r(n, cells=cells):
                try:
                    got = _at(cells, n)
                except KeyError:
                    return "—"
                if got.get("R") is None:
                    return "—"
                return f"{got['R']:.3f}"

            seed_med = float(np.median(cell["seed_Rs"])) if cell.get("seed_Rs") else float("nan")
            gap_cell = f"{tail['gap_med']:.3f}" if head == "walsh" else ""
            frac_cell = (
                f"{tail['n_gap_ge']}/{tail['n_inst']}" if head == "walsh" else ""
            )
            min_cell = f"{tail['gap_min']:.3f}" if head == "walsh" else ""
            peak_cell = f"{tail['peak_med']:.3f}" if head == "walsh" else ""
            lines.append(
                f"| {float(mu):+.2f} | {champ if head == 'walsh' else ''} | {head} | "
                f"{r(12)} | {seed_med:.3f} | {gap_cell} | {frac_cell} | {min_cell} | "
                f"{peak_cell} | {r(6)} | {r(9)} | {r(15)} |"
            )
    lines.append("")
    lines.append("Per-seed n=12 Walsh R, seeds [0, 1, 2]:")
    lines.append("")
    for mu in payload["mus"]:
        cell = _at(payload["rows"][mu_label(mu)]["walsh"], 12)
        rs = ", ".join(f"{x:.3f}" for x in cell["seed_Rs"])
        med = float(np.median(cell["seed_Rs"]))
        lines.append(
            f"- μ={float(mu):+.2f}: {rs} (median of those R = {med:.3f}); "
            f"ratio of medians = {cell['R']:.3f}"
        )
    lines.append("")
    lines.append("## Reading")
    lines.append("")
    curve = [float(_at(payload["rows"][mu_label(mu)]["walsh"], 12)["R"]) for mu in payload["mus"]]
    listed = ", ".join(
        f"{float(mu):+.2f}:{r:.3f}" for mu, r in zip(payload["mus"], curve)
    )
    lines.append(f"Frozen n=12 Walsh R by μ: {listed}.")
    best_i = int(np.argmin(curve))
    lines.append(
        f"Unconstrained argmin on the ratio of medians is μ={float(payload['mus'][best_i]):+.2f} "
        f"at {curve[best_i]:.3f}."
    )
    if best_i not in (0, len(curve) - 1):
        lines.append(
            "That minimum is inside the grid, so R is not monotone in μ on this spacing."
        )
    seed_argmin = []
    for seed_i in range(3):
        seed_curve = []
        for mu in payload["mus"]:
            cell = _at(payload["rows"][mu_label(mu)]["walsh"], 12)
            seed_curve.append(float(cell["seed_Rs"][seed_i]))
        j = int(np.argmin(seed_curve))
        seed_argmin.append(
            f"seed {seed_i} μ={float(payload['mus'][j]):+.2f} at {seed_curve[j]:.3f}"
        )
    seed_med_curve = []
    for mu in payload["mus"]:
        cell = _at(payload["rows"][mu_label(mu)]["walsh"], 12)
        seed_med_curve.append(float(np.median(cell["seed_Rs"])))
    seed_med_i = int(np.argmin(seed_med_curve))
    lines.append("Per-seed argmin of n=12 Walsh R: " + "; ".join(seed_argmin) + ".")
    if abs(float(payload["mus"][seed_med_i]) - float(payload["mus"][best_i])) < 1e-9:
        lines.append(
            "The median of the three per-seed ratios picks the same μ as the ratio of medians, "
            f"at {seed_med_curve[seed_med_i]:.3f}."
        )
    else:
        lines.append(
            "The median of the three per-seed ratios picks "
            f"μ={float(payload['mus'][seed_med_i]):+.2f} at {seed_med_curve[seed_med_i]:.3f}."
        )
    try:
        pp_vals = [
            float(_at(payload["rows"][mu_label(mu)]["power_p"], 12)["R"]) for mu in payload["mus"]
        ]
    except KeyError:
        pp_vals = None
    if pp_vals is not None:
        if all(p > w for p, w in zip(pp_vals, curve)):
            lines.append("power_p stays above Walsh at every μ.")
        pp_i = int(np.argmin(pp_vals))
        lines.append(
            f"power_p's lowest n=12 R is μ={float(payload['mus'][pp_i]):+.2f} "
            f"at {pp_vals[pp_i]:.3f}."
        )
        if abs(float(payload["mus"][pp_i]) - float(payload["mus"][best_i])) > 1e-9:
            lines.append("Walsh's unconstrained minimum and the power_p minimum are different grid points.")
    n_frac_fail = 0
    n_med_fail = 0
    for mu in payload["mus"]:
        t = tails[mu_label(mu)]
        if float(t["frac_gap_ge"]) + 1e-12 < 0.95:
            n_frac_fail += 1
        if float(t["gap_med"]) + 1e-12 < 0.05:
            n_med_fail += 1
    def _removed(n):
        if n == 0:
            return "removes no grid point"
        if n == 1:
            return "removes 1 grid point"
        return f"removes {n} grid points"

    lines.append(
        f"The 95% instance bar {_removed(n_frac_fail)}. "
        f"The median-gap bar {_removed(n_med_fail)}."
    )
    feasible = (payload.get("verdict") or {}).get("feasible") or []
    safe_mu = (payload.get("verdict") or {}).get("safe_mu")
    if safe_mu is not None and feasible:
        others = [c for c in feasible if abs(float(c["mu"]) - float(safe_mu)) > 1e-9]
        if others:
            lines.append(
                "Other points that clear both bars sit at a higher ratio: "
                + ", ".join(f"μ={float(c['mu']):+.2f} at {float(c['R']):.3f}" for c in others)
                + "."
            )
    gaps = []
    for mu in payload["mus"]:
        t = tails[mu_label(mu)]
        gaps.append(
            f"{float(mu):+.2f} med {t['gap_med']:.3f} "
            f"({t['n_gap_ge']}/{t['n_inst']} ≥ 0.02, min {t['gap_min']:.3f})"
        )
    lines.append("n=12 product gap: " + "; ".join(gaps) + ".")
    if v.get("feasible"):
        lines.append(
            "Clears both bars: "
            + ", ".join(f"{c['mu']:+.2f} R={c['R']:.3f}" for c in v["feasible"])
            + "."
        )
    if v.get("infeasible"):
        bits = []
        for c in v["infeasible"]:
            why = (c.get("gap_fail") or c.get("bitstring") or ["ineligible"])[0]
            bits.append(f"{c['mu']:+.2f} R={c['R']:.3f} ({why})")
        lines.append("Misses the safe bar: " + "; ".join(bits) + ".")
    lines.append("")
    if payload.get("cost_scan"):
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
                    f"| {float(mu):+.2f} | {n} | {scan['gamma']:.4f} | {scan['med_eps']:.4f} | {w:.4f} |"
                )
        lines.append("")
    lines.append("## Rule")
    lines.append("")
    lines.append(
        "Operating μ is the SAFE champ: lowest frozen n=12 L=16 Walsh ratio among points "
        "with median product gap ≥ 0.05, at least 95% of the B=64 n=12 instances at gap ≥ 0.02, "
        "and ground-state peak < 0.5 on n=6, n=9, and n=12. "
        "EDGE is reported beside it and does not replace SAFE when the gap thins. "
        "power_p does not choose μ. n=15 is reported and does not choose μ. n=21 is not loaded."
    )
    lines.append("")
    lines.append(
        "PASS when the full grid names both champs and SAFE still beats or ties Fourier. "
        "needs_edit when a grid point, a resumed checkpoint, or the gap tail is missing, "
        "or when no μ clears both bars. "
        "KILL when the SAFE champ's Walsh ratio loses to Fourier or its ground state is a bitstring."
    )
    lines.append("")
    lines.append(
        f"Optimizer time summed over new fits {payload['opt_s']:.1f}s "
        f"(resumed checkpoints {payload['opt_s_resumed']:.1f}s, wall {payload['wall_s']:.1f}s). "
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
        default=str(_ROOT / "Nov_3rd" / "runs" / "quantum_h_explore" / "unsat_mu_safe"),
    )
    ap.add_argument(
        "--resume",
        type=str,
        default=str(_ROOT / "Nov_3rd" / "runs" / "quantum_h_explore" / "unsat_mu_neg"),
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
    if any(int(round(abs(m))) == 21 for m in mus):
        raise SystemExit("refusing n=21")
    out = Path(args_cli.out)
    resume = Path(args_cli.resume)
    if "n21" in str(out) or "n21" in str(resume):
        raise SystemExit("refusing an n=21 path")
    out.mkdir(parents=True, exist_ok=True)
    log(f"out={out}")
    t0 = time.time()
    sw = _load_sweep()
    _copy_resumed(sw, resume, out, mus)
    q_cache = sw._build_caches(resume, out, mus, args_cli.workers, diag_tag=cache_tag(ANCHOR_MU))
    (out / "quantum_cache.json").write_text(json.dumps(sw._jsonable(q_cache), indent=2) + "\n")
    tails_now = _gap_tails(out, mus)
    (out / "gap_tails.json").write_text(json.dumps(tails_now, indent=2) + "\n")
    for mu in mus:
        t = tails_now[mu_label(mu)]
        log(
            f"gap μ={mu:+.2f} med={t['gap_med']:.3f} "
            f"{t['n_gap_ge']}/{t['n_inst']} ≥0.02 min={t['gap_min']:.3f}"
        )
    if args_cli.ed_only:
        log(f"ED only, {time.time() - t0:.1f}s")
        return

    drv = sw._load_driver()

    def _patched_paths(path, L, method, seed):
        return sw._paths_for(drv, path, L, method, seed)

    drv._paths = _patched_paths
    torch.set_num_threads(int(args_cli.threads))
    args = sw._namespace(args_cli.workers)
    rows = {}
    audits = []
    scans = {}
    opt_new = 0.0
    opt_resumed = 0.0
    q_rows = {}
    full = len(mus) == len(MU_GRID) and all(
        abs(float(a) - float(b)) < 5e-4 for a, b in zip(mus, MU_GRID)
    )
    for mu in mus:
        drv.MU = float(mu)
        log(f"=== μ={mu:+.2f} resume={_is_resume(mu)} ===")
        prepared, _sims, quantum = drv.prepare(args, out)
        if any(int(n) == 21 for n, _role in prepared):
            raise RuntimeError("n=21 entered prepare")
        audit = drv.audit(prepared)
        audits.append({"mu": float(mu), **audit})
        train_ns = [sp.n for sp in (prepared[(n, "train")]["split"] for n in (6, 9))]
        if train_ns != [6, 9]:
            raise RuntimeError(f"train split drifted {train_ns}")
        if args_cli.skip_power:
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
            cells = sw._cells_without_power(drv, runs)
        else:
            runs = drv.train(prepared, args, out)
            cells = drv.build_cells(runs)
        for rec in runs:
            if _is_resume(mu):
                opt_resumed += float(rec["wall_s"])
            else:
                opt_new += float(rec["wall_s"])
        qdec = _attach_tails(drv.quantum_for_decide(quantum), tails_now[mu_label(mu)])
        row = sw._row_from_cells(cells, qdec)
        rows[mu_label(mu)] = row
        q_rows[mu_label(mu)] = {str(n): qdec[n] for n in (6, 9, 12, 15)}
        scan = {}
        for n, role in ((6, "hold"), (9, "hold"), (12, "transfer"), (15, "transfer")):
            scan[str(n)] = drv.cost_scan(prepared[(n, role)]["split"].circuit, n)
            log(f"cost μ={mu:+.2f} n={n} γ={scan[str(n)]['gamma']:.4f} ε={scan[str(n)]['med_eps']:.4f}")
        scans[mu_label(mu)] = scan
        w12 = row["walsh"][12]
        log(
            f"μ={mu:+.2f} walsh n12 R={w12['R']:.3f} med ε={w12['med']:.4f} "
            f"gap={qdec[12]['prod_gap_med']:.3f} "
            f"{qdec[12]['n_gap_ge']}/{qdec[12]['n_inst']}"
        )
        partial = {
            "rows": rows,
            "quantum_rows": q_rows,
            "cost_scan": scans,
            "audits": audits,
            "opt_s": opt_new,
            "opt_s_resumed": opt_resumed,
            "mus": [float(m) for m in mus],
            "gap_tails": {mu_label(m): tails_now[mu_label(m)] for m in mus if mu_label(m) in q_rows},
        }
        (out / "partial.json").write_text(json.dumps(sw._jsonable(partial), indent=2) + "\n")

    if not full:
        log(f"partial grid, not writing a verdict. wall={time.time() - t0:.1f}s")
        return
    verdict = decide_mu_safe(rows)
    log(f"VERDICT {verdict['verdict']}: {verdict['reason']}")
    worst_e0 = max(float(a["worst_e0"]) for a in audits)
    worst_psi = max(float(a["worst_psi"]) for a in audits)
    wall = time.time() - t0
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
        "wall_s": wall,
        "branch": BRANCH,
        "published_anchor_R": float(rows[mu_label(ANCHOR_MU)]["walsh"][12]["R"]),
        "skip_power": bool(args_cli.skip_power),
        "gap_tails": tails_now,
    }
    text = render(payload)
    (out / "RESULTS.md").write_text(text)
    (out / "summary.json").write_text(json.dumps(sw._jsonable(payload), indent=2) + "\n")
    (out / "verdict.json").write_text(json.dumps(sw._jsonable(verdict), indent=2) + "\n")
    log(f"wrote {out / 'RESULTS.md'} verdict={verdict['verdict']} wall={wall:.1f}s")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
