#!/usr/bin/env python3
"""2D safe grid λ×μ for unsat Walsh. Same gate and L=16 budget as PR #26.

    H = diag(unsat) − λ Σ_i X_i + μ Σ_cav Σ_{i<j} X_i X_j
    μ ∈ {−0.90, −0.75, −0.60, −0.50}
    λ ∈ {0.75, 0.90, 1.00}

Train n=6 and n=9 only. Freeze n=12 (n=15 is reported). Six cells reuse a
finished checkpoint with the same λ, μ, seeds, and budget. The other six are
new fits on the same unsat diagonals. n=21 is not loaded.

SAFE minimizes frozen n=12 Walsh R subject to median product gap ≥ 0.05 and
≥95% of instances at gap ≥ 0.02. EDGE minimizes that R subject only to the
95% instance bar.
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

from Nov_3rd.unsat_lambda_mu_grid import (
    ANCHOR,
    CELLS,
    cell_key,
    decide_lambda_mu_grid,
    file_stem,
    gap_stats,
    is_resume,
)
from Nov_3rd.unsat_rank2_xx import batch_ground, cache_filename

BRANCH = "cursor/unsat-walsh-lam-mu-grid-1cdb"
METHODS = ("fourier_qaoa", "power_p", "walsh")
SPECS = (
    (6, "train", 128),
    (6, "hold", 128),
    (9, "train", 128),
    (9, "hold", 128),
    (12, "transfer", 64),
    (15, "transfer", 32),
)


def log(*a):
    print(time.strftime("%H:%M:%S"), "[lamMu]", *a, flush=True)


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


def _refuse_n21(*paths: Path):
    for path in paths:
        name = str(path)
        if "n21" in name or "n_21" in name or "/21/" in name:
            raise SystemExit(f"refusing n=21 path {path}")


def _ckpt_src_stem(lam: float, mu: float) -> str:
    """Filename stem in the source sweep. λ=0.75 uses the μ-sweep names."""
    if abs(float(lam) - 0.75) < 1e-12:
        tag = f"{float(mu):+.2f}".replace("+", "").replace(".", "p")
        return f"mu{tag}"
    tag = f"{float(lam):.2f}".replace(".", "p")
    return f"lam{tag}"


def _resume_root(lam: float, mu: float, mu_safe: Path, lam_sweep: Path) -> Path:
    if abs(float(lam) - 0.75) < 1e-12:
        return mu_safe
    if abs(float(mu) - (-0.50)) < 1e-12:
        return lam_sweep
    raise RuntimeError(f"no resume source for λ={lam:.2f} μ={mu:+.2f}")


def _paths_for(out: Path, L: int, method: str, seed: int, lam: float, mu: float):
    stem = Path(out) / "runs" / f"{file_stem(lam, mu)}_L{int(L)}_{method}_seed{int(seed)}"
    return Path(str(stem) + ".json"), Path(str(stem) + ".pt")


def _cache_path(out: Path, n: int, role: str, count: int, lam: float, mu: float) -> Path:
    return out / "cache" / cache_filename(n, role, count, lam, mu)


def _same_diag(anchor: dict, blob: dict, n: int, role: str):
    if not np.array_equal(np.asarray(anchor["score"]), np.asarray(blob["score"])):
        raise RuntimeError(f"unsat diagonal drift n={n} {role}")
    if not np.array_equal(np.asarray(anchor["seeds"]), np.asarray(blob["seeds"])):
        raise RuntimeError(f"formula seed drift n={n} {role}")
    if not np.array_equal(np.asarray(anchor["pulse"]), np.asarray(blob["pulse"])):
        raise RuntimeError(f"pulse drift n={n} {role}")
    if int(np.asarray(anchor["m"])) != int(np.asarray(blob["m"])):
        raise RuntimeError(f"clause-count drift n={n} {role}")


def _copy_resumed(mu_safe: Path, lam_sweep: Path, out: Path, cells):
    """Copy identical (λ, μ) caches and L=16 checkpoints. Not a retrain."""
    _refuse_n21(mu_safe, lam_sweep, out)
    copied = 0
    for lam, mu in cells:
        if not is_resume(lam, mu):
            continue
        src_root = _resume_root(lam, mu, mu_safe, lam_sweep)
        stem = _ckpt_src_stem(lam, mu)
        for n, role, count in SPECS:
            if int(n) == 21:
                raise SystemExit("refusing n=21")
            src = src_root / "cache" / cache_filename(n, role, count, lam, mu)
            dst = _cache_path(out, n, role, count, lam, mu)
            if not src.exists():
                raise FileNotFoundError(src)
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists():
                shutil.copy2(src, dst)
                copied += 1
            blob = np.load(dst)
            if abs(float(blob["lam"]) - float(lam)) > 1e-12 or abs(float(blob["mu"]) - float(mu)) > 1e-12:
                raise RuntimeError(f"resumed cache λ/μ drift {dst.name}")
        runs = out / "runs"
        runs.mkdir(parents=True, exist_ok=True)
        for method in METHODS:
            for seed in (0, 1, 2):
                for ext in (".json", ".pt"):
                    src = src_root / "runs" / f"{stem}_L16_{method}_seed{seed}{ext}"
                    dst_js, dst_pt = _paths_for(out, 16, method, seed, lam, mu)
                    dst = dst_js if ext == ".json" else dst_pt
                    if not src.exists():
                        raise FileNotFoundError(src)
                    if not dst.exists():
                        shutil.copy2(src, dst)
                        copied += 1
                js, pt = _paths_for(out, 16, method, seed, lam, mu)
                rec = json.loads(js.read_text())
                if abs(float(rec["lam"]) - float(lam)) > 1e-12 or abs(float(rec["mu"]) - float(mu)) > 1e-12:
                    raise RuntimeError(f"resumed checkpoint λ/μ drift {js.name}")
                if int(rec["L"]) != 16 or int(rec["rounds_budget"]) != 48:
                    raise RuntimeError(f"resumed budget drift {js.name}")
                if abs(float(rec["lr"]) - 0.35) > 1e-12 or int(rec["max_iter"]) != 8:
                    raise RuntimeError(f"resumed optimizer drift {js.name}")
                hold = {int(k) for k in rec["hold_during_select"]}
                if hold != {6, 9}:
                    raise RuntimeError(f"resumed checkpoint selected outside n=6,9: {js.name}")
                if "12" not in rec["eval"] and 12 not in rec["eval"]:
                    raise RuntimeError(f"resumed checkpoint missing frozen n=12 {js.name}")
                if not pt.exists():
                    raise RuntimeError(f"missing weights {pt.name}")
                if "n21" in js.name:
                    raise RuntimeError(f"n=21 checkpoint {js.name}")
        log(f"resumed λ={lam:.2f} μ={mu:+.2f} from {src_root.name}")
    log(f"resumed checkpoints copied ({copied} new files)")


def _build_caches(out: Path, cells, workers: int):
    """Reuse the PR #26 unsat diagonals. Only E0 changes with λ and μ."""
    anchor_lam, anchor_mu = ANCHOR
    anchors = {}
    for n, role, count in SPECS:
        src = _cache_path(out, n, role, count, anchor_lam, anchor_mu)
        z = np.load(src)
        anchors[(n, role)] = {k: z[k] for k in z.files}
        if int(anchors[(n, role)]["score"].shape[0]) != int(count):
            raise RuntimeError(f"anchor n={n} {role} B drift")
    for lam, mu in cells:
        for n, role, count in SPECS:
            if int(n) == 21:
                raise SystemExit("refusing n=21")
            dst = _cache_path(out, n, role, count, lam, mu)
            base = anchors[(n, role)]
            if dst.exists():
                blob = np.load(dst)
                if abs(float(blob["lam"]) - float(lam)) > 1e-12 or abs(float(blob["mu"]) - float(mu)) > 1e-12:
                    raise RuntimeError(f"cache λ/μ drift {dst.name}")
                _same_diag(base, blob, n, role)
                continue
            t0 = time.time()
            log(f"ED n={n} {role} B={count} λ={lam:.2f} μ={mu:+.2f}")
            gu = batch_ground(base["score"], n, float(lam), float(mu), workers, seeds=base["seeds"])
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
                mu=np.asarray(float(mu), dtype=np.float64),
            )
            log(
                f"  n={n} {role} λ={lam:.2f} μ={mu:+.2f} in {time.time() - t0:.1f}s "
                f"gap={float(np.median(gu['prod_gap'])):.3f}"
            )


def _gap_tails(out: Path, cells) -> dict:
    tails = {}
    for lam, mu in cells:
        path = _cache_path(out, 12, "transfer", 64, lam, mu)
        gap = np.asarray(np.load(path)["prod_gap"], dtype=np.float64)
        peak = np.asarray(np.load(path)["peak"], dtype=np.float64)
        stats = gap_stats(gap)
        stats["peak_max"] = float(peak.max())
        stats["peak_med"] = float(np.median(peak))
        tails[cell_key(lam, mu)] = stats
    return tails


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


def _attach_tails(qdec: dict, tails: dict) -> dict:
    q = {n: dict(qdec[n]) for n in qdec}
    q[12]["frac_gap_ge"] = float(tails["frac_gap_ge"])
    q[12]["n_gap_ge"] = int(tails["n_gap_ge"])
    q[12]["n_inst"] = int(tails["n_inst"])
    q[12]["n_gap_below"] = int(tails["n_gap_below"])
    q[12]["gap_min"] = float(tails["gap_min"])
    q[12]["peak_max"] = float(tails["peak_max"])
    if abs(float(q[12]["prod_gap_med"]) - float(tails["gap_med"])) > 1e-9:
        raise RuntimeError(
            f"n=12 median gap drift {q[12]['prod_gap_med']} vs cache {tails['gap_med']}"
        )
    if abs(float(q[12]["peak_med"]) - float(tails["peak_med"])) > 1e-9:
        raise RuntimeError(
            f"n=12 peak drift {q[12]['peak_med']} vs cache {tails['peak_med']}"
        )
    return q


def _at(block, n):
    if n in block:
        return block[n]
    return block[str(n)]


def _role(lam: float, mu: float, verdict: dict) -> str:
    bits = []
    same_safe = (
        verdict.get("safe_lam") is not None
        and abs(float(lam) - float(verdict["safe_lam"])) < 1e-9
        and abs(float(mu) - float(verdict["safe_mu"])) < 1e-9
    )
    same_edge = (
        verdict.get("edge_lam") is not None
        and abs(float(lam) - float(verdict["edge_lam"])) < 1e-9
        and abs(float(mu) - float(verdict["edge_mu"])) < 1e-9
    )
    if same_safe:
        bits.append("SAFE")
    if same_edge:
        bits.append("EDGE")
    return "+".join(bits)


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


def render(payload: dict) -> str:
    v = payload["verdict"]
    lines = []
    lines.append("# RESULTS — 2D safe grid λ×μ on unsat Walsh, L=16 freeze n=12")
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
    lines.append("H = diag(unsat) − λ Σ_i X_i + μ Σ_cav Σ_{i<j} X_i X_j")
    lines.append("μ ∈ {−0.90, −0.75, −0.60, −0.50}")
    lines.append("λ ∈ {0.75, 0.90, 1.00}")
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
        "λ=0.75 reuses the PR #31 checkpoints at these four μ values. "
        "λ=0.90 and λ=1.00 at μ=−0.50 reuse the PR #29 checkpoints. "
        "The other six cells are new fits on the same diagonals."
    )
    lines.append("")
    lines.append(
        "R = (median over seeds of median ε) / (the same number for Fourier QAOA on this λ and μ). "
        "Lower R is the improvement direction from PR #26. "
        "BEAT if R≤0.98, TIE if R≤1.02, else UNDER. "
        "The SAFE champ is the argmin of frozen n=12 Walsh R among cells with "
        "median product gap ≥ 0.05 and at least 95% of the 64 n=12 instances at gap ≥ 0.02. "
        "The EDGE champ is the argmin of that R among cells with at least 95% of those instances "
        "at gap ≥ 0.02. The median-gap bar does not apply to EDGE. "
        "The operating point is SAFE. A tie on R prefers the thicker median gap, then the higher instance fraction. "
        "power_p does not choose the cell. n=15 does not choose the cell."
    )
    lines.append("")
    lines.append("## Recommendation")
    lines.append("")
    if v.get("safe_lam") is None:
        lines.append("No SAFE champ.")
    else:
        lines.append(
            f"SAFE champ: λ={float(v['safe_lam']):.2f}, μ={float(v['safe_mu']):+.2f}, "
            f"n=12 Walsh R={float(v['safe_R']):.3f}."
        )
    if v.get("edge_lam") is None:
        lines.append("No EDGE champ.")
    else:
        lines.append(
            f"EDGE champ: λ={float(v['edge_lam']):.2f}, μ={float(v['edge_mu']):+.2f}, "
            f"n=12 Walsh R={float(v['edge_R']):.3f}."
        )
    if v.get("operating_lam") is not None:
        lines.append(
            f"Operating point: λ={float(v['operating_lam']):.2f}, μ={float(v['operating_mu']):+.2f}."
        )
    lines.append("")
    lines.append("## Grid")
    lines.append("")
    lines.append(
        "One table. n=6 and n=9 are the selection holdout. n=12 and n=15 are not. "
        "The champ column marks SAFE, EDGE, or both, on the Walsh row. "
        "Gap columns are the frozen n=12 product gap and do not depend on the head."
    )
    lines.append("")
    if payload.get("audit"):
        lines.append(
            f"Worst |ΔE0| Lanczos vs dense over the grid = {payload['audit']['worst_e0']:.2e}. "
            f"Histogram vs ⟨ψ|H|ψ⟩ = {payload['audit']['worst_psi']:.2e}."
        )
        lines.append("")
    lines.append(
        "| λ | μ | champ | head | n=12 R | seed-med R | gap med | ≥0.02 | min gap | peak | n=6 R | n=9 R | n=15 R |"
    )
    lines.append(
        "|--:|--:|:------|:-----|-------:|-----------:|--------:|------:|--------:|-----:|------:|------:|-------:|"
    )
    tails = payload.get("gap_tails") or {}
    for lam, mu in payload["cells"]:
        key = cell_key(lam, mu)
        row = payload["rows"][key]
        tail = tails[key]
        champ = _role(lam, mu, v) or ""
        for head in ("walsh", "power_p"):
            block = row.get(head) or {}
            if not block:
                continue
            try:
                cell = _at(block, 12)
            except KeyError:
                continue

            def r(n, block=block):
                try:
                    got = _at(block, n)
                except KeyError:
                    return "—"
                if got.get("R") is None:
                    return "—"
                return f"{got['R']:.3f}"

            seed_med = float(np.median(cell["seed_Rs"])) if cell.get("seed_Rs") else float("nan")
            if head == "walsh":
                gap_cell = f"{tail['gap_med']:.3f}"
                frac_cell = f"{tail['n_gap_ge']}/{tail['n_inst']}"
                min_cell = f"{tail['gap_min']:.3f}"
                peak_cell = f"{tail['peak_med']:.3f}"
                champ_cell = champ
            else:
                gap_cell = frac_cell = min_cell = peak_cell = champ_cell = ""
            lines.append(
                f"| {float(lam):.2f} | {float(mu):+.2f} | {champ_cell} | {head} | "
                f"{r(12)} | {seed_med:.3f} | {gap_cell} | {frac_cell} | {min_cell} | "
                f"{peak_cell} | {r(6)} | {r(9)} | {r(15)} |"
            )
    lines.append("")
    lines.append("Per-seed n=12 Walsh R, seeds [0, 1, 2]:")
    lines.append("")
    for lam, mu in payload["cells"]:
        cell = _at(payload["rows"][cell_key(lam, mu)]["walsh"], 12)
        rs = ", ".join(f"{x:.3f}" for x in cell["seed_Rs"])
        med = float(np.median(cell["seed_Rs"]))
        lines.append(
            f"- λ={float(lam):.2f} μ={float(mu):+.2f}: {rs} "
            f"(median of those R = {med:.3f}); ratio of medians = {cell['R']:.3f}"
        )
    lines.append("")
    lines.append("## Reading")
    lines.append("")
    listed = ", ".join(
        f"({float(lam):.2f},{float(mu):+.2f}):"
        f"{float(_at(payload['rows'][cell_key(lam, mu)]['walsh'], 12)['R']):.3f}"
        for lam, mu in payload["cells"]
    )
    lines.append(f"Frozen n=12 Walsh R by (λ, μ): {listed}.")
    curve = [
        (
            float(lam),
            float(mu),
            float(_at(payload["rows"][cell_key(lam, mu)]["walsh"], 12)["R"]),
        )
        for lam, mu in payload["cells"]
    ]
    best = min(curve, key=lambda t: t[2])
    lines.append(
        f"Unconstrained argmin on the ratio of medians is λ={best[0]:.2f}, μ={best[1]:+.2f} at {best[2]:.3f}."
    )
    edge_pool = v.get("edge_pool") or []
    safe_pool = v.get("safe_pool") or []
    n_med_fail = 0
    n_frac_fail = 0
    for lam, mu in payload["cells"]:
        t = tails[cell_key(lam, mu)]
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
    if safe_pool:
        lines.append(
            "Clears both SAFE bars: "
            + ", ".join(
                f"λ={c['lam']:.2f} μ={c['mu']:+.2f} R={c['R']:.3f}" for c in safe_pool
            )
            + "."
        )
    if edge_pool:
        lines.append(
            "Clears the EDGE instance bar: "
            + ", ".join(
                f"λ={c['lam']:.2f} μ={c['mu']:+.2f} R={c['R']:.3f} gap={c['gap']:.3f}"
                for c in edge_pool
            )
            + "."
        )
    try:
        pp = [
            float(_at(payload["rows"][cell_key(lam, mu)]["power_p"], 12)["R"])
            for lam, mu in payload["cells"]
        ]
        ww = [t[2] for t in curve]
        if all(p > w for p, w in zip(pp, ww)):
            lines.append("power_p stays above Walsh at every cell.")
        j = int(np.argmin(pp))
        lam_j, mu_j = payload["cells"][j]
        lines.append(
            f"power_p's lowest n=12 R is λ={float(lam_j):.2f}, μ={float(mu_j):+.2f} at {pp[j]:.3f}."
        )
    except KeyError:
        pass
    gaps = []
    for lam, mu in payload["cells"]:
        t = tails[cell_key(lam, mu)]
        gaps.append(
            f"({float(lam):.2f},{float(mu):+.2f}) med {t['gap_med']:.3f} "
            f"({t['n_gap_ge']}/{t['n_inst']} ≥ 0.02, min {t['gap_min']:.3f})"
        )
    lines.append("n=12 product gap: " + "; ".join(gaps) + ".")
    lines.append("")
    if payload.get("cost_scan"):
        lines.append("## One-parameter cost pulse")
        lines.append("")
        lines.append(
            "SNAP phase 0. XX is not in that pulse. The scan is the minimum median ε over γ ∈ [−2, 2]."
        )
        lines.append("")
        lines.append("| λ | μ | n | γ | Med ε | Walsh L=16 Med ε |")
        lines.append("|--:|--:|--:|--:|------:|------------------:|")
        for lam, mu in payload["cells"]:
            for n in (6, 9, 12, 15):
                scan = payload["cost_scan"][cell_key(lam, mu)][str(n)]
                w = _at(payload["rows"][cell_key(lam, mu)]["walsh"], n)["med"]
                lines.append(
                    f"| {float(lam):.2f} | {float(mu):+.2f} | {n} | {scan['gamma']:.4f} | "
                    f"{scan['med_eps']:.4f} | {w:.4f} |"
                )
        lines.append("")
    lines.append("## Rule")
    lines.append("")
    lines.append(
        "Operating point is the SAFE champ: lowest frozen n=12 L=16 Walsh ratio among cells "
        "with median product gap ≥ 0.05, at least 95% of the B=64 n=12 instances at gap ≥ 0.02, "
        "and ground-state peak < 0.5 on n=6, n=9, and n=12. "
        "EDGE is the lowest ratio among cells that clear the 95% instance bar and the same peak gate. "
        "EDGE does not replace SAFE when the median gap is under 0.05. "
        "A tie prefers the thicker median gap. "
        "power_p does not choose the cell. n=15 is reported and does not choose the cell. n=21 is not loaded."
    )
    lines.append("")
    lines.append(
        "PASS when the full grid names both champs and SAFE still beats Fourier "
        "(R≤0.98, at least two seeds beat their own Fourier, n=15 R stays under 1.10, no reversal). "
        "needs_edit when a grid point, a resumed checkpoint, or a gap tail is missing, "
        "when no cell clears a champ's bar, or when SAFE misses a transfer gate. "
        "KILL when the SAFE champ's Walsh ratio loses to Fourier or the L=16 advantage reverses."
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


def _parse_cells(text: str):
    out = []
    for chunk in text.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        lam_s, mu_s = chunk.split(":")
        out.append((float(lam_s), float(mu_s)))
    return tuple(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out",
        type=str,
        default=str(_ROOT / "Nov_3rd" / "runs" / "quantum_h_explore" / "unsat_lambda_mu_grid"),
    )
    ap.add_argument("--mu-safe", type=str, required=True)
    ap.add_argument("--lambda-sweep", type=str, required=True)
    ap.add_argument(
        "--cells",
        type=str,
        default=",".join(f"{lam:.2f}:{mu:.2f}" for lam, mu in CELLS),
        help="Comma-separated lam:mu pairs. The default is the full grid.",
    )
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--ed-only", action="store_true")
    args_cli = ap.parse_args()
    cells = _parse_cells(args_cli.cells)
    for lam, mu in cells:
        if abs(float(lam) - 21.0) < 1e-9 or abs(float(mu) - 21.0) < 1e-9:
            raise SystemExit("refusing n=21")
        if lam < 0:
            raise SystemExit(f"refusing negative λ={lam}")
    out = Path(args_cli.out)
    mu_safe = Path(args_cli.mu_safe)
    lam_sweep = Path(args_cli.lambda_sweep)
    _refuse_n21(out, mu_safe, lam_sweep)
    out.mkdir(parents=True, exist_ok=True)
    log(f"out={out} cells={len(cells)}")
    t0 = time.time()
    _copy_resumed(mu_safe, lam_sweep, out, cells)
    _build_caches(out, cells, args_cli.workers)
    tails_now = _gap_tails(out, cells)
    (out / "gap_tails.json").write_text(json.dumps(tails_now, indent=2) + "\n")
    for lam, mu in cells:
        t = tails_now[cell_key(lam, mu)]
        log(
            f"gap λ={lam:.2f} μ={mu:+.2f} med={t['gap_med']:.3f} "
            f"{t['n_gap_ge']}/{t['n_inst']} ≥0.02 min={t['gap_min']:.3f}"
        )
    if args_cli.ed_only:
        log(f"ED only, {time.time() - t0:.1f}s")
        return

    drv = _load_driver()
    hold = {"lam": float(ANCHOR[0]), "mu": float(ANCHOR[1])}

    def _patched_paths(path, L, method, seed):
        return _paths_for(path, L, method, seed, hold["lam"], hold["mu"])

    drv._paths = _patched_paths
    torch.set_num_threads(int(args_cli.threads))
    args = _namespace(args_cli.workers)
    rows = {}
    audits = []
    scans = {}
    opt_new = 0.0
    opt_resumed = 0.0
    q_rows = {}
    full = list(cells) == list(CELLS)
    for lam, mu in cells:
        hold["lam"] = float(lam)
        hold["mu"] = float(mu)
        drv.LAM = float(lam)
        drv.MU = float(mu)
        log(f"=== λ={lam:.2f} μ={mu:+.2f} resume={is_resume(lam, mu)} ===")
        prepared, _sims, quantum = drv.prepare(args, out)
        if any(int(n) == 21 for n, _role in prepared):
            raise RuntimeError("n=21 entered prepare")
        audit = drv.audit(prepared)
        audits.append({"lam": float(lam), "mu": float(mu), **audit})
        train_ns = [sp.n for sp in (prepared[(n, "train")]["split"] for n in (6, 9))]
        if train_ns != [6, 9]:
            raise RuntimeError(f"train split drifted {train_ns}")
        runs = drv.train(prepared, args, out)
        for rec in runs:
            if abs(float(rec.get("lam", lam)) - float(lam)) > 1e-9:
                raise RuntimeError(f"fit λ drift {rec.get('method')} seed {rec.get('seed')}")
            if abs(float(rec.get("mu", 99)) - float(mu)) > 1e-9:
                raise RuntimeError(f"fit μ drift {rec.get('method')} seed {rec.get('seed')}")
            if is_resume(lam, mu):
                opt_resumed += float(rec["wall_s"])
            else:
                opt_new += float(rec["wall_s"])
        cells_built = drv.build_cells(runs)
        qdec = _attach_tails(drv.quantum_for_decide(quantum), tails_now[cell_key(lam, mu)])
        row = _row_from_cells(cells_built, qdec)
        rows[cell_key(lam, mu)] = row
        q_rows[cell_key(lam, mu)] = {str(n): qdec[n] for n in (6, 9, 12, 15)}
        scan = {}
        for n, role in ((6, "hold"), (9, "hold"), (12, "transfer"), (15, "transfer")):
            scan[str(n)] = drv.cost_scan(prepared[(n, role)]["split"].circuit, n)
            log(
                f"cost λ={lam:.2f} μ={mu:+.2f} n={n} "
                f"γ={scan[str(n)]['gamma']:.4f} ε={scan[str(n)]['med_eps']:.4f}"
            )
        scans[cell_key(lam, mu)] = scan
        w12 = row["walsh"][12]
        log(
            f"λ={lam:.2f} μ={mu:+.2f} walsh n12 R={w12['R']:.3f} med ε={w12['med']:.4f} "
            f"gap={qdec[12]['prod_gap_med']:.3f} {qdec[12]['n_gap_ge']}/{qdec[12]['n_inst']}"
        )
        partial = {
            "rows": rows,
            "quantum_rows": q_rows,
            "cost_scan": scans,
            "audits": audits,
            "opt_s": opt_new,
            "opt_s_resumed": opt_resumed,
            "cells": [[float(a), float(b)] for a, b in cells],
            "gap_tails": {cell_key(a, b): tails_now[cell_key(a, b)] for a, b in cells if cell_key(a, b) in rows},
        }
        (out / "partial.json").write_text(json.dumps(_jsonable(partial), indent=2) + "\n")

    if not full:
        log(f"partial grid, not writing a verdict. wall={time.time() - t0:.1f}s")
        return
    verdict = decide_lambda_mu_grid(rows)
    log(f"VERDICT {verdict['verdict']}: {verdict['reason']}")
    worst_e0 = max(float(a["worst_e0"]) for a in audits)
    worst_psi = max(float(a["worst_psi"]) for a in audits)
    wall = time.time() - t0
    payload = {
        "verdict": verdict,
        "rows": rows,
        "quantum_rows": q_rows,
        "cost_scan": scans,
        "audit": {"worst_e0": worst_e0, "worst_psi": worst_psi, "per_cell": audits},
        "cells": [[float(a), float(b)] for a, b in cells],
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
        "wall_s": wall,
        "branch": BRANCH,
        "gap_tails": tails_now,
    }
    text = render(payload)
    (out / "RESULTS.md").write_text(text)
    (out / "summary.json").write_text(json.dumps(_jsonable(payload), indent=2) + "\n")
    (out / "verdict.json").write_text(json.dumps(_jsonable(verdict), indent=2) + "\n")
    log(f"wrote {out / 'RESULTS.md'} verdict={verdict['verdict']} wall={wall:.1f}s")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
