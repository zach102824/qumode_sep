#!/usr/bin/env python3
"""LN-LA smoke: freeze power_p champs, 9×9 grid on train {6,9}, CORE B=1000.

Opus Idea 2. No optimizer. α, β ∈ linspace(−1, 1, 9).

Train grid uses the fresh250 packs, first B=100 of n=6 and n=9.
Selection score is the champ weight (Med_6 + 2 Med_9) / 3. Lower is better.
Tie-break prefers the cell closer to (0, 0).

Decisive numbers are Med on a fresh CORE pack (base 73e9), n ∈ {6,9,12,15},
disjoint from train. Champ is the α=β=0 cell on that same pack.

Kill (per L) if the grid optimum has both |α| and |β| < 0.05, or the fresh
n=15 Med gain versus the frozen champ is under 0.5%.
Ship only if it also beats same-L QAOA Med on every CORE n.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
_SCR = Path(__file__).resolve().parent
if str(_SCR) not in sys.path:
    sys.path.insert(0, str(_SCR))

from Nov_3rd import settings as S
from Nov_3rd.ansatz import make_sim
from Nov_3rd.lnla import (
    angles_from_champ,
    cavity_loads,
    forward_lnla,
)
from explore_OCT29.lib.qaoa import inv_p_from_psucc, qaoa_psucc
from explore_OCT29.lib.sat_gen import clause_count, gen_instance, instances, pack_clauses
from train_qaoa_train69_fresh import train_qaoa_train69

FRESH = S.OUT / "beat_qaoa_12h" / "train69_ladder_fresh250"
OLD = S.OUT / "beat_qaoa_12h" / "train69"
DEFAULT_OUT = OLD / "lnla"
# New base: not sealed 51e9, not scale1000 54e9, not beat extras 61e9/62e9.
CORE_BASE = 73_000_000_000
TRAIN_NS = (6, 9)
CORE_NS = (6, 9, 12, 15)
W6, W9 = 1.0, 2.0
TIED_KAPPAS = (-1.0, -0.5, 0.0, 0.5, 1.0)
# κ = 0 is the β = 0 row (cost load only). Nonzero κ is a real tie β = κα.
TIED_NONZERO = tuple(k for k in TIED_KAPPAS if k != 0.0)

# Fresh250 ladder has no L=2. Freeze the best-holdout power_p L=2 that exists.
DEFAULT_CHAMP = {
    2: OLD / "power_p" / "pfix0.15" / "params_L2_seed0.pt",
    16: FRESH / "power_p" / "params_L16_seed0.pt",
}
DEFAULT_QAOA = {
    16: FRESH / "qaoa_angles" / "L16" / "angles.json",
}
HISTORICAL_QAOA_L2 = S.QAOA_DIR / "L2" / "angles.json"


def log(*a):
    print(time.strftime("%H:%M:%S"), "[lnla]", *a, flush=True)


def grid_values(n: int) -> np.ndarray:
    """Inclusive grid on [-1, 1]. Odd n puts 0 on a node (exact in float64)."""
    n = int(n)
    if n < 3 or n % 2 == 0:
        raise ValueError(f"grid size must be odd and >= 3, got {n}")
    half = n // 2
    g = np.array([i / half for i in range(-half, half + 1)], dtype=np.float64)
    if g[half] != 0.0 or g[0] != -1.0 or g[-1] != 1.0:
        raise RuntimeError(f"grid {g}")
    return g


def train_score(med6: float, med9: float) -> float:
    return (W6 * med6 + W9 * med9) / (W6 + W9)


def summarize(invs: np.ndarray, leaks: np.ndarray | None = None) -> dict:
    a = np.asarray(invs, dtype=np.float64).reshape(-1)
    out = {
        "Med": float(np.median(a)),
        "B90": float(np.quantile(a, 0.90)),
        "mean": float(np.mean(a)),
        "B": int(a.size),
    }
    if leaks is not None:
        out["mean_leak"] = float(np.mean(np.asarray(leaks, dtype=np.float64)))
    return out


def clause_vars_of(cl) -> np.ndarray:
    return np.asarray([v for v, _s in cl], dtype=np.int64)


def instances_from_seeds(n: int, seeds: np.ndarray, E_check: np.ndarray | None) -> list[dict]:
    out = []
    for i, seed in enumerate(np.asarray(seeds).tolist()):
        _m, cl, Eg = gen_instance(int(n), int(seed), compute_E=True)
        if E_check is not None and not np.array_equal(Eg, np.asarray(E_check[i])):
            raise RuntimeError(f"n{n} seed {seed} regenerated E does not match the pack")
        rho, r = cavity_loads(n, clause_vars_of(cl))
        out.append(
            {
                "seed": int(seed),
                "E": torch.as_tensor(Eg, dtype=torch.float32),
                "r": r,
                "rho": rho,
            }
        )
    return out


def load_train(n: int, B: int, packs: Path) -> list[dict]:
    d = np.load(packs / f"train_n{n}.npz")
    E = np.asarray(d["train_E"][:B])
    seeds = np.asarray(d["train_seeds"][:B])
    if len(seeds) < B:
        raise RuntimeError(f"train n{n} has {len(seeds)} < B={B}")
    log(f"train n{n} B={B} seeds {int(seeds[0])}..{int(seeds[-1])}")
    return instances_from_seeds(n, seeds, E)


def r_stats(pack: list[dict]) -> dict:
    cv = np.array([float(inst["r"].std()) for inst in pack], dtype=np.float64)
    spread = np.array(
        [float(inst["r"].max() - inst["r"].min()) for inst in pack], dtype=np.float64
    )
    return {
        "median_cv": float(np.median(cv)),
        "median_r_span": float(np.median(spread)),
        "min_r": float(min(inst["r"].min() for inst in pack)),
        "max_r": float(max(inst["r"].max() for inst in pack)),
    }


def build_fresh_core(out_dir: Path, ns, B: int, base: int) -> dict:
    """Satisfiable CORE packs. Seeds determine the formulas; E is stored locally."""
    pack_dir = out_dir / "core_packs"
    pack_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "base": int(base),
        "B": int(B),
        "ns": {},
        "note": (
            "Fresh CORE for the LN-LA claim. Disjoint from fresh250 train "
            "(51e9 sealed subsample) and from scale1000 base 54e9."
        ),
    }
    for n in ns:
        path = pack_dir / f"core_n{n}.npz"
        if path.exists():
            d = np.load(path)
            seeds = np.asarray(d["seeds"])
            if int(d["base"]) == int(base) and len(seeds) >= B:
                manifest["ns"][str(n)] = {
                    "path": str(path),
                    "N": int(len(seeds)),
                    "reused": True,
                    "seed_min": int(seeds.min()),
                    "seed_max": int(seeds.max()),
                    "seed_sum": int(np.asarray(seeds, dtype=np.int64).sum()),
                    "sha256_E": str(d["sha256_E"]) if "sha256_E" in d.files else "",
                }
                log(f"reuse core n{n} N={len(seeds)} @ {path.name}")
                continue
        t0 = time.time()
        log(f"generate core n{n} B={B} base={base}")
        drawn = instances(int(n), int(B), int(base), compute_E=True)
        seeds = np.asarray([rec["seed"] for rec in drawn], dtype=np.int64)
        E = np.stack([np.asarray(rec["E"], dtype=np.int16) for rec in drawn], axis=0)
        m = clause_count(int(n))
        cv, cs = pack_clauses([rec["cl"] for rec in drawn], m, int(n))
        digest = hashlib.sha256(np.ascontiguousarray(E).tobytes()).hexdigest()
        np.savez_compressed(
            path,
            n=np.int32(n),
            m=np.int32(m),
            base=np.int64(base),
            seeds=seeds,
            E=E,
            cls_v=cv,
            cls_s=cs,
            sha256_E=digest,
        )
        manifest["ns"][str(n)] = {
            "path": str(path),
            "N": int(len(seeds)),
            "m": int(m),
            "reused": False,
            "seed_min": int(seeds.min()),
            "seed_max": int(seeds.max()),
            "seed_sum": int(seeds.sum()),
            "sha256_E": digest,
            "wall_s": time.time() - t0,
        }
        log(f"wrote {path.name} sha256_E={digest[:12]} ({time.time() - t0:.1f}s)")
    manifest["finished"] = time.strftime("%Y-%m-%d %H:%M:%S %z")
    (pack_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def load_core(out_dir: Path, n: int, B: int) -> list[dict]:
    path = out_dir / "core_packs" / f"core_n{n}.npz"
    d = np.load(path)
    seeds = np.asarray(d["seeds"][:B])
    E = np.asarray(d["E"][:B])
    cv = np.asarray(d["cls_v"][:B])
    out = []
    for i in range(len(seeds)):
        rho, r = cavity_loads(n, cv[i])
        out.append(
            {
                "seed": int(seeds[i]),
                "E": torch.as_tensor(E[i], dtype=torch.float32),
                "r": r,
                "rho": rho,
            }
        )
    return out


def assert_disjoint(train_seeds: dict[int, set[int]], core_dir: Path, ns) -> None:
    for n in ns:
        d = np.load(core_dir / "core_packs" / f"core_n{n}.npz")
        core = set(map(int, np.asarray(d["seeds"]).tolist()))
        for tn, ts in train_seeds.items():
            inter = core & ts
            if inter:
                raise RuntimeError(f"core n{n} overlaps train n{tn}: {list(inter)[:5]}")
        # scale1000, when the pack is on disk
        for scale in (
            S.SCALE1000_DIR / f"data_n{n}.npz",
            _ROOT / "explore_OCT29" / "data" / "scale1000" / f"data_n{n}.npz",
        ):
            if not scale.exists():
                continue
            sd = np.load(scale)
            key = "scale1000_seeds" if "scale1000_seeds" in sd.files else None
            if key is None:
                continue
            inter = core & set(map(int, np.asarray(sd[key]).tolist()))
            if inter:
                raise RuntimeError(f"core n{n} overlaps {scale}: {list(inter)[:5]}")
    log("fresh CORE seeds disjoint from train and from any scale1000 on disk")


def load_angles(path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    data = json.loads(Path(path).read_text())
    best = data["best"]
    betas = torch.tensor(best["betas"], dtype=torch.float32)
    gammas = torch.tensor(best["gammas"], dtype=torch.float32)
    return betas, gammas


def eval_qaoa(betas, gammas, pack: list[dict], n: int) -> dict:
    E = torch.stack([inst["E"] for inst in pack], dim=0)
    chunk = 32 if n <= 12 else 8
    invs = []
    t0 = time.time()
    for s0 in range(0, E.shape[0], chunk):
        ps = qaoa_psucc(betas, gammas, E[s0 : s0 + chunk], int(n))
        invs.append(inv_p_from_psucc(ps.detach().cpu().numpy()))
    return {**summarize(np.concatenate(invs)), "wall_s": time.time() - t0}


@torch.no_grad()
def eval_lnla(sim, pack, gamma, phi, alpha: float, beta: float, *, log_every: int = 0) -> dict:
    invs = []
    leaks = []
    t0 = time.time()
    B = len(pack)
    for i, inst in enumerate(pack):
        out = forward_lnla(sim, inst["E"], gamma, phi, inst["r"], alpha, beta)
        invs.append(float(out["inv_p"].cpu()))
        leaks.append(float(out["p_leak"].cpu()))
        if log_every and (i + 1) % log_every == 0:
            log(
                f"  n{sim.cfg.n} {i + 1}/{B} Med≈{np.median(invs):.4f} "
                f"α={alpha:.2f} β={beta:.2f}"
            )
    stats = summarize(np.asarray(invs), np.asarray(leaks))
    stats["wall_s"] = time.time() - t0
    return stats


def closer_to_origin(a: float, b: float, a2: float, b2: float) -> bool:
    da, db = abs(a) + abs(b), abs(a2) + abs(b2)
    if da < db - 1e-15:
        return True
    if abs(da - db) <= 1e-15 and abs(a) < abs(a2) - 1e-15:
        return True
    return False


def consider(best: dict | None, cell: dict) -> dict:
    if best is None or cell["score"] < best["score"] - 1e-15:
        return cell
    if abs(cell["score"] - best["score"]) <= 1e-15 and closer_to_origin(
        cell["alpha"], cell["beta"], best["alpha"], best["beta"]
    ):
        return cell
    return best


def on_kappa(alpha: float, beta: float, kappa: float, tol: float = 1e-8) -> bool:
    return abs(beta - kappa * alpha) <= tol


def run_grid(L: int, champ, train: dict[int, list[dict]], alphas, betas, device) -> dict:
    angles = {}
    sims = {}
    for n, pack in train.items():
        gamma, phi, a_n = angles_from_champ(champ, n, device)
        if int(gamma.numel()) != L:
            raise RuntimeError(f"champ L={gamma.numel()} != {L}")
        angles[n] = (gamma, phi, a_n)
        sims[n] = make_sim(n, L, device)
        log(f"L={L} n{n} a_n={a_n:.6f} train r {r_stats(pack)}")

    rows = []
    best = None
    best_by_kappa: dict[float, dict | None] = {k: None for k in TIED_KAPPAS}
    t0 = time.time()
    ncell = len(alphas) * len(betas)
    done = 0
    for alpha in alphas:
        for beta in betas:
            per_n = {}
            for n, pack in train.items():
                gamma, phi, _a = angles[n]
                per_n[n] = eval_lnla(sims[n], pack, gamma, phi, float(alpha), float(beta))
            cell = {
                "alpha": float(alpha),
                "beta": float(beta),
                "Med6": per_n[6]["Med"],
                "Med9": per_n[9]["Med"],
                "B90_6": per_n[6]["B90"],
                "B90_9": per_n[9]["B90"],
                "score": train_score(per_n[6]["Med"], per_n[9]["Med"]),
                "mean_leak6": per_n[6]["mean_leak"],
                "mean_leak9": per_n[9]["mean_leak"],
            }
            rows.append(cell)
            best = consider(best, cell)
            for kappa in TIED_KAPPAS:
                if on_kappa(cell["alpha"], cell["beta"], kappa):
                    best_by_kappa[kappa] = consider(best_by_kappa[kappa], cell)
            done += 1
            if done % len(betas) == 0:
                log(
                    f"L={L} grid {done}/{ncell} α={alpha:.2f} "
                    f"best score={best['score']:.4f} at ({best['alpha']:.2f},{best['beta']:.2f}) "
                    f"({time.time() - t0:.0f}s)"
                )
    score00 = next(c["score"] for c in rows if c["alpha"] == 0.0 and c["beta"] == 0.0)
    med00 = {
        6: next(c["Med6"] for c in rows if c["alpha"] == 0.0 and c["beta"] == 0.0),
        9: next(c["Med9"] for c in rows if c["alpha"] == 0.0 and c["beta"] == 0.0),
    }

    def pack_best(cell: dict | None) -> dict | None:
        if cell is None:
            return None
        gain = (score00 - cell["score"]) / score00
        return {**cell, "train_gain": gain, "train_gain_pct": 100.0 * gain}

    tied_cells = [c for c in rows if any(on_kappa(c["alpha"], c["beta"], k) for k in TIED_NONZERO)]
    best_tied = None
    for c in tied_cells:
        best_tied = consider(best_tied, c)
    return {
        "L": L,
        "alphas": [float(a) for a in alphas],
        "betas": [float(b) for b in betas],
        "cells": rows,
        "best_free": pack_best(best),
        "best_tied_nonzero_kappa": pack_best(best_tied),
        "best_by_kappa": {str(k): pack_best(v) for k, v in best_by_kappa.items()},
        "champ_cell": {"score": score00, "Med6": med00[6], "Med9": med00[9]},
        "a_n": {str(n): angles[n][2] for n in angles},
        "r_train": {str(n): r_stats(pack) for n, pack in train.items()},
        "wall_s": time.time() - t0,
        "score_def": "(1*Med6 + 2*Med9) / 3",
    }


def write_heatmap(path: Path, grid: dict) -> None:
    alphas = grid["alphas"]
    betas = grid["betas"]
    by = {(c["alpha"], c["beta"]): c for c in grid["cells"]}
    s00 = grid["champ_cell"]["score"]
    gains = []
    for a in alphas:
        for b in betas:
            gains.append((s00 - by[(a, b)]["score"]) / s00)
    gmax = max(abs(min(gains)), abs(max(gains)), 1e-6)
    cell = 58
    pad_l, pad_b, pad_t, pad_r = 54, 36, 28, 12
    W = pad_l + cell * len(alphas) + pad_r
    H = pad_t + cell * len(betas) + pad_b
    best = grid["best_free"]
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" '
        f'font-family="ui-sans-serif, sans-serif">',
        f'<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{pad_l}" y="18" font-size="13">L={grid["L"]} train score gain vs α=β=0 '
        f'(green = lower Med)</text>',
    ]
    for ib, b in enumerate(betas):
        # β = +1 at the top
        y = pad_t + (len(betas) - 1 - ib) * cell
        parts.append(
            f'<text x="{pad_l - 8}" y="{y + cell / 2 + 4}" font-size="11" '
            f'text-anchor="end">{b:.2f}</text>'
        )
        for ia, a in enumerate(alphas):
            x = pad_l + ia * cell
            c = by[(a, b)]
            g = (s00 - c["score"]) / s00
            t = max(-1.0, min(1.0, g / gmax))
            if t >= 0:
                # white → green
                red = int(255 * (1 - 0.55 * t))
                green = 255
                blue = int(255 * (1 - 0.55 * t))
            else:
                red = 255
                green = int(255 * (1 + 0.55 * t))
                blue = int(255 * (1 + 0.55 * t))
            parts.append(
                f'<rect x="{x}" y="{y}" width="{cell - 2}" height="{cell - 2}" '
                f'fill="rgb({red},{green},{blue})" stroke="#ddd"/>'
            )
            mark = ""
            if abs(a - best["alpha"]) < 1e-9 and abs(b - best["beta"]) < 1e-9:
                mark = "*"
            parts.append(
                f'<text x="{x + (cell - 2) / 2}" y="{y + cell / 2 + 4}" font-size="10" '
                f'text-anchor="middle">{c["score"]:.3f}{mark}</text>'
            )
    for ia, a in enumerate(alphas):
        x = pad_l + ia * cell + (cell - 2) / 2
        y = pad_t + cell * len(betas) + 16
        parts.append(f'<text x="{x}" y="{y}" font-size="11" text-anchor="middle">{a:.2f}</text>')
    parts.append(
        f'<text x="{pad_l}" y="{H - 6}" font-size="11">α →    β ↑    * = best free cell</text>'
    )
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n")


def gain_frac(champ_med: float, other_med: float) -> float:
    return (champ_med - other_med) / champ_med


def mark_lower(snap: float, ref: float) -> str:
    if snap < ref:
        return "BEAT"
    if snap == ref:
        return "TIE"
    return "UNDER"


def _core_row(core_rows: dict, n: int) -> dict:
    if n in core_rows:
        return core_rows[n]
    return core_rows[str(n)]


def verdict_for(alpha: float, beta: float, gain_n15: float, core_rows: dict, qaoa_key: str) -> dict:
    """gain_n15 is a fraction (0.005 = 0.5%)."""
    param_kill = abs(alpha) < 0.05 and abs(beta) < 0.05
    gain_kill = gain_n15 < 0.005
    beats_q = all(
        _core_row(core_rows, n)["lnla"]["Med"] < _core_row(core_rows, n)[qaoa_key]["Med"]
        for n in CORE_NS
    )
    beats_champ_all = all(
        _core_row(core_rows, n)["lnla"]["Med"] < _core_row(core_rows, n)["champ"]["Med"]
        for n in CORE_NS
    )
    reasons = []
    if param_kill:
        reasons.append("grid optimum has |α|<0.05 and |β|<0.05")
    if gain_kill:
        reasons.append(f"n15 Med gain {100.0 * gain_n15:.3f}% < 0.5%")
    if param_kill or gain_kill:
        verdict = "kill"
    elif beats_q and gain_n15 >= 0.005:
        verdict = "ship"
    else:
        verdict = "needs edit"
        if not beats_q:
            reasons.append("CORE Med is not below QAOA on every n in {6,9,12,15}")
        if not beats_champ_all:
            reasons.append("CORE Med is not below the champ on every n")
    return {
        "verdict": verdict,
        "reasons": reasons,
        "param_kill": param_kill,
        "gain_kill": gain_kill,
        "n15_gain": gain_n15,
        "n15_gain_pct": 100.0 * gain_n15,
        "beats_qaoa_all_core": beats_q,
        "beats_champ_all_core": beats_champ_all,
        "alpha": alpha,
        "beta": beta,
    }


def run_core_arm(L, champ, packs, alpha, beta, device) -> dict:
    per = {}
    for n in CORE_NS:
        gamma, phi, a_n = angles_from_champ(champ, n, device)
        sim = make_sim(n, L, device)
        log(f"CORE L={L} n{n} α={alpha:.2f} β={beta:.2f} a_n={a_n:.4f} B={len(packs[n])}")
        every = 250 if n >= 12 else 0
        per[n] = eval_lnla(sim, packs[n], gamma, phi, alpha, beta, log_every=every)
        per[n]["a_n"] = a_n
        log(f"  Med={per[n]['Med']:.4f} B90={per[n]['B90']:.4f} leak={per[n]['mean_leak']:.2e}")
    return {str(n): per[n] for n in CORE_NS}


def md_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    out = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    for row in rows:
        out.append("| " + " | ".join(row) + " |")
    return out


def render_results(payload: dict) -> str:
    lines = [
        "# LN-LA smoke — load-normalized graded H_c × load-adaptive cavity SNAP",
        "",
        "Opus Idea 2. Label: **graded H_c** (positive Walsh weights from cavity load).",
        "Angles γ, c1–c3, p stay frozen. The grid only moves α and β.",
        "",
        "```",
        "ρ_j = clause incidences on the 3 bits of cavity j",
        "r_j = ρ_j / mean_j ρ_j",
        "γ_S = γ_ℓ · Π_{j in cav(S)} r_j^(−α / |cav(S)|)",
        "φ_k^(j) = r_j^β · [c1 k + a_n (c2 k² + c3 k³)]",
        "a_n = (8/(n−1))^p",
        "α = β = 0  →  frozen power_p champ",
        "```",
        "",
        f"Train grid: fresh250 n∈{{6,9}}, first B={payload['train_B']}, "
        f"score (Med6 + 2 Med9) / 3, grid {payload['grid_n']}×{payload['grid_n']} on [−1, 1].",
        f"Decisive CORE: fresh packs base {payload['core_base']}, "
        f"n∈{{6,9,12,15}}, B={payload['core_B']}. Not the train split.",
        "",
        "## Frozen champs",
        "",
    ]
    lines += md_table(
        ["L", "checkpoint", "p", "note"],
        [
            [
                str(L),
                f"`{meta['path']}`",
                f"{meta['p']:.4f}",
                meta["note"],
            ]
            for L, meta in payload["champs"].items()
        ],
    )
    lines += ["", "## Train grid", ""]
    for L, grid in payload["grids"].items():
        best = grid["best_free"]
        champ = grid["champ_cell"]
        lines += [
            f"### L={L}",
            "",
            f"Champ cell α=β=0: score {champ['score']:.4f} "
            f"(Med6 {champ['Med6']:.4f}, Med9 {champ['Med9']:.4f}).",
            f"Best free cell: α={best['alpha']:.2f}, β={best['beta']:.2f}, "
            f"score {best['score']:.4f}, train gain {best['train_gain_pct']:.3f}%.",
            "",
            "Weighted train score (lower is better). `*` marks the selected cell.",
            "",
        ]
        alphas = grid["alphas"]
        betas = grid["betas"]
        by = {(c["alpha"], c["beta"]): c for c in grid["cells"]}
        header = ["β \\\\ α"] + [f"{a:.2f}" for a in alphas]
        body = []
        for b in reversed(betas):
            row = [f"{b:.2f}"]
            for a in alphas:
                c = by[(a, b)]
                star = "*" if abs(a - best["alpha"]) < 1e-9 and abs(b - best["beta"]) < 1e-9 else ""
                row.append(f"{c['score']:.4f}{star}")
            body.append(row)
        lines += md_table(header, body)
        lines += ["", f"Heatmap: `grid_L{L}.svg`.", ""]
        lines += ["Tied lines β = κα (best cell on that line):", ""]
        tied_rows = []
        for k, cell in grid["best_by_kappa"].items():
            if cell is None:
                continue
            tied_rows.append(
                [
                    k,
                    f"{cell['alpha']:.2f}",
                    f"{cell['beta']:.2f}",
                    f"{cell['score']:.4f}",
                    f"{cell['train_gain_pct']:.3f}%",
                ]
            )
        lines += md_table(["κ", "α", "β", "score", "train gain vs champ"], tied_rows)
        bt = grid["best_tied_nonzero_kappa"]
        if bt is not None:
            lines += [
                "",
                f"Best nonzero-κ tie: α={bt['alpha']:.2f}, β={bt['beta']:.2f}, "
                f"train gain {bt['train_gain_pct']:.3f}%.",
            ]
        lines.append("")

    lines += ["## Fresh CORE", ""]
    for L, block in payload["cores"].items():
        lines += [f"### L={L}", ""]
        qkey = block["qaoa_key"]
        headers = ["n", "LN-LA Med", "champ Med", "gain", "QAOA Med", "vs champ", "vs QAOA", "LN-LA B90", "champ B90", "QAOA B90"]
        body = []
        for n in CORE_NS:
            row = block["per_n"][str(n)]
            ln = row["lnla"]["Med"]
            ch = row["champ"]["Med"]
            q = row[qkey]["Med"]
            body.append(
                [
                    str(n),
                    f"{ln:.4f}",
                    f"{ch:.4f}",
                    f"{100.0 * gain_frac(ch, ln):.3f}%",
                    f"{q:.4f}",
                    mark_lower(ln, ch),
                    mark_lower(ln, q),
                    f"{row['lnla']['B90']:.4f}",
                    f"{row['champ']['B90']:.4f}",
                    f"{row[qkey]['B90']:.4f}",
                ]
            )
        lines += md_table(headers, body)
        v = block["verdict_free"]
        lines += [
            "",
            f"Free cell α={v['alpha']:.2f}, β={v['beta']:.2f}.",
            f"**Verdict (free): {v['verdict']}.** " + ("; ".join(v["reasons"]) if v["reasons"] else "Cleared the kill bars and beat QAOA Med on every CORE n."),
            "",
        ]
        if block.get("verdict_tied"):
            vt = block["verdict_tied"]
            lines += [
                f"Tied cell α={vt['alpha']:.2f}, β={vt['beta']:.2f}.",
                f"**Verdict (tied β=κα): {vt['verdict']}.** "
                + ("; ".join(vt["reasons"]) if vt["reasons"] else "Cleared the kill bars and beat QAOA Med on every CORE n."),
                "",
            ]
        if block.get("qaoa_historical_note"):
            lines += [block["qaoa_historical_note"], ""]
    lines += [
        "## Verdict",
        "",
        f"**{payload['overall']}**",
        "",
        payload["overall_why"],
        "",
        "Kill rule (Opus): both |α| and |β| < 0.05 at the selected cell, or n15 Med gain < 0.5% versus the frozen champ on this fresh CORE.",
        "Ship requires clearing that rule and Med < same-L QAOA Med on every n in {6,9,12,15}.",
        "",
    ]
    return "\n".join(lines) + "\n"


def ensure_l2_qaoa(packs: Path, out_dir: Path, device, threads: int) -> Path:
    """Train L=2 QAOA on fresh250 with the ladder recipe. The ladder skipped L=2."""
    dest = out_dir / "qaoa_L2_fresh250"
    dest.mkdir(parents=True, exist_ok=True)
    angles = dest / "angles.json"
    if angles.exists():
        log(f"reuse L=2 QAOA angles {angles}")
        return angles
    torch.set_num_threads(threads)
    blob = train_qaoa_train69(2, packs=packs, device=device, seed=0)
    angles.write_text(json.dumps(blob, indent=2) + "\n")
    log(f"trained L=2 QAOA on fresh250 loss={blob['best']['loss']:.4f}")
    return angles


def main():
    ap = argparse.ArgumentParser(description="LN-LA 9×9 grid + fresh CORE")
    ap.add_argument("--Ls", type=str, default="2,16")
    ap.add_argument("--grid", type=int, default=9)
    ap.add_argument("--train_B", type=int, default=100)
    ap.add_argument("--core_B", type=int, default=1000)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--out", type=str, default=str(DEFAULT_OUT))
    ap.add_argument("--packs", type=str, default=str(FRESH / "train_packs"))
    ap.add_argument("--core_base", type=int, default=CORE_BASE)
    ap.add_argument("--skip_grid", action="store_true")
    ap.add_argument("--skip_core", action="store_true")
    args = ap.parse_args()

    torch.set_num_threads(int(args.threads))
    device = torch.device("cpu")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    packs = Path(args.packs)
    Ls = tuple(int(x) for x in args.Ls.split(",") if x.strip())
    alphas = grid_values(args.grid)
    betas = grid_values(args.grid)

    champs = {}
    champ_meta = {}
    notes = {
        16: "fresh250 power_p seed0, free p, weights w6=1 w9=2",
        2: (
            "no fresh250 L=2 (ladder is 1,4,…,32); best-holdout power_p on the "
            "older train{6,9} packs, p fixed at 0.15, weights w6=0.25 w9=1"
        ),
    }
    for L in Ls:
        path = DEFAULT_CHAMP[L]
        state = torch.load(path, map_location="cpu", weights_only=False)
        if int(state["L"]) != L or state.get("idea") != "power_p":
            raise RuntimeError(f"unexpected champ at {path}: L={state.get('L')} idea={state.get('idea')}")
        champs[L] = state
        champ_meta[str(L)] = {
            "path": str(path),
            "p": float(state["p"]),
            "note": notes[L],
            "weights": {str(k): float(v) for k, v in dict(state.get("weights") or {}).items()},
        }
        log(f"freeze L={L} p={float(state['p']):.4f} from {path.name}")

    log("load train instances")
    train = {n: load_train(n, args.train_B, packs) for n in TRAIN_NS}
    train_seeds: dict[int, set[int]] = {}
    pack_files = [packs / f"train_n{n}.npz" for n in (6, 9, 12, 15)]
    pack_files += [
        S.OUT / "beat_qaoa_12h" / "train_packs" / f"train_n{n}.npz" for n in (6, 9, 12, 15)
    ]
    for tp in pack_files:
        if not tp.exists():
            continue
        d = np.load(tp)
        n = int(np.asarray(d["n"]).reshape(-1)[0]) if "n" in d.files else None
        if n is None:
            continue
        train_seeds.setdefault(n, set()).update(map(int, np.asarray(d["train_seeds"]).tolist()))
    for n, insts in train.items():
        train_seeds.setdefault(n, set()).update(inst["seed"] for inst in insts)

    grids = {}
    if not args.skip_grid:
        for L in Ls:
            log(f"=== grid L={L} ===")
            grids[L] = run_grid(L, champs[L], train, alphas, betas, device)
            (out / f"grid_L{L}.json").write_text(json.dumps(grids[L], indent=2) + "\n")
            write_heatmap(out / f"grid_L{L}.svg", grids[L])
            b = grids[L]["best_free"]
            log(
                f"L={L} best α={b['alpha']:.2f} β={b['beta']:.2f} "
                f"train gain {b['train_gain_pct']:.3f}% wall={grids[L]['wall_s']:.0f}s"
            )
    else:
        for L in Ls:
            grids[L] = json.loads((out / f"grid_L{L}.json").read_text())

    cores = {}
    if not args.skip_core:
        build_fresh_core(out, CORE_NS, args.core_B, args.core_base)
        assert_disjoint(train_seeds, out, CORE_NS)
        core_packs = {n: load_core(out, n, args.core_B) for n in CORE_NS}
        for n, pack in core_packs.items():
            log(f"core n{n} r {r_stats(pack)}")

        qaoa_paths = dict(DEFAULT_QAOA)
        if 2 in Ls:
            qaoa_paths[2] = ensure_l2_qaoa(packs, out, device, args.threads)
        qaoa_angles = {L: load_angles(qaoa_paths[L]) for L in Ls}
        hist_l2 = load_angles(HISTORICAL_QAOA_L2) if 2 in Ls else None

        for L in Ls:
            g = grids[L]
            free = g["best_free"]
            tied = g["best_tied_nonzero_kappa"]
            arms = {"champ": (0.0, 0.0), "lnla": (free["alpha"], free["beta"])}
            tied_is_origin = tied is not None and abs(tied["alpha"]) < 1e-12 and abs(tied["beta"]) < 1e-12
            tied_is_free = tied is not None and abs(tied["alpha"] - free["alpha"]) < 1e-9 and abs(
                tied["beta"] - free["beta"]
            ) < 1e-9
            if tied is not None and not tied_is_origin and not tied_is_free:
                arms["lnla_tied"] = (tied["alpha"], tied["beta"])
            arm_stats = {}
            for name, (a, b) in arms.items():
                # champ and a (0,0) free cell share one eval
                if name == "lnla" and abs(a) == 0.0 and abs(b) == 0.0 and "champ" in arm_stats:
                    arm_stats[name] = arm_stats["champ"]
                    continue
                arm_stats[name] = run_core_arm(L, champs[L], core_packs, a, b, device)

            betas_q, gammas_q = qaoa_angles[L]
            q_per = {}
            for n in CORE_NS:
                log(f"QAOA L={L} n{n} B={args.core_B}")
                q_per[n] = eval_qaoa(betas_q, gammas_q, core_packs[n], n)
                log(f"  QAOA Med={q_per[n]['Med']:.4f}")
            hist_per = None
            if L == 2 and hist_l2 is not None:
                hist_per = {}
                hb, hg = hist_l2
                for n in CORE_NS:
                    hist_per[n] = eval_qaoa(hb, hg, core_packs[n], n)
                    log(f"  historical n6-QAOA n{n} Med={hist_per[n]['Med']:.4f}")

            per_n = {}
            for n in CORE_NS:
                per_n[str(n)] = {
                    "lnla": arm_stats["lnla"][str(n)],
                    "champ": arm_stats["champ"][str(n)],
                    "qaoa": q_per[n],
                }
                if "lnla_tied" in arm_stats:
                    per_n[str(n)]["lnla_tied"] = arm_stats["lnla_tied"][str(n)]
                if hist_per is not None:
                    per_n[str(n)]["qaoa_n6_sealed"] = hist_per[n]

            gain15 = gain_frac(per_n["15"]["champ"]["Med"], per_n["15"]["lnla"]["Med"])
            vfree = verdict_for(free["alpha"], free["beta"], gain15, per_n, "qaoa")
            vtied = None
            if "lnla_tied" in arm_stats and tied is not None:
                # verdict_for reads row['lnla']; temporarily view the tied arm as lnla
                tied_view = {}
                for n in CORE_NS:
                    tied_view[n] = {
                        "lnla": per_n[str(n)]["lnla_tied"],
                        "champ": per_n[str(n)]["champ"],
                        "qaoa": per_n[str(n)]["qaoa"],
                    }
                gain_t = gain_frac(
                    per_n["15"]["champ"]["Med"], per_n["15"]["lnla_tied"]["Med"]
                )
                vtied = verdict_for(tied["alpha"], tied["beta"], gain_t, tied_view, "qaoa")

            note = ""
            if L == 2:
                note = (
                    "QAOA column is trained on the same fresh250 packs "
                    "(paper init, 200 Adam steps, batch 32+32, seed 0) because the "
                    "ladder has no L=2 QAOA. Historical n6-sealed L=2 QAOA Med on this "
                    "CORE: "
                    + ", ".join(
                        f"n{n} {per_n[str(n)]['qaoa_n6_sealed']['Med']:.4f}" for n in CORE_NS
                    )
                    + "."
                )
            elif L == 16:
                note = (
                    "QAOA column re-evaluates the frozen fresh250 L=16 angles "
                    f"(`{qaoa_paths[16]}`) on this fresh CORE, not the published scale1000 medians."
                )
            cores[str(L)] = {
                "per_n": per_n,
                "qaoa_key": "qaoa",
                "qaoa_angles": str(qaoa_paths[L]),
                "verdict_free": vfree,
                "verdict_tied": vtied,
                "qaoa_historical_note": note,
                "r_core": {str(n): r_stats(core_packs[n]) for n in CORE_NS},
            }
            (out / f"core_L{L}.json").write_text(json.dumps(cores[str(L)], indent=2) + "\n")
            log(f"L={L} verdict {vfree['verdict']} n15 gain {vfree['n15_gain_pct']:.3f}%")

    # reload cores if skip
    if args.skip_core:
        for L in Ls:
            cores[str(L)] = json.loads((out / f"core_L{L}.json").read_text())

    verdicts = [cores[str(L)]["verdict_free"]["verdict"] for L in Ls]
    if all(v == "ship" for v in verdicts):
        overall = "ship"
        why = "Both frozen depths cleared the kill rule and beat QAOA Med on fresh CORE."
    elif all(v == "kill" for v in verdicts):
        overall = "kill"
        why = "Both L=2 and L=16 hit a kill rule on the fresh CORE."
    else:
        overall = "needs edit"
        bits = [f"L={L} {cores[str(L)]['verdict_free']['verdict']}" for L in Ls]
        why = "Depths disagree (" + ", ".join(bits) + ")."
        for L in Ls:
            vt = cores[str(L)].get("verdict_tied")
            vf = cores[str(L)]["verdict_free"]["verdict"]
            if vt and vt["verdict"] == "ship" and vf != "ship":
                why += f" L={L} tied β=κα would ship while the free cell does not."

    payload = {
        "idea": "LN-LA",
        "label": "load_normalized_graded_Hc",
        "train_B": args.train_B,
        "core_B": args.core_B,
        "core_base": args.core_base,
        "grid_n": args.grid,
        "champs": champ_meta,
        "grids": {str(L): grids[L] for L in Ls},
        "cores": cores,
        "overall": overall,
        "overall_why": why,
        "finished": time.strftime("%Y-%m-%d %H:%M:%S %z"),
    }
    (out / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    text = render_results(payload)
    (out / "RESULTS.md").write_text(text)
    log(f"OVERALL {overall}")
    log(f"wrote {out / 'RESULTS.md'}")


if __name__ == "__main__":
    main()
