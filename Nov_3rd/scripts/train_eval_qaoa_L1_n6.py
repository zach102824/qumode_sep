#!/usr/bin/env python3
"""Retrain ideal qubit QAOA L=1 on n=6×1000 (fair vs Nov_3rd polish).

Init: Boulebnane–Montanaro arXiv:2208.06909 — β=0.01, γ=-0.01.

Does NOT overwrite Nov_3rd/qaoa_angles/L1/ (n9+n12 baseline).
Writes:
  Nov_3rd/qaoa_angles/L1_train_n6/angles.json
  Nov_3rd/qaoa_angles/L1_train_n6/results.json
  Nov_3rd/runs/QAOA_L1_n6_vs_n912.md
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from explore_OCT29.lib.drivers import fit_b  # noqa: E402
from explore_OCT29.lib.qaoa import (  # noqa: E402
    clause_count,
    inv_p_from_psucc,
    qaoa_psucc,
    summarize_inv,
)
from explore_OCT29.lib.train_loop import _split_arrays, load_pack  # noqa: E402
from Nov_3rd.qaoa_init import (  # noqa: E402
    ANGLE_CONVENTION,
    INIT_NOTES,
    QAOA_NSTART,
    make_starts,
)

PKG = Path(__file__).resolve().parents[1]
OUT = PKG / "qaoa_angles" / "L1_train_n6"
RUNS = PKG / "runs"
SEALED = PKG / "data" / "sealed"
SCALE = PKG / "data" / "scale1000"
OLD_L1_RESULTS = PKG / "qaoa_angles" / "L1" / "results.json"
OLD_L1_ANGLES = PKG / "qaoa_angles" / "L1" / "angles.json"

# Nov_3rd polished L=1 (run_final / debug) — scale1000 B=1000
NOV_POLISHED = {
    6: {"median_inv_p": 9.502, "B90_inv_p": 17.338},
    9: {"median_inv_p": 29.537, "B90_inv_p": 62.135},
    12: {"median_inv_p": 93.283, "B90_inv_p": 217.777},
    15: {"median_inv_p": 318.811, "B90_inv_p": 792.468},
}
# Warm-only (QAOA shell from OLD n9+n12 angles) — exact TIE to old QAOA
NOV_WARM = {
    6: {"median_inv_p": 9.779, "B90_inv_p": 19.191},
    9: {"median_inv_p": 29.279, "B90_inv_p": 63.429},
    12: {"median_inv_p": 90.033, "B90_inv_p": 216.627},
    15: {"median_inv_p": 289.893, "B90_inv_p": 750.515},
}


def log(*a):
    print(time.strftime("%H:%M:%S"), "[qaoa_L1_n6]", *a, flush=True)


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating, float)):
        x = float(o)
        return x
    if isinstance(o, (np.integer, int)):
        return int(o)
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, torch.Tensor):
        return o.detach().cpu().tolist()
    if o is None:
        return None
    return o


def _S(med, b90):
    if b90 is None or not np.isfinite(b90) or not np.isfinite(med):
        return float("nan")
    return 0.5 * (math.log(max(med, 1e-12)) + math.log(max(b90, 1e-12)))


def load_E_split(n: int, data_dir: Path, split: str, max_count: int | None):
    pack = load_pack(n, data_dir=data_dir)
    if f"{split}_seeds" not in pack:
        raise RuntimeError(f"n={n}: missing split {split} in {data_dir}")
    seeds, _, _, _, E = _split_arrays(pack, split, max_count=max_count)
    if E is None or len(seeds) == 0:
        raise RuntimeError(f"n={n}: no dense E for split={split}")
    return seeds, np.asarray(E, dtype=np.float32)


def train_angles(
    L: int,
    train_ns: list[int],
    n_per: dict[int, int],
    data_dir: Path,
    split: str,
    device: torch.device,
    nstart: int,
    steps: int,
    lr: float,
    seed: int,
    chunk: int,
) -> dict:
    rng = np.random.default_rng(seed)
    bundles = {}
    for n in train_ns:
        seeds, E = load_E_split(n, data_dir, split, n_per.get(n))
        bundles[n] = torch.as_tensor(E, dtype=torch.float32, device=device)
        log(f"train load n={n} B={E.shape[0]} dim={E.shape[1]}")

    starts = make_starts(L, nstart, rng)
    best = None
    hist_all = []

    for si, x0 in enumerate(starts):
        betas = torch.nn.Parameter(torch.tensor(x0[:L], dtype=torch.float32, device=device))
        gammas = torch.nn.Parameter(torch.tensor(x0[L:], dtype=torch.float32, device=device))
        opt = torch.optim.Adam([betas, gammas], lr=lr)
        t0 = time.time()
        hist = []
        for step in range(steps):
            opt.zero_grad()
            losses = []
            for n, Efull in bundles.items():
                B = Efull.shape[0]
                acc = []
                for s in range(0, B, chunk):
                    Eb = Efull[s : s + chunk]
                    ps = qaoa_psucc(betas, gammas, Eb, n)
                    acc.append(-torch.log(ps.clamp_min(1e-12)).mean())
                losses.append(torch.stack(acc).mean())
            loss = torch.stack(losses).mean()
            loss.backward()
            opt.step()
            hist.append(float(loss.detach().cpu()))
            if step % 20 == 0 or step == steps - 1:
                log(
                    f"start {si+1}/{len(starts)} step {step:3d} "
                    f"loss={hist[-1]:.4f} β={betas.detach().cpu().tolist()} "
                    f"γ={gammas.detach().cpu().tolist()} wall={time.time()-t0:.0f}s"
                )
        final = hist[-1]
        payload = {
            "start_idx": si,
            "loss": final,
            "betas": betas.detach().cpu().numpy().tolist(),
            "gammas": gammas.detach().cpu().numpy().tolist(),
            "history": hist,
            "wall_s": time.time() - t0,
        }
        hist_all.append({"start_idx": si, "loss": final, "wall_s": payload["wall_s"]})
        if best is None or final < best["loss"]:
            best = payload
            log(f"new best start={si} loss={final:.4f}")

    return {
        "L": L,
        "train_ns": train_ns,
        "n_per": {str(k): int(v) for k, v in n_per.items()},
        "split": split,
        "data_dir": str(data_dir),
        "nstart": nstart,
        "steps": steps,
        "lr": lr,
        "seed": seed,
        "best": best,
        "all_starts": hist_all,
        "angle_convention": ANGLE_CONVENTION,
        "init_notes": INIT_NOTES,
        "purpose": "fair baseline: same train budget as Nov_3rd (n=6×1000)",
    }


@torch.no_grad()
def eval_n(
    n: int,
    betas: torch.Tensor,
    gammas: torch.Tensor,
    data_dir: Path,
    split: str,
    max_count: int,
    device: torch.device,
    chunk: int,
    progress_every: int,
):
    seeds, E = load_E_split(n, data_dir, split, max_count)
    B = len(seeds)
    invs = []
    t0 = time.time()
    for s in range(0, B, chunk):
        Eb = torch.as_tensor(E[s : s + chunk], dtype=torch.float32, device=device)
        ps = qaoa_psucc(betas, gammas, Eb, n).cpu().numpy()
        invs.extend(inv_p_from_psucc(ps).tolist())
        done = min(s + chunk, B)
        if done % progress_every < chunk or done == B:
            med = float(np.median(invs))
            rate = done / max(time.time() - t0, 1e-9)
            eta = (B - done) / max(rate, 1e-9)
            log(f"eval n={n} {done}/{B} Med≈{med:.3f} rate={rate:.2f}/s ETA={eta/60:.1f}min")
    summ = summarize_inv(invs)
    summ.update(
        {
            "n": n,
            "split": split,
            "L": int(betas.numel()),
            "L_times_Med": summ["median_inv_p"] * int(betas.numel())
            if np.isfinite(summ["median_inv_p"])
            else float("inf"),
            "S": _S(summ["median_inv_p"], summ["B90_inv_p"]),
            "wall_s": time.time() - t0,
            "m": clause_count(n),
        }
    )
    return summ, np.asarray(invs, dtype=np.float64)


def gate_label(r: float | None) -> str:
    if r is None or not np.isfinite(r):
        return "—"
    if r < 0.98:
        return "BEAT"
    if r <= 1.02:
        return "TIE"
    return "UNDER"


def write_compare_md(
    path: Path,
    meta: dict,
    ladder: dict,
    old_ladder: dict,
    old_angles: dict,
):
    ns = sorted(set(list(ladder.keys()) + list(old_ladder.keys()) + list(NOV_POLISHED.keys())))
    lines = [
        "# QAOA L=1 — n6×1000 train vs n9+n12 train vs Nov_3rd",
        "",
        f"_Generated {time.strftime('%Y-%m-%d %H:%M:%S %z')} (Asia/Shanghai)_",
        "",
        "**Question:** Is Nov_3rd polished L=1 UNDER only because QAOA was trained on n9+n12?",
        "",
        f"- **New QAOA:** train n=6×{meta['n_per'].get(6) or meta['n_per'].get('6')}, "
        f"Adam from arXiv:2208.06909 init β=0.01 γ=-0.01 (nstart={meta.get('nstart')}, steps={meta.get('steps')}, lr={meta.get('lr')})",
        f"- **Old QAOA:** train n9+n12 (200+128) — `Nov_3rd/qaoa_angles/L1/` "
        f"(β={old_angles.get('betas')}, γ={old_angles.get('gammas')})",
        f"- **New angles:** β={meta.get('betas')}, γ={meta.get('gammas')} "
        f"(train loss={meta.get('train_loss'):.4f})",
        f"- **Nov warm:** QAOA-shell from OLD angles (exact TIE to old QAOA on n9/12/15)",
        f"- **Nov polished:** free LBFGS on n6×1000 (params_best_L1.pt)",
        "",
        "## Med / B90 / R (R = method / QAOA_n6)",
        "",
        "| n | QAOA_n6 Med | QAOA_n6 B90 | QAOA_n912 Med | QAOA_n912 B90 | "
        "warm Med | polished Med | polished B90 | "
        "R_polished/n6 | gate | R_n912/n6 | R_warm/n6 |",
        "|---|-------------|-------------|---------------|---------------|"
        "----------|--------------|--------------|"
        "---------------|------|-----------|-----------|",
    ]
    for n in ns:
        n = int(n)
        q6 = ladder.get(n) or ladder.get(str(n)) or {}
        q912 = old_ladder.get(n) or old_ladder.get(str(n)) or {}
        warm = NOV_WARM.get(n, {})
        pol = NOV_POLISHED.get(n, {})
        m6 = q6.get("median_inv_p")
        b6 = q6.get("B90_inv_p")
        m912 = q912.get("median_inv_p")
        b912 = q912.get("B90_inv_p")
        mw = warm.get("median_inv_p")
        mp = pol.get("median_inv_p")
        bp = pol.get("B90_inv_p")

        def fmt(x, d=3):
            try:
                x = float(x)
                if not np.isfinite(x):
                    return "—"
                return f"{x:.{d}f}"
            except Exception:
                return "—"

        def ratio(a, b):
            try:
                return float(a) / float(b)
            except Exception:
                return None

        r_p = ratio(mp, m6)
        r_912 = ratio(m912, m6)
        r_w = ratio(mw, m6)
        lines.append(
            f"| {n} | {fmt(m6)} | {fmt(b6)} | {fmt(m912)} | {fmt(b912)} | "
            f"{fmt(mw)} | {fmt(mp)} | {fmt(bp)} | "
            f"{fmt(r_p, 3)+'×' if r_p else '—'} | {gate_label(r_p)} | "
            f"{fmt(r_912, 3)+'×' if r_912 else '—'} | "
            f"{fmt(r_w, 3)+'×' if r_w else '—'} |"
        )

    lines += [
        "",
        "## Angle comparison",
        "",
        f"| source | train | β | γ |",
        f"|--------|-------|---|---|",
        f"| QAOA n9+n12 | sealed train 200+128 | {old_angles.get('betas')} | {old_angles.get('gammas')} |",
        f"| QAOA n6×1000 | sealed train 1000 | {meta.get('betas')} | {meta.get('gammas')} |",
        "",
        "## Notes",
        "",
        "- Gate: BEAT if R<0.98, TIE if 0.98≤R≤1.02, UNDER if R>1.02 (same as Nov_3rd).",
        "- R_polished/n6 uses **n6-trained QAOA** as denominator (fair train budget).",
        "- QAOA_n6 init: Boulebnane–Montanaro arXiv:2208.06909 — β=0.01, γ=-0.01 (single start).",
        "- Warm uses OLD n9+n12 angles mapped into Nov shell; remapping warm to n6 angles "
        "not re-eval'd here (qubit QAOA_n6 column is the fair remapped baseline).",
        "- Old L1 angles/results left untouched under `qaoa_angles/L1/`.",
        "",
    ]
    path.write_text("\n".join(lines))
    log(f"wrote {path}")


def main():
    os.environ.setdefault("OMP_NUM_THREADS", "8")
    os.environ.setdefault("MKL_NUM_THREADS", "8")
    torch.set_num_threads(8)

    OUT.mkdir(parents=True, exist_ok=True)
    RUNS.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")
    L = 1
    train_ns = [6]
    n_per = {6: 1000}
    # Paper init (arXiv:2208.06909 §Parameter optimization): nstart=1; Adam steps/lr kept.
    nstart, steps, lr, seed = QAOA_NSTART, 200, 0.08, 0
    eval_ns = [6, 9, 12, 15, 18]
    B = 1000

    log("=== train QAOA L=1 on n=6×1000 ===")
    train_blob = train_angles(
        L, train_ns, n_per, SEALED, "train", device, nstart, steps, lr, seed, chunk=64
    )
    angles_path = OUT / "angles.json"
    angles_path.write_text(json.dumps(_jsonable(train_blob), indent=2))
    log(f"wrote {angles_path} best_loss={train_blob['best']['loss']:.4f}")

    betas = torch.tensor(train_blob["best"]["betas"], dtype=torch.float32, device=device)
    gammas = torch.tensor(train_blob["best"]["gammas"], dtype=torch.float32, device=device)

    meta = {
        "L": L,
        "train_ns": train_ns,
        "n_per": {str(k): v for k, v in n_per.items()},
        "train_split": "train",
        "train_data_dir": str(SEALED),
        "eval_split": "scale1000",
        "eval_data_dir": str(SCALE),
        "B": B,
        "nstart": nstart,
        "steps": steps,
        "lr": lr,
        "seed": seed,
        "betas": train_blob["best"]["betas"],
        "gammas": train_blob["best"]["gammas"],
        "train_loss": train_blob["best"]["loss"],
        "angles_path": str(angles_path),
        "device": str(device),
        "started": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        "angle_convention": train_blob.get("angle_convention"),
        "purpose": "fair vs Nov_3rd (same n=6×1000 train budget)",
    }

    ladder = {}
    for n in eval_ns:
        chunk = 64 if n <= 9 else (16 if n <= 12 else (4 if n <= 15 else 1))
        log(f"=== eval n={n} B={B} chunk={chunk} ===")
        summ, invs = eval_n(n, betas, gammas, SCALE, "scale1000", B, device, chunk, 25 if n <= 12 else 50)
        ladder[n] = summ
        np.save(OUT / f"inv_p_n{n}.npy", invs)
        log(
            f"DONE n={n}: Med={summ['median_inv_p']:.3f} B90={summ['B90_inv_p']:.3f} "
            f"B={summ['B']} wall={summ['wall_s']:.1f}s"
        )

    # b fit 9→18
    ns_fit = [n for n in (9, 12, 15, 18) if n in ladder]
    meds = [ladder[n]["median_inv_p"] for n in ns_fit]
    b90s = [ladder[n]["B90_inv_p"] for n in ns_fit]
    fits = {
        "9_18": {
            "ns": ns_fit,
            "b_Med": fit_b(ns_fit, meds) if len(ns_fit) >= 2 else None,
            "b_B90": fit_b(ns_fit, b90s) if len(ns_fit) >= 2 else None,
        }
    }

    meta["finished"] = time.strftime("%Y-%m-%d %H:%M:%S %z")
    blob = {
        "meta": meta,
        "ladder": {str(k): v for k, v in sorted(ladder.items())},
        "fits": _jsonable(fits),
        "nov_polished_L1_ref": NOV_POLISHED,
        "nov_warm_L1_ref": NOV_WARM,
        "updated": meta["finished"],
    }
    (OUT / "results.json").write_text(json.dumps(_jsonable(blob), indent=2))
    log(f"wrote {OUT / 'results.json'}")

    old = json.loads(OLD_L1_RESULTS.read_text())
    old_ladder = {int(k): v for k, v in (old.get("ladder") or {}).items()}
    old_ang = json.loads(OLD_L1_ANGLES.read_text())
    old_best = old_ang.get("best") or {}
    old_angles_short = {"betas": old_best.get("betas"), "gammas": old_best.get("gammas")}

    write_compare_md(
        RUNS / "QAOA_L1_n6_vs_n912.md",
        meta,
        ladder,
        old_ladder,
        old_angles_short,
    )

    # also a RESULTS.md in OUT
    lines = [
        f"# QAOA L=1 trained n=6×1000 — scale1000",
        "",
        f"_Generated {meta['finished']} (Asia/Shanghai)_",
        "",
        f"**β**={meta['betas']}, **γ**={meta['gammas']}, loss={meta['train_loss']:.4f}",
        "",
        "| n | Med | B90 | B |",
        "|---|-----|-----|---|",
    ]
    for n in eval_ns:
        m = ladder[n]
        lines.append(f"| {n} | {m['median_inv_p']:.3f} | {m['B90_inv_p']:.3f} | {m['B']} |")
    (OUT / "RESULTS.md").write_text("\n".join(lines) + "\n")

    log("ALL DONE")
    for n in eval_ns:
        m6 = ladder[n]["median_inv_p"]
        mp = (NOV_POLISHED.get(n) or {}).get("median_inv_p")
        if mp:
            r = mp / m6
            log(f"compare n={n}: polished={mp:.3f} / QAOA_n6={m6:.3f} → R={r:.3f}× {gate_label(r)}")


if __name__ == "__main__":
    main()
