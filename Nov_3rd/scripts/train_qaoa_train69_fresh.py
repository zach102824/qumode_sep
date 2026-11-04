#!/usr/bin/env python3
"""Train qubit QAOA on fresh train69 packs (250 n6 + 250 n9) and eval scale1000.

Paper init β=0.01, γ=-0.01, nstart=1. Each Adam step: batch 32 n6 + 32 n9.
Writes under --out_root/L{L}/angles.json + results.json (does NOT touch
Nov_3rd/qaoa_angles paper-n6 by default).
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[2]
import sys

if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from Nov_3rd import settings as S
from Nov_3rd.data_io import log
from Nov_3rd.ensure_qaoa import _jsonable, eval_scale_ladder
from Nov_3rd.qaoa_init import (
    ANGLE_CONVENTION,
    INIT_NOTES,
    QAOA_NSTART,
    make_starts,
)
from explore_OCT29.lib.qaoa import inv_p_from_psucc, qaoa_psucc


def load_pack_E(packs: Path, n: int) -> torch.Tensor:
    d = np.load(packs / f"train_n{n}.npz", allow_pickle=True)
    return torch.as_tensor(np.asarray(d["train_E"]), dtype=torch.float32)


def train_qaoa_train69(
    L: int,
    *,
    packs: Path,
    device: torch.device,
    steps: int | None = None,
    lr: float | None = None,
    batch_per_n: int = 32,
    seed: int = 0,
) -> dict:
    L = int(L)
    if steps is None:
        steps = 200 if L <= 4 else (300 if L <= 16 else 400)
    if lr is None:
        lr = 0.08 if L <= 4 else 0.05
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    E6 = load_pack_E(packs, 6)
    E9 = load_pack_E(packs, 9)
    log(f"QAOA train69 L{L}: n6={len(E6)} n9={len(E9)} steps={steps} lr={lr} batch={batch_per_n}+{batch_per_n}")

    x0 = make_starts(L, QAOA_NSTART)[0]
    betas = torch.nn.Parameter(torch.tensor(x0[:L], dtype=torch.float32, device=device))
    gammas = torch.nn.Parameter(torch.tensor(x0[L:], dtype=torch.float32, device=device))
    opt = torch.optim.Adam([betas, gammas], lr=lr)
    t0 = time.time()
    hist = []
    for step in range(steps):
        i6 = rng.choice(len(E6), size=min(batch_per_n, len(E6)), replace=False)
        i9 = rng.choice(len(E9), size=min(batch_per_n, len(E9)), replace=False)
        Eb6 = E6[i6].to(device)
        Eb9 = E9[i9].to(device)
        opt.zero_grad()
        ps6 = qaoa_psucc(betas, gammas, Eb6, 6)
        ps9 = qaoa_psucc(betas, gammas, Eb9, 9)
        loss6 = -torch.log(ps6.clamp_min(1e-12)).mean()
        loss9 = -torch.log(ps9.clamp_min(1e-12)).mean()
        loss = 0.5 * (loss6 + loss9)
        loss.backward()
        opt.step()
        hist.append(float(loss.detach().cpu()))
        if step % 25 == 0 or step == steps - 1:
            log(
                f"QAOA train69 L{L} step {step:3d}/{steps} loss={hist[-1]:.4f} "
                f"wall={time.time()-t0:.0f}s"
            )
    best = {
        "start_idx": 0,
        "loss": hist[-1],
        "betas": betas.detach().cpu().tolist(),
        "gammas": gammas.detach().cpu().tolist(),
        "history": hist,
        "wall_s": time.time() - t0,
    }
    return {
        "L": L,
        "train_ns": [6, 9],
        "n_per": {"6": int(len(E6)), "9": int(len(E9))},
        "split": "train69_fresh250",
        "packs_dir": str(packs),
        "batch_per_n": batch_per_n,
        "nstart": QAOA_NSTART,
        "steps": steps,
        "lr": lr,
        "seed": seed,
        "best": best,
        "all_starts": [{"start_idx": 0, "loss": best["loss"], "wall_s": best["wall_s"]}],
        "angle_convention": ANGLE_CONVENTION,
        "init_notes": INIT_NOTES,
        "purpose": "fair baseline: train69 fresh 250 n6 + 250 n9 mix (batch 32+32), paper init",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=int, required=True)
    ap.add_argument("--packs", type=str, required=True)
    ap.add_argument("--out_root", type=str, required=True)
    ap.add_argument("--eval_ns", type=str, default="6,9,12,15")
    ap.add_argument("--batch_per_n", type=int, default=32)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    import os

    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    os.environ["MKL_NUM_THREADS"] = str(args.threads)
    torch.set_num_threads(args.threads)
    device = torch.device("cpu")

    L = int(args.L)
    packs = Path(args.packs)
    out = Path(args.out_root) / f"L{L}"
    out.mkdir(parents=True, exist_ok=True)
    angles_path = out / "angles.json"
    results_path = out / "results.json"
    ns = [int(x) for x in args.eval_ns.split(",") if x.strip()]

    if angles_path.exists() and not args.force:
        blob = json.loads(angles_path.read_text())
        log(f"QAOA train69 L{L}: reuse angles @ {angles_path}")
    else:
        blob = train_qaoa_train69(
            L,
            packs=packs,
            device=device,
            steps=args.steps,
            lr=args.lr,
            batch_per_n=args.batch_per_n,
            seed=args.seed,
        )
        angles_path.write_text(json.dumps(_jsonable(blob), indent=2) + "\n")
        log(f"QAOA train69 L{L}: wrote {angles_path} loss={blob['best']['loss']:.4f}")

    need_eval = args.force or not results_path.exists()
    if results_path.exists() and not args.force:
        try:
            existing = json.loads(results_path.read_text())
            ladder = {int(k): v for k, v in (existing.get("ladder") or {}).items()}
            need_eval = not all(n in ladder for n in ns)
        except Exception:
            need_eval = True

    if need_eval:
        betas = torch.tensor(blob["best"]["betas"], dtype=torch.float32, device=device)
        gammas = torch.tensor(blob["best"]["gammas"], dtype=torch.float32, device=device)
        if results_path.exists() and not args.force:
            old = json.loads(results_path.read_text())
            ladder = {int(k): v for k, v in (old.get("ladder") or {}).items()}
            missing = [n for n in ns if n not in ladder]
        else:
            ladder = {}
            missing = list(ns)
        missing = [n for n in missing if (S.SCALE1000_DIR / f"data_n{n}.npz").exists()]
        log(f"QAOA train69 L{L}: eval scale1000 ns={missing}")
        fresh = eval_scale_ladder(L, betas, gammas, device=device, ns=missing)
        ladder.update(fresh)
        meta = {
            "L": L,
            "train": "train69 fresh 250 n6 + 250 n9, batch 32+32, paper init β=0.01 γ=-0.01 nstart=1",
            "packs_dir": str(packs),
            "betas": blob["best"]["betas"],
            "gammas": blob["best"]["gammas"],
            "train_loss": blob["best"]["loss"],
            "nstart": QAOA_NSTART,
            "init_notes": INIT_NOTES,
            "angle_convention": ANGLE_CONVENTION,
            "finished": time.strftime("%Y-%m-%d %H:%M:%S %z"),
        }
        payload = {
            "meta": meta,
            "ladder": {str(k): v for k, v in sorted(ladder.items())},
            "updated": meta["finished"],
        }
        results_path.write_text(json.dumps(_jsonable(payload), indent=2) + "\n")
        log(f"QAOA train69 L{L}: wrote {results_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
