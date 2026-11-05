"""Train Hamming-shell SNAP angles with fidelity cost on quantum p-spin.

Joint train n∈{6,9}; also n6-only / n9-only for 1/n interpolation.
Loss = mean(1 - F). Optimizer: L-BFGS.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

PKG = Path(__file__).resolve().parent
ROOT = PKG.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(PKG))

from quantum_p_spin_paths import OUT  # noqa: E402
from pspin import make_dataset, hc_torch, gs_torch  # noqa: E402
from circuit_snap import (  # noqa: E402
    B_SHELL,
    make_ansatz,
    fidelity,
    init_angles,
)


def log(*a):
    print(time.strftime("%H:%M:%S"), "[qpspin]", *a, flush=True)


def pack_instances(insts, device):
    return [(hc_torch(it, device), gs_torch(it, device), it.E0) for it in insts]


@torch.no_grad()
def mean_median_F(sim, gamma, phi, packed, max_count=None) -> dict:
    n = len(packed) if max_count is None else min(len(packed), max_count)
    Fs = []
    for i in range(n):
        Hc, gs, _ = packed[i]
        Fs.append(float(fidelity(sim, Hc, gs, gamma, phi).cpu()))
    arr = np.asarray(Fs, dtype=np.float64)
    return {
        "mean_F": float(arr.mean()),
        "median_F": float(np.median(arr)),
        "B": int(len(arr)),
    }


def train_one(
    *,
    L: int,
    Gamma: float,
    seed: int,
    mode: str,
    B_train: int,
    B_hold: int,
    rounds: int,
    batch: int,
    device,
    data_cache: dict,
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    gamma0, phi0 = init_angles(L, seed=seed)
    gamma = torch.nn.Parameter(gamma0.clone())
    phi = torch.nn.Parameter(phi0.clone())

    sims = {}
    trains = {}
    holds = {}
    ns = (6, 9) if mode == "joint" else ((6,) if mode == "n6" else (9,))
    for n in ns:
        key = (n, float(Gamma))
        if key not in data_cache:
            log(f"build data n={n} Γ={Gamma} B_train={B_train} B_hold={B_hold}")
            tr = make_dataset(n, Gamma, B_train, seed0=1000 * n + int(round(10 * Gamma)) + 0)
            ho = make_dataset(n, Gamma, B_hold, seed0=1000 * n + int(round(10 * Gamma)) + 50_000)
            data_cache[key] = (pack_instances(tr, device), pack_instances(ho, device))
        trains[n], holds[n] = data_cache[key]
        sims[n] = make_ansatz(n, L, device)

    best = {"score": -1.0, "round": -1, "state": None, "hold": {}, "train": {}}

    for r in range(rounds):
        opt = torch.optim.LBFGS(
            [gamma, phi],
            lr=0.25,
            max_iter=8,
            history_size=16,
            line_search_fn="strong_wolfe",
        )

        batches = {}
        for n in ns:
            N = len(trains[n])
            idx = rng.choice(N, size=min(batch, N), replace=False)
            batches[n] = idx

        def closure():
            opt.zero_grad()
            losses = []
            for n in ns:
                sim = sims[n]
                for j in batches[n]:
                    Hc, gs, _ = trains[n][int(j)]
                    F = fidelity(sim, Hc, gs, gamma, phi)
                    losses.append(1.0 - F)
            lt = torch.stack(losses).mean()
            lt.backward()
            torch.nn.utils.clip_grad_norm_([gamma, phi], 5.0)
            return lt

        try:
            loss = opt.step(closure)
            loss_f = float(loss.detach()) if torch.is_tensor(loss) else float(loss)
        except Exception as e:
            log(f"L={L} Γ={Gamma} seed{seed} {mode} r{r} skip: {e}")
            continue

        train_stats = {
            n: mean_median_F(
                sims[n], gamma.detach(), phi.detach(), trains[n], max_count=min(64, len(trains[n]))
            )
            for n in ns
        }
        hold_stats = {
            n: mean_median_F(sims[n], gamma.detach(), phi.detach(), holds[n]) for n in ns
        }
        score = float(np.mean([hold_stats[n]["median_F"] for n in ns]))
        if score > best["score"]:
            best = {
                "score": score,
                "round": r,
                "state": {
                    "gamma": gamma.detach().cpu().clone(),
                    "phi": phi.detach().cpu().clone(),
                    "L": L,
                    "Gamma": Gamma,
                    "seed": seed,
                    "mode": mode,
                    "B_SHELL": B_SHELL,
                },
                "hold": {str(n): hold_stats[n] for n in ns},
                "train": {str(n): train_stats[n] for n in ns},
            }
        if r % 5 == 0 or r + 1 == rounds:
            bits = " ".join(
                f"n{n}:trMedF={train_stats[n]['median_F']:.3f}/hold={hold_stats[n]['median_F']:.3f}"
                for n in ns
            )
            log(
                f"L={L} Γ={Gamma} seed{seed} {mode} r{r}/{rounds} loss≈{loss_f:.4f} "
                f"{bits} bestHold={best['score']:.3f}@r{best['round']}"
            )

    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--Gammas", type=float, nargs="+", default=[0.5, 1.0, 1.5])
    ap.add_argument("--Ls", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--B-train", type=int, default=64)
    ap.add_argument("--B-hold", type=int, default=32)
    ap.add_argument("--rounds", type=int, default=20)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--modes", nargs="+", default=["joint", "n6", "n9"])
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args()

    device = torch.device("cpu")
    torch.set_num_threads(min(8, torch.get_num_threads()))

    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out) if args.out else (OUT / "smoke" / stamp)
    out_dir.mkdir(parents=True, exist_ok=True)
    log(f"out={out_dir}")

    data_cache: dict = {}
    results = []

    for Gamma in args.Gammas:
        for L in args.Ls:
            for mode in args.modes:
                seed_rows = []
                for seed in args.seeds:
                    best = train_one(
                        L=L,
                        Gamma=Gamma,
                        seed=seed,
                        mode=mode,
                        B_train=args.B_train,
                        B_hold=args.B_hold,
                        rounds=args.rounds,
                        batch=args.batch,
                        device=device,
                        data_cache=data_cache,
                    )
                    ckpt_path = out_dir / f"ckpt_G{Gamma}_L{L}_{mode}_seed{seed}.pt"
                    torch.save(best["state"], ckpt_path)
                    row = {
                        "Gamma": Gamma,
                        "L": L,
                        "mode": mode,
                        "seed": seed,
                        "best_hold_score": best["score"],
                        "best_round": best["round"],
                        "train": best["train"],
                        "hold": best["hold"],
                        "ckpt": str(ckpt_path),
                    }
                    seed_rows.append(row)
                    results.append(row)
                    (out_dir / "train_partial.json").write_text(json.dumps(results, indent=2))
                champ = max(seed_rows, key=lambda r: r["best_hold_score"])
                log(
                    f"CHAMP Γ={Gamma} L={L} {mode} seed{champ['seed']} "
                    f"hold={champ['best_hold_score']:.4f}"
                )

    (out_dir / "train_results.json").write_text(json.dumps(results, indent=2))
    log("train done")
    print(str(out_dir))


if __name__ == "__main__":
    main()
