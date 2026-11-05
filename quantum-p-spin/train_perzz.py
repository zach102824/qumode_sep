"""Nonstop per-ZZZ γ explore at L=8 n9 Γ=0.5 vs QAOA hold baseline 0.438."""
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
from pspin import make_dataset, gs_torch  # noqa: E402
from circuit_fourier import shell_phi_from_betas  # noqa: E402
from circuit_perzz import (  # noqa: E402
    B_SHELL,
    make_ansatz,
    n_triples,
    build_terms,
    terms_on_grid,
    fidelity_perzz,
    expand_gamma,
    init_perzz,
    n_params_perzz,
    term_kind,
    uses_free_snap,
)

BASELINE_QAOA_N9 = 0.4380
LEADERBOARD = OUT / "perzz_explore" / "LEADERBOARD.md"


def log(*a):
    print(time.strftime("%H:%M:%S"), "[perzz]", *a, flush=True)


def pack_hold_train(insts, sim, kind: str, device):
    packed = []
    for it in insts:
        terms = torch.as_tensor(build_terms(it.n, it.J, kind), dtype=torch.float32, device=device)
        tgrid = terms_on_grid(sim, terms)
        gs = gs_torch(it, device)
        packed.append((tgrid, gs, it.E0))
    return packed


@torch.no_grad()
def mean_median_F(sim, gamma_LT, phi, packed, max_count=None) -> dict:
    n = len(packed) if max_count is None else min(len(packed), max_count)
    Fs = []
    for i in range(n):
        tgrid, gs, _ = packed[i]
        Fs.append(float(fidelity_perzz(sim, tgrid, gs, gamma_LT, phi).cpu()))
    arr = np.asarray(Fs, dtype=np.float64)
    return {
        "mean_F": float(arr.mean()),
        "median_F": float(np.median(arr)),
        "B": int(len(arr)),
    }


def make_params(recipe: str, raw: dict):
    params = []
    for k, v in raw.items():
        p = torch.nn.Parameter(v.clone())
        raw[k] = p
        params.append(p)
    return params


def phi_from_raw(recipe: str, raw: dict) -> torch.Tensor:
    if uses_free_snap(recipe):
        return raw["phi"]
    return shell_phi_from_betas(raw["beta"])


def append_leaderboard(row: dict):
    LEADERBOARD.parent.mkdir(parents=True, exist_ok=True)
    if not LEADERBOARD.exists():
        LEADERBOARD.write_text(
            "# per-ZZZ explore LEADERBOARD (L=8 n9 Γ=0.5 hold MedF)\n\n"
            f"Baseline QAOA hold MedF = **{BASELINE_QAOA_N9:.4f}**\n\n"
            "| recipe | hold MedF | Δ vs 0.438 | #params | seed | best_round | notes |\n"
            "|--------|-----------|------------|---------|------|------------|-------|\n"
        )
    line = (
        f"| {row['recipe']} | {row['hold_medF']:.4f} | {row['delta']:+.4f} | "
        f"{row['n_params']} | {row['seed']} | {row['best_round']} | {row.get('notes','')} |\n"
    )
    with LEADERBOARD.open("a") as f:
        f.write(line)


def train_one(
    *,
    recipe: str,
    L: int,
    Gamma: float,
    seed: int,
    B_train: int,
    B_hold: int,
    rounds: int,
    batch: int,
    device,
    data_cache: dict,
    rank: int = 4,
    warm_qaoa_ckpt: str = "",
    opt_name: str = "lbfgs",
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)

    n = 9
    T = n_triples(n)
    kind = term_kind(recipe)
    raw = init_perzz(recipe, L, T, seed=seed, rank=rank)
    if warm_qaoa_ckpt or "warm" in recipe:
        ckpt_path = warm_qaoa_ckpt or str(
            OUT / "compare/20261105_G0.5_qaoa_vs_snap/fourier/ckpt_fourier_G0.5_L8_n9_seed2.pt"
        )
        st = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        g = st["gamma"].float().reshape(L)
        # broadcast QAOA γ + tiny per-triple ±
        torch.manual_seed(seed + 123)
        raw["gamma_full"] = g.reshape(L, 1) + 0.01 * torch.randn(L, T)
        if "phi" in raw:
            if "beta" in st:
                from circuit_fourier import shell_phi_from_betas
                raw["phi"] = shell_phi_from_betas(st["beta"].float()) + 0.001 * torch.randn(L, B_SHELL)
            elif "phi" in st:
                raw["phi"] = st["phi"].float()[:L, :B_SHELL] + 0.001 * torch.randn(L, B_SHELL)
        log(f"warm-start from {ckpt_path}")
    params = make_params(recipe, raw)

    key = (n, float(Gamma), kind)
    sim = make_ansatz(n, L, device)
    if key not in data_cache:
        log(f"build data n={n} Γ={Gamma} kind={kind} B_train={B_train} B_hold={B_hold}")
        tr = make_dataset(n, Gamma, B_train, seed0=1000 * n + int(round(10 * Gamma)) + 0)
        ho = make_dataset(n, Gamma, B_hold, seed0=1000 * n + int(round(10 * Gamma)) + 50_000)
        data_cache[key] = (
            pack_hold_train(tr, sim, kind, device),
            pack_hold_train(ho, sim, kind, device),
        )
    trains, holds = data_cache[key]

    best = {"score": -1.0, "round": -1, "state": None, "hold": {}, "train": {}}
    nparam = n_params_perzz(recipe, L, T, rank=rank)

    use_adam = (opt_name == "adam") or ("adam" in recipe)
    for r in range(rounds):
        N = len(trains)
        idx = rng.choice(N, size=min(batch, N), replace=False)

        def loss_fn():
            gLT = expand_gamma(recipe, raw, L, T)
            phi = phi_from_raw(recipe, raw)
            losses = []
            for j in idx:
                tgrid, gs, _ = trains[int(j)]
                F = fidelity_perzz(sim, tgrid, gs, gLT, phi)
                losses.append(1.0 - F)
            return torch.stack(losses).mean()

        try:
            if use_adam:
                opt = torch.optim.Adam(params, lr=0.02)
                loss_f = 0.0
                for _ in range(8):
                    opt.zero_grad()
                    lt = loss_fn()
                    lt.backward()
                    torch.nn.utils.clip_grad_norm_(params, 5.0)
                    opt.step()
                    loss_f = float(lt.detach())
            else:
                opt = torch.optim.LBFGS(
                    params,
                    lr=0.25,
                    max_iter=8,
                    history_size=16,
                    line_search_fn="strong_wolfe",
                )
                def closure():
                    opt.zero_grad()
                    lt = loss_fn()
                    lt.backward()
                    torch.nn.utils.clip_grad_norm_(params, 5.0)
                    return lt
                loss = opt.step(closure)
                loss_f = float(loss.detach()) if torch.is_tensor(loss) else float(loss)
        except Exception as e:
            log(f"{recipe} L={L} seed{seed} r{r} skip: {e}")
            continue

        with torch.no_grad():
            gLT = expand_gamma(recipe, {k: v.detach() for k, v in raw.items()}, L, T)
            if uses_free_snap(recipe):
                phi_det = raw["phi"].detach()
            else:
                phi_det = shell_phi_from_betas(raw["beta"].detach())
        train_stats = mean_median_F(sim, gLT, phi_det, trains, max_count=min(64, len(trains)))
        hold_stats = mean_median_F(sim, gLT, phi_det, holds)
        score = float(hold_stats["median_F"])
        if score > best["score"]:
            state = {
                "recipe": recipe,
                "raw": {k: v.detach().cpu().clone() for k, v in raw.items()},
                "gamma_LT": gLT.detach().cpu().clone(),
                "phi": phi_det.detach().cpu().clone(),
                "L": L,
                "Gamma": Gamma,
                "seed": seed,
                "n_params": nparam,
                "T": T,
            }
            best = {
                "score": score,
                "round": r,
                "state": state,
                "hold": hold_stats,
                "train": train_stats,
            }
        if r % 5 == 0 or r + 1 == rounds:
            log(
                f"{recipe} L={L} Γ={Gamma} seed{seed} r{r}/{rounds} nparam={nparam} "
                f"loss≈{loss_f:.4f} trMedF={train_stats['median_F']:.3f} "
                f"hold={hold_stats['median_F']:.3f} bestHold={best['score']:.3f}@r{best['round']}"
            )
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recipes", nargs="+", required=True)
    ap.add_argument("--Gamma", type=float, default=0.5)
    ap.add_argument("--L", type=int, default=8)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--B-train", type=int, default=64)
    ap.add_argument("--B-hold", type=int, default=32)
    ap.add_argument("--rounds", type=int, default=20)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--rank", type=int, default=4)
    ap.add_argument("--warm-qaoa-ckpt", type=str, default="")
    ap.add_argument("--opt", choices=["lbfgs", "adam"], default="lbfgs")
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args()

    device = torch.device("cpu")
    torch.set_num_threads(min(8, torch.get_num_threads()))

    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_root = Path(args.out) if args.out else (OUT / "perzz_explore")
    out_root.mkdir(parents=True, exist_ok=True)

    data_cache: dict = {}
    all_rows = []

    for recipe in args.recipes:
        out_dir = out_root / recipe
        out_dir.mkdir(parents=True, exist_ok=True)
        log(f"=== recipe={recipe} L={args.L} Γ={args.Gamma} out={out_dir} ===")
        seed_rows = []
        for seed in args.seeds:
            t0 = time.time()
            best = train_one(
                recipe=recipe,
                L=args.L,
                Gamma=args.Gamma,
                seed=seed,
                B_train=args.B_train,
                B_hold=args.B_hold,
                rounds=args.rounds,
                batch=args.batch,
                device=device,
                data_cache=data_cache,
                rank=args.rank,
            )
            ckpt = out_dir / f"ckpt_L{args.L}_seed{seed}.pt"
            torch.save(best["state"], ckpt)
            hold = float(best["score"])
            row = {
                "recipe": recipe,
                "hold_medF": hold,
                "delta": hold - BASELINE_QAOA_N9,
                "n_params": best["state"]["n_params"] if best["state"] else -1,
                "seed": seed,
                "best_round": best["round"],
                "train": best["train"],
                "hold": best["hold"],
                "ckpt": str(ckpt),
                "seconds": time.time() - t0,
                "notes": f"trMed={best['train'].get('median_F', float('nan')):.3f}",
            }
            seed_rows.append(row)
            all_rows.append(row)
            append_leaderboard(row)
            log(
                f"DONE {recipe} seed{seed} hold={hold:.4f} Δ={row['delta']:+.4f} "
                f"in {row['seconds']:.0f}s"
            )
            (out_dir / "train_partial.json").write_text(json.dumps(seed_rows, indent=2))
        champ = max(seed_rows, key=lambda r: r["hold_medF"])
        log(
            f"CHAMP {recipe} seed{champ['seed']} hold={champ['hold_medF']:.4f} "
            f"Δ={champ['delta']:+.4f}"
        )
        (out_dir / "train_results.json").write_text(json.dumps(seed_rows, indent=2))

    (out_root / f"all_partial_{stamp}.json").write_text(json.dumps(all_rows, indent=2))
    log("all recipes done")
    print(str(out_root))


if __name__ == "__main__":
    main()
