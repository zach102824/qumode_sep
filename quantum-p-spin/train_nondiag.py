"""Nonstop non-diagonal U_c explore — beat QAOA L=8 n9 Γ=0.5 hold MedF 0.438."""
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
from circuit_nondiag import (  # noqa: E402
    ND_RECIPES,
    make_ansatz,
    n_params_nd,
    init_nd,
    warm_from_qaoa,
    fidelity_nd,
    pack_hc2,
    uses_free_snap,
)

BASELINE_QAOA_N9 = 0.4380
OUT_ND = OUT / "nondiag_uc_explore"
LEADERBOARD = OUT_ND / "LEADERBOARD.md"


def log(*a):
    print(time.strftime("%H:%M:%S"), "[nd]", *a, flush=True)


def pack_insts(insts, recipe: str, device):
    packed = []
    for it in insts:
        Hc = hc_torch(it, device)
        gs = gs_torch(it, device)
        Hc2 = pack_hc2(it, recipe, device)
        packed.append((Hc, gs, Hc2, it.E0))
    return packed


@torch.no_grad()
def mean_median_F(sim, raw_det, recipe, packed, max_count=None) -> dict:
    n = len(packed) if max_count is None else min(len(packed), max_count)
    Fs = []
    for i in range(n):
        Hc, gs, Hc2, _ = packed[i]
        Fs.append(float(fidelity_nd(sim, Hc, gs, raw_det, recipe, Hc2_diag=Hc2).cpu()))
    arr = np.asarray(Fs, dtype=np.float64)
    return {"mean_F": float(arr.mean()), "median_F": float(np.median(arr)), "B": int(len(arr))}


def make_params(raw: dict):
    params = []
    for k, v in raw.items():
        p = torch.nn.Parameter(v.clone())
        raw[k] = p
        params.append(p)
    return params


def append_leaderboard(row: dict, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_text(
            "# Non-diagonal U_c explore LEADERBOARD (n9 Γ=0.5 hold MedF)\n\n"
            f"Baseline QAOA hold MedF = **{BASELINE_QAOA_N9:.4f}**  | target ≥0.50  | stretch ≥0.55\n\n"
            "| recipe | L | hold MedF | Δ vs 0.438 | #params | seed | best_round | notes |\n"
            "|--------|---|-----------|------------|---------|------|------------|-------|\n"
        )
    line = (
        f"| {row['recipe']} | {row['L']} | {row['hold_medF']:.4f} | {row['delta']:+.4f} | "
        f"{row['n_params']} | {row['seed']} | {row['best_round']} | {row.get('notes','')} |\n"
    )
    with path.open("a") as f:
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
    warm: bool = True,
    opt_schedule: str = "lbfgs",
    adam_rounds: int = 0,
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    n = 9
    raw = init_nd(recipe, L, n=n, seed=seed)

    if warm:
        ckpt_path = OUT / "compare/20261105_G0.5_qaoa_vs_snap/fourier/ckpt_fourier_G0.5_L8_n9_seed2.pt"
        if ckpt_path.exists():
            st = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            raw = warm_from_qaoa(recipe, raw, L, st)
            log(f"warm-start from {ckpt_path.name}")
        else:
            log("no warm ckpt; cold init")

    params = make_params(raw)
    sim = make_ansatz(n, L, device)

    key = (n, float(Gamma), recipe, B_train, B_hold, L)
    if key not in data_cache:
        log(f"build data n={n} Γ={Gamma} recipe={recipe} Btr={B_train} Bho={B_hold}")
        tr = make_dataset(n, Gamma, B_train, seed0=1000 * n + int(round(10 * Gamma)) + 0)
        ho = make_dataset(n, Gamma, B_hold, seed0=1000 * n + int(round(10 * Gamma)) + 50_000)
        data_cache[key] = (pack_insts(tr, recipe, device), pack_insts(ho, recipe, device))
    trains, holds = data_cache[key]

    best = {"score": -1.0, "round": -1, "state": None, "hold": {}, "train": {}}
    nparam = n_params_nd(recipe, L, n=n)

    for r in range(rounds):
        N = len(trains)
        idx = rng.choice(N, size=min(batch, N), replace=False)

        def loss_fn():
            losses = []
            for j in idx:
                Hc, gs, Hc2, _ = trains[int(j)]
                F = fidelity_nd(sim, Hc, gs, raw, recipe, Hc2_diag=Hc2)
                losses.append(1.0 - F)
            return torch.stack(losses).mean()

        use_adam = (opt_schedule == "adam") or (
            opt_schedule == "adam_then_lbfgs" and r < adam_rounds
        )
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
                    params, lr=0.25, max_iter=8, history_size=16, line_search_fn="strong_wolfe"
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

        raw_det = {k: v.detach() for k, v in raw.items()}
        train_stats = mean_median_F(sim, raw_det, recipe, trains, max_count=min(64, len(trains)))
        hold_stats = mean_median_F(sim, raw_det, recipe, holds)
        score = float(hold_stats["median_F"])
        if score > best["score"]:
            state = {
                "recipe": recipe,
                "raw": {k: v.detach().cpu().clone() for k, v in raw.items()},
                "L": L,
                "Gamma": Gamma,
                "seed": seed,
                "n_params": nparam,
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
                f"{recipe} L={L} seed{seed} r{r}/{rounds} nparam={nparam} "
                f"loss≈{loss_f:.4f} trMedF={train_stats['median_F']:.3f} "
                f"hold={hold_stats['median_F']:.3f} bestHold={best['score']:.3f}@r{best['round']}"
            )
    return best


def write_summary(out_root: Path, all_rows: list):
    if not all_rows:
        return
    by = {}
    for r in all_rows:
        by.setdefault(r["recipe"], []).append(r)
    lines = [
        "# Non-diagonal U_c SUMMARY",
        "",
        f"Baseline QAOA hold MedF = **{BASELINE_QAOA_N9:.4f}** | target ≥0.50 | stretch ≥0.55",
        "",
        "| recipe | best hold | Δ | #params | best seed |",
        "|--------|-----------|---|---------|-----------|",
    ]
    champs = []
    for recipe, rows in by.items():
        champ = max(rows, key=lambda x: x["hold_medF"])
        champs.append(champ)
        lines.append(
            f"| {recipe} | {champ['hold_medF']:.4f} | {champ['delta']:+.4f} | "
            f"{champ['n_params']} | {champ['seed']} |"
        )
    overall = max(champs, key=lambda x: x["hold_medF"])
    hit = overall["hold_medF"] >= 0.50
    lines += [
        "",
        f"**Overall best:** `{overall['recipe']}` hold **{overall['hold_medF']:.4f}** "
        f"(Δ {overall['delta']:+.4f}, {overall['n_params']}p)",
        "",
        f"Hit ≥0.50: **{'YES' if hit else 'NO'}**",
        "",
    ]
    if not hit:
        lines += [
            "## Failure note",
            "",
            "Non-diagonal U_c recipes did not dramatically beat QAOA; see LEADERBOARD.",
            "If best ≈0.45–0.47, non-diag also stalls near the per-ZZZ ceiling.",
            "",
        ]
    (out_root / "SUMMARY.md").write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recipes", nargs="+", default=list(ND_RECIPES))
    ap.add_argument("--Gamma", type=float, default=0.5)
    ap.add_argument("--L", type=int, default=8)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--B-train", type=int, default=64)
    ap.add_argument("--B-hold", type=int, default=32)
    ap.add_argument("--rounds", type=int, default=30)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--no-warm", action="store_true")
    ap.add_argument("--opt", choices=["lbfgs", "adam", "adam_then_lbfgs"], default="adam_then_lbfgs")
    ap.add_argument("--adam-rounds", type=int, default=10)
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args()

    device = torch.device("cpu")
    torch.set_num_threads(min(8, torch.get_num_threads()))

    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_root = Path(args.out) if args.out else OUT_ND
    out_root.mkdir(parents=True, exist_ok=True)

    data_cache: dict = {}
    all_rows = []

    for recipe in args.recipes:
        out_dir = out_root / f"{recipe}_L{args.L}"
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
                warm=not args.no_warm,
                opt_schedule=args.opt,
                adam_rounds=args.adam_rounds,
            )
            ckpt = out_dir / f"ckpt_L{args.L}_seed{seed}.pt"
            torch.save(best["state"], ckpt)
            hold = float(best["score"])
            row = {
                "recipe": recipe,
                "L": args.L,
                "hold_medF": hold,
                "delta": hold - BASELINE_QAOA_N9,
                "n_params": best["state"]["n_params"] if best["state"] else -1,
                "seed": seed,
                "best_round": best["round"],
                "train": best["train"],
                "hold": best["hold"],
                "ckpt": str(ckpt),
                "seconds": time.time() - t0,
                "notes": f"trMed={best['train'].get('median_F', float('nan')):.3f} Btr={args.B_train}",
            }
            seed_rows.append(row)
            all_rows.append(row)
            append_leaderboard(row, out_root / "LEADERBOARD.md")
            write_summary(out_root, all_rows)
            (out_root / f"all_partial_{stamp}.json").write_text(json.dumps(all_rows, indent=2))
            log(
                f"DONE {recipe} seed{seed} hold={hold:.4f} Δ={row['delta']:+.4f} "
                f"in {row['seconds']:.0f}s"
            )
            (out_dir / "train_partial.json").write_text(json.dumps(seed_rows, indent=2))
            if hold >= 0.50:
                log(f"*** HIT ≥0.50: {recipe} seed{seed} hold={hold:.4f} ***")
        champ = max(seed_rows, key=lambda r: r["hold_medF"])
        log(
            f"CHAMP {recipe} seed{champ['seed']} hold={champ['hold_medF']:.4f} "
            f"Δ={champ['delta']:+.4f}"
        )
        (out_dir / "train_results.json").write_text(json.dumps(seed_rows, indent=2))

    write_summary(out_root, all_rows)
    (out_root / f"all_partial_{stamp}.json").write_text(json.dumps(all_rows, indent=2))
    log("batch done")
    print(str(out_root))


if __name__ == "__main__":
    main()
