"""Adapt-VQE from QAOA base: screen pool by |∇F|, append, re-BFGS, rank operators.

L=8 n9 Γ=0.5, hold MedF vs QAOA baseline 0.438.
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
from pspin import make_dataset  # noqa: E402
from circuit_adapt import (  # noqa: E402
    POOL,
    make_ansatz,
    init_qaoa,
    warm_qaoa_from_ckpt,
    fidelity_adapt,
    n_params_adapt,
    append_op,
    grad_pool_at_zero,
    pack_aux,
)

BASELINE = 0.4380
OUT_AD = OUT / "nondiag_uc_explore" / "adapt_vqe"
LEADERBOARD = OUT / "nondiag_uc_explore" / "LEADERBOARD.md"
RANKING = OUT_AD / "OPERATOR_RANKING.md"


def log(*a):
    print(time.strftime("%H:%M:%S"), "[adapt]", *a, flush=True)


def pack_data(insts, device):
    return [pack_aux(it, device) for it in insts]


@torch.no_grad()
def eval_medF(sim, raw, packed, max_count=None):
    n = len(packed) if max_count is None else min(len(packed), max_count)
    Fs = []
    raw_det = {
        "beta": raw["beta"].detach(),
        "gamma": raw["gamma"].detach(),
        "adapt_ops": list(raw.get("adapt_ops", [])),
        "adapt_theta": (
            raw["adapt_theta"].detach()
            if torch.is_tensor(raw.get("adapt_theta")) and raw["adapt_theta"].numel()
            else torch.zeros(0)
        ),
    }
    for i in range(n):
        Hc, gs, Hs, Mag = packed[i]
        Fs.append(float(fidelity_adapt(sim, Hc, gs, raw_det, Hc_shift=Hs, Mag=Mag).cpu()))
    arr = np.asarray(Fs, dtype=np.float64)
    return {"mean_F": float(arr.mean()), "median_F": float(np.median(arr)), "B": int(len(arr))}


def make_params(raw: dict):
    params = []
    for k in ("beta", "gamma", "adapt_theta"):
        if k not in raw:
            continue
        v = raw[k]
        if not torch.is_tensor(v) or v.numel() == 0:
            continue
        p = torch.nn.Parameter(v.detach().clone().float())
        raw[k] = p
        params.append(p)
    return params


def bfgs_train(
    sim,
    raw,
    trains,
    holds,
    *,
    rounds: int,
    batch: int,
    seed: int,
    adam_rounds: int = 8,
):
    rng = np.random.default_rng(seed)
    params = make_params(raw)
    best = {"score": -1.0, "round": -1, "raw": None, "hold": {}, "train": {}}
    L = int(raw["beta"].numel())

    for r in range(rounds):
        idx = rng.choice(len(trains), size=min(batch, len(trains)), replace=False)

        def loss_fn():
            losses = []
            for j in idx:
                Hc, gs, Hs, Mag = trains[int(j)]
                F = fidelity_adapt(sim, Hc, gs, raw, Hc_shift=Hs, Mag=Mag)
                losses.append(1.0 - F)
            return torch.stack(losses).mean()

        use_adam = r < adam_rounds
        try:
            if use_adam:
                opt = torch.optim.Adam(params, lr=0.02)
                loss_f = 0.0
                for _ in range(6):
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
            log(f"opt skip r{r}: {e}")
            continue

        tr = eval_medF(sim, raw, trains, max_count=min(64, len(trains)))
        ho = eval_medF(sim, raw, holds)
        score = float(ho["median_F"])
        if score > best["score"]:
            best = {
                "score": score,
                "round": r,
                "raw": {
                    "beta": raw["beta"].detach().cpu().clone(),
                    "gamma": raw["gamma"].detach().cpu().clone(),
                    "adapt_ops": list(raw.get("adapt_ops", [])),
                    "adapt_theta": (
                        raw["adapt_theta"].detach().cpu().clone()
                        if torch.is_tensor(raw.get("adapt_theta")) and raw["adapt_theta"].numel()
                        else torch.zeros(0)
                    ),
                },
                "hold": ho,
                "train": tr,
            }
        if r % 5 == 0 or r + 1 == rounds:
            log(
                f"ops={raw.get('adapt_ops')} r{r}/{rounds} nparam={n_params_adapt(raw)} "
                f"loss≈{loss_f:.4f} trMed={tr['median_F']:.3f} hold={ho['median_F']:.3f} "
                f"best={best['score']:.3f}@r{best['round']}"
            )
    # restore best into raw Parameters
    if best["raw"] is not None:
        with torch.no_grad():
            raw["beta"].copy_(best["raw"]["beta"])
            raw["gamma"].copy_(best["raw"]["gamma"])
            if best["raw"]["adapt_theta"].numel() and "adapt_theta" in raw and isinstance(
                raw["adapt_theta"], torch.nn.Parameter
            ):
                raw["adapt_theta"].copy_(best["raw"]["adapt_theta"])
            elif best["raw"]["adapt_theta"].numel():
                raw["adapt_theta"] = torch.nn.Parameter(best["raw"]["adapt_theta"].clone())
        raw["adapt_ops"] = list(best["raw"]["adapt_ops"])
    return best


def append_leaderboard(row: dict):
    LEADERBOARD.parent.mkdir(parents=True, exist_ok=True)
    if not LEADERBOARD.exists():
        LEADERBOARD.write_text(
            "# Non-diagonal U_c explore LEADERBOARD (n9 Γ=0.5 hold MedF)\n\n"
            f"Baseline QAOA hold MedF = **{BASELINE:.4f}**\n\n"
            "| recipe | L | hold MedF | Δ vs 0.438 | #params | seed | best_round | notes |\n"
            "|--------|---|-----------|------------|---------|------|------------|-------|\n"
        )
    with LEADERBOARD.open("a") as f:
        f.write(
            f"| {row['recipe']} | {row['L']} | {row['hold_medF']:.4f} | {row['delta']:+.4f} | "
            f"{row['n_params']} | {row['seed']} | {row['best_round']} | {row.get('notes','')} |\n"
        )


def write_ranking(path: Path, oneshot: list, adapt_hist: list, baseline_hold: float):
    lines = [
        "# Adapt-VQE operator ranking (L=8 n9 Γ=0.5)",
        "",
        f"QAOA baseline hold MedF = **{baseline_hold:.4f}** (target board 0.438).",
        "",
        "## One-shot: QAOA + single pool op (re-BFGS)",
        "",
        "| rank | operator | |∇F|@0 | hold after | Δ vs QAOA | Δ vs 0.438 | #params |",
        "|-----:|----------|--------:|-----------:|----------:|-----------:|--------:|",
    ]
    ranked = sorted(oneshot, key=lambda r: -r["hold_medF"])
    for i, r in enumerate(ranked, 1):
        lines.append(
            f"| {i} | {r['name']} | {r['grad_abs']:.4g} | {r['hold_medF']:.4f} | "
            f"{r['hold_medF']-baseline_hold:+.4f} | {r['hold_medF']-BASELINE:+.4f} | {r['n_params']} |"
        )
    lines += [
        "",
        "## Sequential Adapt-VQE (greedy by |∇F|)",
        "",
        "| step | added | |∇F|@pick | hold before | hold after | step Δ | #params |",
        "|-----:|-------|----------:|------------:|-----------:|-------:|--------:|",
    ]
    for h in adapt_hist:
        lines.append(
            f"| {h['step']} | {h['added']} | {h['grad_abs']:.4g} | {h['hold_before']:.4f} | "
            f"{h['hold_after']:.4f} | {h['hold_after']-h['hold_before']:+.4f} | {h['n_params']} |"
        )
    if adapt_hist:
        final = adapt_hist[-1]["hold_after"]
        lines += [
            "",
            f"**Final Adapt hold:** {final:.4f} (Δ vs 0.438 = {final-BASELINE:+.4f})",
            "",
            "## Which operators help most",
            "",
        ]
        # rank by oneshot Δ
        help_sorted = sorted(oneshot, key=lambda r: -(r["hold_medF"] - baseline_hold))
        for i, r in enumerate(help_sorted, 1):
            d = r["hold_medF"] - baseline_hold
            lines.append(f"{i}. **{r['name']}** — hold Δ {d:+.4f} vs QAOA, |∇F|={r['grad_abs']:.4g}")
        lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--L", type=int, default=8)
    ap.add_argument("--Gamma", type=float, default=0.5)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--B-train", type=int, default=64)
    ap.add_argument("--B-hold", type=int, default=32)
    ap.add_argument("--rounds-base", type=int, default=25)
    ap.add_argument("--rounds-op", type=int, default=20)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--max-adapt", type=int, default=4)
    ap.add_argument("--min-grad", type=float, default=1e-4)
    ap.add_argument("--min-gain", type=float, default=0.002)
    ap.add_argument("--skip-oneshot", action="store_true")
    args = ap.parse_args()

    device = torch.device("cpu")
    torch.set_num_threads(min(8, torch.get_num_threads()))
    OUT_AD.mkdir(parents=True, exist_ok=True)

    n = 9
    L = args.L
    log(f"build data n={n} Γ={args.Gamma}")
    tr = make_dataset(n, args.Gamma, args.B_train, seed0=1000 * n + int(round(10 * args.Gamma)))
    ho = make_dataset(
        n, args.Gamma, args.B_hold, seed0=1000 * n + int(round(10 * args.Gamma)) + 50_000
    )
    trains = pack_data(tr, device)
    holds = pack_data(ho, device)
    sim = make_ansatz(n, L, device)

    ckpt_path = OUT / "compare/20261105_G0.5_qaoa_vs_snap/fourier/ckpt_fourier_G0.5_L8_n9_seed2.pt"

    all_oneshot = []
    all_adapt = []

    for seed in args.seeds:
        log(f"===== seed {seed} =====")
        raw = init_qaoa(L, seed=seed)
        if ckpt_path.exists():
            st = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            raw = warm_qaoa_from_ckpt(raw, L, st)
            log(f"warm QAOA from {ckpt_path.name}")

        # --- train QAOA base ---
        log("BFGS QAOA base (2L)")
        best0 = bfgs_train(
            sim, raw, trains, holds,
            rounds=args.rounds_base, batch=args.batch, seed=seed, adam_rounds=8,
        )
        base_hold = float(best0["score"])
        log(f"QAOA base hold MedF={base_hold:.4f} Δvs0.438={base_hold-BASELINE:+.4f}")
        append_leaderboard({
            "recipe": f"adapt_qaoa_base_s{seed}",
            "L": L,
            "hold_medF": base_hold,
            "delta": base_hold - BASELINE,
            "n_params": 2 * L,
            "seed": seed,
            "best_round": best0["round"],
            "notes": "QAOA 2L only",
        })
        torch.save(best0["raw"], OUT_AD / f"ckpt_qaoa_base_seed{seed}.pt")

        # --- one-shot each pool op ---
        oneshot_rows = []
        if not args.skip_oneshot:
            log("screen |∇F|@0 on pool")
            # detach raw for screening from frozen base
            raw_frozen = {
                "beta": best0["raw"]["beta"].clone(),
                "gamma": best0["raw"]["gamma"].clone(),
                "adapt_ops": [],
                "adapt_theta": torch.zeros(0),
            }
            # need Parameters? grad_pool uses probe θ only
            ranks = grad_pool_at_zero(sim, raw_frozen, trains, POOL, max_inst=16)
            for rr in ranks:
                log(f"  screen {rr['name']}: |∇F|={rr['grad_abs']:.4g} F0={rr['F_at_0']:.4f}")

            for rr in ranks:
                name = rr["name"]
                log(f"one-shot train QAOA+{name}")
                raw1 = {
                    "beta": best0["raw"]["beta"].clone(),
                    "gamma": best0["raw"]["gamma"].clone(),
                    "adapt_ops": [],
                    "adapt_theta": torch.zeros(0),
                }
                raw1 = append_op(raw1, name, L)
                best1 = bfgs_train(
                    sim, raw1, trains, holds,
                    rounds=args.rounds_op, batch=args.batch, seed=seed + 17, adam_rounds=6,
                )
                row = {
                    "name": name,
                    "grad_abs": rr["grad_abs"],
                    "hold_medF": float(best1["score"]),
                    "n_params": n_params_adapt(raw1),
                    "seed": seed,
                }
                oneshot_rows.append(row)
                all_oneshot.append(row)
                append_leaderboard({
                    "recipe": f"adapt_oneshot_{name}_s{seed}",
                    "L": L,
                    "hold_medF": row["hold_medF"],
                    "delta": row["hold_medF"] - BASELINE,
                    "n_params": row["n_params"],
                    "seed": seed,
                    "best_round": best1["round"],
                    "notes": f"oneshot |∇|={rr['grad_abs']:.3g} vsQAOA={row['hold_medF']-base_hold:+.3f}",
                })
                log(
                    f"  {name}: hold={row['hold_medF']:.4f} ΔQAOA={row['hold_medF']-base_hold:+.4f} "
                    f"Δ0.438={row['hold_medF']-BASELINE:+.4f}"
                )
                torch.save(best1["raw"], OUT_AD / f"ckpt_oneshot_{name}_seed{seed}.pt")
                write_ranking(RANKING, oneshot_rows, [], base_hold)

        # --- sequential Adapt-VQE ---
        log("sequential Adapt-VQE")
        raw_ad = {
            "beta": best0["raw"]["beta"].clone(),
            "gamma": best0["raw"]["gamma"].clone(),
            "adapt_ops": [],
            "adapt_theta": torch.zeros(0),
        }
        # re-wrap as plain tensors; bfgs_train will Parameter-ize
        hold_cur = base_hold
        hist = []
        for step in range(1, args.max_adapt + 1):
            # screening needs non-Parameter or Parameter — use detach tensors
            screen_raw = {
                "beta": raw_ad["beta"].detach() if torch.is_tensor(raw_ad["beta"]) else raw_ad["beta"],
                "gamma": raw_ad["gamma"].detach() if torch.is_tensor(raw_ad["gamma"]) else raw_ad["gamma"],
                "adapt_ops": list(raw_ad.get("adapt_ops", [])),
                "adapt_theta": (
                    raw_ad["adapt_theta"].detach()
                    if torch.is_tensor(raw_ad.get("adapt_theta")) and raw_ad["adapt_theta"].numel()
                    else torch.zeros(0)
                ),
            }
            ranks = grad_pool_at_zero(sim, screen_raw, trains, POOL, max_inst=16)
            if not ranks or ranks[0]["grad_abs"] < args.min_grad:
                log(f"stop Adapt: no grad ≥ {args.min_grad}")
                break
            pick = ranks[0]
            log(f"step{step} pick {pick['name']} |∇F|={pick['grad_abs']:.4g}")
            # ensure raw_ad has plain tensors before append
            raw_ad = {
                "beta": screen_raw["beta"].clone(),
                "gamma": screen_raw["gamma"].clone(),
                "adapt_ops": list(screen_raw["adapt_ops"]),
                "adapt_theta": screen_raw["adapt_theta"].clone()
                if screen_raw["adapt_theta"].numel()
                else torch.zeros(0),
            }
            raw_ad = append_op(raw_ad, pick["name"], L)
            best_s = bfgs_train(
                sim, raw_ad, trains, holds,
                rounds=args.rounds_op, batch=args.batch, seed=seed + 100 * step, adam_rounds=6,
            )
            hold_after = float(best_s["score"])
            entry = {
                "step": step,
                "added": pick["name"],
                "grad_abs": pick["grad_abs"],
                "hold_before": hold_cur,
                "hold_after": hold_after,
                "n_params": n_params_adapt(raw_ad),
                "seed": seed,
            }
            hist.append(entry)
            all_adapt.append(entry)
            append_leaderboard({
                "recipe": f"adapt_seq_s{seed}_step{step}_{pick['name']}",
                "L": L,
                "hold_medF": hold_after,
                "delta": hold_after - BASELINE,
                "n_params": entry["n_params"],
                "seed": seed,
                "best_round": best_s["round"],
                "notes": f"seq +{pick['name']} d={hold_after-hold_cur:+.3f} |∇|={pick['grad_abs']:.3g}",
            })
            log(
                f"  after +{pick['name']}: hold {hold_cur:.4f}→{hold_after:.4f} "
                f"(Δ={hold_after-hold_cur:+.4f})"
            )
            torch.save(best_s["raw"], OUT_AD / f"ckpt_adapt_step{step}_seed{seed}.pt")
            write_ranking(RANKING, oneshot_rows, hist, base_hold)
            if hold_after - hold_cur < args.min_gain:
                log(f"stop Adapt: gain {hold_after-hold_cur:+.4f} < {args.min_gain}")
                hold_cur = hold_after
                break
            hold_cur = hold_after
            if hold_cur >= 0.50:
                log(f"*** HIT ≥0.50 at step {step} hold={hold_cur:.4f} ***")

        write_ranking(RANKING, oneshot_rows, hist, base_hold)
        (OUT_AD / f"oneshot_seed{seed}.json").write_text(json.dumps(oneshot_rows, indent=2))
        (OUT_AD / f"adapt_hist_seed{seed}.json").write_text(json.dumps(hist, indent=2))

    # SUMMARY
    summary = [
        "# Adapt-VQE SUMMARY",
        "",
        f"Baseline board QAOA = **{BASELINE:.4f}**",
        "",
        "See OPERATOR_RANKING.md for |∇F| screens and hold gains.",
        "",
        "Prior non-diag (free SNAP) best: dbl_hc_snap hold 0.5339 (80p).",
        "This run keeps QAOA 2L base and adds few-param pool ops.",
        "",
    ]
    if all_oneshot:
        champ = max(all_oneshot, key=lambda r: r["hold_medF"])
        summary.append(
            f"**Best oneshot:** `{champ['name']}` hold **{champ['hold_medF']:.4f}** "
            f"(Δ {champ['hold_medF']-BASELINE:+.4f}, {champ['n_params']}p)"
        )
    if all_adapt:
        champ = max(all_adapt, key=lambda r: r["hold_after"])
        summary.append(
            f"**Best Adapt seq:** step{champ['step']} +`{champ['added']}` hold "
            f"**{champ['hold_after']:.4f}** ({champ['n_params']}p)"
        )
    (OUT_AD / "SUMMARY.md").write_text("\n".join(summary) + "\n")
    # also mirror short note to parent SUMMARY
    parent_sum = OUT / "nondiag_uc_explore" / "SUMMARY.md"
    parent_sum.write_text(
        "\n".join(
            [
                "# Non-diagonal / Adapt-VQE SUMMARY",
                "",
                f"Board baseline QAOA hold = **{BASELINE:.4f}**",
                "",
                "## Free-SNAP non-diag (earlier)",
                "- `dbl_hc_snap` hold **0.5339** (80p, Δ+0.096) — HIT ≥0.50",
                "",
                "## Adapt-VQE from QAOA base (2L + pool)",
                f"- details: `{OUT_AD}/OPERATOR_RANKING.md`",
                "",
            ]
        )
        + "\n"
    )
    log("done")
    print(str(OUT_AD))


if __name__ == "__main__":
    main()
