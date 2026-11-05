"""Continue Adapt: append ops in oneshot hold-gain order; also givens-first greedy."""
from __future__ import annotations

import json, sys, time
from pathlib import Path
import numpy as np
import torch

PKG = Path(__file__).resolve().parent
ROOT = PKG.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(PKG))

from quantum_p_spin_paths import OUT
from pspin import make_dataset
from circuit_adapt import (
    make_ansatz, init_qaoa, warm_qaoa_from_ckpt, fidelity_adapt,
    n_params_adapt, append_op, pack_aux, POOL, grad_pool_at_zero,
)
from train_adapt import bfgs_train, eval_medF, append_leaderboard, pack_data, log, BASELINE

OUT_AD = OUT / "nondiag_uc_explore" / "adapt_vqe"
RANKING = OUT_AD / "OPERATOR_RANKING.md"


def main():
    device = torch.device("cpu")
    torch.set_num_threads(8)
    L, n, Gamma = 8, 9, 0.5
    Btr, Bho = 64, 32
    seed = 0

    oneshot = json.loads((OUT_AD / "oneshot_seed0.json").read_text())
    # hold-gain order
    order = [r["name"] for r in sorted(oneshot, key=lambda r: -r["hold_medF"])]
    log(f"hold-gain order: {order}")

    tr = make_dataset(n, Gamma, Btr, seed0=1000 * n + int(round(10 * Gamma)))
    ho = make_dataset(n, Gamma, Bho, seed0=1000 * n + int(round(10 * Gamma)) + 50_000)
    trains, holds = pack_data(tr, device), pack_data(ho, device)
    sim = make_ansatz(n, L, device)

    base = torch.load(OUT_AD / "ckpt_qaoa_base_seed0.pt", map_location="cpu", weights_only=False)
    base_hold = float(json.loads((OUT_AD / "adapt_hist_seed0.json").read_text())[0]["hold_before"]) if (OUT_AD/"adapt_hist_seed0.json").exists() else 0.4373
    # re-eval base
    raw0 = {"beta": base["beta"], "gamma": base["gamma"], "adapt_ops": [], "adapt_theta": torch.zeros(0)}
    base_hold = eval_medF(sim, raw0, holds)["median_F"]
    log(f"QAOA base hold={base_hold:.4f}")

    # --- hold-gain sequential ---
    raw = {
        "beta": base["beta"].clone(),
        "gamma": base["gamma"].clone(),
        "adapt_ops": [],
        "adapt_theta": torch.zeros(0),
    }
    hold_cur = base_hold
    hist = []
    for step, name in enumerate(order[:4], 1):
        log(f"holdrank step{step} +{name}")
        raw = {
            "beta": raw["beta"].detach().clone() if torch.is_tensor(raw["beta"]) else raw["beta"],
            "gamma": raw["gamma"].detach().clone(),
            "adapt_ops": list(raw.get("adapt_ops", [])),
            "adapt_theta": raw["adapt_theta"].detach().clone() if torch.is_tensor(raw.get("adapt_theta")) and raw["adapt_theta"].numel() else torch.zeros(0),
        }
        # strip Parameter
        for k in ("beta","gamma","adapt_theta"):
            if isinstance(raw.get(k), torch.nn.Parameter):
                raw[k] = raw[k].detach().clone()
        raw = append_op(raw, name, L)
        best = bfgs_train(sim, raw, trains, holds, rounds=22, batch=32, seed=seed+200*step, adam_rounds=6)
        hold_after = float(best["score"])
        entry = {
            "step": step, "added": name, "grad_abs": next(r["grad_abs"] for r in oneshot if r["name"]==name),
            "hold_before": hold_cur, "hold_after": hold_after,
            "n_params": n_params_adapt(raw), "seed": seed, "method": "hold_gain_order",
        }
        hist.append(entry)
        append_leaderboard({
            "recipe": f"adapt_holdrank_s{seed}_step{step}_{name}",
            "L": L, "hold_medF": hold_after, "delta": hold_after-BASELINE,
            "n_params": entry["n_params"], "seed": seed, "best_round": best["round"],
            "notes": f"holdrank +{name} d={hold_after-hold_cur:+.3f}",
        })
        log(f"  {hold_cur:.4f}→{hold_after:.4f} Δ={hold_after-hold_cur:+.4f}")
        torch.save(best["raw"], OUT_AD / f"ckpt_holdrank_step{step}_seed{seed}.pt")
        if hold_after - hold_cur < 0.002 and step > 1:
            log("stop holdrank: tiny gain")
            hold_cur = hold_after
            break
        hold_cur = hold_after
        if hold_cur >= 0.50:
            log(f"*** HIT ≥0.50 ***")

    # --- givens + xy_match combo (top2 oneshot helpers that are complementary) ---
    log("combo givens+xy_match")
    raw_c = {
        "beta": base["beta"].clone(),
        "gamma": base["gamma"].clone(),
        "adapt_ops": [],
        "adapt_theta": torch.zeros(0),
    }
    raw_c = append_op(raw_c, "givens", L)
    raw_c = append_op(raw_c, "xy_match", L)
    best_c = bfgs_train(sim, raw_c, trains, holds, rounds=25, batch=32, seed=77, adam_rounds=8)
    hc = float(best_c["score"])
    append_leaderboard({
        "recipe": f"adapt_combo_givens_xy_s{seed}",
        "L": L, "hold_medF": hc, "delta": hc-BASELINE,
        "n_params": n_params_adapt(raw_c), "seed": seed, "best_round": best_c["round"],
        "notes": f"combo givens+xy vsQAOA={hc-base_hold:+.3f}",
    })
    log(f"combo givens+xy hold={hc:.4f} ΔQAOA={hc-base_hold:+.4f}")
    torch.save(best_c["raw"], OUT_AD / f"ckpt_combo_givens_xy_seed{seed}.pt")

    # update ranking doc
    prev = (OUT_AD / "OPERATOR_RANKING.md").read_text() if RANKING.exists() else ""
    extra = [
        "",
        "## Hold-gain-ordered Adapt (continuation)",
        "",
        "| step | added | hold before | hold after | step Δ | #params |",
        "|-----:|-------|------------:|-----------:|-------:|--------:|",
    ]
    for h in hist:
        extra.append(
            f"| {h['step']} | {h['added']} | {h['hold_before']:.4f} | {h['hold_after']:.4f} | "
            f"{h['hold_after']-h['hold_before']:+.4f} | {h['n_params']} |"
        )
    extra += [
        "",
        f"**Combo givens+xy_match:** hold **{hc:.4f}** (Δ vs 0.438 {hc-BASELINE:+.4f}, {n_params_adapt(raw_c)}p)",
        "",
        "### Note",
        "Greedy-|∇F| Adapt missed `givens` (mid grad, best hold). Prefer hold-gain or validate with short BFGS.",
        f"Lean Adapt ceiling ~0.46; free-SNAP `dbl_hc_snap` still best overall at **0.5339** (80p).",
        "",
    ]
    RANKING.write_text(prev.rstrip() + "\n" + "\n".join(extra))
    (OUT_AD / "adapt_holdrank_seed0.json").write_text(json.dumps(hist + [{"combo_givens_xy": hc}], indent=2))

    # parent SUMMARY
    (OUT / "nondiag_uc_explore" / "SUMMARY.md").write_text(
        f"""# Non-diagonal / Adapt-VQE SUMMARY

Board baseline QAOA hold = **{BASELINE:.4f}** | target ≥0.50

## Best overall
| recipe | hold | Δ | #params | notes |
|--------|-----:|--:|--------:|-------|
| dbl_hc_snap (free SNAP×2 + double Hc) | **0.5339** | +0.096 | 80 | HIT ≥0.50 |
| adapt oneshot **givens** | 0.4610 | +0.023 | 24 | best lean QAOA+op |
| adapt combo givens+xy | {hc:.4f} | {hc-BASELINE:+.4f} | 32 | |
| adapt greedy\\|∇\\| seq | 0.4442 | +0.006 | 32 | picked xy then yy |

## Operator rank by hold Δ vs QAOA (oneshot, +L params each)
1. givens +0.024
2. xy_match / x_total ~+0.009
3. mag_chan +0.009
4. xx/yy ~+0.002–0.004
5. hc2_chan / shift_chan ~0 (need mid-SNAP + free φ to unlock, cf. dbl_hc_snap)

Details: `adapt_vqe/OPERATOR_RANKING.md`
"""
    )
    log("continuation done")


if __name__ == "__main__":
    main()
