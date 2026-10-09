"""Relabel-round lr scan for L4s10 r200 (RELAYOUT_LOWBUDGET_SUMMARY.md, lr-scan section).

lr 0.05 = the main sweep run lb_xor_vacuum_L4s10_r200 (identical seeds/settings); others are
lblr_xor_vacuum_L4s10_r200_lr<lr>. Writes relayout_lrscan_summary.json and prints markdown tables.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from noiseless.analyze_relayout_lowbudget import RUNS, latest_by_tag, load, per_round

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "noiseless" / "results" / "relayout_lrscan_summary.json"
TAGS = {0.02: "lblr_xor_vacuum_L4s10_r200_lr0.02", 0.05: "lb_xor_vacuum_L4s10_r200",
        0.1: "lblr_xor_vacuum_L4s10_r200_lr0.1", 0.2: "lblr_xor_vacuum_L4s10_r200_lr0.2"}


def main() -> None:
    files = {**latest_by_tag("lb_"), **latest_by_tag("lblr_")}
    res = {}
    for lr, tag in TAGS.items():
        if tag not in files:
            continue
        recs = load(files[tag])
        rows = per_round(recs)
        wrong0 = [r for r in recs if not r["rounds"][0]["next_guess_is_ground"]]
        beta = []
        for k in range(len(recs[0]["rounds"])):
            mab = [r["rounds"][k]["mean_abs_beta"] for r in recs]
            mxb = [r["rounds"][k]["max_abs_beta"] for r in recs]
            beta.append({"round": k, "mean_abs_beta": float(np.mean(mab)),
                         "median_max_abs_beta": float(np.median(mxb)),
                         "p95_max_abs_beta": float(np.percentile(mxb, 95)),
                         "max_max_abs_beta": float(np.max(mxb)),
                         "frac_max_beta_gt3": float(np.mean(np.array(mxb) > 3))})
        sel = [r["mean_abs_beta"] for r in recs]
        res[str(lr)] = {"tag": tag, "rows": rows, "beta": beta,
                        "selected_mean_abs_beta": float(np.mean(sel)),
                        "n_wrong0": len(wrong0),
                        "frac_wrong0_fixed": float(np.mean([r["rounds"][-1]["bsf_success"] for r in wrong0])),
                        "frac_right0_final_success": float(np.mean([r["rounds"][-1]["bsf_success"] for r in recs if r["rounds"][0]["next_guess_is_ground"]]))}
    OUT.write_text(json.dumps(res, indent=2))
    print("| relabel lr | round 0 | round 1 | round 2 | round 3 | round 4 |\n|---|---|---|---|---|---|")
    for lr, v in res.items():
        print(f"| {lr} | " + " | ".join(f"{r['success']:.3f} / {r['mean_p_gs']:.3f} / {r['cum_nfev']:.0f}" for r in v["rows"]) + " |")
    print("\n| relabel lr | wrong guesses fixed by round 4 | right guesses -> final success | guess hit r0..r4 |\n|---|---|---|---|")
    for lr, v in res.items():
        print(f"| {lr} | {v['frac_wrong0_fixed']:.3f} ({round(v['frac_wrong0_fixed']*v['n_wrong0'])}/{v['n_wrong0']}) | {v['frac_right0_final_success']:.3f} | " + " / ".join(f"{r['guess_hit']:.3f}" for r in v["rows"]) + " |")
    print("\n| relabel lr | mean abs beta (rounds 1-4) | median max abs beta (r1-4) | p95 max abs beta (r1-4) | largest max abs beta | share of round trials with max abs beta > 3 |\n|---|---|---|---|---|---|")
    for lr, v in res.items():
        b = v["beta"][1:]
        print(f"| {lr} | " + " / ".join(f"{x['mean_abs_beta']:.2f}" for x in b) + " | "
              + " / ".join(f"{x['median_max_abs_beta']:.2f}" for x in b) + " | "
              + " / ".join(f"{x['p95_max_abs_beta']:.2f}" for x in b) + f" | {max(x['max_max_abs_beta'] for x in b):.2f} | "
              + f"{np.mean([x['frac_max_beta_gt3'] for x in b]):.3f} |")
    print("\nround-0 beta:", res[next(iter(res))]["beta"][0])


if __name__ == "__main__":
    main()
