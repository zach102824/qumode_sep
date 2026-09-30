"""SPSA step scan at n=16 (F1): table from scaling_runs/{main,steps400,steps600,steps800}.

Writes noiseless/results/scaling_step_scan_n16_summary.json and prints a markdown table.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
RUNS = REPO / "noiseless" / "results" / "scaling_runs"
N = 16
SCAN = (("main", 200), ("steps400", 400), ("steps600", 600), ("steps800", 800))


def main():
    rows, out = [], {}
    for arm in ("ry0", "hea1"):
        for tag, steps in SCAN:
            p = RUNS / tag / f"{arm}_n{N:02d}.jsonl"
            if not p.exists():
                continue
            r = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
            assert all(x["nfev"] == 2 * steps + 1 for x in r), (tag, arm)
            pg = np.array([x["p_gs"] for x in r])
            by = {}
            for x in r:
                by.setdefault(x["inst"], []).append(x["success"])
            s = {
                "arm": arm, "steps": steps, "trials": len(r), "instances": len(by),
                "success": float(np.mean([x["success"] for x in r])),
                "mean_p_gs": float(pg.mean()), "median_p_gs": float(np.median(pg)),
                "mean_energy": float(np.mean([x["energy_mean"] for x in r])),
                "inst_with_success": int(sum(any(v) for v in by.values())),
                "s_per_trial": float(np.mean([x["wall_s"] for x in r])),
                "per_instance_success": {int(i): float(np.mean(v)) for i, v in sorted(by.items())},
            }
            out[f"{arm}_{steps}"] = s
            rows.append(
                f"| {arm} | {steps} | {s['trials']} | {s['success']:.3f} | {s['mean_p_gs']:.4f} | "
                f"{s['median_p_gs']:.3g} | {s['mean_energy']:.3f} | {s['inst_with_success']}/{s['instances']} | {s['s_per_trial']:.1f} |"
            )
    (REPO / "noiseless" / "results" / "scaling_step_scan_n16_summary.json").write_text(json.dumps(out, indent=1) + "\n")
    print("| arm | SPSA steps | trials | success | mean p(GS) | median p(GS) | mean <H> | inst with >=1 success | s/trial |")
    print("|---|---|---|---|---|---|---|---|---|")
    print("\n".join(rows))


if __name__ == "__main__":
    main()
