#!/usr/bin/env python3
"""Summarize relayout jsonl runs for IMPROVE_N12_N14: per-explore-run find rate, guess after explore,
final success, mean p(GS), evals, lookups (all classical incl. SA), per-round monotonicity, fixes."""
import json, sys
from pathlib import Path
import numpy as np

def summ(path):
    recs = {}
    for l in Path(path).read_text().splitlines():
        try:
            r = json.loads(l); recs[(r["inst"], r["trial"])] = r
        except Exception:
            pass
    R = list(recs.values())
    if not R:
        return None
    find = [q["polished_is_ground"] for r in R for q in r["rounds"] if q["phase"] == "explore"]
    nx = [sum(q["phase"] == "explore" for q in r["rounds"]) for r in R]
    gx = [r["rounds"][nx[i] - 1]["next_guess_is_ground"] for i, r in enumerate(R)]
    fixed = sum(1 for i, r in enumerate(R) if not gx[i] and r["rounds"][-1]["next_guess_is_ground"])
    lost = sum(1 for i, r in enumerate(R) if gx[i] and not r["rounds"][-1]["next_guess_is_ground"])
    return dict(file=Path(path).stem, trials=len(R), find_rate=float(np.mean(find)), guess_after_explore=float(np.mean(gx)),
                success=float(np.mean([r["success"] for r in R])), mean_p_gs=float(np.mean([r["p_gs"] for r in R])),
                evals=float(np.mean([r["nfev"] for r in R])), lookups=float(np.mean([r["n_lookups_distinct"] for r in R])),
                fixed=fixed, lost=lost, sa_hit=float(np.mean([r.get("sa_best_is_ground", False) for r in R])))

if __name__ == "__main__":
    for p in sys.argv[1:]:
        s = summ(p)
        if s:
            print(" ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in s.items()))
