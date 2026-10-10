#!/usr/bin/env python3
"""Paired comparison of relayout checkpoints on the (inst, trial) pairs both runs share.
Usage: analyze_paired.py BASE.jsonl VAR.jsonl [VAR2.jsonl ...]  (names relative to the kbit run dir if not paths)."""
import json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent / "results" / "relayout_runs" / "kbit"

def load(p):
    p = Path(p) if Path(p).exists() else ROOT / (p if p.endswith(".jsonl") else p + ".jsonl")
    d = {}
    for l in p.read_text().splitlines():
        try:
            r = json.loads(l); d[(r["inst"], r["trial"])] = r
        except Exception:
            pass
    return d

def stats(d, keys):
    R = [d[k] for k in keys]
    return (len(R), np.mean([r["success"] for r in R]), np.mean([r["p_gs"] for r in R]),
            np.mean([r["nfev"] for r in R]), np.mean([r["n_lookups_distinct"] for r in R]))

if __name__ == "__main__":
    base = load(sys.argv[1])
    for v in sys.argv[2:]:
        var = load(v)
        keys = sorted(set(base) & set(var))
        b, x = stats(base, keys), stats(var, keys)
        print(f"{Path(v).stem[:95]}\n  paired n={b[0]}: base succ={b[1]:.3f} p={b[2]:.3f} evals={b[3]:.0f} look={b[4]:.0f} | "
              f"var succ={x[1]:.3f} p={x[2]:.3f} evals={x[3]:.0f} look={x[4]:.0f}")
