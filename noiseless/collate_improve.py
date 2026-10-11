#!/usr/bin/env python3
"""Collate logs/improve/*.json run summaries (+ lookups from the kbit jsonl) into a table."""
import json, sys, re
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "noiseless/results/relayout_runs/kbit"
rows = []
for p in sorted((ROOT / "logs/improve").glob("*.json")):
    t = p.read_text()
    i = t.find("\n{")
    if i < 0 and not t.startswith("{"):
        continue
    try:
        d = json.loads(t[i:] if i >= 0 else t)
    except Exception:
        continue
    lk = float("nan")
    jl = RUNS / f"{d['tag']}.jsonl"
    if jl.exists():
        v = []
        for l in jl.read_text().splitlines():
            try: v.append(json.loads(l)["n_lookups_distinct"])
            except Exception: pass
        if v: lk = float(np.mean(v))
    rows.append((p.stem, d["n_trials"], d["success"], d["mean_p_gs"], d["evals"], lk))
if __name__ == "__main__":
    flt = sys.argv[1] if len(sys.argv) > 1 else ""
    for r in rows:
        if re.search(flt, r[0]):
            print(f"{r[0]:48s} tr={r[1]:3d} succ={r[2]:.2f} p={r[3]:.3f} evals={r[4]:.0f} lookups={r[5]:.0f}")
