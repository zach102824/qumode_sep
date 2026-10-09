#!/usr/bin/env python3
"""Print success / mean p(GS) / leakage / evals / trials for relayout_runs/kbit tags."""
import sys, json, numpy as np
from pathlib import Path
_REPO = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(_REPO))
from noiseless.run_relayout_kbit import RUN_ROOT, load_records
for t in sys.argv[1:]:
    tr = None
    if ":" in t: t, tr = t.split(":"); tr = int(tr)
    r = load_records(RUN_ROOT / f"{t}.jsonl", tr)
    if not r: print(t, "no data"); continue
    print(f"{t} | {np.mean([x['success'] for x in r]):.3f} | {np.mean([x['p_gs'] for x in r]):.3f} | "
          f"{np.mean([x['leakage'] for x in r]):.4f} | {np.mean([x['nfev'] for x in r]):.0f} | {len(r)}"
          + (f" | lookups {np.mean([x.get('n_lookups',0) for x in r]):.0f}" if 'n_lookups' in r[0] else ""))
