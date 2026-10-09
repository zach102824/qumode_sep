#!/usr/bin/env python3
"""Markdown table of every explore-exploit (and reference) full run in relayout_runs/kbit, grouped by n."""
import sys, json, re, numpy as np
from pathlib import Path
_REPO = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(_REPO))
from noiseless.run_relayout_kbit import RUN_ROOT, load_records
REF = {8: ["n08_L4s50_r400_R8"], 10: ["n10_L4s50_r400_R8", "n10_L4s100_r800_R8"], 12: []}
rows = {}
for p in sorted(RUN_ROOT.glob("n*_E*K*.jsonl")) + [RUN_ROOT / f"{t}.jsonl" for v in REF.values() for t in v]:
    r = load_records(p, 5)
    if len(r) < 100:
        continue
    n = r[0]["n"]
    rows.setdefault(n, []).append((p.stem, r))
out = []
for n in sorted(rows):
    out.append(f"\n### n = {n} (2^n = {1 << n})\n")
    out.append("| setting | success | mean p(GS) | leakage | evals | lookups | lookups/2^n | trials |")
    out.append("|---|---|---|---|---|---|---|---|")
    items = []
    for tag, r in rows[n]:
        lk = np.mean([x.get("n_lookups_distinct", x.get("n_lookups", len(x["rounds"]) * (n + 1))) for x in r])
        if "n_lookups_distinct" not in r[0]:
            tag = tag + " (lookups w/ repeats)"
        items.append((np.mean([x["nfev"] for x in r]), tag, np.mean([x["success"] for x in r]), np.mean([x["p_gs"] for x in r]),
                      np.mean([x["leakage"] for x in r]), lk, len(r)))
    for ev, tag, s, p, l, lk, nt in sorted(items):
        out.append(f"| {tag[4:]} | {s:.3f} | {p:.3f} | {l:.4f} | {ev:.0f} | {lk:.0f} | {lk / (1 << n):.3f} | {nt} |")
print("\n".join(out))
