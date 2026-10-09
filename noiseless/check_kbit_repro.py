#!/usr/bin/env python3
"""Compare k-bit relayout (k=3) runs on the legacy n=8 set with the stored headline records
(lblr_xor_vacuum_L4s10_r200_lr0.1): per-trial p(GS) / rounds equality and aggregate metrics."""
import json
import sys
from pathlib import Path

import numpy as np

R = Path(__file__).resolve().parent / "results" / "relayout_runs"
head = json.loads(next((R / "lowbudget").glob("lblr_xor_vacuum_L4s10_r200_lr0.1_*Z.json")).read_text())
old = {(r["ham_file"], int(r["trial"])): r for r in head["records"]}
out = {}
for tag in sys.argv[1:]:
    recs = [json.loads(l) for l in (R / "kbit" / f"{tag}.jsonl").read_text().splitlines() if l.strip()]
    same = sum(1 for r in recs if old[(r["file"], r["trial"])]["p_gs"] == r["p_gs"])
    dp = max(abs(old[(r["file"], r["trial"])]["p_gs"] - r["p_gs"]) for r in recs)
    sel = sum(1 for r in recs if [q["selected"] for q in old[(r["file"], r["trial"])]["rounds"]].index(True) == r["selected_round"])
    out[tag] = {"trials": len(recs), "identical_p_gs": same, "max_abs_dp_gs": dp, "same_selected_round": sel,
                "success": float(np.mean([r["success"] for r in recs])), "mean_p_gs": float(np.mean([r["p_gs"] for r in recs])),
                "evals": float(np.mean([r["nfev"] for r in recs]))}
out["headline_stored"] = {"trials": len(old), "success": float(np.mean([r["success"] for r in old.values()])),
                          "mean_p_gs": float(np.mean([r["p_gs"] for r in old.values()])),
                          "evals": float(np.mean([r["nfev"] for r in old.values()]))}
print(json.dumps(out, indent=1))
