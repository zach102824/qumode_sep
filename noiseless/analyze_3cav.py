"""Summaries for THREE_CAVITY_N16: success, mean p(GS), evals, lookups, leakage, per-run find rate."""
import json, sys
import numpy as np
from noiseless.run_relayout_kbit import RUN_ROOT, load_records
for tag in sys.argv[1:]:
    R = load_records(RUN_ROOT / f"{tag}.jsonl")
    if not R:
        print(tag, "no records"); continue
    fr = [np.mean(r["run_guess_ground"]) for r in R if r.get("run_guess_ground")]
    allmiss = sum(1 for r in R if r.get("run_guess_ground") and not any(r["run_guess_ground"]))
    print(json.dumps(dict(tag=tag, trials=len(R), success=np.mean([r["success"] for r in R]),
          mean_p_gs=round(float(np.mean([r["p_gs"] for r in R])), 4), evals=float(np.mean([r["nfev"] for r in R])),
          lookups=float(np.mean([r["n_lookups"] for r in R])), leak_max=float(max(r["leakage"] for r in R)),
          find_rate=round(float(np.mean(fr)), 3) if fr else None, all_miss_trials=allmiss,
          sa_best_is_gs=float(np.mean([r["sa_best_is_ground"] for r in R])),
          wall_s=round(float(np.mean([r["wall_s"] for r in R])), 1))))
