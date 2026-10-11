import json, glob, numpy as np
for f in sorted(glob.glob("noiseless/results/relayout_runs/kbit/n16_*_xL3_xs*_rawk48t3_*wmaxsat_gray_sab128*fixed_low_*[0-9].jsonl")):
    if "_xc" not in f and "_si" not in f: continue
    R = [json.loads(l) for l in open(f) if l.strip()]
    if not R: continue
    fr = [x for r in R for x in r.get("run_guess_ground", [])]
    print(f"{f.split('/')[-1][:-6]} N={len(R)} succ={np.mean([r['success'] for r in R]):.2f} pgs={np.mean([r['p_gs'] for r in R]):.3f} "
          f"evals={np.mean([r['nfev'] for r in R]):.0f} max={max(r['nfev'] for r in R)} lookups={np.mean([r['n_lookups'] for r in R]):.0f} "
          f"Erun={np.mean([r['n_explore_run'] for r in R]):.1f} find={np.mean(fr) if fr else float('nan'):.2f}")
