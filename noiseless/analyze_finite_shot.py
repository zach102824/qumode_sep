"""Summarise the --shots 1000 sl3 reruns into noiseless/results/FINITE_SHOT_1000.md (tables only section)."""
import glob, json, sys
import numpy as np
SL3 = {8: 0.987, 10: 0.988, 12: 0.991, 14: 0.961, 16: 0.972, 18: 0.963, 20: 0.959}
rows = []
for n in range(8, 21, 2):
    fs = glob.glob(f"noiseless/results/relayout_runs/shots1000/n{n:02d}_*.jsonl")
    if not fs: continue
    r = [json.loads(l) for l in open(fs[0]) if l.strip()]
    N = 2 ** n
    g = lambda k: np.array([x[k] for x in r], float)
    tot = g("n_unique_total") / N
    rows.append(f"| {n} | {len(r)}/10 | {int(sum(x['success'] for x in r))}/{len(r)} | {g('p_gs').mean():.3f} | {g('p_gs_shots').mean():.3f} | {SL3[n]:.3f} | "
                f"{g('nfev').mean():.0f} | {g('n_shot_evals').mean():.0f} | {g('n_lookups_raw').mean():.0f} | {g('n_lookups_distinct').mean():.1f} | "
                f"{g('n_unique_sampled').mean():.1f} | {g('n_unique_total').mean():.1f} | {N} | {100*tot.mean():.2f}% | {100*tot.max():.2f}% |")
    per = " ; ".join(f"i{x['inst']}:{'Y' if x['success'] else 'n'} p={x['p_gs']:.3f} samp={x['n_unique_sampled']} look={x['n_lookups_distinct']} tot={x['n_unique_total']}" for x in sorted(r, key=lambda x: x['inst']))
    rows.append(f"|  | per-trial: {per} |" + " |" * 13)
hdr = ("| n | trials | success | mean exact p(GS) | mean shot p(GS) | sl3 exact-cost p(GS) | SPSA evals (nfev) | shot evals (all circuit evals) | raw lookups | unique lookups | unique sampled | unique total | 2^n | total/2^n mean | total/2^n max |\n"
       "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
print(hdr); print("\n".join(rows))
