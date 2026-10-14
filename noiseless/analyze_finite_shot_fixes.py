"""Tables for noiseless/results/FINITE_SHOT_FIXES.md from relayout_runs/shotfix/<V>_S<S>/*.jsonl."""
import glob, json
import numpy as np, sys
NS = tuple(int(a) for a in sys.argv[1:]) or (8, 10)
NAME = {"V0": "plain", "V1": "lowE readout", "V2": "lowE + CRN", "V3": "lowE + CRN + CVaR0.1"}
print("| variant | S | n | success | mean exact p(GS) | mean shot p(GS) | SPSA evals (nfev) | shot evals | raw lookups | unique lookups | unique total | 2^n | total/2^n mean | total/2^n max |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
per = []
for S in (100, 1000):
    for V in NAME:
        for n in NS:
            fs = glob.glob(f"noiseless/results/relayout_runs/shotfix/{V}_S{S}/n{n:02d}_*.jsonl")
            if not fs: continue
            r = sorted((json.loads(l) for l in open(fs[0]) if l.strip()), key=lambda x: x["inst"])
            N = 2 ** n; g = lambda k: np.array([x[k] for x in r], float); t = g("n_unique_total") / N
            print(f"| {V} {NAME[V]} | {S} | {n} | {int(sum(x['success'] for x in r))}/{len(r)} | {g('p_gs').mean():.3f} | {g('p_gs_shots').mean():.3f} | "
                  f"{g('nfev').mean():.0f} | {g('n_shot_evals').mean():.0f} | {g('n_lookups_raw').mean():.0f} | {g('n_lookups_distinct').mean():.1f} | "
                  f"{g('n_unique_total').mean():.1f} | {N} | {100*t.mean():.2f}% | {100*t.max():.2f}% |")
            per.append(f"- {V} S={S} n={n}: " + " ; ".join(f"i{x['inst']}:{'Y' if x['success'] else 'n'} p={x['p_gs']:.3f} tot={x['n_unique_total']}" for x in r))
print("\nPer-trial:\n" + "\n".join(per))
