"""Tables for FINITE_SHOT_FIXES.md R1-R3 section, from relayout_runs/shotfix2/<V>/*.jsonl (+ shotfix V2_S100 baseline)."""
import glob, json, sys
import numpy as np
VS = sys.argv[1:] or ["B100", "R1", "R2", "R3"]
print("| setting | n | success | mean exact p(GS) | unique strings checked (mean) | share of 2^n mean | share max | total shots (mean) | SPSA evals (nfev) | shot evals |")
print("|---|---|---|---|---|---|---|---|---|---|")
per = []
for V in VS:
    for n in (10, 12, 14, 16):
        fs = glob.glob(f"noiseless/results/relayout_runs/shotfix2/{V}/n{n:02d}_*.jsonl")
        if not fs and V == "B100":
            fs = glob.glob(f"noiseless/results/relayout_runs/shotfix/V2_S100/n{n:02d}_*.jsonl")
        if not fs: continue
        r = sorted((json.loads(l) for l in open(fs[0]) if l.strip()), key=lambda x: x["inst"])
        N = 2 ** n; g = lambda k: np.array([x.get(k, np.nan) for x in r], float); t = g("n_unique_total") / N
        tot = g("n_shots_total") if "n_shots_total" in r[0] else g("n_shot_evals") * r[0]["shots"]
        print(f"| {V} | {n} | {int(sum(x['success'] for x in r))}/{len(r)} | {g('p_gs').mean():.3f} | {g('n_unique_total').mean():.1f} | "
              f"{100*t.mean():.2f}% | {100*t.max():.2f}% | {tot.mean():,.0f} | {g('nfev').mean():.0f} | {g('n_shot_evals').mean():.0f} |")
        per.append(f"- {V} n={n}: " + " ; ".join(f"i{x['inst']}:{'Y' if x['success'] else 'n'} p={x['p_gs']:.3f} tot={x['n_unique_total']}" for x in r))
print("\nPer-trial:\n" + "\n".join(per))
