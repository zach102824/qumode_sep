"""Derive K(n) (top readouts) and polt(n) (fix-up centres) from run_gs_diag records.
Usage: python min_lookup_rule_fit.py OUT.json DIR [DIR...]
Per trial: K_need = 1 + min over explore runs of rank_h1 (rank, in that run's readout order, of the first string within
Hamming 1 of the GS); polt_need = 1 + energy rank of that string in the raw pool at that run (rawk=48 pool, so an upper
bound for smaller K). Rule: smallest K (and polt) whose coverage is within SLACK of the coverage at K=48 (polt=3)."""
import json, glob, sys, collections, numpy as np
SLACK = 0.02
out, dirs = sys.argv[1], sys.argv[2:]
by = collections.defaultdict(list)
for d in dirs:
    for f in glob.glob(f"{d}/n*.jsonl"):
        for l in open(f):
            try: r = json.loads(l)
            except Exception: continue
            dg = [x for x in r.get("run_gs_diag", []) if x]
            if r.get("rawk") != 48 or not dg: continue
            c = [(x["rank_h1"] + 1, x["pool_erank_h1"] + 1) for x in dg if x["rank_h1"] >= 0]
            k = min(c)[0] if c else 10**9
            p = min((pp for kk, pp in c if kk <= 48), default=10**9)
            by[r["n"]].append((k, p, [x["rank_h1"] for x in dg], [x.get("p_gs_run") for x in dg]))
res = {}
for n in sorted(by):
    ks = np.array([a for a, *_ in by[n]]); ps = np.array([b for _, b, *_ in by[n]])
    cov = lambda K: float(np.mean(ks <= K)); cp = lambda P: float(np.mean(ps <= P))
    K = next(K for K in range(1, 49) if cov(K) >= cov(48) - SLACK)
    P = next(P for P in (1, 2, 3) if cp(P) >= cp(3) - SLACK)
    allr = [x for _, _, rr, _ in by[n] for x in rr if x >= 0]
    res[n] = dict(trials=len(ks), cov48=cov(48), K=K, covK=cov(K), polt=P, cov_polt3=cp(3), cov_poltP=cp(P),
                  K_quantiles={q: float(np.quantile(ks[ks < 10**9], q)) for q in (0.5, 0.9, 0.95)} if (ks < 10**9).any() else {},
                  per_run_rank_h1_median=float(np.median(allr)) if allr else None)
    print(n, res[n])
json.dump(res, open(out, "w"), indent=1)
