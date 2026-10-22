"""Fits for Cai 4-SAT WalkSATlm scan (w1=3,w2=1). b,C from log2-linear fit; 95% bootstrap CI (2000 resamples of formulas)."""
import json, numpy as np, common as CM
rng = np.random.default_rng(0); NS = [8, 12, 16, 20]
mb = lambda x: (np.median(x), np.quantile(x, .9))
def fit(ys): s, i = np.polyfit(NS, np.log2(ys), 1); return 2**s, 2**i
for P in ["0.40", "0.45", "0.51", "0.55"]:
    W = {n: np.array(json.load(open(f"{CM.RES}/W4_p{P}_n{n}.json"))["lookups"]) for n in NS}
    st = [mb(W[n]) for n in NS]; out = [P]
    for q in (0, 1):
        b, C = fit([s[q] for s in st])
        bs = [fit([mb(rng.choice(W[n], len(W[n])))[q] for n in NS])[0] for _ in range(2000)]
        lo, hi = np.percentile(bs, [2.5, 97.5])
        out.append(f"{['median','B90'][q]}: " + " / ".join(f"{s[q]:.3g}" for s in st) + f" | b={b:.3f} [{lo:.3f},{hi:.3f}] C={C:.2f}")
    print(" || ".join(out))
