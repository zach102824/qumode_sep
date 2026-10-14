"""FRESH_INSTANCES table: quantum (QM_NOCLS=1) vs SATLike on the same 100 fresh instances per n; fits a*b^n (R2 in log space)."""
import json, glob, sys
import numpy as np
F = json.load(open('logs/fresh/satlike_fresh100.json'))
B = [json.loads(l) for l in open('logs/strong/satlike_manyinst.jsonl')]
def q(h, p):  # smallest L with fraction(h<=L) >= p  (unsolved = inf)
    h = np.sort(np.array(h, float)); i = int(np.ceil(p * len(h))) - 1; return h[max(i, 0)]
def cap(n): return min(40 * 2 ** n, 4_000_000)
def fit(ns, y):
    ns, y = np.array(ns, float), np.log(np.array(y, float)); b, a = np.polyfit(ns, y, 1); r = y - (a + b * ns)
    return np.exp(a), np.exp(b), 1 - (r @ r) / ((y - y.mean()) @ (y - y.mean()))
rows = []
for n in (8, 10, 12, 14, 16, 18, 20):
    fs = glob.glob(f'noiseless/results/relayout_runs/fresh_nocls/n{n:02d}_*.jsonl')
    if not fs: continue
    R = [json.loads(l) for l in open(fs[0]) if l.strip()]
    if len(R) < 100: continue
    S = sum(r['success'] for r in R); lk = np.mean([r['n_lookups_fixed'] for r in R]); ev = np.mean([r['nfev'] for r in R])
    h = [x if x < cap(n) else np.inf for x in F[str(n)]['hits']]
    allh = [x if x < cap(n) else np.inf for b in B if b['n'] == n for x in b['hits']]
    rows.append(dict(n=n, S=S, p=np.mean([r['p_gs'] for r in R]), lk=lk, ev=ev, tot=lk + ev, sm=q(h, S / 100),
                     s99=q(allh, 0.99), s99_100=q(h, 0.99), Nall=len(allh), med=float(np.median(allh))))
out = ["| n | quantum success | mean p(GS) | lookups | evals | total checks | SATLike, same 100, at quantum's rate | SATLike 99%, same 100 | SATLike 99%, all | SATLike median, all | instances (all) |", "|" + "---|" * 11]
for r in rows:
    out.append(f"| {r['n']} | {r['S']}/100 | {r['p']:.3f} | {r['lk']:.0f} | {r['ev']:.0f} | {r['tot']:.0f} | {r['sm']:.0f} | {r['s99_100']:.0f} | {r['s99']:.0f} | {r['med']:.0f} | {r['Nall']} |")
ns = [r['n'] for r in rows]
if len(rows) >= 3:
    out.append("")
    for k, lab in (('tot', 'quantum total checks'), ('sm', 'SATLike at quantum rate, same 100'), ('s99', 'SATLike 99%, all'), ('med', 'SATLike median, all')):
        a, b, r2 = fit(ns, [r[k] for r in rows]); out.append(f"- {lab}: {a:.1f} * {b:.3f}^n, R2 {r2:.3f}")
print("\n".join(out))
