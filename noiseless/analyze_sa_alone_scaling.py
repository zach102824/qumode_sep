import json, math, numpy as np
from pathlib import Path
from scipy.optimize import curve_fit
R = [json.loads(l) for l in Path("logs/3cav/saalone/hits.jsonl").read_text().splitlines()]
HY = {8:152,10:190,12:228,14:266,16:400,18:558,20:740}
def succ(h, B): return np.mean([x is not None and max(x,64) <= B for x in h])
rows = []
for n in sorted(HY):
    h = [r["hit"] for r in R if r["n"] == n]
    assert len(h) == 200
    lo, hi = 0, 64
    while succ(h, hi) < 0.9: lo, hi = hi, hi * 2
    while hi - lo > 1:
        m = (lo + hi) // 2
        if succ(h, m) >= 0.9: hi = m
        else: lo = m
    hh = np.array([max(x,64) if x is not None else np.inf for x in h])
    per = [succ([r["hit"] for r in R if r["n"]==n and r["inst"]==i], HY[n]) for i in range(10)]
    rows.append(dict(n=n, hyb=HY[n], s_hyb=succ(h, HY[n]), inst_min=min(per), inst_max=max(per), B90=hi,
                     med=float(np.median(hh)), mean=float(hh.mean()), fails=int(np.isinf(hh).sum()),
                     rand=2**n*math.log(10), s_2x=succ(h, 2*HY[n]), s_4x=succ(h, 4*HY[n])))
ns = np.array([r["n"] for r in rows], float); B = np.array([r["B90"] for r in rows], float)
def r2(y, f): return 1 - np.sum((y-f)**2)/np.sum((y-y.mean())**2)
le = np.polyfit(ns, np.log(B), 1); a_e, b_e = math.exp(le[1]), math.exp(le[0])
lp = np.polyfit(np.log(ns), np.log(B), 1); a_p, k_p = math.exp(lp[1]), lp[0]
fe = a_e*b_e**ns; fp = a_p*ns**k_p
hy = np.array([r["hyb"] for r in rows], float)
lh = np.polyfit(np.log(ns), np.log(hy), 1); leh = np.polyfit(ns, np.log(hy), 1)
out = ["| n | hybrid lookups | SA-alone success @ hybrid budget (200 runs) | per-instance range | @2x | @4x | B90 (SA-alone) | B90/hybrid | median lookups to GS | mean | random 2^n ln10 | B90/2^n |",
       "|---|---|---|---|---|---|---|---|---|---|---|---|"]
for r in rows:
    out.append(f"| {r['n']} | {r['hyb']} | {r['s_hyb']:.3f} | {r['inst_min']:.2f}-{r['inst_max']:.2f} | {r['s_2x']:.3f} | {r['s_4x']:.3f} | {r['B90']} | {r['B90']/r['hyb']:.2f} | {r['med']:.0f} | {r['mean']:.0f} | {r['rand']:.0f} | {r['B90']/2**r['n']:.4f} |")
out.append("")
out.append(f"Exponential fit B90 = a*b^n: a = {a_e:.3f}, b = {b_e:.4f} (= 2^{math.log2(b_e):.3f}n), R^2(log) = {r2(np.log(B), np.log(fe)):.4f}, R^2(lin) = {r2(B, fe):.4f}")
out.append(f"Power-law fit B90 = a*n^k: a = {a_p:.4g}, k = {k_p:.3f}, R^2(log) = {r2(np.log(B), np.log(fp)):.4f}, R^2(lin) = {r2(B, fp):.4f}")
out.append(f"Hybrid lookups for comparison: power-law k = {lh[0]:.3f} (R^2 log {r2(np.log(hy), np.polyval(lh, np.log(ns))):.4f}); exp b = {math.exp(leh[0]):.4f} (R^2 log {r2(np.log(hy), np.polyval(leh, ns)):.4f})")
print("\n".join(out))
