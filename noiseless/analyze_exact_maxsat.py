import json, numpy as np, statistics as st
R=[json.loads(l) for l in open("logs/3cav/exact_maxsat/results.jsonl")]
out=[]
def fit(ns,ys):
    ns=np.array(ns,float); y=np.log(np.array(ys,float))
    b,a=np.polyfit(ns,y,1); r2=1-((y-(a+b*ns))**2).sum()/((y-y.mean())**2).sum()
    p,c=np.polyfit(np.log(ns),y,1); r2p=1-((y-(c+p*np.log(ns)))**2).sum()/((y-y.mean())**2).sum()
    return f"{np.exp(a):.4g}*{np.exp(b):.4f}^n (R2 {r2:.3f}); {np.exp(c):.4g}*n^{p:.2f} (R2 {r2p:.3f})"
for alg in ("rc2","fm"):
    out.append(f"\n### {alg.upper()} (Glucose4 oracle)\n\n| n | instances | new? | GS match | median t (ms) | mean t (ms) | median SAT calls | mean SAT calls | median cores | median conflicts | mean conflicts | median decisions | median propagations |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    rows={}
    for n in sorted({r["n"] for r in R if r["alg"]==alg}):
        g=[r for r in R if r["alg"]==alg and r["n"]==n]; rows[n]=g
        f=lambda k:[r[k] for r in g if r[k] is not None]
        m="n/a (new)" if g[0]["new"] else f"{sum(r['match_gs'] for r in g)}/{len(g)}"
        cr=f("cores"); out.append(f"| {n} | {len(g)} | {'yes' if g[0]['new'] else 'no'} | {m} | {1e3*st.median(f('t_med')):.3f} | {1e3*np.mean(f('t_med')):.3f} | {st.median(f('calls')):g} | {np.mean(f('calls')):.1f} | {st.median(cr) if cr else '-'} | {st.median(f('conflicts')):g} | {np.mean(f('conflicts')):.1f} | {st.median(f('decisions')):g} | {st.median(f('propagations')):g} |")
    for lab,sel in (("n=8-20 (original)",lambda n:n<=20),("all n",lambda n:True)):
        ns=[n for n in rows if sel(n)]
        for k in ("t_med","calls","conflicts"):
            ys=[max(st.median([r[k] for r in rows[n]]),1e-9 if k=="t_med" else 1) for n in ns]
            out.append(f"\n- fit {lab}, median {k}: {fit(ns,ys)}")
print("\n".join(out))
