import numpy as np, json, collections, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
ns=list(range(8,23)); res={}
for c in (1,2,4):
    for n in ns:
        a=np.loadtxt(f'walksatlm_c/out/n{n}_c{c}_r1.txt',ndmin=2)
        res[(c,n)]=(len(a),np.median(a[:,0]),np.median(a[:,1]),np.median(a[:,2]))
R=json.load(open('/tmp/rows.json')); g=collections.defaultdict(list)
for d in R:
    if 'num_trials' in d and 'other_solvers' in d['_path']: g[d['n']].append(d['num_trials'])
pm={n:np.median(v) for n,v in g.items()}
fit=lambda xs,ys: np.polyfit(xs,np.log2(ys),1)
sel=[n for n in ns if 12<=n<=20]
out={'fits':{}}
for c in (1,2,4):
    for j,name in ((1,'tries'),(2,'flips'),(3,'flips+starts')):
        s,i=fit(sel,[res[(c,n)][j] for n in sel]); out['fits'][f'c{c}_{name}']=(round(s,4),round(i,4))
s,i=fit(sorted(pm),[pm[n] for n in sorted(pm)]); out['fits']['paper_raw']=(round(s,4),round(i,4))
print(out)
print('| n | instances | tries (cutoff n) | flips | flips+starts | paper raw median | Table II fit | tries c=2n | tries c=4n |')
print('|---|---|---|---|---|---|---|---|---|')
for n in ns:
    N,t,f,e=res[(1,n)]; print(f'| {n} | {N} | {t:g} | {f:g} | {e:g} | {pm.get(n,"—")} | {2**(-0.309+0.325*n):.1f} | {res[(2,n)][1]:g} | {res[(4,n)][1]:g} |')
fig,ax=plt.subplots(figsize=(6,4)); x=np.arange(8,23)
ax.semilogy(x,2**(-0.309+0.325*x),'k-',label='Table II WalkSATlm fit (Fig. 3)')
ax.semilogy(sorted(pm),[pm[n] for n in sorted(pm)],'ks',mfc='none',label="paper raw data (Zenodo) median")
ax.semilogy(ns,[res[(1,n)][1] for n in ns],'ro',label='ours: C WalkSATlm, tries of n flips')
s,i=out['fits']['c1_tries']; ax.semilogy(x,2**(i+s*x),'r--',label=f'our fit n=12..20: 2^({i:.3f}+{s:.3f}n)')
ax.semilogy(ns,[res[(1,n)][2] for n in ns],'b^',label='ours: flips (old unit)')
ax.set_xlabel('n'); ax.set_ylabel('median running time'); ax.set_title('Random 8-SAT, r=176.54, WalkSATlm'); ax.legend(fontsize=7)
fig.tight_layout(); fig.savefig('results/fig3_walksatlm_repro.png',dpi=150)
