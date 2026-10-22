import numpy as np, json, collections, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
R=json.load(open('walksatlm_c/walksatlm_rows.json')); g=collections.defaultdict(list)
for d in R:
    if 'num_trials' in d and 'other_solvers' in d['_path']: g[d['n']].append(d['num_trials'])
pm={n:np.median(v) for n,v in g.items()}
ns=list(range(8,23)); med={}; N={}
for n in ns:
    a=np.concatenate([np.loadtxt(f'walksatlm_c/var/p0.30_s{s}_n{n}.txt',ndmin=2)[:,0] for s in (1,2,3)]); med[n]=np.median(a); N[n]=len(a)//3
sel=[n for n in ns if 12<=n<=20]; s,i=np.polyfit(sel,np.log2([med[n] for n in sel]),1)
print(f'FIT {i:.4f} + {s:.4f} n')
print('| n | instances | ours p=0.30 median tries | paper raw median | Table II fit |'); print('|---|---|---|---|---|')
for n in ns: print(f'| {n} | {N[n]} | {med[n]:g} | {pm.get(n,"—")} | {2**(-0.3085+0.3246*n):.1f} |')
x=np.arange(8,23); fig,ax=plt.subplots(figsize=(6,4))
ax.semilogy(x,2**(-0.3085+0.3246*x),'k-',label='Table II / Fig. 3 fit: 2^(-0.309+0.325n)')
ax.semilogy(sorted(pm),[pm[n] for n in sorted(pm)],'ks',mfc='none',label='paper raw (Zenodo) medians')
ax.semilogy(ns,[med[n] for n in ns],'ro',label='ours: C WalkSATlm p=0.30,w1=6,w2=5, n-flip tries')
ax.semilogy(x,2**(i+s*x),'r--',label=f'our fit n=12..20: 2^({i:.3f}+{s:.3f}n)')
ax.set_xlabel('n'); ax.set_ylabel('median running time (tries of n flips)'); ax.set_title('Random 8-SAT r=176.54: WalkSATlm (Fig. 3)'); ax.legend(fontsize=7)
fig.tight_layout(); fig.savefig('results/fig3_walksatlm_repro.png',dpi=150)
