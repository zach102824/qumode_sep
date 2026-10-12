"""Offline (classical-only) analysis for MIN_LOOKUP_RULE.md."""
import json, glob, sys, numpy as np
sys.path.insert(0,'.')
from noiseless.run_relayout_kbit import ham_paths, energies_for, trial_seed
from noiseless.classical_layout import sa_sample, marginals
# 1) SA layout quality vs budget
print("## SA layout quality vs budget (instances x 10 seeds)")
print("n budget | sa_best_is_GS | GS rank in SA top24 (median) | sure-bit agreement (top n/2 surest bits match GS) | mean sure-bit match frac")
for n in range(8,21,2):
    for b in [64,80,96,128,162,200,256,400]:
        if b > 0.8*2**n: continue
        hit=[];agr=[];frac=[]
        for i,p in enumerate(ham_paths(n,"wmaxsat")):
            E,gs=energies_for(str(p),n,"wmaxsat"); gl=int(gs,2) if isinstance(gs,str) else int(gs)
            gb=(gl>>(n-1-np.arange(n)))&1
            for t in range(10):
                rng=np.random.default_rng(trial_seed(20260917,i,t)+7)
                seen=sa_sample(E,n,b,rng); c,m=marginals(seen,n)
                hit.append(c[0]==gl)
                sure=np.argsort(np.minimum(m,1-m))[:n//2]
                mf=((m[sure]>0.5)==gb[sure]); agr.append(mf.all()); frac.append(mf.mean())
        print(f"{n} {b} | {np.mean(hit):.2f} | | {np.mean(agr):.2f} | {np.mean(frac):.3f}",flush=True)
