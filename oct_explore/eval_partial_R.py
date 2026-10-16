import json,sys,os,time,gc; sys.path.insert(0,'.')
import numpy as np, sim
r=json.load(open('results/R_raw.json')); log=lambda *s: print(time.strftime("%H:%M:%S"),*s,flush=True)
for n,N in [(8,100),(10,80),(12,60),(14,80)]:
  ins=sim.instances(n,N,32_000_000_000)
  for L in sorted(r['train'], key=int):
    x=np.array(r['train'][L]['x']); p=[]; ch=10 if n<=12 else 5
    for i in range(0,N,ch):
      c=sim.Circuit(n,ins[i:i+ch],sim.LAYOUTS_R[n],tbit=True)
      p+=list(c.probs(x)); del c; gc.collect()
    s=sim.summarize(p,ins); log(f"R n={n} L={L} med={s['median_inv_p']:.1f} B90={s['B90_inv_p']:.1f} rand={s['rand_median']:.0f}")
log("PARTIALDONE")
