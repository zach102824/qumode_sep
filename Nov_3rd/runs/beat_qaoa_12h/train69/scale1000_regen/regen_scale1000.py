import sys, hashlib, time, numpy as np
sys.path.insert(0,'.')
from explore_OCT29.lib.sat_gen import instances, pack_clauses, clause_count
SCB=54_000_000_000
for n in map(int, sys.argv[1].split(',')):
    t0=time.time(); m=clause_count(n)
    ins=instances(n,1000,SCB,compute_E=True)
    seeds=np.array([i['seed'] for i in ins],np.int64); nsol=np.array([i['nsol'] for i in ins],np.int32)
    cv,cs=pack_clauses([i['cl'] for i in ins],m,n); E=np.stack([i['E'] for i in ins]).astype(np.uint8)
    sp={}
    for p in ('scale1000','test'):
        sp.update({f'{p}_seeds':seeds,f'{p}_nsol':nsol,f'{p}_cls_v':cv,f'{p}_cls_s':cs,f'{p}_E':E})
    out=f'Nov_3rd/data/scale1000/data_n{n}.npz'
    np.savez_compressed(out,n=np.int32(n),m=np.int32(m),store_E=np.array([1],np.int32),**sp)
    h=hashlib.sha256(open(out,'rb').read()).hexdigest()
    print(n, out, 'seeds', seeds.min(), seeds.max(), 'sha', h[:12], f'{time.time()-t0:.0f}s', flush=True)
