import itertools, random, json, numpy as np
RATIO=9.93
for n in (8,10,12,14,16):
    states=np.array(list(itertools.product([0,1],repeat=n)),dtype=np.int8); m=int(round(RATIO*n))
    acc=0; tried=0; seed=0; z=0
    while acc<10:
        sd=100000*n+seed; r=random.Random(sd); seed+=1; tried+=1
        E=np.zeros(len(states),int); cl=[]
        for _ in range(m):
            v=sorted(r.sample(range(n),4)); s=[r.choice([1,-1]) for _ in v]; w=r.randint(1,5)
            viol=np.ones(len(states),bool)
            for i,si in zip(v,s): viol&=(states[:,i]==(0 if si==1 else 1))
            E+=w*viol; cl.append({"vars":[i+1 for i in v],"signs":s,"w":w})
        if (E==E.min()).sum()>1: continue
        o=np.argsort(E,kind="stable"); gs="".join(map(str,states[o[0]]))
        nv=sum(1 for c in cl if all(gs[i-1]==("0" if s==1 else "1") for i,s in zip(c["vars"],c["signs"])))
        h=dict(n=n,m=m,ratio=RATIO,seed=sd,gs=gs,E_gs=int(E[o[0]]),n_viol_gs=nv,E_2nd=int(E[o[1]]),n_first_excited=int((E==E[o[1]]).sum()),clauses=cl)
        json.dump(h,open(f"default_n{n:02d}_{acc}.json","w")); z+=h["E_gs"]==0
        print(f"ROW|{n}|{acc}|{m}|{gs}|{h['E_gs']}|{nv}|{h['E_2nd']}|{h['n_first_excited']}"); acc+=1
    print(f"ACC|{n}|{tried}|{z}")
