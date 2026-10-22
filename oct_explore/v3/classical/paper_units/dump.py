import os,sys; os.environ.setdefault("JAX_PLATFORMS","cpu"); sys.path.insert(0,os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
import sim
TEB=32_000_000_000; R=int(sys.argv[2]); n=int(sys.argv[1])
out=open(f"/workspace/teb_n{n}.txt","w"); seeds=open(f"/workspace/teb_n{n}.seeds","w")
for ins in sim.instances(n,100,TEB):
    m,cl,E=sim.gen_instance(n,ins["seed"]); assert E.min()==0
    line=f"{n} 4 {m} "+" ".join(str((v+1)*s) for vs,ss in cl for v,s in zip(vs,ss))
    for _ in range(R): out.write(line+"\n")
    seeds.write(f"{ins['seed']} {ins['nsol']}\n")
