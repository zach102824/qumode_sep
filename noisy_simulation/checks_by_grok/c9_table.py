"""(9) Parse c9_logs/<mutation>__<check>.log with pass criteria per check; print mutation x check table."""
import re, glob, os, math
L=lambda p: open(p).read()
f=lambda s: [float(x) for x in re.findall(r"[-+]?\d+\.?\d*(?:e[-+]?\d+)?",s)]
def crit(check,t):
    if "Traceback" in t: return "CRASH"
    bad=[]
    if check=="check_noise":
        for m in re.finditer(r"cavity kt=\S+ jmax=\S+: (\S+)|qubit: (\d\S*)$",t,re.M):
            v=float(m.group(1) or m.group(2)); bad+=[v>1e-12]
        bad+=[abs(float(x)-1)>1e-6 for x in re.findall(r"ratio=(\S+)",t)]
        m=re.search(r"P_e=(\S+) expect (\S+);\s+\|rho_ge\|=(\S+) expect (\S+)",t); a=list(map(float,m.groups())); bad+=[abs(a[0]-a[1])>1e-6,abs(a[2]-a[3])>1e-6]
        bad+=[abs(float(x))>1e-5 for x in re.findall(r"tr-1=(\S+)",t)]+[float(x)<-1e-9 for x in re.findall(r"mineig=(\S+)",t)]
        F=[(float(s),float(a),float(b)) for s,a,b in re.findall(r"scale=\s*(\S+): F_fast=(\S+) F_direct=(\S+)",t)]
        bad+=[abs(F[0][1]-1)>1e-9 or abs(F[0][2]-1)>1e-9]+[abs(a-b)>1e-5 for s,a,b in F if s<=1]+[any(F[i+1][1]>F[i][1] for i in range(len(F)-1))]
    elif check=="check_layer_vs_qutip":
        for s,a,b in re.findall(r"scale=(\S+): .*TD\(fast,ME\)=(\S+)\s+TD\(direct,ME\)=(\S+)",t):
            lim=2e-4 if float(s)<=1 else 1e-3; bad+=[float(a)>lim,float(b)>lim]
    elif check=="c1_convergence":
        for line in t.splitlines():
            if not line.startswith("scale="): continue
            sc=float(line.split(":")[0][6:]); e={int(n):float(v) for n,v in re.findall(r"N=(\d+):(\S+)",line)}; sl=float(line.split("slope(N>=8)=")[1])
            bad+=[abs(sl+2)>0.2, e[8]>(2e-4 if sc<=1 else 3e-3)]
    elif check=="c2_first_order":
        r=[float(x) for x in re.findall(r"ratio to prediction=(\S+)",t)]; bad+=[abs(r[0]-1)>1e-3]
    elif check=="c3_isolated":
        blocks=re.split(r"\n(?=\w+ only:)",t)
        for b in blocks:
            for m in re.finditer(r"TD\(de\) noisy-vs-ideal=(\S+), \|de - analytic channel\|max=(\S+); TD\(AB\)=\S+, max\|dFockpop\(AB\)\|=(\S+); .*ratios (\S+),(\S+) vs e\^-kT=(\S+)",b):
                tde,an,dfp,r1,r2,ek=map(lambda s: float(s.rstrip(',')),m.groups())
                bad+=[an>1e-9, abs(r1-ek)>1e-5, abs(r2-ek)>1e-5]
                if b.startswith("kappa"): bad+=[tde>1e-9]
                else: bad+=[dfp>1e-9]
            if b.startswith("kappa"):   # rates must be what the device file says
                k=float(re.search(r"kappa=(\S+)",b).group(1)); bad+=[abs(k-2294)>2]
            if b.startswith("gamma only"): bad+=[abs(float(re.search(r"gamma=(\S+)",b).group(1))-2e4)>10]
    elif check=="c4_longtime":
        bad+=[abs(float(x))>1e-6 for x in re.findall(r"tr-1=(\S+)",t)]+[float(x)<-1e-9 for x in re.findall(r"mineig=(\S+)",t)]
        P=[float(x) for x in re.findall(r"1-P\(vac,g,g\)=(\S+)",t)]; bad+=[P[-2]>1e-5, P[-1]>1e-6]
        bad+=[abs(P[2]-0.3441)>0.01]     # 500-layer value must sit on the e^{-kappa t} curve, not just reach vacuum
    elif check=="c5_mcsolve":
        bad+=[abs(float(z))>4 for z in re.findall(r"z=(\S+)",t)]
    elif check=="c6_fits":
        for v,ref in zip(f(" ".join(re.findall(r"=(\S+) us",t))),[50,30,50,30,436,436,50,30]): bad+=[abs(v/ref-1)>5e-3]
    return "caught" if any(bad) else "pass"
checks=["check_noise","check_layer_vs_qutip","c1_convergence","c2_first_order","c3_isolated","c4_longtime","c5_mcsolve","c6_fits"]
muts=["none","gamma2","gphi_no_half","wrong_cavity","drop_wait2","idle_twice","qubit_adjoint","cavity_adjoint","sign_exponent","jmax1","transmon_wrong_mode","kappa_over_2pi"]
short=["CN","LQ","c1","c2","c3","c4","c5","c6"]
print(f"{'mutation':22s}"+"".join(f"{s:>8s}" for s in short)+"   #caught")
for m in muts:
    row=[crit(c,L(f"c9_logs/{m}__{c}.log")) for c in checks]
    print(f"{m:22s}"+"".join(f"{r:>8s}" for r in row)+f"   {sum(r!='pass' for r in row)}/8")
