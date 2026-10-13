"""QuTiP mcsolve trajectories, chained per segment (idle -> Ry -> wait -> JP_MID -> wait -> Ry^-1),
pulses instantaneous as in the repo. Each trajectory's ket is carried across segments."""
from common import *
nf=3; NT=2000
dev=EICKBUSCH.with_(jmax=None,n_slice=64,noise_scale=5.0)   # 5x rates so noise >> MC error
lay=layer0(nf,sc=0.7); r4=C.ideal_block(C.vacuum_rho(nf),lay,nf)
w,v=np.linalg.eigh(as_matrix(r4)); psi4=v[:,-1]
H0,c=lind5(dev,nf); dims=[[2,2,2,nf,nf],[1]*5]
U=lambda M: op5(qt.Qobj(M),0,nf)
o={"atol":1e-10,"rtol":1e-8,"store_final_state":True,"keep_runs_results":True,"progress_bar":False}
s_idle=qt.MCSolver(0*H0,c,options=o); s_wait=qt.MCSolver(H0,c,options=o)
rng=np.random.default_rng(11); kets=[]
psi0=qt.Qobj(np.kron([1,0],psi4),dims=dims)
for i in range(NT):
    k=s_idle.run(psi0,[0,dev.tau_ecd],ntraj=1,seeds=int(rng.integers(2**31))).runs_final_states[0]
    for M in (C.RY_P,C.JP_MID):
        k=U(M)*k; k=s_wait.run(k,[0,dev.tau_wait],ntraj=1,seeds=int(rng.integers(2**31))).runs_final_states[0]
    k=(U(C.RY_M)*k).full().ravel(); kets.append(k/np.linalg.norm(k))
K=np.array(kets)
rep=as_matrix(C.jp_direct(C.transmon_idle(C.cavity_idle(r4,dev,dev.tau_ecd),dev,dev.tau_ecd),dev,keep_q=True))
ideal=as_matrix(C.jp_direct(r4,NOISELESS,keep_q=True)); ev,V=np.linalg.eigh(ideal); pid=V[:,-1]
a=qt.destroy(nf); obs={"F(ideal)":np.outer(pid,pid.conj()),"<n_A>":op5(a.dag()*a,3,nf).full(),"<n_B>":op5(a.dag()*a,4,nf).full(),"P_e(d)":op5(Pe,1,nf).full(),"P_e(q)":op5(Pe,0,nf).full()}
for l,O in obs.items():
    vals=np.real(np.einsum('ti,ij,tj->t',K.conj(),O,K)); m,s=vals.mean(),vals.std()/np.sqrt(NT); y=np.real(np.trace(O@rep))
    print(f"{l:9s} mcsolve={m:.5f}±{s:.5f}  repo={y:.5f}  z={(m-y)/s:+.2f}")
mc=K.T@K.conj()/NT
# expected TD from sampling alone: bootstrap of trajectories vs their own mean
bs=[td((lambda Kb: Kb.T@Kb.conj()/NT)(K[rng.integers(0,NT,NT)]),mc) for _ in range(20)]
print(f"TD(mc,repo)={td(mc,rep):.4f}  bootstrap TD floor={np.mean(bs):.4f}  noise effect TD(ideal,repo)={td(ideal,rep):.4f}")
