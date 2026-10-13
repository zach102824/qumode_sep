from common import *
nf=6; lay=layer0(nf,sc=0.8); r0=C.vacuum_rho(nf); T=EICKBUSCH.tau_ecd+EICKBUSCH.tau_jp
inf=np.inf
cases={'kappa only':dict(t1_q=inf,t2_q=inf),'gamma only':dict(t1_cav=inf,t2_q=2*50e-6),'gphi only':dict(t1_cav=inf,t1_q=inf)}
ideal=C.noisy_layer(r0,lay,NOISELESS)
def qchan(m,gt,gpt):  # expected single-qubit channel on a 2x2 marginal
    p=1-np.exp(-gt); out=m.copy().astype(complex); out[0,0]=m[0,0]+p*m[1,1]; out[1,1]=(1-p)*m[1,1]
    out[0,1]*=np.exp(-gt/2-gpt); out[1,0]*=np.exp(-gt/2-gpt); return out
nA=np.kron(np.diag(np.arange(nf)),np.eye(nf)); nB=np.kron(np.eye(nf),np.diag(np.arange(nf)))
for name,kw in cases.items():
    dev=EICKBUSCH.with_(**kw); print(f"{name}: kappa={dev.kappa:.4g} gamma={dev.gamma:.4g} gphi={dev.gamma_phi:.4g}")
    for meth in ('fast','direct'):
        out=C.noisy_layer(r0,lay,dev.with_(jmax=None),meth)
        de_i,de_o=marg(ideal,[0,1]),marg(out,[0,1]); ab_i,ab_o=marg(ideal,[2,3]),marg(out,[2,3])
        # expected d,e marginal: product qubit channel over T on each of d,e (JP never touches d,e)
        exp_de=de_i.reshape(2,2,2,2)
        kr=lambda m: m
        # apply channel via repo-independent formula on each axis
        M=de_i.reshape(2,2,2,2).copy()
        def ch_axis(M,ax):
            p=1-np.exp(-dev.gamma*T); f=np.exp(-dev.gamma*T/2-dev.gamma_phi*T); R=M.copy()
            sl=lambda a,b: tuple([a if i==ax else (b if i==ax+2 else slice(None)) for i in range(4)])
            R[sl(0,0)]=M[sl(0,0)]+p*M[sl(1,1)]; R[sl(1,1)]=(1-p)*M[sl(1,1)]; R[sl(0,1)]*=f; R[sl(1,0)]*=f; return R
        exp_de=ch_axis(ch_axis(M,0),1).reshape(4,4)
        nAi,nAo=np.trace(nA@ab_i).real,np.trace(nA@ab_o).real; nBi,nBo=np.trace(nB@ab_i).real,np.trace(nB@ab_o).real
        print(f"  {meth}: TD(de) noisy-vs-ideal={td(de_o,de_i):.2e}, |de - analytic channel|max={np.abs(de_o-exp_de).max():.1e}; "
              f"TD(AB)={td(ab_o,ab_i):.2e}, max|dFockpop(AB)|={np.abs(np.diag(ab_o-ab_i)).max():.1e}; "
              f"<nA> {nAi:.4f}->{nAo:.4f}, <nB> {nBi:.4f}->{nBo:.4f}, ratios {nAo/nAi:.6f},{nBo/nBi:.6f} vs e^-kT={np.exp(-dev.kappa*T):.6f}")
