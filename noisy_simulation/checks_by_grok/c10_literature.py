"""(10) Eickbusch et al., Nat. Phys. 18, 1464 (2022) [arXiv:2111.06414], Table I: Fock |1> prep with ECD,
F_exp = 0.98 +- 0.01, F_sim = 0.99 (all measured decoherence), avg P(g) = 0.96 (Fock states).
We optimise an ECD circuit |0>|g> -> |1>|g> with the repo's own gate code, then run it through the repo's own
lumped Kraus noise (transmon_idle / cavity_idle / apply_conditional) for one (transmon, cavity) pair."""
from common import *
from scipy.optimize import minimize
from noiseless.ecd_kbit import displace_fast
from noisy_simulation.kraus import apply_conditional, apply_unitary
from noisy_simulation.circuit import ecd_rotation_parts, transmon_idle, cavity_idle
nf=14; rng=np.random.default_rng(0)
def ecd(beta): D=displace_fast(beta/2,nf); return np.kron([[0,0],[1,0]],D)+np.kron([[0,1],[0,0]],D.conj().T)
def unpack(p,N): return [(complex(p[4*i],p[4*i+1]),p[4*i+2],p[4*i+3]) for i in range(N)],(p[4*N],p[4*N+1])
def ket_F(p,N):
    s=np.zeros(2*nf,complex); s[0]=1
    steps,(tf,pf)=unpack(p,N)
    for b,t,ph in steps: s=ecd(b)@(np.kron(rotation_np(t,ph),np.eye(nf))@s)
    s=np.kron(rotation_np(tf,pf),np.eye(nf))@s; return abs(s[1])**2   # |g>|1>
best={}
for N in (3,4,5):
    res=min((minimize(lambda p:1-ket_F(p,N),rng.normal(0,1,4*N+2),method="BFGS") for _ in range(60)),key=lambda r:r.fun)
    best[N]=res.x; print(f"N={N}: ideal unitary infidelity {res.fun:.2e}")
N=min(n for n in best if 1-ket_F(best[n],n)<0.01); p=best[N]
def noisy_F(dev,T_total,nth_c0=0.0):
    steps,(tf,pf)=unpack(p,N); tau=T_total/N
    cav=np.zeros((nf,nf)); 
    if nth_c0>0:
        k=np.arange(nf); w=(nth_c0/(1+nth_c0))**k; cav=np.diag(w/w.sum())
    else: cav[0,0]=1
    rho=np.einsum('ab,cd->acbd',np.diag([1.,0]),cav).astype(complex)   # (q, n, q', n')
    for b,t,ph in steps:
        rho=apply_conditional(rho,*ecd_rotation_parts(b,t,ph,nf),0,1)
        rho=cavity_idle(transmon_idle(rho,dev,tau,axes=(0,)),dev,tau,axes=(1,))
    rho=apply_unitary(rho,rotation_np(tf,pf),0)
    Pg=np.real(np.trace(rho[0,:,0,:])); return np.real(rho[0,1,0,1])/Pg, Pg
print(f"using N={N} ECD steps; ideal F={ket_F(p,N):.4f}")
cases=[("repo default device (T1=50us,T2=30us,T1c=436us)",EICKBUSCH.with_(jmax=None),0.0),
       ("Eickbusch error-budget values (T1~30us with nbar=900, gphi=g2E-g1/2~0 -> T2=2T1)",EICKBUSCH.with_(jmax=None,t1_q=30e-6,t2_q=60e-6),0.0),
       ("  + initial cavity thermal nth=0.025",EICKBUSCH.with_(jmax=None,t1_q=30e-6,t2_q=60e-6),0.025)]
for T in (N*224e-9,1.0e-6):
    for lbl,dev,nth in cases:
        F,Pg=noisy_F(dev,T,nth); print(f" T_total={T*1e6:.2f}us {lbl}: F={F:.4f}  P(g)={Pg:.4f}")
