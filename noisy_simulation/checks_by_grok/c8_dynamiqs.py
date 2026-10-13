"""(8) part B: one/two layers rebuilt from NOISE.md device numbers alone in Dynamiqs (JAX), no repo imports.
Run with /workspace/venv-dq/bin/python (dynamiqs 0.2.3, jax 0.4.33; separate venv since venv-qumode's pip is broken)."""
import json, numpy as np, jax, jax.numpy as jnp, dynamiqs as dq
from scipy.linalg import expm
jax.config.update("jax_enable_x64", True)
nf=5; T1c,T1,T2,chi,tecd=436e-6,50e-6,30e-6,2*np.pi*1e6,0.65e-6
kap,gam=1/T1c,1/T1; gphi=1/T2-gam/2; tw=np.pi/chi
lays=json.load(open("c8_params.json")); ref=np.load("c8_repo.npz")
I2=np.eye(2); If=np.eye(nf); X=np.array([[0,1],[1,0]],complex); Y=np.array([[0,-1j],[1j,0]]); Z=np.diag([1.,-1.])
sm=np.array([[0,1],[0,0]],complex); a=np.diag(np.sqrt(np.arange(1,nf)),1).astype(complex); n=a.conj().T@a
def kron(*m):
    out=np.array([[1.0+0j]])
    for x in m: out=np.kron(out,x)
    return out
def emb(M,k,dims):
    return kron(*[M if i==k else np.eye(d) for i,d in enumerate(dims)])
D4=[2,2,nf,nf]; D5=[2,2,2,nf,nf]
R=lambda th,ph: expm(-0.5j*th*(np.cos(ph)*X+np.sin(ph)*Y))
Disp=lambda al: expm(al*a.conj().T-np.conj(al)*a)
def ecd(beta): # on (transmon, cavity): |1><0| D(beta/2) + |0><1| D(-beta/2)
    return np.kron(np.array([[0,0],[1,0]]),Disp(beta/2))+np.kron(np.array([[0,1],[0,0]]),Disp(-beta/2))
def U_block(l):
    bd=complex(*l["beta_d"]); be=complex(*l["beta_e"])
    Ur=kron(R(l["theta_d"],l["phi_d"]),R(l["theta_e"],l["phi_e"]),If,If)
    # ECD_dA on modes (0,2), ECD_eB on (1,3): reorder to (d,A,e,B) then permute
    U=np.kron(ecd(bd),ecd(be)).reshape([2,nf,2,nf]*2).transpose(0,2,1,3,4,6,5,7).reshape(4*nf*nf,4*nf*nf)
    return U@Ur
sol=dq.solver.Dopri8(rtol=1e-11,atol=1e-13,max_steps=10**7); opt=dq.Options(progress_meter=None)
def me(H,c,rho,T): return np.asarray(dq.mesolve(H,c,rho,jnp.array([0.,T]),solver=sol,options=opt).states[-1])
c4=[np.sqrt(kap)*emb(a,2,D4),np.sqrt(kap)*emb(a,3,D4)]+[f*emb(M,k,D4) for k in (0,1) for f,M in ((np.sqrt(gam),sm),(np.sqrt(gphi/2),Z))]
c5=[np.sqrt(kap)*emb(a,3,D5),np.sqrt(kap)*emb(a,4,D5)]+[f*emb(M,k,D5) for k in (0,1,2) for f,M in ((np.sqrt(gam),sm),(np.sqrt(gphi/2),Z))]
Pe=np.diag([0.,1.]); H5=-chi*(emb(n,3,D5)+emb(n,4,D5))@emb(Pe,0,D5)
ry=lambda t: np.array([[np.cos(t/2),-np.sin(t/2)],[np.sin(t/2),np.cos(t/2)]],complex)
Mid=ry(np.pi/2)@np.diag([1,-1j])@ry(-np.pi/2)
def layer(rho,l,noisy=True):
    U=U_block(l); rho=U@rho@U.conj().T
    if noisy: rho=me(0*rho,c4,rho,tecd)
    g=np.diag([1.,0.]); r=np.kron(g,rho)
    for M in (ry(np.pi/2),Mid):
        Uq=emb(M,0,D5); r=Uq@r@Uq.conj().T
        r=me(H5,c5 if noisy else [],r,tw)
    return np.einsum('iaib->ab',r.reshape(2,4*nf*nf,2,4*nf*nf))
td=lambda A,B: 0.5*np.abs(np.linalg.eigvalsh(A-B)).sum()
rho=np.zeros((4*nf*nf,)*2,complex); rho[0,0]=1; rid=rho.copy()
for i,l in enumerate(lays):
    rho=layer(rho,l); rid=layer(rid,l,False); k=i+1
    print(f"layer {k}: TD(dynamiqs ideal, repo ideal)={td(rid,ref[f'ideal{k}']):.1e}  noise effect={td(rho,rid):.4f}  "
          f"TD(dynamiqs, repo fast)={td(rho,ref[f'fast{k}']):.2e}  TD(dynamiqs, repo direct)={td(rho,ref[f'direct{k}']):.2e}")
