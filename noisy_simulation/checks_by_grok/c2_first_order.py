from common import *
from scipy.integrate import simpson
nf=5; lay=layer0(nf); r4=C.ideal_block(C.vacuum_rho(nf),lay,nf)
w,v=np.linalg.eigh(as_matrix(r4)); psi4=v[:,-1]
dev=EICKBUSCH.with_(jmax=None,n_slice=64)
# first-order prediction at scale 1: sum_L rate * int (<L^dag L> - |<L>|^2) dt along ideal path
a=qt.destroy(nf)
def ops(nm):  # nm modes list types
    D=[2 if t=='q' else nf for t in nm]; out={}
    for k,t in enumerate(nm):
        emb=lambda M:qt.tensor(*[M if i==k else qt.qeye(D[i]) for i in range(len(nm))]).full()
        if t=='q': out[(k,'gamma')]=emb(sm); out[(k,'gphi')]=emb(sZ)
        else: out[(k,'kappa')]=emb(a)
    return out
rate={'gamma':dev.gamma,'gphi':dev.gamma_phi/2,'kappa':dev.kappa}
def var(psi,L): return np.real(psi.conj()@(L.conj().T@L)@psi-abs(psi.conj()@L@psi)**2)
contrib={}
O4=ops(['q','q','c','c'])
for key,L in O4.items(): contrib[('idle',)+key]=rate[key[1]]*dev.tau_ecd*var(psi4,L)
O5=ops(['q','q','q','c','c']); g=np.zeros(2); g[0]=1; psi=np.kron(g,psi4)
Hd=C.wait_hamiltonian(dev,nf,with_de=True).ravel()
U=lambda M,p: np.kron(M,np.eye(len(p)//2))@p
psi=U(C.RY_P,psi); ts=np.linspace(0,dev.tau_wait,401)
for wi,M in enumerate((None,C.JP_MID)):
    if M is not None: psi=U(M,psi)
    traj=[np.exp(-1j*Hd*t)*psi for t in ts]
    for key,L in O5.items(): contrib[('wait%d'%wi,)+key]=rate[key[1]]*simpson([var(p,L) for p in traj],x=ts)
    psi=traj[-1]
psi=U(C.RY_M,psi)
pred=sum(contrib.values())
names={0:'q',1:'d',2:'e',3:'A',4:'B'}
print("first-order contributions (scale=1):")
for k,vv in contrib.items():
    lbl=k[0]+':'+(['d','e','A','B'][k[1]] if k[0]=='idle' else names[k[1]])+':'+k[2]
    if vv>1e-9: print(f"  {lbl:16s}{vv:.4e}")
print(f"predicted 1-F per unit noise_scale = {pred:.6e}")
for s in (1e-4,1e-3,1e-2,1e-1):
    d=dev.with_(noise_scale=s); r=C.transmon_idle(C.cavity_idle(r4,d,d.tau_ecd),d,d.tau_ecd)
    rho=as_matrix(C.jp_direct(r,d,keep_q=True)); F=np.real(psi.conj()@rho@psi)
    print(f" scale={s:g}: 1-F={1-F:.6e}  (1-F)/scale={(1-F)/s:.6e}  ratio to prediction={(1-F)/s/pred:.5f}")
