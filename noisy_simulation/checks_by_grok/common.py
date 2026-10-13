import sys, numpy as np, qutip as qt
sys.path.insert(0, '/workspace/qumode_sep')
from noisy_simulation import circuit as C
from noisy_simulation.kraus import as_matrix, trace_out
from noisy_simulation.device import EICKBUSCH, NOISELESS
from noiseless.circuit_local_ecd import unpack_params, n_parameters, rotation_np
I2=qt.qeye(2); sm=qt.destroy(2); sZ=qt.Qobj(np.diag([1.,-1.])); Pe=qt.basis(2,1).proj()
OPT={"atol":1e-13,"rtol":1e-11,"nsteps":10**6}
def td(a,b): return 0.5*np.abs(np.linalg.eigvalsh(a-b)).sum()
def op5(M,k,nf): return qt.tensor(*[M if i==k else (I2 if i<3 else qt.qeye(nf)) for i in range(5)])
def lind5(dev,nf):
    a=qt.destroy(nf)
    H=-dev.chi_qa*(op5(a.dag()*a,3,nf)+op5(a.dag()*a,4,nf))*op5(Pe,0,nf)
    c=[np.sqrt(dev.kappa)*op5(a,3,nf),np.sqrt(dev.kappa)*op5(a,4,nf)]
    for k in (0,1,2): c+=[np.sqrt(dev.gamma)*op5(sm,k,nf),np.sqrt(dev.gamma_phi/2)*op5(sZ,k,nf)]
    return H,[x for x in c if x.norm()>0]
def me_jp(rho4,dev,nf):
    """reference JP (q kept, closing Ry(-pi/2) applied) via mesolve"""
    H,c=lind5(dev,nf); U=lambda M: op5(qt.Qobj(M),0,nf)
    r=qt.tensor(qt.basis(2,0).proj(),qt.Qobj(as_matrix(rho4),dims=[[2,2,nf,nf]]*2))
    for M in (C.RY_P,C.JP_MID):
        r=U(M)*r*U(M).dag(); r=qt.mesolve(H,r,[0,dev.tau_wait],c,options=OPT).states[-1]
    r=U(C.RY_M)*r*U(C.RY_M).dag(); return r.full()
def layer0(nf,seed=3,sc=0.6):
    x=np.random.default_rng(seed).normal(0,sc,n_parameters(1)); return unpack_params(x,1)[0]
def marg(rho,keep):
    N=rho.ndim//2; r=rho
    for k in sorted(set(range(N))-set(keep),reverse=True): r=trace_out(r,k)
    return as_matrix(r)
