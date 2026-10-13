"""(11b) quick qutrit-leakage run: nf=3, 1 layer, pulse model; qubit vs qutrit (same params as c11)."""
from common import *
from pulsemodel import Model
nf=3; dev=EICKBUSCH.with_(jmax=None); lay=unpack_params(np.random.default_rng(11).normal(0,0.5,n_parameters(2)),2)[0]
I=Model(dev,nf).layer(Model(dev,nf).vac(),lay,'ideal').full(); w,v=np.linalg.eigh(I); p=v[:,-1]
for name,kw in (("qubit",dict(dq=2)),("qutrit",dict(dq=3))):
    m=Model(dev,nf,**kw); r=m.jp_fast(m.block(m.vac(),lay,'pulse')).full(); leak=0.0
    if kw['dq']==3:
        R=r.reshape([3,3,nf,nf]*2)[:2,:2,:,:,:2,:2].reshape(4*nf*nf,4*nf*nf); leak=1-np.trace(R).real; r=R
    tvd=0.5*(np.abs(np.real(np.diag(r))-np.real(np.diag(I))).sum()+leak)
    print(f"{name}: 1-F={1-np.real(p.conj()@r@p):.5f} TVD={tvd:.5f} P(f)={leak:.2e}",flush=True)
