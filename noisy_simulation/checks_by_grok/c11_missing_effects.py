"""(11) Thermal population and |f> leakage on top of the c7 finite-pulse model.
Transmon qutrit: anharmonicity -2pi x 193 MHz (Eickbusch Table S1), |f> decays via b (rate 2*gamma, sqrt(2) matrix element),
dephasing 2*gamma_phi D[n_q]; rotations are 24 ns sin^2 pulses WITHOUT DRAG; CD amplitude for |f> = -3 x (g:+1, e:-1).
Thermal: gamma*nth D[b^dag] + gamma*(1+nth) D[b]; cavity kappa*nth_c D[a^dag]."""
from common import *
from pulsemodel import Model
import json
nf=4; L=2; dev=EICKBUSCH.with_(jmax=None)
x=np.random.default_rng(11).normal(0,0.5,n_parameters(L)); lays=unpack_params(x,L)
cfgs={"base (qubit, nth=0)":dict(dq=2),"thermal nth_q=0.0092 nth_c=1e-3":dict(dq=2,nth_q=0.0092,nth_c=1e-3),
      "thermal nth_q=0.05 nth_c=1e-3":dict(dq=2,nth_q=0.05,nth_c=1e-3),"qutrit leakage":dict(dq=3),
      "qutrit + nth_q=0.05 + nth_c=1e-3":dict(dq=3,nth_q=0.05,nth_c=1e-3)}
mi=Model(dev,nf); rid=mi.vac(); ideal=[]
for l in lays: rid=mi.layer(rid,l,'ideal'); ideal.append(rid.full())
def to_qubit(r,dq):   # project qutrit (d,e) onto qubit subspace, keep unnormalised
    if dq==2: return r, 0.0
    R=r.reshape([3,3,nf,nf]*2)[:2,:2,:,:,:2,:2].reshape(4*nf*nf,4*nf*nf); return R, 1-np.trace(R).real
import os, sys
out=json.load(open('c11_results.json')) if os.path.exists('c11_results.json') else {}
for f in sorted(os.listdir('.')):
    if f.startswith('c11_part_'): out.update(json.load(open(f)))
only=sys.argv[1] if len(sys.argv)>1 else None
for name,kw in cfgs.items():
    if name in out or (only and name!=only): continue
    m=Model(dev,nf,**kw); r=m.vac(); res=[]
    for i,l in enumerate(lays):
        r=m.jp_fast(m.block(r,l,'pulse'));   # jp_fast == jp to 1.8e-7 (checked)
        R,leak=to_qubit(r.full(),kw['dq']); I=ideal[i]
        w,v=np.linalg.eigh(I); p=v[:,-1]; F=np.real(p.conj()@R@p)
        tvd=0.5*(np.abs(np.real(np.diag(R))-np.real(np.diag(I))).sum()+leak)
        res.append(dict(layer=i+1,infid=1-F,tvd=tvd,leak=leak)); 
    out[name]=res; print(name, "  ".join(f"L{x['layer']}: 1-F={x['infid']:.5f} TVD(probs)={x['tvd']:.5f} P(f)={x['leak']:.1e}" for x in res), flush=True)
    json.dump({name:res},open(f"c11_part_{abs(hash(name))}.json","w"))
if only: sys.exit()
b=out["base (qubit, nth=0)"]
for name,res in out.items():
    print(f"{name:34s} change vs base after {L} layers: d(1-F)={res[-1]['infid']-b[-1]['infid']:+.5f} ({(res[-1]['infid']/b[-1]['infid']-1)*100:+.1f}%), dTVD={res[-1]['tvd']-b[-1]['tvd']:+.5f}")
json.dump(out,open("c11_results.json","w"),indent=1)
