"""(7) Noise inside finite pulses vs repo's lumped-after-ideal-gate model."""
from common import *
from pulsemodel import Model
nf=5; L=3; dev=EICKBUSCH.with_(jmax=None)
x=np.random.default_rng(5).normal(0,0.5,n_parameters(L)); lays=unpack_params(x,L)
m=Model(dev,nf)
# 0) consistency: my 'ideal' and 'lumped' must equal the repo's noiseless / noisy layer
r_rep=C.vacuum_rho(nf); r_id=m.vac(); r_lu=m.vac(); r_pu=m.vac(); r_repid=C.vacuum_rho(nf); rows=[]
for i,lay in enumerate(lays):
    r_rep=C.noisy_layer(r_rep,lay,dev,'direct'); r_repid=C.noisy_layer(r_repid,lay,NOISELESS)
    r_id=m.layer(r_id,lay,'ideal'); r_lu=m.layer(r_lu,lay,'lumped'); r_pu=m.layer(r_pu,lay,'pulse')
    R=as_matrix(r_rep); I=as_matrix(r_repid); LU=r_lu.full(); PU=r_pu.full(); ID=r_id.full()
    w,v=np.linalg.eigh(I); p=v[:,-1]; F=lambda M: np.real(p.conj()@M@p)
    rows.append((i+1,td(ID,I),td(LU,R),td(R,I),td(PU,I),td(PU,R),1-F(R),1-F(PU)))
    print(f"layer {i+1}: consistency TD(my ideal,repo ideal)={rows[-1][1]:.1e} TD(my lumped,repo)={rows[-1][2]:.1e} | "
          f"noise effect TD(repo,ideal)={rows[-1][3]:.4f}  TD(pulse,ideal)={rows[-1][4]:.4f}  MODEL ERROR TD(pulse,repo)={rows[-1][5]:.4f} "
          f"({rows[-1][5]/rows[-1][3]*100:.1f}% of noise effect) | 1-F repo={rows[-1][6]:.4f} pulse={rows[-1][7]:.4f}")
np.save("/workspace/noisy_checks/c7_rows.npy",np.array(rows))
