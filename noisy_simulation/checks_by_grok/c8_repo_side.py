"""(8) part A: repo output for the c8 comparison (params saved as plain JSON for the independent side)."""
from common import *
import json
nf=5; L=2; x=np.random.default_rng(8).normal(0,0.5,n_parameters(L)); lays=unpack_params(x,L)
json.dump([{k:([v.real,v.imag] if isinstance(v,complex) else v) for k,v in l.items()} for l in lays],open("c8_params.json","w"))
out={}
for meth in ("fast","direct"):
    r=C.vacuum_rho(nf); 
    for i,l in enumerate(lays): r=C.noisy_layer(r,l,EICKBUSCH.with_(jmax=None),meth); out[f"{meth}{i+1}"]=as_matrix(r)
r=C.vacuum_rho(nf)
for i,l in enumerate(lays): r=C.noisy_layer(r,l,NOISELESS); out[f"ideal{i+1}"]=as_matrix(r)
np.savez("c8_repo.npz",**out)
