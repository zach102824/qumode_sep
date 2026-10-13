from common import *
nf=4; lay=layer0(nf); rho=C.ideal_block(C.vacuum_rho(nf),lay,nf)
for sc in (1.0,20.0):
    dev=EICKBUSCH.with_(noise_scale=sc,jmax=None); ref=me_jp(rho,dev,nf); Ns=[1,2,4,8,16,32,64]; e=[]
    for N in Ns: e.append(td(as_matrix(C.jp_direct(rho,dev.with_(n_slice=N),keep_q=True)),ref))
    m=np.array(Ns)>=8; slope=np.polyfit(np.log(np.array(Ns)[m]),np.log(e)[m],1)[0]
    print(f"scale={sc}: "+"  ".join(f"N={n}:{x:.2e}" for n,x in zip(Ns,e))+f"  slope(N>=8)={slope:.3f}")
