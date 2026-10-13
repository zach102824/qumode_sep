from common import *
nf=8; ident=dict(beta_d=0j,beta_e=0j,theta_d=np.pi,theta_e=np.pi,phi_d=0.,phi_e=0.)  # ECD(0)R_x(pi) = -i I
r=C.noisy_layer(C.vacuum_rho(nf),layer0(nf,sc=1.0),NOISELESS)  # excited, entangled start
dev=EICKBUSCH; Tl=dev.tau_ecd+dev.tau_jp; vac=np.zeros(4*nf*nf); vac[0]=1
print(f"start: P(vac,g,g)={np.real(as_matrix(r)[0,0]):.4f}")
for L in range(1,4001):
    r=C.noisy_layer(r,ident,dev)
    if L in (10,100,500,1000,2000,4000):
        m=as_matrix(r); ev=np.linalg.eigvalsh((m+m.conj().T)/2)
        print(f" {L:5d} layers ({L*Tl*1e6:7.1f} us): 1-P(vac,g,g)={1-m[0,0].real:.3e} tr-1={np.trace(m).real-1:+.1e} mineig={ev.min():+.1e}")
# exact long idle via direct channels
r2=C.noisy_layer(C.vacuum_rho(nf),layer0(nf,sc=1.0),NOISELESS)
r2=C.transmon_idle(C.cavity_idle(r2,dev.with_(jmax=None),20*dev.t1_cav),dev,20*dev.t1_cav); m=as_matrix(r2)
print(f" long idle 20*T1cav: 1-P(vac,g,g)={1-m[0,0].real:.1e} tr-1={np.trace(m).real-1:+.1e} mineig={np.linalg.eigvalsh(m).min():+.1e}")
