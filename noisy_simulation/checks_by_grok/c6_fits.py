from common import *
from scipy.optimize import curve_fit
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from noiseless.ecd_kbit import displace_fast
from noisy_simulation.kraus import apply_unitary
dev=EICKBUSCH; nf=16; ts=np.linspace(0,120e-6,41); tc=np.linspace(0,1.5e-3,41)
X=rotation_np(np.pi,0.); H2=rotation_np(np.pi/2,0.)
base=C.vacuum_rho(nf)
def pe(r,ax): m=marg(r,[ax]); return m[1,1].real
res={}; data={}
for ax,nm in ((0,'d'),(1,'e')):
    t1=[pe(C.transmon_idle(apply_unitary(base,X,ax),dev,t),ax) for t in ts]
    ram=[pe(apply_unitary(C.transmon_idle(apply_unitary(base,H2,ax),dev,t),H2,ax),ax) for t in ts]
    # Ramsey after two pi/2 x-pulses: P_e = (1 + c(t))/2 - pop relaxation term; fit with T1 known-free model
    f1=lambda t,A,T,B:A*np.exp(-t/T)+B; p1,_=curve_fit(f1,ts,t1,p0=[1,4e-5,0])
    p2,_=curve_fit(f1,ts,ram,p0=[.5,3e-5,.5]); res[nm]=(p1[1],p2[1]); data[nm]=(t1,ram,p1,p2)
alpha=2.0; D=displace_fast(alpha,nf); rc=apply_unitary(base,D,2); nA=np.diag(np.arange(nf))
nbar=[np.trace(nA@marg(C.cavity_idle(rc,dev.with_(jmax=None),t),[2])).real for t in tc]
p3,_=curve_fit(lambda t,A,T:A*np.exp(-t/T),tc,nbar,p0=[4,4e-4])
# same cavity ring-down through full noisy layers (identity gates + JP + loss, fast path)
ident=dict(beta_d=0j,beta_e=0j,theta_d=np.pi,theta_e=np.pi,phi_d=0.,phi_e=0.); Tl=dev.tau_ecd+dev.tau_jp
r=C.vacuum_rho(10); r=apply_unitary(r,displace_fast(1.0,10),2); r=apply_unitary(r,X,0); r=apply_unitary(r,H2,1)
lay_n,lay_pd,lay_c=[],[],[]
for L in range(0,121):
    if L%4==0:
        lay_n.append(np.trace(np.diag(np.arange(10))@marg(r,[2])).real); lay_pd.append(pe(r,0)); lay_c.append(abs(marg(r,[1])[0,1]))
    r=C.noisy_layer(r,ident,dev)
tl=np.arange(0,121,4)*Tl
q1,_=curve_fit(lambda t,A,T:A*np.exp(-t/T),tl,lay_n,p0=[1,4e-4]); q2,_=curve_fit(lambda t,A,T:A*np.exp(-t/T),tl,lay_pd,p0=[1,5e-5]); q3,_=curve_fit(lambda t,A,T:A*np.exp(-t/T),tl,lay_c,p0=[.5,3e-5])
for nm in 'de': print(f"{nm}: idle fit T1={res[nm][0]*1e6:.4f} us  T2(Ramsey)={res[nm][1]*1e6:.4f} us")
print(f"cavity A ring-down (idle, alpha=2): kappa^-1={p3[1]*1e6:.3f} us, n0={p3[0]:.4f}")
print(f"through {120} full noisy layers (fast path): cavity kappa^-1={q1[1]*1e6:.3f} us, d T1={q2[1]*1e6:.4f} us, e T2 (|rho_ge|)={q3[1]*1e6:.4f} us")
fig,axs=plt.subplots(1,3,figsize=(14,4))
for nm,c in (('d','C0'),('e','C1')):
    t1,ram,p1,p2=data[nm]
    axs[0].plot(ts*1e6,t1,'o',c=c,ms=3,label=f"{nm} sim"); axs[0].plot(ts*1e6,p1[0]*np.exp(-ts/p1[1])+p1[2],'-',c=c,label=f"{nm} fit T1={p1[1]*1e6:.2f}us")
    axs[1].plot(ts*1e6,ram,'o',c=c,ms=3,label=f"{nm} sim"); axs[1].plot(ts*1e6,p2[0]*np.exp(-ts/p2[1])+p2[2],'-',c=c,label=f"{nm} fit T2={p2[1]*1e6:.2f}us")
axs[0].plot(tl*1e6,lay_pd,'kx',label=f"d via noisy layers, T1={q2[1]*1e6:.2f}us")
axs[1].plot(tl*1e6,np.array(lay_c)+0.5,'kx',label=f"e |rho_ge|+0.5 via layers, T2={q3[1]*1e6:.2f}us")
axs[2].plot(tc*1e6,nbar,'o',ms=3,label="idle, alpha=2"); axs[2].plot(tc*1e6,p3[0]*np.exp(-tc/p3[1]),'-',label=f"fit 1/kappa={p3[1]*1e6:.1f}us")
axs[2].plot(tl*1e6,np.array(lay_n),'kx',label=f"via noisy layers, alpha=1: {q1[1]*1e6:.1f}us")
for a_,t in zip(axs,["T1 (pi, wait)","Ramsey (pi/2, wait, pi/2), P_e","cavity ring-down <n>"]): a_.set_title(t); a_.set_xlabel("t (us)"); a_.legend(fontsize=7)
plt.tight_layout(); plt.savefig("/workspace/noisy_checks/fits.png",dpi=130)
