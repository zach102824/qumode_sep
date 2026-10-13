"""Independent sanity checks of noisy_simulation (run from repo root with the venv python)."""
import sys, numpy as np, qutip as qt
from scipy.linalg import expm
sys.path.insert(0, '/workspace/qumode_sep')
from noisy_simulation import circuit as C
from noisy_simulation.kraus import (cavity_loss_kraus, qubit_kraus, as_matrix)
from noisy_simulation.device import EICKBUSCH, NOISELESS
from noiseless.circuit_local_ecd import n_parameters
rng = np.random.default_rng(1)
def mineig(r): m=as_matrix(r); return np.linalg.eigvalsh((m+m.conj().T)/2).min()
def tr(r): return np.trace(as_matrix(r)).real
def td(a,b): return 0.5*np.abs(np.linalg.eigvalsh(a-b)).sum()

print("== 1. Kraus completeness  ||sum K^dag K - I||")
for kt in (1e-3, 0.1, 1.0):
    for jm in (None, 2):
        K = cavity_loss_kraus(kt, 12, jm); print(f" cavity kt={kt} jmax={jm}: {np.abs(sum(k.conj().T@k for k in K)-np.eye(12)).max():.1e}")
K = qubit_kraus(0.3, 0.2); print(f" qubit: {np.abs(sum(k.conj().T@k for k in K)-np.eye(2)).max():.1e}")

print("== 2. analytic: Fock |5>, coherent alpha=1.5 under loss, <n>/(n0 e^-kt)")
nf=30
for kt in (0.05, 0.5, 2.0):
    for name, psi in (("fock5", qt.basis(nf,5)), ("coh1.5", qt.coherent(nf,1.5))):
        r = psi.proj().full(); n0=np.real(np.trace(np.diag(np.arange(nf))@r))
        r2 = sum(k@r@k.conj().T for k in cavity_loss_kraus(kt, nf, None))
        n1 = np.real(np.trace(np.diag(np.arange(nf))@r2)); pur=np.real(np.trace(r2@r2))
        print(f" kt={kt} {name}: ratio={n1/(n0*np.exp(-kt)):.12f} purity={pur:.6f}")
print(" qubit: start |+e>, gt=0.4, gpt=0.3")
psi=np.array([1,1])/np.sqrt(2); r=np.outer(psi,psi); r2=sum(k@r@k.conj().T for k in qubit_kraus(0.4,0.3))
print(f"  P_e={r2[1,1].real:.6f} expect {0.5*np.exp(-0.4):.6f};  |rho_ge|={abs(r2[0,1]):.6f} expect {0.5*np.exp(-0.2-0.3):.6f}")

L=3; nf=10; npar=n_parameters(L)
x = rng.normal(0,0.8,npar)
print("== 3. CPTP along the circuit (trace, min eig, hermiticity) per layer, nf=10, L=3")
from noiseless.circuit_local_ecd import unpack_params
for s in (1.0, 30.0):
    for meth in ("fast","direct"):
        dev=EICKBUSCH.with_(noise_scale=s); r=C.vacuum_rho(nf); out=[]
        for lay in unpack_params(x,L):
            r=C.noisy_layer(r,lay,dev,meth); m=as_matrix(r)
            out.append(f"tr-1={tr(r)-1:+.1e} mineig={mineig(r):+.1e} herm={np.abs(m-m.conj().T).max():.0e}")
        print(f" scale={s} {meth}: "+" | ".join(out))

print("== 4. zero noise == ideal, and fidelity vs noise_scale (monotone?)")
ideal = as_matrix(C.apply_noisy_circuit(x,L,nf,NOISELESS))
w,v=np.linalg.eigh(ideal); psi=v[:,-1]; print(f" ideal purity={np.real(np.trace(ideal@ideal)):.12f}")
for s in (0, 0.01, 0.1, 0.3, 1, 3, 10, 30, 100):
    f=[]; 
    for meth in ("fast","direct"):
        r=as_matrix(C.apply_noisy_circuit(x,L,nf,EICKBUSCH.with_(noise_scale=s),meth)); f.append(np.real(psi.conj()@r@psi))
    print(f" scale={s:>6}: F_fast={f[0]:.8f} F_direct={f[1]:.8f}")
