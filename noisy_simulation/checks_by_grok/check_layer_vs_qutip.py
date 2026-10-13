"""One full noisy layer: repo (fast & direct) vs independent QuTiP mesolve of the same Lindbladian.
Unitary ECD block reuses C.ideal_block (already checked equal to the noiseless code at zero noise)."""
import sys, numpy as np, qutip as qt
sys.path.insert(0, '/workspace/qumode_sep')
from noisy_simulation import circuit as C
from noisy_simulation.kraus import as_matrix
from noisy_simulation.device import EICKBUSCH
from noiseless.circuit_local_ecd import unpack_params, n_parameters
nf=5; rng=np.random.default_rng(7)
def td(a,b): return 0.5*np.abs(np.linalg.eigvalsh(a-b)).sum()
def run(dev, lay, rho0):
    rho = C.ideal_block(rho0, lay, nf)
    I2, If = qt.qeye(2), qt.qeye(nf); sm = qt.destroy(2); sz = qt.sigmaz()
    # NOTE repo convention: index0=|g>, index1=|e>; qt.destroy(2) lowers 1->0, so sm is sigma_-; P_e=|1><1|
    Pe = qt.basis(2,1).proj(); a = qt.destroy(nf); sZ = qt.Qobj(np.diag([-1.,1.]))  # sign irrelevant in D[]
    # idle on d,e,A,B for tau_ecd
    ops4 = lambda M,k: qt.tensor(*[M if i==k else (I2 if i<2 else If) for i in range(4)])
    c4 = [np.sqrt(dev.kappa)*ops4(a,2), np.sqrt(dev.kappa)*ops4(a,3)]
    for k in (0,1): c4 += [np.sqrt(dev.gamma)*ops4(sm,k), np.sqrt(dev.gamma_phi/2)*ops4(sZ,k)]
    r = qt.Qobj(as_matrix(rho), dims=[[2,2,nf,nf]]*2)
    r = qt.mesolve(0*ops4(I2,0), r, [0,dev.tau_ecd], c4, options={"atol":1e-12,"rtol":1e-10}).states[-1]
    # JP: add q in |g>
    ops5 = lambda M,k: qt.tensor(*[M if i==k else (I2 if i<3 else If) for i in range(5)])
    r = qt.tensor(qt.basis(2,0).proj(), r)
    H = -dev.chi_qa*(ops5(a.dag()*a,3)+ops5(a.dag()*a,4))*ops5(Pe,0)
    c5 = [np.sqrt(dev.kappa)*ops5(a,3), np.sqrt(dev.kappa)*ops5(a,4)]
    for k in (0,1,2): c5 += [np.sqrt(dev.gamma)*ops5(sm,k), np.sqrt(dev.gamma_phi/2)*ops5(sZ,k)]
    U = lambda M: ops5(qt.Qobj(M),0)
    r = U(C.RY_P)*r*U(C.RY_P).dag()
    r = qt.mesolve(H, r, [0,dev.tau_wait], c5, options={"atol":1e-12,"rtol":1e-10,"nsteps":100000}).states[-1]
    r = U(C.JP_MID)*r*U(C.JP_MID).dag()
    r = qt.mesolve(H, r, [0,dev.tau_wait], c5, options={"atol":1e-12,"rtol":1e-10,"nsteps":100000}).states[-1]
    return r.ptrace([1,2,3,4]).full()
x = rng.normal(0,0.6,n_parameters(1)); lay = unpack_params(x,1)[0]
rho0 = C.vacuum_rho(nf)
for s in (1.0, 10.0):
    dev = EICKBUSCH.with_(noise_scale=s, jmax=None)
    me = run(dev, lay, rho0)
    ideal = as_matrix(C.noisy_layer(rho0, lay, dev.with_(noise_scale=0)))
    f = as_matrix(C.noisy_layer(rho0, lay, dev, "fast")); d = as_matrix(C.noisy_layer(rho0, lay, dev, "direct"))
    print(f"scale={s}: noise effect TD(ideal,ME)={td(ideal,me):.3e}  TD(fast,ME)={td(f,me):.3e}  TD(direct,ME)={td(d,me):.3e}")
