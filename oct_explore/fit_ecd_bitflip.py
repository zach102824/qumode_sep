"""Fit shared size-relative ECD+SNAP sequence ≈ Gray X mixer. Then plug into QAOA-style train/test."""
import json, sys, time
sys.path.insert(0, '.')
import numpy as np, jax, jax.numpy as jnp
from scipy.optimize import minimize
from scipy.linalg import expm
import sim

def ungray(g):
    g = np.array(g, copy=True); s = g >> 1
    while np.any(s): g ^= s; s >>= 1
    return g

def gray(i):
    i = np.asarray(i); return i ^ (i >> 1)

def gray_X_sum(k, nf):
    N = 1 << k; M = np.zeros((nf, nf), complex)
    for n in range(N):
        bits = int(gray(n))
        for j in range(k):
            n2 = int(ungray(bits ^ (1 << j))); M[n2, n] += 1
    return M

def disp_mat(nf, alpha):
    a = np.diag(np.sqrt(np.arange(1, nf)), 1)
    return expm(alpha * a.conj().T - np.conj(alpha) * a)

def ecd_mat(nf, beta):
    # on (transmon, cavity): |1><0| D(b/2) + |0><1| D(-b/2)
    dp, dm = disp_mat(nf, beta/2), disp_mat(nf, -beta/2)
    sm = np.array([[0,0],[1,0]]); sp = np.array([[0,1],[0,0]])
    return np.kron(sm, dp) + np.kron(sp, dm)

def rot(th, ph):
    c, s = np.cos(th/2), np.sin(th/2)
    return np.array([[c, -1j*s*np.exp(-1j*ph)], [-1j*s*np.exp(1j*ph), c]])

def snap_diag(nf, coeffs, scale):
    u = np.arange(nf) / scale
    ph = sum(c * u**(j+1) for j, c in enumerate(coeffs))
    return np.diag(np.exp(-1j * ph))

# params per sublayer: bre, bim, th, ph, s1..sP  (P=snap)
# Full seq has S sublayers; also overall mixer angle is external (like beta in QAOA)

def seq_unitary(x, k, nf, S, P, angle=np.pi):
    """Build (2*nf)×(2*nf) unitary on (tm, cav). Target: I_tm ⊗ exp(-i angle/2 Xgray) on logical."""
    scale = np.sqrt(2.0 ** k); x = x.reshape(S, 4+P)
    U = np.eye(2*nf, dtype=complex)
    for s in range(S):
        bre,bim,th,ph = x[s,:4]; coeffs = x[s,4:]
        beta = (bre+1j*bim) * scale
        # R on tm
        R = np.kron(rot(th, ph), np.eye(nf))
        E = ecd_mat(nf, beta)
        Sn = np.kron(np.eye(2), snap_diag(nf, coeffs, scale))
        U = Sn @ E @ R @ U
    return U

def target(k, nf, angle):
    X = gray_X_sum(k, nf)
    Uc = expm(-1j * (angle/2) * X)
    return np.kron(np.eye(2), Uc)  # idle transmon

def fid(U, T, k):
    # fidelity on logical subspace: tm any, cav < 2^k
    N = 1 << k; idx = [i for t in range(2) for n in range(N) for i in [t*nf_of + n]]
    # simpler: Frobenius on full then weight logical
    return np.abs(np.vdot(T, U)) / np.linalg.norm(T) / np.linalg.norm(U)

# Fit S=4, P=3 jointly for k in (1,2,3), angle=pi
S, P = 4, 3; ks = [1, 2, 3]
nfs = {k: (1<<k) + max(8, (1<<k)//4) for k in ks}

def loss(x):
    tot = 0
    for k in ks:
        nf = nfs[k]; U = seq_unitary(x, k, nf, S, P, np.pi); T = target(k, nf, np.pi)
        # process fidelity on logical block (cav < 2^k, both tm)
        N = 1 << k
        Ul = U.reshape(2, nf, 2, nf)[:, :N, :, :N].reshape(2*N, 2*N)
        Tl = T.reshape(2, nf, 2, nf)[:, :N, :, :N].reshape(2*N, 2*N)
        tot += -np.abs(np.vdot(Tl, Ul))**2 / (np.vdot(Tl,Tl).real * np.vdot(Ul,Ul).real + 1e-30)
    return tot / len(ks)

rng = np.random.default_rng(0)
best = None
for trial in range(8):
    x0 = 0.3 * rng.standard_normal(S*(4+P))
    r = minimize(loss, x0, method='L-BFGS-B', options={'maxiter': 400})
    print(f'trial {trial} loss={r.fun:.4f}', flush=True)
    if best is None or r.fun < best[0]: best = (r.fun, r.x)
print('BEST', best[0])
# report fids
for k in ks:
    nf = nfs[k]; U = seq_unitary(best[1], k, nf, S, P, np.pi); T = target(k, nf, np.pi)
    N=1<<k; Ul=U.reshape(2,nf,2,nf)[:,:N,:,:N].reshape(2*N,2*N); Tl=T.reshape(2,nf,2,nf)[:,:N,:,:N].reshape(2*N,2*N)
    f = np.abs(np.vdot(Tl,Ul))**2 / (np.linalg.norm(Tl)**2 * np.linalg.norm(Ul)**2)
    print(f'k={k} logical process fid={f:.4f}', flush=True)
json.dump({'S':S,'P':P,'loss':best[0],'x':best[1].tolist(),'ks':ks}, open('results/ecd_bitflip_fit.json','w'), indent=1)
print('ALLDONE', flush=True)
