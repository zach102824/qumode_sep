"""Extra validation of the fast 3cav circuit (see results/ECD_SPEED.md)."""
import json, sys
import numpy as np
import mpmath as mp
from scipy.linalg import expm
import scipy.sparse as sp
from noiseless.ecd_bench import _ecd_mm, apply_circuit_multi_fast, apply_circuit_multi_fast2
from noiseless.ecd_kbit import _destroy, displace_fast, born
from noiseless.relayout_3cav import MultiCavLayout, apply_circuit_multi, make_param_helpers, jp_phase
from noiseless.circuit_local_ecd import rotation_np

OUT = "noiseless/results/ecd_speed/validate.json"
R = {}
def save(): json.dump(R, open(OUT, "w"), indent=1)

# (1) small copy: nf 12/12/8, dense operators from scipy expm
nfs = (12, 12, 8); lay = MultiCavLayout((3, 3, 3), nfs); dims = lay.dims; N = int(np.prod(dims))
rng = np.random.default_rng(7)
betas = rng.normal(size=3) * 1.5 + 1j * rng.normal(size=3) * 1.5
SM = np.array([[0, 0], [1, 0]], complex)  # |1><0|
SP = np.array([[0, 1], [0, 0]], complex)  # |0><1|
def emb(op, pos, sizes):
    out = sp.csr_matrix(np.ones((1, 1), complex))
    for i, s in enumerate(sizes):
        out = sp.kron(out, sp.csr_matrix(op[i]) if i in pos else sp.identity(s, complex, format="csr"), format="csr")
    return out
def ecd_factors(betas):
    fs = []
    for t in range(3):
        a = _destroy(nfs[t]); al = betas[t] / 2
        G = al * a.conj().T - np.conj(al) * a
        Dp, Dm = expm(G), expm(-G)
        ops1 = {t: SM, 3 + t: Dp}; ops2 = {t: SP, 3 + t: Dm}
        sizes = list(dims)
        fs.append((emb(ops1, ops1.keys(), sizes) + emb(ops2, ops2.keys(), sizes)).tocsr())
    return fs
def full_ecd_dense(betas):
    U = np.eye(N, dtype=complex)
    for f in ecd_factors(betas):
        U = f @ U
    return U
Ud = full_ecd_dense(betas)
# fast method on every basis vector
cols = []
B1 = np.empty(dims, complex); B2 = np.empty(dims, complex)
Ds = [displace_fast(complex(b) / 2, nf) for b, nf in zip(betas, nfs)]
for j in range(N):
    B1[...] = 0; B1.reshape(-1)[j] = 1
    r, _ = _ecd_mm(B1.copy(), B2, Ds)
    cols.append(r.reshape(-1).copy())
Uf = np.array(cols).T
R["1_small_one_layer_ECD_maxabs"] = float(np.abs(Uf - Ud).max())
R["1_small_ECD_unitarity_maxabs"] = float(np.abs(Ud.conj().T @ Ud - np.eye(N)).max()); del Uf, Ud; save(); print(R, flush=True)
# full L=4 circuit as dense matrix
h = make_param_helpers(3); L = 4; x = h["random_parameters"](L, rng)
ph = np.tile(jp_phase(lay).reshape(-1), 8)
Ufull = np.eye(N, dtype=complex)
for bet, th, phs in h["unpack"](x, L):
    Rt = np.kron(np.kron(rotation_np(th[0], phs[0]), rotation_np(th[1], phs[1])), rotation_np(th[2], phs[2]))
    Ufull = sp.kron(sp.csr_matrix(Rt), sp.identity(N // 8, complex), format="csr") @ Ufull
    for fct in ecd_factors(bet):
        Ufull = fct @ Ufull
    Ufull *= ph[:, None]
ref = Ufull[:, 0]
R["1_small_L4_state_old_vs_dense"] = float(np.abs(apply_circuit_multi(x, L, lay) - ref).max())
R["1_small_L4_state_fast_vs_dense"] = float(np.abs(apply_circuit_multi_fast(x, L, lay) - ref).max())
R["1_small_L4_probs_fast2_vs_dense"] = float(np.abs(born(apply_circuit_multi_fast2(x, L, lay)) - np.abs(ref) ** 2).max())
# fast circuit operator on every basis vector (generalised start state): use fast path with arbitrary input
def fast_op_on(v, x):
    Aa = v.reshape(dims).copy(); Bb = np.empty(dims, complex); phd = jp_phase(lay)
    for bet, th, phs in h["unpack"](x, L):
        R8 = np.kron(np.kron(rotation_np(th[0], phs[0]), rotation_np(th[1], phs[1])), rotation_np(th[2], phs[2]))
        np.matmul(R8, Aa.reshape(8, -1), out=Bb.reshape(8, -1))
        Aa, Bb = _ecd_mm(Bb, Aa, [displace_fast(complex(b) / 2, nf) for b, nf in zip(bet, nfs)])
        Aa = Aa * phd
    return Aa.reshape(-1)
def unit(j):
    e = np.zeros(N, complex); e[j] = 1; return e
R["1_small_L4_full_operator_fast_vs_dense"] = max(float(np.abs(fast_op_on(unit(j), x) - Ufull[:, j]).max()) for j in range(N))
R["1_small_N"] = N
save(); print(R, flush=True)

# (2) nf=80 D vs mpmath Laguerre (infinite-space D) at 40 digits
mp.mp.dps = 40
def lag_elem(m, n, al):
    al = mp.mpc(al); x = abs(al) ** 2
    if m >= n:
        return mp.sqrt(mp.factorial(n) / mp.factorial(m)) * al ** (m - n) * mp.exp(-x / 2) * mp.laguerre(n, m - n, x)
    return mp.sqrt(mp.factorial(m) / mp.factorial(n)) * (-mp.conj(al)) ** (n - m) * mp.exp(-x / 2) * mp.laguerre(m, n - m, x)
nf = 80; rows = []
for r in (0.25, 0.5, 1.0, 1.5, 2.0, 2.5):
    al = r * np.exp(0.7j)
    Df = displace_fast(al, nf)
    M = np.array([[complex(lag_elem(m, n, al)) for n in range(nf)] for m in range(nf)])
    err = np.abs(Df - M)
    rows.append({"abs_alpha": r, "max_err_all_80x80": float(err.max()), "max_err_block_m,n<60": float(err[:60, :60].max()),
                 "max_err_block_m,n<40": float(err[:40, :40].max())})
    print(rows[-1], flush=True); R["2_laguerre_mpmath_nf80"] = rows; save()
