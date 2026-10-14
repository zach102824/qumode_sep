"""ECD-gate speed benchmark for the n=20 3-cavity device (ks 6,6,5; nf 80,80,48; 3 transmons).

Also provides ``apply_circuit_multi_fast`` (preallocated ping-pong buffers, batched matmul with out=,
no moveaxis/stack copies) used by relayout_3cav when QM_ECD_FAST=1.
"""
from __future__ import annotations

import json
import os
import sys
import time
from functools import lru_cache

import numpy as np
from scipy.linalg import expm
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import expm_multiply
from scipy.special import eval_genlaguerre, gammaln

from noiseless.circuit_local_ecd import rotation_np
from noiseless.ecd_kbit import _destroy, _disp_eig, displace_fast


# ---------------------------------------------------------------- displacement builders
def disp_laguerre(alpha, nf):
    """Closed-form <m|D(alpha)|n> of the infinite-dim D, truncated (not the expm of the truncated generator)."""
    x = abs(alpha) ** 2
    m = np.arange(nf)[:, None]
    n = np.arange(nf)[None, :]
    lo, hi = np.minimum(m, n), np.maximum(m, n)
    d = hi - lo
    L = eval_genlaguerre(lo, d, x)
    with np.errstate(divide="ignore"):
        logpre = 0.5 * (gammaln(lo + 1) - gammaln(hi + 1)) + d * (np.log(abs(alpha)) if alpha != 0 else -np.inf)
    ph = np.where(m >= n, np.exp(1j * np.angle(alpha) * d), (-np.exp(-1j * np.angle(alpha))) ** d)
    pre = np.where(d == 0, 1.0, np.exp(logpre))
    return pre * ph * L * np.exp(-x / 2)


def disp_split(alpha, nf):
    """e^{-|a|^2/2} exp(alpha a^dag) exp(-alpha* a), each a truncated (nilpotent -> exact series) exponential."""
    a = _destroy(nf)
    E1 = expm(alpha * a.conj().T)
    E2 = expm(-np.conj(alpha) * a)
    return np.exp(-abs(alpha) ** 2 / 2) * (E1 @ E2)


# ---------------------------------------------------------------- ECD step on (2,2,2,A,B,C)
def ecd_current(psi, betas, nfs):
    P = 3
    for t in range(P):
        D = displace_fast(complex(betas[t]) / 2.0, nfs[t])
        p0 = np.take(psi, 0, axis=t)
        p1 = np.take(psi, 1, axis=t)
        ax = P + t - 1
        new1 = np.moveaxis(np.tensordot(D, p0, axes=([1], [ax])), 0, ax)
        new0 = np.moveaxis(np.tensordot(D.conj().T, p1, axes=([1], [ax])), 0, ax)
        psi = np.stack([new0, new1], axis=t)
    return psi


def _ecd_mm(psi, out, Ds):
    """Apply 3 ECDs with batched matmul into preallocated buffers; returns final buffer (psi or out)."""
    DA, DB, DC = Ds
    s0 = psi.shape[1] * psi.shape[2], psi.shape[3], psi.shape[4] * psi.shape[5]
    np.matmul(DA, psi[0].reshape(s0), out=out[1].reshape(s0)); np.matmul(DA.conj().T, psi[1].reshape(s0), out=out[0].reshape(s0))
    psi, out = out, psi
    np.matmul(DB, psi[:, 0], out=out[:, 1]); np.matmul(DB.conj().T, psi[:, 1], out=out[:, 0])
    psi, out = out, psi
    np.matmul(psi[:, :, 0], DC.T, out=out[:, :, 1]); np.matmul(psi[:, :, 1], DC.conj(), out=out[:, :, 0])
    return out, psi


def ecd_mm(builder, dtype=complex):
    _buf = {}
    def f(psi, betas, nfs):
        key = (psi.shape, dtype)
        if key not in _buf:
            _buf[key] = (np.empty(psi.shape, dtype), np.empty(psi.shape, dtype))
        b0, b1 = _buf[key]
        b0[...] = psi
        Ds = [builder(complex(b) / 2.0, nf).astype(dtype) for b, nf in zip(betas, nfs)]
        res, other = _ecd_mm(b0, b1, Ds)
        _buf[key] = (res, other)
        return res
    return f



@lru_cache(maxsize=8)
def _gen_sparse(nf):
    a = csr_matrix(_destroy(nf))
    return a, a.conj().T.tocsr()


def ecd_expm_multiply(psi, betas, nfs):
    P = 3
    for t in range(P):
        al = complex(betas[t]) / 2.0
        a, ad = _gen_sparse(nfs[t])
        G = (al * ad - np.conj(al) * a).tocsr()
        ax = P + t - 1
        p0 = np.moveaxis(np.take(psi, 0, axis=t), ax, 0)
        p1 = np.moveaxis(np.take(psi, 1, axis=t), ax, 0)
        sh = p0.shape
        new1 = expm_multiply(G, p0.reshape(sh[0], -1)).reshape(sh)
        new0 = expm_multiply(-G, p1.reshape(sh[0], -1)).reshape(sh)
        psi = np.stack([np.moveaxis(new0, 0, ax), np.moveaxis(new1, 0, ax)], axis=t)
    return psi


def ecd_eig_state(psi, betas, nfs):
    """Eigen trick applied on the state: V^dag psi, diagonal phase, V psi (two basis changes, no D built)."""
    P = 3
    for t in range(P):
        lam, V = _disp_eig(nfs[t])
        al = complex(betas[t]) / 2.0
        W = np.exp(1j * np.angle(al) * np.arange(nfs[t]))[:, None] * V
        e = np.exp(-1j * abs(al) * lam)
        ax = P + t - 1
        sh = [1] * (psi.ndim - 1); sh[ax] = nfs[t]
        p0 = np.take(psi, 0, axis=t); p1 = np.take(psi, 1, axis=t)
        def ap(M, x): return np.moveaxis(np.tensordot(M, x, axes=([1], [ax])), 0, ax)
        new1 = ap(W, ap(W.conj().T, p0) * e.reshape(sh))
        new0 = ap(W, ap(W.conj().T, p1) * e.conj().reshape(sh))
        psi = np.stack([new0, new1], axis=t)
    return psi


# ---------------------------------------------------------------- full fast circuit
def apply_circuit_multi_fast(x, n_layers, layout, u_name="jp", _cache={}):
    from noiseless.relayout_3cav import jp_phase, make_param_helpers, apply_circuit_multi
    P = layout.P
    if P != 3:
        return apply_circuit_multi(x, n_layers, layout, u_name)
    key = (layout.ks, layout.nfs, tuple(layout.edges), u_name)
    if key not in _cache:
        ph = jp_phase(layout) if u_name == "jp" else np.ones(layout.dims[P:], dtype=complex)
        _cache[key] = (ph, np.empty(layout.dims, complex), np.empty(layout.dims, complex))
    ph, A, B = _cache[key]
    h = make_param_helpers(P)
    A[...] = 0
    A[(0,) * len(layout.dims)] = 1.0
    for betas, thetas, phis in h["unpack"](x, n_layers):
        for t in range(P):  # rotations, into B then swap
            R = rotation_np(float(thetas[t]), float(phis[t]))
            idx0 = (slice(None),) * t + (0,); idx1 = (slice(None),) * t + (1,)
            a0, a1 = A[idx0], A[idx1]
            np.multiply(a0, R[0, 0], out=B[idx0]); B[idx0] += R[0, 1] * a1
            np.multiply(a0, R[1, 0], out=B[idx1]); B[idx1] += R[1, 1] * a1
            A, B = B, A
        Ds = [displace_fast(complex(b) / 2.0, nf) for b, nf in zip(betas, layout.nfs)]
        A, B = _ecd_mm(A, B, Ds)
        A *= ph
    _cache[key] = (ph, B, A)  # keep buffers; result returned as a copy
    return A.reshape(-1).copy()


def apply_circuit_multi_fast2(x, n_layers, layout, u_name="jp", _cache={}):
    """fast + (i) layer 1 built analytically as a product of the 3 (transmon, cavity) pair states (vacuum start),
    (ii) the 3 transmon rotations fused into one 8x8 Kronecker matmul, (iii) the final jp skipped
    (a diagonal phase, so |amplitude|^2 is unchanged; only valid when the caller takes born())."""
    from noiseless.relayout_3cav import jp_phase, make_param_helpers, apply_circuit_multi
    P = layout.P
    if P != 3:
        return apply_circuit_multi(x, n_layers, layout, u_name)
    key = (layout.ks, layout.nfs, tuple(layout.edges), u_name)
    if key not in _cache:
        ph = jp_phase(layout) if u_name == "jp" else np.ones(layout.dims[P:], dtype=complex)
        _cache[key] = (ph, np.empty(layout.dims, complex), np.empty(layout.dims, complex))
    ph, A, B = _cache[key]
    h = make_param_helpers(P)
    nfs = layout.nfs
    layers = h["unpack"](x, n_layers)
    for li, (betas, thetas, phis) in enumerate(layers):
        Rs = [rotation_np(float(thetas[t]), float(phis[t])) for t in range(P)]
        Ds = [displace_fast(complex(b) / 2.0, nf) for b, nf in zip(betas, nfs)]
        if li == 0:
            S = []
            for t in range(P):
                v = Rs[t][:, 0]
                st = np.empty((2, nfs[t]), complex)
                st[1] = v[0] * Ds[t][:, 0]
                st[0] = v[1] * Ds[t].conj().T[:, 0]
                S.append(st)
            # psi[i,j,k,a,b,c] = S0[i,a] S1[j,b] S2[k,c]
            s01 = (S[0][:, None, :, None] * S[1][None, :, None, :])  # (2,2,A,B)
            np.multiply(s01[:, :, None, :, :, None], S[2][None, None, :, None, None, :], out=A)
        else:
            R8 = np.kron(np.kron(Rs[0], Rs[1]), Rs[2])
            np.matmul(R8, A.reshape(8, -1), out=B.reshape(8, -1))
            A, B = _ecd_mm(B, A, Ds)
        if li < len(layers) - 1:
            A *= ph
    _cache[key] = (ph, B, A)
    return A.reshape(-1).copy()


# ---------------------------------------------------------------- benchmark
def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "noiseless/results/ecd_speed/bench.json"
    nfs = (80, 80, 48)
    rng = np.random.default_rng(0)
    # realistic state: 2 layers of the real circuit from random params (support spread over Fock levels)
    from noiseless.relayout_3cav import MultiCavLayout, apply_circuit_multi, make_param_helpers
    lay = MultiCavLayout((6, 6, 5), nfs)
    h = make_param_helpers(3)
    x = h["random_parameters"](2, rng)
    psi0 = apply_circuit_multi(x, 2, lay).reshape(lay.dims)
    betas = np.array([2.1 + 1.3j, -1.7 + 0.4j, 0.9 - 2.2j])
    ref = ecd_current(psi0, betas, nfs)
    methods = {
        "a_current(eig D + tensordot/moveaxis/stack)": ecd_current,
        "b_laguerre_D + matmul(out=)": ecd_mm(disp_laguerre),
        "c_expm_multiply": ecd_expm_multiply,
        "d1_eig_D + matmul(out=)": ecd_mm(displace_fast),
        "d2_eig_on_state(two basis changes)": ecd_eig_state,
        "e_eig_D + matmul complex64": ecd_mm(displace_fast, np.complex64),
        "f_split exp(a^dag)exp(a)": ecd_mm(disp_split),
        "g_scipy_expm_D + matmul": ecd_mm(lambda al, nf: expm(al * _destroy(nf).conj().T - np.conj(al) * _destroy(nf))),
    }
    res = {"threads": os.environ.get("OMP_NUM_THREADS"), "dims": list(lay.dims), "rows": []}
    for name, f in methods.items():
        f(psi0, betas, nfs)
        reps = 3 if "expm_multiply" in name else 10
        t0 = time.perf_counter()
        for _ in range(reps):
            r = f(psi0, betas, nfs)
        dt = (time.perf_counter() - t0) / reps
        err = float(np.max(np.abs(np.asarray(r, complex) - ref)))
        res["rows"].append({"method": name, "ms_per_layer": dt * 1e3, "max_abs_err": err})
        print(name, dt * 1e3, err, flush=True)
        json.dump(res, open(out, "w"), indent=1)
    # D-build cost alone
    for nm, bld in (("eig", displace_fast), ("laguerre", disp_laguerre), ("expm", lambda al, nf: expm(al * _destroy(nf).conj().T - np.conj(al) * _destroy(nf)))):
        t0 = time.perf_counter()
        for _ in range(50):
            bld(1.0 + 0.5j, 80)
        res[f"Dbuild_ms_{nm}_nf80"] = (time.perf_counter() - t0) / 50 * 1e3
    # jp is a diagonal phase?
    ph = __import__("noiseless.relayout_3cav", fromlist=["jp_phase"]).jp_phase(lay)
    res["jp_diag_values"] = sorted({complex(v).__repr__() for v in np.unique(np.round(ph, 12))})
    t0 = time.perf_counter()
    for _ in range(10):
        q = psi0 * ph
    res["jp_ms"] = (time.perf_counter() - t0) / 10 * 1e3
    # full circuit L=4 old vs fast
    x4 = h["random_parameters"](4, rng)
    o = apply_circuit_multi(x4, 4, lay); nfast = apply_circuit_multi_fast(x4, 4, lay)
    res["full_L4_err"] = float(np.max(np.abs(o - nfast)))
    for nm, f in (("old", apply_circuit_multi), ("fast", apply_circuit_multi_fast)):
        f(x4, 4, lay); t0 = time.perf_counter()
        for _ in range(5):
            f(x4, 4, lay)
        res[f"full_L4_ms_{nm}"] = (time.perf_counter() - t0) / 5 * 1e3
    from noiseless.ecd_kbit import born
    nf2 = apply_circuit_multi_fast2(x4, 4, lay)
    res["full_L4_prob_err_fast2"] = float(np.max(np.abs(born(o) - born(nf2))))
    apply_circuit_multi_fast2(x4, 4, lay); t0 = time.perf_counter()
    for _ in range(5):
        apply_circuit_multi_fast2(x4, 4, lay)
    res["full_L4_ms_fast2"] = (time.perf_counter() - t0) / 5 * 1e3
    t0 = time.perf_counter()
    for _ in range(5):
        born(o)
    res["born_ms"] = (time.perf_counter() - t0) / 5 * 1e3
    json.dump(res, open(out, "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "rows"}, indent=1))


if __name__ == "__main__":
    main()
