"""Importable calibration helpers (same model as mixcal.py): fit the native sandwich S_out D(a_k) S_{k-1} ... D(a_1) S_in to the
one-bit-flip mixer U(b) = prod_j exp(-i b X_j/2) on the K Gray digits of a cavity at an EXACT beta (warm start from the mixcal grid),
and the native flat-state prep (k pairs D then SNAP from vacuum; final SNAP merged into the first compiled SNAP).
Matrices returned in the LOGICAL basis (index g = logical cavity value), valid block only (leaked amplitude dropped)."""
from __future__ import annotations
import json, os, numpy as np, jax, jax.numpy as jnp
from scipy.optimize import minimize
jax.config.update("jax_enable_x64", True)
gray = lambda i: i ^ (i >> 1)
def ungray(g):
    s = g >> 1
    while s: g ^= s; s >>= 1
    return g
def nf_of(K): N = 1 << K; return N + max(16, N // 2 + 8)
def target(K, b, kind="gray"):
    N = 1 << K
    if kind == "gray":
        U = np.eye(N, dtype=complex)
        for j in range(K):
            P = np.zeros((N, N))
            for f in range(N): P[ungray(gray(f) ^ (1 << j)), f] = 1
            U = (np.cos(b / 2) * np.eye(N) - 1j * np.sin(b / 2) * P) @ U
        return U
    H = np.diag(np.ones(N - 1), 1); H = H + H.T; w, V = np.linalg.eigh(H); return (V * np.exp(-1j * b / 2 * w)) @ V.conj().T
_M = {}
def model(K, k, NF):
    key = (K, k, NF)
    if key in _M: return _M[key]
    N = 1 << K; aa = np.diag(np.sqrt(np.arange(1, NF)), 1); lam, V = np.linalg.eigh(1j * (aa.T - aa)); LAM, VD, NV = jnp.asarray(lam), jnp.asarray(V), jnp.arange(NF)
    def disp(re, im):
        r = jnp.sqrt(re ** 2 + im ** 2 + 1e-30); ph = jnp.arctan2(im, re); W = jnp.exp(1j * ph * NV)[:, None] * VD
        return (W * jnp.exp(-1j * r * LAM)[None]) @ W.conj().T
    def seq(x):
        al = x[:2 * k].reshape(k, 2); o = 2 * k; Sn = x[o:o + (k - 1) * NF].reshape(max(k - 1, 0), NF); o += (k - 1) * NF
        sin, sout = x[o:o + N], x[o + N:o + 2 * N]
        M = jnp.zeros((NF, N), complex).at[:N].set(jnp.diag(jnp.exp(-1j * sin)))
        for i in range(k):
            M = disp(al[i, 0], al[i, 1]) @ M
            if i < k - 1: M = jnp.exp(-1j * Sn[i])[:, None] * M
        return jnp.exp(-1j * sout)[:, None] * M[:N]
    def loss(x, U):
        Vp = seq(x); F = jnp.abs(jnp.trace(U.conj().T @ Vp)) ** 2 / N ** 2; lk = 1 - jnp.sum(jnp.abs(Vp) ** 2) / N; return 1 - F + lk
    _M[key] = (jax.jit(seq), jax.jit(jax.value_and_grad(loss))); return _M[key]
def fit_beta(K, k, b, kind="gray", starts=4, grid_dir="results/v3/native/mixcal", seed=0, maxiter=4000):
    """b any real; reduced to [0, pi] via period 2 pi and U(-b) = conj U(b). -> dict(F, leak, Vlog (N,N) complex, x)"""
    N = 1 << K; NF = nf_of(K); bb = (b + np.pi) % (2 * np.pi) - np.pi; neg = bb < 0; bb = abs(bb)
    seq, vg = model(K, k, NF); U = jnp.asarray(target(K, bb, kind)); rng = np.random.default_rng(seed); x0s = []
    gf = f"{grid_dir}/{kind}_K{K}_k{k}.json"
    if os.path.exists(gf):
        rows = json.load(open(gf))["rows"]; rows = sorted(rows, key=lambda r: abs(r["beta"] - bb))[:2]; x0s += [np.array(r["x"]) for r in rows]
    for _ in range(starts): x0s.append(np.concatenate([rng.normal(0, 0.4 + 1.2 * rng.random(), 2 * k), rng.uniform(0, 2 * np.pi, (k - 1) * NF + 2 * N)]))
    best = None
    for x0 in x0s:
        r = minimize(lambda x: tuple(np.asarray(v, float) for v in vg(x, U)), x0, jac=True, method="L-BFGS-B", options={"maxiter": maxiter, "maxfun": int(1.5 * maxiter)})
        if best is None or r.fun < best.fun: best = r
    Vp = np.asarray(seq(best.x)); F = abs(np.trace(np.asarray(U).conj().T @ Vp)) ** 2 / N ** 2; lk = 1 - np.sum(abs(Vp) ** 2) / N
    if neg: Vp = Vp.conj()
    perm = np.array([ungray(g) for g in range(N)]); Vlog = Vp[np.ix_(perm, perm)]   # Vlog[g', g] = V[ungray g', ungray g]
    return dict(F=float(F), leak=float(lk), Vlog=Vlog, x=best.x.tolist(), beta_red=float(bb), conj=bool(neg))
def fit_prep(K, k=3, starts=12, seed=0):
    """native prep: k x (D then SNAP) from vacuum; target |amp|^2 = 1/N on valid levels (phases removed by the merged SNAP).
    -> logical-order amplitudes on valid levels (real >= 0, norm^2 = 1 - leak), F."""
    N = 1 << K; NF = nf_of(K) + 16; aa = np.diag(np.sqrt(np.arange(1, NF)), 1); lam, V = np.linalg.eigh(1j * (aa.T - aa)); NV = np.arange(NF)
    def disp(re, im):
        r = jnp.sqrt(re ** 2 + im ** 2 + 1e-30); ph = jnp.arctan2(im, re); W = jnp.exp(1j * ph * NV)[:, None] * V
        return (W * jnp.exp(-1j * r * lam)[None]) @ W.conj().T
    def state(x):
        x = x.reshape(k, 2 + NF); psi = jnp.zeros(NF, complex).at[0].set(1)
        for i in range(k): psi = disp(x[i, 0], x[i, 1]) @ psi; psi = psi * jnp.exp(-1j * x[i, 2:])
        return psi
    vg = jax.jit(jax.value_and_grad(lambda x: -(jnp.sum(jnp.abs(state(x)[:N])) / np.sqrt(N)) ** 2)); rng = np.random.default_rng(seed); best = None
    for _ in range(starts):
        x0 = np.concatenate([[rng.uniform(0.3, 2.5 + K / 2), rng.uniform(-1, 1)] + list(rng.uniform(0, 6.28, NF)) for _ in range(k)])
        r = minimize(lambda x: tuple(np.asarray(v, float) for v in vg(x)), x0, jac=True, method="L-BFGS-B", options={"maxiter": 3000})
        if best is None or r.fun < best.fun: best = r
    psi = np.asarray(state(best.x)); amp = np.abs(psi[:N]); perm = np.array([ungray(g) for g in range(N)])
    return dict(F=float(-best.fun), leak=float(np.sum(abs(psi[N:]) ** 2)), amps_logical=amp[perm], x=best.x.tolist())
