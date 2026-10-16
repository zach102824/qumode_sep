"""Gray-bit X mixer on cavity Fock space (exact logical X on Gray bits), + cost phase.
Used as an upper-bound check: if THIS transfers like QAOA, the gap is the ECD mixer specifically.
Params/layer: beta_tm (transmon X angle), beta_cav (cavity Gray-X angle, shared), gamma.
Cavity unitary = exp(-i beta_cav/2 * sum_j X_j^{Gray}) on logical Fock levels 0..2^k-1 (leaked levels: I).
"""
from __future__ import annotations
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, jax, jax.numpy as jnp
from scipy.optimize import minimize
from scipy.linalg import expm
import sim

NP = 3  # b_tm, b_cav, gamma
ap = argparse.ArgumentParser()
ap.add_argument("--tag", default="BF_gray"); ap.add_argument("--layout", default="T")
ap.add_argument("--Ls", type=int, nargs="+", default=[1, 2, 4, 6, 8])
ap.add_argument("--ntrain", type=int, default=100); ap.add_argument("--test", type=int, nargs="*", default=[10, 12, 14, 20])
ap.add_argument("--restarts", type=int, default=4); ap.add_argument("--maxiter", type=int, default=300)
a = ap.parse_args()
LAY = {"T": sim.LAYOUTS_T, "R": sim.LAYOUTS_R, "C": sim.LAYOUTS_C}[a.layout]
log = lambda *s: print(time.strftime("%H:%M:%S"), *s, flush=True)

def gray_X_sum(k, nf):
    """sum_j X_j on Gray-encoded bits, as nf×nf matrix (logical block only)."""
    N = 1 << k; M = np.zeros((nf, nf), complex)
    # Gray code: binary i <-> gray g=i^(i>>1). Ungray already used in sim.
    # Bit j of the Gray label of Fock n: for Gray encoding of bits, Fock level n stores
    # gray_encode(bitstring). Our sim stores ungray(ga) as Fock, with ga = Gray of bits.
    # So Fock n's Gray label is gray(n) = n^(n>>1). Flipping Gray-bit j of the LABEL means
    # XOR the Gray label with 2^j, then convert back to Fock via ungray.
    def ungray(g):
        g = int(g); s = g >> 1
        while s: g ^= s; s >>= 1
        return g
    def gray(i): return i ^ (i >> 1)
    # Wait: sim stores Fock = ungray(ga) where ga are the Gray-coded bits.
    # So bits b (integer) -> ga = gray(b)? Let's re-read sim.
    # fidx = ungray(f) if code=="gray" where f = bits from the register.
    # So Fock level = ungray(bitstring). Thus bitstring = gray(Fock).
    # Flipping bit j of the bitstring: new_bits = bits XOR (1<<j), new_Fock = ungray(new_bits).
    for n in range(N):
        bits = gray(n)  # since n = ungray(bits) => bits = gray(n)
        for j in range(k):
            n2 = ungray(bits ^ (1 << j))
            M[n2, n] += 1.0
    return M

def build(n, insts, ks):
    C = len(ks); tbit = a.layout in ("T", "R")
    assert sum(ks) + (C if tbit else 0) == n
    nfs = [(1 << k) + max(8, (1 << k) // 4) for k in ks]
    # Precompute cavity Gray-X unitaries for a grid of beta, via eigendecomp
    Xs, Vs, lams = [], [], []
    for k, nf in zip(ks, nfs):
        X = gray_X_sum(k, nf); lam, V = np.linalg.eigh(X)
        Xs.append(X); Vs.append(jnp.asarray(V.astype(np.complex64))); lams.append(jnp.asarray(lam.astype(np.float32)))
    # Cost / success like Circuit
    x = np.arange(1 << n); pos = n; tidx, fidx = [], []
    for c, k in enumerate(ks):
        nb = k + (1 if tbit else 0); pos -= nb
        val = (x >> pos) & ((1 << nb) - 1)
        if tbit: tidx.append(val >> k); f = val & ((1 << k) - 1)
        else: tidx.append(np.zeros_like(x)); f = val
        # Gray: Fock = ungray(bits)
        def ug(g):
            g = np.array(g, copy=True); s = g >> 1
            while np.any(s): g ^= s; s >>= 1
            return g
        fidx.append(ug(f))
    shape = [2] * C + nfs; D = int(np.prod(shape))
    flat = np.ravel_multi_index(tuple(tidx + fidx), shape)
    B = len(insts); Cn = np.zeros((B, D), np.float32); succ = np.zeros((B, D), np.float32)
    for i, ins in enumerate(insts):
        E = ins["E"].astype(float); c = E * n / ins["m"]
        row = np.full(D, c.max()); row[flat] = c; Cn[i] = row
        np.add.at(succ[i], flat, (E == 0).astype(float))
    Cn = Cn.reshape([B] + shape); succ = succ.reshape([B] + shape)
    psi0 = np.zeros(D, np.complex64); psi0[flat] = 1 / np.sqrt(len(flat)); psi0 = psi0.reshape(shape)
    Cn_j, succ_j, psi0_j = jnp.asarray(Cn), jnp.asarray(succ), jnp.asarray(psi0)
    def cav_U(c, b):
        return ((Vs[c] * jnp.exp(-1j * (b / 2) * lams[c])[None, :]) @ Vs[c].conj().T).astype(jnp.complex64)
    def apply_M(M, psi, ax):
        return jnp.moveaxis(jnp.tensordot(M, psi, axes=([1], [ax])), 0, ax)
    def state1(x, psi, cn):
        L = x.size // NP; x = x.reshape(L, NP)
        for l in range(L):
            bt, bc, g = x[l, 0], x[l, 1], x[l, 2]
            # transmon X rotation
            M = jnp.array([[jnp.cos(bt / 2), -1j * jnp.sin(bt / 2)], [-1j * jnp.sin(bt / 2), jnp.cos(bt / 2)]]).astype(jnp.complex64)
            for c in range(C):
                psi = apply_M(M, psi, c)
                psi = apply_M(cav_U(c, bc), psi, C + c)
            psi = psi * jnp.exp(-1j * g * cn).astype(jnp.complex64)
        return psi
    def probs(x):
        def one(args):
            cn, sc = args
            return jnp.sum((jnp.abs(state1(x, psi0_j, cn)) ** 2) * sc)
        return jax.lax.map(one, (Cn_j, succ_j))
    pjit = jax.jit(probs)
    vg = jax.jit(jax.value_and_grad(lambda x: -jnp.mean(jnp.log(probs(x) + 1e-12))))
    return pjit, vg

TRB, TEB = 31_000_000_000, 32_000_000_000
pr, vg = build(8, sim.instances(8, a.ntrain, TRB), LAY[8])
f = lambda x: tuple(np.asarray(v, float) for v in vg(jnp.asarray(x, jnp.float32)))
rng = np.random.default_rng(0); res = {"train": {}, "test": {}}; prev = None
OUT = f"results/{a.tag}.json"
for L in a.Ls:
    cands = []
    if prev is not None:
        cands.append(np.concatenate([prev, 0.1 * rng.standard_normal(NP * (L - len(prev) // NP))]))
        xp = prev.reshape(-1, NP); t0, t1 = np.linspace(0, 1, len(xp)), np.linspace(0, 1, L)
        xi = np.stack([np.interp(t1, t0, xp[:, j]) if len(xp) > 1 else np.repeat(xp[0, j], L) for j in range(NP)], 1)
        cands.append(xi.reshape(-1))
    for s in (0.5, 1.0, 1.5):
        cands.append(np.stack([np.linspace(0.9, 0.2, L), np.linspace(0.9, 0.2, L), np.linspace(0.2, 0.9, L)], 1).reshape(-1) * s)
    best = min((minimize(f, c, jac=True, method="L-BFGS-B", options={"maxiter": a.maxiter}) for c in cands[: a.restarts + 1]), key=lambda r: r.fun)
    prev = best.x; p = np.asarray(pr(jnp.asarray(prev, jnp.float32)))
    res["train"][L] = {"x": prev.tolist(), "train": sim.summarize(p)}; log(f"L={L} train med1/p={np.median(1/p):.1f}")
    json.dump(res, open(OUT, "w"), indent=1)
for n, N in [(8, 100)] + [(nn, 200 if nn == 14 else 100) for nn in a.test]:
    ins = sim.instances(n, N, TEB); res["test"][n] = {}
    ch = 20 if n < 14 else (10 if n < 20 else 4)
    for L in a.Ls:
        p = []
        for i in range(0, N, ch):
            pp, _ = build(n, ins[i:i + ch], LAY[n]); p += list(np.asarray(pp(jnp.asarray(res["train"][L]["x"], jnp.float32))))
        s = sim.summarize(p, ins); s["p"] = [float(v) for v in p]; res["test"][n][L] = s
        log(f"test n={n} L={L} med1/p={s['median_inv_p']:.1f} B90={s['B90_inv_p']:.1f} rand={s['rand_median']:.0f}")
        json.dump(res, open(OUT, "w"), indent=1)
log("ALLDONE")
