"""Theory-first joint cavity unitary (hardware model).

HARDWARE layer (Zach): local ECD on each cavity–transmon (optional) + ARBITRARY
unitary on the joint cavity space via GRAPE. No JP.

Theory (exact, GRAPE-friendly later):
  joint cavity U ≈ Gray-X mixer on every cavity  ×  cost diagonal D(C).
  ECD optional: only mixes each cavity with its control transmon (Fock n↔n±1);
  it cannot Hamming-mix cavity bits or couple cavities. Ablate with --ecd 0/1.

Roles:
  - ECD: entangle / move info with transmon bits, or cheap local prep. Optional
    when the joint U already owns cavity mixing.
  - Joint U (GRAPE): the ONLY place that mixes across cavities and applies cost.
    Here: product of local Gray-X (QAOA-analogue hypercube mixer) + diagonal D.

Diagonals (--diag):
  raw   : D = g * Cn, Cn = E*n/m
  poly2 : D = g * Cn + g2 * Cn^2          (size-free; Cn already density-normed)
  soft  : D = g * tanh(a * Cn), a trainable per layer
  step  : D = g * (Cn > t).astype  — soft approx: g * sigmoid(s*(Cn - t))
          with fixed s=8, trainable t,g  (thresholded cost)

Params/layer:
  ecd=0: [b_tm, b_cav, g] (+ g2 or a/t)
  ecd=1: [bre, bim, th, ph, b_cav, g] (+ ...)
"""
from __future__ import annotations
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, jax, jax.numpy as jnp
from scipy.optimize import minimize
import sim

ap = argparse.ArgumentParser()
ap.add_argument("--tag", required=True)
ap.add_argument("--layout", default="T", choices=["T", "C", "R"])
ap.add_argument("--ecd", type=int, default=0, choices=[0, 1])
ap.add_argument("--diag", default="raw", choices=["raw", "poly2", "soft", "step"])
ap.add_argument("--Ls", type=int, nargs="+", default=[1, 2, 4, 6, 8])
ap.add_argument("--ntrain", type=int, default=100)
ap.add_argument("--nt2", type=int, nargs="*", default=[])
ap.add_argument("--test", type=int, nargs="*", default=[10, 12, 14, 20])
ap.add_argument("--restarts", type=int, default=5)
ap.add_argument("--maxiter", type=int, default=300)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

LAY = {"T": sim.LAYOUTS_T, "C": sim.LAYOUTS_C, "R": sim.LAYOUTS_R}[a.layout]
tbit = a.layout in ("T", "R")
log = lambda *s: print(time.strftime("%H:%M:%S"), *s, flush=True)
TRB, TEB = 31_000_000_000, 32_000_000_000
OUT = f"results/{a.tag}.json"

# param layout
BASE = (6 if a.ecd else 3)  # ecd: bre,bim,th,ph,bcav,g | noecd: btm,bcav,g
EXTRA = {"raw": 0, "poly2": 1, "soft": 1, "step": 1}[a.diag]
NP = BASE + EXTRA


def ungray_scalar(g):
    g = int(g); s = g >> 1
    while s:
        g ^= s; s >>= 1
    return g


def gray_scalar(i):
    return i ^ (i >> 1)


def gray_X_sum(k, nf):
    N = 1 << k
    M = np.zeros((nf, nf), complex)
    for n in range(N):
        bits = gray_scalar(n)
        for j in range(k):
            n2 = ungray_scalar(bits ^ (1 << j))
            M[n2, n] += 1.0
    return M


def build(n, insts, ks):
    C = len(ks)
    assert sum(ks) + (C if tbit else 0) == n
    nfs = [(1 << k) + max(8, (1 << k) // 4) for k in ks]
    Vs, lams = [], []
    for k, nf in zip(ks, nfs):
        X = gray_X_sum(k, nf)
        lam, V = np.linalg.eigh(X)
        Vs.append(jnp.asarray(V.astype(np.complex64)))
        lams.append(jnp.asarray(lam.astype(np.float32)))
    # ECD displacement eigendecomps (only used if ecd=1)
    Vd, lamd, nvs, scales = [], [], [], []
    for k, nf in zip(ks, nfs):
        aa = np.diag(np.sqrt(np.arange(1, nf)), 1)
        lam, V = np.linalg.eigh(1j * (aa.T - aa))
        Vd.append(jnp.asarray(V.astype(np.complex64)))
        lamd.append(jnp.asarray(lam.astype(np.float32)))
        nvs.append(jnp.arange(nf))
        scales.append(np.sqrt(2.0 ** k))

    x = np.arange(1 << n); pos = n; tidx, fidx = [], []
    for c, k in enumerate(ks):
        nb = k + (1 if tbit else 0); pos -= nb
        val = (x >> pos) & ((1 << nb) - 1)
        if tbit:
            tidx.append(val >> k); f = val & ((1 << k) - 1)
        else:
            tidx.append(np.zeros_like(x)); f = val
        def ug(g):
            g = np.array(g, copy=True); s = g >> 1
            while np.any(s):
                g ^= s; s >>= 1
            return g
        fidx.append(ug(f))
    shape = [2] * C + nfs
    D = int(np.prod(shape))
    flat = np.ravel_multi_index(tuple(tidx + fidx), shape)
    B = len(insts)
    Cn = np.zeros((B, D), np.float32); succ = np.zeros((B, D), np.float32)
    for i, ins in enumerate(insts):
        E = ins["E"].astype(float); c = E * n / ins["m"]
        row = np.full(D, c.max()); row[flat] = c; Cn[i] = row
        np.add.at(succ[i], flat, (E == 0).astype(float))
    Cn = Cn.reshape([B] + shape); succ = succ.reshape([B] + shape)
    if not tbit:
        sl = (slice(None),) + (0,) * C
        Cn = np.broadcast_to(Cn[sl][(slice(None),) + (None,) * C], Cn.shape).copy()
        succ = np.broadcast_to(succ[sl][(slice(None),) + (None,) * C], succ.shape).copy()
    psi0 = np.zeros(D, np.complex64)
    if tbit:
        psi0[flat] = 1 / np.sqrt(len(flat))
    else:
        fl0 = np.ravel_multi_index(tuple([np.zeros_like(x)] * C + fidx), shape)
        fl0 = np.unique(fl0); psi0[fl0] = 1 / np.sqrt(len(fl0))
    psi0 = psi0.reshape(shape)
    Cn_j = jnp.asarray(Cn); succ_j = jnp.asarray(succ); psi0_j = jnp.asarray(psi0)

    def disp(c, re, im):
        s = scales[c] / 2.0
        r = jnp.sqrt(re ** 2 + im ** 2 + 1e-20) * s
        ph = jnp.arctan2(im, re)
        W = jnp.exp(1j * ph * nvs[c])[:, None] * Vd[c]
        return ((W * jnp.exp(-1j * r * lamd[c])[None, :]) @ W.conj().T).astype(jnp.complex64)

    def cav_U(c, b):
        return ((Vs[c] * jnp.exp(-1j * (b / 2) * lams[c])[None, :]) @ Vs[c].conj().T).astype(jnp.complex64)

    def apply_M(M, psi, ax):
        return jnp.moveaxis(jnp.tensordot(M, psi, axes=([1], [ax])), 0, ax)

    def phase_of(cn, g, extra):
        if a.diag == "raw":
            return g * cn
        if a.diag == "poly2":
            return g * cn + extra * (cn ** 2)
        if a.diag == "soft":
            # extra = a_soft > 0 preferred; allow any
            return g * jnp.tanh(extra * cn)
        # step: soft threshold, extra = t
        return g * jax.nn.sigmoid(8.0 * (cn - extra))

    def state1(x, psi, cn):
        L = x.size // NP
        x = x.reshape(L, NP)
        for l in range(L):
            row = x[l]
            if a.ecd:
                bre, bim, th, ph, bc, g = row[0], row[1], row[2], row[3], row[4], row[5]
                extra = row[6] if EXTRA else 0.0
                cth, sth = jnp.cos(th / 2), jnp.sin(th / 2)
                R = jnp.array([[cth, -1j * sth * jnp.exp(-1j * ph)],
                               [-1j * sth * jnp.exp(1j * ph), cth]]).astype(jnp.complex64)
                for c in range(C):
                    psi = apply_M(R, psi, c)
                    Dp = disp(c, bre, bim)
                    p0 = jnp.take(psi, 0, axis=c); p1 = jnp.take(psi, 1, axis=c)
                    fa = C - 1 + c
                    n0 = apply_M(Dp.conj().T, p1, fa); n1 = apply_M(Dp, p0, fa)
                    psi = jnp.stack([n0, n1], axis=c)
            else:
                bt, bc, g = row[0], row[1], row[2]
                extra = row[3] if EXTRA else 0.0
                Mtm = jnp.array([[jnp.cos(bt / 2), -1j * jnp.sin(bt / 2)],
                                 [-1j * jnp.sin(bt / 2), jnp.cos(bt / 2)]]).astype(jnp.complex64)
                for c in range(C):
                    psi = apply_M(Mtm, psi, c)
            # joint cavity unitary (theory): product Gray-X on cavities, then cost diag
            for c in range(C):
                psi = apply_M(cav_U(c, bc), psi, C + c)
            psi = psi * jnp.exp(-1j * phase_of(cn, g, extra)).astype(jnp.complex64)
        return psi

    def probs(x):
        def one(args):
            cn, sc = args
            return jnp.sum((jnp.abs(state1(x, psi0_j, cn)) ** 2) * sc)
        return jax.lax.map(one, (Cn_j, succ_j))

    pjit = jax.jit(probs)
    vg = jax.jit(jax.value_and_grad(lambda x: -jnp.mean(jnp.log(probs(x) + 1e-12))))
    return pjit, vg


# ---- train / test ----
tr_inst = sim.instances(8, a.ntrain, TRB)
builders = [(8, tr_inst, LAY[8])]
for m in a.nt2:
    builders.append((m, sim.instances(m, max(40, a.ntrain // 2), TRB), LAY[m]))

# build train circuits
tr_p, tr_vg = [], []
for n, insts, ks in builders:
    p, vg = build(n, insts, ks)
    tr_p.append(p); tr_vg.append(vg)

def f(x):
    vs = [vg(jnp.asarray(x, jnp.float32)) for vg in tr_vg]
    loss = sum(float(v[0]) for v in vs) / len(vs)
    grad = sum(np.asarray(v[1], float) for v in vs) / len(vs)
    return loss, grad

rng = np.random.default_rng(a.seed)
res = {"args": vars(a), "train": {}, "test": {}, "np": NP}
prev = None

def init_layer(scale=1.0):
    if a.ecd:
        v = np.array([0.15 * rng.standard_normal(), 0.15 * rng.standard_normal(),
                      rng.uniform(0, np.pi), rng.uniform(0, 2 * np.pi),
                      0.6 * scale, 0.4 * scale])
    else:
        v = np.array([0.6 * scale, 0.6 * scale, 0.4 * scale])
    if a.diag == "poly2":
        v = np.concatenate([v, [0.05 * rng.standard_normal()]])
    elif a.diag == "soft":
        v = np.concatenate([v, [0.8 + 0.2 * rng.standard_normal()]])
    elif a.diag == "step":
        v = np.concatenate([v, [0.5 + 0.2 * rng.standard_normal()]])
    return v

for L in a.Ls:
    t0 = time.time(); cands = []
    if prev is not None:
        Lp = prev.size // NP
        cands.append(np.concatenate([prev, 0.1 * np.concatenate([init_layer() for _ in range(L - Lp)])]))
        xp = prev.reshape(Lp, NP)
        t0s, t1s = np.linspace(0, 1, Lp), np.linspace(0, 1, L)
        xi = np.stack([np.interp(t1s, t0s, xp[:, j]) if Lp > 1 else np.repeat(xp[0, j], L) for j in range(NP)], 1)
        cands.append(xi.reshape(-1))
    for s in (0.4, 0.8, 1.2, 1.6):
        cands.append(np.concatenate([init_layer(s) for _ in range(L)]))
    # QAOA-like decreasing beta / increasing gamma schedule (noecd)
    if not a.ecd:
        sched = np.stack([np.linspace(0.9, 0.2, L), np.linspace(0.9, 0.2, L), np.linspace(0.2, 0.9, L)], 1)
        if EXTRA:
            extra0 = {"poly2": 0.0, "soft": 1.0, "step": 0.5}[a.diag]
            sched = np.concatenate([sched, np.full((L, 1), extra0)], 1)
        cands.append(sched.reshape(-1))
        cands.append(sched.reshape(-1) * 1.4)
    cands = cands[: a.restarts + 2]
    best = None
    for c0 in cands:
        r = minimize(f, c0, jac=True, method="L-BFGS-B", options={"maxiter": a.maxiter})
        log(f"  L={L} try loss={r.fun:.4f}")
        if best is None or r.fun < best[1]:
            best = (r.x, float(r.fun))
    prev = best[0]
    p = np.asarray(tr_p[0](jnp.asarray(prev, jnp.float32)))
    res["train"][L] = {"x": prev.tolist(), "loss": best[1], "train": sim.summarize(p), "sec": time.time() - t0}
    log(f"L={L} train med1/p={np.median(1/np.maximum(p,1e-300)):.1f} mean_p={p.mean():.4f} ({time.time()-t0:.0f}s)")
    json.dump(res, open(OUT, "w"), indent=1)

for n, N in [(8, 100)] + [(nn, 200 if nn == 14 else 100) for nn in a.test]:
    ins = sim.instances(n, N, TEB); res["test"][n] = {}
    ch = 20 if n < 14 else (10 if n < 20 else 4)
    for L in a.Ls:
        p = []
        x = np.asarray(res["train"][L]["x"], np.float32)
        for i in range(0, N, ch):
            pp, _ = build(n, ins[i:i + ch], LAY[n])
            p += list(np.asarray(pp(jnp.asarray(x, jnp.float32))))
        s = sim.summarize(p, ins); s["p"] = [float(v) for v in p]
        res["test"][n][L] = s
        log(f"test n={n} L={L} med1/p={s['median_inv_p']:.1f} B90={s['B90_inv_p']:.1f} rand={s['rand_median']:.0f}")
        json.dump(res, open(OUT, "w"), indent=1)
log("ALLDONE")
