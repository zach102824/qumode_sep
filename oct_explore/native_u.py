"""Native qumode joint cavity U — NO Pauli/Gray-X.

Zach (23:12 CST): prefer ECD + SNAP + displacement/phase on cavities.
Joint cavity U built from SNAP/displacement generators, not Pauli-X.
Gray-X (S_costgx) kept as reference only. Goal: transfer n=8→14/20 ≈ SATLike.

HARDWARE layer: local ECD each cavity–transmon + one joint cavity unitary (GRAPE later).

Native theory forms (--form):
  d_snap     : per cavity  D(α) · SNAP(s)  then cost          (disp then phase)
  snap_d     : per cavity  SNAP(s) · D(α)  then cost
  d_snap_d   : per cavity  D(α1)·SNAP(s)·D(α2) then cost      (echo-like)
  snap_cost  : SNAP only + cost (no extra disp; ECD is the only mixer)
  gen_sum    : U = exp(-i ∑_c [re·x_c + im·p_c + ∑_j s_j n_c^{j}]) · cost
               (single exp of native generators; Trotter-free local)
  bs_snap    : SNAP on each cav + beam-splitter θ between cav pairs + cost

Params/layer (ecd always on): [bre,bim,th,ph, ...] + form-specific + gamma
  SNAP degree --snap (default 2): size-free s_j · (n/sqrt(2^k))^{j+1}
  Displacements α = (are + i aim) * sqrt(2^k)  shared across cavities (size-free amp)
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
ap.add_argument("--form", default="d_snap",
                choices=["d_snap", "snap_d", "d_snap_d", "snap_cost", "gen_sum", "bs_snap"])
ap.add_argument("--snap", type=int, default=2)
ap.add_argument("--Ls", type=int, nargs="+", default=[2, 4, 8])
ap.add_argument("--ntrain", type=int, default=80)
ap.add_argument("--test", type=int, nargs="*", default=[10, 12, 14, 20])
ap.add_argument("--restarts", type=int, default=4)
ap.add_argument("--maxiter", type=int, default=250)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

LAY = {"T": sim.LAYOUTS_T, "C": sim.LAYOUTS_C, "R": sim.LAYOUTS_R}[a.layout]
tbit = a.layout in ("T", "R")
log = lambda *s: print(time.strftime("%H:%M:%S"), *s, flush=True)
TRB, TEB = 31_000_000_000, 32_000_000_000
OUT = f"results/{a.tag}.json"
SNAP = int(a.snap)

# ECD always: bre,bim,th,ph (4)
# + form params + gamma
# d_snap / snap_d: are, aim (2) + snap + g
# d_snap_d: are1,aim1,are2,aim2 (4) + snap + g
# snap_cost: snap + g
# gen_sum: are, aim (quad x,p amps) + snap + g  — same count as d_snap but applied as single exp
# bs_snap: theta_bs + snap + g
ECD_N = 4
if a.form in ("d_snap", "snap_d", "gen_sum"):
    FORM_N = 2 + SNAP  # are,aim + s
elif a.form == "d_snap_d":
    FORM_N = 4 + SNAP
elif a.form == "snap_cost":
    FORM_N = SNAP
elif a.form == "bs_snap":
    FORM_N = 1 + SNAP  # bs angle + s
else:
    raise ValueError(a.form)
NP = ECD_N + FORM_N + 1  # + gamma


def build(n, insts, ks):
    C = len(ks)
    assert sum(ks) + (C if tbit else 0) == n
    nfs = [(1 << k) + max(8, (1 << k) // 4) for k in ks]
    # displacement eigendecomps: D(r e^{iφ}) via eig of i(a†-a) after phase
    Vd, lamd, nvs, scales = [], [], [], []
    for k, nf in zip(ks, nfs):
        aa = np.diag(np.sqrt(np.arange(1, nf)), 1)  # a
        # x = (a+a†)/√2 generator used as 1j*(a†-a) for real displacement axis
        lam, V = np.linalg.eigh(1j * (aa.T - aa))
        Vd.append(jnp.asarray(V.astype(np.complex64)))
        lamd.append(jnp.asarray(lam.astype(np.float32)))
        nvs.append(jnp.arange(nf, dtype=np.float32))
        scales.append(float(np.sqrt(2.0 ** k)))

    # beam-splitter between cavity pairs: need joint 2-mode operators — apply via
    # eigendecomp on truncated joint space per pair (only if C>=2 and form bs_snap)
    bs_pack = None
    if a.form == "bs_snap" and C >= 2:
        # For each adjacent pair (c,c+1), build BS generator K = i(a†b - ab†) on nf0 x nf1
        # Use small nf for speed: exact 2^k block + pad
        packs = []
        for c in range(C - 1):
            n0, n1 = nfs[c], nfs[c + 1]
            # K |n,m> : a†b |n,m> = sqrt(n+1)sqrt(m) |n+1,m-1|; ab† = sqrt(n)sqrt(m+1)|n-1,m+1|
            dim = n0 * n1
            K = np.zeros((dim, dim), complex)
            for i in range(n0):
                for j in range(n1):
                    idx = i * n1 + j
                    if j > 0 and i + 1 < n0:
                        K[(i + 1) * n1 + (j - 1), idx] += 1j * np.sqrt(i + 1) * np.sqrt(j)
                    if i > 0 and j + 1 < n1:
                        K[(i - 1) * n1 + (j + 1), idx] += -1j * np.sqrt(i) * np.sqrt(j + 1)
            # Hermitian: K should be Hermitian (i(a†b-ab†))
            K = 0.5 * (K + K.conj().T)
            lam, V = np.linalg.eigh(K)
            packs.append((c, n0, n1,
                          jnp.asarray(V.astype(np.complex64)),
                          jnp.asarray(lam.astype(np.float32))))
        bs_pack = packs

    x = np.arange(1 << n); pos = n; tidx, fidx = [], []
    for c, k in enumerate(ks):
        nb = k + (1 if tbit else 0); pos -= nb
        val = (x >> pos) & ((1 << nb) - 1)
        if tbit:
            tidx.append(val >> k); f = val & ((1 << k) - 1)
        else:
            tidx.append(np.zeros_like(x)); f = val
        # Gray encoding (bit layout only; mixer is NOT Gray-X)
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

    def disp_mat(c, re, im):
        """D(α) with α=(re+i im)*scale/2 matching ECD convention half-amp on each branch style.
        Full cavity displacement: α = (re+i im)*scale."""
        s = scales[c]
        r = jnp.sqrt(re ** 2 + im ** 2 + 1e-20) * s
        ph = jnp.arctan2(im, re)
        W = jnp.exp(1j * ph * nvs[c])[:, None] * Vd[c]
        return ((W * jnp.exp(-1j * r * lamd[c])[None, :]) @ W.conj().T).astype(jnp.complex64)

    def ecd_disp(c, re, im):
        s = scales[c] / 2.0
        r = jnp.sqrt(re ** 2 + im ** 2 + 1e-20) * s
        ph = jnp.arctan2(im, re)
        W = jnp.exp(1j * ph * nvs[c])[:, None] * Vd[c]
        return ((W * jnp.exp(-1j * r * lamd[c])[None, :]) @ W.conj().T).astype(jnp.complex64)

    def apply_M(M, psi, ax):
        return jnp.moveaxis(jnp.tensordot(M, psi, axes=([1], [ax])), 0, ax)

    def apply_snap_diag(psi, s_coeffs):
        if SNAP <= 0:
            return psi
        for c in range(C):
            u = nvs[c] / scales[c]
            ph = sum(s_coeffs[j] * (u ** (j + 1)) for j in range(SNAP))
            Dph = jnp.exp(-1j * ph).astype(jnp.complex64)
            psi = psi * Dph.reshape([1] * (C + c) + [-1] + [1] * (len(nfs) - 1 - c))
        return psi

    def apply_disp_all(psi, re, im):
        for c in range(C):
            psi = apply_M(disp_mat(c, re, im), psi, C + c)
        return psi

    def apply_bs(psi, theta):
        if bs_pack is None:
            return psi
        for c, n0, n1, V, lam in bs_pack:
            U = ((V * jnp.exp(-1j * theta * lam)[None, :]) @ V.conj().T).astype(jnp.complex64)
            # axes: cavity c and c+1 are C+c, C+c+1
            ax0, ax1 = C + c, C + c + 1
            # move those axes to end, reshape, apply, restore
            psi = jnp.moveaxis(psi, (ax0, ax1), (-2, -1))
            sh = psi.shape
            psi2 = psi.reshape((-1, n0 * n1)) @ U.T
            psi = psi2.reshape(sh)
            psi = jnp.moveaxis(psi, (-2, -1), (ax0, ax1))
        return psi

    def apply_gen_sum(psi, re, im, s_coeffs):
        """exp(-i (re·x + im·p + SNAP poly)) per cavity via: phase frame + disp eig.
        Approximate as SNAP · D — same as snap_d for commuting attempt; use
        D then SNAP then D_phase: actually apply D(α) then SNAP (Trotter of gen_sum).
        Better: in number basis SNAP exact; for x,p use disp_mat.
        True single exp needs joint diag of x/p + n^j — n^j diagonal in Fock,
        x,p not — so eigh of H = re*X + im*P + poly(n) per cavity.
        """
        for c in range(C):
            nf = nfs[c]
            # Build H in Fock basis numerically each call is expensive; use cached a
            # X = (a+a†), P = -i(a-a†)  with a lowering
            # Precompute a as jnp
            pass
        # Fall back implemented below with precomputed a ops
        return apply_gen_sum_fast(psi, re, im, s_coeffs)

    # precompute a, X, P matrices
    a_ops, X_ops, P_ops = [], [], []
    for nf in nfs:
        aa = np.diag(np.sqrt(np.arange(1, nf)), 1).astype(np.complex64)
        X = (aa + aa.T.conj())
        P = -1j * (aa - aa.T.conj())
        a_ops.append(jnp.asarray(aa))
        X_ops.append(jnp.asarray(X))
        P_ops.append(jnp.asarray(P))

    def apply_gen_sum_fast(psi, re, im, s_coeffs):
        for c in range(C):
            H = re * X_ops[c] + im * P_ops[c]
            if SNAP > 0:
                u = nvs[c] / scales[c]
                pot = sum(s_coeffs[j] * (u ** (j + 1)) for j in range(SNAP))
                H = H + jnp.diag(pot)
            # scale displacement amps by cavity size so α magnitude ~ size-free
            # re,im already size-free; X,P matrix elements grow as sqrt(n) — multiply by scale
            H = H * scales[c]
            evals, evecs = jnp.linalg.eigh(H)
            U = (evecs * jnp.exp(-1j * evals)[None, :]) @ evecs.conj().T
            psi = apply_M(U.astype(jnp.complex64), psi, C + c)
        return psi

    def apply_ecd(psi, bre, bim, th, ph):
        cth, sth = jnp.cos(th / 2), jnp.sin(th / 2)
        R = jnp.array([[cth, -1j * sth * jnp.exp(-1j * ph)],
                       [-1j * sth * jnp.exp(1j * ph), cth]]).astype(jnp.complex64)
        for c in range(C):
            psi = apply_M(R, psi, c)
            Dp = ecd_disp(c, bre, bim)
            p0 = jnp.take(psi, 0, axis=c); p1 = jnp.take(psi, 1, axis=c)
            fa = C - 1 + c
            n0 = apply_M(Dp.conj().T, p1, fa); n1 = apply_M(Dp, p0, fa)
            psi = jnp.stack([n0, n1], axis=c)
        return psi

    def apply_cost(psi, cn, g):
        return psi * jnp.exp(-1j * g * cn).astype(jnp.complex64)

    def state1(x, psi, cn):
        L = x.size // NP
        x = x.reshape(L, NP)
        for l in range(L):
            row = x[l]
            bre, bim, th, ph = row[0], row[1], row[2], row[3]
            rest = row[4:]
            g = rest[-1]
            psi = apply_ecd(psi, bre, bim, th, ph)
            form = a.form
            if form == "snap_cost":
                s_coeffs = rest[:SNAP]
                psi = apply_snap_diag(psi, s_coeffs)
                psi = apply_cost(psi, cn, g)
            elif form == "d_snap":
                are, aim = rest[0], rest[1]
                s_coeffs = rest[2:2 + SNAP]
                psi = apply_disp_all(psi, are, aim)
                psi = apply_snap_diag(psi, s_coeffs)
                psi = apply_cost(psi, cn, g)
            elif form == "snap_d":
                are, aim = rest[0], rest[1]
                s_coeffs = rest[2:2 + SNAP]
                psi = apply_snap_diag(psi, s_coeffs)
                psi = apply_disp_all(psi, are, aim)
                psi = apply_cost(psi, cn, g)
            elif form == "d_snap_d":
                a1r, a1i, a2r, a2i = rest[0], rest[1], rest[2], rest[3]
                s_coeffs = rest[4:4 + SNAP]
                psi = apply_disp_all(psi, a1r, a1i)
                psi = apply_snap_diag(psi, s_coeffs)
                psi = apply_disp_all(psi, a2r, a2i)
                psi = apply_cost(psi, cn, g)
            elif form == "gen_sum":
                are, aim = rest[0], rest[1]
                s_coeffs = rest[2:2 + SNAP]
                psi = apply_gen_sum_fast(psi, are, aim, s_coeffs)
                psi = apply_cost(psi, cn, g)
            elif form == "bs_snap":
                theta = rest[0]
                s_coeffs = rest[1:1 + SNAP]
                psi = apply_snap_diag(psi, s_coeffs)
                psi = apply_bs(psi, theta)
                psi = apply_cost(psi, cn, g)
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
log(f"build n=8 form={a.form} snap={SNAP} layout={a.layout} NP={NP}")
tr_p, tr_vg = build(8, tr_inst, LAY[8])


def f(x):
    v, g = tr_vg(jnp.asarray(x, jnp.float32))
    return float(v), np.asarray(g, float)

rng = np.random.default_rng(a.seed)
res = {"args": vars(a), "train": {}, "test": {}, "np": NP}
prev = None


def init_layer(scale=1.0):
    ecd = [0.2 * scale * rng.standard_normal(), 0.2 * scale * rng.standard_normal(),
           rng.uniform(0, np.pi), rng.uniform(0, 2 * np.pi)]
    if a.form in ("d_snap", "snap_d", "gen_sum"):
        form = [0.3 * scale * rng.standard_normal(), 0.3 * scale * rng.standard_normal()]
        form += list(0.15 * rng.standard_normal(SNAP))
    elif a.form == "d_snap_d":
        form = list(0.25 * scale * rng.standard_normal(4))
        form += list(0.15 * rng.standard_normal(SNAP))
    elif a.form == "snap_cost":
        form = list(0.15 * rng.standard_normal(SNAP))
    elif a.form == "bs_snap":
        form = [0.4 * scale * rng.standard_normal()]
        form += list(0.15 * rng.standard_normal(SNAP))
    else:
        form = []
    return np.array(ecd + form + [0.4 * scale], float)

for L in a.Ls:
    t0 = time.time(); cands = []
    if prev is not None:
        Lp = prev.size // NP
        cands.append(np.concatenate([prev, 0.1 * np.concatenate([init_layer() for _ in range(L - Lp)])]))
        xp = prev.reshape(Lp, NP)
        t0s, t1s = np.linspace(0, 1, Lp), np.linspace(0, 1, L)
        xi = np.stack([np.interp(t1s, t0s, xp[:, j]) if Lp > 1 else np.repeat(xp[0, j], L) for j in range(NP)], 1)
        cands.append(xi.reshape(-1))
    for s in (0.5, 1.0, 1.5, 2.0):
        cands.append(np.concatenate([init_layer(s) for _ in range(L)]))
    # schedule: growing |α|, growing gamma
    for sc in (0.8, 1.3):
        rows = []
        for i in range(L):
            t = i / max(L - 1, 1)
            row = [0.1 * sc, 0.1 * sc, 0.4, 0.0]
            if a.form in ("d_snap", "snap_d", "gen_sum"):
                row += [(0.4 + 0.6 * t) * sc, 0.1 * sc] + [0.05] * SNAP
            elif a.form == "d_snap_d":
                row += [(0.5 * sc), 0.0, (-0.5 * sc), 0.0] + [0.1] * SNAP
            elif a.form == "snap_cost":
                row += [0.1] * SNAP
            elif a.form == "bs_snap":
                row += [0.5 * sc] + [0.05] * SNAP
            row += [(0.2 + 0.7 * t) * sc]
            rows.append(row)
        cands.append(np.array(rows, float).reshape(-1))
    cands = cands[: a.restarts + 2]
    best = None
    for c0 in cands:
        r = minimize(f, c0, jac=True, method="L-BFGS-B", options={"maxiter": a.maxiter})
        fun = float(r.fun) if np.isfinite(r.fun) else 1e9
        log(f"  L={L} try loss={fun:.4f}")
        if best is None or fun < best[1]:
            best = (r.x, fun)
    prev = best[0]
    p = np.asarray(tr_p(jnp.asarray(prev, jnp.float32)))
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
