"""Theory-first joint cavity unitary (hardware model) — broad scan.

HARDWARE layer (Zach): EACH layer = local ECD on each cavity–transmon (keep it;
cheap, improves algorithm) + ONE arbitrary unitary on the joint cavity space
(via GRAPE later). Mixer + cost folded into that single U. No JP.
Bits may sit on cavities and/or control transmons.

Theory screens (--u_form / --u_order / --code / --mixer):
  product   : U = GrayX · cost   or   cost · GrayX   (--u_order)
  single_exp: U = exp(-i (a X + b C))  one generator (Trotter-free)
  gx_only   : U = GrayX only (no cost)     ablation
  cost_only : U = cost only (no GrayX)     ablation
  --code gray|binary : X on Gray- or binary-encoded cavity bits
  --mixer local|joint: per-cavity X product vs X on concatenated cavity bits

Params/layer with ecd=1 (default): [bre, bim, th, ph, b_cav, g] (+ diag extras)
  ecd=0: [b_tm, b_cav, g] (+ extras)  — kept for ablations only
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
ap.add_argument("--ecd", type=int, default=1, choices=[0, 1])
ap.add_argument("--diag", default="raw", choices=["raw", "poly2", "soft", "step"])
ap.add_argument("--u_form", default="product",
                choices=["product", "single_exp", "gx_only", "cost_only"])
ap.add_argument("--u_order", default="gx_cost", choices=["gx_cost", "cost_gx"])
ap.add_argument("--code", default="gray", choices=["gray", "binary"])
ap.add_argument("--mixer", default="local", choices=["local", "joint"])
ap.add_argument("--snap", type=int, default=0, help="size-free per-cavity SNAP poly degree")
ap.add_argument("--Ls", type=int, nargs="+", default=[2, 4, 8])
ap.add_argument("--ntrain", type=int, default=80)
ap.add_argument("--nt2", type=int, nargs="*", default=[])
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

BASE = (6 if a.ecd else 3)
EXTRA = {"raw": 0, "poly2": 1, "soft": 1, "step": 1}[a.diag]
SNAP = int(a.snap)
# single_exp / product / gx use b_cav; cost_only still has a slot but unused
# param/layer: BASE + SNAP coeffs + EXTRA diag
NP = BASE + SNAP + EXTRA


def ungray_scalar(g):
    g = int(g); s = g >> 1
    while s:
        g ^= s; s >>= 1
    return g


def gray_scalar(i):
    return i ^ (i >> 1)


def bit_X_sum(k, nf, code="gray"):
    """Sum_j X_j on k encoded bits as nf×nf matrix (logical block only)."""
    N = 1 << k
    M = np.zeros((nf, nf), complex)
    for n in range(N):
        if code == "gray":
            bits = gray_scalar(n)
            for j in range(k):
                n2 = ungray_scalar(bits ^ (1 << j))
                M[n2, n] += 1.0
        else:
            for j in range(k):
                n2 = n ^ (1 << j)
                M[n2, n] += 1.0
    return M


def joint_bit_X(ks, nfs, code="gray"):
    """X on concatenated cavity bits embedded in product Fock space.
    bits = (b0 << k1+k2+...) | (b1 << ...) | ... ; each bi decoded to Fock via code.
    Returns big matrix on ⊗_c C^{nf_c} — only built when dim is small; else None.
    Prefer apply via tensor contractions in build().
    """
    # Handled inside build via multi-index flips; this stub unused.
    return None


def build(n, insts, ks):
    C = len(ks)
    assert sum(ks) + (C if tbit else 0) == n
    nfs = [(1 << k) + max(8, (1 << k) // 4) for k in ks]
    # Per-cavity X eigendecomps (local mixer / product form)
    Vs, lams = [], []
    for k, nf in zip(ks, nfs):
        X = bit_X_sum(k, nf, a.code)
        lam, V = np.linalg.eigh(X)
        Vs.append(jnp.asarray(V.astype(np.complex64)))
        lams.append(jnp.asarray(lam.astype(np.float32)))

    # Joint mixer: precompute X on concatenated logical cavity bits as sparse flips
    # Map: product Fock (n0,...,n_{C-1}) with ni < 2^{k_i} <-> concatenated bitstring
    ktot = int(sum(ks))
    # For joint: list of (src_multi, dst_multi) pairs per bit flip — applied as
    # scatter on computational Fock block. We implement via logical 2^{ktot} matrix
    # acting through an embedding index.
    joint_V = joint_lam = None
    if a.mixer == "joint" and a.u_form != "cost_only":
        # Build X on ktot bits, embed into product of 2^{k_c} blocks (pad levels = 0)
        Nlog = 1 << ktot
        Xj = np.zeros((Nlog, Nlog), complex)
        for bits in range(Nlog):
            for j in range(ktot):
                Xj[bits ^ (1 << j), bits] += 1.0
        # Map logical concatenated bits -> product Fock multi-index
        # packing: high bits = cavity 0
        strides = []
        rem = ktot
        for k in ks:
            rem -= k
            strides.append(rem)
        def bits_to_focks(bits):
            out = []
            for k, sh in zip(ks, strides):
                b = (bits >> sh) & ((1 << k) - 1)
                if a.code == "gray":
                    out.append(ungray_scalar(b))
                else:
                    out.append(b)
            return tuple(out)
        # Embed Xj into full product nf space via logical indices only
        # We'll apply joint U in the logical cavity subspace using a dense Nlog matrix
        # reconstructed each call from eigendecomp of Xj (cost added separately / single_exp).
        jlam, jV = np.linalg.eigh(Xj.real)  # Xj real symmetric
        joint_V = jnp.asarray(jV.astype(np.complex64))
        joint_lam = jnp.asarray(jlam.astype(np.float32))
        # flat index within product of exact 2^k blocks (no pad)
        log_nfs = [1 << k for k in ks]
        log_shape = log_nfs
        # map bits -> ravel index in log_shape
        bit_to_logflat = np.zeros(Nlog, np.int32)
        for bits in range(Nlog):
            fo = bits_to_focks(bits)
            bit_to_logflat[bits] = np.ravel_multi_index(fo, log_shape)
        # inverse: logflat -> bits (via focks -> gray/binary bits)
        logflat_to_bits = np.zeros(Nlog, np.int32)
        for bits in range(Nlog):
            logflat_to_bits[bit_to_logflat[bits]] = bits
        bit_to_logflat_j = jnp.asarray(bit_to_logflat)
        logflat_to_bits_j = jnp.asarray(logflat_to_bits)

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
        fidx.append(ug(f) if a.code == "gray" else f)
    shape = [2] * C + nfs
    D = int(np.prod(shape))
    flat = np.ravel_multi_index(tuple(tidx + fidx), shape)
    B = len(insts)
    Cn = np.zeros((B, D), np.float32); succ = np.zeros((B, D), np.float32)
    # Logical cost on 2^n for single_exp
    Cn_log = np.zeros((B, 1 << n), np.float32)
    for i, ins in enumerate(insts):
        E = ins["E"].astype(float); c = E * n / ins["m"]
        row = np.full(D, c.max()); row[flat] = c; Cn[i] = row
        np.add.at(succ[i], flat, (E == 0).astype(float))
        Cn_log[i] = c.astype(np.float32)
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
    Cn_log_j = jnp.asarray(Cn_log)
    flat_j = jnp.asarray(flat)

    # Logical X_sum on n bits (for single_exp): standard bit-flip X on each var
    Nlog_n = 1 << n
    if a.u_form == "single_exp":
        Xs = np.zeros((Nlog_n, Nlog_n), np.float32)
        for bits in range(Nlog_n):
            for j in range(n):
                Xs[bits ^ (1 << j), bits] += 1.0
        Xs_j = jnp.asarray(Xs)

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
            return g * jnp.tanh(extra * cn)
        return g * jax.nn.sigmoid(8.0 * (cn - extra))

    def apply_local_gx(psi, bc):
        for c in range(C):
            psi = apply_M(cav_U(c, bc), psi, C + c)
        return psi

    def apply_joint_gx(psi, bc):
        # Extract logical cavity block (Fock < 2^k each), apply U=exp(-i bc/2 X) in bit basis, write back
        # psi shape: [2]*C + nfs
        # Take slices [:2^k] on each cavity axis
        sl = [slice(None)] * C + [slice(0, 1 << k) for k in ks]
        block = psi[tuple(sl)]  # [2]*C + log_nfs
        # Flatten cavity dims, keep transmons
        tshape = [2] * C
        cav_dim = 1 << ktot
        block_f = block.reshape(tshape + [cav_dim])  # but ravel order must match log_shape
        # Reorder: current ravel of log_nfs is cavity0 slowest? reshape([2]*C + log_nfs) -> flat cav
        # log flat index = ravel(focks). Convert to bits, apply U in bits basis, convert back.
        # U_bits = V exp(-i bc/2 lam) V^H
        U_bits = ((joint_V * jnp.exp(-1j * (bc / 2) * joint_lam)[None, :]) @ joint_V.conj().T).astype(jnp.complex64)
        # Map logflat -> bits: amp_bits[b] = amp_logflat[bit_to_logflat[b]]
        # block_f last axis is logflat in C-order of log_nfs = focks
        def to_bits(amp_log):
            return amp_log[bit_to_logflat_j]
        def from_bits(amp_bits):
            out = jnp.zeros_like(amp_bits)
            return out.at[bit_to_logflat_j].set(amp_bits)
        # apply over all transmon configs
        amp = block.reshape(-1, cav_dim)
        amp_b = amp[:, bit_to_logflat_j]
        amp_b2 = amp_b @ U_bits.T  # rows are states; U acts on column space: amp @ U.T if amp row vectors
        # state' = U @ state, row vectors: state' = state @ U.T
        amp2 = jnp.zeros_like(amp)
        amp2 = amp2.at[:, bit_to_logflat_j].set(amp_b2)
        new_block = amp2.reshape(tshape + [1 << k for k in ks])
        # write into psi
        psi = psi.at[tuple(sl)].set(new_block)
        return psi

    def apply_gx(psi, bc):
        if a.mixer == "joint" and joint_V is not None:
            return apply_joint_gx(psi, bc)
        return apply_local_gx(psi, bc)

    def apply_cost(psi, cn, g, extra):
        return psi * jnp.exp(-1j * phase_of(cn, g, extra)).astype(jnp.complex64)

    def apply_single_exp(psi, cn_log, bc, g):
        # U = exp(-i (bc * X_sum + g * C)) on logical 2^n amplitudes at flat indices
        H = bc * Xs_j + g * jnp.diag(cn_log)
        # jnp.linalg.eigh for Hermitian
        evals, evecs = jnp.linalg.eigh(H)
        U = (evecs * jnp.exp(-1j * evals)[None, :]) @ evecs.conj().T
        amp = psi.reshape(-1)
        a_log = amp[flat_j]
        a2 = U @ a_log
        amp = amp.at[flat_j].set(a2.astype(jnp.complex64))
        return amp.reshape(psi.shape)

    def apply_snap(psi, s_coeffs):
        """Size-free per-cavity SNAP: phi(n)=sum_j s_j (n/sqrt(2^k))^{j+1}, part of joint U."""
        if SNAP <= 0:
            return psi
        for c in range(C):
            u = nvs[c] / scales[c]
            ph = sum(s_coeffs[j] * (u ** (j + 1)) for j in range(SNAP))
            Dph = jnp.exp(-1j * ph).astype(jnp.complex64)
            # apply diagonal on cavity axis C+c
            psi = psi * Dph.reshape([1] * (C + c) + [-1] + [1] * (len(nfs) - 1 - c))
        return psi

    def apply_ecd_or_tm(psi, row):
        if a.ecd:
            bre, bim, th, ph = row[0], row[1], row[2], row[3]
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
            bc, g = row[4], row[5]
            s_coeffs = row[6:6 + SNAP] if SNAP else jnp.zeros((0,))
            extra = row[6 + SNAP] if EXTRA else 0.0
        else:
            bt, bc, g = row[0], row[1], row[2]
            s_coeffs = row[3:3 + SNAP] if SNAP else jnp.zeros((0,))
            extra = row[3 + SNAP] if EXTRA else 0.0
            Mtm = jnp.array([[jnp.cos(bt / 2), -1j * jnp.sin(bt / 2)],
                             [-1j * jnp.sin(bt / 2), jnp.cos(bt / 2)]]).astype(jnp.complex64)
            for c in range(C):
                psi = apply_M(Mtm, psi, c)
        return psi, bc, g, extra, s_coeffs

    def state1(x, psi, cn, cn_log):
        L = x.size // NP
        x = x.reshape(L, NP)
        for l in range(L):
            psi, bc, g, extra, s_coeffs = apply_ecd_or_tm(psi, x[l])
            # Joint cavity U (theory): SNAP-per-cavity + mixer + cost folded into one U
            psi = apply_snap(psi, s_coeffs)
            form = a.u_form
            if form == "cost_only":
                psi = apply_cost(psi, cn, g, extra)
            elif form == "gx_only":
                psi = apply_gx(psi, bc)
            elif form == "single_exp":
                # one exponential on logical space; bc multiplies X, g multiplies C
                psi = apply_single_exp(psi, cn_log, bc, g)
            else:  # product: one composed U = gx·cost or cost·gx
                if a.u_order == "gx_cost":
                    psi = apply_gx(psi, bc)
                    psi = apply_cost(psi, cn, g, extra)
                else:
                    psi = apply_cost(psi, cn, g, extra)
                    psi = apply_gx(psi, bc)
        return psi

    def probs(x):
        def one(args):
            cn, sc, cn_log = args
            return jnp.sum((jnp.abs(state1(x, psi0_j, cn, cn_log)) ** 2) * sc)
        return jax.lax.map(one, (Cn_j, succ_j, Cn_log_j))

    pjit = jax.jit(probs)
    vg = jax.jit(jax.value_and_grad(lambda x: -jnp.mean(jnp.log(probs(x) + 1e-12))))
    return pjit, vg


# ---- train / test ----
# single_exp: skip n>=14 in default test if present (memory); caller can override
if a.u_form == "single_exp":
    a.test = [t for t in a.test if t <= 12]
    log(f"single_exp: restricting test to {a.test} (logical eigh memory)")

tr_inst = sim.instances(8, a.ntrain, TRB)
builders = [(8, tr_inst, LAY[8])]
for m in a.nt2:
    builders.append((m, sim.instances(m, max(40, a.ntrain // 2), TRB), LAY[m]))

tr_p, tr_vg = [], []
for n, insts, ks in builders:
    log(f"build train n={n} form={a.u_form} order={a.u_order} code={a.code} mixer={a.mixer} ecd={a.ecd} snap={SNAP}")
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
    if SNAP:
        v = np.concatenate([v, 0.1 * rng.standard_normal(SNAP)])
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
    if not a.ecd:
        sched = np.stack([np.linspace(0.9, 0.2, L), np.linspace(0.9, 0.2, L), np.linspace(0.2, 0.9, L)], 1)
        if EXTRA:
            extra0 = {"poly2": 0.0, "soft": 1.0, "step": 0.5}[a.diag]
            sched = np.concatenate([sched, np.full((L, 1), extra0)], 1)
        cands.append(sched.reshape(-1))
        cands.append(sched.reshape(-1) * 1.4)
    else:
        # ECD schedule: small beta, QAOA-like bcav/gamma (+ zero SNAP)
        for sc in (0.7, 1.2):
            rows = []
            for i in range(L):
                tt = i / max(L - 1, 1)
                row = [0.05 * sc, 0.05 * sc, 0.3, 0.0,
                       (0.9 - 0.7 * tt) * sc, (0.2 + 0.7 * tt) * sc]
                row += [0.0] * SNAP
                if EXTRA:
                    row += [{"poly2": 0.0, "soft": 1.0, "step": 0.5}[a.diag]]
                rows.append(row)
            cands.append(np.array(rows, float).reshape(-1))
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
    if a.u_form == "single_exp":
        ch = min(ch, 8 if n <= 10 else 2)
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
