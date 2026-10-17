"""Neural-net parameter predictor for ECD + cost·GrayX (costgx physics).

MORE DIFFERENT from QAOA by design:
  - no QAOA init; start with |beta| ~ beta0 so ECD does real work
  - soft floor on |ECD beta| (penalty if |beta| < beta_min)
  - optional richer diagonal: phase = g0*Cn + g1*Cn^2 (+ g2*Cn^3)
  - optional per-cavity GrayX angles (one bc per register instead of shared)
  - instance-conditioned MLP (clause/E features + n) -> per-instance params
  - curriculum: train at {8}, test 10; add 10, test 12; ... up to 14; eval 20

Modes (--mode):
  agnostic : MLP([n]) -> shared params for that size (size extrapolation)
  cond     : MLP(features) -> per-instance params (base + delta*scale)
  shared   : single shared param vector (ablation; like costgx but with beta floor)

Usage examples:
  python3 -u nn_costgx.py --tag NN_c4 --mode cond --Ls 4 --stages 8 --test 10 12 14 20
  python3 -u nn_costgx.py --tag NN_c48 --mode cond --Ls 4 8 --stages 8 10 12 14 --rich 2
"""
from __future__ import annotations
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, jax, jax.numpy as jnp
from scipy.optimize import minimize
import sim

ap = argparse.ArgumentParser()
ap.add_argument("--tag", required=True)
ap.add_argument("--mode", default="cond", choices=["cond", "agnostic", "shared", "costcond"])
ap.add_argument("--layout", default="T", choices=["T", "C", "R"])
ap.add_argument("--code", default="gray", choices=["gray", "binary"])
ap.add_argument("--Ls", type=int, nargs="+", default=[4, 8])
ap.add_argument("--stages", type=int, nargs="+", default=[8, 10, 12, 14],
                help="curriculum sizes to successively add to the training pool")
ap.add_argument("--ntrain", type=int, default=80)
ap.add_argument("--nt2_count", type=int, default=40)
ap.add_argument("--test", type=int, nargs="*", default=[10, 12, 14, 20])
ap.add_argument("--val_count", type=int, default=40)
ap.add_argument("--hidden", type=int, default=64)
ap.add_argument("--depth", type=int, default=2)
ap.add_argument("--delta_scale", type=float, default=0.3,
                help="cond: network outputs delta * scale added to shared base")
ap.add_argument("--rich", type=int, default=1, help="cost poly degree (1=g*Cn, 2=+g1*Cn^2, ...)")
ap.add_argument("--per_cav", action="store_true", help="per-cavity GrayX angle (pad to max C)")
ap.add_argument("--per_bit", action="store_true", help="per-Gray-digit X angle (pad to 6 bits/cav)")
ap.add_argument("--final_ecd", action="store_true", default=True)
ap.add_argument("--no_final_ecd", action="store_true")
ap.add_argument("--beta0", type=float, default=0.3, help="init |ECD beta| and L2 pull target")
ap.add_argument("--beta_min", type=float, default=0.1, help="soft floor on |beta| (also L2 to beta0)")
ap.add_argument("--beta_w", type=float, default=1.0, help="weight of ECD-beta regulariser")
ap.add_argument("--maxiter", type=int, default=250)
ap.add_argument("--restarts", type=int, default=2)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--eval_only", action="store_true")
ap.add_argument("--joint", action="store_true",
                help="train on all --stages sizes at once (no curriculum)")
ap.add_argument("--bmax", type=float, default=0.0,
                help="if >0, ECD beta = bmax*tanh(raw) (caps displacement)")
ap.add_argument("--fix_beta", action="store_true",
                help="force |ECD beta|=beta0 (r[0]=phase only); stops collapse to 0")
ap.add_argument("--phys_beta", action="store_true",
                help="interpret beta as physical displacement α; undo per-cavity sqrt(2^k) scale")
ap.add_argument("--ebins", type=int, default=0,
                help="if >0, free diagonal cost with ebins phases on round(Cn) (overrides --rich)")
a = ap.parse_args()
if a.no_final_ecd: a.final_ecd = False
if a.ebins > 0:
    a.rich = a.ebins  # NG = ebins; phase_from switches below

LAY = {"T": sim.LAYOUTS_T, "C": sim.LAYOUTS_C, "R": sim.LAYOUTS_R}[a.layout]
tbit = a.layout in ("T", "R")
log = lambda *s: print(time.strftime("%H:%M:%S"), *s, flush=True)
TRB, TEB, VAB = 31_000_000_000, 32_000_000_000, 33_000_000_000
OUT = f"results/{a.tag}.json"
MAX_C = max(len(LAY[n]) for n in LAY)  # for per_cav padding

# ---- feature extraction (size-independent-ish) ----
NF_FEAT = 16  # fixed feature dim


def features_one(n, ins):
    """Clause/energy stats + n. Regenerates clauses from seed for degree info."""
    E = ins["E"].astype(float); m = ins["m"]; N = 1 << n
    em, es = float(E.mean()), float(E.std() + 1e-9)
    # energy histogram (coarse bins relative to mean)
    z = (E - em) / es
    hist = np.histogram(z, bins=[-np.inf, -1.5, -0.5, 0.5, 1.5, np.inf])[0] / N
    # clause degrees
    _, cl, _ = sim.gen_instance(n, ins["seed"])
    deg = np.zeros(n)
    for v, s in cl:
        for vv in v: deg[vv] += 1
    # sign pattern: fraction of clauses with k negative signs
    sgn = np.zeros(5)
    for v, s in cl:
        sgn[sum(1 for ss in s if ss < 0)] += 1
    sgn /= max(m, 1)
    f = np.array([
        n / 20.0,
        m / (9.93 * n),
        em / n,
        es / n,
        ins["nsol"] / N,
        deg.mean() / m,
        deg.std() / (m + 1e-9),
        deg.max() / m,
        *hist.tolist(),
        *sgn.tolist(),
    ], dtype=np.float32)
    # pad/truncate to NF_FEAT
    if f.size < NF_FEAT: f = np.concatenate([f, np.zeros(NF_FEAT - f.size, np.float32)])
    return f[:NF_FEAT]


def features_batch(n, insts):
    return np.stack([features_one(n, ins) for ins in insts]).astype(np.float32)


# ---- circuit param layout ----
# local block: bre, bim, th, ph  (+ optional final)
# U block: bc... (1 or MAX_C), g0..g_{rich-1}
NB = 4
MAX_BITS = 6
if a.per_bit:
    NC = MAX_BITS  # per Gray-digit X angle (shared across cavities; applied digit-wise)
elif a.per_cav:
    NC = MAX_C
else:
    NC = 1
NG = a.rich
USE_EBINS = a.ebins > 0
NP = NB + NC + NG   # per layer
NF = NB if a.final_ecd else 0


def n_params(L):
    return L * NP + NF


# ---- MLP ----
def mlp_sizes(out_dim):
    sizes = [NF_FEAT] + [a.hidden] * a.depth + [out_dim]
    return sizes


def pack_mlp(sizes, rng):
    th = []
    for din, dout in zip(sizes[:-1], sizes[1:]):
        W = rng.normal(0, np.sqrt(2.0 / din), (dout, din)).astype(np.float32)
        b = np.zeros(dout, np.float32)
        th.append(W); th.append(b)
    return th


def mlp_forward(theta, x):
    # x: (B, F) or (F,)
    h = x
    nL = len(theta) // 2
    for i in range(nL):
        W, b = theta[2 * i], theta[2 * i + 1]
        h = h @ W.T + b
        if i < nL - 1: h = jnp.tanh(h)
    return h


def flat_to_theta(v, sizes):
    out = []; i = 0
    for din, dout in zip(sizes[:-1], sizes[1:]):
        nW = dout * din; out.append(v[i:i + nW].reshape(dout, din)); i += nW
        out.append(v[i:i + dout]); i += dout
    return out, i


def theta_dim(sizes):
    return sum(dout * din + dout for din, dout in zip(sizes[:-1], sizes[1:]))


# ---- circuit (costgx physics + rich cost + optional per-cav GrayX) ----
def gray_scalar(i): return i ^ (i >> 1)
def ungray_scalar(g):
    g = int(g); s = g >> 1
    while s: g ^= s; s >>= 1
    return g


def bit_X_ops(k, nf):
    """Return list of X_j matrices (nf,nf) for Gray (or binary) digit j=0..k-1, and their sum."""
    N = 1 << k; ops = []
    for j in range(k):
        M = np.zeros((nf, nf))
        for n in range(N):
            n2 = ungray_scalar(gray_scalar(n) ^ (1 << j)) if a.code == "gray" else n ^ (1 << j)
            M[n2, n] += 1.0
        ops.append(M)
    return ops, sum(ops)


def build(n, insts, ks):
    C = len(ks); assert sum(ks) + (C if tbit else 0) == n
    nfs = [(1 << k) + max(8, (1 << k) // 4) for k in ks]
    Vs, lams, Vd, lamd, nvs, scales, Xops = [], [], [], [], [], [], []
    for k, nf in zip(ks, nfs):
        ops, S = bit_X_ops(k, nf)
        lam, V = np.linalg.eigh(S)
        Vs.append(jnp.asarray(V.astype(np.complex64))); lams.append(jnp.asarray(lam.astype(np.float32)))
        # per-bit: store each X_j's eigen for exact exp(-i b/2 X); X_j^2=I on computational, padded 0
        xeig = []
        for M in ops:
            lamj, Vj = np.linalg.eigh(M)
            xeig.append((jnp.asarray(Vj.astype(np.complex64)), jnp.asarray(lamj.astype(np.float32))))
        Xops.append(xeig)
        aa = np.diag(np.sqrt(np.arange(1, nf)), 1); lam, V = np.linalg.eigh(1j * (aa.T - aa))
        Vd.append(jnp.asarray(V.astype(np.complex64))); lamd.append(jnp.asarray(lam.astype(np.float32)))
        nvs.append(jnp.arange(nf)); scales.append(np.sqrt(2.0 ** k))
    x = np.arange(1 << n); pos = n; tidx, fidx = [], []
    for c, k in enumerate(ks):
        nb = k + (1 if tbit else 0); pos -= nb; val = (x >> pos) & ((1 << nb) - 1)
        if tbit: tidx.append(val >> k); f = val & ((1 << k) - 1)
        else: tidx.append(np.zeros_like(x)); f = val
        fidx.append(sim.ungray(f) if a.code == "gray" else f)
    shape = [2] * C + nfs; D = int(np.prod(shape))
    flat = np.ravel_multi_index(tuple(tidx + fidx), shape)
    B = len(insts); Cn = np.zeros((B, D), np.float32); succ = np.zeros((B, D), np.float32)
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
    if tbit: psi0[flat] = 1 / np.sqrt(len(flat))
    else:
        fl0 = np.unique(np.ravel_multi_index(tuple([np.zeros_like(x)] * C + fidx), shape))
        psi0[fl0] = 1 / np.sqrt(len(fl0))
    psi0 = jnp.asarray(psi0.reshape(shape)); Cn_j = jnp.asarray(Cn); succ_j = jnp.asarray(succ)
    feat = jnp.asarray(features_batch(n, insts))

    def disp(c, re, im):
        s = scales[c] / 2.0; r = jnp.sqrt(re ** 2 + im ** 2 + 1e-20) * s; ph = jnp.arctan2(im, re)
        W = jnp.exp(1j * ph * nvs[c])[:, None] * Vd[c]
        return ((W * jnp.exp(-1j * r * lamd[c])[None, :]) @ W.conj().T).astype(jnp.complex64)

    def cav_U(c, b):
        return ((Vs[c] * jnp.exp(-1j * (b / 2) * lams[c])[None, :]) @ Vs[c].conj().T).astype(jnp.complex64)

    def cav_U_bits(c, bs):
        # bs: (MAX_BITS,) angles; apply exp(-i bs[j]/2 X_j) for j < k_c
        U = jnp.eye(nfs[c], dtype=jnp.complex64)
        for j, (Vj, lamj) in enumerate(Xops[c]):
            bj = bs[j]
            Uj = ((Vj * jnp.exp(-1j * (bj / 2) * lamj)[None, :]) @ Vj.conj().T).astype(jnp.complex64)
            U = Uj @ U
        return U

    def apply_M(M, psi, ax):
        return jnp.moveaxis(jnp.tensordot(M, psi, axes=([1], [ax])), 0, ax)

    def Rm(th, ph):
        ct, st = jnp.cos(th / 2), jnp.sin(th / 2)
        return jnp.array([[ct, -1j * st * jnp.exp(-1j * ph)],
                          [-1j * st * jnp.exp(1j * ph), ct]]).astype(jnp.complex64)

    def local(psi, r):
        R = Rm(r[2], r[3])
        if a.fix_beta:
            # r[0] = phase of displacement; magnitude locked to beta0
            phi = r[0]
            bre = a.beta0 * jnp.cos(phi); bim = a.beta0 * jnp.sin(phi)
        else:
            bre, bim = r[0], r[1]
            if a.bmax > 0:
                bre = a.bmax * jnp.tanh(bre); bim = a.bmax * jnp.tanh(bim)
        for c in range(C):
            psi = apply_M(R, psi, c)
            if a.phys_beta:
                # cancel disp()'s *scales[c]/2 so (bre,bim) is physical α
                s = scales[c] / 2.0
                Dp = disp(c, bre / s, bim / s)
            else:
                Dp = disp(c, bre, bim)
            p0 = jnp.take(psi, 0, axis=c); p1 = jnp.take(psi, 1, axis=c); fa = C - 1 + c
            psi = jnp.stack([apply_M(Dp.conj().T, p1, fa), apply_M(Dp, p0, fa)], axis=c)
        return psi

    def phase_from(row, cn):
        # row[NB:NB+NC] = bc's; row[NB+NC:NB+NC+NG] = g's
        gs = row[NB + NC: NB + NC + NG]
        if USE_EBINS:
            # free diagonal: one phase per integer energy bin (Cn≈E*n/m ∈ [0,n])
            idx = jnp.clip(jnp.round(cn).astype(jnp.int32), 0, NG - 1)
            return gs[idx]
        ph = gs[0] * cn
        cnk = cn
        for j in range(1, NG):
            cnk = cnk * cn; ph = ph + gs[j] * cnk
        return ph

    def state1(xv, cn):
        """xv: (n_params,) for one instance."""
        L = (xv.size - NF) // NP; X = xv[: L * NP].reshape(L, NP); psi = psi0
        for l in range(L):
            psi = local(psi, X[l])
            psi = psi * jnp.exp(-1j * phase_from(X[l], cn)).astype(jnp.complex64)
            bcs = X[l, NB: NB + NC]
            for c in range(C):
                if a.per_bit:
                    psi = apply_M(cav_U_bits(c, bcs), psi, C + c)
                else:
                    bc = bcs[c] if a.per_cav else bcs[0]
                    psi = apply_M(cav_U(c, bc), psi, C + c)
        if NF: psi = local(psi, xv[L * NP:])
        return psi

    def probs_shared(xv):
        return jax.lax.map(lambda t: jnp.sum(jnp.abs(state1(xv, t[0])) ** 2 * t[1]), (Cn_j, succ_j))

    def probs_per(X):
        # X: (B, n_params)
        def one(args):
            xv, cn, sc = args
            return jnp.sum(jnp.abs(state1(xv, cn)) ** 2 * sc)
        return jax.lax.map(one, (X, Cn_j, succ_j))

    def beta_penalty(xv):
        # soft floor below beta_min + quadratic pull of |beta| toward beta0 (keeps ECD meaningful)
        if a.fix_beta:
            return jnp.asarray(0.0, jnp.float32)  # magnitude locked
        L = (xv.size - NF) // NP; X = xv[: L * NP].reshape(L, NP)
        bmag = jnp.sqrt(X[:, 0] ** 2 + X[:, 1] ** 2 + 1e-20)
        pen = jnp.sum(jnp.maximum(0.0, a.beta_min - bmag) ** 2) + jnp.sum((bmag - a.beta0) ** 2)
        if NF:
            t = xv[L * NP:]
            bmagf = jnp.sqrt(t[0] ** 2 + t[1] ** 2 + 1e-20)
            pen = pen + jnp.maximum(0.0, a.beta_min - bmagf) ** 2 + (bmagf - a.beta0) ** 2
        return pen

    return (jax.jit(probs_shared), jax.jit(probs_per), feat, jax.jit(beta_penalty), B)


# ---- init ----
rng = np.random.default_rng(a.seed)


def init_params(L, variant=0):
    """ECD at beta0; mixer/cost schedules. variant diversifies restarts.
    variant 0: equal per-bit angles (= sum-X / QAOA-topology mixer) + linear cost schedule
    variant 1: decaying per-bit spectrum (high digits weaker)
    variant 2: flipped transmon + mid cost
    No published QAOA angles are used.
    """
    rows = []
    for li, t in enumerate(np.linspace(0, 1, L)):
        ang = 0.7 + 1.3 * t + 0.4 * variant
        if a.fix_beta:
            bre, bim = ang, 0.0  # bre slot holds phase; mag locked to beta0
        else:
            bre = a.beta0 * np.cos(ang); bim = a.beta0 * np.sin(ang)
        if variant == 2:
            th = -np.pi + 0.3 * t; ph = 0.0   # near bit-flip on transmon (hardware-natural)
        else:
            th = 0.6 + 0.9 * t + 0.2 * variant; ph = 0.3 * t * (variant + 1)
        bc0 = (0.9 - 0.5 * t) * (0.7 + 0.2 * variant)
        if a.per_bit:
            if variant == 1:
                bcs = [bc0 * (0.85 ** j) for j in range(NC)]
            else:
                bcs = [bc0] * NC   # equal digits ≡ Σ X mixer
        else:
            bcs = [bc0] * NC
        g0 = 0.25 + 0.7 * t
        if USE_EBINS:
            # free energy-bin phases: linear ramp in bin index, scaled by schedule t
            gs = [g0 * (1.0 - k / max(NG - 1, 1)) + 0.02 * (variant - 1) * ((k % 3) - 1)
                  for k in range(NG)]
        else:
            gs = [g0] + ([0.05 * (1 - 2 * t) * (variant + 1)] * (NG - 1) if NG > 1 else [])
        rows.append([bre, bim, th, ph] + bcs + gs)
    x = np.array(rows, float).reshape(-1)
    if NF:
        if a.fix_beta:
            x = np.concatenate([x, [0.5 + 0.2 * variant, 0.0,
                                    (-np.pi + 0.2) if variant == 2 else 0.5, 0.0]])
        else:
            x = np.concatenate([x, [a.beta0 * 0.5, a.beta0 * 0.5,
                                    (-np.pi + 0.2) if variant == 2 else 0.5, 0.0]])
    return x.astype(np.float32)


# ---- training state packing ----
# shared / agnostic: flat params (or MLP weights that map n -> params)
# cond: base params + MLP weights (features -> delta)


def make_pack(L):
    P = n_params(L)
    if a.mode == "shared":
        return {"kind": "shared", "P": P, "sizes": None}
    if a.mode == "agnostic":
        sizes = [1] + [a.hidden] * a.depth + [P]
        return {"kind": "agnostic", "P": P, "sizes": sizes}
    if a.mode == "costcond":
        # MLP outputs only the cost gammas per layer: L * NG
        Cout = L * NG
        sizes = [NF_FEAT] + [a.hidden] * a.depth + [Cout]
        return {"kind": "costcond", "P": P, "sizes": sizes, "Cout": Cout}
    sizes = mlp_sizes(P)
    return {"kind": "cond", "P": P, "sizes": sizes}


def unpack(v, pack):
    if pack["kind"] == "shared":
        return {"base": v[: pack["P"]], "theta": None}, pack["P"]
    sizes = pack["sizes"]
    if pack["kind"] == "agnostic":
        th, i = flat_to_theta(v, sizes)
        return {"base": None, "theta": th}, i
    # cond / costcond: base + mlp
    base = v[: pack["P"]]; th, j = flat_to_theta(v[pack["P"]:], sizes)
    return {"base": base, "theta": th}, pack["P"] + j


def pack_dim(pack):
    if pack["kind"] == "shared": return pack["P"]
    td = theta_dim(pack["sizes"])
    return td if pack["kind"] == "agnostic" else pack["P"] + td


def init_vec(L, pack):
    P = pack["P"]; pieces = []
    if pack["kind"] != "agnostic":
        pieces.append(init_params(L, variant=0))
    if pack["kind"] != "shared":
        th0 = pack_mlp(pack["sizes"], rng)
        flat = np.concatenate([t.ravel() for t in th0])
        # shrink output layer so initial deltas are small
        if pack["kind"] in ("cond", "costcond"):
            # last W, b
            sizes = pack["sizes"]; dout, din = sizes[-1], sizes[-2]
            # rescale last W
            n_before = theta_dim(sizes) - dout * din - dout
            flat[n_before: n_before + dout * din] *= 0.05
            flat[n_before + dout * din:] = 0.0
        else:
            # agnostic: init near init_params via bias of last layer
            sizes = pack["sizes"]; dout = sizes[-1]; din = sizes[-2]
            n_before = theta_dim(sizes) - dout * din - dout
            flat[n_before: n_before + dout * din] *= 0.01
            flat[n_before + dout * din:] = init_params(L)
        pieces.append(flat.astype(np.float32))
    return np.concatenate(pieces).astype(np.float32)


def params_for_batch(state, feat, n, pack):
    """Return (B, P) or (P,) broadcastable params."""
    if pack["kind"] == "shared":
        return state["base"]
    if pack["kind"] == "agnostic":
        # feat not used; n scalar
        x = jnp.array([[n / 20.0]])
        return mlp_forward(state["theta"], x)[0]
    # cond
    delta = mlp_forward(state["theta"], feat)
    return state["base"][None, :] + a.delta_scale * delta


# ---- build training pools ----
def load_pool(sizes):
    pools = []
    for n in sizes:
        N = a.ntrain if n == sizes[0] else a.nt2_count
        # first size in curriculum gets ntrain, later stages get nt2_count each — but when
        # rebuilding full pool use ntrain for n=8 and nt2_count for others
        N = a.ntrain if n == 8 else a.nt2_count
        insts = sim.instances(n, N, TRB)
        log(f"build n={n} N={len(insts)} mode={a.mode} rich={a.rich} per_cav={a.per_cav} per_bit={a.per_bit} beta0={a.beta0}")
        ps, pp, feat, bpen, B = build(n, insts, LAY[n])
        pools.append(dict(n=n, insts=insts, ps=ps, pp=pp, feat=feat, bpen=bpen, B=B))
    return pools


def make_loss(pools, pack, L):
    def loss_and_grad(v):
        st, _ = unpack(jnp.asarray(v, jnp.float32), pack)
        total = 0.0; gacc = None; nterm = 0
        # compute via python loop over sizes; each size's vg via jax
        # We need a single flat loss. Build a jax function per pool.
        return None  # placeholder — filled below
    return loss_and_grad


# Build jax value_and_grad closures per pool
def vg_for_pool(pool, pack, L):
    P = pack["P"]; kind = pack["kind"]

    if kind == "shared":
        def loss(v):
            xv = v[:P]
            nl = -jnp.log(pool["ps"](xv) + 1e-12)
            return jnp.mean(nl) + a.beta_w * pool["bpen"](xv)
        return jax.jit(jax.value_and_grad(loss))

    if kind == "agnostic":
        sizes = pack["sizes"]
        def loss(v):
            th, _ = flat_to_theta(v, sizes)
            xv = mlp_forward(th, jnp.array([[pool["n"] / 20.0]]))[0]
            nl = -jnp.log(pool["ps"](xv) + 1e-12)
            return jnp.mean(nl) + a.beta_w * pool["bpen"](xv)
        return jax.jit(jax.value_and_grad(loss))

    sizes = pack["sizes"]; feat = pool["feat"]
    if pack["kind"] == "costcond":
        Cout = pack["Cout"]; L_ = Cout // NG
        def loss(v):
            base = v[:P]; th, _ = flat_to_theta(v[P:], sizes)
            gs = a.delta_scale * mlp_forward(th, feat)  # (B, L*NG)
            # scatter into full param vectors
            X = jnp.tile(base[None, :], (feat.shape[0], 1))
            def inject(xv_g):
                xv, g = xv_g
                # replace cost slots in each layer
                rows = []
                for l in range(L_):
                    row = xv[l*NP:(l+1)*NP]
                    # cost coeffs sit at indices NB+NC : NB+NC+NG
                    g_l = g[l*NG:(l+1)*NG]
                    row = row.at[NB+NC:NB+NC+NG].set(row[NB+NC:NB+NC+NG] + g_l)
                    rows.append(row)
                out = jnp.concatenate(rows)
                if NF: out = jnp.concatenate([out, xv[L_*NP:]])
                return out
            X = jax.lax.map(inject, (X, gs))
            nl = -jnp.log(pool["pp"](X) + 1e-12)
            return jnp.mean(nl) + a.beta_w * pool["bpen"](base)
        return jax.jit(jax.value_and_grad(loss))

    # cond
    def loss(v):
        base = v[:P]; th, _ = flat_to_theta(v[P:], sizes)
        X = base[None, :] + a.delta_scale * mlp_forward(th, feat)
        nl = -jnp.log(pool["pp"](X) + 1e-12)
        pen = pool["bpen"](base)
        pen = pen + jnp.mean(jax.lax.map(pool["bpen"], X))
        return jnp.mean(nl) + a.beta_w * pen
    return jax.jit(jax.value_and_grad(loss))


def combine_vg(vgs):
    def f(v):
        vs = [vg(jnp.asarray(v, jnp.float32)) for vg in vgs]
        val = sum(float(v[0]) for v in vs) / len(vs)
        g = sum(np.asarray(v[1], float) for v in vs) / len(vs)
        return val, g
    return f


# ---- eval ----
def eval_params_fn(state, pack, n, insts, chunk=None):
    """Return list of success probs on insts."""
    if chunk is None:
        chunk = 20 if n < 14 else (10 if n < 20 else 4)
    out = []
    for i in range(0, len(insts), chunk):
        batch = insts[i:i + chunk]
        ps, pp, feat, bpen, B = build(n, batch, LAY[n])
        if pack["kind"] == "shared":
            p = np.asarray(ps(jnp.asarray(state["base"], jnp.float32)))
        elif pack["kind"] == "agnostic":
            xv = mlp_forward(state["theta"], jnp.array([[n / 20.0]], jnp.float32))[0]
            p = np.asarray(ps(xv))
        elif pack["kind"] == "costcond":
            base = state["base"]; gs = a.delta_scale * mlp_forward(state["theta"], feat)
            B = feat.shape[0]; X = np.tile(np.asarray(base), (B, 1))
            Cout = gs.shape[-1]; L_ = Cout // NG
            for bi in range(B):
                for l in range(L_):
                    sl = slice(l*NP + NB + NC, l*NP + NB + NC + NG)
                    X[bi, sl] = X[bi, sl] + np.asarray(gs[bi, l*NG:(l+1)*NG])
            p = np.asarray(pp(jnp.asarray(X, jnp.float32)))
        else:
            X = state["base"][None, :] + a.delta_scale * mlp_forward(state["theta"], feat)
            p = np.asarray(pp(X))
        out += list(p)
    return out


def beta_mags(xv, L):
    if a.fix_beta:
        return [float(a.beta0)] * L
    X = xv[: L * NP].reshape(L, NP)
    m = np.sqrt(X[:, 0] ** 2 + X[:, 1] ** 2)
    return m.tolist()


# ---- main ----
res = json.load(open(OUT)) if os.path.exists(OUT) else {
    "args": vars(a), "train": {}, "test": {}, "curriculum": [], "np": NP, "nf": NF, "ng": NG, "nc": NC
}
res["args_last"] = vars(a)

# Curriculum: after stage k, training pool = stages[:k+1]
stage_list = [a.stages] if a.joint else [a.stages[:i+1] for i in range(len(a.stages))]
for si, train_sizes in enumerate(stage_list):
    stage_key = "+".join(str(s) for s in train_sizes)
    log(f"=== curriculum stage {stage_key} ===")
    pools = load_pool(train_sizes)

    for L in a.Ls:
        key = f"{stage_key}|L{L}"
        if key in res["train"] and not a.eval_only:
            log(f"skip trained {key}"); continue
        pack = make_pack(L)
        vgs = [vg_for_pool(p, pack, L) for p in pools]
        f = combine_vg(vgs)
        # warm-start from previous stage same L if available
        cands = []
        # previous stage warm-start
        if si > 0:
            pk = "+".join(str(s) for s in a.stages[:si]) + f"|L{L}"
            if pk in res["train"]:
                cands.append(np.array(res["train"][pk]["v"], dtype=np.float32))
        # same stage smaller L grow (shared mode only)
        smaller = [ll for ll in a.Ls if ll < L]
        if smaller:
            sk = stage_key + f"|L{smaller[-1]}"
            if sk in res["train"] and pack["kind"] == "shared":
                prev = np.array(res["train"][sk]["v"]); Lp = (prev.size - NF) // NP
                xp = prev[: Lp * NP].reshape(Lp, NP); tail = prev[Lp * NP:]
                t0s, t1s = np.linspace(0, 1, Lp), np.linspace(0, 1, L)
                xi = np.stack([np.interp(t1s, t0s, xp[:, j]) if Lp > 1 else np.repeat(xp[0, j], L) for j in range(NP)], 1)
                cands.append(np.concatenate([xi.reshape(-1), tail]).astype(np.float32))
        for r in range(a.restarts):
            # diversify: rebuild init_vec with different variant by temp patching
            global_init = init_params
            def _iv(L_, pack_, r=r):
                P = pack_["P"]; pieces = []
                if pack_["kind"] != "agnostic":
                    pieces.append(init_params(L_, variant=r % 3))
                if pack_["kind"] != "shared":
                    th0 = pack_mlp(pack_["sizes"], np.random.default_rng(a.seed + 17 * L_ + r))
                    flat = np.concatenate([t.ravel() for t in th0])
                    if pack_["kind"] in ("cond", "costcond"):
                        sizes = pack_["sizes"]; dout, din = sizes[-1], sizes[-2]
                        n_before = theta_dim(sizes) - dout * din - dout
                        flat[n_before: n_before + dout * din] *= 0.05
                        flat[n_before + dout * din:] = 0.0
                    elif pack_["kind"] == "agnostic":
                        sizes = pack_["sizes"]; dout, din = sizes[-1], sizes[-2]
                        n_before = theta_dim(sizes) - dout * din - dout
                        flat[n_before: n_before + dout * din] *= 0.01
                        flat[n_before + dout * din:] = init_params(L_, variant=r % 3)
                    pieces.append(flat.astype(np.float32))
                out = np.concatenate(pieces).astype(np.float32)
                if r >= 3:  # variants repeat mod 3: jitter so restarts stay distinct
                    out = out + 0.05 * np.random.default_rng(a.seed + 101 * r + L_).standard_normal(out.size).astype(np.float32)
                return out.astype(np.float32)
            cands.append(_iv(L, pack))
        # dedupe length
        cands = [c for c in cands if c.size == pack_dim(pack)]
        if not cands: cands = [init_vec(L, pack)]

        best = None; t0 = time.time()
        for i, c0 in enumerate(cands[: a.restarts + 1]):
            l0 = f(c0)[0]
            r = minimize(f, c0, jac=True, method="L-BFGS-B",
                         options={"maxiter": a.maxiter, "ftol": 1e-10})
            log(f"  {key} cand{i} loss {l0:.4f} -> {r.fun:.4f} ({r.nit} it)")
            if best is None or r.fun < best[1]:
                best = (r.x.copy(), float(r.fun))
        vbest = best[0]; st, _ = unpack(jnp.asarray(vbest, jnp.float32), pack)
        # train summary on first pool
        p0 = eval_params_fn(st, pack, pools[0]["n"], pools[0]["insts"][: min(40, pools[0]["B"])])
        sm = sim.summarize(p0)
        # beta magnitudes
        if pack["kind"] == "shared":
            bm = beta_mags(np.asarray(st["base"]), L)
        elif pack["kind"] == "agnostic":
            xv = np.asarray(mlp_forward(st["theta"], jnp.array([[pools[0]["n"] / 20.0]], jnp.float32))[0])
            bm = beta_mags(xv, L)
        else:
            bm = beta_mags(np.asarray(st["base"]), L)
        res["train"][key] = {
            "v": vbest.tolist(), "loss": best[1], "train": sm, "beta": bm,
            "sec": time.time() - t0, "pack": pack["kind"], "P": pack["P"],
            "stage": stage_key, "L": L,
        }
        log(f"  {key} train med1/p={sm['median_inv_p']:.2f} B90={sm['B90_inv_p']:.2f} "
            f"|beta|={np.mean(bm):.3f} ({time.time()-t0:.0f}s)")
        json.dump(res, open(OUT, "w"), indent=1)

        # evaluate on test sizes (and next curriculum probe)
        probe = sorted(set(a.test) | ({a.stages[si + 1]} if si + 1 < len(a.stages) else set()))
        for n in probe:
            N = 200 if n == 14 else 100
            tkey = f"{key}|n{n}"
            if tkey in res["test"]: continue
            insts = sim.instances(n, N, TEB)
            if n >= 20:  # serialize big evals across jobs (OOM guard, ~4.4GB headroom)
                import fcntl
                with open("/tmp/nn_n20.lock", "w") as _lk:
                    fcntl.flock(_lk, fcntl.LOCK_EX)
                    p = eval_params_fn(st, pack, n, insts, chunk=2)
            else:
                p = eval_params_fn(st, pack, n, insts)
            s = sim.summarize(p, insts); s["p"] = [float(v) for v in p]
            res["test"][tkey] = s
            log(f"  test {tkey} med1/p={s['median_inv_p']:.1f} B90={s['B90_inv_p']:.1f}")
            json.dump(res, open(OUT, "w"), indent=1)

# also store a compact comparison table snapshot
res["curriculum"] = a.stages
json.dump(res, open(OUT, "w"), indent=1)
log("ALLDONE")
