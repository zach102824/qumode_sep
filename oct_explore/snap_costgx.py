"""Local SNAP (instead of ECD) + joint cost·GrayX.

Zach (Sat ~11:00 CST): maybe use SNAP instead of ECD.
Previous: ECD collapsed to |β|≈0; S_snap2 kept ECD+SNAP and hurt n=20 transfer;
native N_* joint SNAP (no GrayX) failed at n=20. This arm DROPS ECD entirely.

Layer (hardware-shaped, theory first):
  1. R(th,ph) on every control transmon (tbit layouts)
  2. Local SNAP on every cavity — kind:
       uncond : SNAP(s) on cavity (size-free poly)
       csnap  : conditional SNAP — SNAP(s0) on |0>, SNAP(s1) on |1|  (ECD replacement)
       free   : free phase on each logical Fock 0..2^k-1 (shared coeffs via interp / pad)
  3. ONE joint U = GrayX(bc) · exp(-i g Cn)   (cost then GrayX; same as S_costgx)

Params/layer (csnap, snap=S): [th, ph, s0_0..s0_{S-1}, s1_0..s1_{S-1}, bc, g]
uncond: [th, ph, s_0..s_{S-1}, bc, g]
free K: [th, ph, φ_0..φ_{K-1}, bc, g]  applied as φ[min(n,K-1)] on Fock n (logical)
"""
from __future__ import annotations
import argparse, json, os, sys, time, fcntl, contextlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, jax, jax.numpy as jnp
from scipy.optimize import minimize
import sim

ap = argparse.ArgumentParser()
ap.add_argument("--tag", required=True)
ap.add_argument("--layout", default="T", choices=["T", "C", "R"])
ap.add_argument("--code", default="gray", choices=["gray", "binary"])
ap.add_argument("--kind", default="csnap", choices=["uncond", "csnap", "free"])
ap.add_argument("--snap", type=int, default=4, help="poly degree (uncond/csnap) or #free phases")
ap.add_argument("--Ls", type=int, nargs="+", default=[2, 4, 8])
ap.add_argument("--ntrain", type=int, default=80)
ap.add_argument("--nt2", type=int, nargs="*", default=[])
ap.add_argument("--nt2_count", type=int, default=40)
ap.add_argument("--test", type=int, nargs="*", default=[10, 12, 14, 20])
ap.add_argument("--restarts", type=int, default=4)
ap.add_argument("--maxiter", type=int, default=300)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--obj", default="meanlog", choices=["meanlog", "cvar", "mix"])
ap.add_argument("--alpha", type=float, default=0.2)
ap.add_argument("--w", type=float, default=1.0)
ap.add_argument("--no_R", action="store_true", help="skip transmon R (ablation)")
ap.add_argument("--rich", type=int, default=1, help="cost poly degree (1=g*Cn, 2=+g1*Cn^2, ...)")
ap.add_argument("--ebins", type=int, default=0,
                help="if >0, free diagonal cost with ebins phases on round(Cn) (overrides --rich)")
a = ap.parse_args()

LAY = {"T": sim.LAYOUTS_T, "C": sim.LAYOUTS_C, "R": sim.LAYOUTS_R}[a.layout]
tbit = a.layout in ("T", "R")
log = lambda *s: print(time.strftime("%H:%M:%S"), *s, flush=True)
TRB, TEB = 31_000_000_000, 32_000_000_000
OUT = f"results/{a.tag}.json"
S = int(a.snap)
if a.ebins > 0:
    a.rich = a.ebins
NG = int(a.rich)
USE_EBINS = a.ebins > 0
# param layout
# R: th,ph (2) unless --no_R
NR = 0 if a.no_R else 2
if a.kind == "uncond":
    NS = S
elif a.kind == "csnap":
    NS = 2 * S
else:  # free
    NS = S
NP = NR + NS + 1 + NG  # + bc + g0..g_{NG-1}


def gray_scalar(i):
    return i ^ (i >> 1)


def ungray_scalar(g):
    g = int(g); s = g >> 1
    while s:
        g ^= s; s >>= 1
    return g


def bit_X_sum(k, nf):
    M = np.zeros((nf, nf)); N = 1 << k
    for n in range(N):
        for j in range(k):
            n2 = ungray_scalar(gray_scalar(n) ^ (1 << j)) if a.code == "gray" else n ^ (1 << j)
            M[n2, n] += 1.0
    return M


def build(n, insts, ks):
    C = len(ks)
    assert sum(ks) + (C if tbit else 0) == n
    nfs = [(1 << k) + max(8, (1 << k) // 4) for k in ks]
    Vs, lams, nvs, scales = [], [], [], []
    for k, nf in zip(ks, nfs):
        lam, V = np.linalg.eigh(bit_X_sum(k, nf))
        Vs.append(jnp.asarray(V.astype(np.complex64)))
        lams.append(jnp.asarray(lam.astype(np.float32)))
        nvs.append(jnp.arange(nf, dtype=np.float32))
        scales.append(float(np.sqrt(2.0 ** k)))

    x = np.arange(1 << n); pos = n; tidx, fidx = [], []
    for c, k in enumerate(ks):
        nb = k + (1 if tbit else 0); pos -= nb
        val = (x >> pos) & ((1 << nb) - 1)
        if tbit:
            tidx.append(val >> k); f = val & ((1 << k) - 1)
        else:
            tidx.append(np.zeros_like(x)); f = val
        fidx.append(sim.ungray(f) if a.code == "gray" else f)
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
        fl0 = np.unique(np.ravel_multi_index(tuple([np.zeros_like(x)] * C + fidx), shape))
        psi0[fl0] = 1 / np.sqrt(len(fl0))
    psi0 = jnp.asarray(psi0.reshape(shape))
    Cn_j = jnp.asarray(Cn); succ_j = jnp.asarray(succ)

    def cav_U(c, b):
        return ((Vs[c] * jnp.exp(-1j * (b / 2) * lams[c])[None, :]) @ Vs[c].conj().T).astype(jnp.complex64)

    def apply_M(M, psi, ax):
        return jnp.moveaxis(jnp.tensordot(M, psi, axes=([1], [ax])), 0, ax)

    def Rm(th, ph):
        ct, st = jnp.cos(th / 2), jnp.sin(th / 2)
        return jnp.array([[ct, -1j * st * jnp.exp(-1j * ph)],
                          [-1j * st * jnp.exp(1j * ph), ct]]).astype(jnp.complex64)

    def snap_phase(c, s_coeffs):
        """Size-free poly SNAP: φ(n) = Σ_j s_j (n/√(2^k))^{j+1}."""
        u = nvs[c] / scales[c]
        ph = sum(s_coeffs[j] * (u ** (j + 1)) for j in range(S))
        return jnp.exp(-1j * ph).astype(jnp.complex64)

    def free_phase(c, phis):
        """φ on Fock n: use phis[min(n, S-1)]; pad levels get last phase."""
        idx = jnp.minimum(nvs[c].astype(jnp.int32), S - 1)
        return jnp.exp(-1j * phis[idx]).astype(jnp.complex64)

    def apply_snap_uncond(psi, s_coeffs):
        for c in range(C):
            Dph = snap_phase(c, s_coeffs)
            psi = psi * Dph.reshape([1] * (C + c) + [-1] + [1] * (len(nfs) - 1 - c))
        return psi

    def apply_snap_free(psi, phis):
        for c in range(C):
            Dph = free_phase(c, phis)
            psi = psi * Dph.reshape([1] * (C + c) + [-1] + [1] * (len(nfs) - 1 - c))
        return psi

    def apply_csnap(psi, s0, s1):
        """Conditional SNAP: |0>⊗SNAP(s0) + |1>⊗SNAP(s1) on each cavity–transmon."""
        for c in range(C):
            D0 = snap_phase(c, s0)
            D1 = snap_phase(c, s1)
            p0 = jnp.take(psi, 0, axis=c)
            p1 = jnp.take(psi, 1, axis=c)
            fa = C - 1 + c  # cavity axis after removing transmon
            # multiply cavity axis by diag phase
            sh0 = [1] * p0.ndim; sh0[fa] = -1
            p0 = p0 * D0.reshape(sh0)
            p1 = p1 * D1.reshape(sh0)
            psi = jnp.stack([p0, p1], axis=c)
        return psi

    def local_snap(psi, row):
        off = 0
        if NR:
            R = Rm(row[0], row[1]); off = 2
            for c in range(C):
                psi = apply_M(R, psi, c)
        sblock = row[off:off + NS]
        if a.kind == "uncond":
            psi = apply_snap_uncond(psi, sblock)
        elif a.kind == "csnap":
            psi = apply_csnap(psi, sblock[:S], sblock[S:])
        else:
            psi = apply_snap_free(psi, sblock)
        return psi

    def state1(xv, cn):
        L = xv.size // NP
        X = xv.reshape(L, NP)
        psi = psi0
        for l in range(L):
            psi = local_snap(psi, X[l])
            bc = X[l, -(1 + NG)]
            gs = X[l, -NG:]
            if USE_EBINS:
                idx = jnp.clip(jnp.round(cn).astype(jnp.int32), 0, NG - 1)
                phase = gs[idx]
            else:
                phase = gs[0] * cn
                for j in range(1, NG):
                    phase = phase + gs[j] * (cn ** (j + 1))
            psi = psi * jnp.exp(-1j * phase).astype(jnp.complex64)
            for c in range(C):
                psi = apply_M(cav_U(c, bc), psi, C + c)
        return psi

    def probs(xv):
        return jax.lax.map(lambda t: jnp.sum(jnp.abs(state1(xv, t[0])) ** 2 * t[1]), (Cn_j, succ_j))

    def loss(xv):
        nl = -jnp.log(probs(xv) + 1e-12)
        k = max(1, int(round(a.alpha * B)))
        tail = jnp.mean(jnp.sort(nl)[-k:])
        return {"meanlog": jnp.mean(nl), "cvar": tail, "mix": jnp.mean(nl) + a.w * tail}[a.obj]

    return jax.jit(probs), jax.jit(jax.value_and_grad(loss))


tr = [(8, sim.instances(8, a.ntrain, TRB))] + [(m, sim.instances(m, a.nt2_count, TRB)) for m in a.nt2]
tr_p, tr_vg = [], []
for n, insts in tr:
    log(f"build train n={n} N={len(insts)} kind={a.kind} snap={S} NP={NP} NG={NG} ebin={USE_EBINS} layout={a.layout}")
    p, vg = build(n, insts, LAY[n]); tr_p.append(p); tr_vg.append(vg)


def f(x):
    vs = [vg(jnp.asarray(x, jnp.float32)) for vg in tr_vg]
    return sum(float(v[0]) for v in vs) / len(vs), sum(np.asarray(v[1], float) for v in vs) / len(vs)


rng = np.random.default_rng(a.seed)
res = json.load(open(OUT)) if os.path.exists(OUT) else {"args": vars(a), "train": {}, "test": {}, "np": NP}
res["args_last"] = vars(a)


def init_layer(scale=1.0, t=0.5):
    row = []
    if NR:
        row += [0.4 * scale, 0.0]
    if a.kind == "csnap":
        row += list(0.15 * scale * rng.standard_normal(S))
        row += list(0.15 * scale * rng.standard_normal(S))
    elif a.kind == "uncond":
        row += list(0.2 * scale * rng.standard_normal(S))
    else:
        row += list(0.3 * scale * rng.standard_normal(S))
    row += [(0.9 - 0.7 * t) * scale]  # bc
    # cost coeffs
    g0 = (0.2 + 0.7 * t) * scale
    if USE_EBINS:
        row += [g0 * (1.0 - k / max(NG - 1, 1)) for k in range(NG)]
    else:
        row += [g0] + [0.05 * (1 - 2 * t) * scale] * (NG - 1)
    return np.array(row, float)


prev = None
for L in a.Ls:
    if str(L) in res["train"]:
        prev = np.array(res["train"][str(L)]["x"]); log(f"L={L} resume (trained)"); continue
    t0 = time.time(); cands = []
    if prev is not None:
        Lp = prev.size // NP; xp = prev.reshape(Lp, NP)
        t0s, t1s = np.linspace(0, 1, Lp), np.linspace(0, 1, L)
        xi = np.stack([np.interp(t1s, t0s, xp[:, j]) if Lp > 1 else np.repeat(xp[0, j], L) for j in range(NP)], 1)
        cands.append(xi.reshape(-1))
        cands.append(np.concatenate([xp.reshape(-1), 0.1 * np.concatenate([init_layer() for _ in range(L - Lp)])]))
    for sc in (0.6, 1.0, 1.4):
        rows = [init_layer(sc, t) for t in np.linspace(0, 1, L)]
        # deterministic SNAP seed for first cand of each sc
        rows[0] = init_layer(sc, 0.0)
        if a.kind == "csnap" and NR:
            # warm: modest conditional SNAP difference
            r = list(rows[0]); r[2:2 + S] = [0.1 * sc] * S; r[2 + S:2 + 2 * S] = [-0.1 * sc] * S
            rows[0] = np.array(r, float)
        cands.append(np.array(rows, float).reshape(-1))
    for s in (0.5, 1.0, 1.5):
        cands.append(np.concatenate([init_layer(s, t) for t in np.linspace(0, 1, L)]))
    # QAOA-like GrayX/cost schedule with SNAP≈0 (sanity basin)
    rows = []
    for t in np.linspace(0, 1, L):
        r = []
        if NR: r += [-np.pi if t < 1e-9 else 0.3, 0.0]
        r += [0.0] * NS
        r += [(0.9 - 0.7 * t)]
        g0 = (0.2 + 0.7 * t)
        if USE_EBINS:
            r += [g0 * (1.0 - k / max(NG - 1, 1)) for k in range(NG)]
        else:
            r += [g0] + [0.0] * (NG - 1)
        rows.append(r)
    cands.append(np.array(rows, float).reshape(-1))
    cands = cands[: max(a.restarts, 3)]
    best = None
    for i, c0 in enumerate(cands):
        l0 = f(c0)[0]
        r = minimize(f, c0, jac=True, method="L-BFGS-B", options={"maxiter": a.maxiter})
        fun = float(r.fun) if np.isfinite(r.fun) else 1e9
        log(f"  L={L} cand{i} loss {l0:.4f} -> {fun:.4f} ({r.nit} it)")
        if best is None or fun < best[1]:
            best = (r.x, fun)
    prev = best[0]
    p = np.asarray(tr_p[0](jnp.asarray(prev, jnp.float32)))
    # SNAP magnitude diagnostic
    X = prev.reshape(L, NP)
    if a.kind == "csnap":
        s0 = X[:, NR:NR + S]; s1 = X[:, NR + S:NR + 2 * S]
        snap_mag = float(np.mean(np.abs(s0 - s1)))
    else:
        snap_mag = float(np.mean(np.abs(X[:, NR:NR + NS])))
    res["train"][str(L)] = {"x": prev.tolist(), "loss": best[1], "train": sim.summarize(p),
                            "sec": time.time() - t0, "snap_mag": snap_mag}
    log(f"L={L} train med1/p={np.median(1/np.maximum(p,1e-300)):.2f} B90={np.quantile(1/np.maximum(p,1e-300),.9):.2f} snap_mag={snap_mag:.4f} ({time.time()-t0:.0f}s)")
    json.dump(res, open(OUT, "w"), indent=1)

for n, N in [(8, 100)] + [(nn, 200 if nn == 14 else 100) for nn in a.test]:
    res["test"].setdefault(str(n), {}); ins = None
    ch = 20 if n < 14 else (10 if n < 20 else 4)
    for L in a.Ls:
        if str(L) in res["test"][str(n)]: continue
        ins = ins or sim.instances(n, N, TEB); p = []
        x = np.asarray(res["train"][str(L)]["x"], np.float32)
        _lk = open("/tmp/nn_n20.lock", "w") if n >= 20 else None
        if _lk: fcntl.flock(_lk, fcntl.LOCK_EX); ch = 2  # shared with nn_costgx: one big eval at a time
        try:
            for i in range(0, N, ch):
                pp, _ = build(n, ins[i:i + ch], LAY[n]); p += list(np.asarray(pp(jnp.asarray(x))))
        finally:
            if _lk: fcntl.flock(_lk, fcntl.LOCK_UN); _lk.close()
        s = sim.summarize(p, ins); s["p"] = [float(v) for v in p]
        res["test"][str(n)][str(L)] = s
        log(f"test n={n} L={L} med1/p={s['median_inv_p']:.1f} B90={s['B90_inv_p']:.1f}")
        json.dump(res, open(OUT, "w"), indent=1)
log("ALLDONE")
