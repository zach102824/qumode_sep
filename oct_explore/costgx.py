"""ECD + cost·GrayX (S_costgx recipe) — improvement driver with checkpoint/resume.

Layer l (hardware): local R(th,ph) + ECD(beta) on every cavity–transmon pair (params shared across pairs),
then ONE joint U = GrayX(bc) · exp(-i g Cn)  (cost first, then Gray-X on cavity bits). Cn = E n/m.
Options beyond theory_u.py S_costgx:
  --final_ecd       one extra trailing R+ECD block after the last U (transmon bits otherwise get only L-1
                    useful mixes; the first R+ECD acts on |+> and is nearly idle).
  --init qaoa       warm start from QAOA_raw angles at the same L (b->bc, g->g, transmon th=b_prev-pi, ph=0, beta~0)
  --obj meanlog|cvar|mix   cvar = mean of worst --alpha fraction of -log p;  mix = meanlog + w*cvar
  --nt2 10 ...      joint multi-size training
  --ecd2            second transmon rotation after the ECD (6 -> 8 params/layer)
Resume: re-running the same --tag reloads results/<tag>.json, skips trained L and finished tests.
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
ap.add_argument("--code", default="gray", choices=["gray", "binary"])
ap.add_argument("--Ls", type=int, nargs="+", default=[2, 4, 8])
ap.add_argument("--ntrain", type=int, default=80)
ap.add_argument("--nt2", type=int, nargs="*", default=[])
ap.add_argument("--nt2_count", type=int, default=40)
ap.add_argument("--test", type=int, nargs="*", default=[10, 12, 14, 20])
ap.add_argument("--restarts", type=int, default=4)
ap.add_argument("--maxiter", type=int, default=300)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--final_ecd", action="store_true")
ap.add_argument("--ecd2", action="store_true")
ap.add_argument("--init", default="grow", choices=["grow", "qaoa", "both"])
ap.add_argument("--qaoa_ref", default="results/QAOA_raw.json")
ap.add_argument("--obj", default="meanlog", choices=["meanlog", "cvar", "mix"])
ap.add_argument("--alpha", type=float, default=0.2)
ap.add_argument("--w", type=float, default=1.0)
ap.add_argument("--beta_l2", type=float, default=0.0, help="L2 penalty on ECD beta (transfer regulariser)")
a = ap.parse_args()

LAY = {"T": sim.LAYOUTS_T, "C": sim.LAYOUTS_C, "R": sim.LAYOUTS_R}[a.layout]
tbit = a.layout in ("T", "R")
log = lambda *s: print(time.strftime("%H:%M:%S"), *s, flush=True)
TRB, TEB = 31_000_000_000, 32_000_000_000
OUT = f"results/{a.tag}.json"
NB = 6 if a.ecd2 else 4          # local block params: bre, bim, th, ph (+ th2, ph2)
NP = NB + 2                      # + bc, g
NF = NB if a.final_ecd else 0    # trailing block


def gray_scalar(i): return i ^ (i >> 1)
def ungray_scalar(g):
    g = int(g); s = g >> 1
    while s: g ^= s; s >>= 1
    return g


def bit_X_sum(k, nf):
    M = np.zeros((nf, nf)); N = 1 << k
    for n in range(N):
        for j in range(k):
            n2 = ungray_scalar(gray_scalar(n) ^ (1 << j)) if a.code == "gray" else n ^ (1 << j)
            M[n2, n] += 1.0
    return M


def build(n, insts, ks):
    C = len(ks); assert sum(ks) + (C if tbit else 0) == n
    nfs = [(1 << k) + max(8, (1 << k) // 4) for k in ks]
    Vs, lams, Vd, lamd, nvs, scales = [], [], [], [], [], []
    for k, nf in zip(ks, nfs):
        lam, V = np.linalg.eigh(bit_X_sum(k, nf)); Vs.append(jnp.asarray(V.astype(np.complex64))); lams.append(jnp.asarray(lam.astype(np.float32)))
        aa = np.diag(np.sqrt(np.arange(1, nf)), 1); lam, V = np.linalg.eigh(1j * (aa.T - aa))
        Vd.append(jnp.asarray(V.astype(np.complex64))); lamd.append(jnp.asarray(lam.astype(np.float32)))
        nvs.append(jnp.arange(nf)); scales.append(np.sqrt(2.0 ** k))
    x = np.arange(1 << n); pos = n; tidx, fidx = [], []
    for c, k in enumerate(ks):
        nb = k + (1 if tbit else 0); pos -= nb; val = (x >> pos) & ((1 << nb) - 1)
        if tbit: tidx.append(val >> k); f = val & ((1 << k) - 1)
        else: tidx.append(np.zeros_like(x)); f = val
        fidx.append(sim.ungray(f) if a.code == "gray" else f)
    shape = [2] * C + nfs; D = int(np.prod(shape)); flat = np.ravel_multi_index(tuple(tidx + fidx), shape)
    B = len(insts); Cn = np.zeros((B, D), np.float32); succ = np.zeros((B, D), np.float32)
    for i, ins in enumerate(insts):
        E = ins["E"].astype(float); c = E * n / ins["m"]
        row = np.full(D, c.max()); row[flat] = c; Cn[i] = row; np.add.at(succ[i], flat, (E == 0).astype(float))
    Cn = Cn.reshape([B] + shape); succ = succ.reshape([B] + shape)
    if not tbit:
        sl = (slice(None),) + (0,) * C
        Cn = np.broadcast_to(Cn[sl][(slice(None),) + (None,) * C], Cn.shape).copy()
        succ = np.broadcast_to(succ[sl][(slice(None),) + (None,) * C], succ.shape).copy()
    psi0 = np.zeros(D, np.complex64)
    if tbit: psi0[flat] = 1 / np.sqrt(len(flat))
    else:
        fl0 = np.unique(np.ravel_multi_index(tuple([np.zeros_like(x)] * C + fidx), shape)); psi0[fl0] = 1 / np.sqrt(len(fl0))
    psi0 = jnp.asarray(psi0.reshape(shape)); Cn_j = jnp.asarray(Cn); succ_j = jnp.asarray(succ)

    def disp(c, re, im):
        s = scales[c] / 2.0; r = jnp.sqrt(re ** 2 + im ** 2 + 1e-20) * s; ph = jnp.arctan2(im, re)
        W = jnp.exp(1j * ph * nvs[c])[:, None] * Vd[c]
        return ((W * jnp.exp(-1j * r * lamd[c])[None, :]) @ W.conj().T).astype(jnp.complex64)
    def cav_U(c, b): return ((Vs[c] * jnp.exp(-1j * (b / 2) * lams[c])[None, :]) @ Vs[c].conj().T).astype(jnp.complex64)
    def apply_M(M, psi, ax): return jnp.moveaxis(jnp.tensordot(M, psi, axes=([1], [ax])), 0, ax)
    def Rm(th, ph):
        ct, st = jnp.cos(th / 2), jnp.sin(th / 2)
        return jnp.array([[ct, -1j * st * jnp.exp(-1j * ph)], [-1j * st * jnp.exp(1j * ph), ct]]).astype(jnp.complex64)
    def local(psi, r):
        R = Rm(r[2], r[3]); R2 = Rm(r[4], r[5]) if a.ecd2 else None
        for c in range(C):
            psi = apply_M(R, psi, c); Dp = disp(c, r[0], r[1])
            p0 = jnp.take(psi, 0, axis=c); p1 = jnp.take(psi, 1, axis=c); fa = C - 1 + c
            psi = jnp.stack([apply_M(Dp.conj().T, p1, fa), apply_M(Dp, p0, fa)], axis=c)
            if R2 is not None: psi = apply_M(R2, psi, c)
        return psi
    def state1(xv, cn):
        L = (xv.size - NF) // NP; X = xv[:L * NP].reshape(L, NP); psi = psi0
        for l in range(L):
            psi = local(psi, X[l]); psi = psi * jnp.exp(-1j * X[l, NB + 1] * cn).astype(jnp.complex64)
            for c in range(C): psi = apply_M(cav_U(c, X[l, NB]), psi, C + c)
        if NF: psi = local(psi, xv[L * NP:])
        return psi
    def probs(xv): return jax.lax.map(lambda t: jnp.sum(jnp.abs(state1(xv, t[0])) ** 2 * t[1]), (Cn_j, succ_j))
    def loss(xv):
        nl = -jnp.log(probs(xv) + 1e-12)
        k = max(1, int(round(a.alpha * B))); tail = jnp.mean(jnp.sort(nl)[-k:])
        L_ = {"meanlog": jnp.mean(nl), "cvar": tail, "mix": jnp.mean(nl) + a.w * tail}[a.obj]
        if a.beta_l2 > 0:
            L = (xv.size - NF) // NP; X = xv[:L * NP].reshape(L, NP)
            L_ = L_ + a.beta_l2 * jnp.sum(X[:, :2] ** 2)
        return L_
    return jax.jit(probs), jax.jit(jax.value_and_grad(loss))


tr = [(8, sim.instances(8, a.ntrain, TRB))] + [(m, sim.instances(m, a.nt2_count, TRB)) for m in a.nt2]
tr_p, tr_vg = [], []
for n, insts in tr:
    log(f"build train n={n} N={len(insts)} final_ecd={a.final_ecd} ecd2={a.ecd2} obj={a.obj} init={a.init}")
    p, vg = build(n, insts, LAY[n]); tr_p.append(p); tr_vg.append(vg)


def f(x):
    vs = [vg(jnp.asarray(x, jnp.float32)) for vg in tr_vg]
    return sum(float(v[0]) for v in vs) / len(vs), sum(np.asarray(v[1], float) for v in vs) / len(vs)


rng = np.random.default_rng(a.seed)
res = json.load(open(OUT)) if os.path.exists(OUT) else {"args": vars(a), "train": {}, "test": {}, "np": NP, "nf": NF}
res["args_last"] = vars(a)


def blk(th=None, ph=0.0, beta=0.01):
    v = [beta, beta, th if th is not None else rng.uniform(0, np.pi), ph if th is not None else rng.uniform(0, 2 * np.pi)]
    if a.ecd2: v += [0.0, 0.0]
    return v


def init_layer(s):
    return blk() + [0.6 * s, 0.4 * s]


def qaoa_init(L):
    q = json.load(open(a.qaoa_ref))["train"]
    if str(L) in q: xq = np.array(q[str(L)]["x"]).reshape(L, 2)
    else:
        Ls = sorted(int(k) for k in q); Lb = max(k for k in Ls if k <= L); xp = np.array(q[str(Lb)]["x"]).reshape(Lb, 2)
        t0, t1 = np.linspace(0, 1, Lb), np.linspace(0, 1, L); xq = np.stack([np.interp(t1, t0, xp[:, j]) for j in range(2)], 1)
    rows = []
    for l in range(L):
        th = -np.pi if l == 0 else xq[l - 1, 0] - np.pi
        rows.append(blk(th, 0.0) + [xq[l, 0], xq[l, 1]])
    out = np.array(rows, float).reshape(-1)
    if NF: out = np.concatenate([out, blk(xq[-1, 0] - np.pi, 0.0)])
    return out


prev = None
for L in a.Ls:
    if str(L) in res["train"]:
        prev = np.array(res["train"][str(L)]["x"]); log(f"L={L} resume (trained)"); continue
    t0 = time.time(); cands = []
    if a.init in ("qaoa", "both"): cands.append(qaoa_init(L))
    if prev is not None and a.init in ("grow", "both"):
        Lp = (prev.size - NF) // NP; xp = prev[:Lp * NP].reshape(Lp, NP); tail = prev[Lp * NP:]
        t0s, t1s = np.linspace(0, 1, Lp), np.linspace(0, 1, L)
        xi = np.stack([np.interp(t1s, t0s, xp[:, j]) if Lp > 1 else np.repeat(xp[0, j], L) for j in range(NP)], 1)
        cands.append(np.concatenate([xi.reshape(-1), tail]))
        cands.append(np.concatenate([xp.reshape(-1), 0.1 * np.concatenate([init_layer(1.0) for _ in range(L - Lp)]), tail]))
    if a.init in ("grow", "both"):
        for sc in (0.7, 1.2):
            rows = [[0.05 * sc, 0.05 * sc, 0.3, 0.0] + ([0.0, 0.0] if a.ecd2 else []) + [(0.9 - 0.7 * t) * sc, (0.2 + 0.7 * t) * sc]
                    for t in np.linspace(0, 1, L)]
            c0 = np.array(rows, float).reshape(-1); cands.append(np.concatenate([c0, blk(0.3, 0.0)]) if NF else c0)
        for s in (0.4, 0.8, 1.2, 1.6):
            c0 = np.concatenate([init_layer(s) for _ in range(L)]); cands.append(np.concatenate([c0, blk()]) if NF else c0)
    cands = cands[: a.restarts + (2 if a.init != "qaoa" else 0)] if a.init != "qaoa" else cands[:1]
    best = None
    for i, c0 in enumerate(cands):
        l0 = f(c0)[0]; r = minimize(f, c0, jac=True, method="L-BFGS-B", options={"maxiter": a.maxiter})
        log(f"  L={L} cand{i} loss {l0:.4f} -> {r.fun:.4f} ({r.nit} it)")
        if best is None or r.fun < best[1]: best = (r.x, float(r.fun))
    prev = best[0]; p = np.asarray(tr_p[0](jnp.asarray(prev, jnp.float32)))
    res["train"][str(L)] = {"x": prev.tolist(), "loss": best[1], "train": sim.summarize(p), "sec": time.time() - t0}
    log(f"L={L} train med1/p={np.median(1/np.maximum(p,1e-300)):.2f} B90={np.quantile(1/np.maximum(p,1e-300),.9):.2f} ({time.time()-t0:.0f}s)")
    json.dump(res, open(OUT, "w"), indent=1)

for n, N in [(8, 100)] + [(nn, 200 if nn == 14 else 100) for nn in a.test]:
    res["test"].setdefault(str(n), {}); ins = None; ch = 20 if n < 14 else (10 if n < 20 else 4)
    for L in a.Ls:
        if str(L) in res["test"][str(n)]: continue
        ins = ins or sim.instances(n, N, TEB); p = []; x = np.asarray(res["train"][str(L)]["x"], np.float32)
        for i in range(0, N, ch):
            pp, _ = build(n, ins[i:i + ch], LAY[n]); p += list(np.asarray(pp(jnp.asarray(x))))
        s = sim.summarize(p, ins); s["p"] = [float(v) for v in p]; res["test"][str(n)][str(L)] = s
        log(f"test n={n} L={L} med1/p={s['median_inv_p']:.1f} B90={s['B90_inv_p']:.1f}")
        json.dump(res, open(OUT, "w"), indent=1)
log("ALLDONE")
