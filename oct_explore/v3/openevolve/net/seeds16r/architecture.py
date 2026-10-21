"""Evolvable NETWORK: formula -> (gamma_l, beta_l, w_jl, p_v) for the locked native cavity-chain circuit.
train(api, budget_s) -> predict;  predict(n, V, SG, m, place)
(+ OPTIONAL dS (F,R,L,C,32) SNAP and dB (F,R,L,C-1,32,32) neighbour-bus additive phase tables; api.place(n, idx) gives cavity contents) -> dict(gamma (F,L), beta (F,L), w (F,m,L), pv (F,n)).
V (F,m,4) clause variable indices, SG (F,m,4) literal signs (+1/-1; literal violated when x_v == (0 if s>0 else 1)).
api.data(n) -> dict(V, SG, m, F) training formulas (n in api.n_list = [10, 15]);
api.batch(n, fidx) -> static circuit inputs; api.p_from(n, outs, batch) -> differentiable p (F, R) over R placements (col 0 = min-cut).
Fair objective = round-robin: mean over the R placements. Allowed imports: numpy, jax, math, functools, itertools, collections, time.
"""
import time, numpy as np, jax, jax.numpy as jnp

# EVOLVE-BLOCK-START
# FAMILY: architecture - network architecture: message passing depth/width, attention, heads
# parent: (net16o ckpt_60 best fd65b9be: selection 56.9/599, fair TEB n=16 84.4/5229)
H = 32
LAM_GLOB = 0.0   # OPTIONAL training-only pull of chain p toward the global-cost p (api.p_from(..., glob=True)); never part of fitness.
TAIL_W = 0.5     # weight of the worst-quarter (hard tail) term in the loss
G0 = np.array([-3.1275, -6.49, 9.6225, 8.4856], np.float32); B0 = np.array([1.1305, 0.3672, -0.5844, -0.3928], np.float32)  # F32_gray_chain (trained n=10+15 only)

def features(n, V, SG, m):
    """unsigned incidence (even) + signed incidence (odd). Flipping SG on var i flips xo_i and sgn[..., i]."""
    F = V.shape[0]; inc = np.zeros((F, m, n), np.float32); sgn = np.zeros((F, m, n), np.float32)
    fi = np.arange(F)[:, None, None]; ci = np.arange(m)[None, :, None]
    np.add.at(inc, (fi, ci, V), 1.0); np.add.at(sgn, (fi, ci, V), np.asarray(SG, np.float32))
    deg = inc.sum(1); sc = 2.0 * m / n
    xe = np.stack([np.ones_like(deg), deg / sc], -1)
    xo = (sgn.sum(1) / sc)[..., None]
    return jnp.asarray(xe), jnp.asarray(xo), jnp.asarray(inc), jnp.asarray(sgn), jnp.asarray(V, jnp.int32)

def init(rng):
    def W(i, o, s=1.0): return jnp.asarray(rng.normal(0, s / np.sqrt(i), (i, o)).astype(np.float32))
    return dict(We=W(2, H), Wo=W(1, H), Wce=W(H, H), Wco=W(H, H), Wve=W(H, H), Wvo=W(H, H),
                Wc=W(2 * H, H), wc=W(H, 4, 0.05), wp=W(H, 1, 0.5)[:, 0],
                g=jnp.asarray(G0), b=jnp.asarray(B0), Wg=W(H, 8, 0.01),
                As=W(H, H), asl=jnp.zeros((4, H)), Ab=W(H, H), bsl=jnp.zeros((4, H)),  # correction heads; zero-init => starts at seed
                Wq1=W(H + 4, H), Wq2=jnp.zeros((H, 4)), Wgg=jnp.zeros((4, 8)), Lv=W(1, H, 0.3)[0])

SUB = np.array([[(s >> i) & 1 for i in range(4)] for s in range(1, 16)], bool)  # 15 nonempty literal subsets, last = full clause

def geom(V, place, n):
    """size-free chain geometry per clause/placement: fraction of Walsh subsets not inside one cavity or neighbour pair."""
    cav = jnp.asarray(place["cav"]); F, R, C, _ = cav.shape; m = V.shape[1]
    eq = (cav[..., None] == jnp.arange(n)).astype(jnp.float32)
    cv = jnp.einsum("frcjn,c->frn", eq, jnp.arange(C, dtype=jnp.float32))
    c = jnp.take_along_axis(jnp.broadcast_to(cv[:, :, None, :], (F, R, m, n)), jnp.broadcast_to(V[:, None], (F, R, m, 4)), axis=-1)
    cs = c[..., None, :]; M = jnp.asarray(SUB)
    mx = jnp.max(jnp.where(M, cs, -1.0), -1); mn = jnp.min(jnp.where(M, cs, 99.0), -1)
    rep = (mx - mn <= 1.5).astype(jnp.float32); lost = 1.0 - rep.mean(-1)
    span = (c[..., None] == jnp.arange(C, dtype=jnp.float32)).any(-2).sum(-1) / 4.0
    ext = (c.max(-1) - c.min(-1)) / 3.0
    return jnp.stack([lost, rep[..., -1], span, ext], -1), lost

def sign_table(k):
    """(32,5): +-1 digit pattern of SNAP/bus table index (MSB first) for a k-bit cavity, 0 for unused slots/indices."""
    i = np.arange(32)[:, None]; j = np.arange(5)[None]; b = (i >> np.maximum(k - 1 - j, 0)) & 1
    return jnp.asarray(np.where((j < k) & (i < (1 << k)), 1.0 - 2.0 * b, 0.0).astype(np.float32))

def corr(p, he, ho, place, lv):
    """size-free tables: even he is SG-invariant; odd ho flips with SG, mixed with ST (Z digit pattern)."""
    cav = jnp.asarray(place["cav"]); ks = place["ks"]; F, R, C, _ = cav.shape; ST = sign_table(ks[0])
    ix = jnp.maximum(cav, 0).reshape(F, R, C * 5); msk = (cav >= 0)[..., None]
    lvc = jnp.take_along_axis(lv, ix, axis=-1).reshape(F, R, C, 5)
    take = lambda h: jax.vmap(lambda u, c: u[jnp.maximum(c, 0)])(h, cav.reshape(F, -1)).reshape(F, R, C, 5, -1)
    hcv = (take(he) + take(ho) + lvc[..., None] * p["Lv"]) * msk
    z = jnp.einsum("kj,frcjh->frckh", ST, hcv)
    us, ub = jnp.tanh(z @ p["As"]), jnp.tanh(z @ p["Ab"])
    dS = jnp.einsum("frckh,lh->frlck", us, p["asl"])
    dB = jnp.einsum("frakh,frbqh,lh->frlakq", ub[:, :, :-1], ub[:, :, 1:], p["bsl"])
    return dS, dB

def apply(p, xe, xo, inc, sgn, V, place=None):
    """3-round shared-weight MP. Sign flip of var i: e unchanged, o_i -> -o_i, pv_i -> 1-pv_i, gamma/beta/w fixed."""
    e = jnp.tanh(xe @ p["We"]); o = jnp.tanh(xo @ p["Wo"]); vd = inc.sum(1)[..., None] + 1.0
    for _ in range(4):
        me = jnp.einsum("fmn,fnh->fmh", inc, e); mo = jnp.einsum("fmn,fnh->fmh", sgn, o)  # |SG|·e even; SG·o even
        hc = jnp.tanh((me @ p["Wce"] + mo @ p["Wco"]) / 4.0)
        e = jnp.tanh(e + (jnp.einsum("fmn,fmh->fnh", inc, hc) / vd) @ p["Wve"])  # even via |SG|
        o = jnp.tanh(o + (jnp.einsum("fmn,fmh->fnh", sgn, hc) / vd) @ p["Wvo"])  # odd via SG
    hc = jnp.tanh(jnp.concatenate([jnp.einsum("fmn,fnh->fmh", inc, e), jnp.einsum("fmn,fnh->fmh", sgn, o)], -1) / 4.0 @ p["Wc"])
    if place is not None:
        gr, lost = geom(V, place, inc.shape[2]); geo = gr.mean(1)
        lv = jnp.einsum("frm,fmn->frn", lost, inc) / vd[:, None, :, 0]
    else:
        geo = jnp.zeros(V.shape[:2] + (4,), jnp.float32)
    gb = jnp.tanh(e.mean(1) @ p["Wg"] + geo.mean(1) @ p["Wgg"])
    w = hc @ p["wc"] + jnp.tanh(jnp.concatenate([hc, geo], -1) @ p["Wq1"]) @ p["Wq2"]
    out = dict(gamma=p["g"][None] + gb[:, :4], beta=p["b"][None] + gb[:, 4:], w=w, pv=jax.nn.sigmoid(o @ p["wp"]))
    if place is not None: out["dS"], out["dB"] = corr(p, e, o, place, lv)
    return out

def train(api, budget_s):
    t0 = time.time(); rng = np.random.default_rng(0); p = init(rng)

    from jax.flatten_util import ravel_pytree
    th, unr = ravel_pytree(p); mo = jnp.zeros_like(th); ve = jnp.zeros_like(th)
    data = {n: api.data(n) for n in api.n_list}; feats = {n: features(n, d["V"], d["SG"], d["m"]) for n, d in data.items()}
    bs = {8: 12, 10: 8, 12: 12, 15: 4}
    def loss(th, n, fx, pl, b):
        pp = unr(th); out = apply(pp, *fx, place=pl)
        P = api.p_from(n, out, b); nl = -jnp.log(P.mean(1) + 1e-12)
        xe, xo, inc, sgn, V = fx; of = apply(pp, xe, -xo, inc, -sgn, V, place=pl)  # all-sign flip: pv->1-pv, even heads fixed
        pen = ((out["gamma"] - of["gamma"]) ** 2).mean() + ((out["beta"] - of["beta"]) ** 2).mean()
        pen = pen + ((out["w"] - of["w"]) ** 2).mean() + ((out["pv"] + of["pv"] - 1.0) ** 2).mean()
        k = max(1, nl.shape[0] // 4); tot = nl.mean() + TAIL_W * jax.lax.top_k(nl, k)[0].mean() + 0.05 * pen
        if LAM_GLOB > 0:  # Bernoulli KL(p_glob || p_chain) per (formula, placement); global p is a stop-gradient target
            Pg = jax.lax.stop_gradient(jnp.clip(api.p_from(n, out, b, glob=True), 1e-6, 1 - 1e-6)); Pc = jnp.clip(P, 1e-6, 1 - 1e-6)
            tot = tot + LAM_GLOB * (Pg * jnp.log(Pg / Pc) + (1 - Pg) * jnp.log((1 - Pg) / (1 - Pc))).mean()
        return tot
    vg = {n: jax.jit(jax.value_and_grad(lambda th, fx, pl, b, n=n: loss(th, n, fx, {"cav": pl, "ks": api.place(n, [0])["ks"]}, b))) for n in api.n_list}
    t = 0; T = budget_s * 0.9; ema = None
    while time.time() - t0 < T:
        g = 0
        for n in api.n_list:
            idx = rng.choice(data[n]["F"], bs.get(n, 8), replace=False); fx = tuple(x[idx] for x in feats[n])
            _, gn = vg[n](th, fx, jnp.asarray(api.place(n, idx)["cav"]), api.batch(n, idx)); g = g + gn
        gnorm = float(jnp.sqrt((g * g).sum()))
        if not np.isfinite(gnorm): continue
        t += 1; g = g * min(1.0, 10.0 / (gnorm + 1e-9))
        fr = (time.time() - t0) / T; lr = 3e-3 * (1.0 if fr < 0.7 else max(0.2, (1.0 - fr) / 0.3))
        mo = 0.9 * mo + 0.1 * g; ve = 0.999 * ve + 0.001 * g * g
        th = th - lr * (mo / (1 - 0.9 ** t)) / (jnp.sqrt(ve / (1 - 0.999 ** t)) + 1e-8)
        if fr > 0.4: ema = th if ema is None else 0.98 * ema + 0.02 * th
    pf = unr(th if ema is None or not bool(jnp.all(jnp.isfinite(ema))) else ema)
    def predict(n, V, SG, m, place=None):
        V, SG = np.asarray(V), np.asarray(SG); res = []
        for s in range(0, V.shape[0], 32):
            pl = None if place is None else dict(cav=np.asarray(place["cav"])[s:s + 32], ks=place["ks"])
            out = apply(pf, *features(n, V[s:s + 32], SG[s:s + 32], m), place=pl)
            res.append({k: np.nan_to_num(np.asarray(v)) for k, v in out.items()})
        return {k: np.concatenate([r[k] for r in res]) for k in res[0]}
    return predict
# EVOLVE-BLOCK-END
