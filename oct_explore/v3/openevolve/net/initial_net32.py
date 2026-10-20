"""Evolvable NETWORK: formula -> (gamma_l, beta_l, w_jl, p_v) for the locked native cavity-chain circuit.
train(api, budget_s) -> predict;  predict(n, V, SG, m) -> dict(gamma (F,L), beta (F,L), w (F,m,L), pv (F,n)).
V (F,m,4) clause variable indices, SG (F,m,4) literal signs (+1/-1; literal violated when x_v == (0 if s>0 else 1)).
api.data(n) -> dict(V, SG, m, F) training formulas (n in api.n_list = [10, 15]);
api.batch(n, fidx) -> static circuit inputs; api.p_from(n, outs, batch) -> differentiable p (F, R) over R placements (col 0 = min-cut).
Fair objective = round-robin: mean over the R placements. Allowed imports: numpy, jax, math, functools, itertools, collections, time.
"""
import time, numpy as np, jax, jax.numpy as jnp

# EVOLVE-BLOCK-START
H = 32
G0 = np.array([-3.0725, -6.2895, 9.5110, 8.4176], np.float32); B0 = np.array([1.1378, 0.3867, -0.5829, -0.3943], np.float32)  # F16_gray_chain_mc (trained n=8+12 only)

def features(n, V, SG, m):
    """per-variable and per-clause features from the formula only."""
    F = V.shape[0]; inc_p = np.zeros((F, m, n), np.float32); inc_n = np.zeros((F, m, n), np.float32)
    fi = np.arange(F)[:, None, None]; ci = np.arange(m)[None, :, None]
    np.add.at(inc_p, (fi, ci, V), (SG > 0).astype(np.float32)); np.add.at(inc_n, (fi, ci, V), (SG < 0).astype(np.float32))
    dp, dn = inc_p.sum(1), inc_n.sum(1); sc = 2.0 * m / n
    xv = np.stack([np.ones_like(dp), dp / sc, dn / sc, (dp - dn) / sc], -1)
    return jnp.asarray(xv), jnp.asarray(inc_p), jnp.asarray(inc_n)

def init(rng):
    def W(i, o, s=1.0): return jnp.asarray(rng.normal(0, s / np.sqrt(i), (i, o)).astype(np.float32))
    return dict(Wv=W(4, H), Wm1=W(2 * H, H), Wm2=W(2 * H, H), Wc=W(2 * H, H), wc=W(H, 4, 0.05), wp=W(H, 1, 0.5)[:, 0], bp=jnp.zeros(()),
                g=jnp.asarray(G0), b=jnp.asarray(B0), Wg=W(H, 8, 0.01))

def apply(p, xv, ip, inn):
    hv = jnp.tanh(xv @ p["Wv"])
    for Wm in (p["Wm1"], p["Wm2"]):
        cp = jnp.einsum("fmn,fnh->fmh", ip, hv); cn = jnp.einsum("fmn,fnh->fmh", inn, hv)
        hc = jnp.concatenate([cp, cn], -1) / 4.0
        mv = jnp.concatenate([jnp.einsum("fmn,fmh->fnh", ip, hc[..., :H]), jnp.einsum("fmn,fmh->fnh", inn, hc[..., H:])], -1)
        hv = jnp.tanh(hv + (mv / (ip.sum(1)[..., None] + inn.sum(1)[..., None] + 1.0)) @ Wm)
    hc = jnp.tanh(jnp.concatenate([jnp.einsum("fmn,fnh->fmh", ip, hv), jnp.einsum("fmn,fnh->fmh", inn, hv)], -1) / 4.0 @ p["Wc"])
    gb = jnp.tanh(hv.mean(1) @ p["Wg"])
    return dict(gamma=p["g"][None] + gb[:, :4], beta=p["b"][None] + gb[:, 4:], w=hc @ p["wc"], pv=jax.nn.sigmoid(hv @ p["wp"] + p["bp"]))

def train(api, budget_s):
    t0 = time.time(); rng = np.random.default_rng(0); p = init(rng)

    from jax.flatten_util import ravel_pytree
    th, unr = ravel_pytree(p); mo = jnp.zeros_like(th); ve = jnp.zeros_like(th)
    data = {n: api.data(n) for n in api.n_list}; feats = {n: features(n, d["V"], d["SG"], d["m"]) for n, d in data.items()}
    bs = {8: 16, 10: 8, 12: 8, 15: 4}
    def loss(th, n, fx, b):
        P = api.p_from(n, apply(unr(th), *fx), b); nl = -jnp.log(P.mean(1) + 1e-12)
        k = max(1, nl.shape[0] // 4); return nl.mean() + 0.5 * jax.lax.top_k(nl, k)[0].mean()
    vg = {n: jax.jit(jax.value_and_grad(lambda th, fx, b, n=n: loss(th, n, fx, b))) for n in api.n_list}
    t = 0; lr = 3e-3
    while time.time() - t0 < budget_s * 0.9:
        t += 1; g = 0
        for n in api.n_list:
            idx = rng.choice(data[n]["F"], bs.get(n, 8), replace=False); fx = tuple(x[idx] for x in feats[n])
            _, gn = vg[n](th, fx, api.batch(n, idx)); g = g + gn
        mo = 0.9 * mo + 0.1 * g; ve = 0.999 * ve + 0.001 * g * g
        th = th - lr * (mo / (1 - 0.9 ** t)) / (jnp.sqrt(ve / (1 - 0.999 ** t)) + 1e-8)
    pf = unr(th)
    def predict(n, V, SG, m):
        out = apply(pf, *features(n, V, SG, m)); return {k: np.asarray(v) for k, v in out.items()}
    return predict
# EVOLVE-BLOCK-END
