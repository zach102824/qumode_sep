"""FROZEN hardware gate sequence + compiler + simulator + readout (hash-locked; do not edit).
Per layer l (L=4): SNAP + neighbour-bus phases = compiled least-squares projection (product measure p_v, chain support)
of nm*gamma_l*sum_j (gamma-shared + w_jl) [clause j violated]  ->  per-cavity mixer M(beta_l) (ideal one-bit-flip
generator, the target the calibrated SNAP-D-SNAP sequence is fitted to; see v3/native/calib.py). Flat prep, readout p = P(sat).
Same convention as v3/native/gnn_bias.forward: Ew = nm * viol @ (gamma_l + w_jl).
Network outputs (per formula): gamma (L,), beta (L,), w (m, L), pv (n,) [var order, P(x_v=1)].
"""
import os, sys, numpy as np, jax, jax.numpy as jnp
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
for p in (ROOT, os.path.join(ROOT, "v3", "native")):
    if p not in sys.path: sys.path.insert(0, p)
import sim, compiler as CP, nsim as NS, assign as AS
L = 4
TRB, TEB, VAB = 31e9, 32e9, 33e9
MIXK = {k: NS.Mixer(k, "gray") for k in (3, 4, 5)}
_XB = {}
def bits(n):
    if n not in _XB: _XB[n] = ((np.arange(1 << n)[:, None] >> (n - 1 - np.arange(n))[None]) & 1).astype(np.int8)
    return _XB[n]
def fwht_j(v, n):
    B = v.shape[0]; t = v.reshape((B,) + (2,) * n)
    for i in range(n):
        t0, t1 = jnp.take(t, 0, axis=i + 1), jnp.take(t, 1, axis=i + 1); t = jnp.stack([t0 + t1, t0 - t1], axis=i + 1)
    return t.reshape(B, -1)
def project_product_j(E, p, n, ks):
    T = E.reshape((E.shape[0],) + (2,) * n); p = jnp.clip(p, 1e-4, 1 - 1e-4)
    for i in range(n):
        pi = p[:, i]; m = 1 - 2 * pi; sd = jnp.sqrt(1 - m * m); ph0, ph1 = (1 - m) / sd, (-1 - m) / sd
        T = jnp.moveaxis(T, i + 1, 1); rest = T.shape[2:]
        pi_ = pi.reshape((pi.shape[0],) + (1,) * len(rest)); ph0_ = ph0.reshape(pi_.shape); ph1_ = ph1.reshape(pi_.shape)
        t0, t1 = T[:, 0], T[:, 1]
        T = jnp.moveaxis(jnp.stack([(1 - pi_) * t0 + pi_ * t1, ((1 - pi_) * ph0_) * t0 + (pi_ * ph1_) * t1], 1), 1, i + 1)
    T = T.reshape(E.shape[0], -1) * jnp.asarray(CP.rep_mask(ks, "chain").astype(np.float32))[None]
    T = T.reshape((E.shape[0],) + (2,) * n)
    for i in range(n):
        pi = p[:, i]; m = 1 - 2 * pi; sd = jnp.sqrt(1 - m * m); ph0, ph1 = (1 - m) / sd, (-1 - m) / sd
        T = jnp.moveaxis(T, i + 1, 1); rest = T.shape[2:]
        ph0_ = ph0.reshape((pi.shape[0],) + (1,) * len(rest)); ph1_ = ph1.reshape(ph0_.shape)
        c0, c1 = T[:, 0], T[:, 1]
        T = jnp.moveaxis(jnp.stack([c0 + ph0_ * c1, c0 + ph1_ * c1], 1), 1, i + 1)
    return T.reshape(E.shape[0], -1)
def circuit_p(n, gamma, beta, w, pv, VP, BB, PERM, S, nm):
    """batched (B entries). gamma,beta (B,L); w (B,m,L); pv (B,n) var order; VP chain positions (B,m,4); BB (B,m,4) violating bit;
    PERM (B,n); S (B,2^n) chain-order sat indicator. Returns p (B,)."""
    ks = CP.LAYOUTS[n]; B = S.shape[0]
    pvc = jnp.take_along_axis(pv, PERM.astype(jnp.int32), axis=1)
    xb = jnp.asarray(bits(n)); lit = xb[:, VP]; viol = jnp.prod((lit == BB[None].astype(jnp.int8)).astype(jnp.float32), -1)
    psi = NS.uniform_psi0(ks, B)
    for l in range(L):
        Ew = nm * jnp.einsum("xbm,bm->bx", viol, gamma[:, l][:, None] + w[:, :, l])
        Ew = project_product_j(Ew, pvc, n, ks)
        psi = psi * jnp.exp(-1j * Ew.reshape(psi.shape)).astype(jnp.complex64)
        for c, k in enumerate(ks): psi = NS.apply_cav(MIXK[k](beta[:, l]), psi, c, len(ks))
    return jnp.sum(jnp.abs(psi.reshape(B, -1)) ** 2 * S, 1)
def make_set(n, count, base, R):
    """formulas + R placements each (r=0 min-cut, r>0 seeded random, same as gnn_bias --rr). S packed, chain order."""
    out = dict(V=[], SG=[], PERM=[], S=[], seed=[], sol=[]); m0 = None; k = c = 0
    while c < count:
        seed = int(base) + 10_000_000 * n + k; k += 1
        m, cl, E = sim.gen_instance(n, seed)
        if E.min() != 0: continue
        c += 1; m0 = m; V = np.array([v for v, s in cl], np.int32); SG = np.array([s for v, s in cl], np.int8)
        perms = []
        for r in range(R):
            if r == 0:
                if len(CP.LAYOUTS[n]) <= 2: perms.append(np.arange(n))
                else:
                    f = os.path.join(ROOT, f"results/v3/native/perm_cache/{int(seed)}.npy")
                    if os.path.exists(f): perms.append(np.load(f))
                    else: pm, _ = AS.anneal(n, cl, CP.LAYOUTS[n], iters=3000 if n < 16 else 5000); np.save(f, pm); perms.append(pm)
            else: perms.append(np.random.default_rng(int(seed) * 31 + r).permutation(n))
        Ss = [np.packbits(CP.permute_vars(E.astype(np.float64), n, pm) == 0) for pm in perms]
        sol = np.flatnonzero(E == 0)[:64]
        out["V"].append(V); out["SG"].append(SG); out["PERM"].append(np.stack(perms)); out["S"].append(np.stack(Ss)); out["seed"].append(seed)
        out["sol"].append(((sol[:, None] >> (n - 1 - np.arange(n))[None]) & 1).astype(np.int8))
    D = {k2: np.stack(v) for k2, v in out.items() if k2 not in ("sol",)}; D["sol"] = out["sol"]; D["m"] = m0; D["n"] = n; D["R"] = R
    return D
def entries(D, fidx, outs):
    """expand per-formula outputs to (formula, r) entries; returns args for circuit_p."""
    n, R = D["n"], D["R"]; fidx = np.asarray(fidx); F = len(fidx)
    PERM = D["PERM"][fidx].reshape(F * R, n); pos = np.argsort(PERM, 1)
    V = np.repeat(D["V"][fidx], R, 0); VP = np.take_along_axis(pos[:, None, :].repeat(V.shape[1], 1), V, 2)
    BB = (np.repeat(D["SG"][fidx], R, 0) < 0).astype(np.int8)
    S = np.unpackbits(D["S"][fidx].reshape(F * R, -1), axis=-1)[:, : 1 << n].astype(np.float32)
    rep = lambda x: jnp.repeat(jnp.asarray(x), R, 0)
    return (rep(outs["gamma"]), rep(outs["beta"]), rep(outs["w"]), rep(outs["pv"]), jnp.asarray(VP), jnp.asarray(BB), jnp.asarray(PERM), jnp.asarray(S), n / D["m"])
