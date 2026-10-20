"""Learned per-variable guess head that reweights the least-squares projection (Zach priority 2).
GNN variable embeddings -> soft probs p_v = sigmoid(hv . Wp); project the clause cost under the product
measure prod_v Bern(p_v), which (when p_v are accurate) keeps more of the cost in the representable span.
p_v is NEVER used for readout — only for weighting the projection. Readout is still photon-number / Gray.
Ablation ladder: --bias 0 = uniform (standard chain projection); --bias 1 = learned p_v.
Warm: S_g4_snapF_s1 backbone + clause_w; base strengths from a fixed.py result (--x0).
"""
import argparse, json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path[:0] = [HERE, os.path.dirname(HERE), os.path.dirname(os.path.dirname(HERE))]
import numpy as np, jax, jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import sim, compiler as CP, nsim as NS, assign as AS, arch_nets as AN
ap = argparse.ArgumentParser(); ap.add_argument("--tag", required=True); ap.add_argument("--train", type=int, nargs="+", default=[8, 12])
ap.add_argument("--test", type=int, nargs="+", default=[8, 12, 16]); ap.add_argument("--L", type=int, default=4)
ap.add_argument("--proj", default="chain"); ap.add_argument("--assign", default="block", choices=["block", "mincut", "random"])
ap.add_argument("--bias", type=int, default=1); ap.add_argument("--bias_mlp", type=int, default=0, help="1: 2-layer MLP for p_v"); ap.add_argument("--clause_w", type=int, default=1); ap.add_argument("--pool_delta", type=int, default=1)
ap.add_argument("--warm", default="results/v3/S_g4_snapF_s1.json#0"); ap.add_argument("--x0", default=""); ap.add_argument("--theta_from", default="", help="gnn_bias json: load theta_best")
ap.add_argument("--steps", type=int, default=3000); ap.add_argument("--bs", type=int, default=32); ap.add_argument("--lr", type=float, default=3e-4)
ap.add_argument("--corr2", type=float, default=0.0); ap.add_argument("--lrbase", type=float, default=0.1); ap.add_argument("--tailw", type=float, default=0.5); ap.add_argument("--npool", type=int, default=2500); ap.add_argument("--npool20", type=int, default=400); ap.add_argument("--two_round", type=int, default=0, help="#5: half shots round 1; round-2 params from round-1 marginals (fed as var features); p_eff=(p1+p2)/2"); ap.add_argument("--shots", type=int, default=0, help="#5: if >0, round-1 marginals estimated from shots/2 samples (binomial noise)"); ap.add_argument("--corr", type=float, default=0.0, help="#3 learned per-cavity SNAP correction, L2 reg weight (0=off)"); ap.add_argument("--teacher", default=""); ap.add_argument("--kd", type=float, default=0.0); ap.add_argument("--kd_mode", default="dist"); ap.add_argument("--val_n", type=int, default=0, help="validation size (default max(train))"); ap.add_argument("--rr", type=int, default=1, help="fair round-robin: R placements per formula (mincut+R-1 random), loss -log mean_r p_r")
ap.add_argument("--nval", type=int, default=200); ap.add_argument("--es", type=int, default=250); ap.add_argument("--es_metric", default="S", choices=["S","b90","med"]); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--smoke", action="store_true"); ap.add_argument("--resw", type=float, default=0.0, help="loss weight 1+resw*(resid/mean resid)"); ap.add_argument("--Hb", type=int, default=0, help="bias MLP width (0=H)")
a = ap.parse_args()
L, H, R, FV, FM = a.L, 64, 3, 7, 32; NPL = 6; NBASE = L * NPL + 4
TRB, TEB, VAB = 31e9, 32e9, 33e9
OUT = f"results/v3/native/gnnw/{a.tag}.json"; os.makedirs(os.path.dirname(OUT), exist_ok=True)
log = lambda *s: print(time.strftime("%H:%M:%S"), *s, flush=True)

def graph(n, seed):
    m, cl, _ = sim.gen_instance(n, seed); Sg = np.zeros((m, n), np.float32)
    for i, (v, s) in enumerate(cl):
        for vv, ss in zip(v, s): Sg[i, vv] = ss
    Ip, In = (Sg > 0).astype(np.float32), (Sg < 0).astype(np.float32)
    fv = np.zeros((n, FV), np.float32); fv[:, 0] = 1; fv[:, 1] = Ip.sum(0) / (2 * m / n); fv[:, 2] = In.sum(0) / (2 * m / n)
    V = np.array([v for v, s in cl], np.int32); Bb = np.array([[0 if ss > 0 else 1 for ss in s] for v, s in cl], np.int8)
    return dict(Ip=Ip, In=In, fv=fv, deg=np.maximum(Ip.sum(0) + In.sum(0), 1)), V, Bb, m, cl

def perm_of(n, seed, cl, rng=None):
    if a.assign == "block" or len(CP.LAYOUTS[n]) <= 2: return np.arange(n)
    if a.assign == "random": return (rng or np.random.default_rng(int(seed) % 2**32)).permutation(n)
    f = f"results/v3/native/perm_cache/{int(seed)}.npy"
    if os.path.exists(f): return np.load(f)
    perm, _ = AS.anneal(n, cl, CP.LAYOUTS[n], iters=3000 if n < 16 else 5000); np.save(f, perm); return perm

PACK = False; LAZY = False
def lazy_instances(n, count, base):
    """memory-lean sim.instances: yields one at a time (E dropped after build uses it)"""
    k = c = 0
    while c < count:
        seed = int(base) + 10_000_000 * n + k; k += 1
        m, cl, E = sim.gen_instance(n, seed)
        if E.min() == 0: c += 1; yield dict(seed=seed, m=m, E=E)
def unpackS(D, n):
    if D["S"].dtype == np.uint8: D = dict(D); D["S"] = np.unpackbits(D["S"], axis=-1)[..., : 1 << n].astype(np.float32)
    return D
def _entries(insts, n, rng, R):
    for ins in insts:
        for r in range(R): yield ins, r, (r == R - 1)
def build(n, insts, rng=None, R=None):
    R = R or a.rr; gs, VP, BB, SS, PERM, RES = [], [], [], [], [], []
    for ins, r, last in _entries(insts, n, rng, R):
        G, V, Bb, m, cl = graph(n, ins["seed"])
        perm = perm_of(n, ins["seed"], cl, rng) if r == 0 else np.random.default_rng(int(ins["seed"]) * 31 + r).permutation(n)
        pos = np.argsort(perm)
        gs.append(G); VP.append(pos[V]); BB.append(Bb); PERM.append(perm); m0 = ins["m"]
        cavp = np.repeat(np.arange(len(CP.LAYOUTS[n])), CP.LAYOUTS[n]); kk_, w2_ = AS.walsh_terms(n, cl)
        RES.append(1 - (w2_ * np.array([AS.rep_ok([cavp[pos[v]] for v in t]) for t in kk_])).sum() / w2_.sum())
        E = ins["E"].astype(np.float64); Sv = (CP.permute_vars(E, n, perm) == 0)
        SS.append(np.packbits(Sv) if PACK else Sv.astype(np.float32)); ins.pop("E", None) if (LAZY and last) else None
    D = {k: np.stack([g[k] for g in gs]) for k in gs[0]}; D["VP"] = np.stack(VP); D["BB"] = np.stack(BB)
    D["S"] = np.stack(SS); D["PERM"] = np.stack(PERM); D["RES"] = np.array(RES, np.float32); D["nm"] = n / m0; return D

XB = {}
def bits(n):
    if n not in XB: XB[n] = ((np.arange(1 << n)[:, None] >> (n - 1 - np.arange(n))[None]) & 1).astype(np.int8)
    return XB[n]

def init_params(rng):
    def W(i, o, s=1.0): return jnp.asarray(rng.normal(0, s / np.sqrt(i), (i, o)).astype(np.float32))
    p = AN.init_backbone(rng, "mp", H, R, FV); p["G1"] = W(2 * H, H); p["g1"] = jnp.zeros(H); p["G2"] = jnp.zeros((H, NBASE))
    p["Wc"] = jnp.zeros((H, L)); p["Wp"] = W(H, 1, 2.0).reshape(H); p["bp"] = jnp.zeros(());
    if a.bias_mlp: HB = a.Hb or H; p["Wb1"] = W(H, HB); p["bb1"] = jnp.zeros(HB); p["Wp2"] = W(HB, 1, 2.0).reshape(HB)
    p["base"] = jnp.zeros(NBASE); return p

def old_template():
    rng = np.random.default_rng(0)
    def W(i, o, s=1.0): return jnp.asarray(rng.normal(0, s / np.sqrt(i), (i, o)).astype(np.float32))
    p = AN.init_backbone(rng, "mp", H, R, FV); p["G1"] = W(2 * H, H); p["g1"] = jnp.zeros(H); p["G2"] = jnp.zeros((H, NBASE))
    p["Wc"] = jnp.zeros((H, L)); p["Wv"] = jnp.zeros((H, L + 1)); p["P0"] = W(H, FM, 2.0); p["P1"] = W(H, FM, 2.0)
    p["u0"] = jnp.zeros(FM); p["u1"] = jnp.zeros(FM); p["fq"] = jnp.zeros(FM); p["Wo"] = jnp.zeros((FM, L))
    return (p, jnp.zeros(NBASE, jnp.float32))

MIXK = {k: NS.Mixer(k, "gray") for k in (3, 4, 5)}

def project_product_j(E, p, n, mode, ks):
    """JAX product-measure projection of E (B, 2^n) with per-instance p (B, n). Orthogonal biased-parity basis."""
    T = E.reshape((E.shape[0],) + (2,) * n); p = jnp.clip(p, 1e-4, 1 - 1e-4)
    for i in range(n):
        pi = p[:, i]; m = 1 - 2 * pi; sd = jnp.sqrt(1 - m * m); ph0, ph1 = (1 - m) / sd, (-1 - m) / sd
        T = jnp.moveaxis(T, i + 1, 1); rest = T.shape[2:]
        pi_ = pi.reshape((pi.shape[0],) + (1,) * len(rest)); ph0_ = ph0.reshape(pi_.shape); ph1_ = ph1.reshape(pi_.shape)
        t0, t1 = T[:, 0], T[:, 1]
        c0 = (1 - pi_) * t0 + pi_ * t1
        c1 = ((1 - pi_) * ph0_) * t0 + (pi_ * ph1_) * t1
        T = jnp.moveaxis(jnp.stack([c0, c1], 1), 1, i + 1)
    T = T.reshape(E.shape[0], -1) * jnp.asarray(CP.rep_mask(ks, mode).astype(np.float32))[None]
    T = T.reshape((E.shape[0],) + (2,) * n)
    for i in range(n):
        pi = p[:, i]; m = 1 - 2 * pi; sd = jnp.sqrt(1 - m * m); ph0, ph1 = (1 - m) / sd, (-1 - m) / sd
        T = jnp.moveaxis(T, i + 1, 1); rest = T.shape[2:]
        ph0_ = ph0.reshape((pi.shape[0],) + (1,) * len(rest)); ph1_ = ph1.reshape(ph0_.shape)
        c0, c1 = T[:, 0], T[:, 1]
        T = jnp.moveaxis(jnp.stack([c0 + ph0_ * c1, c0 + ph1_ * c1], 1), 1, i + 1)
    return T.reshape(E.shape[0], -1)

def fwht_j(v, n):
    B = v.shape[0]; t = v.reshape((B,) + (2,) * n)
    for i in range(n):
        t0, t1 = jnp.take(t, 0, axis=i + 1), jnp.take(t, 1, axis=i + 1); t = jnp.stack([t0 + t1, t0 - t1], axis=i + 1)
    return t.reshape(B, -1)

def forward(p, D, n, proj=None, ret_psi=False):
    if a.two_round and not D.get("_r2", False):
        o1 = forward_one(p, D, n, proj, True); q1 = o1[-1]
        mc = q1 @ jnp.asarray(bits(n), jnp.float32)                                   # chain-order marginals P(x=1)
        if a.shots > 0:
            key = jax.random.PRNGKey(0); mc = jnp.clip(mc + jax.random.normal(key, mc.shape) * jnp.sqrt(mc * (1 - mc) / (a.shots / 2)), 0, 1)
        mv = jnp.take_along_axis(mc, jnp.argsort(D["PERM"].astype(jnp.int32), 1), 1)   # var order
        mv = jax.lax.stop_gradient(mv)
        fv2 = D["fv"].at[..., 3].set(mv).at[..., 4].set((2 * mv - 1) ** 2)
        o2 = forward_one(p, dict(D, fv=fv2, _r2=True), n, proj, True)
        peff = 0.5 * (o1[0] + o2[0])
        return (peff,) + tuple(o2[1:4]) + ((0.5 * (o1[-1] + o2[-1]),) if ret_psi else ())
    return forward_one(p, D, n, proj, ret_psi)
def forward_one(p, D, n, proj=None, ret_psi=False):
    proj = proj or a.proj
    ks = CP.LAYOUTS[n]; G = {k: D[k] for k in ("Ip", "In", "fv", "deg")}; B = D["S"].shape[0]
    hv, hc = AN.run_backbone(p, G, "mp", H); g_ = jnp.concatenate([hv.mean(1), hc.mean(1)], -1)
    bv = p["base"][None] + (jnp.tanh(g_ @ p["G1"] + p["g1"]) @ p["G2"] if a.pool_delta else 0.0)
    X = bv[:, : L * NPL].reshape(B, L, NPL); bl, gl = X[:, :, 4], X[:, :, 5]
    w = hc @ p["Wc"] if a.clause_w else jnp.zeros(hc.shape[:2] + (L,))
    # p_v in formula-var order, then permute to chain order via PERM (chain pos q holds var PERM[q])
    if a.bias:
        hpv = jnp.tanh(hv @ p["Wb1"] + p["bb1"]) if a.bias_mlp else hv
        Wp = p["Wp2"] if a.bias_mlp else p["Wp"]
        pv_f = jax.nn.sigmoid(hpv @ Wp + p["bp"])
    else: pv_f = jnp.full_like(hv[..., 0], 0.5)
    pv = jnp.take_along_axis(pv_f, D["PERM"].astype(jnp.int32), axis=1) if a.bias else pv_f   # chain order
    xb = jnp.asarray(bits(n)); lit = xb[:, D["VP"]]; viol = jnp.prod((lit == D["BB"][None].astype(jnp.int8)).astype(jnp.float32), -1)
    psi = NS.uniform_psi0(ks, B)
    for l in range(L):
        Ew = D["nm"] * jnp.einsum("xbm,bm->bx", viol, gl[:, l][:, None] + w[:, :, l])
        if proj != "full":
            if a.bias: Ew = project_product_j(Ew, pv, n, proj, ks)
            else: Ew = fwht_j(fwht_j(Ew, n) * jnp.asarray(CP.rep_mask(ks, proj).astype(np.float32))[None], n) / (1 << n)
        psi = psi * jnp.exp(-1j * Ew.reshape(psi.shape)).astype(jnp.complex64)
        if "Wsn" in p:
            hvc = jnp.take_along_axis(hv, D["PERM"].astype(jnp.int32)[..., None], axis=1); off = 0
            for c, k in enumerate(ks):
                tab = (hvc[:, off:off + k].mean(1) @ p["Wsn"][l])[:, : 1 << k]; off += k
                sh = [B] + [1] * len(ks); sh[c + 1] = 1 << k
                psi = psi * jnp.exp(-1j * tab.reshape(sh)).astype(jnp.complex64)
        if "Acs" in p:   # SNAP + neighbour-bus correction from signed digit patterns of the cavity's variables (size-free)
            if l == 0:
                hvc = jnp.take_along_axis(hv, D["PERM"].astype(jnp.int32)[..., None], axis=1); k0 = ks[0]
                ib = np.arange(1 << k0)[:, None]; ST = jnp.asarray((1.0 - 2.0 * ((ib >> (k0 - 1 - np.arange(k0))[None]) & 1)).astype(np.float32))
                Z = jnp.stack([jnp.einsum("kj,bjh->bkh", ST, hvc[:, c * k0:(c + 1) * k0]) for c in range(len(ks))], 1)  # (B,C,2^k,H)
                US, UB = jnp.tanh(Z @ p["Acs"]), jnp.tanh(Z @ p["Acb"])
            Ec = jnp.zeros((B,) + (1 << k0,) * len(ks))
            for c in range(len(ks)):
                sh = [B] + [1] * len(ks); sh[c + 1] = 1 << k0; Ec = Ec + (US[:, c] @ p["acs"][l]).reshape(sh)
            for c in range(len(ks) - 1):
                sh = [B] + [1] * len(ks); sh[c + 1] = sh[c + 2] = 1 << k0
                Ec = Ec + jnp.einsum("bkh,bqh,h->bkq", UB[:, c], UB[:, c + 1], p["acb"][l]).reshape(sh)
            psi = psi * jnp.exp(-1j * Ec.reshape(psi.shape)).astype(jnp.complex64)
        for c, k in enumerate(ks): psi = NS.apply_cav(MIXK[k](bl[:, l]), psi, c, len(ks))
    pr_ = jnp.abs(psi.reshape(B, -1)) ** 2
    if ret_psi: return jnp.sum(pr_ * D["S"], 1), bl, gl, pv_f, pr_
    return jnp.sum(pr_ * D["S"], 1), bl, gl, pv_f

TP = None
def lossf(th, D, n, unr):
    if TP is not None and a.kd > 0:
        out = forward(unr(th), D, n, ret_psi=True); pr, qs = out[0], out[-1]
        pt = jax.lax.stop_gradient(forward(TP, D, n, proj="full", ret_psi=True)[-1])
        if a.kd_mode == "dist": kl = jnp.sum(pt * (jnp.log(pt + 1e-12) - jnp.log(qs + 1e-12)), 1)
        else:
            xb = jnp.asarray(bits(n), jnp.float32); mt = jnp.clip(pt @ xb, 1e-4, 1 - 1e-4); ms = jnp.clip(qs @ xb, 1e-4, 1 - 1e-4)
            kl = jnp.sum(mt * jnp.log(mt / ms) + (1 - mt) * jnp.log((1 - mt) / (1 - ms)), 1)
        kd_term = a.kd * jnp.mean(kl)
    else:
        pr = forward(unr(th), D, n)[0]; kd_term = 0.0
    if a.rr > 1: pr = pr.reshape(-1, a.rr).mean(1)
    nl = -jnp.log(pr + 1e-12); k = max(1, nl.shape[0] // 10)
    if a.resw > 0 and a.rr == 1: nl = nl * (1 + a.resw * D["RES"] / (jnp.mean(D["RES"]) + 1e-6))
    reg = a.corr * jnp.mean(unr(th)["Wsn"] ** 2) * 1e3 if a.corr > 0 else 0.0
    if a.corr2 > 0: q_ = unr(th); reg = reg + a.corr2 * (jnp.mean(q_["acs"] ** 2) + jnp.mean(q_["acb"] ** 2)) * 1e2
    return jnp.mean(nl) + a.tailw * jnp.mean(jax.lax.top_k(nl, k)[0]) + kd_term + reg


if __name__ == "__main__":
    rng = np.random.default_rng(1000 + a.seed); p0 = init_params(rng)
    if a.warm:
        pth, k = a.warm.split("#"); Jw = json.load(open(pth))
        try:
            po, bo = ravel_pytree(old_template())[1](jnp.asarray(Jw["runs"][int(k)]["theta"], jnp.float32))
            for kk in list(p0):
                if kk in po and jnp.asarray(p0[kk]).shape == jnp.asarray(po[kk]).shape: p0[kk] = po[kk]
            if jnp.asarray(bo).shape == jnp.asarray(p0["base"]).shape: p0["base"] = bo
            else:
                b = np.asarray(p0["base"]).copy(); bs = np.asarray(bo); m = min(len(b), len(bs)); b[:m] = bs[:m]; p0["base"] = jnp.asarray(b)
            log(f"warm from {a.warm}")
        except Exception as e: log(f"warm skip: {e}")
    if a.x0:
        x = np.array(json.load(open(a.x0))["x"]); b = np.asarray(p0["base"]).copy(); b[4:L * NPL:NPL] = x[0::2]; b[5:L * NPL:NPL] = x[1::2]
        p0["base"] = jnp.asarray(b)
        if a.warm: p0["G2"] = jnp.zeros_like(p0["G2"])
    if a.theta_from:
            Jt = json.load(open(a.theta_from)); th_load = jnp.asarray(Jt["theta_best"], jnp.float32)
            th0s, unr0 = ravel_pytree(p0)
            if th_load.size == th0s.size: p0 = unr0(th_load); log(f"theta_from {a.theta_from}")
            else: log(f"theta_from size mismatch {th_load.size} vs {th0s.size}; skip")
    
    if a.corr2 > 0:
        r_ = np.random.default_rng(77); p0["Acs"] = jnp.asarray(r_.normal(0, 1 / np.sqrt(H), (H, H)), jnp.float32); p0["Acb"] = jnp.asarray(r_.normal(0, 1 / np.sqrt(H), (H, H)), jnp.float32)
        p0["acs"] = jnp.zeros((L, H)); p0["acb"] = jnp.zeros((L, H)); log(f"SNAP+bus pattern correction on, reg {a.corr2}")
    if a.corr > 0: p0["Wsn"] = jnp.zeros((L, H, 32), jnp.float32); log(f"#3 SNAP correction on, reg {a.corr}")
    th0, unr = ravel_pytree(p0)
    if a.teacher:
        Jt_ = json.load(open(a.teacher)); tt_ = jnp.asarray(Jt_["theta_best"], jnp.float32); assert tt_.size == th0.size, ("teacher arch mismatch", tt_.size, th0.size)
        globals()["TP"] = unr(tt_); log(f"teacher {a.teacher} kd={a.kd} mode={a.kd_mode}")
    LRM = ravel_pytree({k: jax.tree_util.tree_map(lambda u, k=k: jnp.full_like(jnp.asarray(u), a.lrbase if k == "base" else 1.0), v) for k, v in p0.items()})[0]
    log(f"gnn_bias tag={a.tag} train {a.train} proj={a.proj} assign={a.assign} bias={a.bias} clause_w={a.clause_w} params {th0.size}")
    VG = {}; EVJ = {}
    def vg(n):
        if n not in VG: VG[n] = jax.jit(jax.value_and_grad(lambda th, D: lossf(th, D, n, unr)))
        return VG[n]
    def evaluate(th, D, n, ch):
        if n not in EVJ: EVJ[n] = jax.jit(lambda th, Dc: forward(unr(th), Dc, n))
        f = EVJ[n]; B = D["S"].shape[0]; ps, bs, gs, pvs = [], [], [], []
        for s in range(0, B, ch):
            Dc = unpackS({k: (v[s:s + ch] if k != "nm" else v) for k, v in D.items()}, n); pr, bl, gl, pv = f(th, Dc)
            ps.append(np.asarray(pr)); bs.append(np.asarray(bl)); gs.append(np.asarray(gl)); pvs.append(np.asarray(pv))
        return np.concatenate(ps).astype(float), np.concatenate(bs), np.concatenate(gs), np.concatenate(pvs)
    if a.smoke:
        for n in a.train:
            D = build(n, sim.instances(n, a.bs, TRB)); t = time.time(); v, g = vg(n)(th0, D); g.block_until_ready(); t1 = time.time() - t
            t = time.time(); v, g = vg(n)(th0, D); g.block_until_ready()
            log(f"smoke n={n}: loss {float(v):.4f} finite {bool(jnp.all(jnp.isfinite(g)))} compile {t1:.1f}s step {time.time()-t:.2f}s")
        for n in a.test[:2]:
            D = build(n, sim.instances(n, 4, TEB)); t = time.time(); pr, bl, gl, pv = evaluate(th0, D, n, 2)
            log(f"smoke eval n={n}: p {np.round(pr, 4)} pv mean {pv.mean():.3f} ({time.time()-t:.1f}s)")
        sys.exit(0)
    pools = {}
    for n in a.train:
        globals()["PACK"] = globals()["LAZY"] = n >= 18; npl = a.npool if n < 18 else min(a.npool, a.npool20)
        t = time.time(); pools[n] = build(n, lazy_instances(n, npl, TRB)); log(f"pool n={n} {a.npool} built {time.time()-t:.0f}s")
    nv = a.val_n or max(a.train); globals()["PACK"] = globals()["LAZY"] = nv >= 18; Dv = build(nv, lazy_instances(nv, a.nval, VAB)); globals()["PACK"] = globals()["LAZY"] = False
    def es_val(th):
        pr, _, _, pv = evaluate(th, Dv, nv, 50 if nv <= 12 else (10 if nv <= 16 else 1)); s = NS.summ(pr.reshape(-1, a.rr).mean(1)); s["single"] = NS.summ(pr[:: a.rr]); s["pv_mean"] = float(pv.mean()); return s
    @jax.jit
    def adam(x, m, v, g, t, lr):
        m = 0.9 * m + 0.1 * g; v = 0.999 * v + 0.001 * g * g; return x - lr * (m / (1 - 0.9 ** t)) / (jnp.sqrt(v / (1 - 0.999 ** t)) + 1e-8), m, v
    x = th0; mom = jnp.zeros_like(x); vel = jnp.zeros_like(x); rb = np.random.default_rng(77 + a.seed)
    s0 = es_val(x)
    def es_key(s):
        if a.es_metric == "b90": return float(np.log(s["B90_inv_p"]))
        if a.es_metric == "med": return float(np.log(s["median_inv_p"]))
        return s["S"]
    BEST = dict(S=es_key(s0), x=np.asarray(x), it=0, summ=s0); HIST = [[0, s0]]; log(f"step 0 val n={nv} {s0['median_inv_p']:.2f}/{s0['B90_inv_p']:.2f} S {s0['S']:.4f} pv {s0['pv_mean']:.3f} metric={a.es_metric}")
    rl = []; t0 = time.time()
    for t in range(1, a.steps + 1):
        lr = a.lr * (0.1 + 0.9 * 0.5 * (1 + np.cos(np.pi * t / a.steps))); gt = 0; vs = 0
        for n in a.train:
            P = pools[n]; fi = rb.choice(P["S"].shape[0] // a.rr, a.bs, replace=False); idx = (fi[:, None] * a.rr + np.arange(a.rr)[None]).ravel()
            Dn = unpackS({k: (v[idx] if k != "nm" else v) for k, v in P.items()}, n)
            v, g = vg(n)(x, Dn); gt = gt + g; vs += float(v)
        x, mom, vel = adam(x, mom, vel, gt, t, lr * LRM); rl.append(vs / len(a.train))
        if t % a.es == 0 or t == a.steps:
            s_ = es_val(x); HIST.append([t, s_])
            if es_key(s_) < BEST["S"]: BEST.update(S=es_key(s_), x=np.asarray(x), it=t, summ=s_)
            log(f"step {t} loss {np.mean(rl[-a.es:]):.4f} lr {lr:.1e} val {s_['median_inv_p']:.2f}/{s_['B90_inv_p']:.2f} S {s_['S']:.4f} pv {s_['pv_mean']:.3f} (best {BEST['S']:.4f} @ {BEST['it']}) [{(time.time()-t0)/t:.2f}s/step]")
            json.dump({"args": vars(a), "hist": HIST, "best_it": BEST["it"], "theta_best": BEST["x"].tolist()}, open(OUT, "w"))
    res = {"args": vars(a), "hist": HIST, "best_it": BEST["it"], "theta_best": BEST["x"].tolist(), "test": {}, "random": {}}
    thb = jnp.asarray(BEST["x"])
    for n in a.test:
        ins = sim.instances(n, 200 if n <= 8 else 100, TEB); globals()["PACK"] = n >= 18; D = build(n, ins); globals()["PACK"] = False; t = time.time()
        pr, bl, gl, pv = evaluate(thb, D, n, 25 if n <= 12 else (4 if n <= 16 else 1))
        pr_all = pr; pr = pr_all[:: a.rr]; bl, gl, pv = bl[:: a.rr], gl[:: a.rr], pv[:: a.rr]; rrs = NS.summ(pr_all.reshape(-1, a.rr).mean(1))
        # polarity accuracy vs a solution (diagnostic)
        acc = []
        for i, ins_i in enumerate(ins):
            sol = np.flatnonzero(ins_i["E"] == 0)[0]; bits_s = np.array([(sol >> (n - 1 - v)) & 1 for v in range(n)])
            acc.append(np.mean((pv[i] > 0.5) == (bits_s == 1)))
        res["test"][str(n)] = NS.summ(pr) | {"p": pr.tolist(), "pv_mean": float(pv.mean()), "pv_acc": float(np.mean(acc)),
                                              "b_mean": bl.mean(0).tolist(), "g_mean": gl.mean(0).tolist(), "rr": rrs, "p_all": pr_all.tolist()}; res["random"][str(n)] = NS.rand_stats(ins, n)
        log(f"TEST n={n}: single {res['test'][str(n)]['median_inv_p']:.2f}/{res['test'][str(n)]['B90_inv_p']:.2f} | rr{a.rr} {rrs['median_inv_p']:.2f}/{rrs['B90_inv_p']:.2f} pv_acc {np.mean(acc):.3f} ({time.time()-t:.0f}s)")
        json.dump(res, open(OUT, "w"))
    log("DONE")
