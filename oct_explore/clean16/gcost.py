# clean16 copy of v3/native/gnnw.py (G16_full_w recipe: --proj full, block, clause_w, pool_delta, --x0 QAOA angles); changes: fresh bases, no warm start from old weights, test 8..20 with 100 formulas.
"""Milestone 4: learned strengths + GNN clause weights on the templated native chain ansatz.
Per layer l:  compiled clause phases  Phi_l(x) = Proj_chain[ nm * sum_j (g_l + w_jl) viol_j(x) ]   (SNAP + neighbour bus tables;
              non-neighbour / 3-cavity terms least-squares projected; --proj full = all-to-all reference)
              -> calibrated one-bit-flip mixer per cavity, strength b_l (ideal matrix during training; native calibrated at eval).
Learned: g_l, b_l (shared base + optional per-formula delta from the pooled GNN embedding, --pool_delta) and clause weights
w_jl = hc_j . Wc[:, l] (--clause_w). Network = arch_nets 'mp' backbone, formula features (as S_g4_snapF_s1); --warm loads
backbone/G1/g1/G2/Wc and the base vector from S_g4_snapF_s1 (old layout [bre,bim,th,ph,bc,g]*L + 4: b_l = bc_l, g_l = g_l).
Training: Adam over fresh pools per train size, loss mean -log p + tailw * mean worst 10%, ES on val (33e9) of the largest train size.
Assignment: block | mincut (cached perms, v3/native/assign.py) | random (augmentation: fresh random perm per minibatch formula).
"""
import argparse, json, os, sys, time
HERE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'v3', 'native'); sys.path[:0] = [HERE, os.path.dirname(HERE), os.path.dirname(os.path.dirname(HERE))]
import numpy as np, jax, jax.numpy as jnp
from jax.flatten_util import ravel_pytree
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), 'clean16')); import common as CM
import sim, compiler as CP, nsim as NS, assign as AS, arch_nets as AN
ap = argparse.ArgumentParser(); ap.add_argument("--tag", required=True); ap.add_argument("--train", type=int, nargs="+", default=[8, 12])
ap.add_argument("--test", type=int, nargs="+", default=[8, 12, 16, 20]); ap.add_argument("--L", type=int, default=4)
ap.add_argument("--proj", default="chain"); ap.add_argument("--assign", default="block", choices=["block", "mincut", "random"])
ap.add_argument("--clause_w", type=int, default=1); ap.add_argument("--pool_delta", type=int, default=1)
ap.add_argument("--warm", default=""); ap.add_argument("--x0", default="", help="fixed.py json: base g/b from its x (overrides warm base)")
ap.add_argument("--steps", type=int, default=3000); ap.add_argument("--bs", type=int, default=32); ap.add_argument("--lr", type=float, default=3e-4)
ap.add_argument("--lrbase", type=float, default=0.1); ap.add_argument("--tailw", type=float, default=0.5); ap.add_argument("--npool", type=int, default=5000)
ap.add_argument("--nval", type=int, default=200); ap.add_argument("--es", type=int, default=250); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--smoke", action="store_true")
a = ap.parse_args()
L, H, R, FV, FM = a.L, 64, 3, 7, 32; NPL = 6; NBASE = L * NPL + 4
TRB, TEB, VAB = CM.TRB, CM.TEB, CM.VAB
OUT = f"{CM.RES}/{a.tag}.json"; os.makedirs(os.path.dirname(OUT), exist_ok=True)
log = lambda *s: print(time.strftime("%H:%M:%S"), *s, flush=True)

def graph(n, seed):
    m, cl, _ = sim.gen_instance(n, seed); Sg = np.zeros((m, n), np.float32)
    for i, (v, s) in enumerate(cl):
        for vv, ss in zip(v, s): Sg[i, vv] = ss
    Ip, In = (Sg > 0).astype(np.float32), (Sg < 0).astype(np.float32)
    fv = np.zeros((n, FV), np.float32); fv[:, 0] = 1; fv[:, 1] = Ip.sum(0) / (2 * m / n); fv[:, 2] = In.sum(0) / (2 * m / n)
    V = np.array([v for v, s in cl], np.int32); Bb = np.array([[0 if ss > 0 else 1 for ss in s] for v, s in cl], np.int8)   # violated iff x_v == Bb
    return dict(Ip=Ip, In=In, fv=fv, deg=np.maximum(Ip.sum(0) + In.sum(0), 1)), V, Bb, m, cl

def perm_of(n, seed, cl, rng=None):
    if a.assign == "block": return np.arange(n)
    if a.assign == "random": return (rng or np.random.default_rng(int(seed) % 2**32)).permutation(n)
    if len(CP.LAYOUTS[n]) <= 2: return np.arange(n)   # 2 cavities: everything representable, assignment irrelevant
    f = f"results/v3/native/perm_cache/{int(seed)}.npy"
    if os.path.exists(f): return np.load(f)
    perm, _ = AS.anneal(n, cl, CP.LAYOUTS[n], iters=3000 if n < 16 else 5000); np.save(f, perm); return perm

def build(n, insts, rng=None):
    """-> dict of numpy arrays: graph, chain-position clause vars VP (B,m,4), bad bits, success mask in chain order."""
    gs, VP, BB, SS = [], [], [], []
    for ins in insts:
        G, V, Bb, m, cl = graph(n, ins["seed"]); perm = perm_of(n, ins["seed"], cl, rng); pos = np.argsort(perm)
        gs.append(G); VP.append(pos[V]); BB.append(Bb)
        E = ins["E"].astype(np.float64); SS.append((CP.permute_vars(E, n, perm) == 0).astype(np.float32))
    D = {k: np.stack([g[k] for g in gs]) for k in gs[0]}; D["VP"] = np.stack(VP); D["BB"] = np.stack(BB); D["S"] = np.stack(SS); D["nm"] = n / insts[0]["m"]
    return D

XB = {}; MASK = {}
def tables(n):
    if n not in XB:
        x = np.arange(1 << n); XB[n] = ((x[:, None] >> (n - 1 - np.arange(n))[None]) & 1).astype(np.int8)
        MASK[n] = CP.rep_mask(CP.LAYOUTS[n], a.proj).astype(np.float32)
    return XB[n], MASK[n]

def fwht_j(v, n):  # (B, 2^n) unnormalised WHT
    B = v.shape[0]; t = v.reshape((B,) + (2,) * n)
    for i in range(n):
        t0 = jnp.take(t, 0, axis=i + 1); t1 = jnp.take(t, 1, axis=i + 1); t = jnp.stack([t0 + t1, t0 - t1], axis=i + 1)
    return t.reshape(B, -1)

def init_params(rng):
    def W(i, o, s=1.0): return jnp.asarray(rng.normal(0, s / np.sqrt(i), (i, o)).astype(np.float32))
    p = AN.init_backbone(rng, "mp", H, R, FV); p["G1"] = W(2 * H, H); p["g1"] = jnp.zeros(H); p["G2"] = jnp.zeros((H, NBASE)); p["Wc"] = jnp.zeros((H, L))
    p["base"] = jnp.zeros(NBASE); return p

def old_template():   # == abl_cav / gnn_snap free+clause_w+var_mix structure, to unravel S_g4_snapF_s1 thetas
    rng = np.random.default_rng(0)
    def W(i, o, s=1.0): return jnp.asarray(rng.normal(0, s / np.sqrt(i), (i, o)).astype(np.float32))
    p = AN.init_backbone(rng, "mp", H, R, FV); p["G1"] = W(2 * H, H); p["g1"] = jnp.zeros(H); p["G2"] = jnp.zeros((H, NBASE))
    p["Wc"] = jnp.zeros((H, L)); p["Wv"] = jnp.zeros((H, L + 1)); p["P0"] = W(H, FM, 2.0); p["P1"] = W(H, FM, 2.0)
    p["u0"] = jnp.zeros(FM); p["u1"] = jnp.zeros(FM); p["fq"] = jnp.zeros(FM); p["Wo"] = jnp.zeros((FM, L))
    return (p, jnp.zeros(NBASE, jnp.float32))

MIXK = {k: NS.Mixer(k, "gray") for k in (3, 4, 5)}
def forward(p, D, n, mixmats=None):
    """mixmats: None (ideal, differentiable) or list over l of dict K -> fixed (N,N) logical matrix (native eval)."""
    ks = CP.LAYOUTS[n]; G = {k: D[k] for k in ("Ip", "In", "fv", "deg")}; B = D["S"].shape[0]
    hv, hc = AN.run_backbone(p, G, "mp", H); g_ = jnp.concatenate([hv.mean(1), hc.mean(1)], -1)
    bv = p["base"][None] + (jnp.tanh(g_ @ p["G1"] + p["g1"]) @ p["G2"] if a.pool_delta else 0.0)
    X = bv[:, : L * NPL].reshape(B, L, NPL); bl, gl = X[:, :, 4], X[:, :, 5]
    w = hc @ p["Wc"] if a.clause_w else jnp.zeros(hc.shape[:2] + (L,))
    xb, mask = tables(n); lit = jnp.asarray(xb)[:, D["VP"]]                       # (2^n, B, m, 4)
    viol = jnp.prod((lit == D["BB"][None].astype(jnp.int8)).astype(jnp.float32), -1)   # (2^n, B, m)
    psi = NS.uniform_psi0(ks, B)
    for l in range(L):
        Ew = D["nm"] * jnp.einsum("xbm,bm->bx", viol, gl[:, l][:, None] + w[:, :, l])
        if a.proj != "full": Ew = fwht_j(fwht_j(Ew, n) * jnp.asarray(mask)[None], n) / (1 << n)
        psi = psi * jnp.exp(-1j * Ew.reshape(psi.shape)).astype(jnp.complex64)
        for c, k in enumerate(ks):
            M = MIXK[k](bl[:, l]) if mixmats is None else jnp.asarray(mixmats[l][k])
            psi = NS.apply_cav(M, psi, c, len(ks))
    return jnp.sum(jnp.abs(psi.reshape(B, -1)) ** 2 * D["S"], 1), bl, gl

def lossf(th, D, n, unr):
    pr = forward(unr(th), D, n)[0]; nl = -jnp.log(pr + 1e-12); k = max(1, nl.shape[0] // 10)
    return jnp.mean(nl) + a.tailw * jnp.mean(jax.lax.top_k(nl, k)[0])

rng = np.random.default_rng(1000 + a.seed); p0 = init_params(rng)
if a.warm:
    pth, k = a.warm.split("#"); Jw = json.load(open(pth)); po, bo = ravel_pytree(old_template())[1](jnp.asarray(Jw["runs"][int(k)]["theta"], jnp.float32))
    for kk in list(p0):
        if kk in po: p0[kk] = po[kk]
    p0["base"] = bo; log(f"warm from {a.warm}: base b {np.round(np.asarray(bo)[4:24:6], 3)} g {np.round(np.asarray(bo)[5:24:6], 3)}")
if a.x0:
    x = np.array(json.load(open(a.x0))["x"]); b = np.asarray(p0["base"]).copy(); b[4:L * NPL:NPL] = x[0::2]; b[5:L * NPL:NPL] = x[1::2]; p0["base"] = jnp.asarray(b)
    if a.warm: p0["G2"] = jnp.zeros_like(p0["G2"])   # keep the net's clause weights, drop its old base deltas
th0, unr = ravel_pytree(p0)
LRM = ravel_pytree({k: jax.tree_util.tree_map(lambda u, k=k: jnp.full_like(jnp.asarray(u), a.lrbase if k == "base" else 1.0), v) for k, v in p0.items()})[0]
log(f"gnnw tag={a.tag} train {a.train} proj={a.proj} assign={a.assign} clause_w={a.clause_w} pool_delta={a.pool_delta} params {th0.size}")
VG = {}; EVJ = {}
def vg(n):
    if n not in VG: VG[n] = jax.jit(jax.value_and_grad(lambda th, D: lossf(th, D, n, unr)))
    return VG[n]
def evaluate(th, D, n, ch, mixmats=None):
    key = (n, mixmats is not None)
    if key not in EVJ or mixmats is not None: EVJ[key] = jax.jit(lambda th, Dc: forward(unr(th), Dc, n, mixmats))
    f = EVJ[key]; B = D["S"].shape[0]; ps, bs, gs = [], [], []
    for s in range(0, B, ch):
        Dc = {k: (v[s:s + ch] if k != "nm" else v) for k, v in D.items()}; pr, bl, gl = f(th, Dc); ps.append(np.asarray(pr)); bs.append(np.asarray(bl)); gs.append(np.asarray(gl))
    return np.concatenate(ps).astype(float), np.concatenate(bs), np.concatenate(gs)
pools = {}
if a.smoke:
    for n in a.train:
        D = build(n, sim.instances(n, a.bs, TRB)); t = time.time(); v, g = vg(n)(th0, D); g.block_until_ready(); t1 = time.time() - t; t = time.time(); v, g = vg(n)(th0, D); g.block_until_ready()
        log(f"smoke n={n}: loss {float(v):.4f} finite {bool(jnp.all(jnp.isfinite(g)))} compile {t1:.1f}s step {time.time()-t:.2f}s")
    for n in a.test:
        ins = sim.instances(n, 4, TEB); D = build(n, ins); t = time.time(); pr, bl, gl = evaluate(th0, D, n, 2); log(f"smoke eval n={n}: p {np.round(pr, 4)} ({time.time()-t:.1f}s)")
    sys.exit(0)
for n in a.train:
    t = time.time(); pools[n] = build(n, sim.instances(n, a.npool, TRB)); log(f"pool n={n} {a.npool} built {time.time()-t:.0f}s")
nv = max(a.train); Dv = build(nv, sim.instances(nv, a.nval, VAB))
def es_val(th):
    pr, _, _ = evaluate(th, Dv, nv, 50 if nv <= 12 else 10); return NS.summ(pr)
@jax.jit
def adam(x, m, v, g, t, lr):
    m = 0.9 * m + 0.1 * g; v = 0.999 * v + 0.001 * g * g; return x - lr * (m / (1 - 0.9 ** t)) / (jnp.sqrt(v / (1 - 0.999 ** t)) + 1e-8), m, v
x = th0; mom = jnp.zeros_like(x); vel = jnp.zeros_like(x); rb = np.random.default_rng(77 + a.seed)
s0 = es_val(x); BEST = dict(S=s0["S"], x=np.asarray(x), it=0); HIST = [[0, s0]]; log(f"step 0 val n={nv} {s0['median_inv_p']:.2f}/{s0['B90_inv_p']:.2f} S {s0['S']:.4f}")
rl = []; t0 = time.time()
for t in range(1, a.steps + 1):
    lr = a.lr * (0.1 + 0.9 * 0.5 * (1 + np.cos(np.pi * t / a.steps))); gt = 0; vs = 0
    for n in a.train:
        P = pools[n]; idx = rb.choice(P["S"].shape[0], a.bs, replace=False)
        if a.assign == "random":   # fresh random layout per formula: re-index clause vars + success mask
            Dn = {k: (v[idx] if k != "nm" else v) for k, v in P.items()}; VP = Dn["VP"].copy(); S = Dn["S"].copy()
            for b in range(a.bs):
                perm = rb.permutation(n); pos = np.argsort(perm); VP[b] = pos[VP[b]]; S[b] = CP.permute_vars(S[b], n, perm)
            Dn["VP"] = VP; Dn["S"] = S
        else: Dn = {k: (v[idx] if k != "nm" else v) for k, v in P.items()}
        v, g = vg(n)(x, Dn); gt = gt + g; vs += float(v)
    x, mom, vel = adam(x, mom, vel, gt, t, lr * LRM); rl.append(vs / len(a.train))
    if t % a.es == 0 or t == a.steps:
        s_ = es_val(x); HIST.append([t, s_])
        if s_["S"] < BEST["S"]: BEST.update(S=s_["S"], x=np.asarray(x), it=t)
        log(f"step {t} loss {np.mean(rl[-a.es:]):.4f} lr {lr:.1e} val {s_['median_inv_p']:.2f}/{s_['B90_inv_p']:.2f} S {s_['S']:.4f} (best {BEST['S']:.4f} @ {BEST['it']}) [{(time.time()-t0)/t:.2f}s/step]")
        json.dump({"args": vars(a), "hist": HIST, "best_it": BEST["it"], "theta_best": BEST["x"].tolist()}, open(OUT, "w"))
res = {"args": vars(a), "hist": HIST, "best_it": BEST["it"], "val_S": BEST["S"], "theta_best": BEST["x"].tolist(), "test": {}, "random": {}}
thb = jnp.asarray(BEST["x"])
for n in a.test:
    ins = sim.instances(n, CM.NTEST, TEB); D = build(n, ins); t = time.time()
    pr, bl, gl = evaluate(thb, D, n, 25 if n <= 12 else (4 if n <= 16 else 1))
    res["test"][str(n)] = NS.summ(pr) | {"p": pr.tolist(), "b_mean": bl.mean(0).tolist(), "b_std": bl.std(0).tolist(), "g_mean": gl.mean(0).tolist()}; res["random"][str(n)] = NS.rand_stats(ins, n)
    log(f"TEST n={n}: {res['test'][str(n)]['median_inv_p']:.2f}/{res['test'][str(n)]['B90_inv_p']:.2f} b {np.round(bl.mean(0), 3)} g {np.round(gl.mean(0), 3)} ({time.time()-t:.0f}s)")
    json.dump(res, open(OUT, "w"))
log("DONE")
