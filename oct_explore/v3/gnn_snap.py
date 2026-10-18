"""v3 SNAP-for-ECD (Zach 08:43/08:44 CST 2026-10-18): replace the (collapsed, |beta|~0) local ECD by a STANDARD SNAP.

Local block per register c, block l (l < L):   S_c(theta) on cavity  ->  R(th,ph) on transmon  ->  X on transmon
   * X on transmon = exactly what ECD(beta=0) does (p0' = p1, p1' = p0), so with S = 1 the circuit equals the ECD
     circuit with beta := 0 (step-0 equivalence, checked numerically).
   * standard SNAP: S = sum_k exp(-i theta_k)|k><k| on the cavity only, no transmon dependence.
     Size-free: theta_k = sum_j a_j b_j(k) [+ sum_{i<j} c_ij b_i(k) b_j(k)], b_j(k) = logical (Gray-code) digit j of Fock k.
     Pads (Fock >= 2^k) get phase 0 (never populated).
   Joint U per layer unchanged: per-digit X mixer . exp(-i cost phase). Final block: R + X only (a cavity-diagonal SNAP right
   before computational-basis measurement cannot change p, so it is omitted).
ABSORBABILITY NOTE: S_c is cavity-diagonal and commutes with R/X on the transmon, so S in block l is exactly a set of extra
   diagonal terms (per-digit local fields a, intra-cavity digit couplings c) added to the cost phase of layer l. It cannot add
   capability beyond "arbitrary diagonal"; what it adds is new *terms* (linear fields per variable, intra-cavity ZZ) that the
   clause-weight parameterisation of the cost phase cannot express. Placing it after the mixer is equivalent (commutes to the
   next layer's cost). Hence any gain = value of per-variable fields / digit couplings.
SNAP forms (--snap_form): lin = per-digit linear; pair = + pairwise; free = deep-set MLP head over each level's digit pattern
   (+ digit node embeddings in GNN), any per-level function, size-free; level = free 2^k table (plain, n=8 only).
Parameters (all zero-init, nothing forced):
   shared: sa[l] (linear, all digits), sc[l] (pair, all digit pairs)       -> plain mode (--plain) trains only base + these
   GNN   : a_{v,l} = sa[l] + h_v.Ws[:,l]   (per variable / digit)
           c_{ij,l} = sc[l] + sum_h h_i h_j Wp[h,l] / H     (--snap_pair, per digit pair inside a cavity)
--snap none = original ECD circuit (control: continue a warm start for the same number of iterations).
"""
from __future__ import annotations
import argparse, json, os, sys, time, fcntl
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import numpy as np, jax, jax.numpy as jnp
from jax.flatten_util import ravel_pytree
from scipy.optimize import minimize
import sim, v3core as V

ap = argparse.ArgumentParser()
ap.add_argument("--tag", required=True); ap.add_argument("--code", default="gray", choices=["gray", "binary"])
ap.add_argument("--L", type=int, required=True); ap.add_argument("--hidden", type=int, default=64); ap.add_argument("--rounds", type=int, default=3)
ap.add_argument("--clause_w", action="store_true"); ap.add_argument("--var_mix", action="store_true")
ap.add_argument("--snap", default="std", choices=["std", "none"]); ap.add_argument("--snap_pair", action="store_true", help="alias for --snap_form pair")
ap.add_argument("--snap_form", default="", choices=["", "lin", "pair", "free", "level"],
                help="lin: theta_k=sum_j a_j b_j(k) (k/SNAP); pair: + sum_{i<j} c_ij b_i b_j (k+k(k-1)/2); free: any per-level function via a "
                     "deep-set MLP head over the level's digit pattern (+ digit node embeddings in the GNN), size-free; level: free table "
                     "theta[l,c,k] (2^k/SNAP, plain only, n=8 only - not transferable)")
ap.add_argument("--free_m", type=int, default=32)
ap.add_argument("--plain", action="store_true", help="shared angles only (no network): base + sa/sc")
ap.add_argument("--warm", default="", help="results/v3/<gnn tag>.json: warm-start from its trained thetas (runs --warm_ids)")
ap.add_argument("--warm_ids", type=int, nargs="*", default=[])
ap.add_argument("--init_from", default="", help="keep-list json (v3_plain or --plain gnn_snap) used as bases")
ap.add_argument("--ninit", type=int, default=3); ap.add_argument("--ntrain", type=int, default=1000); ap.add_argument("--maxiter", type=int, default=1000)
ap.add_argument("--val", type=int, nargs="*", default=[10, 12]); ap.add_argument("--val_count", type=int, default=60)
ap.add_argument("--test", type=int, nargs="*", default=[8, 10, 12, 14, 20])
ap.add_argument("--feat", default="formula", choices=["full", "tq", "formula"])
ap.add_argument("--seed", type=int, default=0); ap.add_argument("--chunk", type=int, default=125)
ap.add_argument("--smoke", type=int, default=0, help="smoke: step-0 equivalence check on this many n=8 instances, then exit")
a = ap.parse_args()
if not a.snap_form: a.snap_form = "pair" if a.snap_pair else "lin"
a.snap_pair = a.snap_form == "pair"
FREE, LEVEL = a.snap_form == "free", a.snap_form == "level"
assert not (LEVEL and not a.plain), "level table only for plain"
TRB, TEB, VAB = 31_000_000_000, 32_000_000_000, 33_000_000_000
os.makedirs("results/v3", exist_ok=True); OUT = f"results/v3/{a.tag}.json"
log = lambda *s: print(time.strftime("%m-%d %H:%M:%S"), *s, flush=True)
L, H = a.L, a.hidden; NPL, NB = 6, 4; NBASE = L * NPL + NB; SNAP = a.snap == "std"
res = json.load(open(OUT)) if (os.path.exists(OUT) and not a.smoke) else {"args": vars(a)}
def save():
    if a.smoke: return
    json.dump(res, open(OUT + ".tmp", "w")); os.replace(OUT + ".tmp", OUT)
FV = 7
SNAPKEYS = ("sa", "sc", "Ws", "Wp", "P0", "P1", "u0", "u1", "fq", "Wo", "T")

# ---------- SNAP-aware layout helpers ----------
_BT = {}
def bits(lay):  # per register (k, nf) numpy table of logical digit j of Fock level f (pads 0).
    # Cache numpy only: caching jnp arrays inside a jitted nf() leaked tracers under n>8 eval (S_g4_snapP_f crash 12:56).
    if id(lay) not in _BT:
        out = []
        for k, nf in zip(lay.ks, lay.nfs):
            T = np.zeros((k, nf), np.float32)
            for f in range(1 << k):
                lg = V.gray_s(f) if lay.code == "gray" else f
                for j in range(k): T[j, f] = (lg >> j) & 1
            out.append(T)
        _BT[id(lay)] = out
    return _BT[id(lay)]


def local_snap(lay, psi, lb, phs):  # lb (B,C,4); phs list over c of (B,nf_c) SNAP phases or None
    C = lay.C
    for c in range(C):
        if phs is not None:
            ph = phs[c]; sh = [ph.shape[0]] + [1] * (psi.ndim - 1); sh[1 + C + c] = ph.shape[1]
            psi = psi * jnp.exp(-1j * ph).reshape(sh).astype(V.C64)
        psi = lay.bapply(lay.Rm(lb[:, c, 2], lb[:, c, 3]), psi, 1 + c)
        psi = jnp.flip(psi, axis=1 + c)                                   # == ECD(beta=0)
    return psi


def probs_any(lay, loc, mix, phase_fn, succ, PH):
    """PH: list over c of (B,L,nf_c) SNAP phases | None. SNAP off -> original ECD local (v3core)."""
    B = succ.shape[0]
    psi = jnp.broadcast_to(lay.psi0, (B,) + tuple(lay.shape))
    for l in range(L):
        if SNAP: psi = local_snap(lay, psi, loc[:, l], [P[:, l] for P in PH])
        else: psi = lay.local(psi, loc[:, l])
        psi = psi * jnp.exp(-1j * phase_fn(l)).astype(V.C64)
        for c in range(lay.C):
            psi = lay.bapply(lay.mixU(c, mix[:, l, c, : lay.ks[c]]), psi, 1 + lay.C + c)
    psi = local_snap(lay, psi, loc[:, L], None) if SNAP else lay.local(psi, loc[:, L])
    return jnp.sum((jnp.abs(psi) ** 2) * succ, axis=tuple(range(1, psi.ndim)))


# ---------- graph data ----------
def graph(n, ins, lay):
    m, cl = V.clauses(n, ins); Sg = np.zeros((m, n), np.float32)
    for i, (v, s) in enumerate(cl):
        for vv, ss in zip(v, s): Sg[i, vv] = ss
    Ip, In = (Sg > 0).astype(np.float32), (Sg < 0).astype(np.float32)
    fv = np.zeros((n, FV), np.float32); fv[:, 0] = 1; fv[:, 1] = Ip.sum(0) / (2 * m / n); fv[:, 2] = In.sum(0) / (2 * m / n)
    for c, k in enumerate(lay.ks):
        fv[lay.tvar[c], 3] = 1; fv[lay.tvar[c], 6] = k / 6
        for j, v in enumerate(lay.cvar[c]): fv[v, 4] = 1; fv[v, 5] = j / 6; fv[v, 6] = k / 6
    if a.feat == "tq": fv[:, 4:] = 0
    elif a.feat == "formula": fv[:, 3:] = 0
    return dict(Ip=Ip, In=In, fv=fv, deg=np.maximum(Ip.sum(0) + In.sum(0), 1)), cl


def batch_graph(n, insts, lay):
    gs, cls = zip(*[graph(n, i, lay) for i in insts])
    return {k: jnp.asarray(np.stack([g[k] for g in gs])) for k in gs[0]}, cls


# ---------- network ----------
def init_params(rng):
    def W(i, o, s=1.0): return jnp.asarray(rng.normal(0, s / np.sqrt(i), (i, o)).astype(np.float32))
    p = {}
    if not a.plain:
        p = {"Ev": W(FV, H), "Ec": jnp.asarray(rng.normal(0, 1, (H,)).astype(np.float32)), "rounds": []}
        for _ in range(a.rounds):
            p["rounds"].append({k: W(H, H) for k in ["Uc", "Ap", "An", "Uv", "Bp", "Bn"]} | {"bc": jnp.zeros(H), "bv": jnp.zeros(H)})
        p["G1"] = W(2 * H, H); p["g1"] = jnp.zeros(H); p["G2"] = jnp.zeros((H, NBASE))
        if a.clause_w: p["Wc"] = jnp.zeros((H, L))
        if a.var_mix: p["Wv"] = jnp.zeros((H, L + 1))
        if SNAP and not FREE: p["Ws"] = jnp.zeros((H, L))
        if SNAP and a.snap_pair: p["Wp"] = jnp.zeros((H, L))
        if SNAP and FREE: p["P0"] = W(H, a.free_m, 2.0); p["P1"] = W(H, a.free_m, 2.0)
    if SNAP and not (FREE or LEVEL): p["sa"] = jnp.zeros(L)
    if SNAP and a.snap_pair: p["sc"] = jnp.zeros(L)
    if SNAP and FREE:  # head: theta_{f,l} = tanh(z_f + fq) . Wo[:,l];  Wo zero-init => identity SNAP at step 0
        p["u0"] = jnp.asarray(rng.normal(0, 1, a.free_m).astype(np.float32)); p["u1"] = jnp.asarray(rng.normal(0, 1, a.free_m).astype(np.float32))
        p["fq"] = jnp.zeros(a.free_m); p["Wo"] = jnp.zeros((a.free_m, L))
    if SNAP and LEVEL: p["T"] = jnp.zeros((L, len(sim.LAYOUTS_T[8]), 1 << max(sim.LAYOUTS_T[8])))
    return p


def rand_base(rng):
    r = []
    for _ in range(L): r += [rng.uniform(-0.4, 0.4), rng.uniform(-0.4, 0.4), rng.uniform(0, np.pi), rng.uniform(0, 2 * np.pi), rng.uniform(0, np.pi), rng.uniform(0, 2.0)]
    r += [rng.uniform(-0.4, 0.4), rng.uniform(-0.4, 0.4), rng.uniform(0, np.pi), rng.uniform(0, 2 * np.pi)]
    return np.array(r, np.float32)


def net(p, base, G):
    """-> bv (B,NBASE), w (B,m,L)|None, e (B,n,L+1)|None, hv (B,n,H)|None"""
    B = G["Ip"].shape[0]
    if a.plain: return jnp.broadcast_to(base[None], (B, NBASE)), None, None, None
    hv = jnp.tanh(G["fv"] @ p["Ev"]); m = G["Ip"].shape[1]
    hc = jnp.broadcast_to(jnp.tanh(p["Ec"]), (B, m, H))
    for r in p["rounds"]:
        ac = (jnp.einsum("bcv,bvh->bch", G["Ip"], hv @ r["Ap"]) + jnp.einsum("bcv,bvh->bch", G["In"], hv @ r["An"])) / 4.0
        hc = hc + jnp.tanh(hc @ r["Uc"] + ac + r["bc"])
        av = (jnp.einsum("bcv,bch->bvh", G["Ip"], hc @ r["Bp"]) + jnp.einsum("bcv,bch->bvh", G["In"], hc @ r["Bn"])) / G["deg"][..., None]
        hv = hv + jnp.tanh(hv @ r["Uv"] + av + r["bv"])
    g = jnp.concatenate([hv.mean(1), hc.mean(1)], -1)
    bv = base[None] + jnp.tanh(g @ p["G1"] + p["g1"]) @ p["G2"]
    w = (hc @ p["Wc"]) if a.clause_w else None
    e = (hv @ p["Wv"]) if a.var_mix else None
    return bv, w, e, hv


def cvpad(lay): return np.array([lay.cvar[c] + [lay.cvar[c][0]] * (lay.kmax - lay.ks[c]) for c in range(lay.C)])


def circuit_args(lay, bv, e):
    B = bv.shape[0]; X = bv[:, : L * NPL].reshape(B, L, NPL); fin = bv[:, L * NPL:]
    blocks = jnp.concatenate([X[:, :, :NB], fin[:, None]], 1)
    loc = jnp.broadcast_to(blocks[:, :, None, :], (B, L + 1, lay.C, NB))
    mix = jnp.broadcast_to(X[:, :, 4][:, :, None, None], (B, L, lay.C, lay.kmax))
    if e is not None:
        tv = jnp.asarray(lay.tvar); et = e[:, tv, :]
        loc = loc.at[:, :, :, 2].add(jnp.transpose(et, (0, 2, 1)))
        ec = e[:, jnp.asarray(cvpad(lay)), :L]
        mix = mix + jnp.transpose(ec, (0, 3, 1, 2))
    return loc, mix, X[:, :, 5]


def snap_args(lay, p, hv, B):
    """-> list over registers c of (B, L, nf_c) SNAP phases (pads 0), or None if SNAP off.
    lin/pair: theta_f = sum_j a_j b_j(f) [+ sum_{i<j} c_ij b_i b_j];  free: deep-set MLP over the digit pattern of level f:
    z_f = mean_j [ b_j(f) U1_j + (1-b_j(f)) U0_j ],  U{0,1}_j = u{0,1} (+ h_vj P{0,1} in the GNN);  theta_f = tanh(z_f + fq) . Wo."""
    if not SNAP: return None
    out = []; cv = cvpad(lay)
    hd_all = None if hv is None else hv[:, jnp.asarray(cv)]                          # (B,C,kmax,H)
    for c, (k, nf) in enumerate(zip(lay.ks, lay.nfs)):
        Bt = jnp.asarray(bits(lay)[c])                                               # (k,nf)
        if LEVEL:
            T = p["T"][:, c, : 1 << k]; ph = jnp.concatenate([T, jnp.zeros((L, nf - (1 << k)))], 1)
            out.append(jnp.broadcast_to(ph[None], (B, L, nf))); continue
        if FREE:
            U0 = jnp.broadcast_to(p["u0"], (B, k, a.free_m)); U1 = jnp.broadcast_to(p["u1"], (B, k, a.free_m))
            if hd_all is not None: hd = hd_all[:, c, :k]; U0 = U0 + hd @ p["P0"]; U1 = U1 + hd @ p["P1"]
            z = (jnp.einsum("bjm,jf->bfm", U1, Bt) + jnp.einsum("bjm,jf->bfm", U0, 1 - Bt)) / k
            mask = (jnp.arange(nf) < (1 << k)).astype(jnp.float32)
            out.append(jnp.transpose(jnp.tanh(z + p["fq"]) @ p["Wo"], (0, 2, 1)) * mask[None, None]); continue
        A = jnp.broadcast_to(p["sa"][None, :, None], (B, L, k))
        if hd_all is not None: A = A + jnp.transpose(hd_all[:, c, :k] @ p["Ws"], (0, 2, 1))
        ph = jnp.einsum("blj,jf->blf", A, Bt)
        if a.snap_pair:
            Cm = jnp.broadcast_to(p["sc"][None, :, None, None], (B, L, k, k))
            if hd_all is not None: hd = hd_all[:, c, :k]; Cm = Cm + jnp.einsum("bih,bjh,hl->blij", hd, hd, p["Wp"]) / H
            ph = ph + jnp.einsum("blij,if,jf->blf", jnp.triu(Cm, 1), Bt, Bt)
        out.append(ph)
    return out


# ---------- training data / loss ----------
lay8 = V.Layout(8, a.code)
def prep8(insts):
    G, cls = batch_graph(8, insts, lay8)
    Viol = jnp.asarray(np.stack([V.viol_matrix(8, c) for c in cls])); S = jnp.asarray(lay8.succ_D(insts))
    return G, Viol, S, 8 / insts[0]["m"]


def probs8(p, base, G, Viol, S, nm):
    bv, w, e, hv = net(p, base, G); loc, mix, g = circuit_args(lay8, bv, e); PH = snap_args(lay8, p, hv, bv.shape[0])
    if w is None: ph = lambda l: lay8.to_D(nm * g[:, l][:, None] * Viol.sum(1))
    else: ph = lambda l: lay8.to_D(nm * jnp.einsum("bc,bcx->bx", g[:, l][:, None] + w[:, :, l], Viol))
    return probs_any(lay8, loc, mix, ph, S, PH)


def lossf(theta, unr, G, Viol, S, nm):
    p, base = unr(theta); return -jnp.mean(jnp.log(probs8(p, base, G, Viol, S, nm) + 1e-12))


# ---------- evaluation at any n ----------
_EV = {}
def eval_n(n, insts, p, base, chunk, mags=None):
    if n not in _EV:
        lay = V.Layout(n, a.code)
        def pr(loc, mix, phD, S, PH): return probs_any(lay, loc, mix, lambda l: phD[:, l], S, PH)
        def nf(p, b, G):
            bv, w, e, hv = net(p, b, G); PH = snap_args(lay, p, hv, bv.shape[0]); return bv, w, e, PH
        _EV[n] = (lay, jax.jit(pr), jax.jit(nf))
    lay, pr, nf = _EV[n]; out = []
    for i in range(0, len(insts), chunk):
        sub = insts[i:i + chunk]; G, cls = batch_graph(n, sub, lay); bv, w, e, PH = nf(p, base, G)
        if mags is not None and PH is not None:   # per layer: rms of the SNAP phase over used Fock levels (global phase removed), max |dev|
            for c in range(lay.C):
                ph = np.asarray(PH[c])[:, :, : 1 << lay.ks[c]]; ph = ph - ph.mean(-1, keepdims=True)
                mags.setdefault("phase_rms", []).append(np.sqrt((ph ** 2).mean(axis=(0, 2)))); mags.setdefault("phase_maxdev", []).append(np.abs(ph).max(axis=(0, 2)))
        loc, mix, g = circuit_args(lay, bv, e); g = np.asarray(g); w = None if w is None else np.asarray(w)
        nm = n / sub[0]["m"]; phs = []
        for b, ins in enumerate(sub):
            rows = []
            for l in range(L):
                if w is None: rows.append(nm * g[b, l] * ins["E"].astype(np.float32))
                else: rows.append(nm * V.weighted_energy(n, cls[b], g[b, l] + w[b, :, l]))
            phs.append(np.stack(rows))
        ph = np.stack(phs)
        phD = jnp.stack([lay.to_D(jnp.asarray(ph[:, l])) for l in range(L)], 1)
        out += list(np.asarray(pr(loc, mix, phD, jnp.asarray(lay.succ_D(sub)), PH)))
    return np.array(out)


# ---------- inits ----------
rng = np.random.default_rng(a.seed + 100 * L + 17 * H + (0 if a.code == "gray" else 3) + (5 if SNAP else 0))
template = (init_params(np.random.default_rng(0)), jnp.zeros(NBASE, jnp.float32))
_, unr = ravel_pytree(template)
def theta_of(p, b): return np.asarray(ravel_pytree((p, jnp.asarray(b, jnp.float32)))[0], np.float64)
inits, origin = [], []
if a.warm:  # GNN thetas trained with the ECD circuit: unravel with the old structure, add zero SNAP params, drop ECD beta
    W = json.load(open(a.warm)); ids = a.warm_ids or [r["i"] for r in W["runs"]]
    pt = template[0]; old_p = {k: v for k, v in pt.items() if k not in SNAPKEYS}
    _, unr_old = ravel_pytree((old_p, template[1]))
    for k in ids[: a.ninit]:
        po, bo = unr_old(jnp.asarray(W["runs"][k]["theta"], jnp.float32)); pn = dict(po); fresh = init_params(rng)
        for kk in SNAPKEYS:  # output-side SNAP params zero (identity SNAP); free-head input projections random (else dead gradient)
            if kk in pt: pn[kk] = fresh[kk] if kk in ("P0", "P1", "u0", "u1") else jnp.zeros_like(pt[kk])
        inits.append((pn, bo)); origin.append(f"warm:{os.path.basename(a.warm)}#{k}")
elif a.init_from:
    J = json.load(open(a.init_from)); ks = J["keep"]
    for k in ks[: a.ninit]:
        x = np.array(k["x"], np.float32); p = init_params(rng)
        if J["args"].get("plain") and SNAP and J["args"].get("snap_form") == a.snap_form and a.snap_form in ("lin", "pair"):
            # plain-SNAP keep: x = theta of ({sa[,sc]}, base); shared SNAP carried over, heads zero
            pp, bb = (ravel_pytree(({kk: jnp.zeros(L) for kk in ("sa", "sc") if kk in p}, jnp.zeros(NBASE)))[1])(jnp.asarray(x))
            p.update(pp); inits.append((p, bb))
        elif J["args"].get("plain") and SNAP:
            raise SystemExit("init_from plain-SNAP keep with a different/free form not supported")
        else: inits.append((p, jnp.asarray(x[:NBASE])))
        origin.append(f"init_from:{os.path.basename(a.init_from)}#{k['id']}")
else:
    for i in range(a.ninit): inits.append((init_params(rng), jnp.asarray(rand_base(rng)))); origin.append(f"rand{i}")
NI = len(inits)
log(f"SNAP-GNN snap={a.snap} form={a.snap_form} plain={a.plain} L={L} H={H} clause_w={a.clause_w} var_mix={a.var_mix} code={a.code} inits={origin}")
log(f"params per model: {template and theta_of(*inits[0]).size}")

vg = jax.jit(jax.value_and_grad(lossf, argnums=0), static_argnums=1)
def make_f(parts, ntot):
    def f(th):
        th = jnp.asarray(th, jnp.float32); tot, gr = 0.0, 0.0
        for G, Vi, S, nm in parts:
            v, gg = vg(th, unr, G, Vi, S, nm); wgt = S.shape[0] / ntot; tot += float(v) * wgt; gr = gr + np.asarray(gg, np.float64) * wgt
        return tot, gr
    return f

# ---------- smoke: step-0 equivalence vs the ECD circuit ----------
if a.smoke:
    import importlib
    trs = sim.instances(8, a.smoke, TRB); parts = [prep8(trs)]; f = make_f(parts, a.smoke)
    for i, (p, b) in enumerate(inits):
        t0 = time.time(); l_snap = f(theta_of(p, b))[0]; t1 = time.time() - t0
        t0 = time.time(); l_snap2, gsn = f(theta_of(p, b)); t2 = time.time() - t0
        # ECD reference using the SAME weights (old structure) with v3core.local; beta kept vs zeroed
        po = {k: v for k, v in p.items() if k not in SNAPKEYS}
        def ecd_loss(base, zero_beta=False):
            G, Vi, S, nm = parts[0]; bv, w, e, hv = net(po, base, G); loc, mix, g = circuit_args(lay8, bv, e)
            if zero_beta: loc = loc.at[..., :2].set(0.0)   # beta := 0 on the per-instance output (base + network delta)
            ph = (lambda l: lay8.to_D(nm * g[:, l][:, None] * Vi.sum(1))) if w is None else (lambda l: lay8.to_D(nm * jnp.einsum("bc,bcx->bx", g[:, l][:, None] + w[:, :, l], Vi)))
            B = S.shape[0]; psi = jnp.broadcast_to(lay8.psi0, (B,) + tuple(lay8.shape))
            for l in range(L):
                psi = lay8.local(psi, loc[:, l]); psi = psi * jnp.exp(-1j * ph(l)).astype(V.C64)
                for c in range(lay8.C): psi = lay8.bapply(lay8.mixU(c, mix[:, l, c, : lay8.ks[c]]), psi, 1 + lay8.C + c)
            psi = lay8.local(psi, loc[:, L]); pr = jnp.sum((jnp.abs(psi) ** 2) * S, axis=tuple(range(1, psi.ndim)))
            return float(-jnp.mean(jnp.log(pr + 1e-12)))
        l_ecd = ecd_loss(jnp.asarray(b)); l_ecd0 = ecd_loss(jnp.asarray(b), True)
        gnorm_snap = {k: float(np.linalg.norm(np.asarray(v))) for k, v in unr(jnp.asarray(gsn, jnp.float32))[0].items() if k in SNAPKEYS}
        log(f"SMOKE init {origin[i]}: loss SNAP=0 {l_snap:.6f} | ECD trained beta {l_ecd:.6f} | ECD beta:=0 {l_ecd0:.6f} | diff(SNAP0-ECDb0) {l_snap-l_ecd0:+.2e} | "
            f"first call {t1:.1f}s, 2nd {t2:.2f}s on {a.smoke} inst | grad-norm on SNAP params {gnorm_snap}")
    sys.exit(0)

# ---------- main ----------
trs = sim.instances(8, a.ntrain, TRB); parts = [prep8(trs[i:i + a.chunk]) for i in range(0, len(trs), a.chunk)]; f = make_f(parts, a.ntrain)
runs = res.setdefault("runs", [])
for i in range(len(runs), NI):
    th0 = theta_of(*inits[i]); t0 = time.time(); l0 = f(th0)[0]
    r = minimize(f, th0, jac=True, method="L-BFGS-B", options={"maxiter": a.maxiter, "maxfun": int(a.maxiter * 1.5)})
    pr_, _ = unr(jnp.asarray(r.x, jnp.float32)); sh = {k: np.asarray(pr_[k]).tolist() for k in ("sa", "sc") if k in pr_}
    runs.append({"i": i, "origin": origin[i], "loss0": l0, "loss": float(r.fun), "nit": int(r.nit), "sec": time.time() - t0, "theta": r.x.tolist(), "snap_shared": sh}); save()
    log(f"  init {i} ({origin[i]}): {l0:.4f} -> {r.fun:.4f} ({r.nit} it, {time.time()-t0:.0f}s) shared SNAP {sh}")
if "winner" not in res:
    for rr in runs:
        p, b = unr(jnp.asarray(rr["theta"], jnp.float32))
        rr["val"] = [float(np.mean(-np.log(eval_n(vn, sim.instances(vn, a.val_count, VAB), p, b, 30) + 1e-12))) for vn in a.val]
        rr["valmean"] = float(np.mean(rr["val"])); log(f"  init {rr['i']}: train {rr['loss']:.4f} val {['%.3f' % v for v in rr['val']]}")
    res["winner"] = int(min(runs, key=lambda r: r["valmean"])["i"]); res["trainbest"] = int(min(runs, key=lambda r: r["loss"])["i"])
    if a.plain: res["keep"] = [{"id": r["i"], "x": r["theta"], "loss": r["loss"], "valmean": r["valmean"]} for r in sorted(runs, key=lambda r: r["loss"])]
    save(); log(f"winner init {res['winner']} (train-best {res['trainbest']})")
te = res.setdefault("test", {}); sm = res.setdefault("snap_mag", {})
for role in ["winner", "trainbest"]:
    k = res[role]
    if role == "trainbest" and k == res["winner"]: continue
    p, b = unr(jnp.asarray(runs[k]["theta"], jnp.float32)); te.setdefault(role, {"id": k}); sm.setdefault(role, {})
    for n in a.test:
        if str(n) in te[role]: continue
        N = 200 if n == 14 else 100; ins = sim.instances(n, N, TEB); ch = 50 if n <= 12 else (10 if n < 20 else 2)
        lk = open("/tmp/v3_n20.lock", "w") if n >= 18 else None
        if lk: fcntl.flock(lk, fcntl.LOCK_EX)
        t0 = time.time(); mg = {}; pp = eval_n(n, ins, p, b, ch, mg)
        if lk: fcntl.flock(lk, fcntl.LOCK_UN); lk.close()
        s = V.summ(pp); te[role][str(n)] = s
        if mg: sm[role][str(n)] = {"phase_rms_per_layer": np.mean(mg["phase_rms"], 0).tolist(), "phase_maxdev_per_layer": np.max(mg["phase_maxdev"], 0).tolist()}
        save()
        log(f"test {role} n={n} L={L} med1/p={s['median_inv_p']:.1f} B90={s['B90_inv_p']:.1f} ({time.time()-t0:.0f}s)"
            + (f" SNAP phase-rms/layer {['%.3f' % v for v in sm[role][str(n)]['phase_rms_per_layer']]} max|dev| {['%.2f' % v for v in sm[role][str(n)]['phase_maxdev_per_layer']]}" if mg else ""))
log("ALLDONE")
