"""v3 PLAN B: graph network that reads each instance's Hamiltonian and outputs that instance's circuit parameters.

Graph: bipartite variable–clause graph, edge type = literal sign (+/-). Node features are size-free:
  var: [1, pos_deg/(2m/n), neg_deg/(2m/n), is_transmon, is_cavity, digit_j/6, k_c/6]   (hardware placement known)
  clause: [1]
R rounds of message passing, weights shared over nodes (works at any n):
  h_c <- h_c + tanh(h_c U_c + (Ip h_v A_p + In h_v A_n)/4 + b_c)
  h_v <- h_v + tanh(h_v U_v + (Ip^T h_c B_p + In^T h_c B_n)/deg_v + b_v)
Heads (zero-initialised so every init starts as a plain random shared-angle circuit `base`):
  global  : [mean h_v, mean h_c] -> MLP -> delta on base = L*[bre,bim,th,ph,bc,g] + final [bre,bim,th,ph]
  clause  (--clause_w): w_{c,l} = g_l + h_c Wc[:,l];  phase_l(x) = (n/m) sum_c w_{c,l} [c violated]   (diagonal)
  variable(--var_mix) : e_{v,l} = h_v Wv[:,l], l=0..L.  Cavity digit j of cavity c (var cvar[c][j]) gets X angle
                        bc_l + e_{v,l} in U_l;  transmon var of register c gets th_{l} + e_{v,l} in local block l.
Circuit unchanged: per layer local R+ECD per register, then ONE joint U = per-digit X mixer . exp(-i phase); final R+ECD.
Train end-to-end at n=8 (mean -log p), L-BFGS-B over all weights; select among inits on val n=10/12; test n=8..20.
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
ap.add_argument("--ninit", type=int, default=6); ap.add_argument("--ntrain", type=int, default=1000); ap.add_argument("--maxiter", type=int, default=1000)
ap.add_argument("--pre_iter", type=int, default=0, help="optional: first fit base only (shared angles) this many its, then all weights")
ap.add_argument("--init_from", default="", help="results/v3/<tag>.json of a v3_plain run: use its keep-list x as bases (from-scratch plain, no QAOA)")
ap.add_argument("--val", type=int, nargs="*", default=[10, 12]); ap.add_argument("--val_count", type=int, default=60)
ap.add_argument("--test", type=int, nargs="*", default=[8, 10, 12, 14, 20])
ap.add_argument("--feat", default="full", choices=["full", "tq", "formula"], help="var features: full=hardware placement (k_c,digit unseen beyond k=3 at n=8); tq=degrees+is_transmon; formula=degrees only")
ap.add_argument("--seed", type=int, default=0); ap.add_argument("--chunk", type=int, default=250)
a = ap.parse_args()
TRB, TEB, VAB = 31_000_000_000, 32_000_000_000, 33_000_000_000
os.makedirs("results/v3", exist_ok=True); OUT = f"results/v3/{a.tag}.json"
log = lambda *s: print(time.strftime("%m-%d %H:%M:%S"), *s, flush=True)
L, H = a.L, a.hidden; NPL, NB = 6, 4; NBASE = L * NPL + NB
res = json.load(open(OUT)) if os.path.exists(OUT) else {"args": vars(a)}
def save(): json.dump(res, open(OUT + ".tmp", "w")); os.replace(OUT + ".tmp", OUT)
FV = 7


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
    p = {"Ev": W(FV, H), "Ec": jnp.asarray(rng.normal(0, 1, (H,)).astype(np.float32)), "rounds": []}
    for _ in range(a.rounds):
        p["rounds"].append({k: W(H, H) for k in ["Uc", "Ap", "An", "Uv", "Bp", "Bn"]} | {"bc": jnp.zeros(H), "bv": jnp.zeros(H)})
    p["G1"] = W(2 * H, H); p["g1"] = jnp.zeros(H); p["G2"] = jnp.zeros((H, NBASE))
    if a.clause_w: p["Wc"] = jnp.zeros((H, L))
    if a.var_mix: p["Wv"] = jnp.zeros((H, L + 1))
    return p


def rand_base(rng):
    r = []
    for _ in range(L): r += [rng.uniform(-0.4, 0.4), rng.uniform(-0.4, 0.4), rng.uniform(0, np.pi), rng.uniform(0, 2 * np.pi), rng.uniform(0, np.pi), rng.uniform(0, 2.0)]
    r += [rng.uniform(-0.4, 0.4), rng.uniform(-0.4, 0.4), rng.uniform(0, np.pi), rng.uniform(0, 2 * np.pi)]
    return np.array(r, np.float32)


def net(p, base, G):
    """G batched graph -> per-instance (base_vec (B,NBASE), w (B,m,L) or None, e (B,n,L+1) or None)."""
    hv = jnp.tanh(G["fv"] @ p["Ev"]); B, m = G["Ip"].shape[0], G["Ip"].shape[1]
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
    return bv, w, e


def circuit_args(lay, bv, e):
    B = bv.shape[0]; X = bv[:, : L * NPL].reshape(B, L, NPL); fin = bv[:, L * NPL:]
    blocks = jnp.concatenate([X[:, :, :NB], fin[:, None]], 1)                      # (B,L+1,4)
    loc = jnp.broadcast_to(blocks[:, :, None, :], (B, L + 1, lay.C, NB))
    mix = jnp.broadcast_to(X[:, :, 4][:, :, None, None], (B, L, lay.C, lay.kmax))
    if e is not None:
        tv = jnp.asarray(lay.tvar); et = e[:, tv, :]                                 # (B,C,L+1)
        loc = loc.at[:, :, :, 2].add(jnp.transpose(et, (0, 2, 1)))
        cv = np.array([lay.cvar[c] + [lay.cvar[c][0]] * (lay.kmax - lay.ks[c]) for c in range(lay.C)])  # pad (unused)
        ec = e[:, jnp.asarray(cv), :L]                                               # (B,C,kmax,L)
        mix = mix + jnp.transpose(ec, (0, 3, 1, 2))
    return loc, mix, X[:, :, 5]


# ---------- training data / loss ----------
lay8 = V.Layout(8, a.code)
def prep8(insts):
    G, cls = batch_graph(8, insts, lay8)
    Viol = jnp.asarray(np.stack([V.viol_matrix(8, c) for c in cls])); S = jnp.asarray(lay8.succ_D(insts))
    return G, Viol, S, 8 / insts[0]["m"]


def probs8(p, base, G, Viol, S, nm):
    bv, w, e = net(p, base, G); loc, mix, g = circuit_args(lay8, bv, e)
    if w is None: ph = lambda l: lay8.to_D(nm * g[:, l][:, None] * Viol.sum(1))
    else: ph = lambda l: lay8.to_D(nm * jnp.einsum("bc,bcx->bx", g[:, l][:, None] + w[:, :, l], Viol))
    return lay8.probs(loc, mix, ph, S)


def lossf(theta, unr, G, Viol, S, nm):
    p, base = unr(theta); return -jnp.mean(jnp.log(probs8(p, base, G, Viol, S, nm) + 1e-12))


# ---------- evaluation at any n ----------
_EV = {}
def eval_n(n, insts, p, base, chunk):
    if n not in _EV:
        lay = V.Layout(n, a.code)
        def pr(loc, mix, phD, S): return lay.probs(loc, mix, lambda l: phD[:, l], S)
        _EV[n] = (lay, jax.jit(pr), jax.jit(lambda p, b, G: net(p, b, G)))
    lay, pr, nf = _EV[n]; out = []
    for i in range(0, len(insts), chunk):
        sub = insts[i:i + chunk]; G, cls = batch_graph(n, sub, lay); bv, w, e = nf(p, base, G)
        loc, mix, g = circuit_args(lay, bv, e); g = np.asarray(g); w = None if w is None else np.asarray(w)
        nm = n / sub[0]["m"]; phs = []
        for b, ins in enumerate(sub):
            rows = []
            for l in range(L):
                if w is None: rows.append(nm * g[b, l] * ins["E"].astype(np.float32))
                else: rows.append(nm * V.weighted_energy(n, cls[b], g[b, l] + w[b, :, l]))
            phs.append(np.stack(rows))
        ph = np.stack(phs)                                                        # (b,L,2^n)
        phD = jnp.stack([lay.to_D(jnp.asarray(ph[:, l])) for l in range(L)], 1)
        out += list(np.asarray(pr(loc, mix, phD, jnp.asarray(lay.succ_D(sub)))))
    return np.array(out)


# ---------- main ----------
rng = np.random.default_rng(a.seed + 100 * L + 17 * H + (0 if a.code == "gray" else 3))
trs = sim.instances(8, a.ntrain, TRB); parts = [prep8(trs[i:i + a.chunk]) for i in range(0, len(trs), a.chunk)]
log(f"GNN L={L} H={H} rounds={a.rounds} clause_w={a.clause_w} var_mix={a.var_mix} code={a.code} ntrain={a.ntrain}")
bases = [rand_base(rng) for _ in range(a.ninit)]
if a.init_from:
    ks = json.load(open(a.init_from))["keep"]; bases = [np.array(k["x"], np.float32) for k in ks][: a.ninit]
inits = [(init_params(rng), jnp.asarray(b)) for b in bases]
_, unr = ravel_pytree(inits[0]); log(f"weights per model: {ravel_pytree(inits[0])[0].size}")
vg = jax.jit(jax.value_and_grad(lossf, argnums=0), static_argnums=1)
def f(th):
    th = jnp.asarray(th, jnp.float32); tot, gr = 0.0, 0.0
    for G, Vi, S, nm in parts:
        v, gg = vg(th, unr, G, Vi, S, nm); wgt = S.shape[0] / a.ntrain; tot += float(v) * wgt; gr = gr + np.asarray(gg, np.float64) * wgt
    return tot, gr
runs = res.setdefault("runs", [])
for i in range(len(runs), a.ninit):
    th0, _ = ravel_pytree(inits[i]); th0 = np.asarray(th0, np.float64); t0 = time.time(); l0 = f(th0)[0]
    if a.pre_iter:  # base-only warmup (net heads zero => shared angles), then everything
        nb = NBASE; mask = np.zeros_like(th0); mask[-nb:] = 1  # ravel order: params dict first, base last
        def fb(z): th = th0.copy(); th[-nb:] = z; v, g = f(th); return v, g[-nb:]
        r0 = minimize(fb, th0[-nb:], jac=True, method="L-BFGS-B", options={"maxiter": a.pre_iter}); th0[-nb:] = r0.x
    r = minimize(f, th0, jac=True, method="L-BFGS-B", options={"maxiter": a.maxiter, "maxfun": int(a.maxiter * 1.5)})
    runs.append({"i": i, "loss0": l0, "loss": float(r.fun), "nit": int(r.nit), "sec": time.time() - t0, "theta": r.x.tolist()}); save()
    log(f"  init {i}: {l0:.4f} -> {r.fun:.4f} ({r.nit} it, {time.time()-t0:.0f}s)")
if "winner" not in res:
    for rr in runs:
        p, b = unr(jnp.asarray(rr["theta"], jnp.float32))
        rr["val"] = [float(np.mean(-np.log(eval_n(vn, sim.instances(vn, a.val_count, VAB), p, b, 30) + 1e-12))) for vn in a.val]
        rr["valmean"] = float(np.mean(rr["val"])); log(f"  init {rr['i']}: train {rr['loss']:.4f} val {['%.3f' % v for v in rr['val']]}")
    res["winner"] = int(min(runs, key=lambda r: r["valmean"])["i"]); res["trainbest"] = int(min(runs, key=lambda r: r["loss"])["i"]); save()
    log(f"winner init {res['winner']} (train-best {res['trainbest']})")
te = res.setdefault("test", {})
for role in ["winner", "trainbest"]:
    k = res[role]
    if role == "trainbest" and k == res["winner"]: continue
    p, b = unr(jnp.asarray(runs[k]["theta"], jnp.float32)); te.setdefault(role, {"id": k})
    for n in a.test:
        if str(n) in te[role]: continue
        N = 200 if n == 14 else 100; ins = sim.instances(n, N, TEB); ch = 50 if n <= 12 else (10 if n < 20 else 2)
        lk = open("/tmp/v3_n20.lock", "w") if n >= 18 else None
        if lk: fcntl.flock(lk, fcntl.LOCK_EX)
        t0 = time.time(); pp = eval_n(n, ins, p, b, ch)
        if lk: fcntl.flock(lk, fcntl.LOCK_UN); lk.close()
        s = V.summ(pp); te[role][str(n)] = s; save()
        log(f"test {role} n={n} L={L} med1/p={s['median_inv_p']:.1f} B90={s['B90_inv_p']:.1f} ({time.time()-t0:.0f}s)")
log("ALLDONE")
