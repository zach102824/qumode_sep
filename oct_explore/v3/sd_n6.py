"""SNAP+displacement (SD) cavity-only design, n=6 feasibility test (Zach 11:3x CST 2026-10-19).

Hardware roles: ALL data bits in cavities (n=6: 2 cavities x 3 bits, Fock 0..7, Gray code: Fock f <-> digits gray(f),
digit j of cavity c <-> variable 3c+j, var0 = MSB of x as sim.fast_energies). Transmons/bus are ancillas always returned
to g -> not simulated; every gate is an ideal cavity unitary. Each cavity truncated at NF levels (default 20); weight on
Fock >= 8 (either cavity) = leakage = failure.
Start ('flat'): vacuum, then FIXED 90-deg Gray mixer on each cavity: prod_j exp(-i pi/4 P_j) (P_j swaps the Gray levels
differing in digit j; identity on spare levels) -> |amplitude| flat over 0..7 (phases (-i)^popcount). 'vac': no prep.
Layer l=1..L (order 'sdc', the spec): on each cavity SNAP S(theta) = sum_f e^{-i theta_f}|f><f| (ALL NF levels) then
D(alpha_{c,l}) (complex, truncated exp(alpha a^dag - alpha* a)), then cost phase
   exp(-i (n/m) sum_c (g_l + w_{c,l}) [x violates c])  (ideal; bus-qubit mediated on hardware).
   Leaked/spare levels get the MAX over valid x of the layer's weighted phase (as sim.Circuit / v3core pads).
   NOTE: in 'sdc' order the last cost phase is diagonal right before measurement -> has no effect on p (3 effective
   cost phases). Order 'csd' (cost, SNAP, D) is an extra variant where all 4 cost phases count.
Measure Fock levels, decode Gray -> x; p = P(x satisfies formula).
GNN-SNAP (mp64 backbone from arch_nets, formula features, H=64, 3 rounds):
   pooled g=[mean hv, mean hc] -> tanh(G1) -> G2 (zero) : delta on shared base [Re a_l, Im a_l, g_l] (per layer)
   cavity embedding hcav_c = mean_j hv_{var(c,j)} -> Wa (zero) : per-cavity delta on alpha_{c,l}
   SNAP used levels: free deep-set head over the level's Gray digit pattern (as gnn_snap --snap_form free), Wo zero-init;
   SNAP spare levels (f>=8): shared learned per-layer per-level offset table (zero-init).
   clause weights w_{c,l} = hc @ Wc (zero-init).
   base init: |alpha_l| ~ U(0.2,0.4), random phase; g_l ~ U(0.2,1.5).
QAOA baseline (--qaoa): qubit QAOA |+>^6, shared (beta_l, gamma_l), Cn = E n/m, same train/test, same loss & optimiser.
"""
from __future__ import annotations
import argparse, json, os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE); sys.path.insert(0, os.path.dirname(HERE))
import numpy as np, jax, jax.numpy as jnp
from jax.flatten_util import ravel_pytree
from scipy.optimize import minimize
import sim, arch_nets as AN

ap = argparse.ArgumentParser()
ap.add_argument("--tag", required=True); ap.add_argument("--L", type=int, default=4); ap.add_argument("--NF", type=int, default=20)
ap.add_argument("--init", type=int, default=0); ap.add_argument("--start", default="flat", choices=["flat", "vac"])
ap.add_argument("--order", default="sdc", choices=["sdc", "csd"])
ap.add_argument("--local", default="disp", choices=["disp", "mix"], help="mix = control: ideal Gray X-mixer per digit (angle per var/layer) instead of D(alpha) (old plan's in-cavity mixer, no leakage)")
ap.add_argument("--ntrain", type=int, default=1000); ap.add_argument("--ntest", type=int, default=200); ap.add_argument("--maxiter", type=int, default=1000)
ap.add_argument("--qaoa", action="store_true"); ap.add_argument("--smoke", action="store_true")
ap.add_argument("--smoke_it", type=int, default=5)
ap.add_argument("--eval_from", default="", help="re-evaluate a trained json at this --NF (truncation check; extra spare SNAP phases = 0)")
a = ap.parse_args()
n, K, C, L, NF, H, R, FV, FM = 6, 3, 2, a.L, a.NF, 64, 3, 7, 32
NU = 1 << K
TRB, TEB = 31_000_000_000, 32_000_000_000
OUT = f"results/v3/sd_n6/{a.tag}.json"
log = lambda *s: print(time.strftime("%m-%d %H:%M:%S"), *s, flush=True)
gray = lambda i: i ^ (i >> 1)

trs = sim.instances(n, a.ntrain, TRB); tes = sim.instances(n, a.ntest, TEB)

def clauses(ins):
    m, cl, E = sim.gen_instance(n, ins["seed"]); assert (E == ins["E"]).all(); return m, cl

def viol(cl):  # (m, 2^n)
    x = np.arange(1 << n); bits = [(x >> (n - 1 - v)) & 1 for v in range(n)]; out = np.zeros((len(cl), 1 << n), np.float32)
    for i, (v, s) in enumerate(cl):
        ok = np.zeros(1 << n, bool)
        for vv, ss in zip(v, s): ok |= (bits[vv] == (1 if ss > 0 else 0))
        out[i] = ~ok
    return out

# (f0, f1) -> x ; spare -> 64
IDX = np.full((NF, NF), 1 << n, np.int64)
for f0 in range(NU):
    for f1 in range(NU):
        bv = [(gray(f0) >> j) & 1 for j in range(K)] + [(gray(f1) >> j) & 1 for j in range(K)]
        IDX[f0, f1] = sum(b << (n - 1 - v) for v, b in enumerate(bv))
IDXf = jnp.asarray(IDX.reshape(-1)); VALID = jnp.asarray((IDX < (1 << n)).astype(np.float32))
BT = np.zeros((K, NF), np.float32)  # digit table, spare 0
for f in range(NU):
    for j in range(K): BT[j, f] = (gray(f) >> j) & 1
BT = jnp.asarray(BT); MASKU = jnp.asarray((np.arange(NF) < NU).astype(np.float32))

# fixed prep: prod_j exp(-i pi/4 P_j) on 0..7, identity on spare
Up = np.eye(NF, dtype=np.complex128)
for j in range(K):
    P = np.zeros((NF, NF)); 
    for f in range(NU):
        g2 = gray(f) ^ (1 << j); f2 = [h for h in range(NU) if gray(h) == g2][0]; P[f2, f] = 1
    M = np.eye(NF, dtype=np.complex128); M[:NU, :NU] = np.cos(np.pi / 4) * np.eye(NU) - 1j * np.sin(np.pi / 4) * P[:NU, :NU]
    Up = M @ Up
e0 = np.zeros(NF, np.complex128); e0[0] = 1; v0 = Up @ e0 if a.start == "flat" else e0
assert a.start == "vac" or np.allclose(np.abs(v0[:NU]) ** 2, 1 / NU)
PSI0 = jnp.asarray(np.outer(v0, v0).astype(np.complex64))
aa = np.diag(np.sqrt(np.arange(1, NF)), 1); LAM, VD = np.linalg.eigh(1j * (aa.T - aa))
LAM = jnp.asarray(LAM.astype(np.float32)); VD = jnp.asarray(VD.astype(np.complex64)); NV = jnp.arange(NF, dtype=jnp.float32)

def disp(re, im):  # batched (B,) -> (B,NF,NF) = exp(alpha a^dag - alpha* a), alpha = re + i im (truncated)
    r = jnp.sqrt(re ** 2 + im ** 2 + 1e-20); ph = jnp.arctan2(im, re)
    W = jnp.exp(1j * ph[:, None] * NV[None])[:, :, None] * VD[None]
    return jnp.einsum("bik,bk,bjk->bij", W, jnp.exp(-1j * r[:, None] * LAM[None]), W.conj())

PJ = np.zeros((K, NF, NF), np.float32)
for j in range(K):
    for f in range(NU):
        g2 = gray(f) ^ (1 << j); f2 = [h for h in range(NU) if gray(h) == g2][0]; PJ[j, f2, f] = 1
PJ = jnp.asarray(PJ.astype(np.complex64)); SUBU = jnp.asarray(np.diag((np.arange(NF) < NU).astype(np.complex64)))
def mixU(bt):  # bt (B,K) per-digit angles -> prod_j [cos(b/2) Sub - i sin(b/2) P_j + Pad] (B,NF,NF)
    B = bt.shape[0]; U = jnp.broadcast_to(jnp.eye(NF, dtype=jnp.complex64), (B, NF, NF))
    for j in range(K):
        M = jnp.cos(bt[:, j] / 2)[:, None, None] * SUBU - 1j * jnp.sin(bt[:, j] / 2)[:, None, None] * PJ[j] + (jnp.eye(NF) - SUBU)
        U = jnp.einsum("bij,bjk->bik", M, U)
    return U

def graph(ins):
    m, cl = clauses(ins); Sg = np.zeros((m, n), np.float32)
    for i, (v, s) in enumerate(cl):
        for vv, ss in zip(v, s): Sg[i, vv] = ss
    Ip, In = (Sg > 0).astype(np.float32), (Sg < 0).astype(np.float32)
    fv = np.zeros((n, FV), np.float32); fv[:, 0] = 1; fv[:, 1] = Ip.sum(0) / (2 * m / n); fv[:, 2] = In.sum(0) / (2 * m / n)
    return dict(Ip=Ip, In=In, fv=fv, deg=np.maximum(Ip.sum(0) + In.sum(0), 1)), cl, m

def prep(insts):
    gs, cls, ms = zip(*[graph(i) for i in insts])
    G = {k: jnp.asarray(np.stack([g[k] for g in gs])) for k in gs[0]}
    Vi = jnp.asarray(np.stack([viol(c) for c in cls]))
    S = np.zeros((len(insts), NF * NF), np.float32)
    for b, ins in enumerate(insts):
        sx = np.append((ins["E"] == 0).astype(np.float32), 0.0); S[b] = sx[IDX.reshape(-1)]
    assert len(set(ms)) == 1
    return G, Vi, jnp.asarray(S.reshape(-1, NF, NF)), n / ms[0]

def phaseD(wx):  # (B, 2^n) logical phase -> (B,NF,NF), spare = max
    ext = jnp.concatenate([wx, wx.max(1, keepdims=True)], 1); return ext[:, IDXf].reshape(-1, NF, NF)

def circuit(psi, A, PH, cost):
    """A (B,L,C,2) alpha re/im; PH (B,L,C,NF) SNAP phases; cost(l) -> (B,NF,NF) phase. Returns final psi."""
    for l in range(L):
        if a.order == "csd": psi = psi * jnp.exp(-1j * cost(l))
        for c in range(C):
            psi = psi * (jnp.exp(-1j * PH[:, l, c])[:, :, None] if c == 0 else jnp.exp(-1j * PH[:, l, c])[:, None, :])
            Dm = disp(A[:, l, c, 0], A[:, l, c, 1]) if a.local == "disp" else mixU(A[:, l, c])
            psi = jnp.einsum("bij,bjk->bik", Dm, psi) if c == 0 else jnp.einsum("bkj,bij->bik", Dm, psi)
        if a.order == "sdc": psi = psi * jnp.exp(-1j * cost(l))
    return psi

# ---------------- GNN-SNAP ----------------
def init_params(rng):
    def W(i, o, s=1.0): return jnp.asarray(rng.normal(0, s / np.sqrt(i), (i, o)).astype(np.float32))
    p = AN.init_backbone(rng, "mp", H, R, FV)
    p["G1"] = W(2 * H, H); p["g1"] = jnp.zeros(H); p["G2"] = jnp.zeros((H, 3 * L)); p["Wa"] = jnp.zeros((H, 2 * L))
    p["Wc"] = jnp.zeros((H, L)); p["P0"] = W(H, FM, 2.0); p["P1"] = W(H, FM, 2.0)
    p["u0"] = jnp.asarray(rng.normal(0, 1, FM).astype(np.float32)); p["u1"] = jnp.asarray(rng.normal(0, 1, FM).astype(np.float32))
    p["fq"] = jnp.zeros(FM); p["Wo"] = jnp.zeros((FM, L)); p["Tsp"] = jnp.zeros((L, NF - NU))
    return p

def rand_base(rng):
    r = rng.uniform(0.2, 0.4, L) if a.local == "disp" else rng.uniform(0.3, 1.5, L); ph = rng.uniform(0, 2 * np.pi, L) if a.local == "disp" else np.zeros(L); g = rng.uniform(0.2, 1.5, L)
    return np.stack([r * np.cos(ph), r * np.sin(ph), g], 1).reshape(-1).astype(np.float32)

CV = jnp.asarray(np.array([[K * c + j for j in range(K)] for c in range(C)]))
def net(p, base, G):
    B = G["Ip"].shape[0]; hv, hc = AN.run_backbone(p, G, "mp", H)
    g = jnp.concatenate([hv.mean(1), hc.mean(1)], -1)
    bv = (base[None] + jnp.tanh(g @ p["G1"] + p["g1"]) @ p["G2"]).reshape(B, L, 3)
    hd = hv[:, CV]                                                               # (B,C,K,H)
    if a.local == "disp":
        A = jnp.broadcast_to(bv[:, :, None, :2], (B, L, C, 2)) + jnp.transpose((hd.mean(2) @ p["Wa"]).reshape(B, C, L, 2), (0, 2, 1, 3))
    else:
        A = bv[:, :, None, None, 0] + jnp.transpose((hd @ p["Wa"][:, :L]), (0, 3, 1, 2))     # (B,L,C,K)
    U0 = p["u0"] + hd @ p["P0"]; U1 = p["u1"] + hd @ p["P1"]                     # (B,C,K,FM)
    z = (jnp.einsum("bcjm,jf->bcfm", U1, BT) + jnp.einsum("bcjm,jf->bcfm", U0, 1 - BT)) / K
    ph_u = jnp.transpose(jnp.tanh(z + p["fq"]) @ p["Wo"], (0, 3, 1, 2)) * MASKU         # (B,L,C,NF)
    sp = jnp.concatenate([jnp.zeros((L, NU)), p["Tsp"]], 1)
    PH = ph_u + sp[None, :, None, :]
    w = hc @ p["Wc"]                                                             # (B,m,L)
    return A, PH, bv[:, :, 2], w

def run(p, base, G, Vi, S, nm):
    A, PH, g, w = net(p, base, G); B = S.shape[0]
    cost = lambda l: phaseD(nm * jnp.einsum("bc,bcx->bx", g[:, l][:, None] + w[:, :, l], Vi))
    psi = circuit(jnp.broadcast_to(PSI0, (B, NF, NF)), A, PH, cost); pr = jnp.abs(psi) ** 2
    return jnp.sum(pr * S, (1, 2)), jnp.sum(pr * (1 - VALID.reshape(NF, NF)), (1, 2)), A

def summ(p):
    inv = 1 / np.maximum(p, 1e-300)
    return {"median_inv_p": float(np.median(inv)), "B90_inv_p": float(np.quantile(inv, 0.9)), "mean_p": float(np.mean(p)), "median_p": float(np.median(p))}

def rand_base_stats(insts):
    rg = np.array([2.0 ** n / i["nsol"] for i in insts]); return {"median_inv_p": float(np.median(rg)), "B90_inv_p": float(np.quantile(rg, 0.9)), "mean_p": float(np.mean(1 / rg))}

t00 = time.time()
Dtr, Dte = prep(trs), prep(tes)
res = {"args": vars(a), "random_test": rand_base_stats(tes), "random_train": rand_base_stats(trs)}

if a.qaoa:
    def qprobs(x, insts_E, S, nm):
        cn = jnp.asarray(insts_E * nm); psi = jnp.full((cn.shape[0],) + (2,) * n, 1 / np.sqrt(2 ** n), jnp.complex64)
        cn = cn.reshape(psi.shape)
        for l in range(L):
            b, gm = x[2 * l], x[2 * l + 1]; psi = psi * jnp.exp(-1j * gm * cn)
            M = jnp.array([[jnp.cos(b / 2), -1j * jnp.sin(b / 2)], [-1j * jnp.sin(b / 2), jnp.cos(b / 2)]]).astype(jnp.complex64)
            for q in range(n): psi = jnp.moveaxis(jnp.tensordot(M, psi, axes=([1], [q + 1])), 0, q + 1)
        return jnp.sum(jnp.abs(psi.reshape(psi.shape[0], -1)) ** 2 * S, 1)
    Etr = np.stack([i["E"] for i in trs]).astype(np.float32); Str = jnp.asarray((Etr == 0).astype(np.float32))
    Ete = np.stack([i["E"] for i in tes]).astype(np.float32); Ste = jnp.asarray((Ete == 0).astype(np.float32))
    nm = n / trs[0]["m"]
    vg = jax.jit(jax.value_and_grad(lambda x: -jnp.mean(jnp.log(qprobs(x, Etr, Str, nm) + 1e-12))))
    f = lambda x: tuple(np.asarray(v, np.float64) for v in vg(jnp.asarray(x, jnp.float32)))
    rng = np.random.default_rng(7); cands = [np.stack([np.linspace(0.8, 0.1, L), np.linspace(0.1, 0.8, L)], 1).reshape(-1) * s for s in (0.5, 1, 2, 4, 8)]
    cands += [np.stack([rng.uniform(0, np.pi, L), rng.uniform(0, 3, L)], 1).reshape(-1) for _ in range(10)]
    runs = []
    for i, c in enumerate(cands):
        r = minimize(f, c, jac=True, method="L-BFGS-B", options={"maxiter": a.maxiter, "maxfun": int(1.5 * a.maxiter)})
        pt = np.asarray(qprobs(jnp.asarray(r.x, jnp.float32), Ete, Ste, nm), float)
        runs.append({"i": i, "loss": float(r.fun), "nit": int(r.nit), "x": r.x.tolist(), "test": summ(pt)}); log(f"QAOA cand {i}: loss {r.fun:.4f} test {runs[-1]['test']}")
    best = min(runs, key=lambda r: r["loss"]); res.update(runs=runs, best=best, sec=time.time() - t00)
    json.dump(res, open(OUT, "w")); log(f"QAOA best (train) loss {best['loss']:.4f} test {best['test']} ({time.time()-t00:.0f}s)"); sys.exit(0)

rng = np.random.default_rng(1000 + a.init)
p0, b0 = init_params(rng), jnp.asarray(rand_base(rng))
th0, unr = ravel_pytree((p0, b0))
if a.eval_from:
    J = json.load(open(a.eval_from)); NFt = J["args"]["NF"]; pt = dict(p0); pt["Tsp"] = jnp.zeros((L, NFt - NU))
    _, unr_t = ravel_pytree((pt, b0)); pj, bj = unr_t(jnp.asarray(J["theta"], jnp.float32))
    pj = dict(pj); pj["Tsp"] = jnp.concatenate([pj["Tsp"], jnp.zeros((L, NF - NFt))], 1)[:, : NF - NU]
    th = ravel_pytree((pj, bj))[0]
    ev = jax.jit(lambda th, G, Vi, S, nm: run(*unr(th), G, Vi, S, nm)); o = {}
    for nmk, D in (("train", Dtr), ("test", Dte)):
        pr, lk, A = (np.asarray(v, float) for v in ev(th, *D)); o[nmk] = summ(pr) | {"mean_leak": float(lk.mean()), "loss": float(np.mean(-np.log(pr + 1e-12)))}
    J.setdefault("nf_check", {})[str(NF)] = o; json.dump(J, open(a.eval_from, "w"))
    log(f"NF-check {a.eval_from} NF {NFt}->{NF}: train loss {J['train']['loss']:.4f}->{o['train']['loss']:.4f} test med/B90 {J['test']['median_inv_p']:.2f}/{J['test']['B90_inv_p']:.2f} -> {o['test']['median_inv_p']:.2f}/{o['test']['B90_inv_p']:.2f} leak {o['test']['mean_leak']:.4f}")
    sys.exit(0)
log(f"SD-GNN tag={a.tag} init={a.init} start={a.start} order={a.order} NF={NF} L={L} params={th0.size} train={len(trs)} test={len(tes)}")
log(f"random baseline test {res['random_test']}")

def loss(th, G, Vi, S, nm):
    p, b = unr(th); pr, lk, _ = run(p, b, G, Vi, S, nm); return -jnp.mean(jnp.log(pr + 1e-12))
vg = jax.jit(jax.value_and_grad(loss))
it = [0]
def f(th):
    v, g = vg(jnp.asarray(th, jnp.float32), *Dtr); it[0] += 1
    if it[0] % 100 == 0: log(f"  fev {it[0]} loss {float(v):.4f}")
    return float(v), np.asarray(g, np.float64)
evalj = jax.jit(lambda th, G, Vi, S, nm: run(*unr(th), G, Vi, S, nm))
l0 = f(np.asarray(th0, np.float64))[0]; log(f"loss0 {l0:.4f} (first call incl. compile {time.time()-t00:.0f}s)")
t0 = time.time()
r = minimize(f, np.asarray(th0, np.float64), jac=True, method="L-BFGS-B", options={"maxiter": a.maxiter if not a.smoke else a.smoke_it, "maxfun": int(1.5 * a.maxiter)})
th = jnp.asarray(r.x, jnp.float32)
out = {}
for nmk, D in (("train", Dtr), ("test", Dte)):
    pr, lk, A = (np.asarray(v, float) for v in evalj(th, *D))
    mag = np.sqrt((A ** 2).sum(-1)) if a.local == "disp" else np.abs(A).mean(-1)   # |alpha| (B,L,C); mix: mean |beta|
    out[nmk] = summ(pr) | {"mean_leak": float(lk.mean()), "max_leak": float(lk.max()), "loss": float(np.mean(-np.log(pr + 1e-12))),
                           "alpha_abs_mean_per_layer": mag.mean((0, 2)).tolist(), "alpha_abs_std_per_layer": mag.std((0, 2)).tolist(),
                           "alpha_abs_per_layer_cav": mag.mean(0).tolist()}
    if nmk == "test": out[nmk]["p"] = pr.tolist(); out[nmk]["leak"] = lk.tolist()
pb, bb = unr(th)
res.update(init=a.init, loss0=l0, loss=float(r.fun), nit=int(r.nit), nfev=int(r.nfev), msg=str(r.message), train_sec=time.time() - t0, sec=time.time() - t00,
           base=np.asarray(bb).reshape(L, 3).tolist(), base_alpha_abs=np.sqrt((np.asarray(bb).reshape(L, 3)[:, :2] ** 2).sum(1)).tolist(),
           base_alpha_abs0=np.sqrt((np.asarray(b0).reshape(L, 3)[:, :2] ** 2).sum(1)).tolist(), **out, theta=r.x.tolist())
json.dump(res, open(OUT, "w"))
log(f"DONE init {a.init}: loss {l0:.4f} -> {r.fun:.4f} ({r.nit} it, {time.time()-t0:.0f}s) | test med/B90 {out['test']['median_inv_p']:.2f}/{out['test']['B90_inv_p']:.2f} "
    f"mean p {out['test']['mean_p']:.3f} leak {out['test']['mean_leak']:.4f} |alpha|/layer {['%.3f' % v for v in out['test']['alpha_abs_mean_per_layer']]} | random {res['random_test']}")
