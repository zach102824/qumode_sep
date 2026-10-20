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
ap.add_argument("--init", type=int, default=0); ap.add_argument("--start", default="flat", choices=["flat", "vac", "learned"], help="learned = shared trainable fixed cavity start state (same on every cavity, init = flat prep); a fixed state prep, not per formula")
ap.add_argument("--order", default="sdc", choices=["sdc", "csd"])
ap.add_argument("--local", default="disp", choices=["disp", "mix"], help="mix = control: ideal Gray X-mixer per digit (angle per var/layer) instead of D(alpha) (old plan's in-cavity mixer, no leakage)")
ap.add_argument("--ntrain", type=int, default=1000); ap.add_argument("--ntest", type=int, default=200); ap.add_argument("--maxiter", type=int, default=1000)
ap.add_argument("--qaoa", action="store_true"); ap.add_argument("--smoke", action="store_true")
ap.add_argument("--smoke_it", type=int, default=5)
ap.add_argument("--final", action="store_true", help="extra final SNAP+D block on each cavity before measurement (no cost)")
ap.add_argument("--amax", type=float, default=0.0, help=">0: bound |alpha| <= amax via amax*tanh(|a|/amax)")
ap.add_argument("--sizefree", action="store_true", help="no per-level shared SNAP table T0 (not size-free); SNAP = free digit-pattern head + snap_lin + spare offsets")
ap.add_argument("--arch", default="mp", choices=["mp", "attn", "tf", "ds"]); ap.add_argument("--H", type=int, default=64); ap.add_argument("--rounds", type=int, default=3)
ap.add_argument("--warm", default="", help="same-structure json: start from its theta")
ap.add_argument("--nval", type=int, default=300)
ap.add_argument("--percase", type=int, default=0, help="ceiling diagnostic: free per-instance params on the first N val instances (no GNN)")
ap.add_argument("--tailw", type=float, default=0.0, help="loss += tailw * mean of worst 10%% of -log p")
ap.add_argument("--leakpen", type=float, default=0.0); ap.add_argument("--ainit", type=float, nargs=2, default=[0.2, 0.4])
ap.add_argument("--ginit", type=float, nargs=2, default=[0.2, 1.5]); ap.add_argument("--seedoff", type=int, default=1000)
ap.add_argument("--nfcheck", type=int, default=40)
ap.add_argument("--pc_set", default="val", choices=["val", "train"]); ap.add_argument("--pc_init", default="", help="plain json: per-case params start at the shared solution")
ap.add_argument("--prox", type=float, default=0.0, help="per-case: + prox * mean_i ||theta_i - theta_shared||^2 (keeps per-case optima in one smooth basin, for distillation)")
ap.add_argument("--distill", default="", help="per-case npz (from --percase --pc_set train --pc_init): supervised pretraining of the GNN heads on these targets, then -log p fine-tune")
ap.add_argument("--wd", type=float, default=0.0, help="L2 on GNN output heads (G2, Wa, Wc, Wo, Ws)")
ap.add_argument("--adam", type=int, default=0, help=">0: minibatch Adam for this many steps on a pool of --npool fresh train instances (instead of L-BFGS on --ntrain)")
ap.add_argument("--npool", type=int, default=20000); ap.add_argument("--bs", type=int, default=256); ap.add_argument("--lr", type=float, default=3e-3)
ap.add_argument("--es", type=int, default=0, help="every N iters evaluate val S=(log med+log B90)/2; keep the best-val theta (early stopping on the 300 val set)")
ap.add_argument("--sup_iters", type=int, default=500); ap.add_argument("--chunk", type=int, default=500, help="train loss/grad accumulated over chunks (memory)")
ap.add_argument("--relabel", default="none", choices=["none", "bias", "antibias"], help="per-formula var->digit map: bias = most polarity-biased vars on Gray digit 2 (only 1 D-edge 3<->4), least on digit 0")
ap.add_argument("--pairs", type=int, default=1, help="SNAP+D pairs per cavity per layer (Zach base design = 1; 2 = fallback)")
ap.add_argument("--snap_lin", action="store_true", help="GNN SNAP += sum_j a_{v(j),l} b_j(f), a = hv @ Ws (zero-init): explicit per-variable local fields")
ap.add_argument("--plain", action="store_true", help="shared params only: SNAP table T0 (NB,NF), alpha A0 (NB,2), g (NB); no clause weights")
ap.add_argument("--init_plain", default="", help="plain json: GNN base alpha/g and shared SNAP table T0 start from it")
ap.add_argument("--eval_from", default="", help="re-evaluate a trained json at this --NF (truncation check; extra spare SNAP phases = 0)")
a = ap.parse_args()
n, K, C, L, NF, H, R, FV, FM = 6, 3, 2, a.L, a.NF, a.H, a.rounds, 7, 32
NB = (L + (1 if a.final else 0)) * a.pairs   # SD units; layer l = units l*P..l*P+P-1; cost of layer l uses g/w column l*P
NU = 1 << K
TRB, TEB, VAB = 31_000_000_000, 32_000_000_000, 33_000_000_000
OUT = f"results/v3/sd_n6/{a.tag}.json"
log = lambda *s: print(time.strftime("%m-%d %H:%M:%S"), *s, flush=True)
gray = lambda i: i ^ (i >> 1)

trs = sim.instances(n, a.ntrain, TRB); tes = sim.instances(n, a.ntest, TEB); vas = sim.instances(n, a.nval, VAB)

def clauses(ins):
    m, cl, E = sim.gen_instance(n, ins["seed"]); assert (E == ins["E"]).all()
    if a.relabel != "none":  # classical compile-time choice of which variable sits on which Gray digit (per formula)
        pos = np.zeros(n); neg = np.zeros(n)
        for v, s_ in cl:
            for vv, ss in zip(v, s_): (pos if ss > 0 else neg)[vv] += 1
        score = np.abs(pos - neg) if a.relabel == "bias" else -np.abs(pos - neg)
        order = np.argsort(score, kind="stable")             # least biased first -> digit 0 (best mixed), most biased -> digit 2
        slots = [K * c + j for j in range(K) for c in range(C)]  # digit-major: d0 of each cavity, then d1, then d2
        new = np.empty(n, int); new[order] = slots
        cl = [([int(new[vv]) for vv in v], s_) for v, s_ in cl]
        cl = [(list(np.array(v)[np.argsort(v)]), list(np.array(s_)[np.argsort(v)])) for v, s_ in cl]
    return m, cl

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
e0 = np.zeros(NF, np.complex128); e0[0] = 1; v0 = Up @ e0 if a.start in ("flat", "learned") else e0
V0RI = np.stack([v0.real, v0.imag], 1).astype(np.float32)
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
    Vi = jnp.asarray(np.stack([viol(c) for c in cls]).astype(np.uint8))   # uint8 (memory); cast to float in run()
    S = np.zeros((len(insts), NF * NF), np.float32)
    for b, ins in enumerate(insts):
        Eb = ins["E"] if a.relabel == "none" else sim.fast_energies(n, clauses(ins)[1])
        sx = np.append((Eb == 0).astype(np.float32), 0.0); S[b] = sx[IDX.reshape(-1)]
    assert len(set(ms)) == 1
    return G, Vi, jnp.asarray(S.reshape(-1, NF, NF)), n / ms[0]

def phaseD(wx):  # (B, 2^n) logical phase -> (B,NF,NF), spare = max
    ext = jnp.concatenate([wx, wx.max(1, keepdims=True)], 1); return ext[:, IDXf].reshape(-1, NF, NF)

def circuit(psi, A, PH, cost):
    """A (B,L,C,2) alpha re/im; PH (B,L,C,NF) SNAP phases; cost(l) -> (B,NF,NF) phase. Returns final psi."""
    P = a.pairs
    for l in range(NB):
        ll, q = divmod(l, P)
        if a.order == "csd" and ll < L and q == 0: psi = psi * jnp.exp(-1j * cost(ll * P))
        for c in range(C):
            psi = psi * (jnp.exp(-1j * PH[:, l, c])[:, :, None] if c == 0 else jnp.exp(-1j * PH[:, l, c])[:, None, :])
            Dm = disp(A[:, l, c, 0], A[:, l, c, 1]) if a.local == "disp" else mixU(A[:, l, c])
            psi = jnp.einsum("bij,bjk->bik", Dm, psi) if c == 0 else jnp.einsum("bkj,bij->bik", Dm, psi)
        if a.order == "sdc" and ll < L and q == P - 1: psi = psi * jnp.exp(-1j * cost(ll * P))
    return psi

# ---------------- GNN-SNAP ----------------
def init_params(rng):
    def W(i, o, s=1.0): return jnp.asarray(rng.normal(0, s / np.sqrt(i), (i, o)).astype(np.float32))
    p = AN.init_backbone(rng, a.arch, H, R, FV)
    p["G1"] = W(2 * H, H); p["g1"] = jnp.zeros(H); p["G2"] = jnp.zeros((H, 3 * NB)); p["Wa"] = jnp.zeros((H, 2 * NB))
    p["Wc"] = jnp.zeros((H, L)); p["P0"] = W(H, FM, 2.0); p["P1"] = W(H, FM, 2.0)
    p["u0"] = jnp.asarray(rng.normal(0, 1, FM).astype(np.float32)); p["u1"] = jnp.asarray(rng.normal(0, 1, FM).astype(np.float32))
    p["fq"] = jnp.zeros(FM); p["Wo"] = jnp.zeros((FM, NB)); p["Tsp"] = jnp.zeros((NB, NF - NU))
    if a.init_plain and not a.sizefree: p["T0"] = jnp.zeros((NB, NF))
    if a.snap_lin: p["Ws"] = jnp.zeros((H, NB))
    if a.start == "learned": p["s0"] = jnp.asarray(V0RI)
    return p

def rand_base(rng):
    r = rng.uniform(*a.ainit, NB) if a.local == "disp" else rng.uniform(0.3, 1.5, NB); ph = rng.uniform(0, 2 * np.pi, NB) if a.local == "disp" else np.zeros(NB); g = rng.uniform(*a.ginit, NB)
    return np.stack([r * np.cos(ph), r * np.sin(ph), g], 1).reshape(-1).astype(np.float32)

CV = jnp.asarray(np.array([[K * c + j for j in range(K)] for c in range(C)]))
def net(p, base, G):
    B = G["Ip"].shape[0]; hv, hc = AN.run_backbone(p, G, a.arch, H)
    g = jnp.concatenate([hv.mean(1), hc.mean(1)], -1)
    bv = (base[None] + jnp.tanh(g @ p["G1"] + p["g1"]) @ p["G2"]).reshape(B, NB, 3)
    hd = hv[:, CV]                                                               # (B,C,K,H)
    if a.local == "disp":
        A = jnp.broadcast_to(bv[:, :, None, :2], (B, NB, C, 2)) + jnp.transpose((hd.mean(2) @ p["Wa"]).reshape(B, C, NB, 2), (0, 2, 1, 3))
    else:
        A = bv[:, :, None, None, 0] + jnp.transpose((hd @ p["Wa"][:, :NB]), (0, 3, 1, 2))     # (B,L,C,K)
    U0 = p["u0"] + hd @ p["P0"]; U1 = p["u1"] + hd @ p["P1"]                     # (B,C,K,FM)
    z = (jnp.einsum("bcjm,jf->bcfm", U1, BT) + jnp.einsum("bcjm,jf->bcfm", U0, 1 - BT)) / K
    ph_u = jnp.transpose(jnp.tanh(z + p["fq"]) @ p["Wo"], (0, 3, 1, 2)) * MASKU         # (B,L,C,NF)
    sp = jnp.concatenate([jnp.zeros((NB, NU)), p["Tsp"]], 1)
    PH = ph_u + sp[None, :, None, :] + (p["T0"][None, :, None, :] if "T0" in p else 0.0)
    if "Ws" in p: PH = PH + jnp.einsum("bcjl,jf->blcf", hd @ p["Ws"], BT)
    w = hc @ p["Wc"]                                                             # (B,m,L)
    return A, PH, bv[:, :, 2], w

def bound(A):
    if a.amax <= 0 or a.local != "disp": return A
    r = jnp.sqrt((A ** 2).sum(-1, keepdims=True) + 1e-20); return A * (a.amax * jnp.tanh(r / a.amax) / r)

def run(p, base, G, Vi, S, nm):
    B = S.shape[0]
    if a.plain:
        A = jnp.broadcast_to(base[:, None, :2], (B, NB, C, 2)); PH = jnp.broadcast_to(p["T0"][None, :, None, :], (B, NB, C, NF))
        g = jnp.broadcast_to(base[None, :, 2], (B, NB)); w = jnp.zeros((B, Vi.shape[1], NB))
    else: A, PH, g, w = net(p, base, G) if not a.percase else (p["A"], p["PH"], p["g"], p["w"])
    A = bound(A)
    Vf = Vi.astype(jnp.float32); cost = lambda l: phaseD(nm * jnp.einsum("bc,bcx->bx", g[:, l][:, None] + w[:, :, l], Vf))
    if a.start == "learned":
        v = p["s0"][:, 0] + 1j * p["s0"][:, 1]; v = v / jnp.sqrt(jnp.sum(jnp.abs(v) ** 2)); ps0 = jnp.outer(v, v).astype(jnp.complex64)
    else: ps0 = PSI0
    psi = circuit(jnp.broadcast_to(ps0, (B, NF, NF)), A, PH, cost); pr = jnp.abs(psi) ** 2
    return jnp.sum(pr * S, (1, 2)), jnp.sum(pr * (1 - VALID.reshape(NF, NF)), (1, 2)), A

def summ(p):
    inv = 1 / np.maximum(p, 1e-300)
    return {"median_inv_p": float(np.median(inv)), "B90_inv_p": float(np.quantile(inv, 0.9)), "mean_p": float(np.mean(p)), "median_p": float(np.median(p))}

def rand_base_stats(insts):
    rg = np.array([2.0 ** n / i["nsol"] for i in insts]); return {"median_inv_p": float(np.median(rg)), "B90_inv_p": float(np.quantile(rg, 0.9)), "mean_p": float(np.mean(1 / rg))}

t00 = time.time()
Dtr, Dte, Dva = prep(trs), prep(tes), prep(vas)
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

rng = np.random.default_rng(a.seedoff + a.init)
if a.percase:  # ceiling diagnostic: per-instance free circuit params on val[:N]
    NPc = a.percase; pcs = (vas if a.pc_set == "val" else trs)[:NPc]; Dtr = prep(pcs); m0 = pcs[0]["m"]
    p0 = {"A": jnp.asarray(np.stack([rand_base(rng).reshape(NB, 3)[:, None, :2].repeat(C, 1) for _ in range(NPc)])),
          "PH": jnp.zeros((NPc, NB, C, NF)), "g": jnp.asarray(rng.uniform(*a.ginit, (NPc, NB)).astype(np.float32)), "w": jnp.zeros((NPc, m0, NB))}
    if a.pc_init:
        Jp = json.load(open(a.pc_init)); bs = np.array(Jp["base"], np.float32).reshape(NB, 3); T0 = np.array(Jp["T0"], np.float32)
        p0 = {"A": jnp.asarray(np.broadcast_to(bs[None, :, None, :2], (NPc, NB, C, 2)).copy()), "PH": jnp.asarray(np.broadcast_to(T0[None, :, None, :], (NPc, NB, C, NF)).copy()),
              "g": jnp.asarray(np.broadcast_to(bs[None, :, 2], (NPc, NB)).copy()), "w": jnp.zeros((NPc, m0, NB))}
    PC0 = {k_: v_ for k_, v_ in p0.items()}
    b0 = jnp.zeros(1)
elif a.plain:
    b0 = jnp.asarray(rand_base(rng).reshape(NB, 3)); p0 = {"T0": jnp.zeros((NB, NF))}
    if a.start == "learned": p0["s0"] = jnp.asarray(V0RI)
else:
    p0, b0 = init_params(rng), jnp.asarray(rand_base(rng))
    if a.init_plain and not a.eval_from:
        Jp = json.load(open(a.init_plain)); b0 = jnp.asarray(np.array(Jp["base"], np.float32).reshape(-1))
        if a.sizefree: p0.pop("T0", None)   # size-free: only base alpha/g from plain; SNAP fully from the free digit-pattern head (+ spare offsets)
        else: p0["T0"] = jnp.asarray(np.array(Jp["T0"], np.float32))
        if a.start == "learned" and "s0" in Jp: p0["s0"] = jnp.asarray(np.stack(Jp["s0"], 1).astype(np.float32))
th0, unr = ravel_pytree((p0, b0))
if a.warm and not a.eval_from:
    th0 = jnp.asarray(json.load(open(a.warm))["theta"], jnp.float32); assert th0.size == ravel_pytree((p0, b0))[0].size
if a.eval_from:
    J = json.load(open(a.eval_from)); NFt = J["args"]["NF"]; pt = dict(p0)
    padax = {"Tsp": 1, "T0": 1, "s0": 0}   # params with a Fock-level axis: zero-pad to the new NF
    for k_, ax in padax.items():
        if k_ in pt: sh = list(pt[k_].shape); sh[ax] += NFt - NF; pt[k_] = jnp.zeros(sh)
    _, unr_t = ravel_pytree((pt, b0)); pj, bj = unr_t(jnp.asarray(J["theta"], jnp.float32)); pj = dict(pj)
    for k_, ax in padax.items():
        if k_ in pj:
            sh = list(pj[k_].shape); sh[ax] = NF - NFt; pj[k_] = jnp.concatenate([pj[k_], jnp.zeros(sh)], ax)
    th = ravel_pytree((pj, bj))[0]
    ev = jax.jit(lambda th, G, Vi, S, nm: run(*unr(th), G, Vi, S, nm)); o = {}
    for nmk, D in (("train", Dtr), ("val", Dva), ("test", Dte)):
        pr, lk, A = (np.asarray(v, float) for v in ev(th, *D)); o[nmk] = summ(pr) | {"mean_leak": float(lk.mean()), "loss": float(np.mean(-np.log(pr + 1e-12)))}
    J.setdefault("nf_check", {})[str(NF)] = o; json.dump(J, open(a.eval_from, "w"))
    log(f"NF-check {a.eval_from} NF {NFt}->{NF}: train loss {J['train']['loss']:.4f}->{o['train']['loss']:.4f} test med/B90 {J['test']['median_inv_p']:.2f}/{J['test']['B90_inv_p']:.2f} -> {o['test']['median_inv_p']:.2f}/{o['test']['B90_inv_p']:.2f} leak {o['test']['mean_leak']:.4f}")
    sys.exit(0)
log(f"SD tag={a.tag} init={a.init} start={a.start} order={a.order} final={a.final} amax={a.amax} NF={NF} L={L} H={H} percase={a.percase} params={th0.size} train={Dtr[2].shape[0]} val={len(vas)} test={len(tes)} warm={a.warm}")
log(f"random baseline test {res['random_test']}")

def loss(th, G, Vi, S, nm):
    p, b = unr(th); pr, lk, _ = run(p, b, G, Vi, S, nm); nl = -jnp.log(pr + 1e-12); out = jnp.mean(nl)
    if a.tailw > 0: k = max(1, nl.shape[0] // 10); out = out + a.tailw * jnp.mean(jax.lax.top_k(nl, k)[0])
    if a.leakpen > 0: out = out + a.leakpen * jnp.mean(lk)
    if a.wd > 0 and not (a.percase or a.plain):
        out = out + a.wd * sum(jnp.sum(x_ ** 2) for x_ in jax.tree_util.tree_leaves({k_: v_ for k_, v_ in p.items() if k_ in ("G2", "Wa", "Wc", "Wo", "Ws")}))
    if a.percase and a.prox > 0: out = out + a.prox * sum(jnp.sum((p[k_] - PC0[k_]) ** 2) for k_ in PC0) / a.percase
    return out
vg = jax.jit(jax.value_and_grad(loss))
it = [0]
def _chunks(D, ch):
    G, Vi, S, nm = D; N = S.shape[0]
    if ch <= 0 or N <= ch: return [D]
    return [({k_: v_[i:i + ch] for k_, v_ in G.items()}, Vi[i:i + ch], S[i:i + ch], nm) for i in range(0, N, ch)]
PARTS = _chunks(Dtr, 0 if a.percase else a.chunk)
def f(th):
    thj = jnp.asarray(th, jnp.float32); v, g = 0.0, 0.0; N = Dtr[2].shape[0]
    for P_ in PARTS:
        vv, gg = vg(thj, *P_); w_ = P_[2].shape[0] / N; v = v + float(vv) * w_; g = g + np.asarray(gg, np.float64) * w_
    it[0] += 1
    if it[0] % 100 == 0: log(f"  fev {it[0]} loss {float(v):.4f}")
    return float(v), g
evalj = jax.jit(lambda th, G, Vi, S, nm: run(*unr(th), G, Vi, S, nm))
if a.distill and not a.percase:
    Z = np.load(a.distill); NDs = Z["A"].shape[0]; assert NDs <= Dtr[2].shape[0]
    Gs = {k_: v_[:NDs] for k_, v_ in Dtr[0].items()}
    tA, tPH, tCW = jnp.asarray(Z["A"]), jnp.asarray(Z["PHc"]), jnp.asarray(Z["CW"])
    sA, sPH, sCW = float(jnp.var(tA)) + 1e-3, float(jnp.var(tPH)) + 1e-3, float(jnp.var(tCW)) + 1e-3
    def sup(th):
        p, b = unr(th); A, PH, g, w = net(p, b, Gs); PHu = PH[..., :NU]; PHu = PHu - PHu.mean(-1, keepdims=True)
        CW = g[:, None, :L] + w[:, :, :L]
        return jnp.mean((A - tA) ** 2) / sA + jnp.mean((PHu - tPH) ** 2) / sPH + jnp.mean((CW - tCW) ** 2) / sCW
    svg = jax.jit(jax.value_and_grad(sup)); t1 = time.time()
    s0_ = float(svg(jnp.asarray(th0, jnp.float32))[0])
    rs = minimize(lambda x: tuple(np.asarray(v, np.float64) for v in svg(jnp.asarray(x, jnp.float32))), np.asarray(th0, np.float64), jac=True, method="L-BFGS-B",
                  options={"maxiter": a.sup_iters, "maxfun": int(1.5 * a.sup_iters)})
    th0 = jnp.asarray(rs.x, jnp.float32); res["sup"] = {"loss0": s0_, "loss": float(rs.fun), "nit": int(rs.nit), "sec": time.time() - t1}
    log(f"distill: supervised loss {s0_:.4f} -> {rs.fun:.4f} ({rs.nit} it, {time.time()-t1:.0f}s); then -log p fine-tune")
    if a.maxiter == 0:
        pass
l0 = f(np.asarray(th0, np.float64))[0]; log(f"loss0 {l0:.4f} (first call incl. compile {time.time()-t00:.0f}s)")
t0 = time.time()
BEST = {"v": np.inf, "x": None, "it": 0}; ITc = [0]; ES_hist = []
def es_val(x):
    pr, lk, _ = evalj(jnp.asarray(x, jnp.float32), *Dva); pr = np.asarray(pr, float); inv = 1 / np.maximum(pr, 1e-300)
    return float((np.log(np.median(inv)) + np.log(np.quantile(inv, 0.9))) / 2), float(np.mean(-np.log(pr + 1e-12)))
def cb(xk):
    ITc[0] += 1
    if a.es and ITc[0] % a.es == 0:
        S_, lv = es_val(xk); ES_hist.append([ITc[0], S_, lv])
        if S_ < BEST["v"]: BEST.update(v=S_, x=np.array(xk), it=ITc[0])
        log(f"  ES it {ITc[0]} val S {S_:.4f} val loss {lv:.4f} (best S {BEST['v']:.4f} @ {BEST['it']})")
if a.adam > 0:  # minibatch Adam over a large fresh-instance pool ("infinite data" regime), early stopping on val S
    from types import SimpleNamespace
    pool = sim.instances(n, a.npool, TRB); t1 = time.time(); Dp = prep(pool); log(f"pool {len(pool)} prepped ({time.time()-t1:.0f}s)")
    vgb = jax.jit(jax.value_and_grad(loss)); x = jnp.asarray(th0, jnp.float32); mom = jnp.zeros_like(x); vel = jnp.zeros_like(x)
    @jax.jit
    def adam_step(x, mom, vel, g, t, lr):
        mom = 0.9 * mom + 0.1 * g; vel = 0.999 * vel + 0.001 * g * g
        mh = mom / (1 - 0.9 ** t); vh = vel / (1 - 0.999 ** t); return x - lr * mh / (jnp.sqrt(vh) + 1e-8), mom, vel
    rngb = np.random.default_rng(77 + a.init); S_, lv = es_val(np.asarray(x)); BEST.update(v=S_, x=np.asarray(x, np.float64), it=0); ES_hist.append([0, S_, lv])
    log(f"  ES step 0 val S {S_:.4f} val loss {lv:.4f}"); run_l = []
    for t in range(1, a.adam + 1):
        idx = rngb.choice(len(pool), a.bs, replace=False); G_, Vi_, S__, nm_ = Dp
        v, g = vgb(x, {k_: v_[idx] for k_, v_ in G_.items()}, Vi_[idx], S__[idx], nm_)
        lr = a.lr * (0.1 + 0.9 * 0.5 * (1 + np.cos(np.pi * t / a.adam)))
        x, mom, vel = adam_step(x, mom, vel, g, t, lr); run_l.append(float(v))
        if t % (a.es or 100) == 0:
            S_, lv = es_val(np.asarray(x)); ES_hist.append([t, S_, lv])
            if S_ < BEST["v"]: BEST.update(v=S_, x=np.asarray(x, np.float64), it=t)
            log(f"  step {t} batch loss(avg) {np.mean(run_l[-(a.es or 100):]):.4f} lr {lr:.2e} | val S {S_:.4f} val loss {lv:.4f} (best {BEST['v']:.4f} @ {BEST['it']})")
    res["es"] = {"best_it": BEST["it"], "hist": ES_hist}; log(f"  Adam done: using step {BEST['it']} (val S {BEST['v']:.4f})")
    r = SimpleNamespace(x=BEST["x"], fun=float(np.mean(run_l[-100:])), nit=a.adam, nfev=a.adam, message="adam")
    a.es = 0
if a.es and not a.percase:
    S_, lv = es_val(np.asarray(th0, np.float64)); BEST.update(v=S_, x=np.asarray(th0, np.float64), it=0); ES_hist.append([0, S_, lv]); log(f"  ES it 0 val S {S_:.4f} val loss {lv:.4f}")
if a.adam <= 0:
    r = minimize(f, np.asarray(th0, np.float64), jac=True, method="L-BFGS-B", callback=cb if (a.es and not a.percase) else None,
                 options={"maxiter": a.maxiter if not a.smoke else a.smoke_it, "maxfun": int(1.5 * a.maxiter)})
if a.es and not a.percase:
    S_, lv = es_val(r.x); ES_hist.append([ITc[0], S_, lv])
    if S_ < BEST["v"]: BEST.update(v=S_, x=np.array(r.x), it=ITc[0])
    log(f"  ES final: using theta from it {BEST['it']} (val S {BEST['v']:.4f}) of {ITc[0]}"); res["es"] = {"best_it": BEST["it"], "hist": ES_hist}
    r.x = BEST["x"]
th = jnp.asarray(r.x, jnp.float32)
out = {}
sets = (("train", Dtr),) if a.percase else (("train", Dtr), ("val", Dva), ("test", Dte))
for nmk, D in sets:
    pr, lk, A = (np.asarray(v, float) for v in evalj(th, *D)); A = np.asarray(bound(jnp.asarray(A)))
    mag = np.sqrt((A ** 2).sum(-1)) if a.local == "disp" else np.abs(A).mean(-1)   # |alpha| (B,NB,C); mix: mean |beta|
    out[nmk] = summ(pr) | {"mean_leak": float(lk.mean()), "max_leak": float(lk.max()), "loss": float(np.mean(-np.log(pr + 1e-12))),
                           "alpha_abs_mean_per_layer": mag.mean((0, 2)).tolist(), "alpha_abs_std_per_layer": mag.std((0, 2)).tolist(),
                           "alpha_abs_per_layer_cav": mag.mean(0).tolist()}
    out[nmk]["S"] = float((np.log(out[nmk]["median_inv_p"]) + np.log(out[nmk]["B90_inv_p"])) / 2)
    if nmk in ("test", "train") : out[nmk]["p"] = pr.tolist(); out[nmk]["leak"] = lk.tolist()
pb, bb = unr(th)
if a.plain: res["T0"] = np.asarray(pb["T0"]).tolist()
if "s0" in pb: v = np.asarray(pb["s0"]); v = v[:, 0] + 1j * v[:, 1]; v /= np.linalg.norm(v); res["s0"] = [v.real.tolist(), v.imag.tolist()]; res["s0_pops"] = (np.abs(v) ** 2).tolist()
extra = {} if a.percase else dict(base=np.asarray(bb).reshape(NB, 3).tolist(), base_alpha_abs=np.sqrt((np.asarray(bb).reshape(NB, 3)[:, :2] ** 2).sum(1)).tolist(),
           base_alpha_abs0=np.sqrt((np.asarray(b0).reshape(NB, 3)[:, :2] ** 2).sum(1)).tolist())
res.update(init=a.init, loss0=l0, loss=float(r.fun), nit=int(r.nit), nfev=int(r.nfev), msg=str(r.message), train_sec=time.time() - t0, sec=time.time() - t00,
           **extra, **out, theta=r.x.tolist())
json.dump(res, open(OUT, "w"))
if a.percase and a.pc_set == "train":
    pp = unr(th)[0]; PHu = np.asarray(pp["PH"])[..., :NU]; PHu = PHu - PHu.mean(-1, keepdims=True)
    np.savez(OUT.replace(".json", "_targets.npz"), A=np.asarray(pp["A"]), PHc=PHu, CW=np.asarray(pp["g"])[:, None, :L] + np.asarray(pp["w"])[:, :, :L], p=np.array(out["train"]["p"]))
k = "train" if a.percase else "test"
log(f"DONE {a.tag}: loss {l0:.4f} -> {r.fun:.4f} ({r.nit} it, {time.time()-t0:.0f}s) | {'val' if not a.percase else 'percase'} " + ("" if a.percase else f"{out['val']['median_inv_p']:.2f}/{out['val']['B90_inv_p']:.2f} S={out['val']['S']:.3f} | test ")
    + f"med/B90 {out[k]['median_inv_p']:.2f}/{out[k]['B90_inv_p']:.2f} mean p {out[k]['mean_p']:.3f} leak {out[k]['mean_leak']:.4f} |alpha|/layer {['%.3f' % v for v in out[k]['alpha_abs_mean_per_layer']]}")
if not a.percase and a.nfcheck and not a.smoke:
    import subprocess
    subprocess.run([sys.executable] + sys.argv + ["--eval_from", OUT, "--NF", str(a.nfcheck)])
