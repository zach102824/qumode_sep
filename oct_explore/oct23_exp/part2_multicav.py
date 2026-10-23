"""Part 2: Multi-Cavity 4-SAT Scaling (C = 2, 3, 4, 5 Cavities, n = 8, 12, 16, 20)
with Physical SNAP+D Gates (0..15 Computational SNAP Only), Global Cost Phase, and Pure GNN (L <= 15).

Strict Audit & Physical Rules:
  1. Every cavity holds K = 4 bits in D = 16 computational Fock levels (0..15) via Gray code.
  2. Every SNAP gate has free phases ONLY on computational Fock levels 0..15 (strictly 0 phase on f >= 16).
  3. Displacements D(alpha) are computed in NF = 40 Fock levels and projected onto the 16x16 valid block
     V_c = P D_{NF=40}(alpha) S(theta_{0..15}) P, so any photon leaking to f >= 16 at any intermediate step
     is permanently lost as leakage (zero refocusing from leaked levels).
  4. State preparation uses the physical k_prep = 2 (98.85% flat, 0.29% leak) or k_prep = 3 (99.998% flat)
     SNAP+D prep from preps.json.
  5. Zero classical pre-solving: canonical variable ordering (relabel = none, flips = 0), and Pure GNN uses
     ONLY raw factor-graph degree features fv[:, :, 0:3] = [1, pos_deg, neg_deg] (fv[:, :, 3:7] = 0).
  6. Reports Test Med / B90 of 1/p, Mean Leakage, Zero-Oracle (gamma=0) Med / B90, and scaling fit b^n
     across n = 8, 12, 16, 20 vs WalkSATlm tries (2.01, 2.92, 4.55, 5.63; 1.092^n) and lookups (14.0, 32.5, 71.0, 110; 1.191^n).
"""
from __future__ import annotations
import argparse, json, os, sys, time
import numpy as np
import jax, jax.numpy as jnp
from jax.flatten_util import ravel_pytree
from scipy.optimize import minimize

ROOT = "/workspace/qumode_sep/oct_explore"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "v3"))
import sim, arch_nets as AN

ap = argparse.ArgumentParser()
ap.add_argument("--tag", required=True)
ap.add_argument("--mode", default="shared", choices=["shared", "gnn"])
ap.add_argument("--L", type=int, default=6)
ap.add_argument("--pairs", type=int, default=1, choices=[1, 2], help="SNAP+D blocks per cavity per cost layer")
ap.add_argument("--kprep", type=int, default=2, choices=[2, 3], help="Physical SNAP+D blocks for initial flat state prep")
ap.add_argument("--ntrain8", type=int, default=5000)
ap.add_argument("--ntrain12", type=int, default=1000)
ap.add_argument("--nstarts", type=int, default=8)
ap.add_argument("--screen_it", type=int, default=60)
ap.add_argument("--maxiter", type=int, default=200)
ap.add_argument("--grow_from", default="", help="JSON of smaller-L shared run to interpolate initial parameters")
ap.add_argument("--init_shared", default="", help="Shared JSON to initialize GNN base parameters")
ap.add_argument("--arch", default="mp", choices=["mp", "mp64", "mp128", "attn", "ds", "mlp"])
ap.add_argument("--H", type=int, default=64)
ap.add_argument("--rounds", type=int, default=3)
ap.add_argument("--clause_w", action="store_true", help="Enable GNN per-clause weights in global cost phase")
ap.add_argument("--steps", type=int, default=1200)
ap.add_argument("--bs8", type=int, default=125)
ap.add_argument("--bs12", type=int, default=50)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--tailw", type=float, default=0.5)
ap.add_argument("--leakpen", type=float, default=0.1)
ap.add_argument("--amax", type=float, default=1.2)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

if a.arch == "mp64":
    a.arch, a.H, a.rounds = "mp", 64, 3
elif a.arch == "mp128":
    a.arch, a.H, a.rounds = "mp", 128, 4

K = 4
NU = 1 << K  # 16 computational Fock levels (0..15) per cavity
NF_DISP = 40
L = a.L
PAIRS = a.pairs
NB = L * PAIRS

TRB, TEB, VAB = 51_000_000_000, 52_000_000_000, 53_000_000_000
OUT_DIR = os.path.join(ROOT, "results", "oct23_exp_part2")
os.makedirs(OUT_DIR, exist_ok=True)
OUT = os.path.join(OUT_DIR, f"{a.tag}.json")
PREPS = json.load(open(os.path.join(ROOT, "results", "oct23_exp", "preps.json")))

WS_TRIES = {8: (2.014, 3.739), 12: (2.92, 6.28), 16: (4.55, 9.09), 20: (5.63, 14.2)}
WS_LOOK  = {8: (13.989, 29.396), 12: (32.5, 76.1), 16: (71.0, 147.0), 20: (110.0, 290.0)}

log = lambda *s: print(time.strftime("%H:%M:%S"), f"[{a.tag}]", *s, flush=True)
gray = lambda i: i ^ (i >> 1)
def ungray(g):
    s = g >> 1
    while s: g ^= s; s >>= 1
    return g

# Permutation mapping logical cavity value g in 0..15 to Fock level f = ungray(g)
PERM_LOG_TO_FOCK = np.array([ungray(g) for g in range(NU)], dtype=np.int64)
# In logical basis (g = 0..15), bit j of g is simply (g >> j) & 1; wait: in sim.gen_instance,
# cavity c holds 4 variables [4*c, 4*c+1, 4*c+2, 4*c+3] where var 4*c is MSB of the 4-bit block!
# In v3/native/nsim.py and compiler.py, logical index g of cavity c has bit j_var (0..3) at bit position (3 - j_var).
# Let's match sd_n6.py / part1_2cav.py exactly: in Fock basis f in 0..15, Gray digit j is (gray(f) >> j) & 1,
# and variable 4*c + j has bit value (gray(f) >> j) & 1!
# So for logical index g in 0..15 (where g = sum_{j=0..3} bit_j * 2^(3 - j)), the corresponding Gray code is
# gray_val = sum_{j=0..3} bit_j * (1 << j) = bit_reverse_4(g), and Fock level is f = ungray(bit_reverse_4(g))!
FOCK_OF_LOG = np.array([ungray(sum(((g >> (3 - j)) & 1) << j for j in range(K))) for g in range(NU)], dtype=np.int64)

# Bit table in Fock basis: BT_FOCK[j, f] = (gray(f) >> j) & 1 for f in 0..15
BT_FOCK = np.zeros((K, NU), np.float32)
for f in range(NU):
    for j in range(K):
        BT_FOCK[j, f] = (gray(f) >> j) & 1
BT_FOCK = jnp.asarray(BT_FOCK)

# Exact single-cavity displacement operator in NF_DISP = 40 Fock levels
_aa = np.diag(np.sqrt(np.arange(1, NF_DISP)), 1)
_lam, _V = np.linalg.eigh(1j * (_aa.T - _aa))
LAM = jnp.asarray(_lam.astype(np.float32))
VD = jnp.asarray(_V.astype(np.complex64))
NV = jnp.arange(NF_DISP, dtype=jnp.float32)

def disp_valid_single(re, im):
    """Scalar (re, im) -> (16, 16) valid Fock-basis block P D_{NF=40}(alpha) P."""
    if a.amax > 0:
        r_raw = jnp.sqrt(re ** 2 + im ** 2 + 1e-20)
        sc = a.amax * jnp.tanh(r_raw / a.amax) / r_raw
        re, im = re * sc, im * sc
    r = jnp.sqrt(re ** 2 + im ** 2 + 1e-20)
    ph = jnp.arctan2(im, re)
    W = jnp.exp(1j * ph * NV)[:, None] * VD
    D_full = (W * jnp.exp(-1j * r * LAM)[None]) @ W.conj().T
    return D_full[:NU, :NU]

def disp_valid_batch(re, im):
    """Batch (B,) -> (B, 16, 16) valid Fock-basis block P D_{NF=40}(alpha) P."""
    if a.amax > 0:
        r_raw = jnp.sqrt(re ** 2 + im ** 2 + 1e-20)
        sc = a.amax * jnp.tanh(r_raw / a.amax) / r_raw
        re, im = re * sc, im * sc
    r = jnp.sqrt(re ** 2 + im ** 2 + 1e-20)
    ph = jnp.arctan2(im, re)
    W = jnp.exp(1j * ph[:, None] * NV[None])[:, :, None] * VD[None]
    D_sub = W[:, :NU, :]
    return jnp.einsum("bik,bk,bjk->bij", D_sub, jnp.exp(-1j * r[:, None] * LAM[None]), D_sub.conj())

# Initial single-cavity state on Fock 0..15 from physical k_prep-block state prep (norm^2 = 1 - leak_prep)
_prep_info = PREPS[f"sd_K4_k{a.kprep}"]
V0_FOCK = jnp.asarray((np.array(_prep_info["psi_re"][:NU]) + 1j * np.array(_prep_info["psi_im"][:NU])).astype(np.complex64))
V0_LOG = V0_FOCK[FOCK_OF_LOG]

def make_psi0_log(C_cav):
    v = V0_LOG
    for _ in range(C_cav - 1):
        v = jnp.multiply.outer(v, V0_LOG)
    return v  # shape (16,)*C_cav

def apply_cav_shared(M_log, psi, c, C_cav):
    let = "defghijk"[:C_cav]
    o = let[c]
    out = let.replace(o, "y")
    return jnp.einsum(f"y{o},b{let}->b{out}", M_log, psi)

def apply_cav_batch(M_log_b, psi, c, C_cav):
    let = "defghijk"[:C_cav]
    o = let[c]
    out = let.replace(o, "y")
    return jnp.einsum(f"by{o},b{let}->b{out}", M_log_b, psi)

# ---------------- Dataset Caching & Loading ----------------
_DATA_CACHE = {}
def get_instances(n, count, base, split_label):
    key = (n, count, int(base))
    if key in _DATA_CACHE:
        return _DATA_CACHE[key]
    cache_file = os.path.join(OUT_DIR, f"inst_n{n}_{split_label}_{count}.npz")
    if os.path.exists(cache_file):
        Z = np.load(cache_file)
        Es = Z["E"]
        nsols = Z["nsol"]
        cls_v = Z["cls_v"]
        cls_s = Z["cls_s"]
    else:
        t0 = time.time()
        # Check if n=8 and we can slice from oct23_exp/data_n8.npz
        p1_n8 = os.path.join(ROOT, "results", "oct23_exp", "data_n8.npz")
        if n == 8 and os.path.exists(p1_n8) and split_label in ("train", "val", "test"):
            Z8 = np.load(p1_n8)
            if len(Z8[f"{split_label}_E"]) >= count:
                Es = Z8[f"{split_label}_E"][:count]
                nsols = Z8[f"{split_label}_nsol"][:count]
                cls_v = Z8[f"{split_label}_cls_v"][:count]
                cls_s = Z8[f"{split_label}_cls_s"][:count]
                _DATA_CACHE[key] = (Es, nsols, cls_v, cls_s)
                return _DATA_CACHE[key]
        insts = sim.instances(n, count, base)
        m = int(round(9.93 * n))
        Es = np.stack([ins["E"] for ins in insts]).astype(np.uint8)
        nsols = np.array([ins["nsol"] for ins in insts], dtype=np.int32)
        cls_v = np.zeros((count, m, 4), dtype=np.uint8)
        cls_s = np.zeros((count, m, 4), dtype=np.int8)
        for i, ins in enumerate(insts):
            _, cl, _ = sim.gen_instance(n, int(ins["seed"]))
            for c_i, (v, s) in enumerate(cl):
                cls_v[i, c_i] = v
                cls_s[i, c_i] = s
        np.savez_compressed(cache_file, E=Es, nsol=nsols, cls_v=cls_v, cls_s=cls_s)
        log(f"Cached {cache_file} ({count} insts at n={n}) in {time.time()-t0:.1f}s")
    _DATA_CACHE[key] = (Es, nsols, cls_v, cls_s)
    return _DATA_CACHE[key]

def build_graphs(cls_v, cls_s, n):
    """Build pure factor-graph tensors (Ip, In, fv[:,:,0:3], deg) with fv[:,:,3:7] = 0 (zero classical pre-solving)."""
    B, m, _ = cls_v.shape
    b_idx = np.arange(B)[:, None, None]
    c_idx = np.arange(m)[None, :, None]
    Ip = np.zeros((B, m, n), np.float32)
    In = np.zeros((B, m, n), np.float32)
    np.add.at(Ip, (b_idx, c_idx, cls_v), (cls_s > 0).astype(np.float32))
    np.add.at(In, (b_idx, c_idx, cls_v), (cls_s < 0).astype(np.float32))
    fv = np.zeros((B, n, 7), np.float32)
    fv[:, :, 0] = 1.0
    fv[:, :, 1] = Ip.sum(1) / (2.0 * m / n)
    fv[:, :, 2] = In.sum(1) / (2.0 * m / n)
    # fv[:, :, 3:7] strictly 0.0!
    deg = np.maximum(Ip.sum(1) + In.sum(1), 1.0)
    bb = (cls_s < 0).astype(np.int8)  # violated iff bit == bb
    return {"Ip": jnp.asarray(Ip), "In": jnp.asarray(In), "fv": jnp.asarray(fv), "deg": jnp.asarray(deg),
            "VP": jnp.asarray(cls_v.astype(np.int32)), "BB": jnp.asarray(bb)}

# ---------------- Shared-Parameter Simulator & Optimizer ----------------
def unpack_shared(x):
    """x has size NB*2 + NB*16 + L:
      A: (NB, 2), TH: (NB, 16), g: (L,)
    """
    nA = NB * 2
    nT = NB * NU
    A = x[:nA].reshape(NB, 2)
    TH = x[nA:nA + nT].reshape(NB, NU)
    g = x[nA + nT:nA + nT + L]
    return A, TH, g

def block_matrix_shared(A, TH, l_blk):
    """Returns 16x16 matrix in LOGICAL basis for block l_blk: V_fock = D_valid(A[l_blk]) * exp(-i TH[l_blk])."""
    V_fock = disp_valid_single(A[l_blk, 0], A[l_blk, 1]) * jnp.exp(-1j * TH[l_blk])[None, :]
    return V_fock[jnp.ix_(FOCK_OF_LOG, FOCK_OF_LOG)]

def sim_shared_n(x, Cn, Succ, n):
    """Cn: (B, 2^n) float32 normalized cost E*n/m; Succ: (B, 2^n) float32."""
    C_cav = n // 4
    B = Cn.shape[0]
    A, TH, g = unpack_shared(x)
    psi0 = make_psi0_log(C_cav)
    psi = jnp.broadcast_to(psi0, (B,) + (NU,) * C_cav)
    Cn_sh = Cn.reshape(psi.shape)
    Ms = [block_matrix_shared(A, TH, l_blk) for l_blk in range(NB)]
    for l in range(L):
        psi = psi * jnp.exp(-1j * g[l] * Cn_sh)
        for q in range(PAIRS):
            M_log = Ms[l * PAIRS + q]
            for c in range(C_cav):
                psi = apply_cav_shared(M_log, psi, c, C_cav)
    pr = jnp.abs(psi.reshape(B, -1)) ** 2
    p_succ = jnp.sum(pr * Succ, axis=1)
    p_leak = 1.0 - jnp.sum(pr, axis=1)
    return p_succ, p_leak

def eval_metrics(p_succ, p_leak, p_zero, Succ_00, n):
    p_succ = np.asarray(p_succ, float)
    p_leak = np.asarray(p_leak, float)
    p_zero = np.asarray(p_zero, float)
    inv = 1.0 / np.maximum(p_succ, 1e-12)
    inv_z = 1.0 / np.maximum(p_zero, 1e-12)
    med = float(np.median(inv))
    b90 = float(np.quantile(inv, 0.9))
    ws_t = WS_TRIES[n]
    ws_l = WS_LOOK[n]
    return {
        "median_inv_p": med,
        "B90_inv_p": b90,
        "S": float((np.log(med) + np.log(b90)) / 2.0),
        "mean_p": float(np.mean(p_succ)),
        "mean_leak": float(np.mean(p_leak)),
        "max_leak": float(np.max(p_leak)),
        "presolve_00_frac": float(np.mean(Succ_00 > 0)),
        "zero_oracle_med": float(np.median(inv_z)),
        "zero_oracle_B90": float(np.quantile(inv_z, 0.9)),
        "beats_ws_tries_med": bool(med < ws_t[0]),
        "beats_ws_tries_b90": bool(b90 < ws_t[1]),
        "beats_ws_look_med": bool(med < ws_l[0]),
        "beats_ws_look_b90": bool(b90 < ws_l[1]),
    }

def fit_scaling(results_by_n):
    ns = np.array([8, 12, 16, 20], dtype=float)
    meds = np.array([results_by_n[str(int(n))]["median_inv_p"] for n in ns], dtype=float)
    b90s = np.array([results_by_n[str(int(n))]["B90_inv_p"] for n in ns], dtype=float)
    slope_m, intercept_m = np.polyfit(ns, np.log2(meds), 1)
    slope_b, intercept_b = np.polyfit(ns, np.log2(b90s), 1)
    return {
        "b_median": float(2.0 ** slope_m),
        "C_median": float(2.0 ** intercept_m),
        "b_B90": float(2.0 ** slope_b),
        "C_B90": float(2.0 ** intercept_b),
    }

def run_shared_pipeline():
    t0 = time.time()
    E8_tr, _, _, _ = get_instances(8, a.ntrain8, TRB, "train")
    E12_tr, _, _, _ = get_instances(12, a.ntrain12, TRB, "train")
    E12_va, _, _, _ = get_instances(12, 200, VAB, "val")

    m8 = int(round(9.93 * 8)); m12 = int(round(9.93 * 12))
    Cn8_tr = jnp.asarray(E8_tr.astype(np.float32) * (8.0 / m8))
    S8_tr = jnp.asarray((E8_tr == 0).astype(np.float32))
    Cn12_tr = jnp.asarray(E12_tr.astype(np.float32) * (12.0 / m12))
    S12_tr = jnp.asarray((E12_tr == 0).astype(np.float32))
    Cn12_va = jnp.asarray(E12_va.astype(np.float32) * (12.0 / m12))
    S12_va = jnp.asarray((E12_va == 0).astype(np.float32))

    def loss_n(x, Cn, S, n):
        p_s, p_l = sim_shared_n(x, Cn, S, n)
        nl = -jnp.log(p_s + 1e-12)
        out = jnp.mean(nl)
        if a.tailw > 0:
            k = max(1, nl.shape[0] // 10)
            out = out + a.tailw * jnp.mean(jax.lax.top_k(nl, k)[0])
        if a.leakpen > 0:
            out = out + a.leakpen * jnp.mean(p_l)
        return out

    vg8 = jax.jit(jax.value_and_grad(lambda x, c, s: loss_n(x, c, s, 8)))
    vg12 = jax.jit(jax.value_and_grad(lambda x, c, s: loss_n(x, c, s, 12)))
    ev12 = jax.jit(lambda x, c, s: sim_shared_n(x, c, s, 12))

    def make_obj( n8_cnt, n12_cnt):
        ch8 = 1000; ch12 = 250
        p8 = [(Cn8_tr[i:i+ch8], S8_tr[i:i+ch8], (min(i+ch8, n8_cnt) - i) / float(n8_cnt)) for i in range(0, n8_cnt, ch8)]
        p12 = [(Cn12_tr[i:i+ch12], S12_tr[i:i+ch12], (min(i+ch12, n12_cnt) - i) / float(n12_cnt)) for i in range(0, n12_cnt, ch12)]
        def obj(x):
            xj = jnp.asarray(x, jnp.float32)
            tot_v = 0.0; tot_g = np.zeros_like(x, dtype=np.float64)
            for c, s, w in p8:
                v, g = vg8(xj, c, s)
                tot_v += 0.5 * float(v) * w; tot_g += 0.5 * np.asarray(g, np.float64) * w
            for c, s, w in p12:
                v, g = vg12(xj, c, s)
                tot_v += 0.5 * float(v) * w; tot_g += 0.5 * np.asarray(g, np.float64) * w
            return tot_v, tot_g
        return obj

    obj_scr = make_obj(min(1000, a.ntrain8), min(250, a.ntrain12))
    obj_full = make_obj(a.ntrain8, a.ntrain12)

    rng = np.random.default_rng(1000 * L + 100 * PAIRS + a.seed)
    inits = []
    if a.grow_from and os.path.exists(a.grow_from):
        J0 = json.load(open(a.grow_from))
        x_prev = np.array(J0["x_best"], np.float32)
        L_prev = J0["args"]["L"]; P_prev = J0["args"]["pairs"]; NB_prev = L_prev * P_prev
        A_p = x_prev[:NB_prev*2].reshape(L_prev, P_prev, 2)
        TH_p = x_prev[NB_prev*2:NB_prev*18].reshape(L_prev, P_prev, NU)
        g_p = x_prev[NB_prev*18:NB_prev*18+L_prev]
        t_old = np.linspace(0, 1, L_prev); t_new = np.linspace(0, 1, L)
        A_new = np.zeros((L, PAIRS, 2), np.float32)
        TH_new = np.zeros((L, PAIRS, NU), np.float32)
        for q in range(min(PAIRS, P_prev)):
            for d in range(2): A_new[:, q, d] = np.interp(t_new, t_old, A_p[:, q, d])
            for f in range(NU): TH_new[:, q, f] = np.interp(t_new, t_old, TH_p[:, q, f])
        g_new = np.interp(t_new, t_old, g_p).astype(np.float32)
        x_interp = np.concatenate([A_new.reshape(-1), TH_new.reshape(-1), g_new])
        inits.append(x_interp)
        for _ in range(min(3, a.nstarts - 1)):
            inits.append(x_interp + rng.normal(0, 0.05, size=x_interp.shape).astype(np.float32))

    while len(inits) < a.nstarts:
        idx = len(inits)
        r = rng.uniform(0.08, 0.25 + 0.1 * (idx % 3), NB)
        ph = rng.uniform(0, 2 * np.pi, NB)
        A0 = np.stack([r * np.cos(ph), r * np.sin(ph)], -1).astype(np.float32)
        TH0 = rng.normal(0, 0.25, (NB, NU)).astype(np.float32)
        g0 = np.linspace(0.6, 2.5 + 1.5 * (idx % 3), L).astype(np.float32)
        inits.append(np.concatenate([A0.reshape(-1), TH0.reshape(-1), g0]))

    log(f"Start Part2 Shared L={L} pairs={PAIRS} kprep={a.kprep} ntrain8={a.ntrain8} ntrain12={a.ntrain12} params={inits[0].size}")
    scr = []
    for i, x0 in enumerate(inits):
        r = minimize(obj_scr, np.asarray(x0, np.float64), jac=True, method="L-BFGS-B",
                     options={"maxiter": a.screen_it, "maxfun": int(1.5 * a.screen_it)})
        scr.append((float(r.fun), i, r.x))
        log(f"  screen {i+1}/{len(inits)}: loss {r.fun:.4f} ({time.time()-t0:.1f}s)")
    scr.sort(key=lambda t: t[0])

    best_val = None
    for rank, (l_scr, idx, x_s) in enumerate(scr[:2]):
        t1 = time.time()
        r = minimize(obj_full, np.asarray(x_s, np.float64), jac=True, method="L-BFGS-B",
                     options={"maxiter": a.maxiter, "maxfun": int(1.5 * a.maxiter)})
        xj = jnp.asarray(r.x, jnp.float32)
        p_v, lk_v = ev12(xj, Cn12_va, S12_va)
        inv_v = 1.0 / np.maximum(np.asarray(p_v, float), 1e-12)
        med_v, b90_v = float(np.median(inv_v)), float(np.quantile(inv_v, 0.9))
        S_v = 0.5 * (np.log(med_v) + np.log(b90_v))
        log(f"  refine #{rank} (init {idx}): loss {r.fun:.4f} ({r.nit} it, {time.time()-t1:.1f}s) | val12 {med_v:.2f}/{b90_v:.2f} S={S_v:.4f}")
        if best_val is None or S_v < best_val["S"]:
            best_val = {"S": S_v, "med": med_v, "B90": b90_v, "x": r.x.astype(np.float32), "loss": float(r.fun), "init": idx}

    x_best = jnp.asarray(best_val["x"], jnp.float32)
    x_zero = jnp.concatenate([x_best[:NB*18], jnp.zeros(L, jnp.float32)])

    test_res = {}
    for n_te in [8, 12, 16, 20]:
        t_ev = time.time()
        E_te, _, _, _ = get_instances(n_te, 100, TEB, "test")
        m_te = int(round(9.93 * n_te))
        ch = 50 if n_te <= 12 else (10 if n_te == 16 else 2)
        ev_fn = jax.jit(lambda x, c, s, n_val=n_te: sim_shared_n(x, c, s, n_val))
        ps, lks, pzs = [], [], []
        for i in range(0, 100, ch):
            Cn_b = jnp.asarray(E_te[i:i+ch].astype(np.float32) * (n_te / float(m_te)))
            S_b = jnp.asarray((E_te[i:i+ch] == 0).astype(np.float32))
            p_b, lk_b = ev_fn(x_best, Cn_b, S_b)
            pz_b, _ = ev_fn(x_zero, Cn_b, S_b)
            ps.append(np.asarray(p_b)); lks.append(np.asarray(lk_b)); pzs.append(np.asarray(pz_b))
        p_all = np.concatenate(ps); lk_all = np.concatenate(lks); pz_all = np.concatenate(pzs)
        test_res[str(n_te)] = eval_metrics(p_all, lk_all, pz_all, (E_te[:, 0] == 0).astype(float), n_te)
        test_res[str(n_te)]["p"] = p_all.tolist()
        log(f"  TEST n={n_te}: med/B90 = {test_res[str(n_te)]['median_inv_p']:.2f}/{test_res[str(n_te)]['B90_inv_p']:.2f} (zero_oracle={test_res[str(n_te)]['zero_oracle_med']:.1f}/{test_res[str(n_te)]['zero_oracle_B90']:.1f}, leak={test_res[str(n_te)]['mean_leak']*100:.2f}%, {time.time()-t_ev:.1f}s)")

    scaling = fit_scaling(test_res)
    A_b, TH_b, g_b = unpack_shared(x_best)
    out = {
        "args": vars(a), "val12": {"median_inv_p": best_val["med"], "B90_inv_p": best_val["B90"], "S": best_val["S"]},
        "test": test_res, "scaling": scaling,
        "alpha_abs": np.hypot(np.asarray(A_b)[:, 0], np.asarray(A_b)[:, 1]).tolist(),
        "g": np.asarray(g_b).tolist(),
        "x_best": best_val["x"].tolist(),
        "sec": time.time() - t0,
    }
    json.dump(out, open(OUT, "w"))
    log(f"DONE Shared {a.tag}: b_med={scaling['b_median']:.4f}^n, b_B90={scaling['b_B90']:.4f}^n ({time.time()-t0:.1f}s)")

# ---------------- Pure GNN Simulator & Optimizer (Size-Transferable n=8,12 -> n=8,12,16,20) ----------------
def run_gnn_pipeline():
    t0 = time.time()
    H = a.H
    FM = 32
    rng = np.random.default_rng(2000 * L + a.seed)
    def W(i, o, s=1.0): return jnp.asarray(rng.normal(0, s / np.sqrt(max(i, 1)), (i, o)).astype(np.float32))
    p0 = AN.init_backbone(rng, a.arch, H, a.rounds, 7)
    p0["G1"] = W(2 * H, H); p0["g1"] = jnp.zeros(H)
    p0["G2_a"] = jnp.zeros((H, NB * 2))
    p0["G2_g"] = jnp.zeros((H, L))
    p0["Wa"] = jnp.zeros((H, NB * 2))
    p0["Ws"] = jnp.zeros((H, NB))
    p0["P0"] = W(H, FM, 1.5); p0["P1"] = W(H, FM, 1.5)
    p0["u0"] = jnp.asarray(rng.normal(0, 0.5, FM).astype(np.float32))
    p0["u1"] = jnp.asarray(rng.normal(0, 0.5, FM).astype(np.float32))
    p0["fq"] = jnp.zeros(FM)
    p0["Wo"] = jnp.zeros((FM, NB))
    p0["Mcav1"] = W(K * H, H, 1.0); p0["bcav1"] = jnp.zeros(H)
    p0["Wfock"] = jnp.zeros((H, NB * NU))
    if a.clause_w:
        p0["Wc"] = jnp.zeros((H, L))

    if a.init_shared and os.path.exists(a.init_shared):
        Js = json.load(open(a.init_shared))
        xs = np.array(Js["x_best"], np.float32)
        A0 = xs[:NB*2].reshape(NB, 2)
        TH0 = xs[NB*2:NB*18].reshape(NB, NU)
        g0 = xs[NB*18:NB*18+L]
    else:
        r = rng.uniform(0.1, 0.25, NB); ph = rng.uniform(0, 2*np.pi, NB)
        A0 = np.stack([r * np.cos(ph), r * np.sin(ph)], -1).astype(np.float32)
        TH0 = np.zeros((NB, NU), np.float32)
        g0 = np.linspace(0.8, 3.0, L).astype(np.float32)

    p0["A0"] = jnp.asarray(A0)
    p0["TH0"] = jnp.asarray(TH0)
    p0["g0"] = jnp.asarray(g0)
    th0, unr = ravel_pytree(p0)

    XB_CACHE = {}
    def get_xb(n):
        if n not in XB_CACHE:
            x = np.arange(1 << n)
            XB_CACHE[n] = jnp.asarray(((x[:, None] >> (n - 1 - np.arange(n))[None]) & 1).astype(np.int8))
        return XB_CACHE[n]

    def forward_gnn(p, G, n):
        C_cav = n // 4
        B = G["Ip"].shape[0]
        hv, hc = AN.run_backbone(p, G, a.arch, H)
        g_pool = jnp.concatenate([hv.mean(1), hc.mean(1)], -1)
        h_g = jnp.tanh(g_pool @ p["G1"] + p["g1"])
        dA_glob = (h_g @ p["G2_a"]).reshape(B, 1, NB, 2)
        dg = p["g0"][None, :] + h_g @ p["G2_g"]  # (B, L)

        hd = hv.reshape(B, C_cav, K, H)  # (B, C, 4, H)
        hcav = jnp.tanh(hd.reshape(B, C_cav, K * H) @ p["Mcav1"] + p["bcav1"])  # (B, C, H)
        dA_cav = ((hd.mean(2) + hcav) @ p["Wa"]).reshape(B, C_cav, NB, 2)
        A_b = p["A0"][None, None, :, :] + dA_glob + dA_cav  # (B, C, NB, 2)

        U0 = p["u0"] + hd @ p["P0"]
        U1 = p["u1"] + hd @ p["P1"]
        z = (jnp.einsum("bcjm,jf->bcfm", U1, BT_FOCK) + jnp.einsum("bcjm,jf->bcfm", U0, 1.0 - BT_FOCK)) / K
        ph_free = jnp.transpose(jnp.tanh(z + p["fq"]) @ p["Wo"], (0, 1, 3, 2))  # (B, C, NB, 16)
        ph_lin = jnp.einsum("bcjl,jf->bclf", hd @ p["Ws"], BT_FOCK)
        ph_fock = (hcav @ p["Wfock"]).reshape(B, C_cav, NB, NU)
        TH_b = p["TH0"][None, None, :, :] + ph_free + ph_lin + ph_fock  # (B, C, NB, 16)
        w_b = (hc @ p["Wc"]) if a.clause_w else jnp.zeros((B, 1, L), jnp.float32)
        return A_b, TH_b, dg, w_b

    def sim_gnn_n(th, G, Cn, Succ, n, zero_oracle=False):
        C_cav = n // 4
        B = Cn.shape[0]
        p = unr(th)
        A_b, TH_b, dg, w_b = forward_gnn(p, G, n)
        if zero_oracle:
            dg = jnp.zeros_like(dg)
            w_b = jnp.zeros_like(w_b)
        psi0 = make_psi0_log(C_cav)
        psi = jnp.broadcast_to(psi0, (B,) + (NU,) * C_cav)
        if a.clause_w and not zero_oracle:
            xb = get_xb(n)
            lit = xb[:, G["VP"]]  # (2^n, B, m, 4)
            viol = jnp.prod((lit == G["BB"][None]).astype(jnp.float32), -1)  # (2^n, B, m)
            nm = n / float(G["Ip"].shape[1])
        else:
            Cn_sh = Cn.reshape(psi.shape)

        for l in range(L):
            if not zero_oracle:
                if a.clause_w:
                    Ew = nm * jnp.einsum("xbm,bm->bx", viol, dg[:, l][:, None] + w_b[:, :, l])
                    psi = psi * jnp.exp(-1j * Ew.reshape(psi.shape))
                else:
                    psi = psi * jnp.exp(-1j * dg[:, l].reshape((B,) + (1,) * C_cav) * Cn_sh)
            for q in range(PAIRS):
                blk = l * PAIRS + q
                for c in range(C_cav):
                    D_val = disp_valid_batch(A_b[:, c, blk, 0], A_b[:, c, blk, 1])  # (B, 16, 16)
                    V_fock = D_val * jnp.exp(-1j * TH_b[:, c, blk])[:, None, :]
                    M_log = V_fock[:, FOCK_OF_LOG, :][:, :, FOCK_OF_LOG]
                    psi = apply_cav_batch(M_log, psi, c, C_cav)

        pr = jnp.abs(psi.reshape(B, -1)) ** 2
        p_succ = jnp.sum(pr * Succ, axis=1)
        p_leak = 1.0 - jnp.sum(pr, axis=1)
        return p_succ, p_leak

    def loss_gnn_n(th, G, Cn, Succ, n):
        p_s, p_l = sim_gnn_n(th, G, Cn, Succ, n, zero_oracle=False)
        nl = -jnp.log(p_s + 1e-12)
        out = jnp.mean(nl)
        if a.tailw > 0:
            k = max(1, nl.shape[0] // 10)
            out = out + a.tailw * jnp.mean(jax.lax.top_k(nl, k)[0])
        if a.leakpen > 0:
            out = out + a.leakpen * jnp.mean(p_l)
        return out

    vg8 = jax.jit(jax.value_and_grad(lambda th, G, c, s: loss_gnn_n(th, G, c, s, 8)))
    vg12 = jax.jit(jax.value_and_grad(lambda th, G, c, s: loss_gnn_n(th, G, c, s, 12)))
    ev12 = jax.jit(lambda th, G, c, s: sim_gnn_n(th, G, c, s, 12, False))

    E8_tr, _, v8_tr, s8_tr = get_instances(8, a.ntrain8, TRB, "train")
    E12_tr, _, v12_tr, s12_tr = get_instances(12, a.ntrain12, TRB, "train")
    E12_va, _, v12_va, s12_va = get_instances(12, 200, VAB, "val")

    G8_tr = build_graphs(v8_tr, s8_tr, 8)
    G12_tr = build_graphs(v12_tr, s12_tr, 12)
    G12_va = build_graphs(v12_va, s12_va, 12)

    m8 = int(round(9.93 * 8)); m12 = int(round(9.93 * 12))
    Cn8_tr = jnp.asarray(E8_tr.astype(np.float32) * (8.0 / m8))
    S8_tr = jnp.asarray((E8_tr == 0).astype(np.float32))
    Cn12_tr = jnp.asarray(E12_tr.astype(np.float32) * (12.0 / m12))
    S12_tr = jnp.asarray((E12_tr == 0).astype(np.float32))
    Cn12_va = jnp.asarray(E12_va.astype(np.float32) * (12.0 / m12))
    S12_va = jnp.asarray((E12_va == 0).astype(np.float32))

    # Verify fv[:,:,3:7] is strictly 0
    assert float(jnp.max(jnp.abs(G12_va["fv"][:, :, 3:7]))) == 0.0

    best_val = {"S": np.inf, "th": np.asarray(th0, np.float32), "step": 0, "med": 0.0, "B90": 0.0}
    def check_val12(th_vec, st):
        ps, lks = [], []
        for i in range(0, 200, 50):
            G_sub = {k_: v_[i:i+50] for k_, v_ in G12_va.items()}
            p_b, lk_b = ev12(jnp.asarray(th_vec, jnp.float32), G_sub, Cn12_va[i:i+50], S12_va[i:i+50])
            ps.append(np.asarray(p_b)); lks.append(np.asarray(lk_b))
        inv = 1.0 / np.maximum(np.concatenate(ps), 1e-12)
        med, b90 = float(np.median(inv)), float(np.quantile(inv, 0.9))
        S_v = 0.5 * (np.log(med) + np.log(b90))
        if S_v < best_val["S"]:
            best_val.update(S=S_v, th=np.array(th_vec, dtype=np.float32, copy=True), step=st, med=med, B90=b90)
        return med, b90, S_v

    m0, b0, s0 = check_val12(th0, 0)
    log(f"Start Part2 Pure GNN L={L} pairs={PAIRS} arch={a.arch} H={H} cw={a.clause_w} params={th0.size} | step 0 val12={m0:.2f}/{b0:.2f} S={s0:.4f}")

    @jax.jit
    def adam_step(x, m, v, g, t, lr):
        gnorm = jnp.sqrt(jnp.sum(g * g) + 1e-12)
        g = g * jnp.minimum(1.0, 3.0 / gnorm)
        m = 0.9 * m + 0.1 * g
        v = 0.999 * v + 0.001 * (g * g)
        mh = m / (1.0 - 0.9 ** t)
        vh = v / (1.0 - 0.999 ** t)
        return x - lr * mh / (jnp.sqrt(vh) + 1e-8), m, v

    x_cur = jnp.asarray(th0, jnp.float32)
    mom = jnp.zeros_like(x_cur); vel = jnp.zeros_like(x_cur)
    rng_b = np.random.default_rng(555 + a.seed)
    for st in range(1, a.steps + 1):
        i8 = rng_b.choice(a.ntrain8, min(a.bs8, a.ntrain8), replace=False)
        i12 = rng_b.choice(a.ntrain12, min(a.bs12, a.ntrain12), replace=False)
        G8_b = {k_: v_[i8] for k_, v_ in G8_tr.items()}
        G12_b = {k_: v_[i12] for k_, v_ in G12_tr.items()}
        v8, g8 = vg8(x_cur, G8_b, Cn8_tr[i8], S8_tr[i8])
        v12, g12 = vg12(x_cur, G12_b, Cn12_tr[i12], S12_tr[i12])
        warm = min(1.0, st / 150.0)
        lr_t = a.lr * warm * (0.1 + 0.9 * 0.5 * (1.0 + np.cos(np.pi * st / a.steps)))
        x_cur, mom, vel = adam_step(x_cur, mom, vel, 0.5 * (g8 + g12), st, lr_t)
        if st % 200 == 0 or st == a.steps:
            med_v, b90_v, S_v = check_val12(x_cur, st)
            log(f"  step {st}/{a.steps} loss={0.5*(float(v8)+float(v12)):.4f} | val12 {med_v:.2f}/{b90_v:.2f} S={S_v:.4f} (best {best_val['S']:.4f} @ {best_val['step']})")

    th_best = jnp.asarray(best_val["th"], jnp.float32)
    np.savez_compressed(OUT.replace(".json", "_theta.npz"), theta=np.asarray(th_best, np.float32))

    test_res = {}
    for n_te in [8, 12, 16, 20]:
        t_ev = time.time()
        E_te, _, v_te, s_te = get_instances(n_te, 100, TEB, "test")
        G_te = build_graphs(v_te, s_te, n_te)
        m_te = int(round(9.93 * n_te))
        ch = 50 if n_te <= 12 else (10 if n_te == 16 else 2)
        ev_fn = jax.jit(lambda th, G_b, c_b, s_b, n_val=n_te: sim_gnn_n(th, G_b, c_b, s_b, n_val, False))
        ev_z  = jax.jit(lambda th, G_b, c_b, s_b, n_val=n_te: sim_gnn_n(th, G_b, c_b, s_b, n_val, True))
        ps, lks, pzs = [], [], []
        for i in range(0, 100, ch):
            G_b = {k_: v_[i:i+ch] for k_, v_ in G_te.items()}
            Cn_b = jnp.asarray(E_te[i:i+ch].astype(np.float32) * (n_te / float(m_te)))
            S_b = jnp.asarray((E_te[i:i+ch] == 0).astype(np.float32))
            p_b, lk_b = ev_fn(th_best, G_b, Cn_b, S_b)
            pz_b, _   = ev_z(th_best, G_b, Cn_b, S_b)
            ps.append(np.asarray(p_b)); lks.append(np.asarray(lk_b)); pzs.append(np.asarray(pz_b))
        p_all = np.concatenate(ps); lk_all = np.concatenate(lks); pz_all = np.concatenate(pzs)
        test_res[str(n_te)] = eval_metrics(p_all, lk_all, pz_all, (E_te[:, 0] == 0).astype(float), n_te)
        test_res[str(n_te)]["p"] = p_all.tolist()
        log(f"  TEST n={n_te}: med/B90 = {test_res[str(n_te)]['median_inv_p']:.2f}/{test_res[str(n_te)]['B90_inv_p']:.2f} (zero_oracle={test_res[str(n_te)]['zero_oracle_med']:.1f}/{test_res[str(n_te)]['zero_oracle_B90']:.1f}, leak={test_res[str(n_te)]['mean_leak']*100:.2f}%, {time.time()-t_ev:.1f}s)")

    scaling = fit_scaling(test_res)
    out = {
        "args": vars(a), "best_step": best_val["step"],
        "val12": {"median_inv_p": best_val["med"], "B90_inv_p": best_val["B90"], "S": best_val["S"]},
        "test": test_res, "scaling": scaling, "sec": time.time() - t0,
    }
    json.dump(out, open(OUT, "w"))
    log(f"DONE Pure GNN {a.tag}: b_med={scaling['b_median']:.4f}^n, b_B90={scaling['b_B90']:.4f}^n ({time.time()-t0:.1f}s)")

if __name__ == "__main__":
    if a.mode == "shared":
        run_shared_pipeline()
    else:
        run_gnn_pipeline()
