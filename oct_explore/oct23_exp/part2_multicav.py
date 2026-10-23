"""Part 2: Multi-Cavity 4-SAT Scaling (C = 2, 3, 4, 5 Cavities, n = 8, 12, 16, 20)
with Physical SNAP+D Gates (0..15 Computational SNAP Only), Global Cost Phase, and Pure GNN (L <= 15).

Strict Audit & Physical Rules:
  1. Every cavity holds K = 4 bits in D = 16 computational Fock levels (0..15) via Gray code.
  2. Every SNAP gate has free phases ONLY on computational Fock levels 0..15 (strictly 0 phase on f >= 16).
  3. Within each layer l = 0..L-1, the k = PAIRS physical SNAP+D blocks evolve in NF = 40 Fock levels:
       M_l = S_out(sout_l) P [ D_{40}(alpha_{l,k-1}) S_{0..15}(theta_{l,k-1}) ... D_{40}(alpha_{l,0}) S_{0..15}(theta_{l,0}) ] P
     where every internal SNAP S_{0..15} has 0 phase on f >= 16, and S_out(sout_l) is merged into the next layer's
     diagonal cost/SNAP on hardware. Any amplitude at f >= 16 at the end of the layer is permanently lost as leakage.
  4. State preparation uses the physical k_prep = 2 (98.85% flat, 0.29% leak) or k_prep = 3 (99.998% flat)
     SNAP+D prep from preps.json.
  5. Zero classical pre-solving: canonical variable ordering (relabel = none, flips = 0), and Pure GNN uses
     ONLY raw factor-graph degree features fv[:, :, 0:3] = [1, pos_deg, neg_deg] (fv[:, :, 3:7] = 0).
  6. 3-Stage Shared BFGS Optimization:
     - Stage 1: Optimize 2L-parameter (beta_l, gamma_l) schedule on n=8+12 from Q_s0 interpolation (for both Gray and Tri targets).
     - Stage 2: Calibrate k = PAIRS physical SNAP+D blocks per layer in NF=40 (with 0..15 SNAP only) to match U_gray(beta_l) and U_tri(beta_l).
     - Stage 3: End-to-end many-body L-BFGS-B joint refinement of all (alpha, theta, sout, gamma) on n=8+12 training instances.
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
ap.add_argument("--mixer", default="sd", choices=["sd", "gray"])
ap.add_argument("--L", type=int, default=6)
ap.add_argument("--pairs", type=int, default=2, choices=[1, 2, 3], help="Physical SNAP+D blocks per cavity per cost layer")
ap.add_argument("--kprep", type=int, default=2, choices=[2, 3], help="Physical SNAP+D blocks for initial flat state prep")
ap.add_argument("--ntrain8", type=int, default=2000)
ap.add_argument("--ntrain12", type=int, default=500)
ap.add_argument("--maxiter", type=int, default=150)
ap.add_argument("--grow_from", default="", help="JSON of smaller-L shared run")
ap.add_argument("--init_shared", default="", help="Shared JSON to initialize GNN base parameters")
ap.add_argument("--arch", default="mp", choices=["mp", "mp64", "mp128", "attn", "ds", "mlp"])
ap.add_argument("--H", type=int, default=64)
ap.add_argument("--rounds", type=int, default=3)
ap.add_argument("--clause_w", action="store_true", help="Enable GNN per-clause weights in global cost phase")
ap.add_argument("--steps", type=int, default=1000)
ap.add_argument("--bs8", type=int, default=125)
ap.add_argument("--bs12", type=int, default=50)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--tailw", type=float, default=0.25)
ap.add_argument("--leakpen", type=float, default=0.2)
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
Q4_PATH = os.path.join(ROOT, "results", "clean16", "Q_s0.json")
Q4_ANGLES = np.array(json.load(open(Q4_PATH))["x"], dtype=np.float32).reshape(4, 2)

WS_TRIES = {8: (2.014, 3.739), 12: (2.92, 6.28), 16: (4.55, 9.09), 20: (5.63, 14.2)}
WS_LOOK  = {8: (13.989, 29.396), 12: (32.5, 76.1), 16: (71.0, 147.0), 20: (110.0, 290.0)}

log = lambda *s: print(time.strftime("%H:%M:%S"), f"[{a.tag}]", *s, flush=True)
gray = lambda i: i ^ (i >> 1)
def ungray(g):
    s = g >> 1
    while s: g ^= s; s >>= 1
    return g

FOCK_OF_LOG = np.array([ungray(sum(((g >> (3 - j)) & 1) << j for j in range(K))) for g in range(NU)], dtype=np.int64)

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

# Ideal mixer generators in Fock basis (0..15) for Stage 1 & Stage 2 calibration
H_GRAY_FOCK = np.zeros((NU, NU), dtype=np.float64)
for f in range(NU):
    for j in range(K):
        H_GRAY_FOCK[ungray(gray(f) ^ (1 << j)), f] = 1.0
W_GRAY, V_GRAY = np.linalg.eigh(H_GRAY_FOCK)
W_GRAY_J = jnp.asarray(W_GRAY.astype(np.float32))
V_GRAY_J = jnp.asarray(V_GRAY.astype(np.complex64))

H_TRI_FOCK = np.diag(np.ones(NU - 1), 1) + np.diag(np.ones(NU - 1), -1)
W_TRI, V_TRI = np.linalg.eigh(H_TRI_FOCK)
W_TRI_J = jnp.asarray(W_TRI.astype(np.float32))
V_TRI_J = jnp.asarray(V_TRI.astype(np.complex64))

def disp40_single(re, im):
    if a.amax > 0:
        r_raw = jnp.sqrt(re ** 2 + im ** 2 + 1e-20)
        sc = a.amax * jnp.tanh(r_raw / a.amax) / r_raw
        re, im = re * sc, im * sc
    r = jnp.sqrt(re ** 2 + im ** 2 + 1e-20)
    ph = jnp.arctan2(im, re)
    W = jnp.exp(1j * ph * NV)[:, None] * VD
    return (W * jnp.exp(-1j * r * LAM)[None]) @ W.conj().T

def disp40_batch(re, im):
    if a.amax > 0:
        r_raw = jnp.sqrt(re ** 2 + im ** 2 + 1e-20)
        sc = a.amax * jnp.tanh(r_raw / a.amax) / r_raw
        re, im = re * sc, im * sc
    r = jnp.sqrt(re ** 2 + im ** 2 + 1e-20)
    ph = jnp.arctan2(im, re)
    W = jnp.exp(1j * ph[:, None] * NV[None])[:, :, None] * VD[None]
    return jnp.einsum("bik,bk,bjk->bij", W, jnp.exp(-1j * r[:, None] * LAM[None]), W.conj())

def layer_fock_shared(A_l, TH_l, sout_l):
    """A_l: (PAIRS, 2), TH_l: (PAIRS, 16), sout_l: (16,) -> (16, 16) Fock-basis block."""
    M = jnp.eye(NF_DISP, NU, dtype=jnp.complex64)
    for q in range(PAIRS):
        th40 = jnp.concatenate([TH_l[q], jnp.zeros(NF_DISP - NU, jnp.float32)])
        M = disp40_single(A_l[q, 0], A_l[q, 1]) @ (jnp.exp(-1j * th40)[:, None] * M)
    return jnp.exp(-1j * sout_l)[:, None] * M[:NU, :NU]

def layer_log_shared(A_l, TH_l, sout_l):
    Vp = layer_fock_shared(A_l, TH_l, sout_l)
    return Vp[jnp.ix_(FOCK_OF_LOG, FOCK_OF_LOG)]

def layer_log_batch(A_b_l, TH_b_l, sout_b_l):
    """A_b_l: (B, PAIRS, 2), TH_b_l: (B, PAIRS, 16), sout_b_l: (B, 16) -> (B, 16, 16) logical block."""
    B = A_b_l.shape[0]
    M = jnp.broadcast_to(jnp.eye(NF_DISP, NU, dtype=jnp.complex64), (B, NF_DISP, NU))
    for q in range(PAIRS):
        th40 = jnp.concatenate([TH_b_l[:, q], jnp.zeros((B, NF_DISP - NU), jnp.float32)], axis=1)
        D40 = disp40_batch(A_b_l[:, q, 0], A_b_l[:, q, 1])
        M = jnp.einsum("bij,bjk->bik", D40, jnp.exp(-1j * th40)[:, :, None] * M)
    Vp = jnp.exp(-1j * sout_b_l)[:, :, None] * M[:, :NU, :NU]
    return Vp[:, FOCK_OF_LOG, :][:, :, FOCK_OF_LOG]

# Initial single-cavity state on Fock 0..15 from physical k_prep-block state prep (norm^2 = 1 - leak_prep)
_prep_info = PREPS[f"sd_K4_k{a.kprep}"]
V0_FOCK = jnp.asarray((np.array(_prep_info["psi_re"][:NU]) + 1j * np.array(_prep_info["psi_im"][:NU])).astype(np.complex64))
V0_LOG = V0_FOCK[FOCK_OF_LOG]

def make_psi0_log(C_cav):
    v = V0_LOG
    for _ in range(C_cav - 1):
        v = jnp.multiply.outer(v, V0_LOG)
    return v

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
    # Check if a larger cached file for the same (n, split_label) already exists
    for cand_cnt in sorted(set([count, 500, 1000, 5000])):
        if cand_cnt < count:
            continue
        cache_file = os.path.join(OUT_DIR, f"inst_n{n}_{split_label}_{cand_cnt}.npz")
        if os.path.exists(cache_file):
            Z = np.load(cache_file)
            _DATA_CACHE[key] = (Z["E"][:count], Z["nsol"][:count], Z["cls_v"][:count], Z["cls_s"][:count])
            return _DATA_CACHE[key]
    p1_n8 = os.path.join(ROOT, "results", "oct23_exp", "data_n8.npz")
    if n == 8 and os.path.exists(p1_n8) and split_label in ("train", "val", "test"):
        Z8 = np.load(p1_n8)
        if len(Z8[f"{split_label}_E"]) >= count:
            _DATA_CACHE[key] = (Z8[f"{split_label}_E"][:count], Z8[f"{split_label}_nsol"][:count],
                                Z8[f"{split_label}_cls_v"][:count], Z8[f"{split_label}_cls_s"][:count])
            return _DATA_CACHE[key]
    t0 = time.time()
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
    cache_file = os.path.join(OUT_DIR, f"inst_n{n}_{split_label}_{count}.npz")
    np.savez_compressed(cache_file, E=Es, nsol=nsols, cls_v=cls_v, cls_s=cls_s)
    log(f"Cached {cache_file} ({count} insts at n={n}) in {time.time()-t0:.1f}s")
    _DATA_CACHE[key] = (Es, nsols, cls_v, cls_s)
    return _DATA_CACHE[key]

def build_graphs(cls_v, cls_s, n):
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
    deg = np.maximum(Ip.sum(1) + In.sum(1), 1.0)
    bb = (cls_s < 0).astype(np.int8)
    return {"Ip": jnp.asarray(Ip), "In": jnp.asarray(In), "fv": jnp.asarray(fv), "deg": jnp.asarray(deg),
            "VP": jnp.asarray(cls_v.astype(np.int32)), "BB": jnp.asarray(bb)}

# ---------------- Shared-Parameter Simulator & 3-Stage Optimizer ----------------
def unpack_shared(x):
    """x has size L*PAIRS*2 + L*PAIRS*16 + L*16 + L:
      A: (L, PAIRS, 2), TH: (L, PAIRS, 16), SOUT: (L, 16), g: (L,)
    """
    nA = L * PAIRS * 2
    nT = L * PAIRS * NU
    nS = L * NU
    A = x[:nA].reshape(L, PAIRS, 2)
    TH = x[nA:nA + nT].reshape(L, PAIRS, NU)
    SOUT = x[nA + nT:nA + nT + nS].reshape(L, NU)
    g = x[nA + nT + nS:nA + nT + nS + L]
    return A, TH, SOUT, g

def sim_shared_n(x, Cn, Succ, n):
    C_cav = n // 4
    B = Cn.shape[0]
    A, TH, SOUT, g = unpack_shared(x)
    psi0 = make_psi0_log(C_cav)
    psi = jnp.broadcast_to(psi0, (B,) + (NU,) * C_cav)
    Cn_sh = Cn.reshape(psi.shape)
    def sh_body(psi_c, xs_l):
        A_l, TH_l, SOUT_l, g_l = xs_l
        psi_c = psi_c * jnp.exp(-1j * g_l * Cn_sh)
        M_log = layer_log_shared(A_l, TH_l, SOUT_l)
        for c in range(C_cav):
            psi_c = apply_cav_shared(M_log, psi_c, c, C_cav)
        return psi_c, None
    psi, _ = jax.lax.scan(sh_body, psi, (A, TH, SOUT, g))
    pr = jnp.abs(psi.reshape(B, -1)) ** 2
    p_succ = jnp.sum(pr * Succ, axis=1)
    p_leak = 1.0 - jnp.sum(pr, axis=1)
    return p_succ, p_leak

def calibrate_single_layer(beta_val, kind="gray", starts=4):
    """Fit 1 layer of PAIRS blocks (A_l, TH_l, sout_l) in NF=40 (with 0..15 SNAP only) to U_{kind}(beta_val)."""
    Vf, w_j = (V_GRAY_J, W_GRAY_J) if kind == "gray" else (V_TRI_J, W_TRI_J)
    U_tgt = (Vf * jnp.exp(-0.5j * beta_val * w_j)[None, :]) @ Vf.conj().T
    def loss_cal(z):
        al = z[:PAIRS*2].reshape(PAIRS, 2)
        th = z[PAIRS*2:PAIRS*18].reshape(PAIRS, NU)
        sout = z[PAIRS*18:PAIRS*18+NU]
        Vp = layer_fock_shared(al, th, sout)
        F = jnp.abs(jnp.trace(U_tgt.conj().T @ Vp)) ** 2 / (NU ** 2)
        lk = 1.0 - jnp.sum(jnp.abs(Vp) ** 2) / NU
        return 1.0 - F + lk
    vg_c = jax.jit(jax.value_and_grad(loss_cal))
    rng = np.random.default_rng(int(abs(beta_val) * 1000) + PAIRS)
    best = None
    for s in range(starts):
        z0 = rng.normal(0, 0.15 + 0.1 * s, PAIRS * 18 + NU).astype(np.float32)
        r = minimize(lambda z: tuple(np.asarray(v, float) for v in vg_c(jnp.asarray(z, jnp.float32))),
                     np.asarray(z0, np.float64), jac=True, method="L-BFGS-B", options={"maxiter": 250})
        if best is None or r.fun < best.fun:
            best = r
    z = best.x.astype(np.float32)
    return z[:PAIRS*2].reshape(PAIRS, 2), z[PAIRS*2:PAIRS*18].reshape(PAIRS, NU), z[PAIRS*18:PAIRS*18+NU], float(best.fun)

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

    # Stage 1: Fit 2L-parameter (beta_l, gamma_l) schedule for both "gray" and "tri" target mixers
    cands_x = []
    if a.grow_from and os.path.exists(a.grow_from) and a.mixer != "gray":
        Jg = json.load(open(a.grow_from))
        cands_x.append((Jg.get("winner_init", "warm"), np.array(Jg["x_best"], dtype=np.float32)))
        log(f"Warm-starting Stage 3 directly from {a.grow_from}")
    else:
        log(f"Stage 1: Optimizing 2L={2*L} (beta, gamma) schedules for Gray and Tri mixers...")
    for kind, Vf, w_j in ([] if cands_x else ([("gray", V_GRAY_J, W_GRAY_J)] if a.mixer == "gray" else [("gray", V_GRAY_J, W_GRAY_J), ("tri", V_TRI_J, W_TRI_J)])):
        def sim_id(bg, Cn, S, n):
            C_cav = n // 4; B = Cn.shape[0]
            psi = jnp.broadcast_to(make_psi0_log(C_cav), (B,) + (NU,) * C_cav)
            Cn_sh = Cn.reshape(psi.shape)
            for l in range(L):
                b, g = bg[2*l], bg[2*l+1]
                psi = psi * jnp.exp(-1j * g * Cn_sh)
                U_f = (Vf * jnp.exp(-0.5j * b * w_j)[None, :]) @ Vf.conj().T
                U_l = U_f[jnp.ix_(FOCK_OF_LOG, FOCK_OF_LOG)]
                for c in range(C_cav):
                    psi = apply_cav_shared(U_l, psi, c, C_cav)
            pr = jnp.abs(psi.reshape(B, -1)) ** 2
            return jnp.sum(pr * S, axis=1)
        vg_id = jax.jit(jax.value_and_grad(
            lambda bg: 0.5 * jnp.mean(-jnp.log(sim_id(bg, Cn8_tr, S8_tr, 8) + 1e-12))
                     + 0.5 * jnp.mean(-jnp.log(sim_id(bg, Cn12_tr, S12_tr, 12) + 1e-12))
        ))
        t_old = np.linspace(0, 1, 4); t_new = np.linspace(0, 1, L)
        bg_inits = [
            np.stack([np.interp(t_new, t_old, Q4_ANGLES[:, 0]), np.interp(t_new, t_old, Q4_ANGLES[:, 1])], 1).reshape(-1),
            np.stack([-np.linspace(0.75, 0.20, L), np.linspace(2.5, 8.5, L)], 1).reshape(-1),
        ]
        gray_out = os.path.join(OUT_DIR, f"p2_gray_L{L}.json" if a.ntrain8 == 2000 else f"p2_gray_L{L}_ntr{a.ntrain8}.json")
        if kind == "gray" and os.path.exists(gray_out) and not a.grow_from:
            g_cached = json.load(open(gray_out))
            class _Res: pass
            best_id = _Res(); best_id.x = np.array(g_cached["bg_opt"], dtype=np.float64); best_id.fun = 0.0
        else:
            best_id = min((minimize(lambda x: tuple(np.asarray(v, float) for v in vg_id(jnp.asarray(x, jnp.float32))),
                           np.asarray(bgi, np.float64), jac=True, method="L-BFGS-B", options={"maxiter": 100})
                           for bgi in bg_inits), key=lambda r: r.fun)
        bg_opt = best_id.x.reshape(L, 2)
        p_id_va = np.asarray(sim_id(jnp.asarray(best_id.x, jnp.float32), Cn12_va, S12_va, 12))
        inv_id_va = 1.0 / np.maximum(p_id_va, 1e-12)
        med_id_va, b90_id_va = float(np.median(inv_id_va)), float(np.quantile(inv_id_va, 0.9))
        log(f"  Stage 1 {kind}: loss={best_id.fun:.4f} | ideal val12 med/B90 = {med_id_va:.2f}/{b90_id_va:.2f}")
        gray_out = os.path.join(OUT_DIR, f"p2_gray_L{L}.json")
        if kind == "gray" and (a.mixer == "gray" or not os.path.exists(gray_out)):
            bg_j = jnp.asarray(best_id.x, jnp.float32)
            bg_z = jnp.asarray(np.stack([bg_opt[:, 0], np.zeros(L)], 1).reshape(-1), jnp.float32)
            te_g = {}
            for n_te in [8, 12, 16, 20]:
                E_te, _, _, _ = get_instances(n_te, 100, TEB, "test")
                m_te = int(round(9.93 * n_te))
                ch = 50 if n_te <= 12 else (25 if n_te == 16 else 10)
                ev_g = jax.jit(lambda bg_v, c_v, s_v, nv=n_te: sim_id(bg_v, c_v, s_v, nv))
                ps_g, pz_g = [], []
                for i in range(0, 100, ch):
                    Cn_b = jnp.asarray(E_te[i:i+ch].astype(np.float32) * (n_te / float(m_te)))
                    S_b = jnp.asarray((E_te[i:i+ch] == 0).astype(np.float32))
                    ps_g.append(np.asarray(ev_g(bg_j, Cn_b, S_b)))
                    pz_g.append(np.asarray(ev_g(bg_z, Cn_b, S_b)))
                p_all_g = np.concatenate(ps_g); pz_all_g = np.concatenate(pz_g)
                lk_g = np.full_like(p_all_g, 1.0 - float(jnp.sum(jnp.abs(V0_LOG)**2)) ** (n_te // 4))
                te_g[str(n_te)] = eval_metrics(p_all_g, lk_g, pz_all_g, (E_te[:, 0] == 0).astype(float), n_te)
                te_g[str(n_te)]["p"] = p_all_g.tolist()
            sc_g = fit_scaling(te_g)
            g_tag = f"p2_gray_L{L}" if a.ntrain8 == 2000 else f"p2_gray_L{L}_ntr{a.ntrain8}"
            g_doc = {
                "args": {**vars(a), "tag": g_tag, "mixer": "gray"},
                "val12": {"median_inv_p": med_id_va, "B90_inv_p": b90_id_va, "S": 0.5*(np.log(med_id_va)+np.log(b90_id_va))},
                "test": te_g, "scaling": sc_g, "bg_opt": best_id.x.tolist(), "sec": time.time() - t0
            }
            json.dump(g_doc, open(gray_out, "w"))
            log(f"  Saved {gray_out}: n=8..20 med = {[round(te_g[str(n)]['median_inv_p'], 2) for n in [8,12,16,20]]}, b_med={sc_g['b_median']:.4f}^n, b_B90={sc_g['b_B90']:.4f}^n")
            if a.mixer == "gray":
                json.dump(g_doc, open(OUT, "w"))
                return

        # Stage 2: Calibrate k = PAIRS physical SNAP+D blocks per layer to match U_{kind}(beta_l)
        A_cal = np.zeros((L, PAIRS, 2), np.float32)
        TH_cal = np.zeros((L, PAIRS, NU), np.float32)
        SOUT_cal = np.zeros((L, NU), np.float32)
        cal_errs = []
        for l in range(L):
            al_l, th_l, sout_l, err_l = calibrate_single_layer(float(bg_opt[l, 0]), kind=kind, starts=4)
            A_cal[l] = al_l; TH_cal[l] = th_l; SOUT_cal[l] = sout_l; cal_errs.append(err_l)
        x_cal = np.concatenate([A_cal.reshape(-1), TH_cal.reshape(-1), SOUT_cal.reshape(-1), bg_opt[:, 1].astype(np.float32)])
        cands_x.append((kind, x_cal))
        log(f"  Stage 2 {kind} (k={PAIRS}): mean layer (1-F+leak) = {np.mean(cal_errs):.5f}")

    # Stage 3: End-to-end many-body L-BFGS-B joint refinement on n=8+12
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

    ch8 = 1000; ch12 = 500
    p8_parts = [(Cn8_tr[i:i+ch8], S8_tr[i:i+ch8], (min(i+ch8, a.ntrain8) - i) / float(a.ntrain8)) for i in range(0, a.ntrain8, ch8)]
    p12_parts = [(Cn12_tr[i:i+ch12], S12_tr[i:i+ch12], (min(i+ch12, a.ntrain12) - i) / float(a.ntrain12)) for i in range(0, a.ntrain12, ch12)]
    def obj_full(x):
        xj = jnp.asarray(x, jnp.float32)
        tot_v = 0.0; tot_g = np.zeros_like(x, dtype=np.float64)
        for c, s, w in p8_parts:
            v, g = vg8(xj, c, s)
            tot_v += 0.5 * float(v) * w; tot_g += 0.5 * np.asarray(g, np.float64) * w
        for c, s, w in p12_parts:
            v, g = vg12(xj, c, s)
            tot_v += 0.5 * float(v) * w; tot_g += 0.5 * np.asarray(g, np.float64) * w
        return tot_v, tot_g

    best_val = None
    for label, x0 in cands_x:
        t1 = time.time()
        p0_va, lk0_va = ev12(jnp.asarray(x0, jnp.float32), Cn12_va, S12_va)
        inv0 = 1.0 / np.maximum(np.asarray(p0_va, float), 1e-12)
        log(f"  Stage 3 start ({label}): pre-refine val12 = {np.median(inv0):.2f}/{np.quantile(inv0, 0.9):.2f} (leak={np.mean(lk0_va)*100:.2f}%)")
        r = minimize(obj_full, np.asarray(x0, np.float64), jac=True, method="L-BFGS-B",
                     options={"maxiter": a.maxiter, "maxfun": int(1.5 * a.maxiter)})
        xj = jnp.asarray(r.x, jnp.float32)
        p_v, lk_v = ev12(xj, Cn12_va, S12_va)
        inv_v = 1.0 / np.maximum(np.asarray(p_v, float), 1e-12)
        med_v, b90_v = float(np.median(inv_v)), float(np.quantile(inv_v, 0.9))
        S_v = 0.5 * (np.log(med_v) + np.log(b90_v))
        log(f"  Stage 3 done ({label}): loss={r.fun:.4f} ({r.nit} it, {time.time()-t1:.1f}s) | val12 = {med_v:.2f}/{b90_v:.2f} (leak={np.mean(lk_v)*100:.2f}%) S={S_v:.4f}")
        if best_val is None or S_v < best_val["S"]:
            best_val = {"S": S_v, "med": med_v, "B90": b90_v, "x": r.x.astype(np.float32), "loss": float(r.fun), "init": label}

    x_best = jnp.asarray(best_val["x"], jnp.float32)
    x_zero = jnp.concatenate([x_best[:-L], jnp.zeros(L, jnp.float32)])

    test_res = {}
    for n_te in [8, 12, 16, 20]:
        t_ev = time.time()
        E_te, _, _, _ = get_instances(n_te, 100, TEB, "test")
        m_te = int(round(9.93 * n_te))
        ch = 50 if n_te <= 12 else (25 if n_te == 16 else 10)
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
    A_b, TH_b, SOUT_b, g_b = unpack_shared(x_best)
    out = {
        "args": vars(a), "winner_init": best_val["init"],
        "val12": {"median_inv_p": best_val["med"], "B90_inv_p": best_val["B90"], "S": best_val["S"]},
        "test": test_res, "scaling": scaling,
        "alpha_abs": np.hypot(np.asarray(A_b)[..., 0], np.asarray(A_b)[..., 1]).tolist(),
        "g": np.asarray(g_b).tolist(),
        "x_best": best_val["x"].tolist(),
        "sec": time.time() - t0,
    }
    json.dump(out, open(OUT, "w"))
    log(f"DONE Shared {a.tag} ({best_val['init']}): b_med={scaling['b_median']:.4f}^n, b_B90={scaling['b_B90']:.4f}^n ({time.time()-t0:.1f}s)")

# ---------------- Pure GNN Simulator & Optimizer (Size-Transferable n=8,12 -> n=8,12,16,20) ----------------
def run_gnn_pipeline():
    t0 = time.time()
    H = a.H
    FM = 32
    rng = np.random.default_rng(2000 * L + a.seed)
    def W(i, o, s=1.0): return jnp.asarray(rng.normal(0, s / np.sqrt(max(i, 1)), (i, o)).astype(np.float32))
    p0 = AN.init_backbone(rng, a.arch, H, a.rounds, 7)
    p0["G1"] = W(2 * H, H); p0["g1"] = jnp.zeros(H)
    p0["G2_a"] = jnp.zeros((H, L * PAIRS * 2))
    p0["G2_g"] = jnp.zeros((H, L))
    p0["Wa"] = jnp.zeros((H, L * PAIRS * 2))
    p0["Ws"] = jnp.zeros((H, L * PAIRS))
    p0["Wsout"] = jnp.zeros((H, L))
    p0["P0"] = W(H, FM, 1.5); p0["P1"] = W(H, FM, 1.5)
    p0["u0"] = jnp.asarray(rng.normal(0, 0.5, FM).astype(np.float32))
    p0["u1"] = jnp.asarray(rng.normal(0, 0.5, FM).astype(np.float32))
    p0["fq"] = jnp.zeros(FM)
    p0["Wo"] = jnp.zeros((FM, L * PAIRS))
    p0["Mcav1"] = W(K * H, H, 1.0); p0["bcav1"] = jnp.zeros(H)
    p0["Wfock"] = jnp.zeros((H, L * PAIRS * NU))
    if a.clause_w:
        p0["Wc"] = jnp.zeros((H, L))

    assert a.init_shared and os.path.exists(a.init_shared), f"init_shared {a.init_shared} required for GNN"
    Js = json.load(open(a.init_shared))
    if a.mixer == "gray":
        bg_init = np.array(Js["bg_opt"], np.float32).reshape(L, 2)
        A0 = jnp.zeros((L, PAIRS, 2), jnp.float32).at[:, 0, 0].set(bg_init[:, 0])
        TH0 = jnp.zeros((L, PAIRS, NU), jnp.float32)
        SOUT0 = jnp.zeros((L, NU), jnp.float32)
        g0 = jnp.asarray(bg_init[:, 1], jnp.float32)
    else:
        xs = jnp.asarray(Js["x_best"], jnp.float32)
        A0, TH0, SOUT0, g0 = unpack_shared(xs)
    p0["A0"] = A0; p0["TH0"] = TH0; p0["SOUT0"] = SOUT0; p0["g0"] = g0
    th0, unr = ravel_pytree(p0)

    # Precompute outside any JIT so get_xb never mutates a cache under a tracer.
    XB_CACHE = {}
    for _n_xb in (8, 12, 16, 20):
        _x = np.arange(1 << _n_xb)
        XB_CACHE[_n_xb] = jnp.asarray(((_x[:, None] >> (_n_xb - 1 - np.arange(_n_xb))[None]) & 1).astype(np.int8))
    def get_xb(n):
        return XB_CACHE[n]

    def forward_gnn(p, G, n):
        C_cav = n // 4
        B = G["Ip"].shape[0]
        hv, hc = AN.run_backbone(p, G, a.arch, H)
        g_pool = jnp.concatenate([hv.mean(1), hc.mean(1)], -1)
        h_g = jnp.tanh(g_pool @ p["G1"] + p["g1"])
        dA_glob = (h_g @ p["G2_a"]).reshape(B, 1, L, PAIRS, 2)
        dg = p["g0"][None, :] + h_g @ p["G2_g"]  # (B, L)

        hd = hv.reshape(B, C_cav, K, H)  # (B, C, 4, H)
        hcav = jnp.tanh(hd.reshape(B, C_cav, K * H) @ p["Mcav1"] + p["bcav1"])  # (B, C, H)
        dA_cav = ((hd.mean(2) + hcav) @ p["Wa"]).reshape(B, C_cav, L, PAIRS, 2)
        A_b = p["A0"][None, None] + dA_glob + dA_cav  # (B, C, L, PAIRS, 2)

        U0 = p["u0"] + hd @ p["P0"]
        U1 = p["u1"] + hd @ p["P1"]
        z = (jnp.einsum("bcjm,jf->bcfm", U1, BT_FOCK) + jnp.einsum("bcjm,jf->bcfm", U0, 1.0 - BT_FOCK)) / K
        ph_free = jnp.transpose(jnp.tanh(z + p["fq"]) @ p["Wo"], (0, 1, 3, 2)).reshape(B, C_cav, L, PAIRS, NU)
        ph_lin = jnp.einsum("bcjn,jf->bcnf", hd @ p["Ws"], BT_FOCK).reshape(B, C_cav, L, PAIRS, NU)
        ph_fock = (hcav @ p["Wfock"]).reshape(B, C_cav, L, PAIRS, NU)
        TH_b = p["TH0"][None, None] + ph_free + ph_lin + ph_fock
        dsout = jnp.einsum("bcjl,jf->bclf", hd @ p["Wsout"], BT_FOCK)  # (B, C, L, 16)
        SOUT_b = p["SOUT0"][None, None] + dsout
        w_b = (hc @ p["Wc"]) if a.clause_w else jnp.zeros((B, 1, L), jnp.float32)
        return A_b, TH_b, SOUT_b, dg, w_b

    def sim_gnn_n(th, G, Cn, Succ, n, zero_oracle=False):
        C_cav = n // 4
        B = Cn.shape[0]
        p = unr(th)
        A_b, TH_b, SOUT_b, dg, w_b = forward_gnn(p, G, n)
        if zero_oracle:
            dg = jnp.zeros_like(dg)
            w_b = jnp.zeros_like(w_b)
        psi0 = make_psi0_log(C_cav)
        psi = jnp.broadcast_to(psi0, (B,) + (NU,) * C_cav)
        # Clause-weighted cost must not materialize viol with shape (2^n, B, m)
        # (OOM at n=20). Build Ew as (B, 2^n) via a batch loop over clauses.
        if a.clause_w and not zero_oracle:
            xb = get_xb(n)  # (2^n, n) int8, precomputed
            VP = G["VP"]    # (B, m, K)
            BB = G["BB"]    # (B, m, K)
            nm = n / float(G["Ip"].shape[1])
            def cost_Ew(w_eff):
                # w_eff: (B, m) -> Ew: (B, 2^n)
                # viol_b[x,c] = prod_k (xb[x, VP[b,c,k]] == BB[b,c,k])
                def one_b(vp_b, bb_b, w_b):
                    # vp_b, bb_b: (m, K); w_b: (m,)
                    lit = xb[:, vp_b]                      # (2^n, m, K)
                    viol = jnp.prod((lit == bb_b[None]).astype(jnp.float32), axis=-1)  # (2^n, m)
                    return viol @ w_b                       # (2^n,)
                return nm * jax.vmap(one_b)(VP, BB, w_eff)  # (B, 2^n)
        else:
            Cn_sh = Cn.reshape(psi.shape)

        # Use jax.lax.scan over L layers so XLA compiles 1 layer body instead of unrolling L*C_cav layers
        A_scan = jnp.transpose(A_b, (2, 0, 1, 3, 4))     # (L, B, C, PAIRS, 2)
        TH_scan = jnp.transpose(TH_b, (2, 0, 1, 3, 4))   # (L, B, C, PAIRS, NU)
        SOUT_scan = jnp.transpose(SOUT_b, (2, 0, 1, 3))  # (L, B, C, NU)
        dg_scan = jnp.transpose(dg, (1, 0))              # (L, B)
        wb_scan = jnp.transpose(w_b, (2, 0, 1))          # (L, B, m_or_1)

        def layer_body(psi_c, xs_l):
            A_l, TH_l, SOUT_l, dg_l, wb_l = xs_l
            if not zero_oracle:
                if a.clause_w:
                    w_eff = dg_l[:, None] + wb_l
                    Ew = cost_Ew(w_eff)
                    psi_c = psi_c * jnp.exp(-1j * Ew.reshape(psi_c.shape))
                else:
                    psi_c = psi_c * jnp.exp(-1j * dg_l.reshape((B,) + (1,) * C_cav) * Cn_sh)
            for c in range(C_cav):
                if a.mixer == "gray":
                    b_cl = A_l[:, c, 0, 0]
                    th_cl = TH_l[:, c, 0]
                    sout_cl = SOUT_l[:, c]
                    e_w = jnp.exp(-0.5j * b_cl[:, None] * W_GRAY_J[None, :])
                    U_f = jnp.einsum("ik,bk,jk->bij", V_GRAY_J, e_w, V_GRAY_J.conj())
                    Vp = jnp.exp(-1j * sout_cl)[:, :, None] * U_f * jnp.exp(-1j * th_cl)[:, None, :]
                    M_log_b = Vp[:, FOCK_OF_LOG, :][:, :, FOCK_OF_LOG]
                else:
                    M_log_b = layer_log_batch(A_l[:, c], TH_l[:, c], SOUT_l[:, c])
                psi_c = apply_cav_batch(M_log_b, psi_c, c, C_cav)
            return psi_c, None

        psi, _ = jax.lax.scan(layer_body, psi, (A_scan, TH_scan, SOUT_scan, dg_scan, wb_scan))

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
