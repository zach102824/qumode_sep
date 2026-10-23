"""Dedicated 5-Block GNN + Clause-Weight Explorer at n=6 and n=8 (`gnn_5block.py`).
Strictly enforces:
  1. <= 5 total SNAP+D blocks per cavity including state prep (`0p5x1`, `1p4x1`, `2p3x1`, `1p2x2`).
  2. All SNAP gates strictly restricted to computational Fock levels 0..D-1 (0 phase on leaked levels f >= D).
  3. All Fock leakage (f >= D) counted as failure, verified at NF=40 (n=6) and NF=56 (n=8).
  4. Supports both `--clause_w` (per-clause weights w_{c,l} in the global cost phase) and `--no_clause_w`
     (exact unweighted global cost phase exp(-i gamma_l (n/m) C(x))).
  5. Supports L-BFGS-B on N_train in {500, 2000, 5000, 10000} AND minibatch Adam with early stopping on validation S.
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
ap.add_argument("--n", type=int, default=6, choices=[6, 8])
ap.add_argument("--sched", default="2p3x1", choices=["0p5x1", "1p4x1", "2p3x1", "1p2x2"])
ap.add_argument("--order", default="csd", choices=["csd", "sdc"])
ap.add_argument("--relabel", default="none", choices=["none", "bias", "biaspol", "bppol", "bpdec", "bpcenter", "mpw4"])
ap.add_argument("--clause_w", action="store_true")
ap.add_argument("--pure_gnn", action="store_true", help="Use only raw factor-graph degrees in fv[:,:,0:3] (fv[:,:,3:7]=0, no BP/decimation features)")
ap.add_argument("--arch", default="mp", choices=["mp", "attn", "ds", "mlp"])
ap.add_argument("--H", type=int, default=64)
ap.add_argument("--rounds", type=int, default=3)
ap.add_argument("--ntrain", type=int, default=10000)
ap.add_argument("--opt", default="adam", choices=["bfgs", "adam", "adam_bfgs"])
ap.add_argument("--maxiter", type=int, default=300)
ap.add_argument("--steps", type=int, default=4000)
ap.add_argument("--bs", type=int, default=250)
ap.add_argument("--lr", type=float, default=1.5e-3)
ap.add_argument("--tailw", type=float, default=0.5)
ap.add_argument("--p_cap", type=float, default=0.0, help=">0: downweight gradient once p_succ > p_cap so optimizer focuses on B90 tail")
ap.add_argument("--leakpen", type=float, default=0.1)
ap.add_argument("--wd", type=float, default=1e-4)
ap.add_argument("--amax", type=float, default=1.8)
ap.add_argument("--init_base", default="", help="JSON from part1_2cav.py to warm-start base A0, TH0, g0")
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
if a.tag.endswith("_nocw") or a.tag.endswith("_mpw4_cw"): a.steps = min(a.steps, 800)
if a.pure_gnn:
    a.ntrain = 10000
    a.steps = 1000
    a.opt = "adam" 

n = a.n
C = 2
K = n // 2
NU = 1 << K
NF = 24 if n == 6 else 36
NF_CHK = 40 if n == 6 else 56

SCHED_MAP = {
    "0p5x1": (0, 5, 1),
    "1p4x1": (1, 4, 1),
    "2p3x1": (2, 3, 1),
    "1p2x2": (1, 2, 2),
}
K_PREP, L_ACT, PAIRS = SCHED_MAP[a.sched]
NB_ACT = L_ACT * PAIRS
assert K_PREP + NB_ACT == 5

OUT_DIR = os.path.join(ROOT, "results", "oct23_exp")
OUT = os.path.join(OUT_DIR, f"{a.tag}.json")
PREPS = json.load(open(os.path.join(OUT_DIR, "preps.json")))
DATA = np.load(os.path.join(OUT_DIR, f"data_n{n}.npz"))
BP_DATA = np.load(os.path.join(OUT_DIR, f"bp_n{n}.npz"))
m_clauses = int(DATA["m"])
nm_scale = float(n) / float(m_clauses)

WS_REF = {
    6: {"tries_med": 1.691, "tries_b90": 2.816, "look_med": 8.466, "look_b90": 16.039},
    8: {"tries_med": 2.014, "tries_b90": 3.739, "look_med": 13.989, "look_b90": 29.396},
}[n]

log = lambda *s: print(time.strftime("%H:%M:%S"), f"[{a.tag}]", *s, flush=True)
gray = lambda i: i ^ (i >> 1)

GRID_TO_X = np.zeros((NU, NU), dtype=np.int64)
for f0 in range(NU):
    for f1 in range(NU):
        bits = [(gray(f0) >> j) & 1 for j in range(K)] + [(gray(f1) >> j) & 1 for j in range(K)]
        GRID_TO_X[f0, f1] = sum(b << (n - 1 - v) for v, b in enumerate(bits))

def build_perm_and_flip(pos, neg, h_bp, mode):
    if mode == "none":
        return np.arange(n, dtype=np.int64), np.zeros(n, dtype=np.int64)
    slots_by_speed = [K * c + j for j in range(K) for c in range(C)]
    if mode in ("bias", "biaspol"):
        diff = pos - neg
        var_by_conf = np.argsort(np.abs(diff), kind="stable")
        order = np.empty(n, dtype=np.int64)
        for rank, v_orig in enumerate(var_by_conf):
            order[slots_by_speed[rank]] = v_orig
        flip = np.zeros(n, dtype=np.int64) if mode == "bias" else (diff >= 0).astype(np.int64)[order]
        return order, flip
    if mode == "bpcenter":
        K_c = n // 2
        fc = (1 << (K_c - 1)) - 1  # f_c = 3 for n=6 (K=3), f_c = 7 for n=8 (K=4)
        g_center = gray(fc)
        center_bits = [(g_center >> j) & 1 for j in range(K_c)]
        df_per_j = [abs([i for i in range(1 << K_c) if gray(i) == (g_center ^ (1 << j))][0] - fc) for j in range(K_c)]
        j_by_speed = np.argsort(df_per_j, kind="stable")
        slots_c = [K_c * c + int(j) for j in j_by_speed for c in range(C)]
        var_by_conf = np.argsort(np.abs(h_bp), kind="stable")
        order = np.empty(n, dtype=np.int64)
        for rank, v_orig in enumerate(var_by_conf):
            order[slots_c[rank]] = v_orig
        pref = (h_bp >= 0).astype(np.int64)[order]
        target_bits = np.array(center_bits * C, dtype=np.int64)
        return order, pref ^ target_bits
    if mode in ("bppol", "bpdec", "mpw4"):
        var_by_conf = np.argsort(np.abs(h_bp), kind="stable")
        order = np.empty(n, dtype=np.int64)
        for rank, v_orig in enumerate(var_by_conf):
            order[slots_by_speed[rank]] = v_orig
        flip = (h_bp >= 0).astype(np.int64)[order]
        return order, flip
    raise ValueError(mode)

def prepare_split(split_name, max_count=None, r_branch=None):
    Es = DATA[f"{split_name}_E"]
    cls_v = DATA[f"{split_name}_cls_v"]
    cls_s = DATA[f"{split_name}_cls_s"]
    pos = BP_DATA[f"{split_name}_pos"]
    neg = BP_DATA[f"{split_name}_neg"]
    if r_branch is not None:
        h_bp = BP_DATA[f"{split_name}_h_bp4"][r_branch]
    else:
        h_bp = (BP_DATA[f"{split_name}_h_dec"] if a.relabel in ("bpdec", "bpcenter") else BP_DATA[f"{split_name}_h_bp"])
    if max_count is not None:
        Es = Es[:max_count]; cls_v = cls_v[:max_count]; cls_s = cls_s[:max_count]
        pos = pos[:max_count]; neg = neg[:max_count]; h_bp = h_bp[:max_count]
    B = len(Es)
    Cn_rel = np.zeros((B, NU, NU), np.float32)
    Succ = np.zeros((B, NU, NU), np.float32)
    Ip = np.zeros((B, m_clauses, n), np.float32)
    In = np.zeros((B, m_clauses, n), np.float32)
    fv = np.zeros((B, n, 7), np.float32)
    deg = np.zeros((B, n), np.float32)
    Viol = np.zeros((B, m_clauses, NU * NU), np.uint8) if a.clause_w else None

    x_grid_flat = GRID_TO_X.reshape(-1)  # (NU*NU,)
    slot_bits = np.array([[(x_val >> (n - 1 - s)) & 1 for x_val in x_grid_flat] for s in range(n)], dtype=np.uint8)  # (n, NU*NU)

    orders = np.zeros((B, n), dtype=np.int64)
    flips = np.zeros((B, n), dtype=np.int64)
    inv_orders = np.zeros((B, n), dtype=np.int64)
    for b in range(B):
        order, flip = build_perm_and_flip(pos[b], neg[b], h_bp[b], a.relabel)
        orders[b] = order; flips[b] = flip
        inv_orders[b, order] = np.arange(n)
        x_old = np.zeros_like(GRID_TO_X)
        for s in range(n):
            bit_s = ((GRID_TO_X >> (n - 1 - s)) & 1) ^ int(flip[s])
            x_old |= (bit_s << (n - 1 - int(order[s])))
        E_grid = Es[b][x_old].astype(np.float32)
        c_grid = E_grid * nm_scale
        Cn_rel[b] = c_grid - np.max(c_grid)
        Succ[b] = (E_grid == 0).astype(np.float32)

    b_idx = np.arange(B)[:, None, None]
    c_idx = np.arange(m_clauses)[None, :, None]
    v_new = inv_orders[b_idx, cls_v]  # (B, m, 4)
    s_new = cls_s * (1 - 2 * flips[b_idx, v_new])  # (B, m, 4)
    np.add.at(Ip, (b_idx, c_idx, v_new), (s_new > 0).astype(np.float32))
    np.add.at(In, (b_idx, c_idx, v_new), (s_new < 0).astype(np.float32))
    if a.clause_w:
        need_val = (s_new < 0).astype(np.uint8)[..., None]  # (B, m, 4, 1): literal is false when bit == (s_new < 0)
        Viol[:] = (
            (slot_bits[v_new[:, :, 0]] == need_val[:, :, 0]) &
            (slot_bits[v_new[:, :, 1]] == need_val[:, :, 1]) &
            (slot_bits[v_new[:, :, 2]] == need_val[:, :, 2]) &
            (slot_bits[v_new[:, :, 3]] == need_val[:, :, 3])
        )
    b_v = np.arange(B)[:, None]
    sgn = (1.0 - 2.0 * flips).astype(np.float32)
    fv[:, :, 0] = 1.0
    fv[:, :, 1] = Ip.sum(1) / (2.0 * m_clauses / n)
    fv[:, :, 2] = In.sum(1) / (2.0 * m_clauses / n)
    if not a.pure_gnn and a.relabel not in ("none", "bias", "biaspol"):
        fv[:, :, 3] = BP_DATA[f"{split_name}_h_bp"][:B][b_v, orders] * sgn
        fv[:, :, 4] = BP_DATA[f"{split_name}_h_dec"][:B][b_v, orders] * sgn
        fv[:, :, 5] = BP_DATA[f"{split_name}_h_bp4"][1, :B][b_v, orders] * sgn
        fv[:, :, 6] = BP_DATA[f"{split_name}_h_bp4"][2, :B][b_v, orders] * sgn
    deg[:] = np.maximum(Ip.sum(1) + In.sum(1), 1.0)

    return {
        "B": B,
        "G": {"Ip": jnp.asarray(Ip), "In": jnp.asarray(In), "fv": jnp.asarray(fv), "deg": jnp.asarray(deg)},
        "Cn_rel": jnp.asarray(Cn_rel),
        "Succ": jnp.asarray(Succ),
        "Viol": jnp.asarray(Viol) if a.clause_w else jnp.zeros((B, 1, 1), jnp.uint8),
    }

def build_sim(nf):
    aa = np.diag(np.sqrt(np.arange(1, nf)), 1)
    lam, V = np.linalg.eigh(1j * (aa.T - aa))
    LAM = jnp.asarray(lam.astype(np.float32))
    VD = jnp.asarray(V.astype(np.complex64))
    NV = jnp.arange(nf, dtype=jnp.float32)

    def disp_batch(re, im):
        if a.amax > 0:
            r_raw = jnp.sqrt(re ** 2 + im ** 2 + 1e-20)
            sc = a.amax * jnp.tanh(r_raw / a.amax) / r_raw
            re, im = re * sc, im * sc
        r = jnp.sqrt(re ** 2 + im ** 2 + 1e-20)
        ph = jnp.arctan2(im, re)
        W = jnp.exp(1j * ph[:, None] * NV[None])[:, :, None] * VD[None]
        return jnp.einsum("bik,bk,bjk->bij", W, jnp.exp(-1j * r[:, None] * LAM[None]), W.conj())

    if K_PREP == 0:
        v0 = np.zeros(nf, np.complex64); v0[0] = 1.0
    else:
        pr = PREPS[f"sd_K{K}_k{K_PREP}"]
        v = jnp.zeros(nf, jnp.complex64).at[0].set(1.0)
        for i in range(K_PREP):
            al = pr["alphas"][i]
            th = jnp.concatenate([jnp.asarray(pr["thetas"][i], jnp.float32), jnp.zeros(nf - NU, jnp.float32)])
            r_p = np.hypot(al[0], al[1]); ph_p = np.arctan2(al[1], al[0])
            W_p = np.exp(1j * ph_p * np.arange(nf))[:, None] * V
            D_p = (W_p * np.exp(-1j * r_p * lam)[None]) @ W_p.conj().T
            v = jnp.asarray(D_p, jnp.complex64) @ v
            v = v * jnp.exp(-1j * th)
        v0 = np.asarray(v, np.complex64)
    PSI0 = jnp.asarray(np.outer(v0, v0))

    def sim_fn(A_b, TH_b, g_b, w_b, Cn_rel, Succ, Viol):
        B = Cn_rel.shape[0]
        psi = jnp.broadcast_to(PSI0, (B, nf, nf))
        if a.clause_w:
            Vf = Viol.astype(jnp.float32)  # (B, m, NU*NU)
        for l in range(NB_ACT):
            ll, q = divmod(l, PAIRS)
            if a.order == "csd" and ll < L_ACT and q == 0:
                if a.clause_w:
                    cw = g_b[:, ll, None] + w_b[:, :, ll]  # (B, m)
                    wx = nm_scale * jnp.einsum("bm,bmx->bx", cw, Vf).reshape(B, NU, NU)
                    wx_rel = wx - jnp.max(wx.reshape(B, -1), axis=1)[:, None, None]
                    psi = psi.at[:, :NU, :NU].multiply(jnp.exp(-1j * wx_rel))
                else:
                    psi = psi.at[:, :NU, :NU].multiply(jnp.exp(-1j * g_b[:, ll, None, None] * Cn_rel))
            # SNAP strictly on 0..NU-1, 0 phase on leaked levels f >= NU
            ph0 = jnp.concatenate([TH_b[:, 0, l], jnp.zeros((B, nf - NU), jnp.float32)], axis=1)
            ph1 = jnp.concatenate([TH_b[:, 1, l], jnp.zeros((B, nf - NU), jnp.float32)], axis=1)
            psi = psi * jnp.exp(-1j * ph0)[:, :, None] * jnp.exp(-1j * ph1)[:, None, :]
            D0 = disp_batch(A_b[:, 0, l, 0], A_b[:, 0, l, 1])
            D1 = disp_batch(A_b[:, 1, l, 0], A_b[:, 1, l, 1])
            psi = jnp.einsum("bij,bjk->bik", D0, psi)
            psi = jnp.einsum("bkj,bij->bik", D1, psi)
            if a.order == "sdc" and ll < L_ACT and q == PAIRS - 1:
                if a.clause_w:
                    cw = g_b[:, ll, None] + w_b[:, :, ll]
                    wx = nm_scale * jnp.einsum("bm,bmx->bx", cw, Vf).reshape(B, NU, NU)
                    wx_rel = wx - jnp.max(wx.reshape(B, -1), axis=1)[:, None, None]
                    psi = psi.at[:, :NU, :NU].multiply(jnp.exp(-1j * wx_rel))
                else:
                    psi = psi.at[:, :NU, :NU].multiply(jnp.exp(-1j * g_b[:, ll, None, None] * Cn_rel))
        pr_valid = jnp.abs(psi[:, :NU, :NU]) ** 2
        p_succ = jnp.sum(pr_valid * Succ, axis=(1, 2))
        p_leak = 1.0 - jnp.sum(pr_valid, axis=(1, 2))
        return p_succ, p_leak

    return sim_fn

SIM_MAIN = build_sim(NF)
SIM_CHK = build_sim(NF_CHK)

BT = np.zeros((K, NU), np.float32)
for f in range(NU):
    for j in range(K): BT[j, f] = (gray(f) >> j) & 1
BT = jnp.asarray(BT)
CV = jnp.asarray(np.array([[K * c + j for j in range(K)] for c in range(C)]))

def init_net(rng):
    H, R, FV, FM = a.H, a.rounds, 7, 32
    def W(i, o, s=1.0): return jnp.asarray(rng.normal(0, s / np.sqrt(max(i, 1)), (i, o)).astype(np.float32))
    p = AN.init_backbone(rng, a.arch, H, R, FV)
    p["G1"] = W(2 * H, H); p["g1"] = jnp.zeros(H)
    p["G2_a"] = jnp.zeros((H, C * NB_ACT * 2))
    p["G2_g"] = jnp.zeros((H, L_ACT))
    p["Wa"] = jnp.zeros((H, NB_ACT * 2))
    p["Ws"] = jnp.zeros((H, NB_ACT))
    p["P0"] = W(H, FM, 1.5); p["P1"] = W(H, FM, 1.5)
    p["u0"] = jnp.asarray(rng.normal(0, 0.5, FM).astype(np.float32))
    p["u1"] = jnp.asarray(rng.normal(0, 0.5, FM).astype(np.float32))
    p["fq"] = jnp.zeros(FM)
    p["Wo"] = jnp.zeros((FM, NB_ACT))
    p["Mcav1"] = W(K * (H + 7), H, 1.0)
    p["bcav1"] = jnp.zeros(H)
    p["Wfock"] = jnp.zeros((H, NB_ACT * NU))
    wbit_np = np.array(W(H + 7, FM, 1.0), np.float32, copy=True)
    if not a.pure_gnn and a.relabel not in ("none", "bias", "biaspol"):
        for m_i in range(4):
            wbit_np[:, m_i] *= 0.1
            wbit_np[H + 3 + m_i, m_i] = 6.0
            wbit_np[:, 4 + m_i] *= 0.2
            wbit_np[H + 3 + m_i, 4 + m_i] = 2.0
    p["Wbit"] = jnp.asarray(wbit_np)
    p["bbit"] = jnp.zeros(FM)
    p["Wprod"] = jnp.zeros((9, FM, NB_ACT))
    p["Wphi_a"] = jnp.zeros((NU * FM, NB_ACT * 2))
    if a.clause_w:
        p["Wc"] = jnp.zeros((H, L_ACT))

    if a.init_base and os.path.exists(a.init_base):
        J = json.load(open(a.init_base))
        xb = np.array(J["x_best"], np.float32)
        nc = C if J["args"].get("asym", False) else 1
        nA = nc * NB_ACT * 2; nT = nc * NB_ACT * NU
        A0 = np.broadcast_to(xb[:nA].reshape(nc, NB_ACT, 2), (C, NB_ACT, 2)).copy()
        TH0 = np.broadcast_to(xb[nA:nA+nT].reshape(nc, NB_ACT, NU), (C, NB_ACT, NU)).copy()
        g0 = xb[nA+nT:nA+nT+L_ACT].copy()
    else:
        r = rng.uniform(0.15, 0.35, (C, NB_ACT)); ph = rng.uniform(0, 2*np.pi, (C, NB_ACT))
        A0 = np.stack([r * np.cos(ph), r * np.sin(ph)], -1).astype(np.float32)
        TH0 = np.zeros((C, NB_ACT, NU), np.float32)
        if K_PREP == 0:
            if a.relabel in ("bppol", "bpdec", "mpw4", "biaspol"):
                A0[:, 0, 0] = 0.85; A0[:, 0, 1] = 0.0
            else:
                pr1 = PREPS[f"sd_K{K}_k1"]
                A0[:, 0, :] = np.array(pr1["alphas"][0])[None, :]
                TH0[:, 0, :] = np.array(pr1["thetas"][0])[None, :]
        g0 = np.linspace(1.5, 4.5, L_ACT).astype(np.float32)

    p["A0"] = jnp.asarray(A0)
    p["TH0"] = jnp.asarray(TH0)
    p["g0"] = jnp.asarray(g0)
    return p

def forward_net(p, G):
    B = G["Ip"].shape[0]
    hv, hc = AN.run_backbone(p, G, a.arch, a.H)
    g_pool = jnp.concatenate([hv.mean(1), hc.mean(1)], -1)
    h_g = jnp.tanh(g_pool @ p["G1"] + p["g1"])
    dA_glob = (h_g @ p["G2_a"]).reshape(B, C, NB_ACT, 2)
    dg = p["g0"][None, :] + h_g @ p["G2_g"]
    hd = hv[:, CV]  # (B, C, K, H)
    hd_aug = jnp.concatenate([hd, G["fv"][:, CV]], axis=-1)  # (B, C, K, H + 7)
    hcav_h = jnp.tanh(hd_aug.reshape(B, C, K * (a.H + 7)) @ p["Mcav1"] + p["bcav1"])  # (B, C, H)
    dA_cav = ((hd.mean(2) + hcav_h) @ p["Wa"]).reshape(B, C, NB_ACT, 2)
    A_b = p["A0"][None] + dA_glob + dA_cav

    U0 = p["u0"] + hd @ p["P0"]
    U1 = p["u1"] + hd @ p["P1"]
    z = (jnp.einsum("bcjm,jf->bcfm", U1, BT) + jnp.einsum("bcjm,jf->bcfm", U0, 1.0 - BT)) / K
    ph_free = jnp.transpose(jnp.tanh(z + p["fq"]) @ p["Wo"], (0, 1, 3, 2))  # (B, C, NB_ACT, NU)
    ph_lin = jnp.einsum("bcjl,jf->bclf", hd @ p["Ws"], BT)
    ph_fock = (hcav_h @ p["Wfock"]).reshape(B, C, NB_ACT, NU)

    # Multilinear K-bit Fock-Product SNAP head + 9-tap Fock-space constructive-interference stencil
    P_bit = jax.nn.sigmoid(hd_aug @ p["Wbit"] + p["bbit"])  # (B, C, K, FM)
    M_jf = P_bit[:, :, :, None, :] * BT[None, None, :, :, None] + (1.0 - P_bit[:, :, :, None, :]) * (1.0 - BT[None, None, :, :, None])
    Phi = jnp.prod(M_jf, axis=2)  # (B, C, NU, FM)
    Phi_pad = jnp.pad(Phi, ((0, 0), (0, 0), (4, 4), (0, 0)))
    Phi_st = jnp.stack([Phi_pad[:, :, d:d+NU, :] for d in range(9)], axis=2)  # (B, C, 9, NU, FM)
    ph_prod = jnp.einsum("bcdfm,dml->bclf", Phi_st, p["Wprod"])  # (B, C, NB_ACT, NU)
    dA_phi = (Phi.reshape(B, C, NU * 32) @ p["Wphi_a"]).reshape(B, C, NB_ACT, 2)
    A_b = A_b + dA_phi

    TH_b = p["TH0"][None] + ph_free + ph_lin + ph_fock + ph_prod
    w_b = (hc @ p["Wc"]) if a.clause_w else jnp.zeros((B, 1, L_ACT), jnp.float32)
    return A_b, TH_b, dg, w_b

def eval_metrics(p_succ, p_leak):
    p_succ = np.asarray(p_succ, float)
    p_leak = np.asarray(p_leak, float)
    inv = 1.0 / np.maximum(p_succ, 1e-12)
    med = float(np.median(inv))
    b90 = float(np.quantile(inv, 0.9))
    return {
        "median_inv_p": med, "B90_inv_p": b90,
        "S": float((np.log(med) + np.log(b90)) / 2.0),
        "mean_p": float(np.mean(p_succ)),
        "mean_leak": float(np.mean(p_leak)), "max_leak": float(np.max(p_leak)),
        "loss": float(np.mean(-np.log(np.maximum(p_succ, 1e-12)))),
        "beats_ws_tries_med": bool(med < WS_REF["tries_med"]),
        "beats_ws_tries_b90": bool(b90 < WS_REF["tries_b90"]),
        "beats_ws_look_med": bool(med < WS_REF["look_med"]),
        "beats_ws_look_b90": bool(b90 < WS_REF["look_b90"]),
    }

def main():
    t0 = time.time()
    D_tr = prepare_split("train", max_count=a.ntrain)
    D_va = prepare_split("val")
    D_te = prepare_split("test")

    rng = np.random.default_rng(777 + a.seed)
    p0 = init_net(rng)
    th0, unr = ravel_pytree(p0)
    log(f"Start GNN 5-block n={n} sched={a.sched} order={a.order} relabel={a.relabel} clause_w={a.clause_w} arch={a.arch} H={a.H} ntrain={a.ntrain} opt={a.opt} params={th0.size}")

    def loss_fn(th, G, Cn_rel, Succ, Viol):
        p = unr(th)
        A_b, TH_b, dg, w_b = forward_net(p, G)
        p_s, p_l = SIM_MAIN(A_b, TH_b, dg, w_b, Cn_rel, Succ, Viol)
        nl = -jnp.log(p_s + 1e-12)
        if a.p_cap > 0:
            cap_nl = -np.log(a.p_cap)
            nl_eff = cap_nl + jax.nn.softplus(4.0 * (nl - cap_nl)) / 4.0
        else:
            nl_eff = nl
        out = jnp.mean(nl_eff)
        if a.tailw > 0:
            k = max(1, int(round(0.16 * nl.shape[0])))
            top_nl = jax.lax.top_k(nl, k)[0]
            out = out + a.tailw * jnp.mean(top_nl) + 0.12 * a.tailw * jnp.mean(jnp.exp(jnp.minimum(top_nl, 3.5)))
        if a.leakpen > 0:
            out = out + a.leakpen * jnp.mean(p_l)
        if a.wd > 0:
            keys = ("G2_a", "G2_g", "Wa", "Ws", "Wo", "Wfock", "Wprod", "Wphi_a") + (("Wc",) if a.clause_w else ())
            out = out + a.wd * sum(jnp.sum(p[k_] ** 2) for k_ in keys)
        return out

    vg = jax.jit(jax.value_and_grad(loss_fn))
    ev_main = jax.jit(lambda th, D: SIM_MAIN(*forward_net(unr(th), D["G"]), D["Cn_rel"], D["Succ"], D["Viol"]))
    ev_chk = jax.jit(lambda th, D: SIM_CHK(*forward_net(unr(th), D["G"]), D["Cn_rel"], D["Succ"], D["Viol"]))

    best_val = {"S": np.inf, "th": np.asarray(th0, np.float32), "step": 0}
    def check_val(th_vec, step_id):
        p_v, lk_v = ev_main(jnp.asarray(th_vec, jnp.float32), D_va)
        mv = eval_metrics(p_v, lk_v)
        if mv["S"] < best_val["S"]:
            best_val.update(S=mv["S"], th=np.array(th_vec, dtype=np.float32, copy=True), step=step_id)
        return mv

    m0 = check_val(th0, 0)
    log(f"  step 0 val med/B90 = {m0['median_inv_p']:.3f}/{m0['B90_inv_p']:.3f} S={m0['S']:.4f}")

    x_cur = jnp.asarray(th0, jnp.float32)
    if a.opt in ("adam", "adam_bfgs"):
        mom = jnp.zeros_like(x_cur)
        vel = jnp.zeros_like(x_cur)
        @jax.jit
        def adam_step(x, m, v, g, t, lr):
            m = 0.9 * m + 0.1 * g
            v = 0.999 * v + 0.001 * (g * g)
            mh = m / (1.0 - 0.9 ** t)
            vh = v / (1.0 - 0.999 ** t)
            return x - lr * mh / (jnp.sqrt(vh) + 1e-8), m, v

        rng_b = np.random.default_rng(1234 + a.seed)
        B_tr = D_tr["B"]; bs = min(a.bs, B_tr)
        for st in range(1, a.steps + 1):
            idx = rng_b.choice(B_tr, bs, replace=False)
            G_sub = {k_: v_[idx] for k_, v_ in D_tr["G"].items()}
            v_b, g_b = vg(x_cur, G_sub, D_tr["Cn_rel"][idx], D_tr["Succ"][idx], D_tr["Viol"][idx])
            gnorm = jnp.sqrt(jnp.sum(g_b * g_b) + 1e-12)
            g_b = g_b * jnp.minimum(1.0, 3.0 / gnorm)
            warm = min(1.0, st / 200.0)
            lr_t = a.lr * warm * (0.08 + 0.92 * 0.5 * (1.0 + np.cos(np.pi * st / a.steps)))
            x_cur, mom, vel = adam_step(x_cur, mom, vel, g_b, st, lr_t)
            if st % 200 == 0 or st == a.steps:
                mv = check_val(x_cur, st)
                log(f"  adam step {st}/{a.steps} batch_loss={float(v_b):.4f} | val med/B90 = {mv['median_inv_p']:.3f}/{mv['B90_inv_p']:.3f} S={mv['S']:.4f} (best {best_val['S']:.4f} @ {best_val['step']})")
        x_cur = jnp.asarray(best_val["th"], jnp.float32)

    if a.opt in ("bfgs", "adam_bfgs"):
        B_tr = D_tr["B"]; ch = min(1000, B_tr)
        parts = [
            ({k_: v_[i:i+ch] for k_, v_ in D_tr["G"].items()}, D_tr["Cn_rel"][i:i+ch], D_tr["Succ"][i:i+ch], D_tr["Viol"][i:i+ch], (min(i+ch, B_tr) - i) / float(B_tr))
            for i in range(0, B_tr, ch)
        ]
        it_c = [0]
        def obj_bfgs(x_vec):
            thj = jnp.asarray(x_vec, jnp.float32)
            tot_v = 0.0; tot_g = np.zeros_like(x_vec, dtype=np.float64)
            for gp, cp, sp, vp, w in parts:
                vv, gg = vg(thj, gp, cp, sp, vp)
                tot_v += float(vv) * w
                tot_g += np.asarray(gg, np.float64) * w
            return tot_v, tot_g
        def cb_bfgs(xk):
            it_c[0] += 1
            if it_c[0] % 15 == 0:
                mv = check_val(xk, a.steps + it_c[0])
                log(f"  bfgs it {it_c[0]}/{a.maxiter} | val med/B90 = {mv['median_inv_p']:.3f}/{mv['B90_inv_p']:.3f} S={mv['S']:.4f} (best {best_val['S']:.4f} @ {best_val['step']})")
        r = minimize(obj_bfgs, np.asarray(x_cur, np.float64), jac=True, method="L-BFGS-B", callback=cb_bfgs,
                     options={"maxiter": a.maxiter, "maxfun": int(1.5 * a.maxiter)})
        check_val(r.x, a.steps + it_c[0])

    th_best = jnp.asarray(best_val["th"], jnp.float32)
    np.savez_compressed(OUT.replace(".json", "_theta.npz"), theta=np.asarray(th_best, np.float32))
    m_val = eval_metrics(*ev_main(th_best, D_va))
    p_te, lk_te = ev_main(th_best, D_te)
    m_test = eval_metrics(p_te, lk_te)
    p_chk, lk_chk = ev_chk(th_best, D_te)
    m_chk = eval_metrics(p_chk, lk_chk)

    if a.relabel in ("bpdec", "bpcenter", "mpw4", "bppol"):
        ps_r, lks_r = [np.asarray(p_chk, float)], [np.asarray(lk_chk, float)]
        for r_b in [1, 2, 3]:
            D_r = prepare_split("test", r_branch=r_b)
            pr_i, lk_i = ev_chk(th_best, D_r)
            ps_r.append(np.asarray(pr_i, float)); lks_r.append(np.asarray(lk_i, float))
        ps_r = np.stack(ps_r, axis=0)
        lks_r = np.stack(lks_r, axis=0)
        # Uniform RR4 and weighted RR4 (0.4, 0.2, 0.2, 0.2) and Best-of-4
        m_rr4 = eval_metrics(np.mean(ps_r, axis=0), np.mean(lks_r, axis=0))
        m_wrr4 = eval_metrics(0.40 * ps_r[0] + 0.20 * ps_r[1] + 0.20 * ps_r[2] + 0.20 * ps_r[3], np.mean(lks_r, axis=0))
        m_best4 = eval_metrics(np.max(ps_r, axis=0), np.mean(lks_r, axis=0))
        for d_tgt in (m_test, m_chk):
            d_tgt["rr4_med"] = m_rr4["median_inv_p"]
            d_tgt["rr4_B90"] = m_rr4["B90_inv_p"]
            d_tgt["wrr4_med"] = m_wrr4["median_inv_p"]
            d_tgt["wrr4_B90"] = m_wrr4["B90_inv_p"]
            d_tgt["best4_med"] = m_best4["median_inv_p"]
            d_tgt["best4_B90"] = m_best4["B90_inv_p"]

    res = {
        "args": vars(a), "ws_ref": WS_REF, "NF": NF, "NF_chk": NF_CHK, "NU": NU,
        "best_step": best_val["step"], "val": m_val, "test": m_test, f"test_NF{NF_CHK}": m_chk,
        "p_test": np.asarray(p_te, float).tolist(), "sec": time.time() - t0
    }
    json.dump(res, open(OUT, "w"))
    log(f"DONE {a.tag}: val {m_val['median_inv_p']:.3f}/{m_val['B90_inv_p']:.3f} | test {m_test['median_inv_p']:.3f}/{m_test['B90_inv_p']:.3f} (NF{NF_CHK}: {m_chk['median_inv_p']:.3f}/{m_chk['B90_inv_p']:.3f}, leak={m_chk['mean_leak']:.4f}) vs WS tries {WS_REF['tries_med']:.2f}/{WS_REF['tries_b90']:.2f} ({time.time()-t0:.1f}s)")

if __name__ == "__main__":
    main()
