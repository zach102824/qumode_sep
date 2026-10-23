"""Part 1: Two-Cavity Experimental Circuit Explorer at n=6 (3+3 bits) and n=8 (4+4 bits).

Strict Hardware Rules:
  - At most 5 total SNAP+D or ECD blocks per cavity (including state prep) and at most 5 fixed cost phase layers.
  - SNAP phases are STRICTLY restricted to computational Fock levels 0..D-1 (D=2^K), with 0 phase on leaked levels f >= D.
  - All Fock leakage outside 0..D-1 (or ancilla transmon in |e> when tbit=False) is counted as failure.
  - Verified against larger N_Fock (NF=40 for n=6, NF=56 for n=8).

Supported schedules (--sched):
  - 0p5x1 : 0 fixed prep blocks + 5 active layers of 1 block each (5 blocks total, 4 cost phases between them, or 5 in csd)
  - 1p4x1 : 1 fixed flat-prep block + 4 active layers of 1 block each (5 blocks total, 4 cost phases)
  - 2p3x1 : 2 fixed flat-prep blocks + 3 active layers of 1 block each (5 blocks total, 3 cost phases)
  - 1p2x2 : 1 fixed flat-prep block + 2 active layers of 2 blocks each (5 blocks total, 2 cost phases)

Supported gate primitives (--gate):
  - sd   : Cavity-only SNAP(theta_{0..D-1}) + D(alpha) (ancilla transmon returned to |g>)
  - ecd  : Cavity-only R(theta, phi) + ECD(beta) (ancilla transmon returned to |g>)
  - ecdt : Hybrid Transmon+Cavity R(theta, phi) + ECD(beta) where each transmon carries 1 data bit and cavity carries K-1 bits
           (1t+2c at n=6 -> D=4 cavity levels; 1t+3c at n=8 -> D=8 cavity levels)

Supported compile-time variable relabeling (--relabel, uses ONLY formula clauses, zero access to E):
  - none    : canonical variable order 0..n-1
  - bias    : sort variables by |pos - neg| (least biased -> fast-mixed Gray LSB, most biased -> slow Gray MSB)
  - biaspol : sort by |pos - neg| AND align variable polarity so preferred bit is 0 (Fock 0 = all-0 in Gray code)
  - bppol   : sort by damped-BP confidence |h_bp| AND align variable polarity so preferred bit is 0 (Fock 0 = BP guess)
  - rr4     : evaluate R=4 deterministic variable permutations/polarities round-robin (1 / mean_r p_r)
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
ap.add_argument("--sched", default="0p5x1", choices=["0p5x1", "1p4x1", "2p3x1", "1p2x2"])
ap.add_argument("--order", default="csd", choices=["csd", "sdc"])
ap.add_argument("--gate", default="sd", choices=["sd", "ecd", "ecdt"])
ap.add_argument("--ntrain", type=int, default=10000)
ap.add_argument("--nscreen", type=int, default=1000)
ap.add_argument("--nstarts", type=int, default=16)
ap.add_argument("--screen_it", type=int, default=120)
ap.add_argument("--refine_top", type=int, default=3)
ap.add_argument("--maxiter", type=int, default=500)
ap.add_argument("--chunk", type=int, default=5000)
ap.add_argument("--NF", type=int, default=0, help="0 -> 24 for n=6, 36 for n=8")
ap.add_argument("--nfcheck", type=int, default=0, help="0 -> 40 for n=6, 56 for n=8")
ap.add_argument("--amax", type=float, default=1.8)
ap.add_argument("--relabel", default="none", choices=["none", "bias", "biaspol", "bppol", "bpdec", "bpcenter", "mpw4", "rr4"])
ap.add_argument("--asym", action="store_true", help="Allow distinct shared parameters per cavity c=0,1")
ap.add_argument("--tailw", type=float, default=0.25, help="CVaR top-10% tail weight in loss")
ap.add_argument("--obj", default="logp", choices=["logp", "invp", "s_direct"])
ap.add_argument("--leakpen", type=float, default=0.1)
ap.add_argument("--gnn", action="store_true", help="Train GNN parameter modulator on top of shared base")
ap.add_argument("--arch", default="mp", choices=["mp", "attn", "ds", "mlp"])
ap.add_argument("--H", type=int, default=64)
ap.add_argument("--rounds", type=int, default=3)
ap.add_argument("--clause_w", action="store_true", help="Allow GNN per-clause weights in global cost phase")
ap.add_argument("--init_from", default="", help="Warm-start shared base from a finished JSON")
ap.add_argument("--wd", type=float, default=1e-4)
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
if a.gate in ("ecd", "ecdt"):
    a.ntrain = min(a.ntrain, 2000)
    a.nstarts = min(a.nstarts, 8)
    a.screen_it = min(a.screen_it, 40)
    a.refine_top = 1
    a.maxiter = min(a.maxiter, 80)
elif a.n == 8:
    a.nstarts = min(a.nstarts, 10)
    a.screen_it = min(a.screen_it, 60)
    a.refine_top = 1
    a.maxiter = min(a.maxiter, 120)

n = a.n
C = 2
K_total = n // 2
tbit = (a.gate == "ecdt")
K_cav = (K_total - 1) if tbit else K_total
NU = 1 << K_cav         # Computational Fock levels per cavity: 8 (n=6 sd/ecd), 16 (n=8 sd/ecd), 4 (n=6 ecdt), 8 (n=8 ecdt)
D_reg = 1 << K_total    # Computational states per register: 8 (n=6) or 16 (n=8)
NF = a.NF or (24 if NU <= 8 else 36)
NF_CHK = a.nfcheck or (40 if NU <= 8 else 56)

SCHED_MAP = {
    "0p5x1": (0, 5, 1),  # (k_prep, L_layers, pairs_per_layer) -> total blocks = 0 + 5*1 = 5
    "1p4x1": (1, 4, 1),  # total blocks = 1 + 4*1 = 5
    "2p3x1": (2, 3, 1),  # total blocks = 2 + 3*1 = 5
    "1p2x2": (1, 2, 2),  # total blocks = 1 + 2*2 = 5
}
K_PREP, L_ACT, PAIRS = SCHED_MAP[a.sched]
NB_ACT = L_ACT * PAIRS
assert K_PREP + NB_ACT == 5, "Total blocks per cavity must be 5"

OUT_DIR = os.path.join(ROOT, "results", "oct23_exp")
os.makedirs(OUT_DIR, exist_ok=True)
OUT = os.path.join(OUT_DIR, f"{a.tag}.json")
PREPS = json.load(open(os.path.join(OUT_DIR, "preps.json")))

log = lambda *s: print(time.strftime("%H:%M:%S"), f"[{a.tag}]", *s, flush=True)
gray = lambda i: i ^ (i >> 1)
def ungray(g):
    g = np.array(g, copy=True); s = g >> 1
    while np.any(s): g ^= s; s >>= 1
    return g

# Load cached dataset & compile-time BP features
DATA = np.load(os.path.join(OUT_DIR, f"data_n{n}.npz"))
BP_DATA = np.load(os.path.join(OUT_DIR, f"bp_n{n}.npz"))
m_clauses = int(DATA["m"])
nm_scale = float(n) / float(m_clauses)

# WalkSATlm baselines for reference
WS_REF = {
    6: {"tries_med": 1.691, "tries_b90": 2.816, "look_med": 8.466, "look_b90": 16.039},
    8: {"tries_med": 2.014, "tries_b90": 3.739, "look_med": 13.989, "look_b90": 29.396},
}[n]

def build_perm_and_flip(pos, neg, h_bp, mode, r_idx=0):
    """Returns (order, flip_mask) for a single formula using ONLY compile-time clause info.
    order[new_slot] = old_var_index (which original variable sits at slot new_slot).
    flip_mask[new_slot] in {0, 1}: if 1, logical bit 0 of new_slot corresponds to original var = 1.
    """
    if mode == "none":
        return np.arange(n, dtype=np.int64), np.zeros(n, dtype=np.int64)
    # Slot ordering from fastest-mixed Gray bit (digit 0 of each cavity) to slowest-mixed (MSB digit K-1 of each cavity)
    # In register c, bits are [j=0 .. K_total-1]; in sim.py/sd_n6.py, var 0 is MSB of x.
    # For cavity-only (sd/ecd): slot K_total*c + j is digit j of cavity c.
    # Digit 0 flips at every odd Fock step (NU/2 edges); digit K_cav-1 only flips at the middle boundary (1 edge)!
    slots_by_speed = [K_total * c + j for j in range(K_total) for c in range(C)]

    if mode in ("bias", "biaspol"):
        diff = pos - neg
        score = np.abs(diff)
        var_by_conf = np.argsort(score, kind="stable")  # least biased first -> fastest slot; most biased -> slowest slot
        order = np.empty(n, dtype=np.int64)
        for rank, v_orig in enumerate(var_by_conf):
            order[slots_by_speed[rank]] = v_orig
        if mode == "bias":
            return order, np.zeros(n, dtype=np.int64)
        # Align polarity so preferred assignment (diff >= 0 -> x=1) maps to logical bit 0 (Fock |0> is all-0!)
        pref = (diff >= 0).astype(np.int64)
        flip = pref[order]
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
        score = np.abs(h_bp)
        var_by_conf = np.argsort(score, kind="stable")
        order = np.empty(n, dtype=np.int64)
        for rank, v_orig in enumerate(var_by_conf):
            order[slots_by_speed[rank]] = v_orig
        pref = (h_bp >= 0).astype(np.int64)
        flip = pref[order]
        return order, flip

    if mode == "rr4":
        # h_bp passed in is already h_bp4[r_idx, b] from the r_idx-th decimated BP branch!
        return build_perm_and_flip(pos, neg, h_bp, "bppol", 0)
    raise ValueError(mode)

# Map logical n-bit index x_new (0..2^n-1) to register state indices (d0, d1) in (D_reg, D_reg)
# Convention: register c holds bits [K_total*c + j for j in range(K_total)], where bit v has weight 2^(n - 1 - v).
GRID_TO_X = np.zeros((D_reg, D_reg), dtype=np.int64)
for d0 in range(D_reg):
    for d1 in range(D_reg):
        if not tbit:
            # All K_total bits in cavity Fock level d_c via Gray code: digit j = (gray(d_c) >> j) & 1
            b0 = [(gray(d0) >> j) & 1 for j in range(K_total)]
            b1 = [(gray(d1) >> j) & 1 for j in range(K_total)]
        else:
            # Transmon holds bit 0 of register (d_c >> K_cav), cavity holds bits 1..K_total-1 via Gray code of (d_c & (NU-1))
            t0, f0 = d0 >> K_cav, d0 & (NU - 1)
            t1, f1 = d1 >> K_cav, d1 & (NU - 1)
            b0 = [t0] + [(gray(f0) >> j) & 1 for j in range(K_cav)]
            b1 = [t1] + [(gray(f1) >> j) & 1 for j in range(K_cav)]
        bits = b0 + b1
        x_val = sum(b << (n - 1 - v) for v, b in enumerate(bits))
        GRID_TO_X[d0, d1] = x_val

def permute_energy_table(E_orig, order, flip):
    """Given E_orig (2^n,) indexed by original variables (var 0 = MSB),
    and new slot s mapped to old variable v = order[s] with polarity flip[s],
    return E_grid (D_reg, D_reg) in the new register coordinates."""
    x_new = GRID_TO_X  # (D_reg, D_reg) integers in 0..2^n-1
    x_old = np.zeros_like(x_new)
    for s in range(n):
        bit_s = ((x_new >> (n - 1 - s)) & 1) ^ int(flip[s])
        v_old = int(order[s])
        x_old |= (bit_s << (n - 1 - v_old))
    return E_orig[x_old]

def prepare_split(split_name, max_count=None, relabel_mode=None, r_idx=0):
    rmode = relabel_mode if relabel_mode is not None else a.relabel
    if rmode == "rr4":
        rmode_use = "rr4"
    else:
        rmode_use = rmode
    Es = DATA[f"{split_name}_E"]
    nsols = DATA[f"{split_name}_nsol"]
    cls_v = DATA[f"{split_name}_cls_v"]
    cls_s = DATA[f"{split_name}_cls_s"]
    h_bp = BP_DATA[f"{split_name}_h_bp4"][r_idx] if rmode_use == "rr4" else (BP_DATA[f"{split_name}_h_mpw4"] if rmode_use == "mpw4" else (BP_DATA[f"{split_name}_h_dec"] if rmode_use in ("bpdec", "bpcenter") else BP_DATA[f"{split_name}_h_bp"]))
    pos = BP_DATA[f"{split_name}_pos"]
    neg = BP_DATA[f"{split_name}_neg"]
    if max_count is not None:
        Es = Es[:max_count]; nsols = nsols[:max_count]
        cls_v = cls_v[:max_count]; cls_s = cls_s[:max_count]
        h_bp = h_bp[:max_count]; pos = pos[:max_count]; neg = neg[:max_count]
    B = len(Es)
    Cn_rel = np.zeros((B, D_reg, D_reg), np.float32)
    Succ = np.zeros((B, D_reg, D_reg), np.float32)

    # Also build GNN graph tensors if needed
    need_graph = a.gnn
    if need_graph:
        Ip = np.zeros((B, m_clauses, n), np.float32)
        In = np.zeros((B, m_clauses, n), np.float32)
        fv = np.zeros((B, n, 7), np.float32)
        deg = np.zeros((B, n), np.float32)
    else:
        Ip = In = fv = deg = None

    for b in range(B):
        order, flip = build_perm_and_flip(pos[b], neg[b], h_bp[b], rmode_use, r_idx)
        E_grid = permute_energy_table(Es[b], order, flip).astype(np.float32)
        c_grid = E_grid * nm_scale
        c_max = np.max(c_grid)
        Cn_rel[b] = c_grid - c_max
        Succ[b] = (E_grid == 0).astype(np.float32)
        if need_graph:
            inv_order = np.empty(n, dtype=np.int64)
            inv_order[order] = np.arange(n)
            for c_i in range(m_clauses):
                for k_i in range(4):
                    v_old = int(cls_v[b, c_i, k_i])
                    s_old = int(cls_s[b, c_i, k_i])
                    s_new = int(inv_order[v_old])
                    sign_new = s_old * (1 if flip[s_new] == 0 else -1)
                    if sign_new > 0:
                        Ip[b, c_i, s_new] = 1.0
                    else:
                        In[b, c_i, s_new] = 1.0
            fv[b, :, 0] = 1.0
            fv[b, :, 1] = Ip[b].sum(0) / (2.0 * m_clauses / n)
            fv[b, :, 2] = In[b].sum(0) / (2.0 * m_clauses / n)
            fv[b, :, 3] = 0.0 if rmode_use in ("none", "bias", "biaspol") else (h_bp[b, order] * (1.0 - 2.0 * flip))
            deg[b] = np.maximum(Ip[b].sum(0) + In[b].sum(0), 1.0)

    G = {"Ip": jnp.asarray(Ip), "In": jnp.asarray(In), "fv": jnp.asarray(fv), "deg": jnp.asarray(deg)} if need_graph else None
    return {
        "B": B,
        "Cn_rel": jnp.asarray(Cn_rel),
        "Succ": jnp.asarray(Succ),
        "nsol": nsols,
        "G": G,
    }

# Build simulator for given NF
def build_simulator(nf):
    aa = np.diag(np.sqrt(np.arange(1, nf)), 1)
    lam, V = np.linalg.eigh(1j * (aa.T - aa))
    LAM = jnp.asarray(lam.astype(np.float32))
    VD = jnp.asarray(V.astype(np.complex64))
    NV = jnp.arange(nf, dtype=jnp.float32)

    def disp_batch(re, im):
        """re, im: (B,) -> (B, nf, nf) complex64."""
        if a.amax > 0:
            r_raw = jnp.sqrt(re ** 2 + im ** 2 + 1e-20)
            sc = a.amax * jnp.tanh(r_raw / a.amax) / r_raw
            re, im = re * sc, im * sc
        r = jnp.sqrt(re ** 2 + im ** 2 + 1e-20)
        ph = jnp.arctan2(im, re)
        W = jnp.exp(1j * ph[:, None] * NV[None])[:, :, None] * VD[None]
        return jnp.einsum("bik,bk,bjk->bij", W, jnp.exp(-1j * r[:, None] * LAM[None]), W.conj())

    def disp_single(re, im):
        """re, im: scalar -> (nf, nf) complex64."""
        if a.amax > 0:
            r_raw = jnp.sqrt(re ** 2 + im ** 2 + 1e-20)
            sc = a.amax * jnp.tanh(r_raw / a.amax) / r_raw
            re, im = re * sc, im * sc
        r = jnp.sqrt(re ** 2 + im ** 2 + 1e-20)
        ph = jnp.arctan2(im, re)
        W = jnp.exp(1j * ph * NV)[:, None] * VD
        return (W * jnp.exp(-1j * r * LAM)[None]) @ W.conj().T

    # Initial state psi0 after K_PREP fixed blocks on each register
    if a.gate == "sd":
        if K_PREP == 0:
            v0 = np.zeros(nf, np.complex64); v0[0] = 1.0
        else:
            pr = PREPS[f"sd_K{K_cav}_k{K_PREP}"]
            # Re-simulate exact K_PREP blocks at current nf (with SNAP strictly restricted to 0..NU-1)
            v = jnp.zeros(nf, jnp.complex64).at[0].set(1.0)
            for i in range(K_PREP):
                al = pr["alphas"][i]
                th = jnp.concatenate([jnp.asarray(pr["thetas"][i], jnp.float32), jnp.zeros(nf - NU, jnp.float32)])
                # Note: fit_sd_prep doesn't bound alpha by amax, use unbounded disp for fixed prep
                r_p = np.hypot(al[0], al[1]); ph_p = np.arctan2(al[1], al[0])
                W_p = np.exp(1j * ph_p * np.arange(nf))[:, None] * V
                D_p = (W_p * np.exp(-1j * r_p * lam)[None]) @ W_p.conj().T
                v = jnp.asarray(D_p, jnp.complex64) @ v
                v = v * jnp.exp(-1j * th)
            v0 = np.asarray(v, np.complex64)
        PSI0 = jnp.asarray(np.outer(v0, v0))  # (nf, nf)

        def sim_shared(base, Cn_rel, Succ):
            """base:
            - if not asym: (NB_ACT, 2 + NU) for alpha_re, alpha_im, theta_{0..NU-1}, plus gamma (L_ACT,)
            Wait: let's pass A (C, NB_ACT, 2), TH (C, NB_ACT, NU), G_cost (L_ACT,).
            """
            A, TH, G_cost = base["A"], base["TH"], base["g"]
            B = Cn_rel.shape[0]
            # Precompute single-cavity unitaries U_c[l] = D(alpha_{c,l}) @ SNAP(theta_{c,l})
            # Wait: in sd_n6, each block applies SNAP(theta) on 0..NU-1 (0 on >=NU) then D(alpha).
            psi = jnp.broadcast_to(PSI0, (B, nf, nf))
            for l in range(NB_ACT):
                ll, q = divmod(l, PAIRS)
                if a.order == "csd" and ll < L_ACT and q == 0:
                    # Global cost phase on valid block (0..NU-1, 0..NU-1); spare levels get 0 since Cn_rel = Cn - cmax
                    psi = psi.at[:, :NU, :NU].multiply(jnp.exp(-1j * G_cost[ll] * Cn_rel))
                # Apply SNAP (strictly on 0..NU-1, zero on f>=NU) + D(alpha) on cavity 0 and cavity 1
                ph0 = jnp.concatenate([TH[0, l], jnp.zeros(nf - NU, jnp.float32)])
                ph1 = jnp.concatenate([TH[1, l], jnp.zeros(nf - NU, jnp.float32)])
                U0 = disp_single(A[0, l, 0], A[0, l, 1]) * jnp.exp(-1j * ph0)[None, :]
                U1 = disp_single(A[1, l, 0], A[1, l, 1]) * jnp.exp(-1j * ph1)[None, :]
                psi = jnp.einsum("ij,bjk->bik", U0, psi)
                psi = jnp.einsum("kj,bij->bik", U1, psi)
                if a.order == "sdc" and ll < L_ACT and q == PAIRS - 1:
                    psi = psi.at[:, :NU, :NU].multiply(jnp.exp(-1j * G_cost[ll] * Cn_rel))
            pr_valid = jnp.abs(psi[:, :NU, :NU]) ** 2
            p_succ = jnp.sum(pr_valid * Succ, axis=(1, 2))
            p_leak = 1.0 - jnp.sum(pr_valid, axis=(1, 2))
            return p_succ, p_leak

        def sim_batch(A_b, TH_b, G_b, Cn_rel, Succ):
            """Per-instance parameters from GNN: A_b (B, C, NB_ACT, 2), TH_b (B, C, NB_ACT, NU), G_b (B, L_ACT)."""
            B = Cn_rel.shape[0]
            psi = jnp.broadcast_to(PSI0, (B, nf, nf))
            for l in range(NB_ACT):
                ll, q = divmod(l, PAIRS)
                if a.order == "csd" and ll < L_ACT and q == 0:
                    psi = psi.at[:, :NU, :NU].multiply(jnp.exp(-1j * G_b[:, ll, None, None] * Cn_rel))
                ph0 = jnp.concatenate([TH_b[:, 0, l], jnp.zeros((B, nf - NU), jnp.float32)], axis=1)
                ph1 = jnp.concatenate([TH_b[:, 1, l], jnp.zeros((B, nf - NU), jnp.float32)], axis=1)
                psi = psi * jnp.exp(-1j * ph0)[:, :, None] * jnp.exp(-1j * ph1)[:, None, :]
                D0 = disp_batch(A_b[:, 0, l, 0], A_b[:, 0, l, 1])
                D1 = disp_batch(A_b[:, 1, l, 0], A_b[:, 1, l, 1])
                psi = jnp.einsum("bij,bjk->bik", D0, psi)
                psi = jnp.einsum("bkj,bij->bik", D1, psi)
                if a.order == "sdc" and ll < L_ACT and q == PAIRS - 1:
                    psi = psi.at[:, :NU, :NU].multiply(jnp.exp(-1j * G_b[:, ll, None, None] * Cn_rel))
            pr_valid = jnp.abs(psi[:, :NU, :NU]) ** 2
            p_succ = jnp.sum(pr_valid * Succ, axis=(1, 2))
            p_leak = 1.0 - jnp.sum(pr_valid, axis=(1, 2))
            return p_succ, p_leak

        return sim_shared, sim_batch

    else:
        # ECD / ECDT gate: each register has state (2, nf), total state is (2, nf, 2, nf) -> reshaped as (2*nf, 2*nf)
        if K_PREP == 0:
            v0 = np.zeros((2, nf), np.complex64); v0[0, 0] = 1.0
        else:
            pr = PREPS[f"{a.gate}_K{K_cav}_k{K_PREP}"]
            v = jnp.zeros((2, nf), jnp.complex64).at[0, 0].set(1.0)
            for i in range(K_PREP):
                bre, bim, th, ph = pr["params"][i]
                ct, st = np.cos(th / 2), np.sin(th / 2)
                e = np.exp(-1j * ph)
                p0 = ct * v[0] - 1j * st * e * v[1]
                p1 = -1j * st * np.conj(e) * v[0] + ct * v[1]
                r_p = np.hypot(bre, bim); ph_p = np.arctan2(bim, bre)
                W_p = np.exp(1j * ph_p * np.arange(nf))[:, None] * V
                Dp = jnp.asarray((W_p * np.exp(-1j * r_p * lam)[None]) @ W_p.conj().T, jnp.complex64)
                Dm = Dp.conj().T
                v = jnp.stack([Dm @ p1, Dp @ p0], axis=0)
            v0 = np.asarray(v, np.complex64)
        # PSI0 has shape (2, nf, 2, nf)
        PSI0 = jnp.asarray(np.multiply.outer(v0, v0))

        def ecd_reg_unitary(par):
            """par: [bre, bim, th, ph] -> (2*nf, 2*nf) unitary on single (transmon, cavity) register."""
            bre, bim, th, ph = par[0], par[1], par[2], par[3]
            ct, st = jnp.cos(th / 2), jnp.sin(th / 2)
            e = jnp.exp(-1j * ph)
            Dp = disp_single(bre, bim)
            Dm = Dp.conj().T
            # R then ECD:
            # |0, f> -> ct |0, f> - i st e* |1, f> --ECD--> -i st e* Dm |0, f> + ct Dp |1, f>
            # |1, f> -> -i st e |0, f> + ct |1, f> --ECD--> ct Dm |0, f> - i st e Dp |1, f>
            U00 = -1j * st * jnp.conj(e) * Dm
            U01 = ct * Dm
            U10 = ct * Dp
            U11 = -1j * st * e * Dp
            return jnp.block([[U00, U01], [U10, U11]])

        def apply_cost_ecd(psi, g_val, Cn_rel):
            if not tbit:
                # Ancilla transmons: cost phase applies identically to all 4 transmon states (t0, t1) on Fock 0..NU-1
                ph = jnp.exp(-1j * g_val * Cn_rel)  # (B, NU, NU)
                return psi.at[:, :, :NU, :, :NU].multiply(ph[:, None, :, None, :])
            else:
                # Hybrid transmon+cavity: Cn_rel is (B, 2*NU, 2*NU) -> reshape to (B, 2, NU, 2, NU)
                B = Cn_rel.shape[0]
                ph = jnp.exp(-1j * g_val * Cn_rel.reshape(B, 2, NU, 2, NU))
                return psi.at[:, :, :NU, :, :NU].multiply(ph)

        def sim_shared(base, Cn_rel, Succ):
            """base["E"]: (C, NB_ACT, 4), base["g"]: (L_ACT,)."""
            E_par, G_cost = base["E"], base["g"]
            B = Cn_rel.shape[0]
            psi = jnp.broadcast_to(PSI0, (B, 2, nf, 2, nf))
            for l in range(NB_ACT):
                ll, q = divmod(l, PAIRS)
                if a.order == "csd" and ll < L_ACT and q == 0:
                    psi = apply_cost_ecd(psi, G_cost[ll], Cn_rel)
                U0 = ecd_reg_unitary(E_par[0, l]).reshape(2, nf, 2, nf)
                U1 = ecd_reg_unitary(E_par[1, l]).reshape(2, nf, 2, nf)
                psi = jnp.einsum("pqij,bijcd->bpqcd", U0, psi)
                psi = jnp.einsum("rsij,bpqij->bpqrs", U1, psi)
                if a.order == "sdc" and ll < L_ACT and q == PAIRS - 1:
                    psi = apply_cost_ecd(psi, G_cost[ll], Cn_rel)
            if not tbit:
                # Ancilla transmon must be in |g> (index 0) and Fock in 0..NU-1
                pr_valid = jnp.abs(psi[:, 0, :NU, 0, :NU]) ** 2
                p_succ = jnp.sum(pr_valid * Succ, axis=(1, 2))
                p_leak = 1.0 - jnp.sum(pr_valid, axis=(1, 2))
            else:
                pr_valid = (jnp.abs(psi[:, :, :NU, :, :NU]) ** 2).reshape(B, 2 * NU, 2 * NU)
                p_succ = jnp.sum(pr_valid * Succ, axis=(1, 2))
                p_leak = 1.0 - jnp.sum(pr_valid, axis=(1, 2))
            return p_succ, p_leak

        return sim_shared, None

SIM_MAIN, SIM_BATCH_MAIN = build_simulator(NF)
SIM_CHK, SIM_BATCH_CHK = build_simulator(NF_CHK)

def unpack_shared(x):
    """Map flat parameter vector x to dict(A, TH, g) or dict(E, g)."""
    nc = C if a.asym else 1
    if a.gate == "sd":
        # A: (nc, NB_ACT, 2), TH: (nc, NB_ACT, NU), g: (L_ACT,)
        nA = nc * NB_ACT * 2
        nT = nc * NB_ACT * NU
        A = x[:nA].reshape(nc, NB_ACT, 2)
        TH = x[nA:nA + nT].reshape(nc, NB_ACT, NU)
        g = x[nA + nT:nA + nT + L_ACT]
        if not a.asym:
            A = jnp.broadcast_to(A, (C, NB_ACT, 2))
            TH = jnp.broadcast_to(TH, (C, NB_ACT, NU))
        return {"A": A, "TH": TH, "g": g}
    else:
        nE = nc * NB_ACT * 4
        E_par = x[:nE].reshape(nc, NB_ACT, 4)
        g = x[nE:nE + L_ACT]
        if not a.asym:
            E_par = jnp.broadcast_to(E_par, (C, NB_ACT, 4))
        return {"E": E_par, "g": g}

def rand_shared_init(rng, idx=0):
    nc = C if a.asym else 1
    if a.gate == "sd":
        # Varied initialization scales across starts
        a_scale = 0.25 + 0.65 * (idx % 4) / 3.0
        g_scale = 1.5 + 3.5 * ((idx // 4) % 4) / 3.0
        r = rng.uniform(0.1, a_scale, (nc, NB_ACT))
        ph = rng.uniform(0, 2 * np.pi, (nc, NB_ACT))
        A = np.stack([r * np.cos(ph), r * np.sin(ph)], axis=-1)
        # If 0p5x1, first block from vacuum needs larger displacement to populate Fock 0..NU-1
        if K_PREP == 0:
            pr1 = PREPS[f"sd_K{K_cav}_k1"]
            if a.relabel in ("bppol", "bpdec", "mpw4", "biaspol", "rr4"):
                # Solutions are concentrated in low Fock levels f=0..4; initialize Block 0 with moderate displacement
                r0 = rng.uniform(0.5, 1.3, size=(nc,))
                p0 = rng.uniform(0, 2 * np.pi, size=(nc,))
                A[:, 0, 0] = r0 * np.cos(p0)
                A[:, 0, 1] = r0 * np.sin(p0)
                TH = rng.normal(0, 0.3, (nc, NB_ACT, NU))
            else:
                A[:, 0, 0] = pr1["alphas"][0][0] * rng.uniform(0.85, 1.15, size=(nc,))
                A[:, 0, 1] = pr1["alphas"][0][1] * rng.uniform(0.85, 1.15, size=(nc,))
                TH = rng.uniform(-0.3, 0.3, (nc, NB_ACT, NU))
                TH[:, 0, :] = np.array(pr1["thetas"][0])[None, :] + rng.normal(0, 0.15, (nc, NU))
        else:
            TH = rng.normal(0, 0.4, (nc, NB_ACT, NU))
        g = np.linspace(0.8, g_scale, L_ACT) * rng.uniform(0.7, 1.3, L_ACT)
        return np.concatenate([A.reshape(-1), TH.reshape(-1), g.reshape(-1)]).astype(np.float32)
    else:
        E_par = np.zeros((nc, NB_ACT, 4), np.float32)
        E_par[:, :, 0] = rng.normal(0, 0.6, (nc, NB_ACT))
        E_par[:, :, 1] = rng.normal(0, 0.6, (nc, NB_ACT))
        E_par[:, :, 2] = rng.uniform(0.2, np.pi - 0.2, (nc, NB_ACT))
        E_par[:, :, 3] = rng.uniform(0, 2 * np.pi, (nc, NB_ACT))
        if K_PREP == 0:
            pr1 = PREPS[f"{a.gate}_K{K_cav}_k1"]
            E_par[:, 0, :] = np.array(pr1["params"][0])[None, :]
        g = np.linspace(0.8, 3.5, L_ACT) * rng.uniform(0.7, 1.3, L_ACT)
        return np.concatenate([E_par.reshape(-1), g.reshape(-1)]).astype(np.float32)

def loss_shared(x, Cn_rel, Succ):
    base = unpack_shared(x)
    p_succ, p_leak = SIM_MAIN(base, Cn_rel, Succ)
    if a.obj == "invp":
        inv = 1.0 / jnp.maximum(p_succ, 1e-4)
        k = max(1, inv.shape[0] // 10)
        out = jnp.mean(inv) + a.tailw * jnp.mean(jax.lax.top_k(inv, k)[0])
    elif a.obj == "s_direct":
        nl = -jnp.log(p_succ + 1e-12)
        s_nl = jnp.sort(nl)
        B = nl.shape[0]
        # Smooth quantile band around 50% (40%..60%) and 90% (85%..95%)
        med_band = jnp.mean(s_nl[int(0.40 * B):max(int(0.40 * B) + 1, int(0.60 * B))])
        b90_band = jnp.mean(s_nl[int(0.85 * B):max(int(0.85 * B) + 1, int(0.95 * B))])
        out = 0.5 * (med_band + b90_band) + 0.2 * jnp.mean(nl)
    else:
        nl = -jnp.log(p_succ + 1e-12)
        out = jnp.mean(nl)
        if a.tailw > 0:
            k = max(1, nl.shape[0] // 10)
            out = out + a.tailw * jnp.mean(jax.lax.top_k(nl, k)[0])
    if a.leakpen > 0:
        out = out + a.leakpen * jnp.mean(p_leak)
    return out

VG_SHARED = jax.jit(jax.value_and_grad(loss_shared))
EVAL_MAIN = jax.jit(lambda x, Cn_rel, Succ: SIM_MAIN(unpack_shared(x), Cn_rel, Succ))
EVAL_CHK = jax.jit(lambda x, Cn_rel, Succ: SIM_CHK(unpack_shared(x), Cn_rel, Succ))

def make_chunked_obj(split_dict):
    B = split_dict["B"]
    ch = min(a.chunk, B)
    parts = [
        (split_dict["Cn_rel"][i:i + ch], split_dict["Succ"][i:i + ch], (min(i + ch, B) - i) / float(B))
        for i in range(0, B, ch)
    ]
    def obj(x):
        xj = jnp.asarray(x, jnp.float32)
        tot_v = 0.0
        tot_g = np.zeros_like(x, dtype=np.float64)
        for c_part, s_part, w in parts:
            v, g = VG_SHARED(xj, c_part, s_part)
            tot_v += float(v) * w
            tot_g += np.asarray(g, np.float64) * w
        return tot_v, tot_g
    return obj

def eval_metrics(p_succ, p_leak):
    p_succ = np.asarray(p_succ, float)
    p_leak = np.asarray(p_leak, float)
    inv = 1.0 / np.maximum(p_succ, 1e-12)
    med = float(np.median(inv))
    b90 = float(np.quantile(inv, 0.9))
    return {
        "median_inv_p": med,
        "B90_inv_p": b90,
        "S": float((np.log(med) + np.log(b90)) / 2.0),
        "mean_p": float(np.mean(p_succ)),
        "mean_leak": float(np.mean(p_leak)),
        "max_leak": float(np.max(p_leak)),
        "loss": float(np.mean(-np.log(np.maximum(p_succ, 1e-12)))),
        "beats_ws_tries_med": bool(med < WS_REF["tries_med"]),
        "beats_ws_tries_b90": bool(b90 < WS_REF["tries_b90"]),
        "beats_ws_look_med": bool(med < WS_REF["look_med"]),
        "beats_ws_look_b90": bool(b90 < WS_REF["look_b90"]),
    }

def evaluate_split(x, split_name, max_count=None, use_chk=False):
    ev_fn = EVAL_CHK if use_chk else EVAL_MAIN
    xj = jnp.asarray(x, jnp.float32)
    if a.relabel != "rr4":
        D_sp = prepare_split(split_name, max_count=max_count)
        B = D_sp["B"]; ch = min(a.chunk, B)
        ps, lks = [], []
        for i in range(0, B, ch):
            p, lk = ev_fn(xj, D_sp["Cn_rel"][i:i+ch], D_sp["Succ"][i:i+ch])
            ps.append(np.asarray(p)); lks.append(np.asarray(lk))
        return eval_metrics(np.concatenate(ps), np.concatenate(lks)), np.concatenate(ps)
    else:
        # Round-robin across R=4 deterministic variable relabelings
        ps_r, lks_r = [], []
        for r_idx in range(4):
            D_sp = prepare_split(split_name, max_count=max_count, relabel_mode="rr4", r_idx=r_idx)
            B = D_sp["B"]; ch = min(a.chunk, B)
            ps, lks = [], []
            for i in range(0, B, ch):
                p, lk = ev_fn(xj, D_sp["Cn_rel"][i:i+ch], D_sp["Succ"][i:i+ch])
                ps.append(np.asarray(p)); lks.append(np.asarray(lk))
            ps_r.append(np.concatenate(ps)); lks_r.append(np.concatenate(lks))
        p_mean = np.mean(ps_r, axis=0)
        lk_mean = np.mean(lks_r, axis=0)
        p_best = np.max(ps_r, axis=0)
        m_rr = eval_metrics(p_mean, lk_mean)
        m_best = eval_metrics(p_best, lk_mean)
        m_rr["best_of_4_med"] = m_best["median_inv_p"]
        m_rr["best_of_4_B90"] = m_best["B90_inv_p"]
        m_rr["r0_med"] = eval_metrics(ps_r[0], lks_r[0])["median_inv_p"]
        m_rr["r0_B90"] = eval_metrics(ps_r[0], lks_r[0])["B90_inv_p"]
        return m_rr, p_mean

# ---------------- GNN Training Mode (if --gnn) ----------------
def run_gnn_mode(x_base):
    """Train a compact GNN on N_train instances that predicts per-instance modulation
    (delta_A, delta_TH, delta_g) on top of the converged shared BFGS solution x_base."""
    assert a.gate == "sd", "GNN mode currently supports SD gate"
    t0 = time.time()
    D_tr = prepare_split("train", max_count=a.ntrain, relabel_mode=("bpdec" if a.relabel == "rr4" else a.relabel))
    D_va = prepare_split("val", relabel_mode=("bpdec" if a.relabel == "rr4" else a.relabel))
    D_te = prepare_split("test", relabel_mode=("bpdec" if a.relabel == "rr4" else a.relabel))

    base_unp = unpack_shared(jnp.asarray(x_base, jnp.float32))
    A0, TH0, g0 = base_unp["A"], base_unp["TH"], base_unp["g"]  # (C, NB_ACT, 2), (C, NB_ACT, NU), (L_ACT,)

    # Digit bit table BT: (K_cav, NU)
    BT = np.zeros((K_cav, NU), np.float32)
    for f in range(NU):
        for j in range(K_cav):
            BT[j, f] = (gray(f) >> j) & 1
    BT = jnp.asarray(BT)
    CV = jnp.asarray(np.array([[K_cav * c + j for j in range(K_cav)] for c in range(C)]))

    rng = np.random.default_rng(100 + a.seed)
    H, R, FV, FM = a.H, a.rounds, 7, 32
    def W(i, o, s=1.0): return jnp.asarray(rng.normal(0, s / np.sqrt(max(i, 1)), (i, o)).astype(np.float32))
    p0 = AN.init_backbone(rng, a.arch, H, R, FV)
    p0["G1"] = W(2 * H, H); p0["g1"] = jnp.zeros(H)
    p0["G2_a"] = jnp.zeros((H, C * NB_ACT * 2))
    p0["G2_g"] = jnp.zeros((H, L_ACT))
    p0["Wa"] = jnp.zeros((H, NB_ACT * 2))
    # Linear + Walsh digit-pattern SNAP head (strictly on 0..NU-1; zero on >=NU!)
    p0["Ws"] = jnp.zeros((H, NB_ACT))
    p0["P0"] = W(H, FM, 1.5); p0["P1"] = W(H, FM, 1.5)
    p0["u0"] = jnp.asarray(rng.normal(0, 0.5, FM).astype(np.float32))
    p0["u1"] = jnp.asarray(rng.normal(0, 0.5, FM).astype(np.float32))
    p0["fq"] = jnp.zeros(FM)
    p0["Wo"] = jnp.zeros((FM, NB_ACT))
    p0["A0"] = A0; p0["TH0"] = TH0; p0["g0"] = g0

    th0, unr = ravel_pytree(p0)
    log(f"GNN init from shared base: params={th0.size}, arch={a.arch}, H={H}, ntrain={a.ntrain}")

    def forward_gnn(p, G):
        B = G["Ip"].shape[0]
        hv, hc = AN.run_backbone(p, G, a.arch, H)
        g_pool = jnp.concatenate([hv.mean(1), hc.mean(1)], -1)
        h_g = jnp.tanh(g_pool @ p["G1"] + p["g1"])
        dA_glob = (h_g @ p["G2_a"]).reshape(B, C, NB_ACT, 2)
        dg = p["g0"][None, :] + h_g @ p["G2_g"]
        hd = hv[:, CV]  # (B, C, K_cav, H)
        dA_cav = (hd.mean(2) @ p["Wa"]).reshape(B, C, NB_ACT, 2)
        A_b = p["A0"][None] + dA_glob + dA_cav

        # SNAP phases on 0..NU-1
        U0 = p["u0"] + hd @ p["P0"]
        U1 = p["u1"] + hd @ p["P1"]
        z = (jnp.einsum("bcjm,jf->bcfm", U1, BT) + jnp.einsum("bcjm,jf->bcfm", U0, 1.0 - BT)) / K_cav
        ph_free = jnp.transpose(jnp.tanh(z + p["fq"]) @ p["Wo"], (0, 1, 3, 2))  # (B, C, NB_ACT, NU)
        ph_lin = jnp.einsum("bcjl,jf->bclf", hd @ p["Ws"], BT)
        TH_b = p["TH0"][None] + ph_free + ph_lin
        return A_b, TH_b, dg

    def loss_gnn(th, G, Cn_rel, Succ):
        p = unr(th)
        A_b, TH_b, dg = forward_gnn(p, G)
        p_succ, p_leak = SIM_BATCH_MAIN(A_b, TH_b, dg, Cn_rel, Succ)
        nl = -jnp.log(p_succ + 1e-12)
        out = jnp.mean(nl)
        if a.tailw > 0:
            k = max(1, nl.shape[0] // 10)
            out = out + a.tailw * jnp.mean(jax.lax.top_k(nl, k)[0])
        if a.leakpen > 0:
            out = out + a.leakpen * jnp.mean(p_leak)
        if a.wd > 0:
            reg = sum(jnp.sum(p[k_] ** 2) for k_ in ("G2_a", "G2_g", "Wa", "Ws", "Wo"))
            out = out + a.wd * reg
        return out

    vg_gnn = jax.jit(jax.value_and_grad(loss_gnn))
    eval_gnn_main = jax.jit(lambda th, G, Cn_rel, Succ: SIM_BATCH_MAIN(*forward_gnn(unr(th), G), Cn_rel, Succ))
    eval_gnn_chk = jax.jit(lambda th, G, Cn_rel, Succ: SIM_BATCH_CHK(*forward_gnn(unr(th), G), Cn_rel, Succ))

    B_tr = D_tr["B"]; ch = min(a.chunk, B_tr)
    parts = [
        ({k_: v_[i:i+ch] for k_, v_ in D_tr["G"].items()}, D_tr["Cn_rel"][i:i+ch], D_tr["Succ"][i:i+ch], (min(i+ch, B_tr) - i) / float(B_tr))
        for i in range(0, B_tr, ch)
    ]

    best_val = {"S": np.inf, "th": np.asarray(th0, np.float64), "it": 0}
    it_cnt = [0]

    def eval_val(x_vec):
        thj = jnp.asarray(x_vec, jnp.float32)
        p_s, p_l = eval_gnn_main(thj, D_va["G"], D_va["Cn_rel"], D_va["Succ"])
        return eval_metrics(p_s, p_l)

    m0 = eval_val(th0)
    best_val.update(S=m0["S"], th=np.asarray(th0, np.float64), it=0)
    log(f"GNN step 0 val med/B90 = {m0['median_inv_p']:.3f}/{m0['B90_inv_p']:.3f} (S={m0['S']:.4f})")

    def obj_gnn(x_vec):
        thj = jnp.asarray(x_vec, jnp.float32)
        tot_v = 0.0; tot_g = np.zeros_like(x_vec, dtype=np.float64)
        for g_p, c_p, s_p, w in parts:
            v, g = vg_gnn(thj, g_p, c_p, s_p)
            tot_v += float(v) * w
            tot_g += np.asarray(g, np.float64) * w
        return tot_v, tot_g

    def cb_gnn(xk):
        it_cnt[0] += 1
        if it_cnt[0] % 15 == 0:
            mv = eval_val(xk)
            if mv["S"] < best_val["S"]:
                best_val.update(S=mv["S"], th=np.array(xk, copy=True), it=it_cnt[0])
            log(f"  GNN it {it_cnt[0]} val med/B90 = {mv['median_inv_p']:.3f}/{mv['B90_inv_p']:.3f} S={mv['S']:.4f} (best {best_val['S']:.4f} @ it {best_val['it']})")

    r = minimize(obj_gnn, np.asarray(th0, np.float64), jac=True, method="L-BFGS-B", callback=cb_gnn,
                 options={"maxiter": a.maxiter, "maxfun": int(1.5 * a.maxiter)})
    mv_end = eval_val(r.x)
    if mv_end["S"] < best_val["S"]:
        best_val.update(S=mv_end["S"], th=np.array(r.x, copy=True), it=it_cnt[0])

    th_best = jnp.asarray(best_val["th"], jnp.float32)
    m_val = eval_metrics(*eval_gnn_main(th_best, D_va["G"], D_va["Cn_rel"], D_va["Succ"]))
    p_te, lk_te = eval_gnn_main(th_best, D_te["G"], D_te["Cn_rel"], D_te["Succ"])
    m_test = eval_metrics(p_te, lk_te)
    p_chk, lk_chk = eval_gnn_chk(th_best, D_te["G"], D_te["Cn_rel"], D_te["Succ"])
    m_chk = eval_metrics(p_chk, lk_chk)

    # Also check RR4 if requested
    if a.relabel == "rr4":
        ps_r, lks_r = [], []
        for r_idx in range(4):
            D_r = prepare_split("test", relabel_mode="rr4", r_idx=r_idx)
            pr_i, lk_i = eval_gnn_main(th_best, D_r["G"], D_r["Cn_rel"], D_r["Succ"])
            ps_r.append(np.asarray(pr_i)); lks_r.append(np.asarray(lk_i))
        m_rr = eval_metrics(np.mean(ps_r, axis=0), np.mean(lks_r, axis=0))
        m_best4 = eval_metrics(np.max(ps_r, axis=0), np.mean(lks_r, axis=0))
        m_test["rr4_med"] = m_rr["median_inv_p"]
        m_test["rr4_B90"] = m_rr["B90_inv_p"]
        m_test["best4_med"] = m_best4["median_inv_p"]
        m_test["best4_B90"] = m_best4["B90_inv_p"]

    res = {
        "args": vars(a), "ws_ref": WS_REF, "best_it": best_val["it"],
        "val": m_val, "test": m_test, f"test_NF{NF_CHK}": m_chk,
        "sec": time.time() - t0, "theta": best_val["th"].tolist()
    }
    json.dump(res, open(OUT, "w"))
    log(f"DONE GNN {a.tag}: val {m_val['median_inv_p']:.3f}/{m_val['B90_inv_p']:.3f} | test {m_test['median_inv_p']:.3f}/{m_test['B90_inv_p']:.3f} | NF{NF_CHK} {m_chk['median_inv_p']:.3f}/{m_chk['B90_inv_p']:.3f} leak={m_chk['mean_leak']:.4f}")
    return

# ---------------- Main Shared-Parameter L-BFGS-B Pipeline ----------------
def main():
    t0 = time.time()
    if a.init_from and a.gnn:
        J = json.load(open(a.init_from))
        x_base = np.array(J["x_best"], dtype=np.float32)
        run_gnn_mode(x_base)
        return

    train_rmode = "bpdec" if a.relabel == "rr4" else a.relabel
    D_scr = prepare_split("train", max_count=min(a.nscreen, a.ntrain), relabel_mode=train_rmode)
    D_tr = prepare_split("train", max_count=a.ntrain, relabel_mode=train_rmode)
    D_va = prepare_split("val", relabel_mode=train_rmode)

    obj_scr = make_chunked_obj(D_scr)
    obj_tr = make_chunked_obj(D_tr)

    rng = np.random.default_rng(1000 * n + a.seed)
    inits = [rand_shared_init(rng, i) for i in range(a.nstarts)]
    if a.init_from and os.path.exists(a.init_from):
        J = json.load(open(a.init_from))
        inits[0] = np.array(J["x_best"], dtype=np.float32)

    log(f"Start n={n} sched={a.sched} order={a.order} gate={a.gate} relabel={a.relabel} asym={a.asym} ntrain={a.ntrain} NF={NF}->{NF_CHK} params={inits[0].size}")
    screen_res = []
    for i, x0 in enumerate(inits):
        r = minimize(obj_scr, np.asarray(x0, np.float64), jac=True, method="L-BFGS-B",
                     options={"maxiter": a.screen_it, "maxfun": int(1.5 * a.screen_it)})
        screen_res.append({"i": i, "loss": float(r.fun), "nit": int(r.nit), "x": r.x})
        if (i + 1) % 4 == 0 or i == len(inits) - 1:
            best_s = min(s["loss"] for s in screen_res)
            log(f"  screen {i+1}/{len(inits)}: best loss {best_s:.4f} ({time.time()-t0:.1f}s)")

    top_scr = sorted(screen_res, key=lambda s: s["loss"])[:a.refine_top]
    refine_res = []
    for rank, s in enumerate(top_scr):
        t1 = time.time()
        r = minimize(obj_tr, np.asarray(s["x"], np.float64), jac=True, method="L-BFGS-B",
                     options={"maxiter": a.maxiter, "maxfun": int(1.5 * a.maxiter)})
        # Evaluate on validation set
        p_va, lk_va = EVAL_MAIN(jnp.asarray(r.x, jnp.float32), D_va["Cn_rel"], D_va["Succ"])
        m_va = eval_metrics(p_va, lk_va)
        refine_res.append({
            "id": s["i"], "loss_scr": s["loss"], "loss_train": float(r.fun),
            "nit": int(r.nit), "sec": time.time() - t1,
            "val": m_va, "x": r.x.tolist()
        })
        log(f"  refine #{rank} (init {s['i']}): train loss {r.fun:.4f} ({r.nit} it, {time.time()-t1:.1f}s) | val med/B90 = {m_va['median_inv_p']:.3f}/{m_va['B90_inv_p']:.3f} S={m_va['S']:.4f}")

    # Pick winner by validation S = (log med + log B90)/2
    winner = min(refine_res, key=lambda r: r["val"]["S"])
    x_best = np.array(winner["x"], dtype=np.float32)

    m_train, _ = evaluate_split(x_best, "train", max_count=min(2000, a.ntrain))
    m_val, _ = evaluate_split(x_best, "val")
    m_test, p_test = evaluate_split(x_best, "test", use_chk=False)
    m_test_chk, _ = evaluate_split(x_best, "test", use_chk=True)

    unp = unpack_shared(jnp.asarray(x_best))
    extra = {"g": np.asarray(unp["g"]).tolist()}
    if a.gate == "sd":
        A_np = np.asarray(unp["A"])
        extra["alpha_abs"] = np.hypot(A_np[..., 0], A_np[..., 1]).tolist()

    out = {
        "args": vars(a),
        "ws_ref": WS_REF,
        "NF": NF, "NF_chk": NF_CHK, "NU": NU,
        "winner_init": winner["id"],
        "refine": [{k: v for k, v in r.items() if k != "x"} for r in refine_res],
        "train": m_train,
        "val": m_val,
        "test": m_test,
        f"test_NF{NF_CHK}": m_test_chk,
        "x_best": x_best.tolist(),
        "p_test": p_test.tolist(),
        "sec": time.time() - t0,
        **extra
    }
    json.dump(out, open(OUT, "w"))
    log(f"DONE {a.tag}: test med/B90 = {m_test['median_inv_p']:.3f}/{m_test['B90_inv_p']:.3f} (NF{NF_CHK}: {m_test_chk['median_inv_p']:.3f}/{m_test_chk['B90_inv_p']:.3f}, leak={m_test_chk['mean_leak']:.4f}) vs WS tries {WS_REF['tries_med']:.2f}/{WS_REF['tries_b90']:.2f} ({time.time()-t0:.1f}s)")

    if a.gnn:
        run_gnn_mode(x_best)

if __name__ == "__main__":
    main()
