"""Rigorous quantitative wavefunction diagnostic for 16-level cavity-chain ansatz on 4-SAT (clean16) and 8-SAT (clean16_8sat).

Compares across n = 8, 12, 16, 20 (100 unseen test formulas per n):
  1. All-to-all QAOA L4 (Q) and Global Cost L4 (G)
  2. Pure Chain-Projected QAOA L4 (QC_block and QC_r0, no neural network)
  3. Native ckpt_110 (N_full: r=0..3 and R=4 round-robin) + component ablations:
     - N_pv_only:     Proj_pv(E_w) only (dS=0, dB=0)
     - N_corr_only:   Proj_pv(E_w) + corr(dS, dB) (grover=0)
     - N_grover_only: Proj_pv(E_w) + grover(dS) (corr=0)
     - N_unif_dSdB:   Proj_unif(E_w) + dS + dB (pv=0.5)
     - N_unif_w:      Proj_unif(E_w) only (pv=0.5, dS=0, dB=0)

Measures all 5 diagnostic families:
  1. Smearing vs. Locking into Wrong States (H(P), N_eff, P_top1/10/100 wrong vs P_sol, Easy/Median/Hard B90 split)
  2. True Energy vs. Projected Energy of Top Wrong States ("Ghost Minima" counts N_ghost and z-scores)
  3. Hamming Distance Structure (d_H to solutions, pv, local optimistic min, argmin Proj(E); within- vs across-cavity errors)
  4. Layer-by-Layer Evolution (l = 0..4) & Ablation of ckpt_110 components
  5. Across R = 4 Placements (min-cut r=0 vs random r=1..3, spread, and whether placements lock into same or different states)
"""
import argparse, importlib.util, json, os, sys, time
import numpy as np
import jax, jax.numpy as jnp
from jax.flatten_util import ravel_pytree

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (HERE, ROOT, os.path.join(ROOT, "v3"), os.path.join(ROOT, "v3", "native"), os.path.join(ROOT, "v3", "openevolve", "net")):
    if p not in sys.path:
        sys.path.insert(0, p)

import compiler as CP, nsim as NS, frozen_circuit as FC, arch_nets as AN
CP.LAYOUTS[20] = [4, 4, 4, 4, 4]

# Precompute 16-bit popcount and 4-bit nibble popcount
PC16 = np.array([bin(x).count("1") for x in range(65536)], dtype=np.int8)
PC4 = np.array([bin(x).count("1") for x in range(16)], dtype=np.int8)
I16 = np.arange(16)[:, None]
J4 = np.arange(4)[None]
BT4 = jnp.asarray(((I16 >> (3 - J4)) & 1).astype(np.float32))  # (16, 4)

def popcount_arr(x):
    x = np.asarray(x, dtype=np.int32)
    return PC16[x & 0xFFFF] + PC16[(x >> 16) & 0xFFFF]

def ham_to_sols_np(sols, n):
    ids = np.arange(1 << n, dtype=np.int32)
    out = np.full(1 << n, n, dtype=np.int8)
    for s in sols[:64]:
        d = popcount_arr(ids ^ int(s))
        out = np.minimum(out, d)
    return out

def cav_err_stats_np(x_arr, sols, n):
    """For each state in x_arr (K,), find nearest solution in sols ( by total Hamming distance)
    and return (ham (K,), cav_wrong (K,), bits_per_wrong_cav (K,))."""
    C = n // 4
    x_arr = np.asarray(x_arr, dtype=np.int32)
    best_ham = np.full(len(x_arr), n + 1, dtype=np.int32)
    best_cw = np.zeros(len(x_arr), dtype=np.int32)
    for s in sols[:64]:
        diff = x_arr ^ int(s)
        h = popcount_arr(diff).astype(np.int32)
        cw = np.zeros(len(x_arr), dtype=np.int32)
        for c in range(C):
            nib = (diff >> (4 * (C - 1 - c))) & 0xF
            cw += (nib != 0).astype(np.int32)
        upd = h < best_ham
        best_ham = np.where(upd, h, best_ham)
        best_cw = np.where(upd, cw, best_cw)
    bpw = np.where(best_cw > 0, best_ham / np.maximum(best_cw, 1), 0.0)
    return best_ham, best_cw, bpw

def chain_to_var_int(x_chain, perm, n):
    x_chain = np.asarray(x_chain, dtype=np.int64)
    x_var = np.zeros_like(x_chain, dtype=np.int64)
    for q, v in enumerate(perm):
        x_var |= ((x_chain >> (n - 1 - q)) & 1) << (n - 1 - int(v))
    return x_var

def var_to_chain_int(x_var, perm, n):
    x_var = np.asarray(x_var, dtype=np.int64)
    x_chain = np.zeros_like(x_var, dtype=np.int64)
    for q, v in enumerate(perm):
        x_chain |= ((x_var >> (n - 1 - int(v))) & 1) << (n - 1 - q)
    return x_chain

def load_gcost_params(json_path):
    J = json.load(open(json_path))
    L, H, R, FV = 4, 64, 3, 7
    NPL = 6; NBASE = L * NPL + 4
    rng = np.random.default_rng(1000)
    W = lambda i, o, s=1.0: jnp.asarray(rng.normal(0, s / np.sqrt(i), (i, o)).astype(np.float32))
    p0 = AN.init_backbone(rng, "mp", H, R, FV)
    p0["G1"] = W(2 * H, H); p0["g1"] = jnp.zeros(H); p0["G2"] = jnp.zeros((H, NBASE)); p0["Wc"] = jnp.zeros((H, L)); p0["base"] = jnp.zeros(NBASE)
    _, unr = ravel_pytree(p0)
    return unr(jnp.asarray(J["theta_best"], jnp.float32))

def gcost_forward_params(pG, V, SG, n):
    """Run GNN backbone of Global Cost L4 on batch of formulas (V, SG) -> (bl (F,4), gl (F,4), wG (F,m,4))."""
    F, m, _ = V.shape
    L, H, FV, NPL = 4, 64, 7, 6
    Ip = np.zeros((F, m, n), np.float32)
    In = np.zeros((F, m, n), np.float32)
    fi = np.arange(F)[:, None, None]; ci = np.arange(m)[None, :, None]
    np.add.at(Ip, (fi, ci, V), (SG > 0).astype(np.float32))
    np.add.at(In, (fi, ci, V), (SG < 0).astype(np.float32))
    Ip = (Ip > 0).astype(np.float32); In = (In > 0).astype(np.float32)
    mr = np.maximum((Ip.sum(-1) + In.sum(-1) > 0).sum(1, keepdims=True), 1.0)
    fv = np.zeros((F, n, FV), np.float32)
    fv[:, :, 0] = 1.0
    fv[:, :, 1] = Ip.sum(1) / (2.0 * mr / n)
    fv[:, :, 2] = In.sum(1) / (2.0 * mr / n)
    deg = np.maximum(Ip.sum(1) + In.sum(1), 1.0)
    G = dict(Ip=jnp.asarray(Ip), In=jnp.asarray(In), fv=jnp.asarray(fv), deg=jnp.asarray(deg))
    hv, hc = AN.run_backbone(pG, G, "mp", H)
    g_ = jnp.concatenate([hv.mean(1), hc.mean(1)], -1)
    bv = pG["base"][None] + jnp.tanh(g_ @ pG["G1"] + pG["g1"]) @ pG["G2"]
    X = bv[:, : L * NPL].reshape(F, L, NPL)
    bl, gl = X[:, :, 4], X[:, :, 5]
    wG = hc @ pG["Wc"]
    return np.asarray(bl), np.asarray(gl), np.asarray(wG)

# Per-n JIT-compiled helpers
_JIT_CACHE = {}
def get_jit_fns(n):
    if n in _JIT_CACHE:
        return _JIT_CACHE[n]
    ks = CP.LAYOUTS[n]
    C = len(ks)
    mask_chain = jnp.asarray(CP.rep_mask(ks, "chain").astype(np.float32))
    I_cav = [jnp.asarray(x) for x in FC.cav_idx(n)]

    @jax.jit
    def contract_perm(cpos, cneg, perm, W):
        # cpos, cneg: (m, n), perm: (n,), W: (m, K_ch) -> (K_ch, 2^n)
        cp = cpos[:, perm].reshape(-1, C, 4)
        cn = cneg[:, perm].reshape(-1, C, 4)
        sat = jnp.einsum("mcj,ij->mci", cp, BT4) + jnp.einsum("mcj,ij->mci", cn, 1.0 - BT4)
        Lc = (sat < 0.5).astype(jnp.float32)
        cl = C // 2
        Ll = Lc[:, 0]
        for c in range(1, cl):
            Ll = (Ll[:, :, None] * Lc[:, c, None, :]).reshape(Lc.shape[0], -1)
        Lr = Lc[:, cl]
        for c in range(cl + 1, C):
            Lr = (Lr[:, :, None] * Lc[:, c, None, :]).reshape(Lc.shape[0], -1)
        return jnp.einsum("mp,mq,mk->kpq", Ll, Lr, W).reshape(W.shape[1], 1 << n)

    @jax.jit
    def proj_unif(E_batch):
        # E_batch: (B, 2^n) -> uniform chain projection (B, 2^n)
        return FC.fwht_j(FC.fwht_j(E_batch, n) * mask_chain[None, :], n) / float(1 << n)

    @jax.jit
    def proj_pv(E_batch, pvc_batch):
        # E_batch: (B, 2^n), pvc_batch: (B, n) -> product-measure chain projection (B, 2^n)
        return FC.project_product_j(E_batch, pvc_batch, n, ks)

    @jax.jit
    def add_tables(dS_b, dB_b):
        # dS_b: (B, 4, C, 32), dB_b: (B, 4, C-1, 32, 32) -> (B, 4, 2^n)
        B = dS_b.shape[0]
        out = jnp.zeros((B, 4, 1 << n), jnp.float32)
        for l in range(4):
            tl = jnp.zeros((B, 1 << n), jnp.float32)
            for c in range(C):
                tl = tl + jnp.take(dS_b[:, l, c], I_cav[c], axis=1)
            for a in range(C - 1):
                tl = tl + jnp.take(dB_b[:, l, a].reshape(B, -1), I_cav[a] * 32 + I_cav[a + 1], axis=1)
            out = out.at[:, l].set(tl)
        return out

    @jax.jit
    def sim_and_metrics(Phi, beta, E_ord, PE_ord, H_ord, ord_idx):
        """Simulate B circuits for 4 layers.
        Phi: (B, 4, 2^n), beta: (B, 4)
        E_ord: (5, 2^n) int16, PE_ord: (5, 2^n) float32 (normalized z-score), H_ord: (5, 2^n) int8
        ord_idx: (B,) int32 in 0..4 mapping each circuit b to its variable ordering (0=block, 1..4=r0..r3).
        """
        B = Phi.shape[0]
        E_b = E_ord[ord_idx]
        PE_b = PE_ord[ord_idx]
        H_b = H_ord[ord_idx]
        S_b = (E_b == 0).astype(jnp.float32)

        psi = NS.uniform_psi0(ks, B)
        p_sol_l, neff_l, ent_l, mE_l, mPE_l, mH_l, top10_l = [], [], [], [], [], [], []
        for l in range(4):
            psi = psi * jnp.exp(-1j * Phi[:, l].reshape(psi.shape)).astype(jnp.complex64)
            for c, k in enumerate(ks):
                psi = NS.apply_cav(FC.MIXK[k](beta[:, l]), psi, c, C)
            P = jnp.abs(psi.reshape(B, -1)) ** 2
            p_sol_l.append(jnp.sum(P * S_b, axis=-1))
            neff_l.append(1.0 / jnp.sum(P * P, axis=-1))
            ent_l.append(-jnp.sum(jnp.where(P > 1e-15, P * jnp.log2(jnp.maximum(P, 1e-15)), 0.0), axis=-1))
            mE_l.append(jnp.sum(P * E_b, axis=-1))
            mPE_l.append(jnp.sum(P * PE_b, axis=-1))
            mH_l.append(jnp.sum(P * H_b, axis=-1))
            Pw = jnp.where(E_b > 0, P, -1.0)
            if l < 3:
                tv10, _ = jax.lax.top_k(Pw, 10)
                top10_l.append(jnp.sum(jnp.maximum(tv10, 0.0), axis=-1))
            else:
                tv100, ti100 = jax.lax.top_k(Pw, 100)
                top10_l.append(jnp.sum(jnp.maximum(tv100[:, :10], 0.0), axis=-1))
                tv_all, _ = jax.lax.top_k(P, 100)

        e_shells = jnp.stack([jnp.sum(P * (E_b == k), axis=-1) for k in range(5)] + [jnp.sum(P * (E_b >= 5), axis=-1)], axis=-1)
        h_shells = jnp.stack([
            jnp.sum(P * (H_b == 0), axis=-1),
            jnp.sum(P * ((H_b >= 1) & (H_b <= 2)), axis=-1),
            jnp.sum(P * ((H_b >= 3) & (H_b <= 5)), axis=-1),
            jnp.sum(P * ((H_b >= 6) & (H_b <= 9)), axis=-1),
            jnp.sum(P * (H_b >= 10), axis=-1),
        ], axis=-1)
        return (jnp.stack(p_sol_l), jnp.stack(neff_l), jnp.stack(ent_l),
                jnp.stack(mE_l), jnp.stack(mPE_l), jnp.stack(mH_l), jnp.stack(top10_l),
                e_shells, h_shells, tv100, ti100, tv_all)

    _JIT_CACHE[n] = (contract_perm, proj_unif, proj_pv, add_tables, sim_and_metrics)
    return _JIT_CACHE[n]

def ham_and_cw_to_sols_np(sols, n):
    C = n // 4
    ids = np.arange(1 << n, dtype=np.int32)
    best_h = np.full(1 << n, n + 1, dtype=np.int8)
    best_cw = np.full(1 << n, C + 1, dtype=np.int8)
    for s in sols[:32]:
        diff = ids ^ int(s)
        h = PC16[diff & 0xFFFF] + PC16[(diff >> 16) & 0xFFFF]
        cw = np.zeros(1 << n, dtype=np.int8)
        for c in range(C):
            cw += (((diff >> (4 * (C - 1 - c))) & 0xF) != 0).astype(np.int8)
        upd = h < best_h
        best_h = np.where(upd, h, best_h)
        best_cw = np.where(upd, cw, best_cw)
    return best_h, best_cw

# Update get_jit_fns to also compute cw_shells (0, 1, 2, >=3 cavities wrong)
_JIT_CACHE.clear()
def get_jit_fns(n):
    if n in _JIT_CACHE:
        return _JIT_CACHE[n]
    ks = CP.LAYOUTS[n]
    C = len(ks)
    mask_chain = jnp.asarray(CP.rep_mask(ks, "chain").astype(np.float32))
    I_cav = [jnp.asarray(x) for x in FC.cav_idx(n)]

    @jax.jit
    def contract_perm(cpos, cneg, perm, W):
        cp = cpos[:, perm].reshape(-1, C, 4)
        cn = cneg[:, perm].reshape(-1, C, 4)
        sat = jnp.einsum("mcj,ij->mci", cp, BT4) + jnp.einsum("mcj,ij->mci", cn, 1.0 - BT4)
        Lc = (sat < 0.5).astype(jnp.float32)
        cl = C // 2
        Ll = Lc[:, 0]
        for c in range(1, cl):
            Ll = (Ll[:, :, None] * Lc[:, c, None, :]).reshape(Lc.shape[0], -1)
        Lr = Lc[:, cl]
        for c in range(cl + 1, C):
            Lr = (Lr[:, :, None] * Lc[:, c, None, :]).reshape(Lc.shape[0], -1)
        return jnp.einsum("mp,mq,mk->kpq", Ll, Lr, W).reshape(W.shape[1], 1 << n)

    @jax.jit
    def proj_unif(E_batch):
        return FC.fwht_j(FC.fwht_j(E_batch, n) * mask_chain[None, :], n) / float(1 << n)

    @jax.jit
    def proj_pv(E_batch, pvc_batch):
        return FC.project_product_j(E_batch, pvc_batch, n, ks)

    @jax.jit
    def add_tables(dS_b, dB_b):
        B = dS_b.shape[0]
        out = jnp.zeros((B, 4, 1 << n), jnp.float32)
        for l in range(4):
            tl = jnp.zeros((B, 1 << n), jnp.float32)
            for c in range(C):
                tl = tl + jnp.take(dS_b[:, l, c], I_cav[c], axis=1)
            for a in range(C - 1):
                tl = tl + jnp.take(dB_b[:, l, a].reshape(B, -1), I_cav[a] * 32 + I_cav[a + 1], axis=1)
            out = out.at[:, l].set(tl)
        return out

    @jax.jit
    def sim_and_metrics(Phi, beta, E_ord, PE_ord, H_ord, CW_ord, ord_idx):
        B = Phi.shape[0]
        E_b = E_ord[ord_idx]
        PE_b = PE_ord[ord_idx]
        H_b = H_ord[ord_idx]
        CW_b = CW_ord[ord_idx]
        S_b = (E_b == 0).astype(jnp.float32)

        psi = NS.uniform_psi0(ks, B)
        p_sol_l, neff_l, ent_l, mE_l, mPE_l, mH_l, top10_l = [], [], [], [], [], [], []
        for l in range(4):
            psi = psi * jnp.exp(-1j * Phi[:, l].reshape(psi.shape)).astype(jnp.complex64)
            for c, k in enumerate(ks):
                psi = NS.apply_cav(FC.MIXK[k](beta[:, l]), psi, c, C)
            P = jnp.abs(psi.reshape(B, -1)) ** 2
            p_sol_l.append(jnp.sum(P * S_b, axis=-1))
            neff_l.append(1.0 / jnp.sum(P * P, axis=-1))
            ent_l.append(-jnp.sum(jnp.where(P > 1e-15, P * jnp.log2(jnp.maximum(P, 1e-15)), 0.0), axis=-1))
            mE_l.append(jnp.sum(P * E_b, axis=-1))
            mPE_l.append(jnp.sum(P * PE_b, axis=-1))
            mH_l.append(jnp.sum(P * H_b, axis=-1))
            Pw = jnp.where(E_b > 0, P, -1.0)
            if l < 3:
                tv10, _ = jax.lax.top_k(Pw, 10)
                top10_l.append(jnp.sum(jnp.maximum(tv10, 0.0), axis=-1))
            else:
                tv100, ti100 = jax.lax.top_k(Pw, 100)
                top10_l.append(jnp.sum(jnp.maximum(tv100[:, :10], 0.0), axis=-1))
                tv_all, _ = jax.lax.top_k(P, 100)

        e_shells = jnp.stack([jnp.sum(P * (E_b == k), axis=-1) for k in range(5)] + [jnp.sum(P * (E_b >= 5), axis=-1)], axis=-1)
        h_shells = jnp.stack([
            jnp.sum(P * (H_b == 0), axis=-1),
            jnp.sum(P * ((H_b >= 1) & (H_b <= 2)), axis=-1),
            jnp.sum(P * ((H_b >= 3) & (H_b <= 5)), axis=-1),
            jnp.sum(P * ((H_b >= 6) & (H_b <= 9)), axis=-1),
            jnp.sum(P * (H_b >= 10), axis=-1),
        ], axis=-1)
        cw_shells = jnp.stack([
            jnp.sum(P * (CW_b == 0), axis=-1),
            jnp.sum(P * (CW_b == 1), axis=-1),
            jnp.sum(P * (CW_b == 2), axis=-1),
            jnp.sum(P * (CW_b >= 3), axis=-1),
        ], axis=-1)
        return (jnp.stack(p_sol_l), jnp.stack(neff_l), jnp.stack(ent_l),
                jnp.stack(mE_l), jnp.stack(mPE_l), jnp.stack(mH_l), jnp.stack(top10_l),
                e_shells, h_shells, cw_shells, tv100, ti100, tv_all)

    _JIT_CACHE[n] = (contract_perm, proj_unif, proj_pv, add_tables, sim_and_metrics)
    return _JIT_CACHE[n]

CIRCUIT_NAMES = [
    "Q",            # 0: All-to-all QAOA L4 (block order)
    "G",            # 1: Global Cost L4 (block order)
    "QC_block",     # 2: Pure Chain-Projected QAOA L4 (block order)
    "QC_r0",        # 3: Pure Chain-Projected QAOA L4 (r=0 min-cut)
    "QC_r1",        # 4: Pure Chain-Projected QAOA L4 (r=1 random)
    "QC_r2",        # 5: Pure Chain-Projected QAOA L4 (r=2 random)
    "QC_r3",        # 6: Pure Chain-Projected QAOA L4 (r=3 random)
    "N_r0",         # 7: Native ckpt_110 full (r=0 min-cut)
    "N_r1",         # 8: Native ckpt_110 full (r=1 random)
    "N_r2",         # 9: Native ckpt_110 full (r=2 random)
    "N_r3",         # 10: Native ckpt_110 full (r=3 random)
    "N_pv_r0",      # 11: ckpt_110 Proj_pv(Ew) only (r=0)
    "N_pv_r1",      # 12: ckpt_110 Proj_pv(Ew) only (r=1)
    "N_pv_r2",      # 13: ckpt_110 Proj_pv(Ew) only (r=2)
    "N_pv_r3",      # 14: ckpt_110 Proj_pv(Ew) only (r=3)
    "N_corr_r0",    # 15: ckpt_110 Proj_pv(Ew) + corr(dS,dB) (r=0)
    "N_grover_r0",  # 16: ckpt_110 Proj_pv(Ew) + grover(dS) (r=0)
    "N_unif_dSdB",  # 17: ckpt_110 Proj_unif(Ew) + dS + dB (r=0)
    "N_unif_w",     # 18: ckpt_110 Proj_unif(Ew) only (r=0)
]
ORD_IDX = jnp.array([0, 0, 0, 1, 2, 3, 4, 1, 2, 3, 4, 1, 2, 3, 4, 1, 1, 1, 1], dtype=jnp.int32)

def load_dataset_inputs(sat_type, n):
    """Load V, SG, PERM, seeds, and model parameters for 4sat or 8sat at size n."""
    ks = CP.LAYOUTS[n]
    C = len(ks)
    if sat_type == "4sat":
        res_dir = os.path.join(ROOT, "results", "clean16")
        dc_file = os.path.join(ROOT, "v3", "openevolve", "runs", "datacache", f"n{n}_c100_b52000000000_R4.npz")
        z = np.load(dc_file, allow_pickle=True)
        V, SG, PERM, seeds = z["V"], z["SG"], z["PERM"], z["seed"]
        m_fixed = int(z["m"])
        nm = float(n) / float(m_fixed)
        x_Q = np.array(json.load(open(os.path.join(res_dir, "Q_s0.json")))["x"], dtype=np.float32)
        x_QC = np.array(json.load(open(os.path.join(res_dir, "discard", "Q_s0_projchain_mistake.json")))["x"], dtype=np.float32)
        pG = load_gcost_params(os.path.join(res_dir, "G_s2.json"))
        spec = importlib.util.spec_from_file_location("prog4", os.path.join(ROOT, "clean16", "native_prog.py"))
        modN = importlib.util.module_from_spec(spec); spec.loader.exec_module(modN)
        zp = np.load(os.path.join(res_dir, "N_s2_pf.npz"))
        pfN = {k: jnp.asarray(zp[k]) for k in zp.files}
        pfN_nog = dict(pfN)
        pfN_nog["gopt"] = jnp.zeros_like(pfN["gopt"])
        pfN_nog["gS"] = jnp.zeros_like(pfN["gS"])
        pfN_nog["gfin"] = jnp.zeros_like(pfN["gfin"])
        place = dict(cav=FC.cavities(n, PERM.reshape(-1, n)).reshape(100, 4, C, 5), ks=tuple(ks))
        fx = modN.features(n, V, SG, m_fixed)
        out_full = {k: np.asarray(v) for k, v in modN.apply(pfN, *fx, place=place).items()}
        out_nog = {k: np.asarray(v) for k, v in modN.apply(pfN_nog, *fx, place=place).items()}
        # Also compute local_tables E_loc and grover mark for each formula/placement
        xe, xo, inc_j, sgn_j, V_j = fx
        e = jnp.tanh(xe @ pfN["We"]); o = jnp.tanh(xo @ pfN["Wo"]); vd = inc_j.sum(1)[..., None] + 1.0
        for _ in range(4):
            hc = jnp.tanh((jnp.einsum("fmn,fnh->fmh", inc_j, e) @ pfN["Wce"] + jnp.einsum("fmn,fnh->fmh", sgn_j, o) @ pfN["Wco"]) / 4.0)
            e = jnp.tanh(e + (jnp.einsum("fmn,fmh->fnh", inc_j, hc) / vd) @ pfN["Wve"])
            o = jnp.tanh(o + (jnp.einsum("fmn,fmh->fnh", sgn_j, hc) / vd) @ pfN["Wvo"])
        hc = jnp.tanh(jnp.concatenate([jnp.einsum("fmn,fnh->fmh", inc_j, e), jnp.einsum("fmn,fnh->fmh", sgn_j, o)], -1) / 4.0 @ pfN["Wc"])
        E1, hop_pen, valid = modN.local_tables(pfN, hc, V_j, sgn_j, place, n)
        E_loc = np.asarray(E1 + pfN["ghop"] * hop_pen)[..., :16]  # (100, 4, C, 16)
        r_rank = modN.soft_rank(E1 + pfN["ghop"] * hop_pen, valid)
        q_cap = modN.GROVER_QMAX * jax.nn.sigmoid(pfN["rq"])
        mark = np.asarray(jax.nn.sigmoid((q_cap - r_rank) / 0.03) * valid)[..., :16]
    else:
        res_dir = os.path.join(ROOT, "results", "clean16_8sat")
        J_N = json.load(open(os.path.join(res_dir, "N_s2.json")))
        seeds = np.array(J_N["test"][str(n)]["seeds"], dtype=np.int64)
        R_8 = 176.54; K_8 = 8; nm = 1.0 / R_8
        M_pad = int(np.ceil(R_8 * n + 8 * np.sqrt(R_8 * n)))
        V = np.zeros((100, M_pad, K_8), np.int32)
        SG = np.zeros((100, M_pad, K_8), np.int8)
        PERM = np.zeros((100, 4, n), np.int32)
        for i, s in enumerate(seeds):
            rng = np.random.default_rng(int(s)); m_i = int(rng.poisson(R_8 * n)); L_i = rng.integers(0, 2 * n, size=(m_i, K_8))
            V[i, :m_i] = L_i // 2; SG[i, :m_i] = 1 - 2 * (L_i % 2)
            if C <= 2:
                PERM[i, 0] = np.arange(n)
            else:
                PERM[i, 0] = np.load(os.path.join(ROOT, "results", "v3", "native", "perm_cache", f"{int(s)}.npy"))
            for r in range(1, 4):
                PERM[i, r] = np.random.default_rng(int(s) * 31 + r).permutation(n)
        x_Q = np.array(json.load(open(os.path.join(res_dir, "Q_s0.json")))["x"], dtype=np.float32)
        x_QC = np.array(json.load(open(os.path.join(res_dir, "QC_s0.json")))["x"], dtype=np.float32)
        pG = load_gcost_params(os.path.join(res_dir, "G_s0.json"))
        spec = importlib.util.spec_from_file_location("prog8", os.path.join(ROOT, "clean16_8sat", "native_prog.py"))
        modN = importlib.util.module_from_spec(spec); spec.loader.exec_module(modN)
        zp = np.load(os.path.join(res_dir, "N_s2_pf.npz"))
        pfN = {k: jnp.asarray(zp[k]) for k in zp.files}
        pfN_nog = dict(pfN)
        pfN_nog["gopt"] = jnp.zeros_like(pfN["gopt"])
        pfN_nog["gS"] = jnp.zeros_like(pfN["gS"])
        pfN_nog["gfin"] = jnp.zeros_like(pfN["gfin"])
        place = dict(cav=FC.cavities(n, PERM.reshape(-1, n)).reshape(100, 4, C, 5), ks=tuple(ks))
        out_full_l, out_nog_l, E_loc_l, mark_l = [], [], [], []
        cm_all = (SG != 0).any(-1).astype(np.float32)
        for s0 in range(0, 100, 10):
            s1 = min(100, s0 + 10)
            pl_s = dict(cav=place["cav"][s0:s1], ks=place["ks"])
            fx_s = modN.features(n, V[s0:s1], SG[s0:s1], M_pad)
            cm_s = jnp.asarray(cm_all[s0:s1])
            of_s = modN.apply(pfN, *fx_s, place=pl_s, cm=cm_s)
            on_s = modN.apply(pfN_nog, *fx_s, place=pl_s, cm=cm_s)
            out_full_l.append({k: np.asarray(v) for k, v in of_s.items()})
            out_nog_l.append({k: np.asarray(v) for k, v in on_s.items()})
            xe, xo, inc_j, sgn_j, V_j = fx_s
            K_f = 8.0; e = jnp.tanh(xe @ pfN["We"]); o = jnp.tanh(xo @ pfN["Wo"]); vd = inc_j.sum(1)[..., None] + 1.0
            for _ in range(4):
                hc = jnp.tanh((jnp.einsum("fmn,fnh->fmh", inc_j, e) @ pfN["Wce"] + jnp.einsum("fmn,fnh->fmh", sgn_j, o) @ pfN["Wco"]) / K_f)
                e = jnp.tanh(e + (jnp.einsum("fmn,fmh->fnh", inc_j, hc) / vd) @ pfN["Wve"])
                o = jnp.tanh(o + (jnp.einsum("fmn,fmh->fnh", sgn_j, hc) / vd) @ pfN["Wvo"])
            hc = jnp.tanh(jnp.concatenate([jnp.einsum("fmn,fnh->fmh", inc_j, e), jnp.einsum("fmn,fnh->fmh", sgn_j, o)], -1) / K_f @ pfN["Wc"])
            gr, _, _ = modN.geom(V_j, pl_s, n, cm=cm_s)
            mr_s = cm_s.sum(1)
            E1, hop_pen, valid = modN.local_tables(pfN, hc, V_j, inc_j, sgn_j, pl_s, n, gr[..., 3], mr_s)
            E_loc_s = E1 + pfN["ghop"] * hop_pen
            r_rank = modN.soft_rank(E_loc_s, valid)
            q_cap = modN.GROVER_QMAX * jax.nn.sigmoid(pfN["rq"])
            mark_s = jax.nn.sigmoid((q_cap - r_rank) / 0.03) * valid
            E_loc_l.append(np.asarray(E_loc_s)[..., :16])
            mark_l.append(np.asarray(mark_s)[..., :16])
        out_full = {k: np.concatenate([x[k] for x in out_full_l], axis=0) for k in out_full_l[0]}
        out_nog = {k: np.concatenate([x[k] for x in out_nog_l], axis=0) for k in out_nog_l[0]}
        E_loc = np.concatenate(E_loc_l, axis=0)
        mark = np.concatenate(mark_l, axis=0)

    # Also compute Global Cost L4 forward parameters in chunks of 10
    bl_G_l, gl_G_l, wG_l = [], [], []
    for s0 in range(0, 100, 10):
        s1 = min(100, s0 + 10)
        b_s, g_s, w_s = gcost_forward_params(pG, V[s0:s1], SG[s0:s1], n)
        bl_G_l.append(b_s); gl_G_l.append(g_s); wG_l.append(w_s)
    bl_G = np.concatenate(bl_G_l, axis=0)
    gl_G = np.concatenate(gl_G_l, axis=0)
    w_G = np.concatenate(wG_l, axis=0)

    return dict(V=V, SG=SG, PERM=PERM, seeds=seeds, nm=nm,
                x_Q=x_Q, x_QC=x_QC, bl_G=bl_G, gl_G=gl_G, w_G=w_G,
                out_full=out_full, out_nog=out_nog, E_loc=E_loc, mark=mark)

def analyze_n(sat_type, n, max_formulas=100):
    t0 = time.time()
    ks = CP.LAYOUTS[n]
    C = len(ks)
    N = 1 << n
    inp = load_dataset_inputs(sat_type, n)
    V, SG, PERM, seeds, nm = inp["V"], inp["SG"], inp["PERM"], inp["seeds"], inp["nm"]
    x_Q, x_QC = inp["x_Q"], inp["x_QC"]
    beta_Q, gamma_Q = jnp.asarray(x_Q[0::2]), jnp.asarray(x_Q[1::2])
    beta_QC, gamma_QC = jnp.asarray(x_QC[0::2]), jnp.asarray(x_QC[1::2])
    bl_G, gl_G, w_G = inp["bl_G"], inp["gl_G"], inp["w_G"]
    out_full, out_nog = inp["out_full"], inp["out_nog"]
    E_loc_all, mark_all = inp["E_loc"], inp["mark"]

    contract_perm, proj_unif, proj_pv, add_tables, sim_and_metrics = get_jit_fns(n)
    m_dim = V.shape[1]
    ci = np.arange(m_dim)[:, None]
    perm0 = jnp.arange(n, dtype=jnp.int32)

    formulas_out = []
    for i in range(min(max_formulas, len(seeds))):
        inc = np.zeros((m_dim, n), np.float32)
        sgn = np.zeros((m_dim, n), np.float32)
        np.add.at(inc, (ci, V[i]), (SG[i] != 0).astype(np.float32))
        np.add.at(sgn, (ci, V[i]), SG[i].astype(np.float32))
        cpos = jnp.asarray(0.5 * (inc + sgn))
        cneg = jnp.asarray(0.5 * (inc - sgn))
        cm = jnp.asarray((inc.sum(-1) > 0).astype(np.float32))
        m_real = int(cm.sum())

        gamma_N = jnp.asarray(out_full["gamma"][i])
        beta_N = jnp.asarray(out_full["beta"][i])
        w_N = jnp.asarray(out_full["w"][i])
        pv_i = np.asarray(out_full["pv"][i])
        dS_f = jnp.asarray(out_full["dS"][i])       # (4, 4, C, 32)
        dB_f = jnp.asarray(out_full["dB"][i])       # (4, 4, C-1, 32, 32)
        dS_c = jnp.asarray(out_nog["dS"][i, 0:1])   # (1, 4, C, 32)
        dB_c = jnp.asarray(out_nog["dB"][i, 0:1])   # (1, 4, C-1, 32, 32)
        dS_g = dS_f[0:1] - dS_c                     # (1, 4, C, 32)
        dB_z = jnp.zeros_like(dB_c)

        # Order 0 (block): E_var, Ew_G (4), Ew_N_var (4)
        W0 = jnp.concatenate([
            cm[:, None],
            cm[:, None] * nm * (jnp.asarray(gl_G[i])[None, :] + jnp.asarray(w_G[i])),
            cm[:, None] * nm * (gamma_N[None, :] + w_N),
        ], axis=1)  # (m, 9)
        res0 = contract_perm(cpos, cneg, perm0, W0)
        E_0 = jnp.round(res0[0]).astype(jnp.int16)
        Ew_G = res0[1:5]

        # Orders 1..4 (r = 0..3)
        Wr = jnp.concatenate([cm[:, None], cm[:, None] * nm * (gamma_N[None, :] + w_N)], axis=1)  # (m, 5)
        E_r_list, Ew_r_list = [], []
        for r in range(4):
            if np.array_equal(PERM[i, r], np.arange(n)):
                E_r_list.append(E_0)
                Ew_r_list.append(res0[5:9])
            else:
                res_r = contract_perm(cpos, cneg, jnp.asarray(PERM[i, r], jnp.int32), Wr)
                E_r_list.append(jnp.round(res_r[0]).astype(jnp.int16))
                Ew_r_list.append(res_r[1:5])
        E_ord = jnp.stack([E_0] + E_r_list, axis=0)      # (5, 2^n) int16
        Ew_r = jnp.stack(Ew_r_list, axis=0)              # (4, 4, 2^n) float32

        # Projections (fused single batch calls)
        PE_unif_all = proj_unif(jnp.concatenate([E_ord.astype(jnp.float32), Ew_r[0]], axis=0))  # (9, 2^n)
        PE_unif_ord = PE_unif_all[:5]
        PE_unif_Ew_r0 = PE_unif_all[5:]
        pe_mean = jnp.mean(PE_unif_ord, axis=-1, keepdims=True)
        pe_std = jnp.maximum(jnp.std(PE_unif_ord, axis=-1, keepdims=True), 1e-6)
        PE_ord_z = (PE_unif_ord - pe_mean) / pe_std

        pvc_r = jnp.asarray(np.stack([pv_i[PERM[i, r]] for r in range(4)], axis=0), jnp.float32)  # (4, n)
        PE_pv_all = proj_pv(
            jnp.concatenate([Ew_r.reshape(16, N), E_ord[1:5].astype(jnp.float32)], axis=0),
            jnp.concatenate([jnp.repeat(pvc_r, 4, axis=0), pvc_r], axis=0)
        )  # (20, 2^n)
        PE_pv_Ew = PE_pv_all[:16].reshape(4, 4, N)
        PE_pv_E_r = PE_pv_all[16:]

        # Tables dS + dB
        dS_cat = jnp.concatenate([dS_f, dS_c, dS_g], axis=0)  # (6, 4, C, 32)
        dB_cat = jnp.concatenate([dB_f, dB_c, dB_z], axis=0)  # (6, 4, C-1, 32, 32)
        T_all = add_tables(dS_cat, dB_cat)                    # (6, 4, 2^n)
        T_full = T_all[:4]
        T_corr_r0 = T_all[4]
        T_grov_r0 = T_all[5]

        # Build Phi (19, 4, 2^n) and beta (19, 4)
        Phi_Q = (gamma_Q[:, None] * nm * E_ord[0].astype(jnp.float32)[None, :])[None]
        Phi_G = Ew_G[None]
        Phi_QC_blk = (gamma_QC[:, None] * nm * PE_unif_ord[0][None, :])[None]
        Phi_QC_r = gamma_QC[None, :, None] * nm * PE_unif_ord[1:5, None, :]
        Phi_N_r = PE_pv_Ew + T_full
        Phi_N_pv_r = PE_pv_Ew
        Phi_N_corr = (PE_pv_Ew[0] + T_corr_r0)[None]
        Phi_N_grov = (PE_pv_Ew[0] + T_grov_r0)[None]
        Phi_N_ud = (PE_unif_Ew_r0 + T_full[0])[None]
        Phi_N_uw = PE_unif_Ew_r0[None]

        Phi = jnp.concatenate([
            Phi_Q, Phi_G, Phi_QC_blk, Phi_QC_r, Phi_N_r, Phi_N_pv_r,
            Phi_N_corr, Phi_N_grov, Phi_N_ud, Phi_N_uw
        ], axis=0)  # (19, 4, 2^n)

        beta = jnp.concatenate([
            beta_Q[None], jnp.asarray(bl_G[i])[None], beta_QC[None],
            jnp.repeat(beta_QC[None], 4, axis=0),
            jnp.repeat(beta_N[None], 12, axis=0)
        ], axis=0)  # (19, 4)

        # Hamming & cavity-error arrays per ordering (0..4)
        E_ord_np = np.asarray(E_ord)
        PE_unif_np = np.asarray(PE_unif_ord)
        PE_ord_z_np = np.asarray(PE_ord_z)
        PE_pv_E_np = np.asarray(PE_pv_E_r)
        Phi_N_r3_np = np.asarray(Phi_N_r[:, 3, :]) / float(gamma_N[3])  # layer 4 effective cost in r=0..3

        sols_ord = [np.flatnonzero(E_ord_np[k] == 0) for k in range(5)]
        H_ord_np = np.zeros((5, N), dtype=np.int8)
        CW_ord_np = np.zeros((5, N), dtype=np.int8)
        H_ord_np[0], CW_ord_np[0] = ham_and_cw_to_sols_np(sols_ord[0], n)
        for r in range(4):
            if np.array_equal(PERM[i, r], np.arange(n)):
                H_ord_np[1 + r], CW_ord_np[1 + r] = H_ord_np[0], CW_ord_np[0]
            else:
                H_ord_np[1 + r], CW_ord_np[1 + r] = ham_and_cw_to_sols_np(sols_ord[1 + r], n)

        sim_out = sim_and_metrics(
            Phi, beta, E_ord, PE_ord_z,
            jnp.asarray(H_ord_np), jnp.asarray(CW_ord_np), ORD_IDX
        )
        p_sol_l, neff_l, ent_l, mE_l, mPE_l, mH_l, top10_l, e_sh, h_sh, cw_sh, tv100, ti100, tv_all = [
            np.asarray(x) for x in sim_out
        ]

        # Formula-level & landscape diagnostics
        sols_var = sols_ord[0]
        nsol = int(len(sols_var))
        n_E1 = int(np.sum(E_ord_np[0] == 1))
        n_E2 = int(np.sum(E_ord_np[0] == 2))
        unif_E = float(np.mean(E_ord_np[0]))
        unif_H = float(np.mean(H_ord_np[0]))

        # pv bitstring in variable order
        pv_bits = (pv_i > 0.5).astype(np.int64)
        x_pv_var = int(np.sum(pv_bits * (1 << np.arange(n - 1, -1, -1))))
        ham_pv_sol = int(H_ord_np[0, x_pv_var])
        pv_acc = float(1.0 - ham_pv_sol / float(n))
        pv_conf = float(np.mean(np.abs(2.0 * pv_i - 1.0)))
        E_at_pv = int(E_ord_np[0, x_pv_var])

        # Local optimistic minimum x_E1 and grover mark accuracy per placement r=0..3
        x_E1_chain = np.zeros(4, dtype=np.int64)
        E_at_E1 = np.zeros(4, dtype=np.int32)
        ham_E1_sol = np.zeros(4, dtype=np.int32)
        sol_in_mark = np.zeros(4, dtype=np.float32)
        for r in range(4):
            min_lvls = np.argmin(E_loc_all[i, r], axis=-1)  # (C,) in 0..15
            x_c = 0
            for c in range(C):
                x_c = (x_c << 4) | int(min_lvls[c])
            x_E1_chain[r] = x_c
            E_at_E1[r] = int(E_ord_np[1 + r, x_c])
            ham_E1_sol[r] = int(H_ord_np[1 + r, x_c])
            # Check how many cavities of the nearest ground state fall in mark > 0.5
            best_m = 0.0
            for s_c in sols_ord[1 + r][:16]:
                cnt_m = 0.0
                for c in range(C):
                    lvl_s = (int(s_c) >> (4 * (C - 1 - c))) & 0xF
                    if mark_all[i, r, c, lvl_s] > 0.5:
                        cnt_m += 1.0
                best_m = max(best_m, cnt_m / C)
            sol_in_mark[r] = best_m

        # Ghost minima & projection stats for order 0 (block) and r=0..3
        var_E = float(np.var(E_ord_np[0].astype(np.float64)))
        proj_stats = []
        for k in range(5):
            pe_k = PE_unif_np[k]
            e_k = E_ord_np[k]
            vk = float(np.var(pe_k.astype(np.float64)) / max(var_E, 1e-12))
            cr = float(np.corrcoef(pe_k, e_k)[0, 1])
            amin = int(np.argmin(pe_k))
            sol_min_pe = float(np.min(pe_k[sols_ord[k]]))
            sol_min_pe_z = float(np.min(PE_ord_z_np[k, sols_ord[k]]))
            n_ghost_u = int(np.sum((e_k > 0) & (pe_k < sol_min_pe - 1e-6)))
            d_info = dict(
                var_kept=vk, corr_PE_E=cr,
                argmin_is_sol=int(e_k[amin] == 0),
                E_at_argmin=int(e_k[amin]),
                ham_argmin_sol=int(H_ord_np[k, amin]),
                sol_min_PE_z=sol_min_pe_z,
                n_ghost_unif=n_ghost_u,
                argmin_idx=amin,
            )
            if k >= 1:
                r = k - 1
                pe_pv_r = PE_pv_E_np[r]
                sol_min_pv = float(np.min(pe_pv_r[sols_ord[k]]))
                d_info["n_ghost_pv"] = int(np.sum((e_k > 0) & (pe_pv_r < sol_min_pv - 1e-6)))
                eff_r = Phi_N_r3_np[r]
                sol_min_eff = float(np.min(eff_r[sols_ord[k]]))
                d_info["n_ghost_eff"] = int(np.sum((e_k > 0) & (eff_r < sol_min_eff - 1e-6)))
            proj_stats.append(d_info)

        # Per-circuit metrics dictionary
        circ_res = {}
        for b, cname in enumerate(CIRCUIT_NAMES):
            ok = int(ORD_IDX[b])
            perm_b = np.arange(n) if ok == 0 else PERM[i, ok - 1]
            ti_b = ti100[b]       # (100,) in ordering ok
            tv_b = np.maximum(tv100[b], 0.0)
            tva_b = np.maximum(tv_all[b], 0.0)

            top1_idx = int(ti_b[0])
            top10_idx = ti_b[:10]
            top10_w = tv_b[:10] / max(float(np.sum(tv_b[:10])), 1e-15)

            e_top10 = E_ord_np[ok, top10_idx].astype(np.float64)
            pez_top10 = PE_ord_z_np[ok, top10_idx].astype(np.float64)
            h_top10, cw_top10, bpw_top10 = cav_err_stats_np(top10_idx, sols_ord[ok], n)

            sol_min_pez = proj_stats[ok]["sol_min_PE_z"]
            top1_beats_sol_u = int(pez_top10[0] < sol_min_pez - 1e-6)
            top10_beats_sol_u = float(np.mean(pez_top10 < sol_min_pez - 1e-6))

            if ok >= 1:
                r_idx = ok - 1
                pe_pv_b = PE_pv_E_np[r_idx]
                sol_min_pv_b = float(np.min(pe_pv_b[sols_ord[ok]]))
                top1_beats_pv = int(pe_pv_b[top1_idx] < sol_min_pv_b - 1e-6)
                x_pv_chain = int(var_to_chain_int(np.array([x_pv_var]), perm_b, n)[0])
                ham_top1_E1 = int(popcount_arr(top1_idx ^ int(x_E1_chain[r_idx])))
            else:
                top1_beats_pv = top1_beats_sol_u
                x_pv_chain = x_pv_var
                ham_top1_E1 = int(popcount_arr(top1_idx ^ int(x_E1_chain[0])))

            ham_top1_pv = int(popcount_arr(top1_idx ^ x_pv_chain))
            ham_top1_amin = int(popcount_arr(top1_idx ^ proj_stats[ok]["argmin_idx"]))
            top10_var = chain_to_var_int(top10_idx, perm_b, n)

            circ_res[cname] = dict(
                p_sol=float(p_sol_l[3, b]),
                inv_p=float(1.0 / max(float(p_sol_l[3, b]), 1e-12)),
                neff=float(neff_l[3, b]),
                ent=float(ent_l[3, b]),
                mE=float(mE_l[3, b]),
                mPE_z=float(mPE_l[3, b]),
                mH=float(mH_l[3, b]),
                p_top1_wrong=float(tv_b[0]),
                p_top10_wrong=float(np.sum(tv_b[:10])),
                p_top100_wrong=float(np.sum(tv_b)),
                p_top1_all=float(tva_b[0]),
                p_top10_all=float(np.sum(tva_b[:10])),
                p_top100_all=float(np.sum(tva_b)),
                p_sol_l=[float(p_sol_l[l, b]) for l in range(4)],
                neff_l=[float(neff_l[l, b]) for l in range(4)],
                ent_l=[float(ent_l[l, b]) for l in range(4)],
                mE_l=[float(mE_l[l, b]) for l in range(4)],
                mPE_l=[float(mPE_l[l, b]) for l in range(4)],
                mH_l=[float(mH_l[l, b]) for l in range(4)],
                top10_l=[float(top10_l[l, b]) for l in range(4)],
                e_shells=[float(x) for x in e_sh[b]],
                h_shells=[float(x) for x in h_sh[b]],
                cw_shells=[float(x) for x in cw_sh[b]],
                top1_E=int(e_top10[0]),
                top10_mean_E=float(np.mean(e_top10)),
                top10_wmean_E=float(np.sum(e_top10 * top10_w)),
                top1_PE_z=float(pez_top10[0]),
                top10_PE_z=float(np.mean(pez_top10)),
                top1_beats_sol_unif=top1_beats_sol_u,
                top10_beats_sol_unif=top10_beats_sol_u,
                top1_beats_sol_pv=top1_beats_pv,
                top1_ham_sol=int(h_top10[0]),
                top10_ham_sol=float(np.mean(h_top10)),
                top1_cav_wrong=int(cw_top10[0]),
                top10_cav_wrong=float(np.mean(cw_top10)),
                top1_bpw_cav=float(bpw_top10[0]),
                top10_bpw_cav=float(np.mean(bpw_top10)),
                top1_ham_pv=ham_top1_pv,
                top1_ham_E1=ham_top1_E1,
                top1_ham_argmin_PE=ham_top1_amin,
                top10_var=[int(x) for x in top10_var],
            )

        # Across R=4 placements comparison for QC, N_full, and N_pv
        placement_comp = {}
        for fam, pref in [("QC", "QC_r"), ("N_full", "N_r"), ("N_pv", "N_pv_r")]:
            ps_r = np.array([circ_res[f"{pref}{r}"]["p_sol"] for r in range(4)], dtype=np.float64)
            t1_vars = [circ_res[f"{pref}{r}"]["top10_var"][0] for r in range(4)]
            t10_sets = [set(circ_res[f"{pref}{r}"]["top10_var"]) for r in range(4)]
            pw_hams, pw_ovl = [], []
            for r1 in range(4):
                for r2 in range(r1 + 1, 4):
                    pw_hams.append(int(popcount_arr(t1_vars[r1] ^ t1_vars[r2])))
                    pw_ovl.append(len(t10_sets[r1] & t10_sets[r2]) / 10.0)
            union_sz = len(set().union(*t10_sets))
            placement_comp[fam] = dict(
                p_r=ps_r.tolist(),
                p_rr=float(np.mean(ps_r)),
                inv_p_rr=float(1.0 / max(float(np.mean(ps_r)), 1e-12)),
                p_max=float(np.max(ps_r)),
                p_min=float(np.min(ps_r)),
                spread=float(np.max(ps_r) / max(float(np.min(ps_r)), 1e-12)),
                n_distinct_top1=len(set(t1_vars)),
                mean_pw_ham_top1=float(np.mean(pw_hams)),
                mean_pw_top10_overlap=float(np.mean(pw_ovl)),
                top10_union_size=int(union_sz),
            )

        formulas_out.append(dict(
            i=i, seed=int(seeds[i]), m=m_real, nsol=nsol, n_E1=n_E1, n_E2=n_E2,
            unif_E=unif_E, unif_H=unif_H,
            pv_acc=pv_acc, pv_conf=pv_conf, ham_pv_sol=ham_pv_sol, E_at_pv=E_at_pv,
            E_at_E1=E_at_E1.tolist(), ham_E1_sol=ham_E1_sol.tolist(), sol_in_mark=sol_in_mark.tolist(),
            proj_block=proj_stats[0], proj_r=proj_stats[1:],
            circuits=circ_res, placement_comp=placement_comp,
        ))
        if (i + 1) % 25 == 0 or i + 1 == max_formulas:
            print(f"[{sat_type} n={n}] {i+1}/{max_formulas} done ({time.time()-t0:.1f}s)", flush=True)

    return formulas_out

def run_all_and_save(sat_types=("4sat", "8sat"), ns=(8, 12, 16, 20), max_formulas=100):
    for sat_type in sat_types:
        out_dir = os.path.join(ROOT, "results", "clean16" if sat_type == "4sat" else "clean16_8sat")
        out_path = os.path.join(out_dir, "diagnosis_data.json")
        data = {}
        if os.path.exists(out_path):
            try:
                data = json.load(open(out_path))
            except Exception:
                data = {}
        for n in ns:
            t0 = time.time()
            print(f"=== Running {sat_type} n={n} ({max_formulas} formulas) ===", flush=True)
            res_n = analyze_n(sat_type, n, max_formulas=max_formulas)
            data[str(n)] = res_n
            with open(out_path, "w") as f:
                json.dump(data, f)
            print(f"=== Saved {sat_type} n={n} to {out_path} ({time.time()-t0:.1f}s) ===", flush=True)
            _JIT_CACHE.clear()
            try:
                jax.clear_caches()
            except Exception:
                pass

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sat", nargs="+", default=["4sat", "8sat"], choices=["4sat", "8sat"])
    ap.add_argument("--n", type=int, nargs="+", default=[8, 12, 16, 20])
    ap.add_argument("--max_formulas", type=int, default=100)
    args = ap.parse_args()
    run_all_and_save(args.sat, args.n, args.max_formulas)
