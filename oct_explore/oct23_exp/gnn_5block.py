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
ap.add_argument("--relabel", default="none", choices=["none", "bias", "biaspol", "bppol", "bpdec"])
ap.add_argument("--clause_w", action="store_true")
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
ap.add_argument("--leakpen", type=float, default=0.1)
ap.add_argument("--wd", type=float, default=1e-4)
ap.add_argument("--amax", type=float, default=1.8)
ap.add_argument("--init_base", default="", help="JSON from part1_2cav.py to warm-start base A0, TH0, g0")
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()

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
    if mode in ("bppol", "bpdec"):
        var_by_conf = np.argsort(np.abs(h_bp), kind="stable")
        order = np.empty(n, dtype=np.int64)
        for rank, v_orig in enumerate(var_by_conf):
            order[slots_by_speed[rank]] = v_orig
        flip = (h_bp >= 0).astype(np.int64)[order]
        return order, flip
    raise ValueError(mode)

def prepare_split(split_name, max_count=None):
    Es = DATA[f"{split_name}_E"]
    cls_v = DATA[f"{split_name}_cls_v"]
    cls_s = DATA[f"{split_name}_cls_s"]
    pos = BP_DATA[f"{split_name}_pos"]
    neg = BP_DATA[f"{split_name}_neg"]
    h_bp = BP_DATA[f"{split_name}_h_dec"] if a.relabel == "bpdec" else BP_DATA[f"{split_name}_h_bp"]
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

    for b in range(B):
        order, flip = build_perm_and_flip(pos[b], neg[b], h_bp[b], a.relabel)
        x_old = np.zeros_like(GRID_TO_X)
        for s in range(n):
            bit_s = ((GRID_TO_X >> (n - 1 - s)) & 1) ^ int(flip[s])
            x_old |= (bit_s << (n - 1 - int(order[s])))
        E_grid = Es[b][x_old].astype(np.float32)
        c_grid = E_grid * nm_scale
        Cn_rel[b] = c_grid - np.max(c_grid)
        Succ[b] = (E_grid == 0).astype(np.float32)

        inv_order = np.empty(n, dtype=np.int64)
        inv_order[order] = np.arange(n)
        for c_i in range(m_clauses):
            v0, v1, v2, v3 = [int(inv_order[int(cls_v[b, c_i, k_i])]) for k_i in range(4)]
            s0, s1, s2, s3 = [int(cls_s[b, c_i, k_i]) * (1 if flip[int(inv_order[int(cls_v[b, c_i, k_i])])] == 0 else -1) for k_i in range(4)]
            for s_new, sign_new in ((v0, s0), (v1, s1), (v2, s2), (v3, s3)):
                if sign_new > 0: Ip[b, c_i, s_new] = 1.0
                else: In[b, c_i, s_new] = 1.0
            if a.clause_w:
                # Clause is violated iff all 4 literals are false
                viol_c = (
                    (slot_bits[v0] == (0 if s0 > 0 else 1)) &
                    (slot_bits[v1] == (0 if s1 > 0 else 1)) &
                    (slot_bits[v2] == (0 if s2 > 0 else 1)) &
                    (slot_bits[v3] == (0 if s3 > 0 else 1))
                )
                Viol[b, c_i] = viol_c
        fv[b, :, 0] = 1.0
        fv[b, :, 1] = Ip[b].sum(0) / (2.0 * m_clauses / n)
        fv[b, :, 2] = In[b].sum(0) / (2.0 * m_clauses / n)
        fv[b, :, 3] = h_bp[b, order] * (1.0 - 2.0 * flip)
        deg[b] = np.maximum(Ip[b].sum(0) + In[b].sum(0), 1.0)

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
    dA_cav = (hd.mean(2) @ p["Wa"]).reshape(B, C, NB_ACT, 2)
    A_b = p["A0"][None] + dA_glob + dA_cav

    U0 = p["u0"] + hd @ p["P0"]
    U1 = p["u1"] + hd @ p["P1"]
    z = (jnp.einsum("bcjm,jf->bcfm", U1, BT) + jnp.einsum("bcjm,jf->bcfm", U0, 1.0 - BT)) / K
    ph_free = jnp.transpose(jnp.tanh(z + p["fq"]) @ p["Wo"], (0, 1, 3, 2))  # (B, C, NB_ACT, NU)
    ph_lin = jnp.einsum("bcjl,jf->bclf", hd @ p["Ws"], BT)
    TH_b = p["TH0"][None] + ph_free + ph_lin
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
        out = jnp.mean(nl)
        if a.tailw > 0:
            k = max(1, nl.shape[0] // 10)
            out = out + a.tailw * jnp.mean(jax.lax.top_k(nl, k)[0])
        if a.leakpen > 0:
            out = out + a.leakpen * jnp.mean(p_l)
        if a.wd > 0:
            keys = ("G2_a", "G2_g", "Wa", "Ws", "Wo") + (("Wc",) if a.clause_w else ())
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
            lr_t = a.lr * (0.08 + 0.92 * 0.5 * (1.0 + np.cos(np.pi * st / a.steps)))
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
    m_val = eval_metrics(*ev_main(th_best, D_va))
    p_te, lk_te = ev_main(th_best, D_te)
    m_test = eval_metrics(p_te, lk_te)
    p_chk, lk_chk = ev_chk(th_best, D_te)
    m_chk = eval_metrics(p_chk, lk_chk)

    res = {
        "args": vars(a), "ws_ref": WS_REF, "NF": NF, "NF_chk": NF_CHK, "NU": NU,
        "best_step": best_val["step"], "val": m_val, "test": m_test, f"test_NF{NF_CHK}": m_chk,
        "p_test": np.asarray(p_te, float).tolist(), "sec": time.time() - t0
    }
    json.dump(res, open(OUT, "w"))
    log(f"DONE {a.tag}: val {m_val['median_inv_p']:.3f}/{m_val['B90_inv_p']:.3f} | test {m_test['median_inv_p']:.3f}/{m_test['B90_inv_p']:.3f} (NF{NF_CHK}: {m_chk['median_inv_p']:.3f}/{m_chk['B90_inv_p']:.3f}, leak={m_chk['mean_leak']:.4f}) vs WS tries {WS_REF['tries_med']:.2f}/{WS_REF['tries_b90']:.2f} ({time.time()-t0:.1f}s)")

if __name__ == "__main__":
    main()
