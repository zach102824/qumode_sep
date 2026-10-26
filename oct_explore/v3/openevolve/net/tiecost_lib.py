"""Tied-cost GNN trainer (library) for gnn_tiecost OpenEvolve.
Locked: vacuum | SNAP+D -> cost; ONE cost unitary per formula (gamma + clause_w tied).
Pure GNN + clause_w. SNAP+D params free. Same L train/test.
"""
from __future__ import annotations
import json, os, sys, time
import numpy as np
import jax, jax.numpy as jnp
from jax.flatten_util import ravel_pytree
ROOT = "/workspace/qumode_sep/oct_explore"
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "v3"))
import arch_nets as AN
OUT_DIR = os.path.join(ROOT, "results", "oct23_exp")
PREPS = json.load(open(os.path.join(OUT_DIR, "preps.json")))
WS = {6: {"look_med": 8.466, "look_b90": 16.039}, 8: {"look_med": 13.989, "look_b90": 29.396}}

def _assert_tied_cost_heads(p):
    """Hardening (qumode_checker): Wc/G2_g last dim must be 1.
    Otherwise w_tied=(hc@Wc)[:,:,0] / g_tied=...[:,0] silently drop an L axis.
    """
    wc_sh = tuple(p["Wc"].shape)
    g2_sh = tuple(p["G2_g"].shape)
    assert wc_sh[-1] == 1, f"Wc last dim must be 1 (tied clause_w); got shape {wc_sh}"
    assert g2_sh[-1] == 1, f"G2_g last dim must be 1 (tied gamma); got shape {g2_sh}"

def _gray(i): return i ^ (i >> 1)
def eval_metrics(p_succ, p_leak, n=None):
    p_succ = np.asarray(p_succ, float); p_leak = np.asarray(p_leak, float)
    inv = 1.0 / np.maximum(p_succ, 1e-12)
    med = float(np.median(inv)); b90 = float(np.quantile(inv, 0.9))
    out = {"median_inv_p": med, "B90_inv_p": b90, "S": float((np.log(med)+np.log(b90))/2.0),
           "mean_p": float(np.mean(p_succ)), "mean_leak": float(np.mean(p_leak)), "max_leak": float(np.max(p_leak))}
    if n in WS:
        r = WS[n]; out["beats_ws_look_med"] = bool(med < r["look_med"]); out["beats_ws_look_b90"] = bool(b90 < r["look_b90"])
    return out


def train_one(n, L, hypers, screen=False, log_fn=None):
    log = log_fn or (lambda *a: print(__import__("time").strftime("%H:%M:%S"), f"[tie n={n} L={L}]", *a, flush=True))
    H = int(hypers.get("H", 128)); rounds = int(hypers.get("rounds", 4)); arch = str(hypers.get("arch", "mp"))
    lr = float(hypers.get("lr", 1.2e-3)); steps = int(hypers.get("steps", 1000)); bs = int(hypers.get("bs", 250))
    wd = float(hypers.get("wd", 1e-4)); leakpen = float(hypers.get("leakpen", 0.1)); tailw = float(hypers.get("tailw", 1.0))
    ntrain = int(hypers.get("ntrain", 10000)); amax = float(hypers.get("amax", 1.8)); seed = int(hypers.get("seed", 0))
    if screen:
        steps = min(steps, int(os.environ.get("OE_SCREEN_STEPS", "200")))
        ntrain = min(ntrain, int(os.environ.get("OE_NTRAIN_SCREEN", "2000")))
    assert 1 <= L <= 20
    C = 2; K = n // 2; NU = 1 << K
    NF = 24 if n == 6 else 36; NF_CHK = 40 if n == 6 else 56
    K_PREP = 0; L_ACT = L; PAIRS = 1; NB_ACT = L
    DATA = np.load(os.path.join(OUT_DIR, f"data_n{n}.npz"))
    BP_DATA = np.load(os.path.join(OUT_DIR, f"bp_n{n}.npz"))
    m_clauses = int(DATA["m"]); nm_scale = float(n) / float(m_clauses)
    GRID_TO_X = np.zeros((NU, NU), dtype=np.int64)
    for f0 in range(NU):
        for f1 in range(NU):
            bits = [(_gray(f0) >> j) & 1 for j in range(K)] + [(_gray(f1) >> j) & 1 for j in range(K)]
            GRID_TO_X[f0, f1] = sum(b << (n - 1 - v) for v, b in enumerate(bits))

    def prepare_split(split_name, max_count=None):
        Es = DATA[f"{split_name}_E"]; cls_v = DATA[f"{split_name}_cls_v"]; cls_s = DATA[f"{split_name}_cls_s"]
        pos = BP_DATA[f"{split_name}_pos"]; neg = BP_DATA[f"{split_name}_neg"]
        if max_count is not None:
            Es, cls_v, cls_s = Es[:max_count], cls_v[:max_count], cls_s[:max_count]
            pos, neg = pos[:max_count], neg[:max_count]
        B = len(Es)
        Cn_rel = np.zeros((B, NU, NU), np.float32); Succ = np.zeros((B, NU, NU), np.float32)
        Ip = np.zeros((B, m_clauses, n), np.float32); In = np.zeros((B, m_clauses, n), np.float32)
        fv = np.zeros((B, n, 7), np.float32); deg = np.zeros((B, n), np.float32)
        Viol = np.zeros((B, m_clauses, NU * NU), np.uint8)
        x_grid_flat = GRID_TO_X.reshape(-1)
        slot_bits = np.array([[(x_val >> (n - 1 - s)) & 1 for x_val in x_grid_flat] for s in range(n)], dtype=np.uint8)
        orders = np.arange(n, dtype=np.int64)[None].repeat(B, 0); flips = np.zeros((B, n), dtype=np.int64)
        inv_orders = orders.copy()
        for b in range(B):
            E_grid = Es[b][GRID_TO_X].astype(np.float32); c_grid = E_grid * nm_scale
            Cn_rel[b] = c_grid - np.max(c_grid); Succ[b] = (E_grid == 0).astype(np.float32)
        b_idx = np.arange(B)[:, None, None]; c_idx = np.arange(m_clauses)[None, :, None]
        v_new = inv_orders[b_idx, cls_v]; s_new = cls_s * (1 - 2 * flips[b_idx, v_new])
        np.add.at(Ip, (b_idx, c_idx, v_new), (s_new > 0).astype(np.float32))
        np.add.at(In, (b_idx, c_idx, v_new), (s_new < 0).astype(np.float32))
        need_val = (s_new < 0).astype(np.uint8)[..., None]
        Viol[:] = ((slot_bits[v_new[:,:,0]]==need_val[:,:,0]) & (slot_bits[v_new[:,:,1]]==need_val[:,:,1]) &
                   (slot_bits[v_new[:,:,2]]==need_val[:,:,2]) & (slot_bits[v_new[:,:,3]]==need_val[:,:,3]))
        fv[:,:,0]=1.0; fv[:,:,1]=Ip.sum(1)/(2.0*m_clauses/n); fv[:,:,2]=In.sum(1)/(2.0*m_clauses/n)
        deg[:] = np.maximum(Ip.sum(1)+In.sum(1), 1.0)
        return {"B": B, "G": {"Ip": jnp.asarray(Ip), "In": jnp.asarray(In), "fv": jnp.asarray(fv), "deg": jnp.asarray(deg)},
                "Cn_rel": jnp.asarray(Cn_rel), "Succ": jnp.asarray(Succ), "Viol": jnp.asarray(Viol)}

    def build_sim(nf):
        aa = np.diag(np.sqrt(np.arange(1, nf)), 1)
        lam, V = np.linalg.eigh(1j * (aa.T - aa))
        LAM = jnp.asarray(lam.astype(np.float32)); VD = jnp.asarray(V.astype(np.complex64)); NV = jnp.arange(nf, dtype=jnp.float32)
        def disp_batch(re, im):
            if amax > 0:
                r_raw = jnp.sqrt(re**2 + im**2 + 1e-20); sc = amax * jnp.tanh(r_raw/amax)/r_raw; re, im = re*sc, im*sc
            r = jnp.sqrt(re**2 + im**2 + 1e-20); ph = jnp.arctan2(im, re)
            W = jnp.exp(1j * ph[:,None] * NV[None])[:,:,None] * VD[None]
            return jnp.einsum("bik,bk,bjk->bij", W, jnp.exp(-1j * r[:,None] * LAM[None]), W.conj())
        v0 = np.zeros(nf, np.complex64); v0[0] = 1.0; PSI0 = jnp.asarray(np.outer(v0, v0))
        def sim_fn(A_b, TH_b, g_tied, w_tied, Cn_rel, Succ, Viol):
            B = Cn_rel.shape[0]; psi = jnp.broadcast_to(PSI0, (B, nf, nf)); Vf = Viol.astype(jnp.float32)
            cw = g_tied[:, None] + w_tied
            wx = nm_scale * jnp.einsum("bm,bmx->bx", cw, Vf).reshape(B, NU, NU)
            wx_rel = wx - jnp.max(wx.reshape(B, -1), axis=1)[:, None, None]
            cost_u = jnp.exp(-1j * wx_rel)
            for l in range(NB_ACT):
                ph0 = jnp.concatenate([TH_b[:,0,l], jnp.zeros((B, nf-NU), jnp.float32)], axis=1)
                ph1 = jnp.concatenate([TH_b[:,1,l], jnp.zeros((B, nf-NU), jnp.float32)], axis=1)
                psi = psi * jnp.exp(-1j*ph0)[:,:,None] * jnp.exp(-1j*ph1)[:,None,:]
                D0 = disp_batch(A_b[:,0,l,0], A_b[:,0,l,1]); D1 = disp_batch(A_b[:,1,l,0], A_b[:,1,l,1])
                psi = jnp.einsum("bij,bjk->bik", D0, psi); psi = jnp.einsum("bkj,bij->bik", D1, psi)
                psi = psi.at[:,:NU,:NU].multiply(cost_u)
            pr_valid = jnp.abs(psi[:,:NU,:NU])**2
            return jnp.sum(pr_valid*Succ, axis=(1,2)), 1.0 - jnp.sum(pr_valid, axis=(1,2)), wx_rel
        return sim_fn

    SIM_MAIN = build_sim(NF); SIM_CHK = build_sim(NF_CHK)
    BT = np.zeros((K, NU), np.float32)
    for f in range(NU):
        for j in range(K): BT[j,f] = (_gray(f)>>j)&1
    BT = jnp.asarray(BT); CV = jnp.asarray(np.array([[K*c+j for j in range(K)] for c in range(C)]))

    def init_net(rng):
        R, FV, FM = rounds, 7, 32
        def W(i,o,s=1.0): return jnp.asarray(rng.normal(0, s/np.sqrt(max(i,1)), (i,o)).astype(np.float32))
        p = AN.init_backbone(rng, arch, H, R, FV)
        p["G1"]=W(2*H,H); p["g1"]=jnp.zeros(H)
        p["G2_a"]=jnp.zeros((H, C*NB_ACT*2)); p["G2_g"]=jnp.zeros((H,1))
        p["Wa"]=jnp.zeros((H, NB_ACT*2)); p["Ws"]=jnp.zeros((H, NB_ACT))
        p["P0"]=W(H,FM,1.5); p["P1"]=W(H,FM,1.5)
        p["u0"]=jnp.asarray(rng.normal(0,0.5,FM).astype(np.float32)); p["u1"]=jnp.asarray(rng.normal(0,0.5,FM).astype(np.float32))
        p["fq"]=jnp.zeros(FM); p["Wo"]=jnp.zeros((FM, NB_ACT))
        p["Mcav1"]=W(K*(H+7),H,1.0); p["bcav1"]=jnp.zeros(H)
        p["Wfock"]=jnp.zeros((H, NB_ACT*NU)); p["Wbit"]=jnp.asarray(W(H+7,FM,1.0)); p["bbit"]=jnp.zeros(FM)
        p["Wprod"]=jnp.zeros((9,FM,NB_ACT)); p["Wphi_a"]=jnp.zeros((NU*FM, NB_ACT*2)); p["Wc"]=jnp.zeros((H,1))
        r = rng.uniform(0.15,0.35,(C,NB_ACT)); ph = rng.uniform(0,2*np.pi,(C,NB_ACT))
        A0 = np.stack([r*np.cos(ph), r*np.sin(ph)], -1).astype(np.float32); TH0 = np.zeros((C,NB_ACT,NU), np.float32)
        pr1 = PREPS[f"sd_K{K}_k1"]; A0[:,0,:] = np.array(pr1["alphas"][0])[None,:]; TH0[:,0,:] = np.array(pr1["thetas"][0])[None,:]
        p["A0"]=jnp.asarray(A0); p["TH0"]=jnp.asarray(TH0); p["g0"]=jnp.asarray(np.array([3.0], np.float32))
        _assert_tied_cost_heads(p)
        return p

    def forward_net(p, G):
        B = G["Ip"].shape[0]
        _assert_tied_cost_heads(p)
        hv, hc = AN.run_backbone(p, G, arch, H)
        g_pool = jnp.concatenate([hv.mean(1), hc.mean(1)], -1); h_g = jnp.tanh(g_pool @ p["G1"] + p["g1"])
        dA_glob = (h_g @ p["G2_a"]).reshape(B, C, NB_ACT, 2)
        g_tied = (p["g0"][None,:] + h_g @ p["G2_g"])[:,0]
        hd = hv[:, CV]; hd_aug = jnp.concatenate([hd, G["fv"][:,CV]], axis=-1)
        hcav_h = jnp.tanh(hd_aug.reshape(B, C, K*(H+7)) @ p["Mcav1"] + p["bcav1"])
        dA_cav = ((hd.mean(2) + hcav_h) @ p["Wa"]).reshape(B, C, NB_ACT, 2)
        A_b = p["A0"][None] + dA_glob + dA_cav
        U0 = p["u0"] + hd @ p["P0"]; U1 = p["u1"] + hd @ p["P1"]
        z = (jnp.einsum("bcjm,jf->bcfm", U1, BT) + jnp.einsum("bcjm,jf->bcfm", U0, 1.0-BT)) / K
        ph_free = jnp.transpose(jnp.tanh(z + p["fq"]) @ p["Wo"], (0,1,3,2))
        ph_lin = jnp.einsum("bcjl,jf->bclf", hd @ p["Ws"], BT)
        ph_fock = (hcav_h @ p["Wfock"]).reshape(B, C, NB_ACT, NU)
        P_bit = jax.nn.sigmoid(hd_aug @ p["Wbit"] + p["bbit"])
        M_jf = P_bit[:,:,:,None,:] * BT[None,None,:,:,None] + (1.0-P_bit[:,:,:,None,:]) * (1.0-BT[None,None,:,:,None])
        Phi = jnp.prod(M_jf, axis=2); Phi_pad = jnp.pad(Phi, ((0,0),(0,0),(4,4),(0,0)))
        Phi_st = jnp.stack([Phi_pad[:,:,d:d+NU,:] for d in range(9)], axis=2)
        ph_prod = jnp.einsum("bcdfm,dml->bclf", Phi_st, p["Wprod"])
        dA_phi = (Phi.reshape(B, C, NU*32) @ p["Wphi_a"]).reshape(B, C, NB_ACT, 2)
        A_b = A_b + dA_phi; TH_b = p["TH0"][None] + ph_free + ph_lin + ph_fock + ph_prod
        w_tied = (hc @ p["Wc"])[:,:,0]
        return A_b, TH_b, g_tied, w_tied

    D_tr = prepare_split("train", max_count=ntrain); D_va = prepare_split("val"); D_te = prepare_split("test")
    rng = np.random.default_rng(777 + seed); p0 = init_net(rng); th0, unr = ravel_pytree(p0)
    log(f"start params={th0.size} steps={steps} ntrain={ntrain} H={H} rounds={rounds} arch={arch} lr={lr}")

    def loss_fn(th, G, Cn_rel, Succ, Viol):
        p = unr(th); A_b, TH_b, g_t, w_t = forward_net(p, G)
        p_s, p_l, _ = SIM_MAIN(A_b, TH_b, g_t, w_t, Cn_rel, Succ, Viol)
        nl = -jnp.log(p_s + 1e-12); out = jnp.mean(nl)
        if tailw > 0:
            k = max(1, int(round(0.16 * nl.shape[0]))); top_nl = jax.lax.top_k(nl, k)[0]
            out = out + tailw * jnp.mean(top_nl) + 0.12 * tailw * jnp.mean(jnp.exp(jnp.minimum(top_nl, 3.5)))
        if leakpen > 0: out = out + leakpen * jnp.mean(p_l)
        if wd > 0:
            keys = ("G2_a","G2_g","Wa","Ws","Wo","Wfock","Wprod","Wphi_a","Wc")
            out = out + wd * sum(jnp.sum(p[k_]**2) for k_ in keys)
        return out
    vg = jax.jit(jax.value_and_grad(loss_fn))
    def _ev(th, D, sim=SIM_MAIN):
        A_b, TH_b, g_t, w_t = forward_net(unr(th), D["G"])
        p_s, p_l, wx = sim(A_b, TH_b, g_t, w_t, D["Cn_rel"], D["Succ"], D["Viol"])
        return p_s, p_l, wx, g_t, w_t
    ev_jit = jax.jit(lambda th, D: _ev(th, D, SIM_MAIN))
    m = np.zeros_like(np.asarray(th0)); v = np.zeros_like(m); th = np.asarray(th0, np.float32)
    best = {"S": np.inf, "th": th.copy(), "step": 0}; Btr = D_tr["B"]; t0 = time.time()
    for step in range(1, steps+1):
        sel = rng.choice(Btr, size=min(bs, Btr), replace=False)
        G = {k: jnp.asarray(np.asarray(D_tr["G"][k])[sel]) for k in D_tr["G"]}
        Cn = jnp.asarray(np.asarray(D_tr["Cn_rel"])[sel]); Suc = jnp.asarray(np.asarray(D_tr["Succ"])[sel])
        Viol = jnp.asarray(np.asarray(D_tr["Viol"])[sel])
        loss, g = vg(jnp.asarray(th), G, Cn, Suc, Viol)
        g = np.nan_to_num(np.asarray(g), nan=0.0, posinf=0.0, neginf=0.0)
        m = 0.9*m + 0.1*g; v = 0.999*v + 0.001*(g*g)
        mh = m/(1-0.9**step); vh = v/(1-0.999**step); th = th - lr * mh / (np.sqrt(vh)+1e-8)
        if step % max(1, steps//5) == 0 or step == steps:
            p_v, lk_v, wx_v, g_v, w_v = ev_jit(jnp.asarray(th), D_va)
            mv = eval_metrics(p_v, lk_v, n)
            tie_ok = bool(np.isfinite(np.asarray(wx_v)).all() and np.isfinite(np.asarray(g_v)).all())
            if mv["S"] < best["S"] and tie_ok: best.update(S=mv["S"], th=th.copy(), step=step)
            log(f"  step {step} val med/B90={mv['median_inv_p']:.3f}/{mv['B90_inv_p']:.3f} S={mv['S']:.4f} tie_ok={tie_ok}")
    th_b = jnp.asarray(best["th"])
    p_te, lk_te, wx_te, g_te, w_te = ev_jit(th_b, D_te)
    p_va, lk_va, _, _, _ = ev_jit(th_b, D_va)
    assert np.asarray(g_te).ndim == 1 and np.asarray(w_te).ndim == 2
    mt = eval_metrics(p_te, lk_te, n); mv = eval_metrics(p_va, lk_va, n)
    p_chk, lk_chk, _, _, _ = _ev(th_b, D_te, SIM_CHK); mc = eval_metrics(p_chk, lk_chk, n)
    out = {"n": n, "L": L, "screen": screen,
           "hypers": dict(H=H, rounds=rounds, arch=arch, lr=lr, steps=steps, bs=bs, wd=wd, leakpen=leakpen, tailw=tailw, ntrain=ntrain, amax=amax, seed=seed),
           "val": mv, "test": mt, "test_NFchk": mc, "cost_tied": True,
           "tie_ok": bool(np.isfinite(np.asarray(wx_te)).all()), "mean_leak_test": mt["mean_leak"],
           "sec": time.time()-t0, "best_step": best["step"], "order": "sdc", "vacuum": True, "clause_w": True, "pure_gnn": True}
    log(f"done test med/B90={mt['median_inv_p']:.3f}/{mt['B90_inv_p']:.3f} leak={mt['mean_leak']:.4f} {out['sec']:.0f}s")
    return out


def train_rule(L, hypers, screen=False):
    train_ns = tuple(hypers.get("train_ns", (6, 8)))
    supported = [n for n in train_ns if n in (6, 8)]
    requested_extra = [n for n in train_ns if n not in (6, 8)]
    if not supported: supported = [6, 8]
    # For screen speed: train only the first supported n
    if screen:
        supported = supported[:1]
    results = {}
    for n in supported:
        results[n] = train_one(n, L, hypers, screen=screen)
    scores = []
    for n, r in results.items():
        if not r.get("tie_ok") or not r.get("cost_tied"):
            return {"combined_score": -20.0, "error": "cost not tied or nonfinite", "results": {str(k):v for k,v in results.items()}, "L": L}
        if r["test"]["mean_leak"] > 0.35:
            return {"combined_score": -15.0, "error": f"leak {r['test']['mean_leak']:.3f}", "results": {str(k):v for k,v in results.items()}, "L": L}
        med, b90 = r["test"]["median_inv_p"], r["test"]["B90_inv_p"]
        scores.append(-(np.log(med) + np.log(b90)) / 2.0)
    growth = None; growth_term = 0.0
    if 6 in results and 8 in results:
        m6 = results[6]["test"]["median_inv_p"]; m8 = results[8]["test"]["median_inv_p"]
        growth = float((m8 / max(m6, 1e-9)) ** 0.5)
        growth_term = -np.log(max(growth, 1.01))
    combined = float(np.mean(scores) + 0.5 * growth_term)
    # serialize results lightly
    ser = {}
    for k, v in results.items():
        ser[str(k)] = {kk: vv for kk, vv in v.items()}
    return {"combined_score": combined, "growth_proxy": growth, "requested_extra_ns": list(requested_extra),
            "supported_ns": supported, "results": ser, "L": L, "screen": screen}

if __name__ == "__main__":
    import argparse, json as _json
    ap = argparse.ArgumentParser(); ap.add_argument("--L", type=int, default=5); ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--screen", action="store_true"); ap.add_argument("--steps", type=int, default=0)
    a = ap.parse_args()
    h = dict(H=128, rounds=4, lr=1.2e-3, steps=a.steps or (200 if a.screen else 1000), ntrain=2000 if a.screen else 10000, train_ns=(a.n,))
    print(_json.dumps(train_one(a.n, a.L, h, screen=a.screen), indent=2, default=str))
