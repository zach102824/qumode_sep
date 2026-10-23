"""Stage-2 Physical Compilation of Ideal Gray Mixer U_gray(beta_l) into k=3,4 Physical 0..15 SNAP+D Blocks.

For each layer l = 0..L-1 of p2_gray_L15.json (which has (beta_l, gamma_l)):
  1. Multi-start + bidirectional continuation L-BFGS-B in NF = 40 Fock levels with strictly 0 SNAP phase
     on f >= 16, optimizing k physical SNAP+D blocks per layer:
       M_l = S_out(sout_l) P [ D_40(alpha_{l,k-1}) S_{0..15}(theta_{l,k-1}) ... D_40(alpha_{l,0}) S_{0..15}(theta_{l,0}) ] P
     to maximize subspace unitary fidelity F_l = |Tr(U_gray(beta_l)^dagger M_l)|^2 / 16^2 while penalizing
     leakage lk_l = 1 - ||M_l||_F^2 / 16 outside 0..15.
  2. Also evaluates a state-trajectory distilled candidate on n=8,12 training instances (matching the exact
     p2_gray_L15 state evolution |psi_gray> on n=8,12 with physical 0..15 SNAP+D blocks) and stores both
     compiled initializations in cands_x for Stage-3 L-BFGS-B refinement.
"""
from __future__ import annotations
import argparse, json, os, sys, time
import numpy as np
import jax, jax.numpy as jnp
from scipy.optimize import minimize

ROOT = "/workspace/qumode_sep/oct_explore"
sys.path.insert(0, ROOT)
import sim

ap = argparse.ArgumentParser()
ap.add_argument("--gray_json", default=os.path.join(ROOT, "results", "oct23_exp_part2", "p2_gray_L15.json"))
ap.add_argument("--pairs", type=int, required=True, choices=[1, 2, 3, 4])
ap.add_argument("--kprep", type=int, default=3, choices=[2, 3])
ap.add_argument("--starts", type=int, default=20, help="Random multi-starts per layer in addition to continuation")
ap.add_argument("--maxiter_layer", type=int, default=400)
ap.add_argument("--maxiter_traj", type=int, default=120)
ap.add_argument("--amax", type=float, default=1.2)
ap.add_argument("--out", required=True, help="Output init JSON path (used by part2_multicav.py --grow_from)")
a = ap.parse_args()

K = 4
NU = 1 << K  # 16 computational Fock states 0..15
NF_DISP = 40
PAIRS = a.pairs
TRB, VAB = 51_000_000_000, 53_000_000_000
OUT_DIR = os.path.join(ROOT, "results", "oct23_exp_part2")

log = lambda *s: print(time.strftime("%H:%M:%S"), f"[compile_k{PAIRS}]", *s, flush=True)

gray = lambda i: i ^ (i >> 1)
def ungray(g):
    s = g >> 1
    while s:
        g ^= s
        s >>= 1
    return g

FOCK_OF_LOG = np.array([ungray(sum(((g >> (3 - j)) & 1) << j for j in range(K))) for g in range(NU)], dtype=np.int64)

_aa = np.diag(np.sqrt(np.arange(1, NF_DISP)), 1)
_lam, _V = np.linalg.eigh(1j * (_aa.T - _aa))
LAM = jnp.asarray(_lam.astype(np.float32))
VD = jnp.asarray(_V.astype(np.complex64))
NV = jnp.arange(NF_DISP, dtype=jnp.float32)

H_GRAY_FOCK = np.zeros((NU, NU), dtype=np.float64)
for f in range(NU):
    for j in range(K):
        H_GRAY_FOCK[ungray(gray(f) ^ (1 << j)), f] = 1.0
W_GRAY, V_GRAY = np.linalg.eigh(H_GRAY_FOCK)
W_GRAY_J = jnp.asarray(W_GRAY.astype(np.float32))
V_GRAY_J = jnp.asarray(V_GRAY.astype(np.complex64))

PREPS = json.load(open(os.path.join(ROOT, "results", "oct23_exp", "preps.json")))
_prep_info = PREPS[f"sd_K4_k{a.kprep}"]
V0_FOCK = jnp.asarray((np.array(_prep_info["psi_re"][:NU]) + 1j * np.array(_prep_info["psi_im"][:NU])).astype(np.complex64))
V0_LOG = V0_FOCK[FOCK_OF_LOG]

def make_psi0_log(C_cav):
    v = V0_LOG
    for _ in range(C_cav - 1):
        v = jnp.multiply.outer(v, V0_LOG)
    return v

def disp40_single(re, im):
    if a.amax > 0:
        r_raw = jnp.sqrt(re ** 2 + im ** 2 + 1e-20)
        sc = a.amax * jnp.tanh(r_raw / a.amax) / r_raw
        re, im = re * sc, im * sc
    r = jnp.sqrt(re ** 2 + im ** 2 + 1e-20)
    ph = jnp.arctan2(im, re)
    W = jnp.exp(1j * ph * NV)[:, None] * VD
    return (W * jnp.exp(-1j * r * LAM)[None]) @ W.conj().T

def layer_fock_shared(A_l, TH_l, sout_l):
    M = jnp.eye(NF_DISP, NU, dtype=jnp.complex64)
    for q in range(PAIRS):
        th40 = jnp.concatenate([TH_l[q], jnp.zeros(NF_DISP - NU, jnp.float32)])
        M = disp40_single(A_l[q, 0], A_l[q, 1]) @ (jnp.exp(-1j * th40)[:, None] * M)
    return jnp.exp(-1j * sout_l)[:, None] * M[:NU, :NU]

def layer_log_shared(A_l, TH_l, sout_l):
    Vp = layer_fock_shared(A_l, TH_l, sout_l)
    return Vp[jnp.ix_(FOCK_OF_LOG, FOCK_OF_LOG)]

def apply_cav_shared(M_log, psi, c, C_cav):
    let = "defghijk"[:C_cav]
    o = let[c]
    out = let.replace(o, "y")
    return jnp.einsum(f"y{o},b{let}->b{out}", M_log, psi)

def get_cached_instances(n, count, split_label):
    for cand_cnt in sorted(set([count, 500, 1000, 2000, 5000])):
        if cand_cnt < count:
            continue
        cf = os.path.join(OUT_DIR, f"inst_n{n}_{split_label}_{cand_cnt}.npz")
        if os.path.exists(cf):
            Z = np.load(cf)
            return Z["E"][:count]
    p1_n8 = os.path.join(ROOT, "results", "oct23_exp", "data_n8.npz")
    if n == 8 and os.path.exists(p1_n8):
        Z8 = np.load(p1_n8)
        return Z8[f"{split_label}_E"][:count]
    raise RuntimeError(f"Missing cached instances for n={n} {split_label}")

def main():
    t0 = time.time()
    g_doc = json.load(open(a.gray_json))
    bg_opt = np.array(g_doc["bg_opt"], dtype=np.float32).reshape(-1, 2)
    L = bg_opt.shape[0]
    log(f"Compiling L={L} layers from {a.gray_json} into k={PAIRS} physical 0..15 SNAP+D blocks (NF={NF_DISP}, amax={a.amax})")

    def layer_loss(z, U_tgt, w_lk):
        al = z[:PAIRS*2].reshape(PAIRS, 2)
        th = z[PAIRS*2:PAIRS*18].reshape(PAIRS, NU)
        sout = z[PAIRS*18:PAIRS*18+NU]
        Vp = layer_fock_shared(al, th, sout)
        F = jnp.abs(jnp.trace(U_tgt.conj().T @ Vp)) ** 2 / (NU ** 2)
        lk = 1.0 - jnp.sum(jnp.abs(Vp) ** 2) / NU
        return 1.0 - F + w_lk * lk, (F, lk)

    vg_layer = jax.jit(jax.value_and_grad(layer_loss, has_aux=True))

    def compile_schedule(w_lk, seed_off=0):
        rng = np.random.default_rng(1000 + 100 * PAIRS + seed_off)
        dim_z = PAIRS * 18 + NU
        best_by_l = [None] * L

        # Pass 1: sweep l = L-1 down to 0 (small |beta_l| -> large |beta_l|) with continuation + multi-start
        prev_z = None
        for l in range(L - 1, -1, -1):
            beta_val = float(bg_opt[l, 0])
            U_tgt = (V_GRAY_J * jnp.exp(-0.5j * beta_val * W_GRAY_J)[None, :]) @ V_GRAY_J.conj().T
            cands = []
            if prev_z is not None:
                cands.append(prev_z.copy())
                cands.append(prev_z + rng.normal(0, 0.05, dim_z))
                cands.append(prev_z + rng.normal(0, 0.12, dim_z))
            for s in range(a.starts):
                sig_a = 0.25 + 0.05 * s
                A0 = rng.normal(0, sig_a, PAIRS * 2)
                TH0 = rng.uniform(-np.pi, np.pi, PAIRS * NU + NU)
                cands.append(np.concatenate([A0, TH0]))
            best = None
            for z0 in cands:
                def f_obj(z):
                    (val, _), g = vg_layer(jnp.asarray(z, jnp.float32), U_tgt, float(w_lk))
                    return float(val), np.asarray(g, np.float64)
                r = minimize(f_obj, np.asarray(z0, np.float64), jac=True, method="L-BFGS-B",
                             options={"maxiter": a.maxiter_layer})
                if best is None or r.fun < best.fun:
                    best = r
            best_by_l[l] = best
            prev_z = best.x.copy()

        # Pass 2: sweep l = 0 up to L-1 checking continuation from l-1
        for l in range(1, L):
            beta_val = float(bg_opt[l, 0])
            U_tgt = (V_GRAY_J * jnp.exp(-0.5j * beta_val * W_GRAY_J)[None, :]) @ V_GRAY_J.conj().T
            for z0 in [best_by_l[l - 1].x, best_by_l[l - 1].x + rng.normal(0, 0.05, dim_z)]:
                def f_obj(z):
                    (val, _), g = vg_layer(jnp.asarray(z, jnp.float32), U_tgt, float(w_lk))
                    return float(val), np.asarray(g, np.float64)
                r = minimize(f_obj, np.asarray(z0, np.float64), jac=True, method="L-BFGS-B",
                             options={"maxiter": a.maxiter_layer})
                if r.fun < best_by_l[l].fun:
                    best_by_l[l] = r

        A_cal = np.zeros((L, PAIRS, 2), np.float32)
        TH_cal = np.zeros((L, PAIRS, NU), np.float32)
        SOUT_cal = np.zeros((L, NU), np.float32)
        fids, lks = [], []
        for l in range(L):
            z = best_by_l[l].x.astype(np.float32)
            A_cal[l] = z[:PAIRS*2].reshape(PAIRS, 2)
            TH_cal[l] = z[PAIRS*2:PAIRS*18].reshape(PAIRS, NU)
            SOUT_cal[l] = z[PAIRS*18:PAIRS*18+NU]
            beta_val = float(bg_opt[l, 0])
            U_tgt = (V_GRAY_J * jnp.exp(-0.5j * beta_val * W_GRAY_J)[None, :]) @ V_GRAY_J.conj().T
            (_, (F_l, lk_l)), _ = vg_layer(jnp.asarray(z), U_tgt, float(w_lk))
            fids.append(float(F_l))
            lks.append(float(lk_l))
        x_cal = np.concatenate([A_cal.reshape(-1), TH_cal.reshape(-1), SOUT_cal.reshape(-1), bg_opt[:, 1].astype(np.float32)])
        log(f"  w_lk={w_lk}: mean F={np.mean(fids):.5f} (1-F={1-np.mean(fids):.5f}), mean layer leak={np.mean(lks)*100:.3f}%")
        log(f"    per-layer F: {[round(x, 3) for x in fids]}")
        return x_cal, fids, lks

    # Evaluate compiled schedules on n=8, 12 and also run Stage-2B state-trajectory distillation
    E8_tr = get_cached_instances(8, 2000, "train")
    E12_tr = get_cached_instances(12, 500, "train")
    E12_va = get_cached_instances(12, 200, "val")
    m8 = int(round(9.93 * 8)); m12 = int(round(9.93 * 12))
    Cn8_tr = jnp.asarray(E8_tr.astype(np.float32) * (8.0 / m8))
    S8_tr = jnp.asarray((E8_tr == 0).astype(np.float32))
    Cn12_tr = jnp.asarray(E12_tr.astype(np.float32) * (12.0 / m12))
    S12_tr = jnp.asarray((E12_tr == 0).astype(np.float32))
    Cn12_va = jnp.asarray(E12_va.astype(np.float32) * (12.0 / m12))
    S12_va = jnp.asarray((E12_va == 0).astype(np.float32))

    def unpack_shared(x):
        nA = L * PAIRS * 2; nT = L * PAIRS * NU; nS = L * NU
        return (x[:nA].reshape(L, PAIRS, 2),
                x[nA:nA+nT].reshape(L, PAIRS, NU),
                x[nA+nT:nA+nT+nS].reshape(L, NU),
                x[nA+nT+nS:nA+nT+nS+L])

    def sim_shared_state(x, Cn, n):
        C_cav = n // 4; B = Cn.shape[0]
        A, TH, SOUT, g = unpack_shared(x)
        psi = jnp.broadcast_to(make_psi0_log(C_cav), (B,) + (NU,) * C_cav)
        Cn_sh = Cn.reshape(psi.shape)
        def body(psi_c, xs_l):
            A_l, TH_l, SOUT_l, g_l = xs_l
            psi_c = psi_c * jnp.exp(-1j * g_l * Cn_sh)
            M_log = layer_log_shared(A_l, TH_l, SOUT_l)
            for c in range(C_cav):
                psi_c = apply_cav_shared(M_log, psi_c, c, C_cav)
            return psi_c, None
        psi, _ = jax.lax.scan(body, psi, (A, TH, SOUT, g))
        return psi.reshape(B, -1)

    def sim_gray_state(bg, Cn, n):
        C_cav = n // 4; B = Cn.shape[0]
        psi = jnp.broadcast_to(make_psi0_log(C_cav), (B,) + (NU,) * C_cav)
        Cn_sh = Cn.reshape(psi.shape)
        bg_mat = bg.reshape(L, 2)
        def body(psi_c, bg_l):
            b, g = bg_l[0], bg_l[1]
            psi_c = psi_c * jnp.exp(-1j * g * Cn_sh)
            U_f = (V_GRAY_J * jnp.exp(-0.5j * b * W_GRAY_J)[None, :]) @ V_GRAY_J.conj().T
            U_l = U_f[jnp.ix_(FOCK_OF_LOG, FOCK_OF_LOG)]
            for c in range(C_cav):
                psi_c = apply_cav_shared(U_l, psi_c, c, C_cav)
            return psi_c, None
        psi, _ = jax.lax.scan(body, psi, bg_mat)
        return psi.reshape(B, -1)

    ev12 = jax.jit(lambda x, c, s: (
        jnp.sum((jnp.abs(sim_shared_state(x, c, 12)) ** 2) * s, axis=1),
        1.0 - jnp.sum(jnp.abs(sim_shared_state(x, c, 12)) ** 2, axis=1)
    ))

    def score_val12(x, label):
        p_v, lk_v = ev12(jnp.asarray(x, jnp.float32), Cn12_va, S12_va)
        inv = 1.0 / np.maximum(np.asarray(p_v, float), 1e-12)
        med, b90 = float(np.median(inv)), float(np.quantile(inv, 0.9))
        S_v = 0.5 * (np.log(med) + np.log(b90))
        log(f"  [{label}] val12 med/B90 = {med:.2f}/{b90:.2f} (leak={float(jnp.mean(lk_v))*100:.2f}%) S={S_v:.4f}")
        return S_v, med, b90, float(jnp.mean(lk_v))

    cands = []
    for w_lk in [1.0, 2.5]:
        x_cal, fids, lks = compile_schedule(w_lk)
        S_v, med, b90, lk_v = score_val12(x_cal, f"unitary_wlk{w_lk}")
        cands.append({"label": f"gray_unitary_wlk{w_lk}", "x": x_cal, "S": S_v, "med": med, "B90": b90, "fids": fids, "lks": lks})

    # Stage 2B: State-trajectory distillation against ideal Gray state on n=8 (500 insts) + n=12 (200 insts)
    psi8_tgt = jax.jit(lambda b, c: sim_gray_state(b, c, 8))(jnp.asarray(bg_opt.reshape(-1)), Cn8_tr[:500])
    psi12_tgt = jax.jit(lambda b, c: sim_gray_state(b, c, 12))(jnp.asarray(bg_opt.reshape(-1)), Cn12_tr[:200])
    psi8_norm = jnp.sum(jnp.abs(psi8_tgt) ** 2, axis=1)
    psi12_norm = jnp.sum(jnp.abs(psi12_tgt) ** 2, axis=1)

    def loss_distill(x):
        psi8 = sim_shared_state(x, Cn8_tr[:500], 8)
        psi12 = sim_shared_state(x, Cn12_tr[:200], 12)
        ov8 = jnp.abs(jnp.sum(jnp.conj(psi8_tgt) * psi8, axis=1)) ** 2 / jnp.maximum(psi8_norm, 1e-12)
        ov12 = jnp.abs(jnp.sum(jnp.conj(psi12_tgt) * psi12, axis=1)) ** 2 / jnp.maximum(psi12_norm, 1e-12)
        lk8 = 1.0 - jnp.sum(jnp.abs(psi8) ** 2, axis=1)
        lk12 = 1.0 - jnp.sum(jnp.abs(psi12) ** 2, axis=1)
        p8 = jnp.sum((jnp.abs(psi8) ** 2) * S8_tr[:500], axis=1)
        p12 = jnp.sum((jnp.abs(psi12) ** 2) * S12_tr[:200], axis=1)
        nl8 = jnp.mean(-jnp.log(p8 + 1e-12))
        nl12 = jnp.mean(-jnp.log(p12 + 1e-12))
        return 2.0 * (1.0 - 0.5 * (jnp.mean(ov8) + jnp.mean(ov12))) + 0.5 * (nl8 + nl12) + 0.5 * (jnp.mean(lk8) + jnp.mean(lk12))

    vg_dist = jax.jit(jax.value_and_grad(loss_distill))
    for base_cand in list(cands):
        x0 = base_cand["x"]
        r = minimize(lambda x: tuple(np.asarray(v, float) for v in vg_dist(jnp.asarray(x, jnp.float32))),
                     np.asarray(x0, np.float64), jac=True, method="L-BFGS-B", options={"maxiter": a.maxiter_traj})
        x_d = r.x.astype(np.float32)
        lbl = base_cand["label"] + "_distill"
        S_v, med, b90, lk_v = score_val12(x_d, lbl)
        cands.append({"label": lbl, "x": x_d, "S": S_v, "med": med, "B90": b90, "fids": base_cand["fids"], "lks": base_cand["lks"]})

    best_unitary = min([c for c in cands if not c["label"].endswith("_distill")], key=lambda c: c["S"])
    best_distill = min([c for c in cands if c["label"].endswith("_distill")], key=lambda c: c["S"])
    best_cand = min(cands, key=lambda c: c["S"])
    top_cands = [best_unitary, best_distill]
    log(f"Selected best Gray-compiled init: {best_cand['label']} (val12={best_cand['med']:.2f}/{best_cand['B90']:.2f}, S={best_cand['S']:.4f})")

    out_doc = {
        "winner_init": best_cand["label"],
        "L": L,
        "pairs": PAIRS,
        "kprep": a.kprep,
        "val12_pre": {"median_inv_p": best_cand["med"], "B90_inv_p": best_cand["B90"], "S": best_cand["S"]},
        "layer_fids": best_cand["fids"],
        "layer_leaks": best_cand["lks"],
        "x_best": best_cand["x"].tolist(),
        "cands_x": [{"label": c["label"], "S": c["S"], "med": c["med"], "B90": c["B90"], "x": c["x"].tolist()} for c in top_cands],
        "sec": time.time() - t0,
    }
    json.dump(out_doc, open(a.out, "w"))
    log(f"Saved {a.out} in {time.time()-t0:.1f}s")

if __name__ == "__main__":
    main()
