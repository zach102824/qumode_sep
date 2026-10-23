"""Precompute compile-time variable features and damped-BP marginals (including BP-decimation and R=4 decimated BP tree)
for n=6 and n=8 datasets (train=10,000, val=200, test=100) and save to bp_n{n}.npz.
All quantities use ONLY the clause list (var, sign) — zero access to true E or nsol."""
import os, time
import numpy as np

OUT_DIR = "/workspace/qumode_sep/oct_explore/results/oct23_exp"

def compute_bp_all(cls_v, cls_s, n):
    B, m, _ = cls_v.shape
    b_idx = np.arange(B)[:, None, None]
    pos_cnt = np.zeros((B, n), np.float32)
    neg_cnt = np.zeros((B, n), np.float32)
    np.add.at(pos_cnt, (b_idx, cls_v), (cls_s > 0).astype(np.float32))
    np.add.at(neg_cnt, (b_idx, cls_v), (cls_s < 0).astype(np.float32))
    h0_base = 0.15 * (pos_cnt - neg_cnt)

    def run_bp_once(pinned_mask, pin_signs, iters=12, beta=3.0):
        eta = np.zeros((B, m, 4), np.float32)
        for _ in range(iters):
            h = np.where(pinned_mask, 12.0 * pin_signs, h0_base)
            np.add.at(h, (b_idx, cls_v), cls_s * eta)
            h = np.where(pinned_mask, 12.0 * pin_signs, h)
            h_cav = h[b_idx, cls_v] - cls_s * eta
            q_false = 1.0 / (1.0 + np.exp(np.clip(beta * cls_s * h_cav, -20.0, 20.0)))
            new_eta = np.zeros_like(eta)
            for k in range(4):
                others = [j for j in range(4) if j != k]
                prod_other = q_false[:, :, others[0]] * q_false[:, :, others[1]] * q_false[:, :, others[2]]
                new_eta[:, :, k] = -0.5 * np.log(np.clip(1.0 - 0.96 * prod_other, 1e-5, 1.0))
            eta = 0.5 * eta + 0.5 * new_eta
        h_final = h0_base.copy()
        np.add.at(h_final, (b_idx, cls_v), cls_s * eta)
        return h_final.astype(np.float32)

    # 1. Plain damped BP (16 iters)
    h_bp = run_bp_once(np.zeros((B, n), bool), np.zeros((B, n), np.float32), iters=16, beta=2.6)

    # 2. BP-Decimation (1 branch): fix variables from most confident to least confident
    pinned = np.zeros((B, n), dtype=bool)
    pin_val = np.zeros((B, n), dtype=np.float32)
    h_step0 = run_bp_once(pinned, pin_val, iters=14, beta=3.0)
    conf0 = np.abs(h_step0)
    v_first = np.argmax(conf0, axis=1)
    pinned[np.arange(B), v_first] = True
    pin_val[np.arange(B), v_first] = np.where(h_step0[np.arange(B), v_first] >= 0, 1.0, -1.0)
    for step in range(1, n):
        h_cur = run_bp_once(pinned, pin_val, iters=10, beta=3.0)
        conf = np.where(pinned, -1.0, np.abs(h_cur))
        v_best = np.argmax(conf, axis=1)
        pinned[np.arange(B), v_best] = True
        pin_val[np.arange(B), v_best] = np.where(h_cur[np.arange(B), v_best] >= 0, 1.0, -1.0)
    # Combine sign of decimated solution with confidence magnitude of step-0 BP
    h_dec = (pin_val * (conf0 + 0.1)).astype(np.float32)

    # 3. 4-branch BP-Decimation Tree (branching on 2 least-confident step-0 variables)
    uncert_order = np.argsort(conf0, axis=1)
    u0 = uncert_order[:, 0]
    u1 = uncert_order[:, 1]
    s0_pref = np.where(h_step0[np.arange(B), u0] >= 0, 1.0, -1.0)
    s1_pref = np.where(h_step0[np.arange(B), u1] >= 0, 1.0, -1.0)
    branches = []
    for r, (m0, m1) in enumerate([(1.0, 1.0), (-1.0, 1.0), (1.0, -1.0), (-1.0, -1.0)]):
        p_mask = np.zeros((B, n), dtype=bool)
        p_sign = np.zeros((B, n), dtype=np.float32)
        p_mask[np.arange(B), u0] = True; p_sign[np.arange(B), u0] = s0_pref * m0
        p_mask[np.arange(B), u1] = True; p_sign[np.arange(B), u1] = s1_pref * m1
        for step in range(n - 2):
            h_cur = run_bp_once(p_mask, p_sign, iters=9, beta=3.0)
            conf = np.where(p_mask, -1.0, np.abs(h_cur))
            v_best = np.argmax(conf, axis=1)
            p_mask[np.arange(B), v_best] = True
            p_sign[np.arange(B), v_best] = np.where(h_cur[np.arange(B), v_best] >= 0, 1.0, -1.0)
        branches.append((p_sign * (conf0 + 0.1)).astype(np.float32))
    h_bp4 = np.stack(branches, axis=0)

    return h_bp, h_dec, h_bp4, pos_cnt, neg_cnt

for n in [6, 8]:
    t0 = time.time()
    D = np.load(os.path.join(OUT_DIR, f"data_n{n}.npz"))
    out = {}
    for split in ["train", "val", "test"]:
        h_bp, h_dec, h_bp4, pos, neg = compute_bp_all(D[f"{split}_cls_v"], D[f"{split}_cls_s"], n)
        out[f"{split}_h_bp"] = h_bp
        out[f"{split}_h_dec"] = h_dec
        out[f"{split}_h_bp4"] = h_bp4
        out[f"{split}_pos"] = pos
        out[f"{split}_neg"] = neg
    out_path = os.path.join(OUT_DIR, f"bp_n{n}.npz")
    np.savez_compressed(out_path, **out)
    print(f"n={n}: saved {out_path} in {time.time()-t0:.2f}s")
