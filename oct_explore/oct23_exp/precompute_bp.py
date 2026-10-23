"""Precompute compile-time variable features and message-passing marginals (Damped BP, Soft Min-Sum Decimation,
and 4-branch Decimated Tree) for n=6 and n=8 datasets (train=10,000, val=200, test=100) and save to bp_n{n}.npz.
All quantities use ONLY the factor graph (cls_v, cls_s) — zero access to true E or nsol."""
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

    def run_minsum(pinned, pin_val, steps=35, cubic=False):
        eta = np.zeros((B, m, 4), np.float32)
        h_reinf = np.zeros((B, n), np.float32)
        for step in range(steps):
            t = step / float(steps)
            beta = 0.6 + 6.0 * (t ** 1.3)
            gamma_r = 0.10 * (t ** 2)
            h = np.where(pinned, 20.0 * pin_val, h_reinf)
            np.add.at(h, (b_idx, cls_v), cls_s * eta)
            h = np.where(pinned, 20.0 * pin_val, h)
            norm = np.maximum(np.max(np.abs(h), axis=1, keepdims=True), 1.0)
            h_norm = h / norm
            h_cav = h_norm[b_idx, cls_v] - (cls_s * eta) / norm[:, :, None]
            h_cav = np.where(pinned[b_idx, cls_v], 2.0 * pin_val[b_idx, cls_v], h_cav)
            lit_sat = cls_s * h_cav
            new_eta = np.zeros_like(eta)
            for k in range(4):
                others = [j for j in range(4) if j != k]
                o_stack = np.stack([lit_sat[:, :, others[0]], lit_sat[:, :, others[1]], lit_sat[:, :, others[2]]], axis=-1)
                mx = np.max(o_stack, axis=-1, keepdims=True)
                max_other = mx[..., 0] + (1.0 / beta) * np.log(np.sum(np.exp(beta * (o_stack - mx)), axis=-1))
                new_eta[:, :, k] = np.log1p(np.exp(np.clip(-beta * max_other, -20.0, 20.0))) / beta
            eta = 0.6 * eta + 0.4 * new_eta
            h_reinf = (1.0 - gamma_r) * h_reinf + gamma_r * (h_norm ** 3 if cubic else h_norm)
        h_final = np.where(pinned, 20.0 * pin_val, h_reinf)
        np.add.at(h_final, (b_idx, cls_v), cls_s * eta)
        return h_final.astype(np.float32)

    # 1. Continuous Cubic-Reinforced Soft Min-Sum (h_bp)
    h_bp = run_minsum(np.zeros((B, n), bool), np.zeros((B, n), np.float32), steps=45, cubic=True)

    # 2. Soft Min-Sum Decimation (1-branch) combined with Cubic-Reinf confidence ordering
    pinned = np.zeros((B, n), bool)
    pin_val = np.zeros((B, n), np.float32)
    h_step0 = run_minsum(pinned, pin_val, steps=30, cubic=False)
    for step in range(n):
        h_cur = h_step0 if step == 0 else run_minsum(pinned, pin_val, steps=14, cubic=False)
        conf = np.where(pinned, -1.0, np.abs(h_cur))
        v_best = np.argmax(conf, axis=1)
        pinned[np.arange(B), v_best] = True
        pin_val[np.arange(B), v_best] = np.where(h_cur[np.arange(B), v_best] >= 0, 1.0, -1.0)
    h_dec = (pin_val * (np.abs(h_bp) + 0.05)).astype(np.float32)

    # 3. 4-branch Soft Min-Sum Tree (branching on 2 least-confident variables from h_bp)
    uncert_order = np.argsort(np.abs(h_bp), axis=1)
    u0 = uncert_order[:, 0]
    u1 = uncert_order[:, 1]
    s0_pref = np.where(h_dec[np.arange(B), u0] >= 0, 1.0, -1.0)
    s1_pref = np.where(h_dec[np.arange(B), u1] >= 0, 1.0, -1.0)
    branches = [h_dec]
    for m0, m1 in [(-1.0, 1.0), (1.0, -1.0), (-1.0, -1.0)]:
        p_mask = np.zeros((B, n), bool)
        p_sign = np.zeros((B, n), np.float32)
        p_mask[np.arange(B), u0] = True; p_sign[np.arange(B), u0] = s0_pref * m0
        p_mask[np.arange(B), u1] = True; p_sign[np.arange(B), u1] = s1_pref * m1
        for step in range(n - 2):
            h_cur = run_minsum(p_mask, p_sign, steps=10, cubic=False)
            conf = np.where(p_mask, -1.0, np.abs(h_cur))
            v_best = np.argmax(conf, axis=1)
            p_mask[np.arange(B), v_best] = True
            p_sign[np.arange(B), v_best] = np.where(h_cur[np.arange(B), v_best] >= 0, 1.0, -1.0)
        branches.append((p_sign * (np.abs(h_bp) + 0.05)).astype(np.float32))
    h_bp4 = np.stack(branches, axis=0)

    return h_bp, h_dec, h_bp4, pos_cnt, neg_cnt

if __name__ == "__main__":
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
        print(f"n={n}: saved {out_path} in {time.time()-t0:.2f}s", flush=True)
