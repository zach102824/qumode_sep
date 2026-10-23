"""Add train_h_mpw4, val_h_mpw4, test_h_mpw4 and updated flip_step=(0,1,2) h_bp4 branches to bp_n6.npz and bp_n8.npz
while keeping h_bp, h_dec, pos, neg bit-for-bit identical."""
import os, time
import numpy as np

OUT_DIR = "/workspace/qumode_sep/oct_explore/results/oct23_exp"

def compute_mpw4_split(cls_v, cls_s, h_bp, n):
    B, m, _ = cls_v.shape
    b_idx = np.arange(B)[:, None, None]

    def run_minsum(pinned, pin_val, steps=28):
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
            h_reinf = (1.0 - gamma_r) * h_reinf + gamma_r * h_norm
        h_final = np.where(pinned, 20.0 * pin_val, h_reinf)
        np.add.at(h_final, (b_idx, cls_v), cls_s * eta)
        return h_final

    h_step0 = run_minsum(np.zeros((B, n), bool), np.zeros((B, n), np.float32), steps=30)

    def decimate_branch(flip_steps=()):
        pinned = np.zeros((B, n), bool); pin_val = np.zeros((B, n), np.float32)
        for step in range(n):
            h_cur = h_step0 if step == 0 else run_minsum(pinned, pin_val, steps=12)
            conf = np.where(pinned, -1.0, np.abs(h_cur))
            v_best = np.argmax(conf, axis=1)
            pinned[np.arange(B), v_best] = True
            sgn = np.where(h_cur[np.arange(B), v_best] >= 0, 1.0, -1.0)
            if step in flip_steps:
                sgn = -sgn
            pin_val[np.arange(B), v_best] = sgn
        beta = 8.0
        lit_sat = cls_s * (2.0 * pin_val[b_idx, cls_v])
        warn_contrad = np.zeros(B, np.float32)
        for k in range(4):
            others = [j for j in range(4) if j != k]
            o_stack = np.stack([lit_sat[:, :, others[0]], lit_sat[:, :, others[1]], lit_sat[:, :, others[2]]], axis=-1)
            max_other = np.max(o_stack, axis=-1)
            eta_end = np.log1p(np.exp(np.clip(-beta * max_other, -20.0, 20.0))) / beta
            warn_contrad += np.sum(eta_end * (lit_sat[:, :, k] < 0), axis=1)
        return pin_val, warn_contrad

    brs = [decimate_branch(fs) for fs in [(), (0,), (1,), (2,)]]
    p4 = np.stack([b[0] for b in brs], axis=0)  # (4, B, n)
    c4 = np.stack([b[1] for b in brs], axis=0)  # (4, B)
    w_c = np.exp(-5.0 * (c4 - np.min(c4, axis=0, keepdims=True)))
    w_c = w_c / np.sum(w_c, axis=0, keepdims=True)
    h_back = np.sum(w_c[:, :, None] * p4, axis=0)
    h_mpw4 = (np.where(h_back >= 0, 1.0, -1.0) * (np.abs(h_back) + 1e-3 * np.abs(h_bp))).astype(np.float32)
    h_bp4_new = (p4 * (np.abs(h_bp)[None, :, :] + 0.05)).astype(np.float32)
    return h_mpw4, h_bp4_new

if __name__ == "__main__":
    for n in [6, 8]:
        t0 = time.time()
        D = np.load(os.path.join(OUT_DIR, f"data_n{n}.npz"))
        bp_path = os.path.join(OUT_DIR, f"bp_n{n}.npz")
        BP = dict(np.load(bp_path))
        for split in ["train", "val", "test"]:
            h_mpw4, h_bp4_new = compute_mpw4_split(D[f"{split}_cls_v"], D[f"{split}_cls_s"], BP[f"{split}_h_bp"], n)
            BP[f"{split}_h_mpw4"] = h_mpw4
            BP[f"{split}_h_bp4"] = h_bp4_new
        tmp_path = os.path.join(OUT_DIR, f"bp_n{n}.tmp.npz")
        np.savez_compressed(tmp_path, **BP)
        os.replace(tmp_path, bp_path)
        print(f"n={n}: updated {bp_path} with h_mpw4 and root-flip h_bp4 in {time.time()-t0:.1f}s", flush=True)
