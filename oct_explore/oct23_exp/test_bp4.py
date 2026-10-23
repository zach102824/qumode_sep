import os, time
import numpy as np

OUT_DIR = "/workspace/qumode_sep/oct_explore/results/oct23_exp"

def compute_bp4(cls_v, cls_s, n, iters=20, damping=0.5, beta=2.8):
    """Returns 4 deterministic BP marginal vectors (4, B, n) per formula:
    r=0: damped BP from literal bias init
    r=1..3: damped BP where the 1st, 2nd, 3rd lowest-confidence variable from r=0 is pinned to opposite sign!
    """
    B, m, _ = cls_v.shape
    b_idx = np.arange(B)[:, None, None]
    pos_cnt = np.zeros((B, n), np.float32)
    neg_cnt = np.zeros((B, n), np.float32)
    np.add.at(pos_cnt, (b_idx, cls_v), (cls_s > 0).astype(np.float32))
    np.add.at(neg_cnt, (b_idx, cls_v), (cls_s < 0).astype(np.float32))
    h0 = 0.15 * (pos_cnt - neg_cnt)

    def run_bp(pin_var=None, pin_sign=None):
        eta = np.zeros((B, m, 4), np.float32)
        for _ in range(iters):
            h = h0.copy()
            if pin_var is not None:
                h[np.arange(B), pin_var] += 6.0 * pin_sign
            np.add.at(h, (b_idx, cls_v), cls_s * eta)
            h_cav = h[b_idx, cls_v] - cls_s * eta
            q_false = 1.0 / (1.0 + np.exp(np.clip(beta * cls_s * h_cav, -20.0, 20.0)))
            new_eta = np.zeros_like(eta)
            for k in range(4):
                others = [j for j in range(4) if j != k]
                prod_other = q_false[:, :, others[0]] * q_false[:, :, others[1]] * q_false[:, :, others[2]]
                new_eta[:, :, k] = -0.5 * np.log(np.clip(1.0 - 0.95 * prod_other, 1e-5, 1.0))
            eta = (1.0 - damping) * eta + damping * new_eta
        h_final = h0.copy()
        if pin_var is not None:
            h_final[np.arange(B), pin_var] += 6.0 * pin_sign
        np.add.at(h_final, (b_idx, cls_v), cls_s * eta)
        return h_final

    h_r0 = run_bp()
    order_uncert = np.argsort(np.abs(h_r0), axis=1)  # (B, n) least confident first
    hs = [h_r0]
    for r in range(1, 4):
        v_pin = order_uncert[:, r - 1]
        s_pin = np.where(h_r0[np.arange(B), v_pin] >= 0, -1.0, 1.0)
        hs.append(run_bp(v_pin, s_pin))
    return np.stack(hs, axis=0)  # (4, B, n)

for n in [6, 8]:
    D = np.load(os.path.join(OUT_DIR, f"data_n{n}.npz"))
    hs = compute_bp4(D["test_cls_v"], D["test_cls_s"], n)
    Es = D["test_E"]
    B = len(Es)
    min_dH_r0, min_dH_best4 = [], []
    for b in range(B):
        sols = np.where(Es[b] == 0)[0]
        sol_bits = np.array([[(s >> (n - 1 - v)) & 1 for v in range(n)] for s in sols])
        dHs = []
        for r in range(4):
            x_r = (hs[r, b] >= 0).astype(int)
            dHs.append(np.min(np.sum(np.abs(sol_bits - x_r[None, :]), axis=1)))
        min_dH_r0.append(dHs[0])
        min_dH_best4.append(min(dHs))
    print(f"n={n}: r=0 dH med/90%/max = {np.median(min_dH_r0):.1f}/{np.quantile(min_dH_r0,0.9):.1f}/{np.max(min_dH_r0)} | best-of-4 dH med/90%/max = {np.median(min_dH_best4):.1f}/{np.quantile(min_dH_best4,0.9):.1f}/{np.max(min_dH_best4)} (exact hit={np.mean(np.array(min_dH_best4)==0):.2f})")
