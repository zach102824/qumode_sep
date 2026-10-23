"""Test pure compile-time clause-driven GSAT / Walk-prior (zero access to E or nsol)
on n=6 and n=8 test sets to see how close the compile-time polarity alignment is to a satisfying assignment."""
import os, time
import numpy as np

OUT_DIR = "/workspace/qumode_sep/oct_explore/results/oct23_exp"

def gsat_prior_batch(cls_v, cls_s, n, n_restarts=4, max_flips=8, seed=42):
    """Pure compile-time local clause-repair prior using ONLY cls_v (B, m, 4) and cls_s (B, m, 4).
    Does NOT use the 2^n energy table E. Evaluates only the m clauses on a single candidate n-bit vector.
    Actually wait: let's also compare against pure damped BP + decimation (message passing only, zero candidate evaluation)!
    """
    B, m, _ = cls_v.shape
    rng = np.random.default_rng(seed)

    # First: pure message-passing decimation (100% continuous message passing, no clause evaluation on bitstrings!)
    b_idx = np.arange(B)[:, None, None]
    pos_cnt = np.zeros((B, n), np.float32)
    neg_cnt = np.zeros((B, n), np.float32)
    np.add.at(pos_cnt, (b_idx, cls_v), (cls_s > 0).astype(np.float32))
    np.add.at(neg_cnt, (b_idx, cls_v), (cls_s < 0).astype(np.float32))

    # Decimated BP: fix variables one by one from most confident to least confident!
    pinned = np.zeros((B, n), dtype=bool)
    pin_val = np.zeros((B, n), dtype=np.float32)  # +1 or -1
    for step in range(n):
        eta = np.zeros((B, m, 4), np.float32)
        h0 = 0.15 * (pos_cnt - neg_cnt)
        for _ in range(12):
            h = h0.copy()
            h = np.where(pinned, 12.0 * pin_val, h)
            np.add.at(h, (b_idx, cls_v), cls_s * eta)
            h = np.where(pinned, 12.0 * pin_val, h)
            h_cav = h[b_idx, cls_v] - cls_s * eta
            q_false = 1.0 / (1.0 + np.exp(np.clip(3.0 * cls_s * h_cav, -20.0, 20.0)))
            new_eta = np.zeros_like(eta)
            for k in range(4):
                others = [j for j in range(4) if j != k]
                prod_other = q_false[:, :, others[0]] * q_false[:, :, others[1]] * q_false[:, :, others[2]]
                new_eta[:, :, k] = -0.5 * np.log(np.clip(1.0 - 0.96 * prod_other, 1e-5, 1.0))
            eta = 0.5 * eta + 0.5 * new_eta
        h_final = h0.copy()
        np.add.at(h_final, (b_idx, cls_v), cls_s * eta)
        # Pick unpinned variable with largest |h_final|
        conf = np.where(pinned, -1.0, np.abs(h_final))
        v_best = np.argmax(conf, axis=1)
        pinned[np.arange(B), v_best] = True
        pin_val[np.arange(B), v_best] = np.where(h_final[np.arange(B), v_best] >= 0, 1.0, -1.0)
        if step == 0:
            h_first = h_final.copy()

    return h_first, pin_val

for n in [6, 8]:
    D = np.load(os.path.join(OUT_DIR, f"data_n{n}.npz"))
    h_first, pin_val = gsat_prior_batch(D["test_cls_v"], D["test_cls_s"], n)
    Es = D["test_E"]
    B = len(Es)
    dH_dec = []
    for b in range(B):
        sols = np.where(Es[b] == 0)[0]
        sol_bits = np.array([[(s >> (n - 1 - v)) & 1 for v in range(n)] for s in sols])
        x_dec = (pin_val[b] > 0).astype(int)
        dH_dec.append(np.min(np.sum(np.abs(sol_bits - x_dec[None, :]), axis=1)))
    dH_dec = np.array(dH_dec)
    print(f"n={n} BP-Decimation: mean dH={dH_dec.mean():.2f}, med={np.median(dH_dec):.1f}, 90%={np.quantile(dH_dec, 0.9):.1f}, exact hit={np.mean(dH_dec==0):.2f}")

def bp_tree4(cls_v, cls_s, n):
    """Run 4 branches of BP-decimation:
    At step 0, run BP to get h0. Find the 2 least-confident variables u0, u1 (smallest |h0|).
    Branch r in {0, 1, 2, 3} pins (u0, u1) to one of the 4 sign combinations (+,+), (+,-), (-,+), (-,-),
    then runs BP-decimation on the remaining n-2 variables!
    Also returns the step-0 confidence order so we know which variables are least confident (fastest Gray bits).
    """
    B, m, _ = cls_v.shape
    b_idx = np.arange(B)[:, None, None]
    pos_cnt = np.zeros((B, n), np.float32)
    neg_cnt = np.zeros((B, n), np.float32)
    np.add.at(pos_cnt, (b_idx, cls_v), (cls_s > 0).astype(np.float32))
    np.add.at(neg_cnt, (b_idx, cls_v), (cls_s < 0).astype(np.float32))
    h0_base = 0.15 * (pos_cnt - neg_cnt)

    def run_bp_once(pinned_mask, pin_signs, iters=12):
        eta = np.zeros((B, m, 4), np.float32)
        for _ in range(iters):
            h = np.where(pinned_mask, 12.0 * pin_signs, h0_base)
            np.add.at(h, (b_idx, cls_v), cls_s * eta)
            h = np.where(pinned_mask, 12.0 * pin_signs, h)
            h_cav = h[b_idx, cls_v] - cls_s * eta
            q_false = 1.0 / (1.0 + np.exp(np.clip(3.0 * cls_s * h_cav, -20.0, 20.0)))
            new_eta = np.zeros_like(eta)
            for k in range(4):
                others = [j for j in range(4) if j != k]
                prod_other = q_false[:, :, others[0]] * q_false[:, :, others[1]] * q_false[:, :, others[2]]
                new_eta[:, :, k] = -0.5 * np.log(np.clip(1.0 - 0.96 * prod_other, 1e-5, 1.0))
            eta = 0.5 * eta + 0.5 * new_eta
        h_final = h0_base.copy()
        np.add.at(h_final, (b_idx, cls_v), cls_s * eta)
        return h_final

    h_init = run_bp_once(np.zeros((B, n), bool), np.zeros((B, n), np.float32), iters=16)
    uncert_order = np.argsort(np.abs(h_init), axis=1)  # (B, n) least confident first
    u0 = uncert_order[:, 0]
    u1 = uncert_order[:, 1]
    s0_pref = np.where(h_init[np.arange(B), u0] >= 0, 1.0, -1.0)
    s1_pref = np.where(h_init[np.arange(B), u1] >= 0, 1.0, -1.0)

    branches = []
    for r, (m0, m1) in enumerate([(1.0, 1.0), (-1.0, 1.0), (1.0, -1.0), (-1.0, -1.0)]):
        pinned = np.zeros((B, n), dtype=bool)
        pin_val = np.zeros((B, n), dtype=np.float32)
        pinned[np.arange(B), u0] = True; pin_val[np.arange(B), u0] = s0_pref * m0
        pinned[np.arange(B), u1] = True; pin_val[np.arange(B), u1] = s1_pref * m1
        for step in range(n - 2):
            h_cur = run_bp_once(pinned, pin_val, iters=10)
            conf = np.where(pinned, -1.0, np.abs(h_cur))
            v_best = np.argmax(conf, axis=1)
            pinned[np.arange(B), v_best] = True
            pin_val[np.arange(B), v_best] = np.where(h_cur[np.arange(B), v_best] >= 0, 1.0, -1.0)
        # Encode both confidence magnitude from h_init and final decimated sign from pin_val!
        branches.append(pin_val * (np.abs(h_init) + 0.1))
    return np.stack(branches, axis=0)

for n in [6, 8]:
    D = np.load(os.path.join(OUT_DIR, f"data_n{n}.npz"))
    br = bp_tree4(D["test_cls_v"], D["test_cls_s"], n)
    Es = D["test_E"]
    B = len(Es)
    dH_r0, dH_b4 = [], []
    for b in range(B):
        sols = np.where(Es[b] == 0)[0]
        sol_bits = np.array([[(s >> (n - 1 - v)) & 1 for v in range(n)] for s in sols])
        dHs = [np.min(np.sum(np.abs(sol_bits - (br[r, b] > 0).astype(int)[None, :]), axis=1)) for r in range(4)]
        dH_r0.append(dHs[0]); dH_b4.append(min(dHs))
    print(f"n={n} Tree4: r0 med/90%={np.median(dH_r0):.1f}/{np.quantile(dH_r0,0.9):.1f} (hit={np.mean(np.array(dH_r0)==0):.2f}) | best4 med/90%={np.median(dH_b4):.1f}/{np.quantile(dH_b4,0.9):.1f} (hit={np.mean(np.array(dH_b4)==0):.2f})")
