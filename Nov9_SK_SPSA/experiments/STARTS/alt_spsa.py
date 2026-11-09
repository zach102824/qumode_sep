"""Block-alternating C0 SPSA. Frozen spsa.py is not modified.

Odd k: update block A only. Even k: update block B only.
Calibration is the same 4 full-vector two-sided C0 probes (charged, not a JOINT K prefix).
Each K step still draws a full Rademacher vector, then zeros the inactive block.
"""
from __future__ import annotations

import numpy as np

import spsa as SP


def block_masks(name, L):
    if name == "Nov3":
        d = 3 + 3 * int(L)
        a = np.zeros(d)
        b = np.zeros(d)
        a[:3] = 1.0
        b[3:] = 1.0
    elif name == "QAOA":
        L = int(L)
        d = 2 * L
        a = np.zeros(d)
        b = np.zeros(d)
        a[:L] = 1.0
        b[L:] = 1.0
    else:
        raise ValueError(name)
    if a.size != b.size or a.size == 0 or np.any((a + b) != 1.0) or np.any(a * b != 0.0):
        raise ValueError("blocks must partition coordinates")
    return a, b


def spsa_start_alt(B, x0, sig, rng, K, mask_a, mask_b, a_override=None, ncal=4, A=25.0, c=1.0, target=0.5, clip=2.0, avg_tail=64):
    d = x0.size
    mask_a = np.asarray(mask_a, float)
    mask_b = np.asarray(mask_b, float)
    if mask_a.size != d or mask_b.size != d:
        raise ValueError("mask size")
    y = np.zeros(d)
    gabs = np.zeros(d)
    for _ in range(ncal):
        De = rng.choice([-1.0, 1.0], size=d)
        ep = B(x0 + sig * c * De, "calibration")
        em = B(x0 - sig * c * De, "calibration")
        gabs += np.abs((ep - em) / (2 * c) * De)
    gabs /= ncal
    med = float(np.median(gabs))
    a_cal = target * (A + 1.0) ** 0.602 / max(med, 1e-12)
    a = a_cal if a_override is None else float(a_override)
    hist = []
    n_a = n_b = 0
    for k in range(K):
        ak = a / (k + 1.0 + A) ** 0.602
        ck = c / (k + 1.0) ** 0.101
        De = rng.choice([-1.0, 1.0], size=d)
        if k % 2 == 0:
            De = De * mask_a
            n_a += 1
        else:
            De = De * mask_b
            n_b += 1
        ep = B(x0 + sig * (y + ck * De), "perturb")
        em = B(x0 + sig * (y - ck * De), "perturb")
        g = (ep - em) / (2 * ck) * De
        y = y - np.clip(ak * g, -clip, clip)
        hist.append(y.copy())
    tail = hist[-min(avg_tail, len(hist)):] if hist else [y.copy()]
    cands = [("last", y.copy()), ("polyak", np.mean(tail, axis=0))]
    out = []
    for name, yy in cands:
        xx = x0 + sig * yy
        out.append((B(xx, "candidate"), name, xx))
    meta = {
        "a_used": float(a), "a_calibrated": float(a_cal), "median_absg": med,
        "gain_frozen": a_override is not None, "schedule": "odd=A even=B",
        "n_block_A_steps": n_a, "n_block_B_steps": n_b,
        "mask_A": mask_a.tolist(), "mask_B": mask_b.tolist(),
        "joint_prefix_K": 0,
    }
    return out, meta


def spsa_run_alt(f, starts, sig, rng_for_start, cap, mask_a, mask_b, a_override=None):
    per, K = SP.split_cap(cap, len(starts))
    B = SP.Budget(f, cap)
    allc, meta = [], []
    for si, x0 in enumerate(starts):
        cands, m = spsa_start_alt(
            B, np.asarray(x0, float), sig, rng_for_start(si), K, mask_a, mask_b, a_override=a_override
        )
        for e, name, xx in cands:
            allc.append((e, si, name, xx))
        meta.append(m)
    e, si, name, xx = min(allc, key=lambda t: t[0])
    return {
        "E_sel": e, "start": si, "cand": name, "x": xx, "queries": B.n, "query_cat": dict(B.cat),
        "K_per_start": K, "per_start": per, "start_meta": meta, "all_cand_E": [t[0] for t in allc],
    }
