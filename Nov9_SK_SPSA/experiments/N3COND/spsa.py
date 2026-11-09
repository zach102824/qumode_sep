"""Scalar SPSA (C0) and diagonally scaled SPSA (C1).

Both arms spend the same queries inside the matched cap: 4 starts, each
8 calibration + 2K perturbations + 2 candidates. C1 still executes the 8
calibration queries and counts them. It does not use them to set the gain.
The gain and the per-coordinate scale are the frozen dev values.
"""
from __future__ import annotations

import numpy as np


class Budget:
    def __init__(self, f, cap):
        self.f, self.cap, self.n = f, cap, 0
        self.cat = {"calibration": 0, "perturb": 0, "candidate": 0, "monitor": 0}

    def __call__(self, x, cat):
        if self.n >= self.cap:
            raise RuntimeError("query cap exceeded")
        self.n += 1
        self.cat[cat] += 1
        e = float(self.f(x))
        if not np.isfinite(e):
            raise FloatingPointError("non-finite energy")
        return e


def spsa_start(B, x0, sig, rng, K, a_override=None, ncal=4, A=25.0, c=1.0, target=0.5, clip=2.0, avg_tail=64):
    d = x0.size
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
    for k in range(K):
        ak = a / (k + 1.0 + A) ** 0.602
        ck = c / (k + 1.0) ** 0.101
        De = rng.choice([-1.0, 1.0], size=d)
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
    meta = {"a_used": float(a), "a_calibrated": float(a_cal), "median_absg": med, "gain_frozen": a_override is not None}
    return out, meta


def split_cap(cap, nstarts=4, ncal=4):
    if cap % nstarts != 0:
        raise ValueError(f"cap {cap} not divisible by {nstarts} starts")
    per = cap // nstarts
    K = (per - 2 * ncal - 2) // 2
    if 2 * ncal + 2 * K + 2 != per or K < 1:
        raise ValueError(f"cap {cap} does not give a positive integer K (per={per}, K={K})")
    return per, K


def spsa_run(f, starts, sig, rng_for_start, cap, a_override=None):
    per, K = split_cap(cap, len(starts))
    B = Budget(f, cap)
    allc = []
    meta = []
    for si, x0 in enumerate(starts):
        cands, m = spsa_start(B, np.asarray(x0, float), sig, rng_for_start(si), K, a_override=a_override)
        for e, name, xx in cands:
            allc.append((e, si, name, xx))
        meta.append(m)
    e, si, name, xx = min(allc, key=lambda t: t[0])
    return {
        "E_sel": e, "start": si, "cand": name, "x": xx, "queries": B.n, "query_cat": dict(B.cat),
        "K_per_start": K, "per_start": per, "start_meta": meta, "all_cand_E": [t[0] for t in allc],
    }


def probes_at(f, x0, sig, rng, nprobe, c=1.0):
    """nprobe two-sided SPSA probes. Returns (nprobe, dim) absolute components. 2*nprobe queries."""
    rows = []
    x0 = np.asarray(x0, float)
    sig = np.asarray(sig, float)
    for _ in range(nprobe):
        De = rng.choice([-1.0, 1.0], size=x0.size)
        ep = float(f(x0 + sig * c * De))
        em = float(f(x0 - sig * c * De))
        if not np.isfinite(ep) or not np.isfinite(em):
            raise FloatingPointError("non-finite scale probe")
        rows.append(np.abs((ep - em) / (2 * c) * De))
    return np.vstack(rows)


LAM_LO, LAM_HI = 0.25, 4.0
TARGET, A_SPALL = 0.5, 25.0


def freeze_lambda(gabs_rows):
    """Per-coordinate scale from a stack of |g| rows, then the matching frozen gain.

    g_j = median_row |sigma_j dE/dx_j| at the base sigma.
    lambda_j = clip(median_j(g) / g_j, 1/4, 4).
    a0 uses the median of lambda*g, which is the predicted equalized |g| after clipping.
    """
    g = np.median(np.asarray(gabs_rows, float), axis=0)
    g = np.maximum(g, 1e-12)
    g_ref = float(np.median(g))
    lam = np.clip(g_ref / g, LAM_LO, LAM_HI)
    g_eq = lam * g
    med_eq = float(np.median(g_eq))
    a0 = TARGET * (A_SPALL + 1.0) ** 0.602 / max(med_eq, 1e-12)
    return lam, a0, g, g_ref
