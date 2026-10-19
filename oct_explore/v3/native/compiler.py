"""Milestone 2: deterministic compiler for the templated native ansatz (no learning).

Variables -> cavities: cavity c holds K_c bits (Gray code inside the cavity). Default assignment = contiguous blocks
(cavity 0 = vars 0..K_0-1 = MSBs of the sim.fast_energies index). A general assignment is a permutation `perm`
(position q in the chain-ordered bit list holds variable perm[q]); we apply it by permuting the energy vector.
Representable diagonals on the chain = span{ functions of one cavity } + span{ functions of two NEIGHBOUR cavities }.
In the Walsh (parity) basis chi_S(x) = prod_{i in S} (1-2x_i) this is exactly span{chi_S : cavities(S) subset of {c} or {c,c+1}},
so the least-squares (uniform measure) projection P of any cost vector = FWHT -> zero the non-representable coefficients -> iFWHT.
Clauses inside one cavity / one neighbour pair are reproduced exactly; others (non-neighbour, 3+ cavities) are projected.
compile_tables(): P C -> per-cavity SNAP table theta_c(level) and per-pair coupling table PhiTilde_c(kA,kB) (first row/col 0,
local parts folded into the SNAPs), checked to sum back to P C.
"""
from __future__ import annotations
import numpy as np

LAYOUTS = {6: [3, 3], 8: [4, 4], 9: [3, 3, 3], 10: [5, 5], 12: [4, 4, 4], 14: [5, 5, 4], 15: [5, 5, 5], 16: [4, 4, 4, 4], 18: [5, 5, 4, 4], 20: [5, 5, 5, 5]}
gray = lambda i: i ^ (i >> 1)

def ungray_arr(g):
    g = np.array(g, copy=True); s = g >> 1
    while np.any(s): g ^= s; s >>= 1
    return g

def fwht(a):
    """unnormalised Walsh-Hadamard along the last axis (length 2^n); self-inverse up to 1/2^n."""
    a = np.array(a, dtype=np.float64, copy=True); h = 1; N = a.shape[-1]; sh = a.shape[:-1]
    while h < N:
        a = a.reshape(sh + (N // (2 * h), 2, h)); x, y = a[..., 0, :].copy(), a[..., 1, :].copy()
        a[..., 0, :] = x + y; a[..., 1, :] = x - y; a = a.reshape(sh + (N,)); h *= 2
    return a

_MASK = {}
def rep_mask(ks, mode="chain"):
    """bool over Walsh index s (same bit layout as x: cavity 0 = top bits). mode chain | local (bus off) | full."""
    key = (tuple(ks), mode)
    if key in _MASK: return _MASK[key]
    n = sum(ks); s = np.arange(1 << n); occ = np.zeros(1 << n, np.int64); pos = n
    for c, k in enumerate(ks):
        pos -= k; occ |= (((s >> pos) & ((1 << k) - 1)) != 0).astype(np.int64) << c
    pc = np.zeros_like(occ)
    for c in range(len(ks)): pc += (occ >> c) & 1
    if mode == "full": m = np.ones(1 << n, bool)
    elif mode == "pairs": m = pc <= 2          # any two cavities (e.g. chain + cavity SWAP network)
    elif mode == "local": m = pc <= 1
    else:
        adj = np.zeros_like(occ, dtype=bool)
        for c in range(len(ks) - 1): adj |= occ == (3 << c)
        m = (pc <= 1) | adj
    _MASK[key] = m; return m

def project(E, ks, mode="chain"):
    """least-squares projection of cost vector(s) E (..., 2^n) onto the representable space."""
    W = fwht(E); W = W * rep_mask(ks, mode); return fwht(W) / E.shape[-1]

def residual_frac(E, ks, mode="chain"):
    """||E - P E||^2 / ||E - mean E||^2 (fraction of the cost variance the chain cannot apply)."""
    W = fwht(E); W[..., 0] = 0; tot = (W ** 2).sum(-1); return ((W ** 2) * ~rep_mask(ks, mode)).sum(-1) / tot

def permute_vars(E, n, perm):
    """E indexed by x with var v at bit (n-1-v). Return E' with chain position q holding variable perm[q]:
    E'[x'] = E[x] where bit (n-1-q) of x' = bit (n-1-perm[q]) of x."""
    x = np.arange(1 << n); xo = np.zeros_like(x)
    for q, v in enumerate(perm): xo |= ((x >> (n - 1 - q)) & 1) << (n - 1 - v)
    return E[..., xo]

def compile_tables(PE, ks):
    """PE: projected cost (2^n,) in chain order. -> (snap list over c of (2^K_c,) indexed by LEVEL f, bus list of (N_c, N_{c+1})
    coupling tables indexed by levels, const). Uses ANOVA on the Walsh coefficients grouped by cavity support."""
    n = sum(ks); C = len(ks); W = fwht(PE) / (1 << n); s = np.arange(1 << n); pos = n; occ = np.zeros(1 << n, np.int64); offs = []
    for c, k in enumerate(ks):
        pos -= k; offs.append(pos); occ |= (((s >> pos) & ((1 << k) - 1)) != 0).astype(np.int64) << c
    const = W[0]; snaps, buses = [], []
    for c, k in enumerate(ks):
        Wc = np.where(occ == (1 << c), W, 0); fx = fwht(Wc)           # function of x depending only on cavity c's bits
        g = np.arange(1 << k); xs = g << offs[c]; tab_logical = fx[xs]  # value at logical cavity value g (other bits 0)
        snaps.append(tab_logical[gray(np.arange(1 << k))])            # level f holds logical digits gray(f)
    for c in range(C - 1):
        Wp = np.where(occ == (3 << c), W, 0); fx = fwht(Wp); kA, kB = ks[c], ks[c + 1]
        gA = np.arange(1 << kA)[:, None]; gB = np.arange(1 << kB)[None, :]; xs = (gA << offs[c]) | (gB << offs[c + 1])
        T = fx[xs]; T = T[gray(np.arange(1 << kA))][:, gray(np.arange(1 << kB))]
        T = T - T[:, :1] - T[:1, :] + T[0, 0]                         # coupling part; local parts are in the snaps (T[.,0] etc. are 0 here
        buses.append(T)                                               #  already up to the ANOVA convention -> fold residual into snaps below)
    # fold the (row/col) parts removed from the bus back into the SNAPs so that the decomposition is exact
    # (with the Walsh/ANOVA convention a pure pair term has zero mean over each argument, the shift goes to the snaps)
    for c in range(C - 1):
        Wp = np.where(occ == (3 << c), W, 0); fx = fwht(Wp); kA, kB = ks[c], ks[c + 1]
        gA = np.arange(1 << kA)[:, None]; gB = np.arange(1 << kB)[None, :]; xs = (gA << offs[c]) | (gB << offs[c + 1])
        T0 = fx[xs][gray(np.arange(1 << kA))][:, gray(np.arange(1 << kB))]
        snaps[c] = snaps[c] + T0[:, 0] - T0[0, 0]; snaps[c + 1] = snaps[c + 1] + T0[0, :]
    return snaps, buses, const

def reassemble(snaps, buses, const, ks):
    n = sum(ks); x = np.arange(1 << n); pos = n; lev = []
    for k in ks:
        pos -= k; lev.append(ungray_arr((x >> pos) & ((1 << k) - 1)))
    out = np.full(1 << n, const, float)
    for c in range(len(ks)): out += snaps[c][lev[c]]
    for c in range(len(ks) - 1): out += buses[c][lev[c], lev[c + 1]]
    return out

def project_product(E, ks, p, mode="chain"):
    """least-squares projection of E (2^n,) under the PRODUCT measure mu(x) = prod_i p_i^{x_i} (1-p_i)^{1-x_i}
    (p_i = P(x_i = 1), var i at bit n-1-i). Orthonormal biased parity basis phi_i = (s_i - m_i)/sqrt(1-m_i^2), s = 1-2x, m = 1-2p.
    p = 1/2 recovers project(). Same representable mask (cavity support)."""
    n = sum(ks); T = np.asarray(E, float).reshape((2,) * n)
    for i in range(n):
        pi = float(np.clip(p[i], 1e-4, 1 - 1e-4)); m = 1 - 2 * pi; sd = np.sqrt(1 - m * m)
        ph0, ph1 = (1 - m) / sd, (-1 - m) / sd                     # phi at x=0 (s=+1), x=1 (s=-1)
        A = np.array([[1 - pi, pi], [(1 - pi) * ph0, pi * ph1]])  # values -> (c0, c1)
        T = np.moveaxis(np.tensordot(A, T, axes=([1], [i])), 0, i)
    T = T.reshape(-1) * rep_mask(ks, mode); T = T.reshape((2,) * n)
    for i in range(n):
        pi = float(np.clip(p[i], 1e-4, 1 - 1e-4)); m = 1 - 2 * pi; sd = np.sqrt(1 - m * m); ph0, ph1 = (1 - m) / sd, (-1 - m) / sd
        B = np.array([[1, ph0], [1, ph1]])                          # coeffs -> values
        T = np.moveaxis(np.tensordot(B, T, axes=([1], [i])), 0, i)
    return T.reshape(-1)
