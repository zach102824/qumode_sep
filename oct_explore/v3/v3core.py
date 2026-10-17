"""v3 core: batched simulator for the fixed hardware structure (Zach-approved 15:18 CST 2026-10-17).

Per layer l:  on every register c (cavity + its own transmon):  R(th,ph) on transmon -> ECD(beta) on (transmon,cavity)
              then ONE joint cavity unitary  U_l = Mixer_l . exp(-i Phase_l(x))   (diagonal cost phase first, then X-mixer)
Final block:  R + ECD on every register.
Phase_l(x) = (n/m) * sum_c w_{c,l} [clause c violated by x]   (plain: w_{c,l} = g_l  ->  g_l * E(x) n/m, as costgx)
Mixer_l    = prod_c prod_j exp(-i b_{c,j,l}/2 Xt_{c,j}),  Xt_{c,j} flips logical bit j of cavity c in its Fock code
             (gray: Fock f <-> logical gray(f); binary: f <-> f). All Xt commute; pads (Fock >= 2^k) untouched.
             plain: all b_{c,j,l} = bc_l (== exp(-i bc/2 sum_j Xt_j), identical to costgx).
Variable map (LAYOUTS_T, var0 = MSB): register c occupies x-bits [pos_c, pos_c+k_c] ; cavity digit j <-> var n-1-(pos_c+j),
             transmon <-> var n-1-(pos_c+k_c).  Layout.cvar / Layout.tvar hold these.
All parameter arrays carry a leading instance-batch axis B (plain = broadcast), so per-instance (GNN) params are native.
loc  : (B, L+1, C, 4) = [bre, bim, th, ph] per block/register (ECD beta = (bre + i bim) * sqrt(2^k)/2 as costgx)
mix  : (B, L, C, KMAX) per-digit mixer angles
phase: list of L arrays (B, 2^n) in logical order (pads get max over x, as costgx)
"""
from __future__ import annotations
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, jax, jax.numpy as jnp
import sim

C64 = jnp.complex64


def gray_s(i): return i ^ (i >> 1)
def ungray_s(g):
    g = int(g); s = g >> 1
    while s: g ^= s; s >>= 1
    return g


class Layout:
    def __init__(self, n, code="gray"):
        ks = sim.LAYOUTS_T[n]; self.n, self.ks, self.C, self.code = n, list(ks), len(ks), code
        self.kmax = max(ks)
        self.nfs = [(1 << k) + max(8, (1 << k) // 4) for k in ks]
        x = np.arange(1 << n); pos = n; tidx, fidx = [], []; self.tvar, self.cvar = [], []
        for c, k in enumerate(ks):
            nb = k + 1; pos -= nb; val = (x >> pos) & ((1 << nb) - 1)
            tidx.append(val >> k); f = val & ((1 << k) - 1)
            fidx.append(sim.ungray(f) if code == "gray" else f)
            self.tvar.append(n - 1 - (pos + k)); self.cvar.append([n - 1 - (pos + j) for j in range(k)])
        self.shape = [2] * self.C + self.nfs; self.D = int(np.prod(self.shape))
        flat = np.ravel_multi_index(tuple(tidx + fidx), self.shape)
        idx = np.full(self.D, 1 << n, np.int64); idx[flat] = x; self.idxD = idx
        psi0 = np.zeros(self.D, np.complex64); psi0[flat] = 1 / np.sqrt(len(flat)); self.psi0 = jnp.asarray(psi0.reshape(self.shape))
        self.P, self.Sub, self.Pad, self.Vd, self.lamd, self.nv, self.sc = [], [], [], [], [], [], []
        for k, nf in zip(ks, self.nfs):
            N = 1 << k; Ps = np.zeros((k, nf, nf), np.float32)
            for j in range(k):
                for f in range(N):
                    f2 = ungray_s(gray_s(f) ^ (1 << j)) if code == "gray" else f ^ (1 << j)
                    Ps[j, f2, f] = 1.0
            self.P.append(jnp.asarray(Ps)); d = (np.arange(nf) < N).astype(np.float32)
            self.Sub.append(jnp.asarray(np.diag(d))); self.Pad.append(jnp.asarray(np.diag(1 - d)))
            aa = np.diag(np.sqrt(np.arange(1, nf)), 1); lam, V = np.linalg.eigh(1j * (aa.T - aa))
            self.Vd.append(jnp.asarray(V.astype(np.complex64))); self.lamd.append(jnp.asarray(lam.astype(np.float32)))
            self.nv.append(jnp.arange(nf, dtype=jnp.float32)); self.sc.append(np.sqrt(2.0 ** k) / 2.0)

    # ---- data ----
    def succ_D(self, insts):
        S = np.stack([(i["E"] == 0).astype(np.float32) for i in insts]); S = np.concatenate([S, np.zeros((len(insts), 1), np.float32)], 1)
        return S[:, self.idxD].reshape([len(insts)] + self.shape)

    def to_D(self, ph):  # ph (B, 2^n) jnp/np -> (B, *shape), pad = max over x
        ph = jnp.concatenate([ph, jnp.max(ph, axis=1, keepdims=True)], 1)
        return ph[:, self.idxD].reshape((ph.shape[0],) + tuple(self.shape))

    # ---- gates (batched) ----
    def bapply(self, M, psi, ax):  # M (B,d,d) or (1,d,d) shared, psi (B,...), act on axis ax (>=1)
        L = "abcdefghijk"[: psi.ndim - 1]; o = L[ax - 1]
        if M.shape[0] == 1 and psi.shape[0] != 1:
            return jnp.einsum(f"Y{o},z{L}->z{L.replace(o, 'Y')}", M[0], psi)
        return jnp.einsum(f"zY{o},z{L}->z{L.replace(o, 'Y')}", M, psi)

    def Rm(self, th, ph):
        ct, st = jnp.cos(th / 2), jnp.sin(th / 2)
        e = jnp.exp(-1j * ph)
        return jnp.stack([jnp.stack([ct + 0j, -1j * st * e], -1), jnp.stack([-1j * st * jnp.conj(e), ct + 0j], -1)], -2).astype(C64)

    def disp(self, c, re, im):
        r = jnp.sqrt(re ** 2 + im ** 2 + 1e-20) * self.sc[c]; ph = jnp.arctan2(im, re)
        W = jnp.exp(1j * ph[:, None] * self.nv[c][None])[:, :, None] * self.Vd[c][None]
        return jnp.einsum("bik,bk,bjk->bij", W, jnp.exp(-1j * r[:, None] * self.lamd[c][None]), jnp.conj(W)).astype(C64)

    def mixU(self, c, b):  # b (B,k)
        U = None
        for j in range(self.ks[c]):
            cj, sj = jnp.cos(b[:, j] / 2), jnp.sin(b[:, j] / 2)
            Uj = (cj[:, None, None] * self.Sub[c][None] - 1j * sj[:, None, None] * self.P[c][j][None] + self.Pad[c][None]).astype(C64)
            U = Uj if U is None else jnp.einsum("bij,bjk->bik", Uj, U)
        return U

    def local(self, psi, lb):  # lb (B, C, 4)
        C = self.C
        for c in range(C):
            psi = self.bapply(self.Rm(lb[:, c, 2], lb[:, c, 3]), psi, 1 + c)
            Dp = self.disp(c, lb[:, c, 0], lb[:, c, 1]); Dh = jnp.conj(jnp.swapaxes(Dp, 1, 2))
            p0 = jnp.take(psi, 0, axis=1 + c); p1 = jnp.take(psi, 1, axis=1 + c)
            psi = jnp.stack([self.bapply(Dh, p1, C + c), self.bapply(Dp, p0, C + c)], axis=1 + c)
        return psi

    def probs(self, loc, mix, phase_fn, succ):
        """phase_fn(l) -> (B, *shape) real phase for layer l. Returns p (B,)."""
        B = succ.shape[0]; L = loc.shape[1] - 1  # loc/mix may have batch 1 (shared params)
        psi = jnp.broadcast_to(self.psi0, (B,) + tuple(self.shape))
        for l in range(L):
            psi = self.local(psi, loc[:, l])
            psi = psi * jnp.exp(-1j * phase_fn(l)).astype(C64)
            for c in range(self.C):
                psi = self.bapply(self.mixU(c, mix[:, l, c, : self.ks[c]]), psi, 1 + self.C + c)
        psi = self.local(psi, loc[:, L])
        return jnp.sum((jnp.abs(psi) ** 2) * succ, axis=tuple(range(1, psi.ndim)))


# ---- clause structure ----
def clauses(n, ins):
    m, cl, _ = sim.gen_instance(n, ins["seed"]); return m, cl


def viol_matrix(n, cl):  # (m, 2^n) float32 indicator, var0 = MSB
    x = np.arange(1 << n); out = np.ones((len(cl), 1 << n), np.float32)
    for i, (v, s) in enumerate(cl):
        for vv, ss in zip(v, s):
            bit = (x >> (n - 1 - vv)) & 1
            out[i] *= (bit == (0 if ss > 0 else 1))  # violated literal: s>0 needs x_v=1, violated if 0
    return out


def weighted_energy(n, cl, w):  # numpy, w (m,) -> (2^n,)
    T = np.zeros([2] * n, np.float32)
    for (v, s), ww in zip(cl, w):
        ix = [slice(None)] * n
        for vv, ss in zip(v, s): ix[vv] = 0 if ss > 0 else 1
        T[tuple(ix)] += ww
    return T.reshape(-1)


def B90(inv): return float(np.quantile(inv, 0.9))
def summ(p):
    p = np.asarray(p, float); inv = 1 / np.maximum(p, 1e-300)
    return {"median_inv_p": float(np.median(inv)), "B90_inv_p": B90(inv), "mean_p": float(p.mean()), "p": [float(v) for v in p]}
