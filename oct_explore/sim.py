"""ECD + diagonal cost-phase circuit on C independent (transmon, cavity) registers, no inter-cavity coupling.

Register c holds k_c bits in cavity Fock levels 0..2^k_c-1 (binary or Gray) plus optionally 1 bit in its control
transmon (tbit=True). Cavity simulated with nf_c = 2^k_c + pad levels; leaked population (Fock >= 2^k_c) = failure.
Layer l (parameters SHARED by all cavities, so they are defined for any number of cavities):
    R(theta_l, phi_l) on every transmon -> ECD(beta_c) on every pair, beta_c = (bre+i bim) * sqrt(2^k_c)
    -> exp(-i gamma_l * Cn(x)) diagonal on the joint computational space.
ECD(b) = |1><0| D(b/2) + |0><1| D(-b/2)  (same as noiseless/circuit_local_ecd.py).
Cost: 'raw'  Cn = E (unsat clause count) * n/m   (clause-density normalised; ~E/9.93)
      'std'  Cn = (E - mean E)/std E over all 2^n strings.
Leaked levels get phase of max(Cn). Cavity-only layouts: transmon bits are ancillas; cost does not depend on them,
success marginalises over them.
"""
from __future__ import annotations
import random
import numpy as np
import jax, jax.numpy as jnp

NP = 5  # per layer: bre, bim, theta, phi, gamma


def fast_energies(n, cl):
    T = np.zeros([2] * n, np.int16)
    for v, s in cl:
        ix = [slice(None)] * n
        for vv, ss in zip(v, s): ix[vv] = 0 if ss > 0 else 1
        T[tuple(ix)] += 1
    return T.reshape(-1)  # index: bit of var 0 is the MSB


def gen_instance(n, seed):
    """F-sat arm generator (noiseless/sat4_poisson_run.py 'fs'): m=round(9.93n), 4 distinct sorted vars, random signs."""
    r = random.Random(seed); m = int(round(9.93 * n)); cl = []
    for _ in range(m):
        v = sorted(r.sample(range(n), 4)); s = [r.choice([1, -1]) for _ in v]; cl.append((v, s))
    return m, cl, fast_energies(n, cl)


def instances(n, count, base):
    """Keep satisfiable ones. base e.g. 3.1e10 (train) / 3.2e10 (test); seed = base + 1e7 n + k."""
    out, k = [], 0
    while len(out) < count:
        seed = int(base) + 10_000_000 * n + k; k += 1
        m, cl, E = gen_instance(n, seed)
        if E.min() == 0: out.append(dict(seed=seed, m=m, E=E, nsol=int((E == 0).sum())))
    return out


LAYOUTS_T = {8: [3, 3], 10: [4, 4], 12: [5, 5], 14: [4, 4, 3], 16: [5, 4, 4], 18: [5, 5, 5], 20: [6, 6, 5]}
LAYOUTS_C = {8: [4, 4], 10: [5, 5], 12: [6, 6], 14: [5, 5, 4], 16: [6, 5, 5], 18: [6, 6, 6], 20: [7, 7, 6]}


def ungray(g):
    g = np.array(g, copy=True); s = g >> 1
    while np.any(s): g ^= s; s >>= 1
    return g


class Circuit:
    def __init__(self, n, insts, ks, tbit=True, code="gray", cost="raw", prep="uniform", pad=None, dtype=np.complex64,
                 diag_fn=None, snap=0):
        self.snap = snap; self.np = NP + snap
        self.n, self.ks, self.tbit, self.C = n, list(ks), tbit, len(ks)
        assert sum(ks) + (self.C if tbit else 0) == n
        self.nfs = [(1 << k) + (pad if pad is not None else max(8, (1 << k) // 4)) for k in ks]
        self.cdt = dtype; self.rdt = np.float32 if dtype == np.complex64 else np.float64
        C = self.C
        # logical index x (n bits, var0 = MSB) -> register values in order; register c = [tbit?] + k_c bits
        x = np.arange(1 << n); pos = n; tidx, fidx = [], []
        for c, k in enumerate(ks):
            nb = k + (1 if tbit else 0); pos -= nb
            val = (x >> pos) & ((1 << nb) - 1)
            if tbit: tidx.append(val >> k); f = val & ((1 << k) - 1)
            else: tidx.append(np.zeros_like(x)); f = val
            fidx.append(ungray(f) if code == "gray" else f)
        shape = [2] * C + self.nfs
        flat = np.ravel_multi_index(tuple(tidx + fidx), shape)
        self.shape = shape; D = int(np.prod(shape))
        B = len(insts)
        Cn = np.zeros((B, D), self.rdt); succ = np.zeros((B, D), self.rdt)
        for i, ins in enumerate(insts):
            E = ins["E"].astype(float)
            if cost == "raw": c = E * n / ins["m"]
            elif cost == "std": c = (E - E.mean()) / E.std()
            else: raise ValueError(cost)
            if diag_fn is not None: c = diag_fn(c, E)
            row = np.full(D, c.max()); row[flat] = c; Cn[i] = row
            np.add.at(succ[i], flat, (E == 0).astype(float))
        Cn = Cn.reshape([B] + shape); succ = succ.reshape([B] + shape)
        if not tbit:  # transmons are ancillas: cost/success independent of their state
            sl = (slice(None),) + (0,) * C
            Cn = np.broadcast_to(Cn[sl][(slice(None),) + (None,) * C], Cn.shape).copy()
            succ = np.broadcast_to(succ[sl][(slice(None),) + (None,) * C], succ.shape).copy()
        self.Cn = jnp.asarray(Cn); self.succ = jnp.asarray(succ)
        psi0 = np.zeros(D, dtype)
        if prep == "uniform":
            fl0 = np.ravel_multi_index(tuple([np.zeros_like(x)] * C + fidx), shape) if not tbit else flat
            fl0 = np.unique(fl0); psi0[fl0] = 1 / np.sqrt(len(fl0))
        else: psi0[0] = 1
        self.psi0 = jnp.asarray(psi0.reshape(shape))
        self.V, self.lam, self.nv = [], [], []
        for nf in self.nfs:
            a = np.diag(np.sqrt(np.arange(1, nf)), 1); lam, V = np.linalg.eigh(1j * (a.T - a))
            self.V.append(jnp.asarray(V.astype(dtype))); self.lam.append(jnp.asarray(lam.astype(self.rdt))); self.nv.append(jnp.arange(nf))
        self.scales = [np.sqrt(2.0 ** k) for k in ks]
        self._p = jax.jit(self._probs)
        self._vg = jax.jit(jax.value_and_grad(self._loss), static_argnums=1)

    def _disp(self, c, re, im):
        s = self.scales[c] / 2.0
        r = jnp.sqrt(re ** 2 + im ** 2 + 1e-20) * s; ph = jnp.arctan2(im, re)
        W = jnp.exp(1j * ph * self.nv[c])[:, None] * self.V[c]
        return ((W * jnp.exp(-1j * r * self.lam[c])[None, :]) @ W.conj().T).astype(self.cdt)

    @staticmethod
    def _apply(M, psi, ax):
        return jnp.moveaxis(jnp.tensordot(M, psi, axes=([1], [ax])), 0, ax)

    def _state1(self, x, psi, cn):
        C = self.C; L = x.size // self.np; x = x.reshape(L, self.np)
        for l in range(L):
            bre, bim, th, ph, g = [x[l, j] for j in range(NP)]
            cth, sth = jnp.cos(th / 2), jnp.sin(th / 2)
            R = jnp.array([[cth, -1j * sth * jnp.exp(-1j * ph)], [-1j * sth * jnp.exp(1j * ph), cth]]).astype(self.cdt)
            for c in range(C):
                psi = self._apply(R, psi, c)
                Dp = self._disp(c, bre, bim)
                if self.snap:  # size-free cavity SNAP phase phi(n) = sum_j s_j (n / sqrt(2^k))^j  (applied before ECD)
                    u = self.nv[c] / self.scales[c]
                    ph = sum(x[l, NP + j] * u ** (j + 1) for j in range(self.snap))
                    Dp = Dp * jnp.exp(-1j * ph)[None, :].astype(self.cdt)
                p0 = jnp.take(psi, 0, axis=c); p1 = jnp.take(psi, 1, axis=c)
                fa = C - 1 + c  # cavity axis after removing transmon axis
                n0 = self._apply(Dp.conj().T, p1, fa); n1 = self._apply(Dp, p0, fa)
                psi = jnp.stack([n0, n1], axis=c)
            psi = psi * jnp.exp(-1j * g * cn).astype(self.cdt)
        return psi

    def _probs(self, x, Cn, succ, psi0):
        def one(args):
            cn, sc = args
            psi = self._state1(x, psi0, cn)
            return jnp.sum((jnp.abs(psi) ** 2) * sc)
        return jax.lax.map(one, (Cn, succ))

    def _loss(self, x, kind, Cn, succ, psi0):
        p = self._probs(x, Cn, succ, psi0)
        return -jnp.mean(jnp.log(p + 1e-12)) if kind == "logp" else -jnp.mean(p)

    def probs(self, x): return np.asarray(self._p(jnp.asarray(x, self.rdt), self.Cn, self.succ, self.psi0), float)

    def vg(self, x, kind="meanp"):
        v, g = self._vg(jnp.asarray(x, self.rdt), kind, self.Cn, self.succ, self.psi0)
        return float(v), np.asarray(g, float)


def summarize(p, insts=None):
    p = np.asarray(p); inv = 1 / np.maximum(p, 1e-300)
    out = {"median_inv_p": float(np.median(inv)), "B90_inv_p": float(np.quantile(inv, 0.9)), "mean_p": float(p.mean()),
           "median_p": float(np.median(p))}
    if insts is not None:
        rg = np.array([2.0 ** len(np.binary_repr(len(i["E"]) - 1)) / i["nsol"] for i in insts])
        out["rand_median"] = float(np.median(rg)); out["rand_B90"] = float(np.quantile(rg, 0.9))
    return out
