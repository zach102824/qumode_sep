"""Simulator for the templated native chain ansatz (all data in cavities, Gray code, chain of neighbour buses).
State in the LOGICAL basis: x (n bits, var0 = MSB, cavity 0 = top K_0 bits); cavity c's logical value g_c = contiguous bit block.
(level f of cavity c holds logical digits gray(f); a level-basis matrix V maps to logical basis by V_log[g',g] = V[ungray g', ungray g].)
Layer l: diag phase exp(-i gamma_l * nm * PC_w)  [PC_w = chain-projected weighted cost = compiled SNAP + bus tables, exact sum]
         -> per-cavity mixer M_c(beta_l)        [ideal 'gray' = prod_j exp(-i b X_j/2) (= qubit QAOA mixer), ideal 'tri',
                                                  or calibrated native P V P from mixcal (non-unitary: leaked part dropped)]
Start: prep state per cavity (ideal uniform or native prep amplitudes on valid levels). p = sum_{x sat} |psi_x|^2.
"""
from __future__ import annotations
import numpy as np, jax, jax.numpy as jnp
gray = lambda i: i ^ (i >> 1)
def ungray(g):
    s = g >> 1
    while s: g ^= s; s >>= 1
    return g

def gen_matrix(K, kind):
    """generator H with ideal mixer U(b) = exp(-i b/2 H) in the LOGICAL basis of a K-bit cavity."""
    N = 1 << K; H = np.zeros((N, N))
    if kind == "gray":
        for g in range(N):
            for j in range(K): H[g ^ (1 << j), g] = 1
    elif kind == "tri":   # level hop f <-> f+1, levels hold gray(f)
        for f in range(N - 1): H[gray(f + 1), gray(f)] = H[gray(f), gray(f + 1)] = 1
    else: raise ValueError(kind)
    return H

class Mixer:
    """ideal mixer family, differentiable in beta: U(b) = Q diag(exp(-i b/2 lam)) Q^T."""
    def __init__(self, K, kind):
        lam, Q = np.linalg.eigh(gen_matrix(K, kind)); self.lam = lam.astype(np.float32); self.Q = Q.astype(np.complex64)
    def __call__(self, b):  # b scalar or (B,)
        b = jnp.asarray(b)
        e = jnp.exp(-0.5j * b[..., None] * self.lam)
        return jnp.einsum("ik,...k,jk->...ij", self.Q, e, self.Q)

def apply_cav(M, psi, c, C):
    """M (N,N) or (B,N,N) acting on cavity axis c of psi (B, N_0..N_{C-1})."""
    let = "defghijk"[:C]; o = let[c]; out = let.replace(o, "y")
    if M.ndim == 2: return jnp.einsum(f"y{o},b{let}->b{out}", M, psi)
    return jnp.einsum(f"by{o},b{let}->b{out}", M, psi)

def run_state(psi0, phases, mixers, ks):
    """psi0 (B, Ns), phases list over l of (B, 2^n) real (already multiplied by gamma*nm), mixers list over l of list over c of matrices."""
    B = psi0.shape[0]; C = len(ks); psi = psi0
    for l in range(len(phases)):
        psi = psi * jnp.exp(-1j * phases[l].reshape(psi.shape)).astype(jnp.complex64)
        for c in range(C): psi = apply_cav(mixers[l][c], psi, c, C)
    return psi

def uniform_psi0(ks, B=1, amps=None):
    """product start state; amps: list over c of (N_c,) amplitudes in LOGICAL order (None = ideal uniform)."""
    v = None
    for c, k in enumerate(ks):
        a = np.full(1 << k, 1 / np.sqrt(1 << k), np.complex64) if amps is None else np.asarray(amps[c], np.complex64)
        v = a if v is None else np.multiply.outer(v, a)
    return jnp.broadcast_to(jnp.asarray(v), (B,) + v.shape)

def summ(p, lk=None):
    p = np.asarray(p, float); inv = 1 / np.maximum(p, 1e-300)
    o = {"median_inv_p": float(np.median(inv)), "B90_inv_p": float(np.quantile(inv, 0.9)), "mean_p": float(np.mean(p)), "median_p": float(np.median(p)),
         "loss": float(np.mean(-np.log(np.maximum(p, 1e-300)))), "S": float((np.log(np.median(inv)) + np.log(np.quantile(inv, 0.9))) / 2)}
    return o
def rand_stats(insts, n):
    rg = np.array([2.0 ** n / i["nsol"] for i in insts]); return {"median_inv_p": float(np.median(rg)), "B90_inv_p": float(np.quantile(rg, 0.9))}
