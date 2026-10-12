"""Finite-time Kraus channels of the Lindblad dissipators, and helpers that act on one mode of a
multi-mode density tensor.

Density tensor convention: for modes with dimensions ``dims = (d_0, ..., d_{N-1})`` the state is an
array of shape ``dims + dims``; axis ``k`` is the ket index of mode ``k`` and axis ``k + N`` its bra
index. Transmon index 0 is |g>, 1 is |e>.

Channels (each is the exact solution of its own Lindblad term over a time tau):

* photon loss  kappa D[a]:  K_j = sqrt((1 - e^{-kt})^j / j!) e^{-kt n/2} a^j,  kt = kappa tau.
  ``K_j = sum_n w_j[n] |n - j><n|`` is stored as weights (shift form). With ``jmax`` below the
  cutoff the dropped jumps are folded into K_0 via K_0 = (I - sum_{j>=1} K_j^dag K_j)^{1/2}
  (Dutta et al. eqs. 38-39).
* transmon  gamma D[sigma_-] + (gamma_phi / 2) D[sigma_z]: amplitude damping (p = 1 - e^{-gt}) composed
  with phase damping (coherence factor e^{-gpt}); the two commute. Populations relax as e^{-gamma tau},
  coherences as e^{-tau / T2} with 1/T2 = gamma/2 + gamma_phi.
"""

from __future__ import annotations

from functools import lru_cache
from math import lgamma

import numpy as np


# ---------------------------------------------------------------------------
# channel construction
# ---------------------------------------------------------------------------
@lru_cache(maxsize=64)
def cavity_loss_weights(kt: float, nf: int, jmax: int | None = None,
                        complete: bool = True) -> tuple[tuple[int, np.ndarray], ...]:
    """Photon-loss Kraus operators in shift form: ((j, w_j), ...), K_j = sum_n w_j[n] |n-j><n|.

    ``complete=False`` keeps the exact K_0 = e^{-kt n/2} even when jumps above ``jmax`` are dropped
    (the channel is then trace-decreasing by exactly the probability of > jmax losses)."""
    kt = float(kt)
    nf = int(nf)
    if kt < 0:
        raise ValueError("kappa*tau must be >= 0")
    n = np.arange(nf)
    if kt == 0.0:
        return ((0, np.ones(nf)),)
    top = nf - 1 if jmax is None else min(int(jmax), nf - 1)
    logp = np.log(-np.expm1(-kt))
    out = []
    for j in range(1, top + 1):
        w = np.zeros(nf)
        m = n[j:]
        lg = np.array([lgamma(v + 1) - lgamma(v - j + 1) for v in m])
        w[j:] = np.exp(0.5 * (j * logp - lgamma(j + 1) + lg) - 0.5 * kt * (m - j))
        out.append((j, w))
    if top == nf - 1 or not complete:
        w0 = np.exp(-0.5 * kt * n)
    else:
        jumped = sum(w ** 2 for _, w in out)
        w0 = np.sqrt(np.clip(1.0 - jumped, 0.0, None))
    return ((0, w0),) + tuple(out)


def shift_to_dense(j: int, w: np.ndarray) -> np.ndarray:
    nf = w.size
    K = np.zeros((nf, nf), dtype=complex)
    for col in range(j, nf):
        K[col - j, col] = w[col]
    return K


def cavity_loss_kraus(kt: float, nf: int, jmax: int | None = None, complete: bool = True) -> list[np.ndarray]:
    """Dense nf x nf photon-loss Kraus operators (for tests / small systems)."""
    return [shift_to_dense(j, w) for j, w in cavity_loss_weights(kt, nf, jmax, complete)]


def qubit_kraus(gt: float, gpt: float) -> list[np.ndarray]:
    """Kraus set of gamma D[sigma_-] + (gamma_phi/2) D[sigma_z] for gt = gamma tau, gpt = gamma_phi tau."""
    p = -np.expm1(-float(gt))
    lam = np.exp(-float(gpt))
    ad = [np.array([[1.0, 0.0], [0.0, np.sqrt(1.0 - p)]], dtype=complex),
          np.array([[0.0, np.sqrt(p)], [0.0, 0.0]], dtype=complex)]
    pd = [np.sqrt((1.0 + lam) / 2.0) * np.eye(2, dtype=complex),
          np.sqrt((1.0 - lam) / 2.0) * np.diag([1.0, -1.0]).astype(complex)]
    out = [a @ b for a in ad for b in pd]
    return [K for K in out if np.abs(K).max() > 0.0]


def superop(kraus: list[np.ndarray]) -> np.ndarray:
    """S[i, i', j, j'] = sum_K K[i, j] conj(K[i', j']), so rho'[i, i'] = sum S[i, i', j, j'] rho[j, j']."""
    return sum(np.einsum("ij,kl->ikjl", K, K.conj()) for K in kraus)


# ---------------------------------------------------------------------------
# applying operators to one mode of a density tensor
# ---------------------------------------------------------------------------
def n_modes(rho: np.ndarray) -> int:
    if rho.ndim % 2:
        raise ValueError("density tensor must have ket + bra axes")
    return rho.ndim // 2


def apply_axis(t: np.ndarray, M: np.ndarray, axis: int) -> np.ndarray:
    """Contract M[i, j] with index ``axis`` of t (batched matmul on a reshaped view, no transposes)."""
    shp = t.shape
    pre = int(np.prod(shp[:axis], dtype=np.int64))
    v = np.ascontiguousarray(t).reshape(pre, shp[axis], -1)
    return np.matmul(M, v).reshape(shp[:axis] + (M.shape[0],) + shp[axis + 1:])


def apply_left(rho: np.ndarray, M: np.ndarray, k: int) -> np.ndarray:
    """M rho on mode k."""
    return apply_axis(rho, M, k)


def apply_right_dag(rho: np.ndarray, M: np.ndarray, k: int) -> np.ndarray:
    """rho M^dag on mode k."""
    return apply_axis(rho, M.conj(), k + n_modes(rho))


def apply_unitary(rho: np.ndarray, U: np.ndarray, k: int) -> np.ndarray:
    return apply_right_dag(apply_left(rho, U, k), U, k)


def conditional_left(t: np.ndarray, coef: np.ndarray, ops, kq: int, kc: int) -> np.ndarray:
    """U t for U = sum_{i, c} coef[i, c] |i><c|_kq (x) ops[i]_kc  (qubit axis kq, cavity axis kc).

    ECD(beta) R = |0><1|R (x) D^dag + |1><0|R (x) D has this form with coef = R[::-1], ops = (D^dag, D)."""
    mixed = apply_axis(t, coef, kq)
    kc_red = kc - (kc > kq)
    parts = [apply_axis(mixed[(slice(None),) * kq + (i,)], ops[i], kc_red) for i in range(2)]
    return np.stack(parts, axis=kq)


def pair_conditional_left(t: np.ndarray, part_1, part_2) -> np.ndarray:
    """(U_2 U_1) t for two ``conditional_left`` unitaries on (axis 0, axis 2) and (axis 1, axis 3).

    Same result as two ``conditional_left`` calls; the transmon mixing is one 4 x 4 product on the
    merged (0, 1) index and the cavity operators then act on contiguous blocks."""
    (c1, o1), (c2, o2) = part_1, part_2
    shp = t.shape
    nf = shp[2]
    mixed = (np.kron(c1, c2) @ np.ascontiguousarray(t).reshape(4, -1)).reshape(shp)
    out = np.empty_like(mixed)
    for i in range(2):
        for j in range(2):
            blk = np.matmul(o1[i], mixed[i, j].reshape(nf, -1)).reshape(nf, nf, -1)
            out[i, j] = np.matmul(o2[j], blk).reshape(shp[2:])
    return out


def apply_conditional(rho: np.ndarray, coef: np.ndarray, ops, kq: int, kc: int) -> np.ndarray:
    """U rho U^dag for the conditional unitary of ``conditional_left`` (any rho)."""
    N = n_modes(rho)
    out = conditional_left(rho, coef, ops, kq, kc)
    return conditional_left(out, coef.conj(), [m.conj() for m in ops], kq + N, kc + N)


def dagger(rho: np.ndarray) -> np.ndarray:
    """Conjugate transpose of the density tensor viewed as a matrix."""
    m = as_matrix(rho)
    return np.ascontiguousarray(m.conj().T).reshape(rho.shape)


def apply_two_mode_unitary(rho: np.ndarray, U4: np.ndarray, k1: int, k2: int) -> np.ndarray:
    """U rho U^dag with U given as U4[i1, i2, j1, j2] on modes (k1, k2)."""
    N = n_modes(rho)
    out = np.moveaxis(np.tensordot(U4, rho, axes=([2, 3], [k1, k2])), [0, 1], [k1, k2])
    out = np.tensordot(out, U4.conj(), axes=([k1 + N, k2 + N], [2, 3]))
    return np.moveaxis(out, [-2, -1], [k1 + N, k2 + N])


def apply_kraus(rho: np.ndarray, kraus: list[np.ndarray], k: int) -> np.ndarray:
    """sum_K K rho K^dag on mode k (dense, any mode dimension)."""
    out = None
    for K in kraus:
        term = apply_right_dag(apply_left(rho, K, k), K, k)
        out = term if out is None else out + term
    return out


def _ax(N: int, k: int, sl, kb=None, slb=None) -> tuple:
    idx = [slice(None)] * (2 * N)
    idx[k] = sl
    idx[k + N if kb is None else kb] = sl if slb is None else slb
    return tuple(idx)


def apply_superop_small(rho: np.ndarray, S: np.ndarray, k: int) -> np.ndarray:
    """Apply a one-mode superoperator S[i, i', j, j'] by block slicing (cheap for a qubit: only the
    nonzero entries of S are touched, no transposes of the big tensor)."""
    N = n_modes(rho)
    d = S.shape[0]
    out = np.zeros_like(rho)
    for i in range(d):
        for ip in range(d):
            dst = out[_ax(N, k, i, slb=ip)]
            for j in range(d):
                for jp in range(d):
                    s = S[i, ip, j, jp]
                    if s != 0:
                        dst += s * rho[_ax(N, k, j, slb=jp)]
    return out


def _is_phase_covariant(S: np.ndarray) -> bool:
    keep = np.zeros((2, 2, 2, 2), dtype=bool)
    for idx in ((0, 0, 0, 0), (0, 0, 1, 1), (1, 1, 0, 0), (1, 1, 1, 1), (0, 1, 0, 1), (1, 0, 1, 0)):
        keep[idx] = True
    return S.shape == (2, 2, 2, 2) and not np.any(S[~keep])


def apply_qubit_channel(rho: np.ndarray, kraus: list[np.ndarray], k: int) -> np.ndarray:
    S = superop(kraus)
    if not _is_phase_covariant(S):
        return apply_superop_small(rho, S, k)
    N = n_modes(rho)
    gg, ge, eg, ee = (_ax(N, k, a, slb=b) for a, b in ((0, 0), (0, 1), (1, 0), (1, 1)))
    out = rho.copy()
    old_gg = rho[gg]
    out[gg] = S[0, 0, 0, 0] * old_gg + S[0, 0, 1, 1] * rho[ee]
    out[ee] = S[1, 1, 1, 1] * rho[ee] + S[1, 1, 0, 0] * old_gg
    out[ge] *= S[0, 1, 0, 1]
    out[eg] *= S[1, 0, 1, 0]
    return out


def apply_cavity_loss(rho: np.ndarray, weights, k: int) -> np.ndarray:
    """sum_j K_j rho K_j^dag on cavity mode k with K_j in shift form (no matrix products)."""
    N = n_modes(rho)
    nf = rho.shape[k]
    shp_k = [1] * (2 * N)
    shp_b = [1] * (2 * N)
    out = np.zeros_like(rho)
    for j, w in weights:
        m = nf - j
        shp_k[k] = m
        shp_b[k + N] = m
        wk = w[j:].reshape(shp_k)
        wb = w[j:].reshape(shp_b)
        out[_ax(N, k, slice(0, m))] += wk * wb * rho[_ax(N, k, slice(j, nf))]
        shp_k[k] = 1
        shp_b[k + N] = 1
    return out


def apply_diag_unitary(rho: np.ndarray, phase: np.ndarray) -> np.ndarray:
    """U rho U^dag for U diagonal, phase[mode indices] = U's diagonal, shape = dims."""
    N = n_modes(rho)
    return rho * phase.reshape(phase.shape + (1,) * N) * phase.conj().reshape((1,) * N + phase.shape)


# ---------------------------------------------------------------------------
# adding / removing a transmon mode
# ---------------------------------------------------------------------------
def tensor_in_ground(rho: np.ndarray, pos: int = 0) -> np.ndarray:
    """rho -> |g><g| (x) rho with the new 2-level mode at position ``pos``."""
    N = n_modes(rho)
    dims = rho.shape[:N]
    new_dims = dims[:pos] + (2,) + dims[pos:]
    out = np.zeros(new_dims + new_dims, dtype=rho.dtype)
    idx = [slice(None)] * (2 * (N + 1))
    idx[pos] = 0
    idx[pos + N + 1] = 0
    out[tuple(idx)] = rho
    return out


def trace_out(rho: np.ndarray, k: int) -> np.ndarray:
    N = n_modes(rho)
    return np.trace(rho, axis1=k, axis2=k + N)


def as_matrix(rho: np.ndarray) -> np.ndarray:
    N = n_modes(rho)
    D = int(np.prod(rho.shape[:N]))
    return rho.reshape(D, D)


def probabilities(rho: np.ndarray) -> np.ndarray:
    return np.real(np.diagonal(as_matrix(rho))).copy()
