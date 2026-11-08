"""Statevector harness for the fixed-field power-law family.

Circuit propagation never builds a dense Hamiltonian. Dense matrices are used
only for N<=12 validation and the ground-state reference. N>12 is refused by
the dense helpers so a later size cannot accidentally allocate 2^N by 2^N.
"""

from __future__ import annotations

import hashlib
from math import pow as mpow

import numpy as np
from scipy.linalg import expm

_I = np.eye(2, dtype=np.complex128)
_X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)


def _on(bit: int) -> np.ndarray:
    mats = [_X if r == bit else _I for r in range(3)]
    return np.kron(np.kron(mats[2], mats[1]), mats[0])


G1 = _on(0) + _on(1) + _on(2)
G2 = _on(0) @ _on(1) + _on(0) @ _on(2) + _on(1) @ _on(2)
G3 = _on(0) @ _on(1) @ _on(2)


def couplings(n: int, alpha: float = 1.1) -> np.ndarray:
    """Open chain, J_ij = |i-j|^{-alpha}, J_nn = 1. No Kac factor."""
    J = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            J[i, j] = J[j, i] = float(mpow(j - i, -alpha))
    return J


def packing_interleaved(n: int):
    if n % 3 != 0:
        raise ValueError("n must be a multiple of 3")
    m = n // 3
    return [[a, a + m, a + 2 * m] for a in range(m)]


def upper(mat: np.ndarray) -> np.ndarray:
    n = mat.shape[0]
    return np.array([mat[i, j] for i in range(n) for j in range(i + 1, n)], dtype=np.float64)


def rms(v: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(v, dtype=np.float64) ** 2)))


def z_table(n: int) -> np.ndarray:
    x = np.arange(1 << n, dtype=np.uint32)
    z = np.empty((n, 1 << n), dtype=np.float64)
    for q in range(n):
        z[q] = 1.0 - 2.0 * ((x >> np.uint32(q)) & np.uint32(1))
    return z


def pair_diag(coeff: np.ndarray, z: np.ndarray) -> np.ndarray:
    n = coeff.shape[0]
    out = np.zeros(z.shape[1], dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            if coeff[i, j] != 0.0:
                out += coeff[i, j] * z[i] * z[j]
    return out


def fk_matrix(J: np.ndarray) -> np.ndarray:
    K = np.array(J @ J, dtype=np.float64, copy=True)
    np.fill_diagonal(K, 0.0)
    K = 0.5 * (K + K.T)
    np.fill_diagonal(K, 0.0)
    return K


def block_mean_matrix(J: np.ndarray, packing) -> np.ndarray:
    n = J.shape[0]
    M = np.zeros((n, n), dtype=np.float64)
    m = len(packing)
    for a in range(m):
        for b in range(a + 1, m):
            acc = 0.0
            for i in packing[a]:
                for j in packing[b]:
                    acc += float(J[i, j])
            jbar = acc / 9.0
            for i in packing[a]:
                for j in packing[b]:
                    M[i, j] = M[j, i] = jbar
    return M


def cavity_index(n: int, sites) -> np.ndarray:
    free = [b for b in range(n) if b not in set(sites)]
    nloc = 1 << len(sites)
    index = np.empty((1 << len(free), nloc), dtype=np.int64)
    for f in range(index.shape[0]):
        base = 0
        for k, b in enumerate(free):
            if (f >> k) & 1:
                base |= 1 << b
        for local in range(nloc):
            ind = base
            for r, site in enumerate(sites):
                if (local >> r) & 1:
                    ind |= 1 << site
            index[f, local] = ind
    return index


class Target:
    def __init__(self, n: int, B: float, alpha: float = 1.1):
        self.n = n
        self.B = float(B)
        self.alpha = alpha
        self.J = couplings(n, alpha)
        self.packing = packing_interleaved(n)
        self.z = z_table(n)
        self.hc = pair_diag(self.J, self.z)
        self.K = fk_matrix(self.J)
        self.rK = rms(upper(self.K)) / rms(upper(self.J))
        self.khat_coeff = self.K / self.rK
        self.khat = pair_diag(self.khat_coeff, self.z)
        self.M = block_mean_matrix(self.J, self.packing)
        self.rM = rms(upper(self.M)) / rms(upper(self.J))
        self.mhat_coeff = self.M / self.rM
        self.mhat = pair_diag(self.mhat_coeff, self.z)
        cols = np.stack([upper(self.J), upper(self.K), upper(self.M)])
        self.feature_rank = int(np.linalg.matrix_rank(cols, tol=1e-10))
        self.cav = [cavity_index(n, c) for c in self.packing]
        self.dim = 1 << n

    def g_diag(self, theta_k: float, theta_m: float) -> np.ndarray:
        return self.hc + float(theta_k) * self.khat + float(theta_m) * self.mhat


def mixer(c1, c2, c3, with_grad=False):
    H = c1 * G1 + c2 * G2 + c3 * G3
    w, V = np.linalg.eigh(np.real(H).astype(np.float64))
    U = (V * np.exp(-1j * w)) @ V.T
    if not with_grad:
        return U
    wp, wq = w[:, None], w[None, :]
    half = 0.5 * (wp - wq)
    mid = 0.5 * (wp + wq)
    sinc = np.ones_like(half)
    nz = np.abs(half) > 1e-8
    sinc[nz] = np.sin(half[nz]) / half[nz]
    phi = -1j * np.exp(-1j * mid) * sinc
    dUs = []
    for G in (np.real(G1), np.real(G2), np.real(G3)):
        dtil = V.T @ G @ V
        dUs.append((V @ (phi * dtil)) @ V.T)
    return U, dUs


def apply_cav(psi, U, index):
    out = np.array(psi, copy=True)
    out[index] = out[index] @ U.T
    return out


def apply_sum_x(psi, n):
    acc = np.zeros_like(psi)
    for q in range(n):
        half = 1 << q
        x = psi.reshape(-1, 2, half)
        y = np.empty_like(x)
        y[:, 0, :] = x[:, 1, :]
        y[:, 1, :] = x[:, 0, :]
        acc += y.reshape(-1)
    return acc


def apply_rx_all(psi, beta, n):
    c = np.cos(beta)
    s = np.sin(beta)
    out = np.asarray(psi, dtype=np.complex128)
    for q in range(n):
        half = 1 << q
        x = out.reshape(-1, 2, half)
        a, b = x[:, 0, :], x[:, 1, :]
        y = np.empty_like(x)
        y[:, 0, :] = c * a - 1j * s * b
        y[:, 1, :] = -1j * s * a + c * b
        out = y.reshape(-1)
    return out


def plus(n):
    return np.full(1 << n, 1.0 / np.sqrt(1 << n), dtype=np.complex128)


def logical_candidate(target: Target, gamma, theta_k, theta_m, shells):
    phase = np.exp(-1j * float(gamma) * target.g_diag(theta_k, theta_m))
    psi = plus(target.n)
    for row in shells:
        psi = psi * phase
        U = mixer(float(row[0]), float(row[1]), float(row[2]))
        for index in target.cav:
            psi = apply_cav(psi, U, index)
    return psi


def logical_grad(target: Target, x):
    L = (len(x) - 3) // 3
    gamma, thk, thm = float(x[0]), float(x[1]), float(x[2])
    shells = np.asarray(x[3:], dtype=np.float64).reshape(L, 3)
    G = target.g_diag(thk, thm)
    phase = np.exp(-1j * gamma * G)
    psi = plus(target.n)
    frames = []
    for ell, row in enumerate(shells):
        psi = psi * phase
        frames.append(("d", psi))
        U, dUs = mixer(float(row[0]), float(row[1]), float(row[2]), True)
        for index in target.cav:
            psi = apply_cav(psi, U, index)
        frames.append(("s", psi, U, dUs, ell))
    Hpsi = target.hc * psi + target.B * apply_sum_x(psi, target.n)
    E = float(np.vdot(psi, Hpsi).real)
    var = float(np.vdot(Hpsi, Hpsi).real - E * E)
    lam = Hpsi
    grad = np.zeros(3 + 3 * L)
    for kind, *rest in reversed(frames):
        if kind == "d":
            psi_d = rest[0]
            im = np.imag(np.conj(lam) * psi_d)
            grad[0] += 2.0 * float(np.sum(G * im))
            grad[1] += 2.0 * float(np.sum((gamma * target.khat) * im))
            grad[2] += 2.0 * float(np.sum((gamma * target.mhat) * im))
            lam = np.exp(1j * gamma * G) * lam
        else:
            psi_s, U, dUs, ell = rest
            for k in range(3):
                acc = np.zeros_like(psi_s)
                Va = dUs[k] @ np.conjugate(U).T
                for index in target.cav:
                    acc += apply_cav(psi_s, Va, index)
                grad[3 + 3 * ell + k] += 2.0 * float(np.real(np.vdot(lam, acc)))
            for index in target.cav:
                lam = apply_cav(lam, np.conjugate(U).T, index)
    return psi, E, grad, var


def qaoa_state(target: Target, gammas, betas):
    psi = plus(target.n)
    for g, b in zip(gammas, betas):
        psi = np.exp(-1j * float(g) * target.hc) * psi
        psi = apply_rx_all(psi, float(b), target.n)
    return psi


def qaoa_grad(target: Target, x):
    p = len(x) // 2
    gammas = np.asarray(x[0::2], dtype=np.float64)
    betas = np.asarray(x[1::2], dtype=np.float64)
    psi = plus(target.n)
    frames = []
    for ell in range(p):
        psi = np.exp(-1j * gammas[ell] * target.hc) * psi
        frames.append(("g", psi, float(gammas[ell]), ell))
        psi = apply_rx_all(psi, betas[ell], target.n)
        frames.append(("b", psi, float(betas[ell]), ell))
    Hpsi = target.hc * psi + target.B * apply_sum_x(psi, target.n)
    E = float(np.vdot(psi, Hpsi).real)
    var = float(np.vdot(Hpsi, Hpsi).real - E * E)
    lam = Hpsi
    grad = np.zeros(2 * p)
    for kind, psi_new, angle, ell in reversed(frames):
        if kind == "g":
            im = np.imag(np.conj(lam) * psi_new)
            grad[2 * ell] += 2.0 * float(np.sum(target.hc * im))
            lam = np.exp(1j * angle * target.hc) * lam
        else:
            grad[2 * ell + 1] += 2.0 * float(np.real(np.vdot(lam, -1j * apply_sum_x(psi_new, target.n))))
            lam = apply_rx_all(lam, -angle, target.n)
    return psi, E, grad, var


def dense_h(target: Target) -> np.ndarray:
    if target.n > 12:
        raise RuntimeError("dense Hamiltonian refused above N=12")
    n = target.n
    dim = 1 << n
    H = np.diag(target.hc.astype(np.float64))
    idx = np.arange(dim)
    for q in range(n):
        H[idx, idx ^ (1 << q)] += target.B
    return H


def ground_state(target: Target):
    H = dense_h(target)
    evals, evecs = np.linalg.eigh(H)
    gs = evecs[:, 0]
    k = int(np.argmax(np.abs(gs)))
    gs = gs * np.exp(-1j * np.angle(gs[k]))
    if float(np.sum(gs.real)) < 0:
        gs = -gs
    residual = float(np.linalg.norm(H @ gs - evals[0] * gs))
    parity = float(np.vdot(gs, gs[::-1]).real)
    return {
        "E0": float(evals[0]),
        "Emax": float(evals[-1]),
        "full_gap": float(evals[1] - evals[0]),
        "gs": gs,
        "residual": residual,
        "parity": parity,
        "evals": evals,
    }


def parity_weight(psi) -> float:
    return 0.5 * (1.0 + float(np.vdot(psi, psi[::-1]).real))


def fidelity(gs, psi) -> float:
    return float(abs(np.vdot(gs, psi)) ** 2)


def eta(E, E0, Emax) -> float:
    return float((E - Emax) / (E0 - Emax))


def state_hash(psi) -> str:
    return hashlib.sha256(np.ascontiguousarray(psi).tobytes()).hexdigest()


def dense_qaoa_state(target: Target, gammas, betas):
    """Independent dense-matrix reference. Not used for training."""
    if target.n > 12:
        raise RuntimeError("dense QAOA reference refused above N=12")
    dim = target.dim
    Hc = np.diag(target.hc)
    Xsum = np.zeros((dim, dim), dtype=np.complex128)
    idx = np.arange(dim)
    for q in range(target.n):
        Xsum[idx, idx ^ (1 << q)] = 1.0
    psi = plus(target.n)
    for g, b in zip(gammas, betas):
        psi = expm(-1j * float(g) * Hc) @ psi
        psi = expm(-1j * float(b) * Xsum) @ psi
    return psi
