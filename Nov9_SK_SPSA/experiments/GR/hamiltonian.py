"""Logical SK Hamiltonian. Site i is bit i (LSB = site 0). Z|0> = +|0>.

H = sum_{i<j} J_ij Z_i Z_j - b sum_i X_i.
The same J is used at every field and by every method. Grouping never permutes H.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from common import assert_dev_seed


def upper(mat: np.ndarray) -> np.ndarray:
    n = mat.shape[0]
    return np.array([mat[i, j] for i in range(n) for j in range(i + 1, n)], dtype=np.float64)


def rms(v: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(np.asarray(v, dtype=np.float64)))))


def sha256_f64(vec: np.ndarray) -> str:
    raw = np.ascontiguousarray(np.asarray(vec, dtype="<f8")).tobytes()
    return hashlib.sha256(raw).hexdigest()


def make_J(seed: int, n: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    u = rng.normal(0.0, 1.0 / np.sqrt(n), size=n * (n - 1) // 2)
    J = np.zeros((n, n), dtype=np.float64)
    t = 0
    for i in range(n):
        for j in range(i + 1, n):
            J[i, j] = J[j, i] = float(u[t])
            t += 1
    return J


def z_table(n: int) -> np.ndarray:
    dim = 1 << n
    x = np.arange(dim, dtype=np.uint32)
    z = np.empty((n, dim), dtype=np.float64)
    for q in range(n):
        z[q] = 1.0 - 2.0 * ((x >> np.uint32(q)) & np.uint32(1))
    return z


def pair_diag(coeff: np.ndarray, z: np.ndarray | None = None) -> np.ndarray:
    n = coeff.shape[0]
    if z is None:
        z = z_table(n)
    out = np.zeros(z.shape[1], dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            cij = float(coeff[i, j])
            if cij != 0.0:
                out += cij * z[i] * z[j]
    return out


def pair_K(J: np.ndarray) -> np.ndarray:
    K = np.array(J @ J, dtype=np.float64, copy=True)
    np.fill_diagonal(K, 0.0)
    K = 0.5 * (K + K.T)
    np.fill_diagonal(K, 0.0)
    return K


def block_coeff(source: np.ndarray, packing: list[list[int]]) -> np.ndarray:
    """Inter-cavity blocks replaced by their mean. Intra-cavity entries stay 0."""
    C = np.zeros_like(source, dtype=np.float64)
    m = len(packing)
    for a in range(m):
        for b in range(a + 1, m):
            pairs = [(i, j) for i in packing[a] for j in packing[b]]
            acc = 0.0
            for i, j in pairs:
                acc += float(source[i, j])
            w = acc / float(len(pairs))
            for i, j in pairs:
                C[i, j] = C[j, i] = w
    return C


def hat(raw: np.ndarray, rms_j: float) -> tuple[np.ndarray, float]:
    r = rms(upper(raw))
    if r <= 0.0:
        raise RuntimeError("zero feature RMS")
    scale = float(rms_j) / r
    return raw * scale, scale


def apply_X_sum(psi: np.ndarray, n: int) -> np.ndarray:
    out = np.zeros_like(psi)
    dim = psi.shape[0]
    for i in range(n):
        step = 1 << i
        view = psi.reshape(dim // (2 * step), 2, step)
        flipped = np.empty_like(view)
        flipped[:, 0, :] = view[:, 1, :]
        flipped[:, 1, :] = view[:, 0, :]
        out += flipped.reshape(dim)
    return out


def apply_H_site(psi: np.ndarray, site: int) -> np.ndarray:
    step = 1 << site
    dim = psi.shape[0]
    view = psi.reshape(dim // (2 * step), 2, step)
    a = view[:, 0, :].copy()
    b = view[:, 1, :].copy()
    s = np.sqrt(0.5)
    view[:, 0, :] = s * (a + b)
    view[:, 1, :] = s * (a - b)
    return view.reshape(dim)


def apply_rx_site(psi: np.ndarray, site: int, beta: float) -> np.ndarray:
    step = 1 << site
    dim = psi.shape[0]
    view = psi.reshape(dim // (2 * step), 2, step)
    a = view[:, 0, :].copy()
    b = view[:, 1, :].copy()
    c = np.cos(beta)
    s = np.sin(beta)
    view[:, 0, :] = c * a - 1j * s * b
    view[:, 1, :] = -1j * s * a + c * b
    return view.reshape(dim)


def parity_of_vector(vec: np.ndarray) -> float:
    dim = vec.shape[0]
    flip = np.arange(dim) ^ (dim - 1)
    return float(np.vdot(vec, vec[flip]).real)


def dense_H(hc: np.ndarray, b: float, n: int) -> np.ndarray:
    dim = 1 << n
    H = np.diag(np.asarray(hc, dtype=np.float64))
    bb = float(b)
    for i in range(n):
        step = 1 << i
        for base in range(dim):
            if (base & step) == 0:
                j = base | step
                H[base, j] -= bb
                H[j, base] -= bb
    return H


@dataclass
class LogicalInstance:
    n: int
    seed: int
    b: float
    J: np.ndarray
    z: np.ndarray
    hc: np.ndarray
    K_raw: np.ndarray
    K_hat: np.ndarray
    scale_K: float
    rms_J: float
    E0: float
    E1: float
    E0_odd: float
    lambda_max: float
    gap: float
    delta_odd: float
    gs: np.ndarray
    F_plus: float
    j_upper_sha: str
    gs_sha: str
    gs_min: float
    gs_parity: float
    residual: float


def build_instance(seed: int, n: int, b: float, *, pilot: bool) -> LogicalInstance:
    assert_dev_seed(seed, pilot=pilot)
    if n > 12:
        raise RuntimeError("dense ground state is only built for N<=12")
    J = make_J(seed, n)
    z = z_table(n)
    hc = pair_diag(J, z)
    K_raw = pair_K(J)
    rms_J = rms(upper(J))
    K_hat, scale_K = hat(K_raw, rms_J)
    H = dense_H(hc, b, n)
    evals, evecs = np.linalg.eigh(H)
    gs = np.asarray(evecs[:, 0], dtype=np.float64).copy()
    if float(np.sum(gs)) < 0.0:
        gs *= -1.0
    E0 = float(evals[0])
    E1 = float(evals[1])
    lam_max = float(evals[-1])
    odd_vals = []
    gs_parity = parity_of_vector(gs)
    for k in range(evals.shape[0]):
        p = parity_of_vector(evecs[:, k])
        if p < -0.5:
            odd_vals.append(float(evals[k]))
    if not odd_vals:
        raise RuntimeError("no odd-sector eigenvalue found")
    E0_odd = float(min(odd_vals))
    Hpsi = hc * gs - float(b) * apply_X_sum(gs, n)
    residual = float(np.linalg.norm(Hpsi - E0 * gs))
    dim = 1 << n
    F_plus = float((np.sum(gs) / np.sqrt(dim)) ** 2)
    return LogicalInstance(
        n=n,
        seed=int(seed),
        b=float(b),
        J=J,
        z=z,
        hc=hc,
        K_raw=K_raw,
        K_hat=K_hat,
        scale_K=float(scale_K),
        rms_J=float(rms_J),
        E0=E0,
        E1=E1,
        E0_odd=E0_odd,
        lambda_max=lam_max,
        gap=float(E1 - E0),
        delta_odd=float(E0_odd - E0),
        gs=gs,
        F_plus=F_plus,
        j_upper_sha=sha256_f64(upper(J)),
        gs_sha=sha256_f64(gs),
        gs_min=float(np.min(gs)),
        gs_parity=float(gs_parity),
        residual=residual,
    )


def logical_energy(inst: LogicalInstance, psi: np.ndarray) -> float:
    Hpsi = inst.hc * psi - inst.b * apply_X_sum(psi, inst.n)
    return float(np.vdot(psi, Hpsi).real)


def logical_zz(inst: LogicalInstance, psi: np.ndarray) -> float:
    return float(np.vdot(psi, inst.hc * psi).real)


def plus_state(n: int) -> np.ndarray:
    return np.full(1 << n, 2.0 ** (-0.5 * n), dtype=np.complex128)
