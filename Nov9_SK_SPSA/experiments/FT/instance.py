"""Instance generation, RMS-matched features, and the logical spectrum.

J is drawn once per (N, seed) and shared across b and every method.
R1 is the off-diagonal part of λ_min v_min v_min^T of J. It is not an
eigenfeature of H and does not use the ground state.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from Nov9_SK_SPSA.experiments.FT.params import assert_seed_allowed


def packing(N: int) -> list[list[int]]:
    if N % 3 != 0:
        raise ValueError(f"N={N} is not divisible by 3")
    m = N // 3
    return [[a, a + m, a + 2 * m] for a in range(m)]


def upper(mat: np.ndarray) -> np.ndarray:
    n = mat.shape[0]
    return np.array([mat[i, j] for i in range(n) for j in range(i + 1, n)], dtype=np.float64)


def rms(v: np.ndarray) -> float:
    v = np.asarray(v, dtype=np.float64)
    return float(np.sqrt(np.mean(np.square(v))))


def sha256_f64(v: np.ndarray) -> str:
    raw = np.ascontiguousarray(v, dtype="<f8").tobytes()
    return hashlib.sha256(raw).hexdigest()


def make_J(seed: int, N: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    u = rng.normal(0.0, 1.0 / np.sqrt(N), size=N * (N - 1) // 2)
    J = np.zeros((N, N), dtype=np.float64)
    t = 0
    for i in range(N):
        for j in range(i + 1, N):
            J[i, j] = J[j, i] = float(u[t])
            t += 1
    return J


def pair_K(J: np.ndarray) -> np.ndarray:
    K = np.array(J @ J, dtype=np.float64, copy=True)
    np.fill_diagonal(K, 0.0)
    K = 0.5 * (K + K.T)
    np.fill_diagonal(K, 0.0)
    return K


def block_coeff(source: np.ndarray, groups: list[list[int]]) -> np.ndarray:
    C = np.zeros_like(source)
    m = len(groups)
    for a in range(m):
        for b in range(a + 1, m):
            pairs = [(i, j) for i in groups[a] for j in groups[b]]
            w = float(sum(source[i, j] for i, j in pairs) / len(pairs))
            for i, j in pairs:
                C[i, j] = C[j, i] = w
    return C


def most_negative_pair(J: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    evals, evecs = np.linalg.eigh(np.asarray(J, dtype=np.float64))
    lam = float(evals[0])
    v = np.real(evecs[:, 0]).astype(np.float64)
    if v.sum() < 0:
        v = -v
    R = lam * np.outer(v, v)
    np.fill_diagonal(R, 0.0)
    return R, lam, v


def hat(raw: np.ndarray, rms_j: float) -> tuple[np.ndarray, float]:
    r = rms(upper(raw))
    if r <= 0.0:
        raise RuntimeError("zero feature")
    scale = float(rms_j) / r
    return raw * scale, scale


def z_table(N: int) -> np.ndarray:
    dim = 1 << N
    x = np.arange(dim, dtype=np.uint32)
    z = np.empty((N, dim), dtype=np.float64)
    for q in range(N):
        z[q] = 1.0 - 2.0 * ((x >> np.uint32(q)) & np.uint32(1))
    return z


def pair_diag(coeff: np.ndarray, z: np.ndarray) -> np.ndarray:
    n = coeff.shape[0]
    out = np.zeros(z.shape[1], dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            cij = float(coeff[i, j])
            if cij != 0.0:
                out += cij * z[i] * z[j]
    return out


def theta_K_prior(b: float, K_raw: np.ndarray, J: np.ndarray) -> float:
    """Hat-unit init. Raw-unit coefficient b^2/2 is only a sign/scale prior."""
    raw = (float(b) ** 2) / 2.0
    return raw * rms(upper(K_raw)) / rms(upper(J))


def feature_report(J, K_raw, M_raw, R_raw) -> dict:
    mats = {"J": J, "K": K_raw, "M": M_raw, "R1": R_raw}
    ups = {k: upper(v) for k, v in mats.items()}
    keys = list(ups)
    corr = {}
    for a in keys:
        for b in keys:
            va, vb = ups[a], ups[b]
            if np.std(va) == 0 or np.std(vb) == 0:
                corr[f"{a}_{b}"] = float("nan")
            else:
                corr[f"{a}_{b}"] = float(np.corrcoef(va, vb)[0, 1])
    A = np.column_stack([ups["J"], ups["K"], ups["M"]])
    coef, *_ = np.linalg.lstsq(A, ups["R1"], rcond=None)
    resid = float(np.linalg.norm(A @ coef - ups["R1"]) / max(np.linalg.norm(ups["R1"]), 1e-30))
    # Singular values of the column-normalized feature matrix, for rank.
    cols = np.column_stack([ups[k] / max(np.linalg.norm(ups[k]), 1e-30) for k in keys])
    s = np.linalg.svd(cols, compute_uv=False)
    return {
        "pearson_upper": corr,
        "R1_residual_vs_span_JKM": resid,
        "feature_singular_values": [float(x) for x in s],
    }


def dense_H(zz: np.ndarray, b: float) -> np.ndarray:
    dim = zz.shape[0]
    N = int(np.log2(dim))
    H = np.diag(np.asarray(zz, dtype=np.float64))
    idx = np.arange(dim)
    for q in range(N):
        H[idx, idx ^ (1 << q)] += -float(b)
    return H


def spin_parity(v: np.ndarray) -> float:
    n = int(round(np.log2(v.size)))
    flip = np.arange(v.size) ^ ((1 << n) - 1)
    return float(np.vdot(v, v[flip]).real)


@dataclass
class Instance:
    N: int
    seed: int
    b: float
    J: np.ndarray
    J_upper: np.ndarray
    J_sha256: str
    groups: list[list[int]]
    K_raw: np.ndarray
    M_raw: np.ndarray
    R1_raw: np.ndarray
    K_hat: np.ndarray
    M_hat: np.ndarray
    R1_hat: np.ndarray
    scale_K: float
    scale_M: float
    scale_R1: float
    rms_J: float
    lam_min_J: float
    v_min: np.ndarray
    theta_K_prior: float
    zz_J: np.ndarray
    zz_K: np.ndarray
    zz_M: np.ndarray
    zz_R1: np.ndarray
    E0: float
    E1: float
    E0_odd: float
    Delta_odd: float
    lam_max: float
    gs: np.ndarray
    F_plus: float
    residual: float
    feature_diag: dict
    classical_seconds: float

    def zz_cost(self, theta: dict) -> np.ndarray:
        g = self.zz_J.copy()
        if "K" in theta:
            g = g + float(theta["K"]) * self.zz_K
        if "M" in theta:
            g = g + float(theta["M"]) * self.zz_M
        if "R1" in theta:
            g = g + float(theta["R1"]) * self.zz_R1
        return g


def build_instance(N: int, seed: int, b: float, role: str = "pilot", with_spectrum: bool = True) -> Instance:
    import time

    assert_seed_allowed(seed, role)
    t0 = time.perf_counter()
    J = make_J(seed, N)
    Ju = upper(J)
    groups = packing(N)
    K = pair_K(J)
    M = block_coeff(J, groups)
    R, lam, v = most_negative_pair(J)
    rms_j = rms(Ju)
    Kh, sK = hat(K, rms_j)
    Mh, sM = hat(M, rms_j)
    Rh, sR = hat(R, rms_j)
    z = z_table(N)
    zzJ = pair_diag(J, z)
    dim = 1 << N
    if with_spectrum:
        if N > 12:
            raise RuntimeError("dense spectrum is only built for N<=12; use Lanczos above that")
        H = dense_H(zzJ, b)
        evals, evecs = np.linalg.eigh(H)
        gs = np.real(evecs[:, 0]).astype(np.float64)
        gs /= np.linalg.norm(gs)
        if gs.sum() < 0:
            gs = -gs
        E0 = float(evals[0])
        E1 = float(evals[1])
        residual = float(np.linalg.norm(H @ gs - E0 * gs))
        E0_odd = None
        for i in range(evals.size):
            if spin_parity(evecs[:, i]) < -0.5:
                E0_odd = float(evals[i])
                break
        if E0_odd is None:
            raise RuntimeError("no odd-parity eigenstate")
        F_plus = float(np.sum(gs) / np.sqrt(dim)) ** 2
        lam_max = float(evals[-1])
        delta_odd = float(E0_odd - E0)
    else:
        gs = np.zeros(dim, dtype=np.float64)
        E0 = E1 = E0_odd = lam_max = F_plus = residual = delta_odd = float("nan")
    feat = feature_report(J, K, M, R)
    t1 = time.perf_counter()
    return Instance(
        N=N,
        seed=int(seed),
        b=float(b),
        J=J,
        J_upper=Ju,
        J_sha256=sha256_f64(Ju),
        groups=groups,
        K_raw=K,
        M_raw=M,
        R1_raw=R,
        K_hat=Kh,
        M_hat=Mh,
        R1_hat=Rh,
        scale_K=sK,
        scale_M=sM,
        scale_R1=sR,
        rms_J=rms_j,
        lam_min_J=lam,
        v_min=v,
        theta_K_prior=theta_K_prior(b, K, J),
        zz_J=zzJ,
        zz_K=pair_diag(Kh, z),
        zz_M=pair_diag(Mh, z),
        zz_R1=pair_diag(Rh, z),
        E0=E0,
        E1=E1,
        E0_odd=E0_odd,
        Delta_odd=delta_odd,
        lam_max=lam_max,
        gs=gs,
        F_plus=F_plus,
        residual=residual,
        feature_diag=feat,
        classical_seconds=float(t1 - t0),
    )
