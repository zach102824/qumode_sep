"""Quantum p=3 spin glass + power-law TFIM helpers.

Glass: H = -s ∑ J_ijk ZZZ - Γ ∑ X, J~N(0,1), s=sqrt(3)/n.
Power-law p=3 (ring): J_ijk = ξ r^{-α}/N, N=(∑r^{-α})/n, r=ring diameter;
ξ=±1 (signed) or ξ=1 (ferro). See make_powerlaw_instance.
Power-law 2-body TFIM: J_ij = ξ |i-j|^{-α}/N (open 1D chain). See make_tfim2_plaw_instance.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Iterable

import numpy as np
import torch

P = 3


def scale_p3(n: int) -> float:
    """Intensive p=3 SK scale: sqrt(p! / (2 n^{p-1})) = sqrt(3)/n."""
    return float(np.sqrt(3.0) / float(n))


def all_triples(n: int) -> np.ndarray:
    return np.asarray(list(combinations(range(n), 3)), dtype=np.int64)


@dataclass
class PSpinInstance:
    n: int
    Gamma: float
    seed: int
    J: np.ndarray          # (n_triples,) float64
    triples: np.ndarray    # (n_triples, 3)
    scale: float
    Hc_diag: np.ndarray    # (2**n,) eigenvalues of H_c = -s ∑ J ZZZ
    E0: float
    gs: np.ndarray         # (2**n,) complex128 ground state (Z basis, var0=MSB)


def _bit_z(x: np.ndarray, bit: int, n: int) -> np.ndarray:
    """Z eigenvalue ±1 for variable `bit` (0=MSB) on bitstring indices x."""
    shift = n - 1 - bit
    return np.where(((x >> shift) & 1) == 0, 1.0, -1.0)


def hc_diagonal(n: int, triples: np.ndarray, J: np.ndarray, scale: float) -> np.ndarray:
    """H_c = -scale ∑ J_ijk Z_i Z_j Z_k on computational basis (var0=MSB)."""
    dim = 1 << n
    x = np.arange(dim, dtype=np.int64)
    Hc = np.zeros(dim, dtype=np.float64)
    for (i, j, k), jval in zip(triples, J):
        zzz = _bit_z(x, int(i), n) * _bit_z(x, int(j), n) * _bit_z(x, int(k), n)
        Hc -= float(scale) * float(jval) * zzz
    return Hc


def apply_H(v: np.ndarray, Hc_diag: np.ndarray, Gamma: float, n: int) -> np.ndarray:
    """H|v⟩ = Hc_diag * v - Γ ∑_i X_i |v⟩."""
    out = Hc_diag * v
    dim = v.shape[0]
    idx = np.arange(dim)
    for i in range(n):
        shift = n - 1 - i
        partner = idx ^ (1 << shift)
        out = out - float(Gamma) * v[partner]
    return out


def ground_state_dense(Hc_diag: np.ndarray, Gamma: float, n: int) -> tuple[float, np.ndarray]:
    """Dense Hermitian eigh for n <= 12 (dim<=4096)."""
    dim = 1 << n
    H = np.diag(Hc_diag.astype(np.float64)).astype(np.complex128)
    idx = np.arange(dim)
    for i in range(n):
        shift = n - 1 - i
        partner = idx ^ (1 << shift)
        mask = partner > idx
        ii = idx[mask]
        jj = partner[mask]
        H[ii, jj] -= float(Gamma)
        H[jj, ii] -= float(Gamma)
    evals, evecs = np.linalg.eigh(H)
    return float(evals[0].real), evecs[:, 0].astype(np.complex128)


def ground_state_lanczos(
    Hc_diag: np.ndarray,
    Gamma: float,
    n: int,
    k: int = 64,
    n_iter: int = 120,
    seed: int = 0,
) -> tuple[float, np.ndarray]:
    """Memory-light Lanczos for smallest eigenvalue (n=15)."""
    dim = Hc_diag.shape[0]
    rng = np.random.default_rng(seed)
    # Shift-invert not needed; use plain Lanczos on H, take min Ritz.
    v0 = rng.normal(size=dim) + 1j * rng.normal(size=dim)
    v0 = v0 / np.linalg.norm(v0)

    V = []
    alphas = []
    betas = []
    v_prev = np.zeros(dim, dtype=np.complex128)
    v = v0.astype(np.complex128)
    beta = 0.0
    for j in range(n_iter):
        V.append(v.copy())
        w = apply_H(v, Hc_diag, Gamma, n)
        alpha = np.vdot(v, w).real
        alphas.append(alpha)
        w = w - beta * v_prev - alpha * v
        # full reorthogonalization (dim=32k, n_iter~80 → OK)
        for vi in V:
            w = w - np.vdot(vi, w) * vi
        beta = float(np.linalg.norm(w))
        betas.append(beta)
        if beta < 1e-12:
            break
        v_prev = v
        v = w / beta
        if len(V) >= k and (j + 1) % 8 == 0:
            # early Ritz
            m = len(alphas)
            T = np.zeros((m, m), dtype=np.float64)
            for i, a in enumerate(alphas):
                T[i, i] = a
            for i in range(m - 1):
                T[i, i + 1] = betas[i]
                T[i + 1, i] = betas[i]
            evals, evecs = np.linalg.eigh(T)
            if j + 1 >= n_iter - 1:
                break

    m = len(alphas)
    T = np.zeros((m, m), dtype=np.float64)
    for i, a in enumerate(alphas):
        T[i, i] = a
    for i in range(m - 1):
        T[i, i + 1] = betas[i]
        T[i + 1, i] = betas[i]
    evals, evecs = np.linalg.eigh(T)
    E0 = float(evals[0])
    coeffs = evecs[:, 0]
    gs = np.zeros(dim, dtype=np.complex128)
    for i, vi in enumerate(V[:m]):
        gs += coeffs[i] * vi
    gs = gs / np.linalg.norm(gs)
    # Rayleigh polish
    for _ in range(3):
        w = apply_H(gs, Hc_diag, Gamma, n)
        E0 = float(np.vdot(gs, w).real)
        # one inverse-iteration-ish residual step toward GS: gs ← gs - τ (H-E)gs
        r = w - E0 * gs
        gs = gs - 0.5 * r
        gs = gs / np.linalg.norm(gs)
    E0 = float(np.vdot(gs, apply_H(gs, Hc_diag, Gamma, n)).real)
    return E0, gs


def ground_state(Hc_diag: np.ndarray, Gamma: float, n: int, seed: int = 0) -> tuple[float, np.ndarray]:
    # dense only for tiny n; Lanczos for n>=10 (n=12 dense ~12s/inst)
    if n <= 9:
        return ground_state_dense(Hc_diag, Gamma, n)
    return ground_state_lanczos(Hc_diag, Gamma, n, seed=seed)


def make_instance(n: int, Gamma: float, seed: int) -> PSpinInstance:
    rng = np.random.default_rng(seed)
    triples = all_triples(n)
    J = rng.normal(0.0, 1.0, size=len(triples))
    s = scale_p3(n)
    Hc = hc_diagonal(n, triples, J, s)
    E0, gs = ground_state(Hc, Gamma, n, seed=seed + 17)
    # fix global phase: largest-magnitude entry → real positive
    k = int(np.argmax(np.abs(gs)))
    phase = np.abs(gs[k]) / (gs[k] + 1e-30)
    gs = gs * phase
    return PSpinInstance(
        n=n,
        Gamma=float(Gamma),
        seed=int(seed),
        J=J.astype(np.float64),
        triples=triples,
        scale=s,
        Hc_diag=Hc.astype(np.float64),
        E0=E0,
        gs=gs.astype(np.complex128),
    )


def make_dataset(
    n: int,
    Gamma: float,
    n_instances: int,
    seed0: int,
) -> list[PSpinInstance]:
    return [make_instance(n, Gamma, seed0 + i) for i in range(n_instances)]


def hc_torch(inst: PSpinInstance, device) -> torch.Tensor:
    return torch.as_tensor(inst.Hc_diag, dtype=torch.float32, device=device)


def gs_torch(inst: PSpinInstance, device) -> torch.Tensor:
    return torch.as_tensor(inst.gs, dtype=torch.complex64, device=device)


# ---------------------------------------------------------------------------
# Power-law p=3 TFIM (ring geometry)
# ---------------------------------------------------------------------------
# H = -Σ_{i<j<k} J_ijk Z_i Z_j Z_k - Γ Σ_i X_i
# J_ijk = N^{-1} r_ijk^{-α},  r = ring diameter of {i,j,k}.
# Kac-style norm: N = (Σ_t r_t^{-α}) / n  so classical all-+ energy = -n
# (intensive per site; same spirit as √3/n all-to-all glass).
# α=0 → flat all-to-all ferro. Geometry: RING (documented).


def ring_diameter(i: int, j: int, k: int, n: int) -> int:
    """Shortest arc length (edges) covering {i,j,k} on a ring of n sites."""
    pts = sorted((int(i) % n, int(j) % n, int(k) % n))
    if pts[0] == pts[1] or pts[1] == pts[2] or pts[0] == pts[2]:
        raise ValueError(f"need 3 distinct sites, got {pts}")
    gaps = (pts[1] - pts[0], pts[2] - pts[1], pts[0] + n - pts[2])
    return int(n - max(gaps))


def powerlaw_weights(n: int, alpha: float) -> tuple[np.ndarray, np.ndarray, float]:
    """Return (triples, raw_w, N_norm) with raw_w = r^{-α}, N_norm = sum(w)/n."""
    triples = all_triples(n)
    w = np.empty(len(triples), dtype=np.float64)
    for t, (i, j, k) in enumerate(triples):
        r = ring_diameter(int(i), int(j), int(k), n)
        # r >= 2 for distinct triples; guard r>=1
        w[t] = float(r) ** (-float(alpha))
    N_norm = float(np.sum(w) / float(n))
    if N_norm <= 0:
        raise RuntimeError("N_norm nonpositive")
    return triples, w, N_norm


def make_powerlaw_instance(
    n: int,
    Gamma: float,
    alpha: float,
    seed: int = 0,
    signed: bool = False,
) -> PSpinInstance:
    """Ferro (default) or Rademacher-signed power-law p=3 on a ring.

    seed only affects GS Lanczos polish (and signs if signed=True).
    scale field stores N_norm; J stores J_ijk = (±) w / N_norm.
    """
    triples, w, N_norm = powerlaw_weights(n, alpha)
    if signed:
        rng = np.random.default_rng(seed)
        signs = rng.choice(np.array([-1.0, 1.0]), size=len(w))
        J = (signs * w) / N_norm
    else:
        J = w / N_norm
    Hc = hc_diagonal(n, triples, J, scale=1.0)  # J already includes norm
    E0, gs = ground_state(Hc, Gamma, n, seed=seed + 17)
    k = int(np.argmax(np.abs(gs)))
    phase = np.abs(gs[k]) / (gs[k] + 1e-30)
    gs = gs * phase
    return PSpinInstance(
        n=n,
        Gamma=float(Gamma),
        seed=int(seed),
        J=J.astype(np.float64),
        triples=triples,
        scale=float(N_norm),
        Hc_diag=Hc.astype(np.float64),
        E0=E0,
        gs=gs.astype(np.complex128),
    )


def make_powerlaw_dataset(
    n: int,
    Gamma: float,
    alpha: float,
    n_instances: int,
    seed0: int,
    signed: bool = True,
) -> list[PSpinInstance]:
    """Signed power-law instances (different ξ). Ferro → n_instances=1."""
    if not signed:
        return [make_powerlaw_instance(n, Gamma, alpha, seed=seed0, signed=False)]
    return [
        make_powerlaw_instance(n, Gamma, alpha, seed=seed0 + i, signed=True)
        for i in range(n_instances)
    ]


# ---------------------------------------------------------------------------
# Power-law 2-body TFIM (open 1D chain)
# ---------------------------------------------------------------------------
# H = -Σ_{i<j} J_ij Z_i Z_j - Γ Σ_i X_i
# J_ij = ξ_ij r_ij^{-α} / N,  r = |i-j| (NO wrap-around), N = (Σ r^{-α})/n
# Kac-style: classical ferro all-+ energy = -n.


def all_pairs(n: int) -> np.ndarray:
    return np.asarray(list(combinations(range(n), 2)), dtype=np.int64)


def chain_pair_dist(i: int, j: int) -> int:
    """Open 1D chain distance: |i-j| (no periodic wrap)."""
    return int(abs(int(i) - int(j)))


def ring_pair_dist(i: int, j: int, n: int) -> int:
    """Legacy ring distance (kept for reference; 2-body TFIM uses chain)."""
    d = abs(int(i) - int(j))
    return int(min(d, n - d))


def hc_diagonal_pairs(n: int, pairs: np.ndarray, J: np.ndarray, scale: float = 1.0) -> np.ndarray:
    """H_c = -scale ∑ J_ij Z_i Z_j on computational basis (var0=MSB)."""
    dim = 1 << n
    x = np.arange(dim, dtype=np.int64)
    Hc = np.zeros(dim, dtype=np.float64)
    for (i, j), jval in zip(pairs, J):
        zz = _bit_z(x, int(i), n) * _bit_z(x, int(j), n)
        Hc -= float(scale) * float(jval) * zz
    return Hc


def powerlaw_pair_weights(n: int, alpha: float) -> tuple[np.ndarray, np.ndarray, float]:
    """Return (pairs, raw_w, N_norm) with raw_w = |i-j|^{-α}, N_norm = sum(w)/n."""
    pairs = all_pairs(n)
    w = np.empty(len(pairs), dtype=np.float64)
    for t, (i, j) in enumerate(pairs):
        r = chain_pair_dist(int(i), int(j))
        w[t] = float(r) ** (-float(alpha))
    N_norm = float(np.sum(w) / float(n))
    if N_norm <= 0:
        raise RuntimeError("N_norm nonpositive")
    return pairs, w, N_norm


def make_tfim2_plaw_instance(
    n: int,
    Gamma: float,
    alpha: float,
    seed: int = 0,
    signed: bool = True,
) -> PSpinInstance:
    """Signed (default) or ferro 2-body power-law TFIM on an open 1D chain.

    Distance r_ij = |i-j| (no wrap-around). Reuses PSpinInstance; `triples`
    holds pairs (shape n_pairs×2). scale stores N_norm; J = (±) w / N_norm.
    """
    pairs, w, N_norm = powerlaw_pair_weights(n, alpha)
    if signed:
        rng = np.random.default_rng(seed)
        signs = rng.choice(np.array([-1.0, 1.0]), size=len(w))
        J = (signs * w) / N_norm
    else:
        J = w / N_norm
    Hc = hc_diagonal_pairs(n, pairs, J, scale=1.0)
    E0, gs = ground_state(Hc, Gamma, n, seed=seed + 17)
    k = int(np.argmax(np.abs(gs)))
    phase = np.abs(gs[k]) / (gs[k] + 1e-30)
    gs = gs * phase
    return PSpinInstance(
        n=n,
        Gamma=float(Gamma),
        seed=int(seed),
        J=J.astype(np.float64),
        triples=pairs,  # pairs stored here
        scale=float(N_norm),
        Hc_diag=Hc.astype(np.float64),
        E0=E0,
        gs=gs.astype(np.complex128),
    )


def half_cut_entropy(gs: np.ndarray, n: int) -> float:
    """Von Neumann entropy of first n//2 qubits for a pure state vector."""
    nA = n // 2
    nB = n - nA
    psi = np.asarray(gs, dtype=np.complex128).reshape((1 << nA, 1 << nB))
    # Schmidt via SVD
    svals = np.linalg.svd(psi, compute_uv=False)
    p = (svals.real ** 2) + (svals.imag ** 2)  # already real nonneg; svals real
    p = np.clip(p.real, 0.0, None)
    p = p[p > 1e-16]
    p = p / p.sum()
    return float(-np.sum(p * np.log(p)))
