"""Statevector QAOA with the paper's half-angle convention (Definition 2 / Eq. 13).

Mixer: ``exp(-i β X_j / 2)`` on every qubit (Yao ``Rx(β)``).

Cost: the PDF writes ``exp(-i γ H / 2)``, but the published angles and the
authors' analytic helper (``exp(+i γ/2)`` on the forward branch) were trained
with the opposite cost sign. Using Eq. (13) as written with those positive γ
makes p=1 *worse* than random; ``exp(+i γ H / 2)`` beats random on every
satisfiable 8-SAT instance we tried. That is Eq. (13) with γ → −γ_published.
"""

from __future__ import annotations

import numpy as np

# +1 matches published angles / analytic code; −1 is Eq. (13) literally.
COST_PHASE_SIGN = +1.0


def plus_state(n: int) -> np.ndarray:
    dim = 1 << int(n)
    return np.full(dim, 1.0 / np.sqrt(dim), dtype=np.complex128)


def apply_mixer(psi: np.ndarray, beta: float, n: int) -> np.ndarray:
    """Apply ⊗_j exp(-i β X_j / 2). Qubit j is bit j of the computational index (LSB = x_0)."""
    c = np.cos(0.5 * float(beta))
    s = -1j * np.sin(0.5 * float(beta))
    t = np.ascontiguousarray(psi, dtype=np.complex128).reshape(-1)
    n = int(n)
    for q in range(n):
        x = t.reshape(1 << (n - 1 - q), 2, 1 << q)
        t0 = np.ascontiguousarray(x[:, 0, :])
        t1 = np.ascontiguousarray(x[:, 1, :])
        y = np.empty_like(x)
        y[:, 0, :] = c * t0 + s * t1
        y[:, 1, :] = s * t0 + c * t1
        t = np.ascontiguousarray(y.reshape(-1))
    return t


def qaoa_state(
    betas: np.ndarray,
    gammas: np.ndarray,
    energy: np.ndarray,
    n: int | None = None,
    *,
    cost_phase_sign: float = COST_PHASE_SIGN,
) -> np.ndarray:
    betas = np.asarray(betas, dtype=float).reshape(-1)
    gammas = np.asarray(gammas, dtype=float).reshape(-1)
    if betas.size != gammas.size:
        raise ValueError("betas and gammas must have the same length")
    e = np.asarray(energy, dtype=float).reshape(-1)
    n = int(n) if n is not None else int(np.log2(e.size))
    if e.size != (1 << n):
        raise ValueError(f"energy length {e.size} is not 2^{n}")
    psi = plus_state(n)
    sign = float(cost_phase_sign)
    for beta, gamma in zip(betas, gammas):
        psi = psi * np.exp(1j * sign * 0.5 * float(gamma) * e)
        psi = apply_mixer(psi, float(beta), n)
    return psi


def qaoa_success(
    betas: np.ndarray,
    gammas: np.ndarray,
    energy: np.ndarray,
    n: int | None = None,
    **kwargs,
) -> float:
    return success_probability(qaoa_state(betas, gammas, energy, n, **kwargs), energy)


def success_probability(psi: np.ndarray, energy: np.ndarray) -> float:
    e = np.asarray(energy).reshape(-1)
    p = np.abs(np.asarray(psi, dtype=np.complex128).reshape(-1)) ** 2
    total = float(p.sum())
    if total <= 0.0:
        return 0.0
    sat = e == 0
    if not np.any(sat):
        return 0.0
    return float(p.real[sat].sum() / total)
