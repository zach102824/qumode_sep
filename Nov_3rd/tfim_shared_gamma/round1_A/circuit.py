"""Logical Nov_3rd circuit, QAOA, energy, and analytic gradients.

Logical frame, matrix order rightmost-first:
    |psi> = M_L D M_{L-1} D ... M_1 D |+>
    D = exp(-i gamma G), G = Hc + theta * F
    M = exp[-i sum_cavities (c1 sum X + c2 sum XX + c3 XXX)]

Physical frame (verified against this one, not used for the sweep):
    |0> -> [W exp(-i gamma G) W -> shell]^L -> W
which equals the logical circuit because |+> = W|0> and M = W shell W.

Independent-angle QAOA is the same skeleton with theta=0, c2=c3=0, and a
separate gamma_l, beta_l=c1_l each layer. Shared-gamma QAOA keeps one gamma.
Native D keeps one gamma, theta fixed at 0, and a free (c1,c2,c3) each layer.
"""
from __future__ import annotations

import numpy as np

from .hamiltonian import Problem, apply_X_sum

THETA_METHODS = ("family_A", "random_K", "refl_K")
FEATURE_OF = {"family_A": "physical", "random_K": "random", "refl_K": "refl"}


def fwht(v: np.ndarray) -> np.ndarray:
    """Normalized W = H^{otimes n} with site 0 as LSB. W is Hermitian and W^2 = I."""
    x = np.asarray(v, dtype=np.complex128).copy()
    n = int(x.shape[0])
    h = 1
    while h < n:
        x = x.reshape(n // (2 * h), 2, h)
        a = x[:, 0, :].copy()
        b = x[:, 1, :].copy()
        x[:, 0, :] = a + b
        x[:, 1, :] = a - b
        x = np.reshape(x, (n,))
        h *= 2
    x *= n ** -0.5
    return x


def phi_table(c1: float, c2: float, c3: float) -> np.ndarray:
    k = np.arange(4, dtype=np.float64)
    return c1 * (3.0 - 2.0 * k) + c2 * (((3.0 - 2.0 * k) ** 2 - 3.0) / 2.0) + c3 * ((-1.0) ** k)


def unpack(method: str, x: np.ndarray, L: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return per-layer (gammas, thetas, cs) with shared scalars already broadcast."""
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if method == "qaoa":
        if x.size != 2 * L:
            raise ValueError(f"qaoa expects {2*L} params, got {x.size}")
        gammas = x[:L].copy()
        thetas = np.zeros(L, dtype=np.float64)
        cs = np.zeros((L, 3), dtype=np.float64)
        cs[:, 0] = x[L:]
        return gammas, thetas, cs
    if method == "shared_qaoa":
        if x.size != 1 + L:
            raise ValueError(f"shared_qaoa expects {1+L} params, got {x.size}")
        gammas = np.full(L, float(x[0]))
        thetas = np.zeros(L, dtype=np.float64)
        cs = np.zeros((L, 3), dtype=np.float64)
        cs[:, 0] = x[1:]
        return gammas, thetas, cs
    if method == "native_D":
        if x.size != 1 + 3 * L:
            raise ValueError(f"native_D expects {1+3*L} params, got {x.size}")
        gammas = np.full(L, float(x[0]))
        thetas = np.zeros(L, dtype=np.float64)
        cs = x[1:].reshape(L, 3).copy()
        return gammas, thetas, cs
    if method not in THETA_METHODS:
        raise KeyError(method)
    if x.size != 2 + 3 * L:
        raise ValueError(f"{method} expects {2+3*L} params, got {x.size}")
    gammas = np.full(L, float(x[0]))
    thetas = np.full(L, float(x[1]))
    cs = x[2:].reshape(L, 3).copy()
    return gammas, thetas, cs


def pack_shell(gamma: float, theta: float | None, cs: np.ndarray, with_theta: bool) -> np.ndarray:
    cs = np.asarray(cs, dtype=np.float64).reshape(-1)
    if with_theta:
        return np.concatenate([[float(gamma), float(0.0 if theta is None else theta)], cs])
    return np.concatenate([[float(gamma)], cs])


def pack_qaoa(gammas: np.ndarray, betas: np.ndarray) -> np.ndarray:
    return np.concatenate([np.asarray(gammas, dtype=np.float64), np.asarray(betas, dtype=np.float64)])


def fk_diag_for(problem: Problem, method: str) -> np.ndarray | None:
    if method in FEATURE_OF:
        return problem.features[FEATURE_OF[method]].fk_diag
    return None


def forward(
    problem: Problem,
    gammas: np.ndarray,
    thetas: np.ndarray,
    cs: np.ndarray,
    fk: np.ndarray | None,
    store: bool = False,
):
    """Apply the logical circuit. Returns the output state and optional snaps."""
    psi = problem.plus.copy()
    hc = problem.hc_diag
    snaps = [] if store else None
    L = int(gammas.shape[0])
    for ell in range(L):
        gdiag = hc if fk is None else hc + float(thetas[ell]) * fk
        phase = float(gammas[ell]) * gdiag
        psi_before = psi
        psi_mid = psi_before * np.exp(-1j * phase)
        x1 = fwht(psi_mid)
        phi = cs[ell, 0] * problem.w1 + cs[ell, 1] * problem.w2 + cs[ell, 2] * problem.w3
        x2 = x1 * np.exp(-1j * phi)
        psi = fwht(x2)
        if store:
            snaps.append((psi_before, phase, gdiag, x1, phi, x2, psi_mid))
    return psi, snaps


def apply_H(problem: Problem, psi: np.ndarray) -> np.ndarray:
    return problem.hc_diag * psi + problem.B * apply_X_sum(psi, problem.n)


def energy_of_state(problem: Problem, psi: np.ndarray) -> float:
    return float(np.vdot(psi, apply_H(problem, psi)).real)


def fidelity_of_state(problem: Problem, psi: np.ndarray) -> float:
    return float(abs(np.vdot(problem.gs, psi)) ** 2)


def eta_of(E: float, E0: float, Emax: float) -> float:
    den = E0 - Emax
    if abs(den) < 1e-15:
        return float("nan")
    return float((E - Emax) / den)


def _grad_from_snaps(problem: Problem, psi: np.ndarray, snaps, gammas, thetas, cs, fk):
    lam = apply_H(problem, psi)
    L = len(snaps)
    g_gamma = np.zeros(L, dtype=np.float64)
    g_theta = np.zeros(L, dtype=np.float64)
    g_c = np.zeros((L, 3), dtype=np.float64)
    for ell in range(L - 1, -1, -1):
        _psi_before, phase, gdiag, x1, phi, x2, psi_mid = snaps[ell]
        lam_x2 = fwht(lam)
        overlap = np.conj(lam_x2) * x2
        g_c[ell, 0] = 2.0 * float(np.imag(np.sum(overlap * problem.w1)))
        g_c[ell, 1] = 2.0 * float(np.imag(np.sum(overlap * problem.w2)))
        g_c[ell, 2] = 2.0 * float(np.imag(np.sum(overlap * problem.w3)))
        lam_x1 = lam_x2 * np.exp(1j * phi)
        lam_mid = fwht(lam_x1)
        ovD = np.conj(lam_mid) * psi_mid
        # phase = gamma * gdiag, gdiag = Hc + theta * F
        g_gamma[ell] = 2.0 * float(np.imag(np.sum(ovD * gdiag)))
        if fk is not None:
            g_theta[ell] = 2.0 * float(np.imag(np.sum(ovD * (float(gammas[ell]) * fk))))
        lam = lam_mid * np.exp(1j * phase)
    return g_gamma, g_theta, g_c


def energy_only(problem: Problem, method: str, x: np.ndarray, L: int):
    """Energy and state. The ground state is not referenced."""
    gammas, thetas, cs = unpack(method, x, L)
    psi, _snaps = forward(problem, gammas, thetas, cs, fk_diag_for(problem, method), store=False)
    return energy_of_state(problem, psi), psi


def energy_and_grad(problem: Problem, method: str, x: np.ndarray, L: int):
    """Objective energy and the gradient of the packed parameter vector.

    The ground state is not referenced.
    """
    gammas, thetas, cs = unpack(method, x, L)
    fk = fk_diag_for(problem, method)
    psi, snaps = forward(problem, gammas, thetas, cs, fk, store=True)
    E = energy_of_state(problem, psi)
    g_gamma, g_theta, g_c = _grad_from_snaps(problem, psi, snaps, gammas, thetas, cs, fk)
    if method == "qaoa":
        grad = np.concatenate([g_gamma, g_c[:, 0]])
    elif method == "shared_qaoa":
        grad = np.concatenate([[g_gamma.sum()], g_c[:, 0]])
    elif method == "native_D":
        grad = np.concatenate([[g_gamma.sum()], g_c.reshape(-1)])
    else:
        grad = np.concatenate([[g_gamma.sum(), g_theta.sum()], g_c.reshape(-1)])
    return E, grad, psi


def state_from_params(problem: Problem, method: str, x: np.ndarray, L: int) -> np.ndarray:
    gammas, thetas, cs = unpack(method, x, L)
    psi, _ = forward(problem, gammas, thetas, cs, fk_diag_for(problem, method), store=False)
    return psi


def layer_observables(problem: Problem, method: str, x: np.ndarray, L: int) -> list[dict]:
    """Energy pieces after each completed layer. Fidelity is diagnostic only."""
    gammas, thetas, cs = unpack(method, x, L)
    fk = fk_diag_for(problem, method)
    psi = problem.plus.copy()
    rows = []
    hc = problem.hc_diag
    for ell in range(L):
        gdiag = hc if fk is None else hc + float(thetas[ell]) * fk
        phase = float(gammas[ell]) * gdiag
        psi = psi * np.exp(-1j * phase)
        x1 = fwht(psi)
        phi = cs[ell, 0] * problem.w1 + cs[ell, 1] * problem.w2 + cs[ell, 2] * problem.w3
        psi = fwht(x1 * np.exp(-1j * phi))
        E = energy_of_state(problem, psi)
        Hc = float(np.vdot(psi, hc * psi).real)
        Fk = float(np.vdot(psi, (fk if fk is not None else 0.0) * psi).real) if fk is not None else 0.0
        rows.append(
            {
                "layer": ell + 1,
                "E": E,
                "E_minus_E0": E - problem.E0,
                "Hc": Hc,
                "F_normalized": Fk,
                "F": fidelity_of_state(problem, psi),
                "eta": eta_of(E, problem.E0, problem.Emax),
            }
        )
    return rows


def metrics_from_params(problem: Problem, method: str, x: np.ndarray, L: int) -> dict:
    psi = state_from_params(problem, method, x, L)
    E = energy_of_state(problem, psi)
    norm = float(np.vdot(psi, psi).real)
    return {
        "E": E,
        "E_minus_E0": E - problem.E0,
        "F": fidelity_of_state(problem, psi),
        "eta": eta_of(E, problem.E0, problem.Emax),
        "norm": norm,
        "psi": psi,
    }


def apply_physical_from_vacuum(problem: Problem, gamma: float, theta: float, cs: np.ndarray, fk):
    """Hardware-order circuit on the vacuum, final W included.

    UcX = W exp(-i gamma G) W, then shell, repeated, then final W.
    """
    psi = np.zeros(problem.dim, dtype=np.complex128)
    psi[0] = 1.0
    hc = problem.hc_diag
    gdiag = hc if fk is None else hc + float(theta) * fk
    phase = np.exp(-1j * float(gamma) * gdiag)
    for ell in range(cs.shape[0]):
        psi = fwht(psi)
        psi = psi * phase
        psi = fwht(psi)
        phi = cs[ell, 0] * problem.w1 + cs[ell, 1] * problem.w2 + cs[ell, 2] * problem.w3
        psi = psi * np.exp(-1j * phi)
    psi = fwht(psi)
    return psi


def apply_physical_unitary(problem: Problem, psi_in: np.ndarray, gamma: float, theta: float, cs, fk):
    """Same gates as the vacuum circuit, acting on an arbitrary Fock state."""
    psi = np.asarray(psi_in, dtype=np.complex128).copy()
    hc = problem.hc_diag
    gdiag = hc if fk is None else hc + float(theta) * fk
    phase = np.exp(-1j * float(gamma) * gdiag)
    for ell in range(cs.shape[0]):
        psi = fwht(psi)
        psi = psi * phase
        psi = fwht(psi)
        phi = cs[ell, 0] * problem.w1 + cs[ell, 1] * problem.w2 + cs[ell, 2] * problem.w3
        psi = psi * np.exp(-1j * phi)
    psi = fwht(psi)
    return psi


def logical_on_state(problem: Problem, psi_in: np.ndarray, gamma: float, theta: float, cs, fk):
    """Logical [D -> M]^L acting on an arbitrary state (not necessarily |+>)."""
    psi = np.asarray(psi_in, dtype=np.complex128).copy()
    hc = problem.hc_diag
    gdiag = hc if fk is None else hc + float(theta) * fk
    phase_factor = np.exp(-1j * float(gamma) * gdiag)
    for ell in range(cs.shape[0]):
        psi = psi * phase_factor
        x1 = fwht(psi)
        phi = cs[ell, 0] * problem.w1 + cs[ell, 1] * problem.w2 + cs[ell, 2] * problem.w3
        psi = fwht(x1 * np.exp(-1j * phi))
    return psi
