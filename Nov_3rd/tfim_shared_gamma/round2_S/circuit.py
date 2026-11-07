"""Family S circuit and analytic energy gradients.

Logical order is |+> -> [D -> M]^L.
D = exp(-i gamma G) with one gamma.
Family S: G = Hc_inter + thetaIn Hc_intra + thetaK FK_inter.
thetaIn = 1 with a free thetaK is inter-K A, not the original full-K A.
"""
from __future__ import annotations

import numpy as np

from Nov_3rd.tfim_shared_gamma.round1_A.circuit import fwht
from Nov_3rd.tfim_shared_gamma.round1_A.hamiltonian import apply_X_sum

from .frozen import FrozenSystem, g_diag, g_partials

SHELL_METHODS = ("family_S", "shuffled_intra", "full_K_A", "inter_K_A", "thetaK0")
TWO_THETA = ("family_S", "shuffled_intra", "same_G_qaoa")


def param_count(method: str, L: int) -> int:
    if method == "qaoa":
        return 2 * L
    if method == "same_G_qaoa":
        return 2 * L + 2
    if method in ("family_S", "shuffled_intra"):
        return 3 + 3 * L
    if method in ("full_K_A", "inter_K_A", "thetaK0"):
        return 2 + 3 * L
    raise KeyError(method)


def unpack(method: str, x: np.ndarray, L: int):
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if x.size != param_count(method, L):
        raise ValueError(f"{method} L={L} expects {param_count(method, L)} params, got {x.size}")
    cs = np.zeros((L, 3), dtype=np.float64)
    if method == "qaoa":
        return x[:L].copy(), 1.0, 0.0, _put_beta(cs, x[L:])
    if method == "same_G_qaoa":
        return x[:L].copy(), float(x[-2]), float(x[-1]), _put_beta(cs, x[L:-2])
    if method in ("family_S", "shuffled_intra"):
        cs = x[3:].reshape(L, 3).copy()
        return np.full(L, float(x[0])), float(x[1]), float(x[2]), cs
    if method in ("full_K_A", "inter_K_A"):
        cs = x[2:].reshape(L, 3).copy()
        return np.full(L, float(x[0])), 1.0, float(x[1]), cs
    if method == "thetaK0":
        cs = x[2:].reshape(L, 3).copy()
        return np.full(L, float(x[0])), float(x[1]), 0.0, cs
    raise KeyError(method)


def _put_beta(cs, beta):
    cs = cs.copy()
    cs[:, 0] = np.asarray(beta, dtype=np.float64)
    return cs


def pack_shell(method: str, gamma: float, theta_in: float, theta_k: float, cs: np.ndarray) -> np.ndarray:
    cs = np.asarray(cs, dtype=np.float64).reshape(-1)
    if method in ("family_S", "shuffled_intra"):
        return np.concatenate([[gamma, theta_in, theta_k], cs])
    if method in ("full_K_A", "inter_K_A"):
        return np.concatenate([[gamma, theta_k], cs])
    if method == "thetaK0":
        return np.concatenate([[gamma, theta_in], cs])
    raise KeyError(method)


def pack_qaoa(method: str, gammas, betas, theta_in: float = 1.0, theta_k: float = 0.0) -> np.ndarray:
    g = np.asarray(gammas, dtype=np.float64).reshape(-1)
    b = np.asarray(betas, dtype=np.float64).reshape(-1)
    if method == "qaoa":
        return np.concatenate([g, b])
    if method == "same_G_qaoa":
        return np.concatenate([g, b, [theta_in, theta_k]])
    raise KeyError(method)


def _prepare(system: FrozenSystem, method: str, theta_in: float, theta_k: float):
    g = g_diag(system, method, theta_in, theta_k)
    d_in, d_k = g_partials(system, method)
    return g, d_in, d_k


def energy_only(system: FrozenSystem, method: str, x: np.ndarray, L: int):
    gammas, theta_in, theta_k, cs = unpack(method, x, L)
    g, _din, _dk = _prepare(system, method, theta_in, theta_k)
    psi = system.plus.copy()
    for ell in range(L):
        psi = psi * np.exp(-1j * float(gammas[ell]) * g)
        x1 = fwht(psi)
        phi = cs[ell, 0] * system.w1 + cs[ell, 1] * system.w2 + cs[ell, 2] * system.w3
        psi = fwht(x1 * np.exp(-1j * phi))
    return _energy(system, psi), psi


def energy_and_grad(system: FrozenSystem, method: str, x: np.ndarray, L: int):
    gammas, theta_in, theta_k, cs = unpack(method, x, L)
    g, d_in, d_k = _prepare(system, method, theta_in, theta_k)
    psi = system.plus.copy()
    snaps = []
    for ell in range(L):
        phase = float(gammas[ell]) * g
        psi_mid = psi * np.exp(-1j * phase)
        x1 = fwht(psi_mid)
        phi = cs[ell, 0] * system.w1 + cs[ell, 1] * system.w2 + cs[ell, 2] * system.w3
        x2 = x1 * np.exp(-1j * phi)
        psi_out = fwht(x2)
        snaps.append((phase, psi_mid, phi, x2, psi_out))
        psi = psi_out
    E = _energy(system, psi)
    lam = system.hc_full * psi + system.B * apply_X_sum(psi, system.n)
    g_gamma = np.zeros(L)
    g_in = 0.0
    g_k = 0.0
    g_c = np.zeros((L, 3))
    for ell in range(L - 1, -1, -1):
        phase, psi_mid, phi, x2, _psi_out = snaps[ell]
        lam_x2 = fwht(lam)
        overlap = np.conj(lam_x2) * x2
        g_c[ell, 0] = 2.0 * float(np.imag(np.sum(overlap * system.w1)))
        g_c[ell, 1] = 2.0 * float(np.imag(np.sum(overlap * system.w2)))
        g_c[ell, 2] = 2.0 * float(np.imag(np.sum(overlap * system.w3)))
        lam_mid = fwht(lam_x2 * np.exp(1j * phi))
        ov = np.conj(lam_mid) * psi_mid
        # phase = gamma * g, g = g(theta)
        g_gamma[ell] = 2.0 * float(np.imag(np.sum(ov * g)))
        if d_in is not None:
            g_in += 2.0 * float(np.imag(np.sum(ov * (float(gammas[ell]) * d_in))))
        if d_k is not None:
            g_k += 2.0 * float(np.imag(np.sum(ov * (float(gammas[ell]) * d_k))))
        lam = lam_mid * np.exp(1j * phase)
    grad = _assemble(method, L, g_gamma, g_in, g_k, g_c)
    return E, grad, psi


def _assemble(method, L, g_gamma, g_in, g_k, g_c):
    if method == "qaoa":
        return np.concatenate([g_gamma, g_c[:, 0]])
    if method == "same_G_qaoa":
        return np.concatenate([g_gamma, g_c[:, 0], [g_in, g_k]])
    if method in ("family_S", "shuffled_intra"):
        return np.concatenate([[g_gamma.sum(), g_in, g_k], g_c.reshape(-1)])
    if method in ("full_K_A", "inter_K_A"):
        return np.concatenate([[g_gamma.sum(), g_k], g_c.reshape(-1)])
    if method == "thetaK0":
        return np.concatenate([[g_gamma.sum(), g_in], g_c.reshape(-1)])
    raise KeyError(method)


def _energy(system: FrozenSystem, psi: np.ndarray) -> float:
    Hpsi = system.hc_full * psi + system.B * apply_X_sum(psi, system.n)
    return float(np.vdot(psi, Hpsi).real)


def fidelity(system: FrozenSystem, psi: np.ndarray) -> float:
    return float(abs(np.vdot(system.gs, psi)) ** 2)


def eta(E: float, E0: float, Emax: float) -> float:
    den = E0 - Emax
    return float("nan") if abs(den) < 1e-15 else float((E - Emax) / den)


def layer_rows(system: FrozenSystem, method: str, x: np.ndarray, L: int) -> list[dict]:
    gammas, theta_in, theta_k, cs = unpack(method, x, L)
    g, _a, _b = _prepare(system, method, theta_in, theta_k)
    psi = system.plus.copy()
    rows = []
    for ell in range(L):
        psi = psi * np.exp(-1j * float(gammas[ell]) * g)
        phi = cs[ell, 0] * system.w1 + cs[ell, 1] * system.w2 + cs[ell, 2] * system.w3
        psi = fwht(fwht(psi) * np.exp(-1j * phi))
        E = _energy(system, psi)
        rows.append(
            {
                "layer": ell + 1,
                "E": E,
                "energy_error": E - system.E0,
                "F": fidelity(system, psi),
            }
        )
    return rows
