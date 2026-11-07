"""Extra gradients: same-G linear-mixer diagnostic, and free theta_Q."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

R2 = Path(__file__).resolve().parents[1] / "round2_H"
if str(R2) not in sys.path:
    sys.path.insert(0, str(R2))

from ansatz import mixer_unitary  # noqa: E402


def sameg_energy_grad(model, L: int, x: np.ndarray):
    """Independent gamma_l, shared theta, standard beta mixer. Not a legal Nov3 candidate."""
    x = np.asarray(x, dtype=np.float64)
    gammas = x[:L]
    theta = float(x[L])
    betas = x[L + 1 : 2 * L + 1]
    gdiag = model.g_diag(theta)
    psi = model.plus.copy()
    snaps = []
    for ell in range(L):
        psi = psi * np.exp(-1j * float(gammas[ell]) * gdiag)
        snaps.append(psi.copy())
        psi = model.apply_rx_all(psi, float(betas[ell]))
        snaps.append(psi.copy())
    energy = model.energy(psi)
    chi = model.apply_H(psi)
    g_gamma = np.zeros(L)
    g_beta = np.zeros(L)
    g_theta = 0.0
    for ell in range(L - 1, -1, -1):
        g_beta[ell] = 2.0 * np.imag(model.overlap_sum_x(chi, snaps[2 * ell + 1]))
        chi = model.apply_rx_all(chi, float(-betas[ell]))
        psi_d = snaps[2 * ell]
        g_gamma[ell] = 2.0 * float(np.imag(np.vdot(chi, gdiag * psi_d)))
        g_theta += 2.0 * float(np.imag(float(gammas[ell]) * np.vdot(chi, model.fk * psi_d)))
        chi = chi * np.exp(1j * float(gammas[ell]) * gdiag)
    grad = np.concatenate([g_gamma, [g_theta], g_beta])
    return float(energy), grad


def sameg_state(model, L: int, x: np.ndarray):
    x = np.asarray(x, dtype=np.float64)
    gammas, theta, betas = x[:L], float(x[L]), x[L + 1 :]
    gdiag = model.g_diag(theta)
    psi = model.plus.copy()
    for ell in range(L):
        psi = psi * np.exp(-1j * float(gammas[ell]) * gdiag)
        psi = model.apply_rx_all(psi, float(betas[ell]))
    return psi


def free_q4_energy_grad(model, L: int, x: np.ndarray, feat_k: np.ndarray, feat_q: np.ndarray):
    """G = Hc + thetaK * FK/rK + thetaQ * Q4hat. Not a polynomial of Hc."""
    x = np.asarray(x, dtype=np.float64)
    gamma = float(x[0])
    thk = float(x[1])
    thq = float(x[2])
    shell = x[3:].reshape(L, 3)
    gdiag = model.hc + thk * feat_k + thq * feat_q
    phase = np.exp(-1j * gamma * gdiag)
    us = [mixer_unitary(float(r[0]), float(r[1]), float(r[2])) for r in shell]
    psi = model.plus.copy()
    snaps = []
    for ell in range(L):
        psi = psi * phase
        snaps.append(psi.copy())
        psi = model.apply_cavities(psi, us[ell])
        snaps.append(psi.copy())
    energy = model.energy(psi)
    chi = model.apply_H(psi)
    g_shell = np.zeros_like(shell)
    g_gamma = 0.0
    g_k = 0.0
    g_q = 0.0
    phase_dag = np.conj(phase)
    for ell in range(L - 1, -1, -1):
        g_shell[ell] = 2.0 * np.imag(model.overlap_generators(chi, snaps[2 * ell + 1]))
        chi = model.apply_cavities(chi, us[ell].conj().T)
        psi_d = snaps[2 * ell]
        g_gamma += 2.0 * float(np.imag(np.vdot(chi, gdiag * psi_d)))
        g_k += 2.0 * float(np.imag(gamma * np.vdot(chi, feat_k * psi_d)))
        g_q += 2.0 * float(np.imag(gamma * np.vdot(chi, feat_q * psi_d)))
        chi = chi * phase_dag
    grad = np.concatenate([[g_gamma, g_k, g_q], g_shell.reshape(-1)])
    return float(energy), grad
