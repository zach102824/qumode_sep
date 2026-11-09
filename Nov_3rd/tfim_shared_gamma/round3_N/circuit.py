"""Shared-gamma family N. Energy is the objective. Fidelity is not in the gradient."""
from __future__ import annotations

import numpy as np

from Nov_3rd.tfim_shared_gamma.round1_A.circuit import fwht
from Nov_3rd.tfim_shared_gamma.round1_A.hamiltonian import apply_X_sum

from .features import Features

# Free theta names, in vector order after gamma (or after gammas, betas for qaoa).
SPECS = {
    "family_N": ("K", "M", "N"),
    "thetaN0": ("K", "M"),
    "thetaM0": ("K", "N"),
    "A": ("K",),
    "thetaK0": ("M", "N"),
    "full_intra": ("K", "M", "N"),
    "wrong_sel": ("K", "M", "N"),
    "shuf_jbar": ("K", "M", "N"),
    "same_G_qaoa": ("K", "M", "N"),
    "qaoa": (),
}


def param_count(method: str, L: int) -> int:
    if method == "qaoa":
        return 2 * L
    if method == "same_G_qaoa":
        return 2 * L + len(SPECS[method])
    return 1 + len(SPECS[method]) + 3 * L


def unpack(method: str, x: np.ndarray, L: int):
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if x.size != param_count(method, L):
        raise ValueError(f"{method} L={L} wants {param_count(method, L)}, got {x.size}")
    names = SPECS[method]
    cs = np.zeros((L, 3), dtype=np.float64)
    theta = {k: 0.0 for k in ("K", "M", "N")}
    if method == "qaoa":
        cs[:, 0] = x[L:]
        return x[:L].copy(), theta, cs
    if method == "same_G_qaoa":
        cs[:, 0] = x[L : 2 * L]
        for i, name in enumerate(names):
            theta[name] = float(x[2 * L + i])
        return x[:L].copy(), theta, cs
    gammas = np.full(L, float(x[0]))
    for i, name in enumerate(names):
        theta[name] = float(x[1 + i])
    cs = x[1 + len(names) :].reshape(L, 3).copy()
    return gammas, theta, cs


def pack(method: str, gamma, theta: dict, cs: np.ndarray, gammas=None, betas=None) -> np.ndarray:
    names = SPECS[method]
    if method == "qaoa":
        return np.concatenate([np.asarray(gammas, dtype=np.float64), np.asarray(betas, dtype=np.float64)])
    if method == "same_G_qaoa":
        th = [float(theta.get(n, 0.0)) for n in names]
        return np.concatenate([np.asarray(gammas, dtype=np.float64), np.asarray(betas, dtype=np.float64), th])
    flat = [float(gamma)] + [float(theta.get(n, 0.0)) for n in names]
    return np.concatenate([flat, np.asarray(cs, dtype=np.float64).reshape(-1)])


def _g_and_parts(feat: Features, method: str, theta: dict):
    spec = "family_N" if method in ("thetaN0", "thetaM0", "A", "thetaK0", "same_G_qaoa", "qaoa") else method
    if method == "qaoa":
        return feat.hc, {}
    g = feat.g_diag(spec, theta)
    parts = feat.partials(spec)
    # Only free thetas contribute. Fixed zeros are already in theta.
    return g, {name: parts[name] for name in SPECS[method]}


def energy_only(feat: Features, method: str, x: np.ndarray, L: int):
    gammas, theta, cs = unpack(method, x, L)
    g, _ = _g_and_parts(feat, method, theta)
    sys = feat.system
    psi = sys.plus.copy()
    for ell in range(L):
        psi = psi * np.exp(-1j * float(gammas[ell]) * g)
        phi = cs[ell, 0] * sys.w1 + cs[ell, 1] * sys.w2 + cs[ell, 2] * sys.w3
        psi = fwht(fwht(psi) * np.exp(-1j * phi))
    return _energy(sys, psi), psi


def energy_and_grad(feat: Features, method: str, x: np.ndarray, L: int):
    gammas, theta, cs = unpack(method, x, L)
    g, parts = _g_and_parts(feat, method, theta)
    sys = feat.system
    psi = sys.plus.copy()
    snaps = []
    for ell in range(L):
        phase = float(gammas[ell]) * g
        mid = psi * np.exp(-1j * phase)
        phi = cs[ell, 0] * sys.w1 + cs[ell, 1] * sys.w2 + cs[ell, 2] * sys.w3
        x2 = fwht(mid) * np.exp(-1j * phi)
        psi = fwht(x2)
        snaps.append((phase, mid, phi, x2))
    E = _energy(sys, psi)
    lam = sys.hc_full * psi + sys.B * apply_X_sum(psi, sys.n)
    g_gamma = np.zeros(L)
    g_th = {name: 0.0 for name in parts}
    g_c = np.zeros((L, 3))
    for ell in range(L - 1, -1, -1):
        phase, mid, phi, x2 = snaps[ell]
        lam_x2 = fwht(lam)
        ov = np.conj(lam_x2) * x2
        g_c[ell, 0] = 2.0 * float(np.imag(np.sum(ov * sys.w1)))
        g_c[ell, 1] = 2.0 * float(np.imag(np.sum(ov * sys.w2)))
        g_c[ell, 2] = 2.0 * float(np.imag(np.sum(ov * sys.w3)))
        lam_mid = fwht(lam_x2 * np.exp(1j * phi))
        ovD = np.conj(lam_mid) * mid
        g_gamma[ell] = 2.0 * float(np.imag(np.sum(ovD * g)))
        for name, part in parts.items():
            g_th[name] += 2.0 * float(np.imag(np.sum(ovD * (float(gammas[ell]) * part))))
        lam = lam_mid * np.exp(1j * phase)
    names = SPECS[method]
    if method == "qaoa":
        grad = np.concatenate([g_gamma, g_c[:, 0]])
    elif method == "same_G_qaoa":
        grad = np.concatenate([g_gamma, g_c[:, 0], [g_th[n] for n in names]])
    else:
        grad = np.concatenate([[g_gamma.sum()], [g_th[n] for n in names], g_c.reshape(-1)])
    return E, grad, psi


def _energy(system, psi) -> float:
    Hpsi = system.hc_full * psi + system.B * apply_X_sum(psi, system.n)
    return float(np.vdot(psi, Hpsi).real)


def variance_H(system, psi) -> float:
    Hpsi = system.hc_full * psi + system.B * apply_X_sum(psi, system.n)
    E = float(np.vdot(psi, Hpsi).real)
    return float(np.vdot(Hpsi, Hpsi).real - E * E)


def fidelity(system, psi) -> float:
    return float(abs(np.vdot(system.gs, psi)) ** 2)


def eta(E, E0, Emax) -> float:
    den = E0 - Emax
    return float("nan") if abs(den) < 1e-15 else float((E - Emax) / den)


def fock_occupations(psi, n: int) -> dict:
    """Per-cavity probability of each Fock state, in the logical basis."""
    dim = 1 << n
    probs = np.abs(np.asarray(psi)) ** 2
    m = n // 3
    out = []
    for a in range(m):
        p = np.zeros(8)
        shift = 3 * a
        idx = np.arange(dim)
        local = (idx >> shift) & 7
        for j in range(8):
            p[j] = float(np.sum(probs[local == j]))
        out.append({"p_fock": p.tolist(), "p_neel_2_or_5": float(p[2] + p[5])})
    return {"cavities": out, "mean_p_neel": float(np.mean([c["p_neel_2_or_5"] for c in out]))}


def layer_rows(feat, method, x, L):
    gammas, theta, cs = unpack(method, x, L)
    g, _ = _g_and_parts(feat, method, theta)
    sys = feat.system
    psi = sys.plus.copy()
    rows = []
    for ell in range(L):
        psi = psi * np.exp(-1j * float(gammas[ell]) * g)
        phi = cs[ell, 0] * sys.w1 + cs[ell, 1] * sys.w2 + cs[ell, 2] * sys.w3
        psi = fwht(fwht(psi) * np.exp(-1j * phi))
        E = _energy(sys, psi)
        rows.append({"layer": ell + 1, "E": E, "energy_error": E - sys.E0, "F": fidelity(sys, psi)})
    return rows
