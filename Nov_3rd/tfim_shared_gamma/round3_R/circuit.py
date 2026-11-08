"""Packing-aware shared-gamma shell. The state vector stays in the logical basis."""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

import model


def plus_state(n: int) -> np.ndarray:
    return np.full(1 << n, 1.0 / np.sqrt(1 << n), dtype=np.complex128)


def apply_X(psi, qubit: int):
    half = 1 << qubit
    x = np.asarray(psi).reshape(-1, 2, half)
    out = np.empty_like(x)
    out[:, 0, :] = x[:, 1, :]
    out[:, 1, :] = x[:, 0, :]
    return out.reshape(-1)


def apply_sum_X(psi, n: int):
    acc = np.zeros_like(psi)
    for q in range(n):
        acc += apply_X(psi, q)
    return acc


def apply_rx_all(psi, beta: float, n: int):
    c = np.cos(beta)
    s = np.sin(beta)
    out = np.asarray(psi, dtype=np.complex128)
    for q in range(n):
        half = 1 << q
        x = out.reshape(-1, 2, half)
        a = x[:, 0, :]
        b = x[:, 1, :]
        y = np.empty_like(x)
        y[:, 0, :] = c * a - 1j * s * b
        y[:, 1, :] = -1j * s * a + c * b
        out = y.reshape(-1)
    return out


def apply_H(psi, hc, B, n):
    out = hc * psi
    if B != 0.0:
        out = out + float(B) * apply_sum_X(psi, n)
    return out


def energy_var(psi, hc, B, n):
    Hpsi = apply_H(psi, hc, B, n)
    E = float(np.vdot(psi, Hpsi).real)
    second = float(np.vdot(Hpsi, Hpsi).real)
    return E, second - E * E


def n_params(kind: str, L: int) -> int:
    if kind == "qaoa":
        return 2 * L
    if kind == "k0":
        return 1 + 3 * L
    if kind == "A":
        return 2 + 3 * L
    if kind == "M":
        return 3 + 3 * L
    if kind == "sameG":
        return 1 + 2 * L
    raise KeyError(kind)


def matched_depth(kind: str, L: int) -> int:
    return int(np.ceil(n_params(kind, L) / 2.0))


def bounds_for(kind: str, L: int):
    g = (-2.0 * np.pi, 2.0 * np.pi)
    c = (-2.0 * np.pi, 2.0 * np.pi)
    th = (-5.0, 5.0)
    if kind == "qaoa":
        return [g, c] * L
    if kind == "k0":
        return [g] + [c] * (3 * L)
    if kind == "A":
        return [g, th] + [c] * (3 * L)
    if kind == "M":
        return [g, th, th] + [c] * (3 * L)
    if kind == "sameG":
        return [th] + [g, c] * L
    raise KeyError(kind)


def unpack(kind: str, x, L: int):
    x = np.asarray(x, dtype=np.float64)
    if kind == "qaoa":
        return {"gammas": x[0::2].copy(), "betas": x[1::2].copy()}
    if kind == "k0":
        shells = x[1:].reshape(L, 3)
        return {"gamma": float(x[0]), "theta_k": 0.0, "theta_m": 0.0,
                "c1": shells[:, 0].copy(), "c2": shells[:, 1].copy(), "c3": shells[:, 2].copy()}
    if kind == "A":
        shells = x[2:].reshape(L, 3)
        return {"gamma": float(x[0]), "theta_k": float(x[1]), "theta_m": 0.0,
                "c1": shells[:, 0].copy(), "c2": shells[:, 1].copy(), "c3": shells[:, 2].copy()}
    if kind == "M":
        shells = x[3:].reshape(L, 3)
        return {"gamma": float(x[0]), "theta_k": float(x[1]), "theta_m": float(x[2]),
                "c1": shells[:, 0].copy(), "c2": shells[:, 1].copy(), "c3": shells[:, 2].copy()}
    shells = x[1:].reshape(L, 2)
    return {"theta_k": float(x[0]), "gammas": shells[:, 0].copy(), "betas": shells[:, 1].copy()}


def pack_angles(parsed):
    out = {}
    for k, v in parsed.items():
        if np.isscalar(v):
            out[k] = float(v)
        else:
            out[k] = [float(t) for t in np.asarray(v)]
    return out


def simulate(kind, x, L, hc, fk, fm, B, n, indices, want_grad=False):
    p = unpack(kind, x, L)
    psi = plus_state(n)
    frames = []
    if kind == "qaoa":
        for ell in range(L):
            gell = float(p["gammas"][ell])
            bell = float(p["betas"][ell])
            psi = np.exp(-1j * gell * hc) * psi
            frames.append(("qdiag", psi, gell, ell))
            psi = apply_rx_all(psi, bell, n)
            frames.append(("rx", psi, bell, ell))
    elif kind == "sameG":
        G = hc + float(p["theta_k"]) * fk
        for ell in range(L):
            gell = float(p["gammas"][ell])
            bell = float(p["betas"][ell])
            psi = np.exp(-1j * gell * G) * psi
            frames.append(("sdiag", psi, gell, ell))
            psi = apply_rx_all(psi, bell, n)
            frames.append(("srx", psi, bell, ell))
    else:
        gamma = float(p["gamma"])
        G = hc + float(p["theta_k"]) * fk + float(p["theta_m"]) * fm
        phase = np.exp(-1j * gamma * G)
        for ell in range(L):
            psi = phase * psi
            frames.append(("diag", psi))
            if want_grad:
                U, dUs = model.shell_unitary(float(p["c1"][ell]), float(p["c2"][ell]), float(p["c3"][ell]), True)
            else:
                U = model.shell_unitary(float(p["c1"][ell]), float(p["c2"][ell]), float(p["c3"][ell]), False)
                dUs = None
            for idx in indices:
                psi = model.apply_indexed(psi, U, idx)
            frames.append(("shell", psi, U, dUs, ell))
    E, var = energy_var(psi, hc, B, n)
    if not want_grad:
        return psi, float(E), None, float(var)
    grad = np.zeros(n_params(kind, L), dtype=np.float64)
    lam = apply_H(psi, hc, B, n)
    if kind not in ("qaoa", "sameG"):
        G = hc + float(p["theta_k"]) * fk + float(p["theta_m"]) * fm
        gamma = float(p["gamma"])
    elif kind == "sameG":
        G = hc + float(p["theta_k"]) * fk
    for fr in reversed(frames):
        kind_f = fr[0]
        if kind_f == "qdiag":
            _, psi_new, gell, ell = fr
            im = np.imag(np.conj(lam) * psi_new)
            grad[2 * ell] += 2.0 * float(np.sum(hc * im))
            lam = np.exp(1j * gell * hc) * lam
        elif kind_f == "sdiag":
            _, psi_new, gell, ell = fr
            im = np.imag(np.conj(lam) * psi_new)
            grad[1 + 2 * ell] += 2.0 * float(np.sum(G * im))
            grad[0] += 2.0 * float(np.sum((gell * fk) * im))
            lam = np.exp(1j * gell * G) * lam
        elif kind_f == "diag":
            _, psi_new = fr
            im = np.imag(np.conj(lam) * psi_new)
            grad[0] += 2.0 * float(np.sum(G * im))
            if kind in ("A", "M"):
                grad[1] += 2.0 * float(np.sum((gamma * fk) * im))
            if kind == "M":
                grad[2] += 2.0 * float(np.sum((gamma * fm) * im))
            lam = np.exp(1j * gamma * G) * lam
        elif kind_f == "rx":
            _, psi_new, beta, ell = fr
            gbeta = 2.0 * float(np.real(np.vdot(lam, -1j * apply_sum_X(psi_new, n))))
            grad[2 * ell + 1] += gbeta
            lam = apply_rx_all(lam, -beta, n)
        elif kind_f == "srx":
            _, psi_new, beta, ell = fr
            gbeta = 2.0 * float(np.real(np.vdot(lam, -1j * apply_sum_X(psi_new, n))))
            grad[2 + 2 * ell] += gbeta
            lam = apply_rx_all(lam, -beta, n)
        elif kind_f == "shell":
            _, psi_new, U, dUs, ell = fr
            base = 1 if kind == "k0" else (2 if kind == "A" else 3)
            for k in range(3):
                Va = dUs[k] @ np.conjugate(U).T
                acc = np.zeros_like(psi_new)
                for idx in indices:
                    acc += model.apply_indexed(psi_new, Va, idx)
                grad[base + 3 * ell + k] += 2.0 * float(np.real(np.vdot(lam, acc)))
            for idx in indices:
                lam = model.apply_indexed(lam, np.conjugate(U).T, idx)
        else:
            raise RuntimeError(kind_f)
    return psi, float(E), grad, float(var)


def numerical_grad(kind, x, L, hc, fk, fm, B, n, indices, eps=1e-6):
    x = np.asarray(x, dtype=np.float64)
    g = np.zeros_like(x)
    for i in range(x.size):
        e = np.zeros_like(x)
        e[i] = eps
        _, fp, _, _ = simulate(kind, x + e, L, hc, fk, fm, B, n, indices, False)
        _, fm_, _, _ = simulate(kind, x - e, L, hc, fk, fm, B, n, indices, False)
        g[i] = (fp - fm_) / (2 * eps)
    return g


def fidelity(psi, gs) -> float:
    return float(abs(np.vdot(gs, psi)) ** 2)


def parity_expectation(psi) -> float:
    return float(np.vdot(psi, psi[::-1]).real)


def reflection_indices(n: int):
    dim = 1 << n
    x = np.arange(dim, dtype=np.uint32)
    y = np.zeros(dim, dtype=np.uint32)
    for src in range(n):
        bit = (x >> np.uint32(src)) & np.uint32(1)
        y |= bit << np.uint32(n - 1 - src)
    return y.astype(np.int64)


def reflection_expectation(psi, ridx) -> float:
    return float(np.vdot(psi, psi[ridx]).real)


def dense_spectrum(hc, B, n):
    dim = hc.shape[0]
    H = np.diag(np.asarray(hc, dtype=np.float64))
    x = np.arange(dim)
    for q in range(n):
        H[x, x ^ (1 << q)] = float(B)
    evals, evecs = np.linalg.eigh(H)
    gs = evecs[:, 0]
    k = int(np.argmax(np.abs(gs)))
    gs = gs * np.exp(-1j * np.angle(gs[k]))
    if float(np.sum(gs.real)) < 0:
        gs = -gs
    ridx = reflection_indices(n)
    return {
        "E0": float(evals[0]),
        "Emax": float(evals[-1]),
        "full_gap": float(evals[1] - evals[0]),
        "gs": gs,
        "evals": evals,
        "ridx": ridx,
        "gs_parity": parity_expectation(gs),
        "gs_reflection": reflection_expectation(gs, ridx),
        "gs_residual": float(np.linalg.norm(H @ gs - evals[0] * gs)),
    }
