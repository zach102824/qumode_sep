"""Logical Nov_3rd circuit and its physical-frame twin.

Logical frame, verified against the physical sequence:
    |+>^N -> [exp(-i gamma G(Z)) -> M_l ]^L

M_l = exp(-i sum_cavities (c1 sum X + c2 sum XX + c3 XXX))
with the same (c1, c2, c3) on every cavity. Equivalently the shell phases
    phi_k = c1 (3-2k) + c2 (((3-2k)^2 - 3)/2) + c3 (-1)^k
on Fock states of Hamming weight k, conjugated by W = H^{tensor N}.

Physical order (rightmost acts first on vacuum):
    W S_L U_c^X ... S_1 U_c^X |0>,   U_c^X = W exp(-i gamma G) W.

gamma and the generator coefficients are shared across layers. Shell
controls are per layer. QAOA is the independent-angle comparator
    product_l exp(-i beta_l sum X) exp(-i gamma_l Hc) |+>
and does not put B into the mixer.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from hamiltonian import apply_H_psi, apply_sum_X, energy_from_state, plus_state

_I2 = np.eye(2, dtype=np.float64)
_X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64)
_H1 = np.array([[1.0, 1.0], [1.0, -1.0]], dtype=np.float64) / np.sqrt(2.0)


def _kron3(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    return np.kron(np.kron(a, b), c)


G1 = _kron3(_X, _I2, _I2) + _kron3(_I2, _X, _I2) + _kron3(_I2, _I2, _X)
G2 = _kron3(_X, _X, _I2) + _kron3(_X, _I2, _X) + _kron3(_I2, _X, _X)
G3 = _kron3(_X, _X, _X)
GENS = (G1, G2, G3)

METHODS = ("qaoa", "shared_linear", "D", "A", "C", "C_unrestricted")


def n_params(method: str, L: int) -> int:
    if method == "qaoa":
        return 2 * L
    if method == "shared_linear":
        return 1 + L
    if method == "D":
        return 1 + 3 * L
    if method == "A":
        return 2 + 3 * L
    if method in ("C", "C_unrestricted"):
        return 3 + 3 * L
    raise KeyError(method)


def bounds_for(method: str, L: int) -> List[Tuple[float, float]]:
    g = (-2.0 * np.pi, 2.0 * np.pi)
    c = (-2.0 * np.pi, 2.0 * np.pi)
    th = (-5.0, 5.0)
    if method == "qaoa":
        return [g, c] * L
    if method == "shared_linear":
        return [g] + [c] * L
    if method == "D":
        return [g] + [c] * (3 * L)
    if method == "A":
        return [g, th] + [c] * (3 * L)
    if method in ("C", "C_unrestricted"):
        return [g, th, th] + [c] * (3 * L)
    raise KeyError(method)


def unpack(method: str, x: np.ndarray, L: int) -> Dict:
    x = np.asarray(x, dtype=np.float64)
    if x.shape != (n_params(method, L),):
        raise ValueError(f"{method} L={L} expected {n_params(method, L)} params, got {x.shape}")
    if method == "qaoa":
        return {"gammas": x[0::2].copy(), "betas": x[1::2].copy()}
    if method == "shared_linear":
        return {
            "gamma": float(x[0]),
            "theta_k": 0.0,
            "theta_c": 0.0,
            "c1": x[1:].copy(),
            "c2": np.zeros(L),
            "c3": np.zeros(L),
        }
    if method == "D":
        shells = x[1:].reshape(L, 3)
        return {
            "gamma": float(x[0]),
            "theta_k": 0.0,
            "theta_c": 0.0,
            "c1": shells[:, 0].copy(),
            "c2": shells[:, 1].copy(),
            "c3": shells[:, 2].copy(),
        }
    if method == "A":
        shells = x[2:].reshape(L, 3)
        return {
            "gamma": float(x[0]),
            "theta_k": float(x[1]),
            "theta_c": 0.0,
            "c1": shells[:, 0].copy(),
            "c2": shells[:, 1].copy(),
            "c3": shells[:, 2].copy(),
        }
    shells = x[3:].reshape(L, 3)
    return {
        "gamma": float(x[0]),
        "theta_k": float(x[1]),
        "theta_c": float(x[2]),
        "c1": shells[:, 0].copy(),
        "c2": shells[:, 1].copy(),
        "c3": shells[:, 2].copy(),
    }


def pack_angles(parsed: Dict) -> Dict:
    """JSON-ready angle record. Scalars stay scalars."""
    out = {}
    for k, v in parsed.items():
        if np.isscalar(v):
            out[k] = float(v)
        else:
            out[k] = [float(t) for t in np.asarray(v).tolist()]
    return out


def shell_phases(c1: float, c2: float, c3: float) -> np.ndarray:
    k = np.arange(4, dtype=np.float64)
    return c1 * (3.0 - 2.0 * k) + c2 * (((3.0 - 2.0 * k) ** 2 - 3.0) / 2.0) + c3 * ((-1.0) ** k)


def shell_unitary(c1: float, c2: float, c3: float, with_grad: bool = False):
    """Single-cavity mixer and, optionally, dU/dc_k for k=1,2,3."""
    H = c1 * G1 + c2 * G2 + c3 * G3
    w, V = np.linalg.eigh(H)
    phase = np.exp(-1j * w)
    U = (V * phase) @ V.T
    if not with_grad:
        return U
    dUs = []
    wp = w[:, None]
    wq = w[None, :]
    half = 0.5 * (wp - wq)
    mid = 0.5 * (wp + wq)
    sinc = np.ones_like(half, dtype=np.float64)
    nz = np.abs(half) > 1e-8
    sinc[nz] = np.sin(half[nz]) / half[nz]
    phi = -1j * np.exp(-1j * mid) * sinc
    for G in GENS:
        dtil = V.T @ G @ V
        dU = (V @ (phi * dtil)) @ V.T
        dUs.append(dU)
    return U, dUs


def apply_all_cavities(psi: np.ndarray, U: np.ndarray, n: int) -> np.ndarray:
    m = n // 3
    t = np.asarray(psi, dtype=np.complex128).reshape([8] * m)
    for a in range(m):
        t = np.moveaxis(t, a, 0)
        t = np.tensordot(U, t, axes=([1], [0]))
        t = np.moveaxis(t, 0, a)
    return np.ascontiguousarray(t).reshape(-1)


def apply_one_cavity(psi: np.ndarray, U: np.ndarray, n: int, cav: int) -> np.ndarray:
    m = n // 3
    t = np.asarray(psi, dtype=np.complex128).reshape([8] * m)
    t = np.moveaxis(t, cav, 0)
    t = np.tensordot(U, t, axes=([1], [0]))
    t = np.moveaxis(t, 0, cav)
    return np.ascontiguousarray(t).reshape(-1)


def apply_rx_all(psi: np.ndarray, beta: float, n: int) -> np.ndarray:
    """exp(-i beta sum_i X_i) = [exp(-i beta X)]^{tensor n}."""
    c = np.cos(beta)
    s = np.sin(beta)
    out = np.asarray(psi, dtype=np.complex128)
    for q in range(n):
        half = 1 << (n - 1 - q)
        x = out.reshape(-1, 2, half)
        a = x[:, 0, :]
        b = x[:, 1, :]
        y = np.empty_like(x)
        y[:, 0, :] = c * a - 1j * s * b
        y[:, 1, :] = -1j * s * a + c * b
        out = y.reshape(-1)
    return out


def apply_hadamard_all(psi: np.ndarray, n: int) -> np.ndarray:
    h = 1.0 / np.sqrt(2.0)
    out = np.asarray(psi, dtype=np.complex128)
    for q in range(n):
        half = 1 << (n - 1 - q)
        x = out.reshape(-1, 2, half)
        a = x[:, 0, :]
        b = x[:, 1, :]
        y = np.empty_like(x)
        y[:, 0, :] = h * (a + b)
        y[:, 1, :] = h * (a - b)
        out = y.reshape(-1)
    return out


def shell_diag_unitary(n: int, c1: float, c2: float, c3: float) -> np.ndarray:
    """Bare shell SNAP, diagonal in the Fock / computational basis."""
    dim = 1 << n
    x = np.arange(dim, dtype=np.int64)
    phi = np.zeros(dim, dtype=np.float64)
    table = shell_phases(c1, c2, c3)
    for a in range(n // 3):
        pop = np.zeros(dim, dtype=np.int64)
        for k in range(3):
            q = 3 * a + k
            pop += (x >> (n - 1 - q)) & 1
        phi += table[pop]
    return np.exp(-1j * phi)


def physical_readout(
    psi_fock: np.ndarray,
    n: int,
    gamma: float,
    gdiag: np.ndarray,
    shells: Sequence[Tuple[float, float, float]],
) -> np.ndarray:
    """Physical Nov_3rd sequence ending with W, so the output is the logical readout state."""
    psi = np.asarray(psi_fock, dtype=np.complex128).copy()
    phase_g = None
    for c1, c2, c3 in shells:
        psi = apply_hadamard_all(psi, n)
        if phase_g is None or gamma != 0.0:
            phase_g = np.exp(-1j * float(gamma) * gdiag)
        psi = phase_g * psi
        psi = apply_hadamard_all(psi, n)
        psi = shell_diag_unitary(n, c1, c2, c3) * psi
    psi = apply_hadamard_all(psi, n)
    return psi


def _shell_parameter_index(method: str, ell: int, k: int) -> int:
    if method == "D":
        base = 1
    elif method == "A":
        base = 2
    elif method in ("C", "C_unrestricted"):
        base = 3
    else:
        raise KeyError(method)
    return base + 3 * ell + k



def _simulate_impl(
    method: str,
    x: np.ndarray,
    L: int,
    hc: np.ndarray,
    fk: np.ndarray,
    fs: np.ndarray,
    B: float,
    n: int,
    want_grad: bool = False,
):
    p = unpack(method, np.asarray(x, dtype=np.float64), L)
    psi = plus_state(n)
    frames = []
    G = None
    gamma = None
    if method == "qaoa":
        for ell in range(L):
            gell = float(p["gammas"][ell])
            bell = float(p["betas"][ell])
            psi = np.exp(-1j * gell * hc) * psi
            frames.append(("qdiag", psi, gell, ell))
            psi = apply_rx_all(psi, bell, n)
            frames.append(("rx", psi, bell, ell))
    else:
        gamma = float(p["gamma"])
        G = hc + float(p["theta_k"]) * fk + float(p["theta_c"]) * fs
        phase = np.exp(-1j * gamma * G)
        for ell in range(L):
            psi = phase * psi
            frames.append(("diagG", psi))
            if method == "shared_linear":
                beta = float(p["c1"][ell])
                psi = apply_rx_all(psi, beta, n)
                frames.append(("rx", psi, beta, ell))
            else:
                c1 = float(p["c1"][ell])
                c2 = float(p["c2"][ell])
                c3 = float(p["c3"][ell])
                if want_grad:
                    U, dUs = shell_unitary(c1, c2, c3, with_grad=True)
                else:
                    U = shell_unitary(c1, c2, c3, with_grad=False)
                    dUs = None
                psi = apply_all_cavities(psi, U, n)
                frames.append(("shell", psi, U, dUs, ell))
    energy = energy_from_state(psi, hc, B, n)
    if not want_grad:
        return psi, float(energy), None

    grad = np.zeros(n_params(method, L), dtype=np.float64)
    lam = apply_H_psi(psi, hc, B, n)
    for fr in reversed(frames):
        kind = fr[0]
        if kind == "qdiag":
            _, psi_new, gell, ell = fr
            im = np.imag(np.conj(lam) * psi_new)
            grad[2 * ell] += 2.0 * float(np.sum(hc * im))
            lam = np.exp(1j * gell * hc) * lam
        elif kind == "diagG":
            _, psi_new = fr
            im = np.imag(np.conj(lam) * psi_new)
            grad[0] += 2.0 * float(np.sum(G * im))
            if method in ("A", "C", "C_unrestricted"):
                grad[1] += 2.0 * float(np.sum((gamma * fk) * im))
            if method in ("C", "C_unrestricted"):
                grad[2] += 2.0 * float(np.sum((gamma * fs) * im))
            lam = np.exp(1j * gamma * G) * lam
        elif kind == "rx":
            _, psi_new, beta, ell = fr
            sx = apply_sum_X(psi_new, n)
            gbeta = 2.0 * float(np.real(np.vdot(lam, -1j * sx)))
            if method == "qaoa":
                grad[2 * ell + 1] += gbeta
            else:
                grad[1 + ell] += gbeta
            lam = apply_rx_all(lam, -beta, n)
        elif kind == "shell":
            _, psi_new, U, dUs, ell = fr
            for k in range(3):
                Va = dUs[k] @ np.conjugate(U).T
                acc = None
                for cav in range(n // 3):
                    piece = apply_one_cavity(psi_new, Va, n, cav)
                    acc = piece if acc is None else acc + piece
                grad[_shell_parameter_index(method, ell, k)] += 2.0 * float(np.real(np.vdot(lam, acc)))
            lam = apply_all_cavities(lam, np.conjugate(U).T, n)
        else:
            raise RuntimeError(kind)
    return psi, float(energy), grad


simulate = _simulate_impl


def logical_state(method: str, x: np.ndarray, L: int, hc, fk, fs, B: float, n: int) -> np.ndarray:
    psi, _, _ = simulate(method, x, L, hc, fk, fs, B, n, want_grad=False)
    return psi


def eta_score(E: float, E0: float, Emax: float) -> float:
    den = E0 - Emax
    if abs(den) < 1e-15:
        return float("nan")
    return float((E - Emax) / den)


def fidelity(psi: np.ndarray, gs: np.ndarray) -> float:
    return float(abs(np.vdot(gs, psi)) ** 2)


def numerical_energy_grad(method, x, L, hc, fk, fs, B, n, eps: float = 1e-6) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    g = np.zeros_like(x)
    for i in range(x.size):
        e = np.zeros_like(x)
        e[i] = eps
        _, fp, _ = simulate(method, x + e, L, hc, fk, fs, B, n, False)
        _, fm, _ = simulate(method, x - e, L, hc, fk, fs, B, n, False)
        g[i] = (fp - fm) / (2.0 * eps)
    return g


def zero_shell_is_not_identity(hc, B, n, gamma: float = 0.3) -> Dict:
    """A zero shell does not cancel a shared-gamma cost block.

    Two zero-shell layers equal one layer at 2*gamma, and differ from one layer
    at gamma, for a generic spectrum.
    """
    fk = np.zeros_like(hc)
    fs = np.zeros_like(hc)
    x2 = np.array([gamma, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)  # D, L=2
    # D L=2 is 1+6=7 params. Yes.
    x1 = np.array([gamma, 0.0, 0.0, 0.0], dtype=np.float64)
    x1b = np.array([2.0 * gamma, 0.0, 0.0, 0.0], dtype=np.float64)
    _, e2, _ = simulate("D", x2, 2, hc, fk, fs, B, n, False)
    _, e1, _ = simulate("D", x1, 1, hc, fk, fs, B, n, False)
    _, e1b, _ = simulate("D", x1b, 1, hc, fk, fs, B, n, False)
    return {
        "E_two_zero_shells": e2,
        "E_one_layer_same_gamma": e1,
        "E_one_layer_double_gamma": e1b,
        "matches_double_gamma": bool(abs(e2 - e1b) < 1e-8),
        "differs_from_single_gamma": bool(abs(e2 - e1) > 1e-6),
    }
