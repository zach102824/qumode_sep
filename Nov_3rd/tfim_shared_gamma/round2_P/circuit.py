"""Little-endian Nov_3rd logical circuit for family P.

|+> -> [exp(-i gamma G) -> M_l]^L
M_l uses the same (c1, c2, c3) on every cavity.
Local Fock j = bit0 + 2*bit1 + 4*bit2 inside the cavity, bit0 = lowest qubit index.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np

_I2 = np.eye(2, dtype=np.float64)
_X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64)
_H1 = np.array([[1.0, 1.0], [1.0, -1.0]], dtype=np.float64) / np.sqrt(2.0)


def _kron3(slow: np.ndarray, mid: np.ndarray, fast: np.ndarray) -> np.ndarray:
    """fast is the least-significant cavity bit (lowest qubit index)."""
    return np.kron(slow, np.kron(mid, fast))


# Local qubits (q_low, q_mid, q_high) = (fast, mid, slow).
_Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.float64)
G1 = _kron3(_I2, _I2, _X) + _kron3(_I2, _X, _I2) + _kron3(_X, _I2, _I2)
G2 = _kron3(_I2, _X, _X) + _kron3(_X, _I2, _X) + _kron3(_X, _X, _I2)
G3 = _kron3(_X, _X, _X)
ZZZ = _kron3(_Z, _Z, _Z)
GENS = (G1, G2, G3)

METHODS = ("qaoa", "A", "P", "Sblock", "Pshuffle", "P_thetaK0", "qaoa_sameG", "P_indgamma")


def n_params(method: str, L: int) -> int:
    if method == "qaoa":
        return 2 * L
    if method == "A":
        return 2 + 3 * L
    if method == "P_thetaK0":
        return 2 + 3 * L
    if method in ("P", "Sblock", "Pshuffle"):
        return 3 + 3 * L
    if method == "qaoa_sameG":
        return 2 + 2 * L
    if method == "P_indgamma":
        return 2 + 4 * L
    raise KeyError(method)


def matched_qaoa_depth(L: int) -> int:
    """ceil(P/2) for the primary family, P = 3L+3."""
    return int(np.ceil((3 * L + 3) / 2.0))


def bounds_for(method: str, L: int) -> List[Tuple[float, float]]:
    g = (-2.0 * np.pi, 2.0 * np.pi)
    c = (-2.0 * np.pi, 2.0 * np.pi)
    th = (-5.0, 5.0)
    if method == "qaoa":
        return [g, c] * L
    if method == "A":
        return [g, th] + [c] * (3 * L)
    if method == "P_thetaK0":
        return [g, th] + [c] * (3 * L)
    if method in ("P", "Sblock", "Pshuffle"):
        return [g, th, th] + [c] * (3 * L)
    if method == "qaoa_sameG":
        return [th, th] + [g, c] * L
    if method == "P_indgamma":
        # thetaK, thetaP, then per layer (gamma, c1, c2, c3)
        return [th, th] + [g, c, c, c] * L
    raise KeyError(method)


def unpack(method: str, x: np.ndarray, L: int) -> Dict:
    x = np.asarray(x, dtype=np.float64)
    if x.shape != (n_params(method, L),):
        raise ValueError(f"{method} L={L} expected {n_params(method, L)} got {x.size}")
    if method == "qaoa":
        return {"gammas": x[0::2].copy(), "betas": x[1::2].copy(), "theta_k": 0.0, "theta_p": 0.0}
    if method == "A":
        shells = x[2:].reshape(L, 3)
        return {"gamma": float(x[0]), "theta_k": float(x[1]), "theta_p": 0.0,
                "c1": shells[:, 0].copy(), "c2": shells[:, 1].copy(), "c3": shells[:, 2].copy()}
    if method == "P_thetaK0":
        shells = x[2:].reshape(L, 3)
        return {"gamma": float(x[0]), "theta_k": 0.0, "theta_p": float(x[1]),
                "c1": shells[:, 0].copy(), "c2": shells[:, 1].copy(), "c3": shells[:, 2].copy()}
    if method in ("P", "Sblock", "Pshuffle"):
        shells = x[3:].reshape(L, 3)
        return {"gamma": float(x[0]), "theta_k": float(x[1]), "theta_p": float(x[2]),
                "c1": shells[:, 0].copy(), "c2": shells[:, 1].copy(), "c3": shells[:, 2].copy()}
    if method == "qaoa_sameG":
        return {"theta_k": float(x[0]), "theta_p": float(x[1]),
                "gammas": x[2::2].copy(), "betas": x[3::2].copy()}
    # P_indgamma: thetaK, thetaP, (gamma, c1, c2, c3) * L
    block = x[2:].reshape(L, 4)
    return {"theta_k": float(x[0]), "theta_p": float(x[1]),
            "gammas": block[:, 0].copy(), "c1": block[:, 1].copy(),
            "c2": block[:, 2].copy(), "c3": block[:, 3].copy()}


def pack_angles(parsed: Dict) -> Dict:
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
    sinc = np.ones_like(half)
    nz = np.abs(half) > 1e-8
    sinc[nz] = np.sin(half[nz]) / half[nz]
    phi = -1j * np.exp(-1j * mid) * sinc
    for G in GENS:
        dtil = V.T @ G @ V
        dUs.append((V @ (phi * dtil)) @ V.T)
    return U, dUs


def plus_state(n: int) -> np.ndarray:
    return np.full(1 << n, 1.0 / np.sqrt(1 << n), dtype=np.complex128)


def apply_X(psi: np.ndarray, qubit: int, n: int) -> np.ndarray:
    half = 1 << qubit
    x = np.asarray(psi).reshape(-1, 2, half)
    out = np.empty_like(x)
    out[:, 0, :] = x[:, 1, :]
    out[:, 1, :] = x[:, 0, :]
    return out.reshape(-1)


def apply_sum_X(psi: np.ndarray, n: int) -> np.ndarray:
    acc = np.zeros_like(psi)
    for q in range(n):
        acc += apply_X(psi, q, n)
    return acc


def energy_from_state(psi: np.ndarray, hc: np.ndarray, B: float, n: int) -> float:
    e = float(np.vdot(psi, hc * psi).real)
    for q in range(n):
        e += float(B) * float(np.vdot(psi, apply_X(psi, q, n)).real)
    return e


def apply_H(psi: np.ndarray, hc: np.ndarray, B: float, n: int) -> np.ndarray:
    out = hc * psi
    if B != 0.0:
        out = out + float(B) * apply_sum_X(psi, n)
    return out


def _cavity_axis(cavity: int, m: int) -> int:
    # C-order reshape([8]*m): last axis is cavity 0 (little-endian).
    return m - 1 - cavity


def apply_one_cavity(psi: np.ndarray, U: np.ndarray, n: int, cavity: int) -> np.ndarray:
    m = n // 3
    t = np.asarray(psi, dtype=np.complex128).reshape([8] * m)
    ax = _cavity_axis(cavity, m)
    t = np.moveaxis(t, ax, 0)
    t = np.tensordot(U, t, axes=([1], [0]))
    t = np.moveaxis(t, 0, ax)
    return np.ascontiguousarray(t).reshape(-1)


def apply_all_cavities(psi: np.ndarray, U: np.ndarray, n: int) -> np.ndarray:
    out = psi
    for a in range(n // 3):
        out = apply_one_cavity(out, U, n, a)
    return out


def apply_rx_all(psi: np.ndarray, beta: float, n: int) -> np.ndarray:
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


def apply_hadamard_all(psi: np.ndarray, n: int) -> np.ndarray:
    h = 1.0 / np.sqrt(2.0)
    out = np.asarray(psi, dtype=np.complex128)
    for q in range(n):
        half = 1 << q
        x = out.reshape(-1, 2, half)
        a = x[:, 0, :]
        b = x[:, 1, :]
        y = np.empty_like(x)
        y[:, 0, :] = h * (a + b)
        y[:, 1, :] = h * (a - b)
        out = y.reshape(-1)
    return out


def shell_diag(n: int, c1: float, c2: float, c3: float) -> np.ndarray:
    dim = 1 << n
    x = np.arange(dim, dtype=np.int64)
    table = shell_phases(c1, c2, c3)
    phi = np.zeros(dim, dtype=np.float64)
    for a in range(n // 3):
        pop = np.zeros(dim, dtype=np.int64)
        for r in range(3):
            pop += (x >> (3 * a + r)) & 1
        phi += table[pop]
    return np.exp(-1j * phi)


def physical_readout(psi_fock, n, gamma, gdiag, shells: Sequence[Tuple[float, float, float]]):
    psi = np.asarray(psi_fock, dtype=np.complex128).copy()
    phase = np.exp(-1j * float(gamma) * gdiag)
    for c1, c2, c3 in shells:
        psi = apply_hadamard_all(psi, n)
        psi = phase * psi
        psi = apply_hadamard_all(psi, n)
        psi = shell_diag(n, c1, c2, c3) * psi
    return apply_hadamard_all(psi, n)


def _feature_diag(method: str, fk: np.ndarray, fp: np.ndarray, fp_shuf: np.ndarray, fs: np.ndarray):
    if method == "Sblock":
        return fs
    if method == "Pshuffle":
        return fp_shuf
    return fp


def simulate(method, x, L, hc, fk, fp, fp_shuf, fs, B, n, want_grad=False):
    p = unpack(method, x, L)
    feat = _feature_diag(method, fk, fp, fp_shuf, fs)
    psi = plus_state(n)
    frames = []
    if method == "qaoa":
        for ell in range(L):
            gell = float(p["gammas"][ell])
            bell = float(p["betas"][ell])
            psi = np.exp(-1j * gell * hc) * psi
            frames.append(("qdiag", psi, gell, ell))
            psi = apply_rx_all(psi, bell, n)
            frames.append(("rx", psi, bell, ell, False))
    elif method == "qaoa_sameG":
        G = hc + float(p["theta_k"]) * fk + float(p["theta_p"]) * fp
        for ell in range(L):
            gell = float(p["gammas"][ell])
            bell = float(p["betas"][ell])
            psi = np.exp(-1j * gell * G) * psi
            frames.append(("sgdiag", psi, gell, ell))
            psi = apply_rx_all(psi, bell, n)
            frames.append(("rx", psi, bell, ell, True))
    elif method == "P_indgamma":
        for ell in range(L):
            gell = float(p["gammas"][ell])
            G = hc + float(p["theta_k"]) * fk + float(p["theta_p"]) * fp
            psi = np.exp(-1j * gell * G) * psi
            frames.append(("sgdiag", psi, gell, ell))
            if want_grad:
                U, dUs = shell_unitary(float(p["c1"][ell]), float(p["c2"][ell]), float(p["c3"][ell]), True)
            else:
                U = shell_unitary(float(p["c1"][ell]), float(p["c2"][ell]), float(p["c3"][ell]), False)
                dUs = None
            psi = apply_all_cavities(psi, U, n)
            frames.append(("shell", psi, U, dUs, ell, "ind"))
    else:
        thk = float(p["theta_k"])
        thp = float(p["theta_p"])
        gamma = float(p["gamma"])
        G = hc + thk * fk + thp * feat
        phase = np.exp(-1j * gamma * G)
        for ell in range(L):
            psi = phase * psi
            frames.append(("diag", psi))
            if want_grad:
                U, dUs = shell_unitary(float(p["c1"][ell]), float(p["c2"][ell]), float(p["c3"][ell]), True)
            else:
                U = shell_unitary(float(p["c1"][ell]), float(p["c2"][ell]), float(p["c3"][ell]), False)
                dUs = None
            psi = apply_all_cavities(psi, U, n)
            frames.append(("shell", psi, U, dUs, ell, "shared"))
    energy = energy_from_state(psi, hc, B, n)
    if not want_grad:
        return psi, float(energy), None
    grad = np.zeros(n_params(method, L), dtype=np.float64)
    lam = apply_H(psi, hc, B, n)
    if method not in ("qaoa", "qaoa_sameG", "P_indgamma"):
        thk = float(p["theta_k"])
        thp = float(p["theta_p"])
        gamma = float(p["gamma"])
        G = hc + thk * fk + thp * feat
    elif method in ("qaoa_sameG", "P_indgamma"):
        G = hc + float(p["theta_k"]) * fk + float(p["theta_p"]) * fp
    for fr in reversed(frames):
        kind = fr[0]
        if kind == "qdiag":
            _, psi_new, gell, ell = fr
            im = np.imag(np.conj(lam) * psi_new)
            grad[2 * ell] += 2.0 * float(np.sum(hc * im))
            lam = np.exp(1j * gell * hc) * lam
        elif kind == "sgdiag":
            _, psi_new, gell, ell = fr
            im = np.imag(np.conj(lam) * psi_new)
            if method == "qaoa_sameG":
                grad[2 + 2 * ell] += 2.0 * float(np.sum(G * im))
            else:
                # P_indgamma: block starts at index 2, stride 4, gamma is offset 0
                grad[2 + 4 * ell] += 2.0 * float(np.sum(G * im))
            grad[0] += 2.0 * float(np.sum((gell * fk) * im))
            grad[1] += 2.0 * float(np.sum((gell * fp) * im))
            lam = np.exp(1j * gell * G) * lam
        elif kind == "diag":
            _, psi_new = fr
            im = np.imag(np.conj(lam) * psi_new)
            grad[0] += 2.0 * float(np.sum(G * im))
            if method == "A":
                grad[1] += 2.0 * float(np.sum((gamma * fk) * im))
            elif method == "P_thetaK0":
                grad[1] += 2.0 * float(np.sum((gamma * feat) * im))
            else:
                grad[1] += 2.0 * float(np.sum((gamma * fk) * im))
                grad[2] += 2.0 * float(np.sum((gamma * feat) * im))
            lam = np.exp(1j * gamma * G) * lam
        elif kind == "rx":
            _, psi_new, beta, ell, same = fr
            sx = apply_sum_X(psi_new, n)
            gbeta = 2.0 * float(np.real(np.vdot(lam, -1j * sx)))
            if same:
                grad[3 + 2 * ell] += gbeta
            else:
                grad[2 * ell + 1] += gbeta
            lam = apply_rx_all(lam, -beta, n)
        elif kind == "shell":
            _, psi_new, U, dUs, ell, tag = fr
            for k in range(3):
                Va = dUs[k] @ np.conjugate(U).T
                acc = None
                for cav in range(n // 3):
                    piece = apply_one_cavity(psi_new, Va, n, cav)
                    acc = piece if acc is None else acc + piece
                gshell = 2.0 * float(np.real(np.vdot(lam, acc)))
                if tag == "ind":
                    grad[2 + 4 * ell + 1 + k] += gshell
                elif method == "A" or method == "P_thetaK0":
                    grad[2 + 3 * ell + k] += gshell
                else:
                    grad[3 + 3 * ell + k] += gshell
            lam = apply_all_cavities(lam, np.conjugate(U).T, n)
        else:
            raise RuntimeError(kind)
    return psi, float(energy), grad


def numerical_grad(method, x, L, hc, fk, fp, fp_shuf, fs, B, n, eps=1e-6) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    g = np.zeros_like(x)
    for i in range(x.size):
        e = np.zeros_like(x)
        e[i] = eps
        _, fp_, _ = simulate(method, x + e, L, hc, fk, fp, fp_shuf, fs, B, n, False)
        _, fm, _ = simulate(method, x - e, L, hc, fk, fp, fp_shuf, fs, B, n, False)
        g[i] = (fp_ - fm) / (2 * eps)
    return g


def fidelity(psi, gs) -> float:
    return float(abs(np.vdot(gs, psi)) ** 2)


def eta_score(E, E0, Emax) -> float:
    den = E0 - Emax
    return float("nan") if abs(den) < 1e-15 else float((E - Emax) / den)
