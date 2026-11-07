"""Family H circuits on the frozen LSB basis.

Legal candidate: one gamma, one theta_K, shell (c1,c2,c3) per layer.
  D = exp(-i gamma (Hc + theta_K FK_norm))
  M = exp(-i sum_cavities (c1 sum X + c2 sum XX + c3 XXX))

Relaxed initializer: independent gamma_l, same shared theta and shell.
It is not a legal candidate and not a certified performance ceiling.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import expm

import thread_env  # noqa: F401
from target import apply_H, pair_diagonal, z_table


def shell_phi_levels(c1: float, c2: float, c3: float) -> np.ndarray:
    out = np.empty(4, dtype=np.float64)
    for k in range(4):
        s2 = ((3 - 2 * k) ** 2 - 3) / 2.0
        out[k] = c1 * (3 - 2 * k) + c2 * s2 + c3 * ((-1) ** k)
    return out


def _local_generators():
    I2 = np.eye(2, dtype=np.complex128)
    X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)

    def on_bit(bit: int) -> np.ndarray:
        mats = [X if r == bit else I2 for r in range(3)]
        M = mats[2]
        M = np.kron(M, mats[1])
        M = np.kron(M, mats[0])
        return M

    X0, X1, X2 = on_bit(0), on_bit(1), on_bit(2)
    G1 = X0 + X1 + X2
    G2 = X0 @ X1 + X0 @ X2 + X1 @ X2
    G3 = X0 @ X1 @ X2
    return G1, G2, G3


G1, G2, G3 = _local_generators()


def mixer_unitary(c1: float, c2: float, c3: float) -> np.ndarray:
    return expm(-1j * (c1 * G1 + c2 * G2 + c3 * G3))


def blocks_lsb(n: int, sites_fast_to_slow: list[int]) -> np.ndarray:
    """Local index bit r is global bit sites[r]. Site index equals bit index."""
    bitpos = list(sites_fast_to_slow)
    free = [b for b in range(n) if b not in set(bitpos)]
    nloc = 1 << len(bitpos)
    index = np.empty((1 << len(free), nloc), dtype=np.int64)
    for f in range(index.shape[0]):
        base = 0
        for k, b in enumerate(free):
            if (f >> k) & 1:
                base |= 1 << b
        for local in range(nloc):
            ind = base
            for r, bp in enumerate(bitpos):
                if (local >> r) & 1:
                    ind |= 1 << bp
            index[f, local] = ind
    return index


class Model:
    def __init__(self, n: int, J: np.ndarray, B: float, fk: np.ndarray):
        self.n = int(n)
        self.nc = n // 3
        self.J = np.asarray(J, dtype=np.float64)
        self.B = float(B)
        self.z = z_table(n)
        self.hc = pair_diagonal(self.J, self.z)
        self.fk = np.asarray(fk, dtype=np.float64)
        self.dim = 1 << n
        self.plus = np.full(self.dim, 1.0 / np.sqrt(self.dim), dtype=np.complex128)
        self.cav_index = [blocks_lsb(n, [3 * a, 3 * a + 1, 3 * a + 2]) for a in range(self.nc)]
        self.x_index = [blocks_lsb(n, [i]) for i in range(self.n)]
        self.E0 = None
        self.Emax = None
        self.gs = None

    def g_diag(self, theta: float) -> np.ndarray:
        return self.hc + float(theta) * self.fk

    def energy(self, psi: np.ndarray) -> float:
        p = np.abs(psi) ** 2
        corr = self.z.T @ (self.z * p[:, None])
        e_zz = 0.5 * float(np.sum(self.J * corr))
        e_x = 0.0
        for index in self.x_index:
            amps = psi[index]
            e_x += float(2.0 * np.real(np.vdot(amps[:, 0], amps[:, 1])))
        return float(e_zz + self.B * e_x)

    def apply_H(self, psi: np.ndarray) -> np.ndarray:
        return apply_H(psi, self.hc, self.B, self.n)

    def apply_cavities(self, psi: np.ndarray, U: np.ndarray) -> np.ndarray:
        out = np.array(psi, copy=True)
        for index in self.cav_index:
            out[index] = out[index] @ U.T
        return out

    def apply_rx_all(self, psi: np.ndarray, beta: float) -> np.ndarray:
        c = np.cos(beta)
        s = np.sin(beta)
        U = np.array([[c, -1j * s], [-1j * s, c]], dtype=np.complex128)
        out = np.array(psi, copy=True)
        for index in self.x_index:
            out[index] = out[index] @ U.T
        return out

    def overlap_generators(self, chi: np.ndarray, psi: np.ndarray) -> np.ndarray:
        out = np.zeros(3, dtype=np.complex128)
        for index in self.cav_index:
            ac = chi[index]
            ap = psi[index]
            for k, G in enumerate((G1, G2, G3)):
                out[k] += np.vdot(ac, ap @ G.T)
        return out

    def overlap_sum_x(self, chi: np.ndarray, psi: np.ndarray) -> complex:
        X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
        total = 0j
        for index in self.x_index:
            total += np.vdot(chi[index], psi[index] @ X.T)
        return complex(total)


def fidelity(gs: np.ndarray, psi: np.ndarray) -> float:
    ov = np.vdot(gs, psi)
    return float((ov.real * ov.real) + (ov.imag * ov.imag))


def eta_score(E: float, E0: float, Emax: float) -> float:
    denom = E0 - Emax
    return float((E - Emax) / denom)


def n_params_legal(L: int) -> int:
    return 2 + 3 * L


def n_params_relaxed(L: int) -> int:
    return L + 1 + 3 * L


def n_params_theta0(L: int) -> int:
    return 1 + 3 * L


def unpack_legal(L: int, x: np.ndarray) -> tuple[float, float, np.ndarray]:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if x.size != n_params_legal(L):
        raise ValueError("legal width")
    return float(x[0]), float(x[1]), x[2:].reshape(L, 3).copy()


def unpack_relaxed(L: int, x: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if x.size != n_params_relaxed(L):
        raise ValueError("relaxed width")
    return x[:L].copy(), float(x[L]), x[L + 1 :].reshape(L, 3).copy()


def unpack_theta0(L: int, x: np.ndarray) -> tuple[float, np.ndarray]:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if x.size != n_params_theta0(L):
        raise ValueError("theta0 width")
    return float(x[0]), x[1:].reshape(L, 3).copy()


def _shell_forward(model: Model, diag_phase_factors: list[np.ndarray], shell: np.ndarray) -> np.ndarray:
    psi = model.plus.copy()
    for ell, phase in enumerate(diag_phase_factors):
        psi = psi * phase
        row = shell[ell]
        psi = model.apply_cavities(psi, mixer_unitary(float(row[0]), float(row[1]), float(row[2])))
    return psi


def legal_state(model: Model, gamma: float, theta: float, shell: np.ndarray) -> np.ndarray:
    phase = np.exp(-1j * float(gamma) * model.g_diag(theta))
    return _shell_forward(model, [phase] * shell.shape[0], shell)


def relaxed_state(model: Model, gammas: np.ndarray, theta: float, shell: np.ndarray) -> np.ndarray:
    gdiag = model.g_diag(theta)
    phases = [np.exp(-1j * float(g) * gdiag) for g in gammas]
    return _shell_forward(model, phases, shell)


def qaoa_state(model: Model, gammas: np.ndarray, betas: np.ndarray) -> np.ndarray:
    psi = model.plus.copy()
    for g, b in zip(gammas, betas):
        psi = psi * np.exp(-1j * float(g) * model.hc)
        psi = model.apply_rx_all(psi, float(b))
    return psi


def _backprop_shell(model: Model, phases: list[np.ndarray], Us: list[np.ndarray], snaps: list[np.ndarray]):
    """snaps alternate after-D, after-M. Returns chi at the end is not needed; layer grads of the diagonal generator expectation and shell."""
    L = len(Us)
    psi = snaps[-1]
    E = model.energy(psi)
    chi = model.apply_H(psi)
    g_shell = np.zeros((L, 3), dtype=np.float64)
    diag_ov = np.zeros(L, dtype=np.complex128)
    for ell in range(L - 1, -1, -1):
        g_shell[ell] = 2.0 * np.imag(model.overlap_generators(chi, snaps[2 * ell + 1]))
        chi = model.apply_cavities(chi, Us[ell].conj().T)
        # chi and snaps[2*ell] are after the diagonal gate. The derivative below is filled by the caller
        # using this chi and the stored state. We stash chi by applying the overlap with a unit diagonal later.
        diag_ov[ell] = np.vdot(chi, snaps[2 * ell])  # placeholder overwritten
        # Keep chi and the state for the caller via a list.
        phases  # silence lint; phases applied by caller
        chi = chi * np.conj(phases[ell])
    return E, g_shell


def legal_energy_grad(model: Model, L: int, x: np.ndarray):
    gamma, theta, shell = unpack_legal(L, x)
    gdiag = model.g_diag(theta)
    phase = np.exp(-1j * gamma * gdiag)
    Us = [mixer_unitary(float(r[0]), float(r[1]), float(r[2])) for r in shell]
    psi = model.plus.copy()
    snaps = []
    for ell in range(L):
        psi = psi * phase
        snaps.append(psi.copy())
        psi = model.apply_cavities(psi, Us[ell])
        snaps.append(psi.copy())
    E = model.energy(psi)
    chi = model.apply_H(psi)
    g_shell = np.zeros_like(shell)
    g_gamma = 0.0
    g_theta = 0.0
    phase_dag = np.conj(phase)
    for ell in range(L - 1, -1, -1):
        g_shell[ell] = 2.0 * np.imag(model.overlap_generators(chi, snaps[2 * ell + 1]))
        chi = model.apply_cavities(chi, Us[ell].conj().T)
        psi_d = snaps[2 * ell]
        ov_g = np.vdot(chi, gdiag * psi_d)
        ov_f = np.vdot(chi, model.fk * psi_d)
        g_gamma += 2.0 * float(np.imag(ov_g))
        g_theta += 2.0 * float(np.imag(gamma * ov_f))
        chi = chi * phase_dag
    grad = np.concatenate([[g_gamma, g_theta], g_shell.reshape(-1)])
    return float(E), grad


def relaxed_energy_grad(model: Model, L: int, x: np.ndarray, lam: float):
    gammas, theta, shell = unpack_relaxed(L, x)
    gdiag = model.g_diag(theta)
    phases = [np.exp(-1j * float(g) * gdiag) for g in gammas]
    Us = [mixer_unitary(float(r[0]), float(r[1]), float(r[2])) for r in shell]
    psi = model.plus.copy()
    snaps = []
    for ell in range(L):
        psi = psi * phases[ell]
        snaps.append(psi.copy())
        psi = model.apply_cavities(psi, Us[ell])
        snaps.append(psi.copy())
    E = model.energy(psi)
    chi = model.apply_H(psi)
    g_shell = np.zeros_like(shell)
    g_gamma = np.zeros(L, dtype=np.float64)
    g_theta = 0.0
    for ell in range(L - 1, -1, -1):
        g_shell[ell] = 2.0 * np.imag(model.overlap_generators(chi, snaps[2 * ell + 1]))
        chi = model.apply_cavities(chi, Us[ell].conj().T)
        psi_d = snaps[2 * ell]
        ov_g = np.vdot(chi, gdiag * psi_d)
        ov_f = np.vdot(chi, model.fk * psi_d)
        g_gamma[ell] = 2.0 * float(np.imag(ov_g))
        g_theta += 2.0 * float(np.imag(float(gammas[ell]) * ov_f))
        chi = chi * np.conj(phases[ell])
    mean = float(np.mean(gammas))
    dev = gammas - mean
    penalty = float(lam * np.dot(dev, dev))
    g_gamma = g_gamma + (2.0 * lam) * dev
    grad = np.concatenate([g_gamma, [g_theta], g_shell.reshape(-1)])
    return float(E), penalty, grad


def theta0_energy_grad(model: Model, L: int, x: np.ndarray):
    gamma, shell = unpack_theta0(L, x)
    full = np.concatenate([[gamma, 0.0], shell.reshape(-1)])
    E, g = legal_energy_grad(model, L, full)
    # Drop the theta component. It is not a free parameter.
    grad = np.concatenate([[g[0]], g[2:]])
    return E, grad


def qaoa_energy_grad(model: Model, gammas: np.ndarray, betas: np.ndarray):
    gammas = np.asarray(gammas, dtype=np.float64)
    betas = np.asarray(betas, dtype=np.float64)
    L = gammas.size
    psi = model.plus.copy()
    snaps = []
    for ell in range(L):
        psi = psi * np.exp(-1j * gammas[ell] * model.hc)
        snaps.append(psi.copy())
        psi = model.apply_rx_all(psi, float(betas[ell]))
        snaps.append(psi.copy())
    E = model.energy(psi)
    chi = model.apply_H(psi)
    g_gamma = np.zeros(L)
    g_beta = np.zeros(L)
    for ell in range(L - 1, -1, -1):
        g_beta[ell] = 2.0 * np.imag(model.overlap_sum_x(chi, snaps[2 * ell + 1]))
        chi = model.apply_rx_all(chi, float(-betas[ell]))
        g_gamma[ell] = 2.0 * float(np.imag(np.vdot(chi, model.hc * snaps[2 * ell])))
        chi = chi * np.exp(1j * gammas[ell] * model.hc)
    return float(E), g_gamma, g_beta


def qaoa_pack(gammas, betas) -> np.ndarray:
    return np.concatenate([np.asarray(gammas, dtype=np.float64), np.asarray(betas, dtype=np.float64)])


def project_relaxed_to_legal(L: int, x: np.ndarray) -> np.ndarray:
    gammas, theta, shell = unpack_relaxed(L, x)
    gamma = float(np.mean(gammas))
    return np.concatenate([[gamma, theta], shell.reshape(-1)])


def legal_to_relaxed(L: int, x: np.ndarray) -> np.ndarray:
    gamma, theta, shell = unpack_legal(L, x)
    return np.concatenate([np.full(L, gamma), [theta], shell.reshape(-1)])
