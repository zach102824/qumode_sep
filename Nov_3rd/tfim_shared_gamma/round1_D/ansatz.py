"""Logical Nov_3rd family D and independent-angle QAOA.

Logical frame, matrix-product order rightmost-first:
  |psi> = M_L D M_{L-1} D ... M_1 D |+>
  D = exp(-i * gamma * Hc)          # one shared gamma for every non-QAOA method
  M = exp(-i * sum_cavities (c1 sum X + c2 sum XX + c3 XXX))

Shell phases on Fock popcount k (contract formula):
  phi(k) = c1*(3-2k) + c2*(((3-2k)^2-3)/2) + c3*(-1)^k

QAOA (strong baseline, NOT shared-gamma):
  |psi> = prod_l exp(-i beta_l sum_i X_i) exp(-i gamma_l Hc) |+>
B enters only the energy, not the mixer.

G1, G2, G3 are simultaneous X-type Paulis, so they commute and
d/dc exp(-i (c1 G1+c2 G2+c3 G3)) = -i G_k U. The energy derivative at a
gate output is 2 Im <chi| G |psi>, with chi the adjoint of H propagated
back to that cut.
"""
from __future__ import annotations

import thread_env  # noqa: F401
import numpy as np
from scipy.linalg import expm

from common import apply_H, hc_diagonal, plus_state, z_table


def shell_phi_levels(c1: float, c2: float, c3: float) -> np.ndarray:
    """Four shell phases, k = 0,1,2,3. No additive gauge constant."""
    out = np.empty(4, dtype=np.float64)
    for k in range(4):
        s2 = ((3 - 2 * k) ** 2 - 3) / 2.0
        out[k] = c1 * (3 - 2 * k) + c2 * s2 + c3 * ((-1) ** k)
    return out


def _pauli_generators() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    I2 = np.eye(2, dtype=np.complex128)
    X = np.array([[0, 1], [1, 0]], dtype=np.complex128)

    def kron3(a, b, c):
        return np.kron(np.kron(a, b), c)

    X1 = kron3(X, I2, I2)
    X2 = kron3(I2, X, I2)
    X3 = kron3(I2, I2, X)
    G1 = X1 + X2 + X3
    G2 = X1 @ X2 + X1 @ X3 + X2 @ X3
    G3 = X1 @ X2 @ X3
    return G1, G2, G3


G1, G2, G3 = _pauli_generators()


def mixer_unitary(c1: float, c2: float, c3: float) -> np.ndarray:
    A = c1 * G1 + c2 * G2 + c3 * G3
    return expm(-1j * A)


def hadamard_w(n: int) -> np.ndarray:
    H = np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2.0)
    W = np.array([[1.0 + 0j]])
    for _ in range(n):
        W = np.kron(W, H)
    return W


def blocks_for_sites(n: int, sites: list[int]) -> np.ndarray:
    """Index array (nblocks, 2**len(sites)).

    sites are slow-to-fast (site 0 is MSB). Local index bit (len-1-k) is sites[k].
    """
    bitpos = [n - 1 - s for s in sites]
    free = [b for b in range(n) if b not in set(bitpos)]
    nloc = 1 << len(sites)
    index = np.empty((1 << len(free), nloc), dtype=np.int64)
    for f in range(index.shape[0]):
        base = 0
        for k, b in enumerate(free):
            if (f >> k) & 1:
                base |= 1 << b
        for local in range(nloc):
            ind = base
            for k, bp in enumerate(bitpos):
                # sites[0] is the slow local bit.
                if (local >> (len(sites) - 1 - k)) & 1:
                    ind |= 1 << bp
            index[f, local] = ind
    return index


class Model:
    def __init__(self, n: int, J: np.ndarray, B: float, hc: np.ndarray | None = None):
        if n % 3 != 0:
            raise ValueError("n must be a multiple of 3")
        self.n = int(n)
        self.nc = n // 3
        self.J = np.asarray(J, dtype=np.float64)
        self.B = float(B)
        self.z = z_table(n)
        self.hc = np.asarray(hc if hc is not None else hc_diagonal(self.J, self.z), dtype=np.float64)
        self.plus = plus_state(n)
        self.dim = 1 << n
        self.cav_index = [blocks_for_sites(n, [3 * a, 3 * a + 1, 3 * a + 2]) for a in range(self.nc)]
        self.x_index = [blocks_for_sites(n, [i]) for i in range(n)]
        self.pc = self._popcounts()
        self._W = None

    def _popcounts(self) -> np.ndarray:
        bits = (1 - self.z.astype(np.int8)) // 2  # bit 1 where Z=-1
        pc = np.zeros((self.nc, self.dim), dtype=np.int64)
        for a in range(self.nc):
            pc[a] = bits[:, 3 * a : 3 * a + 3].sum(axis=1)
        return pc

    @property
    def W(self) -> np.ndarray:
        if self._W is None:
            if self.n > 6:
                raise RuntimeError("dense W is only built for the N=6 physical check")
            self._W = hadamard_w(self.n)
        return self._W

    def energy(self, psi: np.ndarray) -> float:
        p = np.abs(psi) ** 2
        corr = self.z.T @ (self.z * p[:, None])
        e_zz = 0.5 * float(np.sum(self.J * corr))
        e_x = 0.0
        for i, index in enumerate(self.x_index):
            amps = psi[index]
            e_x += float(2.0 * np.real(np.vdot(amps[:, 0], amps[:, 1])))
        return float(e_zz + self.B * e_x)

    def apply_H(self, psi: np.ndarray) -> np.ndarray:
        return apply_H(psi, self.hc, self.B, self.n)

    def apply_cavities(self, psi: np.ndarray, U: np.ndarray) -> np.ndarray:
        # Cavities partition the qubits. Apply sequentially to the updated state:
        # writing every cavity from the original vector would let the last cavity overwrite the rest.
        out = np.array(psi, copy=True)
        for index in self.cav_index:
            out[index] = out[index] @ U.T
        return out

    def apply_rx_all(self, psi: np.ndarray, beta: float) -> np.ndarray:
        c = np.cos(beta)
        s = np.sin(beta)
        U = np.array([[c, -1j * s], [-1j * s, c]], dtype=np.complex128)
        # Gates on different qubits commute, but each must act on the updated state.
        out = np.array(psi, copy=True)
        for index in self.x_index:
            out[index] = out[index] @ U.T
        return out

    def shell_phases(self, c1: float, c2: float, c3: float) -> np.ndarray:
        tab = shell_phi_levels(c1, c2, c3)
        ph = tab[self.pc[0]]
        for a in range(1, self.nc):
            ph = ph + tab[self.pc[a]]
        return ph

    def apply_shell_diagonal(self, psi: np.ndarray, c1: float, c2: float, c3: float) -> np.ndarray:
        return psi * np.exp(-1j * self.shell_phases(c1, c2, c3))

    def overlap_generators(self, chi: np.ndarray, psi: np.ndarray) -> np.ndarray:
        out = np.zeros(3, dtype=np.complex128)
        gens = (G1, G2, G3)
        for index in self.cav_index:
            ac = chi[index]
            ap = psi[index]
            for k, G in enumerate(gens):
                out[k] += np.vdot(ac, ap @ G.T)
        return out

    def overlap_sum_x(self, chi: np.ndarray, psi: np.ndarray) -> complex:
        total = 0j
        X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
        for index in self.x_index:
            total += np.vdot(chi[index], psi[index] @ X.T)
        return complex(total)


def qaoa_forward(model: Model, gammas: np.ndarray, betas: np.ndarray) -> np.ndarray:
    psi = model.plus.copy()
    for g, b in zip(gammas, betas):
        psi = psi * np.exp(-1j * float(g) * model.hc)
        psi = model.apply_rx_all(psi, float(b))
    return psi


def shell_forward(model: Model, gamma: float, shell: np.ndarray) -> np.ndarray:
    """shell shape (L, 3) columns c1, c2, c3. One gamma multiplies every cost block."""
    psi = model.plus.copy()
    phase = np.exp(-1j * float(gamma) * model.hc)
    for row in shell:
        psi = psi * phase
        U = mixer_unitary(float(row[0]), float(row[1]), float(row[2]))
        psi = model.apply_cavities(psi, U)
    return psi


def qaoa_energy_grad(model: Model, gammas: np.ndarray, betas: np.ndarray):
    gammas = np.asarray(gammas, dtype=np.float64)
    betas = np.asarray(betas, dtype=np.float64)
    L = gammas.shape[0]
    psi = model.plus.copy()
    snaps = []
    for ell in range(L):
        psi = psi * np.exp(-1j * gammas[ell] * model.hc)
        snaps.append(psi.copy())
        psi = model.apply_rx_all(psi, float(betas[ell]))
        snaps.append(psi.copy())
    E = model.energy(psi)
    chi = model.apply_H(psi)
    g_gamma = np.zeros(L, dtype=np.float64)
    g_beta = np.zeros(L, dtype=np.float64)
    for ell in range(L - 1, -1, -1):
        psi_m = snaps[2 * ell + 1]
        g_beta[ell] = 2.0 * np.imag(model.overlap_sum_x(chi, psi_m))
        chi = model.apply_rx_all(chi, float(-betas[ell]))
        psi_d = snaps[2 * ell]
        g_gamma[ell] = 2.0 * np.imag(np.vdot(chi, model.hc * psi_d))
        chi = chi * np.exp(1j * gammas[ell] * model.hc)
    return E, g_gamma, g_beta


def shell_energy_grad(model: Model, gamma: float, shell: np.ndarray):
    shell = np.asarray(shell, dtype=np.float64)
    L = shell.shape[0]
    Us = [mixer_unitary(float(r[0]), float(r[1]), float(r[2])) for r in shell]
    psi = model.plus.copy()
    phase = np.exp(-1j * float(gamma) * model.hc)
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
    phase_dag = np.exp(1j * float(gamma) * model.hc)
    for ell in range(L - 1, -1, -1):
        psi_m = snaps[2 * ell + 1]
        g_shell[ell] = 2.0 * np.imag(model.overlap_generators(chi, psi_m))
        chi = model.apply_cavities(chi, Us[ell].conj().T)
        psi_d = snaps[2 * ell]
        g_gamma += 2.0 * float(np.imag(np.vdot(chi, model.hc * psi_d)))
        chi = chi * phase_dag
    return E, g_gamma, g_shell


def n_params(method: str, L: int) -> int:
    if method == "qaoa":
        return 2 * L
    if method == "shared_beta":
        return 1 + L
    if method == "nov3_D":
        return 1 + 3 * L
    if method in ("drop_c2", "drop_c3"):
        return 1 + 2 * L
    raise KeyError(method)


def unpack(method: str, L: int, x: np.ndarray) -> dict:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if x.size != n_params(method, L):
        raise ValueError(f"{method} L={L} expected {n_params(method, L)} params, got {x.size}")
    if method == "qaoa":
        return {"gammas": x[:L].copy(), "betas": x[L:].copy()}
    if method == "shared_beta":
        return {"gammas": np.full(L, x[0]), "betas": x[1:].copy(), "gamma": float(x[0])}
    if method == "nov3_D":
        return {"gamma": float(x[0]), "shell": x[1:].reshape(L, 3).copy()}
    if method == "drop_c2":
        pairs = x[1:].reshape(L, 2)
        shell = np.column_stack([pairs[:, 0], np.zeros(L), pairs[:, 1]])
        return {"gamma": float(x[0]), "shell": shell}
    if method == "drop_c3":
        pairs = x[1:].reshape(L, 2)
        shell = np.column_stack([pairs[:, 0], pairs[:, 1], np.zeros(L)])
        return {"gamma": float(x[0]), "shell": shell}
    raise KeyError(method)


def pack_grad(method: str, g_gamma_vec_or_scalar, g_beta=None, g_shell=None) -> np.ndarray:
    if method == "qaoa":
        return np.concatenate([g_gamma_vec_or_scalar, g_beta])
    if method == "shared_beta":
        return np.concatenate([[float(np.sum(g_gamma_vec_or_scalar))], g_beta])
    shell = np.asarray(g_shell, dtype=np.float64)
    if method == "nov3_D":
        return np.concatenate([[float(g_gamma_vec_or_scalar)], shell.reshape(-1)])
    if method == "drop_c2":
        # parameters store (c1, c3)
        reduced = np.column_stack([shell[:, 0], shell[:, 2]]).reshape(-1)
        return np.concatenate([[float(g_gamma_vec_or_scalar)], reduced])
    if method == "drop_c3":
        reduced = np.column_stack([shell[:, 0], shell[:, 1]]).reshape(-1)
        return np.concatenate([[float(g_gamma_vec_or_scalar)], reduced])
    raise KeyError(method)


def energy_and_grad(model: Model, method: str, L: int, x: np.ndarray):
    spec = unpack(method, L, x)
    if method in ("qaoa", "shared_beta"):
        E, gg, gb = qaoa_energy_grad(model, spec["gammas"], spec["betas"])
        return E, pack_grad(method, gg, g_beta=gb)
    E, gg, gs = shell_energy_grad(model, spec["gamma"], spec["shell"])
    return E, pack_grad(method, gg, g_shell=gs)


def final_state(model: Model, method: str, L: int, x: np.ndarray) -> np.ndarray:
    spec = unpack(method, L, x)
    if method in ("qaoa", "shared_beta"):
        return qaoa_forward(model, spec["gammas"], spec["betas"])
    return shell_forward(model, spec["gamma"], spec["shell"])


def physical_evolve(model: Model, gamma: float, shell: np.ndarray, psi0: np.ndarray) -> np.ndarray:
    """Vacuum-style physical circuit on an arbitrary Fock input.

    |out> = W S_L UcX ... S_1 UcX |psi0>, UcX = W D W, D = exp(-i gamma Hc).
    Equivalent logical action is [M D]^L W |psi0|.
    """
    W = model.W
    psi = np.asarray(psi0, dtype=np.complex128).copy()
    phase = np.exp(-1j * float(gamma) * model.hc)
    for row in shell:
        psi = W @ psi
        psi = phase * psi
        psi = W @ psi
        psi = model.apply_shell_diagonal(psi, float(row[0]), float(row[1]), float(row[2]))
    psi = W @ psi
    return psi


def logical_evolve_from_physical_input(model: Model, gamma: float, shell: np.ndarray, psi0: np.ndarray) -> np.ndarray:
    W = model.W
    psi = W @ np.asarray(psi0, dtype=np.complex128)
    phase = np.exp(-1j * float(gamma) * model.hc)
    for row in shell:
        psi = phase * psi
        U = mixer_unitary(float(row[0]), float(row[1]), float(row[2]))
        psi = model.apply_cavities(psi, U)
    return psi


def fidelity(gs: np.ndarray, psi: np.ndarray) -> float:
    ov = np.vdot(gs, psi)
    return float(np.real(ov.conj() * ov))


def eta_score(E: float, E0: float, Emax: float) -> float:
    denom = E0 - Emax
    if abs(denom) < 1e-15:
        return float("nan")
    return float((E - Emax) / denom)


def canonicalize_angles(method: str, L: int, x: np.ndarray) -> np.ndarray:
    """Fold verified pi/2 mixer periods into (-pi/4, pi/4], then time-reverse so the cost angle is >= 0.

    Gamma of a weighted Hc has no short period and is not folded by pi.
    The caller must re-evaluate the energy; this is a reporting map, not an optimizer constraint.
    """
    y = np.array(x, dtype=np.float64, copy=True)
    half = 0.5 * np.pi

    def fold_component(v):
        # Map to (-pi/4, pi/4].
        z = (v + 0.25 * np.pi) / half
        z = z - np.floor(z)
        return z * half - 0.25 * np.pi

    if method == "qaoa":
        y[L:] = fold_component(y[L:])
        if y[0] < 0.0:
            y = -y
            y[L:] = fold_component(y[L:])
        return y
    if method == "shared_beta":
        y[1:] = fold_component(y[1:])
        if y[0] < 0.0:
            y = -y
            y[1:] = fold_component(y[1:])
        return y
    # shell families: every coefficient after gamma is a shell angle with period pi/2
    y[1:] = fold_component(y[1:])
    if y[0] < 0.0:
        y = -y
        y[1:] = fold_component(y[1:])
    return y
