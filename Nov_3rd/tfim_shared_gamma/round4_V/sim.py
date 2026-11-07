"""Fresh logical and physical simulators for sublattice Rfull.

Does not import the round-3 R simulator. Basis: index = sum bit_i * 2**i,
Z_i = (-1)**bit_i. Target H is never permuted. Packing only chooses which
three logical sites share a shell.
"""
from __future__ import annotations

import hashlib
import json
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.linalg import expm

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "source_data"
MANIFEST_SHA = "14f40daa135640c9eb18d4ff64d2608216f139b51d87e9ede2fcd20939851da9"
PACKING_SHA = "4418b3c33126dd3fd29a6fc74a20314c2e1606764b942301ac4ad01d13de4717"


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def load_manifest():
    raw = (DATA / "FROZEN_TARGET_R2.json").read_bytes()
    if sha256_bytes(raw) != MANIFEST_SHA:
        raise RuntimeError("manifest hash mismatch")
    return json.loads(raw.decode()), raw


def load_packings():
    raw = (DATA / "packings.json").read_bytes()
    if sha256_bytes(raw) != PACKING_SHA:
        raise RuntimeError("packing hash mismatch")
    return json.loads(raw.decode()), raw


def parse_J(manifest, n: int):
    system = manifest["systems"][str(n)]
    B = float(system["B"])
    J = np.zeros((n, n), dtype=np.float64)
    for i, j, s in system["Jij"]:
        J[int(i), int(j)] = J[int(j), int(i)] = float(s)
    return J, B


def z_table(n: int) -> np.ndarray:
    dim = 1 << n
    bits = (np.arange(dim)[:, None] >> np.arange(n)) & 1
    return 1.0 - 2.0 * bits


def upper_rms(mat: np.ndarray) -> float:
    iu = np.triu_indices(mat.shape[0], 1)
    v = mat[iu]
    return float(np.sqrt(np.mean(v * v)))


def pair_diag(coeff: np.ndarray, z: np.ndarray) -> np.ndarray:
    return 0.5 * np.einsum("ai,ij,aj->a", z, coeff, z, optimize=True)


def fk_matrix(J: np.ndarray) -> np.ndarray:
    K = J @ J
    np.fill_diagonal(K, 0.0)
    return 0.5 * (K + K.T)


def block_mean_matrix(J: np.ndarray, packing: list[list[int]]) -> tuple[np.ndarray, np.ndarray]:
    """M_ij = Jbar_ab on inter-cavity pairs and 0 inside a cavity.

    Jbar_ab = (1/9) sum_{i in a, j in b} J_ij. This is the block-constant
    projection of J onto cavity pairs.
    """
    n = J.shape[0]
    M = np.zeros((n, n), dtype=np.float64)
    jb = np.zeros((len(packing), len(packing)), dtype=np.float64)
    for a, ca in enumerate(packing):
        for b, cb in enumerate(packing):
            if a >= b:
                continue
            block = J[np.ix_(ca, cb)]
            jbar = float(block.mean())
            jb[a, b] = jb[b, a] = jbar
            for i in ca:
                for j in cb:
                    M[i, j] = M[j, i] = jbar
    return M, jb


def shell_phi(c1, c2, c3) -> np.ndarray:
    out = np.empty(4)
    for k in range(4):
        s2 = ((3 - 2 * k) ** 2 - 3) / 2.0
        out[k] = c1 * (3 - 2 * k) + c2 * s2 + c3 * ((-1) ** k)
    return out


def _generators():
    I = np.eye(2, dtype=np.complex128)
    X = np.array([[0, 1], [1, 0]], dtype=np.complex128)

    def on(bit):
        mats = [X if r == bit else I for r in range(3)]
        M = mats[2]
        M = np.kron(M, mats[1])
        return np.kron(M, mats[0])

    X0, X1, X2 = on(0), on(1), on(2)
    return X0 + X1 + X2, X0 @ X1 + X0 @ X2 + X1 @ X2, X0 @ X1 @ X2


G1, G2, G3 = _generators()


def mixer(c1, c2, c3) -> np.ndarray:
    return expm(-1j * (c1 * G1 + c2 * G2 + c3 * G3))


def blocks(n: int, sites: list[int]) -> np.ndarray:
    """Local bit r (value 2**r) is global bit sites[r]. First listed site is the low bit."""
    free = [b for b in range(n) if b not in set(sites)]
    nloc = 1 << len(sites)
    index = np.empty((1 << len(free), nloc), dtype=np.int64)
    for f in range(index.shape[0]):
        base = 0
        for k, b in enumerate(free):
            if (f >> k) & 1:
                base |= 1 << b
        for local in range(nloc):
            ind = base
            for r, site in enumerate(sites):
                if (local >> r) & 1:
                    ind |= 1 << site
            index[f, local] = ind
    return index


class System:
    def __init__(self, J, B, packing):
        self.J = np.asarray(J, dtype=np.float64)
        self.B = float(B)
        self.n = self.J.shape[0]
        self.packing = [list(c) for c in packing]
        if sorted(s for c in self.packing for s in c) != list(range(self.n)):
            raise ValueError("packing is not a partition")
        self.z = z_table(self.n)
        self.hc = pair_diag(self.J, self.z)
        self.K = fk_matrix(self.J)
        self.fk = pair_diag(self.K, self.z)
        self.rK = upper_rms(self.K) / upper_rms(self.J)
        self.khat = self.fk / self.rK
        self.M, self.jbar = block_mean_matrix(self.J, self.packing)
        self.md = pair_diag(self.M, self.z)
        self.rM = upper_rms(self.M) / upper_rms(self.J)
        self.mhat = self.md / self.rM
        self.dim = 1 << self.n
        self.plus = np.full(self.dim, 1.0 / np.sqrt(self.dim), dtype=np.complex128)
        self.cav = [blocks(self.n, c) for c in self.packing]
        self.x_index = [blocks(self.n, [i]) for i in range(self.n)]
        # physical bit p=3a+r <-> logical site packing[a][r]
        self.phys_to_log = np.empty(self.n, dtype=np.int64)
        self.log_to_phys = np.empty(self.n, dtype=np.int64)
        for a, cav in enumerate(self.packing):
            for r, site in enumerate(cav):
                self.phys_to_log[3 * a + r] = site
                self.log_to_phys[site] = 3 * a + r
        self.logical_of = self._basis_map(self.phys_to_log)

    def _basis_map(self, bit_image: np.ndarray) -> np.ndarray:
        """logical_of[phys] if bit_image[p] is the logical site of physical bit p."""
        dim = 1 << self.n
        out = np.zeros(dim, dtype=np.int64)
        for p, site in enumerate(bit_image):
            bit = (np.arange(dim) >> p) & 1
            out |= bit << int(site)
        return out

    def g_diag(self, theta_k: float, theta_m: float) -> np.ndarray:
        return self.hc + float(theta_k) * self.khat + float(theta_m) * self.mhat

    def energy(self, psi) -> float:
        p = np.abs(psi) ** 2
        corr = self.z.T @ (self.z * p[:, None])
        e_zz = 0.5 * float(np.sum(self.J * corr))
        e_x = 0.0
        for index in self.x_index:
            amps = psi[index]
            e_x += 2.0 * float(np.real(np.vdot(amps[:, 0], amps[:, 1])))
        return float(e_zz + self.B * e_x)

    def apply_H(self, psi):
        out = self.hc * psi
        acc = np.zeros_like(psi)
        for i, index in enumerate(self.x_index):
            flipped = np.empty_like(psi)
            amps = psi[index]
            flipped[index[:, 0]] = amps[:, 1]
            flipped[index[:, 1]] = amps[:, 0]
            acc += flipped
        return out + self.B * acc

    def apply_cav(self, psi, U):
        out = np.array(psi, copy=True)
        for index in self.cav:
            out[index] = out[index] @ U.T
        return out

    def apply_rx(self, psi, beta):
        c = np.cos(beta)
        s = np.sin(beta)
        U = np.array([[c, -1j * s], [-1j * s, c]], dtype=np.complex128)
        out = np.array(psi, copy=True)
        for index in self.x_index:
            out[index] = out[index] @ U.T
        return out

    def apply_h_all(self, psi):
        h = np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2)
        out = np.array(psi, copy=True)
        for index in self.x_index:
            out[index] = out[index] @ h.T
        return out

    def overlap_shell(self, chi, psi):
        out = np.zeros(3, dtype=np.complex128)
        for index in self.cav:
            ac, ap = chi[index], psi[index]
            for k, G in enumerate((G1, G2, G3)):
                out[k] += np.vdot(ac, ap @ G.T)
        return out

    def overlap_x(self, chi, psi):
        X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.complex128)
        total = 0j
        for index in self.x_index:
            total += np.vdot(chi[index], psi[index] @ X.T)
        return complex(total)


def unpack_rfull(L, x):
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    gamma, thk, thm = float(x[0]), float(x[1]), float(x[2])
    shell = x[3:].reshape(L, 3)
    return gamma, thk, thm, shell


def logical_state(sys: System, L, x) -> np.ndarray:
    gamma, thk, thm, shell = unpack_rfull(L, x)
    phase = np.exp(-1j * gamma * sys.g_diag(thk, thm))
    psi = sys.plus.copy()
    for row in shell:
        psi = psi * phase
        psi = sys.apply_cav(psi, mixer(float(row[0]), float(row[1]), float(row[2])))
    return psi


def physical_state(sys: System, L, x) -> np.ndarray:
    """Physical Fock circuit, returned in the logical basis."""
    gamma, thk, thm, shell = unpack_rfull(L, x)
    g_log = sys.g_diag(thk, thm)
    g_phys = np.empty_like(g_log)
    g_phys[np.arange(sys.dim)] = g_log[sys.logical_of]
    # inverse: physical index whose logical image is i
    phys_of = np.empty(sys.dim, dtype=np.int64)
    phys_of[sys.logical_of] = np.arange(sys.dim)
    phase = np.exp(-1j * gamma * g_phys)
    psi = np.zeros(sys.dim, dtype=np.complex128)
    psi[0] = 1.0  # physical vacuum
    for row in shell:
        psi = sys.apply_h_all(psi)
        psi = psi * phase
        psi = sys.apply_h_all(psi)
        # shell on physical cavities: bits 3a+r, which is contiguous in the physical register
        phi = shell_phi(float(row[0]), float(row[1]), float(row[2]))
        # popcount of each physical cavity
        idx = np.arange(sys.dim)
        ph = np.zeros(sys.dim)
        for a in range(len(sys.packing)):
            bits = 0
            for r in range(3):
                bits += (idx >> (3 * a + r)) & 1
            ph += phi[bits]
        psi = psi * np.exp(-1j * ph)
    psi = sys.apply_h_all(psi)
    logical = np.zeros_like(psi)
    logical[sys.logical_of] = psi
    return logical


def rfull_energy_grad(sys: System, L, x):
    gamma, thk, thm, shell = unpack_rfull(L, x)
    gdiag = sys.g_diag(thk, thm)
    phase = np.exp(-1j * gamma * gdiag)
    us = [mixer(*map(float, row)) for row in shell]
    psi = sys.plus.copy()
    snaps = []
    for ell, U in enumerate(us):
        psi = psi * phase
        snaps.append(psi.copy())
        psi = sys.apply_cav(psi, U)
        snaps.append(psi.copy())
    energy = sys.energy(psi)
    chi = sys.apply_H(psi)
    g_shell = np.zeros_like(shell)
    g_gamma = 0.0
    g_k = 0.0
    g_m = 0.0
    phase_dag = np.conj(phase)
    for ell in range(L - 1, -1, -1):
        g_shell[ell] = 2.0 * np.imag(sys.overlap_shell(chi, snaps[2 * ell + 1]))
        chi = sys.apply_cav(chi, us[ell].conj().T)
        psi_d = snaps[2 * ell]
        g_gamma += 2.0 * float(np.imag(np.vdot(chi, gdiag * psi_d)))
        g_k += 2.0 * float(np.imag(gamma * np.vdot(chi, sys.khat * psi_d)))
        g_m += 2.0 * float(np.imag(gamma * np.vdot(chi, sys.mhat * psi_d)))
        chi = chi * phase_dag
    grad = np.concatenate([[g_gamma, g_k, g_m], g_shell.reshape(-1)])
    return float(energy), grad


def qaoa_energy_grad(sys: System, gammas, betas):
    gammas = np.asarray(gammas, dtype=np.float64)
    betas = np.asarray(betas, dtype=np.float64)
    L = gammas.size
    psi = sys.plus.copy()
    snaps = []
    for ell in range(L):
        psi = psi * np.exp(-1j * gammas[ell] * sys.hc)
        snaps.append(psi.copy())
        psi = sys.apply_rx(psi, float(betas[ell]))
        snaps.append(psi.copy())
    energy = sys.energy(psi)
    chi = sys.apply_H(psi)
    gg = np.zeros(L)
    gb = np.zeros(L)
    for ell in range(L - 1, -1, -1):
        gb[ell] = 2.0 * np.imag(sys.overlap_x(chi, snaps[2 * ell + 1]))
        chi = sys.apply_rx(chi, float(-betas[ell]))
        gg[ell] = 2.0 * float(np.imag(np.vdot(chi, sys.hc * snaps[2 * ell])))
        chi = chi * np.exp(1j * gammas[ell] * sys.hc)
    return float(energy), np.concatenate([gg, gb])


def qaoa_state(sys: System, x) -> np.ndarray:
    L = len(x) // 2
    x = np.asarray(x, dtype=np.float64)
    psi = sys.plus.copy()
    for g, b in zip(x[:L], x[L:]):
        psi = psi * np.exp(-1j * g * sys.hc)
        psi = sys.apply_rx(psi, float(b))
    return psi


def fidelity(gs, psi) -> float:
    ov = np.vdot(gs, psi)
    return float(ov.real ** 2 + ov.imag ** 2)


def overlap(a, b) -> float:
    return float(abs(np.vdot(a, b)))
