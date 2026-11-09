"""Original Nov3 and p=L QAOA. Features enter only the cost unitary."""
from __future__ import annotations

import numpy as np

from Nov9_SK_SPSA.experiments.FT.instance import (
    block_coeff,
    hat,
    make_J,
    packing,
    pair_K,
    pair_diag,
    rms,
    upper,
    z_table,
)
from Nov9_SK_SPSA.experiments.FT.sim import apply_rx_all, fwht


def shell_phase_from_pops(pops: list[np.ndarray], s1: float, s2: float, s3: float) -> np.ndarray:
    ph = np.zeros(pops[0].shape[0], dtype=np.float64)
    for pop in pops:
        ph += s1 * pop + s2 * (pop * (pop - 1.0) / 2.0) + np.where(pop == 3.0, s3, 0.0)
    return ph


def shell_phase_percavity(pops: list[np.ndarray], shells: np.ndarray) -> np.ndarray:
    """Independent (s1, s2, s3) per cavity. Same polynomial as the shared shell."""
    shells = np.asarray(shells, dtype=np.float64)
    if shells.shape != (len(pops), 3):
        raise ValueError(f"shells shape {shells.shape} != ({len(pops)}, 3)")
    ph = np.zeros(pops[0].shape[0], dtype=np.float64)
    for pop, (s1, s2, s3) in zip(pops, shells):
        ph += float(s1) * pop + float(s2) * (pop * (pop - 1.0) / 2.0) + np.where(pop == 3.0, float(s3), 0.0)
    return ph


def pops_of(n: int, groups: list[list[int]]) -> list[np.ndarray]:
    idx = np.arange(1 << n)
    pops = []
    for cav in groups:
        pop = np.zeros(idx.shape[0], dtype=np.float64)
        for q in cav:
            pop += ((idx >> int(q)) & 1).astype(np.float64)
        pops.append(pop)
    return pops


class Logical:
    def __init__(self, n: int, seed: int, b: float, gs: np.ndarray):
        self.n = int(n)
        self.seed = int(seed)
        self.b = float(b)
        self.gs = np.asarray(gs, dtype=np.float64)
        J = make_J(self.seed, self.n)
        self.groups = packing(self.n)
        z = z_table(self.n)
        self.j_upper = upper(J)
        zz_pairs = []
        for i in range(self.n):
            for j in range(i + 1, self.n):
                zz_pairs.append(z[i] * z[j])
        self.zz_pairs = np.stack(zz_pairs, axis=0)
        self.hc = pair_diag(J, z)
        rms_j = rms(upper(J))
        k_hat, _sk = hat(pair_K(J), rms_j)
        m_hat, _sm = hat(block_coeff(J, self.groups), rms_j)
        self.d_k = pair_diag(k_hat, z)
        self.d_m = pair_diag(m_hat, z)
        self.pops = pops_of(self.n, self.groups)
        self.plus = np.full(1 << self.n, 2.0 ** (-0.5 * self.n), dtype=np.complex128)

    def energy(self, psi: np.ndarray) -> float:
        e_zz = float(np.vdot(psi, self.hc * psi).real)
        ex = 0.0
        for q in range(self.n):
            half = 1 << q
            rest = psi.size // (2 * half)
            a = psi.reshape(rest, 2, half)
            ex += float(np.vdot(a[:, 0, :], a[:, 1, :]).real)
            ex += float(np.vdot(a[:, 1, :], a[:, 0, :]).real)
        return e_zz - self.b * ex

    def nov3(self, x: np.ndarray, L: int) -> np.ndarray:
        gamma, th_k, th_m = float(x[0]), float(x[1]), float(x[2])
        shells = np.asarray(x[3:], dtype=np.float64).reshape(L, 3)
        gdiag = self.hc + th_k * self.d_k + th_m * self.d_m
        eph = np.exp(-1j * gamma * gdiag)
        psi = self.plus.copy()
        for ell in range(L):
            psi *= eph
            psi = fwht(psi)
            s1, s2, s3 = shells[ell]
            psi *= np.exp(-1j * shell_phase_from_pops(self.pops, float(s1), float(s2), float(s3)))
            psi = fwht(psi)
        return psi

    def nov3_pair(self, x: np.ndarray, L: int) -> np.ndarray:
        """Shared Uc (γ, θK, θM) + per-cavity SNAP shells. Layout (L, C, 3)."""
        C = len(self.groups)
        gamma, th_k, th_m = float(x[0]), float(x[1]), float(x[2])
        shells = np.asarray(x[3:], dtype=np.float64).reshape(L, C, 3)
        gdiag = self.hc + th_k * self.d_k + th_m * self.d_m
        eph = np.exp(-1j * gamma * gdiag)
        psi = self.plus.copy()
        for ell in range(L):
            psi *= eph
            psi = fwht(psi)
            psi *= np.exp(-1j * shell_phase_percavity(self.pops, shells[ell]))
            psi = fwht(psi)
        return psi

    def qaoa(self, x: np.ndarray, L: int) -> np.ndarray:
        psi = self.plus.copy()
        for ell in range(L):
            psi *= np.exp(-1j * float(x[2 * ell]) * self.hc)
            psi = apply_rx_all(psi, float(x[2 * ell + 1]), self.n)
        return psi

    def vm(self, x: np.ndarray, L: int) -> np.ndarray:
        """Paper V(M)=exp(-i/4 ∑_{n<m} M_nm Z_n Z_m) then shared Rx(β) per layer.

        Layout per layer: upper-triangle M (N(N-1)/2) then β. Not Nov3 shells.
        """
        n_pairs = self.n * (self.n - 1) // 2
        width = n_pairs + 1
        x = np.asarray(x, dtype=np.float64)
        if int(x.size) != L * width:
            raise ValueError(f"VM x len {x.size} != {L * width}")
        psi = self.plus.copy()
        for ell in range(L):
            sl = ell * width
            phase = 0.25 * (x[sl : sl + n_pairs] @ self.zz_pairs)
            psi *= np.exp(-1j * phase)
            psi = apply_rx_all(psi, float(x[sl + n_pairs]), self.n)
        return psi

    def propagate(self, method: str, L: int, x: np.ndarray) -> tuple[float, np.ndarray]:
        if method == "Nov3rd":
            psi = self.nov3(x, L)
        elif method == "Nov3pair":
            psi = self.nov3_pair(x, L)
        elif method == "VM":
            psi = self.vm(x, L)
        elif method == "QAOA":
            psi = self.qaoa(x, L)
        else:
            raise ValueError(method)
        energy = self.energy(psi)
        if not np.isfinite(energy):
            raise RuntimeError("non-finite energy")
        return float(energy), psi

    def fidelity(self, psi: np.ndarray) -> float:
        return float(np.abs(np.vdot(self.gs, psi)) ** 2)
