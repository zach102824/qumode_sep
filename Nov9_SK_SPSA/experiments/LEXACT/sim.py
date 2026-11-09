"""Candidate exact simulators. Physics matches N3COND/nov3.py. Optimizer recipe unchanged."""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "N3COND"))
sys.path.insert(0, str(HERE.parent / "LE"))
import core as C  # noqa: E402
import kernel as K  # noqa: E402
import kernel_m2 as K2  # noqa: E402
import nov3 as N  # noqa: E402


def hzz_diag_fast(J):
    z = C.z_table(J.shape[0])
    return 0.5 * np.sum(z * (J @ z), axis=0)


class Nov3Fast:
    def __init__(self, J, b, L):
        Nq = J.shape[0]
        if Nq % 3 != 0:
            raise ValueError(Nq)
        self.N, self.L, self.b = Nq, L, float(b)
        self.J = J
        self.pack = C.packing(Nq)
        Khat = N._hat_or_zero(C.feature_K(J), J)
        Mhat = N._hat_or_zero(C.feature_M(J, self.pack), J)
        self.hJ = hzz_diag_fast(J)
        self.hK = hzz_diag_fast(Khat)
        self.hM = hzz_diag_fast(Mhat)
        self.c1, self.c2, self.c3 = N.shell_coeff(Nq, self.pack)
        dim = 1 << Nq
        self.dim = dim
        self.zero = np.zeros(dim, dtype=np.complex128)
        self.zero[0] = 1.0
        self.plus = np.full(dim, 2.0 ** (-0.5 * Nq), dtype=np.complex128)
        self.flip_all = (dim - 1)

    def nparam(self):
        return 3 + 3 * self.L

    def sigma0(self):
        sig = np.empty(self.nparam())
        sig[0] = 0.10
        sig[1:3] = 0.25
        sig[3:] = 0.15
        return sig

    def unpack(self, x):
        x = np.asarray(x, float)
        if x.size != self.nparam():
            raise ValueError(x.size)
        return float(x[0]), x[1:3].copy(), x[3:].reshape(self.L, 3).copy()

    def state(self, x):
        gamma, theta, s = self.unpack(x)
        diag = self.hJ + float(theta[0]) * self.hK + float(theta[1]) * self.hM
        uz = np.exp(-1j * gamma * diag)
        snaps = np.empty((self.L, self.dim), dtype=np.complex128)
        for ell in range(self.L):
            ph = s[ell, 0] * self.c1 + s[ell, 1] * self.c2 + s[ell, 2] * self.c3
            snaps[ell] = np.exp(-1j * ph)
        psi = self.zero.copy()
        K.nov3_evolve(psi, uz, snaps)
        return psi

    def energy(self, x):
        return K.energy_zz_x(self.state(x), self.hJ, self.b)

    def diagnostics(self, x, gs):
        psi = self.state(x)
        F = float(K.overlap2(np.asarray(gs, dtype=np.float64), psi))
        par = float(np.real(np.vdot(psi, psi[::-1])))
        return {
            "E": float(K.energy_zz_x(psi, self.hJ, self.b)),
            "F": F,
            "p_leak": 0.0,
            "parity": par,
            "norm": float(np.sum(psi.real ** 2 + psi.imag ** 2)),
        }


class QAOAFast:
    def __init__(self, J, b, p):
        self.N, self.L, self.b = J.shape[0], p, float(b)
        self.hzz = hzz_diag_fast(J)
        self.dim = 1 << self.N
        self.plus = np.full(self.dim, 2.0 ** (-0.5 * self.N), dtype=np.complex128)

    def nparam(self):
        return 2 * self.L

    def sigma0(self):
        return np.full(2 * self.L, 0.10)

    def state(self, x):
        x = np.asarray(x, float)
        psi = self.plus.copy()
        K.qaoa_evolve(psi, self.hzz, x[: self.L].copy(), x[self.L :].copy())
        return psi

    def energy(self, x):
        return K.energy_zz_x(self.state(x), self.hzz, self.b)

    def diagnostics(self, x, gs):
        psi = self.state(x)
        F = float(K.overlap2(np.asarray(gs, dtype=np.float64), psi))
        par = float(np.real(np.vdot(psi, psi[::-1])))
        return {
            "E": float(K.energy_zz_x(psi, self.hzz, self.b)),
            "F": F,
            "p_leak": 0.0,
            "parity": par,
            "norm": float(np.sum(psi.real ** 2 + psi.imag ** 2)),
        }


class Nov3M2(Nov3Fast):
    def state(self, x):
        gamma, theta, s = self.unpack(x)
        uz_ang = gamma * (self.hJ + float(theta[0]) * self.hK + float(theta[1]) * self.hM)
        snap_ang = np.empty((self.L, self.dim))
        for ell in range(self.L):
            snap_ang[ell] = s[ell, 0] * self.c1 + s[ell, 1] * self.c2 + s[ell, 2] * self.c3
        psi = self.zero.copy()
        K2.nov3_evolve(psi, uz_ang, snap_ang)
        return psi

    def energy(self, x):
        return K2.energy_zz_x(self.state(x), self.hJ, self.b)

    def diagnostics(self, x, gs):
        psi = self.state(x)
        F = float(K2.overlap2(np.asarray(gs, dtype=np.float64), psi))
        par = float(np.real(np.vdot(psi, psi[::-1])))
        return {
            "E": float(K2.energy_zz_x(psi, self.hJ, self.b)),
            "F": F, "p_leak": 0.0, "parity": par,
            "norm": float(np.sum(psi.real ** 2 + psi.imag ** 2)),
        }


class QAOAM2(QAOAFast):
    def state(self, x):
        x = np.asarray(x, float)
        psi = self.plus.copy()
        K2.qaoa_evolve(psi, self.hzz, x[: self.L].copy(), x[self.L :].copy())
        return psi

    def energy(self, x):
        return K2.energy_zz_x(self.state(x), self.hzz, self.b)

    def diagnostics(self, x, gs):
        psi = self.state(x)
        F = float(K2.overlap2(np.asarray(gs, dtype=np.float64), psi))
        par = float(np.real(np.vdot(psi, psi[::-1])))
        return {
            "E": float(K2.energy_zz_x(psi, self.hzz, self.b)),
            "F": F, "p_leak": 0.0, "parity": par,
            "norm": float(np.sum(psi.real ** 2 + psi.imag ** 2)),
        }


def make_sim(name, J, b, L, fast=True, engine=None):
    if engine is None:
        engine = "m1" if fast else "ref"
    if engine in (False, "ref"):
        return N.Nov3(J, b, L) if name == "Nov3" else N.QAOA(J, b, L)
    if engine in (True, "m1"):
        return Nov3Fast(J, b, L) if name == "Nov3" else QAOAFast(J, b, L)
    if engine == "m2":
        return Nov3M2(J, b, L) if name == "Nov3" else QAOAM2(J, b, L)
    raise ValueError((name, engine))
