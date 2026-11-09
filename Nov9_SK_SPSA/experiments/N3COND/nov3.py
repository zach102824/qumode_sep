"""Original Nov3rd on the quantum SK Hamiltonian, and QAOA p=L.

Circuit, fixed by the owner for this stream (no per-layer gamma, no displacement):

    |0> -> [U_c^X -> bare shell SNAP]^L -> W

    U_c^X = W exp(-i gamma (J + theta_K Khat + theta_M Mhat) · ZZ) W
    W = H^{⊗N} (binary Fock, interleaved cavities of 3 qubits)
    SNAP phase on a cavity with popcount p:
        s1 * p + s2 * p*(p-1)/2 + s3 * [p==3]
    One shared gamma and one (theta_K, theta_M) for every layer. s is per layer.

The |+> rewrite |+> -> [U_cZ -> W S W]^L is an identity of this circuit, checked in tests.
With theta=0 and s2=s3=0 it is QAOA with gamma_l = gamma and beta_l = -s1_l/2.

Objective H = sum_{i<j} J_ij Z_i Z_j - B sum_i X_i. Features K, M enter only the ansatz.
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "LE"))
import core as C  # noqa: E402


def fwht(psi: np.ndarray) -> np.ndarray:
    """Normalized Hadamard transform, H^{⊗N} on a length-2^N vector."""
    n = psi.shape[0]
    y = np.array(psi, copy=True, dtype=np.complex128)
    h = 1
    while h < n:
        y = y.reshape(n // (2 * h), 2, h)
        a = y[:, 0, :].copy()
        b = y[:, 1, :].copy()
        y[:, 0, :] = a + b
        y[:, 1, :] = a - b
        y = y.reshape(n)
        h *= 2
    y /= np.sqrt(n)
    return y


def _hat_or_zero(F, J):
    """RMS-match F to J. A structurally zero feature (one cavity) contributes 0."""
    if C.rms(C.upper(F)) <= 0:
        return np.zeros_like(F)
    return C.hat(F, J)


def shell_coeff(N, pack):
    """Per-basis-state shell coefficients c1, c2, c3 summed over cavities."""
    dim = 1 << N
    idx = np.arange(dim)
    c1 = np.zeros(dim)
    c2 = np.zeros(dim)
    c3 = np.zeros(dim)
    for cav in pack:
        p = np.zeros(dim, dtype=np.int64)
        for site in cav:
            p += (idx >> site) & 1
        c1 += p
        c2 += p * (p - 1) / 2.0
        c3 += (p == 3).astype(np.float64)
    return c1, c2, c3


class Nov3:
    """Shared-gamma original Nov3rd. Parameter layout [gamma, theta_K, theta_M, (s1,s2,s3)_l]."""

    def __init__(self, J, b, L):
        N = J.shape[0]
        if N % 3 != 0:
            raise ValueError(f"N={N} is not divisible by 3")
        self.N, self.L, self.b = N, L, float(b)
        self.J = J
        self.pack = C.packing(N)
        Khat = _hat_or_zero(C.feature_K(J), J)
        Mhat = _hat_or_zero(C.feature_M(J, self.pack), J)
        z = C.z_table(N)
        self.hJ = C.hzz_diag(J, z)
        self.hK = C.hzz_diag(Khat, z)
        self.hM = C.hzz_diag(Mhat, z)
        self.c1, self.c2, self.c3 = shell_coeff(N, self.pack)
        dim = 1 << N
        self.zero = np.zeros(dim, dtype=np.complex128)
        self.zero[0] = 1.0
        self.plus = np.full(dim, 2.0 ** (-0.5 * N), dtype=np.complex128)
        self.idx = np.arange(dim)
        self.xflip = [self.idx ^ (1 << i) for i in range(N)]
        self.flip_all = self.idx ^ (dim - 1)

    def nparam(self):
        return 3 + 3 * self.L

    def sigma0(self):
        """Protocol group scales: gamma 0.10, theta 0.25, s 0.15. No alpha."""
        sig = np.empty(self.nparam())
        sig[0] = 0.10
        sig[1:3] = 0.25
        sig[3:] = 0.15
        return sig

    def families(self):
        fam = {"gamma": [0], "thetaK": [1], "thetaM": [2], "s1": [], "s2": [], "s3": []}
        for ell in range(self.L):
            fam["s1"].append(3 + 3 * ell)
            fam["s2"].append(3 + 3 * ell + 1)
            fam["s3"].append(3 + 3 * ell + 2)
        return fam

    def unpack(self, x):
        x = np.asarray(x, float)
        if x.size != self.nparam():
            raise ValueError(f"got {x.size} params, want {self.nparam()}")
        return float(x[0]), x[1:3].copy(), x[3:].reshape(self.L, 3).copy()

    def cost_phase(self, gamma, theta):
        diag = self.hJ + float(theta[0]) * self.hK + float(theta[1]) * self.hM
        return np.exp(-1j * gamma * diag)

    def snap_phase(self, s):
        ph = s[0] * self.c1 + s[1] * self.c2 + s[2] * self.c3
        return np.exp(-1j * ph)

    def state(self, x):
        """Owner circuit: |0> -> [W exp(-i gamma E) W -> SNAP]^L -> W."""
        gamma, theta, s = self.unpack(x)
        uz = self.cost_phase(gamma, theta)
        psi = self.zero.copy()
        for ell in range(self.L):
            psi = fwht(psi)
            psi *= uz
            psi = fwht(psi)
            psi *= self.snap_phase(s[ell])
        psi = fwht(psi)
        return psi

    def state_plus_rewrite(self, x):
        """|+> -> [U_cZ -> W S W]^L. Must match state()."""
        gamma, theta, s = self.unpack(x)
        uz = self.cost_phase(gamma, theta)
        psi = self.plus.copy()
        for ell in range(self.L):
            psi *= uz
            psi = fwht(psi)
            psi *= self.snap_phase(s[ell])
            psi = fwht(psi)
        return psi

    def energy_of_state(self, psi):
        prob = psi.real ** 2 + psi.imag ** 2
        ex = 0.0
        for ix in self.xflip:
            ex += float(np.real(np.vdot(psi, psi[ix])))
        return float(prob @ self.hJ) - self.b * ex

    def energy(self, x):
        return self.energy_of_state(self.state(x))

    def diagnostics(self, x, gs):
        psi = self.state(x)
        F = float(abs(np.vdot(gs, psi)) ** 2)
        par = float(np.real(np.vdot(psi, psi[self.flip_all])))
        return {
            "E": self.energy_of_state(psi),
            "F": F,
            "p_leak": 0.0,
            "parity": par,
            "norm": float(np.sum(psi.real ** 2 + psi.imag ** 2)),
        }


class QAOA:
    """p=L QAOA on the same H. Angles [gamma_0..gamma_{p-1}, beta_0..beta_{p-1}]."""

    def __init__(self, J, b, p):
        self.inner = C.QAOA(J, b, p)
        self.N, self.L, self.b = self.inner.N, p, float(b)
        self.J = J

    def nparam(self):
        return self.inner.nparam()

    def sigma0(self):
        return self.inner.sigma()

    def families(self):
        return {"gamma": list(range(self.L)), "beta": list(range(self.L, 2 * self.L))}

    def energy(self, x):
        return self.inner.energy(x)

    def state(self, x):
        return self.inner.state(x)

    def diagnostics(self, x, gs):
        d = self.inner.diagnostics(x, gs)
        d["p_leak"] = 0.0
        return d


def circuit_id(name):
    return {"QAOA": 0, "Nov3": 40}[name]


def method_id(name, cond):
    base = circuit_id(name)
    return base if cond == "C0" else base + 1


def rng_for(seed, N, L, mid, start, stream):
    return np.random.default_rng(np.random.SeedSequence([int(seed), int(N), 0, int(L), int(mid), int(start), int(stream)]))


def tqa_nov3(L):
    """Shared-gamma embedding of the protocol TQA ramp.

    QAOA start 0 uses gamma_l = t_l*0.75, beta_l = -(1-t_l)*0.75, t_l=(l+0.5)/L.
    sum_l gamma_l = 0.375*L, so one shared gamma applied L times is 0.375.
    s1_l = -2 beta_l, s2=s3=0, theta=0. Then beta_mixer = -s1/2 equals beta_l.
    """
    t = (np.arange(L) + 0.5) / L
    beta = -(1.0 - t) * 0.75
    s = np.zeros((L, 3))
    s[:, 0] = -2.0 * beta
    return np.concatenate([[0.375], [0.0, 0.0], s.reshape(-1)])


def tqa_qaoa(L):
    t = (np.arange(L) + 0.5) / L
    return np.concatenate([t * 0.75, -(1.0 - t) * 0.75])


def starts_for(name, seed, N, L):
    """Four starts. Start 0 is the TQA embedding. Starts 1..3 are U(-0.6, 0.6), stream 2.

    The seed uses circuit_id, so C0 and C1 share starts. Not tuned.
    """
    cid = circuit_id(name)
    nparam = 3 + 3 * L if name == "Nov3" else 2 * L
    out = [tqa_nov3(L) if name == "Nov3" else tqa_qaoa(L)]
    for s in range(1, 4):
        out.append(rng_for(seed, N, L, cid, s, 2).uniform(-0.6, 0.6, size=nparam))
    return out


def make_sim(name, J, b, L):
    if name == "Nov3":
        return Nov3(J, b, L)
    if name == "QAOA":
        return QAOA(J, b, L)
    raise ValueError(name)
