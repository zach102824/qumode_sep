"""Sublattice split of full K. Energy never sees the ground state.

Kin keeps FK pairs inside a cavity. Kout keeps every other pair. Their sum is FK.
rX = RMS(X) / RMS(J) over all i<j slots, zeros included. Xhat = X / rX.
G = Hc + thetaIn * Kin / rIn + thetaOut * Kout / rOut + thetaM * Mhat.
The nested Rfull point is thetaIn = thetaK * rIn / rK, thetaOut = thetaK * rOut / rK,
with the same gamma, thetaM, and shell. A zero feature RMS raises.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from Nov_3rd.tfim_shared_gamma.round4_N.model import (
    fwht,
    logical_perm,
    pair_diag,
    shell_weights,
    upper,
    rms,
)

ROOT = Path(__file__).resolve().parent
MANIFEST_SHA = "14f40daa135640c9eb18d4ff64d2608216f139b51d87e9ede2fcd20939851da9"
PACKING_SHA = "4418b3c33126dd3fd29a6fc74a20314c2e1606764b942301ac4ad01d13de4717"

ARMS = {
    "KS": {"packing": "sublattice", "thetas": ("In", "Out", "M"), "features": ("In", "Out", "M")},
    "Rfull": {"packing": "sublattice", "thetas": ("K", "M"), "features": ("K", "M")},
    "qaoa": {"packing": "none", "thetas": (), "features": ()},
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def split_K(K: np.ndarray, packing: list[list[int]]) -> tuple[np.ndarray, np.ndarray]:
    Kin = np.zeros_like(K)
    for cav in packing:
        for a in range(3):
            for b in range(a + 1, 3):
                i, j = cav[a], cav[b]
                Kin[i, j] = Kin[j, i] = K[i, j]
    return Kin, K - Kin


def pair_K(J: np.ndarray) -> np.ndarray:
    K = np.array(J @ J, dtype=np.float64, copy=True)
    np.fill_diagonal(K, 0.0)
    K = 0.5 * (K + K.T)
    np.fill_diagonal(K, 0.0)
    return K


def block_coeff(source: np.ndarray, packing: list[list[int]]) -> np.ndarray:
    C = np.zeros_like(source)
    m = len(packing)
    for a in range(m):
        for b in range(a + 1, m):
            pairs = [(i, j) for i in packing[a] for j in packing[b]]
            w = float(sum(source[i, j] for i, j in pairs)) / float(len(pairs))
            for i, j in pairs:
                C[i, j] = C[j, i] = w
    return C


def z_table(n: int) -> np.ndarray:
    dim = 1 << n
    x = np.arange(dim, dtype=np.uint32)
    z = np.empty((n, dim), dtype=np.float64)
    for q in range(n):
        z[q] = 1.0 - 2.0 * ((x >> np.uint32(q)) & np.uint32(1))
    return z


class Target:
    def __init__(self, n: int):
        man = ROOT / "FROZEN_TARGET_R2.json"
        pak = ROOT / "packings.json"
        if _sha(man) != MANIFEST_SHA or _sha(pak) != PACKING_SHA:
            raise RuntimeError("manifest or packing hash mismatch")
        raw = json.loads(man.read_text())
        packs = json.loads(pak.read_text())
        block = raw["systems"][str(n)]
        J = np.zeros((n, n), dtype=np.float64)
        for i, j, s in block["Jij"]:
            J[int(i), int(j)] = J[int(j), int(i)] = float(s)
        self.n = n
        self.B = float(block["B"])
        self.J = J
        self.z = z_table(n)
        self.hc = pair_diag(J, self.z)
        self.hc_sha = hashlib.sha256(np.ascontiguousarray(self.hc, dtype="<f8").tobytes()).hexdigest()
        self.rms_J = rms(upper(J))
        self.K = pair_K(J)
        self.rK = rms(upper(self.K)) / self.rms_J
        if self.rK <= 0:
            raise RuntimeError("FK RMS is zero")
        self.packings = {
            "contiguous": [list(map(int, t)) for t in packs["contiguous"][str(n)]],
            "sublattice": [list(map(int, t)) for t in packs["sublattice"][str(n)]],
        }
        pack = self.packings["sublattice"]
        self.Kin, self.Kout = split_K(self.K, pack)
        self.rIn = rms(upper(self.Kin)) / self.rms_J
        self.rOut = rms(upper(self.Kout)) / self.rms_J
        if self.rIn <= 0 or self.rOut <= 0:
            raise RuntimeError("Kin or Kout RMS is zero")
        M = block_coeff(J, pack)
        self.rM = rms(upper(M)) / self.rms_J
        if self.rM <= 0:
            raise RuntimeError("M RMS is zero")
        self.diags = {
            "K": pair_diag(self.K / self.rK, self.z),
            "In": pair_diag(self.Kin / self.rIn, self.z),
            "Out": pair_diag(self.Kout / self.rOut, self.z),
            "M": pair_diag(M / self.rM, self.z),
            "M_raw": pair_diag(M, self.z),
        }
        self.weights = shell_weights(n, pack)
        self.weights_contiguous = shell_weights(n, self.packings["contiguous"])
        self.M_raw = M
        self.checks = self._checks(pack, M)

    def _checks(self, pack, M) -> dict:
        err_sum = float(np.max(np.abs(self.Kin + self.Kout - self.K)))
        # Disjoint support: an intra slot of Kin is zero in Kout.
        overlap = float(np.max(np.abs(self.Kin * self.Kout)))
        Hinter = np.array(self.J, copy=True)
        for cav in pack:
            for a in range(3):
                for b in range(a + 1, 3):
                    i, j = cav[a], cav[b]
                    Hinter[i, j] = Hinter[j, i] = 0.0
        err_proj = float(np.max(np.abs(upper(block_coeff(Hinter, pack)) - upper(M))))
        theta = 0.7
        left = self.hc + theta * self.diags["M"]
        right = (
            pair_diag(self.J - Hinter, self.z)
            + pair_diag(Hinter - M, self.z)
            + (1.0 + theta / self.rM) * self.diags["M_raw"]
        )
        err_id = float(np.max(np.abs(left - right)))
        # Random nested point: the two K diagonals reproduce thetaK * Khat.
        rng = np.random.default_rng(20261112)
        thetaK = float(rng.normal())
        tin, tout = nest_thetas(thetaK, self.rIn, self.rOut, self.rK)
        diag_err = float(np.max(np.abs(
            tin * self.diags["In"] + tout * self.diags["Out"] - thetaK * self.diags["K"]
        )))
        r_square = abs((self.rIn ** 2 + self.rOut ** 2) - self.rK ** 2)
        return {
            "Kin_plus_Kout_maxabs": err_sum,
            "Kin_Kout_product_maxabs": overlap,
            "M_projection_maxabs": err_proj,
            "M_identity_maxabs": err_id,
            "nested_diagonal_maxabs": diag_err,
            "r_square_identity": r_square,
            "rK": self.rK,
            "rIn": self.rIn,
            "rOut": self.rOut,
            "rM": self.rM,
            "rms_J": self.rms_J,
            "hc_sha256": self.hc_sha,
            "passed": (
                err_sum < 1e-12 and overlap < 1e-18 and err_proj < 1e-12
                and err_id < 1e-10 and diag_err < 1e-10 and r_square < 1e-12
            ),
        }

    def g_diag(self, arm: str, theta: dict[str, float]) -> np.ndarray:
        g = np.array(self.hc, copy=True)
        for name in ARMS[arm]["features"]:
            g = g + float(theta.get(name, 0.0)) * self.diags[name]
        return g


def nest_thetas(thetaK: float, rIn: float, rOut: float, rK: float) -> tuple[float, float]:
    return float(thetaK * rIn / rK), float(thetaK * rOut / rK)


def coefficient_distance(thetaIn: float, thetaOut: float, rIn: float, rOut: float, rK: float) -> dict:
    """Distance from the nested line c_in = c_out, in units of RMS(J).

    Kin and Kout have disjoint pair support, so the residual RMS of the K
    coefficient list, divided by RMS(J), is |c_in - c_out| * rIn * rOut / rK.
    This does not divide by thetaOut.
    """
    c_in = thetaIn / rIn
    c_out = thetaOut / rOut
    split = c_in - c_out
    distance = abs(split) * rIn * rOut / rK
    return {
        "c_in": float(c_in),
        "c_out": float(c_out),
        "split_c_in_minus_c_out": float(split),
        "normalized_distance": float(distance),
    }


def param_count(arm: str, L: int) -> int:
    if arm == "qaoa":
        return 2 * L
    return 1 + len(ARMS[arm]["thetas"]) + 3 * L


def unpack(arm: str, x: np.ndarray, L: int):
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if x.size != param_count(arm, L):
        raise ValueError(f"{arm} L={L} wants {param_count(arm, L)} got {x.size}")
    if arm == "qaoa":
        return x[:L].copy(), {}, np.column_stack([x[L:], np.zeros(L), np.zeros(L)])
    names = ARMS[arm]["thetas"]
    theta = {name: float(x[1 + i]) for i, name in enumerate(names)}
    cs = x[1 + len(names):].reshape(L, 3).copy()
    return np.full(L, float(x[0])), theta, cs


def pack(arm: str, gamma: float, theta: dict, cs: np.ndarray, gammas=None, betas=None) -> np.ndarray:
    if arm == "qaoa":
        return np.concatenate([np.asarray(gammas, dtype=np.float64), np.asarray(betas, dtype=np.float64)])
    names = ARMS[arm]["thetas"]
    head = [float(gamma)] + [float(theta.get(n, 0.0)) for n in names]
    return np.concatenate([head, np.asarray(cs, dtype=np.float64).reshape(-1)])


def embed_rfull(x_rfull: np.ndarray, L: int, target: Target) -> np.ndarray:
    """Map an Rfull vector onto the nested KS vector at the same depth."""
    gammas, theta, cs = unpack("Rfull", x_rfull, L)
    tin, tout = nest_thetas(theta["K"], target.rIn, target.rOut, target.rK)
    return pack("KS", float(gammas[0]), {"In": tin, "Out": tout, "M": theta["M"]}, cs)


class Runner:
    def __init__(self, target: Target):
        self.t = target
        self.plus = np.full(1 << target.n, 2 ** (-0.5 * target.n), dtype=np.complex128)
        self.gs = None
        self.E0 = None
        self.Emax = None
        self.gap_full = None
        self.gap_acc = None

    def set_spectrum(self, gs, E0, Emax, gap_full, gap_acc):
        self.gs = np.asarray(gs, dtype=np.float64)
        self.E0 = float(E0)
        self.Emax = float(Emax)
        self.gap_full = float(gap_full)
        self.gap_acc = float(gap_acc)

    def _weights(self, arm: str):
        if arm == "qaoa":
            return self.t.weights_contiguous
        return self.t.weights

    def forward(self, arm, x, L, store=False):
        gammas, theta, cs = unpack(arm, x, L)
        g = self.t.hc if arm == "qaoa" else self.t.g_diag(arm, theta)
        parts = {} if arm == "qaoa" else {name: self.t.diags[name] for name in ARMS[arm]["thetas"]}
        w1, w2, w3 = self._weights(arm)
        psi = self.plus.copy()
        snaps = []
        for ell in range(L):
            phase = float(gammas[ell]) * g
            mid = psi * np.exp(-1j * phase)
            phi = cs[ell, 0] * w1 + cs[ell, 1] * w2 + cs[ell, 2] * w3
            x2 = fwht(mid) * np.exp(-1j * phi)
            psi = fwht(x2)
            if store:
                snaps.append((phase, mid, phi, x2, g, parts, w1, w2, w3))
        return psi, snaps, gammas

    def energy(self, psi):
        from Nov_3rd.tfim_shared_gamma.round1_A.hamiltonian import apply_X_sum
        Hpsi = self.t.hc * psi + self.t.B * apply_X_sum(psi, self.t.n)
        return float(np.vdot(psi, Hpsi).real), Hpsi

    def energy_only(self, arm, x, L):
        psi, _, _ = self.forward(arm, x, L, store=False)
        E, _ = self.energy(psi)
        return E, psi

    def energy_grad(self, arm, x, L):
        from Nov_3rd.tfim_shared_gamma.round1_A.hamiltonian import apply_X_sum
        psi, snaps, gammas = self.forward(arm, x, L, store=True)
        E, Hpsi = self.energy(psi)
        lam = Hpsi
        names = () if arm == "qaoa" else ARMS[arm]["thetas"]
        g_gamma = np.zeros(L)
        g_th = {n: 0.0 for n in names}
        g_c = np.zeros((L, 3))
        for ell in range(L - 1, -1, -1):
            phase, mid, phi, x2, g, parts, w1, w2, w3 = snaps[ell]
            lam_x = fwht(lam)
            ov = np.conj(lam_x) * x2
            g_c[ell, 0] = 2.0 * float(np.imag(np.sum(ov * w1)))
            g_c[ell, 1] = 2.0 * float(np.imag(np.sum(ov * w2)))
            g_c[ell, 2] = 2.0 * float(np.imag(np.sum(ov * w3)))
            lam_mid = fwht(lam_x * np.exp(1j * phi))
            ovD = np.conj(lam_mid) * mid
            g_gamma[ell] = 2.0 * float(np.imag(np.sum(ovD * g)))
            for name, part in parts.items():
                g_th[name] += 2.0 * float(np.imag(np.sum(ovD * (float(gammas[ell]) * part))))
            lam = lam_mid * np.exp(1j * phase)
        if arm == "qaoa":
            grad = np.concatenate([g_gamma, g_c[:, 0]])
        else:
            grad = np.concatenate([[g_gamma.sum()], [g_th[n] for n in names], g_c.reshape(-1)])
        return E, grad, psi

    def fidelity(self, psi) -> float:
        return float(abs(np.vdot(self.gs, psi)) ** 2)

    def variance(self, psi) -> float:
        E, Hpsi = self.energy(psi)
        return float(np.vdot(Hpsi, Hpsi).real - E * E)

    def parity_weight(self, psi) -> float:
        n = self.t.n
        x = np.arange(1 << n)
        flip = x ^ ((1 << n) - 1)
        p = float(np.vdot(psi, psi[flip]).real)
        return p
