"""Fock Nov3: D0=NF=8 rewrite, NF=10 pad, ideal W_L and illustrative W_ε.

v4 center-4 sims are the logical rewrite of Nov3rd on Fock 0..7. Ideal pad is
W_L = H^{⊗3} ⊕ I (PR106, algebraic invariance). Illustrative
W_ε = G_{7,8}(ε) W_BinH with ε=0.10 is a phenomenological knob, not a
compiled-W or hardware-calibrated model. PROTOCOL §3.1 names the 7–8
a†-a edge; it does not specify this Givens. SNAP phase 0 on f≥8, Uz, H^ext
unconditional. A SNAP–D pulse compile of W_L is still unmodelled.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from Nov9_SK_SPSA.experiments.FT.instance import (
    block_coeff,
    hat,
    make_J,
    most_negative_pair,
    pair_K,
    rms,
    upper,
)
from Nov9_SK_SPSA.experiments.FT.sim import (
    FockTables,
    _apply_matrix,
    _code_amps,
    metrics_from_fock,
    snap_level_phase,
)

D0 = 8
D0_PLUS_2 = 10
CODE = 8
# Residual 7↔8 mixing after an intended code Hadamard. PROTOCOL §3.1 names
# 7–8 as the unique a†-a code-leaving edge; it does not specify this Givens
# or ε. ε=0.10 is an illustrative phenomenological knob, not a calibrated
# error model and not a SNAP–D compile.
EPS_WL = 0.10


def w_binh(nf: int) -> np.ndarray:
    """H^{⊗3} on Fock 0..7, identity pad on 8..nf-1. Same as build_W_BinH."""
    nf = int(nf)
    if nf < CODE:
        raise ValueError(f"nf={nf} is below the 8-level code")
    h = np.array([[1.0, 1.0], [1.0, -1.0]], dtype=np.float64) / np.sqrt(2.0)
    w3 = h
    for _ in range(2):
        w3 = np.kron(w3, h)
    w = np.eye(nf, dtype=np.float64)
    w[:CODE, :CODE] = w3
    return w


def givens_78(nf: int, eps: float) -> np.ndarray:
    """Real rotation in span{|7⟩,|8⟩}. Identity if nf<9 or ε=0."""
    nf = int(nf)
    g = np.eye(nf, dtype=np.float64)
    if nf < 9 or float(eps) == 0.0:
        return g
    c = float(np.cos(eps))
    s = float(np.sin(eps))
    g[7, 7] = c
    g[8, 8] = c
    g[7, 8] = -s
    g[8, 7] = s
    return g


def w_nonideal(nf: int, eps: float) -> np.ndarray:
    """Ideal W_L then residual 7↔8 Givens. ε=0 recovers w_binh."""
    return givens_78(nf, eps) @ w_binh(nf)


def inst_from_logical(logical) -> SimpleNamespace:
    n, seed, b = logical.n, logical.seed, logical.b
    j = make_J(seed, n)
    groups = logical.groups
    rms_j = rms(upper(j))
    kh, _ = hat(pair_K(j), rms_j)
    mh, _ = hat(block_coeff(j, groups), rms_j)
    r, _, _ = most_negative_pair(j)
    rh, _ = hat(r, rms_j)
    return SimpleNamespace(N=n, groups=groups, J=j, K_hat=kh, M_hat=mh, R1_hat=rh, b=b, gs=logical.gs)


def leakage_fock(psi: np.ndarray, tables: FockTables) -> dict:
    nf, c = tables.nf, tables.C
    tens = np.asarray(psi, dtype=np.complex128).reshape((nf,) * c)
    probs = np.abs(tens) ** 2
    p_code = float(probs[(slice(0, CODE),) * c].sum())
    levels = np.arange(nf, dtype=np.float64)
    mean_ns = []
    for cav in range(c):
        marg = np.moveaxis(probs, cav, 0).reshape(nf, -1).sum(axis=1)
        mean_ns.append(float(marg @ levels))
    p_below_top = float(probs[(slice(0, nf - 1),) * c].sum()) if nf > 1 else 0.0
    norm = float(np.vdot(psi, psi).real)
    return {
        "p_leak": float(1.0 - p_code),
        "p_code": p_code,
        "mean_n_per_cavity": float(np.mean(mean_ns)),
        "mean_n_total": float(np.sum(mean_ns)),
        "p_highest_any_cavity": float(1.0 - p_below_top),
        "norm": norm,
        "norm_deficit": float(abs(1.0 - norm)),
        "nf": int(nf),
        "formula": "p_leak=1-P(all cavities in 0..7); mean_n=E[n_c]; p_highest=1-P(all n_c < nf-1); H^ext energy",
    }


def leakage_logical(psi: np.ndarray, groups: list[list[int]]) -> dict:
    psi = np.asarray(psi, dtype=np.complex128)
    p = np.abs(psi) ** 2
    idx = np.arange(psi.size)
    mean_ns = []
    for cav in groups:
        pop = np.zeros(psi.size, dtype=np.float64)
        for q in cav:
            pop += ((idx >> int(q)) & 1).astype(np.float64)
        mean_ns.append(float(p @ pop))
    norm = float(np.vdot(psi, psi).real)
    return {
        "p_leak": 0.0,
        "p_code": 1.0,
        "mean_n_per_cavity": float(np.mean(mean_ns)),
        "mean_n_total": float(np.sum(mean_ns)),
        "p_highest_any_cavity": 0.0,
        "norm": norm,
        "norm_deficit": float(abs(1.0 - norm)),
        "nf": None,
        "formula": "qubit QAOA: p_leak=0 by construction; mean_n is mean Hamming weight per 3-qubit cavity",
    }


class FockNov3:
    """Fock Nov3 / QAOA with a chosen W_L.

    order='rewrite': |+⟩ → [UcZ → W S W]^L (PR106 / logical identity at ε=0).
    order='hardware': |0⟩ → [W UcZ W → S]^L → W (Nov3rd native; needed when W²≠I).
    """

    def __init__(self, logical, nf: int, eps: float = 0.0, order: str = "rewrite"):
        if order not in ("rewrite", "hardware"):
            raise ValueError(order)
        self.logical = logical
        self.nf = int(nf)
        self.eps = float(eps)
        self.order = order
        self.tables = FockTables(inst_from_logical(logical), self.nf)
        self.W = w_nonideal(self.nf, self.eps)
        self.C = self.tables.C

    def _snap(self, s1: float, s2: float, s3: float, psi: np.ndarray) -> np.ndarray:
        snap = np.exp(-1j * snap_level_phase(self.nf, float(s1), float(s2), float(s3)))
        t = psi.reshape((self.nf,) * self.C)
        for cav in range(self.C):
            shape = [1] * self.C
            shape[cav] = self.nf
            t = t * snap.reshape(shape)
        return t.reshape(-1)

    def _vacuum(self) -> np.ndarray:
        psi = np.zeros(self.tables.dim, dtype=np.complex128)
        psi.reshape((self.nf,) * self.C)[(0,) * self.C] = 1.0
        return psi

    def state_nov3(self, x: np.ndarray, L: int) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        gamma, th_k, th_m = float(x[0]), float(x[1]), float(x[2])
        shells = x[3:].reshape(int(L), 3)
        g = self.tables.uz["J"] + th_k * self.tables.uz["K"] + th_m * self.tables.uz["M"]
        eph = np.exp(-1j * gamma * g)
        nf, c = self.nf, self.C
        if self.order == "rewrite":
            psi = self.tables.plus()
            for ell in range(int(L)):
                psi = psi * eph
                psi = _apply_matrix(psi, self.W, nf, c)
                psi = self._snap(shells[ell, 0], shells[ell, 1], shells[ell, 2], psi)
                psi = _apply_matrix(psi, self.W, nf, c)
            return psi
        psi = self._vacuum()
        for ell in range(int(L)):
            psi = _apply_matrix(psi, self.W, nf, c)
            psi = psi * eph
            psi = _apply_matrix(psi, self.W, nf, c)
            psi = self._snap(shells[ell, 0], shells[ell, 1], shells[ell, 2], psi)
        return _apply_matrix(psi, self.W, nf, c)

    def state_qaoa(self, x: np.ndarray, L: int) -> np.ndarray:
        """Same W embedding: mixer W SNAP(s1=-2β) W, cost Uz J only."""
        x = np.asarray(x, dtype=np.float64)
        if int(x.size) != 2 * int(L):
            raise ValueError(f"QAOA x len {x.size} != {2 * L}")
        nf, c = self.nf, self.C
        if self.order == "rewrite":
            psi = self.tables.plus()
            for ell in range(int(L)):
                psi = psi * np.exp(-1j * float(x[2 * ell]) * self.tables.uz["J"])
                psi = _apply_matrix(psi, self.W, nf, c)
                psi = self._snap(-2.0 * float(x[2 * ell + 1]), 0.0, 0.0, psi)
                psi = _apply_matrix(psi, self.W, nf, c)
            return psi
        psi = self._vacuum()
        for ell in range(int(L)):
            psi = _apply_matrix(psi, self.W, nf, c)
            psi = psi * np.exp(-1j * float(x[2 * ell]) * self.tables.uz["J"])
            psi = _apply_matrix(psi, self.W, nf, c)
            psi = self._snap(-2.0 * float(x[2 * ell + 1]), 0.0, 0.0, psi)
        return _apply_matrix(psi, self.W, nf, c)

    def state(self, x: np.ndarray, L: int) -> np.ndarray:
        return self.state_nov3(x, L)

    def propagate(self, x: np.ndarray, L: int) -> tuple[float, np.ndarray]:
        psi = self.state_nov3(x, L)
        met = metrics_from_fock(psi, self.tables)
        if not met["ok"]:
            raise RuntimeError(f"non-finite Fock energy or norm {met['norm']}")
        return float(met["E"]), psi

    def propagate_qaoa(self, x: np.ndarray, L: int) -> tuple[float, np.ndarray]:
        psi = self.state_qaoa(x, L)
        met = metrics_from_fock(psi, self.tables)
        if not met["ok"]:
            raise RuntimeError(f"non-finite Fock energy or norm {met['norm']}")
        return float(met["E"]), psi

    def fidelity(self, psi: np.ndarray) -> float:
        amp = _code_amps(psi, self.tables)
        return float(np.abs(np.vdot(self.logical.gs, amp)) ** 2)

    def leakage(self, psi: np.ndarray) -> dict:
        return leakage_fock(psi, self.tables)
