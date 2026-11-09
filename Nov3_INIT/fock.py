"""Fock-cutoff Nov3: frozen v4 D0=NF=8, pad D0+2=10, ideal W_L.

v4 center-4 sims are the logical rewrite of Nov3rd on Fock 0..7 (Hilbert dim 8
per cavity). Compiled W_L leakage is not modelled (D5). Off-code rules: SNAP
phase 0 on f≥8, Uz pair phases, H^ext energy.
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
    """|+⟩ → [UcZ → W_L S W_L]^L on nf levels per cavity. Ideal W_L pad."""

    def __init__(self, logical, nf: int):
        self.logical = logical
        self.nf = int(nf)
        self.tables = FockTables(inst_from_logical(logical), self.nf)
        self.W = w_binh(self.nf)
        self.C = self.tables.C

    def state(self, x: np.ndarray, L: int) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        gamma, th_k, th_m = float(x[0]), float(x[1]), float(x[2])
        shells = x[3:].reshape(int(L), 3)
        g = self.tables.uz["J"] + th_k * self.tables.uz["K"] + th_m * self.tables.uz["M"]
        eph = np.exp(-1j * gamma * g)
        psi = self.tables.plus()
        nf, c = self.nf, self.C
        for ell in range(int(L)):
            psi = psi * eph
            psi = _apply_matrix(psi, self.W, nf, c)
            snap = np.exp(-1j * snap_level_phase(nf, float(shells[ell, 0]), float(shells[ell, 1]), float(shells[ell, 2])))
            t = psi.reshape((nf,) * c)
            for cav in range(c):
                shape = [1] * c
                shape[cav] = nf
                t = t * snap.reshape(shape)
            psi = t.reshape(-1)
            psi = _apply_matrix(psi, self.W, nf, c)
        return psi

    def propagate(self, x: np.ndarray, L: int) -> tuple[float, np.ndarray]:
        psi = self.state(x, L)
        met = metrics_from_fock(psi, self.tables)
        if not met["ok"]:
            raise RuntimeError(f"non-finite Fock energy or norm {met['norm']}")
        return float(met["E"]), psi

    def fidelity(self, psi: np.ndarray) -> float:
        amp = _code_amps(psi, self.tables)
        return float(np.abs(np.vdot(self.logical.gs, amp)) ** 2)

    def leakage(self, psi: np.ndarray) -> dict:
        return leakage_fock(psi, self.tables)
