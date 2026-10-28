"""Ansatz: |0> -> D(alpha0) -> [SNAP -> D -> U_phase^q]^L  (multi-cavity, K=3).

A0: U_phase from exact cost Hamiltonian on the product Gray grid.
q in {1..10}: continuous q_tilde = 1 + 9*sigmoid(u), raised as U_phase ** q
  (phase multiply by q) — equivalent to applying cost phase q times.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from .packing import K_BITS, NU, n_cavities, default_nf
from .gray_grid import build_grid_to_x, gray


@dataclass
class AnsatzConfig:
    n: int
    L: int = 5
    nf: Optional[int] = None
    amax: float = 1.8
    mode: str = "A0"  # A0 exact cost; A1 reserved

    def __post_init__(self):
        self.C = n_cavities(self.n)
        self.nf = int(self.nf or default_nf(self.n))
        assert self.nf >= NU


def _disp_matrix(nf: int, re: float, im: float, amax: float = 1.8) -> np.ndarray:
    """Single-cavity displacement D(re+i*im) as nf x nf complex matrix."""
    re = float(re); im = float(im)
    r_raw = math_hypot(re, im)
    if amax > 0 and r_raw > 1e-12:
        sc = amax * np.tanh(r_raw / amax) / r_raw
        re, im = re * sc, im * sc
    r = math_hypot(re, im)
    ph = np.arctan2(im, re)
    a = np.diag(np.sqrt(np.arange(1, nf, dtype=np.float64)), 1)
    lam, V = np.linalg.eigh(1j * (a.T - a))
    nv = np.arange(nf, dtype=np.float64)
    W = np.exp(1j * ph * nv)[:, None] * V
    return (W * np.exp(-1j * r * lam)[None, :]) @ W.conj().T


def math_hypot(a, b):
    return float(np.sqrt(a * a + b * b + 1e-20))


def q_from_logit(u: float) -> float:
    """Continuous q in (1, 10)."""
    return 1.0 + 9.0 / (1.0 + np.exp(-float(u)))


def integer_q(u: float) -> int:
    return int(np.clip(np.round(q_from_logit(u)), 1, 10))


class MultiCavityAnsatz:
    """Numpy reference simulator for smoke / small-n forward.

    State is a complex array of shape (nf,)*C (product Fock).
    Computational subspace: levels 0..NU-1 per cavity.
    """

    def __init__(self, cfg: AnsatzConfig):
        self.cfg = cfg
        self.C = cfg.C
        self.nf = cfg.nf
        self.L = cfg.L
        self.grid_to_x = build_grid_to_x(cfg.n)
        # Precompute displacement eigenbasis once
        a = np.diag(np.sqrt(np.arange(1, self.nf, dtype=np.float64)), 1)
        lam, V = np.linalg.eigh(1j * (a.T - a))
        self._lam = lam.astype(np.float64)
        self._V = V.astype(np.complex128)
        self._nv = np.arange(self.nf, dtype=np.float64)

    def _D(self, re: float, im: float) -> np.ndarray:
        return _disp_matrix(self.nf, re, im, self.cfg.amax)

    def _apply_local(self, psi: np.ndarray, M: np.ndarray, cav: int) -> np.ndarray:
        """Apply single-cavity matrix M on axis `cav`."""
        return np.moveaxis(np.tensordot(M, psi, axes=([1], [cav])), 0, cav)

    def vacuum(self) -> np.ndarray:
        psi = np.zeros((self.nf,) * self.C, dtype=np.complex128)
        psi[(0,) * self.C] = 1.0
        return psi

    def build_cost_phase(self, E: np.ndarray, gamma: float = 1.0) -> np.ndarray:
        """Diagonal phase on computational product grid: exp(-i * gamma * E_grid * n/m).

        Returns array shape (NU,)*C complex. Leakage levels get phase 0 here
        (applied only on computational slice).
        """
        n = self.cfg.n
        m = int(round(9.93 * n))
        E_grid = E[self.grid_to_x].astype(np.float64)
        Cn = E_grid * (n / m)
        Cn = Cn - Cn.max()  # relative
        return np.exp(-1j * gamma * Cn)

    def forward(
        self,
        E: np.ndarray,
        alpha0: np.ndarray,
        A: np.ndarray,
        TH: np.ndarray,
        q_logits: np.ndarray,
        gamma: float = 1.0,
    ) -> dict:
        """Forward one instance.

        alpha0: (C, 2) re/im for init displacement
        A: (C, L, 2) per-layer per-cavity displacement
        TH: (C, L, NU) SNAP phases on computational levels
        q_logits: (L,) continuous logits -> q in (1,10)
        """
        C, L, nf, NU_ = self.C, self.L, self.nf, NU
        assert alpha0.shape == (C, 2)
        assert A.shape == (C, L, 2)
        assert TH.shape == (C, L, NU_)
        assert q_logits.shape == (L,)

        psi = self.vacuum()
        # init coherent
        for c in range(C):
            psi = self._apply_local(psi, self._D(alpha0[c, 0], alpha0[c, 1]), c)

        cost_u = self.build_cost_phase(E, gamma=gamma)  # (NU,)*C

        for ell in range(L):
            # SNAP
            for c in range(C):
                ph = np.zeros(nf, dtype=np.float64)
                ph[:NU_] = TH[c, ell]
                snap = np.diag(np.exp(-1j * ph))
                psi = self._apply_local(psi, snap, c)
            # D
            for c in range(C):
                psi = self._apply_local(psi, self._D(A[c, ell, 0], A[c, ell, 1]), c)
            # U_phase^q on computational slice
            q = q_from_logit(q_logits[ell])
            phase = cost_u ** q  # elementwise; cost_u already unit-modulus
            # multiply computational subspace
            sl = tuple(slice(0, NU_) for _ in range(C))
            psi = psi.copy()
            psi[sl] = psi[sl] * phase

        # success = population on E==0 computational cells; leakage = pop outside NU
        probs = np.abs(psi) ** 2
        comp_mask = np.zeros_like(probs, dtype=bool)
        comp_mask[tuple(slice(0, NU_) for _ in range(C))] = True
        p_leak = float(probs[~comp_mask].sum())
        E_grid = E[self.grid_to_x]
        succ_mask = (E_grid == 0)
        p_succ = float(probs[tuple(slice(0, NU_) for _ in range(C))][succ_mask].sum())
        return {
            "p_succ": p_succ,
            "p_leak": p_leak,
            "inv_p": 1.0 / max(p_succ, 1e-12),
            "q_int": [integer_q(u) for u in q_logits],
            "q_cont": [float(q_from_logit(u)) for u in q_logits],
        }


def random_params(cfg: AnsatzConfig, rng: np.random.Generator | None = None) -> dict:
    rng = rng or np.random.default_rng(0)
    C, L = cfg.C, cfg.L
    return {
        "alpha0": rng.normal(0, 0.3, size=(C, 2)).astype(np.float64),
        "A": rng.normal(0, 0.3, size=(C, L, 2)).astype(np.float64),
        "TH": rng.normal(0, 0.3, size=(C, L, NU)).astype(np.float64),
        "q_logits": rng.normal(0, 0.5, size=(L,)).astype(np.float64),
        "gamma": 1.0,
    }
