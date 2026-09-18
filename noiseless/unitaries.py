"""Fixed bus unitaries U on A⊗B (identity on transmons d, e).

CLI / results names (approved plan):
  identity, bs_pi6, bs_pi4, bs_pi3, bs_pi2, cz_nm, snap_a_pi, snap_b_pi,
  ck_pi2, ck_pi4, cphase_nn
"""

from __future__ import annotations

import numpy as np
import qutip as qt

from .encoding import NFOCK, identity as full_identity

U_NAMES = (
    "identity",
    "bs_pi4",
    "bs_pi2",
    "cz_nm",
    "snap_a_pi",
    "snap_b_pi",
    "ck_pi2",
    "ck_pi4",
    "cphase_nn",
    "bs_pi6",
    "bs_pi3",
)


def _embed_ab(u_ab: qt.Qobj) -> qt.Qobj:
    """Embed (NFOCK×NFOCK)⊗(NFOCK×NFOCK) operator as I_d ⊗ I_e ⊗ U_AB."""
    return qt.tensor(qt.qeye(2), qt.qeye(2), u_ab)


def beamsplitter_ab(theta: float) -> qt.Qobj:
    """U_BS(θ) = exp(-i θ (a†b + ab†)) on A⊗B."""
    a = qt.destroy(NFOCK)
    b = qt.destroy(NFOCK)
    gen = qt.tensor(a.dag(), b) + qt.tensor(a, b.dag())
    return _embed_ab((-1j * float(theta) * gen).expm())


def cz_nm_ab() -> qt.Qobj:
    """|n,m⟩ ↦ (-1)^{n m} |n,m⟩."""
    diag = np.ones(NFOCK * NFOCK, dtype=complex)
    for n in range(NFOCK):
        for m in range(NFOCK):
            if (n * m) % 2 == 1:
                diag[n * NFOCK + m] = -1.0
    u_ab = qt.Qobj(np.diag(diag), dims=[[NFOCK, NFOCK], [NFOCK, NFOCK]])
    return _embed_ab(u_ab)


def ck_phase_ab(phi: float) -> qt.Qobj:
    """|n,m⟩ ↦ exp(-i φ n m) |n,m⟩ (Fock-diagonal controlled-phase family)."""
    diag = np.ones(NFOCK * NFOCK, dtype=complex)
    for n in range(NFOCK):
        for m in range(NFOCK):
            diag[n * NFOCK + m] = np.exp(-1j * float(phi) * n * m)
    u_ab = qt.Qobj(np.diag(diag), dims=[[NFOCK, NFOCK], [NFOCK, NFOCK]])
    return _embed_ab(u_ab)


def cphase_nn_ab() -> qt.Qobj:
    """|n,n⟩ ↦ -|n,n⟩ for n≥1; |n,m⟩ unchanged if n≠m (and |0,0⟩ unchanged)."""
    diag = np.ones(NFOCK * NFOCK, dtype=complex)
    for n in range(1, NFOCK):
        diag[n * NFOCK + n] = -1.0
    u_ab = qt.Qobj(np.diag(diag), dims=[[NFOCK, NFOCK], [NFOCK, NFOCK]])
    return _embed_ab(u_ab)


def snap_a_pi_ab() -> qt.Qobj:
    """|n,m⟩ ↦ e^{-i π n} |n,m⟩ = (-1)^n |n,m⟩."""
    diag = np.ones(NFOCK * NFOCK, dtype=complex)
    for n in range(NFOCK):
        phase = (-1.0) ** n
        for m in range(NFOCK):
            diag[n * NFOCK + m] = phase
    u_ab = qt.Qobj(np.diag(diag), dims=[[NFOCK, NFOCK], [NFOCK, NFOCK]])
    return _embed_ab(u_ab)


def snap_b_pi_ab() -> qt.Qobj:
    """|n,m⟩ ↦ e^{-i π m} |n,m⟩ = (-1)^m |n,m⟩."""
    diag = np.ones(NFOCK * NFOCK, dtype=complex)
    for n in range(NFOCK):
        for m in range(NFOCK):
            diag[n * NFOCK + m] = (-1.0) ** m
    u_ab = qt.Qobj(np.diag(diag), dims=[[NFOCK, NFOCK], [NFOCK, NFOCK]])
    return _embed_ab(u_ab)


def build_fixed_u(name: str) -> qt.Qobj:
    key = str(name).lower().strip()
    if key == "identity":
        return full_identity()
    if key == "bs_pi6":
        return beamsplitter_ab(np.pi / 6.0)
    if key == "bs_pi4":
        return beamsplitter_ab(np.pi / 4.0)
    if key == "bs_pi3":
        return beamsplitter_ab(np.pi / 3.0)
    if key == "bs_pi2":
        return beamsplitter_ab(np.pi / 2.0)
    if key == "cz_nm":
        return cz_nm_ab()
    if key == "ck_pi2":
        return ck_phase_ab(np.pi / 2.0)
    if key == "ck_pi4":
        return ck_phase_ab(np.pi / 4.0)
    if key == "cphase_nn":
        return cphase_nn_ab()
    if key == "snap_a_pi":
        return snap_a_pi_ab()
    if key == "snap_b_pi":
        return snap_b_pi_ab()
    raise ValueError(f"Unknown fixed U {name!r}; choose from {U_NAMES}")


def ab_matrix_elements(u_full: qt.Qobj, n: int, m: int) -> complex:
    """⟨0,0,n,m| U |0,0,n,m⟩ for diagonal checks (d=e=0)."""
    ket = qt.tensor(
        qt.basis(2, 0), qt.basis(2, 0), qt.basis(NFOCK, n), qt.basis(NFOCK, m)
    )
    return complex((ket.dag() * u_full * ket))
