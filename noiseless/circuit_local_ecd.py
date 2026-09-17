"""Local ECD ansatz on diagonal pairs (d–A) and (e–B), then frozen U.

One layer: (ECD_dA R_d ‖ ECD_eB R_e) → U_fixed.
Repeat L* times with the SAME frozen U; fresh ECD params each layer.

Params (Cartesian, 8 L* reals per trial):
  [Reβ_d, Reβ_e, Imβ_d, Imβ_e, θ_d, θ_e, φ_d, φ_e] × L*
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import qutip as qt
from scipy.linalg import expm

from .encoding import NFOCK, identity as full_identity
from .unitaries import build_fixed_u

_DIM = 2 * 2 * NFOCK * NFOCK
NFOCK_SQ = NFOCK * NFOCK

_SM = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=complex)
_SP = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=complex)
_SX = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
_SY = np.array([[0.0, -1j], [1j, 0.0]], dtype=complex)


def n_parameters(n_layers: int) -> int:
    return 8 * int(n_layers)


def qubit_rotation(theta: float, phi: float) -> qt.Qobj:
    gen = np.cos(phi) * qt.sigmax() + np.sin(phi) * qt.sigmay()
    return (-1j * (theta / 2.0) * gen).expm()


def _sigma_minus() -> qt.Qobj:
    return qt.Qobj(np.array([[0.0, 0.0], [1.0, 0.0]], dtype=complex))


def _sigma_plus() -> qt.Qobj:
    return qt.Qobj(np.array([[0.0, 1.0], [0.0, 0.0]], dtype=complex))


@lru_cache(maxsize=1)
def _destroy_np() -> np.ndarray:
    a = np.zeros((NFOCK, NFOCK), dtype=complex)
    for n in range(1, NFOCK):
        a[n - 1, n] = np.sqrt(n)
    return a


def displace_np(alpha: complex) -> np.ndarray:
    a = _destroy_np()
    gen = complex(alpha) * a.conj().T - np.conj(complex(alpha)) * a
    return expm(gen)


def rotation_np(theta: float, phi: float) -> np.ndarray:
    gen = np.cos(phi) * _SX + np.sin(phi) * _SY
    return expm(-1j * (theta / 2.0) * gen)


def ecd_2x8_np(beta: complex) -> np.ndarray:
    """16×16 ECD(β) on one (transmon, cavity) pair."""
    dp = displace_np(beta / 2.0)
    dm = displace_np(-beta / 2.0)
    return np.kron(_SM, dp) + np.kron(_SP, dm)


def vacuum_np() -> np.ndarray:
    v = np.zeros(_DIM, dtype=complex)
    v[0] = 1.0
    return v


def qobj_to_np(u: qt.Qobj) -> np.ndarray:
    return np.asarray(u.full(), dtype=complex)


def extract_u_ab(u_full: np.ndarray) -> np.ndarray:
    """Extract U_AB (64×64) from I_d ⊗ I_e ⊗ U_AB (256×256)."""
    u4 = np.asarray(u_full, dtype=complex).reshape(2, 2, NFOCK, NFOCK, 2, 2, NFOCK, NFOCK)
    return u4[0, 0, :, :, 0, 0, :, :].reshape(NFOCK_SQ, NFOCK_SQ).copy()


def apply_layer_factored(
    ket: np.ndarray,
    beta_d: complex,
    beta_e: complex,
    theta_d: float,
    theta_e: float,
    phi_d: float,
    phi_e: float,
    u_ab: np.ndarray,
) -> np.ndarray:
    """Apply R_d, R_e, ECD_dA, ECD_eB, U_AB → matches U·(ECD_e R_e)·(ECD_d R_d)."""
    psi = np.asarray(ket, dtype=complex).reshape(2, 2, NFOCK, NFOCK)

    rd = rotation_np(theta_d, phi_d)
    psi = np.einsum("ij,jklm->iklm", rd, psi, optimize=True)

    re = rotation_np(theta_e, phi_e)
    psi = np.einsum("ij,kjlm->kilm", re, psi, optimize=True)

    ecd_d = ecd_2x8_np(beta_d)
    tmp = np.transpose(psi, (1, 3, 0, 2)).reshape(2, NFOCK, 16)
    tmp = np.einsum("ij,abj->abi", ecd_d, tmp, optimize=True)
    psi = np.transpose(tmp.reshape(2, NFOCK, 2, NFOCK), (2, 0, 3, 1))

    ecd_e = ecd_2x8_np(beta_e)
    tmp = np.transpose(psi, (0, 2, 1, 3)).reshape(2, NFOCK, 16)
    tmp = np.einsum("ij,abj->abi", ecd_e, tmp, optimize=True)
    psi = np.transpose(tmp.reshape(2, NFOCK, 2, NFOCK), (0, 2, 1, 3))

    flat = psi.reshape(2, 2, NFOCK_SQ)
    flat = np.einsum("ij,dej->dei", u_ab, flat, optimize=True)
    return flat.reshape(-1)


def unpack_params(xvec: np.ndarray, n_layers: int) -> list[dict]:
    x = np.asarray(xvec, dtype=float).reshape(-1)
    expected = n_parameters(n_layers)
    if x.size != expected:
        raise ValueError(f"Expected {expected} params for L*={n_layers}, got {x.size}")
    layers = []
    for ell in range(int(n_layers)):
        base = 8 * ell
        layers.append(
            {
                "beta_d": complex(x[base], x[base + 2]),
                "beta_e": complex(x[base + 1], x[base + 3]),
                "theta_d": float(x[base + 4]),
                "theta_e": float(x[base + 5]),
                "phi_d": float(x[base + 6]),
                "phi_e": float(x[base + 7]),
            }
        )
    return layers


def apply_circuit_np(
    xvec: np.ndarray,
    n_layers: int,
    u_fixed_np: np.ndarray,
    ket0: np.ndarray | None = None,
    u_ab: np.ndarray | None = None,
) -> np.ndarray:
    ket = vacuum_np() if ket0 is None else np.asarray(ket0, dtype=complex).reshape(-1).copy()
    if u_ab is None:
        u_ab = u_fixed_np if u_fixed_np.shape == (NFOCK_SQ, NFOCK_SQ) else extract_u_ab(u_fixed_np)
    for layer in unpack_params(xvec, n_layers):
        ket = apply_layer_factored(
            ket,
            layer["beta_d"],
            layer["beta_e"],
            layer["theta_d"],
            layer["theta_e"],
            layer["phi_d"],
            layer["phi_e"],
            u_ab,
        )
    return ket


def random_parameters(
    n_layers: int,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    rng = rng or np.random.default_rng()
    out = np.empty(n_parameters(n_layers), dtype=float)
    for ell in range(int(n_layers)):
        base = 8 * ell
        for pair in (0, 1):
            mag = rng.uniform(0.0, 3.0)
            arg = rng.uniform(0.0, np.pi)
            beta = mag * np.exp(1j * arg)
            out[base + pair] = beta.real
            out[base + 2 + pair] = beta.imag
            out[base + 4 + pair] = rng.uniform(0.0, np.pi)
            out[base + 6 + pair] = rng.uniform(0.0, np.pi)
    return out


def ecd_d_a(beta: complex) -> qt.Qobj:
    sm, sp = _sigma_minus(), _sigma_plus()
    dp = qt.displace(NFOCK, beta / 2.0)
    dm = qt.displace(NFOCK, -beta / 2.0)
    return qt.tensor(sm, qt.qeye(2), dp, qt.qeye(NFOCK)) + qt.tensor(
        sp, qt.qeye(2), dm, qt.qeye(NFOCK)
    )


def ecd_e_b(beta: complex) -> qt.Qobj:
    sm, sp = _sigma_minus(), _sigma_plus()
    dp = qt.displace(NFOCK, beta / 2.0)
    dm = qt.displace(NFOCK, -beta / 2.0)
    return qt.tensor(qt.qeye(2), sm, qt.qeye(NFOCK), dp) + qt.tensor(
        qt.qeye(2), sp, qt.qeye(NFOCK), dm
    )


def rotation_d(theta: float, phi: float) -> qt.Qobj:
    return qt.tensor(qubit_rotation(theta, phi), qt.qeye(2), qt.qeye(NFOCK), qt.qeye(NFOCK))


def rotation_e(theta: float, phi: float) -> qt.Qobj:
    return qt.tensor(qt.qeye(2), qubit_rotation(theta, phi), qt.qeye(NFOCK), qt.qeye(NFOCK))


def parallel_ecd_layer(
    beta_d: complex,
    beta_e: complex,
    theta_d: float,
    theta_e: float,
    phi_d: float,
    phi_e: float,
) -> qt.Qobj:
    pair_d = ecd_d_a(beta_d) * rotation_d(theta_d, phi_d)
    pair_e = ecd_e_b(beta_e) * rotation_e(theta_e, phi_e)
    return pair_e * pair_d


def build_ansatz_unitary(
    xvec: np.ndarray,
    n_layers: int,
    u_fixed: qt.Qobj | None = None,
    u_name: str | None = None,
) -> qt.Qobj:
    if u_fixed is None:
        if u_name is None:
            raise ValueError("Provide u_fixed or u_name")
        u_fixed = build_fixed_u(u_name)
    uni = full_identity()
    for layer in unpack_params(xvec, n_layers):
        ecd = parallel_ecd_layer(
            layer["beta_d"],
            layer["beta_e"],
            layer["theta_d"],
            layer["theta_e"],
            layer["phi_d"],
            layer["phi_e"],
        )
        uni = u_fixed * ecd * uni
    return uni


def apply_circuit(
    xvec: np.ndarray,
    n_layers: int,
    u_fixed: qt.Qobj,
    ket0: qt.Qobj | None = None,
) -> qt.Qobj:
    u_np = qobj_to_np(u_fixed)
    k0 = None if ket0 is None else np.asarray(ket0.full(), dtype=complex).reshape(-1)
    vec = apply_circuit_np(xvec, n_layers, u_np, k0)
    return qt.Qobj(vec.reshape(-1, 1), dims=[[2, 2, NFOCK, NFOCK], [1]])


def born_probs_np(ket: np.ndarray) -> np.ndarray:
    p = np.abs(np.asarray(ket, dtype=complex).reshape(-1)) ** 2
    s = float(p.sum())
    return p if s <= 0.0 else p / s


def born_probs(ket: qt.Qobj | np.ndarray) -> np.ndarray:
    if isinstance(ket, qt.Qobj):
        vec = np.asarray(ket.full(), dtype=complex).reshape(-1)
    else:
        vec = np.asarray(ket, dtype=complex).reshape(-1)
    return born_probs_np(vec)

extract_u_ab = extract_u_ab
