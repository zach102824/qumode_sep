"""Local ECD ansatz on diagonal pairs (d–A) and (e–B), then frozen U.

One layer: (ECD_dA(β_d) R_d(θ_d,φ_d) ‖ ECD_eB(β_e) R_e(θ_e,φ_e)) → U_fixed.
Repeat L* times with the SAME frozen U; fresh ECD params each layer.

Parameter layout (Cartesian, 8 L* reals):
  [Reβ_d, Reβ_e, Imβ_d, Imβ_e, θ_d, θ_e, φ_d, φ_e]  per layer, concatenated.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import qutip as qt

from .encoding import DIMS, NFOCK, identity as full_identity, vacuum
from .unitaries import build_fixed_u


def n_parameters(n_layers: int) -> int:
    return 8 * int(n_layers)


def qubit_rotation(theta: float, phi: float) -> qt.Qobj:
    """R(θ, φ) = exp[-i (θ/2) (X cos φ + Y sin φ)]."""
    gen = np.cos(phi) * qt.sigmax() + np.sin(phi) * qt.sigmay()
    return (-1j * (theta / 2.0) * gen).expm()


def _sigma_minus() -> qt.Qobj:
    return qt.Qobj(np.array([[0.0, 0.0], [1.0, 0.0]], dtype=complex))


def _sigma_plus() -> qt.Qobj:
    return qt.Qobj(np.array([[0.0, 1.0], [0.0, 0.0]], dtype=complex))


def ecd_d_a(beta: complex) -> qt.Qobj:
    """ECD(β) on (d, A); identity on e, B.

    ECD(β) = σ- ⊗ D(β/2) + σ+ ⊗ D(-β/2) on the (transmon, cavity) pair.
    """
    sm, sp = _sigma_minus(), _sigma_plus()
    d_p = qt.displace(NFOCK, beta / 2.0)
    d_m = qt.displace(NFOCK, -beta / 2.0)
    # tensor order: d, e, A, B
    return qt.tensor(sm, qt.qeye(2), d_p, qt.qeye(NFOCK)) + qt.tensor(
        sp, qt.qeye(2), d_m, qt.qeye(NFOCK)
    )


def ecd_e_b(beta: complex) -> qt.Qobj:
    """ECD(β) on (e, B); identity on d, A."""
    sm, sp = _sigma_minus(), _sigma_plus()
    d_p = qt.displace(NFOCK, beta / 2.0)
    d_m = qt.displace(NFOCK, -beta / 2.0)
    return qt.tensor(qt.qeye(2), sm, qt.qeye(NFOCK), d_p) + qt.tensor(
        qt.qeye(2), sp, qt.qeye(NFOCK), d_m
    )


def rotation_d(theta: float, phi: float) -> qt.Qobj:
    return qt.tensor(qubit_rotation(theta, phi), qt.qeye(2), qt.qeye(NFOCK), qt.qeye(NFOCK))


def rotation_e(theta: float, phi: float) -> qt.Qobj:
    return qt.tensor(qt.qeye(2), qubit_rotation(theta, phi), qt.qeye(NFOCK), qt.qeye(NFOCK))


def ecd_pair_d_a(beta: complex, theta: float, phi: float) -> qt.Qobj:
    """ECD(β) R(θ,φ) on (d, A)."""
    return ecd_d_a(beta) * rotation_d(theta, phi)


def ecd_pair_e_b(beta: complex, theta: float, phi: float) -> qt.Qobj:
    """ECD(β) R(θ,φ) on (e, B)."""
    return ecd_e_b(beta) * rotation_e(theta, phi)


def parallel_ecd_layer(
    beta_d: complex,
    beta_e: complex,
    theta_d: float,
    theta_e: float,
    phi_d: float,
    phi_e: float,
) -> qt.Qobj:
    """(ECD on A–d ‖ ECD on B–e) as product of commuting pairs."""
    pair_d = ecd_pair_d_a(beta_d, theta_d, phi_d)
    pair_e = ecd_pair_e_b(beta_e, theta_e, phi_e)
    # Disjoint supports → commute; apply e then d (matches qumode pair1*pair0 style).
    return pair_e * pair_d


def unpack_params(xvec: np.ndarray, n_layers: int) -> list[dict]:
    x = np.asarray(xvec, dtype=float).reshape(-1)
    expected = n_parameters(n_layers)
    if x.size != expected:
        raise ValueError(f"Expected {expected} params for L*={n_layers}, got {x.size}")
    layers = []
    for ell in range(int(n_layers)):
        base = 8 * ell
        re_d, re_e = x[base], x[base + 1]
        im_d, im_e = x[base + 2], x[base + 3]
        th_d, th_e = x[base + 4], x[base + 5]
        ph_d, ph_e = x[base + 6], x[base + 7]
        layers.append(
            {
                "beta_d": complex(re_d, im_d),
                "beta_e": complex(re_e, im_e),
                "theta_d": float(th_d),
                "theta_e": float(th_e),
                "phi_d": float(ph_d),
                "phi_e": float(ph_e),
            }
        )
    return layers


def random_parameters(
    n_layers: int,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Random ECD init (Cartesian). Mag in [0,3], angles in [0,π]."""
    rng = rng or np.random.default_rng()
    out = np.empty(n_parameters(n_layers), dtype=float)
    for ell in range(int(n_layers)):
        base = 8 * ell
        for pair in (0, 1):  # d, e
            mag = rng.uniform(0.0, 3.0)
            arg = rng.uniform(0.0, np.pi)
            beta = mag * np.exp(1j * arg)
            out[base + pair] = beta.real
            out[base + 2 + pair] = beta.imag
            out[base + 4 + pair] = rng.uniform(0.0, np.pi)
            out[base + 6 + pair] = rng.uniform(0.0, np.pi)
    return out


def build_ansatz_unitary(
    xvec: np.ndarray,
    n_layers: int,
    u_fixed: qt.Qobj | None = None,
    u_name: str | None = None,
) -> qt.Qobj:
    """Full circuit unitary: ∏_ℓ [U_fixed · ECD_parallel(ℓ)]."""
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
        # Apply ECD first (rightmost), then U_fixed: U * ECD * |ψ⟩
        uni = u_fixed * ecd * uni
    return uni


def apply_circuit(
    xvec: np.ndarray,
    n_layers: int,
    u_fixed: qt.Qobj,
    ket0: qt.Qobj | None = None,
) -> qt.Qobj:
    """Apply circuit to ket without building full U each time (layer-wise)."""
    ket = vacuum() if ket0 is None else ket0
    for layer in unpack_params(xvec, n_layers):
        ecd = parallel_ecd_layer(
            layer["beta_d"],
            layer["beta_e"],
            layer["theta_d"],
            layer["theta_e"],
            layer["phi_d"],
            layer["phi_e"],
        )
        ket = ecd * ket
        ket = u_fixed * ket
    return ket


def born_probs(ket: qt.Qobj) -> np.ndarray:
    """|⟨basis|ψ⟩|² flattened in (d,e,n_A,n_B) C-order."""
    vec = np.asarray(ket.full(), dtype=complex).reshape(-1)
    p = np.abs(vec) ** 2
    s = float(p.sum())
    if s <= 0.0:
        return p
    return p / s
