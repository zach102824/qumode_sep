"""Noisy density-matrix circuit for the local-ECD + JP ansatz on (d, e, A, B), with the coupler q
entering the master equation during every JP gate.

One layer (same gate order as noiseless/ecd_kbit._layer_fast):

    R_d, R_e, ECD_dA, ECD_eB          ideal unitaries
    idle noise for tau_ECD            cavity loss on A, B; T1/T2 on d, e
    JP gate                           q tensored in at |g>, then
                                      Ry(+pi/2) - wait - Ry(+pi/2) S^dag Ry(-pi/2) - wait - Ry(-pi/2),
                                      q traced out (coupler reset)

Each wait (chi_qA t = pi) is the exact diagonal evolution under the interaction-picture dispersive
Hamiltonian (eq:Htilde of docs/joint_parity_gate.tex) Strang-split with the exact Kraus channels of
every dissipator of eq:lindblad: K(h/2) U(h) K(h) ... U(h) K(h/2), h = tau_wait / n_slice.

Two implementations of the JP gate, identical up to dropping > J photon losses per cavity per gate:

* ``direct``: literally tensors q into rho (dim 2 * 4 nf^2) and runs the sequence. Supports every
  option, including the spectator shifts chi_dA, chi_eB that couple d, e into the waits.
* ``fast`` (default): when d, e do not enter the wait Hamiltonian (chi_spectator = 0) the gate acts on
  d, e only through their own T1/T2 channels, which commute with everything else in the gate. The
  cavity part, traced over q, is then a fixed linear map of shift form
      rho[.., n, m, .., n', m'] -> sum_{j,k} C_jk[n, m, n', m'] rho[.., n-j, m-k, .., n'-j, m'-k]
  (photon loss lowers ket and bra together). C is built once per device by running the same
  q-sequence on every Fock matrix element, and also absorbs the preceding tau_ECD cavity loss.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path

import numpy as np

from noiseless.circuit_local_ecd import rotation_np, unpack_params
from noiseless.ecd_kbit import displace_fast

from .device import Device
from .kraus import (
    apply_cavity_loss,
    apply_diag_unitary,
    apply_qubit_channel,
    apply_unitary,
    apply_conditional,
    cavity_loss_weights,
    dagger,
    pair_conditional_left,
    qubit_kraus,
    superop,
    tensor_in_ground,
    trace_out,
)

# 4-mode axes
D_, E_, A_, B_ = 0, 1, 2, 3


def ry(theta: float) -> np.ndarray:
    c, s = np.cos(theta / 2.0), np.sin(theta / 2.0)
    return np.array([[c, -s], [s, c]], dtype=complex)


RY_P = ry(np.pi / 2.0)
RY_M = ry(-np.pi / 2.0)
S_DAG = np.diag([1.0, -1j])
JP_MID = RY_P @ S_DAG @ RY_M  # applied between the two waits


def vacuum_rho(nf: int) -> np.ndarray:
    dims = (2, 2, nf, nf)
    rho = np.zeros(dims + dims, dtype=complex)
    rho[(0,) * 8] = 1.0
    return rho


def ecd_u4(beta: complex, nf: int) -> np.ndarray:
    """ECD(beta) = |1><0| (x) D(beta/2) + |0><1| (x) D(-beta/2) as U4[i, a, j, b] on (transmon, cavity)."""
    D = displace_fast(beta / 2.0, nf)
    U = np.zeros((2, nf, 2, nf), dtype=complex)
    U[1, :, 0, :] = D
    U[0, :, 1, :] = D.conj().T
    return U


def wait_hamiltonian(dev: Device, nf: int, with_de: bool) -> np.ndarray:
    """Diagonal of the interaction-picture wait Hamiltonian (rad/s), axes (q, [d, e,] A, B)."""
    n = np.arange(nf, dtype=float)
    a = n[:, None]
    b = n[None, :]
    cav = -0.5 * dev.kerr * (a * (a - 1) + b * (b - 1)) - dev.cross_kerr * a * b
    shared = -dev.chi_qa * (a + b)
    if not with_de:
        return np.stack([cav, cav + shared])
    H = np.zeros((2, 2, 2, nf, nf))
    for q in (0, 1):
        for d in (0, 1):
            for e in (0, 1):
                H[q, d, e] = cav + q * shared - dev.chi_spectator * (d * a + e * b)
    return H


# ---------------------------------------------------------------------------
# idle noise on the 4-mode state
# ---------------------------------------------------------------------------
def transmon_idle(rho: np.ndarray, dev: Device, t: float, axes=(D_, E_)) -> np.ndarray:
    if dev.noise_scale == 0.0 or t == 0.0:
        return rho
    kr = qubit_kraus(dev.gamma * t, dev.gamma_phi * t)
    for ax in axes:
        rho = apply_qubit_channel(rho, kr, ax)
    return rho


def cavity_idle(rho: np.ndarray, dev: Device, t: float, axes=(A_, B_)) -> np.ndarray:
    if dev.noise_scale == 0.0 or t == 0.0:
        return rho
    nf = rho.shape[axes[0]]
    w = cavity_loss_weights(dev.kappa * t, nf, dev.jmax)
    for ax in axes:
        rho = apply_cavity_loss(rho, w, ax)
    return rho


# ---------------------------------------------------------------------------
# JP gate, direct: q tensored into the density matrix
# ---------------------------------------------------------------------------
def _wait_direct(rho5: np.ndarray, dev: Device) -> np.ndarray:
    """One wait on (q, d, e, A, B)."""
    nf = rho5.shape[3]
    h = dev.tau_wait / dev.n_slice
    phase = np.exp(-1j * h * wait_hamiltonian(dev, nf, with_de=True))

    def noise(r, t):
        r = cavity_idle(r, dev, t, axes=(3, 4))
        return transmon_idle(r, dev, t, axes=(0, 1, 2))

    rho5 = noise(rho5, h / 2.0)
    for s in range(dev.n_slice):
        rho5 = apply_diag_unitary(rho5, phase)
        rho5 = noise(rho5, h if s < dev.n_slice - 1 else h / 2.0)
    return rho5


def jp_direct(rho: np.ndarray, dev: Device, keep_q: bool = False) -> np.ndarray:
    """Physical JP sequence with q live; returns the (d, e, A, B) state, or (q, d, e, A, B) if keep_q."""
    r = tensor_in_ground(rho, 0)
    r = apply_unitary(r, RY_P, 0)
    r = _wait_direct(r, dev)
    r = apply_unitary(r, JP_MID, 0)
    r = _wait_direct(r, dev)
    if keep_q:
        return apply_unitary(r, RY_M, 0)
    # the closing Ry(-pi/2) is a unitary on q alone, so it does not change Tr_q
    return trace_out(r, 0)


# ---------------------------------------------------------------------------
# JP gate, fast: precomputed cavity channel
# ---------------------------------------------------------------------------
def _x_cavity_loss(X: np.ndarray, weights, jump_axis: int, n_axis: int, np_axis: int) -> np.ndarray:
    """Photon loss in the propagated-element representation, in place, highest jump count first.

    X[..., j, ..., n, ..., n', ...] is the amplitude of input element (n, n') after j losses, so its
    current Fock levels are (n - j, n' - j)."""
    J = X.shape[jump_axis] - 1
    nf = X.shape[n_axis]
    n = np.arange(nf)

    def at(j):
        idx = [slice(None)] * X.ndim
        idx[jump_axis] = j
        return tuple(idx)

    def wvec(w, j, axis):
        cur = n - j
        v = np.where(cur >= 0, w[np.clip(cur, 0, None)], 0.0)
        shp = [1] * (X.ndim - 1)
        shp[axis - (axis > jump_axis)] = nf
        return v.reshape(shp)

    wl = dict(weights)
    for jt in range(J, -1, -1):
        new = wvec(wl[0], jt, n_axis) * wvec(wl[0], jt, np_axis) * X[at(jt)]
        for ell, w in wl.items():
            if ell == 0 or jt - ell < 0:
                continue
            j0 = jt - ell
            new = new + wvec(w, j0, n_axis) * wvec(w, j0, np_axis) * X[at(j0)]
        X[at(jt)] = new
    return X


@lru_cache(maxsize=4)
def jp_cavity_channel(dev: Device, nf: int, jumps: int = 3, pre_time: float = 0.0) -> np.ndarray:
    """C[j, k, n, m, n', m'] of the cavity map (tau = pre_time of idle loss, then the JP gate, traced over q).

    Exact (same q-sequence, Kraus channels and Strang splitting as ``jp_direct``) except that more than
    ``jumps`` photon losses per cavity are dropped; their probability is the trace deficit."""
    if dev.chi_spectator != 0.0:
        raise ValueError("fast JP channel requires chi_spectator = 0 (use method='direct')")
    J = int(jumps)
    sh = (2, 2, J + 1, J + 1, nf, nf, nf, nf)  # q, q', j, k, n, m, n', m'
    X = np.zeros(sh, dtype=complex)
    X[0, 0, 0, 0] = 1.0
    noisy = dev.noise_scale != 0.0

    def cav_loss(X, t):
        if not noisy or t == 0.0:
            return X
        w = cavity_loss_weights(dev.kappa * t, nf, min(J, nf - 1), complete=False)
        X = _x_cavity_loss(X, w, 2, 4, 6)
        return _x_cavity_loss(X, w, 3, 5, 7)

    def q_op(X, S):
        return np.tensordot(S, X, axes=([2, 3], [0, 1]))

    def q_noise(X, t):
        if not noisy or t == 0.0:
            return X
        return q_op(X, superop(qubit_kraus(dev.gamma * t, dev.gamma_phi * t)))

    h = dev.tau_wait / dev.n_slice
    P = np.exp(-1j * h * wait_hamiltonian(dev, nf, with_de=False))  # (q, a, b)
    n = np.arange(nf)

    def phase(X):
        for j in range(J + 1):
            ia = np.clip(n - j, 0, None)
            for k in range(J + 1):
                ib = np.clip(n - k, 0, None)
                Pk = P[:, ia][:, :, ib]  # (q, n, m) at current levels
                X[:, :, j, k] *= Pk[:, None, :, :, None, None] * Pk.conj()[None, :, None, None, :, :]
        return X

    def noise(X, t):
        return q_noise(cav_loss(X, t), t)

    def wait(X):
        X = noise(X, h / 2.0)
        for s in range(dev.n_slice):
            X = phase(X)
            X = noise(X, h if s < dev.n_slice - 1 else h / 2.0)
        return X

    X = cav_loss(X, pre_time)
    X = q_op(X, superop([RY_P]))
    X = wait(X)
    X = q_op(X, superop([JP_MID]))
    X = wait(X)
    C = X[0, 0] + X[1, 1]
    C.flags.writeable = False
    return C


def apply_cavity_channel(rho: np.ndarray, C: np.ndarray) -> np.ndarray:
    """Apply C[j, k, n, m, n', m'] to the (A, B) indices of the 4-mode state.

    With rho viewed as (de, P, d'e', P'), P = n nf + m, the (j, k) term is a contiguous shift by
    o = j nf + k on P and P'. Sources with m < k would wrap into the previous row, but C is exactly
    zero there (fewer than k photons cannot lose k)."""
    nf = rho.shape[A_]
    P = nf * nf
    r = np.ascontiguousarray(rho).reshape(4, P, 4, P)
    out = np.zeros_like(r)
    tmp = np.empty_like(r)
    for j in range(C.shape[0]):
        for k in range(C.shape[1]):
            o = j * nf + k
            c = C[j, k].reshape(P, P)[o:, o:]
            if not np.any(c):
                continue
            t = tmp[:, : P - o, :, : P - o]
            np.multiply(c[None, :, None, :], r[:, o:, :, o:], out=t)
            out[:, : P - o, :, : P - o] += t
    return out.reshape(rho.shape)


# ---------------------------------------------------------------------------
# full circuit
# ---------------------------------------------------------------------------
def ecd_rotation_parts(beta: complex, theta: float, phi: float, nf: int):
    """ECD(beta) R(theta, phi) on one (transmon, cavity) pair in ``conditional_left`` form."""
    R = rotation_np(theta, phi)
    D = displace_fast(beta / 2.0, nf)
    return R[::-1], (D.conj().T, D)


def ideal_block(rho: np.ndarray, layer: dict, nf: int) -> np.ndarray:
    """(ECD_eB R_e)(ECD_dA R_d) rho (...)^dag, using U rho U^dag = U (U rho)^dag for Hermitian rho."""
    pd = ecd_rotation_parts(layer["beta_d"], layer["theta_d"], layer["phi_d"], nf)
    pe = ecd_rotation_parts(layer["beta_e"], layer["theta_e"], layer["phi_e"], nf)
    return pair_conditional_left(dagger(pair_conditional_left(rho, pd, pe)), pd, pe)


def ideal_block_reference(rho: np.ndarray, layer: dict, nf: int) -> np.ndarray:
    """Same map as ``ideal_block`` with each side of rho handled separately (no Hermiticity assumed)."""
    pd = ecd_rotation_parts(layer["beta_d"], layer["theta_d"], layer["phi_d"], nf)
    pe = ecd_rotation_parts(layer["beta_e"], layer["theta_e"], layer["phi_e"], nf)
    rho = apply_conditional(rho, *pd, D_, A_)
    return apply_conditional(rho, *pe, E_, B_)


CHANNEL_CACHE_ENV = "NOISY_CHANNEL_CACHE"


def channel_cache_path(dev: Device, nf: int, jumps: int, pre_time: float, cache_dir) -> Path:
    key = hashlib.sha1(repr((asdict(dev), int(nf), int(jumps), float(pre_time))).encode()).hexdigest()[:16]
    return Path(cache_dir) / f"jp_channel_nf{nf}_J{jumps}_{key}.npy"


@lru_cache(maxsize=4)
def get_cavity_channel(dev: Device, nf: int, jumps: int, pre_time: float) -> np.ndarray:
    """``jp_cavity_channel``, shared through a read-only .npy memmap when $NOISY_CHANNEL_CACHE is set."""
    cache_dir = os.environ.get(CHANNEL_CACHE_ENV)
    if not cache_dir:
        return jp_cavity_channel(dev, nf, jumps, pre_time)
    path = channel_cache_path(dev, nf, jumps, pre_time, cache_dir)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f".{os.getpid()}.tmp.npy")
        np.save(tmp, jp_cavity_channel(dev, nf, jumps, pre_time))
        os.replace(tmp, path)
    return np.load(path, mmap_mode="r")


def noisy_layer(rho: np.ndarray, layer: dict, dev: Device, method: str = "fast", jumps: int = 3) -> np.ndarray:
    nf = rho.shape[A_]
    rho = ideal_block(rho, layer, nf)
    if method == "fast":
        C = get_cavity_channel(dev, nf, jumps, dev.tau_ecd)
        rho = transmon_idle(rho, dev, dev.tau_ecd + dev.tau_jp)
        return apply_cavity_channel(rho, C)
    if method == "direct":
        rho = cavity_idle(rho, dev, dev.tau_ecd)
        rho = transmon_idle(rho, dev, dev.tau_ecd)
        return jp_direct(rho, dev)
    raise ValueError(f"unknown method {method!r}")


def apply_noisy_circuit(x, n_layers: int, nf: int, dev: Device, method: str = "fast",
                        jumps: int = 3, rho0: np.ndarray | None = None) -> np.ndarray:
    rho = vacuum_rho(nf) if rho0 is None else rho0
    for layer in unpack_params(x, n_layers):
        rho = noisy_layer(rho, layer, dev, method, jumps)
    return rho
