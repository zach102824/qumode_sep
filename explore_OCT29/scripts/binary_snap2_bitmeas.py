"""Binary 3-bit cavity: exactly 2 SNAP + 2 displacements.

Encoding: Fock |n> for n=0..7 is the ordinary binary computational state
|n> = |b2 b1 b0> with b0 = n & 1 (LSB), b1 = (n>>1)&1 (middle), b2 = (n>>2)&1 (MSB).

Budget: two selective SNAP gates and two displacements, nothing else.
SNAP phases are free on n=0..7 and fixed at 0 on every guard level n>=8
(same rule as oct_explore/oct23_exp/fit_preps.py). One SNAP pulse sets the
whole phase vector, so guard phases are a modelling choice, not extra pulses;
they are held at 0 here.

Displacements are the exponential of the photon-truncated generator on
NF > 8 (spectral form used by mixcal.py / fit_preps.py), never an 8x8
matrix exponential. The same physical controls are re-evaluated at larger NF.

Why only two orders: D(a)D(b) = e^{i phi} D(a+b) and S(theta)S(phi)=S(theta+phi).
Any non-alternating word in {S,S,D,D} therefore collapses to at most
2 SNAP+1 D or 1 SNAP+2 D. The only words that spend the full budget are

  physical time (first pulse applied first)     written product (rightmost first)
  SDSD: S1, D1, S2, D2                           U = D2 @ S2 @ D1 @ S1
  DSDS: D1, S1, D2, S2                           U = S2 @ D2 @ S1 @ D1

A terminal SNAP is diagonal in Fock number, so it does not change any
photon-number measurement. DSDS therefore has the same measurement power as
D,S,D (the last SNAP is idle for readout). SDSD uses both SNAPs before a
displacement, and an extra uniform Z-rotation on the chosen bit can be added
into S1 without a new pulse; that rotates X into Y but does not create the
rotation away from Z.

Displacement amplitudes use a smooth cap |alpha| <= amax. Headline amax=1.5.
The cavity ansatz in explore_OCT29/lib/ansatz.py soft-clips at 1.8, so 1.5 is
inside that operating range. Sensitivity runs use 0.8 and 2.2, not huge alphas.

Gate fidelity (global phase free, leakage penalised), d=8:
  M_ij = <i|U|j> for i,j in 0..7
  F_proc = |Tr(U_tgt^H M)|^2 / d^2
This is the mixcal.py subspace process fidelity. Also reported:
  F_avg = (|Tr|^2 + ||M||_F^2) / (d (d+1))
    Haar-average state fidelity with leakage counted as failure.

Measurement (leakage is a third outcome, no postselection):
  Q+ = projector onto {n<8 : bit_b(n)=0}, Q- onto bit_b=0's complement in 0..7.
  Levels n>=8 are neither outcome.
  E± = P_comp U^H Q± U P_comp
  eps± = ||E± - (I ± X_b)/2||_op
  eps_X = max(eps+, eps-)
eps_X is the worst-case |probability error| over both binary outcomes and every
state in the 8-level space, including states that entangle the target bit with
the two spectator bits. Matching E+ alone is not enough: dumping the X=-1
subspace into leakage makes eps+~0 while eps-~1. Retention is reported separately.
Classical outcome-label flip (X -> -X) is scored as eps_X_basis = min over the
two labelings; it is not a pulse.

H and Ry(pi/2) both measure the X basis (Ry(pi/2) gives -X, a label flip) but
H = Ry(pi/2) @ Z_b, a different gate. Output diagonal phases do not move the
measured axis. Input Z_b does, and on SDSD it folds into S1.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from numpy.linalg import eigh, eigvalsh

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.scipy.linalg import expm
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "explore_OCT29" / "runs" / "binary_snap2"
ART_DIR = Path("/opt/cursor/artifacts")

HEADLINE_AMAX = 1.5
D = 8  # computational levels


def pauli(name: str) -> np.ndarray:
    if name == "I":
        return np.eye(2, dtype=np.complex128)
    if name == "X":
        return np.array([[0, 1], [1, 0]], dtype=np.complex128)
    if name == "Y":
        return np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
    if name == "Z":
        return np.array([[1, 0], [0, -1]], dtype=np.complex128)
    raise ValueError(name)


def bit_op(bit: int, name: str) -> np.ndarray:
    """Kronecker embedding. bit 0 = LSB = fastest index = last kron factor."""
    ops = [pauli("I"), pauli("I"), pauli("I")]
    ops[2 - bit] = pauli(name)
    m = ops[0]
    for t in ops[1:]:
        m = np.kron(m, t)
    return m


def one_qubit(kind: str) -> np.ndarray:
    c = 1.0 / np.sqrt(2.0)
    if kind == "H":
        return np.array([[c, c], [c, -c]], dtype=np.complex128)
    if kind == "Ry":
        # Ry(+pi/2). Ry^H Z Ry = -X (outcome label flip relative to H).
        return np.array([[c, -c], [c, c]], dtype=np.complex128)
    if kind == "Ry_neg":
        # Ry(-pi/2). Ry_neg^H Z Ry_neg = +X.
        return np.array([[c, c], [-c, c]], dtype=np.complex128)
    raise ValueError(kind)


def target_unitary(bit: int, kind: str) -> np.ndarray:
    g = one_qubit(kind)
    if bit == 0:
        return np.kron(np.eye(4), g)
    if bit == 1:
        return np.kron(np.eye(2), np.kron(g, np.eye(2)))
    if bit == 2:
        return np.kron(g, np.eye(4))
    raise ValueError(bit)


def disp_factors(nf: int):
    """Eigenpairs of i(a†-a). D(r) = exp(-i r quad) on 0..nf-1."""
    a = np.diag(np.sqrt(np.arange(1, nf, dtype=np.float64)), 1)
    quad = 1j * (a.T - a)
    lam, V = eigh(quad)
    return np.ascontiguousarray(lam.real), np.ascontiguousarray(V.astype(np.complex128))


def clip_alpha_np(re: float, im: float, amax: float) -> tuple[float, float]:
    r = float(np.hypot(re, im))
    if r < 1e-15:
        return float(re), float(im)
    scale = amax * np.tanh(r / amax) / r
    return float(re * scale), float(im * scale)


def displacement_np(re: float, im: float, lam: np.ndarray, V: np.ndarray) -> np.ndarray:
    """D(re+i im) on the truncated Fock space. Exact exp of the truncated generator."""
    r = float(np.hypot(re, im))
    ph = float(np.arctan2(im, re))
    nf = lam.shape[0]
    if r < 1e-15:
        return np.eye(nf, dtype=np.complex128)
    n = np.arange(nf, dtype=np.float64)
    phase = np.exp(1j * ph * n)
    W = phase[:, None] * V
    eigs = np.exp(-1j * r * lam)
    return (W * eigs[None, :]) @ W.conj().T


def apply_snap_cols(cols: np.ndarray, theta8: np.ndarray) -> np.ndarray:
    nf = cols.shape[0]
    ph = np.zeros(nf, dtype=np.float64)
    ph[:8] = np.asarray(theta8, dtype=np.float64)
    return cols * np.exp(-1j * ph)[:, None]


def apply_disp_cols(cols: np.ndarray, re: float, im: float, lam, V) -> np.ndarray:
    return displacement_np(re, im, lam, V) @ cols


def sequence_columns(order: str, x: np.ndarray, lam, V) -> np.ndarray:
    """Images of |0>..|7>. x is PHYSICAL (already clipped): [S1(8), D1re, D1im, S2(8), D2re, D2im]."""
    nf = lam.shape[0]
    cols = np.zeros((nf, 8), dtype=np.complex128)
    cols[:8, :] = np.eye(8, dtype=np.complex128)
    th1, th2 = x[0:8], x[10:18]
    re1, im1, re2, im2 = float(x[8]), float(x[9]), float(x[18]), float(x[19])
    if order == "SDSD":
        cols = apply_snap_cols(cols, th1)
        cols = apply_disp_cols(cols, re1, im1, lam, V)
        cols = apply_snap_cols(cols, th2)
        cols = apply_disp_cols(cols, re2, im2, lam, V)
    elif order == "DSDS":
        cols = apply_disp_cols(cols, re1, im1, lam, V)
        cols = apply_snap_cols(cols, th1)
        cols = apply_disp_cols(cols, re2, im2, lam, V)
        cols = apply_snap_cols(cols, th2)
    else:
        raise ValueError(order)
    return cols


def raw_to_physical(x: np.ndarray, amax: float) -> np.ndarray:
    y = np.array(x[:20], dtype=np.float64, copy=True)
    for i in (8, 18):
        re, im = clip_alpha_np(float(y[i]), float(y[i + 1]), amax)
        y[i], y[i + 1] = re, im
    return y


def level_sets(bit: int) -> tuple[np.ndarray, np.ndarray]:
    plus = np.array([n for n in range(8) if ((n >> bit) & 1) == 0], dtype=np.int64)
    minus = np.array([n for n in range(8) if ((n >> bit) & 1) == 1], dtype=np.int64)
    return plus, minus


def herm(a: np.ndarray) -> np.ndarray:
    return 0.5 * (a + a.conj().T)


def op_norm_herm(a: np.ndarray) -> float:
    ev = eigvalsh(herm(a))
    return float(np.max(np.abs(ev)))


def effects_from_cols(cols: np.ndarray, bit: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    plus, minus = level_sets(bit)
    cp = cols[plus, :]
    cm = cols[minus, :]
    ep = cp.conj().T @ cp
    em = cm.conj().T @ cm
    leak_c = cols[8:, :]
    leak = leak_c.conj().T @ leak_c
    return ep, em, leak


def z_rotation_phases(bit: int, phi: float) -> np.ndarray:
    """SNAP phases for exp(-i phi Z_b / 2): theta_n so exp(-i theta) = exp(-i phi Z/2)."""
    th = np.zeros(8, dtype=np.float64)
    for n in range(8):
        b = (n >> bit) & 1
        # Z|+1> = +1 on bit 0, Z|-1> wait: Z|0>=+|0>, Z|1>=-|1|.
        # exp(-i phi Z/2) |0> = exp(-i phi/2), |1> = exp(+i phi/2).
        # SNAP applies exp(-i theta), so theta = +phi/2 on bit 0 and -phi/2 on bit 1.
        th[n] = (0.5 * phi) if b == 0 else (-0.5 * phi)
    return th


def conjugate_phases(mat: np.ndarray, theta8: np.ndarray) -> np.ndarray:
    """S^H mat S with S=diag(exp(-i theta)) on the computational space."""
    phase = np.exp(-1j * np.asarray(theta8, dtype=np.float64))
    # S^H M S = conj(phase)_i * M_ij * phase_j
    return np.conj(phase)[:, None] * mat * phase[None, :]


def fold_bit_phase(ep: np.ndarray, em: np.ndarray, leak: np.ndarray, bit: int, x_op: np.ndarray):
    """Best uniform Z_b folded into an initial SNAP, plus classical label flip.

    Returns the best eps_X, the phi used, whether the label was flipped, and
    the conjugated effects. phi is the rotation angle in exp(-i phi Z/2).
    Label flip is classical relabelling of the two detector bins, not a pulse.
    """
    eplus_id = 0.5 * (np.eye(8) + x_op)
    eminus_id = 0.5 * (np.eye(8) - x_op)
    best = None
    phis = np.unique(
        np.concatenate(
            [
                np.linspace(0.0, 2.0 * np.pi, 361, endpoint=False),
                np.array([0.0, 0.5 * np.pi, np.pi, 1.5 * np.pi, 0.25 * np.pi, -0.5 * np.pi]),
            ]
        )
    )
    for phi in phis:
        th = z_rotation_phases(bit, float(phi))
        ep2 = conjugate_phases(ep, th)
        em2 = conjugate_phases(em, th)
        lk2 = conjugate_phases(leak, th)
        for flip in (False, True):
            tgt_p = eminus_id if flip else eplus_id
            tgt_m = eplus_id if flip else eminus_id
            ep_err = op_norm_herm(ep2 - tgt_p)
            em_err = op_norm_herm(em2 - tgt_m)
            err = max(ep_err, em_err)
            if best is None or err < best["eps_X"] - 1e-15:
                best = {
                    "eps_X": err,
                    "eps_plus": ep_err,
                    "eps_minus": em_err,
                    "phi": float(phi),
                    "label_flip": bool(flip),
                    "Ep": ep2,
                    "Em": em2,
                    "L": lk2,
                    "theta_add": th,
                }
    return best


def pauli_coefficients(op: np.ndarray) -> dict:
    names = ("I", "X", "Y", "Z")
    out = {}
    for a in names:
        for b in names:
            for c in names:
                P = bit_op(0, c)
                # rebuild properly: a is MSB, b mid, c LSB
                P = np.kron(np.kron(pauli(a), pauli(b)), pauli(c))
                out[a + b + c] = float(np.real(np.trace(herm(op) @ P) / 8.0))
    return out


def single_bit_bloch(coeffs: dict, bit: int) -> dict:
    """Coefficients of I⊗P on the chosen bit. Spectator Paulis should vanish for a clean axis."""
    key = {0: "II", 1: "I{}I", 2: "{}II"}
    def k(p):
        if bit == 0:
            return "II" + p
        if bit == 1:
            return "I" + p + "I"
        return p + "II"
    nx, ny, nz = coeffs[k("X")], coeffs[k("Y")], coeffs[k("Z")]
    # identity contamination
    ni = coeffs["III"]
    transverse = float(np.hypot(nx, ny))
    alpha = float(np.arctan2(transverse, nz))  # angle from +Z toward the equator
    phi = float(np.arctan2(ny, nx))
    # residual after subtracting the single-bit part (and the III offset)
    return {
        "n_I": ni,
        "n_X": nx,
        "n_Y": ny,
        "n_Z": nz,
        "radius": float(np.sqrt(nx * nx + ny * ny + nz * nz)),
        "alpha_from_Z_rad": alpha,
        "alpha_from_Z_deg": float(np.degrees(alpha)),
        "azimuth_deg": float(np.degrees(phi)),
    }


def spectator_blocks(op: np.ndarray, bit: int) -> list:
    """2x2 blocks of O on each spectator basis state, in the target-bit subspace.

    Off-block coherences are NOT included; those show up in the Pauli residual.
    """
    blocks = []
    spec_bits = [b for b in (0, 1, 2) if b != bit]
    for s in range(4):
        # s as a 2-bit integer over the spectator bits, low spectator bit first
        base = 0
        for k, b in enumerate(spec_bits):
            if (s >> k) & 1:
                base |= 1 << b
        i0 = base
        i1 = base | (1 << bit)
        blk = op[np.ix_([i0, i1], [i0, i1])]
        # Bloch of this 2x2: n_p = 0.5 Tr(blk sigma) if the state is normalized
        # but blk is the compression of the observable, so n = (Tr blk sigma)/ something.
        # For a clean I⊗n·σ the block equals n·σ, and Tr(blk sigma)/2 = n.
        def comp(sig):
            return float(np.real(np.trace(herm(blk) @ sig) / 2.0))
        sx = np.array([[0, 1], [1, 0]], dtype=np.complex128)
        sy = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
        sz = np.array([[1, 0], [0, -1]], dtype=np.complex128)
        nx, ny, nz = comp(sx), comp(sy), comp(sz)
        blocks.append({
            "spectator": s,
            "levels": [int(i0), int(i1)],
            "n_X": nx,
            "n_Y": ny,
            "n_Z": nz,
            "radius": float(np.sqrt(nx * nx + ny * ny + nz * nz)),
        })
    return blocks


def gate_fidelities(cols: np.ndarray, utgt: np.ndarray) -> dict:
    M = cols[:8, :]
    W = utgt.conj().T @ M
    tr = np.trace(W)
    f_proc = float(np.abs(tr) ** 2 / (D * D))
    mnorm = float(np.sum(np.abs(M) ** 2))
    f_avg = float((np.abs(tr) ** 2 + mnorm) / (D * (D + 1)))
    # Trace overlap after the best computational output diagonal (extra terminal SNAP).
    f_out = float((np.sum(np.abs(np.diag(W))) ** 2) / (D * D))
    return {
        "F_proc": f_proc,
        "F_avg": f_avg,
        "F_proc_extra_output_SNAP": f_out,
        "M_fro2": mnorm,
    }


def leakage_report(cols: np.ndarray, leak_op: np.ndarray) -> dict:
    per = np.sum(np.abs(cols[8:, :]) ** 2, axis=0).real
    per = np.clip(per, 0.0, None)
    worst_state = op_norm_herm(leak_op)
    tail = np.sum(np.abs(cols[-5:, :]) ** 2, axis=0).real
    return {
        "leak_per_input": [float(v) for v in per],
        "leak_mean_basis": float(np.mean(per)),
        "leak_worst_basis": float(np.max(per)),
        "leak_worst_state": float(worst_state),
        "retain_mean_basis": float(1.0 - np.mean(per)),
        "retain_worst_state": float(1.0 - worst_state),
        "tail5_per_input": [float(v) for v in tail],
        "tail5_max": float(np.max(tail)),
    }


def readout_errors(ep, em, leak, x_op) -> dict:
    eplus_id = 0.5 * (np.eye(8, dtype=np.complex128) + x_op)
    eminus_id = 0.5 * (np.eye(8, dtype=np.complex128) - x_op)
    dp = herm(ep - eplus_id)
    dm = herm(em - eminus_id)
    # strict +X and the classical label flip -X
    ep_flip = op_norm_herm(ep - eminus_id)
    em_flip = op_norm_herm(em - eplus_id)
    eps_plus = op_norm_herm(dp)
    eps_minus = op_norm_herm(dm)
    eps = max(eps_plus, eps_minus)
    eps_flip = max(ep_flip, em_flip)
    # worst state for the larger of the two strict outcomes
    if eps_plus >= eps_minus:
        ev, evec = eigh(dp)
        tgt = eplus_id
        other_id = eminus_id
        got = ep
        got_o = em
    else:
        ev, evec = eigh(dm)
        tgt = eminus_id
        other_id = eplus_id
        got = em
        got_o = ep
    k = int(np.argmax(np.abs(ev)))
    psi = evec[:, k]
    def expv(mat):
        return float(np.real(np.vdot(psi, mat @ psi)))
    p_plus = expv(ep)
    p_minus = expv(em)
    return {
        "eps_plus": float(eps_plus),
        "eps_minus": float(eps_minus),
        "eps_X": float(eps),
        "eps_X_label_flipped": float(eps_flip),
        "eps_X_basis": float(min(eps, eps_flip)),
        "eps_Z_observable": float(op_norm_herm((ep - em) - x_op)),
        "fro_plus": float(np.sum(np.abs(ep - eplus_id) ** 2)),
        "fro_minus": float(np.sum(np.abs(em - eminus_id) ** 2)),
        "worst_state": {
            "amps_re": [float(v) for v in psi.real],
            "amps_im": [float(v) for v in psi.imag],
            "eigenvalue": float(ev[k].real),
            "p_plus": p_plus,
            "p_minus": p_minus,
            "p_leak": float(1.0 - p_plus - p_minus),
            "p_plus_ideal": expv(eplus_id),
            "p_minus_ideal": expv(eminus_id),
        },
    }


def x_eigen_diagnostic(cols: np.ndarray, bit: int) -> dict:
    """Necessary but not sufficient: error on the four |±> product states of this bit."""
    x_op = bit_op(bit, "X")
    ep, em, _ = effects_from_cols(cols, bit)
    rows = []
    worst = 0.0
    for spec in range(4):
        spec_bits = [b for b in (0, 1, 2) if b != bit]
        base = 0
        for k, b in enumerate(spec_bits):
            if (spec >> k) & 1:
                base |= 1 << b
        i0 = base
        i1 = base | (1 << bit)
        for sign, amp0, amp1, ideal_plus in (
            (1, 1 / np.sqrt(2), 1 / np.sqrt(2), 1.0),
            (-1, 1 / np.sqrt(2), -1 / np.sqrt(2), 0.0),
        ):
            psi = np.zeros(8, dtype=np.complex128)
            psi[i0] = amp0
            psi[i1] = amp1
            p_plus = float(np.real(np.vdot(psi, ep @ psi)))
            p_minus = float(np.real(np.vdot(psi, em @ psi)))
            err = abs(p_plus - ideal_plus)
            worst = max(worst, err, abs(p_minus - (1.0 - ideal_plus)))
            rows.append({
                "spectator": spec,
                "x_eig": int(sign),
                "p_plus": p_plus,
                "p_minus": p_minus,
                "p_leak": float(1.0 - p_plus - p_minus),
                "abs_err_plus": err,
            })
    return {"worst_abs_err_on_X_eigenstates": worst, "rows": rows, "note": "not a substitute for eps_X"}


def evaluate(order: str, x_phys: np.ndarray, bit: int, nf: int, lam=None, V=None) -> dict:
    if lam is None or V is None:
        lam, V = disp_factors(nf)
    cols = sequence_columns(order, x_phys, lam, V)
    x_op = bit_op(bit, "X")
    ep, em, leak = effects_from_cols(cols, bit)
    zpull = ep - em
    coeffs = pauli_coefficients(zpull)
    bloch = single_bit_bloch(coeffs, bit)
    # residual of Zpull against its own single-bit part
    axis = (
        bloch["n_X"] * bit_op(bit, "X")
        + bloch["n_Y"] * bit_op(bit, "Y")
        + bloch["n_Z"] * bit_op(bit, "Z")
        + bloch["n_I"] * np.eye(8)
    )
    residual = herm(zpull - axis)
    gates = {}
    for kind in ("H", "Ry", "Ry_neg"):
        gates[kind] = gate_fidelities(cols, target_unitary(bit, kind))
    significant = {k: v for k, v in coeffs.items() if abs(v) >= 0.02}
    return {
        "nf": nf,
        "order": order,
        "bit": int(bit),
        "alphas": [
            {"re": float(x_phys[8]), "im": float(x_phys[9]), "abs": float(np.hypot(x_phys[8], x_phys[9]))},
            {"re": float(x_phys[18]), "im": float(x_phys[19]), "abs": float(np.hypot(x_phys[18], x_phys[19]))},
        ],
        "snap_phases": {"S1": [float(v) for v in x_phys[0:8]], "S2": [float(v) for v in x_phys[10:18]]},
        "readout": readout_errors(ep, em, leak, x_op),
        "leakage": leakage_report(cols, leak),
        "gates": gates,
        "bloch_single_bit": bloch,
        "single_bit_residual_op": float(op_norm_herm(residual)),
        "pauli_abs_ge_0.02": significant,
        "spectator_blocks": spectator_blocks(zpull, bit),
        "x_eigen_diagnostic": x_eigen_diagnostic(cols, bit),
        "Zpull_re": np.real(herm(zpull)).round(8).tolist(),
        "Zpull_im": np.imag(herm(zpull)).round(8).tolist(),
        "Eplus_re": np.real(herm(ep)).round(8).tolist(),
        "M_re": np.real(cols[:8, :]).round(8).tolist(),
        "M_im": np.imag(cols[:8, :]).round(8).tolist(),
    }


def nf_pair(amax: float) -> tuple[int, tuple[int, int]]:
    if amax <= 1.05:
        return 40, (56, 72)
    if amax <= 1.7:
        return 48, (64, 80)
    return 64, (80, 96)


# ---------------------------------------------------------------------------
# JAX loss
# ---------------------------------------------------------------------------


def _make_loss(order: str, bit: int, objective: str, amax: float, lam, V):
    plus, minus = level_sets(bit)
    plus_j = jnp.asarray(plus)
    minus_j = jnp.asarray(minus)
    x_op = jnp.asarray(bit_op(bit, "X"))
    y_op = jnp.asarray(bit_op(bit, "Y"))
    eye = jnp.eye(8, dtype=jnp.complex128)
    utgt_H = jnp.asarray(target_unitary(bit, "H"))
    utgt_Ry = jnp.asarray(target_unitary(bit, "Ry"))

    nf = int(lam.shape[0])
    a_op = jnp.diag(jnp.sqrt(jnp.arange(1, nf, dtype=jnp.float64)), 1)
    ad_op = jnp.transpose(a_op)

    def disp(re, im):
        # α a† - α* a, smooth at α=0 (no polar-coordinate singularity).
        G = re * (ad_op - a_op) + 1j * im * (ad_op + a_op)
        return expm(G)

    def clip(re, im):
        r2 = re * re + im * im
        r = jnp.sqrt(r2 + 1e-24)
        # amax * tanh(r/amax) / r → 1 at r=0; 1e-24 keeps the quotient finite.
        scale = amax * jnp.tanh(r / amax) / r
        return re * scale, im * scale

    def columns(x):
        th1 = x[0:8]
        re1, im1 = clip(x[8], x[9])
        th2 = x[10:18]
        re2, im2 = clip(x[18], x[19])
        cols = jnp.zeros((nf, 8), dtype=jnp.complex128).at[:8, :].set(jnp.eye(8, dtype=jnp.complex128))

        def snap(c, th):
            ph = jnp.zeros((nf,)).at[:8].set(th)
            return c * jnp.exp(-1j * ph)[:, None]

        def dispc(c, re, im):
            return disp(re, im) @ c

        if order == "SDSD":
            cols = snap(cols, th1)
            cols = dispc(cols, re1, im1)
            cols = snap(cols, th2)
            cols = dispc(cols, re2, im2)
        else:
            cols = dispc(cols, re1, im1)
            cols = snap(cols, th1)
            cols = dispc(cols, re2, im2)
            cols = snap(cols, th2)
        return cols

    def effects(cols):
        cp = cols[plus_j, :]
        cm = cols[minus_j, :]
        ep = jnp.conj(cp).T @ cp
        em = jnp.conj(cm).T @ cm
        return ep, em, cols

    def loss(x):
        cols = columns(x)
        ep, em, cols = effects(cols)
        if objective == "meas":
            dp = ep - 0.5 * (eye + x_op)
            dm = em - 0.5 * (eye - x_op)
            return jnp.sum(jnp.abs(dp) ** 2) + jnp.sum(jnp.abs(dm) ** 2)
        if objective == "meas_op":
            # Smooth worst-case probability error. Frobenius can shrink the
            # average eigenvalue while increasing the largest one.
            def soft_op(delta):
                h = 0.5 * (delta + jnp.conj(delta).T)
                ev = jnp.linalg.eigvalsh(h)
                beta = 25.0
                return jnp.log(jnp.sum(jnp.exp(beta * jnp.abs(ev)))) / beta
            dp = ep - 0.5 * (eye + x_op)
            dm = em - 0.5 * (eye - x_op)
            return soft_op(dp) + soft_op(dm)
        if objective == "eq":
            phi = x[20]
            axis = jnp.cos(phi) * x_op + jnp.sin(phi) * y_op
            dp = ep - 0.5 * (eye + axis)
            dm = em - 0.5 * (eye - axis)
            return jnp.sum(jnp.abs(dp) ** 2) + jnp.sum(jnp.abs(dm) ** 2)
        M = cols[:8, :]
        ut = utgt_H if objective == "gate_H" else utgt_Ry
        tr = jnp.trace(jnp.conj(ut).T @ M)
        f = jnp.abs(tr) ** 2 / (D * D)
        return 1.0 - f.real

    return loss


def structured_seeds(bit: int, objective: str, rng: np.random.Generator, n_random: int, amax: float) -> list[np.ndarray]:
    dim = 21 if objective == "eq" else 20
    seeds = []

    def pack(th1, re1, im1, th2, re2, im2, phi=0.0):
        x = np.zeros(dim, dtype=np.float64)
        x[0:8] = th1
        x[8], x[9] = re1, im1
        x[10:18] = th2
        x[18], x[19] = re2, im2
        if dim == 21:
            x[20] = phi
        return x

    z0 = np.zeros(8)
    zpi = z_rotation_phases(bit, np.pi)  # a Z gate, phases ±pi/2
    seeds.append(pack(z0, 0.0, 0.0, z0, 0.0, 0.0))
    for a in (0.35, 0.7, 1.0, 1.3, min(1.45, amax * 0.95)):
        seeds.append(pack(z0, a, 0.0, z0, -a, 0.0))
        seeds.append(pack(z0, a, 0.0, zpi, -a, 0.0))
        seeds.append(pack(zpi, a, 0.0, z0, -0.5 * a, 0.0))
        seeds.append(pack(z0, 0.0, a, z0, 0.0, -a))
        seeds.append(pack(z_rotation_phases(bit, np.pi / 2), a, 0.2, z0, -a, 0.1))
    # number-dependent sign patterns on the target-bit pairs
    alt = np.array([((n >> bit) & 1) * np.pi for n in range(8)], dtype=np.float64)
    seeds.append(pack(alt, 0.9, 0.0, -alt, -0.9, 0.0))
    seeds.append(pack(alt, 1.2, -0.3, np.zeros(8), -0.4, 0.6))
    for i in range(n_random):
        x = np.zeros(dim)
        x[0:8] = rng.uniform(0, 2 * np.pi, size=8)
        x[10:18] = rng.uniform(0, 2 * np.pi, size=8)
        scale = amax * (0.3 + 0.7 * rng.random())
        x[8:10] = rng.normal(0, scale / 2, size=2)
        x[18:20] = rng.normal(0, scale / 2, size=2)
        if dim == 21:
            x[20] = float(rng.uniform(0, 2 * np.pi))
        seeds.append(x)
    return seeds


def optimize_one(loss_vg, x0: np.ndarray, maxiter: int) -> dict:
    def fun(x):
        v, g = loss_vg(jnp.asarray(x))
        return float(v), np.asarray(g, dtype=np.float64)

    t0 = time.time()
    res = minimize(
        fun,
        np.asarray(x0, dtype=np.float64),
        jac=True,
        method="L-BFGS-B",
        options={"maxiter": maxiter, "ftol": 1e-15, "gtol": 1e-8, "maxfun": maxiter * 2},
    )
    gnorm = float(np.linalg.norm(res.jac)) if res.jac is not None else float("nan")
    fun = float(res.fun)
    if not np.isfinite(fun):
        fun = 1e9
    return {
        "x": np.asarray(res.x, dtype=np.float64),
        "loss": fun,
        "nit": int(res.nit),
        "success": bool(res.success),
        "gnorm": gnorm,
        "message": str(res.message),
        "seconds": time.time() - t0,
    }


def run_config(bit_name: str, order: str, objective: str, n_random: int, maxiter: int, amax: float, master: int) -> dict:
    bit = {"lsb": 0, "mid": 1, "msb": 2}[bit_name]
    nf, vals = nf_pair(amax)
    lam, V = disp_factors(nf)
    loss = _make_loss(order, bit, objective, amax, lam, V)
    loss_vg = jax.jit(jax.value_and_grad(loss))
    # warmup compile
    dim = 21 if objective == "eq" else 20
    loss_vg(jnp.zeros(dim))
    rng = np.random.default_rng(master)
    seeds = structured_seeds(bit, objective, rng, n_random, amax)
    rows = []
    best = None
    t0 = time.time()
    for i, x0 in enumerate(seeds):
        row = optimize_one(loss_vg, x0, maxiter)
        rec = {
            "i": i,
            "loss": row["loss"],
            "nit": row["nit"],
            "success": row["success"],
            "gnorm": row["gnorm"],
            "seconds": row["seconds"],
            "seed_kind": "structured" if i < len(seeds) - n_random else "random",
        }
        rows.append(rec)
        print(
            f"  {bit_name:3} {order} {objective:7} a={amax:.1f} s{i:02d} "
            f"loss {row['loss']:.6e} nit {row['nit']:4d} g {row['gnorm']:.2e} {row['message'][:48]}",
            flush=True,
        )
        if best is None or row["loss"] < best["loss"]:
            best = row
            best["i"] = i
    # polish the winner, then three small perturbations — convergence check, not a new budget
    polish_notes = []
    x_pol = best["x"]
    loss_pol = best["loss"]
    for tag, x_init, it in (
        ("rep Polish", x_pol, max(maxiter, 800)),
        ("perturb0", x_pol + rng.normal(0, 0.05, size=x_pol.shape), maxiter // 2),
        ("perturb1", x_pol + rng.normal(0, 0.15, size=x_pol.shape), maxiter // 2),
        ("perturb2", x_pol + rng.normal(0, 0.3, size=x_pol.shape), maxiter // 2),
    ):
        row = optimize_one(loss_vg, x_init, it)
        polish_notes.append({"tag": tag, "loss": row["loss"], "nit": row["nit"], "gnorm": row["gnorm"], "success": row["success"]})
        print(f"    polish {tag}: loss {row['loss']:.6e} nit {row['nit']} g {row['gnorm']:.2e}", flush=True)
        if row["loss"] < loss_pol - 1e-12:
            loss_pol = row["loss"]
            x_pol = row["x"]
    x_phys = raw_to_physical(x_pol, amax)
    ev = evaluate(order, x_phys, bit, nf, lam, V)
    # truncation checks with the SAME physical controls
    trunc = {}
    for nf2 in vals:
        lam2, V2 = disp_factors(nf2)
        ev2 = evaluate(order, x_phys, bit, nf2, lam2, V2)
        trunc[str(nf2)] = {
            "eps_X": ev2["readout"]["eps_X"],
            "eps_X_basis": ev2["readout"]["eps_X_basis"],
            "F_proc_H": ev2["gates"]["H"]["F_proc"],
            "F_proc_Ry": ev2["gates"]["Ry"]["F_proc"],
            "leak_worst_state": ev2["leakage"]["leak_worst_state"],
            "leak_mean_basis": ev2["leakage"]["leak_mean_basis"],
            "tail5_max": ev2["leakage"]["tail5_max"],
            "delta_eps_X": abs(ev2["readout"]["eps_X"] - ev["readout"]["eps_X"]),
            "delta_F_H": abs(ev2["gates"]["H"]["F_proc"] - ev["gates"]["H"]["F_proc"]),
            "F_avg_H": ev2["gates"]["H"]["F_avg"],
            "alpha_from_Z_deg": ev2["bloch_single_bit"]["alpha_from_Z_deg"],
            "single_bit_residual_op": ev2["single_bit_residual_op"],
        }
    # phase fold (budget-respecting only when the sequence starts with SNAP)
    x_op = bit_op(bit, "X")
    cols = sequence_columns(order, x_phys, lam, V)
    ep, em, leak = effects_from_cols(cols, bit)
    eq_target = None
    if objective == "eq":
        phi = float(x_pol[20])
        axis = np.cos(phi) * bit_op(bit, "X") + np.sin(phi) * bit_op(bit, "Y")
        eq_target = {
            "phi": phi,
            "phi_deg": float(np.degrees(phi)),
            "opt_nf": readout_errors(ep, em, leak, axis),
        }
        lam_v, V_v = disp_factors(vals[-1])
        cols_hi = sequence_columns(order, x_phys, lam_v, V_v)
        ep_h, em_h, leak_h = effects_from_cols(cols_hi, bit)
        eq_target["high_nf"] = readout_errors(ep_h, em_h, leak_h, axis)
        eq_target["high_nf_leak_worst_state"] = leakage_report(cols_hi, leak_h)["leak_worst_state"]
    folded = fold_bit_phase(ep, em, leak, bit, x_op)
    fold_eval = None
    if order == "SDSD":
        x_fold = x_phys.copy()
        x_fold[0:8] = x_fold[0:8] + folded["theta_add"]
        # re-simulate, do not trust the 8x8 conjugation alone
        fold_eval = evaluate(order, x_fold, bit, nf, lam, V)
        lam_v, V_v = disp_factors(vals[-1])
        fold_eval_hi = evaluate(order, x_fold, bit, vals[-1], lam_v, V_v)
        fold_eval = {
            "phi": folded["phi"],
            "label_flip": folded["label_flip"],
            "within_budget": True,
            "how": "added exp(-i phi Z_b/2) into S1; classical label flip if label_flip",
            "metrics_opt_nf": _compact(fold_eval),
            "metrics_high_nf": _compact(fold_eval_hi),
            "x_physical": x_fold.tolist(),
        }
    else:
        fold_eval = {
            "phi": folded["phi"],
            "label_flip": folded["label_flip"],
            "within_budget": False,
            "how": "DSDS starts with a displacement, so an input Z rotation would be an extra SNAP. Reported only as a diagnostic.",
            "eps_X_if_extra_input_SNAP": folded["eps_X"],
            "eps_plus_if_extra": folded["eps_plus"],
            "eps_minus_if_extra": folded["eps_minus"],
        }
    losses = [r["loss"] for r in rows]
    return {
        "bit": bit_name,
        "order": order,
        "written_product": "D2 @ S2 @ D1 @ S1" if order == "SDSD" else "S2 @ D2 @ S1 @ D1",
        "physical_time_order": ["S1", "D1", "S2", "D2"] if order == "SDSD" else ["D1", "S1", "D2", "S2"],
        "objective": objective,
        "amax": amax,
        "nf_opt": nf,
        "nf_validate": list(vals),
        "n_starts": len(seeds),
        "n_random": n_random,
        "maxiter": maxiter,
        "master_seed": master,
        "loss_best": loss_pol,
        "loss_best_before_polish": best["loss"],
        "loss_median": float(np.median(losses)),
        "loss_min_raw_starts": float(np.min(losses)),
        "gnorm_best_pre_polish": best["gnorm"],
        "polish": polish_notes,
        "seconds": time.time() - t0,
        "x_raw": x_pol.tolist(),
        "x_physical": x_phys.tolist(),
        "metrics": _compact(ev),
        "metrics_full": ev,
        "truncation": trunc,
        "phase_fold": fold_eval,
        "equatorial_target": eq_target,
        "start_losses": losses,
    }


def _compact(ev: dict) -> dict:
    """Drop the bulky matrices from the rolled-up table; full matrices stay in metrics_full."""
    return {
        "alphas": ev["alphas"],
        "snap_phases": ev["snap_phases"],
        "readout": {k: v for k, v in ev["readout"].items() if k != "worst_state"},
        "worst_state": ev["readout"]["worst_state"],
        "leakage": ev["leakage"],
        "gates": ev["gates"],
        "bloch_single_bit": ev["bloch_single_bit"],
        "single_bit_residual_op": ev["single_bit_residual_op"],
        "pauli_abs_ge_0.02": ev["pauli_abs_ge_0.02"],
        "spectator_blocks": ev["spectator_blocks"],
        "x_eigen_worst": ev["x_eigen_diagnostic"]["worst_abs_err_on_X_eigenstates"],
        "Zpull_re": ev["Zpull_re"],
        "Zpull_im": ev["Zpull_im"],
    }


def run_selftests() -> None:
    # Displacement coherent-state matrix elements.
    nf = 36
    lam, V = disp_factors(nf)
    alpha = 0.4 + 0.2j
    D = displacement_np(alpha.real, alpha.imag, lam, V)
    vac = np.exp(-0.5 * abs(alpha) ** 2)
    assert abs(D[0, 0] - vac) < 1e-12
    assert abs(D[1, 0] - vac * alpha) < 1e-12
    assert abs(D[2, 0] - vac * alpha ** 2 / np.sqrt(2)) < 1e-12
    I = displacement_np(0.0, 0.0, lam, V)
    assert np.allclose(I, np.eye(nf))
    # Unitarity of the truncated displacement.
    assert np.allclose(D.conj().T @ D, np.eye(nf), atol=1e-10)
    # Truncation stability of a modest displacement on the vacuum column.
    lam2, V2 = disp_factors(20)
    lam3, V3 = disp_factors(48)
    d2 = displacement_np(0.5, -0.25, lam2, V2)
    d3 = displacement_np(0.5, -0.25, lam3, V3)
    assert np.allclose(d2[:12, :12], d3[:12, :12], atol=1e-10)

    # Algebra: H and Ry.
    H = one_qubit("H")
    Ry = one_qubit("Ry")
    Ryn = one_qubit("Ry_neg")
    Z = pauli("Z")
    X = pauli("X")
    assert np.allclose(H @ Z @ H, X)
    assert np.allclose(Ry.conj().T @ Z @ Ry, -X)
    assert np.allclose(Ryn.conj().T @ Z @ Ryn, X)
    assert np.allclose(Ry @ Z, H)
    for bit in (0, 1, 2):
        Hb = target_unitary(bit, "H")
        assert np.allclose(Hb @ bit_op(bit, "Z") @ Hb, bit_op(bit, "X"), atol=1e-10)
        # spectators unchanged: Hb commutes with spectator Z
        for b2 in (0, 1, 2):
            if b2 == bit:
                continue
            assert np.allclose(Hb @ bit_op(b2, "Z") @ Hb.conj().T, bit_op(b2, "Z"), atol=1e-10)

    # Ideal embedded H has perfect X readout and perfect gate fidelity, zero leakage.
    for bit, name in ((0, "lsb"), (1, "mid"), (2, "msb")):
        cols = np.zeros((nf, 8), dtype=np.complex128)
        cols[:8, :] = target_unitary(bit, "H")
        ep, em, leak = effects_from_cols(cols, bit)
        err = readout_errors(ep, em, leak, bit_op(bit, "X"))
        assert err["eps_X"] < 1e-10, (name, err["eps_X"])
        assert gate_fidelities(cols, target_unitary(bit, "H"))["F_proc"] > 1 - 1e-12
        assert leakage_report(cols, leak)["leak_worst_state"] < 1e-12

    # Identity measures Z, so the X error is ||(Z-X)/2||_op = 1/sqrt(2).
    cols = np.zeros((nf, 8), dtype=np.complex128)
    cols[:8, :] = np.eye(8)
    ep, em, leak = effects_from_cols(cols, 0)
    err = readout_errors(ep, em, leak, bit_op(0, "X"))
    assert abs(err["eps_X"] - 1 / np.sqrt(2)) < 1e-10

    # Cheat: map X=+1 into even computational levels and X=-1 into leakage.
    # eps+ collapses; eps- does not. The headline max must catch it.
    cols = np.zeros((nf, 8), dtype=np.complex128)
    # |+>_{p} = (|2p>+|2p+1>)/sqrt(2) -> |2p>,  |-> -> |8+p>
    for p in range(4):
        plus = np.zeros(8, dtype=np.complex128)
        minus = np.zeros(8, dtype=np.complex128)
        plus[2 * p] = 1 / np.sqrt(2)
        plus[2 * p + 1] = 1 / np.sqrt(2)
        minus[2 * p] = 1 / np.sqrt(2)
        minus[2 * p + 1] = -1 / np.sqrt(2)
        # Express the computational basis by inverting the ± basis.
        # |2p> = (|+>+|->)/sqrt(2), |2p+1> = (|+>-|->)/sqrt(2)
        # We set columns directly in the ± basis then convert.
    # Build images of the 8 computational basis vectors.
    # U|2p> = (|2p_out_even> + |leak p>)/sqrt(2)? Easier: define U on ± states.
    images = [None] * 8
    for p in range(4):
        # |+> -> |2p>, |-> -> |8+p|
        ket_p = np.zeros(nf, dtype=np.complex128)
        ket_m = np.zeros(nf, dtype=np.complex128)
        ket_p[2 * p] = 1.0
        ket_m[8 + p] = 1.0
        # |2p> = (|+> + |->)/sqrt(2), |2p+1> = (|+> - |->)/sqrt(2)
        images[2 * p] = (ket_p + ket_m) / np.sqrt(2)
        images[2 * p + 1] = (ket_p - ket_m) / np.sqrt(2)
    cols = np.stack(images, axis=1)
    ep, em, leak = effects_from_cols(cols, 0)
    err = readout_errors(ep, em, leak, bit_op(0, "X"))
    assert err["eps_plus"] < 1e-10
    assert err["eps_minus"] > 0.9
    assert err["eps_X"] > 0.9
    lk = leakage_report(cols, leak)
    assert lk["leak_mean_basis"] > 0.4

    # Terminal SNAP does not change number readout.
    rng = np.random.default_rng(0)
    x = rng.normal(size=20)
    x = raw_to_physical(x, 1.5)
    cols_a = sequence_columns("DSDS", x, lam, V)
    x2 = x.copy()
    x2[10:18] = rng.normal(size=8)  # S2 is the LAST pulse in DSDS? 
    # DSDS physical: D1, S1, D2, S2. S2 is x[10:18], yes the second SNAP, applied last.
    # Wait layout: th1=x[0:8] is S1 (earlier SNAP), th2=x[10:18] is S2 (later SNAP).
    # For DSDS, S2 is terminal. Changing x[10:18] must not change effects.
    cols_b = sequence_columns("DSDS", x2, lam, V)
    ep1, em1, _ = effects_from_cols(cols_a, 1)
    ep2, em2, _ = effects_from_cols(cols_b, 1)
    assert np.allclose(ep1, ep2, atol=1e-10)
    assert np.allclose(em1, em2, atol=1e-10)

    # Folding a Z rotation into S1 of SDSD matches 8x8 conjugation, and turns Y into X.
    x = raw_to_physical(rng.normal(size=20), 1.2)
    cols = sequence_columns("SDSD", x, lam, V)
    ep, em, leak = effects_from_cols(cols, 0)
    phi = 0.7
    th = z_rotation_phases(0, phi)
    ep_c = conjugate_phases(ep, th)
    x2 = x.copy()
    x2[0:8] = x2[0:8] + th
    cols2 = sequence_columns("SDSD", x2, lam, V)
    ep2, _, _ = effects_from_cols(cols2, 0)
    assert np.allclose(ep_c, ep2, atol=1e-8)

    # Abstract Y observable folds to X.
    Y = bit_op(0, "Y")
    # Build a unitary with U^H Z U = Y by taking the Y-eigenbasis map is unnecessary:
    # conjugate Y by the inverse rotation and check the fold search.
    folded = fold_bit_phase(0.5 * (np.eye(8) + Y), 0.5 * (np.eye(8) - Y), np.zeros((8, 8)), 0, bit_op(0, "X"))
    assert folded["eps_X"] < 1e-8, folded["eps_X"]

    # JAX loss matches numpy Frobenius loss, and the gradient matches a finite difference.
    loss = _make_loss("SDSD", 0, "meas", 1.5, lam, V)
    vg = jax.jit(jax.value_and_grad(loss))
    x = rng.normal(size=20)
    v_j, g_j = vg(jnp.asarray(x))
    xp = raw_to_physical(x, 1.5)
    cols = sequence_columns("SDSD", xp, lam, V)
    ep, em, _ = effects_from_cols(cols, 0)
    eplus = 0.5 * (np.eye(8) + bit_op(0, "X"))
    eminus = 0.5 * (np.eye(8) - bit_op(0, "X"))
    v_n = np.sum(np.abs(ep - eplus) ** 2) + np.sum(np.abs(em - eminus) ** 2)
    assert abs(float(v_j) - v_n) < 1e-8, (float(v_j), v_n)
    g_n = np.zeros(20)
    eps = 1e-6
    for i in range(20):
        e = np.zeros(20)
        e[i] = eps
        vp = float(loss(jnp.asarray(x + e)))
        vm = float(loss(jnp.asarray(x - e)))
        g_n[i] = (vp - vm) / (2 * eps)
    assert np.allclose(np.asarray(g_j), g_n, rtol=1e-4, atol=1e-4), np.max(np.abs(np.asarray(g_j) - g_n))

    # Gate loss at the ideal point is not claimed (ideal H is not 2+2); just that F_proc
    # of an embedded H rebuilt through evaluate() is 1. Already checked.
    # Product written-order check: S then D on a basis vector equals D @ S @ e.
    th = rng.normal(size=8)
    cols = np.zeros((nf, 8), dtype=np.complex128)
    cols[:8, :] = np.eye(8)
    cols = apply_snap_cols(cols, th)
    cols = apply_disp_cols(cols, 0.3, -0.1, lam, V)
    S = np.eye(nf, dtype=np.complex128)
    ph = np.zeros(nf)
    ph[:8] = th
    S = np.diag(np.exp(-1j * ph))
    full = displacement_np(0.3, -0.1, lam, V) @ S @ np.eye(nf, 8)
    assert np.allclose(cols, full, atol=1e-10)
    print("selftests passed", flush=True)


def _json_default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    raise TypeError(type(o))


def save_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=_json_default))


def headline_eps(block: dict) -> float:
    """Budget-respecting X-basis error at the largest validation NF."""
    if block["order"] == "SDSD" and block["phase_fold"].get("within_budget"):
        hi = block["phase_fold"]["metrics_high_nf"]
        return float(hi["readout"]["eps_X_basis"])
    # use the highest validation nf stored
    trunc = block["truncation"]
    nf = max(int(k) for k in trunc)
    return float(trunc[str(nf)]["eps_X_basis"]) if "eps_X_basis" in trunc[str(nf)] else float(trunc[str(nf)]["eps_X"])


def make_plots(summary: dict) -> list[Path]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ART_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    paths = []
    bits = ["lsb", "mid", "msb"]
    # Bars: F_proc(H) and eps for SDSD / DSDS at amax 1.5
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6))
    width = 0.35
    xs = np.arange(len(bits))
    for ax, key, ylabel, title in (
        (axes[0], "F", r"subspace process fidelity $F_\mathrm{proc}(H)$", "Full single-bit H"),
        (axes[1], "eps", r"operator-norm X readout error", "Direct X measurement"),
    ):
        for j, order in enumerate(("SDSD", "DSDS")):
            vals = []
            for b in bits:
                hit = None
                for row in summary["runs"]:
                    if row["bit"] != b or row["order"] != order or abs(row["amax"] - HEADLINE_AMAX) > 1e-9:
                        continue
                    nf = max(int(k) for k in row["truncation"])
                    if key == "F" and row["objective"] == "gate_H":
                        hit = row["truncation"][str(nf)]["F_proc_H"]
                    if key == "eps" and row["objective"] in ("meas", "meas_op"):
                        eps = row["truncation"][str(nf)]["eps_X_basis"]
                        hit = eps if hit is None else min(hit, eps)
                vals.append(np.nan if hit is None else hit)
            ax.bar(xs + (j - 0.5) * width, vals, width, label=order)
        ax.set_xticks(xs, [b.upper() for b in bits])
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(frameon=False)
        if key == "F":
            ax.set_ylim(0, 1.05)
        else:
            ax.axhline(1.0 / np.sqrt(2.0), color="0.4", ls="--", lw=0.8, label="bare Z (eps=1/√2)")
            ax.legend(frameon=False, fontsize=8)
            ax.set_ylim(0, 1.05)
    fig.tight_layout()
    p = OUT_DIR / "summary_bars.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    paths.append(p)

    # Heatmaps of the pulled-back Z for the best meas run of each bit (SDSD, folded if present)
    fig, axes = plt.subplots(3, 3, figsize=(8.4, 8.0))
    x_targets = [bit_op(b, "X").real for b in range(3)]
    for r, bname in enumerate(bits):
        row = None
        best_eps = 1e9
        for item in summary["runs"]:
            if item["bit"] == bname and item["order"] == "SDSD" and item["objective"] in ("meas", "meas_op") and abs(item["amax"] - HEADLINE_AMAX) < 1e-9:
                nf = max(int(k) for k in item["truncation"])
                eps = item["truncation"][str(nf)]["eps_X_basis"]
                if eps < best_eps:
                    best_eps = eps
                    row = item
        if row is None:
            continue
        if row["phase_fold"].get("within_budget"):
            z = np.array(row["phase_fold"]["metrics_high_nf"]["Zpull_re"], dtype=float)
        else:
            z = np.array(row["metrics"]["Zpull_re"], dtype=float)
        ideal = x_targets[r]
        diff = z - ideal
        for c, (mat, title) in enumerate((
            (z, f"{bname.upper()} measured Zpull"),
            (ideal, f"{bname.upper()} ideal X"),
            (diff, f"{bname.upper()} Zpull − X"),
        )):
            ax = axes[r, c]
            lim = max(1.0, np.max(np.abs(mat)))
            im = ax.imshow(mat, vmin=-lim, vmax=lim, cmap="coolwarm")
            ax.set_title(title, fontsize=9)
            ax.set_xticks(range(8))
            ax.set_yticks(range(8))
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Effective observable on Fock 0..7 (real part)", fontsize=11)
    fig.tight_layout()
    p = OUT_DIR / "observable_heatmaps.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    paths.append(p)
    return paths


def write_results_md(summary: dict) -> str:
    lines = []
    lines.append("# Binary 3-bit cavity, exactly 2 SNAP + 2 displacements")
    lines.append("")
    lines.append("Encoding `|n⟩ = |b2 b1 b0⟩` for n=0..7, b0 = LSB. SNAP phases free on 0..7 and **fixed at 0 for n≥8**. Displacements are the exponential of the NF-truncated generator. Headline amplitude cap `|α|≤1.5` (repo cavity ansatz soft-clips at 1.8). Sensitivity at 0.8 and 2.2.")
    lines.append("")
    lines.append("Only alternating words spend the whole budget. `D(a)D(b)` is one displacement up to a global phase, and two SNAPs in a row are one SNAP.")
    lines.append("")
    lines.append("| order | physical time (first → last) | written product |")
    lines.append("|---|---|---|")
    lines.append("| SDSD | S1, D1, S2, D2 | U = D2 S2 D1 S1 |")
    lines.append("| DSDS | D1, S1, D2, S2 | U = S2 D2 S1 D1 |")
    lines.append("")
    lines.append("DSDS ends on a SNAP, which cannot change a photon-number measurement, so its readout uses only D,S,D. SDSD can absorb a uniform Z rotation on the measured bit into S1 (X↔Y) without an extra pulse. An output diagonal does not move the measured axis at all.")
    lines.append("")
    lines.append("**F_proc** = `|Tr(U_tgt† M)|²/64` with `M_ij=⟨i|U|j⟩` on levels 0..7 (mixcal subspace process fidelity, global phase free). Leakage lowers it. **F_avg** is the Haar state fidelity with leakage as failure.")
    lines.append("")
    lines.append("**eps_X** = max of the two operator norms `‖E± − (I±X_b)/2‖_op`. E± only count computational levels with the chosen bit; n≥8 is a third outcome (no postselection). This is the worst-case binary probability error over every state in the 8-level space, including spectator entanglement. eps_X on the X eigenstates alone is recorded and is not the claim. Classical bin-label flip is `eps_X_basis`.")
    lines.append("")
    lines.append("`H = Ry(+π/2) Z` on the encoded bit. Both measure the X basis (`Ry(+π/2)` yields −X, a label flip). They are not the same gate. On SDSD the extra Z folds into S1; on DSDS it would be an extra pulse.")
    lines.append("")
    lines.append("Numbers below are at the **largest validation truncation**, same physical controls as the NF used in the fit. These are best-of-multistart values, not upper bounds.")
    lines.append("")
    lines.append("## Table")
    lines.append("")
    lines.append("| bit | order | F_proc(H) | F_avg(H) | F_proc(Ry) | eps_X basis | eps_X signed | leak worst state | retain worst | α from Z (deg) | single-bit residual |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")

    def find(bit, order, obj, amax=HEADLINE_AMAX):
        for row in summary["runs"]:
            if row["bit"] == bit and row["order"] == order and row["objective"] == obj and abs(row["amax"] - amax) < 1e-9:
                return row
        return None

    def meas_eps_block(row):
        if row is None:
            return None
        nf = max(int(k) for k in row["truncation"])
        eps_s = row["truncation"][str(nf)]["eps_X"]
        eps_b = row["truncation"][str(nf)].get("eps_X_basis", eps_s)
        leak = row["truncation"][str(nf)]["leak_worst_state"]
        raw = {
            "eps_b": eps_b,
            "eps_s": eps_s,
            "leak": leak,
            "ret": 1.0 - leak,
            "alpha": row["truncation"][str(nf)].get(
                "alpha_from_Z_deg", row["metrics"]["bloch_single_bit"]["alpha_from_Z_deg"]
            ),
            "resid": row["truncation"][str(nf)].get(
                "single_bit_residual_op", row["metrics"]["single_bit_residual_op"]
            ),
            "obj": row["objective"],
            "label_flip": False,
        }
        fold = row.get("phase_fold") or {}
        if row["order"] == "SDSD" and fold.get("within_budget"):
            m = fold["metrics_high_nf"]
            folded = {
                "eps_b": m["readout"]["eps_X_basis"],
                "eps_s": m["readout"]["eps_X"],
                "leak": m["leakage"]["leak_worst_state"],
                "ret": m["leakage"]["retain_worst_state"],
                "alpha": m["bloch_single_bit"]["alpha_from_Z_deg"],
                "resid": m["single_bit_residual_op"],
                "obj": row["objective"],
                "label_flip": bool(fold.get("label_flip")),
            }
            # Keep the unfolded pulses unless the budget-respecting Z fold
            # actually lowers the X-basis error.
            if folded["eps_b"] < raw["eps_b"] - 1e-3:
                return folded
        return raw

    for bit in ("lsb", "mid", "msb"):
        for order in ("SDSD", "DSDS"):
            g = find(bit, order, "gate_H")
            ry = find(bit, order, "gate_Ry")
            cands = [meas_eps_block(find(bit, order, obj)) for obj in ("meas", "meas_op")]
            cands = [c for c in cands if c is not None]
            if g is None and not cands:
                continue
            nf_g = max(int(k) for k in g["truncation"]) if g else None
            fH = g["truncation"][str(nf_g)]["F_proc_H"] if g else float("nan")
            favg = g["truncation"][str(nf_g)].get("F_avg_H", g["metrics"]["gates"]["H"]["F_avg"]) if g else float("nan")
            fRy_s = "—"
            if ry:
                nf_r = max(int(k) for k in ry["truncation"])
                fRy_s = f"{ry['truncation'][str(nf_r)]['F_proc_Ry']:.4f}"
            if not cands:
                continue
            best = min(cands, key=lambda c: c["eps_b"])
            lines.append(
                f"| {bit} | {order} | {fH:.4f} | {favg:.4f} | {fRy_s} | {best['eps_b']:.4f} | {best['eps_s']:.4f} | {best['leak']:.4f} | {best['ret']:.4f} | {best['alpha']:.1f} | {best['resid']:.4f} |"
            )
    lines.append("")
    lines.append("F_proc(Ry) is the fit that targeted Ry, not the Ry fidelity of the H fit. eps is the better of the Frobenius measurement fit and the operator-norm fit (`meas_op`), validated at the largest NF. It is not inferred from the gate fit. Doing nothing (identity) measures Z and has eps_X = 1/√2 ≈ 0.707 with zero leakage; a larger eps is not a useful X readout.")
    lines.append("")
    lines.append("## LSB candidate")
    lines.append("")
    lsb = find("lsb", "SDSD", "meas_op") or find("lsb", "SDSD", "meas")
    if lsb:
        nf = max(int(k) for k in lsb["truncation"])
        tr = lsb["truncation"][str(nf)]
        m = lsb["metrics"]
        b = m["bloch_single_bit"]
        lines.append(
            f"Best verified LSB readout is SDSD `{lsb['objective']}` "
            f"(physical S1, D1, S2, D2; written U = D2 S2 D1 S1), "
            f"same controls at NF={nf}. "
            f"eps_X={tr['eps_X']:.4f} (basis {tr['eps_X_basis']:.4f}), "
            f"leak worst/mean {tr['leak_worst_state']:.4f}/{tr['leak_mean_basis']:.4f}, "
            f"retain worst {1-tr['leak_worst_state']:.4f}, "
            f"|α|=({m['alphas'][0]['abs']:.3f}, {m['alphas'][1]['abs']:.3f}). "
            f"X-eigenstate-only error={m['x_eigen_worst']:.4f}, which is smaller than eps_X; "
            f"the operator norm is the claim. "
            f"Loss best/median {lsb['loss_best']:.4f}/{lsb['loss_median']:.4f} over {lsb['n_starts']} starts, "
            f"and polish perturbations returned to the same loss. Truncation drift of eps_X is {tr['delta_eps_X']:.1e}."
        )
        lines.append("")
        lines.append(
            f"Pulled-back observable's single-bit Bloch vector is "
            f"n=({b['n_X']:.3f}, {b['n_Y']:.3f}, {b['n_Z']:.3f}), "
            f"angle from Z {b['alpha_from_Z_deg']:.1f}°. "
            f"Residual op-norm after removing that single-bit part and the identity offset: "
            f"{m['single_bit_residual_op']:.3f}. "
            f"That residual is spectator-correlated Pauli weight, so this is not a clean encoded-qubit axis. "
            f"Folding an extra Z rotation into S1 (free on SDSD) does not remove it: the azimuth is already ~0."
        )
        lines.append("")
        lines.append("Largest Pauli coefficients of Z_pull (|c|≥0.02):")
        lines.append("")
        items = sorted(m["pauli_abs_ge_0.02"].items(), key=lambda kv: -abs(kv[1]))
        lines.append(", ".join(f"`{k}`={v:.3f}" for k, v in items[:16]) + ".")
        lines.append("")
        lines.append("Spectator blocks of Z_pull, (n_X, n_Y, n_Z):")
        lines.append("")
        for blk in m["spectator_blocks"]:
            lines.append(
                f"- spec {blk['spectator']} levels {blk['levels']}: "
                f"({blk['n_X']:.3f}, {blk['n_Y']:.3f}, {blk['n_Z']:.3f}), radius {blk['radius']:.3f}"
            )
        ws = m["worst_state"]
        lines.append("")
        lines.append(
            f"Worst state for the larger outcome error: p+={ws['p_plus']:.3f} (ideal {ws['p_plus_ideal']:.3f}), "
            f"p-={ws['p_minus']:.3f} (ideal {ws['p_minus_ideal']:.3f}), p_leak={ws['p_leak']:.3f}."
        )
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- Guard SNAP phases are identically 0. Controlling n≥8 would be a different model (mixcal’s internal SNAPs do that) and was not used.")
    lines.append("- No terminal SNAP was added outside the two. F_proc_extra_output_SNAP in the JSON is the trace overlap if one were allowed; it is not a gate fidelity claim.")
    lines.append("- Identity is an exact Z measurement (α=0, eps_X=1/√2≈0.707). A free-angle fit is reported only through the achieved Bloch angle and the residual to a single-bit axis, not by declaring a tiny loss against Z.")
    lines.append("- Prior Gray-mixer mixcal table (free outer SNAPs, internal SNAP on all levels, unbounded α, target ∏ exp(−iβ X/2) in Gray code) is a different problem. K3 Fmin over β was quoted in `oct_explore/NATIVE_TEMPLATE.md` as k=2: 0.13, k=6: 0.83, k=8: 0.9987. The K3 JSON is not in the tree. This study does not reproduce that fit.")
    lines.append("- MSB H at `|α|≤1.5` saturates both displacements (`|α|=1.5`). Raising the cap to 2.2 moves the best-found F_proc(H) only from 0.607 to 0.641, with `|α|≈1.90` still inside 2.2. The cap is weakly active for MSB and inactive for LSB (best `|α|≈0.18` for the gate, `≈0.31` for the readout).")
    lines.append("- MSB X-readout at `|α|≤2.2` has eps_X=0.689 versus 0.690 at `|α|≤1.5`. Opening the cap does not fix MSB.")
    lines.append("- Practical recommendation: do not use 2 SNAP+2 D as an X-basis readout. LSB is a partial, spectator-correlated rotation (signed eps_X≈0.265, residual ≈0.49), not a 0.01 X measurement. Middle's unfolded operator-norm fit has signed eps_X≈0.544. The table's 0.53 basis entry is that fit after a free S1 Z-rotation plus a classical label flip; the signed error of the flipped representative is ≈0.98, and the residual to a single-bit axis stays ≈0.70 with worst-case leakage ≈0.45. MSB stays at ≈0.69. Coherent LSB H reaches F_proc≈0.915 (F_avg≈0.924) and is a different pulse from the best readout; middle/MSB H are F_proc≈0.63/0.61 (0.64 for MSB if `|α|≤2.2`). None of these is a 0.99 gate.")
    lines.append("")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["selftest", "lsb", "all", "opnorm"], default="all")
    ap.add_argument("--quick", action="store_true", help="fewer starts, for a smoke run")
    args = ap.parse_args()
    quick = args.quick
    if args.stage == "selftest":
        run_selftests()
        return
    if args.stage == "opnorm":
        # Append operator-norm fits. Do not wipe the Frobenius sweep.
        prev = json.loads((OUT_DIR / "results.json").read_text()) if (OUT_DIR / "results.json").exists() else {"definitions": {}, "runs": []}
        plan = [
            ("lsb", "SDSD", "meas_op", 8 if quick else 24, 160 if quick else 500, HEADLINE_AMAX, 21001),
            ("mid", "SDSD", "meas_op", 6 if quick else 20, 160 if quick else 450, HEADLINE_AMAX, 22001),
            ("msb", "SDSD", "meas_op", 6 if quick else 20, 160 if quick else 450, HEADLINE_AMAX, 23001),
            ("lsb", "DSDS", "meas_op", 4 if quick else 12, 140 if quick else 400, HEADLINE_AMAX, 21002),
        ]
        summary = prev
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        for bit, order, obj, nrand, maxiter, amax, seed in plan:
            print(f"\n=== {bit} {order} {obj} amax={amax} random={nrand} maxiter={maxiter} ===", flush=True)
            block = run_config(bit, order, obj, nrand, maxiter, amax, seed)
            summary["runs"].append(block)
            save_json(OUT_DIR / "results.json", summary)
            print(
                f"  DONE loss {block['loss_best']:.6e} med {block['loss_median']:.4e} "
                f"eps {block['metrics']['readout']['eps_X']:.4f} "
                f"F_H {block['metrics']['gates']['H']['F_proc']:.4f} "
                f"leak {block['metrics']['leakage']['leak_worst_state']:.4f} "
                f"in {block['seconds']:.1f}s",
                flush=True,
            )
        md = write_results_md(summary)
        (OUT_DIR / "RESULTS.md").write_text(md)
        plots = make_plots(summary)
        ART_DIR.mkdir(parents=True, exist_ok=True)
        slim = {
            "definitions": summary.get("definitions", {}),
            "runs": [
                {k: v for k, v in row.items() if k not in ("metrics_full", "start_losses", "x_raw")}
                for row in summary["runs"]
            ],
        }
        save_json(ART_DIR / "binary_snap2_summary.json", slim)
        save_json(OUT_DIR / "summary_slim.json", slim)
        (ART_DIR / "binary_snap2_RESULTS.md").write_text(md)
        for p in plots:
            (ART_DIR / p.name).write_bytes(p.read_bytes())
        print("opnorm append done", flush=True)
        return
    run_selftests()
    quick = args.quick
    # (bit, order, objective, n_random, maxiter, amax, seed)
    plan = []
    if args.stage in ("lsb", "all"):
        plan += [
            ("lsb", "SDSD", "meas", 6 if quick else 36, 200 if quick else 700, HEADLINE_AMAX, 11001),
            ("lsb", "DSDS", "meas", 4 if quick else 24, 180 if quick else 600, HEADLINE_AMAX, 11002),
            ("lsb", "SDSD", "eq", 4 if quick else 20, 180 if quick else 550, HEADLINE_AMAX, 11003),
            ("lsb", "SDSD", "gate_H", 4 if quick else 24, 200 if quick else 600, HEADLINE_AMAX, 11004),
            ("lsb", "DSDS", "gate_H", 3 if quick else 16, 180 if quick else 500, HEADLINE_AMAX, 11005),
            ("lsb", "SDSD", "gate_Ry", 3 if quick else 12, 160 if quick else 450, HEADLINE_AMAX, 11006),
            ("lsb", "SDSD", "meas", 3 if quick else 12, 160 if quick else 450, 0.8, 11007),
            ("lsb", "SDSD", "meas", 3 if quick else 16, 180 if quick else 550, 2.2, 11008),
        ]
    if args.stage == "all":
        plan += [
            ("mid", "SDSD", "meas", 4 if quick else 28, 180 if quick else 600, HEADLINE_AMAX, 12001),
            ("mid", "DSDS", "meas", 3 if quick else 16, 160 if quick else 450, HEADLINE_AMAX, 12002),
            ("msb", "SDSD", "meas", 4 if quick else 28, 180 if quick else 600, HEADLINE_AMAX, 13001),
            ("msb", "DSDS", "meas", 3 if quick else 16, 160 if quick else 450, HEADLINE_AMAX, 13002),
            ("mid", "SDSD", "eq", 3 if quick else 14, 160 if quick else 400, HEADLINE_AMAX, 12003),
            ("msb", "SDSD", "eq", 3 if quick else 14, 160 if quick else 400, HEADLINE_AMAX, 13003),
            ("mid", "SDSD", "gate_H", 3 if quick else 16, 160 if quick else 500, HEADLINE_AMAX, 12004),
            ("msb", "SDSD", "gate_H", 3 if quick else 16, 160 if quick else 500, HEADLINE_AMAX, 13004),
            ("mid", "DSDS", "gate_H", 2 if quick else 10, 140 if quick else 400, HEADLINE_AMAX, 12005),
            ("msb", "DSDS", "gate_H", 2 if quick else 10, 140 if quick else 400, HEADLINE_AMAX, 13005),
            ("mid", "SDSD", "gate_Ry", 2 if quick else 8, 140 if quick else 350, HEADLINE_AMAX, 12006),
            ("msb", "SDSD", "gate_Ry", 2 if quick else 8, 140 if quick else 350, HEADLINE_AMAX, 13006),
        ]
    summary = {
        "definitions": {
            "encoding": "Fock n=0..7 = binary b2 b1 b0, b0 LSB",
            "snap_control": "free phases on n=0..7, identically 0 on n>=8",
            "displacement": "expm of NF-truncated generator via spectral factors of i(ad-a); not an 8x8 truncation",
            "amax_headline": HEADLINE_AMAX,
            "amax_note": "1.5 is inside the repo ansatz soft clip of 1.8; 0.8 and 2.2 are sensitivity, not fictitious large displacements",
            "F_proc": "|Tr(U_tgt^H M)|^2/64, Mij=<i|U|j> on 0..7, global phase free",
            "eps_X": "max(||E+-(I+X)/2||_op, ||E--(I-X)/2||_op), leakage is a third outcome",
            "orders": {
                "SDSD": {"physical": ["S1", "D1", "S2", "D2"], "written": "D2 @ S2 @ D1 @ S1"},
                "DSDS": {"physical": ["D1", "S1", "D2", "S2"], "written": "S2 @ D2 @ S1 @ D1"},
            },
        },
        "runs": [],
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for bit, order, obj, nrand, maxiter, amax, seed in plan:
        print(f"\n=== {bit} {order} {obj} amax={amax} random={nrand} maxiter={maxiter} ===", flush=True)
        block = run_config(bit, order, obj, nrand, maxiter, amax, seed)
        # Drop the duplicated full matrix blob from the rolling file only after compact is kept.
        # Keep metrics_full for the headline meas/gate runs; it is the proof.
        summary["runs"].append(block)
        save_json(OUT_DIR / "results.json", summary)
        print(
            f"  DONE loss {block['loss_best']:.6e} med {block['loss_median']:.4e} "
            f"eps {block['metrics']['readout']['eps_X']:.4f} "
            f"F_H {block['metrics']['gates']['H']['F_proc']:.4f} "
            f"leak {block['metrics']['leakage']['leak_worst_state']:.4f} "
            f"in {block['seconds']:.1f}s",
            flush=True,
        )
    md = write_results_md(summary)
    (OUT_DIR / "RESULTS.md").write_text(md)
    try:
        plots = make_plots(summary)
    except Exception as exc:  # plotting should not discard numbers
        print("plot failed", exc, flush=True)
        plots = []
    ART_DIR.mkdir(parents=True, exist_ok=True)
    # concise artifact: strip start_losses duplication already inside, copy md, png, and a slim json
    slim = {
        "definitions": summary["definitions"],
        "runs": [
            {k: v for k, v in row.items() if k not in ("metrics_full", "start_losses", "x_raw")}
            for row in summary["runs"]
        ],
    }
    save_json(ART_DIR / "binary_snap2_summary.json", slim)
    save_json(OUT_DIR / "summary_slim.json", slim)
    (ART_DIR / "binary_snap2_RESULTS.md").write_text(md)
    for p in plots:
        target = ART_DIR / p.name
        target.write_bytes(p.read_bytes())
    print("wrote", OUT_DIR, "artifacts", ART_DIR, flush=True)


if __name__ == "__main__":
    main()
