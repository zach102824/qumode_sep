"""SNAP + true displacement compile of the W_BinH isometry.

Real α, SNAP pad phase 0, full Fock output, no crop/renorm/postselect.
Compiler evaluations only; not an SK energy query.
"""
from __future__ import annotations

import numpy as np

from Nov3_INIT.fock import CODE, w_binh
from Nov9_SK_SPSA.experiments.FT.sim import displacement_matrix

K_PAIRS = 3
NF_OPT = 16
NF_REPLAY = 32
N_CODE = CODE
N_PARAMS = K_PAIRS * (1 + N_CODE)
A_MAX = 1.8
Q_COMPILE = 2048
COMPILER_SEED = 33001
SIG_U = 0.15
SIG_TH = 0.25


def target_isometry(nf: int) -> np.ndarray:
    """NF×8: H^{⊗3} on code rows, zeros on the pad."""
    t = np.zeros((int(nf), N_CODE), dtype=np.complex128)
    t[:N_CODE, :] = w_binh(N_CODE).astype(np.float64)
    return t


def unpack(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float64).reshape(K_PAIRS, 1 + N_CODE)
    u = x[:, 0]
    theta = x[:, 1:]
    alpha = A_MAX * np.tanh(u)
    return alpha, theta


def snap_diag(nf: int, theta8: np.ndarray) -> np.ndarray:
    ph = np.zeros(int(nf), dtype=np.float64)
    ph[:N_CODE] = np.asarray(theta8, dtype=np.float64)
    return np.exp(-1j * ph)


def compiled_map(x: np.ndarray, nf: int) -> np.ndarray:
    """Apply (D, SNAP)^K to each code basis ket. Columns are full NF states."""
    alpha, theta = unpack(x)
    nf = int(nf)
    v = np.zeros((nf, N_CODE), dtype=np.complex128)
    v[:N_CODE, :] = np.eye(N_CODE, dtype=np.complex128)
    for k in range(K_PAIRS):
        d = displacement_matrix(float(alpha[k]), nf, "restricted").astype(np.complex128)
        s = snap_diag(nf, theta[k])
        v = d @ v
        v = s[:, None] * v
    return v


def global_phase(t: np.ndarray, v: np.ndarray) -> float:
    hs = np.vdot(t.ravel(), v.ravel())
    return float(np.angle(hs))


def metrics(x: np.ndarray, nf: int) -> dict:
    t = target_isometry(nf)
    v = compiled_map(x, nf)
    phi = global_phase(t, v)
    vp = np.exp(-1j * phi) * v
    col_ov = np.array([np.vdot(t[:, k], vp[:, k]) for k in range(N_CODE)])
    col_fid = np.abs(col_ov) ** 2
    col_norm = np.array([float(np.vdot(v[:, k], v[:, k]).real) for k in range(N_CODE)])
    p_code = np.array([float(np.vdot(v[:N_CODE, k], v[:N_CODE, k]).real) for k in range(N_CODE)])
    p_leak = 1.0 - p_code
    mse = float(np.sum(np.abs(vp - t) ** 2) / N_CODE)
    hs = np.vdot(t.ravel(), v.ravel())
    f_hs = float(np.abs(hs) ** 2 / (N_CODE * N_CODE))
    plus_in = np.full(N_CODE, N_CODE ** -0.5, dtype=np.complex128)
    plus_out = v @ plus_in
    plus_tgt = t @ plus_in
    plus_out_p = np.exp(-1j * phi) * plus_out
    plus_fid = float(np.abs(np.vdot(plus_tgt, plus_out_p)) ** 2)
    plus_leak = float(1.0 - np.vdot(plus_out[:N_CODE], plus_out[:N_CODE]).real)
    rng = np.random.default_rng(COMPILER_SEED)
    amp = rng.normal(size=N_CODE) + 1j * rng.normal(size=N_CODE)
    amp /= np.linalg.norm(amp)
    rnd_out = v @ amp
    rnd_tgt = t @ amp
    rnd_fid = float(np.abs(np.vdot(rnd_tgt, np.exp(-1j * phi) * rnd_out)) ** 2)
    rnd_leak = float(1.0 - np.vdot(rnd_out[:N_CODE], rnd_out[:N_CODE]).real)
    f_avg = float(col_fid.mean())
    f_worst = float(col_fid.min())
    return {
        "ok": bool(np.isfinite(mse) and np.isfinite(f_avg)),
        "L": mse,
        "phi": phi,
        "F_avg": f_avg,
        "F_worst": f_worst,
        "F_hs": f_hs,
        "infidelity_avg": float(1.0 - f_avg),
        "infidelity_worst": float(1.0 - f_worst),
        "p_leak_mean": float(p_leak.mean()),
        "p_leak_worst": float(p_leak.max()),
        "p_code_mean": float(p_code.mean()),
        "col_fid": [float(z) for z in col_fid],
        "col_norm": [float(z) for z in col_norm],
        "p_leak": [float(z) for z in p_leak],
        "plus_fid": plus_fid,
        "plus_leak": plus_leak,
        "random_fid": rnd_fid,
        "random_leak": rnd_leak,
        "nf": int(nf),
        "norm_deficit_mean": float(np.mean(np.abs(1.0 - col_norm))),
        "no_renorm": True,
        "no_postselect": True,
        "global_phase_one_overall": True,
    }


def sigmas() -> np.ndarray:
    sig = np.empty(N_PARAMS, dtype=np.float64)
    for k in range(K_PAIRS):
        sl = k * (1 + N_CODE)
        sig[sl] = SIG_U
        sig[sl + 1 : sl + 1 + N_CODE] = SIG_TH
    return sig


def x0_identity() -> np.ndarray:
    return np.zeros(N_PARAMS, dtype=np.float64)


def x0_spread(start_id: int, rng: np.random.Generator) -> np.ndarray:
    if int(start_id) == 0:
        return x0_identity()
    x = np.empty(N_PARAMS, dtype=np.float64)
    for k in range(K_PAIRS):
        sl = k * (1 + N_CODE)
        x[sl] = float(rng.uniform(-1.0, 1.0))
        x[sl + 1 : sl + 1 + N_CODE] = rng.uniform(-np.pi, np.pi, size=N_CODE)
    return x


def pulse_dict(x: np.ndarray) -> dict:
    alpha, theta = unpack(x)
    return {
        "K": K_PAIRS,
        "order": "D then SNAP, repeated K times",
        "A_MAX": A_MAX,
        "alpha_real": [float(a) for a in alpha],
        "theta_code": [[float(v) for v in row] for row in theta],
        "snap_pad_forced_0": True,
        "displacement": "restricted true D from NF+32, real alpha",
        "x": [float(v) for v in np.asarray(x, dtype=np.float64)],
    }


def candidate_gates(m16: dict, m32: dict) -> dict:
    dF = abs(float(m32["F_avg"]) - float(m16["F_avg"]))
    ok16 = bool(
        m16["F_avg"] >= 0.99
        and m16["F_worst"] >= 0.95
        and m16["p_leak_mean"] <= 0.01
        and m16["p_leak_worst"] <= 0.05
        and m16["F_hs"] >= 0.99
    )
    replay_ok = bool(dF <= 1e-3)
    label = "WCOMP_CANDIDATE" if ok16 and replay_ok else "WCOMP_FAIL_BUDGET"
    return {
        "label": label,
        "nf16_bars": ok16,
        "replay_dF_avg": dF,
        "replay_ok": replay_ok,
    }
