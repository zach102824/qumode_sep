"""Workspace-correct W_BinH compile. Common Fock space, no per-gate P_16."""
from __future__ import annotations

import numpy as np

from Nov3_INIT.fock import CODE, w_binh
from Nov3_INIT.wcompile import (
    A_MAX,
    COMPILER_SEED,
    K_PAIRS,
    N_CODE,
    N_PARAMS,
    Q_COMPILE,
    SIG_TH,
    SIG_U,
    global_phase,
    sigmas,
    unpack,
    x0_identity,
    x0_spread,
)
from Nov9_SK_SPSA.experiments.FT.sim import displacement_matrix

NF_FIT = 96
NF_REPLAY = 128
SEED_NF_TAG = 16  # PR109 seed tuple, not the workspace
P_LOST_BAR = 1e-8
L_BAR = 0.02
FHS_BAR = 0.99
P_CODE_BAR = 0.99
REPLAY_OV_BAR = 0.999
REPLAY_DL_BAR = 1e-3


def target_isometry(nf: int) -> np.ndarray:
    t = np.zeros((int(nf), N_CODE), dtype=np.complex128)
    t[:N_CODE, :] = w_binh(N_CODE).astype(np.float64)
    return t


def snap_diag(nf: int, theta8: np.ndarray) -> np.ndarray:
    ph = np.zeros(int(nf), dtype=np.float64)
    ph[:N_CODE] = np.asarray(theta8, dtype=np.float64)
    return np.exp(-1j * ph)


def workspace_map(x: np.ndarray, nf: int) -> np.ndarray:
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


def partition(v: np.ndarray) -> dict:
    p_tot = np.array([float(np.vdot(v[:, k], v[:, k]).real) for k in range(N_CODE)])
    p_code = np.array([float(np.vdot(v[:N_CODE, k], v[:N_CODE, k]).real) for k in range(N_CODE)])
    p_pad = p_tot - p_code
    p_lost = 1.0 - p_tot
    n_b = min(4, v.shape[0])
    p_b = np.array([float(np.vdot(v[-n_b:, k], v[-n_b:, k]).real) for k in range(N_CODE)])
    return {
        "p_code": [float(z) for z in p_code],
        "p_pad_retained": [float(z) for z in p_pad],
        "p_lost": [float(z) for z in p_lost],
        "p_total": [float(z) for z in p_tot],
        "p_boundary_last4": [float(z) for z in p_b],
        "mean_p_code": float(p_code.mean()),
        "mean_p_pad_retained": float(p_pad.mean()),
        "mean_p_lost": float(p_lost.mean()),
        "mean_p_total": float(p_tot.mean()),
        "mean_p_boundary": float(p_b.mean()),
        "partition_sum_mean": float((p_code + p_pad + p_lost).mean()),
    }


def metrics(x: np.ndarray, nf: int) -> dict:
    v = workspace_map(x, nf)
    t = target_isometry(nf)
    phi = global_phase(t, v)
    vp = np.exp(-1j * phi) * v
    col_fid = np.array([float(np.abs(np.vdot(t[:, k], vp[:, k])) ** 2) for k in range(N_CODE)])
    L = float(np.sum(np.abs(vp - t) ** 2) / N_CODE)
    hs = np.vdot(t.ravel(), v.ravel())
    f_hs = float(np.abs(hs) ** 2 / (N_CODE * N_CODE))
    plus_in = np.full(N_CODE, N_CODE ** -0.5, dtype=np.complex128)
    plus_fid = float(np.abs(np.vdot(t @ plus_in, np.exp(-1j * phi) * (v @ plus_in))) ** 2)
    rng = np.random.default_rng(COMPILER_SEED)
    amp = rng.normal(size=N_CODE) + 1j * rng.normal(size=N_CODE)
    amp /= np.linalg.norm(amp)
    rnd_fid = float(np.abs(np.vdot(t @ amp, np.exp(-1j * phi) * (v @ amp))) ** 2)
    part = partition(v)
    return {
        "ok": bool(np.isfinite(L) and np.isfinite(f_hs)),
        "L": L,
        "phi": phi,
        "F_hs": f_hs,
        "F_avg": float(col_fid.mean()),
        "F_worst": float(col_fid.min()),
        "col_fid": [float(z) for z in col_fid],
        "plus_fid": plus_fid,
        "random_fid": rnd_fid,
        "nf": int(nf),
        "partition": part,
        "selector_is_L_not_Favg": True,
        "no_renorm": True,
        "no_postselect": True,
    }


def embed(v: np.ndarray, nf_big: int) -> np.ndarray:
    out = np.zeros((int(nf_big), N_CODE), dtype=np.complex128)
    n = min(v.shape[0], int(nf_big))
    out[:n, :] = v[:n, :]
    return out


def column_overlap(a: np.ndarray, b: np.ndarray) -> dict:
    n = max(a.shape[0], b.shape[0])
    aa, bb = embed(a, n), embed(b, n)
    phi = global_phase(aa, bb)
    bp = np.exp(-1j * phi) * bb
    fids = np.array([float(np.abs(np.vdot(aa[:, k], bp[:, k])) ** 2) for k in range(N_CODE)])
    return {
        "mean_col_overlap_sq": float(fids.mean()),
        "worst_col_overlap_sq": float(fids.min()),
        "code_block_fro": float(np.linalg.norm(aa[:N_CODE, :] - bp[:N_CODE, :])),
        "frobenius_per_col": float(np.linalg.norm(aa - bp) / np.sqrt(N_CODE)),
    }


def compiler_rng(start_id: int, stream: int) -> np.random.Generator:
    return np.random.default_rng(
        np.random.SeedSequence([int(COMPILER_SEED), int(SEED_NF_TAG), int(K_PAIRS), int(start_id), int(stream)])
    )


def pulse_dict(x: np.ndarray) -> dict:
    alpha, theta = unpack(x)
    return {
        "K": K_PAIRS,
        "order": "D then SNAP, repeated K times, common workspace",
        "A_MAX": A_MAX,
        "alpha_real": [float(a) for a in alpha],
        "theta_code": [[float(v) for v in row] for row in theta],
        "snap_pad_forced_0": True,
        "displacement": "restricted true D from W+32, applied on common W, no per-gate P_16",
        "nf_fit": NF_FIT,
        "nf_replay": NF_REPLAY,
        "x": [float(v) for v in np.asarray(x, dtype=np.float64)],
    }


def label_of(m_fit: dict, m_rep: dict, ov: dict) -> str:
    lost_fail = (
        m_fit["partition"]["mean_p_lost"] > P_LOST_BAR
        or m_rep["partition"]["mean_p_lost"] > P_LOST_BAR
    )
    if lost_fail:
        return "WCOMP_WS_INCONCLUSIVE"
    cand = bool(
        m_fit["L"] <= L_BAR
        and m_fit["F_hs"] >= FHS_BAR
        and m_fit["partition"]["mean_p_code"] >= P_CODE_BAR
        and ov["mean_col_overlap_sq"] >= REPLAY_OV_BAR
        and abs(m_rep["L"] - m_fit["L"]) <= REPLAY_DL_BAR
    )
    return "WCOMP_WS_CANDIDATE" if cand else "WCOMP_WS_FAIL_BUDGET"
