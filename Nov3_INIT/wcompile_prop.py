"""Norm-preserving SNAP+D on a common Fock workspace.

Historical PR109 compiled_map restricted D from nf+32 down to nf after
EVERY displacement. That is P D_{nf+32} P, applied K times, and is not
oscillator evolution. This module keeps all amplitudes on one workspace
for the whole sequence and only reads code/pad/boundary at the end.
"""
from __future__ import annotations

from collections import Counter

import numpy as np
from scipy.linalg import expm

from Nov3_INIT.fock import CODE, w_binh
from Nov3_INIT.wcompile import COMPILER_SEED, K_PAIRS, N_CODE, global_phase, unpack
from Nov9_SK_SPSA.experiments.FT.sim import boson_generator, displacement_matrix, generator

LADDER = (32, 64, 96)
BOUNDARY = 4
N_CODE = CODE
EVALS = Counter()


def _count(kind: str, n: int = 1) -> None:
    EVALS[kind] += int(n)
    EVALS["total"] += int(n)


def displacement_on_workspace(alpha: float, nf: int) -> np.ndarray:
    """True D on this workspace: exp in nf+32, restrict once to nf.

    The restriction defines the gate on C^{nf}. It is not re-applied between
    later gates; the sequence stays in C^{nf}.
    """
    _count("displacement_build")
    return displacement_matrix(float(alpha), int(nf), "restricted").astype(np.complex128)


def displacement_expm_cropped(alpha: float, nf: int) -> np.ndarray:
    """exp[α(a†-a)] on nf levels (cropped a). Not a true displacement."""
    _count("displacement_expm_cropped")
    g = boson_generator(int(nf))
    return expm(float(alpha) * g).astype(np.complex128)


def snap_diag(nf: int, theta8: np.ndarray) -> np.ndarray:
    ph = np.zeros(int(nf), dtype=np.float64)
    ph[:N_CODE] = np.asarray(theta8, dtype=np.float64)
    return np.exp(-1j * ph)


def workspace_map(alpha: np.ndarray, theta: np.ndarray, nf: int, inverse: bool = False) -> np.ndarray:
    """(D, SNAP)^K on a common workspace. Columns = U|k>, k=0..7."""
    nf = int(nf)
    _count("sequence")
    v = np.zeros((nf, N_CODE), dtype=np.complex128)
    v[:N_CODE, :] = np.eye(N_CODE, dtype=np.complex128)
    ds = [displacement_on_workspace(float(a), nf) for a in alpha]
    ss = [snap_diag(nf, th) for th in theta]
    if not inverse:
        for k in range(K_PAIRS):
            v = ds[k] @ v
            v = ss[k][:, None] * v
    else:
        for k in range(K_PAIRS - 1, -1, -1):
            v = np.conj(ss[k])[:, None] * v
            v = ds[k].T.conj() @ v
    return v


def target_isometry(nf: int) -> np.ndarray:
    t = np.zeros((int(nf), N_CODE), dtype=np.complex128)
    t[:N_CODE, :] = w_binh(N_CODE).astype(np.float64)
    return t


def partition_columns(v: np.ndarray) -> dict:
    """Mean over 8 code inputs. Probabilities are ‖·‖², not Euclidean norms."""
    nf = v.shape[0]
    p_tot = np.array([float(np.vdot(v[:, k], v[:, k]).real) for k in range(N_CODE)])
    p_code = np.array([float(np.vdot(v[:N_CODE, k], v[:N_CODE, k]).real) for k in range(N_CODE)])
    if nf > N_CODE:
        p_pad = np.array([float(np.vdot(v[N_CODE:, k], v[N_CODE:, k]).real) for k in range(N_CODE)])
    else:
        p_pad = np.zeros(N_CODE)
    p_lost = 1.0 - p_tot
    n_b = min(BOUNDARY, nf)
    p_boundary = np.array([float(np.vdot(v[-n_b:, k], v[-n_b:, k]).real) for k in range(N_CODE)])
    euc = np.sqrt(np.clip(p_tot, 0.0, None))
    return {
        "p_code": [float(z) for z in p_code],
        "p_pad_retained": [float(z) for z in p_pad],
        "p_lost": [float(z) for z in p_lost],
        "p_total": [float(z) for z in p_tot],
        "p_boundary_last4": [float(z) for z in p_boundary],
        "euclid_norm": [float(z) for z in euc],
        "mean_p_code": float(p_code.mean()),
        "mean_p_pad_retained": float(p_pad.mean()),
        "mean_p_lost": float(p_lost.mean()),
        "mean_p_total": float(p_tot.mean()),
        "mean_p_boundary": float(p_boundary.mean()),
        "mean_euclid_norm": float(euc.mean()),
        "mean_p_total_is_norm_squared": True,
        "old_p_leak_was_one_minus_p_code": float((1.0 - p_code).mean()),
    }


def isometry_metrics(v: np.ndarray, nf: int) -> dict:
    t = target_isometry(nf)
    part = partition_columns(v)
    phi = global_phase(t, v)
    vp = np.exp(-1j * phi) * v
    col_fid = np.array([float(np.abs(np.vdot(t[:, k], vp[:, k])) ** 2) for k in range(N_CODE)])
    mse = float(np.sum(np.abs(vp - t) ** 2) / N_CODE)
    hs = np.vdot(t.ravel(), v.ravel())
    f_hs = float(np.abs(hs) ** 2 / (N_CODE * N_CODE))
    plus_in = np.full(N_CODE, N_CODE ** -0.5, dtype=np.complex128)
    plus_out = v @ plus_in
    plus_tgt = t @ plus_in
    plus_fid = float(np.abs(np.vdot(plus_tgt, np.exp(-1j * phi) * plus_out)) ** 2)
    rng = np.random.default_rng(COMPILER_SEED)
    amp = rng.normal(size=N_CODE) + 1j * rng.normal(size=N_CODE)
    amp /= np.linalg.norm(amp)
    rnd_out = v @ amp
    rnd_tgt = t @ amp
    rnd_fid = float(np.abs(np.vdot(rnd_tgt, np.exp(-1j * phi) * rnd_out)) ** 2)
    return {
        "L": mse,
        "phi": phi,
        "F_avg": float(col_fid.mean()),
        "F_worst": float(col_fid.min()),
        "F_hs": f_hs,
        "col_fid": [float(z) for z in col_fid],
        "plus_fid": plus_fid,
        "random_fid": rnd_fid,
        "nf": int(nf),
        "F_avg_unnormalized_overlap_sq": True,
        "partition": part,
        "no_renorm": True,
        "no_postselect": True,
    }


def pulse_from_json(doc: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(doc["x"], dtype=np.float64)
    alpha, theta = unpack(x)
    return x, alpha, theta


def embed_cols(v: np.ndarray, nf_big: int) -> np.ndarray:
    out = np.zeros((int(nf_big), N_CODE), dtype=np.complex128)
    n = min(v.shape[0], int(nf_big))
    out[:n, :] = v[:n, :]
    return out


def column_overlap(a: np.ndarray, b: np.ndarray) -> dict:
    """Compare two 8-column maps after padding the smaller and one global phase."""
    n = max(a.shape[0], b.shape[0])
    aa = embed_cols(a, n)
    bb = embed_cols(b, n)
    phi = global_phase(aa, bb)
    bp = np.exp(-1j * phi) * bb
    fids = [float(np.abs(np.vdot(aa[:, k], bp[:, k])) ** 2) for k in range(N_CODE)]
    fro = float(np.linalg.norm(aa - bp) / np.sqrt(N_CODE))
    return {
        "mean_col_overlap_sq": float(np.mean(fids)),
        "worst_col_overlap_sq": float(np.min(fids)),
        "frobenius_per_col": fro,
        "code_block_fro": float(np.linalg.norm(aa[:N_CODE, :] - bp[:N_CODE, :])),
    }


def tests(alpha_pulse: np.ndarray, theta_pulse: np.ndarray, nf: int = 96) -> dict:
    nf = int(nf)
    a = float(alpha_pulse[0])
    # α=0 identity
    d0 = displacement_on_workspace(0.0, nf)
    ident_err = float(np.linalg.norm(d0 - np.eye(nf)))
    # D(a) D(-a)
    dp = displacement_on_workspace(a, nf)
    dm = displacement_on_workspace(-a, nf)
    rec = dm @ dp
    rec_err_full = float(np.linalg.norm(rec - np.eye(nf)) / nf)
    rec_code = []
    for k in (0, 3, 7):
        e = np.zeros(nf, dtype=np.complex128)
        e[k] = 1.0
        rec_code.append(float(np.linalg.norm(dm @ dp @ e)))
    plus = np.zeros(nf, dtype=np.complex128)
    plus[:N_CODE] = N_CODE ** -0.5
    rec_plus = float(np.linalg.norm(dm @ dp @ plus))
    # SNAP unitarity
    s = snap_diag(nf, theta_pulse[0])
    snap_u = float(np.max(np.abs(np.abs(s) - 1.0)))
    # sequence then inverse
    v = workspace_map(alpha_pulse, theta_pulse, nf, inverse=False)
    # apply inverse to those columns
    _count("sequence_inverse_extra")
    ds = [displacement_on_workspace(float(ai), nf) for ai in alpha_pulse]
    ss = [snap_diag(nf, th) for th in theta_pulse]
    w = v.copy()
    for k in range(K_PAIRS - 1, -1, -1):
        w = np.conj(ss[k])[:, None] * w
        w = ds[k].T.conj() @ w
    eye = np.zeros((nf, N_CODE), dtype=np.complex128)
    eye[:N_CODE, :] = np.eye(N_CODE)
    inv_err = float(np.linalg.norm(w - eye) / np.sqrt(N_CODE))
    inv_norms = [float(np.linalg.norm(w[:, k])) for k in range(N_CODE)]
    # cropped vs +32 restricted
    d_true = dp
    d_crop = displacement_expm_cropped(a, nf)
    crop_vs_true = float(np.linalg.norm(d_true - d_crop) / nf)
    d_ws = generator(nf).exp(a)
    ws_vs_true = float(np.linalg.norm(d_true - d_ws) / nf)
    return {
        "nf": nf,
        "alpha_used": a,
        "D0_minus_I_fro": ident_err,
        "Dminus_Dplus_I_fro_over_n": rec_err_full,
        "roundtrip_norm_fock_0_3_7": rec_code,
        "roundtrip_norm_plus": rec_plus,
        "snap_unimodular_max_err": snap_u,
        "sequence_inverse_fro_per_col": inv_err,
        "sequence_inverse_col_norms": inv_norms,
        "cropped_expm_vs_restricted_fro_over_n": crop_vs_true,
        "generator_nf_vs_restricted_fro_over_n": ws_vs_true,
        "D0_ok": ident_err < 1e-10,
        "snap_ok": snap_u < 1e-15,
        "roundtrip_code_ok": all(abs(z - 1.0) < 1e-8 for z in rec_code) and abs(rec_plus - 1.0) < 1e-8,
        "inverse_ok": inv_err < 1e-6 and all(abs(z - 1.0) < 1e-6 for z in inv_norms),
    }


def pr109_partition_explanation() -> dict:
    """Reconstruct the 0.402 vs 0.672 arithmetic from the stored NF=16 columns."""
    return {
        "col_norm_field": "mean of <psi|psi> = mean retained probability = mean ||psi||^2, NOT Euclidean ||psi||",
        "mean_p_total_0.402": "retained in the 16-level register after three P_16 D_48 P_16",
        "mean_p_code_0.328": "mass on Fock 0..7",
        "mean_p_pad_in_nf": "0.402-0.328 ~ 0.074 (levels 8..15 kept)",
        "mean_p_lost": "1-0.402 ~ 0.598 (thrown away at each restriction)",
        "old_p_leak_0.672": "1-p_code = pad_in_nf + lost = 0.074+0.598",
        "euclid_mean": "mean sqrt(p_total) is a different number; do not equate to 0.402",
    }
