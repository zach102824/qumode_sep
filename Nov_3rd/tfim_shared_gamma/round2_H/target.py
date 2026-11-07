"""Frozen Round-2 target. Couplings and B come from the manifest strings.

Basis: index = sum_i bit_i * 2**i, Z_i = (-1)**bit_i, site i is bit i.
Do not regenerate J from a power law.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import thread_env  # noqa: F401
import numpy as np
from scipy.linalg import eigh

ROOT = Path(__file__).resolve().parent
MANIFEST_PATH = ROOT / "FROZEN_TARGET_R2.json"
MANIFEST_SHA256 = "14f40daa135640c9eb18d4ff64d2608216f139b51d87e9ede2fcd20939851da9"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_array(a: np.ndarray) -> str:
    arr = np.ascontiguousarray(a)
    return hashlib.sha256(arr.tobytes()).hexdigest()


def load_manifest() -> tuple[dict, bytes]:
    raw = MANIFEST_PATH.read_bytes()
    digest = sha256_bytes(raw)
    if digest != MANIFEST_SHA256:
        raise RuntimeError(f"manifest hash {digest} != {MANIFEST_SHA256}")
    return json.loads(raw.decode("utf-8")), raw


def parse_system(manifest: dict, n: int) -> tuple[np.ndarray, float]:
    system = manifest["systems"][str(n)]
    if int(system["N"]) != n:
        raise RuntimeError("N mismatch")
    B = float(system["B"])
    J = np.zeros((n, n), dtype=np.float64)
    seen = set()
    for i, j, value in system["Jij"]:
        i = int(i)
        j = int(j)
        if i > j:
            i, j = j, i
        if (i, j) in seen or i == j:
            raise RuntimeError(f"bad pair {(i, j)}")
        seen.add((i, j))
        J[i, j] = J[j, i] = float(value)
    n_pairs = n * (n - 1) // 2
    if len(seen) != n_pairs:
        raise RuntimeError(f"expected {n_pairs} pairs, got {len(seen)}")
    return J, B


def z_table(n: int) -> np.ndarray:
    """z[x, i] = (-1)**bit_i with bit_i = (x >> i) & 1."""
    dim = 1 << n
    bits = ((np.arange(dim, dtype=np.int64)[:, None] >> np.arange(n, dtype=np.int64)) & 1).astype(np.float64)
    return 1.0 - 2.0 * bits


def pair_diagonal(coeff: np.ndarray, z: np.ndarray) -> np.ndarray:
    """sum_{i<j} coeff_ij z_i z_j. coeff is symmetric with zero diagonal."""
    return 0.5 * np.einsum("ai,ij,aj->a", z, coeff, z, optimize=True)


def upper_rms(mat: np.ndarray) -> float:
    n = mat.shape[0]
    iu = np.triu_indices(n, 1)
    vals = np.asarray(mat, dtype=np.float64)[iu]
    return float(np.sqrt(np.mean(vals * vals)))


def fk_matrix(J: np.ndarray) -> np.ndarray:
    """K_jk = sum_{i != j,k} J_ij J_ik = (J @ J)_jk for j != k, zero diagonal.

    J has a zero diagonal, so the i=j and i=k terms vanish and the sum is the
    full matrix product.
    """
    K = J @ J
    np.fill_diagonal(K, 0.0)
    return 0.5 * (K + K.T)


def normalize_feature(K: np.ndarray, rms_J: float) -> tuple[np.ndarray, float, float]:
    """Rescale the full upper triangle, including any zeros, onto rms_J."""
    rms_K = upper_rms(K)
    if rms_K <= 0:
        raise RuntimeError("feature RMS is zero")
    scale = float(rms_J / rms_K)
    return K * scale, rms_K, scale


def random_feature(n: int, rms_J: float, seed: int = 20261107) -> np.ndarray:
    """Frozen Gaussian upper triangle, zero diagonal, RMS matched to Hc."""
    rng = np.random.Generator(np.random.PCG64(seed))
    K = np.zeros((n, n), dtype=np.float64)
    iu = np.triu_indices(n, 1)
    vals = rng.normal(size=iu[0].size)
    K[iu] = vals
    K[(iu[1], iu[0])] = vals
    Kn, _rms, _scale = normalize_feature(K, rms_J)
    return Kn


def hc_hash_le(hc: np.ndarray) -> str:
    le = np.ascontiguousarray(hc, dtype="<f8")
    return hashlib.sha256(le.tobytes()).hexdigest()


def apply_H(psi: np.ndarray, hc: np.ndarray, B: float, n: int) -> np.ndarray:
    out = hc * psi
    tensor = np.reshape(psi, [2] * n)
    acc = np.zeros_like(tensor)
    for i in range(n):
        acc += np.flip(tensor, axis=n - 1 - i)
    return out + B * acc.reshape(psi.shape)


def dense_H(hc: np.ndarray, B: float, n: int) -> np.ndarray:
    dim = 1 << n
    H = np.diag(np.asarray(hc, dtype=np.float64)).copy()
    idx = np.arange(dim, dtype=np.int64)
    for i in range(n):
        partner = idx ^ (1 << i)
        H[partner, idx] += float(B)
    return H


def bit_reverse_perm(n: int) -> np.ndarray:
    dim = 1 << n
    perm = np.empty(dim, dtype=np.int64)
    for x in range(dim):
        y = 0
        for b in range(n):
            if (x >> b) & 1:
                y |= 1 << (n - 1 - b)
        perm[x] = y
    return perm


def parity_weight(psi: np.ndarray) -> float:
    dim = psi.shape[0]
    flipped = psi[np.arange(dim) ^ (dim - 1)]
    proj = 0.5 * (psi + flipped)
    return float(np.vdot(proj, proj).real)


def reflection_weight(psi: np.ndarray, perm: np.ndarray) -> float:
    proj = 0.5 * (psi + psi[perm])
    return float(np.vdot(proj, proj).real)


def canonicalize_gs(gs: np.ndarray) -> np.ndarray:
    v = np.asarray(gs, dtype=np.complex128).copy()
    k = int(np.argmax(np.abs(v)))
    v *= abs(v[k]) / v[k]
    if np.linalg.norm(v.imag) <= 1e-10 * max(1.0, np.linalg.norm(v.real)):
        v = v.real.astype(np.float64)
        if float(np.sum(v)) < 0:
            v = -v
        return v.astype(np.complex128)
    return v


def half_chain_entropy_bits(psi: np.ndarray, n: int) -> float:
    """Entropy of sites 0..n/2-1. Those sites are the low bits."""
    nA = n // 2
    mat = np.asarray(psi, dtype=np.complex128).reshape(1 << (n - nA), 1 << nA)
    s = np.linalg.svd(mat, compute_uv=False)
    p = np.clip(s.real ** 2, 0.0, None)
    p = p[p > 1e-16]
    p = p / p.sum()
    return float(-(p * np.log2(p)).sum())


def spectrum_diagnostics(hc: np.ndarray, B: float, n: int, n_states: int = 12) -> dict:
    H = dense_H(hc, B, n)
    evals, evecs = eigh(H)
    perm = bit_reverse_perm(n)
    rows = []
    accessible = None
    n_keep = min(n_states, evals.shape[0])
    for k in range(n_keep):
        psi = canonicalize_gs(evecs[:, k])
        pw = parity_weight(psi)
        rw = reflection_weight(psi, perm)
        gap = float(evals[k] - evals[0])
        rows.append({"k": k, "E": float(evals[k]), "gap_from_gs": gap, "P_plus_weight": pw, "R_plus_weight": rw})
        if k > 0 and accessible is None and pw > 0.99 and rw > 0.99:
            accessible = gap
    gs = canonicalize_gs(evecs[:, 0])
    residual = float(np.linalg.norm(apply_H(gs, hc, B, n) - evals[0] * gs))
    ev = np.asarray(evals, dtype=np.float64)
    return {
        "E0": float(ev[0]),
        "E1": float(ev[1]),
        "full_space_gap": float(ev[1] - ev[0]),
        "accessible_PR_plus_gap": accessible,
        "Emax": float(ev[-1]),
        "entropy_bits": half_chain_entropy_bits(gs, n),
        "gs_residual": residual,
        "gs_min_real": float(np.min(gs.real)),
        "P_plus_weight": parity_weight(gs),
        "R_plus_weight": reflection_weight(gs, perm),
        "lowest_states": rows,
        "spectrum_sha256": sha256_array(np.asarray(ev, dtype="<f8")),
        "gs": gs,
        "gap_note": (
            "full_space_gap is E1-E0. The partner is P=-1, R=-1 and is inaccessible. "
            "accessible_PR_plus_gap is the gap to the next state in P=+1, R=+1. "
            "Do not substitute the accessible gap for the full-space gap."
        ),
    }


def jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def dump_json(path: Path, obj) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(jsonable(obj), indent=2, allow_nan=False) + "\n")
