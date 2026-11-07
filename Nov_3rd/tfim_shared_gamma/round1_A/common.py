"""Shared constants, hashes, and the round-1 A contract record.

Bit convention (frozen):
  Site i is bit i of the computational index, with site 0 the LSB.
  Cavity a holds sites 3a, 3a+1, 3a+2.
  Natural binary inside a cavity: j = b0 + 2 b1 + 4 b2.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

PKG = Path(__file__).resolve().parent
RESULTS = PKG / "results"

ALPHA = 1.1
J0 = 1.0
B_COARSE_LO = -0.8
B_COARSE_HI = -0.1
B_COARSE_STEP = 0.005
B_REFINE_TOL = 1e-4
# Entropy ties flatter than this (bits) collapse to one frozen B.
ENTROPY_TIE_BITS = 1e-6

RANDOM_K_SEED = 20261103
REFL_K_SEED = 20261104
OPT_SEED_BASE = 20261107

# Predeclared before looking at N=12 energies. Raw and RMS are sensitivity checks.
PRIMARY_TRANSFER_SCALE = "kappa"

METHOD_ORDER = (
    "qaoa",
    "shared_qaoa",
    "native_D",
    "family_A",
    "random_K",
    "refl_K",
)

SHELL_METHODS = ("native_D", "family_A", "random_K", "refl_K")
THETA_METHODS = ("family_A", "random_K", "refl_K")

# Comparable energy-evaluation caps per (N, L, method), including grid samples.
EVAL_CAP = {1: 3500, 2: 5000, 3: 6000, 4: 7000, 5: 7000, 6: 7000}
LBFGS_MAXFUN = 220
LBFGS_MAXITER = 80

CONTRACT = {
    "round": "1",
    "direction": "A",
    "model": "open-chain AF TFIM, ideal power-law, not experimental matrices",
    "H": "Hc + B * sum_i X_i",
    "Hc": "sum_{i<j} J_ij Z_i Z_j",
    "J": "J_ij = J0 / |i-j|^alpha, J0=1, alpha=1.1",
    "G": "Hc + thetaK * F_K_normalized",
    "F_K": "sum_{j<k} K_jk Z_j Z_k",
    "K_jk": "sum_{i != j,k} J_ij J_ik = off-diagonal (J @ J)",
    "normalization": (
        "sK = rms(upper J) / rms(upper K); "
        "F_K_normalized uses sK * K; Hc coefficient fixed at 1; "
        "rms = sqrt(mean of squared upper-triangle entries)"
    ),
    "shared_across_layers": ["gamma", "thetaK", "K"],
    "per_layer": ["c1", "c2", "c3"],
    "shell_phi": "phi_k = c1*(3-2k) + c2*(((3-2k)^2-3)/2) + c3*(-1)^k",
    "logical_circuit": "|+>^N -> [D(gamma,G) -> M_l]^L",
    "physical_circuit": "vacuum -> [UcX -> shell_SNAP]^L -> W, UcX = W exp(-i gamma G) W",
    "objective": "energy only; fidelity and eta are post-hoc",
    "primary_comparator": "independent-angle QAOA, 2L parameters",
    "controls": ["native_D (thetaK=0, nonlinear shell)", "shared-gamma linear QAOA"],
    "ablations": ["thetaK=0 via native_D", "equal-RMS random K", "equal-RMS reflection-symmetric random K"],
    "no_gs_oracle": "K and training use only J; exact GS is for B selection and reported metrics",
    "primary_transfer_scale": PRIMARY_TRANSFER_SCALE,
    "gamma_ramp": "design heuristic; measured, not assumed",
    "bit_convention": "LSB = site 0; cavity a = sites 3a..3a+2; j = b0+2*b1+4*b2",
}


def log(msg: str) -> None:
    print(msg, flush=True)


def _json_default(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    raise TypeError(f"not JSON serializable: {type(obj).__name__}")


def json_dump(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=_json_default) + "\n")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_state(psi: np.ndarray, amp_tol: float = 1e-15) -> np.ndarray:
    """Fix global phase so the largest component is real and positive."""
    v = np.asarray(psi, dtype=np.complex128).copy()
    k = int(np.argmax(np.abs(v)))
    ph = v[k]
    if abs(ph) > 0:
        v *= np.conj(ph) / abs(ph)
    v[np.abs(v) < amp_tol] = 0
    return v


def sha256_state(psi: np.ndarray, decimals: int | None = None) -> str:
    c = canonical_state(psi)
    if decimals is not None:
        c = np.round(c.real, decimals) + 1j * np.round(c.imag, decimals)
    raw = np.ascontiguousarray(np.stack([np.real(c), np.imag(c)]))
    return sha256_bytes(raw.tobytes())


def param_count(method: str, L: int) -> int:
    if method == "qaoa":
        return 2 * L
    if method == "shared_qaoa":
        return 1 + L
    if method == "native_D":
        return 1 + 3 * L
    if method in THETA_METHODS:
        return 2 + 3 * L
    raise KeyError(method)


def method_index(method: str) -> int:
    return METHOD_ORDER.index(method)


def opt_rng(n: int, L: int, method: str, tag: int) -> np.random.Generator:
    seed = (
        OPT_SEED_BASE
        + int(n) * 10007
        + int(L) * 97
        + method_index(method) * 17
        + int(tag) * 104729
    )
    return np.random.default_rng(seed % (2**32 - 1))
