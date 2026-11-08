"""Frozen Round-2 target. The manifest bytes are authoritative.

Basis: index = sum_i bit_i * 2^i, Z_i = (-1)^bit_i.
Packing: contiguous cavities (0,1,2), (3,4,5), ...; local Fock
j = bit_(3a) + 2*bit_(3a+1) + 4*bit_(3a+2).
Do not regenerate J or B from the power law. Do not run a new entropy scan.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

MANIFEST = Path(__file__).resolve().parent / "FROZEN_TARGET_R2.json"
EXPECTED_SHA256 = "14f40daa135640c9eb18d4ff64d2608216f139b51d87e9ede2fcd20939851da9"


def manifest_sha256() -> str:
    return hashlib.sha256(MANIFEST.read_bytes()).hexdigest()


def load_manifest() -> dict:
    digest = manifest_sha256()
    if digest != EXPECTED_SHA256:
        raise RuntimeError(f"frozen manifest hash {digest} != {EXPECTED_SHA256}")
    return json.loads(MANIFEST.read_text())


def system(n: int) -> dict:
    raw = load_manifest()["systems"][str(n)]
    if int(raw["N"]) != n:
        raise RuntimeError("manifest N mismatch")
    J = np.zeros((n, n), dtype=np.float64)
    for i, j, s in raw["Jij"]:
        J[int(i), int(j)] = J[int(j), int(i)] = float(s)
    B = float(raw["B"])
    hc = hc_diagonal(J)
    return {
        "n": n,
        "B_string": raw["B"],
        "B": B,
        "J": J,
        "hc": hc,
        "manifest_sha256": EXPECTED_SHA256,
        "hc_sha256": hashlib.sha256(np.ascontiguousarray(hc, dtype="<f8").tobytes()).hexdigest(),
        "basis": "index=sum(bit_i*2**i); Z_i=(-1)**bit_i",
    }


def z_table(n: int) -> np.ndarray:
    """Z eigenvalues, shape (n, 2^n), little-endian basis."""
    dim = 1 << n
    x = np.arange(dim, dtype=np.uint32)
    z = np.empty((n, dim), dtype=np.float64)
    for q in range(n):
        bit = (x >> np.uint32(q)) & np.uint32(1)
        z[q] = 1.0 - 2.0 * bit.astype(np.float64)
    return z


def hc_diagonal(J: np.ndarray, z: np.ndarray | None = None) -> np.ndarray:
    n = J.shape[0]
    if z is None:
        z = z_table(n)
    hc = np.zeros(z.shape[1], dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            if J[i, j] != 0.0:
                hc += J[i, j] * z[i] * z[j]
    return hc


def rms_list(values: np.ndarray) -> float:
    v = np.asarray(values, dtype=np.float64).ravel()
    if v.size == 0:
        raise RuntimeError("empty coefficient list")
    return float(np.sqrt(np.mean(v * v)))


def upper_values(mat: np.ndarray) -> np.ndarray:
    n = mat.shape[0]
    return np.array([mat[i, j] for i in range(n) for j in range(i + 1, n)], dtype=np.float64)
