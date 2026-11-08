"""Logical target plus packing-dependent shells.

The Hamiltonian stays in the manifest basis. A packing only chooses which
three logical qubits share a cavity shell. States are mapped back by never
leaving that basis. A simultaneous relabeling of H and the state is used
only as an equivalence test.
"""

from __future__ import annotations

import hashlib
import json
from math import comb
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "FROZEN_TARGET_R2.json"
PACKING_FILE = ROOT / "packings.json"
EXPECTED_MANIFEST = "14f40daa135640c9eb18d4ff64d2608216f139b51d87e9ede2fcd20939851da9"

_I2 = np.eye(2, dtype=np.float64)
_X = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64)
_Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.float64)


def _kron3(slow, mid, fast):
    return np.kron(slow, np.kron(mid, fast))


G1 = _kron3(_I2, _I2, _X) + _kron3(_I2, _X, _I2) + _kron3(_X, _I2, _I2)
G2 = _kron3(_I2, _X, _X) + _kron3(_X, _I2, _X) + _kron3(_X, _X, _I2)
G3 = _kron3(_X, _X, _X)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_manifest():
    digest = sha256_file(MANIFEST)
    if digest != EXPECTED_MANIFEST:
        raise RuntimeError(f"manifest {digest} != {EXPECTED_MANIFEST}")
    raw = json.loads(MANIFEST.read_text())
    return raw, digest


def load_packings():
    raw_bytes = PACKING_FILE.read_bytes()
    digest = hashlib.sha256(raw_bytes).hexdigest()
    return json.loads(raw_bytes), digest


def system(n: int):
    raw, digest = load_manifest()
    block = raw["systems"][str(n)]
    J = np.zeros((n, n), dtype=np.float64)
    for i, j, s in block["Jij"]:
        J[int(i), int(j)] = J[int(j), int(i)] = float(s)
    z = z_table(n)
    hc = pair_diag(J, z)
    K = pair_K(J)
    rms_j = rms(upper(J))
    scale_k = rms_j / rms(upper(K))
    fk = pair_diag(K * scale_k, z)
    hc_hash = hashlib.sha256(np.ascontiguousarray(hc, dtype="<f8").tobytes()).hexdigest()
    return {
        "n": n,
        "B": float(block["B"]),
        "B_string": block["B"],
        "J": J,
        "hc": hc,
        "fk": fk,
        "K": K,
        "scale_K": float(scale_k),
        "rms_J": float(rms_j),
        "z": z,
        "manifest_sha256": digest,
        "hc_sha256": hc_hash,
    }


def z_table(n: int) -> np.ndarray:
    dim = 1 << n
    x = np.arange(dim, dtype=np.uint32)
    z = np.empty((n, dim), dtype=np.float64)
    for q in range(n):
        z[q] = 1.0 - 2.0 * ((x >> np.uint32(q)) & np.uint32(1)).astype(np.float64)
    return z


def upper(mat: np.ndarray) -> np.ndarray:
    n = mat.shape[0]
    return np.array([mat[i, j] for i in range(n) for j in range(i + 1, n)], dtype=np.float64)


def rms(v: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(v, dtype=np.float64) ** 2)))


def pair_K(J: np.ndarray) -> np.ndarray:
    K = J @ J
    K = np.array(K, dtype=np.float64, copy=True)
    np.fill_diagonal(K, 0.0)
    K = 0.5 * (K + K.T)
    np.fill_diagonal(K, 0.0)
    return K


def pair_diag(coeff: np.ndarray, z: np.ndarray) -> np.ndarray:
    n = coeff.shape[0]
    out = np.zeros(z.shape[1], dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            if coeff[i, j] != 0.0:
                out += coeff[i, j] * z[i] * z[j]
    return out


def block_mean_diag(J: np.ndarray, packing, z: np.ndarray):
    """Normalized sum_{a<b} Jbar_ab S_a S_b on this packing. RMS over all pair slots."""
    n = J.shape[0]
    m = len(packing)
    coeff = np.zeros((n, n), dtype=np.float64)
    jbar = []
    for a in range(m):
        for b in range(a + 1, m):
            s = 0.0
            for i in packing[a]:
                for j in packing[b]:
                    s += float(J[i, j])
            w = s / 9.0
            jbar.append({"a": a, "b": b, "Jbar": w})
            for i in packing[a]:
                for j in packing[b]:
                    coeff[i, j] = coeff[j, i] = w
    scale = rms(upper(J)) / rms(upper(coeff))
    return pair_diag(coeff * scale, z), float(scale), jbar


def packing(name: str, n: int):
    raw, digest = load_packings()
    if name.startswith("random"):
        idx = int(name.replace("random", ""))
        trips = raw["random12"][idx]
        if n != 12:
            raise KeyError("random packings are N=12")
    elif name == "stride4":
        trips = raw["stride4"][str(n)]
    else:
        trips = raw[name][str(n)]
    trips = [list(map(int, t)) for t in trips]
    _validate(trips, n)
    return trips, digest


def _validate(trips, n):
    flat = [q for t in trips for q in t]
    if sorted(flat) != list(range(n)) or any(len(t) != 3 for t in trips):
        raise RuntimeError(f"bad packing {trips}")


def triple_index(q0, q1, q2, n: int) -> np.ndarray:
    others = [q for q in range(n) if q not in (q0, q1, q2)]
    n_ext = 1 << (n - 3)
    ext = np.arange(n_ext, dtype=np.int64)
    base = np.zeros(n_ext, dtype=np.int64)
    for k, q in enumerate(others):
        base |= ((ext >> k) & 1) << q
    local = np.arange(8, dtype=np.int64)
    idx = np.broadcast_to(base, (8, n_ext)).copy()
    idx += ((local & 1) << q0)[:, None]
    idx += (((local >> 1) & 1) << q1)[:, None]
    idx += (((local >> 2) & 1) << q2)[:, None]
    return idx


def precompute(packing_trips, n: int):
    return [triple_index(t[0], t[1], t[2], n) for t in packing_trips]


def apply_indexed(psi, U, idx):
    block = psi[idx]
    out = np.empty_like(psi)
    out[idx] = U @ block
    return out


def shell_unitary(c1, c2, c3, with_grad=False):
    H = c1 * G1 + c2 * G2 + c3 * G3
    w, V = np.linalg.eigh(H)
    phase = np.exp(-1j * w)
    U = (V * phase) @ V.T
    if not with_grad:
        return U
    wp = w[:, None]
    wq = w[None, :]
    half = 0.5 * (wp - wq)
    mid = 0.5 * (wp + wq)
    sinc = np.ones_like(half)
    nz = np.abs(half) > 1e-8
    sinc[nz] = np.sin(half[nz]) / half[nz]
    phi = -1j * np.exp(-1j * mid) * sinc
    dUs = []
    for G in (G1, G2, G3):
        dtil = V.T @ G @ V
        dUs.append((V @ (phi * dtil)) @ V.T)
    return U, dUs


def reflection_closed(trips, n: int) -> bool:
    images = []
    for cav in trips:
        images.append(frozenset(n - 1 - q for q in cav))
    return set(images) == {frozenset(c) for c in trips}


def intra_bonds(trips):
    bonds = []
    for cav in trips:
        a, b, c = cav
        bonds.extend([(a, b), (a, c), (b, c)])
    return bonds


def sym_weight(psi: np.ndarray, trips) -> float:
    """Weight in the product of per-cavity S3-symmetric subspaces.

    This is a diagnostic. It is not an upper bound on circuit fidelity when G
    is not symmetric inside the cavity.
    """
    n = int(round(np.log2(psi.shape[0])))
    m = len(trips)
    dim = 1 << n
    x = np.arange(dim, dtype=np.int64)
    idx = np.zeros(dim, dtype=np.int64)
    for a, cav in enumerate(trips):
        pop = np.zeros(dim, dtype=np.int64)
        for q in cav:
            pop += (x >> q) & 1
        idx += pop * (4 ** a)
    acc = np.bincount(idx, weights=np.asarray(psi).real, minlength=4 ** m).astype(np.complex128)
    acc += 1j * np.bincount(idx, weights=np.asarray(psi).imag, minlength=4 ** m)
    den = np.ones(4 ** m, dtype=np.float64)
    for a in range(m):
        kk = (np.arange(4 ** m) // (4 ** a)) % 4
        den *= np.array([comb(3, int(k)) for k in kk], dtype=np.float64)
    return float(np.sum((np.abs(acc) ** 2) / den).real)


def relabel_maps(trips, n: int):
    """src[contiguous slot] = logical qubit. new_of_old maps logical index -> contiguous-frame index."""
    src = [q for cav in trips for q in cav]
    dim = 1 << n
    old = np.arange(dim, dtype=np.int64)
    new = np.zeros(dim, dtype=np.int64)
    for slot, logical in enumerate(src):
        new |= ((old >> logical) & 1) << slot
    inv = np.empty(dim, dtype=np.int64)
    inv[new] = old
    return src, new, inv


def permute_state(psi, map_index):
    out = np.empty_like(psi)
    out[map_index] = psi
    return out


def permute_J(J, src):
    n = len(src)
    out = np.zeros_like(J)
    for a in range(n):
        for b in range(n):
            out[a, b] = J[src[a], src[b]]
    return out
