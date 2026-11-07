"""Independent stride-4 simulator. Logical shells and a packed physical circuit.

Bit 0 of a cavity is the first listed site. States are returned in the logical basis.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.linalg import expm

ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "FROZEN_TARGET_R2.json"
EXPECTED = "14f40daa135640c9eb18d4ff64d2608216f139b51d87e9ede2fcd20939851da9"

_I = np.eye(2, dtype=np.complex128)
_X = np.array([[0, 1], [1, 0]], dtype=np.complex128)
_Z = np.array([[1, 0], [0, -1]], dtype=np.float64)


def _on(bit: int) -> np.ndarray:
    mats = [_X if r == bit else _I for r in range(3)]
    return np.kron(np.kron(mats[2], mats[1]), mats[0])


G1 = _on(0) + _on(1) + _on(2)
G2 = _on(0) @ _on(1) + _on(0) @ _on(2) + _on(1) @ _on(2)
G3 = _on(0) @ _on(1) @ _on(2)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_manifest():
    digest = sha256_file(MANIFEST)
    if digest != EXPECTED:
        raise RuntimeError(digest)
    raw = json.loads(MANIFEST.read_text())
    return raw, digest


def upper(mat: np.ndarray) -> np.ndarray:
    n = mat.shape[0]
    return np.array([mat[i, j] for i in range(n) for j in range(i + 1, n)], dtype=np.float64)


def rms(v: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(v, dtype=np.float64) ** 2)))


def z_table(n: int) -> np.ndarray:
    x = np.arange(1 << n, dtype=np.uint32)
    z = np.empty((n, 1 << n), dtype=np.float64)
    for q in range(n):
        z[q] = 1.0 - 2.0 * ((x >> np.uint32(q)) & 1)
    return z


def pair_diag(coeff: np.ndarray, z: np.ndarray) -> np.ndarray:
    n = coeff.shape[0]
    out = np.zeros(z.shape[1], dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            if coeff[i, j] != 0.0:
                out += coeff[i, j] * z[i] * z[j]
    return out


def fk_matrix(J: np.ndarray) -> np.ndarray:
    K = np.array(J @ J, dtype=np.float64, copy=True)
    np.fill_diagonal(K, 0.0)
    K = 0.5 * (K + K.T)
    np.fill_diagonal(K, 0.0)
    return K


def block_mean_matrix(J: np.ndarray, packing) -> np.ndarray:
    n = J.shape[0]
    M = np.zeros((n, n), dtype=np.float64)
    m = len(packing)
    for a in range(m):
        for b in range(a + 1, m):
            acc = 0.0
            for i in packing[a]:
                for j in packing[b]:
                    acc += float(J[i, j])
            jbar = acc / 9.0
            for i in packing[a]:
                for j in packing[b]:
                    M[i, j] = M[j, i] = jbar
    return M


def parse_system(packing, bit_reverse=False):
    raw, digest = load_manifest()
    n = 12 if len(packing) == 4 else len(packing) * 3
    block = raw["systems"][str(n)]
    J = np.zeros((n, n), dtype=np.float64)
    for i, j, s in block["Jij"]:
        J[int(i), int(j)] = J[int(j), int(i)] = float(s)
    trips = []
    for cav in packing:
        sites = [int(q) for q in cav]
        trips.append(list(reversed(sites)) if bit_reverse else sites)
    flat = [q for c in trips for q in c]
    if sorted(flat) != list(range(n)):
        raise RuntimeError("packing is not a partition")
    z = z_table(n)
    hc = pair_diag(J, z)
    K = fk_matrix(J)
    rK = rms(upper(K)) / rms(upper(J))
    M = block_mean_matrix(J, trips)
    rM = rms(upper(M)) / rms(upper(J))
    return {
        "n": n,
        "B": float(block["B"]),
        "J": J,
        "packing": trips,
        "z": z,
        "hc": hc,
        "hc_sha256": hashlib.sha256(np.ascontiguousarray(hc, dtype="<f8").tobytes()).hexdigest(),
        "fk": pair_diag(K, z),
        "rK": float(rK),
        "khat": pair_diag(K, z) / rK,
        "M": M,
        "rM": float(rM),
        "mhat": pair_diag(M, z) / rM,
        "manifest_sha256": digest,
        "cav_index": [cavity_index(n, c) for c in trips],
    }


def cavity_index(n: int, sites: list[int]) -> np.ndarray:
    """index[exterior, local], local bit r = sites[r]."""
    free = [b for b in range(n) if b not in set(sites)]
    nloc = 1 << len(sites)
    index = np.empty((1 << len(free), nloc), dtype=np.int64)
    for f in range(index.shape[0]):
        base = 0
        for k, b in enumerate(free):
            if (f >> k) & 1:
                base |= 1 << b
        for local in range(nloc):
            ind = base
            for r, site in enumerate(sites):
                if (local >> r) & 1:
                    ind |= 1 << site
            index[f, local] = ind
    return index


def mixer(c1, c2, c3, with_grad=False):
    H = c1 * G1 + c2 * G2 + c3 * G3
    w, V = np.linalg.eigh(H)
    U = (V * np.exp(-1j * w)) @ V.conj().T
    if not with_grad:
        return U
    wp, wq = w[:, None], w[None, :]
    half = 0.5 * (wp - wq)
    mid = 0.5 * (wp + wq)
    sinc = np.ones_like(half)
    nz = np.abs(half) > 1e-8
    sinc[nz] = np.sin(half[nz]) / half[nz]
    phi = -1j * np.exp(-1j * mid) * sinc
    dUs = []
    for G in (G1, G2, G3):
        dtil = V.conj().T @ G @ V
        dUs.append((V @ (phi * dtil)) @ V.conj().T)
    return U, dUs


def apply_cav(psi, U, index):
    out = np.array(psi, copy=True)
    out[index] = out[index] @ U.T
    return out


def unpack(x, L):
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    return float(x[0]), float(x[1]), float(x[2]), x[3:].reshape(L, 3).copy()


def g_diag(sys, theta_k, theta_m):
    return sys["hc"] + float(theta_k) * sys["khat"] + float(theta_m) * sys["mhat"]


def logical_state(sys, L, x):
    gamma, thk, thm, shell = unpack(x, L)
    phase = np.exp(-1j * gamma * g_diag(sys, thk, thm))
    psi = np.full(1 << sys["n"], 1.0 / np.sqrt(1 << sys["n"]), dtype=np.complex128)
    for row in shell:
        psi = psi * phase
        U = mixer(float(row[0]), float(row[1]), float(row[2]))
        for index in sys["cav_index"]:
            psi = apply_cav(psi, U, index)
    return psi


def _basis_map(bit_image, n):
    dim = 1 << n
    out = np.zeros(dim, dtype=np.int64)
    for p, site in enumerate(bit_image):
        bit = (np.arange(dim) >> p) & 1
        out |= bit.astype(np.int64) << int(site)
    return out


def physical_state(sys, L, x, invert_map=False):
    """Vacuum, Hadamards, diagonal G, popcount shell, final Hadamards, mapped back."""
    n = sys["n"]
    dim = 1 << n
    gamma, thk, thm, shell = unpack(x, L)
    phys_to_log = np.empty(n, dtype=np.int64)
    for a, cav in enumerate(sys["packing"]):
        for r, site in enumerate(cav):
            phys_to_log[3 * a + r] = site
    logical_of = _basis_map(phys_to_log, n)
    g_log = g_diag(sys, thk, thm)
    g_phys = g_log[logical_of]
    phase = np.exp(-1j * gamma * g_phys)
    psi = np.zeros(dim, dtype=np.complex128)
    psi[0] = 1.0
    h = np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2)
    single = [cavity_index(n, [q]) for q in range(n)]

    def had(vec):
        out = np.array(vec, copy=True)
        for index in single:
            out[index] = out[index] @ h.T
        return out

    for row in shell:
        psi = had(psi)
        psi = psi * phase
        psi = had(psi)
        phi = np.empty(4)
        for k in range(4):
            s2 = ((3 - 2 * k) ** 2 - 3) / 2.0
            phi[k] = row[0] * (3 - 2 * k) + row[1] * s2 + row[2] * ((-1) ** k)
        idx = np.arange(dim)
        ph = np.zeros(dim)
        for a in range(len(sys["packing"])):
            bits = np.zeros(dim, dtype=np.int64)
            for r in range(3):
                bits += (idx >> (3 * a + r)) & 1
            ph += phi[bits]
        psi = psi * np.exp(-1j * ph)
    psi = had(psi)
    logical = np.empty_like(psi)
    if invert_map:
        logical[np.arange(dim)] = psi[logical_of]
    else:
        logical[logical_of] = psi
    return logical


def apply_sum_x(psi, n):
    acc = np.zeros_like(psi)
    for q in range(n):
        half = 1 << q
        x = psi.reshape(-1, 2, half)
        flipped = np.empty_like(x)
        flipped[:, 0, :] = x[:, 1, :]
        flipped[:, 1, :] = x[:, 0, :]
        acc += flipped.reshape(-1)
    return acc


def energy_var(psi, sys):
    n = sys["n"]
    Hpsi = sys["hc"] * psi + sys["B"] * apply_sum_x(psi, n)
    E = float(np.vdot(psi, Hpsi).real)
    second = float(np.vdot(Hpsi, Hpsi).real)
    return E, second - E * E


def logical_grad(sys, L, x):
    gamma, thk, thm, shell = unpack(x, L)
    G = g_diag(sys, thk, thm)
    phase = np.exp(-1j * gamma * G)
    psi = np.full(1 << sys["n"], 1.0 / np.sqrt(1 << sys["n"]), dtype=np.complex128)
    frames = []
    for ell, row in enumerate(shell):
        psi = psi * phase
        frames.append(("diag", psi))
        U, dUs = mixer(float(row[0]), float(row[1]), float(row[2]), True)
        for index in sys["cav_index"]:
            psi = apply_cav(psi, U, index)
        frames.append(("shell", psi, U, dUs, ell))
    E, var = energy_var(psi, sys)
    lam = sys["hc"] * psi + sys["B"] * apply_sum_x(psi, sys["n"])
    grad = np.zeros(3 + 3 * L)
    for kind, *rest in reversed(frames):
        if kind == "diag":
            psi_d = rest[0]
            im = np.imag(np.conj(lam) * psi_d)
            grad[0] += 2.0 * float(np.sum(G * im))
            grad[1] += 2.0 * float(np.sum((gamma * sys["khat"]) * im))
            grad[2] += 2.0 * float(np.sum((gamma * sys["mhat"]) * im))
            lam = np.exp(1j * gamma * G) * lam
        else:
            psi_s, U, dUs, ell = rest
            for k in range(3):
                acc = np.zeros_like(psi_s)
                Va = dUs[k] @ np.conjugate(U).T
                for index in sys["cav_index"]:
                    acc += apply_cav(psi_s, Va, index)
                grad[3 + 3 * ell + k] += 2.0 * float(np.real(np.vdot(lam, acc)))
            for index in sys["cav_index"]:
                lam = apply_cav(lam, np.conjugate(U).T, index)
    return psi, E, grad, var


def spectrum(sys):
    n = sys["n"]
    dim = 1 << n
    H = np.diag(sys["hc"].astype(np.float64))
    x = np.arange(dim)
    for q in range(n):
        H[x, x ^ (1 << q)] += sys["B"]
    evals, evecs = np.linalg.eigh(H)
    gs = evecs[:, 0]
    k = int(np.argmax(np.abs(gs)))
    gs = gs * np.exp(-1j * np.angle(gs[k]))
    if float(np.sum(gs.real)) < 0:
        gs = -gs
    return {"E0": float(evals[0]), "full_gap": float(evals[1] - evals[0]), "gs": gs, "evals": evals}


def fidelity(gs, psi) -> float:
    return float(abs(np.vdot(gs, psi)) ** 2)


def permute_joint(sys, perm):
    """perm[old] = new site. Returns a system on relabeled J and packing, plus the basis map old->new."""
    n = sys["n"]
    perm = np.asarray(perm, dtype=np.int64)
    J = np.zeros_like(sys["J"])
    for i in range(n):
        for j in range(n):
            J[perm[i], perm[j]] = sys["J"][i, j]
    packing = [[int(perm[q]) for q in cav] for cav in sys["packing"]]
    # rebuild from J directly
    z = z_table(n)
    hc = pair_diag(J, z)
    K = fk_matrix(J)
    rK = rms(upper(K)) / rms(upper(J))
    M = block_mean_matrix(J, packing)
    rM = rms(upper(M)) / rms(upper(J))
    new = {
        "n": n,
        "B": sys["B"],
        "J": J,
        "packing": packing,
        "z": z,
        "hc": hc,
        "fk": pair_diag(K, z),
        "rK": float(rK),
        "khat": pair_diag(K, z) / rK,
        "M": M,
        "rM": float(rM),
        "mhat": pair_diag(M, z) / rM,
        "cav_index": [cavity_index(n, c) for c in packing],
        "hc_sha256": "",
        "manifest_sha256": sys["manifest_sha256"],
    }
    dim = 1 << n
    old = np.arange(dim)
    new_index = np.zeros(dim, dtype=np.int64)
    for q in range(n):
        new_index |= ((old >> q) & 1).astype(np.int64) << int(perm[q])
    return new, new_index
