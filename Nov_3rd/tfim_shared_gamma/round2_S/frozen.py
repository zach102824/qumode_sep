"""Load the exact frozen manifest. Do not regenerate J or choose a new field.

The raw file bytes are the authority. Decimal strings are parsed with float();
they are not replaced by a fresh power-law evaluation.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from Nov_3rd.tfim_shared_gamma.round1_A.hamiltonian import (
    apply_X_sum,
    half_chain_entropy,
    parity_flip_perm,
    reflection_perm,
    shell_weights,
    zz_diag,
)

MANIFEST_PATH = Path(__file__).resolve().parent / "FROZEN_TARGET_R2.json"
MANIFEST_SHA256 = "14f40daa135640c9eb18d4ff64d2608216f139b51d87e9ede2fcd20939851da9"
SHUFFLE_SEED = 20261108


def manifest_bytes() -> bytes:
    return MANIFEST_PATH.read_bytes()


def assert_manifest() -> str:
    raw = manifest_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != MANIFEST_SHA256:
        raise RuntimeError(f"frozen manifest hash {digest} != {MANIFEST_SHA256}")
    return digest


def _parse_J(n: int, triples: list) -> np.ndarray:
    J = np.zeros((n, n), dtype=np.float64)
    for i, j, val in triples:
        J[int(i), int(j)] = J[int(j), int(i)] = float(val)
    return J


def upper_rms(A: np.ndarray) -> float:
    """RMS over every i<j slot, including structural zeros."""
    iu = np.triu_indices(A.shape[0], k=1)
    v = np.asarray(A, dtype=np.float64)[iu]
    return float(np.sqrt(np.mean(np.square(v))))


def sha256_f64_le(arr: np.ndarray) -> str:
    raw = np.ascontiguousarray(arr, dtype="<f8").tobytes()
    return hashlib.sha256(raw).hexdigest()


def cavity_of(n: int) -> np.ndarray:
    return np.arange(n) // 3


def split_intra_inter(J: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = J.shape[0]
    cav = cavity_of(n)
    intra = np.zeros_like(J)
    inter = np.zeros_like(J)
    for i in range(n):
        for j in range(i + 1, n):
            if cav[i] == cav[j]:
                intra[i, j] = intra[j, i] = J[i, j]
            else:
                inter[i, j] = inter[j, i] = J[i, j]
    return intra, inter


def physical_K(J: np.ndarray) -> np.ndarray:
    K = J @ J
    K = np.array(K, dtype=np.float64, copy=True)
    np.fill_diagonal(K, 0.0)
    return K


def mask_inter(K: np.ndarray) -> np.ndarray:
    n = K.shape[0]
    cav = cavity_of(n)
    out = np.array(K, dtype=np.float64, copy=True)
    for i in range(n):
        for j in range(i + 1, n):
            if cav[i] == cav[j]:
                out[i, j] = out[j, i] = 0.0
    np.fill_diagonal(out, 0.0)
    return out


def reflection_symmetrize(M: np.ndarray) -> np.ndarray:
    n = M.shape[0]
    R = np.arange(n)[::-1]
    out = 0.5 * (M + M[np.ix_(R, R)])
    np.fill_diagonal(out, 0.0)
    return out


def shuffled_intra(J_intra: np.ndarray, n: int) -> dict:
    """Reflection-symmetric reassignment of the intra couplings.

    The multiset of intra values is permuted, then averaged with its reflection
    so the control stays in the accessible sector. The result is rescaled to the
    original intra RMS over the full upper triangle, zeros included. thetaIn=1
    on this matrix is therefore not the physical intra pattern.
    """
    pairs = []
    for a in range(n // 3):
        s = 3 * a
        pairs.extend([(s, s + 1), (s + 1, s + 2), (s, s + 2)])
    vals = np.array([J_intra[i, j] for i, j in pairs], dtype=np.float64)
    perm = np.random.default_rng(SHUFFLE_SEED).permutation(len(pairs))
    raw = np.zeros_like(J_intra)
    for src, dst in enumerate(perm):
        i, j = pairs[int(dst)]
        raw[i, j] = raw[j, i] = vals[int(src)]
    R = np.arange(n)[::-1]
    refl_err = float(np.max(np.abs(raw - raw[np.ix_(R, R)])))
    sym = reflection_symmetrize(raw)
    rms_phys = upper_rms(J_intra)
    rms_sym = upper_rms(sym)
    scale = 1.0 if rms_sym == 0 else rms_phys / rms_sym
    scaled = scale * sym
    diff = float(np.linalg.norm(scaled[np.triu_indices(n, 1)] - J_intra[np.triu_indices(n, 1)]))
    return {
        "matrix": scaled,
        "pre_sym_reflection_max_abs": refl_err,
        "scale_to_intra_rms": float(scale),
        "rms": upper_rms(scaled),
        "distance_to_physical_intra": diff,
        "seed": SHUFFLE_SEED,
        "note": (
            "Intra values are permuted, then reflection-averaged so the control "
            "does not leave the P=R=+1 sector by itself. RMS matches physical intra "
            "over all i<j slots, zeros included."
        ),
    }


def dense_spectrum(hc: np.ndarray, B: float, n: int, n_label: int = 12) -> dict:
    """Full-space dense spectrum. A symmetric start vector is not used."""
    dim = 1 << n
    H = np.diag(np.asarray(hc, dtype=np.float64))
    idx = np.arange(dim)
    for bit in range(n):
        partner = idx ^ (1 << bit)
        mask = partner > idx
        a = idx[mask]
        b = partner[mask]
        H[a, b] = B
        H[b, a] = B
    evals, evecs = np.linalg.eigh(H)
    Rperm = reflection_perm(n)
    Pperm = parity_flip_perm(n)
    rows = []
    for i in range(min(n_label, dim)):
        v = np.real(evecs[:, i])
        v = v / np.linalg.norm(v)
        if float(np.sum(v)) < 0:
            v = -v
        Hv = hc * v + B * apply_X_sum(v, n)
        rows.append(
            {
                "index": i,
                "E": float(evals[i]),
                "gap_from_lowest": float(evals[i] - evals[0]),
                "parity": float(np.dot(v, v[Pperm])),
                "reflection": float(np.dot(v, v[Rperm])),
                "residual": float(np.linalg.norm(Hv - evals[i] * v)),
                "negative_mass": float(np.sum(np.square(v[v < 0]))),
                "vector": v,
            }
        )
    gs = rows[0]["vector"]
    accessible = [r for r in rows if r["parity"] > 0.5 and r["reflection"] > 0.5]
    # Search beyond the labeled window if the partner is not in the first rows.
    # The full gap is always evals[1]-evals[0], independent of sector.
    gap_full = float(evals[1] - evals[0])
    gap_access = None
    if len(accessible) >= 2:
        gap_access = float(accessible[1]["E"] - accessible[0]["E"])
    elif rows[0]["parity"] > 0.5 and rows[0]["reflection"] > 0.5:
        # Walk further eigenvectors until the next accessible state.
        for i in range(n_label, dim):
            v = np.real(evecs[:, i])
            v = v / np.linalg.norm(v)
            p = float(np.dot(v, v[Pperm]))
            r = float(np.dot(v, v[Rperm]))
            if p > 0.5 and r > 0.5:
                gap_access = float(evals[i] - evals[0])
                accessible.append(
                    {
                        "index": i,
                        "E": float(evals[i]),
                        "gap_from_lowest": float(evals[i] - evals[0]),
                        "parity": p,
                        "reflection": r,
                    }
                )
                break
    S2, Sln = half_chain_entropy(gs, n)
    return {
        "E0": float(evals[0]),
        "E1": float(evals[1]),
        "gap_full": gap_full,
        "Emax": float(evals[-1]),
        "gap_accessible_PR_plus": gap_access,
        "S_log2": S2,
        "S_ln": Sln,
        "gs": gs,
        "gs_parity": rows[0]["parity"],
        "gs_reflection": rows[0]["reflection"],
        "gs_residual": rows[0]["residual"],
        "gs_negative_mass": rows[0]["negative_mass"],
        "lowest": [{k: r[k] for k in r if k != "vector"} for r in rows],
        "n_evals": int(dim),
        "method": "numpy.linalg.eigh full dense Hamiltonian, no sector projection and no symmetric start",
    }


@dataclass
class FrozenSystem:
    n: int
    B: float
    B_string: str
    J: np.ndarray
    J_intra: np.ndarray
    J_inter: np.ndarray
    J_intra_shuf: np.ndarray
    K_full: np.ndarray
    K_inter: np.ndarray
    K_full_norm: np.ndarray
    K_inter_norm: np.ndarray
    sK_full: float
    sK_inter: float
    hc_full: np.ndarray
    hc_intra: np.ndarray
    hc_inter: np.ndarray
    hc_intra_shuf: np.ndarray
    fk_full: np.ndarray
    fk_inter: np.ndarray
    spectrum: dict
    hc_diag_sha256: str
    manifest_sha256: str
    shuffle: dict
    w1: np.ndarray
    w2: np.ndarray
    w3: np.ndarray
    Rperm: np.ndarray
    Pperm: np.ndarray
    plus: np.ndarray
    ledger: dict

    @property
    def dim(self) -> int:
        return 1 << self.n

    @property
    def E0(self) -> float:
        return self.spectrum["E0"]

    @property
    def Emax(self) -> float:
        return self.spectrum["Emax"]

    @property
    def gs(self) -> np.ndarray:
        return self.spectrum["gs"]


def intra_is_not_shell(hc_intra: np.ndarray, n: int) -> dict:
    """Hamming-shell SNAP cannot represent intra ZZ when equal-weight states differ."""
    dim = 1 << n
    by_pop: dict[int, list[float]] = {}
    # One cavity's local diagonal, read from the global intra diagonal on states
    # that are vacuum outside cavity 0.
    for j in range(8):
        pop = int(j.bit_count())
        by_pop.setdefault(pop, []).append(float(hc_intra[j]))
    spreads = {str(k): float(max(v) - min(v)) for k, v in by_pop.items()}
    return {
        "popcount_spreads_cavity0": spreads,
        "not_a_hamming_shell": any(s > 1e-8 for s in spreads.values()),
        "n_distinct_fock_phases_cavity0": len({round(float(hc_intra[j]), 10) for j in range(8)}),
    }


def build_system(n: int) -> FrozenSystem:
    digest = assert_manifest()
    manifest = json.loads(manifest_bytes().decode("utf-8"))
    block = manifest["systems"][str(n)]
    if int(block["N"]) != n:
        raise RuntimeError("manifest N mismatch")
    B_string = str(block["B"])
    B = float(B_string)
    J = _parse_J(n, block["Jij"])
    J_intra, J_inter = split_intra_inter(J)
    K_full = physical_K(J)
    K_inter = mask_inter(K_full)
    rmsJ = upper_rms(J)
    rmsK = upper_rms(K_full)
    rmsKi = upper_rms(K_inter)
    sK_full = rmsJ / rmsK
    sK_inter = rmsJ / rmsKi
    K_full_norm = sK_full * K_full
    K_inter_norm = sK_inter * K_inter
    shuf = shuffled_intra(J_intra, n)
    hc_full = zz_diag(n, J)
    hc_intra = zz_diag(n, J_intra)
    hc_inter = zz_diag(n, J_inter)
    hc_shuf = zz_diag(n, shuf["matrix"])
    # Numerical identity: intra + inter = full.
    if np.max(np.abs(hc_intra + hc_inter - hc_full)) > 1e-10:
        raise RuntimeError("intra/inter split does not rebuild Hc")
    spec = dense_spectrum(hc_full, B, n)
    w1, w2, w3 = shell_weights(n)
    shell_info = intra_is_not_shell(hc_intra, n)
    m = n // 3
    ledger = {
        "k_body_max": 2,
        "intra_level_resolved_not_shell_snap": shell_info,
        "n_cavities": m,
        "n_cavity_pairs": m * (m - 1) // 2,
        "n_intra_bonds": int(np.count_nonzero(np.triu(J_intra, 1))),
        "n_inter_bonds": int(np.count_nonzero(np.triu(J_inter, 1))),
        "n_fk_inter_nonzero": int(np.count_nonzero(np.triu(K_inter, 1))),
        "applications_per_layer": {
            "inter_two_body_diagonal_profiles": 1,
            "fk_inter_shares_that_inter_diagonal": True,
            "intra_level_resolved_diagonals": m,
            "shared_hamming_shell_snap": 1,
            "note": (
                "Intra phases depend on the Fock index, not only the Hamming weight, "
                "and they differ by cavity because the local distances do. "
                "They are not the shared 4-phase shell. Shell SNAP remains the mixer only."
            ),
        },
        "trained_controls": {
            "shared_gamma": 1,
            "shared_thetaIn": 1,
            "shared_thetaK": 1,
            "shell_per_layer": 3,
        },
        "no_free_compile_claim": True,
    }
    return FrozenSystem(
        n=n,
        B=B,
        B_string=B_string,
        J=J,
        J_intra=J_intra,
        J_inter=J_inter,
        J_intra_shuf=shuf["matrix"],
        K_full=K_full,
        K_inter=K_inter,
        K_full_norm=K_full_norm,
        K_inter_norm=K_inter_norm,
        sK_full=float(sK_full),
        sK_inter=float(sK_inter),
        hc_full=hc_full,
        hc_intra=hc_intra,
        hc_inter=hc_inter,
        hc_intra_shuf=hc_shuf,
        fk_full=zz_diag(n, K_full_norm),
        fk_inter=zz_diag(n, K_inter_norm),
        spectrum=spec,
        hc_diag_sha256=sha256_f64_le(hc_full),
        manifest_sha256=digest,
        shuffle=shuf,
        w1=w1,
        w2=w2,
        w3=w3,
        Rperm=reflection_perm(n),
        Pperm=parity_flip_perm(n),
        plus=np.full(1 << n, 2 ** (-0.5 * n), dtype=np.complex128),
        ledger=ledger,
    )


def g_diag(system: FrozenSystem, method: str, theta_in: float, theta_k: float) -> np.ndarray:
    if method == "qaoa":
        return system.hc_full
    if method == "full_K_A":
        return system.hc_full + float(theta_k) * system.fk_full
    if method == "inter_K_A":
        # thetaIn fixed at 1. This is Hc + thetaK * FK_inter, not full-K A.
        return system.hc_full + float(theta_k) * system.fk_inter
    if method in ("family_S", "same_G_qaoa"):
        return (
            system.hc_inter
            + float(theta_in) * system.hc_intra
            + float(theta_k) * system.fk_inter
        )
    if method == "thetaK0":
        return system.hc_inter + float(theta_in) * system.hc_intra
    if method == "shuffled_intra":
        return (
            system.hc_inter
            + float(theta_in) * system.hc_intra_shuf
            + float(theta_k) * system.fk_inter
        )
    raise KeyError(method)


def g_partials(system: FrozenSystem, method: str) -> tuple[np.ndarray | None, np.ndarray | None]:
    """dg/dthetaIn, dg/dthetaK. None means that scalar is not a free parameter."""
    if method == "qaoa":
        return None, None
    if method == "full_K_A":
        return None, system.fk_full
    if method == "inter_K_A":
        return None, system.fk_inter
    if method in ("family_S", "same_G_qaoa", "shuffled_intra"):
        intra = system.hc_intra_shuf if method == "shuffled_intra" else system.hc_intra
        return intra, system.fk_inter
    if method == "thetaK0":
        return system.hc_intra, None
    raise KeyError(method)
