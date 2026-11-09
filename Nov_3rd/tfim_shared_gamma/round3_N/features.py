"""Contiguous block-mean M and Neel selector N. No ground-state input.

G = Hc + thetaK Khat + thetaM Mhat + thetaN Nhat.
Each hat multiplies the raw two-body matrix by RMS(J)/RMS(feature), and both
RMS values run over every i<j slot, including zeros.

This generator does not contain round-2 S. S used inter-only K and a
multiplicative weight on the raw intra matrix. N is only the antisymmetric
piece of that intra matrix, and K here is the full K.
"""
from __future__ import annotations

import hashlib

import numpy as np

from Nov_3rd.tfim_shared_gamma.round1_A.hamiltonian import zz_diag
from Nov_3rd.tfim_shared_gamma.round2_S.frozen import (
    MANIFEST_SHA256,
    FrozenSystem,
    assert_manifest,
    build_system,
    physical_K,
    upper_rms,
)

SHUFFLE_SEED = 20261109


def _scale(raw: np.ndarray, rms_J: float) -> tuple[np.ndarray, float]:
    r = upper_rms(raw)
    if r <= 0:
        raise RuntimeError("feature RMS is zero")
    s = float(rms_J / r)
    return s * raw, s


def selector_matrix(n: int, kind: str) -> np.ndarray:
    """Intra selector. kind 'neel' is Z0Z1+Z1Z2-2 Z0Z2. kind 'wrong' swaps the far bond."""
    C = np.zeros((n, n), dtype=np.float64)
    for a in range(n // 3):
        s = 3 * a
        if kind == "neel":
            bonds = ((s, s + 1, 1.0), (s + 1, s + 2, 1.0), (s, s + 2, -2.0))
        elif kind == "wrong":
            bonds = ((s, s + 1, 1.0), (s, s + 2, 1.0), (s + 1, s + 2, -2.0))
        else:
            raise KeyError(kind)
        for i, j, w in bonds:
            C[i, j] += w
            C[j, i] += w
    return C


def jbar(J: np.ndarray) -> np.ndarray:
    n = J.shape[0]
    m = n // 3
    out = np.zeros((m, m), dtype=np.float64)
    for a in range(m):
        ia = slice(3 * a, 3 * a + 3)
        for b in range(a + 1, m):
            block = J[ia, 3 * b : 3 * b + 3]
            out[a, b] = out[b, a] = float(np.mean(block))
    return out


def coupling_from_jbar(jb: np.ndarray, n: int) -> np.ndarray:
    C = np.zeros((n, n), dtype=np.float64)
    m = n // 3
    for a in range(m):
        for b in range(a + 1, m):
            w = float(jb[a, b])
            for i in range(3):
                for j in range(3):
                    ii = 3 * a + i
                    jj = 3 * b + j
                    C[ii, jj] = C[jj, ii] = w
    return C


def shuffled_jbar(jb: np.ndarray) -> dict:
    """Seeded permutation of cavity-pair weights, then reflection-averaged.

    The seed is fixed before any optimization. Averaging keeps the control
    reflection-symmetric. The result is not the physical Jbar.
    """
    m = jb.shape[0]
    pairs = [(a, b) for a in range(m) for b in range(a + 1, m)]
    vals = np.array([jb[a, b] for a, b in pairs], dtype=np.float64)
    perm = np.random.default_rng(SHUFFLE_SEED).permutation(len(pairs))
    raw = np.zeros_like(jb)
    for src, dst in enumerate(perm):
        a, b = pairs[int(dst)]
        raw[a, b] = raw[b, a] = float(vals[int(src)])
    rc = (m - 1) - np.arange(m)
    sym = 0.5 * (raw + raw[np.ix_(rc, rc)])
    dist = float(np.linalg.norm(sym[np.triu_indices(m, 1)] - jb[np.triu_indices(m, 1)]))
    return {"jbar": sym, "distance_to_physical": dist, "seed": SHUFFLE_SEED, "perm": perm.tolist()}


def intra_identity(J: np.ndarray) -> dict:
    """Hintra = jbar_in * sum_{i<j in a} ZiZj + c * N, for contiguous triples.

    jbar_in = (2 + 2^{-1.1})/3 and c = (1 - 2^{-1.1})/3. This uses the actual
    nearest-neighbor and distance-2 values stored in the manifest, and it is
    not a claim that S's generator equals N's.
    """
    n = J.shape[0]
    j2 = float(J[0, 2])
    alpha = (2.0 + j2) / 3.0
    c = (1.0 - j2) / 3.0
    sym = np.zeros_like(J)
    for a in range(n // 3):
        s = 3 * a
        for i, j in ((s, s + 1), (s + 1, s + 2), (s, s + 2)):
            sym[i, j] = sym[j, i] = alpha
    recon = sym + c * selector_matrix(n, "neel")
    # Compare only where the manifest intra couplings live, and require the
    # reconstructed matrix to match J on those bonds.
    err = 0.0
    for a in range(n // 3):
        s = 3 * a
        for i, j in ((s, s + 1), (s + 1, s + 2), (s, s + 2)):
            err = max(err, abs(recon[i, j] - J[i, j]))
    return {
        "j2": j2,
        "alpha": alpha,
        "c": c,
        "max_abs_error": err,
        "passed": err < 1e-12,
    }


def selector_eigenvalues(n: int = 3) -> dict:
    raw = selector_matrix(n, "neel")
    wrong = selector_matrix(n, "wrong")
    d = zz_diag(n, raw)
    w = zz_diag(n, wrong)
    # One cavity, the low 8 computational states.
    neel = {int(k): float(d[k]) for k in range(8)}
    wrong_e = {int(k): float(w[k]) for k in range(8)}
    expect = {0: 0.0, 7: 0.0, 2: -4.0, 5: -4.0, 1: 2.0, 3: 2.0, 4: 2.0, 6: 2.0}
    err = max(abs(neel[k] - expect[k]) for k in expect)
    return {
        "neel_fock_eigenvalues": neel,
        "wrong_fock_eigenvalues": wrong_e,
        "neel_matches_minus4_on_2_and_5": err < 1e-12,
        "max_abs_error": err,
    }


class Features:
    def __init__(self, system: FrozenSystem):
        self.system = system
        self.n = system.n
        self.rms_J = upper_rms(system.J)
        self.K_raw = physical_K(system.J)
        self.K_hat, self.sK = _scale(self.K_raw, self.rms_J)
        self.N_raw = selector_matrix(system.n, "neel")
        self.W_raw = selector_matrix(system.n, "wrong")
        self.intra_raw = np.array(system.J_intra, copy=True)
        self.N_hat, self.sN = _scale(self.N_raw, self.rms_J)
        self.W_hat, self.sW = _scale(self.W_raw, self.rms_J)
        self.intra_hat, self.sI = _scale(self.intra_raw, self.rms_J)
        jb = jbar(system.J)
        self.jbar = jb
        self.M_raw = coupling_from_jbar(jb, system.n)
        self.M_hat, self.sM = _scale(self.M_raw, self.rms_J)
        sh = shuffled_jbar(jb)
        self.shuffle = sh
        M_sh = coupling_from_jbar(sh["jbar"], system.n)
        # Match the physical M RMS, not a fresh RMS(J) scale, so thetaM is comparable.
        rM = upper_rms(self.M_hat)
        rS = upper_rms(M_sh)
        self.M_shuf_hat = M_sh * (rM / rS)
        self.hc = system.hc_full
        self.diag = {
            "K": zz_diag(system.n, self.K_hat),
            "M": zz_diag(system.n, self.M_hat),
            "N": zz_diag(system.n, self.N_hat),
            "W": zz_diag(system.n, self.W_hat),
            "I": zz_diag(system.n, self.intra_hat),
            "Ms": zz_diag(system.n, self.M_shuf_hat),
        }
        self.identity = intra_identity(system.J)
        self.eigs = selector_eigenvalues(3)
        if not self.identity["passed"] or not self.eigs["neel_matches_minus4_on_2_and_5"]:
            raise RuntimeError(f"feature identity failed {self.identity} {self.eigs}")

    def g_diag(self, spec: str, theta: dict[str, float]) -> np.ndarray:
        g = np.array(self.hc, copy=True)
        g = g + float(theta.get("K", 0.0)) * self.diag["K"]
        if spec == "shuf_jbar":
            g = g + float(theta.get("M", 0.0)) * self.diag["Ms"]
        else:
            g = g + float(theta.get("M", 0.0)) * self.diag["M"]
        if spec == "wrong_sel":
            g = g + float(theta.get("N", 0.0)) * self.diag["W"]
        elif spec == "full_intra":
            g = g + float(theta.get("N", 0.0)) * self.diag["I"]
        else:
            g = g + float(theta.get("N", 0.0)) * self.diag["N"]
        return g

    def partials(self, spec: str) -> dict[str, np.ndarray]:
        mkey = "Ms" if spec == "shuf_jbar" else "M"
        nkey = {"wrong_sel": "W", "full_intra": "I"}.get(spec, "N")
        return {"K": self.diag["K"], "M": self.diag[mkey], "N": self.diag[nkey]}


def level_spacings(feat: Features) -> dict:
    """Intra eigenvalues on cavity 0, and why one 2π shift is not an alias."""
    d = zz_diag(feat.n, feat.intra_raw)
    ev = {k: float(d[k]) for k in range(8)}
    # Two gaps inside the classical intra spectrum.
    gap_neel = abs(ev[2] - ev[1])  # Neel fock 2 versus a non-Neel weight-1 state
    gap_vac = abs(ev[0] - ev[1])
    ratio = gap_vac / gap_neel if gap_neel else float("nan")
    # A shift that returns the Neel gap to the same phase does not do so for the other gap
    # unless the ratio is an integer.
    slip = abs(ratio - round(ratio))
    return {
        "cavity0_intra_eigenvalues": ev,
        "gap_neel_vs_other_weight1": gap_neel,
        "gap_vacuum_vs_other_weight1": gap_vac,
        "ratio": ratio,
        "distance_of_ratio_to_nearest_integer": slip,
        "one_gap_shift_is_exact_alias": bool(slip < 1e-8),
        "note": (
            "A theta shift of 2π/(gamma*gap) for one intra spacing changes the "
            "phase of the other spacing by 2π*ratio, which is not a multiple of "
            "2π when the ratio is not an integer. The two spacings come from "
            "2+2*J_2 and 2*(1-J_2), with J_2=2^{-1.1}."
        ),
    }


def manifest_sha() -> str:
    digest = assert_manifest()
    if digest != MANIFEST_SHA256:
        raise RuntimeError(digest)
    return digest


def load_features(n: int) -> tuple[FrozenSystem, Features]:
    manifest_sha()
    system = build_system(n)
    return system, Features(system)


def low_eigenpairs(system: FrozenSystem, k: int = 8):
    """Full-space lowest states for post-hoc weights. Not used to train."""
    n = system.n
    dim = 1 << n
    H = np.diag(system.hc_full)
    idx = np.arange(dim)
    for bit in range(n):
        partner = idx ^ (1 << bit)
        mask = partner > idx
        a = idx[mask]
        b = partner[mask]
        H[a, b] = system.B
        H[b, a] = system.B
    evals, evecs = np.linalg.eigh(H)
    return evals[:k], evecs[:, :k]


def sha256_hc(system: FrozenSystem) -> str:
    raw = np.ascontiguousarray(system.hc_full, dtype="<f8").tobytes()
    return hashlib.sha256(raw).hexdigest()
