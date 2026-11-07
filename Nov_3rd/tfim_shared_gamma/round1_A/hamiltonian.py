"""Open-chain power-law TFIM, K feature, and entropy-maximizing field.

H = Hc + B sum_i X_i
Hc = sum_{i<j} J_ij Z_i Z_j
J_ij = |i-j|^(-alpha), alpha=1.1, J_0=1 (nearest neighbor is 1; not rescaled).

B is chosen separately at each N by the half-chain von Neumann entropy of the
full-space ground state. The search is the deterministic contract grid
[-0.8, -0.1] step 0.005, expanded if the maximum sits on an endpoint, then
refined to 1e-4 inside the neighboring coarse interval.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

import numpy as np
from scipy.sparse.linalg import LinearOperator, eigsh

from .common import (
    ALPHA,
    B_COARSE_HI,
    B_COARSE_LO,
    B_COARSE_STEP,
    B_REFINE_TOL,
    ENTROPY_TIE_BITS,
    J0,
    RANDOM_K_SEED,
    REFL_K_SEED,
    sha256_bytes,
    sha256_state,
)


def build_J(n: int, alpha: float = ALPHA, J0: float = J0) -> np.ndarray:
    J = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            J[i, j] = J[j, i] = J0 / (float(j - i) ** float(alpha))
    return J


def physical_K(J: np.ndarray) -> np.ndarray:
    """K_jk = sum_{i != j,k} J_ij J_ik, zero diagonal.

    With J_ii = 0 this is exactly the off-diagonal part of J @ J.
    """
    J2 = J @ J
    K = np.array(J2, dtype=np.float64, copy=True)
    np.fill_diagonal(K, 0.0)
    return K


def upper_rms(A: np.ndarray) -> float:
    iu = np.triu_indices(A.shape[0], k=1)
    v = np.asarray(A, dtype=np.float64)[iu]
    return float(np.sqrt(np.mean(np.square(v))))


def normalize_like_J(K: np.ndarray, rms_J: float) -> tuple[np.ndarray, float]:
    """Return sK * K with the same upper-triangle RMS as J, and sK."""
    r = upper_rms(K)
    if r <= 0:
        raise ValueError("K RMS is zero")
    sK = float(rms_J / r)
    return sK * K, sK


def upper_cosine(A: np.ndarray, B: np.ndarray) -> float:
    iu = np.triu_indices(A.shape[0], k=1)
    a = A[iu]
    b = B[iu]
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def random_symmetric_K(n: int, seed: int, reflect: bool) -> np.ndarray:
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(n, n))
    K = 0.5 * (a + a.T)
    np.fill_diagonal(K, 0.0)
    if reflect:
        R = np.arange(n)[::-1]
        K = 0.5 * (K + K[np.ix_(R, R)])
        np.fill_diagonal(K, 0.0)
    return K


def zz_diag(n: int, coeff: np.ndarray) -> np.ndarray:
    """Diagonal of sum_{i<j} coeff_ij Z_i Z_j. Z|0>=+1, Z|1>=-1, site 0 is LSB."""
    dim = 1 << n
    idx = np.arange(dim, dtype=np.int64)
    z = np.empty((n, dim), dtype=np.float64)
    for i in range(n):
        bit = (idx >> i) & 1
        z[i] = 1.0 - 2.0 * bit
    diag = np.zeros(dim, dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            cij = float(coeff[i, j])
            if cij != 0.0:
                diag += cij * z[i] * z[j]
    return diag


def apply_X_sum(psi: np.ndarray, n: int) -> np.ndarray:
    """(sum_i X_i) |psi>. Bit i is axis i with site 0 the LSB."""
    out = np.zeros_like(psi)
    dim = psi.shape[0]
    for i in range(n):
        step = 1 << i
        block = psi.reshape(dim // (2 * step), 2, step)
        flipped = block[:, ::-1, :].reshape(dim)
        out += flipped
    return out


def reflection_perm(n: int) -> np.ndarray:
    """Site i <-> site n-1-i. perm[x] = R(x), and perm is an involution."""
    x = np.arange(1 << n, dtype=np.int64)
    y = np.zeros_like(x)
    for i in range(n):
        bit = (x >> i) & 1
        y |= bit << (n - 1 - i)
    return y


def parity_flip_perm(n: int) -> np.ndarray:
    """Basis permutation of P = prod X_i: y = x with all bits flipped."""
    return np.arange(1 << n, dtype=np.int64) ^ ((1 << n) - 1)


def apply_bit_perm(psi: np.ndarray, perm: np.ndarray) -> np.ndarray:
    """(U psi)[y] = psi[U y] for a real involution that permutes basis states."""
    return psi[perm]


def shell_weights(n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """X-basis phase weights so phi = c1 w1 + c2 w2 + c3 w3.

    w1 = sum_cavities (3 - 2 pop)
    w2 = sum_cavities ((3-2pop)^2 - 3)/2
    w3 = sum_cavities (-1)^pop
    """
    if n % 3 != 0:
        raise ValueError(f"n={n} is not a multiple of 3")
    dim = 1 << n
    ncav = n // 3
    idx = np.arange(dim, dtype=np.int64)
    c1_table = np.array([3.0, 1.0, -1.0, -3.0])
    c2_table = np.array([3.0, -1.0, -1.0, 3.0])
    c3_table = np.array([1.0, -1.0, 1.0, -1.0])
    w1 = np.zeros(dim, dtype=np.float64)
    w2 = np.zeros(dim, dtype=np.float64)
    w3 = np.zeros(dim, dtype=np.float64)
    for a in range(ncav):
        bits = (idx >> (3 * a)) & 7
        pop = (bits & 1) + ((bits >> 1) & 1) + ((bits >> 2) & 1)
        pop = pop.astype(np.int64)
        w1 += c1_table[pop]
        w2 += c2_table[pop]
        w3 += c3_table[pop]
    return w1, w2, w3


def half_chain_entropy(gs: np.ndarray, n: int) -> tuple[float, float]:
    """Return (S_log2, S_ln) of the contiguous cut sites [0, n/2) vs the rest."""
    nA = n // 2
    amp = np.asarray(gs, dtype=np.complex128)
    amp = amp / np.linalg.norm(amp)
    mat = amp.reshape((1 << (n - nA), 1 << nA))
    svals = np.linalg.svd(mat, compute_uv=False)
    p = np.square(svals.real if np.isrealobj(svals) else np.abs(svals))
    p = p[p > 1e-16]
    p = p / p.sum()
    S_ln = float(-np.sum(p * np.log(p)))
    return S_ln / float(np.log(2.0)), S_ln


def _matvec_factory(hc: np.ndarray, B: float, n: int):
    def mv(v: np.ndarray) -> np.ndarray:
        v = np.asarray(v, dtype=np.float64)
        if v.ndim == 2:
            cols = [
                hc * v[:, i] + B * apply_X_sum(v[:, i], n) for i in range(v.shape[1])
            ]
            return np.column_stack(cols)
        return hc * v + B * apply_X_sum(v, n)

    return mv


def _eigs(hc: np.ndarray, B: float, n: int, which: str, k: int, v0: np.ndarray):
    dim = hc.shape[0]
    op = LinearOperator((dim, dim), matvec=_matvec_factory(hc, B, n), dtype=np.float64)
    ncv = min(dim - 1, max(8 * k + 8, 48))
    w, vecs = eigsh(
        op, k=k, which=which, v0=v0, ncv=ncv, tol=1e-12, maxiter=8000
    )
    order = np.argsort(w)
    if which == "LA":
        order = order[::-1]
    return np.asarray(w[order], dtype=np.float64), np.asarray(vecs[:, order], dtype=np.float64)


def _gauge_gs(vec: np.ndarray) -> np.ndarray:
    v = np.asarray(vec, dtype=np.float64)
    if np.max(np.abs(np.imag(v))) > 1e-8:
        raise RuntimeError("ground-state vector is not real")
    v = np.real(v)
    v = v / np.linalg.norm(v)
    if float(np.sum(v)) < 0:
        v = -v
    return v


def spectrum_at_B(hc: np.ndarray, B: float, n: int, v0_sa: np.ndarray, v0_la: np.ndarray):
    """Full-space GS, first gap, and highest eigenvalue. No sector projection."""
    w_lo, v_lo = _eigs(hc, B, n, "SA", 2, v0_sa)
    w_hi, _v_hi = _eigs(hc, B, n, "LA", 1, v0_la)
    gs = _gauge_gs(v_lo[:, 0])
    mv = _matvec_factory(hc, B, n)
    Hgs = mv(gs)
    E0 = float(np.dot(gs, Hgs))
    residual = float(np.linalg.norm(Hgs - E0 * gs))
    if residual > 1e-8:
        # One harder retry from the current vector.
        w_lo, v_lo = _eigs(hc, B, n, "SA", 3, gs)
        gs = _gauge_gs(v_lo[:, 0])
        Hgs = mv(gs)
        E0 = float(np.dot(gs, Hgs))
        residual = float(np.linalg.norm(Hgs - E0 * gs))
    E1 = float(w_lo[1])
    # Prefer the Rayleigh quotient for E0; keep E1 from the Ritz value but
    # recompute it if the returned SA value disagrees with a second vector.
    gap = float(E1 - E0)
    Emax = float(w_hi[0])
    S2, Sln = half_chain_entropy(gs, n)
    return {
        "E0": E0,
        "E1": E1,
        "gap": gap,
        "Emax": Emax,
        "S_log2": S2,
        "S_ln": Sln,
        "residual": residual,
        "gs": gs,
    }


def dense_spectrum(hc: np.ndarray, B: float, n: int):
    """Exact dense eigh. Used to certify N=6 and any small diagnostic."""
    dim = 1 << n
    H = np.zeros((dim, dim), dtype=np.float64)
    mv = _matvec_factory(hc, B, n)
    eye = np.eye(dim)
    for i in range(dim):
        H[:, i] = mv(eye[:, i])
    H = 0.5 * (H + H.T)
    w, vecs = np.linalg.eigh(H)
    gs = _gauge_gs(vecs[:, 0])
    S2, Sln = half_chain_entropy(gs, n)
    return {
        "E0": float(w[0]),
        "E1": float(w[1]),
        "gap": float(w[1] - w[0]),
        "Emax": float(w[-1]),
        "S_log2": S2,
        "S_ln": Sln,
        "residual": float(np.linalg.norm(H @ gs - w[0] * gs)),
        "gs": gs,
        "evals": w,
    }


def _low_lying(hc, B, n, Rperm, Pperm, gs, k: int = 4) -> list[dict]:
    """Lowest states of the full Hamiltonian, with parity and reflection.

    This is a diagnostic of which excitations sit next to the ground state.
    The benchmark GS remains the full-space ground state.
    """
    k = min(k, (1 << n) - 1)
    if n <= 8:
        dim = 1 << n
        H = np.zeros((dim, dim), dtype=np.float64)
        mv = _matvec_factory(hc, B, n)
        for i in range(dim):
            e = np.zeros(dim)
            e[i] = 1.0
            H[:, i] = mv(e)
        H = 0.5 * (H + H.T)
        w, vecs = np.linalg.eigh(H)
    else:
        v0 = np.full(hc.shape[0], 1.0 / np.sqrt(hc.shape[0]))
        w, vecs = _eigs(hc, B, n, "SA", k, v0)
    rows = []
    for i in range(min(k, vecs.shape[1])):
        v = _gauge_gs(vecs[:, i])
        rows.append(
            {
                "index": i,
                "E": float(w[i]),
                "gap_from_E0": float(w[i] - w[0]),
                "parity": float(np.dot(v, v[Pperm])),
                "reflection": float(np.dot(v, v[Rperm])),
                "overlap_gs": float(abs(np.dot(v, gs))),
            }
        )
    accessible = [r for r in rows if r["parity"] > 0.5 and r["reflection"] > 0.5]
    gap_sector = None
    if len(accessible) >= 2:
        gap_sector = float(accessible[1]["E"] - accessible[0]["E"])
    return {"states": rows, "gap_within_PR_plus": gap_sector}


def _coarse_grid(lo: float, hi: float) -> np.ndarray:
    if hi < lo:
        return np.zeros(0, dtype=np.float64)
    nstep = int(round((hi - lo) / B_COARSE_STEP))
    return np.round(lo + np.arange(nstep + 1) * B_COARSE_STEP, 10)


def _entropy_curve(hc, n, grid, v0_sa, v0_la):
    rows = []
    for B in grid:
        # N<=8 is cheap to diagonalize exactly, so the frozen field does not
        # depend on ARPACK tolerances. Larger N uses a deterministic sparse solve.
        if n <= 8:
            spec = dense_spectrum(hc, float(B), n)
        else:
            spec = spectrum_at_B(hc, float(B), n, v0_sa, v0_la)
        rows.append(
            {
                "B": float(B),
                "E0": spec["E0"],
                "E1": spec["E1"],
                "gap": spec["gap"],
                "Emax": spec["Emax"],
                "S_log2": spec["S_log2"],
                "S_ln": spec["S_ln"],
                "residual": spec["residual"],
                "gs": spec["gs"],
            }
        )
        if spec["residual"] > 1e-7:
            raise RuntimeError(f"N={n} B={B} GS residual {spec['residual']}")
    return rows


def _argmax_with_tie(rows: list[dict], anchor: float | None) -> int:
    S = np.array([r["S_log2"] for r in rows])
    smax = float(S.max())
    cand = np.flatnonzero(S >= smax - ENTROPY_TIE_BITS)
    if anchor is None:
        return int(cand[0])
    best = int(cand[0])
    best_key = (abs(rows[best]["B"] - anchor), rows[best]["B"])
    for i in cand[1:]:
        key = (abs(rows[int(i)]["B"] - anchor), rows[int(i)]["B"])
        if key < best_key:
            best = int(i)
            best_key = key
    return best


def select_field(n: int) -> dict:
    """Deterministic entropy-max field. Returns metadata plus the GS vector."""
    J = build_J(n)
    hc = zz_diag(n, J)
    dim = 1 << n
    v0_sa = np.full(dim, 1.0 / np.sqrt(dim))
    rng = np.random.default_rng(0)
    v0_la = rng.normal(size=dim)
    v0_la /= np.linalg.norm(v0_la)

    grid = _coarse_grid(B_COARSE_LO, B_COARSE_HI)
    rows = _entropy_curve(hc, n, grid, v0_sa, v0_la)
    diagnosis = []
    expansions = 0
    while expansions < 4:
        imax = _argmax_with_tie(rows, None)
        at_lo = imax == 0
        at_hi = imax == len(rows) - 1
        if not (at_lo or at_hi):
            break
        expansions += 1
        if at_lo:
            new_lo = float(rows[0]["B"] - 0.5)
            extra_B = _coarse_grid(new_lo, rows[0]["B"])[:-1]
            diagnosis.append(
                f"coarse max at low endpoint B={rows[0]['B']}; expanded to {new_lo}"
            )
            extra = _entropy_curve(hc, n, extra_B, v0_sa, v0_la)
            rows = extra + rows
        else:
            new_hi = float(min(-B_REFINE_TOL, rows[-1]["B"] + 0.5))
            if new_hi <= rows[-1]["B"] + 0.5 * B_COARSE_STEP:
                diagnosis.append(
                    f"coarse max at high endpoint B={rows[-1]['B']} and B<0 blocks further expansion"
                )
                break
            extra_B = _coarse_grid(rows[-1]["B"], new_hi)[1:]
            diagnosis.append(
                f"coarse max at high endpoint B={rows[-1]['B']}; expanded to {new_hi}"
            )
            extra = _entropy_curve(hc, n, extra_B, v0_sa, v0_la)
            rows = rows + extra

    # Drop GS vectors from the stored curve; keep the chosen one.
    imax = _argmax_with_tie(rows, None)
    B_coarse = float(rows[imax]["B"])
    if imax == 0 or imax == len(rows) - 1:
        diagnosis.append(
            f"WARNING: entropy maximum still at endpoint after expansion, frozen B={B_coarse}"
        )
        fine_rows = [rows[imax]]
        B_frozen = B_coarse
    else:
        lo = float(rows[imax - 1]["B"])
        hi = float(rows[imax + 1]["B"])
        nstep = int(round((hi - lo) / B_REFINE_TOL))
        fine_B = np.round(lo + np.arange(nstep + 1) * B_REFINE_TOL, 10)
        fine_rows_full = _entropy_curve(hc, n, fine_B, v0_sa, v0_la)
        fidx = _argmax_with_tie(fine_rows_full, B_coarse)
        B_frozen = float(fine_rows_full[fidx]["B"])
        fine_rows = fine_rows_full

    chosen = None
    for r in fine_rows:
        if abs(r["B"] - B_frozen) < 5e-12:
            chosen = r
            break
    if chosen is None:
        raise RuntimeError("frozen B missing from the refined curve")

    # Certify N=6 against dense eigh. N=12 stays on the sparse full-space solve.
    dense_check = None
    if n <= 8:
        dense = dense_spectrum(hc, B_frozen, n)
        fid = float(np.dot(dense["gs"], chosen["gs"]) ** 2)
        dense_check = {
            "E0": dense["E0"],
            "gap": dense["gap"],
            "Emax": dense["Emax"],
            "S_log2": dense["S_log2"],
            "fidelity_sparse_vs_dense": fid,
            "E0_abs_diff": abs(dense["E0"] - chosen["E0"]),
        }
        if fid < 1 - 1e-8 or dense_check["E0_abs_diff"] > 1e-8:
            raise RuntimeError(f"N={n} sparse GS disagrees with dense eigh: {dense_check}")
        # Prefer the dense vector and eigenvalues as the frozen exact target.
        chosen = {
            **chosen,
            "E0": dense["E0"],
            "E1": dense["E1"],
            "gap": dense["gap"],
            "Emax": dense["Emax"],
            "S_log2": dense["S_log2"],
            "S_ln": dense["S_ln"],
            "residual": dense["residual"],
            "gs": dense["gs"],
        }

    gs = chosen["gs"]
    Rperm = reflection_perm(n)
    Pperm = parity_flip_perm(n)
    R_exp = float(np.dot(gs, gs[Rperm]))
    P_exp = float(np.dot(gs, gs[Pperm]))
    neg_mass = float(np.sum(np.square(gs[gs < 0])))
    low_lying = _low_lying(hc, B_frozen, n, Rperm, Pperm, gs)

    def _public(row, keep_gs=False):
        out = {k: row[k] for k in ("B", "E0", "E1", "gap", "Emax", "S_log2", "S_ln", "residual")}
        return out

    Svals = np.array([r["S_log2"] for r in rows])
    smax = float(Svals.max())
    plateau = [r["B"] for r in rows if r["S_log2"] >= smax - 1e-4]

    H_hash = hamiltonian_hash(n, B_frozen, J)
    return {
        "n": n,
        "alpha": ALPHA,
        "J0": J0,
        "B": B_frozen,
        "B_coarse": B_coarse,
        "E0": chosen["E0"],
        "E1": chosen["E1"],
        "gap": chosen["gap"],
        "Emax": chosen["Emax"],
        "S_log2": chosen["S_log2"],
        "S_ln": chosen["S_ln"],
        "residual": chosen["residual"],
        "parity_expectation": P_exp,
        "reflection_expectation": R_exp,
        "gs_negative_mass": neg_mass,
        "low_lying": low_lying,
        "H_hash": H_hash,
        "gs_sha256_f64": sha256_state(gs),
        "gs_sha256_round10": sha256_state(gs, decimals=10),
        "diagnosis": diagnosis,
        "endpoint_after_expansion": bool(imax == 0 or imax == len(rows) - 1),
        "coarse_plateau_1e-4_bits": plateau,
        "dense_check": dense_check,
        "coarse_curve": [_public(r) for r in rows],
        "fine_curve": [_public(r) for r in fine_rows],
        "J": J,
        "hc": hc,
        "gs": gs,
    }


def hamiltonian_hash(n: int, B: float, J: np.ndarray) -> str:
    """Cross-agent hash of the Hamiltonian inputs, not of the eigensolver output."""
    iu = np.triu_indices(n, k=1)
    payload = {
        "alpha": f"{ALPHA:.1f}",
        "J0": f"{J0:.1f}",
        "n": int(n),
        "B": f"{float(B):.10f}",
        "J_upper": [f"{float(v):.12e}" for v in J[iu]],
        "convention": "open chain ZZ + B X, LSB=site 0, Jij=|i-j|^-1.1",
    }
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()


def kappa_sites(J: np.ndarray) -> np.ndarray:
    return np.sum(J, axis=1)


def intra_inter_weight(coeff: np.ndarray) -> dict:
    n = coeff.shape[0]
    intra = 0.0
    inter = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            w = abs(float(coeff[i, j]))
            if i // 3 == j // 3:
                intra += w
            else:
                inter += w
    total = intra + inter
    return {
        "intra_abs": intra,
        "inter_abs": inter,
        "intra_fraction": intra / total if total else float("nan"),
    }


def local_energy_identity(J: np.ndarray, K: np.ndarray, n_samples: int = 0) -> dict:
    """Check sum_i e_i^2 = const + 2 F_K on classical configs.

    e_i = z_i sum_{j!=i} J_ij z_j, F_K = sum_{j<k} K_jk z_j z_k.
    Z_i^2 = I is what makes the diagonal of J@J a pure constant.
    """
    n = J.shape[0]
    const = float(np.sum(J * J))  # sum_i sum_{j!=i} J_ij^2 since diag J is 0
    dim = 1 << n
    if n <= 8:
        idx = np.arange(dim)
    else:
        rng = np.random.default_rng(1)
        idx = rng.choice(dim, size=min(n_samples or 256, dim), replace=False)
    max_abs = 0.0
    for x in idx:
        z = np.empty(n, dtype=np.float64)
        for i in range(n):
            z[i] = 1.0 - 2.0 * ((int(x) >> i) & 1)
        h = J @ z  # diag is 0
        sum_e2 = float(np.dot(h, h))
        F = 0.0
        for j in range(n):
            for k in range(j + 1, n):
                F += K[j, k] * z[j] * z[k]
        max_abs = max(max_abs, abs(sum_e2 - (const + 2.0 * F)))
    return {"const": const, "max_abs_identity_error": max_abs, "n_checked": int(len(idx))}


@dataclass
class FeatureSet:
    name: str
    K_raw: np.ndarray
    K_norm: np.ndarray
    sK: float
    cosine_with_J: float
    fk_diag: np.ndarray
    reflection_max_abs: float
    intra: dict


@dataclass
class Problem:
    n: int
    B: float
    J: np.ndarray
    hc_diag: np.ndarray
    gs: np.ndarray
    E0: float
    E1: float
    gap: float
    Emax: float
    S_log2: float
    S_ln: float
    residual: float
    H_hash: str
    parity_expectation: float
    reflection_expectation: float
    gs_negative_mass: float
    kappa_mean: float
    kappa_per_site: np.ndarray
    rms_J: float
    features: dict[str, FeatureSet]
    w1: np.ndarray
    w2: np.ndarray
    w3: np.ndarray
    Rperm: np.ndarray
    Pperm: np.ndarray
    plus: np.ndarray
    meta: dict = field(default_factory=dict)

    @property
    def dim(self) -> int:
        return 1 << self.n


def problem_at_fixed_B(n: int, B: float, use_dense: bool | None = None) -> Problem:
    """Build a problem at a caller-specified B. Used by validation, not as the frozen field."""
    J = build_J(n)
    hc = zz_diag(n, J)
    dim = 1 << n
    if use_dense is None:
        use_dense = n <= 8
    if use_dense:
        spec = dense_spectrum(hc, B, n)
    else:
        v0 = np.full(dim, 1.0 / np.sqrt(dim))
        rng = np.random.default_rng(0)
        v0_la = rng.normal(size=dim)
        v0_la /= np.linalg.norm(v0_la)
        spec = spectrum_at_B(hc, B, n, v0, v0_la)
    gs = spec["gs"]
    Rperm = reflection_perm(n)
    Pperm = parity_flip_perm(n)
    sel = {
        "n": n,
        "B": float(B),
        "E0": spec["E0"],
        "E1": spec["E1"],
        "gap": spec["gap"],
        "Emax": spec["Emax"],
        "S_log2": spec["S_log2"],
        "S_ln": spec["S_ln"],
        "residual": spec["residual"],
        "parity_expectation": float(np.dot(gs, gs[Pperm])),
        "reflection_expectation": float(np.dot(gs, gs[Rperm])),
        "gs_negative_mass": float(np.sum(np.square(gs[gs < 0]))),
        "H_hash": hamiltonian_hash(n, float(B), J),
        "gs_sha256_f64": sha256_state(gs),
        "gs_sha256_round10": sha256_state(gs, decimals=10),
        "diagnosis": ["fixed-B validation build, not the entropy-selected field"],
        "B_coarse": float(B),
        "endpoint_after_expansion": False,
        "coarse_plateau_1e-4_bits": [],
        "dense_check": None,
        "J": J,
        "hc": hc,
        "gs": gs,
    }
    return build_problem_from_selection(sel)


def build_problem_from_selection(sel: dict) -> Problem:
    n = int(sel["n"])
    J = np.asarray(sel["J"], dtype=np.float64)
    rmsJ = upper_rms(J)
    K_phys = physical_K(J)
    # off-diagonal identity
    J2 = J @ J
    if not np.allclose(K_phys, J2 - np.diag(np.diag(J2)), atol=1e-10):
        raise RuntimeError("K is not the off-diagonal of J@J")
    K_rand = random_symmetric_K(n, RANDOM_K_SEED, reflect=False)
    K_refl = random_symmetric_K(n, REFL_K_SEED, reflect=True)
    Rperm = reflection_perm(n)
    Ridx = np.arange(n)[::-1]

    def _refl_err(K: np.ndarray) -> float:
        return float(np.max(np.abs(K - K[np.ix_(Ridx, Ridx)])))

    def _feat(name: str, Kraw: np.ndarray) -> FeatureSet:
        Kn, sK = normalize_like_J(Kraw, rmsJ)
        return FeatureSet(
            name=name,
            K_raw=Kraw,
            K_norm=Kn,
            sK=sK,
            cosine_with_J=upper_cosine(J, Kraw),
            fk_diag=zz_diag(n, Kn),
            reflection_max_abs=_refl_err(Kraw),
            intra=intra_inter_weight(Kraw),
        )

    features = {
        "physical": _feat("physical", K_phys),
        "random": _feat("random", K_rand),
        "refl": _feat("refl", K_refl),
    }
    kapp = kappa_sites(J)
    return Problem(
        n=n,
        B=float(sel["B"]),
        J=J,
        hc_diag=np.asarray(sel["hc"], dtype=np.float64),
        gs=np.asarray(sel["gs"], dtype=np.float64),
        E0=float(sel["E0"]),
        E1=float(sel["E1"]),
        gap=float(sel["gap"]),
        Emax=float(sel["Emax"]),
        S_log2=float(sel["S_log2"]),
        S_ln=float(sel["S_ln"]),
        residual=float(sel["residual"]),
        H_hash=str(sel["H_hash"]),
        parity_expectation=float(sel["parity_expectation"]),
        reflection_expectation=float(sel["reflection_expectation"]),
        gs_negative_mass=float(sel["gs_negative_mass"]),
        kappa_mean=float(np.mean(kapp)),
        kappa_per_site=kapp,
        rms_J=rmsJ,
        features=features,
        w1=(sw := shell_weights(n))[0],
        w2=sw[1],
        w3=sw[2],
        Rperm=Rperm,
        Pperm=parity_flip_perm(n),
        plus=np.full(1 << n, 2 ** (-0.5 * n), dtype=np.complex128),
        meta={
            "diagnosis": sel["diagnosis"],
            "B_coarse": sel["B_coarse"],
            "endpoint_after_expansion": sel["endpoint_after_expansion"],
            "coarse_plateau_1e-4_bits": sel["coarse_plateau_1e-4_bits"],
            "dense_check": sel["dense_check"],
            "gs_sha256_f64": sel["gs_sha256_f64"],
            "gs_sha256_round10": sel["gs_sha256_round10"],
            "J_intra": intra_inter_weight(J),
            "identity": local_energy_identity(J, K_phys, n_samples=256),
        },
    )


def feature_public(feat: FeatureSet) -> dict:
    return {
        "name": feat.name,
        "sK": feat.sK,
        "rms_raw": upper_rms(feat.K_raw),
        "rms_norm": upper_rms(feat.K_norm),
        "cosine_with_J": feat.cosine_with_J,
        "reflection_max_abs": feat.reflection_max_abs,
        "intra": feat.intra,
        "K_norm_sha256": sha256_bytes(np.ascontiguousarray(feat.K_norm).tobytes()),
    }


def selection_public(sel: dict) -> dict:
    """JSON-safe target record. Curves kept; vectors omitted (stored in npz)."""
    skip = {"J", "hc", "gs"}
    out = {}
    for k, v in sel.items():
        if k in skip:
            continue
        out[k] = v
    out["coupling_table"] = np.asarray(sel["J"]).tolist()
    out["kappa_per_site"] = kappa_sites(sel["J"]).tolist()
    out["kappa_mean"] = float(np.mean(out["kappa_per_site"]))
    out["hash_recipe"] = (
        "sha256 of canonical JSON {alpha,J0,n,B:.10f,J_upper:.12e,convention}; "
        "GS hash is sha256 of canonical-phase float64 [real; imag], plus a decimals=10 variant"
    )
    return out
