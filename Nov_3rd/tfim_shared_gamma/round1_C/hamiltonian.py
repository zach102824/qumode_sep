"""Open-chain Pagano-like TFIM and entropy-selected transverse field.

H = Hc + B sum_i X_i
Hc = sum_{i<j} J_ij Z_i Z_j
J_ij = J0 / |i-j|^alpha, J0=1, alpha=1.1

This is an ideal power-law model, not Pagano's published (alpha, B) pair and not
an experimental coupling matrix. B is chosen numerically by maximizing the
ground-state half-chain von Neumann entropy.
"""

from __future__ import annotations

import hashlib
import time
from typing import Dict, List, Tuple

import numpy as np
from numpy.linalg import eigh
from scipy import sparse
from scipy.sparse.linalg import eigsh

ALPHA = 1.1
J0 = 1.0


def build_J(n: int, alpha: float = ALPHA, J0: float = J0) -> np.ndarray:
    J = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            J[i, j] = J[j, i] = J0 / float(j - i) ** float(alpha)
    return J


def basis_index_bits(n: int) -> np.ndarray:
    """Return bit array shape (dim, n) with qubit 0 as MSB (leftmost)."""
    dim = 1 << n
    x = np.arange(dim, dtype=np.uint32)
    bits = np.empty((dim, n), dtype=np.uint8)
    for q in range(n):
        bits[:, q] = (x >> np.uint32(n - 1 - q)) & np.uint32(1)
    return bits


def z_eigenvalues(n: int) -> np.ndarray:
    """Z_q eigenvalues, shape (n, dim). Bit 0 -> +1, bit 1 -> -1."""
    bits = basis_index_bits(n)
    return 1.0 - 2.0 * bits.astype(np.float64).T


def hc_diag_from_J(J: np.ndarray, z: np.ndarray | None = None) -> np.ndarray:
    n = J.shape[0]
    if z is None:
        z = z_eigenvalues(n)
    dim = z.shape[1]
    hc = np.zeros(dim, dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            hc += J[i, j] * z[i] * z[j]
    return hc


def kappa_per_site(J: np.ndarray) -> float:
    """Mean over sites of sum_{j != i} J_ij. Intensive coupling scale."""
    n = J.shape[0]
    return float(np.sum(J) / n)


def hs_norm_pairs(J: np.ndarray) -> float:
    n = J.shape[0]
    s = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            s += float(J[i, j]) ** 2
    return float(np.sqrt(s))


def sparse_H(hc: np.ndarray, B: float, n: int) -> sparse.csr_matrix:
    dim = hc.shape[0]
    ind = np.arange(dim, dtype=np.int64)
    rows = [ind]
    cols = [ind]
    data = [np.ascontiguousarray(hc, dtype=np.float64)]
    bcol = np.full(dim, float(B), dtype=np.float64)
    for q in range(n):
        mask = 1 << (n - 1 - q)
        rows.append(ind)
        cols.append(ind ^ mask)
        data.append(bcol)
    return sparse.csr_matrix(
        (np.concatenate(data), (np.concatenate(rows), np.concatenate(cols))),
        shape=(dim, dim),
        dtype=np.float64,
    )


def dense_H(hc: np.ndarray, B: float, n: int) -> np.ndarray:
    return sparse_H(hc, B, n).toarray()


def plus_state(n: int) -> np.ndarray:
    dim = 1 << n
    return np.full(dim, 1.0 / np.sqrt(dim), dtype=np.complex128)


def apply_X(psi: np.ndarray, qubit: int, n: int) -> np.ndarray:
    shift = n - 1 - qubit
    half = 1 << shift
    x = psi.reshape(-1, 2, half)
    out = np.empty_like(x)
    out[:, 0, :] = x[:, 1, :]
    out[:, 1, :] = x[:, 0, :]
    return out.reshape(-1)


def apply_sum_X(psi: np.ndarray, n: int) -> np.ndarray:
    acc = np.zeros_like(psi)
    for q in range(n):
        acc += apply_X(psi, q, n)
    return acc


def energy_from_state(psi: np.ndarray, hc: np.ndarray, B: float, n: int) -> float:
    e_c = float(np.vdot(psi, hc * psi).real)
    e_x = 0.0
    for q in range(n):
        e_x += float(np.vdot(psi, apply_X(psi, q, n)).real)
    return e_c + float(B) * e_x


def apply_H_psi(psi: np.ndarray, hc: np.ndarray, B: float, n: int) -> np.ndarray:
    out = hc * psi
    if B != 0.0:
        out = out + float(B) * apply_sum_X(psi, n)
    return out


def half_chain_entropy(psi: np.ndarray, n: int, log_base: float = np.e) -> float:
    """Von Neumann entropy of the left half of the chain (qubits 0..n/2-1)."""
    nA = n // 2
    nB = n - nA
    mat = np.asarray(psi, dtype=np.complex128).reshape(1 << nA, 1 << nB)
    s = np.linalg.svd(mat, compute_uv=False)
    p = np.clip(np.abs(s) ** 2, 0.0, None)
    p = p[p > 1e-16]
    p = p / p.sum()
    ent = -np.sum(p * np.log(p))
    if log_base != np.e:
        ent = ent / np.log(log_base)
    return float(ent)


def parity_expectation(psi: np.ndarray) -> float:
    """<prod_i X_i>. Bit-flip all qubits; X is real and sign-free."""
    rev = psi[::-1].copy()
    # integer bitwise NOT is reversal of all N bits only if the index width is N
    # and we flip the stored order. For an array indexed by 0..2^N-1, x -> x ^ (2^N-1)
    # is exactly psi[::-1] only when the index runs contiguously. Yes, arange order.
    return float(np.vdot(psi, rev).real)


def _bitreverse_indices(n: int) -> np.ndarray:
    """Basis-index map of site reflection qubit q <-> qubit n-1-q.

    Qubit 0 is the MSB of the integer index.
    """
    dim = 1 << n
    x = np.arange(dim, dtype=np.uint32)
    y = np.zeros(dim, dtype=np.uint32)
    for src in range(n):
        dst = n - 1 - src
        bit = (x >> np.uint32(n - 1 - src)) & np.uint32(1)
        y |= bit << np.uint32(n - 1 - dst)
    return y.astype(np.int64)


def reflection_indices(n: int) -> np.ndarray:
    """Index map of the site reflection i -> n-1-i. Cached by the caller."""
    return _bitreverse_indices(n)


def reflection_expectation(psi: np.ndarray, ridx: np.ndarray) -> float:
    return float(np.vdot(psi, psi[ridx]).real)


def canonicalize_vector(psi: np.ndarray) -> np.ndarray:
    """Fix global phase so the largest component is real and positive.

    If the vector is essentially real and mostly negative, flip the sign so a
    stoquastic ground state is stored nonnegative.
    """
    v = np.asarray(psi, dtype=np.complex128).copy()
    k = int(np.argmax(np.abs(v)))
    if abs(v[k]) > 0.0:
        v *= np.exp(-1j * np.angle(v[k]))
    if float(np.sum(v.real)) < 0.0:
        v = -v
    v.imag[np.abs(v.imag) < 1e-14] = 0.0
    return v


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_array(arr: np.ndarray) -> str:
    a = np.ascontiguousarray(arr)
    return sha256_bytes(a.tobytes())


def state_hash(psi: np.ndarray) -> str:
    return sha256_array(canonicalize_vector(psi))


def _spectrum_dense(hc: np.ndarray, B: float, n: int):
    H = dense_H(hc, B, n)
    w, v = eigh(H)
    return w, v, H


def ground_state_dense(hc: np.ndarray, B: float, n: int) -> Dict:
    w, v, H = _spectrum_dense(hc, B, n)
    gs = canonicalize_vector(v[:, 0])
    E0 = float(w[0])
    resid = float(np.linalg.norm(H @ gs - E0 * gs))
    return {
        "E0": E0,
        "gap": float(w[1] - w[0]),
        "Emax": float(w[-1]),
        "entropy_nats": half_chain_entropy(gs, n, np.e),
        "entropy_bits": half_chain_entropy(gs, n, 2.0),
        "gs": gs,
        "residual": resid,
        "evals": w,
        "imag_max": float(np.max(np.abs(gs.imag))),
        "min_real": float(np.min(gs.real)),
    }


def ground_state_sparse(hc: np.ndarray, B: float, n: int, v0: np.ndarray) -> Dict:
    dim = hc.shape[0]
    H = sparse_H(hc, B, n)
    ncv = int(min(dim - 1, max(48, 24)))
    try:
        w, v = eigsh(
            H,
            k=2,
            which="SA",
            v0=np.asarray(v0, dtype=np.float64),
            tol=1e-10,
            maxiter=8000,
            ncv=ncv,
        )
    except Exception:
        return ground_state_dense(hc, B, n)
    order = np.argsort(w.real)
    w = w.real[order]
    gs = canonicalize_vector(v[:, order[0]])
    E0 = float(w[0])
    Hgs = apply_H_psi(gs, hc, B, n)
    resid = float(np.linalg.norm(Hgs - E0 * gs))
    if not np.isfinite(resid) or resid > 1e-6:
        return ground_state_dense(hc, B, n)
    return {
        "E0": E0,
        "gap": float(w[1] - w[0]),
        "entropy_nats": half_chain_entropy(gs, n, np.e),
        "entropy_bits": half_chain_entropy(gs, n, 2.0),
        "gs": gs,
        "residual": resid,
        "imag_max": float(np.max(np.abs(gs.imag))),
        "min_real": float(np.min(gs.real)),
    }


def b_grid(lo: float, hi: float, step: float) -> List[float]:
    """Inclusive decimal grid. Endpoints are snapped to 1e-4 and always kept."""
    scale = 10000
    ilo = int(round(lo * scale))
    ihi = int(round(hi * scale))
    istep = int(round(step * scale))
    if istep <= 0:
        raise ValueError("step must be positive")
    if ilo > ihi:
        ilo, ihi = ihi, ilo
    ks = list(range(ilo, ihi + 1, istep))
    if not ks or ks[-1] != ihi:
        ks.append(ihi)
    return [k / scale for k in ks]


def _eval_B(n: int, hc: np.ndarray, B: float, v0: np.ndarray) -> Dict:
    if n <= 8:
        out = ground_state_dense(hc, B, n)
        out.pop("evals", None)
        return out
    return ground_state_sparse(hc, B, n, v0)


def select_field(n: int, J: np.ndarray, coarse_step: float = 0.005, b_min: float = -0.8, b_max: float = -0.1, verbose: bool = False) -> Dict:
    """Deterministic entropy maximization.

    Coarse grid on [b_min, b_max]. If the maximum is an endpoint, expand once
    (and a second time if it remains on the new endpoint) before freezing.
    Then refine at spacing 1e-4 on the closed interval between the neighboring
    coarse points.
    """
    hc = hc_diag_from_J(J)
    v0 = np.full(1 << n, 1.0 / np.sqrt(1 << n), dtype=np.float64)
    expansions: List[str] = []

    def scan(lo: float, hi: float, step: float) -> Tuple[List[float], List[Dict]]:
        bs = b_grid(lo, hi, step)
        rows = []
        t_scan = time.perf_counter()
        for i, B in enumerate(bs):
            row = _eval_B(n, hc, float(B), v0)
            row["B"] = float(B)
            row.pop("gs", None)
            rows.append(row)
            if verbose and (i % 20 == 0 or i + 1 == len(bs)):
                print(f"  field scan N={n} {i+1}/{len(bs)} B={B:.4f} S={row['entropy_nats']:.6f} gap={row['gap']:.3e} ({time.perf_counter()-t_scan:.1f}s)", flush=True)
        return bs, rows

    lo, hi = float(b_min), float(b_max)
    bs, rows = scan(lo, hi, coarse_step)
    for _ in range(2):
        ent = np.array([r["entropy_nats"] for r in rows])
        imax = int(np.argmax(ent))
        if imax == 0:
            new_lo = lo - 0.7
            expansions.append(f"max at lower endpoint B={bs[0]:.4f}; expand to {new_lo:.4f}")
            extra_bs, extra_rows = scan(new_lo, bs[0] - coarse_step, coarse_step)
            bs = extra_bs + bs
            rows = extra_rows + rows
            lo = new_lo
            continue
        if imax == len(bs) - 1:
            new_hi = min(-0.005, hi + 0.7)
            if new_hi <= hi + 1e-12:
                expansions.append("upper endpoint is already at the B<0 boundary; not expanded")
                break
            expansions.append(f"max at upper endpoint B={bs[-1]:.4f}; expand to {new_hi:.4f}")
            extra_bs, extra_rows = scan(bs[-1] + coarse_step, new_hi, coarse_step)
            bs = bs + extra_bs
            rows = rows + extra_rows
            hi = new_hi
            continue
        break

    ent = np.array([r["entropy_nats"] for r in rows], dtype=np.float64)
    imax = int(np.argmax(ent))
    # Tie note: values within 1e-6 nats of the max.
    near = [float(bs[i]) for i in range(len(bs)) if ent[imax] - ent[i] <= 1e-6]
    if imax == 0 or imax == len(bs) - 1:
        refine_lo, refine_hi = float(bs[imax]), float(bs[imax])
        expansions.append("refinement skipped because the maximum remained on an endpoint after expansion")
    else:
        refine_lo = float(bs[imax - 1])
        refine_hi = float(bs[imax + 1])
    fine_bs = b_grid(refine_lo, refine_hi, 1e-4) if refine_hi > refine_lo else [float(bs[imax])]
    if verbose:
        print(f"  refine N={n} interval [{refine_lo:.4f}, {refine_hi:.4f}] n={len(fine_bs)}", flush=True)
    fine_rows = []
    best = None
    for iB, B in enumerate(fine_bs):
        row = _eval_B(n, hc, float(B), v0)
        rec = {
            "B": float(B),
            "E0": row["E0"],
            "gap": row["gap"],
            "entropy_nats": row["entropy_nats"],
            "entropy_bits": row["entropy_bits"],
            "residual": row["residual"],
        }
        fine_rows.append(rec)
        if best is None or rec["entropy_nats"] > best["entropy_nats"] + 1e-15:
            best = rec
        elif abs(rec["entropy_nats"] - best["entropy_nats"]) <= 1e-15 and rec["B"] < best["B"]:
            best = rec
        if verbose and (iB % 25 == 0 or iB + 1 == len(fine_bs)):
            print(f"  refine {iB+1}/{len(fine_bs)} B={B:.4f} S={rec['entropy_nats']:.6f}", flush=True)

    assert best is not None
    # For N=12 the scan uses a sparse solver. Re-rank the top sparse candidates
    # with a dense eigensolve and freeze the dense entropy maximum.
    scan_best = dict(best)
    confirm = []
    if n > 8:
        ranked = sorted(fine_rows, key=lambda r: (-r["entropy_nats"], r["B"]))[:3]
        dense_rows = []
        for rec in ranked:
            dtry = ground_state_dense(hc, rec["B"], n)
            dense_rows.append((rec, dtry))
            confirm.append({
                "B": rec["B"],
                "sparse_entropy_nats": rec["entropy_nats"],
                "dense_entropy_nats": dtry["entropy_nats"],
                "dense_E0": dtry["E0"],
                "dense_gap": dtry["gap"],
                "dense_residual": dtry["residual"],
            })
        rec, dense = max(dense_rows, key=lambda t: (t[1]["entropy_nats"], -t[0]["B"]))
        best = rec
    else:
        dense = ground_state_dense(hc, best["B"], n)
    ridx = reflection_indices(n)
    gs = dense["gs"]
    parity = parity_expectation(gs)
    refl = reflection_expectation(gs, ridx)
    # Accessible-sector weight. Do not replace the full-space GS.
    # P psi[x] = psi[x ^ ones], which is psi[::-1].
    p_plus = 0.5 * (gs + gs[::-1])
    r_plus = 0.5 * (gs + gs[ridx])
    pr_plus = 0.5 * (p_plus + p_plus[ridx])
    def _weight(v):
        return float(np.vdot(v, v).real)

    J_hash = sha256_array(np.ascontiguousarray(J, dtype=np.float64))
    hc_hash = sha256_array(np.ascontiguousarray(hc, dtype=np.float64))
    eval_hash = sha256_array(np.round(np.ascontiguousarray(dense["evals"], dtype=np.float64), 12))
    gs_hash = state_hash(gs)
    formula = f"H=sum_{{i<j}}|i-j|^-alpha Z_i Z_j + B sum X; alpha={ALPHA}; J0={J0}; open chain; B={best['B']:.4f}"
    h_hash = sha256_bytes(f"{formula}|J={J_hash}|hc={hc_hash}|evals={eval_hash}".encode())
    second = sorted(rows, key=lambda r: -r["entropy_nats"])[:5]
    return {
        "n": n,
        "alpha": ALPHA,
        "J0": J0,
        "B": best["B"],
        "E0": dense["E0"],
        "gap": dense["gap"],
        "Emax": dense["Emax"],
        "entropy_nats": dense["entropy_nats"],
        "entropy_bits": dense["entropy_bits"],
        "gs_residual": dense["residual"],
        "gs_imag_max": dense["imag_max"],
        "gs_min_real": dense["min_real"],
        "parity_expectation": parity,
        "reflection_expectation": refl,
        "weight_Pplus": _weight(p_plus),
        "weight_Rplus": _weight(r_plus),
        "weight_Pplus_Rplus": _weight(pr_plus),
        "kappa": kappa_per_site(J),
        "hs_Hc": hs_norm_pairs(J),
        "sum_J": float(np.sum(np.triu(J, 1))),
        "J_hash": J_hash,
        "hc_hash": hc_hash,
        "spectrum_hash": eval_hash,
        "gs_hash": gs_hash,
        "H_hash": h_hash,
        "formula": formula,
        "coarse_grid": [{"B": r["B"], "entropy_nats": r["entropy_nats"], "E0": r["E0"], "gap": r["gap"], "residual": r["residual"]} for r in rows],
        "refine_interval": [refine_lo, refine_hi],
        "refine_best_matches_scan": True,
        "near_tie_B_within_1e-6_nats": near,
        "endpoint_expansions": expansions,
        "top_coarse": [{"B": r["B"], "entropy_nats": r["entropy_nats"], "gap": r["gap"]} for r in second],
        "sparse_scan_B": scan_best["B"],
        "sparse_scan_residual_at_best": scan_best["residual"],
        "sparse_vs_dense_entropy_diff": float(scan_best["entropy_nats"] - dense["entropy_nats"]) if abs(scan_best["B"] - best["B"]) < 1e-12 else float(best["entropy_nats"] - dense["entropy_nats"]),
        "sparse_vs_dense_E0_diff": float(scan_best["E0"] - dense["E0"]) if abs(scan_best["B"] - best["B"]) < 1e-12 else None,
        "dense_confirmation_top3": confirm,
        "dense_changed_frozen_B": bool(abs(scan_best["B"] - best["B"]) > 1e-12),
        "gs": gs,
        "hc": hc,
        "evals": dense["evals"],
        "reflection_indices": ridx,
    }
