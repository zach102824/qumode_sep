"""Pagano-like open-chain AF TFIM target for round1 direction D.

Contract (overrides older briefs):
  H = Hc + B sum_i X_i
  Hc = sum_{i<j} J_ij Z_i Z_j
  J_ij = |i-j| ** (-alpha), alpha = 1.1, J0 = 1
  B < 0 chosen by maximum half-chain entanglement of the full-space ground state.

Site 0 is the leftmost site and the most-significant bit of the basis index.
Z|0> = +|0>, Z|1> = -|1>. Cavities are contiguous blocks of 3 sites.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import thread_env  # noqa: F401
import numpy as np
from scipy.linalg import eigh
from scipy.sparse.linalg import LinearOperator, eigsh

ALPHA = float(np.float64("1.1"))
J0 = 1.0
# Inclusive coarse grid required by the contract.
B_GRID_LO = -0.8
B_GRID_HI = -0.1
B_GRID_STEP = 0.005
B_REFINE_STEP = 1e-4


def sha256_array(a: np.ndarray) -> str:
    arr = np.ascontiguousarray(a)
    return hashlib.sha256(arr.view(np.uint8)).hexdigest()


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
    if isinstance(obj, complex):
        return {"re": float(obj.real), "im": float(obj.imag)}
    return obj


def dump_json(path: Path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(obj), indent=2, allow_nan=False) + "\n")


def build_J(n: int, alpha: float = ALPHA, J0: float = J0) -> np.ndarray:
    """Open chain. Nearest-neighbour entries equal J0 because distance 1 ** -alpha = 1."""
    J = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            J[i, j] = J[j, i] = float(J0) / (float(j - i) ** float(alpha))
    return J


def z_table(n: int) -> np.ndarray:
    """z[x, i] = ±1 for site i (0 = MSB) in basis state x. Bit 0 -> +1."""
    dim = 1 << n
    shifts = np.arange(n - 1, -1, -1, dtype=np.int64)
    bits = ((np.arange(dim, dtype=np.int64)[:, None] >> shifts) & 1).astype(np.int8)
    return (1 - 2 * bits).astype(np.float64)


def hc_diagonal(J: np.ndarray, z: np.ndarray | None = None) -> np.ndarray:
    n = J.shape[0]
    if z is None:
        z = z_table(n)
    # sum_{i<j} J_ij z_i z_j = 0.5 * sum_{ij} J_ij z_i z_j with J symmetric, zero diagonal.
    return 0.5 * np.einsum("ai,ij,aj->a", z, J, z, optimize=True)


def plus_state(n: int) -> np.ndarray:
    dim = 1 << n
    return np.full(dim, 1.0 / np.sqrt(dim), dtype=np.complex128)


def apply_H(psi: np.ndarray, hc: np.ndarray, B: float, n: int) -> np.ndarray:
    """H|psi> with H = diag(Hc) + B sum_i X_i. Real H, accepts real or complex psi."""
    out = hc * psi
    tensor = np.reshape(psi, [2] * n)
    acc = np.zeros_like(tensor)
    for i in range(n):
        acc += np.flip(tensor, axis=i)
    out = out + (B * acc.reshape(psi.shape))
    return out


def dense_H(hc: np.ndarray, B: float, n: int) -> np.ndarray:
    dim = 1 << n
    H = np.diag(np.asarray(hc, dtype=np.float64)).copy()
    idx = np.arange(dim, dtype=np.int64)
    for i in range(n):
        mask = 1 << (n - 1 - i)
        partner = idx ^ mask
        H[partner, idx] += float(B)
    return H


def bit_reverse_perm(n: int) -> np.ndarray:
    """Site reflection i <-> n-1-i is bit reversal when site 0 is MSB."""
    dim = 1 << n
    perm = np.empty(dim, dtype=np.int64)
    for x in range(dim):
        y = 0
        for b in range(n):
            if (x >> b) & 1:
                y |= 1 << (n - 1 - b)
        perm[x] = y
    return perm


def parity_expectations(psi: np.ndarray) -> dict:
    """P = prod_i X_i (real bit-flip, no extra phase)."""
    dim = psi.shape[0]
    flipped = psi[np.arange(dim) ^ (dim - 1)]
    exp = complex(np.vdot(psi, flipped))
    proj = 0.5 * (psi + flipped)
    weight_plus = float(np.vdot(proj, proj).real)
    return {
        "P_expectation_re": float(exp.real),
        "P_expectation_im": float(exp.imag),
        "P_plus_weight": weight_plus,
    }


def reflection_expectations(psi: np.ndarray, perm: np.ndarray) -> dict:
    reflected = psi[perm]
    exp = complex(np.vdot(psi, reflected))
    proj = 0.5 * (psi + reflected)
    weight_plus = float(np.vdot(proj, proj).real)
    return {
        "R_expectation_re": float(exp.real),
        "R_expectation_im": float(exp.imag),
        "R_plus_weight": weight_plus,
    }


def canonicalize_gs(gs: np.ndarray) -> np.ndarray:
    """Real nonnegative representative when the vector is real up to a phase."""
    v = np.asarray(gs, dtype=np.complex128).copy()
    k = int(np.argmax(np.abs(v)))
    if abs(v[k]) < 1e-30:
        return v
    v = v * (abs(v[k]) / v[k])
    if np.linalg.norm(v.imag) <= 1e-10 * max(1.0, np.linalg.norm(v.real)):
        v = v.real.astype(np.float64)
        if float(np.sum(v)) < 0.0:
            v = -v
        return v.astype(np.complex128)
    return v


def half_chain_entropy_bits(psi: np.ndarray, n: int) -> float:
    """Von Neumann entropy of sites 0..n/2-1, in bits. Site 0 is MSB / first reshape axis."""
    if n % 2 != 0:
        raise ValueError("half-chain entropy requires even n")
    nA = n // 2
    mat = np.asarray(psi, dtype=np.complex128).reshape(1 << nA, 1 << (n - nA))
    s = np.linalg.svd(mat, compute_uv=False)
    p = np.clip(s.real ** 2, 0.0, None)
    p = p[p > 1e-16]
    p = p / p.sum()
    return float(-(p * np.log2(p)).sum())


def cavity_packing(n: int, bits: int = 3) -> dict:
    if n % bits != 0:
        raise ValueError(f"n={n} is not a multiple of {bits}")
    cavities = [list(range(a * bits, (a + 1) * bits)) for a in range(n // bits)]
    reflected = [[n - 1 - i for i in cav] for cav in cavities]
    # Reflection should map the set of cavities to itself, reversing each block.
    reflected_sets = [tuple(sorted(c)) for c in reflected]
    original_sets = [tuple(c) for c in cavities]
    maps_to_cavities = sorted(reflected_sets) == sorted(original_sets)
    return {
        "bits_per_cavity": bits,
        "cavities_left_to_right": cavities,
        "reflection_maps_cavities_to_cavities": bool(maps_to_cavities),
        "reflected_site_lists": reflected,
    }


def kappa_mean_degree(J: np.ndarray) -> float:
    """mean_i sum_{j != i} J_ij. Uses J only, no ground state."""
    n = J.shape[0]
    return float(J.sum() / n)


def full_spectrum(hc: np.ndarray, B: float, n: int):
    H = dense_H(hc, B, n)
    evals, evecs = eigh(H, check_finite=False)
    return H, evals, evecs


def _gs_from_eigh(evals, evecs):
    gs = canonicalize_gs(evecs[:, 0])
    E0 = float(evals[0])
    E1 = float(evals[1])
    Emax = float(evals[-1])
    return gs, E0, E1, Emax


def _residual(psi, E, hc, B, n) -> float:
    r = apply_H(psi, hc, B, n) - E * psi
    return float(np.linalg.norm(r) / max(1.0, np.linalg.norm(psi)))


def spectrum_point_dense(hc, B, n, perm) -> dict:
    _H, evals, evecs = full_spectrum(hc, B, n)
    gs, E0, E1, Emax = _gs_from_eigh(evals, evecs)
    ent = half_chain_entropy_bits(gs, n)
    return {
        "B": float(B),
        "E0": E0,
        "E1": E1,
        "gap": E1 - E0,
        "Emax": Emax,
        "entropy_bits": ent,
        "entropy_nats": ent * float(np.log(2.0)),
        "residual": _residual(gs, E0, hc, B, n),
        "gs": gs,
        "parity": parity_expectations(gs),
        "reflection": reflection_expectations(gs, perm),
        "min_amplitude_real": float(np.min(gs.real)),
        "imag_norm": float(np.linalg.norm(gs.imag)),
        "solver": "dense_eigh",
    }


def _eigsh_pair(hc, B, n, v0) -> tuple[np.ndarray, np.ndarray]:
    dim = hc.shape[0]

    def mv(v):
        return np.real(apply_H(np.asarray(v, dtype=np.float64), hc, B, n))

    op = LinearOperator((dim, dim), matvec=mv, dtype=np.float64)
    evals, evecs = eigsh(op, k=2, which="SA", tol=1e-12, maxiter=4000, v0=np.real(v0))
    order = np.argsort(evals.real)
    return evals.real[order], evecs[:, order]


def spectrum_point_eigsh(hc, B, n, perm, v0) -> dict:
    evals, evecs = _eigsh_pair(hc, B, n, v0)
    gs = canonicalize_gs(evecs[:, 0])
    E0 = float(evals[0])
    E1 = float(evals[1])
    ent = half_chain_entropy_bits(gs, n)
    return {
        "B": float(B),
        "E0": E0,
        "E1": E1,
        "gap": E1 - E0,
        "Emax": None,
        "entropy_bits": ent,
        "entropy_nats": ent * float(np.log(2.0)),
        "residual": _residual(gs, E0, hc, B, n),
        "gs": gs,
        "parity": parity_expectations(gs),
        "reflection": reflection_expectations(gs, perm),
        "min_amplitude_real": float(np.min(gs.real)),
        "imag_norm": float(np.linalg.norm(gs.imag)),
        "solver": "eigsh_SA_k2",
    }


def coarse_B_grid() -> np.ndarray:
    nstep = int(round((B_GRID_HI - B_GRID_LO) / B_GRID_STEP))
    return np.linspace(B_GRID_LO, B_GRID_HI, nstep + 1)


def _local_max_indices(values: np.ndarray) -> list[int]:
    out = []
    for i, s in enumerate(values):
        left = values[i - 1] if i > 0 else -np.inf
        right = values[i + 1] if i + 1 < len(values) else -np.inf
        if s >= left and s >= right:
            out.append(i)
    return out


def select_field(n: int, alpha: float = ALPHA, log=print) -> dict:
    """Entropy-maximizing B on the contract grid, with endpoint expansion and 1e-4 refine.

    Ground-state data are used only to choose and document the benchmark field.
    They are not used as a training oracle for angles.
    """
    J = build_J(n, alpha=alpha)
    z = z_table(n)
    hc = hc_diagonal(J, z)
    perm = bit_reverse_perm(n)
    packing = cavity_packing(n)
    grid = coarse_B_grid()
    expanded = []
    diagnosis = []

    def evaluate_many(Bs: np.ndarray) -> list[dict]:
        rows = []
        v0 = plus_state(n).real.copy()
        for i, B in enumerate(Bs):
            if n <= 8:
                row = spectrum_point_dense(hc, float(B), n, perm)
            else:
                row = spectrum_point_eigsh(hc, float(B), n, perm, v0)
                if row["residual"] > 1e-8 or row["gap"] < 0:
                    log(f"  eigsh residual {row['residual']:.2e} at B={B:.6f}; dense fallback")
                    row = spectrum_point_dense(hc, float(B), n, perm)
                else:
                    # Warm-start the next point from this GS (field changes slowly).
                    v0 = row["gs"].real.copy()
            rows.append(row)
            if i % 20 == 0 or i == len(Bs) - 1:
                log(
                    f"  scan n={n} {i+1}/{len(Bs)} B={float(B):.4f} "
                    f"S={row['entropy_bits']:.6f} gap={row['gap']:.6e} res={row['residual']:.2e}"
                )
        return rows

    log(f"coarse entropy scan n={n} points={len(grid)} alpha={alpha!r}")
    coarse = evaluate_many(grid)
    ent = np.array([r["entropy_bits"] for r in coarse])
    best_i = int(np.argmax(ent))

    # Endpoint is not an acceptable freeze. Expand by 0.3, twice if needed.
    guard = 0
    while best_i in (0, len(grid) - 1) and guard < 2:
        guard += 1
        if best_i == 0:
            new_hi = float(grid[0]) - B_GRID_STEP
            new_lo = float(grid[0]) - 0.3
            side = "more_negative"
        else:
            new_lo = float(grid[-1]) + B_GRID_STEP
            new_hi = float(grid[-1]) + 0.3
            side = "less_negative"
        extra = np.linspace(new_lo, new_hi, int(round((new_hi - new_lo) / B_GRID_STEP)) + 1)
        # Keep B finite and record that we left the nominal window.
        diagnosis.append(
            {
                "event": "endpoint_expansion",
                "side": side,
                "nominal_best_B": float(grid[best_i]),
                "nominal_best_S": float(ent[best_i]),
                "extra_lo": float(extra[0]),
                "extra_hi": float(extra[-1]),
            }
        )
        log(f"  endpoint at B={grid[best_i]:.4f}; expanding {side} over [{extra[0]:.4f},{extra[-1]:.4f}]")
        extra_rows = evaluate_many(extra)
        expanded.append(extra_rows)
        # Rebuild a merged grid in increasing B order.
        merged = list(zip(grid.tolist(), coarse)) + [
            (float(B), row) for B, row in zip(extra, extra_rows)
        ]
        merged.sort(key=lambda t: t[0])
        # Drop duplicate B.
        dedup = []
        for B, row in merged:
            if dedup and abs(dedup[-1][0] - B) < 1e-12:
                continue
            dedup.append((B, row))
        grid = np.array([t[0] for t in dedup], dtype=np.float64)
        coarse = [t[1] for t in dedup]
        ent = np.array([r["entropy_bits"] for r in coarse])
        best_i = int(np.argmax(ent))

    endpoint_unresolved = bool(best_i in (0, len(grid) - 1))
    if endpoint_unresolved:
        diagnosis.append(
            {
                "event": "endpoint_unresolved",
                "B": float(grid[best_i]),
                "note": "Entropy maximum remained on the expanded boundary. Field is frozen there and flagged.",
            }
        )

    # Refine every near-best local maximum on its neighboring coarse interval.
    peak_S = float(ent[best_i])
    refine_ids = []
    for i in _local_max_indices(ent):
        if peak_S - float(ent[i]) <= 0.05:
            refine_ids.append(i)
    if best_i not in refine_ids:
        refine_ids.append(best_i)

    fine_rows = []
    for i in refine_ids:
        lo = float(grid[max(0, i - 1)])
        hi = float(grid[min(len(grid) - 1, i + 1)])
        if hi <= lo:
            continue
        nfine = int(round((hi - lo) / B_REFINE_STEP)) + 1
        fine_B = np.linspace(lo, hi, nfine)
        log(f"  refine local max coarse B={grid[i]:.4f} on [{lo:.4f},{hi:.4f}] ({nfine} pts)")
        fine_rows.extend(evaluate_many(fine_B))

    # Choose the global entropy max among coarse and fine samples.
    pool = list(coarse) + fine_rows
    pool_S = np.array([r["entropy_bits"] for r in pool])
    winner = int(np.argmax(pool_S))
    # Numerical ties: median B among samples within 1e-5 bits of the best.
    tie = [r for r in pool if pool_S[winner] - r["entropy_bits"] <= 1e-5]
    tie_B = np.median([r["B"] for r in tie])
    # Snap to the evaluated sample nearest that median, then confirm it is still at the max.
    nearest = min(pool, key=lambda r: (abs(r["B"] - tie_B), -(r["entropy_bits"])))
    if pool_S[winner] - nearest["entropy_bits"] > 1e-5:
        nearest = pool[winner]

    # One dense solve freezes Emax, the spectrum hash, and the published GS.
    log(f"  dense freeze B={nearest['B']:.6f}")
    H = dense_H(hc, float(nearest["B"]), n)
    evals, evecs = eigh(H, check_finite=False)
    gs, E0, E1, Emax = _gs_from_eigh(evals, evecs)
    frozen = {
        "B": float(nearest["B"]),
        "E0": E0,
        "E1": E1,
        "gap": E1 - E0,
        "Emax": Emax,
        "entropy_bits": half_chain_entropy_bits(gs, n),
        "residual": _residual(gs, E0, hc, float(nearest["B"]), n),
        "gs": gs,
        "parity": parity_expectations(gs),
        "reflection": reflection_expectations(gs, perm),
        "min_amplitude_real": float(np.min(gs.real)),
        "imag_norm": float(np.linalg.norm(gs.imag)),
        "solver": "dense_eigh",
    }
    frozen["entropy_nats"] = frozen["entropy_bits"] * float(np.log(2.0))

    gs = frozen["gs"]
    curve = [
        {
            "B": r["B"],
            "entropy_bits": r["entropy_bits"],
            "E0": r["E0"],
            "gap": r["gap"],
            "residual": r["residual"],
            "solver": r["solver"],
        }
        for r in coarse
    ]
    top = sorted(coarse, key=lambda r: -r["entropy_bits"])[:5]
    result = {
        "n": n,
        "alpha": alpha,
        "alpha_hex": float(alpha).hex(),
        "J0": J0,
        "J": J,
        "kappa_mean_degree": kappa_mean_degree(J),
        "sum_pairs_J": float(np.sum(np.triu(J, 1))),
        "hc_diag": hc,
        "B": frozen["B"],
        "B_hex": float(frozen["B"]).hex(),
        "E0": frozen["E0"],
        "E1": frozen["E1"],
        "gap": frozen["gap"],
        "Emax": frozen["Emax"],
        "entropy_bits": frozen["entropy_bits"],
        "entropy_nats": frozen["entropy_nats"],
        "gs_residual": frozen["residual"],
        "gs_min_amplitude_real": frozen["min_amplitude_real"],
        "gs_imag_norm": frozen["imag_norm"],
        "parity": frozen["parity"],
        "reflection": frozen["reflection"],
        "packing": packing,
        "endpoint_unresolved": endpoint_unresolved,
        "diagnosis": diagnosis,
        "coarse_curve": curve,
        "top_coarse": [
            {"B": r["B"], "entropy_bits": r["entropy_bits"], "gap": r["gap"], "E0": r["E0"]}
            for r in top
        ],
        "n_fine_evaluations": len(fine_rows),
        "tie_width_1e-5_bits": int(len(tie)),
        "hashes": {
            "J_sha256": sha256_array(J),
            "hc_diag_sha256": sha256_array(hc),
            "H_dense_sha256": sha256_array(H),
            "spectrum_sha256": sha256_array(np.asarray(evals, dtype=np.float64)),
            "gs_sha256": sha256_array(np.asarray(gs, dtype=np.complex128)),
            "layout": "C-order float64; GS complex128 after real-nonnegative canonicalization; site0=MSB",
        },
        "gs": gs,
        "selection_rule": (
            "Maximize half-chain von Neumann entropy (bits, log2) of the full-space ground state. "
            "Coarse B in [-0.8,-0.1] step 0.005 inclusive. If the max is an endpoint, expand by 0.3 "
            "up to twice and record the diagnosis. Refine every local maximum within 0.05 bits of the "
            "best on its neighboring coarse interval at step 1e-4. Freeze the median-B near-tie, "
            "re-solved by dense eigh. This B is our numerical realization, not a published Pagano value."
        ),
        "stochastic_note": (
            "B<0 makes off-diagonal X matrix elements negative, so H is stoquastic in the Z basis. "
            "The ground state is taken from the full 2^N space, not a parity or reflection sector."
        ),
    }
    return result


def public_target(target: dict) -> dict:
    """JSON-safe target record. The GS vector is stored separately as .npy."""
    skip = {"gs", "J", "hc_diag"}
    out = {k: jsonable(v) for k, v in target.items() if k not in skip}
    out["J"] = target["J"].tolist()
    out["hashes"] = target["hashes"]
    return out
