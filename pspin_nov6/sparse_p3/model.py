"""Graphs, Pauli Hamiltonians, and observables for the sparse 3-body model.

Model (Family B), deliberately not the collective p-spin and not a spin glass::

    H = - sum_{edges (i,j,k)} Z_i Z_j Z_k  -  Gamma * sum_i X_i

Edges come from a 3-regular 3-uniform simple hypergraph: M = N distinct
triples, every vertex in exactly three triples, entries of each triple
distinct. Couplings are uniformly +1 (no random signs).

Bit convention: qubit q is bit q of the computational index (qubit 0 = LSB).
"""

from __future__ import annotations

from itertools import combinations

import numpy as np

PERIOD = np.pi  # exact period of both gamma and beta; derived in validation


def configuration_hypergraph(n: int, seed: int):
    """One configuration-model draw. Return a sorted edge tuple or None.

    Stubs: three labeled stubs per vertex, shuffled with PCG64(seed), then
    cut into N consecutive triples. Reject loops (a vertex repeated inside a
    triple) and repeated edges. A successful draw is exactly 3-regular.
    """
    if n < 3:
        raise ValueError("n >= 3 required")
    rng = np.random.default_rng(np.random.PCG64(int(seed)))
    stubs = np.repeat(np.arange(n, dtype=np.int64), 3)
    rng.shuffle(stubs)
    edges = []
    for e in range(n):
        tri = [int(x) for x in stubs[3 * e : 3 * e + 3]]
        if len(set(tri)) < 3:
            return None
        edges.append(tuple(sorted(tri)))
    if len(set(edges)) < n:
        return None
    deg = np.zeros(n, dtype=np.int64)
    for e in edges:
        for v in e:
            deg[v] += 1
    if not np.all(deg == 3):
        return None
    return tuple(sorted(edges))


def is_linear(edges) -> bool:
    """True when every pair of edges shares at most one vertex."""
    for a, b in combinations(edges, 2):
        if len(set(a) & set(b)) > 1:
            return False
    return True


def incidence_matrix(edges, n: int) -> np.ndarray:
    A = np.zeros((len(edges), n), dtype=np.uint8)
    for r, e in enumerate(edges):
        for v in e:
            A[r, int(v)] = 1
    return A


def gf2_rank(A: np.ndarray) -> int:
    """Rank of a 0/1 matrix over GF(2)."""
    M = np.array(A, dtype=np.uint8, copy=True) & 1
    rows, cols = M.shape
    rank = 0
    row = 0
    for col in range(cols):
        pivot = None
        for r in range(row, rows):
            if M[r, col]:
                pivot = r
                break
        if pivot is None:
            continue
        if pivot != row:
            M[[row, pivot]] = M[[pivot, row]]
        for r in range(rows):
            if r != row and M[r, col]:
                M[r] ^= M[row]
        rank += 1
        row += 1
        if row == rows:
            break
    return int(rank)


def linear_impossible(n: int) -> bool:
    """A linear 3-uniform N-edge hypergraph needs 3N <= C(N, 2)."""
    return 3 * n > n * (n - 1) // 2


def select_graphs(n: int, k: int = 4, max_seed: int = 200000) -> dict:
    """First k distinct full-rank simple 3-regular hypergraphs, seed order.

    Selection does not look at energies or fidelities. Seeds 0,1,2,... are
    tried until k unique full-rank graphs are stored or max_seed is hit.
    """
    chosen = []
    seen = set()
    n_simple = 0
    n_unique = 0
    n_full = 0
    n_linear = 0
    n_linear_full = 0
    first_linear_full = []
    for seed in range(int(max_seed)):
        edges = configuration_hypergraph(n, seed)
        if edges is None:
            continue
        n_simple += 1
        if edges in seen:
            continue
        seen.add(edges)
        n_unique += 1
        lin = is_linear(edges)
        if lin:
            n_linear += 1
        rank = gf2_rank(incidence_matrix(edges, n))
        full = rank == n
        if full:
            n_full += 1
        if lin and full:
            n_linear_full += 1
            if len(first_linear_full) < 2:
                first_linear_full.append(int(seed))
        if full and len(chosen) < k:
            chosen.append(
                {
                    "graph_id": f"N{n}_seed{seed}",
                    "n": int(n),
                    "seed": int(seed),
                    "edges": [list(e) for e in edges],
                    "gf2_rank": int(rank),
                    "full_rank": True,
                    "linear": bool(lin),
                    "simple": True,
                    "regular_degree": 3,
                    "n_edges": int(n),
                }
            )
        if len(chosen) >= k and n_linear_full >= 1:
            # Keep a little extra scan context, but stop once the sample
            # and at least one linearity observation are in hand. For n where
            # linear graphs are impossible this second condition never blocks:
            # linear_impossible is reported separately and we stop at k graphs
            # after a short confirmation window.
            pass
        if len(chosen) >= k and (linear_impossible(n) or seed > chosen[-1]["seed"] + 5000):
            break
    return {
        "n": int(n),
        "requested": int(k),
        "max_seed_cap": int(max_seed),
        "seeds_scanned_through": int(chosen[-1]["seed"]) if chosen else int(max_seed),
        "n_simple_draws": int(n_simple),
        "n_unique_simple": int(n_unique),
        "n_unique_full_rank_seen": int(n_full),
        "n_unique_linear_seen": int(n_linear),
        "n_unique_linear_full_rank_seen": int(n_linear_full),
        "linear_impossible_by_pair_count": bool(linear_impossible(n)),
        "first_linear_full_rank_seeds": first_linear_full,
        "selection_rule": (
            "Ascending PCG64 configuration-model seeds; keep the first k "
            "distinct simple 3-regular hypergraphs whose GF(2) incidence "
            "matrix has rank N. No energy or fidelity filter."
        ),
        "graphs": chosen,
    }


def z_eigenvalues(n: int, qubit: int) -> np.ndarray:
    idx = np.arange(1 << n, dtype=np.int64)
    bit = (idx >> int(qubit)) & 1
    return np.where(bit == 0, 1.0, -1.0).astype(np.float64)


def triple_eigenvalues(n: int, edge) -> np.ndarray:
    z = np.ones(1 << n, dtype=np.float64)
    for q in edge:
        z *= z_eigenvalues(n, int(q))
    return z


def cost_diagonal(n: int, edges, couplings) -> np.ndarray:
    """Diagonal of H_C = - sum_a J_a Z_i Z_j Z_k."""
    diag = np.zeros(1 << n, dtype=np.float64)
    for edge, J in zip(edges, couplings):
        diag -= float(J) * triple_eigenvalues(n, edge)
    return diag


def uniform_couplings(n_edges: int, sign: float = 1.0) -> np.ndarray:
    return np.full(int(n_edges), float(sign), dtype=np.float64)


def pauli_zzz_diagonal_kron(n: int, edge) -> np.ndarray:
    """Independent construction: Kronecker product of Z and I, then diagonal."""
    Z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.float64)
    I = np.eye(2, dtype=np.float64)
    acc = np.array([[1.0]], dtype=np.float64)
    edge_set = set(int(q) for q in edge)
    # Kronecker from high qubit down to qubit 0 so qubit 0 is the LSB.
    for q in range(n - 1, -1, -1):
        acc = np.kron(acc, Z if q in edge_set else I)
    return np.diag(acc).real.copy()


def build_HX(n: int) -> np.ndarray:
    """Dense sum_i X_i with qubit q on bit q."""
    dim = 1 << n
    H = np.zeros((dim, dim), dtype=np.float64)
    idx = np.arange(dim, dtype=np.int64)
    for q in range(n):
        partner = idx ^ (1 << q)
        upper = partner > idx
        ii = idx[upper]
        jj = partner[upper]
        H[ii, jj] = 1.0
        H[jj, ii] = 1.0
    return H


def apply_H_matvec(v: np.ndarray, hc_diag: np.ndarray, gamma: float, n: int) -> np.ndarray:
    """(H_C - Gamma * sum X) v, matrix-free."""
    out = hc_diag * v
    idx = np.arange(v.shape[0], dtype=np.int64)
    for q in range(n):
        out = out - float(gamma) * v[idx ^ (1 << q)]
    return out


def dense_hamiltonian(hc_diag: np.ndarray, gamma: float, n: int) -> np.ndarray:
    return np.diag(hc_diag.astype(np.float64)) - float(gamma) * build_HX(n)


def _phase_fix_gs(gs: np.ndarray) -> tuple[np.ndarray, int]:
    gs = np.real(np.asarray(gs, dtype=np.float64)).copy()
    gs /= np.linalg.norm(gs)
    if float(np.sum(gs)) < 0.0:
        gs *= -1.0
    n_neg = int(np.sum(gs < -1e-8))
    return gs, n_neg


def ground_state_dense(hc_diag: np.ndarray, gamma: float, n: int):
    """Full dense eigh. Used to audit the Lanczos ground state, not as the scan default."""
    H = dense_hamiltonian(hc_diag, gamma, n)
    evals, evecs = np.linalg.eigh(H)
    gs, n_neg = _phase_fix_gs(evecs[:, 0])
    gap = float(evals[1] - evals[0])
    return {
        "evals0": float(evals[0]),
        "evals1": float(evals[1]),
        "evals_max": float(evals[-1]),
        "gap": gap,
        "degenerate": bool(gap < 1e-8),
        "gs": gs,
        "n_negative_amps": n_neg,
        "dim": int(H.shape[0]),
        "method": "dense_eigh",
        "residual": float(np.linalg.norm(H @ gs - evals[0] * gs)),
    }


def ground_state(hc_diag: np.ndarray, gamma: float, n: int):
    """Lowest two eigenpairs of the full 2^N Hamiltonian via Lanczos.

    No symmetry sector is projected out. The matvec is H_C - Gamma * sum X
    on the complete computational basis. For Gamma > 0 the matrix is
    stoquastic, so a nondegenerate ground state can be chosen nonnegative.

    Odd p with a pure transverse field is antisymmetric under conjugation by
    prod_i Y_i, which sends H to -H. The spectrum is symmetric and
    Emax = -E0. That identity is checked against a separate largest-eigenvalue
    Lanczos solve; the returned Emax is the computed one, not the substitute.
    """
    from scipy.sparse.linalg import LinearOperator, eigsh

    dim = 1 << n
    hc = np.asarray(hc_diag, dtype=np.float64)
    gamma = float(gamma)

    def matvec(v):
        return apply_H_matvec(np.asarray(v, dtype=np.float64), hc, gamma, n)

    op = LinearOperator((dim, dim), matvec=matvec, dtype=np.float64)
    # k=2 is enough for the gap. A tiny gap still returns two Ritz values.
    evals_lo, evecs_lo = eigsh(op, k=2, which="SA", tol=1e-12, maxiter=max(2000, 4 * dim))
    order = np.argsort(evals_lo)
    e0 = float(evals_lo[order[0]])
    e1 = float(evals_lo[order[1]])
    gs, n_neg = _phase_fix_gs(evecs_lo[:, order[0]])
    residual = float(np.linalg.norm(matvec(gs) - e0 * gs))
    evals_hi, _ = eigsh(op, k=1, which="LA", tol=1e-10, maxiter=max(2000, 4 * dim))
    emax = float(evals_hi[0])
    gap = float(e1 - e0)
    return {
        "evals0": e0,
        "evals1": e1,
        "evals_max": emax,
        "gap": gap,
        "degenerate": bool(gap < 1e-8),
        "gs": gs,
        "n_negative_amps": n_neg,
        "dim": int(dim),
        "method": "eigsh_full_space",
        "residual": residual,
        "emax_plus_e0": float(emax + e0),
    }


def plus_state(n: int) -> np.ndarray:
    dim = 1 << n
    return np.full(dim, 1.0 / np.sqrt(dim), dtype=np.complex128)


def classical_ground_index() -> int:
    """All-plus Z product is computational basis index 0 (every bit 0)."""
    return 0


def count_classical_ground_states(hc_diag: np.ndarray, n_edges: int) -> dict:
    """Satisfying assignments of uniform +1 couplings have H_C = -M."""
    target = -float(n_edges)
    hits = np.flatnonzero(np.isclose(hc_diag, target))
    return {
        "n_ground": int(hits.size),
        "indices": [int(i) for i in hits[:8]],
        "all_plus_index_is_ground": bool(hits.size > 0 and int(hits[0]) == 0 and np.isclose(hc_diag[0], target)),
        "energy": target,
    }


def flip_state(psi: np.ndarray, mask: int) -> np.ndarray:
    """Apply prod_{q in mask} X_q."""
    idx = np.arange(psi.shape[0], dtype=np.int64)
    out = np.empty_like(psi)
    out[idx] = psi[idx ^ int(mask)]
    return out


def gauge_couplings(edges, couplings, flipped_qubits) -> np.ndarray:
    """Conjugation by prod_{q in S} X_q flips each edge with odd overlap with S.

    For odd p this is the standard gauge of a p-spin interaction. Degree 3 is
    odd, so flipping every qubit sends every coupling J_a -> -J_a.
    """
    mask = 0
    for q in flipped_qubits:
        mask |= 1 << int(q)
    out = np.array(couplings, dtype=np.float64, copy=True)
    for a, edge in enumerate(edges):
        parity = 0
        for q in edge:
            parity ^= (mask >> int(q)) & 1
        if parity:
            out[a] = -out[a]
    return out


def half_cut_entropy(psi: np.ndarray, n: int) -> dict:
    """Von Neumann entropy, bits, of the low n//2 qubits (bits 0 .. na-1).

    High bits are traced out. The cut is the integer partition of the index
    into the least-significant na bits and the remaining high bits. Product
    states have entropy 0. The value is not a contiguous spatial cut of a
    chain, because the hypergraph is not a chain; the bit partition is fixed
    and the same for every instance of a given N.
    """
    na = n // 2
    nb = n - na
    amp = np.asarray(psi, dtype=np.complex128)
    V = amp.reshape(1 << nb, 1 << na)
    rho = V.conj().T @ V
    ev = np.linalg.eigvalsh(rho).real
    ev = np.clip(ev, 0.0, None)
    ev = ev[ev > 1e-15]
    if ev.size == 0:
        S = 0.0
    else:
        S = float(-(ev * np.log2(ev)).sum())
    return {
        "S_half_bits": S,
        "cut_low_qubits": list(range(na)),
        "cut_high_qubits": list(range(na, n)),
        "n_low": int(na),
        "n_high": int(nb),
        "cut_definition": "low bits 0..n//2-1 versus the complementary high bits",
    }


def z_expectations(psi: np.ndarray, n: int) -> np.ndarray:
    p = np.abs(np.asarray(psi)) ** 2
    idx = np.arange(p.shape[0], dtype=np.int64)
    out = np.empty(n, dtype=np.float64)
    for q in range(n):
        z = np.where(((idx >> q) & 1) == 0, 1.0, -1.0)
        out[q] = float(np.dot(p, z))
    return out


def x_expectations(psi: np.ndarray, n: int) -> np.ndarray:
    amp = np.asarray(psi, dtype=np.complex128)
    idx = np.arange(amp.shape[0], dtype=np.int64)
    out = np.empty(n, dtype=np.float64)
    for q in range(n):
        out[q] = float(np.vdot(amp, amp[idx ^ (1 << q)]).real)
    return out


def clause_expectations(psi: np.ndarray, n: int, edges) -> np.ndarray:
    p = np.abs(np.asarray(psi)) ** 2
    idx = np.arange(p.shape[0], dtype=np.int64)
    out = np.empty(len(edges), dtype=np.float64)
    for a, edge in enumerate(edges):
        z = np.ones(p.shape[0], dtype=np.float64)
        for q in edge:
            z *= np.where(((idx >> int(q)) & 1) == 0, 1.0, -1.0)
        out[a] = float(np.dot(p, z))
    return out


def product_state(theta: np.ndarray, n: int) -> np.ndarray:
    """Real product state. theta[q] is the Bloch polar angle of qubit q.

    |θ⟩ = cos(θ/2)|0⟩ + sin(θ/2)|1⟩, qubit 0 = LSB.
    """
    psi = np.array([1.0], dtype=np.float64)
    for q in range(n - 1, -1, -1):
        a0 = np.cos(float(theta[q]) / 2.0)
        a1 = np.sin(float(theta[q]) / 2.0)
        psi = np.kron(psi, np.array([a0, a1], dtype=np.float64))
    return psi


def best_product_overlap(gs: np.ndarray, n: int, seed: int = 0) -> dict:
    """Maximize |⟨product|GS⟩|^2 over real nonnegative product states.

    The dense ground state is real. Starts: all-zero (classical product),
    all-plus, single-site marginal angles, and a few deterministic perturbations.
    """
    from scipy.optimize import minimize

    gs = np.real(np.asarray(gs, dtype=np.float64))
    gs = gs / np.linalg.norm(gs)

    def overlap(theta):
        prod = product_state(np.asarray(theta, dtype=np.float64), n)
        return float(np.dot(prod, gs))

    def loss(theta):
        return -overlap(theta) ** 2

    z = z_expectations(gs, n)
    z = np.clip(z, -1.0, 1.0)
    marginal = np.arccos(z)
    starts = [
        np.zeros(n, dtype=np.float64),
        np.full(n, np.pi / 2.0),
        marginal,
    ]
    rng = np.random.default_rng(np.random.PCG64(1000 + int(seed) + 17 * n))
    for _ in range(3):
        starts.append(np.clip(marginal + rng.normal(0.0, 0.35, size=n), 0.0, np.pi))

    best_ov = -1.0
    best_theta = starts[0]
    n_eval = 0
    for th0 in starts:
        res = minimize(
            loss,
            np.asarray(th0, dtype=np.float64),
            method="L-BFGS-B",
            bounds=[(0.0, np.pi)] * n,
            options={"maxiter": 80, "ftol": 1e-14},
        )
        n_eval += int(res.nfev)
        ov2 = float(-res.fun)
        if ov2 > best_ov:
            best_ov = ov2
            best_theta = np.asarray(res.x, dtype=np.float64)
    # Also record the computational-basis and |+> overlaps exactly.
    f_classical = float(gs[0] ** 2)
    f_plus = float(np.dot(gs, np.full(gs.shape[0], 1.0 / np.sqrt(gs.shape[0]))) ** 2)
    f_bit = float(np.max(gs**2))
    return {
        "F_product": float(best_ov),
        "theta": [float(t) for t in best_theta],
        "F_classical_allplusZ": f_classical,
        "F_plus": f_plus,
        "F_best_bitstring": f_bit,
        "n_fev": int(n_eval),
    }


def state_observables(psi, n, edges, gs, E0, Emax, energy) -> dict:
    amp = np.asarray(psi)
    zn = z_expectations(amp, n)
    xn = x_expectations(amp, n)
    clauses = clause_expectations(amp, n, edges)
    ent = half_cut_entropy(amp, n)
    F = float(abs(np.vdot(np.asarray(gs, dtype=np.complex128), amp)) ** 2)
    denom = float(Emax - E0)
    eta = float((Emax - energy) / denom) if abs(denom) > 1e-15 else float("nan")
    p = np.abs(amp) ** 2
    return {
        "F": F,
        "energy": float(energy),
        "eta": eta,
        "S_half_bits": ent["S_half_bits"],
        "m_z": float(np.mean(zn)),
        "m_x": float(np.mean(xn)),
        "q_z": float(np.mean(zn**2)),
        "clause_mean": float(np.mean(clauses)),
        "F_best_bitstring_of_state": float(np.max(p)),
        "prob_allplusZ": float(p[0]),
    }


def collective_p3_diagonal(n: int) -> np.ndarray:
    """Diagonal of (sum_i Z_i)^3, for the bookkeeping identity only."""
    s = np.zeros(1 << n, dtype=np.float64)
    for q in range(n):
        s += z_eigenvalues(n, q)
    return s**3


def distinct_index_identity_residual(n: int) -> float:
    """Max | (sum Z)^3 - 6 sum_{i<j<k} ZZZ - (3N-2) sum Z | on the diagonal."""
    s = np.zeros(1 << n, dtype=np.float64)
    for q in range(n):
        s += z_eigenvalues(n, q)
    left = s**3
    acc = np.zeros(1 << n, dtype=np.float64)
    for i, j, k in combinations(range(n), 3):
        acc += z_eigenvalues(n, i) * z_eigenvalues(n, j) * z_eigenvalues(n, k)
    right = 6.0 * acc + (3.0 * n - 2.0) * s
    return float(np.max(np.abs(left - right)))


def term_list(n: int, edges, gamma: float) -> list:
    """Explicit Pauli terms of the full target H at a given Gamma."""
    terms = []
    for e in edges:
        terms.append(
            {
                "pauli": "ZZZ",
                "qubits": [int(q) for q in e],
                "coeff": -1.0,
            }
        )
    for q in range(int(n)):
        terms.append({"pauli": "X", "qubits": [int(q)], "coeff": -float(gamma)})
    return terms
