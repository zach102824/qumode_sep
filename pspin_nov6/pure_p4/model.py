"""Family C: fully connected DISTINCT-index pure 4-body ferro / antiferro.

This is not a random p-spin glass. Every quadruple has the same coupling.

Conventions (fixed for every comparison in this folder)
-------------------------------------------------------
J = 1.
a_N = N / C(N, 4) = 24 / ((N-1)(N-2)(N-3))     # extensive
H_C^pure = s * J * a_N * sum_{i<j<k<l} Z_i Z_j Z_k Z_l
H_C^ctrl = s * J * N * m_z^4 = s * J * M^4 / N^3     # collective CONTROL only
M = sum_i Z_i,  m_z = M/N,  Z|0>=+|0>, Z|1>=-|1>, qubit 0 is MSB.
s = -1 ferro, s = +1 antiferro.
H = H_C - Gamma * sum_i X_i.

a_N is chosen so the fully polarized configurations have pure cost ±N,
the same polarized cost as the collective control. The two operators still
disagree at every other magnetization: expanding M^4 with Z_i^2 = I produces
repeated-index (lower-body) contractions, and the distinct four-body
coefficient in N m_z^4 is 24/N^3 rather than a_N.

Repeated-index identity (operator equation, Z_i commuting involutions)
-----------------------------------------------------------------------
(sum_i Z_i)^4
  = sum_i Z_i^4                                             # type (4)
  + sum_{a != b} 4 Z_a^3 Z_b                                # type (3,1)
  + sum_{a < b} 6 Z_a^2 Z_b^2                               # type (2,2)
  + sum_{distinct a,b,c} 6 Z_a^2 Z_b Z_c                    # type (2,1,1)
  + sum_{distinct i,j,k,l ordered} Z_i Z_j Z_k Z_l         # type (1,1,1,1)

Using Z^2 = I and Z^3 = Z, and writing e_4 = sum_{i<j<k<l} Z_i Z_j Z_k Z_l,
  M^4 = N
      + 4 (M^2 - N)
      + 3 N (N-1)
      + 6 (N-2) (M^2 - N)
      + 24 e_4.

Hence
  e_4 = [ M^4 - (6N-8) M^2 + 3 N (N-2) I ] / 24.

The (2,1,1) coefficient is 6, not 12: the 12 ordered strings with pattern
{a,a,b,c} are shared by the labels (b,c) and (c,b). Checked by partitioning
N^4 and by direct enumeration in validate.py.

Odd and even N are the same identity. Admissible M have M ≡ N (mod 2).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations
from typing import Iterable

import numpy as np
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import eigsh

J = 1.0


def a_N(n: int) -> float:
    """Extensive distinct-quadruple prefactor. Polarized cost equals ±N."""
    if n < 4:
        raise ValueError("a_N requires N>=4")
    return float(n) / float(math.comb(n, 4))


def a_N_fraction(n: int) -> tuple[int, int]:
    """a_N = 24 / ((N-1)(N-2)(N-3)) as a reduced rational numerator, denominator."""
    den = (n - 1) * (n - 2) * (n - 3)
    num = 24
    g = math.gcd(num, den)
    return num // g, den // g


def e4_int(M: int, n: int) -> int:
    """Eigenvalue of sum_{i<j<k<l} ZZZZ on any basis state with sum Z = M."""
    if (M - n) % 2 != 0 or abs(M) > n:
        raise ValueError(f"inadmissible M={M} for N={n}")
    val = M**4 - (6 * n - 8) * M * M + 3 * n * (n - 2)
    if val % 24 != 0:
        raise RuntimeError(f"e4 not integral at N={n}, M={M}: {val}")
    return val // 24


def e4_poly_from_M(M: np.ndarray, n: int) -> np.ndarray:
    """Float copy of e4_int, vectorized. Exact for |M|<=N<=20 in float64."""
    M = np.asarray(M)
    val = M**4 - (6 * n - 8) * (M**2) + 3 * n * (n - 2)
    return val / 24.0


def magnetization_from_weight(n: int, weight: np.ndarray) -> np.ndarray:
    """weight = number of Z=-1 bits. M = n_up - n_down = N - 2 weight."""
    return n - 2 * np.asarray(weight)


def pure_cost_from_M(M: np.ndarray, n: int, s: int) -> np.ndarray:
    return float(s) * J * a_N(n) * e4_poly_from_M(M, n)


def ctrl_cost_from_M(M: np.ndarray, n: int, s: int) -> np.ndarray:
    """s * N * m^4 = s * M^4 / N^3."""
    M = np.asarray(M, dtype=np.float64)
    return float(s) * J * (M**4) / float(n**3)


def lower_body_delta(M: np.ndarray, n: int) -> np.ndarray:
    """a_N e_4 - N m^4. Zero at |M|=N; nonzero in between (finite-N terms)."""
    return pure_cost_from_M(M, n, s=1) - ctrl_cost_from_M(M, n, s=1)


def gamma_period(n: int) -> dict:
    """Minimal spectral unit of H_C^pure / (s J).

    e_4 eigenvalues differ by multiples of g = gcd{e_4 - min e_4}.
    H_C differences are multiples of a_N * g, so
        exp(-i (γ+T) H_C) = exp(-i γ H_C) * (global phase on the diagonal)
    at T = 2π / (a_N g) = 2π * D / (24 g), D=(N-1)(N-2)(N-3).
    This is a valid period. validate.py reduces it empirically if a proper
    divisor already closes every diagonal phase.
    Beta period is separate and comes from sum X, not from H_C.
    """
    e4s = [e4_int(m, n) for m in range(-n, n + 1, 2)]
    emin = min(e4s)
    g = 0
    for e in e4s:
        g = math.gcd(g, abs(e - emin))
    num, den = a_N_fraction(n)  # a_N = num/den
    # unit = a_N * g = num*g / den
    unit_num = num * g
    unit_den = den
    gg = math.gcd(unit_num, unit_den)
    unit_num //= gg
    unit_den //= gg
    # T = 2π * unit_den / unit_num
    T = (2.0 * math.pi * unit_den) / float(unit_num)
    return {
        "g_e4": int(g),
        "a_N": a_N(n),
        "a_N_num": num,
        "a_N_den": den,
        "energy_unit_num": int(unit_num),
        "energy_unit_den": int(unit_den),
        "T_gamma": float(T),
        "beta_state_period": float(math.pi),
        "beta_note": (
            "exp(-i(β+π) sum_i X_i) = (-1)^N exp(-i β sum_i X_i). "
            "Projective period of every β is π. Period π/2 is not a symmetry: "
            "eigenvalues of sum X differ by 2, and (-i)^{λ+2} = -(-i)^λ."
        ),
    }


def enumerate_quadruples(n: int) -> list[tuple[int, int, int, int]]:
    return list(combinations(range(n), 4))


def hc_diag_enumerated(n: int, s: int) -> tuple[np.ndarray, list[tuple[int, int, int, int]]]:
    """Dense diagonal of H_C^pure by writing every distinct quadruple once."""
    dim = 1 << n
    x = np.arange(dim, dtype=np.int64)
    hc = np.zeros(dim, dtype=np.float64)
    terms = enumerate_quadruples(n)
    scale = float(s) * J * a_N(n)
    for comb in terms:
        acc = np.ones(dim, dtype=np.float64)
        for b in comb:
            bit = (x >> np.int64(n - 1 - b)) & 1
            acc *= np.where(bit == 0, 1.0, -1.0)
        hc += scale * acc
    return hc, terms


def hc_diag_closed(n: int, s: int, kind: str) -> np.ndarray:
    dim = 1 << n
    wt = np.bitwise_count(np.arange(dim, dtype=np.uint32)).astype(np.int64)
    M = magnetization_from_weight(n, wt).astype(np.float64)
    if kind == "pure":
        return np.asarray(pure_cost_from_M(M, n, s), dtype=np.float64)
    if kind == "collective":
        return np.asarray(ctrl_cost_from_M(M, n, s), dtype=np.float64)
    raise ValueError(kind)


def dicke_weights(n: int) -> np.ndarray:
    return np.arange(n + 1, dtype=np.int64)


def dicke_cost(n: int, s: int, kind: str) -> np.ndarray:
    w = dicke_weights(n)
    M = magnetization_from_weight(n, w).astype(np.float64)
    if kind == "pure":
        return np.asarray(pure_cost_from_M(M, n, s), dtype=np.float64)
    if kind == "collective":
        return np.asarray(ctrl_cost_from_M(M, n, s), dtype=np.float64)
    raise ValueError(kind)


def dicke_X(n: int) -> np.ndarray:
    """Matrix of sum_i X_i in the Dicke basis |D_w>, w = Hamming weight.

    <D_w| sum X |D_{w+1}> = sqrt((w+1)(N-w)).
    """
    X = np.zeros((n + 1, n + 1), dtype=np.float64)
    for w in range(n):
        me = math.sqrt((w + 1) * (n - w))
        X[w, w + 1] = me
        X[w + 1, w] = me
    return X


def plus_dicke(n: int) -> np.ndarray:
    """|+>^{⊗N} in the Dicke basis. Real, P=+1, permutation symmetric."""
    c = np.array(
        [math.sqrt(math.comb(n, w) / float(1 << n)) for w in range(n + 1)],
        dtype=np.complex128,
    )
    return c


def parity_basis(n: int, parity: int) -> np.ndarray:
    """Columns are orthonormal Dicke vectors of definite flip parity.

    parity = +1 or -1. P|D_w> = |D_{N-w}>.
    Even N has a self-paired weight N/2 in the +1 sector only.
    Odd N has no self-paired weight. Both parities are built for either.
    """
    if parity not in (+1, -1):
        raise ValueError(parity)
    cols = []
    # Pairs (w, N-w) with w < N-w. Count is N//2 for every N
    # (odd N has (N+1)/2 pairs and no fixed weight; (N+1)//2 handles both).
    for w in range((n + 1) // 2):
        if n - w == w:
            continue
        v = np.zeros(n + 1, dtype=np.float64)
        v[w] = 1.0 / math.sqrt(2.0)
        v[n - w] = float(parity) / math.sqrt(2.0)
        cols.append(v)
    if n % 2 == 0 and parity == +1:
        v = np.zeros(n + 1, dtype=np.float64)
        v[n // 2] = 1.0
        cols.append(v)
    if not cols:
        return np.zeros((n + 1, 0), dtype=np.float64)
    return np.column_stack(cols)


def dicke_parity_expectation(c: np.ndarray) -> float:
    """<c|P|c> with (Pc)[w] = c[N-w]."""
    return float(np.vdot(c, c[::-1]).real)


def sector_hamiltonian(n: int, s: int, gamma: float, kind: str, parity: int):
    """Dense H restricted to one flip-parity block of the symmetric irrep."""
    hc = dicke_cost(n, s, kind)
    X = dicke_X(n)
    H = np.diag(hc) - float(gamma) * X
    B = parity_basis(n, parity)
    if B.shape[1] == 0:
        raise RuntimeError("empty parity sector")
    Hs = B.T @ H @ B
    Hs = 0.5 * (Hs + Hs.T)
    evals, evecs = np.linalg.eigh(Hs)
    # columns in full Dicke coordinates
    vecs = B @ evecs
    return evals.real.astype(np.float64), vecs.astype(np.complex128), hc, X


def symmetric_dicke_spectrum(n: int, s: int, gamma: float, kind: str) -> dict:
    even_e, even_v, hc, X = sector_hamiltonian(n, s, gamma, kind, +1)
    odd_e, odd_v, _, _ = sector_hamiltonian(n, s, gamma, kind, -1)
    return {
        "hc": hc,
        "X": X,
        "even_evals": even_e,
        "even_vecs": even_v,
        "odd_evals": odd_e,
        "odd_vecs": odd_v,
        "E0_even": float(even_e[0]),
        "E1_even": float(even_e[1]) if len(even_e) > 1 else float("nan"),
        "E0_odd": float(odd_e[0]),
        "mult_even": int(np.sum(even_e <= even_e[0] + 1e-8)),
        "mult_odd": int(np.sum(odd_e <= odd_e[0] + 1e-8)),
    }


def build_sparse_H(n: int, hc_diag: np.ndarray, gamma: float):
    dim = hc_diag.shape[0]
    H = diags(hc_diag.astype(np.float64), offsets=0, format="csr", dtype=np.float64)
    idx = np.arange(dim, dtype=np.int64)
    blocks = []
    g = float(gamma)
    for q in range(n):
        bit = 1 << (n - 1 - q)
        a = idx[(idx & bit) == 0]
        b = a + bit
        data = np.full(a.shape[0], -g, dtype=np.float64)
        blocks.append(coo_matrix((data, (a, b)), shape=(dim, dim)))
        blocks.append(coo_matrix((data, (b, a)), shape=(dim, dim)))
    for blk in blocks:
        H = H + blk.tocsr()
    return H.tocsr()


def _phase_fix(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.complex128)
    s = np.sum(v)
    if abs(s) < 1e-12:
        k = int(np.argmax(np.abs(v)))
        s = v[k]
    if abs(s) < 1e-18:
        return v
    return v * (abs(s) / s)


def full_spectrum_edges(
    n: int,
    s: int,
    gamma: float,
    kind: str,
    k_bottom: int = 24,
    tol: float = 1e-7,
) -> dict:
    """Extremal eigenpairs of the full 2^N Hamiltonian.

    N<=10 uses dense Hermitian eigh (exact multiplicity). Larger N uses
    ARPACK on the sparse matrix (diagonal cost + N bit flips).
    """
    hc = hc_diag_closed(n, s, kind)
    dim = hc.shape[0]
    if n <= 10:
        H = np.diag(hc)
        idx = np.arange(dim, dtype=np.int64)
        g = float(gamma)
        for q in range(n):
            bit = 1 << (n - 1 - q)
            a = idx[(idx & bit) == 0]
            b = a + bit
            H[a, b] -= g
            H[b, a] -= g
        evals, evecs = np.linalg.eigh(H)
        evals = evals.real.astype(np.float64)
        e0 = float(evals[0])
        emax = float(evals[-1])
        mask = evals <= e0 + tol
        mult = int(mask.sum())
        n_store = min(evals.size, max(mult, min(k_bottom, mult + 4)))
        above = np.flatnonzero(evals > e0 + tol)
        e1 = float(evals[above[0]]) if above.size else float("nan")
        vecs = np.stack([_phase_fix(evecs[:, i]) for i in range(min(mult, 8))], axis=1)
        return {
            "E0": e0,
            "E1": e1,
            "gap": float(e1 - e0) if e1 == e1 else float("nan"),
            "Emax": emax,
            "evals_bottom": evals[:n_store].copy(),
            "vecs_bottom": vecs,
            "multiplicity": mult,
            "truncated": False,
            "method": "dense_eigh",
            "hc_diag": hc,
        }
    k_bottom = min(k_bottom, dim - 2)
    H = build_sparse_H(n, hc, gamma)
    maxiter = 1500
    ev_b, vec_b = eigsh(H, k=k_bottom, which="SA", return_eigenvectors=True, tol=1e-8, maxiter=maxiter)
    ev_t = eigsh(H, k=2, which="LA", return_eigenvectors=False, tol=1e-8, maxiter=maxiter)
    order = np.argsort(ev_b.real)
    ev_b = ev_b.real[order]
    vec_b = vec_b[:, order]
    e0 = float(ev_b[0])
    emax = float(np.max(ev_t.real))
    mask = ev_b <= e0 + tol
    mult_found = int(mask.sum())
    truncated = bool(mult_found == k_bottom)
    above = np.flatnonzero(ev_b > e0 + tol)
    e1 = float(ev_b[above[0]]) if above.size else float("nan")
    nvec = min(mult_found, 8)
    vecs = np.stack([_phase_fix(vec_b[:, i]) for i in range(nvec)], axis=1)
    return {
        "E0": e0,
        "E1": e1,
        "gap": float(e1 - e0) if e1 == e1 else float("nan"),
        "Emax": emax,
        "evals_bottom": ev_b.copy(),
        "vecs_bottom": vecs,
        "multiplicity": mult_found,
        "truncated": truncated,
        "method": "eigsh",
        "hc_diag": hc,
    }


def flip_expectation(psi: np.ndarray) -> float:
    """<ψ|∏_i X_i|ψ>. Flip is index XOR (dim-1) in the MSB statevector."""
    dim = psi.shape[0]
    flipped = psi[np.arange(dim) ^ (dim - 1)]
    return float(np.vdot(psi, flipped).real)


def symmetric_weight(psi: np.ndarray, n: int) -> float:
    """|P_{S=N/2} ψ|^2. Dicke |D_w> is the uniform vector on weight w."""
    dim = 1 << n
    wt = np.bitwise_count(np.arange(dim, dtype=np.uint32))
    acc = 0.0
    for w in range(n + 1):
        m = wt == w
        cw = int(m.sum())
        mean = psi[m].mean()
        acc += cw * float(abs(mean) ** 2)
    return float(acc)


def full_to_dicke(psi: np.ndarray, n: int) -> np.ndarray:
    dim = 1 << n
    wt = np.bitwise_count(np.arange(dim, dtype=np.uint32))
    c = np.zeros(n + 1, dtype=np.complex128)
    for w in range(n + 1):
        m = wt == w
        cw = float(m.sum())
        c[w] = psi[m].sum() / math.sqrt(cw)
    return c


def dicke_to_full(c: np.ndarray, n: int) -> np.ndarray:
    dim = 1 << n
    wt = np.bitwise_count(np.arange(dim, dtype=np.uint32))
    scale = np.array([1.0 / math.sqrt(math.comb(n, w)) for w in range(n + 1)])
    return np.asarray(c, dtype=np.complex128)[wt] * scale[wt]


def half_cut_entropy_full(psi: np.ndarray, n: int) -> dict:
    """Von Neumann entropy of qubits [0, n//2), qubit 0 = MSB = slowest index.

    Returns nats and bits. For a permutation-symmetric state every subsystem
    of this size has the same entropy; that equivalence is NOT assumed when
    the vector is a general full-space state.
    """
    nA = n // 2
    nB = n - nA
    mat = np.asarray(psi, dtype=np.complex128).reshape(1 << nA, 1 << nB)
    # ρ_A = ψ ψ†
    rho = mat @ mat.conj().T
    evals = np.linalg.eigvalsh(rho).real
    evals = evals[evals > 1e-16]
    evals = evals / evals.sum()
    nats = float(-np.sum(evals * np.log(evals)))
    return {
        "S_nats": nats,
        "S_bits": float(nats / math.log(2.0)),
        "subsystem": f"qubits[0:{nA}) MSB-first, partner qubits[{nA}:{n})",
        "nA": nA,
        "nB": nB,
    }


def half_cut_entropy_dicke(c: np.ndarray, n: int) -> dict:
    """Half-cut entropy of a permutation-symmetric state given by Dicke coeffs.

    |D_w> = sum_k sqrt( C(nA,k) C(nB,w-k) / C(N,w) ) |D_k>_A |D_{w-k}>_B.
    Different (k, m) are orthogonal, so the Schmidt matrix is (nA+1) by (nB+1).
    """
    nA = n // 2
    nB = n - nA
    amp = np.zeros((nA + 1, nB + 1), dtype=np.complex128)
    for k in range(nA + 1):
        for m in range(nB + 1):
            w = k + m
            if w > n:
                continue
            if math.comb(n, w) == 0:
                continue
            amp[k, m] = c[w] * math.sqrt(
                math.comb(nA, k) * math.comb(nB, m) / math.comb(n, w)
            )
    svals = np.linalg.svd(amp, compute_uv=False)
    p = np.clip(svals.real**2, 0.0, None)
    p = p[p > 1e-16]
    if p.size == 0:
        return {
            "S_nats": 0.0,
            "S_bits": 0.0,
            "subsystem": f"qubits[0:{nA}) MSB-first; symmetric-state Schmidt",
            "nA": nA,
            "nB": nB,
        }
    p = p / p.sum()
    nats = float(-np.sum(p * np.log(p)))
    return {
        "S_nats": nats,
        "S_bits": float(nats / math.log(2.0)),
        "subsystem": (
            f"qubits[0:{nA}) MSB-first; entropy via Dicke Schmidt. "
            "Permutation symmetry => every nA-qubit subsystem matches."
        ),
        "nA": nA,
        "nB": nB,
    }


def moments_from_dicke(c: np.ndarray, n: int, s: int, kind: str) -> dict:
    """Magnetization moments and spatially averaged correlators.

    C2 = average_{i<j} <Z_i Z_j> = (<M^2> - N) / (N(N-1))
    C4 = average distinct <Z_i Z_j Z_k Z_l> = <e_4> / C(N,4)
    Both identities hold without permutation symmetry of the *state*, because
    they only use operator averages. For a Dicke state, populations |c_w|^2
    determine them exactly.
    """
    p = np.abs(np.asarray(c)) ** 2
    p = p / p.sum()
    w = np.arange(n + 1, dtype=np.float64)
    M = (n - 2.0 * w)
    m1 = float(np.sum(p * M) / n)
    m2 = float(np.sum(p * (M**2)) / n**2)
    m4 = float(np.sum(p * (M**4)) / n**4)
    M2 = float(np.sum(p * (M**2)))
    C2 = (M2 - n) / float(n * (n - 1))
    e4 = e4_poly_from_M(M, n)
    e4_avg = float(np.sum(p * e4))
    C4 = e4_avg / float(math.comb(n, 4))
    hc = dicke_cost(n, s, kind)
    X = dicke_X(n)
    cc = np.asarray(c, dtype=np.complex128)
    hc_exp = float(np.vdot(cc, hc * cc).real)
    x_exp = float(np.vdot(cc, X @ cc).real)
    # classical-like population entropy of M
    pnz = p[p > 1e-16]
    S_M = float(-np.sum(pnz * np.log(pnz)))
    return {
        "m1": m1,
        "m2": m2,
        "m4": m4,
        "C2": float(C2),
        "C4": float(C4),
        "e4_avg": e4_avg,
        "Hc": hc_exp,
        "sumX": x_exp,
        "S_M_nats": S_M,
        "p_weight": p.tolist(),
    }


def symmetric_product_overlap(c: np.ndarray, n: int) -> dict:
    """Max |<⊗|θ><ψ>|^2 over permutation-symmetric real product states.

    |θ> = cos(θ/2)|0> + sin(θ/2)|1>, θ∈[0, π].
    Overlap = sum_w c_w * sqrt(C(N,w)) * cos(θ/2)^{N-w} * sin(θ/2)^w.
    For a permutation-symmetric target this is the mean-field product overlap.
    """
    cc = np.asarray(c, dtype=np.complex128)
    ws = np.arange(n + 1)
    sqrtC = np.array([math.sqrt(math.comb(n, w)) for w in ws])
    thetas = np.linspace(0.0, math.pi, 361)
    best_F = -1.0
    best_th = 0.0
    best_amp = 0.0
    for th in thetas:
        ov = _sym_prod_amp(cc, sqrtC, n, float(th))
        F = abs(ov) ** 2
        if F > best_F:
            best_F = F
            best_th = float(th)
            best_amp = ov
    # local polish
    th = best_th
    for step in (1e-3, 2e-4, 5e-5):
        improved = True
        while improved:
            improved = False
            for trial in (th - step, th + step):
                if trial < 0.0 or trial > math.pi:
                    continue
                ov = _sym_prod_amp(cc, sqrtC, n, trial)
                F = abs(ov) ** 2
                if F > best_F + 1e-15:
                    best_F = F
                    th = trial
                    best_th = trial
                    best_amp = ov
                    improved = True
    return {
        "F_product_symmetric": float(best_F),
        "theta": float(best_th),
        "overlap_amp": complex(best_amp),
        "note": "best real symmetric product; not an unrestricted product search",
    }


def _sym_prod_amp(c, sqrtC, n, theta: float) -> complex:
    cth = math.cos(theta / 2.0)
    sth = math.sin(theta / 2.0)
    # guard 0**0
    acc = 0.0j
    for w in range(n + 1):
        amp_w = (cth ** (n - w)) * (sth ** w)
        acc += c[w] * sqrtC[w] * amp_w
    return acc


def cat_overlap(c: np.ndarray) -> float:
    """Overlap with the even GHZ cat (|0...0> + |1...1>)/sqrt(2) = (|D_0>+|D_N>)/sqrt(2)."""
    ov = (c[0] + c[-1]) / math.sqrt(2.0)
    return float(abs(ov) ** 2)


def best_basis_population(c: np.ndarray, n: int) -> float:
    """max_x |<x|ψ>|^2 for a Dicke state. Each x of weight w has amp c_w/sqrt(C)."""
    p = 0.0
    for w in range(n + 1):
        cw = math.comb(n, w)
        p = max(p, float(abs(c[w]) ** 2) / cw)
    return p


def classical_optimum(n: int, s: int, kind: str) -> dict:
    """Minima of the diagonal cost over admissible M. Gamma = 0 manifold."""
    rows = []
    for M in range(-n, n + 1, 2):
        if kind == "pure":
            e = float(pure_cost_from_M(np.array(M, dtype=float), n, s))
            e4 = e4_int(M, n)
        else:
            e = float(ctrl_cost_from_M(np.array(M, dtype=float), n, s))
            e4 = None
        rows.append((e, M, e4))
    emin = min(r[0] for r in rows)
    mins = [r for r in rows if abs(r[0] - emin) <= 1e-10]
    deg = 0
    for _, M, _ in mins:
        ndown = (n - M) // 2
        deg += math.comb(n, ndown)
    return {
        "E_classical": emin,
        "M_list": [int(m) for _, m, _ in mins],
        "e4_list": [e for _, _, e in mins],
        "degeneracy": int(deg),
    }


def apply_mixer_full(psi: np.ndarray, beta: float, n: int) -> np.ndarray:
    """exp(-i β sum_i X_i) with qubit 0 on the slowest reshape axis (MSB)."""
    c = math.cos(beta)
    s = math.sin(beta)
    # Copy: reshape of a complex128 buffer is a view, and the slice assigns
    # below would otherwise overwrite the caller's statevector.
    a = np.array(psi, dtype=np.complex128, copy=True).reshape([2] * n)
    for q in range(n):
        a = np.moveaxis(a, q, 0)
        p0 = a[0].copy()
        p1 = a[1].copy()
        a[0] = c * p0 - 1j * s * p1
        a[1] = -1j * s * p0 + c * p1
        a = np.moveaxis(a, 0, q)
    return np.ascontiguousarray(a).reshape(-1)


def evolve_full(psi0: np.ndarray, hc: np.ndarray, n: int, betas, gammas) -> np.ndarray:
    psi = np.asarray(psi0, dtype=np.complex128).copy()
    for beta, gamma in zip(betas, gammas):
        psi = psi * np.exp(-1j * float(gamma) * hc)
        psi = apply_mixer_full(psi, float(beta), n)
    return psi


def plus_full(n: int) -> np.ndarray:
    dim = 1 << n
    return np.full(dim, 1.0 / math.sqrt(dim), dtype=np.complex128)


@dataclass
class DickeModel:
    """Fast exact QAOA generators inside the symmetric irrep (dimension N+1).

    The circuit that starts at |+> and alternates diagonal H_C with sum X
    never leaves this subspace. That is a theorem for this family (both
    generators commute with every permutation) plus a numerical check against
    the full statevector in validate.py. It is NOT a theorem that the
    full-space ground state lies in the subspace: callers must compare
    E0_full with E0_even.
    """

    n: int
    s: int
    gamma: float
    kind: str
    hc: np.ndarray
    X: np.ndarray
    X_evals: np.ndarray
    X_vecs: np.ndarray
    c0: np.ndarray
    T_gamma: float
    period_info: dict

    @staticmethod
    def build(n: int, s: int, gamma: float, kind: str = "pure") -> "DickeModel":
        hc = dicke_cost(n, s, kind)
        X = dicke_X(n)
        evals, vecs = np.linalg.eigh(X)
        period = gamma_period(n) if kind == "pure" else {
            "T_gamma": float("nan"),
            "beta_state_period": float(math.pi),
            "note": "collective control spectrum is integer powers of M/N; period derived separately if needed",
        }
        # Collective cost eigenvalues are M^4 / N^3. Differences share a unit
        # 1/N^3 times gcd of M^4 differences. Compute it so anneal/QAOA logs
        # have a real number for both kinds.
        if kind == "collective":
            period = collective_gamma_period(n)
        return DickeModel(
            n=n,
            s=int(s),
            gamma=float(gamma),
            kind=kind,
            hc=hc.astype(np.float64),
            X=X,
            X_evals=evals.real.astype(np.float64),
            X_vecs=vecs.real.astype(np.float64),
            c0=plus_dicke(n),
            T_gamma=float(period["T_gamma"]),
            period_info=period,
        )

    def evolve(self, betas, gammas, c0: np.ndarray | None = None) -> np.ndarray:
        c = self.c0.copy() if c0 is None else np.asarray(c0, dtype=np.complex128).copy()
        hc = self.hc
        ev = self.X_evals
        V = self.X_vecs
        for beta, gamma in zip(np.asarray(betas, dtype=float), np.asarray(gammas, dtype=float)):
            c = c * np.exp(-1j * float(gamma) * hc)
            coeff = c @ V  # V real orthogonal, coeff = V^T c
            coeff = np.exp(-1j * float(beta) * ev) * coeff
            c = coeff @ V.T
        return c

    def energy(self, c: np.ndarray) -> float:
        hc_e = float(np.vdot(c, self.hc * c).real)
        x_e = float(np.vdot(c, self.X @ c).real)
        return hc_e - self.gamma * x_e

    def observables(self, c: np.ndarray) -> dict:
        mom = moments_from_dicke(c, self.n, self.s, self.kind)
        ent = half_cut_entropy_dicke(c, self.n)
        prod = symmetric_product_overlap(c, self.n)
        return {
            **mom,
            **{f"cut_{k}": v for k, v in ent.items() if k != "p_weight"},
            "S_half_bits": ent["S_bits"],
            "S_half_nats": ent["S_nats"],
            "subsystem": ent["subsystem"],
            "F_product": prod["F_product_symmetric"],
            "F_cat": cat_overlap(c),
            "F_best_basis": best_basis_population(c, self.n),
            "parity": dicke_parity_expectation(c),
            "E": self.energy(c),
        }


def collective_gamma_period(n: int) -> dict:
    """Period of exp(-i γ s N m^4). Eigenvalues s M^4 / N^3.

    M^4 differences are multiples of g4. Unit = g4 / N^3, T = 2π N^3 / g4.
    """
    vals = []
    for M in range(-n, n + 1, 2):
        vals.append(M**4)
    vmin = min(vals)
    g = 0
    for v in vals:
        g = math.gcd(g, abs(v - vmin))
    # unit of N m^4 is g / N^3
    unit_num = g
    unit_den = n**3
    gg = math.gcd(unit_num, unit_den)
    unit_num //= gg
    unit_den //= gg
    T = (2.0 * math.pi * unit_den) / float(unit_num)
    return {
        "g_M4": int(g),
        "energy_unit_num": int(unit_num),
        "energy_unit_den": int(unit_den),
        "T_gamma": float(T),
        "beta_state_period": float(math.pi),
        "beta_note": "same sum-X projective period π as the pure family",
    }


def subspace_fidelity(c: np.ndarray, vecs: np.ndarray) -> float:
    """||P ψ||^2 for orthonormal columns of vecs, ψ a Dicke vector."""
    if vecs.size == 0:
        return 0.0
    amps = vecs.conj().T @ np.asarray(c, dtype=np.complex128)
    return float(np.vdot(amps, amps).real)


def eta_score(E: float, E0: float, Emax: float) -> float:
    """η = (E - Emax) / (E0 - Emax). Equals 1 at the ground state, 0 at Emax.

    Same convention as quantum-p-spin/run_pagano_af_qaoa.py. E0 and Emax are
    full-H edges unless a caller deliberately passes sector edges.
    """
    d = E0 - Emax
    if abs(d) < 1e-15:
        return float("nan")
    return float((E - Emax) / d)


def canonical_angles(betas: np.ndarray, gammas: np.ndarray, T_gamma: float) -> tuple[np.ndarray, np.ndarray]:
    """Fold each angle into one projective period, then fix the global sign.

    Folding is for reporting only. Bootstrap seeds use the raw optimizer
    output so a real branch switch is not averaged away. Global sign flip
    (β,γ) -> (-β,-γ) is the complex-conjugation symmetry of a real initial
    state evolved under real generators; it preserves energy and |overlap|^2.
    """
    b = np.array(betas, dtype=float, copy=True)
    g = np.array(gammas, dtype=float, copy=True)
    # projective β period π, representative (-π/2, π/2]
    b = (b + 0.5 * math.pi) % math.pi - 0.5 * math.pi
    if T_gamma == T_gamma and T_gamma > 0:  # not NaN
        half = 0.5 * T_gamma
        g = (g + half) % T_gamma - half
    if g.sum() < -1e-10 or (abs(g.sum()) <= 1e-10 and b.sum() < 0):
        b = -b
        g = -g
    return b, g


def l1_duration(betas, gammas) -> float:
    """Sum |β_k| + |γ_k|. Proxy for total generator-time, not a lab clock.

    Circuit convention: layer k applies exp(-i γ_k H_C) then exp(-i β_k sum X),
    k increasing. Angles already include every factor of the normalized
    generators (no extra 1/J0). A β shift of π on one layer multiplies the
    state by the global phase (-1)^N.
    """
    return float(np.sum(np.abs(np.asarray(betas, dtype=float))) + np.sum(np.abs(np.asarray(gammas, dtype=float))))
