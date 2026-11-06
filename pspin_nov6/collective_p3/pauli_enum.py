"""Independent Pauli enumeration for the collective cubic ferromagnet.

This file does not import the symmetric-subspace solver. It builds the
diagonal cost from explicit products of Z eigenvalues and checks the
operator identity

    (sum_i Z_i)^3 = 6 * sum_{i<j<k} Z_i Z_j Z_k + (3N - 2) * sum_i Z_i

by exact integer arithmetic on every computational-basis configuration.
"""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np


def bit_z(x: int, i: int) -> int:
    """Z eigenvalue on qubit i. Convention: bit 0 is |0> with Z=+1."""
    return 1 if ((x >> i) & 1) == 0 else -1


def sumz_cubes_identity_failures(n: int) -> int:
    """Return how many basis states violate the cubic identity (exact ints)."""
    failures = 0
    for x in range(1 << n):
        z = [bit_z(x, i) for i in range(n)]
        s = sum(z)
        lhs = s * s * s
        triples = 0
        for i in range(n):
            for j in range(i + 1, n):
                zij = z[i] * z[j]
                for k in range(j + 1, n):
                    triples += zij * z[k]
        rhs = 6 * triples + (3 * n - 2) * s
        if lhs != rhs:
            failures += 1
    return failures


def pauli_coefficients(n: int) -> tuple[Fraction, Fraction]:
    """Coefficients of each ZZZ triple and each Z in H_cost = -N m_z^3.

    H_cost = -(sum Z)^3 / N^2
           = -6/N^2 * sum_{i<j<k} ZZZ - (3N-2)/N^2 * sum_i Z.
    """
    n2 = n * n
    return Fraction(-6, n2), Fraction(-(3 * n - 2), n2)


def enumerated_cost_diagonal(n: int) -> np.ndarray:
    """Cost diagonal from the expanded Pauli strings, not from m^3."""
    c3, c1 = pauli_coefficients(n)
    c3f = float(c3)
    c1f = float(c1)
    diag = np.empty(1 << n, dtype=np.float64)
    for x in range(1 << n):
        z = [bit_z(x, i) for i in range(n)]
        s = float(sum(z))
        triples = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                zij = z[i] * z[j]
                for k in range(j + 1, n):
                    triples += zij * z[k]
        diag[x] = c3f * triples + c1f * s
    return diag


def closed_form_cost_diagonal(n: int) -> np.ndarray:
    """Same diagonal from -(sum Z)^3 / N^2. Kept here only as the compare target."""
    diag = np.empty(1 << n, dtype=np.float64)
    n2 = float(n * n)
    for x in range(1 << n):
        # popcount of bits that are |1> (Z = -1)
        w = x.bit_count()
        m = n - 2 * w
        diag[x] = -(m ** 3) / n2
    return diag


def distinct_index_cost_diagonal(n: int) -> np.ndarray:
    """Three-body piece only: -6/N^2 sum_{i<j<k} ZZZ, no longitudinal field."""
    c3, _c1 = pauli_coefficients(n)
    c3f = float(c3)
    diag = np.empty(1 << n, dtype=np.float64)
    for x in range(1 << n):
        z = [bit_z(x, i) for i in range(n)]
        triples = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                zij = z[i] * z[j]
                for k in range(j + 1, n):
                    triples += zij * z[k]
        diag[x] = c3f * triples
    return diag


def write_term_list(path: str, n: int) -> None:
    """Exact fractional Pauli list of H = H_cost - Gamma * sum X."""
    c3, c1 = pauli_coefficients(n)
    lines = [
        f"# Collective ferromagnetic p=3, N={n}",
        "# H = -N * m_z^3 - Gamma * sum_i X_i,  m_z = (sum_i Z_i)/N",
        "# Z convention: |0> eigenvalue +1, |1> eigenvalue -1.",
        "# Distinct-index expansion (exact):",
        "#   -N m_z^3 = c3 * sum_{i<j<k} Z_i Z_j Z_k + c1 * sum_i Z_i",
        f"#   c3 = {c3} = {float(c3)!r}",
        f"#   c1 = {c1} = {float(c1)!r}",
        "# The c1 term is the finite-N lower-body field. It is absent from a",
        "# pure distinct-index triple sum. Its total energy is O(1), not O(N):",
        f"#   c1 * <sum Z> = {c1} * N * <m_z> = {Fraction(c1.numerator * n, c1.denominator)} <m_z>",
        "#   = -(3 - 2/N) <m_z>.",
        "body,coeff_exact,coeff_float,qubits",
    ]
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                lines.append(f"ZZZ,{c3},{float(c3):.16e},{i} {j} {k}")
    for i in range(n):
        lines.append(f"Z,{c1},{float(c1):.16e},{i}")
    for i in range(n):
        lines.append(f"X,-Gamma,-Gamma,{i}")
    lines.append(f"# n_triples={math.comb(n, 3)} n_field={n} n_mixer={n}")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def dense_hamiltonian(n: int, gamma: float) -> np.ndarray:
    """Full real symmetric matrix from the enumerated diagonal plus -Gamma sum X."""
    diag = enumerated_cost_diagonal(n)
    dim = 1 << n
    h = np.diag(diag)
    for i in range(n):
        for x in range(dim):
            y = x ^ (1 << i)
            if y > x:
                h[x, y] -= gamma
                h[y, x] -= gamma
    return h


def apply_sum_x(psi: np.ndarray, n: int) -> np.ndarray:
    """Independent full-space application of sum_i X_i."""
    out = np.zeros_like(psi)
    dim = psi.shape[0]
    for i in range(n):
        # X flips bit i. Iterate even/odd pairs.
        step = 1 << i
        for base in range(0, dim, step << 1):
            a = slice(base, base + step)
            b = slice(base + step, base + 2 * step)
            out[a] += psi[b]
            out[b] += psi[a]
    return out


def embed_dicke(c: np.ndarray, n: int) -> np.ndarray:
    """Spread symmetric-irrep amplitudes uniformly onto computational basis states."""
    out = np.zeros(1 << n, dtype=np.complex128)
    scale = [math.sqrt(math.comb(n, w)) for w in range(n + 1)]
    for x in range(1 << n):
        w = x.bit_count()
        out[x] = c[w] / scale[w]
    return out


def evolve_statevector(
    psi0: np.ndarray,
    n: int,
    gammas: np.ndarray,
    betas: np.ndarray,
    cost_diag: np.ndarray,
) -> np.ndarray:
    """Replay QAOA in the full Hilbert space.

    Layer order: cost then mixer, layer 0 applied first.
    U_C(gamma) = exp(-i gamma H_cost), H_cost diagonal.
    U_M(beta) = exp(-i beta sum X) = tensor_i (cos(beta) I - i sin(beta) X).
    """
    psi = np.array(psi0, dtype=np.complex128, copy=True)
    dim = 1 << n
    for gamma, beta in zip(gammas, betas):
        psi *= np.exp(-1j * float(gamma) * cost_diag)
        cb = math.cos(float(beta))
        sb = math.sin(float(beta))
        # Apply exp(-i beta X) = cb I - i sb X on each qubit, low index first.
        for i in range(n):
            step = 1 << i
            new = np.empty_like(psi)
            for base in range(0, dim, step << 1):
                a = slice(base, base + step)
                b = slice(base + step, base + 2 * step)
                pa = psi[a]
                pb = psi[b]
                new[a] = cb * pa - 1j * sb * pb
                new[b] = cb * pb - 1j * sb * pa
            psi = new
    return psi
