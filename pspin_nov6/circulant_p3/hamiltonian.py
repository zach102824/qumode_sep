"""Two-window circulant 3-body ring.

H = -sum_i Z_i Z_{i+1} Z_{i+3} + kappa * sum_i Z_i Z_{i+2} Z_{i+3} - Gamma * sum_i X_i

Indices are periodic. Bit i of a computational basis integer is qubit i.
Bit 0 means |0> with Z = +1. Bit 1 means |1> with Z = -1.

This module builds the diagonal from clause parities. A separate Kronecker
builder in verify_algebra.py must agree before any QAOA result is trusted.
"""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np


def sites_of(i: int, offsets: tuple[int, int, int], n: int) -> tuple[int, int, int]:
    return tuple(sorted((i + d) % n for d in offsets))


def clause_list(n: int, kappa: float) -> list[tuple[tuple[int, int, int], Fraction]]:
    """Signed triples. Coefficients are exact rationals when kappa is rational."""
    k = Fraction(kappa).limit_denominator(100)
    clauses: dict[tuple[int, int, int], Fraction] = {}
    for i in range(n):
        a = sites_of(i, (0, 1, 3), n)
        b = sites_of(i, (0, 2, 3), n)
        clauses[a] = clauses.get(a, Fraction(0)) + Fraction(-1)
        clauses[b] = clauses.get(b, Fraction(0)) + k
    # Drop exact cancellations. Merged supports are real, not a coding accident.
    return [(supp, coeff) for supp, coeff in sorted(clauses.items()) if coeff != 0]


def z_product(x: int, sites: tuple[int, int, int]) -> int:
    """Product of Z eigenvalues on a basis state. +1 or -1."""
    parity = 0
    for s in sites:
        parity ^= (x >> s) & 1
    return 1 if parity == 0 else -1


def cost_diagonal(n: int, kappa: float) -> np.ndarray:
    clauses = clause_list(n, kappa)
    diag = np.zeros(1 << n, dtype=np.float64)
    for supp, coeff in clauses:
        c = float(coeff)
        for x in range(1 << n):
            diag[x] += c * z_product(x, supp)
    return diag


def cost_diagonal_from(clauses: list[tuple[tuple[int, int, int], Fraction]], n: int) -> np.ndarray:
    diag = np.zeros(1 << n, dtype=np.float64)
    for supp, coeff in clauses:
        c = float(coeff)
        for x in range(1 << n):
            diag[x] += c * z_product(x, supp)
    return diag


def apply_sum_x(psi: np.ndarray, n: int) -> np.ndarray:
    out = np.zeros_like(psi)
    dim = psi.shape[0]
    for i in range(n):
        step = 1 << i
        for base in range(0, dim, step << 1):
            a = slice(base, base + step)
            b = slice(base + step, base + 2 * step)
            out[a] += psi[b]
            out[b] += psi[a]
    return out


def energy_of(psi: np.ndarray, diag: np.ndarray, gamma: float, n: int) -> float:
    hc = float(np.real(np.vdot(psi, diag * psi)))
    hx = float(np.real(np.vdot(psi, apply_sum_x(psi, n))))
    return hc - gamma * hx


def translate_bits(x: int, n: int, shift: int = 1) -> int:
    y = 0
    for i in range(n):
        if (x >> i) & 1:
            y |= 1 << ((i + shift) % n)
    return y


def reflect_bits(x: int, n: int) -> int:
    """i -> -i mod N. Exchanges windows (0,1,3) and (0,2,3)."""
    y = 0
    for i in range(n):
        if (x >> i) & 1:
            y |= 1 << ((-i) % n)
    return y


def permute_state(psi: np.ndarray, n: int, perm) -> np.ndarray:
    out = np.zeros_like(psi)
    for x in range(psi.shape[0]):
        out[perm(x, n)] += psi[x]
    return out


def gf2_satisfiability(n: int) -> dict:
    """Can a gauge make every clause prefer the product +1?

    Family A already wants product +1, family B wants product -1 when kappa>0.
    Gauge g_i = (-1)^{x_i} multiplies a clause sign by the product of its g's.
    Over GF(2), satisfiability (existence of a fully satisfying assignment) is:

        x_i + x_{i+1} + x_{i+3} = 0
        x_i + x_{i+2} + x_{i+3} = 1

    Consistent => gauge-equivalent to a ferromagnet on this hypergraph
    (all-+ is a classical ground state after local X's). Inconsistent =>
    frustrated. Rank is computed here; divisibility by 7 is not used.
    """
    n_eq = 2 * n
    aug = np.zeros((n_eq, n + 1), dtype=np.uint8)
    for i in range(n):
        for d in (0, 1, 3):
            aug[i, (i + d) % n] ^= 1
        aug[i, n] = 0
        for d in (0, 2, 3):
            aug[n + i, (i + d) % n] ^= 1
        aug[n + i, n] = 1
    rank, consistent, pivots = _rref(aug)
    nullity = n - rank
    return {
        "n": n,
        "n_equations": n_eq,
        "rank": int(rank),
        "consistent": bool(consistent),
        "nullity": int(nullity),
        "n_satisfying_assignments": int(2 ** nullity) if consistent else 0,
    }


def _rref(aug: np.ndarray) -> tuple[int, bool, list[int]]:
    a = aug.copy()
    rows, cols = a.shape
    n_var = cols - 1
    pivots: list[int] = []
    r = 0
    for c in range(n_var):
        pivot = None
        for i in range(r, rows):
            if a[i, c]:
                pivot = i
                break
        if pivot is None:
            continue
        if pivot != r:
            a[[r, pivot]] = a[[pivot, r]]
        for i in range(rows):
            if i != r and a[i, c]:
                a[i] ^= a[r]
        pivots.append(c)
        r += 1
    consistent = True
    for i in range(r, rows):
        if a[i, n_var]:
            consistent = False
            break
    return r, consistent, pivots


def classical_summary(n: int, kappa: float) -> dict:
    diag = cost_diagonal(n, kappa)
    e_min = float(np.min(diag))
    e_max = float(np.max(diag))
    deg = int(np.sum(np.abs(diag - e_min) < 1e-10))
    # Fully satisfied energy: every clause at its lower value -|coeff|.
    sat = 0.0
    for _supp, coeff in clause_list(n, kappa):
        sat += -abs(float(coeff))
    return {
        "e_min": e_min,
        "e_max": e_max,
        "degeneracy": deg,
        "fully_satisfied_energy": sat,
        "frustrated": e_min > sat + 1e-8,
        "violation_above_sat": e_min - sat,
    }


def gamma_period(diag: np.ndarray) -> dict:
    """Minimal T>0 with exp(-i T E) a global phase on every basis energy.

    Energies are converted to exact rationals on a rounded 1e-12 grid, then
    the gcd of all differences fixes T = 2π / (gcd in energy units).
    """
    # Represent E = p/q with q dividing a common denominator.
    scaled = np.rint(diag * 2.0).astype(np.int64)  # works for halves and integers
    if np.max(np.abs(diag - scaled / 2.0)) > 1e-8:
        raise ValueError("diagonal is not on the half-integer lattice")
    g = 0
    # gcd of all differences; use a sample of consecutive unique values plus extremes
    uniq = np.unique(scaled)
    for i in range(len(uniq)):
        g = math.gcd(g, int(uniq[i] - uniq[0]))
    if g == 0:
        return {"gcd_half_units": 0, "period": math.inf, "half_is_not_period": False}
    # T * (g/2) = 2π  => T = 4π / g
    period = 4.0 * math.pi / float(g)
    phase = np.exp(-1j * period * diag)
    dev = float(np.max(np.abs(phase - phase[0])))
    half = np.exp(-1j * (0.5 * period) * diag)
    half_dev = float(np.max(np.abs(half / half[0] - 1.0))) if abs(half[0]) > 0 else 1.0
    # half_dev ~ 0 means half period is also a period
    return {
        "gcd_half_units": int(g),
        "period": period,
        "full_period_phase_spread": dev,
        "half_period_relative_spread": half_dev,
        "half_is_not_period": half_dev > 1e-6,
    }


def write_term_list(path: str, n: int, kappa: float) -> None:
    clauses = clause_list(n, kappa)
    lines = [
        f"# circulant two-window 3-body ring N={n} kappa={kappa}",
        "# H = -sum_i Z_i Z_{i+1} Z_{i+3} + kappa * sum_i Z_i Z_{i+2} Z_{i+3} - Gamma * sum_i X_i",
        "# periodic indices. Z|0>=+1, Z|1>=-1.",
        "# Identical supports from the two windows are merged; a zero coefficient is omitted.",
        "body,coeff_exact,coeff_float,qubits",
    ]
    for supp, coeff in clauses:
        lines.append(f"ZZZ,{coeff},{float(coeff):.16e},{' '.join(str(s) for s in supp)}")
    for i in range(n):
        lines.append(f"X,-Gamma,-Gamma,{i}")
    lines.append(f"# n_zzz={len(clauses)} n_x={n}")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
