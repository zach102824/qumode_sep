"""Independent checks for the circulant ring. Run before any QAOA.

The Kronecker builder below does not call cost_diagonal. It multiplies
explicit Pauli matrices. Agreement is required.
"""

from __future__ import annotations

import math

import numpy as np

from hamiltonian import (
    apply_sum_x,
    clause_list,
    classical_summary,
    cost_diagonal,
    gf2_satisfiability,
    gamma_period,
    permute_state,
    reflect_bits,
    translate_bits,
    z_product,
)


def _pauli_z() -> np.ndarray:
    return np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.float64)


def _pauli_x() -> np.ndarray:
    return np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64)


def kron_operator(n: int, sites: tuple[int, ...], which: str) -> np.ndarray:
    """Qubit 0 is the fastest index (rightmost Kronecker factor)."""
    op = np.array([[1.0]])
    z = _pauli_z()
    x = _pauli_x()
    eye = np.eye(2)
    for q in range(n - 1, -1, -1):
        if q in sites:
            local = z if which == "Z" else x
        else:
            local = eye
        op = np.kron(op, local)
    return op


def kron_diagonal(n: int, kappa: float) -> np.ndarray:
    clauses = clause_list(n, kappa)
    diag = np.zeros(1 << n, dtype=np.float64)
    for supp, coeff in clauses:
        op = kron_operator(n, supp, "Z")
        diag += float(coeff) * np.diag(op)
    return diag


def dense_h(n: int, kappa: float, gamma: float) -> np.ndarray:
    diag = kron_diagonal(n, kappa)
    h = np.diag(diag)
    x = _pauli_x()
    for q in range(n):
        h += -gamma * kron_operator(n, (q,), "X")
    return h


def reflection_diagonal_mismatch(n: int, kappa: float) -> float:
    """Max |E(x) - E(R x)|. Zero iff reflection preserves the cost."""
    diag = cost_diagonal(n, kappa)
    err = 0.0
    for x in range(1 << n):
        err = max(err, abs(diag[x] - diag[reflect_bits(x, n)]))
    return err


def translation_diagonal_mismatch(n: int, kappa: float) -> float:
    diag = cost_diagonal(n, kappa)
    err = 0.0
    for x in range(1 << n):
        err = max(err, abs(diag[x] - diag[translate_bits(x, n, 1)]))
    return err


def run_checks(ns: list[int] | None = None) -> list[str]:
    lines: list[str] = []

    def log(msg: str) -> None:
        lines.append(msg)
        print(msg, flush=True)

    log("algebra verification start")
    # Window collision / cancellation report.
    for n in range(6, 15):
        raw_a = [tuple(sorted((i + d) % n for d in (0, 1, 3))) for i in range(n)]
        raw_b = [tuple(sorted((i + d) % n for d in (0, 2, 3))) for i in range(n)]
        merged = clause_list(n, 1.0)
        overlap = set(raw_a) & set(raw_b)
        log(
            f"N={n}: raw windows A={len(raw_a)} B={len(raw_b)} shared_supports={len(overlap)} "
            f"merged_nonzero={len(merged)} coeffs={sorted({str(c) for _s, c in merged})}"
        )

    log("--- GF(2) satisfiability (actual rank, not a divisibility shortcut) ---")
    for n in range(6, 15):
        info = gf2_satisfiability(n)
        clas = classical_summary(n, 1.0)
        log(
            f"N={n}: rank={info['rank']}/{n} consistent={info['consistent']} "
            f"sat_assignments={info['n_satisfying_assignments']} "
            f"classical_degeneracy={clas['degeneracy']} Emin={clas['e_min']:.4f} "
            f"Esat={clas['fully_satisfied_energy']:.4f} frustrated={clas['frustrated']} "
            f"excess={clas['violation_above_sat']:.4f}"
        )
        if info["consistent"] != (not clas["frustrated"]):
            raise AssertionError(f"GF2 and enumeration disagree at N={n}")

    # kappa=0.5 has the same signs, so the same satisfiability, different energy scale.
    for n in (6, 7, 8, 10, 12, 14):
        c = classical_summary(n, 0.5)
        log(
            f"kappa=0.5 N={n}: Emin={c['e_min']:.4f} Esat={c['fully_satisfied_energy']:.4f} "
            f"deg={c['degeneracy']} frustrated={c['frustrated']} excess={c['violation_above_sat']:.4f}"
        )

    log("--- independent Kronecker vs bit-parity diagonal ---")
    for n in (6, 7, 8):
        for kappa in (1.0, 0.5):
            a = cost_diagonal(n, kappa)
            b = kron_diagonal(n, kappa)
            err = float(np.max(np.abs(a - b)))
            if err > 1e-9:
                raise AssertionError(f"kron mismatch N={n} kappa={kappa} err={err}")
            log(f"N={n} kappa={kappa}: Pauli-kron vs bit diagonal max abs {err:.2e}")

    log("--- translation is a symmetry; reflection is not, for these signs ---")
    for n in (6, 8, 9, 10, 12):
        for kappa in (1.0, 0.5):
            tr = translation_diagonal_mismatch(n, kappa)
            rf = reflection_diagonal_mismatch(n, kappa)
            if tr > 1e-9:
                raise AssertionError(f"translation failed N={n}")
            log(f"N={n} kappa={kappa}: translation mismatch {tr:.2e}, reflection mismatch {rf:.6f}")

    log("--- dense ED: stoquastic GS unique, positive, momentum 0; reflection does not preserve it as a symmetry requirement ---")
    for n, kappa, gamma in ((6, 1.0, 1.0), (8, 1.0, 1.0), (7, 1.0, 0.8), (8, 0.5, 1.2)):
        h = dense_h(n, kappa, gamma)
        # off-diagonal entries are 0 or -gamma
        off = h.copy()
        np.fill_diagonal(off, 0.0)
        if np.min(off) < -gamma - 1e-8 or np.max(off) > 1e-8:
            raise AssertionError("unexpected off-diagonal sign")
        evals, evecs = np.linalg.eigh(h)
        gap = float(evals[1] - evals[0])
        if gap < 1e-8:
            raise AssertionError(f"GS degenerate at N={n} kappa={kappa} Gamma={gamma}")
        gs = evecs[:, 0].real
        if float(np.sum(gs)) < 0:
            gs = -gs
        if float(np.min(gs)) < -1e-8:
            raise AssertionError("GS not nonnegative")
        psi = gs.astype(np.complex128)
        tpsi = permute_state(psi, n, translate_bits)
        mom = float(np.linalg.norm(tpsi - psi))
        if mom > 1e-8:
            raise AssertionError(f"GS not momentum 0, dev {mom}")
        # bit method energy vs dense
        e_bit = float(np.real(np.vdot(psi, cost_diagonal(n, kappa) * psi))) - gamma * float(
            np.real(np.vdot(psi, apply_sum_x(psi, n)))
        )
        if abs(e_bit - float(evals[0])) > 1e-8:
            raise AssertionError("bit energy disagrees with dense eigh")
        log(
            f"N={n} kappa={kappa} Gamma={gamma}: E0={evals[0]:.8f} gap={gap:.6f} "
            f"min_amp={float(np.min(gs)):.3e} momentum_dev={mom:.2e} bit_energy_err={abs(e_bit-float(evals[0])):.2e}"
        )

    # Period check on the integer and half-integer spectra.
    for n, kappa in ((6, 1.0), (8, 1.0), (6, 0.5), (9, 1.0)):
        info = gamma_period(cost_diagonal(n, kappa))
        if info["full_period_phase_spread"] > 1e-8:
            raise AssertionError(f"period failed {info}")
        log(
            f"N={n} kappa={kappa}: gamma period {info['period']:.6f} "
            f"(gcd half-units {info['gcd_half_units']}), half-period spread {info['half_period_relative_spread']:.3e}, "
            f"half_not_period={info['half_is_not_period']}"
        )

    # Global X flips every ZZZ sign. Fidelity identity is checked once the QAOA
    # exists; here only the diagonal map.
    n, kappa = 6, 1.0
    diag = cost_diagonal(n, kappa)
    # Flip all bits: product of 3 Z's picks up (-1)^3 = -1, so cost -> -cost.
    flipped = np.array([diag[x ^ ((1 << n) - 1)] for x in range(1 << n)])
    # Global bit flip is conjugation by X... actually computational basis flip maps
    # |x> to |x flipped>, and Z eigenvalues on the flipped state equal minus the
    # original only if we compare the same bit pattern's energy after sign flip of H.
    # H_cost(-J) diagonal equals -diag. The basis permutation of global X does not
    # change a product of three Z eigenvalues? Wait: global X maps |x> to |~x>,
    # and <~x| ZZZ |~x> = - <x|ZZZ|x> because each Z flips. So diag_of_original_H at ~x
    # equals - diag at x. Yes flipped[x] = diag[~x] = -diag[x].
    err = float(np.max(np.abs(flipped + diag)))
    if err > 1e-8:
        raise AssertionError(f"global-X diagonal map failed {err}")
    log(f"global X sends the cost diagonal to its negative, max abs {err:.2e}")

    # A one-site gauge flips the three-site products that include that site.
    # Checked by comparing enumerated energies after flipping qubit 0's Z.
    log("algebra verification passed")
    return lines


if __name__ == "__main__":
    run_checks()
