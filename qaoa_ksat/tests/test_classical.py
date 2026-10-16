"""Classical solvers find a tiny planted SAT formula."""

from __future__ import annotations

import numpy as np

from qaoa_ksat.classical import run_solver
from qaoa_ksat.ksat import Clause, KSATInstance, clauses_as_arrays, energy_table


def _unit_clauses_all_true(n: int = 6) -> KSATInstance:
    # (x0) ∧ (x1) ∧ ... forces the all-1 assignment. Extra tautology padding to k=2.
    clauses = []
    for v in range(n):
        clauses.append(
            Clause(
                variables=np.array([v, v], dtype=np.int32),
                signs=np.array([1, 1], dtype=np.int8),
            )
        )
    return KSATInstance(n=n, k=2, r=1.0, clauses=tuple(clauses))


def test_energy_all_ones_is_zero():
    inst = _unit_clauses_all_true()
    energy = energy_table(inst)
    assert int(energy[(1 << inst.n) - 1]) == 0
    assert int(energy[0]) == inst.n


def test_all_solvers_solve_unit_formula():
    inst = _unit_clauses_all_true()
    cv, cs = clauses_as_arrays(inst)
    for name in ("walksat_simple", "walksatlm", "satlike"):
        flips, lookups = run_solver(name, cv, cs, inst.n, seed=0, cap=10_000)
        assert flips >= 0, name
        assert lookups >= 1
