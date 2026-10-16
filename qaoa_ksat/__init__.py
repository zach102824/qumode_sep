"""Independent QAOA simulator for random k-SAT (Boulebnane & Montanaro, PRX Quantum 5, 030348).

This package is separate from ``noiseless.qaoa``, which is the 8-qubit four_sat
Gibbs/SPSA bake-off (no 1/2-angle convention, different Hamiltonian).
"""

from .circuit import apply_mixer, plus_state, qaoa_state, success_probability
from .ksat import (
    THRESHOLD_R,
    Clause,
    KSATInstance,
    energy_table,
    random_ksat,
)

__all__ = [
    "THRESHOLD_R",
    "Clause",
    "KSATInstance",
    "apply_mixer",
    "energy_table",
    "plus_state",
    "qaoa_state",
    "random_ksat",
    "success_probability",
]
