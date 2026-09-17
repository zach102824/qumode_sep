"""Noiseless local-ECD + fixed-U SPSA campaign for 8-qubit 4-SAT."""

from .encoding import DIMS, NFOCK, N_QUBITS, load_four_sat_npz, vacuum
from .unitaries import U_NAMES, build_fixed_u
from .circuit_local_ecd import n_parameters, random_parameters
from .spsa_gibbs import NoiselessSimulator, optimize_trial, scale_spsa_a

__all__ = [
    "DIMS",
    "NFOCK",
    "N_QUBITS",
    "U_NAMES",
    "build_fixed_u",
    "load_four_sat_npz",
    "vacuum",
    "n_parameters",
    "random_parameters",
    "NoiselessSimulator",
    "optimize_trial",
    "scale_spsa_a",
]
