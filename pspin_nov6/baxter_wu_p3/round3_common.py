"""Shared constants for the round-3 matched grid.

Strip geometries only (3×Ly). Width-4 tori belong to the other worker.
Hamiltonian normalization is the same at every size: triangle coefficients
are ±1 and are not divided by N. Gamma is the coefficient of -sum X in

    H = -sum_▲ ZZZ + sum_▼ ZZZ - Gamma * sum_i X_i

Bit convention: site (x, y) has index x + Lx*y, and that index is bit q
of the computational basis (qubit 0 = LSB).
"""

from __future__ import annotations

MATCHED_GAMMAS = (0.6, 0.8, 1.0, 1.2)
STRIP_GEOMS = ((3, 3), (3, 4), (3, 5), (3, 6))
# Reported round-2 points. 3×5@0.6 and 3×6@1.0 also sit on the matched grid.
REPLAY_ONLY = ((3, 4, 0.7),)
HASH_NOTE = (
    "canonical JSON is utf-8 with sort_keys and separators (',', ':'). "
    "statevector hash is sha256 of little-endian float64 real bytes followed "
    "by float64 imag bytes, C-order, of the vector as produced here. "
    "A second hash rounds real and imag to 8 decimal places before the same packing."
)
