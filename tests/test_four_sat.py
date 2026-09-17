"""8-qubit 4-SAT NPZ: unique SAT ground state and clause consistency."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
HAM_DIR = ROOT / "Hamiltonians" / "four_sat"
N_QUBITS = 8


def _paths() -> list[Path]:
    return sorted(HAM_DIR.glob("four_sat_[0-9][0-9][0-9].npz"))


def _bits_from_index(idx: int, n: int = N_QUBITS) -> np.ndarray:
    return np.array([(idx >> (n - 1 - k)) & 1 for k in range(n)], dtype=int)


def _clause_energy(bits: np.ndarray, clauses: np.ndarray, polarities: np.ndarray) -> int:
    energy = 0
    for variables, poles in zip(clauses, polarities, strict=True):
        pattern = np.where(np.asarray(poles, dtype=int) > 0, 0, 1)
        energy += int(np.all(bits[np.asarray(variables, dtype=int)] == pattern))
    return energy


def test_all_instances_unique_eight_qubit():
    paths = _paths()
    assert len(paths) == 20
    for path in paths:
        data = np.load(path)
        assert int(data["num_spins"]) == 8
        assert 14 <= int(data["num_clauses"]) <= 23
        clauses = np.asarray(data["clauses"])
        polarities = np.asarray(data["polarities"])
        energies = np.array(
            [_clause_energy(_bits_from_index(i), clauses, polarities) for i in range(1 << N_QUBITS)]
        )
        assert int(np.sum(energies == 0)) == 1
        gs = _bits_from_index(int(np.argmin(energies)))
        assert gs.sum() not in (0, N_QUBITS)
        bitstring = "".join(str(int(b)) for b in gs)
        assert bitstring not in {"00000000", "11111111", "01010101", "10101010"}


def test_noiseless_loader_ground_state():
    from noiseless.encoding import list_four_sat_npz, load_four_sat_npz

    paths = list_four_sat_npz(HAM_DIR)
    assert len(paths) >= 1
    inst = load_four_sat_npz(paths[0])
    assert inst["num_spins"] == 8
    assert inst["n_ground"] == 1
    assert inst["ground_bitstring"] not in {"00000000", "11111111"}
    assert abs(float(inst["untilted_flat"].min())) < 1e-12
