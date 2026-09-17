"""8-bit noiseless encoding: |d⟩⊗|e⟩⊗|A⟩⊗|B⟩ with Fock cutoff 8.

Bit map (MSB-first): (q_d, q_e | n_A[2:0] | n_B[2:0]).
Physical middle bus qubit is omitted; the bus is an ideal unitary on A⊗B.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import qutip as qt

N_QUBITS = 8
NFOCK = 8
DIMS = (2, 2, NFOCK, NFOCK)  # d, e, A, B
HILBERT_DIM = 2 * 2 * NFOCK * NFOCK  # 256
N_A_BITS = 3
N_B_BITS = 3


def vacuum() -> qt.Qobj:
    """|0⟩⊗|0⟩⊗|0⟩⊗|0⟩ computational vacuum."""
    return qt.tensor(
        qt.basis(2, 0),
        qt.basis(2, 0),
        qt.basis(NFOCK, 0),
        qt.basis(NFOCK, 0),
    )


def identity() -> qt.Qobj:
    return qt.tensor(qt.qeye(2), qt.qeye(2), qt.qeye(NFOCK), qt.qeye(NFOCK))


def bits_from_denm(d: int, e: int, n_a: int, n_b: int) -> np.ndarray:
    """MSB-first 8 bits from (d, e, n_A, n_B)."""
    bits = np.zeros(N_QUBITS, dtype=int)
    bits[0] = int(d) & 1
    bits[1] = int(e) & 1
    for k in range(N_A_BITS):
        bits[2 + k] = (int(n_a) >> (N_A_BITS - 1 - k)) & 1
    for k in range(N_B_BITS):
        bits[2 + N_A_BITS + k] = (int(n_b) >> (N_B_BITS - 1 - k)) & 1
    return bits


def denm_from_bits(bits: Sequence[int]) -> tuple[int, int, int, int]:
    x = np.asarray(bits, dtype=int).reshape(-1)
    if x.size != N_QUBITS:
        raise ValueError(f"Expected {N_QUBITS} bits, got {x.size}")
    d, e = int(x[0]), int(x[1])
    n_a = 0
    for b in x[2 : 2 + N_A_BITS]:
        n_a = (n_a << 1) | int(b)
    n_b = 0
    for b in x[2 + N_A_BITS :]:
        n_b = (n_b << 1) | int(b)
    return d, e, n_a, n_b


def bitstring_from_bits(bits: Sequence[int]) -> str:
    return "".join(str(int(b)) for b in np.asarray(bits).reshape(-1))


def bits_from_bitstring(s: str) -> np.ndarray:
    if len(s) != N_QUBITS:
        raise ValueError(f"Expected {N_QUBITS}-char bitstring, got {len(s)}")
    return np.array([int(c) for c in s], dtype=int)


def flat_index(d: int, e: int, n_a: int, n_b: int) -> int:
    return ((int(d) * 2 + int(e)) * NFOCK + int(n_a)) * NFOCK + int(n_b)


def denm_from_flat(index: int) -> tuple[int, int, int, int]:
    n_b = index % NFOCK
    rem = index // NFOCK
    n_a = rem % NFOCK
    rem = rem // NFOCK
    e = rem % 2
    d = rem // 2
    return int(d), int(e), int(n_a), int(n_b)


def _z_eigenvalue(bits: np.ndarray, sites: Sequence[int]) -> float:
    """⟨x|∏_{i∈S} Z_i|x⟩ with Z|0⟩=+1, Z|1⟩=−1."""
    val = 1.0
    for i in sites:
        val *= 1.0 - 2.0 * float(bits[int(i)])
    return val


def energy_from_z_terms(
    bits: Sequence[int],
    terms: Sequence[tuple[tuple[int, ...], float]],
    identity: float = 0.0,
) -> float:
    x = np.asarray(bits, dtype=int).reshape(N_QUBITS)
    e = float(identity)
    for sites, coeff in terms:
        e += float(coeff) * _z_eigenvalue(x, sites)
    return e


def z_terms_from_npz(path: Path | str) -> tuple[list[tuple[tuple[int, ...], float]], dict]:
    path = Path(path)
    data = np.load(path)
    required = ("sites", "orders", "coefficients", "num_spins")
    missing = [k for k in required if k not in data.files]
    if missing:
        raise ValueError(f"{path.name} missing {missing}")
    num_spins = int(np.asarray(data["num_spins"]).reshape(-1)[0])
    if num_spins != N_QUBITS:
        raise ValueError(f"{path.name} has num_spins={num_spins}, need {N_QUBITS}")
    sites = np.asarray(data["sites"])
    orders = np.asarray(data["orders"], dtype=np.int64).reshape(-1)
    coeffs = np.asarray(data["coefficients"], dtype=float).reshape(-1)
    terms: list[tuple[tuple[int, ...], float]] = []
    for row, coeff in enumerate(coeffs):
        order = int(orders[row])
        row_sites = tuple(int(v) for v in sites[row, :order])
        terms.append((row_sites, float(coeff)))
    identity = float(np.asarray(data["identity"]).reshape(-1)[0]) if "identity" in data.files else 0.0
    meta = {
        "file": path.name,
        "path": str(path),
        "num_spins": num_spins,
        "identity": identity,
        "n_terms": len(terms),
        "num_clauses": int(np.asarray(data["num_clauses"]).reshape(-1)[0])
        if "num_clauses" in data.files
        else None,
    }
    return terms, meta


def energy_tensor_from_terms(
    terms: Sequence[tuple[tuple[int, ...], float]],
    identity: float = 0.0,
    tilt: float = 0.0,
) -> np.ndarray:
    """Diagonal energies shaped (2, 2, 8, 8) for |d,e,n_A,n_B⟩."""
    out = np.empty(DIMS, dtype=float)
    for d in range(2):
        for e in range(2):
            for n_a in range(NFOCK):
                for n_b in range(NFOCK):
                    bits = bits_from_denm(d, e, n_a, n_b)
                    out[d, e, n_a, n_b] = energy_from_z_terms(bits, terms, identity)
    if tilt:
        out = out + float(tilt) * np.arange(out.size, dtype=float).reshape(out.shape)
    return out


def load_four_sat_npz(path: Path | str, tilt: float = 0.0) -> dict:
    """Load one 8-qubit four_sat NPZ into energy tensor + ground-state info."""
    terms, meta = z_terms_from_npz(path)
    tensor = energy_tensor_from_terms(terms, meta["identity"], tilt=tilt)
    flat = tensor.reshape(-1)
    gmin = float(np.min(flat))
    ground_idx = int(np.argmin(flat))
    # unique check on untilted energies
    untilted = energy_tensor_from_terms(terms, meta["identity"], tilt=0.0).reshape(-1)
    n_ground = int(np.count_nonzero(np.isclose(untilted, untilted.min(), atol=1e-12)))
    d, e, n_a, n_b = denm_from_flat(ground_idx)
    bits = bits_from_denm(d, e, n_a, n_b)
    return {
        **meta,
        "terms": terms,
        "energy_tensor": tensor,
        "energies_flat": flat,
        "ground_energy": gmin if tilt else float(untilted.min()),
        "ground_denm": (d, e, n_a, n_b),
        "ground_bitstring": bitstring_from_bits(bits),
        "n_ground": n_ground,
        "untilted_flat": untilted,
    }


def list_four_sat_npz(directory: Path | str) -> list[Path]:
    directory = Path(directory)
    return sorted(directory.glob("four_sat_[0-9][0-9][0-9].npz"))
