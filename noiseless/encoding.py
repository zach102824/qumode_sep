"""8-bit noiseless encoding: |d⟩⊗|e⟩⊗|A⟩⊗|B⟩ with Fock cutoff 8.

Bit map (MSB-first): (q_d, q_e | n_A[2:0] | n_B[2:0]).
Physical middle bus qubit is omitted; the bus is an ideal unitary on A⊗B.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, permutations
from pathlib import Path
from typing import Sequence, Union

import numpy as np
import qutip as qt

N_QUBITS = 8
NFOCK = 8
DIMS = (2, 2, NFOCK, NFOCK)  # d, e, A, B
HILBERT_DIM = 2 * 2 * NFOCK * NFOCK  # 256
N_A_BITS = 3
N_B_BITS = 3
ENCODINGS = ("binary", "gray")


# ---------------------------------------------------------------------------
# General encoding spec: logical variable → physical bit slot (+ per-cavity codeword → Fock)
# ---------------------------------------------------------------------------

# Physical bit slots, in the MSB-first order of the legacy 8-bit vector (q_d, q_e | n_A | n_B).
# Cavity bit k = bit k of the cavity's 3-bit codeword, bit 0 = LSB.
SLOT_NAMES = ("d", "e", "A2", "A1", "A0", "B2", "B1", "B0")
# The ansatz symmetry: swap (transmon d, cavity A) ↔ (transmon e, cavity B), slot j → SWAP[j].
SWAP_DE_AB = (1, 0, 5, 6, 7, 2, 3, 4)
IDENTITY_PERM = tuple(range(N_QUBITS))
BINARY_CAVITY_MAP = tuple(range(NFOCK))


@dataclass(frozen=True)
class EncodingSpec:
    """How the 8 logical Hamiltonian variables Z1..Z8 sit on the chip.

    ``perm[i]`` = physical slot (index into :data:`SLOT_NAMES`) carrying logical variable i
    (Z_{i+1}; logical bit i is position i of the MSB-first logical bitstring). The identity
    perm is the legacy layout (Z1→d, Z2→e, Z3→A2, Z4→A1, Z5→A0, Z6→B2, Z7→B1, Z8→B0).

    ``cavity_a`` / ``cavity_b``: codeword → Fock level map per cavity; entry c is the Fock
    level holding the 3-bit codeword c (c read MSB-first from slots A2 A1 A0). Identity =
    binary; ``tuple(gray_decode(c) for c in range(8))`` = Gray (legacy ``encoding="gray"``).
    Designed so codewords can later live on Fock levels 0..15 with a larger truncation
    (entries must then be distinct levels < nfock, with unused levels = leakage); only the
    nfock=8 bijection is implemented today.
    """

    perm: tuple[int, ...] = IDENTITY_PERM
    cavity_a: tuple[int, ...] = BINARY_CAVITY_MAP
    cavity_b: tuple[int, ...] = BINARY_CAVITY_MAP

    def __post_init__(self) -> None:
        object.__setattr__(self, "perm", tuple(int(v) for v in self.perm))
        object.__setattr__(self, "cavity_a", tuple(int(v) for v in self.cavity_a))
        object.__setattr__(self, "cavity_b", tuple(int(v) for v in self.cavity_b))
        if sorted(self.perm) != list(range(N_QUBITS)):
            raise ValueError(f"perm must be a permutation of 0..7, got {self.perm}")
        for name, cm in (("cavity_a", self.cavity_a), ("cavity_b", self.cavity_b)):
            if len(cm) != 2**N_A_BITS or len(set(cm)) != len(cm):
                raise ValueError(f"{name} must map the 8 codewords to distinct Fock levels")
            if sorted(cm) != list(range(NFOCK)):
                raise NotImplementedError(
                    f"{name}={cm}: codewords outside Fock 0..{NFOCK - 1} need a larger truncation"
                )

    @classmethod
    def from_name(cls, name: str) -> "EncodingSpec":
        enc = str(name).lower().strip()
        if enc == "binary":
            return cls()
        if enc == "gray":
            g = tuple(gray_decode(c) for c in range(NFOCK))
            return cls(cavity_a=g, cavity_b=g)
        raise ValueError(f"unknown encoding {name!r}")

    def swapped(self) -> "EncodingSpec":
        """Image under the (d,A)↔(e,B) ansatz symmetry."""
        return EncodingSpec(tuple(SWAP_DE_AB[j] for j in self.perm), self.cavity_b, self.cavity_a)

    def label(self) -> str:
        out = "perm=" + "".join(str(v) for v in self.perm)
        if self.cavity_a != BINARY_CAVITY_MAP or self.cavity_b != BINARY_CAVITY_MAP:
            out += ",A=" + "".join(map(str, self.cavity_a)) + ",B=" + "".join(map(str, self.cavity_b))
        return out

    def __str__(self) -> str:  # records store str(encoding)
        return self.label()

    def slot_of_variable(self) -> dict[str, str]:
        return {f"Z{i + 1}": SLOT_NAMES[j] for i, j in enumerate(self.perm)}


Encoding = Union[str, EncodingSpec]


def _spec_phys_bits(d: int, e: int, n_a: int, n_b: int, spec: EncodingSpec) -> np.ndarray:
    p = np.zeros(N_QUBITS, dtype=int)
    p[0], p[1] = int(d) & 1, int(e) & 1
    ca = spec.cavity_a.index(int(n_a))
    cb = spec.cavity_b.index(int(n_b))
    for k in range(N_A_BITS):
        p[2 + k] = (ca >> (N_A_BITS - 1 - k)) & 1
        p[2 + N_A_BITS + k] = (cb >> (N_B_BITS - 1 - k)) & 1
    return p


def canonical_perm(perm: Sequence[int]) -> tuple[int, ...]:
    """Lexicographically smaller of perm and its (d,A)↔(e,B) image (class representative)."""
    a = tuple(int(v) for v in perm)
    b = tuple(SWAP_DE_AB[j] for j in a)
    return min(a, b)


def distinct_assignments() -> list[tuple[int, ...]]:
    """All 8!/2 = 20160 symmetry classes of variable→slot assignments, as sorted canonical
    perms. Index 0 is the identity (legacy layout)."""
    return [q for q in permutations(range(N_QUBITS)) if q <= tuple(SWAP_DE_AB[j] for j in q)]


def logical_index_of_flat(spec: EncodingSpec) -> np.ndarray:
    """int array (256,): logical bitstring (as MSB-first integer) of each physical flat index."""
    out = np.empty(HILBERT_DIM, dtype=np.int64)
    for idx in range(HILBERT_DIM):
        x = bits_from_denm(*denm_from_flat(idx), spec)
        out[idx] = int("".join(map(str, x)), 2)
    return out


def logical_energies_from_terms(terms, identity_shift: float = 0.0) -> np.ndarray:
    """Energy of every logical bitstring (index = MSB-first integer), same arithmetic as
    :func:`energy_from_z_terms`."""
    out = np.empty(2**N_QUBITS, dtype=float)
    for v in range(2**N_QUBITS):
        bits = np.array([(v >> (N_QUBITS - 1 - k)) & 1 for k in range(N_QUBITS)], dtype=int)
        out[v] = energy_from_z_terms(bits, terms, identity_shift)
    return out


def energy_tensor_for_spec(logical_energies: np.ndarray, spec: EncodingSpec) -> np.ndarray:
    """Physical (2,2,8,8) energy tensor for ``spec`` from the 256 logical energies (fast path;
    identical values to :func:`energy_tensor_from_terms` with ``encoding=spec``)."""
    return np.asarray(logical_energies, dtype=float)[logical_index_of_flat(spec)].reshape(DIMS)


def polish_bitstring(bitstring: str, logical_energies: np.ndarray, radius: int = 1) -> str:
    """Lowest-energy logical bitstring within Hamming ``radius`` of ``bitstring``.

    Free classical post-processing (table lookups into the 256 logical energies, no
    circuit runs). On the H0–H7 encoding screen the final most-likely bitstring is within
    Hamming 1 of the GS in 98.4 % of trials (results/LAYOUT_PATTERNS.md §5), so the
    default ``radius=1`` turns almost every near-miss into the exact GS. ``radius=0`` is
    a no-op. Energy ties keep the smaller bitstring integer (deterministic).
    """
    e = np.asarray(logical_energies, dtype=float).reshape(-1)
    if e.size != 2**N_QUBITS:
        raise ValueError(f"logical_energies must have {2**N_QUBITS} entries, got {e.size}")
    v0 = int(str(bitstring), 2)
    best = v0
    for r in range(1, int(radius) + 1):
        for pos in combinations(range(N_QUBITS), r):
            v = v0
            for k in pos:
                v ^= 1 << (N_QUBITS - 1 - k)
            if (e[v], v) < (e[best], best):
                best = v
    return format(best, f"0{N_QUBITS}b")


def corner_spec_for_bitstring(bitstring: str, base: EncodingSpec = EncodingSpec()) -> EncodingSpec:
    """EncodingSpec with ``base.perm`` whose cavity maps put ``bitstring`` at Fock (0, 0).

    XOR-relabels each cavity's codeword→Fock map by the bitstring's codeword on that
    cavity, so the bitstring's physical state has n_A = n_B = 0 — both cavities at a
    Fock-ladder end, the tier-0 geometry that the encoding screen found optimal
    (results/LAYOUT_PATTERNS.md: tier-0/0 mean p(GS) ≈ 0.82 vs 0.51 for a random
    layout). Its Hamming-1 cavity neighbours land on Fock 1, 2 and 4, the same geometry
    as a binary-code GS at codeword 000. Only ``base.perm`` is read (``base``'s cavity
    maps are replaced), so the map is idempotent: re-deriving from the same bitstring
    returns an equal spec.
    """
    x = bits_from_bitstring(bitstring)
    p = np.zeros(N_QUBITS, dtype=int)
    p[list(base.perm)] = x
    ca = int("".join(map(str, p[2 : 2 + N_A_BITS])), 2)
    cb = int("".join(map(str, p[2 + N_A_BITS :])), 2)
    return EncodingSpec(
        base.perm,
        tuple(c ^ ca for c in range(NFOCK)),
        tuple(c ^ cb for c in range(NFOCK)),
    )


def _check_encoding(encoding: Encoding):
    if isinstance(encoding, EncodingSpec):
        return encoding
    enc = str(encoding).lower().strip()
    if enc not in ENCODINGS:
        raise ValueError(f"unknown encoding {encoding!r}; choose from {ENCODINGS}")
    return enc


def gray_encode(n: int) -> int:
    """Binary-reflected Gray code of a Fock number: g = n ^ (n >> 1)."""
    n = int(n)
    return n ^ (n >> 1)


def gray_decode(g: int) -> int:
    """Inverse of :func:`gray_encode` (prefix XOR)."""
    g = int(g)
    n = 0
    while g:
        n ^= g
        g >>= 1
    return n


def _cavity_code(n: int, encoding: str) -> int:
    return gray_encode(n) if encoding == "gray" else int(n)


def _cavity_decode(v: int, encoding: str) -> int:
    return gray_decode(v) if encoding == "gray" else int(v)


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


def bits_from_denm(
    d: int, e: int, n_a: int, n_b: int, encoding: str = "binary"
) -> np.ndarray:
    """MSB-first 8 bits from (d, e, n_A, n_B).

    ``encoding="binary"`` (default): cavity bits are the binary digits of n.
    ``encoding="gray"``: cavity bits are the digits of n ^ (n >> 1). Transmon bits unchanged.
    """
    enc = _check_encoding(encoding)
    if isinstance(enc, EncodingSpec):
        return _spec_phys_bits(d, e, n_a, n_b, enc)[list(enc.perm)]
    n_a = _cavity_code(n_a, enc)
    n_b = _cavity_code(n_b, enc)
    bits = np.zeros(N_QUBITS, dtype=int)
    bits[0] = int(d) & 1
    bits[1] = int(e) & 1
    for k in range(N_A_BITS):
        bits[2 + k] = (int(n_a) >> (N_A_BITS - 1 - k)) & 1
    for k in range(N_B_BITS):
        bits[2 + N_A_BITS + k] = (int(n_b) >> (N_B_BITS - 1 - k)) & 1
    return bits


def denm_from_bits(
    bits: Sequence[int], encoding: str = "binary"
) -> tuple[int, int, int, int]:
    enc = _check_encoding(encoding)
    x = np.asarray(bits, dtype=int).reshape(-1)
    if x.size != N_QUBITS:
        raise ValueError(f"Expected {N_QUBITS} bits, got {x.size}")
    if isinstance(enc, EncodingSpec):
        p = np.zeros(N_QUBITS, dtype=int)
        p[list(enc.perm)] = x
        ca = int("".join(map(str, p[2 : 2 + N_A_BITS])), 2)
        cb = int("".join(map(str, p[2 + N_A_BITS :])), 2)
        return int(p[0]), int(p[1]), enc.cavity_a[ca], enc.cavity_b[cb]
    d, e = int(x[0]), int(x[1])
    n_a = 0
    for b in x[2 : 2 + N_A_BITS]:
        n_a = (n_a << 1) | int(b)
    n_b = 0
    for b in x[2 + N_A_BITS :]:
        n_b = (n_b << 1) | int(b)
    return d, e, _cavity_decode(n_a, enc), _cavity_decode(n_b, enc)


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
    identity_shift: float = 0.0,
) -> float:
    x = np.asarray(bits, dtype=int).reshape(N_QUBITS)
    e = float(identity_shift)
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
    identity_shift = (
        float(np.asarray(data["identity"]).reshape(-1)[0]) if "identity" in data.files else 0.0
    )
    meta = {
        "file": path.name,
        "path": str(path),
        "num_spins": num_spins,
        "identity": identity_shift,
        "n_terms": len(terms),
        "num_clauses": int(np.asarray(data["num_clauses"]).reshape(-1)[0])
        if "num_clauses" in data.files
        else None,
    }
    return terms, meta


def energy_tensor_from_terms(
    terms: Sequence[tuple[tuple[int, ...], float]],
    identity_shift: float = 0.0,
    tilt: float = 0.0,
    encoding: str = "binary",
) -> np.ndarray:
    """Diagonal energies shaped (2, 2, 8, 8) for |d,e,n_A,n_B⟩ (physical Fock basis).

    The Hamiltonian is defined on logical bits; ``encoding`` fixes the map
    physical (d, e, n_A, n_B) → logical bits.
    """
    enc = _check_encoding(encoding)
    out = np.empty(DIMS, dtype=float)
    for d in range(2):
        for e in range(2):
            for n_a in range(NFOCK):
                for n_b in range(NFOCK):
                    bits = bits_from_denm(d, e, n_a, n_b, enc)
                    out[d, e, n_a, n_b] = energy_from_z_terms(bits, terms, identity_shift)
    if tilt:
        out = out + float(tilt) * np.arange(out.size, dtype=float).reshape(out.shape)
    return out


def load_four_sat_npz(
    path: Path | str, tilt: float = 0.0, encoding: str = "binary"
) -> dict:
    """Load one 8-qubit four_sat NPZ into energy tensor + ground-state info.

    ``ground_bitstring`` is always the LOGICAL bitstring (encoding-independent);
    ``ground_denm`` / ``ground_flat_index`` are the physical Fock-basis location.
    """
    enc = _check_encoding(encoding)
    terms, meta = z_terms_from_npz(path)
    tensor = energy_tensor_from_terms(terms, meta["identity"], tilt=tilt, encoding=enc)
    flat = tensor.reshape(-1)
    gmin = float(np.min(flat))
    ground_idx = int(np.argmin(flat))
    untilted = energy_tensor_from_terms(
        terms, meta["identity"], tilt=0.0, encoding=enc
    ).reshape(-1)
    n_ground = int(np.count_nonzero(np.isclose(untilted, untilted.min(), atol=1e-12)))
    d, e, n_a, n_b = denm_from_flat(ground_idx)
    bits = bits_from_denm(d, e, n_a, n_b, enc)
    return {
        **meta,
        "encoding": enc,
        "ground_flat_index": ground_idx,
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
