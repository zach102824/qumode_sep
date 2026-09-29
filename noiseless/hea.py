"""8-qubit hardware-efficient ansatz (HEA) for four_sat bake-offs.

Lattice: 2×4 square lattice with snake numbering (1-indexed q1..q8)::

    q1 - q2 - q3 - q4
    |    |    |    |
    q8 - q7 - q6 - q5

Qubit mapping to the repo: 1-indexed q_k ↔ repo qubit index k-1 ↔ bitstring
character k-1 (MSB-first; basis index = int(bitstring, 2)), i.e. q1 is the
most-significant bit and the Z-term site 0 of the four_sat Hamiltonian.

Circuit (RY only, no RZ):
  start |0⟩^⊗8
  for l = 1..L:
      RY(θ_{l,q}) on all 8 qubits
      CZ sublayer A: (1,2)(3,4)(5,6)(7,8)
      CZ sublayer B: (2,3)(4,5)(6,7)
      CZ sublayer C (rungs): (1,8)(2,7)(3,6)
  final RY layer on all 8 qubits
L = 0 ("hea_ry0" arm) is the product-state ansatz: a single RY layer on
|0⟩^⊗8, no CZ, 8 params.
n_params = 8 (L + 1).  Parameter layout: x.reshape(L+1, 8), row l = RY layer l,
column k = qubit q_{k+1}.

Note: the 4-5 rung is realized by sublayer B's (4,5); with snake numbering the
10 CZs are exactly the 10 edges of the 2×4 lattice.
RY(θ) = exp(-i θ Y / 2) = [[cos θ/2, -sin θ/2], [sin θ/2, cos θ/2]].
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .encoding import N_QUBITS
from .qaoa import bitstring_from_index, born_probs, index_from_bitstring
from .spsa_gibbs import SampledTailEta, gibbs_objective, run_spsa, scale_spsa_a

N = N_QUBITS
DIM = 1 << N

# 1-indexed CZ pairs per sublayer
CZ_SUBLAYER_A = ((1, 2), (3, 4), (5, 6), (7, 8))
CZ_SUBLAYER_B = ((2, 3), (4, 5), (6, 7))
CZ_SUBLAYER_C = ((1, 8), (2, 7), (3, 6))
CZ_SUBLAYERS = (CZ_SUBLAYER_A, CZ_SUBLAYER_B, CZ_SUBLAYER_C)
CZ_PAIRS_1IDX = CZ_SUBLAYER_A + CZ_SUBLAYER_B + CZ_SUBLAYER_C


def lattice_edges_1idx() -> set[tuple[int, int]]:
    """The 10 edges of the 2×4 square lattice, snake numbering (1-indexed)."""
    top = [1, 2, 3, 4]
    bot = [8, 7, 6, 5]
    edges = set()
    for row in (top, bot):
        for a, b in zip(row[:-1], row[1:]):
            edges.add((min(a, b), max(a, b)))
    for a, b in zip(top, bot):
        edges.add((min(a, b), max(a, b)))
    return edges


def n_hea_parameters(n_layers: int) -> int:
    return N * (int(n_layers) + 1)


def hea_layers_for_params(n_params: int) -> int:
    if int(n_params) % N:
        raise ValueError(f"HEA n_params must be a multiple of {N}, got {n_params}")
    return int(n_params) // N - 1


def _cz_layer_signs(n: int = N) -> np.ndarray:
    """Diagonal ±1 of the product of all 10 CZs (they commute)."""
    idx = np.arange(1 << n)
    signs = np.ones(1 << n, dtype=float)
    for a, b in CZ_PAIRS_1IDX:
        qa, qb = a - 1, b - 1
        ba = (idx >> (n - 1 - qa)) & 1
        bb = (idx >> (n - 1 - qb)) & 1
        signs = np.where((ba & bb) == 1, -signs, signs)
    return signs


CZ_SIGNS = _cz_layer_signs()


def apply_ry_layer(psi: np.ndarray, thetas: np.ndarray, n: int = N) -> np.ndarray:
    """Apply RY(θ_q) on each qubit q (0-indexed repo order)."""
    t = np.asarray(psi).reshape((2,) * int(n))
    th = np.asarray(thetas, dtype=float).reshape(-1)
    for q in range(int(n)):
        c = np.cos(0.5 * th[q])
        s = np.sin(0.5 * th[q])
        t0 = np.take(t, 0, axis=q)
        t1 = np.take(t, 1, axis=q)
        t = np.stack((c * t0 - s * t1, s * t0 + c * t1), axis=q)
    return t.reshape(-1)


def zero_state(n: int = N) -> np.ndarray:
    psi = np.zeros(1 << int(n), dtype=float)
    psi[0] = 1.0
    return psi


def hea_state(params: np.ndarray, n: int = N) -> np.ndarray:
    x = np.asarray(params, dtype=float).reshape(-1)
    if x.size % N or x.size < N:
        raise ValueError(f"HEA parameter length must be 8(L+1) with L>=0, got {x.size}")
    rows = x.reshape(-1, N)
    n_layers = rows.shape[0] - 1
    psi = zero_state(n)
    for l in range(n_layers):
        psi = apply_ry_layer(psi, rows[l], n)
        psi = psi * CZ_SIGNS
    psi = apply_ry_layer(psi, rows[n_layers], n)
    return psi


def random_hea_params(n_layers: int, rng: np.random.Generator) -> np.ndarray:
    """Uniform[0, π) for every RY angle (same draw rule as random_qaoa_params)."""
    return rng.random(n_hea_parameters(n_layers)) * np.pi


@dataclass
class HEATrialResult:
    success: bool
    p_gs: float
    most_likely_bitstring: str
    ground_bitstring: str
    fun: float
    eta: float
    nfev: int
    nit: int
    x: np.ndarray
    energy_mean: float


@dataclass
class HEASimulator:
    """HEA SPSA Gibbs on the full diagonal four_sat spectrum; score vs true GS."""

    energies: np.ndarray
    ground_bitstring: str
    n: int = N
    eta_ctrl: SampledTailEta = field(default_factory=SampledTailEta)
    _current_eta: float = 1.0

    def __post_init__(self) -> None:
        self.energies = np.asarray(self.energies, dtype=float).reshape(-1)
        if self.energies.size != (1 << int(self.n)):
            raise ValueError(f"energies must have length {1 << int(self.n)}")
        self.ground_index = index_from_bitstring(self.ground_bitstring)

    def probs_from_x(self, x: np.ndarray) -> np.ndarray:
        return born_probs(hea_state(x, self.n))

    def cost(self, x: np.ndarray, eta: float | None = None) -> float:
        probs = self.probs_from_x(x)
        use_eta = self._current_eta if eta is None else float(eta)
        return gibbs_objective(probs, self.energies, use_eta)

    def refresh_eta(self, x: np.ndarray, step: int) -> float:
        probs = self.probs_from_x(x)
        self._current_eta = self.eta_ctrl.update(self.energies, probs, step)
        return self._current_eta

    def evaluate(self, x: np.ndarray) -> dict:
        probs = self.probs_from_x(x)
        idx = int(np.argmax(probs))
        ml = bitstring_from_index(idx, self.n)
        return {
            "most_likely_bitstring": ml,
            "p_gs": float(probs[self.ground_index]),
            "success": ml == self.ground_bitstring,
            "energy_mean": float(np.dot(probs, self.energies)),
            "probs": probs,
        }


def optimize_hea_trial(
    sim: HEASimulator,
    n_layers: int,
    *,
    maxiter: int = 200,
    rng: np.random.Generator | None = None,
    x0: np.ndarray | None = None,
    a: float | None = None,
    c: float = 0.15,
    A: float = 10.0,
) -> HEATrialResult:
    """Same SPSA / η-refresh cadence as optimize_qaoa_trial."""
    rng = rng or np.random.default_rng()
    n_params = n_hea_parameters(n_layers)
    x0 = random_hea_params(n_layers, rng) if x0 is None else np.asarray(x0, dtype=float)
    if a is None:
        a = scale_spsa_a(n_params)
    sim.eta_ctrl = SampledTailEta()
    sim._current_eta = 1.0

    def on_before(step: int, x: np.ndarray) -> None:
        if (step - 1) % sim.eta_ctrl.refresh_every == 0:
            sim.refresh_eta(x, step)

    x_final, fun_final, nfev = run_spsa(
        sim.cost,
        x0,
        maxiter=maxiter,
        rng=rng,
        a=float(a),
        c=c,
        A=A,
        on_before_step=on_before,
    )
    ev = sim.evaluate(x_final)
    return HEATrialResult(
        success=bool(ev["success"]),
        p_gs=float(ev["p_gs"]),
        most_likely_bitstring=str(ev["most_likely_bitstring"]),
        ground_bitstring=sim.ground_bitstring,
        fun=float(fun_final),
        eta=float(sim._current_eta),
        nfev=int(nfev),
        nit=int(maxiter),
        x=x_final,
        energy_mean=float(ev["energy_mean"]),
    )
