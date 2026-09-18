"""8-qubit QAOA for four_sat bake-offs (full spectrum and NN-ring truncation).

Circuit: start |+⟩^⊗8; each layer applies exp(-i γ H_P) then exp(-i β H_M)
with H_M = Σ_j X_j. H_P is diagonal in the computational basis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from .encoding import (
    N_QUBITS,
    bitstring_from_bits,
    bits_from_bitstring,
    energy_from_z_terms,
)
from .spsa_gibbs import (
    SampledTailEta,
    gibbs_objective,
    run_spsa,
    scale_spsa_a,
)

N = N_QUBITS
DIM = 1 << N


def bitstring_from_index(index: int, n: int = N) -> str:
    return format(int(index), f"0{int(n)}b")


def index_from_bitstring(bits: str) -> int:
    return int(bits, 2)


def bits_from_index(index: int, n: int = N) -> np.ndarray:
    return np.array([(int(index) >> (int(n) - 1 - k)) & 1 for k in range(int(n))], dtype=int)


def ring_nn_pairs(n: int = N) -> set[tuple[int, int]]:
    """Unordered nearest-neighbor edges on an n-qubit ring."""
    edges: set[tuple[int, int]] = set()
    for i in range(int(n)):
        j = (i + 1) % int(n)
        edges.add((min(i, j), max(i, j)))
    return edges


def truncate_terms_nn_ring(
    terms: Sequence[tuple[tuple[int, ...], float]],
    n: int = N,
) -> tuple[list[tuple[tuple[int, ...], float]], dict]:
    """Keep 1-local Z and ring-NN ZZ; drop weight≥3 and non-NN ZZ."""
    nn = ring_nn_pairs(n)
    kept: list[tuple[tuple[int, ...], float]] = []
    n_w1 = n_w2_nn = n_w2_other = n_w3plus = 0
    dropped_coeff_l1 = 0.0
    kept_coeff_l1 = 0.0
    for sites, coeff in terms:
        order = len(sites)
        c = float(coeff)
        if order == 1:
            kept.append((tuple(int(s) for s in sites), c))
            n_w1 += 1
            kept_coeff_l1 += abs(c)
        elif order == 2:
            a, b = int(sites[0]), int(sites[1])
            key = (min(a, b), max(a, b))
            if key in nn:
                kept.append((tuple(sorted((a, b))), c))
                n_w2_nn += 1
                kept_coeff_l1 += abs(c)
            else:
                n_w2_other += 1
                dropped_coeff_l1 += abs(c)
        else:
            n_w3plus += 1
            dropped_coeff_l1 += abs(c)
    n_terms = len(terms)
    n_kept = len(kept)
    n_dropped = n_terms - n_kept
    stats = {
        "n_terms_full": int(n_terms),
        "n_terms_kept": int(n_kept),
        "n_terms_dropped": int(n_dropped),
        "n_weight1": int(n_w1),
        "n_weight2_nn": int(n_w2_nn),
        "n_weight2_non_nn": int(n_w2_other),
        "n_weight_ge3": int(n_w3plus),
        "frac_terms_dropped": float(n_dropped / n_terms) if n_terms else 0.0,
        "kept_coeff_l1": float(kept_coeff_l1),
        "dropped_coeff_l1": float(dropped_coeff_l1),
        "frac_coeff_l1_dropped": float(
            dropped_coeff_l1 / (kept_coeff_l1 + dropped_coeff_l1)
            if (kept_coeff_l1 + dropped_coeff_l1) > 0
            else 0.0
        ),
    }
    return kept, stats


def diagonal_spectrum_from_terms(
    terms: Sequence[tuple[tuple[int, ...], float]],
    identity_shift: float = 0.0,
    n: int = N,
) -> np.ndarray:
    """Energies in MSB-first binary order: index = int(bitstring, 2)."""
    dim = 1 << int(n)
    out = np.empty(dim, dtype=float)
    for idx in range(dim):
        bits = bits_from_index(idx, n)
        out[idx] = energy_from_z_terms(bits, terms, identity_shift)
    return out


def plus_state(n: int = N) -> np.ndarray:
    dim = 1 << int(n)
    return np.full(dim, 1.0 / np.sqrt(dim), dtype=complex)


def apply_mixer(psi: np.ndarray, beta: float, n: int = N) -> np.ndarray:
    """Apply ⊗_j exp(-i β X_j)."""
    b = float(beta)
    c = np.cos(b)
    s = -1j * np.sin(b)
    t = np.asarray(psi, dtype=complex).reshape((2,) * int(n))
    for q in range(int(n)):
        t0 = np.take(t, 0, axis=q)
        t1 = np.take(t, 1, axis=q)
        t = np.stack((c * t0 + s * t1, s * t0 + c * t1), axis=q)
    return t.reshape(-1)


def qaoa_state(params: np.ndarray, energies: np.ndarray, n: int = N) -> np.ndarray:
    """Layer: exp(-i γ H_P) then exp(-i β H_M), H_P diagonal = energies."""
    x = np.asarray(params, dtype=float).reshape(-1)
    if x.size % 2:
        raise ValueError(f"QAOA parameter length must be even, got {x.size}")
    p = x.size // 2
    gamma = x[:p]
    beta = x[p:]
    psi = plus_state(n)
    e = np.asarray(energies, dtype=float).reshape(-1)
    for k in range(p):
        psi = psi * np.exp(-1j * float(gamma[k]) * e)
        psi = apply_mixer(psi, float(beta[k]), n)
    return psi


def born_probs(psi: np.ndarray) -> np.ndarray:
    p = np.abs(np.asarray(psi, dtype=complex).reshape(-1)) ** 2
    total = float(p.sum())
    if total <= 0.0:
        return np.full(p.shape, 1.0 / max(p.size, 1))
    return p.real / total


def random_qaoa_params(p: int, rng: np.random.Generator) -> np.ndarray:
    """Uniform[0, π) for every γ and β."""
    return rng.random(2 * int(p)) * np.pi


def n_qaoa_parameters(p_layers: int) -> int:
    return 2 * int(p_layers)


@dataclass
class QAOATrialResult:
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
    energy_mean_cost: float


@dataclass
class QAOASimulator:
    """QAOA SPSA Gibbs: optimize w.r.t. cost_energies; score vs true GS."""

    cost_energies: np.ndarray
    ground_bitstring: str
    eval_energies: np.ndarray | None = None
    n: int = N
    eta_ctrl: SampledTailEta = field(default_factory=SampledTailEta)
    _current_eta: float = 1.0

    def __post_init__(self) -> None:
        self.cost_energies = np.asarray(self.cost_energies, dtype=float).reshape(-1)
        if self.cost_energies.size != (1 << int(self.n)):
            raise ValueError(f"cost_energies must have length {1 << int(self.n)}")
        if self.eval_energies is None:
            self.eval_energies = self.cost_energies
        else:
            self.eval_energies = np.asarray(self.eval_energies, dtype=float).reshape(-1)
        self.ground_index = index_from_bitstring(self.ground_bitstring)

    def probs_from_x(self, x: np.ndarray) -> np.ndarray:
        return born_probs(qaoa_state(x, self.cost_energies, self.n))

    def cost(self, x: np.ndarray, eta: float | None = None) -> float:
        probs = self.probs_from_x(x)
        use_eta = self._current_eta if eta is None else float(eta)
        return gibbs_objective(probs, self.cost_energies, use_eta)

    def refresh_eta(self, x: np.ndarray, step: int) -> float:
        probs = self.probs_from_x(x)
        self._current_eta = self.eta_ctrl.update(self.cost_energies, probs, step)
        return self._current_eta

    def evaluate(self, x: np.ndarray) -> dict:
        probs = self.probs_from_x(x)
        idx = int(np.argmax(probs))
        ml = bitstring_from_index(idx, self.n)
        return {
            "most_likely_bitstring": ml,
            "p_gs": float(probs[self.ground_index]),
            "success": ml == self.ground_bitstring,
            "energy_mean": float(np.dot(probs, self.eval_energies)),
            "energy_mean_cost": float(np.dot(probs, self.cost_energies)),
            "probs": probs,
        }


def optimize_qaoa_trial(
    sim: QAOASimulator,
    p_layers: int,
    *,
    maxiter: int = 200,
    rng: np.random.Generator | None = None,
    x0: np.ndarray | None = None,
    a: float | None = None,
    c: float = 0.15,
    A: float = 10.0,
) -> QAOATrialResult:
    rng = rng or np.random.default_rng()
    n_params = n_qaoa_parameters(p_layers)
    x0 = random_qaoa_params(p_layers, rng) if x0 is None else np.asarray(x0, dtype=float)
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
    return QAOATrialResult(
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
        energy_mean_cost=float(ev["energy_mean_cost"]),
    )
