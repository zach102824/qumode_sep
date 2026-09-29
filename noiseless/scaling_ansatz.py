"""n-qubit HEA and RY-only product ansatz for the planted 4-SAT scaling study.

Generalizes ``noiseless/hea.py`` (n = 8) to any even n with the SAME conventions:
start |0>^n, RY(θ) = exp(-iθY/2), MSB-first basis (site 0 = bitstring char 0),
parameters x.reshape(L+1, n), init Uniform[0, π), SPSA (a = scale_spsa_a(n_params),
c = 0.15, A = 10, α = 0.602, γ = 0.101), Gibbs cost ``gibbs_objective`` with the
``SampledTailEta`` η controller refreshed every 5 steps, success = argmax bitstring
== GS, p(GS) = Born probability of the GS.

Lattice: 2 × n/2 snake numbering (1-indexed)::

    q1   - q2     - ... - q_{n/2}
    |      |               |
    q_n  - q_{n-1} - ... - q_{n/2+1}

CZ sublayer A: chain edges (k, k+1), k odd; B: chain edges, k even; C: rungs
(i, n+1-i), i = 1..n/2-1 (the rung (n/2, n/2+1) is a chain edge).  All 3n/2 - 2
lattice edges per layer; for n = 8 this is exactly hea.CZ_SUBLAYERS.

HEA (L >= 1) is simulated by a real statevector (n <= 20 in the study).

RY-only (L = 0) is a product state with P(x_q = 1) = sin^2(θ_q / 2):
  * <H> = Σ_c Π_{j in c} P(x_j = pattern_j)  and  p(GS) = Π_q P(x_q = gs_q), analytic;
  * argmax bitstring = bitwise (P(x_q=1) > 1/2);
  * Gibbs cost ``exact``: product distribution on the enumerated spectrum (n <= 20);
  * Gibbs cost ``sampled``: N_s bitstrings drawn from the product distribution,
    G = -log( mean_s exp(-η E_s) )  (E_min = 0 is known for unique planted SAT, the
    same shift the exact cost uses), η refreshed from the samples' weighted
    quantiles (uniform weights).  Common random numbers: one uniform matrix
    U ∈ [0,1)^{N_s × n} is drawn per SPSA step (bit_q = U_q < P(x_q = 1)) and shared by
    the y+ / y- evaluations and the η refresh of that step, from a dedicated sampling
    rng so the SPSA Δ stream / x0 are identical to the exact-cost run at equal seed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .spsa_gibbs import (
    SampledTailEta,
    clamp_eta,
    eta_from_tail,
    gibbs_objective,
    run_spsa,
    scale_spsa_a,
)

DEFAULT_N_SAMPLES = 4096


# ---------------------------------------------------------------------------
# lattice / circuit
# ---------------------------------------------------------------------------
def snake_cz_sublayers(n: int) -> tuple[tuple[tuple[int, int], ...], ...]:
    if n % 2 or n < 4:
        raise ValueError("n must be even and >= 4")
    a = tuple((k, k + 1) for k in range(1, n, 2))
    b = tuple((k, k + 1) for k in range(2, n, 2))
    c = tuple((i, n + 1 - i) for i in range(1, n // 2))
    return a, b, c


def snake_lattice_edges(n: int) -> set[tuple[int, int]]:
    h = n // 2
    top = list(range(1, h + 1))
    bot = [n + 1 - i for i in top]
    e = set()
    for row in (top, bot):
        for x, y in zip(row[:-1], row[1:]):
            e.add((min(x, y), max(x, y)))
    for x, y in zip(top, bot):
        e.add((min(x, y), max(x, y)))
    return e


def cz_signs(n: int) -> np.ndarray:
    idx = np.arange(1 << n, dtype=np.int64)
    par = np.zeros(1 << n, dtype=np.int8)
    for sub in snake_cz_sublayers(n):
        for x, y in sub:
            par ^= (((idx >> (n - x)) & 1) & ((idx >> (n - y)) & 1)).astype(np.int8)
    return (1 - 2 * par).astype(float)


def apply_ry_layer_inplace(psi: np.ndarray, thetas: np.ndarray, n: int) -> np.ndarray:
    for q in range(n):
        c = math.cos(0.5 * float(thetas[q]))
        s = math.sin(0.5 * float(thetas[q]))
        t = psi.reshape(1 << q, 2, 1 << (n - q - 1))
        a = t[:, 0, :].copy()
        b = t[:, 1, :]
        t[:, 0, :] *= c
        t[:, 0, :] -= s * b
        b *= c
        b += s * a
    return psi


_CZ_CACHE: dict[int, np.ndarray] = {}


def hea_state_n(params: np.ndarray, n: int) -> np.ndarray:
    x = np.asarray(params, dtype=float).reshape(-1)
    if x.size % n or x.size < n:
        raise ValueError(f"HEA parameter length must be n(L+1), got {x.size} for n={n}")
    rows = x.reshape(-1, n)
    n_layers = rows.shape[0] - 1
    if n_layers and n not in _CZ_CACHE:
        _CZ_CACHE[n] = cz_signs(n)
    psi = np.zeros(1 << n, dtype=float)
    psi[0] = 1.0
    for l in range(n_layers):
        apply_ry_layer_inplace(psi, rows[l], n)
        psi *= _CZ_CACHE[n]
    apply_ry_layer_inplace(psi, rows[n_layers], n)
    return psi


def n_params(n: int, n_layers: int) -> int:
    return int(n) * (int(n_layers) + 1)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def weighted_quantiles_presorted(v_sorted: np.ndarray, w_sorted: np.ndarray, qs) -> list[float]:
    """Same arithmetic as spsa_gibbs.weighted_quantile, with the sort precomputed."""
    w = np.clip(w_sorted, 0.0, None)
    total = float(w.sum())
    if v_sorted.size == 0 or total <= 0.0:
        return [float("nan")] * len(qs)
    cdf = np.cumsum(w / total)
    cdf = np.clip(cdf, 0.0, 1.0)
    cdf[-1] = 1.0
    return [float(np.interp(float(q), cdf, v_sorted)) for q in qs]


def eta_update_presorted(ctrl: SampledTailEta, v_sorted, w_sorted, step: int) -> float:
    """SampledTailEta.update with presorted energies (identical numerics)."""
    if step - ctrl.last_step_updated < ctrl.refresh_every and ctrl.history:
        return ctrl.eta
    q05, q25, q75 = weighted_quantiles_presorted(v_sorted, w_sorted, (0.05, 0.25, 0.75))
    floor = max(q25 - q05, 0.25 * max(q75 - q25, 0.0), 1e-8)
    target, fallback = eta_from_tail(q05, q25, floor)
    if ctrl.history:
        target = (1.0 - ctrl.ema) * ctrl.eta + ctrl.ema * target
    eta, clamped = clamp_eta(target)
    if clamped:
        fallback = fallback or "clamped"
    ctrl.eta = float(eta)
    ctrl.last_step_updated = int(step)
    ctrl.history.append({"step": int(step), "eta": float(eta), "fallback": fallback, "clamped": bool(clamped)})
    return ctrl.eta


def clause_masks(clauses, polarities, n):
    clauses = np.asarray(clauses, dtype=np.int64)
    pats = np.where(np.asarray(polarities) > 0, 0, 1).astype(np.int64)
    sh = (n - 1 - clauses).astype(np.int64)
    masks = np.bitwise_or.reduce(np.int64(1) << sh, axis=1)
    vals = np.bitwise_or.reduce(pats << sh, axis=1)
    return masks, vals


def spectrum(clauses, polarities, n) -> np.ndarray:
    masks, vals = clause_masks(clauses, polarities, n)
    idx = np.arange(1 << n, dtype=np.int64)
    e = np.zeros(1 << n, dtype=np.float64)
    for m_, v_ in zip(masks, vals, strict=True):
        e += (idx & m_) == v_
    return e


@dataclass
class TrialOut:
    success: bool
    p_gs: float
    energy_mean: float
    most_likely_bitstring: str
    fun: float
    eta: float
    nfev: int
    x: np.ndarray


# ---------------------------------------------------------------------------
# HEA (statevector)
# ---------------------------------------------------------------------------
@dataclass
class HEAnSimulator:
    energies: np.ndarray
    ground_bitstring: str
    n: int
    n_layers: int
    eta_ctrl: SampledTailEta = field(default_factory=SampledTailEta)
    _current_eta: float = 1.0

    def __post_init__(self):
        self.energies = np.asarray(self.energies, dtype=float).reshape(-1)
        self.ground_index = int(self.ground_bitstring, 2)
        self._order = np.argsort(self.energies, kind="mergesort")
        self._e_sorted = self.energies[self._order]

    def probs_from_x(self, x):
        psi = hea_state_n(x, self.n)
        p = psi * psi
        return p / p.sum()

    def cost(self, x, eta=None):
        return gibbs_objective(self.probs_from_x(x), self.energies, self._current_eta if eta is None else eta)

    def refresh_eta(self, x, step):
        p = self.probs_from_x(x)
        self._current_eta = eta_update_presorted(self.eta_ctrl, self._e_sorted, p[self._order], step)
        return self._current_eta

    def evaluate(self, x):
        p = self.probs_from_x(x)
        idx = int(np.argmax(p))
        ml = format(idx, f"0{self.n}b")
        return {"most_likely_bitstring": ml, "p_gs": float(p[self.ground_index]),
                "success": ml == self.ground_bitstring, "energy_mean": float(np.dot(p, self.energies))}


# ---------------------------------------------------------------------------
# RY-only product
# ---------------------------------------------------------------------------
@dataclass
class ProductRYSimulator:
    clauses: np.ndarray
    polarities: np.ndarray
    ground_bitstring: str
    n: int
    cost_mode: str = "exact"  # "exact" | "sampled"
    n_samples: int = DEFAULT_N_SAMPLES
    energies: np.ndarray | None = None
    eta_ctrl: SampledTailEta = field(default_factory=SampledTailEta)
    _current_eta: float = 1.0

    def __post_init__(self):
        self.clauses = np.asarray(self.clauses, dtype=np.int64)
        self.patterns = np.where(np.asarray(self.polarities) > 0, 0, 1).astype(np.int64)
        self.gs_bits = np.array([int(c) for c in self.ground_bitstring], dtype=np.int64)
        self.masks, self.vals = clause_masks(self.clauses, self.polarities, self.n)
        self.weights = (np.int64(1) << np.arange(self.n - 1, -1, -1, dtype=np.int64))
        if self.cost_mode == "exact":
            if self.energies is None:
                self.energies = spectrum(self.clauses, self.polarities, self.n)
            self._order = np.argsort(self.energies, kind="mergesort")
            self._e_sorted = self.energies[self._order]
        elif self.cost_mode != "sampled":
            raise ValueError(self.cost_mode)
        self._U: np.ndarray | None = None

    # analytic quantities
    @staticmethod
    def p1(x):
        return np.sin(0.5 * np.asarray(x, dtype=float)) ** 2

    def analytic(self, x):
        p1 = self.p1(x)
        pv = np.where(self.patterns == 1, p1[self.clauses], 1.0 - p1[self.clauses])
        e_mean = float(np.prod(pv, axis=1).sum())
        p_gs = float(np.prod(np.where(self.gs_bits == 1, p1, 1.0 - p1)))
        ml = "".join("1" if v > 0.5 else "0" for v in p1)
        return e_mean, p_gs, ml

    def probs_from_x(self, x):
        p = np.ones(1)
        for v in self.p1(x):
            p = np.outer(p, (1.0 - v, v)).reshape(-1)  # MSB-first
        return p

    # sampling
    def resample(self, rng):
        self._U = rng.random((self.n_samples, self.n))

    def sample_energies(self, x):
        bits = (self._U < self.p1(x)[None, :]).astype(np.int64)
        idx = bits @ self.weights
        e = np.zeros(idx.size, dtype=float)
        for m_, v_ in zip(self.masks, self.vals, strict=True):
            e += (idx & m_) == v_
        return e

    def cost(self, x, eta=None):
        eta = self._current_eta if eta is None else float(eta)
        if self.cost_mode == "exact":
            return gibbs_objective(self.probs_from_x(x), self.energies, eta)
        e = self.sample_energies(x)
        avg = float(np.mean(np.exp(-eta * e)))
        return float(-np.log(max(avg, 1e-300)))

    def refresh_eta(self, x, step):
        if self.cost_mode == "exact":
            p = self.probs_from_x(x)
            self._current_eta = eta_update_presorted(self.eta_ctrl, self._e_sorted, p[self._order], step)
        else:
            e = np.sort(self.sample_energies(x), kind="mergesort")
            self._current_eta = eta_update_presorted(self.eta_ctrl, e, np.ones(e.size), step)
        return self._current_eta

    def evaluate(self, x):
        e_mean, p_gs, ml = self.analytic(x)
        return {"most_likely_bitstring": ml, "p_gs": p_gs, "success": ml == self.ground_bitstring, "energy_mean": e_mean}


def optimize_trial(sim, n_par: int, *, maxiter=200, rng=None, sample_rng=None, x0=None, a=None, c=0.15, A=10.0) -> TrialOut:
    """SPSA + η refresh cadence of hea.optimize_hea_trial; resample per step if sampled."""
    rng = rng or np.random.default_rng()
    x0 = rng.random(n_par) * np.pi if x0 is None else np.asarray(x0, dtype=float)
    a = scale_spsa_a(n_par) if a is None else a
    sim.eta_ctrl = SampledTailEta()
    sim._current_eta = 1.0
    sampled = getattr(sim, "cost_mode", "exact") == "sampled"

    def on_before(step, x):
        if sampled:
            sim.resample(sample_rng)
        if (step - 1) % sim.eta_ctrl.refresh_every == 0:
            sim.refresh_eta(x, step)

    xf, fun, nfev = run_spsa(sim.cost, x0, maxiter=maxiter, rng=rng, a=float(a), c=c, A=A, on_before_step=on_before)
    ev = sim.evaluate(xf)
    return TrialOut(bool(ev["success"]), float(ev["p_gs"]), float(ev["energy_mean"]), ev["most_likely_bitstring"],
                    float(fun), float(sim._current_eta), int(nfev), xf)
