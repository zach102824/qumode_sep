"""SPSA + sampled_tail Gibbs cost on the noiseless local-ECD circuit.

Mirrors qumode HybridSimulator / optimize_gibbs style:
  f = −ln ⟨e^{−η E}⟩ with η from probability-weighted 5%/25% energy quantiles
  (no known E_min during optimization).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np
import qutip as qt

from .circuit_local_ecd import (
    apply_circuit,
    born_probs,
    n_parameters,
    random_parameters,
)
from .encoding import (
    DIMS,
    bitstring_from_bits,
    bits_from_denm,
    denm_from_flat,
    flat_index,
)

ETA_MIN = 1e-4
ETA_MAX = 50.0
LN20 = math.log(20.0)
EMA_ALPHA = 0.35
DEFAULT_REFRESH_EVERY = 5

# Baseline a≈0.2 at n_params=37 (old n=7 joint); scale ∝ 1/√n_params
BASELINE_A = 0.2
BASELINE_NPARAMS = 37


def scale_spsa_a(n_params: int, baseline_a: float = BASELINE_A) -> float:
    return float(baseline_a) * math.sqrt(float(BASELINE_NPARAMS) / max(int(n_params), 1))


def weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    v = np.asarray(values, dtype=float).reshape(-1)
    w = np.clip(np.asarray(weights, dtype=float).reshape(-1), 0.0, None)
    total = float(w.sum())
    if v.size == 0 or total <= 0.0:
        return float("nan")
    w = w / total
    order = np.argsort(v, kind="mergesort")
    v = v[order]
    cdf = np.cumsum(w[order])
    cdf = np.clip(cdf, 0.0, 1.0)
    cdf[-1] = 1.0
    return float(np.interp(float(q), cdf, v))


def robust_scale(values: np.ndarray, weights: np.ndarray) -> float:
    q05 = weighted_quantile(values, weights, 0.05)
    q25 = weighted_quantile(values, weights, 0.25)
    q75 = weighted_quantile(values, weights, 0.75)
    iqr = max(q75 - q25, 0.0)
    tail = max(q25 - q05, 0.0)
    return max(tail, 0.25 * iqr, 1e-8)


def clamp_eta(eta: float) -> tuple[float, bool]:
    x = float(eta)
    if not math.isfinite(x):
        return ETA_MIN, True
    clamped = min(max(x, ETA_MIN), ETA_MAX)
    return clamped, clamped != x


def eta_from_tail(q05: float, q25: float, floor: float) -> tuple[float, str | None]:
    span = float(q25 - q05)
    fallback = None
    if not math.isfinite(span) or span <= 1e-12:
        span = max(float(floor), 1e-8)
        fallback = "degenerate_tail"
    eta, clamped = clamp_eta(LN20 / span)
    if clamped:
        fallback = fallback or "clamped"
    return eta, fallback


@dataclass
class SampledTailEta:
    refresh_every: int = DEFAULT_REFRESH_EVERY
    ema: float = EMA_ALPHA
    eta: float = 1.0
    history: list[dict] = field(default_factory=list)
    last_step_updated: int = -10**9

    def update(self, energies: np.ndarray, probs: np.ndarray, step: int) -> float:
        if step - self.last_step_updated < self.refresh_every and self.history:
            return self.eta
        e = np.asarray(energies, dtype=float).reshape(-1)
        p = np.asarray(probs, dtype=float).reshape(-1)
        q05 = weighted_quantile(e, p, 0.05)
        q25 = weighted_quantile(e, p, 0.25)
        floor = robust_scale(e, p)
        target, fallback = eta_from_tail(q05, q25, floor)
        if self.history:
            target = (1.0 - self.ema) * self.eta + self.ema * target
        eta, clamped = clamp_eta(target)
        if clamped:
            fallback = fallback or "clamped"
        self.eta = float(eta)
        self.last_step_updated = int(step)
        self.history.append(
            {
                "step": int(step),
                "eta": float(eta),
                "fallback": fallback,
                "clamped": bool(clamped),
            }
        )
        return self.eta


def gibbs_objective(probs: np.ndarray, energies: np.ndarray, eta: float) -> float:
    """f = −ln ⟨e^{−ηE}⟩, shifted by min E for numerics (not used as known E_min)."""
    p = np.asarray(probs, dtype=float).reshape(-1)
    e = np.asarray(energies, dtype=float).reshape(-1)
    p = np.clip(p, 0.0, None)
    total = float(p.sum())
    if total <= 0.0:
        return 0.0
    p = p / total
    emin = float(np.min(e))
    avg = float(np.dot(p, np.exp(-float(eta) * (e - emin))))
    return float(-np.log(max(avg, 1e-300)) + float(eta) * emin)


@dataclass
class TrialResult:
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
class NoiselessSimulator:
    """Cached U_fixed + energy tensor; ECD params only."""

    u_fixed: qt.Qobj
    energy_tensor: np.ndarray
    n_layers: int
    ground_bitstring: str
    ground_flat_index: int

    def __post_init__(self) -> None:
        self.energies_flat = np.asarray(self.energy_tensor, dtype=float).reshape(-1)
        if self.energies_flat.size != int(np.prod(DIMS)):
            raise ValueError("energy_tensor must match dims (2,2,8,8)")
        self.eta_ctrl = SampledTailEta()
        self._current_eta = 1.0

    def probs_from_x(self, x: np.ndarray) -> np.ndarray:
        ket = apply_circuit(x, self.n_layers, self.u_fixed)
        return born_probs(ket)

    def cost(self, x: np.ndarray, eta: float | None = None) -> float:
        probs = self.probs_from_x(x)
        use_eta = self._current_eta if eta is None else float(eta)
        return gibbs_objective(probs, self.energies_flat, use_eta)

    def refresh_eta(self, x: np.ndarray, step: int) -> float:
        probs = self.probs_from_x(x)
        self._current_eta = self.eta_ctrl.update(self.energies_flat, probs, step)
        return self._current_eta

    def evaluate(self, x: np.ndarray) -> dict:
        probs = self.probs_from_x(x)
        idx = int(np.argmax(probs))
        d, e, n_a, n_b = denm_from_flat(idx)
        bits = bits_from_denm(d, e, n_a, n_b)
        ml = bitstring_from_bits(bits)
        p_gs = float(probs[self.ground_flat_index])
        e_mean = float(np.dot(probs, self.energies_flat))
        return {
            "most_likely_bitstring": ml,
            "p_gs": p_gs,
            "success": ml == self.ground_bitstring,
            "energy_mean": e_mean,
            "probs": probs,
        }


def run_spsa(
    fun: Callable[[np.ndarray], float],
    x0: np.ndarray,
    *,
    maxiter: int,
    rng: np.random.Generator,
    a: float = 0.2,
    c: float = 0.15,
    A: float = 10.0,
    alpha: float = 0.602,
    gamma: float = 0.101,
    on_before_step: Callable[[int, np.ndarray], None] | None = None,
) -> tuple[np.ndarray, float, int]:
    x = np.asarray(x0, dtype=float).copy()
    nfev = 0
    last_fun = 0.0
    for k in range(1, int(maxiter) + 1):
        if on_before_step is not None:
            on_before_step(k, x)
        ak = a / (k + A) ** alpha
        ck = c / k**gamma
        delta = rng.choice([-1.0, 1.0], size=x.size)
        xp = x + ck * delta
        xm = x - ck * delta
        yp = float(fun(xp))
        ym = float(fun(xm))
        nfev += 2
        ghat = (yp - ym) / (2.0 * ck) * delta
        x = x - ak * ghat
        last_fun = 0.5 * (yp + ym)
    return x, float(fun(x)), nfev + 1


def optimize_trial(
    sim: NoiselessSimulator,
    *,
    maxiter: int = 200,
    rng: np.random.Generator | None = None,
    x0: np.ndarray | None = None,
    a: float | None = None,
    c: float = 0.15,
    A: float = 10.0,
) -> TrialResult:
    rng = rng or np.random.default_rng()
    n_params = n_parameters(sim.n_layers)
    if x0 is None:
        x0 = random_parameters(sim.n_layers, rng)
    else:
        x0 = np.asarray(x0, dtype=float)
    if a is None:
        a = scale_spsa_a(n_params)
    sim.eta_ctrl = SampledTailEta()
    sim._current_eta = 1.0

    def on_before(step: int, x: np.ndarray) -> None:
        sim.refresh_eta(x, step)

    def fun(x: np.ndarray) -> float:
        return sim.cost(x)

    x_final, fun_final, nfev = run_spsa(
        fun,
        x0,
        maxiter=maxiter,
        rng=rng,
        a=float(a),
        c=c,
        A=A,
        on_before_step=on_before,
    )
    ev = sim.evaluate(x_final)
    return TrialResult(
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


def ground_flat_from_bitstring(bitstring: str) -> int:
    from .encoding import bits_from_bitstring, denm_from_bits

    d, e, n_a, n_b = denm_from_bits(bits_from_bitstring(bitstring))
    return flat_index(d, e, n_a, n_b)
