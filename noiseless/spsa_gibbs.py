"""SPSA + sampled_tail Gibbs cost on the noiseless local-ECD circuit."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import qutip as qt

from .circuit_local_ecd import (
    apply_circuit_np,
    born_probs_np,
    extract_u_ab,
    n_parameters,
    qobj_to_np,
    random_parameters,
    unpack_params,
)
from .encoding import (
    DIMS,
    bitstring_from_bits,
    bits_from_bitstring,
    bits_from_denm,
    denm_from_bits,
    denm_from_flat,
    flat_index,
)

ETA_MIN = 1e-4
ETA_MAX = 50.0
LN20 = math.log(20.0)
EMA_ALPHA = 0.35
DEFAULT_REFRESH_EVERY = 5
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
    return max(q25 - q05, 0.25 * max(q75 - q25, 0.0), 1e-8)


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
            {"step": int(step), "eta": float(eta), "fallback": fallback, "clamped": bool(clamped)}
        )
        return self.eta


def gibbs_objective(probs: np.ndarray, energies: np.ndarray, eta: float) -> float:
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


def betas_from_x(x: np.ndarray, n_layers: int) -> np.ndarray:
    """Complex β_d, β_e per layer from Cartesian parameter vector (same layout as unpack_params)."""
    layers = unpack_params(x, n_layers)
    out: list[complex] = []
    for layer in layers:
        out.append(complex(layer["beta_d"]))
        out.append(complex(layer["beta_e"]))
    return np.asarray(out, dtype=complex)


def _finite_beta_max(beta_max: float | None) -> float | None:
    if beta_max is None:
        return None
    bm = float(beta_max)
    if not math.isfinite(bm):
        return None
    return bm


def beta_regularizer(
    betas: np.ndarray,
    *,
    lambda1: float = 0.0,
    lambda3: float = 0.0,
    beta_max: float | None = None,
) -> float:
    """λ1 * Σ|β_i| + λ3 * Σ max(|β_i|-β_max, 0)^2. Zero when λ1=λ3=0 (no extra work)."""
    l1 = float(lambda1)
    l3 = float(lambda3)
    if l1 == 0.0 and l3 == 0.0:
        return 0.0
    abs_b = np.abs(np.asarray(betas, dtype=complex).reshape(-1))
    reg = l1 * float(np.sum(abs_b))
    bm = _finite_beta_max(beta_max)
    if l3 != 0.0 and bm is not None:
        excess = np.maximum(abs_b - bm, 0.0)
        reg += l3 * float(np.sum(excess * excess))
    return float(reg)


def beta_abs_stats(
    betas: np.ndarray, beta_max: float | None = None
) -> tuple[float, float, float]:
    """Return (mean_|β|, max_|β|, frac_|β|>β_max). frac is 0 when β_max is None/inf."""
    abs_b = np.abs(np.asarray(betas, dtype=complex).reshape(-1))
    if abs_b.size == 0:
        return 0.0, 0.0, 0.0
    mean_abs = float(np.mean(abs_b))
    max_abs = float(np.max(abs_b))
    bm = _finite_beta_max(beta_max)
    if bm is None:
        frac = 0.0
    else:
        frac = float(np.mean(abs_b > bm))
    return mean_abs, max_abs, frac


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
    mean_abs_beta: float = 0.0
    max_abs_beta: float = 0.0
    frac_over_beta_max: float = 0.0


@dataclass
class NoiselessSimulator:
    u_fixed: qt.Qobj | np.ndarray
    energy_tensor: np.ndarray
    n_layers: int
    ground_bitstring: str
    ground_flat_index: int
    lambda1: float = 0.0
    lambda3: float = 0.0
    beta_max: float | None = None

    def __post_init__(self) -> None:
        self.energies_flat = np.asarray(self.energy_tensor, dtype=float).reshape(-1)
        if self.energies_flat.size != int(np.prod(DIMS)):
            raise ValueError("energy_tensor must match dims (2,2,8,8)")
        if isinstance(self.u_fixed, qt.Qobj):
            self.u_np = qobj_to_np(self.u_fixed)
        else:
            self.u_np = np.asarray(self.u_fixed, dtype=complex)
        self.u_ab = extract_u_ab(self.u_np) if self.u_np.shape == (256, 256) else self.u_np
        self.eta_ctrl = SampledTailEta()
        self._current_eta = 1.0

    def probs_from_x(self, x: np.ndarray) -> np.ndarray:
        ket = apply_circuit_np(x, self.n_layers, self.u_np, u_ab=self.u_ab)
        return born_probs_np(ket)

    def cost(self, x: np.ndarray, eta: float | None = None) -> float:
        probs = self.probs_from_x(x)
        use_eta = self._current_eta if eta is None else float(eta)
        gibbs = gibbs_objective(probs, self.energies_flat, use_eta)
        # Default λ1=λ3=0: return Gibbs alone so the path matches pre-β-aware numerics exactly.
        if float(self.lambda1) == 0.0 and float(self.lambda3) == 0.0:
            return gibbs
        betas = betas_from_x(x, self.n_layers)
        return float(
            gibbs
            + beta_regularizer(
                betas,
                lambda1=self.lambda1,
                lambda3=self.lambda3,
                beta_max=self.beta_max,
            )
        )

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
        betas = betas_from_x(x, self.n_layers)
        mean_abs, max_abs, frac = beta_abs_stats(betas, self.beta_max)
        return {
            "most_likely_bitstring": ml,
            "p_gs": float(probs[self.ground_flat_index]),
            "success": ml == self.ground_bitstring,
            "energy_mean": float(np.dot(probs, self.energies_flat)),
            "probs": probs,
            "mean_abs_beta": mean_abs,
            "max_abs_beta": max_abs,
            "frac_over_beta_max": frac,
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
    for k in range(1, int(maxiter) + 1):
        if on_before_step is not None:
            on_before_step(k, x)
        ak = a / (k + A) ** alpha
        ck = c / k**gamma
        delta = rng.choice([-1.0, 1.0], size=x.size)
        yp = float(fun(x + ck * delta))
        ym = float(fun(x - ck * delta))
        nfev += 2
        ghat = (yp - ym) / (2.0 * ck) * delta
        x = x - ak * ghat
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
    lambda1: float | None = None,
    lambda3: float | None = None,
    beta_max: float | None = None,
) -> TrialResult:
    """SPSA trial. Optional ``lambda1``/``lambda3``/``beta_max`` override ``sim`` fields.

    When omitted, existing ``sim.lambda1`` / ``sim.lambda3`` / ``sim.beta_max`` are
    used (defaults 0 / 0 / None → Gibbs-only cost, identical to pre-β-aware).
    Pass ``beta_max=float('inf')`` (or None on the simulator) for no cap term.
    """
    rng = rng or np.random.default_rng()
    n_params = n_parameters(sim.n_layers)
    x0 = random_parameters(sim.n_layers, rng) if x0 is None else np.asarray(x0, dtype=float)
    if a is None:
        a = scale_spsa_a(n_params)
    if lambda1 is not None:
        sim.lambda1 = float(lambda1)
    if lambda3 is not None:
        sim.lambda3 = float(lambda3)
    if beta_max is not None:
        sim.beta_max = _finite_beta_max(beta_max)
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
        mean_abs_beta=float(ev["mean_abs_beta"]),
        max_abs_beta=float(ev["max_abs_beta"]),
        frac_over_beta_max=float(ev["frac_over_beta_max"]),
    )


def ground_flat_from_bitstring(bitstring: str) -> int:
    d, e, n_a, n_b = denm_from_bits(bits_from_bitstring(bitstring))
    return flat_index(d, e, n_a, n_b)


LocalEcdGibbsSim = NoiselessSimulator
