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
    lam: float = 0.0,
    lambda3: float | None = None,
    beta_max: float | None = None,
) -> float:
    """λ1 * Σ|β_i| + λ * Σ max(|β_i|-β_max, 0)^2. Zero when λ1=λ=0 (no extra work).

    ``lam`` is the soft-cap weight λ. Legacy ``lambda3`` overrides ``lam`` when not None.
    """
    l1 = float(lambda1)
    l3 = float(lambda3) if lambda3 is not None else float(lam)
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
    final_lam: float = 0.0
    mean_lam_post_warmup: float = 0.0
    optimizer: str = "spsa"
    opt_status: int | None = None  # scipy status (BFGS only)
    opt_message: str | None = None  # scipy message (BFGS only)
    stages: list[dict] | None = None  # per-stage metrics (grow_trial only)


# Adaptive soft-cap λ defaults (opt-in via optimize_trial adapt_lambda=True)
ADAPT_WARMUP_FRAC = 0.25
ADAPT_EVERY = 25
ADAPT_F_HI = 0.20
ADAPT_F_LO = 0.05
ADAPT_LAM_MIN = 0.5
ADAPT_LAM_MAX = 5.0
ADAPT_LAM_TINY = 1e-6


@dataclass
class NoiselessSimulator:
    u_fixed: qt.Qobj | np.ndarray
    energy_tensor: np.ndarray
    n_layers: int
    ground_bitstring: str
    ground_flat_index: int
    lambda1: float = 0.0
    lam: float = 0.0  # soft-cap weight λ (preferred name)
    beta_max: float | None = None
    lambda3: float = 0.0  # legacy alias of lam; kept in sync in __post_init__ / set_lam
    encoding: str = "binary"  # physical Fock → logical bit map used to decode argmax
    # Multiplier on the controller's η when it is refreshed (η_used = eta_scale · η_ctrl).
    # η is an INVERSE temperature in gibbs_objective: smaller = hotter (→ η·<E>, mean
    # energy), larger = colder (→ emphasizes -log p(GS)). 1.0 = legacy (x*1.0 == x exactly).
    eta_scale: float = 1.0

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
        # Sync lam ↔ lambda3 (prefer non-zero lam; else adopt legacy lambda3).
        if float(self.lam) != 0.0:
            self.lambda3 = float(self.lam)
        elif float(self.lambda3) != 0.0:
            self.lam = float(self.lambda3)
        else:
            self.lam = 0.0
            self.lambda3 = 0.0

    def set_lam(self, value: float) -> None:
        """Set soft-cap λ and keep legacy ``lambda3`` mirrored."""
        v = float(value)
        self.lam = v
        self.lambda3 = v

    def probs_from_x(self, x: np.ndarray) -> np.ndarray:
        ket = apply_circuit_np(x, self.n_layers, self.u_np, u_ab=self.u_ab)
        return born_probs_np(ket)

    def cost(self, x: np.ndarray, eta: float | None = None) -> float:
        probs = self.probs_from_x(x)
        use_eta = self._current_eta if eta is None else float(eta)
        gibbs = gibbs_objective(probs, self.energies_flat, use_eta)
        # Default λ1=λ=0: return Gibbs alone so the path matches pre-β-aware numerics exactly.
        if float(self.lambda1) == 0.0 and float(self.lam) == 0.0:
            return gibbs
        betas = betas_from_x(x, self.n_layers)
        return float(
            gibbs
            + beta_regularizer(
                betas,
                lambda1=self.lambda1,
                lam=self.lam,
                beta_max=self.beta_max,
            )
        )

    def refresh_eta(self, x: np.ndarray, step: int) -> float:
        probs = self.probs_from_x(x)
        eta = self.eta_ctrl.update(self.energies_flat, probs, step)
        self._current_eta = eta * float(self.eta_scale)
        return self._current_eta

    def evaluate(self, x: np.ndarray) -> dict:
        probs = self.probs_from_x(x)
        idx = int(np.argmax(probs))
        d, e, n_a, n_b = denm_from_flat(idx)
        bits = bits_from_denm(d, e, n_a, n_b, self.encoding)
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


ADAM_LR = 0.05
ADAM_BETA1 = 0.9
ADAM_BETA2 = 0.999
ADAM_EPS = 1e-8


def run_spsa_adam(
    fun: Callable[[np.ndarray], float],
    x0: np.ndarray,
    *,
    maxiter: int,
    rng: np.random.Generator,
    lr: float = ADAM_LR,
    beta1: float = ADAM_BETA1,
    beta2: float = ADAM_BETA2,
    eps: float = ADAM_EPS,
    c: float = 0.15,
    gamma: float = 0.101,
    on_before_step: Callable[[int, np.ndarray], None] | None = None,
) -> tuple[np.ndarray, float, int]:
    """SPSA gradient estimate + Adam update.

    Identical to :func:`run_spsa` in everything except the parameter update: same
    ``on_before_step`` hook (η refresh), same c_k = c / k**gamma, same Rademacher draw
    ``rng.choice([-1, 1], size=n)`` per step (so the rng stream is consumed identically),
    2 cost evals per step + 1 final eval. Update: Adam with bias correction,
    x -= lr * m_hat / (sqrt(v_hat) + eps). The SPSA gain a_k (a, A, alpha) is unused.
    """
    x = np.asarray(x0, dtype=float).copy()
    m = np.zeros_like(x)
    v = np.zeros_like(x)
    b1 = float(beta1)
    b2 = float(beta2)
    nfev = 0
    for k in range(1, int(maxiter) + 1):
        if on_before_step is not None:
            on_before_step(k, x)
        ck = c / k**gamma
        delta = rng.choice([-1.0, 1.0], size=x.size)
        yp = float(fun(x + ck * delta))
        ym = float(fun(x - ck * delta))
        nfev += 2
        ghat = (yp - ym) / (2.0 * ck) * delta
        m = b1 * m + (1.0 - b1) * ghat
        v = b2 * v + (1.0 - b2) * ghat * ghat
        m_hat = m / (1.0 - b1**k)
        v_hat = v / (1.0 - b2**k)
        x = x - float(lr) * m_hat / (np.sqrt(v_hat) + float(eps))
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
    lam: float | None = None,
    lambda3: float | None = None,
    beta_max: float | None = None,
    adapt_lambda: bool = False,
    adapt_warmup_frac: float = ADAPT_WARMUP_FRAC,
    adapt_every: int = ADAPT_EVERY,
    adapt_f_hi: float = ADAPT_F_HI,
    adapt_f_lo: float = ADAPT_F_LO,
    adapt_lam_min: float = ADAPT_LAM_MIN,
    adapt_lam_max: float = ADAPT_LAM_MAX,
    optimizer: str = "spsa",
    adam_lr: float = ADAM_LR,
) -> TrialResult:
    """SPSA (default), SPSA-Adam or BFGS trial. Optional ``lambda1``/``lam``/``beta_max`` override ``sim`` fields.

    Soft-cap weight is ``lam`` (CLI ``--lambda``). Legacy ``lambda3`` still works as an
    alias when ``lam`` is omitted.

    When ``adapt_lambda`` is False (default), behavior matches today's fixed
    lambda1/lam/beta_max path exactly. When True:
      - steps 1..floor(warmup_frac*maxiter): lam=0 (pure Gibbs warm-up)
      - afterwards every ``adapt_every`` steps, raise/lower lam from frac(|β|>β_max)

    ``optimizer="bfgs"`` uses scipy BFGS (finite-difference gradient) from the same
    x0 (drawn from ``rng`` exactly as for SPSA), refreshing η at iteration 1 and then
    after every 5th iteration (mirrors SPSA refreshes at steps 1, 6, 11, ...).
    ``nfev`` counts true cost evaluations. Adaptive λ is not supported for BFGS.

    ``optimizer="spsa_adam"`` (:func:`run_spsa_adam`) uses the same x0, the same SPSA
    two-point gradient (same c_k, same Rademacher rng draws, same η refresh hook) and the
    same number of cost evaluations (2·maxiter + 1) as SPSA, but updates with Adam
    (lr=``adam_lr``, β1=0.9, β2=0.999, eps=1e-8, bias-corrected). ``a``/``A`` are ignored.
    """
    optimizer = str(optimizer).lower()
    if optimizer not in ("spsa", "bfgs", "spsa_adam"):
        raise ValueError(
            f"unknown optimizer {optimizer!r} (expected 'spsa', 'spsa_adam' or 'bfgs')"
        )
    if optimizer == "bfgs" and adapt_lambda:
        raise ValueError("adapt_lambda is not supported with optimizer='bfgs'")
    rng = rng or np.random.default_rng()
    n_params = n_parameters(sim.n_layers)
    x0 = random_parameters(sim.n_layers, rng) if x0 is None else np.asarray(x0, dtype=float)
    if a is None:
        a = scale_spsa_a(n_params)
    if lambda1 is not None:
        sim.lambda1 = float(lambda1)
    if lam is not None:
        sim.set_lam(lam)
    elif lambda3 is not None:
        sim.set_lam(lambda3)
    if beta_max is not None:
        sim.beta_max = _finite_beta_max(beta_max)
    sim.eta_ctrl = SampledTailEta()
    sim._current_eta = 1.0

    warmup_end = int(math.floor(float(adapt_warmup_frac) * int(maxiter)))
    every = max(int(adapt_every), 1)
    f_hi = float(adapt_f_hi)
    f_lo = float(adapt_f_lo)
    lam_min = float(adapt_lam_min)
    lam_max = float(adapt_lam_max)
    lam_post: list[float] = []

    def on_before(step: int, x: np.ndarray) -> None:
        if (step - 1) % sim.eta_ctrl.refresh_every == 0:
            sim.refresh_eta(x, step)
        if not adapt_lambda:
            return
        if step <= warmup_end:
            sim.set_lam(0.0)
            return
        # Post warm-up: adapt every K steps (aligned to global step index).
        if step % every == 0:
            betas = betas_from_x(x, sim.n_layers)
            abs_b = np.abs(np.asarray(betas, dtype=complex).reshape(-1))
            bm = _finite_beta_max(sim.beta_max)
            if bm is None or abs_b.size == 0:
                frac = 0.0
            else:
                frac = float(np.mean(abs_b > bm))
            cur = float(sim.lam)
            if frac > f_hi:
                if cur == 0.0:
                    sim.set_lam(lam_min)
                else:
                    sim.set_lam(min(lam_max, cur * 1.5))
            elif frac < f_lo:
                nxt = cur / 1.5
                sim.set_lam(0.0 if nxt < ADAPT_LAM_TINY else nxt)
            # else hold
        lam_post.append(float(sim.lam))

    if optimizer == "bfgs":
        return _optimize_trial_bfgs(sim, x0, maxiter=int(maxiter))

    if optimizer == "spsa_adam":
        x_final, fun_final, nfev = run_spsa_adam(
            sim.cost,
            x0,
            maxiter=maxiter,
            rng=rng,
            lr=float(adam_lr),
            c=c,
            on_before_step=on_before,
        )
    else:
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
    final_lam = float(sim.lam)
    if adapt_lambda and lam_post:
        mean_lam_pw = float(np.mean(lam_post))
    else:
        mean_lam_pw = final_lam
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
        final_lam=final_lam,
        mean_lam_post_warmup=mean_lam_pw,
        optimizer=optimizer,
    )


def _optimize_trial_bfgs(sim: NoiselessSimulator, x0: np.ndarray, *, maxiter: int) -> TrialResult:
    """BFGS path of ``optimize_trial`` (sim already reset: fresh eta_ctrl, η=1)."""
    from scipy.optimize import minimize

    x0 = np.asarray(x0, dtype=float).copy()
    counter = {"nfev": 0}

    def fun(x: np.ndarray) -> float:
        counter["nfev"] += 1
        return float(sim.cost(x))

    # Same as SPSA step 1: refresh η from x0 before any cost evaluation.
    sim.refresh_eta(x0, 1)
    state = {"k": 0}
    refresh_every = int(sim.eta_ctrl.refresh_every)

    def callback(xk: np.ndarray) -> None:
        state["k"] += 1
        k = state["k"]
        if k % refresh_every == 0:
            sim.refresh_eta(np.asarray(xk, dtype=float), k + 1)

    res = minimize(fun, x0, method="BFGS", options={"maxiter": int(maxiter)}, callback=callback)
    x_final = np.asarray(res.x, dtype=float)
    fun_final = fun(x_final)
    ev = sim.evaluate(x_final)
    final_lam = float(sim.lam)
    return TrialResult(
        success=bool(ev["success"]),
        p_gs=float(ev["p_gs"]),
        most_likely_bitstring=str(ev["most_likely_bitstring"]),
        ground_bitstring=sim.ground_bitstring,
        fun=float(fun_final),
        eta=float(sim._current_eta),
        nfev=int(counter["nfev"]),
        nit=int(res.nit),
        x=x_final,
        energy_mean=float(ev["energy_mean"]),
        mean_abs_beta=float(ev["mean_abs_beta"]),
        max_abs_beta=float(ev["max_abs_beta"]),
        frac_over_beta_max=float(ev["frac_over_beta_max"]),
        final_lam=final_lam,
        mean_lam_post_warmup=final_lam,
        optimizer="bfgs",
        opt_status=int(res.status),
        opt_message=str(res.message),
    )


def ground_flat_from_bitstring(bitstring: str, encoding: str = "binary") -> int:
    """Physical flat index of a LOGICAL bitstring under ``encoding``."""
    d, e, n_a, n_b = denm_from_bits(bits_from_bitstring(bitstring), encoding)
    return flat_index(d, e, n_a, n_b)


# ---------------------------------------------------------------------------
# Layer growth (append a probability-transparent last layer, kick, retrain)
# ---------------------------------------------------------------------------

GROW_KICK_SIGMA = 0.05


def transparent_layer_params() -> np.ndarray:
    """8 params of a layer that leaves Born probabilities unchanged when appended last.

    β_d = β_e = 0, θ_d = θ_e = π, φ_d = φ_e = 0. With this code's conventions
    R(π, 0) = exp(-i π/2 σx) = -i σx and ECD(0) = σ⁻ ⊗ I + σ⁺ ⊗ I = σx ⊗ I, so
    ECD(0)·R(π,0) = -i · I per transmon (global phase), and the fixed U (jp) is
    diagonal in the Fock basis, hence probability-preserving. (For a non-diagonal U
    the appended layer is NOT transparent.)
    """
    return np.array([0.0, 0.0, 0.0, 0.0, np.pi, np.pi, 0.0, 0.0], dtype=float)


def split_steps(total_steps: int, n_stages: int) -> list[int]:
    """Split ``total_steps`` evenly over stages; remainder goes to the earliest stages.

    200 over 2 → [100, 100]; over 3 → [67, 67, 66]; over 4 → [50, 50, 50, 50].
    """
    n = max(int(n_stages), 1)
    base, rem = divmod(int(total_steps), n)
    return [base + (1 if i < rem else 0) for i in range(n)]


def grow_trial(
    sim: NoiselessSimulator,
    *,
    final_layers: int,
    total_steps: int = 200,
    rng: np.random.Generator | None = None,
    start_layers: int = 1,
    kick_sigma: float = GROW_KICK_SIGMA,
    a: float | None = None,
    c: float = 0.15,
    A: float = 10.0,
    optimizer: str = "spsa",
    steps_per_stage: int | None = None,
    adam_lr: float = ADAM_LR,
    steps_schedule: list[int] | None = None,
    lr_schedule: list[float] | None = None,
    eta_scale_schedule: list[float] | None = None,
    c_schedule: list[float] | None = None,
) -> TrialResult:
    """Layer-growth trial: train L=start_layers from random init, then repeatedly append a
    transparent LAST layer (+ Gaussian kick σ on its 8 params) and retrain, up to
    ``final_layers``. The total budget ``total_steps`` is split evenly over the stages
    (:func:`split_steps`), unless ``steps_per_stage`` is given, in which case every stage
    gets that many steps (generous budget; ``total_steps`` is then ignored). Each stage is a fresh :func:`optimize_trial` call, so SPSA gain
    ``a`` is rescaled for the stage's parameter count (unless ``a`` is given) and the η
    controller restarts. ``sim.n_layers`` is set per stage and ends at ``final_layers``.

    RNG draw order: x0 = random_parameters(start_layers, rng); stage-1 SPSA; kick
    (rng.normal, 8 draws); stage-2 SPSA; ...

    With ``optimizer="spsa_adam"`` each stage starts a FRESH Adam state (m=v=0, bias
    correction restarts at k=1), consistent with the η controller restart; the appended
    layer's params have no history, and the old moments would be mis-scaled anyway.

    Optional per-stage schedules (length must equal the number of stages,
    ``final_layers - start_layers + 1``; None = legacy behaviour, bit-for-bit):
      - ``steps_schedule``: SPSA steps per stage (overrides ``total_steps`` /
        ``steps_per_stage``);
      - ``lr_schedule``: Adam lr per stage (overrides ``adam_lr``);
      - ``eta_scale_schedule``: multiplier on the η controller output per stage
        (``sim.eta_scale``; η is an inverse temperature, so <1 = hotter, >1 = colder).
        ``sim.eta_scale`` is restored afterwards;
      - ``c_schedule``: SPSA perturbation c per stage (overrides ``c``).
    """
    rng = rng or np.random.default_rng()
    final_layers = int(final_layers)
    start_layers = int(start_layers)
    if final_layers < start_layers:
        raise ValueError("final_layers must be >= start_layers")
    layer_seq = list(range(start_layers, final_layers + 1))
    n_st = len(layer_seq)

    def _sched(name: str, vals, default, cast):
        if vals is None:
            return [default] * n_st
        vals = [cast(v) for v in vals]
        if len(vals) != n_st:
            raise ValueError(
                f"{name} has {len(vals)} entries but growth {start_layers}->{final_layers} "
                f"has {n_st} stages"
            )
        return vals

    if steps_schedule is not None:
        steps_seq = _sched("steps_schedule", steps_schedule, None, int)
    elif steps_per_stage is None:
        steps_seq = split_steps(total_steps, n_st)
    else:
        steps_seq = [int(steps_per_stage)] * n_st
    lr_seq = _sched("lr_schedule", lr_schedule, adam_lr, float)
    eta_seq = _sched("eta_scale_schedule", eta_scale_schedule, None, float)
    c_seq = _sched("c_schedule", c_schedule, c, float)
    eta_scale0 = sim.eta_scale
    stages: list[dict] = []
    x = None
    total_nfev = 0
    result: TrialResult | None = None
    for si, (L, steps) in enumerate(zip(layer_seq, steps_seq)):
        sim.n_layers = int(L)
        if eta_seq[si] is not None:
            sim.eta_scale = float(eta_seq[si])
        insert_info: dict = {}
        if x is None:
            x0 = random_parameters(L, rng)
        else:
            x_tr = np.concatenate([np.asarray(x, dtype=float), transparent_layer_params()])
            ev_tr = sim.evaluate(x_tr)
            kick = rng.normal(0.0, float(kick_sigma), size=8)
            x0 = x_tr.copy()
            x0[-8:] += kick
            ev_k = sim.evaluate(x0)
            insert_info = {
                "p_gs_after_insert": float(ev_tr["p_gs"]),
                "p_gs_after_kick": float(ev_k["p_gs"]),
            }
        stage_a = scale_spsa_a(n_parameters(L)) if a is None else float(a)
        result = optimize_trial(
            sim,
            maxiter=int(steps),
            rng=rng,
            x0=x0,
            a=stage_a,
            c=c_seq[si],
            A=A,
            optimizer=optimizer,
            adam_lr=lr_seq[si],
        )
        total_nfev += int(result.nfev)
        x = np.asarray(result.x, dtype=float)
        stages.append(
            {
                "stage": si,
                "n_layers": int(L),
                "n_params": int(x.size),
                "steps": int(steps),
                "spsa_a": float(stage_a),
                "spsa_c": float(c_seq[si]),
                "adam_lr": float(lr_seq[si]),
                "eta_scale": float(sim.eta_scale),
                "success": bool(result.success),
                "p_gs": float(result.p_gs),
                "fun": float(result.fun),
                "eta": float(result.eta),
                "energy_mean": float(result.energy_mean),
                "mean_abs_beta": float(result.mean_abs_beta),
                "max_abs_beta": float(result.max_abs_beta),
                "most_likely_bitstring": result.most_likely_bitstring,
                "nfev": int(result.nfev),
                **insert_info,
            }
        )
    sim.eta_scale = eta_scale0
    assert result is not None
    result.stages = stages
    result.nfev = total_nfev
    result.nit = int(sum(steps_seq))
    return result


LocalEcdGibbsSim = NoiselessSimulator
