"""SPSA with the preregistered 4-start inclusive query budget.

Natural gradient and BFGS are not implemented. Fidelity is not a selection key.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from common import ALPHA_MAX


class OptimizationFailure(RuntimeError):
    pass


@dataclass
class QueryCounter:
    total: int = 0
    calibration: int = 0
    iteration: int = 0
    candidate: int = 0
    failure_score: int = 0

    def add(self, kind: str) -> None:
        self.total += 1
        setattr(self, kind, getattr(self, kind) + 1)

    def as_dict(self) -> dict:
        return {
            "total": self.total,
            "calibration": self.calibration,
            "iteration": self.iteration,
            "candidate": self.candidate,
            "failure_score": self.failure_score,
            "monitoring": 0,
            "continuation": 0,
        }


def iterations_for_cap(cap: int, starts: int = 4) -> int:
    if cap % starts != 0:
        raise ValueError(f"cap {cap} is not divisible by {starts} starts")
    per = cap // starts
    if per < 10 or (per - 10) % 2 != 0:
        raise ValueError(f"per-start budget {per} does not fit 8 calibration + 2 candidates + even iterations")
    return (per - 10) // 2


def _sigma_eff(x: np.ndarray, sigmas: np.ndarray, alpha_index: np.ndarray) -> np.ndarray:
    eff = np.array(sigmas, dtype=np.float64, copy=True)
    for i in alpha_index:
        i = int(i)
        sech2 = 1.0 - float(np.tanh(x[i])) ** 2
        if sech2 < 1e-4:
            eff[i] = 0.0
        else:
            eff[i] = float(sigmas[i]) / (ALPHA_MAX * sech2)
    return eff


def _rademacher(rng: np.random.Generator, n: int) -> np.ndarray:
    return rng.integers(0, 2, size=n).astype(np.float64) * 2.0 - 1.0


def _grad(ep: float, em: float, c_k: float, sig: np.ndarray, delta: np.ndarray) -> np.ndarray:
    g = np.zeros_like(sig)
    scale = ep - em
    for j in range(sig.shape[0]):
        denom = 2.0 * c_k * sig[j] * delta[j]
        if sig[j] == 0.0 or denom == 0.0:
            g[j] = 0.0
        else:
            g[j] = scale / denom
    return g


@dataclass
class SPSAResult:
    x: np.ndarray
    energy: float
    payload: dict
    counter: QueryCounter
    start_id: int
    candidate: str
    n_iter: int
    a_values: list = field(default_factory=list)
    failed: bool = False
    failure: str = ""


def run_spsa(
    evaluate,
    x0_list: list[np.ndarray],
    sigmas: np.ndarray,
    cap: int,
    rngs: list[np.random.Generator],
    *,
    c: float = 0.1,
    A: float = 25.0,
    alpha_exp: float = 0.602,
    gamma_exp: float = 0.101,
    alpha_index: np.ndarray | None = None,
    polyak_window: int = 64,
) -> SPSAResult:
    """evaluate(x) -> dict with finite E and norm. Selection uses E only."""
    starts = len(x0_list)
    if len(rngs) != starts:
        raise ValueError("one rng per start")
    n_iter = iterations_for_cap(cap, starts)
    counter = QueryCounter()
    alpha_index = np.zeros(0, dtype=np.int64) if alpha_index is None else np.asarray(alpha_index, dtype=np.int64)
    best = None
    a_values = []
    try:
        for start_id, (x0, rng) in enumerate(zip(x0_list, rngs)):
            x = np.array(x0, dtype=np.float64, copy=True)
            sig0 = _sigma_eff(x, sigmas, alpha_index)
            grads = []
            for _ in range(4):
                delta = _rademacher(rng, x.size)
                step = c * sig0 * delta
                ep = _call(evaluate, x + step, counter, "calibration")
                em = _call(evaluate, x - step, counter, "calibration")
                grads.append(_grad(ep, em, c, sig0, delta))
            ratios = []
            for g in grads:
                for j in range(x.size):
                    if sig0[j] > 0.0:
                        ratios.append(abs(g[j]) / sig0[j])
            med = float(np.median(ratios)) if ratios else 0.0
            a0 = 0.0 if med < 1e-15 else 0.5 / med
            a = a0 * (1.0 + A) ** alpha_exp
            a_values.append(float(a))
            history = []
            for k in range(n_iter):
                ck = c / (k + 1) ** gamma_exp
                ak = a / (k + 1 + A) ** alpha_exp
                sig = _sigma_eff(x, sigmas, alpha_index)
                delta = _rademacher(rng, x.size)
                step = ck * sig * delta
                ep = _call(evaluate, x + step, counter, "iteration")
                em = _call(evaluate, x - step, counter, "iteration")
                g = _grad(ep, em, ck, sig, delta)
                update = ak * g
                for j in range(x.size):
                    cap_j = 2.0 * sig[j]
                    if update[j] > cap_j:
                        update[j] = cap_j
                    elif update[j] < -cap_j:
                        update[j] = -cap_j
                x = x - update
                history.append(x.copy())
            x_last = history[-1]
            window = history[-min(polyak_window, len(history)) :]
            x_poly = np.mean(np.stack(window, axis=0), axis=0)
            out_last = _call_full(evaluate, x_last, counter, "candidate")
            out_poly = _call_full(evaluate, x_poly, counter, "candidate")
            if out_poly["E"] < out_last["E"]:
                chosen_x, chosen_out, which = x_poly, out_poly, "polyak"
            else:
                chosen_x, chosen_out, which = x_last, out_last, "last"
            if best is None or chosen_out["E"] < best.energy:
                best = SPSAResult(
                    x=np.array(chosen_x, dtype=np.float64, copy=True),
                    energy=float(chosen_out["E"]),
                    payload=chosen_out,
                    counter=counter,
                    start_id=start_id,
                    candidate=which,
                    n_iter=n_iter,
                    a_values=list(a_values),
                    failed=False,
                )
        assert best is not None
        if counter.total != cap:
            raise OptimizationFailure(f"query counter {counter.total} != cap {cap}")
        best.a_values = list(a_values)
        return best
    except OptimizationFailure as exc:
        fallback = evaluate(np.array(x0_list[0], dtype=np.float64, copy=True))
        counter.add("failure_score")
        return SPSAResult(
            x=np.array(x0_list[0], dtype=np.float64, copy=True),
            energy=float(fallback["E"]),
            payload=fallback,
            counter=counter,
            start_id=0,
            candidate="failure_x0",
            n_iter=n_iter,
            a_values=a_values,
            failed=True,
            failure=str(exc),
        )


def _call(evaluate, x, counter: QueryCounter, kind: str) -> float:
    out = _call_full(evaluate, x, counter, kind)
    return float(out["E"])


def _call_full(evaluate, x, counter: QueryCounter, kind: str) -> dict:
    counter.add(kind)
    out = evaluate(np.array(x, dtype=np.float64, copy=True))
    E = out.get("E", None)
    norm = out.get("norm", None)
    if E is None or not np.isfinite(E):
        raise OptimizationFailure("non-finite energy")
    if norm is None or not np.isfinite(norm) or abs(float(norm) - 1.0) > 1e-9:
        raise OptimizationFailure(f"norm drift {norm}")
    return out
