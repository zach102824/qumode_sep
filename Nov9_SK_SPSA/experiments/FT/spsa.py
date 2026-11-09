"""SPSA only. The objective's returned energy is the sole selection signal.

Fidelity may be present in the objective payload for logging. It is not read
by the step, the candidate choice, or the multi-start choice.
"""
from __future__ import annotations

import numpy as np

from Nov9_SK_SPSA.experiments.FT.params import (
    A_SPALL,
    ALPHA_EXP,
    C_BASE,
    CLIP_OVER_SIGMA,
    GAMMA_EXP,
    TARGET_STEP_OVER_SIGMA,
    Spec,
    physical,
    shift_physical,
)


def _rademacher(rng: np.random.Generator, n: int) -> np.ndarray:
    return rng.choice(np.array([-1.0, 1.0]), size=n)


def _grad(Ep: float, Em: float, x, xp, xm, spec: Spec) -> np.ndarray:
    dp = physical(xp, spec) - physical(x, spec)
    dm = physical(x, spec) - physical(xm, spec)
    denom = dp + dm
    diff = float(Ep) - float(Em)
    g = np.zeros(spec.n, dtype=np.float64)
    for i in range(spec.n):
        if abs(float(denom[i])) < 1e-14:
            g[i] = 0.0
        else:
            g[i] = diff / float(denom[i])
    return g


def spsa_start(
    objective,
    x0: np.ndarray,
    spec: Spec,
    rng: np.random.Generator,
    n_iter: int,
    make_context=None,
    c: float = C_BASE,
) -> dict:
    """One start. objective(x, ctx) -> dict with E, ok, n_queries, and optional F.

    make_context(kind) is called once per two-sided pair and once for the
    shared candidate context, so both sides see the same minibatch.
    """
    x = np.asarray(x0, dtype=np.float64).copy()
    queries = 0
    trace_E = []
    trace_F = []

    def ev(x_eval, ctx):
        nonlocal queries
        out = objective(x_eval, ctx)
        queries += int(out["n_queries"])
        trace_E.append(float(out["E"]))
        if "F" in out and out["F"] is not None:
            trace_F.append(float(out["F"]))
        return out

    def ctx_for(kind):
        if make_context is None:
            return None
        return make_context(kind)

    grads = []
    for _ in range(4):
        ctx = ctx_for("cal")
        delta = _rademacher(rng, spec.n)
        dphys = c * spec.sigma * delta
        xp = shift_physical(x, dphys, spec)
        xm = shift_physical(x, -dphys, spec)
        op = ev(xp, ctx)
        om = ev(xm, ctx)
        if not op["ok"] or not om["ok"]:
            return _fail(x0, queries, trace_E, trace_F, "calibration")
        grads.append(_grad(op["E"], om["E"], x, xp, xm, spec))

    stacked = np.abs(np.vstack(grads)) / spec.sigma
    med = float(np.median(stacked))
    if not np.isfinite(med) or med < 1e-18:
        a_coeff = 0.0
    else:
        a0 = TARGET_STEP_OVER_SIGMA / med
        a_coeff = a0 * (1.0 + A_SPALL) ** ALPHA_EXP

    xs = []
    for k in range(int(n_iter)):
        a_k = a_coeff / (k + 1.0 + A_SPALL) ** ALPHA_EXP
        c_k = c / (k + 1.0) ** GAMMA_EXP
        ctx = ctx_for("iter")
        delta = _rademacher(rng, spec.n)
        dphys = c_k * spec.sigma * delta
        xp = shift_physical(x, dphys, spec)
        xm = shift_physical(x, -dphys, spec)
        op = ev(xp, ctx)
        om = ev(xm, ctx)
        if not op["ok"] or not om["ok"]:
            return _fail(x0, queries, trace_E, trace_F, "iteration")
        g = _grad(op["E"], om["E"], x, xp, xm, spec)
        step = np.clip(a_k * g, -CLIP_OVER_SIGMA * spec.sigma, CLIP_OVER_SIGMA * spec.sigma)
        x = shift_physical(x, -step, spec)
        xs.append(x.copy())

    x_last = x.copy()
    x_poly = x_last.copy() if not xs else np.mean(np.stack(xs[-64:], axis=0), axis=0)
    ctx_c = ctx_for("cand")
    ol = ev(x_last, ctx_c)
    opk = ev(x_poly, ctx_c)
    if not ol["ok"] or not opk["ok"]:
        return _fail(x0, queries, trace_E, trace_F, "candidate")
    if float(opk["E"]) < float(ol["E"]):
        chosen, cout, which = x_poly, opk, "polyak"
    else:
        chosen, cout, which = x_last, ol, "last"
    return {
        "failed": False,
        "failure_stage": None,
        "x": chosen,
        "x0": np.asarray(x0, dtype=np.float64).copy(),
        "which": which,
        "E": float(cout["E"]),
        "payload": cout,
        "queries": queries,
        "a_coeff": float(a_coeff),
        "trace_E": trace_E,
        "trace_F_oracle_not_used": trace_F,
        "n_iter": int(n_iter),
    }


def _fail(x0, queries, trace_E, trace_F, stage) -> dict:
    return {
        "failed": True,
        "failure_stage": stage,
        "x": np.asarray(x0, dtype=np.float64).copy(),
        "x0": np.asarray(x0, dtype=np.float64).copy(),
        "which": "x0_after_failure",
        "E": None,
        "payload": None,
        "queries": queries,
        "a_coeff": None,
        "trace_E": trace_E,
        "trace_F_oracle_not_used": trace_F,
        "n_iter": None,
    }
