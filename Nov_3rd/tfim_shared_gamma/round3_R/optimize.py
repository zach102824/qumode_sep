"""Ascending-depth seeds. Linear and cubic shell seeds are both kept at L>=3."""

from __future__ import annotations

from typing import Optional

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize

import circuit


def unwrap(values, period: Optional[float]):
    out = np.asarray(values, dtype=np.float64).copy()
    if period is None or len(out) < 2:
        return out
    for i in range(1, len(out)):
        k = np.round((out[i - 1] - out[i]) / period)
        out[i] += k * period
    return out


def resample(values, L_new, kind, period=None):
    y = unwrap(values, period)
    if len(y) == 1 or L_new == 1:
        return np.full(L_new, float(y[0]))
    s_old = np.linspace(0, 1, len(y))
    s_new = np.linspace(0, 1, L_new)
    if kind == "linear" or len(y) < 3:
        return np.interp(s_new, s_old, y)
    return np.asarray(CubicSpline(s_old, y, bc_type="natural")(s_new), dtype=np.float64)


def _shells_from(prev, L, kind, B, periods, shift=False):
    if shift:
        c1 = np.array([float(prev["c1"][0]), float(prev["c1"][0]) - 0.2 * B])
        c2 = np.repeat(float(prev["c2"][0]), 2)
        c3 = np.repeat(float(prev["c3"][0]), 2)
    else:
        c1 = resample(prev["c1"], L, kind, periods.get("c1"))
        c2 = resample(prev["c2"], L, kind, periods.get("c2"))
        c3 = resample(prev["c3"], L, kind, periods.get("c3"))
    return np.column_stack([c1, c2, c3]).reshape(-1)


def continuation(kind, previous, L_prev, L, B, periods, how="auto"):
    prev = circuit.unpack(kind, previous, L_prev)
    if kind == "qaoa":
        if L == 2:
            g, b = float(prev["gammas"][0]), float(prev["betas"][0])
            return np.array([g, b, g + 0.2, b - 0.2 * B])
        kind_i = "linear" if (how == "auto" and L == 3) or how == "linear" else "cubic"
        if how == "linear":
            kind_i = "linear"
        if how == "cubic":
            kind_i = "cubic"
        g = resample(prev["gammas"], L, kind_i, None)
        b = resample(prev["betas"], L, kind_i, periods.get("beta"))
        x = np.empty(2 * L)
        x[0::2] = g
        x[1::2] = b
        return x
    if kind == "sameG":
        if L == 2:
            g, b = float(prev["gammas"][0]), float(prev["betas"][0])
            return np.array([prev["theta_k"], g, b, g + 0.2, b - 0.2 * B])
        kind_i = "cubic" if how == "cubic" or (how == "auto" and L > 3) else "linear"
        g = resample(prev["gammas"], L, kind_i, None)
        b = resample(prev["betas"], L, kind_i, periods.get("beta"))
        x = np.empty(1 + 2 * L)
        x[0] = prev["theta_k"]
        x[1::2] = g
        x[2::2] = b
        return x
    kind_i = "linear" if L == 2 or how == "linear" or (how == "auto" and L == 3) else "cubic"
    if how == "cubic":
        kind_i = "cubic"
    if how == "linear":
        kind_i = "linear"
    shells = _shells_from(prev, L, kind_i, B, periods, shift=(L == 2 and how == "auto"))
    if kind == "k0":
        return np.concatenate([[prev["gamma"]], shells])
    if kind == "A":
        return np.concatenate([[prev["gamma"], prev["theta_k"]], shells])
    return np.concatenate([[prev["gamma"], prev["theta_k"], prev["theta_m"]], shells])


def fourier_schedule(values, n_modes=2):
    y = np.asarray(values, dtype=np.float64)
    L = len(y)
    if L < 2:
        return y.copy()
    cols = [[np.sin((k - 0.5) * (i + 0.5) * np.pi / L) for i in range(L)] for k in range(1, n_modes + 1)]
    A = np.column_stack(cols)
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return A @ coef


def clip(x, bounds):
    y = np.array(x, dtype=np.float64, copy=True)
    for i, (lo, hi) in enumerate(bounds):
        y[i] = min(max(y[i], lo), hi)
    return y


def run_starts(objective, starts, bounds, maxiter, maxfun):
    rows = []
    best = None
    for spec in starts:
        trace = []

        def fun(x, _trace=trace):
            E, g = objective(np.asarray(x, dtype=np.float64))
            _trace.append(float(E))
            return float(E), np.asarray(g, dtype=np.float64)

        res = minimize(
            fun, np.asarray(spec["x0"], dtype=np.float64), method="L-BFGS-B", jac=True, bounds=bounds,
            options={"maxiter": int(maxiter), "maxfun": int(maxfun), "ftol": 1e-14, "gtol": 1e-8},
        )
        row = {
            "name": spec["name"],
            "E": float(res.fun),
            "x": [float(v) for v in res.x],
            "nit": int(res.nit),
            "nfev": int(res.nfev),
            "success": bool(res.success),
            "trace": trace[:: max(1, len(trace) // 80)][:80],
        }
        rows.append(row)
        if best is None or row["E"] < best["E"]:
            best = row
    energies = np.array([r["E"] for r in rows], dtype=np.float64)
    q1, q3 = np.percentile(energies, [25, 75])
    return {
        "best": best,
        "starts": rows,
        "E_median": float(np.median(energies)),
        "E_iqr": float(q3 - q1),
        "nfev_sum": int(sum(r["nfev"] for r in rows)),
        "n_starts": len(rows),
    }
