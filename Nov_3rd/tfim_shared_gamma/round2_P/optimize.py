"""Pagano-style continuation. Gamma and theta stay shared except where the
method is explicitly an independent-gamma diagnostic.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize

import circuit

METHOD_ID = {
    "qaoa": 1, "A": 2, "P": 3, "Sblock": 4, "Pshuffle": 5,
    "P_thetaK0": 6, "qaoa_sameG": 7, "P_indgamma": 8,
}


def unwrap(values: np.ndarray, period: Optional[float]) -> np.ndarray:
    out = np.asarray(values, dtype=np.float64).copy()
    if period is None or len(out) < 2:
        return out
    for i in range(1, len(out)):
        k = np.round((out[i - 1] - out[i]) / period)
        out[i] = out[i] + k * period
    return out


def resample(values, L_new: int, kind: str, period: Optional[float] = None) -> np.ndarray:
    y = unwrap(np.asarray(values, dtype=np.float64), period)
    if L_new <= 1:
        return np.array([float(y[0])])
    if len(y) == 1:
        return np.full(L_new, float(y[0]))
    s_old = np.linspace(0.0, 1.0, len(y))
    s_new = np.linspace(0.0, 1.0, L_new)
    if kind == "linear" or len(y) < 3:
        return np.interp(s_new, s_old, y)
    return np.asarray(CubicSpline(s_old, y, bc_type="natural")(s_new), dtype=np.float64)


def _clip(x, bounds):
    y = np.array(x, dtype=np.float64, copy=True)
    for i, (lo, hi) in enumerate(bounds):
        y[i] = min(max(y[i], lo), hi)
    return y


def continuation(method: str, previous_x, L_prev: int, L: int, B: float, periods: Dict[str, Optional[float]]):
    prev = circuit.unpack(method, previous_x, L_prev)
    kind = "linear" if L == 3 else "cubic"
    if method == "qaoa":
        if L == 2 and L_prev == 1:
            g1, b1 = float(prev["gammas"][0]), float(prev["betas"][0])
            return np.array([g1, b1, g1 + 0.2, b1 - 0.2 * float(B)]), "qaoa L2 shift beta2=beta1-0.2*B, gamma2=gamma1+0.2"
        g = resample(prev["gammas"], L, kind, None)
        b = resample(prev["betas"], L, kind, periods.get("beta"))
        x = np.empty(2 * L)
        x[0::2] = g
        x[1::2] = b
        return x, f"qaoa {kind}"
    if method == "qaoa_sameG":
        if L == 2 and L_prev == 1:
            g1, b1 = float(prev["gammas"][0]), float(prev["betas"][0])
            return np.array([prev["theta_k"], prev["theta_p"], g1, b1, g1 + 0.2, b1 - 0.2 * float(B)]), "sameG L2 shift, thetas copied"
        g = resample(prev["gammas"], L, kind, None)
        b = resample(prev["betas"], L, kind, periods.get("beta"))
        x = np.empty(2 + 2 * L)
        x[0] = prev["theta_k"]
        x[1] = prev["theta_p"]
        x[2::2] = g
        x[3::2] = b
        return x, f"sameG {kind}, thetas copied"
    if method == "P_indgamma":
        if L == 2 and L_prev == 1:
            return np.array([
                prev["theta_k"], prev["theta_p"],
                float(prev["gammas"][0]), float(prev["c1"][0]), float(prev["c2"][0]), float(prev["c3"][0]),
                float(prev["gammas"][0]) + 0.2, float(prev["c1"][0]), float(prev["c2"][0]), float(prev["c3"][0]),
            ]), "indgamma L2, gamma split, shells duplicated, thetas copied"
        g = resample(prev["gammas"], L, kind, None)
        c1 = resample(prev["c1"], L, kind, periods.get("c1"))
        c2 = resample(prev["c2"], L, kind, periods.get("c2"))
        c3 = resample(prev["c3"], L, kind, periods.get("c3"))
        x = np.empty(2 + 4 * L)
        x[0] = prev["theta_k"]
        x[1] = prev["theta_p"]
        block = np.column_stack([g, c1, c2, c3])
        x[2:] = block.reshape(-1)
        return x, f"indgamma {kind} diagnostic, thetas copied"
    # shared-gamma shells
    if L == 2 and L_prev == 1:
        shift = -0.2 * float(B)
        c1 = np.array([float(prev["c1"][0]), float(prev["c1"][0]) + shift])
        c2 = np.repeat(float(prev["c2"][0]), 2)
        c3 = np.repeat(float(prev["c3"][0]), 2)
        rule = "duplicate shells; c1_layer2 += -0.2*B; gamma and theta copied"
    else:
        c1 = resample(prev["c1"], L, kind, periods.get("c1"))
        c2 = resample(prev["c2"], L, kind, periods.get("c2"))
        c3 = resample(prev["c3"], L, kind, periods.get("c3"))
        rule = f"shell {kind}; gamma and theta copied"
    shells = np.column_stack([c1, c2, c3]).reshape(-1)
    if method == "A":
        x = np.concatenate([[prev["gamma"], prev["theta_k"]], shells])
    elif method == "P_thetaK0":
        x = np.concatenate([[prev["gamma"], prev["theta_p"]], shells])
    else:
        x = np.concatenate([[prev["gamma"], prev["theta_k"], prev["theta_p"]], shells])
    return np.asarray(x, dtype=np.float64), rule


def build_starts(method, n, L, B, previous_x, n_starts, energy, periods, nested=None) -> List[Dict]:
    rng = np.random.default_rng(41011 + METHOD_ID[method] * 1000 + n * 17 + L)
    bounds = circuit.bounds_for(method, L)
    pdim = circuit.n_params(method, L)
    starts: List[Dict] = []

    def add(x, name):
        starts.append({"name": name, "x0": _clip(x, bounds)})

    if L == 1 and previous_x is None:
        if method == "qaoa":
            ranked = []
            for g in np.linspace(-np.pi, np.pi, 21):
                for b in np.linspace(-np.pi, np.pi, 21):
                    x = np.array([g, b])
                    ranked.append((energy(x), x))
            ranked.sort(key=lambda t: t[0])
            for i, (_, x) in enumerate(ranked[:8]):
                add(x, f"grid{i}")
        elif method == "qaoa_sameG":
            ranked = []
            for g in np.linspace(-1.2, 1.2, 9):
                for b in np.linspace(-1.0, 1.0, 9):
                    x = np.zeros(pdim)
                    x[2] = g
                    x[3] = b
                    ranked.append((energy(x), x.copy()))
            ranked.sort(key=lambda t: t[0])
            for i, (_, x) in enumerate(ranked[:4]):
                add(x, f"grid{i}")
            x = np.zeros(pdim)
            x[0] = -0.5
            x[1] = 0.3
            x[2] = 0.3
            x[3] = 0.3
            add(x, "theta_seed")
        elif method == "P_indgamma":
            x = np.zeros(pdim)
            x[2] = 0.3
            x[3] = 0.3
            add(x, "small")
        else:
            ranked = []
            for g in np.linspace(-1.2, 1.2, 9):
                for c1 in np.linspace(-1.0, 1.0, 9):
                    x = np.zeros(pdim)
                    x[0] = g
                    # c1 index
                    if method in ("A", "P_thetaK0"):
                        x[2] = c1
                    else:
                        x[3] = c1
                    ranked.append((energy(x), x.copy()))
            ranked.sort(key=lambda t: t[0])
            for i, (_, x) in enumerate(ranked[:4]):
                add(x, f"gc{i}")
            x = np.zeros(pdim)
            x[0] = 0.35
            if method == "A":
                x[1] = -0.5
                x[2] = 0.3
            elif method == "P_thetaK0":
                x[1] = 0.4
                x[2] = 0.3
            else:
                x[1] = -0.5
                x[2] = 0.3
                x[3] = 0.3
            add(x, "structured")
    else:
        if previous_x is None:
            raise RuntimeError(f"{method} L={L} missing previous depth")
        x_boot, rule = continuation(method, previous_x, L - 1, L, B, periods)
        add(x_boot, "bootstrap:" + rule)
        x_jit = np.array(x_boot, copy=True)
        x_jit[0] = x_jit[0] + 0.12
        add(x_jit, "bootstrap_shift0")
        for s in (0.05, 0.12, 0.25):
            add(np.asarray(x_boot, float) + rng.normal(0.0, s, size=pdim), f"perturb{s}")
    while len(starts) < n_starts:
        x = rng.normal(0.0, 0.35, size=pdim)
        add(x, f"random{len(starts)}")
    # Nested inits replace the last random slots. Count stays n_starts.
    slots = [i for i, s in enumerate(starts) if str(s["name"]).startswith("random")]
    for slot, (name, xnest) in zip(reversed(slots), nested or []):
        add_x = _clip(xnest, bounds)
        starts[slot] = {"name": name, "x0": add_x}
    return starts[:n_starts]


def run_multistart(objective, starts, bounds, maxiter, maxfun, ftol, gtol) -> Dict:
    rows = []
    best = None
    for spec in starts:
        trace: List[float] = []

        def fun(x, _trace=trace):
            E, g = objective(np.asarray(x, dtype=np.float64))
            _trace.append(float(E))
            return float(E), np.asarray(g, dtype=np.float64)

        res = minimize(
            fun, np.asarray(spec["x0"], dtype=np.float64), method="L-BFGS-B", jac=True, bounds=bounds,
            options={"maxiter": int(maxiter), "ftol": float(ftol), "gtol": float(gtol), "maxfun": int(maxfun)},
        )
        xfin = np.asarray(res.x, dtype=np.float64)
        hit = [i for i, ((lo, hi), v) in enumerate(zip(bounds, xfin)) if abs(v - lo) <= 1e-5 or abs(v - hi) <= 1e-5]
        row = {
            "name": spec["name"],
            "E": float(res.fun),
            "x": [float(v) for v in xfin],
            "x0": [float(v) for v in spec["x0"]],
            "nit": int(res.nit),
            "nfev": int(res.nfev),
            "success": bool(res.success),
            "message": str(res.message),
            "bound_hit_indices": hit,
            "trace": trace[:: max(1, len(trace) // 120)][:120],
        }
        rows.append(row)
        if best is None or row["E"] < best["E"]:
            best = row
    energies = np.array([r["E"] for r in rows])
    return {
        "best": best,
        "starts": rows,
        "E_best": float(energies.min()),
        "E_median": float(np.median(energies)),
        "E_worst": float(energies.max()),
        "nfev_sum": int(sum(r["nfev"] for r in rows)),
        "n_starts": len(rows),
    }


def fourier_schedule(values: Sequence[float], n_modes: int = 2) -> np.ndarray:
    y = np.asarray(values, dtype=np.float64)
    L = len(y)
    if L == 1:
        return y.copy()
    cols = []
    for k in range(1, n_modes + 1):
        cols.append([np.sin((k - 0.5) * (i + 0.5) * np.pi / L) for i in range(L)])
    A = np.column_stack(cols)
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    return A @ coef
