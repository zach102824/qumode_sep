"""Energy-only multi-start L-BFGS with Pagano-style depth continuation.

QAOA keeps independent gamma_l and beta_l. Every Nov_3rd method keeps one gamma
and one theta vector for the whole depth. Shell schedules are interpolated;
shared scalars are copied and reoptimized, never promoted to per-layer values.

The L=2 QAOA shift follows Pagano et al. after converting the mixer. Their
mixer is exp(-i beta_paper B sum Y / J0). Ours is exp(-i beta sum X), and J0=1,
so beta = beta_paper * B and the published (beta, beta-0.2) seed becomes
    beta_2 = beta_1 - 0.2 * B.
Their gamma multiplies Hc/J0 with the same J0=1 normalization used here, so
the gamma shift stays +0.2. Shared-gamma ansätze cannot split gamma; only the
shell controls are shifted at L=2.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize

from circuit import bounds_for, n_params, unpack

METHOD_ID = {
    "qaoa": 1,
    "shared_linear": 2,
    "D": 3,
    "A": 4,
    "C": 5,
    "C_unrestricted": 6,
}


def unwrap_series(values: np.ndarray, period: Optional[float]) -> np.ndarray:
    out = np.asarray(values, dtype=np.float64).copy()
    if period is None or len(out) < 2:
        return out
    for i in range(1, len(out)):
        k = np.round((out[i - 1] - out[i]) / period)
        out[i] = out[i] + k * period
    return out


def resample_schedule(values: Sequence[float], L_new: int, kind: str, period: Optional[float] = None) -> np.ndarray:
    y = unwrap_series(np.asarray(values, dtype=np.float64), period)
    if L_new <= 1:
        return np.array([float(y[0])], dtype=np.float64)
    if len(y) == 1:
        return np.full(L_new, float(y[0]), dtype=np.float64)
    s_old = np.linspace(0.0, 1.0, len(y))
    s_new = np.linspace(0.0, 1.0, L_new)
    if kind == "linear" or len(y) < 3:
        return np.interp(s_new, s_old, y).astype(np.float64)
    spline = CubicSpline(s_old, y, bc_type="natural")
    return np.asarray(spline(s_new), dtype=np.float64)


def _clip(x: np.ndarray, bounds: Sequence[Tuple[float, float]]) -> np.ndarray:
    y = np.array(x, dtype=np.float64, copy=True)
    for i, (lo, hi) in enumerate(bounds):
        y[i] = min(max(y[i], lo), hi)
    return y


def _rng(method: str, n: int, L: int) -> np.random.Generator:
    return np.random.default_rng(10007 + METHOD_ID[method] * 1000 + int(n) * 17 + int(L))


def continuation_seed(
    method: str,
    previous_x: np.ndarray,
    L_prev: int,
    L: int,
    B: float,
    periods: Dict[str, Optional[float]],
) -> Tuple[np.ndarray, str]:
    """Primary bootstrap seed at depth L from the best energy at L-1.

    Returns the parameter vector and a short description of the rule.
    """
    prev = unpack(method, previous_x, L_prev)
    kind = "linear" if L == 3 else "cubic"
    if method == "qaoa":
        if L == 2 and L_prev == 1:
            g1 = float(prev["gammas"][0])
            b1 = float(prev["betas"][0])
            x = np.array([g1, b1, g1 + 0.2, b1 - 0.2 * float(B)], dtype=np.float64)
            return x, "qaoa L2 Pagano shift, mixer converted by beta_2=beta_1-0.2*B, gamma_2=gamma_1+0.2"
        g = resample_schedule(prev["gammas"], L, kind, period=None)
        b = resample_schedule(prev["betas"], L, kind, period=periods.get("beta"))
        x = np.empty(2 * L, dtype=np.float64)
        x[0::2] = g
        x[1::2] = b
        return x, f"qaoa schedule {kind} interpolation; gamma not unwrapped; beta period={periods.get('beta')}"
    # Shared scalars are copied. Shells are duplicated at L=2 and interpolated later.
    c1 = prev["c1"]
    c2 = prev["c2"]
    c3 = prev["c3"]
    if L == 2 and L_prev == 1:
        shift = -0.2 * float(B)
        c1n = np.array([float(c1[0]), float(c1[0]) + shift])
        c2n = np.array([float(c2[0]), float(c2[0])])
        c3n = np.array([float(c3[0]), float(c3[0])])
        rule = "duplicate shells; c1_layer2 += -0.2*B (converted mixer shift). gamma stays one scalar"
    else:
        c1n = resample_schedule(c1, L, kind, periods.get("c1"))
        c2n = resample_schedule(c2, L, kind, periods.get("c2"))
        c3n = resample_schedule(c3, L, kind, periods.get("c3"))
        rule = f"shell {kind} interpolation with verified gauge periods; gamma/theta copied as scalars"
    if method == "shared_linear":
        x = np.concatenate([[prev["gamma"]], c1n])
    elif method == "D":
        shells = np.column_stack([c1n, c2n, c3n]).reshape(-1)
        x = np.concatenate([[prev["gamma"]], shells])
    elif method == "A":
        shells = np.column_stack([c1n, c2n, c3n]).reshape(-1)
        x = np.concatenate([[prev["gamma"], prev["theta_k"]], shells])
    else:
        shells = np.column_stack([c1n, c2n, c3n]).reshape(-1)
        x = np.concatenate([[prev["gamma"], prev["theta_k"], prev["theta_c"]], shells])
    return np.asarray(x, dtype=np.float64), rule


def build_starts(
    method: str,
    n: int,
    L: int,
    B: float,
    previous_x: Optional[np.ndarray],
    n_starts: int,
    objective_energy: Callable[[np.ndarray], float],
    periods: Dict[str, Optional[float]],
) -> List[Dict]:
    """Exactly n_starts initial vectors. Grid evaluations are initialization only."""
    rng = _rng(method, n, L)
    bounds = bounds_for(method, L)
    pdim = n_params(method, L)
    starts: List[Dict] = []

    def add(x, name):
        starts.append({"name": name, "x0": _clip(np.asarray(x, dtype=np.float64), bounds)})

    if L == 1:
        gammas = np.linspace(-np.pi, np.pi, 25)
        mix = np.linspace(-np.pi, np.pi, 25)
        if method == "qaoa":
            best = []
            for g in gammas:
                for b in mix:
                    x = np.array([g, b], dtype=np.float64)
                    best.append((objective_energy(x), x))
            best.sort(key=lambda t: t[0])
            for i, (_, x) in enumerate(best[:8]):
                add(x, f"grid_top{i}")
        else:
            g_vals = np.linspace(-1.5, 1.5, 13)
            c_vals = np.linspace(-1.2, 1.2, 13)
            ranked = []
            for g in g_vals:
                for c1 in c_vals:
                    x = np.zeros(pdim, dtype=np.float64)
                    x[0] = g
                    # c1 sits at index 1 for every non-qaoa packing (shared_linear, D, and after thetas it is NOT index 1).
                    if method == "shared_linear":
                        x[1] = c1
                    elif method == "D":
                        x[1] = c1
                    elif method == "A":
                        x[2] = c1
                    else:
                        x[3] = c1
                    ranked.append((objective_energy(x), x.copy()))
            ranked.sort(key=lambda t: t[0])
            for i, (_, x) in enumerate(ranked[:4]):
                add(x, f"gc1_top{i}")
            if method in ("A", "C", "C_unrestricted"):
                ths = np.linspace(-1.5, 1.5, 7)
                local = []
                for _, base in ranked[:3]:
                    for th in ths:
                        x = base.copy()
                        x[1] = th
                        local.append((objective_energy(x), x.copy()))
                local.sort(key=lambda t: t[0])
                for i, (_, x) in enumerate(local[:2]):
                    add(x, f"thetaK_scan{i}")
            if method in ("C", "C_unrestricted"):
                ths = np.linspace(-1.0, 1.0, 5)
                local = []
                for _, base in ranked[:2]:
                    for thk in ths:
                        for thc in ths:
                            x = base.copy()
                            x[1] = thk
                            x[2] = thc
                            local.append((objective_energy(x), x.copy()))
                local.sort(key=lambda t: t[0])
                add(local[0][1], "thetaKC_scan0")
            # Structured shells. These are initial guesses, not physical claims.
            def shell_vec(c1, c2, c3, thk=0.0, thc=0.0):
                if method == "shared_linear":
                    return np.array([0.35, c1])
                if method == "D":
                    return np.array([0.35, c1, c2, c3])
                if method == "A":
                    return np.array([0.35, thk, c1, c2, c3])
                return np.array([0.35, thk, thc, c1, c2, c3])
            add(shell_vec(0.25, 0.0, 0.45), "c3_heavy")
            add(shell_vec(0.25, 0.35, 0.0), "c2_heavy")
            if method in ("A", "C", "C_unrestricted"):
                add(shell_vec(0.3, 0.1, 0.2, thk=-0.5), "neg_thetaK")
            if method in ("C", "C_unrestricted"):
                add(shell_vec(0.3, 0.1, 0.2, thk=-0.3, thc=0.4), "star_positive")
    else:
        if previous_x is None:
            raise RuntimeError("depth > 1 requires a previous energy optimum")
        x_boot, rule = continuation_seed(method, previous_x, L - 1, L, B, periods)
        add(x_boot, "bootstrap:" + rule)
        # Opposite mixer shift, still one shared gamma for non-QAOA.
        x_alt, _ = continuation_seed(method, previous_x, L - 1, L, B, periods)
        if method == "qaoa" and L == 2:
            g1 = float(previous_x[0])
            b1 = float(previous_x[1])
            add(np.array([g1, b1, g1 - 0.2, b1 + 0.2 * float(B)]), "qaoa_opposite_shift")
            add(np.array([g1, b1, g1, b1]), "qaoa_duplicate")
        elif L == 2:
            # A pure duplicate, without the c1 shift, as the second structured seed.
            x_dup = np.array(x_boot, copy=True)
            if method == "shared_linear":
                x_dup[2] = x_dup[1]
            elif method == "D":
                x_dup[4:7] = x_dup[1:4]
            elif method == "A":
                x_dup[5:8] = x_dup[2:5]
            else:
                x_dup[6:9] = x_dup[3:6]
            add(x_dup, "duplicate_shells_no_shift")
        else:
            # Not a second copy of the same seed. Shift the shared cost angle, or
            # for QAOA the first-layer gamma, and keep the interpolated shells.
            x_jit = np.array(x_boot, copy=True)
            x_jit[0] = float(x_jit[0]) + 0.15
            add(x_jit, "bootstrap_gamma0_plus_0.15")
        scales = (0.04, 0.08, 0.15, 0.30)
        for s in scales:
            noise = rng.normal(0.0, s, size=pdim)
            add(np.asarray(x_boot, dtype=np.float64) + noise, f"perturb_std{s}")

    # Fill with modest random starts. Not uniform on the wide bounds.
    while len(starts) < n_starts:
        x = np.zeros(pdim, dtype=np.float64)
        if method == "qaoa":
            x[0::2] = rng.uniform(-1.2, 1.2, size=L)
            x[1::2] = rng.uniform(-1.0, 1.0, size=L)
        else:
            x[0] = rng.uniform(-1.2, 1.2)
            cursor = 1
            if method in ("A", "C", "C_unrestricted"):
                x[cursor] = rng.uniform(-1.0, 1.0)
                cursor += 1
            if method in ("C", "C_unrestricted"):
                x[cursor] = rng.uniform(-1.0, 1.0)
                cursor += 1
            x[cursor:] = rng.uniform(-0.8, 0.8, size=pdim - cursor)
        add(x, f"random{len(starts)}")
    return starts[:n_starts]


def run_multistart(
    objective: Callable[[np.ndarray], Tuple[float, np.ndarray]],
    starts: List[Dict],
    bounds: Sequence[Tuple[float, float]],
    maxiter: int,
    maxfun: int,
    ftol: float,
    gtol: float,
) -> Dict:
    """Same L-BFGS-B budget at every start. The reported state is the best energy."""
    rows = []
    best = None
    for spec in starts:
        trace: List[float] = []

        def fun(x, _trace=trace):
            E, g = objective(np.asarray(x, dtype=np.float64))
            _trace.append(float(E))
            return float(E), np.asarray(g, dtype=np.float64)

        res = minimize(
            fun,
            np.asarray(spec["x0"], dtype=np.float64),
            method="L-BFGS-B",
            jac=True,
            bounds=bounds,
            options={"maxiter": int(maxiter), "ftol": float(ftol), "gtol": float(gtol), "maxfun": int(maxfun)},
        )
        xfin = np.asarray(res.x, dtype=np.float64)
        hit = []
        for i, (lo, hi) in enumerate(bounds):
            if abs(xfin[i] - lo) <= 1e-5 or abs(xfin[i] - hi) <= 1e-5:
                hit.append(i)
        row = {
            "name": spec["name"],
            "x0": [float(v) for v in spec["x0"]],
            "x": [float(v) for v in xfin],
            "E": float(res.fun),
            "nit": int(res.nit),
            "nfev": int(res.nfev),
            "njev": int(res.njev) if res.njev is not None else None,
            "success": bool(res.success),
            "message": str(res.message),
            "bound_hit_indices": hit,
            "trace": [float(v) for v in _downsample(trace, 250)],
        }
        rows.append(row)
        if best is None or row["E"] < best["E"] - 1e-15:
            best = row
    assert best is not None
    energies = np.array([r["E"] for r in rows], dtype=np.float64)
    return {
        "best": best,
        "starts": rows,
        "n_starts": len(rows),
        "E_best": float(energies.min()),
        "E_median": float(np.median(energies)),
        "E_worst": float(energies.max()),
        "nfev_sum": int(sum(r["nfev"] for r in rows)),
        "nit_sum": int(sum(r["nit"] for r in rows)),
    }


def _downsample(xs: List[float], limit: int) -> List[float]:
    if len(xs) <= limit:
        return xs
    idx = np.linspace(0, len(xs) - 1, limit).astype(int)
    return [xs[i] for i in idx]
