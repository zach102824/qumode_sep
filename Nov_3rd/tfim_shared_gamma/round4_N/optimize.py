"""Bootstrap and restarts. Seeds are frozen by seed_base 20261110 in protocol.json."""
from __future__ import annotations

import time

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize

from .model import ARMS, pack, param_count, unpack

SCALAR = (-12.0, 12.0)
SHELL = (-2.0 * np.pi, 2.0 * np.pi)
MAXITER = 1500
MAXFUN = 500


def _bounds(arm, L):
    if arm == "qaoa":
        return [SCALAR] * L + [SHELL] * L
    return [SCALAR] * (1 + len(ARMS[arm]["thetas"])) + [SHELL] * (3 * L)


def _clip(x, bounds):
    y = np.asarray(x, dtype=np.float64).copy()
    for i, (lo, hi) in enumerate(bounds):
        y[i] = min(max(y[i], lo), hi)
    return y


def _rng(n, L, arm, tag):
    names = list(ARMS)
    mid = names.index(arm) if arm in names else 9
    return np.random.default_rng((20261110 + n * 10007 + L * 97 + tag * 13 + mid * 17) % (2**32 - 1))


def _resample(values, Lnew, kind):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size == 1:
        return np.full(Lnew, float(values[0]))
    if Lnew == values.size:
        return values.copy()
    sold = np.linspace(0.0, 1.0, values.size)
    snew = np.linspace(0.0, 1.0, Lnew)
    if kind == "linear" or values.size < 3:
        return np.interp(snew, sold, values)
    return np.asarray(CubicSpline(sold, values, bc_type="natural")(snew))


def _unwrap(cs):
    out = np.array(cs, dtype=np.float64, copy=True)
    period = np.pi  # checked numerically as a global phase for each shell coefficient
    for ell in range(1, out.shape[0]):
        for j in range(3):
            k = np.round((out[ell - 1, j] - out[ell, j]) / period)
            out[ell, j] += k * period
    return out


def _fresh(arm, L, rng, n, prefix):
    out = []
    for i in range(n):
        if arm == "qaoa":
            out.append((f"{prefix}{i}", pack("qaoa", 0, {}, np.zeros((L, 3)), rng.uniform(-0.8, 0.8, L), rng.uniform(-0.8, 0.8, L))))
        else:
            theta = {name: float(rng.uniform(-1.0, 1.0)) for name in ARMS[arm]["thetas"]}
            if "K" in theta:
                theta["K"] = float(rng.uniform(-1.0, 0.2))
            cs = rng.uniform(-0.6, 0.6, size=(L, 3))
            out.append((f"{prefix}{i}", pack(arm, float(rng.uniform(-0.6, 0.6)), theta, cs)))
    return out


def build_starts(arm, L, prev, rng, n_fresh):
    starts = []
    if prev is None:
        if arm == "qaoa":
            starts.append(("zero", np.zeros(2 * L)))
            starts.append(("small", pack("qaoa", 0, {}, np.zeros((L, 3)), np.full(L, 0.15), np.full(L, 0.15))))
        else:
            theta = {name: 0.0 for name in ARMS[arm]["thetas"]}
            if "K" in theta:
                theta["K"] = -0.5
            if "M" in theta:
                theta["M"] = 0.4
            if "N" in theta:
                theta["N"] = 0.4
            if "B" in theta:
                theta["B"] = -0.4
            cs = np.zeros((L, 3))
            cs[:, 0] = 0.2
            starts.append(("small", pack(arm, 0.25, theta, cs)))
    else:
        gammas, theta, cs = unpack(arm, prev, L - 1)
        cs = _unwrap(cs)
        if arm == "qaoa":
            beta = cs[:, 0]
            if L == 2:
                g0, b0 = float(gammas[0]), float(beta[0])
                for off in (0.2, 0.0, 0.1):
                    starts.append((f"pagano_{off}", pack("qaoa", 0, {}, np.zeros((2, 3)), [g0, g0 + off], [b0, b0 - off])))
            else:
                kind = "linear" if L == 3 else "cubic"
                starts.append((f"interp_{kind}", pack("qaoa", 0, {}, np.zeros((L, 3)), _resample(gammas, L, kind), _resample(beta, L, kind))))
            t = (np.arange(L) + 0.5) / L
            ag, ab = float(np.max(np.abs(gammas))), float(np.max(np.abs(beta)))
            for mode in (1, 2):
                shape = np.sin(mode * np.pi * t)
                starts.append((f"fourier{mode}", pack("qaoa", 0, {}, np.zeros((L, 3)), ag * shape, ab * shape)))
        else:
            gamma = float(gammas[0])
            if L == 2:
                row = cs[0]
                for off in (0.0, 0.2, -0.2):
                    c = np.vstack([row, row])
                    c[1, 0] = row[0] - off
                    starts.append((f"dup_{off}", pack(arm, gamma, theta, c)))
            else:
                kind = "linear" if L == 3 else "cubic"
                c_new = np.column_stack([_resample(cs[:, j], L, kind) for j in range(3)])
                starts.append((f"interp_{kind}", pack(arm, gamma, theta, c_new)))
                starts.append((f"interp_noise", pack(arm, gamma, theta, c_new + rng.normal(scale=0.08, size=c_new.shape))))
    starts.extend(_fresh(arm, L, rng, n_fresh, "fresh_"))
    return starts


def optimize(runner, arm, L, prev, n_fresh, seed_tag, eval_cap, extra=None, screen=True):
    t0 = time.perf_counter()
    bounds = _bounds(arm, L)
    rng = _rng(runner.s.n, L, arm, seed_tag)
    n_eval = 0
    best_E, best_x, best_label, best_trace = float("inf"), None, None, []
    best_so_far = []
    rows = []

    def consider(x, E, label, trace):
        nonlocal best_E, best_x, best_label, best_trace
        if E < best_E:
            best_E, best_x, best_label, best_trace = float(E), np.array(x, copy=True), label, list(trace)

    seeds = []
    screen_pts = []
    if screen and L == 1 and arm != "qaoa":
        # A short axis screen, charged.
        for g in (-0.4, 0.2, 0.5):
            theta = {name: 0.0 for name in ARMS[arm]["thetas"]}
            for val in (-0.6, 0.0, 0.5):
                if theta:
                    theta[next(iter(theta))] = val
                screen_pts.append(pack(arm, g, theta, np.array([[0.2, 0.0, 0.0]])))
    elif screen and arm == "qaoa" and L == 1:
        for g in np.linspace(-1.0, 1.0, 13):
            for b in np.linspace(-0.8, 0.8, 9):
                screen_pts.append(np.array([g, b], dtype=np.float64))
    if screen_pts:
        scored = []
        for x0 in screen_pts:
            if n_eval >= eval_cap // 5:
                break
            x0 = _clip(x0, bounds)
            E, _ = runner.energy_only(arm, x0, L)
            n_eval += 1
            scored.append((E, x0))
        scored.sort(key=lambda t: t[0])
        for E, x0 in scored[:5]:
            seeds.append(("screen", x0))
    seeds.extend(build_starts(arm, L, prev, rng, n_fresh))
    if extra:
        seeds = list(extra) + seeds
    seen = set()
    for label, x0 in seeds:
        if n_eval >= eval_cap - 15:
            rows.append({"label": label, "skipped": True})
            break
        x0 = _clip(np.asarray(x0, dtype=np.float64), bounds)
        key = tuple(np.round(x0, 5))
        if key in seen:
            continue
        seen.add(key)
        trace, cache = [], {}
        local_E, local_x = float("inf"), x0.copy()

        def fun(x, _tr=trace, _c=cache):
            nonlocal n_eval, local_E, local_x
            if n_eval >= eval_cap:
                return local_E
            E, grad, _ = runner.energy_grad(arm, x, L)
            n_eval += 1
            _c["x"] = np.array(x, copy=True)
            _c["g"] = grad
            _tr.append(float(E))
            if E < local_E:
                local_E, local_x = float(E), np.array(x, copy=True)
            consider(x, E, label, _tr)
            best_so_far.append(float(best_E))
            return float(E)

        def jac(x, _tr=trace, _c=cache):
            nonlocal n_eval, local_E, local_x
            if _c.get("x") is not None and np.allclose(_c["x"], x, atol=1e-12, rtol=0):
                return _c["g"]
            if n_eval >= eval_cap:
                return np.zeros_like(x)
            E, grad, _ = runner.energy_grad(arm, x, L)
            n_eval += 1
            _c["x"] = np.array(x, copy=True)
            _c["g"] = grad
            _tr.append(float(E))
            if E < local_E:
                local_E, local_x = float(E), np.array(x, copy=True)
            consider(x, E, label, _tr)
            best_so_far.append(float(best_E))
            return grad

        try:
            res = minimize(fun, x0, method="L-BFGS-B", jac=jac, bounds=bounds,
                           options={"maxiter": MAXITER, "maxfun": min(MAXFUN, max(30, eval_cap - n_eval)), "ftol": 1e-14, "gtol": 1e-8})
            ok, nit = bool(res.success), int(res.nit)
        except Exception:
            ok, nit = False, 0
        E_fin, psi = runner.energy_only(arm, local_x, L)
        consider(local_x, E_fin, label, trace)
        rows.append({"label": label, "E": float(E_fin), "F": runner.fidelity(psi), "success": ok, "nit": nit, "skipped": False})
    if best_x is None:
        raise RuntimeError(f"no start {arm} L={L}")
    E, _, psi = runner.energy_grad(arm, best_x, L)
    energies = np.array([r["E"] for r in rows if "E" in r], dtype=np.float64)
    q25, q75 = np.percentile(energies, [25, 75])
    gammas, theta, cs = unpack(arm, best_x, L)
    return {
        "arm": arm, "n": runner.s.n, "L": L, "x": best_x, "E": float(E), "F": runner.fidelity(psi),
        "psi": psi, "init": best_label, "n_eval": int(n_eval), "n_starts": int(energies.size),
        "seconds": time.perf_counter() - t0, "seed_best": float(energies.min()),
        "seed_median": float(np.median(energies)), "seed_iqr": float(q75 - q25),
        "gamma": [float(v) for v in gammas], "theta": {k: float(v) for k, v in theta.items()},
        "shell": cs.tolist(), "trace": [float(v) for v in best_trace[:250]], "starts": rows,
        "variance_H": runner.variance(psi),
        "best_so_far": best_so_far[:: max(1, len(best_so_far) // 400)],
        "n_success": int(sum(1 for r in rows if r.get("success"))),
        "n_failed": int(sum(1 for r in rows if r.get("success") is False)),
    }
