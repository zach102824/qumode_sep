"""Ascending bootstrap, interpolation, and a frozen list of restarts.

maxiter is 1500. Each start also has a maxfun cap so one flat start cannot
consume the round. Fourier and interpolation seeds are initializations only.
"""
from __future__ import annotations

import time

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize

from .circuit import SPECS, energy_and_grad, energy_only, fidelity, pack, param_count, unpack

SCALAR = (-20.0, 20.0)
SHELL = (-2.0 * np.pi, 2.0 * np.pi)
PERIODS = (0.5 * np.pi, 0.5 * np.pi, 0.5 * np.pi)
MAXITER = 1500
MAXFUN = 700


def _bounds(method: str, L: int):
    if method == "qaoa":
        return [SCALAR] * L + [SHELL] * L
    if method == "same_G_qaoa":
        ntheta = len(SPECS[method])
        return [SCALAR] * L + [SHELL] * L + [SCALAR] * ntheta
    ntheta = len(SPECS[method])
    return [SCALAR] * (1 + ntheta) + [SHELL] * (3 * L)


def _clip(x, bounds):
    y = np.asarray(x, dtype=np.float64).copy()
    for i, (lo, hi) in enumerate(bounds):
        y[i] = min(hi, max(lo, y[i]))
    return y


def _resample(values, Lnew, kind):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size == 1 or Lnew == values.size:
        return np.full(Lnew, float(values[0])) if values.size == 1 else values.copy()
    sold = np.linspace(0.0, 1.0, values.size)
    snew = np.linspace(0.0, 1.0, Lnew)
    if kind == "linear" or values.size < 3:
        return np.interp(snew, sold, values)
    return np.asarray(CubicSpline(sold, values, bc_type="natural")(snew), dtype=np.float64)


def _unwrap(cs):
    out = np.array(cs, dtype=np.float64, copy=True)
    for ell in range(1, out.shape[0]):
        for j, per in enumerate(PERIODS):
            k = np.round((out[ell - 1, j] - out[ell, j]) / per)
            out[ell, j] += k * per
    return out


def _rng(n, L, method, tag):
    names = (
        "family_N", "thetaN0", "A", "thetaM0", "full_intra", "wrong_sel",
        "shuf_jbar", "thetaK0", "qaoa", "same_G_qaoa",
    )
    mid = names.index(method)
    return np.random.default_rng((20261109 + n * 10007 + L * 97 + tag * 13 + mid * 17) % (2**32 - 1))


def _fresh(method, L, rng, n_fresh, prefix):
    starts = []
    for i in range(n_fresh):
        if method in ("qaoa", "same_G_qaoa"):
            g = rng.uniform(-0.8, 0.8, size=L)
            b = rng.uniform(-0.8, 0.8, size=L)
            theta = {k: float(rng.uniform(-1.2, 1.2)) for k in SPECS[method]}
            starts.append((f"{prefix}{i}", pack(method, 0.0, theta, np.zeros((L, 3)), g, b)))
        else:
            theta = {k: float(rng.uniform(-1.2, 1.2)) for k in SPECS[method]}
            if "K" in theta:
                theta["K"] = float(rng.uniform(-1.0, 0.2))
            cs = rng.uniform(-0.7, 0.7, size=(L, 3))
            starts.append((f"{prefix}{i}", pack(method, float(rng.uniform(-0.8, 0.8)), theta, cs)))
    return starts


def build_starts(method, L, prev, rng, n_fresh):
    starts = []
    if prev is None:
        if method in ("qaoa", "same_G_qaoa"):
            starts.append(("zero", np.zeros(param_count(method, L))))
            theta = {k: 0.0 for k in SPECS[method]}
            starts.append(("small", pack(method, 0.0, theta, np.zeros((L, 3)), np.full(L, 0.15), np.full(L, 0.15))))
        else:
            theta = {k: 0.0 for k in SPECS[method]}
            if "K" in theta:
                theta["K"] = -0.4
            cs = np.zeros((L, 3))
            cs[:, 0] = 0.2
            starts.append(("small", pack(method, 0.25, theta, cs)))
            theta2 = dict(theta)
            if "N" in theta2:
                theta2["N"] = 0.8
            if "M" in theta2:
                theta2["M"] = -0.3
            starts.append(("feature_seed", pack(method, 0.3, theta2, cs)))
    elif L == 1:
        pass
    else:
        gammas, theta, cs = unpack(method, prev, L - 1)
        cs_u = _unwrap(cs)
        if method in ("qaoa", "same_G_qaoa"):
            beta = cs[:, 0]
            if L == 2:
                g0, b0 = float(gammas[0]), float(beta[0])
                for off in (0.2, 0.0, 0.1, 0.4):
                    starts.append((f"pagano_{off}", pack(method, 0, theta, np.zeros((2, 3)), [g0, g0 + off], [b0, b0 - off])))
                    starts.append((f"pagano_flip_{off}", pack(method, 0, theta, np.zeros((2, 3)), [g0, g0 - off], [b0, b0 + off])))
            else:
                kind = "linear" if L == 3 else "cubic"
                starts.append((f"interp_{kind}", pack(method, 0, theta, np.zeros((L, 3)), _resample(gammas, L, kind), _resample(beta, L, kind))))
            t = (np.arange(L) + 0.5) / L
            ag = float(np.max(np.abs(gammas)))
            ab = float(np.max(np.abs(beta)))
            for mode in (1, 2):
                shape = np.sin(mode * np.pi * t)
                starts.append((f"fourier{mode}", pack(method, 0, theta, np.zeros((L, 3)), ag * shape, ab * shape)))
        else:
            gamma = float(gammas[0])
            if L == 2:
                row = cs_u[0]
                for off in (0.0, 0.2, -0.2, 0.1):
                    c = np.vstack([row, row])
                    c[1, 0] = row[0] - off
                    starts.append((f"dup_{off}", pack(method, gamma, theta, c)))
            else:
                kind = "linear" if L == 3 else "cubic"
                c_new = np.column_stack([_resample(cs_u[:, j], L, kind) for j in range(3)])
                starts.append((f"interp_{kind}", pack(method, gamma, theta, c_new)))
                for scale in (0.05, 0.15):
                    starts.append((f"interp_noise_{scale}", pack(method, gamma, theta, c_new + rng.normal(scale=scale, size=c_new.shape))))
    starts.extend(_fresh(method, L, rng, n_fresh, "fresh_"))
    return starts


def _screen(method, L, rng, feat, cap_screen, bounds):
    if L != 1 or method in ("qaoa", "same_G_qaoa"):
        pts = []
        if method == "qaoa" and L == 1:
            for g in np.linspace(-1.2, 1.2, 21):
                for b in np.linspace(-1.0, 1.0, 17):
                    pts.append(np.array([g, b]))
        return pts
    pts = []
    gammas = (-0.6, -0.2, 0.25, 0.6)
    thetas_k = (-0.8, -0.4, 0.0)
    thetas_m = (-0.4, 0.0, 0.4)
    thetas_n = (-0.5, 0.0, 0.8)
    names = SPECS[method]
    for g in gammas:
        for tk in thetas_k:
            for tm in thetas_m:
                for tn in thetas_n:
                    th = {"K": tk, "M": tm, "N": tn}
                    # Skip combinations that are not free, to keep the screen finite.
                    if "M" not in names and tm != 0.0:
                        continue
                    if "N" not in names and tn != 0.0:
                        continue
                    if "K" not in names and tk != 0.0:
                        continue
                    if (abs(tm) > 0 and abs(tn) > 0 and tk not in (-0.4, 0.0)):
                        continue
                    vec = pack(method, g, th, np.array([[0.25, 0.0, 0.0]]))
                    pts.append(vec)
    for _ in range(60):
        pts.append(_fresh(method, 1, rng, 1, "cloud_")[0][1])
    return pts


def optimize_method(feat, method, L, prev, eval_cap, n_fresh, seed_tag, extra_starts=None, do_screen=True):
    t0 = time.perf_counter()
    bounds = _bounds(method, L)
    rng = _rng(feat.n, L, method, seed_tag)
    n_eval = 0
    best_E = float("inf")
    best_x = None
    best_label = None
    best_trace = []
    rows = []

    def eval_e(x):
        nonlocal n_eval
        E, psi = energy_only(feat, method, x, L)
        n_eval += 1
        return E, psi

    seeds = []
    if do_screen and L == 1:
        cloud = _screen(method, L, rng, feat, eval_cap // 4, bounds)
        xs, Es = [], []
        for x0 in cloud:
            if n_eval >= max(80, eval_cap // 4):
                break
            x0 = _clip(x0, bounds)
            E, _ = eval_e(x0)
            xs.append(x0)
            Es.append(E)
        if xs:
            for i in np.argsort(Es)[:6]:
                seeds.append(("screen", xs[int(i)]))
    seeds.extend(build_starts(method, L, prev, rng, n_fresh))
    if extra_starts:
        seeds = list(extra_starts) + seeds
    seen = set()
    for label, x0 in seeds:
        if n_eval >= eval_cap - 20:
            rows.append({"label": label, "skipped": True})
            break
        x0 = _clip(np.asarray(x0, dtype=np.float64), bounds)
        key = tuple(np.round(x0, 5))
        if key in seen:
            continue
        seen.add(key)
        trace = []
        cache = {}
        local_E = float("inf")
        local_x = x0.copy()

        def fun(x, _trace=trace, _cache=cache):
            nonlocal n_eval, local_E, local_x
            if n_eval >= eval_cap:
                return local_E
            E, grad, _psi = energy_and_grad(feat, method, x, L)
            n_eval += 1
            _cache["x"] = np.array(x, copy=True)
            _cache["g"] = grad
            _trace.append(float(E))
            if E < local_E:
                local_E = float(E)
                local_x = np.array(x, copy=True)
            return float(E)

        def jac(x, _cache=cache, _trace=trace):
            nonlocal n_eval, local_E, local_x
            if _cache.get("x") is not None and np.allclose(_cache["x"], x, atol=1e-12, rtol=0):
                return _cache["g"]
            if n_eval >= eval_cap:
                return np.zeros_like(x, dtype=np.float64)
            E, grad, _psi = energy_and_grad(feat, method, x, L)
            n_eval += 1
            _cache["x"] = np.array(x, copy=True)
            _cache["g"] = grad
            _trace.append(float(E))
            if E < local_E:
                local_E = float(E)
                local_x = np.array(x, copy=True)
            return grad

        try:
            res = minimize(
                fun, x0, method="L-BFGS-B", jac=jac, bounds=bounds,
                options={"maxiter": MAXITER, "maxfun": min(MAXFUN, max(40, eval_cap - n_eval)), "ftol": 1e-14, "gtol": 1e-8},
            )
            ok, nit, msg = bool(res.success), int(res.nit), str(res.message)
        except Exception as exc:
            ok, nit, msg = False, 0, f"{type(exc).__name__}: {exc}"
        E_fin, psi = eval_e(local_x)
        F_fin = fidelity(feat.system, psi)
        if E_fin < best_E:
            best_E = float(E_fin)
            best_x = local_x.copy()
            best_label = label
            best_trace = list(trace)
        rows.append({"label": label, "E": float(E_fin), "F": float(F_fin), "success": ok, "nit": nit, "message": msg, "skipped": False})
    if best_x is None:
        raise RuntimeError(f"no start finished for {method} L={L}")
    E_check, _g, psi = energy_and_grad(feat, method, best_x, L)
    energies = np.array([r["E"] for r in rows if not r.get("skipped")], dtype=np.float64)
    q25, q75 = np.percentile(energies, [25, 75])
    gammas, theta, cs = unpack(method, best_x, L)
    return {
        "method": method,
        "n": feat.n,
        "L": L,
        "x": best_x,
        "E": float(E_check),
        "F": fidelity(feat.system, psi),
        "psi": psi,
        "init_label": best_label,
        "n_eval": int(n_eval),
        "eval_cap": int(eval_cap),
        "n_starts": int(energies.size),
        "seconds": time.perf_counter() - t0,
        "seed_best": float(energies.min()),
        "seed_median": float(np.median(energies)),
        "seed_iqr": float(q75 - q25),
        "seed_q25": float(q25),
        "seed_q75": float(q75),
        "seed_std": float(energies.std()),
        "gamma": [float(v) for v in gammas],
        "theta": {k: float(v) for k, v in theta.items()},
        "shell": cs.tolist(),
        "trace_best_start": [float(v) for v in best_trace[:300]],
        "starts": rows,
    }
