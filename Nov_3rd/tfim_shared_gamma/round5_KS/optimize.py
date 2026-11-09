"""Energy-only L-BFGS-B. Seeds come from SeedSequence entropy 20261112.

Layer-center knots are x_l = (l+0.5)/L. Fewer than three source layers are
resampled linearly. Three or more use a natural cubic inside the source knots
and a constant clamp outside them. Gamma and theta are copied, not wrapped.
"""
from __future__ import annotations

import time

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize

from .model import ARMS, pack, unpack

SEED_ENTROPY = 20261112
SCALAR = (-12.0, 12.0)
SHELL = (-2.0 * np.pi, 2.0 * np.pi)
MAXITER = 1500
MAXFUN = 400
ARM_ID = {"KS_fresh": 1, "Rfull_fresh": 2, "KS_nested": 3, "KS_transfer": 4, "qaoa": 5, "KS_smoke": 6}


def generator_of(role: str) -> str:
    if role.startswith("KS"):
        return "KS"
    if role.startswith("Rfull"):
        return "Rfull"
    if role == "qaoa":
        return "qaoa"
    raise KeyError(role)


def rng_for(role: str, L: int, k: int, tag: int) -> np.random.Generator:
    seq = np.random.SeedSequence([SEED_ENTROPY, ARM_ID[role], int(L), int(k), int(tag)])
    return np.random.default_rng(seq)


def _bounds(gen: str, L: int):
    if gen == "qaoa":
        return [SCALAR] * L + [SHELL] * L
    return [SCALAR] * (1 + len(ARMS[gen]["thetas"])) + [SHELL] * (3 * L)


def _clip(x, bounds):
    y = np.asarray(x, dtype=np.float64).copy()
    for i, (lo, hi) in enumerate(bounds):
        y[i] = min(max(float(y[i]), lo), hi)
    return y


def unwrap_shell(cs: np.ndarray) -> np.ndarray:
    out = np.array(cs, dtype=np.float64, copy=True)
    period = np.pi
    for ell in range(1, out.shape[0]):
        for j in range(3):
            k = np.round((out[ell - 1, j] - out[ell, j]) / period)
            out[ell, j] += k * period
    return out


def resample_rows(values: np.ndarray, L_new: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    L_old = values.size
    if L_new == L_old:
        return values.copy()
    xs = (np.arange(L_old) + 0.5) / L_old
    xt = (np.arange(L_new) + 0.5) / L_new
    if L_old < 3:
        return np.interp(xt, xs, values)
    spline = CubicSpline(xs, values, bc_type="natural")
    return np.asarray(spline(np.clip(xt, xs[0], xs[-1])), dtype=np.float64)


def resample_shell(cs: np.ndarray, L_new: int) -> np.ndarray:
    cs = unwrap_shell(cs)
    return np.column_stack([resample_rows(cs[:, j], L_new) for j in range(3)])


def fresh_vector(gen: str, L: int, rng: np.random.Generator) -> np.ndarray:
    if gen == "qaoa":
        return pack("qaoa", 0.0, {}, np.zeros((L, 3)), rng.uniform(-0.8, 0.8, L), rng.uniform(-0.8, 0.8, L))
    theta = {name: float(rng.uniform(-1.0, 1.0)) for name in ARMS[gen]["thetas"]}
    cs = rng.uniform(-0.6, 0.6, size=(L, 3))
    return pack(gen, float(rng.uniform(-0.6, 0.6)), theta, cs)


def ascending_seeds(gen: str, L: int, prev: np.ndarray | None, role: str, n_fresh: int):
    seeds = []
    if prev is None:
        if gen == "qaoa":
            seeds.append(("zero", np.zeros(2 * L), 0.0))
        else:
            theta = {name: 0.0 for name in ARMS[gen]["thetas"]}
            cs = np.zeros((L, 3))
            cs[:, 0] = 0.2
            seeds.append(("plain", pack(gen, 0.25, theta, cs), 0.0))
    else:
        gammas, theta, cs = unpack(gen, prev, L - 1)
        cs = unwrap_shell(cs)
        if gen == "qaoa":
            beta = cs[:, 0]
            if L == 2:
                g0, b0 = float(gammas[0]), float(beta[0])
                for k, off in enumerate((0.2, 0.0, 0.1)):
                    seeds.append((f"pagano_{off}", pack("qaoa", 0, {}, np.zeros((2, 3)), [g0, g0 + off], [b0, b0 - off]), 0.0))
            else:
                kind = "linear" if L == 3 else "cubic"
                seeds.append((f"interp_{kind}", pack("qaoa", 0, {}, np.zeros((L, 3)), resample_rows(gammas, L), resample_rows(beta, L)), 0.0))
            t = (np.arange(L) + 0.5) / L
            ag, ab = float(np.max(np.abs(gammas))), float(np.max(np.abs(beta)))
            for mode in (1, 2):
                shape = np.sin(mode * np.pi * t)
                seeds.append((f"fourier{mode}", pack("qaoa", 0, {}, np.zeros((L, 3)), ag * shape, ab * shape), 0.0))
        else:
            gamma = float(gammas[0])
            if L == 2:
                row = cs[0]
                for off in (0.0, 0.2, -0.2):
                    c = np.vstack([row, row])
                    c[1, 0] = row[0] - off
                    seeds.append((f"pagano_{off}", pack(gen, gamma, theta, c), 0.0))
            else:
                kind = "linear" if L == 3 else "cubic"
                seeds.append((f"interp_{kind}", pack(gen, gamma, theta, resample_shell(cs, L)), 0.0))
    for k in range(n_fresh):
        seeds.append((f"fresh_{k}", fresh_vector(gen, L, rng_for(role, L, k, 3)), 0.0))
    return seeds


def perturb_seeds(x0: np.ndarray, role: str, L: int, sigmas=(0.02, 0.1), n_each=12, label="nested"):
    seeds = [(f"{label}_exact", np.array(x0, dtype=np.float64), 0.0)]
    for tag, sigma in enumerate(sigmas, start=1):
        for k in range(n_each):
            rng = rng_for(role, L, k, tag)
            seeds.append((f"{label}_s{sigma}_{k}", np.array(x0, dtype=np.float64) + rng.normal(scale=sigma, size=np.shape(x0)), float(sigma)))
    return seeds


def optimize(runner, gen: str, L: int, seeds, role: str, eval_cap: int, screen: bool = False):
    t0 = time.perf_counter()
    bounds = _bounds(gen, L)
    n_eval = 0
    best_E, best_x, best_label = float("inf"), None, None
    best_so_far = []
    rows = []

    def consider(x, E, label):
        nonlocal best_E, best_x, best_label
        if E < best_E:
            best_E = float(E)
            best_x = np.array(x, copy=True)
            best_label = label

    use = []
    if screen and L == 1 and gen != "qaoa":
        pts = []
        for g in (-0.4, 0.2, 0.5):
            theta = {name: 0.0 for name in ARMS[gen]["thetas"]}
            for val in (-0.6, 0.0, 0.5):
                theta[next(iter(theta))] = val
                pts.append(pack(gen, g, theta, np.array([[0.2, 0.0, 0.0]])))
        scored = []
        for x0 in pts:
            if n_eval >= max(20, eval_cap // 8):
                break
            x0 = _clip(x0, bounds)
            E, _ = runner.energy_only(gen, x0, L)
            n_eval += 1
            best_so_far.append(float(min(best_E, E)))
            scored.append((E, x0))
        scored.sort(key=lambda t: t[0])
        for _, x0 in scored[:4]:
            use.append(("screen", x0, 0.0))
    elif screen and gen == "qaoa" and L == 1:
        pts = [np.array([g, b], dtype=np.float64) for g in np.linspace(-1.0, 1.0, 9) for b in np.linspace(-0.8, 0.8, 7)]
        scored = []
        for x0 in pts:
            if n_eval >= max(30, eval_cap // 8):
                break
            E, _ = runner.energy_only(gen, x0, L)
            n_eval += 1
            best_so_far.append(float(min(best_E, E) if np.isfinite(best_E) else E))
            scored.append((E, x0))
        scored.sort(key=lambda t: t[0])
        for _, x0 in scored[:4]:
            use.append(("screen", x0, 0.0))
    use.extend(seeds)
    seen = set()
    for label, x0, sigma in use:
        if n_eval >= eval_cap - 15:
            rows.append({"label": label, "sigma": sigma, "skipped": True})
            break
        x0 = _clip(np.asarray(x0, dtype=np.float64), bounds)
        key = tuple(np.round(x0, 6))
        if key in seen:
            continue
        seen.add(key)
        trace, cache = [], {}
        local_E, local_x = float("inf"), x0.copy()

        def fun(x, _tr=trace, _c=cache, _label=label):
            nonlocal n_eval, local_E, local_x
            if n_eval >= eval_cap:
                return local_E
            E, grad, _ = runner.energy_grad(gen, x, L)
            n_eval += 1
            _c["x"] = np.array(x, copy=True)
            _c["g"] = grad
            _tr.append(float(E))
            if E < local_E:
                local_E, local_x = float(E), np.array(x, copy=True)
            consider(x, E, _label)
            best_so_far.append(float(best_E))
            return float(E)

        def jac(x, _tr=trace, _c=cache, _label=label):
            nonlocal n_eval, local_E, local_x
            if _c.get("x") is not None and np.allclose(_c["x"], x, atol=1e-12, rtol=0):
                return _c["g"]
            if n_eval >= eval_cap:
                return np.zeros_like(x)
            E, grad, _ = runner.energy_grad(gen, x, L)
            n_eval += 1
            _c["x"] = np.array(x, copy=True)
            _c["g"] = grad
            _tr.append(float(E))
            if E < local_E:
                local_E, local_x = float(E), np.array(x, copy=True)
            consider(x, E, _label)
            best_so_far.append(float(best_E))
            return grad

        try:
            res = minimize(
                fun, x0, method="L-BFGS-B", jac=jac, bounds=bounds,
                options={"maxiter": MAXITER, "maxfun": min(MAXFUN, max(30, eval_cap - n_eval)), "ftol": 1e-14, "gtol": 1e-8},
            )
            ok, nit = bool(res.success), int(res.nit)
        except Exception:
            ok, nit = False, 0
        E_fin, psi = runner.energy_only(gen, local_x, L)
        consider(local_x, E_fin, label)
        rows.append({
            "label": label, "sigma": sigma, "E": float(E_fin), "F": runner.fidelity(psi),
            "success": ok, "nit": nit, "skipped": False,
        })
    if best_x is None:
        raise RuntimeError(f"no start {role} L={L}")
    E, _, psi = runner.energy_grad(gen, best_x, L)
    energies = np.array([r["E"] for r in rows if "E" in r], dtype=np.float64)
    q25, q75 = np.percentile(energies, [25, 75])
    gammas, theta, cs = unpack(gen, best_x, L)
    return {
        "role": role, "arm": gen, "n": runner.t.n, "L": L, "x": best_x, "E": float(E),
        "F": runner.fidelity(psi), "psi": psi, "init": best_label, "n_eval": int(n_eval),
        "n_starts": int(energies.size), "n_success": int(sum(1 for r in rows if r.get("success"))),
        "n_failed": int(sum(1 for r in rows if r.get("success") is False)),
        "n_skipped": int(sum(1 for r in rows if r.get("skipped"))),
        "seconds": time.perf_counter() - t0,
        "seed_median": float(np.median(energies)), "seed_iqr": float(q75 - q25),
        "seed_best": float(energies.min()),
        "gamma": [float(v) for v in gammas], "theta": {k: float(v) for k, v in theta.items()},
        "shell": cs.tolist(), "starts": rows,
        "best_so_far": best_so_far[:: max(1, len(best_so_far) // 400)],
        "variance_H": runner.variance(psi),
    }
