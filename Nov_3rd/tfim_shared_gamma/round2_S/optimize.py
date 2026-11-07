"""Ascending-depth bootstrap. Offsets and Fourier seeds are initializations only.

A zero shell is not an identity pad, so schedules are not grown by appending
a zero shell as if it did nothing.
"""
from __future__ import annotations

import time

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize

from .circuit import (
    SHELL_METHODS,
    energy_and_grad,
    energy_only,
    fidelity,
    pack_qaoa,
    pack_shell,
    param_count,
    unpack,
)
from .frozen import FrozenSystem

SCALAR = (-20.0, 20.0)
SHELL = (-2.0 * np.pi, 2.0 * np.pi)


def bounds_for(method: str, L: int):
    p = param_count(method, L)
    if method == "qaoa":
        return [SCALAR] * L + [SHELL] * L
    if method == "same_G_qaoa":
        return [SCALAR] * L + [SHELL] * L + [SCALAR, SCALAR]
    if method in ("family_S", "shuffled_intra"):
        return [SCALAR, SCALAR, SCALAR] + [SHELL] * (3 * L)
    return [SCALAR, SCALAR] + [SHELL] * (3 * L)


def _clip(x, bounds):
    y = np.array(x, dtype=np.float64, copy=True)
    for i, (lo, hi) in enumerate(bounds):
        y[i] = min(hi, max(lo, y[i]))
    return y


def _resample(values, Lnew, kind):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if Lnew == values.size:
        return values.copy()
    if values.size == 1:
        return np.full(Lnew, float(values[0]))
    s_old = np.linspace(0.0, 1.0, values.size)
    s_new = np.linspace(0.0, 1.0, Lnew)
    if kind == "linear" or values.size < 3:
        return np.interp(s_new, s_old, values)
    return np.asarray(CubicSpline(s_old, values, bc_type="natural")(s_new), dtype=np.float64)


def _unwrap(cs, periods):
    out = np.array(cs, dtype=np.float64, copy=True)
    for ell in range(1, out.shape[0]):
        for j, per in enumerate(periods):
            if per is None:
                continue
            k = np.round((out[ell - 1, j] - out[ell, j]) / per)
            out[ell, j] = out[ell, j] + k * per
    return out


def _shell_of(method, x, L):
    gammas, tin, tk, cs = unpack(method, x, L)
    return float(gammas[0]), float(tin), float(tk), cs


def build_starts(method, L, prev, rng, periods, n_fresh: int):
    starts = []
    if method in ("qaoa", "same_G_qaoa"):
        if prev is None or L == 1:
            z = np.zeros(param_count(method, L))
            starts.append(("zero", z))
            small = np.concatenate([np.full(L, 0.15), np.full(L, 0.15)])
            if method == "same_G_qaoa":
                small = np.concatenate([small, [1.0, 0.0]])
            starts.append(("small", small))
        else:
            g, tin, tk, cs = _shell_of(method, prev, L - 1)
            beta = cs[:, 0]
            if L == 2:
                for off in (0.2, 0.0, 0.1, 0.4):
                    starts.append((f"pagano_off_{off}", pack_qaoa(method, [g, g + off], [beta[0], beta[0] - off], tin, tk)))
                    starts.append((f"pagano_flip_{off}", pack_qaoa(method, [g, g - off], [beta[0], beta[0] + off], tin, tk)))
            else:
                kind = "linear" if L == 3 else "cubic"
                gi = _resample(_gammas(method, prev, L - 1), L, kind)
                bi = _resample(beta, L, kind)
                starts.append((f"interp_{kind}", pack_qaoa(method, gi, bi, tin, tk)))
            # Fourier-style seeds from the previous scalar magnitudes. Initialization only.
            amp_g = float(np.max(np.abs(_gammas(method, prev, L - 1))))
            amp_b = float(np.max(np.abs(beta)))
            t = (np.arange(L) + 0.5) / L
            for mode, name in ((1, "fourier1"), (2, "fourier2")):
                shape = np.sin(mode * np.pi * t)
                starts.append((name, pack_qaoa(method, amp_g * shape, amp_b * shape, tin, tk)))
        for i in range(n_fresh):
            g = rng.uniform(-0.8, 0.8, size=L)
            b = rng.uniform(-0.8, 0.8, size=L)
            tin = float(rng.uniform(-1.0, 2.0)) if method == "same_G_qaoa" else 1.0
            tk = float(rng.uniform(-1.0, 1.0)) if method == "same_G_qaoa" else 0.0
            starts.append((f"fresh_{i}", pack_qaoa(method, g, b, tin, tk)))
        return starts

    if prev is None or L == 1:
        if method in ("family_S", "shuffled_intra"):
            starts.append(("zero_theta", pack_shell(method, 0.2, 1.0, 0.0, np.array([0.2, 0.0, 0.0]))))
            starts.append(("theta_in_low", pack_shell(method, 0.3, 0.2, -0.4, np.array([0.2, 0.0, 0.1]))))
            starts.append(("theta_in_high", pack_shell(method, 0.3, 2.0, -0.4, np.array([0.2, 0.05, 0.0]))))
        elif method == "thetaK0":
            starts.append(("theta_in_1", pack_shell(method, 0.25, 1.0, 0.0, np.array([0.2, 0.0, 0.0]))))
            starts.append(("theta_in_low", pack_shell(method, 0.25, 0.2, 0.0, np.array([0.2, 0.1, 0.0]))))
        else:
            starts.append(("theta0", pack_shell(method, 0.25, 1.0, 0.0, np.array([0.2, 0.0, 0.0]))))
            starts.append(("theta_neg", pack_shell(method, 0.3, 1.0, -0.5, np.array([0.2, 0.0, 0.1]))))
    else:
        gamma, tin, tk, cs = _shell_of(method, prev, L - 1)
        cs_u = _unwrap(cs, periods)
        if L == 2:
            row = cs_u[0]
            for off in (0.0, 0.2, -0.2, 0.1):
                c = np.vstack([row, row])
                c[1, 0] = row[0] - off
                starts.append((f"dup_c1_{off}", pack_shell(method, gamma, tin, tk, c)))
            for dg in (0.05, -0.05):
                c = np.vstack([row, row])
                c[1, 0] = row[0] - 0.2
                starts.append((f"dup_gamma_{dg}", pack_shell(method, gamma + dg, tin, tk, c)))
        else:
            kind = "linear" if L == 3 else "cubic"
            c_new = np.column_stack([_resample(cs_u[:, j], L, kind) for j in range(3)])
            starts.append((f"interp_{kind}", pack_shell(method, gamma, tin, tk, c_new)))
            for scale in (0.05, 0.15):
                noise = rng.normal(scale=scale, size=c_new.shape)
                starts.append((f"interp_noise_{scale}", pack_shell(method, gamma, tin, tk, c_new + noise)))
    for i in range(n_fresh):
        c = rng.uniform(-0.7, 0.7, size=(L, 3))
        gamma = float(rng.uniform(-0.8, 0.8))
        tin = float(rng.uniform(-0.5, 2.0))
        tk = float(rng.uniform(-1.2, 1.2))
        starts.append((f"fresh_{i}", pack_shell(method, gamma, tin, tk, c)))
    return starts


def _fresh_only(method, L, rng, n_fresh: int):
    starts = []
    for i in range(n_fresh):
        if method in ("qaoa", "same_G_qaoa"):
            g = rng.uniform(-0.8, 0.8, size=L)
            b = rng.uniform(-0.8, 0.8, size=L)
            tin = float(rng.uniform(0.0, 2.0))
            tk = float(rng.uniform(-1.0, 1.0))
            starts.append((f"topup_{i}", pack_qaoa(method, g, b, tin, tk)))
        else:
            c = rng.uniform(-0.7, 0.7, size=(L, 3))
            starts.append(
                (
                    f"topup_{i}",
                    pack_shell(
                        method,
                        float(rng.uniform(-0.8, 0.8)),
                        float(rng.uniform(-0.5, 2.0)),
                        float(rng.uniform(-1.2, 1.2)),
                        c,
                    ),
                )
            )
    return starts


def _gammas(method, x, L):
    gammas, _tin, _tk, _cs = unpack(method, x, L)
    return gammas


def _screen_points(method, L, rng):
    if L != 1:
        return []
    pts = []
    if method == "qaoa":
        for g in np.linspace(-1.2, 1.2, 25):
            for b in np.linspace(-1.0, 1.0, 21):
                pts.append(np.array([g, b]))
        return pts
    if method == "same_G_qaoa":
        for g in np.linspace(-0.8, 0.8, 9):
            for b in np.linspace(-0.6, 0.6, 7):
                for tin in (0.5, 1.0, 1.5):
                    for tk in (-0.6, 0.0, 0.6):
                        pts.append(np.array([g, b, tin, tk]))
        return pts
    # Shell L=1 cloud.
    gammas = (-0.6, -0.2, 0.2, 0.6)
    c1s = (-0.4, 0.0, 0.4)
    extras = (0.0, 0.3, -0.3)
    if method in ("family_S", "shuffled_intra"):
        for g in gammas:
            for tin in (0.3, 1.0, 1.8):
                for tk in (-0.8, -0.3, 0.0, 0.4):
                    for c1 in c1s:
                        pts.append(np.array([g, tin, tk, c1, 0.0, 0.0]))
                        if c1 == 0.0:
                            pts.append(np.array([g, tin, tk, c1, 0.25, -0.2]))
    elif method == "thetaK0":
        for g in gammas:
            for tin in (0.3, 1.0, 1.8):
                for c1 in c1s:
                    pts.append(np.array([g, tin, c1, 0.0, 0.0]))
    else:
        for g in gammas:
            for tk in (-0.8, -0.3, 0.0, 0.4):
                for c1 in c1s:
                    pts.append(np.array([g, tk, c1, 0.0, 0.0], dtype=np.float64))
    for _ in range(80):
        pts.append(rng.normal(scale=0.35, size=param_count(method, 1)))
    return pts


def optimize_method(
    system: FrozenSystem,
    method: str,
    L: int,
    prev: np.ndarray | None,
    periods,
    eval_cap: int,
    n_fresh: int,
    seed_tag: int,
    do_screen: bool = True,
    extra_starts: list | None = None,
    fresh_only: bool = False,
) -> dict:
    t0 = time.perf_counter()
    bounds = bounds_for(method, L)
    rng = np.random.default_rng((20261108 + system.n * 1009 + L * 97 + seed_tag * 13 + _mid(method)) % (2**32 - 1))
    n_eval = 0
    best_E = float("inf")
    best_x = None
    best_label = None
    best_trace: list[float] = []
    rows = []

    def consider(x, E, F, label, trace):
        nonlocal best_E, best_x, best_label, best_trace
        if E < best_E:
            best_E = float(E)
            best_x = np.array(x, dtype=np.float64, copy=True)
            best_label = label
            best_trace = list(trace)

    def eval_energy(x):
        nonlocal n_eval
        E, psi = energy_only(system, method, x, L)
        n_eval += 1
        return E, psi

    seeds = []
    if fresh_only:
        seeds.extend(_fresh_only(method, L, rng, n_fresh))
    elif do_screen and L == 1:
        cloud = _screen_points(method, L, rng)
        xs, Es = [], []
        for x0 in cloud:
            if n_eval >= eval_cap // 3:
                break
            x0 = _clip(x0, bounds)
            E, _ = eval_energy(x0)
            xs.append(x0)
            Es.append(E)
        order = np.argsort(Es)[:8]
        for i in order:
            seeds.append(("screen", xs[int(i)]))
    if not fresh_only:
        seeds.extend(build_starts(method, L, prev, rng, periods, n_fresh=n_fresh))
    if extra_starts:
        seeds.extend(extra_starts)

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
        trace: list[float] = []
        cache = {"x": None}
        local_E = float("inf")
        local_x = x0.copy()

        def fun(x, _trace=trace, _cache=cache):
            nonlocal n_eval, local_E, local_x
            if n_eval >= eval_cap:
                return local_E
            E, grad, _psi = energy_and_grad(system, method, x, L)
            n_eval += 1
            _cache["x"] = np.array(x, copy=True)
            _cache["g"] = grad
            _trace.append(float(E))
            if E < local_E:
                local_E = float(E)
                local_x = np.array(x, copy=True)
            return float(E)

        def jac(x, _cache=cache):
            nonlocal n_eval, local_E, local_x
            if _cache["x"] is not None and np.allclose(_cache["x"], x, atol=1e-12, rtol=0):
                return _cache["g"]
            if n_eval >= eval_cap:
                return np.zeros_like(x)
            E, grad, _psi = energy_and_grad(system, method, x, L)
            n_eval += 1
            _cache["x"] = np.array(x, copy=True)
            _cache["g"] = grad
            trace.append(float(E))
            if E < local_E:
                local_E = float(E)
                local_x = np.array(x, copy=True)
            return grad

        try:
            res = minimize(
                fun,
                x0,
                method="L-BFGS-B",
                jac=jac,
                bounds=bounds,
                options={"maxiter": 80, "maxfun": min(220, max(30, eval_cap - n_eval)), "ftol": 1e-14, "gtol": 1e-8},
            )
            ok = bool(res.success)
            nit = int(res.nit)
            msg = str(res.message)
        except Exception as exc:
            ok = False
            nit = 0
            msg = f"{type(exc).__name__}: {exc}"
        E_fin, psi = eval_energy(local_x)
        F_fin = fidelity(system, psi)
        consider(local_x, E_fin, F_fin, label, trace)
        rows.append({"label": label, "E": E_fin, "F": F_fin, "success": ok, "nit": nit, "message": msg, "skipped": False})

    if best_x is None:
        raise RuntimeError(f"no completed start for {method} L={L}")
    E_check, _g, psi = energy_and_grad(system, method, best_x, L)
    if abs(E_check - best_E) > 1e-6:
        # Keep the recomputed value; the stored winner is the replay.
        best_E = float(E_check)
    energies = np.array([r["E"] for r in rows if not r.get("skipped")], dtype=np.float64)
    order = np.argsort(energies)
    top = energies[order[: max(1, len(energies) // 2)]]
    gammas, tin, tk, cs = unpack(method, best_x, L)
    return {
        "method": method,
        "n": system.n,
        "L": L,
        "x": best_x,
        "E": float(E_check),
        "F": fidelity(system, psi),
        "init_label": best_label,
        "n_eval": int(n_eval),
        "eval_cap": int(eval_cap),
        "n_starts": int(energies.size),
        "seconds": time.perf_counter() - t0,
        "seed_median": float(np.median(energies)),
        "seed_median_best_half": float(np.median(top)),
        "seed_best": float(energies.min()),
        "seed_std": float(energies.std()),
        "seed_spread_median_minus_best": float(np.median(energies) - energies.min()),
        "gamma": [float(v) for v in gammas],
        "thetaIn": None if method in ("qaoa", "full_K_A", "inter_K_A") else float(tin),
        "thetaK": None if method in ("qaoa", "thetaK0") else float(tk),
        "shell": cs.tolist(),
        "trace_best_start": [float(v) for v in best_trace[:200]],
        "starts": rows,
        "manifest_sha256": system.manifest_sha256,
        "hc_diag_sha256": system.hc_diag_sha256,
        "psi": psi,
    }


def _mid(method: str) -> int:
    names = (
        "qaoa",
        "thetaK0",
        "inter_K_A",
        "full_K_A",
        "shuffled_intra",
        "family_S",
        "same_G_qaoa",
    )
    return names.index(method)


def top_up(system, method, L, current: dict, periods, target_evals: int, seed_tag: int) -> dict:
    """Add fresh starts until the evaluation count reaches target_evals."""
    if current["n_eval"] >= target_evals:
        current["topped_up"] = False
        return current
    guard = 0
    cur = current
    while cur["n_eval"] < target_evals and guard < 8:
        guard += 1
        extra_cap = target_evals - cur["n_eval"] + 40
        more = _top_up_once(system, method, L, cur, periods, extra_cap, seed_tag + guard)
        cur = more
    return cur


def _top_up_once(system, method, L, current, periods, extra_cap, seed_tag) -> dict:
    more = optimize_method(
        system,
        method,
        L,
        None,
        periods,
        eval_cap=extra_cap,
        n_fresh=max(4, extra_cap // 80),
        seed_tag=seed_tag,
        do_screen=False,
        fresh_only=True,
    )
    more["n_eval"] = int(current["n_eval"] + more["n_eval"])
    more["seconds"] = float(current["seconds"] + more["seconds"])
    more["topped_up"] = True
    more["pre_topup_E"] = current["E"]
    more["pre_topup_n_eval"] = current["n_eval"]
    # Keep the better energy. Medians are from the added round only; store both.
    if current["E"] <= more["E"]:
        kept = dict(current)
        kept["n_eval"] = more["n_eval"]
        kept["seconds"] = more["seconds"]
        kept["topped_up"] = True
        kept["topup_did_not_improve"] = True
        kept["topup_E"] = more["E"]
        kept["starts"] = list(current["starts"]) + list(more["starts"])
        energies = np.array([r["E"] for r in kept["starts"] if "E" in r], dtype=np.float64)
        kept["seed_median"] = float(np.median(energies))
        kept["seed_std"] = float(energies.std())
        kept["seed_spread_median_minus_best"] = float(np.median(energies) - energies.min())
        kept["n_starts"] = int(energies.size)
        kept["seed_best"] = float(energies.min())
        return kept
    more["starts"] = list(current["starts"]) + list(more["starts"])
    energies = np.array([r["E"] for r in more["starts"] if "E" in r], dtype=np.float64)
    more["seed_median"] = float(np.median(energies))
    more["seed_std"] = float(energies.std())
    more["seed_spread_median_minus_best"] = float(np.median(energies) - energies.min())
    more["n_starts"] = int(energies.size)
    more["seed_best"] = float(energies.min())
    return more
