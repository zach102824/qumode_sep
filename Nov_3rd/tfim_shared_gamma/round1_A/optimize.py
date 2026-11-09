"""Pagano-style depth bootstrap with a shared evaluation budget.

Independent-angle QAOA at L=2 is seeded by the published offset pattern
(beta, beta-0.2), (gamma, gamma+0.2) in J0=1 units, plus other offsets.
Those offsets are initializations. They are not treated as an optimal ramp.

Nov_3rd keeps a single gamma and a single theta. Only the three shell
controls are interpolated across normalized layer time. A zero shell is not
used as an identity pad.
"""
from __future__ import annotations

import time

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize

from .circuit import (
    THETA_METHODS,
    energy_and_grad,
    energy_only,
    fidelity_of_state,
    pack_qaoa,
    pack_shell,
    unpack,
)
from .common import EVAL_CAP, LBFGS_MAXFUN, LBFGS_MAXITER, opt_rng
from .hamiltonian import Problem

SHELL_BOUNDS = (-2.0 * np.pi, 2.0 * np.pi)
SCALAR_BOUNDS = (-8.0, 8.0)


def bounds_for(method: str, L: int) -> list[tuple[float, float]]:
    if method == "qaoa":
        return [SCALAR_BOUNDS] * L + [SHELL_BOUNDS] * L
    if method == "shared_qaoa":
        return [SCALAR_BOUNDS] + [SHELL_BOUNDS] * L
    if method == "native_D":
        return [SCALAR_BOUNDS] + [SHELL_BOUNDS] * (3 * L)
    return [SCALAR_BOUNDS, SCALAR_BOUNDS] + [SHELL_BOUNDS] * (3 * L)


def _clip(x: np.ndarray, bounds) -> np.ndarray:
    y = np.array(x, dtype=np.float64, copy=True)
    for i, (lo, hi) in enumerate(bounds):
        if y[i] < lo:
            y[i] = lo
        elif y[i] > hi:
            y[i] = hi
    return y


def _on_bound(x: np.ndarray, bounds, tol: float = 1e-4) -> list[int]:
    hit = []
    for i, (lo, hi) in enumerate(bounds):
        if abs(x[i] - lo) <= tol or abs(x[i] - hi) <= tol:
            hit.append(i)
    return hit


def resample_schedule(values: np.ndarray, Lnew: int, kind: str) -> np.ndarray:
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


def continue_branch(prev_value: float, value: float, period: float | None) -> float:
    if period is None:
        return float(value)
    k = np.round((prev_value - value) / period)
    return float(value + k * period)


def unwrap_schedule(cs: np.ndarray, periods: list[float | None]) -> np.ndarray:
    out = np.array(cs, dtype=np.float64, copy=True)
    for ell in range(1, out.shape[0]):
        for j, per in enumerate(periods):
            if per is not None:
                out[ell, j] = continue_branch(out[ell - 1, j], out[ell, j], per)
    return out


def _qaoa_vectors(x: np.ndarray, L: int) -> tuple[np.ndarray, np.ndarray]:
    gammas, _thetas, cs = unpack("qaoa", x, L)
    return gammas, cs[:, 0].copy()


def _shell_pieces(method: str, x: np.ndarray, L: int):
    gammas, thetas, cs = unpack(method, x, L)
    theta = None if method not in THETA_METHODS else float(thetas[0])
    return float(gammas[0]), theta, cs


def build_starts(
    method: str,
    L: int,
    prev: dict | None,
    siblings: dict | None,
    periods: list[float | None],
    rng: np.random.Generator,
    n_fresh: int,
) -> list[tuple[str, np.ndarray]]:
    """Deterministic list of (label, x0). Fresh draws use the supplied rng."""
    starts: list[tuple[str, np.ndarray]] = []
    siblings = siblings or {}

    if method == "qaoa":
        if L == 1 or prev is None:
            starts.append(("zero", np.zeros(2 * L)))
            starts.append(("small", np.concatenate([np.full(L, 0.15), np.full(L, 0.15)])))
        else:
            g, b = _qaoa_vectors(np.asarray(prev["x"], dtype=np.float64), L - 1)
            if L == 2:
                g1, b1 = float(g[0]), float(b[0])
                # 0.2 is Pagano's published seed in J0=1 units. Other offsets
                # are here so that number is tested as a heuristic.
                for off in (0.2, 0.0, 0.1, 0.4):
                    starts.append(
                        (f"pagano_off_{off}", pack_qaoa([g1, g1 + off], [b1, b1 - off]))
                    )
                    starts.append(
                        (f"pagano_flip_{off}", pack_qaoa([g1, g1 - off], [b1, b1 + off]))
                    )
            else:
                kind = "linear" if L == 3 else "cubic"
                gi = resample_schedule(g, L, kind)
                bi = resample_schedule(b, L, kind)
                starts.append((f"interp_{kind}", pack_qaoa(gi, bi)))
                starts.append(("dup_edge", pack_qaoa(np.concatenate([g, g[-1:]]), np.concatenate([b, b[-1:]]))))
        for i in range(n_fresh):
            starts.append(
                (
                    f"fresh_{i}",
                    np.concatenate(
                        [rng.uniform(-0.8, 0.8, size=L), rng.uniform(-0.8, 0.8, size=L)]
                    ),
                )
            )
        return starts

    if method == "shared_qaoa":
        if L == 1 or prev is None:
            starts.append(("zero", np.zeros(1 + L)))
            starts.append(("small", np.concatenate([[0.15], np.full(L, 0.15)])))
        else:
            gamma, _th, cs = _shell_pieces("shared_qaoa", np.asarray(prev["x"]), L - 1)
            # shared_qaoa unpack works; cs[:,0] is beta
            beta = cs[:, 0]
            if L == 2:
                b1 = float(beta[0])
                for off in (0.2, 0.0, 0.1, -0.2):
                    starts.append(
                        (
                            f"beta_shift_{off}",
                            pack_qaoa_shared(gamma, [b1, b1 - off]),
                        )
                    )
                for dg in (0.05, -0.05, 0.1):
                    starts.append(
                        (f"gamma_shift_{dg}", pack_qaoa_shared(gamma + dg, [b1, b1 - 0.2]))
                    )
            else:
                kind = "linear" if L == 3 else "cubic"
                bi = resample_schedule(beta, L, kind)
                starts.append((f"interp_{kind}", pack_qaoa_shared(gamma, bi)))
                for dg in (0.05, -0.05):
                    starts.append((f"interp_gamma_{dg}", pack_qaoa_shared(gamma + dg, bi)))
        qaoa_sib = siblings.get("qaoa")
        if qaoa_sib is not None:
            g, b = _qaoa_vectors(np.asarray(qaoa_sib["x"]), L)
            starts.append(("qaoa_mean_gamma", pack_qaoa_shared(float(np.mean(g)), b)))
        for i in range(n_fresh):
            starts.append(
                (
                    f"fresh_{i}",
                    np.concatenate([[rng.uniform(-0.8, 0.8)], rng.uniform(-0.8, 0.8, size=L)]),
                )
            )
        return starts

    with_theta = method in THETA_METHODS
    if L == 1 or prev is None:
        if with_theta:
            starts.append(("zero", np.zeros(2 + 3 * L)))
            starts.append(("small", np.concatenate([[0.2, 0.0], np.array([0.2, 0.0, 0.0])])))
            starts.append(("theta_neg", np.concatenate([[0.3, -0.5], np.array([0.2, 0.0, 0.0])])))
            starts.append(("theta_pos", np.concatenate([[0.3, 0.5], np.array([0.2, 0.0, 0.0])])))
        else:
            starts.append(("zero", np.zeros(1 + 3 * L)))
            starts.append(("small_linear", np.concatenate([[0.2], np.array([0.2, 0.0, 0.0])])))
            starts.append(("small_c3", np.concatenate([[0.2], np.array([0.2, 0.0, 0.25])])))
    else:
        gamma, theta, cs = _shell_pieces(method, np.asarray(prev["x"]), L - 1)
        cs_u = unwrap_schedule(cs, periods)
        if L == 2:
            row = cs_u[0]
            for off in (0.0, 0.2, -0.2, 0.1):
                c = np.vstack([row, row])
                c[1, 0] = row[0] - off
                starts.append((f"dup_c1_shift_{off}", _pack(gamma, theta, c, with_theta)))
            for dg in (0.05, -0.05):
                c = np.vstack([row, row])
                c[1, 0] = row[0] - 0.2
                starts.append((f"dup_gamma_{dg}", _pack(gamma + dg, theta, c, with_theta)))
            if with_theta and theta is not None:
                c = np.vstack([row, row])
                for dth in (0.15, -0.15):
                    starts.append((f"dup_theta_{dth}", _pack(gamma, theta + dth, c, True)))
        else:
            kind = "linear" if L == 3 else "cubic"
            c_new = np.column_stack(
                [resample_schedule(cs_u[:, j], L, kind) for j in range(3)]
            )
            starts.append((f"interp_{kind}", _pack(gamma, theta, c_new, with_theta)))
            for scale in (0.05, 0.15):
                noise = rng.normal(scale=scale, size=c_new.shape)
                starts.append(
                    (
                        f"interp_noise_{scale}",
                        _pack(gamma, theta, c_new + noise, with_theta),
                    )
                )
            # A raw (not unwrapped) interpolation, if the branch choice mattered.
            if not np.allclose(cs, cs_u):
                c_raw = np.column_stack(
                    [resample_schedule(cs[:, j], L, kind) for j in range(3)]
                )
                starts.append((f"interp_raw_{kind}", _pack(gamma, theta, c_raw, with_theta)))

    native = siblings.get("native_D")
    shared = siblings.get("shared_qaoa")
    if method == "native_D" and shared is not None:
        gamma, _th, cs = _shell_pieces("shared_qaoa", np.asarray(shared["x"]), L)
        c = np.zeros((L, 3))
        c[:, 0] = cs[:, 0]
        starts.append(("warm_shared_linear", _pack(gamma, None, c, False)))
    if with_theta and native is not None:
        gamma, _th, cs = _shell_pieces("native_D", np.asarray(native["x"]), L)
        starts.append(("warm_native_theta0", _pack(gamma, 0.0, cs, True)))
    if with_theta and shared is not None:
        gamma, _th, cs = _shell_pieces("shared_qaoa", np.asarray(shared["x"]), L)
        c = np.zeros((L, 3))
        c[:, 0] = cs[:, 0]
        starts.append(("warm_shared_theta0", _pack(gamma, 0.0, c, True)))

    for i in range(n_fresh):
        c = rng.uniform(-0.7, 0.7, size=(L, 3))
        gamma = float(rng.uniform(-0.8, 0.8))
        if with_theta:
            theta = float(rng.uniform(-1.0, 1.0))
            starts.append((f"fresh_{i}", _pack(gamma, theta, c, True)))
        else:
            starts.append((f"fresh_{i}", _pack(gamma, None, c, False)))
    return starts


def pack_qaoa_shared(gamma: float, betas) -> np.ndarray:
    betas = np.asarray(betas, dtype=np.float64).reshape(-1)
    return np.concatenate([[float(gamma)], betas])


def _pack(gamma: float, theta: float | None, cs: np.ndarray, with_theta: bool) -> np.ndarray:
    return pack_shell(gamma, theta, cs, with_theta)


def _l1_screening_points(method: str, rng: np.random.Generator, siblings: dict | None) -> list[np.ndarray]:
    pts = []
    if method == "qaoa":
        gammas = np.linspace(-1.5, 1.5, 31)
        betas = np.linspace(-1.2, 1.2, 25)
        for g in gammas:
            for b in betas:
                pts.append(np.array([g, b], dtype=np.float64))
        return pts
    if method == "shared_qaoa":
        # Same 2-parameter family as QAOA at L=1. A shorter grid still polishes.
        gammas = np.linspace(-1.5, 1.5, 21)
        betas = np.linspace(-1.2, 1.2, 21)
        for g in gammas:
            for b in betas:
                pts.append(np.array([g, b], dtype=np.float64))
        return pts
    with_theta = method in THETA_METHODS
    # Structured axes plus a deterministic random cloud. Not a GS-informed design.
    gammas = (-0.8, -0.4, -0.15, 0.15, 0.4, 0.8)
    c1s = (-0.6, -0.2, 0.2, 0.6)
    c2s = (0.0, 0.35, -0.35)
    c3s = (0.0, 0.35, -0.35)
    thetas = (0.0, -0.6, 0.6, -1.2, 1.2) if with_theta else (None,)
    # Full cartesian product is large; take a strided subset plus the axes.
    acc = []
    for ig, g in enumerate(gammas):
        for it, th in enumerate(thetas):
            for i1, c1 in enumerate(c1s):
                for i2, c2 in enumerate(c2s):
                    for i3, c3 in enumerate(c3s):
                        if (ig + it + i1 + i2 + i3) % 2 == 1 and not (
                            c2 == 0.0 and c3 == 0.0
                        ):
                            continue
                        if with_theta:
                            acc.append(np.array([g, th, c1, c2, c3], dtype=np.float64))
                        else:
                            acc.append(np.array([g, c1, c2, c3], dtype=np.float64))
    pts.extend(acc)
    n_cloud = 240 if with_theta else 180
    for _ in range(n_cloud):
        if with_theta:
            pts.append(
                np.array(
                    [
                        rng.uniform(-1.0, 1.0),
                        rng.uniform(-1.5, 1.5),
                        rng.uniform(-1.0, 1.0),
                        rng.uniform(-0.8, 0.8),
                        rng.uniform(-0.8, 0.8),
                    ],
                    dtype=np.float64,
                )
            )
        else:
            pts.append(
                np.array(
                    [
                        rng.uniform(-1.0, 1.0),
                        rng.uniform(-1.0, 1.0),
                        rng.uniform(-0.8, 0.8),
                        rng.uniform(-0.8, 0.8),
                    ],
                    dtype=np.float64,
                )
            )
    if method == "native_D" and siblings and siblings.get("qaoa") is not None:
        g, b = _qaoa_vectors(np.asarray(siblings["qaoa"]["x"]), 1)
        pts.append(np.array([float(g[0]), float(b[0]), 0.0, 0.0], dtype=np.float64))
    if with_theta and siblings and siblings.get("native_D") is not None:
        g, _th, cs = _shell_pieces("native_D", np.asarray(siblings["native_D"]["x"]), 1)
        pts.append(np.array([g, 0.0, cs[0, 0], cs[0, 1], cs[0, 2]], dtype=np.float64))
    return pts


def _diverse_top(xs: list[np.ndarray], energies: list[float], k: int, min_dist: float):
    order = np.argsort(energies)
    chosen = []
    for i in order:
        x = xs[int(i)]
        if any(np.linalg.norm(x - y) < min_dist for y in chosen):
            continue
        chosen.append(x)
        if len(chosen) >= k:
            break
    return chosen


def optimize_method(
    problem: Problem,
    method: str,
    L: int,
    prev: dict | None,
    siblings: dict | None,
    periods: list[float | None],
    eval_cap: int | None = None,
    n_fresh: int = 8,
    extra_starts: list[tuple[str, np.ndarray]] | None = None,
    seed_tag: int = 1,
    do_screen: bool = True,
) -> dict:
    """Minimize energy. Returns the best-energy point and the per-start spread.

    Fidelity is computed after each local solve for the record. It does not
    choose the retained point.
    """
    t0 = time.perf_counter()
    cap = int(EVAL_CAP.get(L, 7000) if eval_cap is None else eval_cap)
    bounds = bounds_for(method, L)
    rng = opt_rng(problem.n, L, method, seed_tag)
    n_eval = 0
    best_E = float("inf")
    best_x = None
    best_label = None
    best_trace: list[float] = []
    best_F_seen = -1.0
    best_F_E = None
    best_F_x = None
    best_F_label = None
    start_rows = []
    screened = 0

    def consider(x, E, F, label, trace):
        nonlocal best_E, best_x, best_label, best_trace
        nonlocal best_F_seen, best_F_E, best_F_x, best_F_label
        if E < best_E:
            best_E = float(E)
            best_x = np.array(x, dtype=np.float64, copy=True)
            best_label = label
            best_trace = list(trace)
        if F > best_F_seen:
            best_F_seen = float(F)
            best_F_E = float(E)
            best_F_x = np.array(x, dtype=np.float64, copy=True)
            best_F_label = label

    def eval_energy(x):
        nonlocal n_eval
        E, psi = energy_only(problem, method, x, L)
        n_eval += 1
        return E, psi

    # Bootstrap and same-depth simpler-method warms run before the screen so a
    # nested ansatz is not cut off from its own theta=0 point.
    seeds = build_starts(
        method, L, prev, siblings, periods, rng, n_fresh=0 if L == 1 else n_fresh
    )
    if extra_starts:
        seeds = list(extra_starts) + seeds
    if L == 1 and do_screen:
        screen_rng = opt_rng(problem.n, L, method, seed_tag + 1)
        cloud = _l1_screening_points(method, screen_rng, siblings)
        xs = []
        Es = []
        screen_budget = max(0, cap // 2 - n_eval)
        for x in cloud:
            if screened >= screen_budget:
                break
            x = _clip(x, bounds)
            E, _psi = eval_energy(x)
            xs.append(x)
            Es.append(E)
            screened += 1
        seeds.extend(("screen_top", x) for x in _diverse_top(xs, Es, k=8, min_dist=0.08))
        if xs:
            pick = opt_rng(problem.n, L, method, seed_tag + 2).choice(
                len(xs), size=min(4, len(xs)), replace=False
            )
            for j, i in enumerate(pick):
                seeds.append((f"screen_rand_{j}", xs[int(i)]))
    if L == 1:
        fresh_rng = opt_rng(problem.n, L, method, seed_tag + 3)
        seeds.extend(build_starts(method, L, None, siblings, periods, fresh_rng, n_fresh=n_fresh))

    seen = set()
    unique_seeds = []
    for label, x0 in seeds:
        x0 = _clip(np.asarray(x0, dtype=np.float64), bounds)
        key = tuple(np.round(x0, 6))
        if key in seen:
            continue
        seen.add(key)
        unique_seeds.append((label, x0))

    for label, x0 in unique_seeds:
        remaining = cap - n_eval
        if remaining < 20:
            start_rows.append({"label": label, "skipped": True, "reason": "eval cap"})
            break
        this_trace: list[float] = []
        cache = {"x": None}
        local_best_E = float("inf")
        local_best_x = np.array(x0, copy=True)

        def fun(x, _label=label):
            nonlocal n_eval, local_best_E, local_best_x
            if n_eval >= cap:
                return float(local_best_E if np.isfinite(local_best_E) else 0.0)
            E, g, _psi = energy_and_grad(problem, method, x, L)
            n_eval += 1
            cache["x"] = np.array(x, copy=True)
            cache["g"] = g
            cache["E"] = E
            this_trace.append(float(E))
            if E < local_best_E:
                local_best_E = float(E)
                local_best_x = np.array(x, copy=True)
            return float(E)

        def jac(x):
            nonlocal n_eval, local_best_E, local_best_x
            if cache["x"] is not None and np.allclose(cache["x"], x, atol=1e-12, rtol=0):
                return cache["g"]
            if n_eval >= cap:
                return np.zeros_like(x)
            E, g, _psi = energy_and_grad(problem, method, x, L)
            n_eval += 1
            cache["x"] = np.array(x, copy=True)
            cache["g"] = g
            this_trace.append(float(E))
            if E < local_best_E:
                local_best_E = float(E)
                local_best_x = np.array(x, copy=True)
            return g

        maxfun = int(min(LBFGS_MAXFUN, max(20, remaining)))
        try:
            res = minimize(
                fun,
                x0,
                method="L-BFGS-B",
                jac=jac,
                bounds=bounds,
                options={
                    "maxiter": LBFGS_MAXITER,
                    "maxfun": maxfun,
                    "ftol": 1e-14,
                    "gtol": 1e-8,
                    "maxls": 30,
                },
            )
            x_fin = np.array(local_best_x, copy=True)
            success = bool(res.success)
            nit = int(res.nit)
            message = str(res.message)
        except Exception as exc:  # numerical failures stay in the log, not silent
            x_fin = np.array(local_best_x, copy=True)
            success = False
            nit = 0
            message = f"{type(exc).__name__}: {exc}"
        E_fin, psi = eval_energy(x_fin)
        # eval_energy already counted. If the cap forced a dummy fun value,
        # this recomputation is the one we keep.
        F_fin = fidelity_of_state(problem, psi)
        consider(x_fin, E_fin, F_fin, label, this_trace)
        start_rows.append(
            {
                "label": label,
                "E": E_fin,
                "F": F_fin,
                "success": success,
                "nit": nit,
                "n_trace": len(this_trace),
                "message": message,
                "skipped": False,
            }
        )

    # One widened-bound polish if the energy winner sits on a bound.
    bound_hit = []
    widened = False
    if best_x is not None:
        bound_hit = _on_bound(best_x, bounds)
        if bound_hit and n_eval < cap - 20:
            wide = []
            for lo, hi in bounds:
                span = hi - lo
                wide.append((lo - 0.5 * span, hi + 0.5 * span))
            x0 = best_x.copy()
            this_trace = []
            cache = {"x": None}
            local_best_E = float("inf")
            local_best_x = np.array(x0, copy=True)

            def fun(x):
                nonlocal n_eval, local_best_E, local_best_x
                E, g, _psi = energy_and_grad(problem, method, x, L)
                n_eval += 1
                cache["x"] = np.array(x, copy=True)
                cache["g"] = g
                this_trace.append(float(E))
                if E < local_best_E:
                    local_best_E = float(E)
                    local_best_x = np.array(x, copy=True)
                return float(E)

            def jac(x):
                nonlocal n_eval, local_best_E, local_best_x
                if cache["x"] is not None and np.allclose(cache["x"], x, atol=1e-12, rtol=0):
                    return cache["g"]
                E, g, _psi = energy_and_grad(problem, method, x, L)
                n_eval += 1
                cache["x"] = np.array(x, copy=True)
                cache["g"] = g
                this_trace.append(float(E))
                if E < local_best_E:
                    local_best_E = float(E)
                    local_best_x = np.array(x, copy=True)
                return g

            res = minimize(
                fun,
                x0,
                method="L-BFGS-B",
                jac=jac,
                bounds=wide,
                options={"maxiter": 60, "maxfun": min(180, cap), "ftol": 1e-14, "gtol": 1e-8},
            )
            x_fin = np.array(local_best_x, dtype=np.float64, copy=True)
            E_fin, psi = eval_energy(x_fin)
            F_fin = fidelity_of_state(problem, psi)
            consider(x_fin, E_fin, F_fin, "widen_bound", this_trace)
            start_rows.append(
                {
                    "label": "widen_bound",
                    "E": E_fin,
                    "F": F_fin,
                    "success": bool(res.success),
                    "nit": int(res.nit),
                    "message": str(res.message),
                    "skipped": False,
                }
            )
            widened = True
            bound_hit = _on_bound(x_fin, wide)

    if best_x is None:
        raise RuntimeError(f"no start completed for {method} L={L}")

    # Replay so the stored energy matches an independent forward pass.
    E_check, _g, psi = energy_and_grad(problem, method, best_x, L)
    F_check = fidelity_of_state(problem, psi)
    finished = [r for r in start_rows if not r.get("skipped")]
    energies = np.array([r["E"] for r in finished], dtype=np.float64)
    gammas, thetas, cs = unpack(method, best_x, L)
    elapsed = time.perf_counter() - t0
    return {
        "method": method,
        "L": L,
        "n": problem.n,
        "x": best_x,
        "E": float(E_check),
        "F": float(F_check),
        "init_label": best_label,
        "n_eval": int(n_eval),
        "eval_cap": cap,
        "n_starts": len(finished),
        "n_screen": screened,
        "seconds": elapsed,
        "restart_E_min": float(energies.min()) if energies.size else None,
        "restart_E_median": float(np.median(energies)) if energies.size else None,
        "restart_E_max": float(energies.max()) if energies.size else None,
        "restart_E_std": float(energies.std()) if energies.size else None,
        "bound_indices": bound_hit,
        "widened_bounds": widened,
        "best_F_diagnostic": {
            "F": best_F_seen,
            "E": best_F_E,
            "label": best_F_label,
            "same_as_best_E": bool(
                best_F_x is not None and np.allclose(best_F_x, best_x, atol=1e-8)
            ),
            "promoted": False,
        },
        "gamma": gammas.tolist(),
        "theta": None if method not in THETA_METHODS else float(thetas[0]),
        "shell": cs.tolist(),
        "beta": cs[:, 0].tolist() if method in ("qaoa", "shared_qaoa") else None,
        "trace_best_start": _downsample(best_trace, 200),
        "starts": finished,
        "periods_used_for_interp": periods,
    }


def _downsample(values: list[float], limit: int) -> list[float]:
    if len(values) <= limit:
        return [float(v) for v in values]
    idx = np.linspace(0, len(values) - 1, limit).astype(int)
    return [float(values[int(i)]) for i in idx]
