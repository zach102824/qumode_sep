"""Budgets were fixed before the N=12 search.

Homotopy minimizes E + lambda * sum (gamma_l - mean)^2, then projects onto one
gamma and polishes. The independent-gamma stage is an initializer. A weak
relaxed optimum is not a certified ceiling and is not a legal candidate.
"""
from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize

import thread_env  # noqa: F401
from ansatz import (
    legal_energy_grad,
    legal_state,
    legal_to_relaxed,
    n_params_legal,
    n_params_relaxed,
    n_params_theta0,
    project_relaxed_to_legal,
    qaoa_energy_grad,
    qaoa_pack,
    qaoa_state,
    relaxed_energy_grad,
    relaxed_state,
    theta0_energy_grad,
    unpack_legal,
    unpack_relaxed,
)

# Pre-registered search sizes. Do not retune after seeing energies.
BUDGET = {
    "legal_starts": 16,
    "legal_maxfun": 220,
    "l1_screen": 1000,
    "l1_keep": 20,
    "homotopy_trajectories": 5,
    "lambdas": (0.0, 0.03, 0.3, 3.0, 30.0, 300.0),
    "lambda_maxfun": (180, 60, 60, 60, 60, 50),
    "polish_maxfun": 220,
    "theta0_starts": 12,
    "theta0_maxfun": 180,
    "random_starts": 8,
    "random_maxfun": 160,
    "qaoa_grid_b": 81,
    "qaoa_grid_g": 121,
    "qaoa_g_lo": -2.5,
    "qaoa_g_hi": 2.5,
    "qaoa_starts_low": 20,
    "qaoa_starts_high": 12,
    "qaoa_maxfun_low": 220,
    "qaoa_maxfun_high": 160,
    "workers": 4,
}

_MODEL = None


def _init_worker(n, J, B, fk_h, fk_rand):
    global _MODEL
    os.environ["OMP_NUM_THREADS"] = "1"
    from ansatz import Model

    _MODEL = Model(int(n), np.asarray(J), float(B), np.asarray(fk_h))
    _MODEL.fk_H = np.asarray(fk_h, dtype=np.float64)
    _MODEL.fk_rand = np.asarray(fk_rand, dtype=np.float64)


def _set_feature(name: str):
    _MODEL.fk = _MODEL.fk_H if name == "fk" else _MODEL.fk_rand


def _lbfgs(fun, x0, maxfun: int):
    best = {"E": np.inf, "x": np.asarray(x0, dtype=np.float64).copy()}
    nfev = {"n": 0}

    def wrapped(x):
        val, grad = fun(x)
        nfev["n"] += 1
        if val < best["E"]:
            best["E"] = float(val)
            best["x"] = np.array(x, dtype=np.float64, copy=True)
        return float(val), np.asarray(grad, dtype=np.float64)

    res = minimize(
        wrapped,
        np.asarray(x0, dtype=np.float64),
        method="L-BFGS-B",
        jac=True,
        options={"maxfun": int(maxfun), "maxiter": int(maxfun), "ftol": 1e-15, "gtol": 1e-8, "maxls": 30},
    )
    return best["x"], float(best["E"]), int(getattr(res, "nfev", nfev["n"])), bool(res.success), str(res.message)


def _eval_kind(kind: str, L: int, x: np.ndarray, lam: float, feature: str):
    _set_feature(feature)
    if kind == "legal":
        return legal_energy_grad(_MODEL, L, x)
    if kind == "theta0":
        return theta0_energy_grad(_MODEL, L, x)
    if kind == "relaxed":
        E, pen, g = relaxed_energy_grad(_MODEL, L, x, lam)
        return E + pen, g
    if kind == "qaoa":
        L = x.size // 2
        E, gg, gb = qaoa_energy_grad(_MODEL, x[:L], x[L:])
        return E, np.concatenate([gg, gb])
    raise KeyError(kind)


def _polish(task: dict) -> dict:
    L = int(task["L"])
    kind = task["kind"]
    feature = task.get("feature", "fk")
    lam = float(task.get("lam", 0.0))

    def fun(x):
        return _eval_kind(kind, L, x, lam, feature)

    try:
        x, _tracked, nfev, success, message = _lbfgs(fun, task["x0"], int(task["maxfun"]))
        # Re-evaluate pure energy (no penalty) at the tracked point.
        _set_feature(feature)
        if kind == "relaxed":
            E, _pen, _g = relaxed_energy_grad(_MODEL, L, x, 0.0)
        elif kind == "legal":
            E, _g = legal_energy_grad(_MODEL, L, x)
        elif kind == "theta0":
            E, _g = theta0_energy_grad(_MODEL, L, x)
        else:
            E, _gg, _gb = qaoa_energy_grad(_MODEL, x[:L], x[L:])
        return {
            "index": int(task["index"]),
            "tag": task["tag"],
            "x": x.tolist(),
            "E": float(E),
            "nfev": nfev,
            "success": success,
            "message": message,
            "error": None,
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "index": int(task["index"]),
            "tag": task["tag"],
            "x": list(map(float, np.asarray(task["x0"]).reshape(-1))),
            "E": None,
            "nfev": 0,
            "success": False,
            "message": "exception",
            "error": repr(exc),
        }


def _homotopy(task: dict) -> dict:
    """One trajectory: anneal lambda, project, polish. All evals are charged."""
    L = int(task["L"])
    feature = task.get("feature", "fk")
    x = np.asarray(task["x0"], dtype=np.float64)
    nfev = 0
    stages = []
    relaxed_E0 = None
    try:
        for lam, maxfun in zip(task["lambdas"], task["lambda_maxfun"]):
            def fun(z, lam=lam):
                return _eval_kind("relaxed", L, z, float(lam), feature)

            x, _penE, n_this, success, message = _lbfgs(fun, x, int(maxfun))
            nfev += n_this
            _set_feature(feature)
            E, _pen, _g = relaxed_energy_grad(_MODEL, L, x, 0.0)
            gammas, theta, _shell = unpack_relaxed(L, x)
            if relaxed_E0 is None:
                relaxed_E0 = float(E)
            stages.append(
                {
                    "lambda": float(lam),
                    "E": float(E),
                    "gamma_std": float(np.std(gammas)),
                    "gamma_mean": float(np.mean(gammas)),
                    "nfev": n_this,
                    "success": success,
                    "message": message,
                }
            )
        legal0 = project_relaxed_to_legal(L, x)

        def fun_legal(z):
            return _eval_kind("legal", L, z, 0.0, feature)

        x_legal, _e, n_pol, success, message = _lbfgs(fun_legal, legal0, int(task["polish_maxfun"]))
        nfev += n_pol
        _set_feature(feature)
        E, _g = legal_energy_grad(_MODEL, L, x_legal)
        gamma, theta, shell = unpack_legal(L, x_legal)
        return {
            "index": int(task["index"]),
            "tag": task["tag"],
            "x": x_legal.tolist(),
            "E": float(E),
            "nfev": nfev,
            "nfev_relaxed": nfev - n_pol,
            "nfev_polish": n_pol,
            "relaxed_E_after_lambda0": relaxed_E0,
            "gamma": gamma,
            "theta": theta,
            "gamma_spread_before_projection": stages[-1]["gamma_std"] if stages else None,
            "stages": stages,
            "success": success,
            "message": message,
            "error": None,
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "index": int(task["index"]),
            "tag": task["tag"],
            "x": None,
            "E": None,
            "nfev": nfev,
            "error": repr(exc),
            "stages": stages,
            "relaxed_E_after_lambda0": relaxed_E0,
        }


def _energy_batch(tasks: list) -> list:
    out = []
    for task in tasks:
        _set_feature(task.get("feature", "fk"))
        kind = task["kind"]
        x = np.asarray(task["x0"], dtype=np.float64)
        L = int(task["L"])
        if kind == "qaoa":
            psi = qaoa_state(_MODEL, x[:L], x[L:])
        elif kind == "legal":
            g, th, shell = unpack_legal(L, x)
            psi = legal_state(_MODEL, g, th, shell)
        elif kind == "theta0":
            from ansatz import unpack_theta0

            g, shell = unpack_theta0(L, x)
            psi = legal_state(_MODEL, g, 0.0, shell)
        else:
            gs, th, shell = unpack_relaxed(L, x)
            psi = relaxed_state(_MODEL, gs, th, shell)
        out.append((int(task["index"]), float(_MODEL.energy(psi))))
    return out


class Pool:
    def __init__(self, model, fk_rand: np.ndarray, workers: int):
        self.model = model
        self.workers = max(1, int(workers))
        self.ex = ProcessPoolExecutor(
            max_workers=self.workers,
            initializer=_init_worker,
            initargs=(model.n, model.J, model.B, model.fk, fk_rand),
        )

    def map_polish(self, tasks: list[dict]) -> list[dict]:
        if not tasks:
            return []
        return list(self.ex.map(_polish, tasks, chunksize=1))

    def map_homotopy(self, tasks: list[dict]) -> list[dict]:
        if not tasks:
            return []
        return list(self.ex.map(_homotopy, tasks, chunksize=1))

    def map_energy(self, tasks: list[dict]) -> np.ndarray:
        if not tasks:
            return np.zeros(0)
        chunks = np.array_split(np.arange(len(tasks)), self.workers * 4)
        batches = []
        for chunk in chunks:
            if len(chunk) == 0:
                continue
            batches.append([tasks[int(i)] for i in chunk])
        energies = np.empty(len(tasks))
        for batch_out in self.ex.map(_energy_batch, batches):
            for index, E in batch_out:
                energies[index] = E
        return energies

    def close(self):
        self.ex.shutdown(wait=True)


def _dedup(starts, decimals=6):
    seen = set()
    out = []
    for tag, x in starts:
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        if not np.all(np.isfinite(x)):
            continue
        key = tuple(np.round(x, decimals))
        if key in seen:
            continue
        seen.add(key)
        out.append((tag, x))
    return out


def _summarize(rows: list[dict]) -> dict:
    finished = [r for r in rows if r.get("E") is not None and np.isfinite(r["E"])]
    if not finished:
        return {"ok": False, "error": "no finite start", "starts": rows, "nfev_sum": int(sum(r.get("nfev") or 0 for r in rows))}
    finished.sort(key=lambda r: (r["E"], r["index"]))
    best = finished[0]
    energies = np.array([r["E"] for r in finished], dtype=np.float64)
    return {
        "ok": True,
        "best": best,
        "x": best["x"],
        "E_best": float(best["E"]),
        "E_median": float(np.median(energies)),
        "E_std": float(np.std(energies)),
        "n_starts": len(rows),
        "n_finished": len(finished),
        "nfev_sum": int(sum(r.get("nfev") or 0 for r in rows)),
        "best_tag": best.get("tag"),
        "starts": [
            {"tag": r.get("tag"), "E": r.get("E"), "nfev": r.get("nfev"), "success": r.get("success"), "error": r.get("error")}
            for r in rows
        ],
    }


def resample(values, L_new: int, mode: str) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if L_new <= 1:
        return values[:1].copy()
    if values.size == 1:
        return np.full(L_new, float(values[0]))
    s_old = np.linspace(0.0, 1.0, values.size)
    s_new = np.linspace(0.0, 1.0, L_new)
    if mode == "cubic" and values.size >= 3:
        return np.asarray(CubicSpline(s_old, values, bc_type="natural")(s_new), dtype=np.float64)
    return np.interp(s_new, s_old, values).astype(np.float64)


def _greedy(xs, energies, k, min_dist):
    order = np.argsort(energies)
    kept = []
    for i in order:
        p = xs[int(i)]
        if all(np.linalg.norm(p - xs[j]) >= min_dist for j in kept):
            kept.append(int(i))
        if len(kept) >= k:
            break
    return kept


def fourier_qaoa(rng, L: int) -> np.ndarray:
    q = min(3, L)
    u = rng.normal(scale=0.35, size=q)
    v = rng.normal(scale=0.25, size=q)
    g = np.zeros(L)
    b = np.zeros(L)
    for i in range(1, L + 1):
        for k in range(1, q + 1):
            arg = (k - 0.5) * (i - 0.5) * np.pi / L
            g[i - 1] += u[k - 1] * np.sin(arg)
            b[i - 1] += v[k - 1] * np.cos(arg)
    return qaoa_pack(g, b)


def qaoa_starts(L: int, prev, B: float, rng, cap: int):
    starts = []
    if prev is not None:
        prev = np.asarray(prev, dtype=np.float64)
        g_prev, b_prev = prev[: L - 1], prev[L - 1 :]
        off_b = 0.2 * abs(float(B))
        if L == 2:
            g1, b1 = float(g_prev[0]), float(b_prev[0])
            named = {
                "paper_shift_primary": qaoa_pack([g1, g1 + 0.2], [b1, b1 + off_b]),
                "paper_shift_opposite": qaoa_pack([g1, g1 - 0.2], [b1, b1 - off_b]),
                "duplicate": qaoa_pack([g1, g1], [b1, b1]),
            }
            for tag, x in named.items():
                starts.append((tag, x))
                starts.append((tag + "_tr", -x))
            base = named["paper_shift_primary"]
        else:
            mode = "linear" if L <= 3 else "cubic"
            base = qaoa_pack(resample(g_prev, L, mode), resample(b_prev, L, mode))
            starts.append((f"interp_{mode}", base))
            if mode == "cubic":
                starts.append(("interp_linear", qaoa_pack(resample(g_prev, L, "linear"), resample(b_prev, L, "linear"))))
        for scale in (0.03, 0.08, 0.2):
            starts.append((f"perturb_{scale}", base + rng.normal(scale=scale, size=base.size)))
    s = np.linspace(0.0, 1.0, L)
    for gmax, bmax in ((0.4, 0.3), (1.0, 0.5), (2.0, 0.8)):
        starts.append((f"adiab_{gmax}", qaoa_pack(gmax * s, bmax * (1.0 - s))))
    for i in range(4):
        starts.append((f"fourier_{i}", fourier_qaoa(rng, L)))
    for i in range(4):
        starts.append((f"random_{i}", qaoa_pack(rng.normal(scale=0.4, size=L), rng.uniform(0, np.pi / 2, size=L))))
    return _dedup(starts)[:cap]


def legal_starts(L, prev, qaoa_x, theta0_x, B, rng, cap, kind="legal"):
    """kind legal includes theta; theta0 does not."""
    starts = []

    def pack(gamma, theta, shell):
        shell = np.asarray(shell, dtype=np.float64).reshape(L, 3)
        if kind == "theta0":
            return np.concatenate([[float(gamma)], shell.reshape(-1)])
        return np.concatenate([[float(gamma), float(theta)], shell.reshape(-1)])

    if prev is not None:
        prev = np.asarray(prev, dtype=np.float64)
        if kind == "legal":
            gamma, theta, shell_prev = unpack_legal(L - 1, prev)
        else:
            gamma, shell_prev = float(prev[0]), prev[1:].reshape(L - 1, 3)
            theta = 0.0
        if L == 2:
            c = shell_prev[0]
            for i, second in enumerate(
                (
                    c,
                    c + np.array([0.08, 0, 0]),
                    c + np.array([0, 0.08, 0]),
                    c + np.array([0, 0, 0.08]),
                    c + np.array([0.05, -0.05, 0.05]),
                )
            ):
                starts.append((f"dup_{i}", pack(gamma, theta, np.vstack([c, second]))))
            starts.append(("dup_theta", pack(gamma, theta + 0.2, np.vstack([c, c]))))
            base = pack(gamma, theta, np.vstack([c, c]))
        else:
            mode = "linear" if L <= 3 else "cubic"
            shell = np.column_stack([resample(shell_prev[:, k], L, mode) for k in range(3)])
            base = pack(gamma, theta, shell)
            starts.append((f"interp_{mode}", base))
        for scale in (0.04, 0.12):
            starts.append((f"perturb_{scale}", base + rng.normal(scale=scale, size=base.size)))
    if qaoa_x is not None and L == len(qaoa_x) // 2:
        g = np.asarray(qaoa_x[:L], dtype=np.float64)
        b = np.asarray(qaoa_x[L:], dtype=np.float64)
        shell = np.zeros((L, 3))
        shell[:, 0] = b
        starts.append(("qaoa_c1", pack(float(np.mean(g)), 0.0, shell)))
    if theta0_x is not None and kind == "legal":
        g = float(theta0_x[0])
        shell = np.asarray(theta0_x[1:], dtype=np.float64).reshape(L, 3)
        starts.append(("from_theta0", pack(g, 0.15, shell)))
        starts.append(("from_theta0_zero", pack(g, 0.0, shell)))
    width = n_params_legal(L) if kind == "legal" else n_params_theta0(L)
    for i in range(4):
        x = rng.normal(scale=0.25, size=width)
        x[0] = rng.uniform(-1.5, 1.5)
        if kind == "legal":
            x[1] = rng.uniform(-1.0, 1.0)
        starts.append((f"random_{i}", x))
    return _dedup(starts)[:cap]


def homotopy_seeds(L, legal_prev, legal_here, theta0_x, rng):
    """Relaxed vectors. At least one seed is exactly legal (all gamma equal)."""
    seeds = []

    def from_legal(tag, x_legal):
        seeds.append((tag, legal_to_relaxed(L, x_legal)))

    if legal_here is not None:
        from_legal("from_plain", legal_here)
        x = np.asarray(legal_here, dtype=np.float64).copy()
        # Split gammas around the legal scalar. This is initialization only.
        g = float(x[0])
        gammas = g + np.linspace(-0.15, 0.15, L)
        theta = float(x[1])
        shell = x[2:]
        seeds.append(("split_plain", np.concatenate([gammas, [theta], shell])))
    elif legal_prev is not None:
        # Duplicate previous legal shell into L and split gammas.
        g, theta, shell_prev = unpack_legal(L - 1, legal_prev)
        if L == 2:
            shell = np.vstack([shell_prev[0], shell_prev[0]])
        else:
            mode = "linear" if L <= 3 else "cubic"
            shell = np.column_stack([resample(shell_prev[:, k], L, mode) for k in range(3)])
        x_legal = np.concatenate([[g, theta], shell.reshape(-1)])
        from_legal("from_prev_legal", x_legal)
        gammas = g + 0.1 * np.linspace(-1, 1, L)
        seeds.append(("split_prev", np.concatenate([gammas, [theta], shell.reshape(-1)])))
    if theta0_x is not None:
        g = float(theta0_x[0])
        shell = np.asarray(theta0_x[1:], dtype=np.float64)
        seeds.append(("from_theta0", np.concatenate([np.full(L, g), [0.2], shell])))
    width = n_params_relaxed(L)
    for i in range(2):
        x = rng.normal(scale=0.2, size=width)
        x[:L] = rng.normal(loc=0.4, scale=0.25, size=L)
        seeds.append((f"random_{i}", x))
    return _dedup(seeds)[: BUDGET["homotopy_trajectories"]]


def polish_many(pool: Pool, tasks: list[dict]) -> dict:
    rows = pool.map_polish(tasks)
    for i, row in enumerate(rows):
        row["index"] = i
    return _summarize(rows)


def make_tasks(kind, L, starts, maxfun, feature="fk"):
    return [
        {"kind": kind, "L": L, "index": i, "tag": tag, "x0": np.asarray(x, dtype=np.float64), "maxfun": maxfun, "feature": feature, "lam": 0.0}
        for i, (tag, x) in enumerate(starts)
    ]
