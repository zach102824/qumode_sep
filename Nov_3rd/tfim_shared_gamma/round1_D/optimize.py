"""Depth-bootstrapped energy minimization.

Objective is <H> only. Fidelity is not consulted here.
QAOA keeps independent (gamma_l, beta_l). Shared-gamma methods have one scalar gamma.
No split gamma and no per-layer generator coefficients.
"""
from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor

import thread_env  # noqa: F401
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize

from ansatz import energy_and_grad, final_state, n_params

# Evaluation budgets. N=6 is cheap, so it gets a denser L=1 grid and more starts.
# N=12 stays inside the same ladder with fewer starts. Both are energy-only.
BUDGET = {
    6: dict(
        grid_b=161,
        grid_g=241,
        g_lo=-3.0,
        g_hi=3.0,
        l1_screen=4000,
        l1_keep=48,
        higher_cap=48,
        maxfun=500,
        workers=4,
    ),
    12: dict(
        grid_b=121,
        grid_g=181,
        g_lo=-2.5,
        g_hi=2.5,
        l1_screen=3000,
        l1_keep=36,
        higher_cap=36,
        maxfun=400,
        workers=4,
    ),
}

_MODEL = None


def cfg_for(n: int) -> dict:
    return BUDGET[6 if n <= 6 else 12]


def _init_worker(n, J, B, hc):
    global _MODEL
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    from ansatz import Model

    _MODEL = Model(int(n), np.asarray(J, dtype=np.float64), float(B), np.asarray(hc, dtype=np.float64))


def _polish(task: dict) -> dict:
    method = task["method"]
    L = int(task["L"])
    maxfun = int(task["maxfun"])
    x0 = np.asarray(task["x0"], dtype=np.float64)
    best = {"E": np.inf, "x": x0.copy()}
    trace = []

    def fun(x):
        E, g = energy_and_grad(_MODEL, method, L, x)
        E = float(E)
        trace.append(E)
        if E < best["E"]:
            best["E"] = E
            best["x"] = np.array(x, dtype=np.float64, copy=True)
        return E, np.asarray(g, dtype=np.float64)

    try:
        res = minimize(
            fun,
            x0,
            method="L-BFGS-B",
            jac=True,
            options={
                "maxfun": maxfun,
                "maxiter": maxfun,
                "ftol": 1e-15,
                "gtol": 1e-8,
                "maxls": 40,
            },
        )
        # Recompute the gradient at the tracked best point. res.x can be a
        # line-search endpoint that is slightly worse than the best evaluation.
        E2, g2 = energy_and_grad(_MODEL, method, L, best["x"])
        return {
            "index": int(task["index"]),
            "tag": task["tag"],
            "x": best["x"].tolist(),
            "E": float(E2),
            "nfev": int(getattr(res, "nfev", len(trace))),
            "success": bool(res.success),
            "grad_norm": float(np.linalg.norm(g2)),
            "message": str(res.message),
            "trace": trace,
            "trace_len": len(trace),
            "trace_head": trace[:8],
            "trace_tail": trace[-8:],
            "error": None,
        }
    except Exception as exc:  # noqa: BLE001 — record and continue the ladder
        return {
            "index": int(task["index"]),
            "tag": task["tag"],
            "x": x0.tolist(),
            "E": None,
            "nfev": len(trace),
            "success": False,
            "grad_norm": None,
            "message": "exception",
            "trace": trace,
            "trace_len": len(trace),
            "trace_head": trace[:8],
            "trace_tail": trace[-8:],
            "error": repr(exc),
        }


def _energy_batch(tasks: list) -> list:
    out = []
    for task in tasks:
        E = float(_MODEL.energy(final_state(_MODEL, task["method"], int(task["L"]), np.asarray(task["x0"]))))
        out.append((int(task["index"]), E))
    return out


class Pool:
    def __init__(self, model, workers: int):
        self.model = model
        self.workers = max(1, int(workers))
        self.ex = ProcessPoolExecutor(
            max_workers=self.workers,
            initializer=_init_worker,
            initargs=(model.n, model.J, model.B, model.hc),
        )

    def map_polish(self, tasks: list[dict]) -> list[dict]:
        if not tasks:
            return []
        return list(self.ex.map(_polish, tasks))

    def map_energy(self, method: str, L: int, xs: list[np.ndarray]) -> np.ndarray:
        if not xs:
            return np.zeros(0)
        batches = []
        chunks = np.array_split(np.arange(len(xs)), self.workers * 4)
        for chunk in chunks:
            if len(chunk) == 0:
                continue
            batches.append(
                [
                    {"method": method, "L": L, "index": int(i), "x0": np.asarray(xs[i], dtype=np.float64)}
                    for i in chunk
                ]
            )
        energies = np.empty(len(xs), dtype=np.float64)
        for batch_out in self.ex.map(_energy_batch, batches):
            for index, E in batch_out:
                energies[index] = E
        return energies

    def close(self):
        self.ex.shutdown(wait=True)


def resample(values: np.ndarray, L_new: int, mode: str) -> np.ndarray:
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


def _dedup(starts: list[tuple[str, np.ndarray]], decimals: int = 6) -> list[tuple[str, np.ndarray]]:
    seen = set()
    out = []
    for tag, x in starts:
        key = tuple(np.round(np.asarray(x, dtype=np.float64), decimals))
        if key in seen or not np.all(np.isfinite(x)):
            continue
        seen.add(key)
        out.append((tag, np.asarray(x, dtype=np.float64)))
    return out


def _fourier_qaoa(rng: np.random.Generator, L: int, scale_g: float = 0.35, scale_b: float = 0.25) -> np.ndarray:
    """Zhou et al. Fourier seed, optimized afterwards in the free angle coordinates."""
    q = min(3, L)
    u = rng.normal(scale=scale_g, size=q)
    v = rng.normal(scale=scale_b, size=q)
    g = np.zeros(L)
    b = np.zeros(L)
    for i in range(1, L + 1):
        for k in range(1, q + 1):
            arg = (k - 0.5) * (i - 0.5) * np.pi / L
            g[i - 1] += u[k - 1] * np.sin(arg)
            b[i - 1] += v[k - 1] * np.cos(arg)
    return np.concatenate([g, b])


def _pack_qaoa(gammas, betas) -> np.ndarray:
    return np.concatenate([np.asarray(gammas, dtype=np.float64), np.asarray(betas, dtype=np.float64)])


def _unpack_qaoa(x, L):
    x = np.asarray(x, dtype=np.float64)
    return x[:L].copy(), x[L:].copy()


def qaoa_starts(L: int, prev: np.ndarray | None, B: float, rng: np.random.Generator, cap: int) -> list:
    starts = []
    if prev is None:
        return starts
    prev = np.asarray(prev, dtype=np.float64)
    g_prev, b_prev = _unpack_qaoa(prev, L - 1)
    off_b = 0.2 * abs(float(B))
    off_g = 0.2
    if L == 2:
        g1 = float(g_prev[0])
        b1 = float(b_prev[0])
        # Paper beta is the coefficient of H_B/J0 = B sum(transverse).
        # Our beta multiplies sum X, so beta_ours = beta_paper * B.
        # beta_paper,2 = beta_paper,1 - 0.2 implies beta_ours,2 = beta_ours,1 + 0.2*|B|.
        # gamma multiplies the same J0=1 Ising generator, so the +0.2 gamma offset is unchanged.
        named = {
            "paper_shift_primary": _pack_qaoa([g1, g1 + off_g], [b1, b1 + off_b]),
            "paper_shift_opposite": _pack_qaoa([g1, g1 - off_g], [b1, b1 - off_b]),
            "paper_shift_beta_only": _pack_qaoa([g1, g1], [b1, b1 + off_b]),
            "paper_shift_gamma_only": _pack_qaoa([g1, g1 + off_g], [b1, b1]),
            "duplicate": _pack_qaoa([g1, g1], [b1, b1]),
            "shift_0.1": _pack_qaoa([g1, g1 + 0.1], [b1, b1 + 0.5 * off_b]),
            "shift_0.3": _pack_qaoa([g1, g1 + 0.3], [b1, b1 + 1.5 * off_b]),
        }
        for tag, x in named.items():
            starts.append((tag, x))
            starts.append((tag + "_time_reverse", -x))
        base = named["paper_shift_primary"]
    else:
        mode = "linear" if L <= 3 else "cubic"
        g = resample(g_prev, L, mode)
        b = resample(b_prev, L, mode)
        base = _pack_qaoa(g, b)
        starts.append((f"interp_{mode}", base))
        if mode == "cubic":
            starts.append(("interp_linear_alt", _pack_qaoa(resample(g_prev, L, "linear"), resample(b_prev, L, "linear"))))
    for scale in (0.02, 0.05, 0.1, 0.2):
        for rep in range(3):
            noise = rng.normal(scale=scale, size=base.size)
            starts.append((f"perturb_{scale}_{rep}", base + noise))
    for i in range(8):
        starts.append((f"fourier_{i}", _fourier_qaoa(rng, L)))
    for i in range(10):
        g = rng.normal(scale=0.45, size=L) + np.linspace(-0.2, 0.5, L) * float(rng.normal())
        b = rng.uniform(0.0, np.pi / 2, size=L)
        starts.append((f"random_{i}", _pack_qaoa(g, b)))
    return _dedup(starts)[:cap]


def shared_starts(L, prev, qaoa_x, B, rng, cap):
    starts = []
    if prev is not None:
        prev = np.asarray(prev, dtype=np.float64)
        gamma = float(prev[0])
        betas = np.asarray(prev[1:], dtype=np.float64)
        if L == 2:
            b1 = float(betas[0])
            off_b = 0.2 * abs(float(B))
            for tag, b2, g in (
                ("duplicate", b1, gamma),
                ("beta_shift_primary", b1 + off_b, gamma),
                ("beta_shift_opposite", b1 - off_b, gamma),
                ("gamma_plus", b1, gamma + 0.05),
                ("gamma_minus", b1, gamma - 0.05),
            ):
                starts.append((tag, np.array([g, b1, b2], dtype=np.float64)))
            base = np.array([gamma, b1, b1 + off_b])
        else:
            mode = "linear" if L <= 3 else "cubic"
            b = resample(betas, L, mode)
            base = np.concatenate([[gamma], b])
            starts.append((f"interp_{mode}", base))
            for dg in (-0.1, 0.1):
                starts.append((f"gamma_shift_{dg}", np.concatenate([[gamma + dg], b])))
        for scale in (0.03, 0.08, 0.15, 0.3):
            for rep in range(2):
                starts.append((f"perturb_{scale}_{rep}", base + rng.normal(scale=scale, size=base.size)))
    if qaoa_x is not None:
        g, b = _unpack_qaoa(qaoa_x, L)
        starts.append(("qaoa_mean_gamma", np.concatenate([[float(np.mean(g))], b])))
        # Keep per-layer mixer/cost ratios where the QAOA cost angles are not tiny.
        safe = np.where(np.abs(g) > 1e-3, g, np.sign(g + 1e-12) * 1e-3)
        gamma = float(np.mean(g))
        starts.append(("qaoa_ratio_rescale", np.concatenate([[gamma], b * (gamma / safe)])))
    for i in range(8):
        starts.append(
            (
                f"random_{i}",
                np.concatenate([rng.normal(scale=0.5, size=1), rng.uniform(0, np.pi / 2, size=L)]),
            )
        )
    return _dedup(starts)[:cap]


def _shell_pack(gamma, shell) -> np.ndarray:
    return np.concatenate([[float(gamma)], np.asarray(shell, dtype=np.float64).reshape(-1)])


def shell_starts(method, L, prev, qaoa_x, shared_x, donor_x, B, rng, cap):
    """method is nov3_D, drop_c2, or drop_c3. prev/donor are that method's own vectors."""
    starts = []

    def project(full_shell):
        shell = np.array(full_shell, dtype=np.float64, copy=True)
        if method == "drop_c2":
            shell[:, 1] = 0.0
        elif method == "drop_c3":
            shell[:, 2] = 0.0
        return shell

    def pack_method(gamma, shell):
        shell = project(shell)
        if method == "nov3_D":
            return _shell_pack(gamma, shell)
        if method == "drop_c2":
            cols = np.column_stack([shell[:, 0], shell[:, 2]])
            return np.concatenate([[float(gamma)], cols.reshape(-1)])
        cols = np.column_stack([shell[:, 0], shell[:, 1]])
        return np.concatenate([[float(gamma)], cols.reshape(-1)])

    def prev_shell(x):
        # Decode a same-method vector at depth L-1 into (gamma, full shell).
        x = np.asarray(x, dtype=np.float64)
        if method == "nov3_D":
            return float(x[0]), x[1:].reshape(L - 1, 3)
        if method == "drop_c2":
            pairs = x[1:].reshape(L - 1, 2)
            shell = np.column_stack([pairs[:, 0], np.zeros(L - 1), pairs[:, 1]])
            return float(x[0]), shell
        pairs = x[1:].reshape(L - 1, 2)
        shell = np.column_stack([pairs[:, 0], pairs[:, 1], np.zeros(L - 1)])
        return float(x[0]), shell

    if prev is not None:
        gamma, shell_prev = prev_shell(prev)
        if L == 2:
            c = shell_prev[0]
            variants = [
                c.copy(),
                c + np.array([0.08, 0.0, 0.0]),
                c + np.array([-0.08, 0.0, 0.0]),
                c + np.array([0.0, 0.08, 0.0]),
                c + np.array([0.0, -0.08, 0.0]),
                c + np.array([0.0, 0.0, 0.08]),
                c + np.array([0.0, 0.0, -0.08]),
                c + np.array([0.05, 0.05, 0.05]),
                c + np.array([-0.05, 0.05, -0.05]),
                c + np.array([0.15, -0.1, 0.1]),
            ]
            for i, second in enumerate(variants):
                shell = np.vstack([c, second])
                starts.append((f"duplicate_{i}", pack_method(gamma, shell)))
            for dg, tag in ((0.05, "g+"), (-0.05, "g-"), (0.15, "g++"), (-0.15, "g--")):
                starts.append((f"duplicate_{tag}", pack_method(gamma + dg, np.vstack([c, c]))))
            base = pack_method(gamma, np.vstack([c, c]))
        else:
            mode = "linear" if L <= 3 else "cubic"
            cols = [resample(shell_prev[:, k], L, mode) for k in range(3)]
            shell = np.column_stack(cols)
            base = pack_method(gamma, shell)
            starts.append((f"interp_{mode}", base))
            if mode == "cubic":
                cols_l = [resample(shell_prev[:, k], L, "linear") for k in range(3)]
                starts.append(("interp_linear_alt", pack_method(gamma, np.column_stack(cols_l))))
            for dg in (-0.1, 0.1):
                starts.append((f"gamma_shift_{dg}", pack_method(gamma + dg, shell)))
        for scale in (0.03, 0.08, 0.15, 0.3):
            for rep in range(2):
                noise = rng.normal(scale=scale, size=base.size)
                starts.append((f"perturb_{scale}_{rep}", base + noise))
    if qaoa_x is not None:
        g, b = _unpack_qaoa(qaoa_x, L)
        gamma = float(np.mean(g))
        shell = np.zeros((L, 3))
        shell[:, 0] = b
        starts.append(("qaoa_c1_only", pack_method(gamma, shell)))
        safe = np.where(np.abs(g) > 1e-3, g, np.sign(g + 1e-12) * 1e-3)
        shell_r = np.zeros((L, 3))
        shell_r[:, 0] = b * (gamma / safe)
        starts.append(("qaoa_ratio_c1", pack_method(gamma, shell_r)))
    if shared_x is not None and method != "shared_beta":
        shared_x = np.asarray(shared_x, dtype=np.float64)
        gamma = float(shared_x[0])
        shell = np.zeros((L, 3))
        shell[:, 0] = shared_x[1:]
        starts.append(("shared_beta_c1", pack_method(gamma, shell)))
    if donor_x is not None and method != "nov3_D":
        # donor is full D at this same depth: zero the dropped coefficient and re-optimize.
        donor_x = np.asarray(donor_x, dtype=np.float64)
        gamma = float(donor_x[0])
        shell = donor_x[1:].reshape(L, 3)
        starts.append(("projected_D", pack_method(gamma, shell)))
    width = n_params(method, L)
    for i in range(8):
        x = rng.normal(scale=0.25, size=width)
        x[0] = rng.uniform(-1.5, 1.5)
        starts.append((f"random_{i}", x))
    return _dedup(starts)[:cap]


def _greedy(points: list[np.ndarray], energies: np.ndarray, k: int, min_dist: float) -> list[int]:
    order = np.argsort(energies)
    kept = []
    for i in order:
        p = points[int(i)]
        if all(np.linalg.norm(p - points[j]) >= min_dist for j in kept):
            kept.append(int(i))
        if len(kept) >= k:
            break
    return kept


def polish_starts(pool: Pool, method: str, L: int, starts: list, maxfun: int) -> dict:
    tasks = []
    for i, (tag, x) in enumerate(starts):
        tasks.append(
            {
                "method": method,
                "L": L,
                "index": i,
                "tag": tag,
                "x0": np.asarray(x, dtype=np.float64),
                "maxfun": maxfun,
                "keep_trace": False,
            }
        )
    rows = pool.map_polish(tasks)
    finished = [r for r in rows if r["E"] is not None and np.isfinite(r["E"])]
    if not finished:
        return {
            "ok": False,
            "error": "no finite start",
            "starts": rows,
        }
    finished.sort(key=lambda r: (r["E"], r["grad_norm"] if r["grad_norm"] is not None else 1e9, r["index"]))
    best = finished[0]
    # One traced rerun is unnecessary: recompute trace by a single kept polish of the winner
    # only if we did not store it. Store the winner's x; a short confirmation eval is done by caller.
    energies = np.array([r["E"] for r in finished], dtype=np.float64)
    return {
        "ok": True,
        "best": best,
        "starts": [
            {
                "tag": r["tag"],
                "x": r["x"],
                "E": r["E"],
                "nfev": r["nfev"],
                "success": r["success"],
                "grad_norm": r["grad_norm"],
                "message": r["message"],
                "error": r["error"],
                "trace_len": r["trace_len"],
                "trace_head": r["trace_head"],
                "trace_tail": r["trace_tail"],
            }
            for r in sorted(rows, key=lambda r: r["index"])
        ],
        "E_best": float(best["E"]),
        "E_median": float(np.median(energies)),
        "E_mean": float(np.mean(energies)),
        "E_std": float(np.std(energies)),
        "n_starts": len(starts),
        "n_finished": len(finished),
        "nfev_sum": int(sum(r["nfev"] for r in rows)),
        "x": best["x"],
        "grad_norm": best["grad_norm"],
        "best_tag": best["tag"],
        "best_trace": best.get("trace") or [],
    }


def optimize_l1_qaoa(pool: Pool, B: float, rng: np.random.Generator, conf: dict) -> dict:
    betas = np.linspace(0.0, 0.5 * np.pi, conf["grid_b"], endpoint=False)
    gammas = np.linspace(conf["g_lo"], conf["g_hi"], conf["grid_g"])
    xs = [_pack_qaoa([g], [b]) for g in gammas for b in betas]
    energies = pool.map_energy("qaoa", 1, xs)
    # Boundary extension if the best gamma sits on the edge of the window.
    grid_note = {"extended": False}
    best_flat = int(np.argmin(energies))
    g_index = best_flat // len(betas)
    if g_index <= 1 or g_index >= len(gammas) - 2:
        span = conf["g_hi"] - conf["g_lo"]
        if g_index <= 1:
            g_lo, g_hi = conf["g_lo"] - 0.5 * span, conf["g_lo"]
        else:
            g_lo, g_hi = conf["g_hi"], conf["g_hi"] + 0.5 * span
        extra_g = np.linspace(g_lo, g_hi, max(20, conf["grid_g"] // 2))
        extra_x = [_pack_qaoa([g], [b]) for g in extra_g for b in betas]
        extra_e = pool.map_energy("qaoa", 1, extra_x)
        xs = xs + extra_x
        energies = np.concatenate([energies, extra_e])
        grid_note = {"extended": True, "extra_lo": float(g_lo), "extra_hi": float(g_hi), "n_extra": len(extra_x)}
    points = [np.asarray(x) for x in xs]
    keep = _greedy(points, energies, k=12, min_dist=0.08)
    starts = [(f"grid_{i}", points[i]) for i in keep]
    for i in range(8):
        starts.append(("random_box", _pack_qaoa([rng.uniform(-3, 3)], [rng.uniform(0, np.pi / 2)])))
    for i in range(4):
        starts.append(("random_wide_gamma", _pack_qaoa([rng.uniform(-6, 6)], [rng.uniform(0, np.pi / 2)])))
    polished = polish_starts(pool, "qaoa", 1, _dedup(starts), conf["maxfun"])
    polished["grid"] = {
        "n_points": int(len(xs)),
        "grid_min_E": float(np.min(energies)),
        "beta_period": "pi/2 on the P=+1 ray",
        "gamma_window": [conf["g_lo"], conf["g_hi"]],
        **grid_note,
    }
    polished["beta_offset_conversion"] = {
        "paper_beta2_minus_beta1": -0.2,
        "our_beta2_minus_beta1": 0.2 * abs(float(B)),
        "reason": "beta_ours = beta_paper * B with B<0 and J0=1; gamma offset +0.2 is unscaled",
    }
    return polished


def optimize_l1_family(pool, method, rng, conf, qaoa_x=None, donor_x=None) -> dict:
    """Random screen plus QAOA-matched and, for ablations, projected-D seeds."""
    width = n_params(method, 1)
    xs = []
    tags = []
    for i in range(conf["l1_screen"]):
        if method == "shared_beta":
            x = np.array([rng.uniform(-2.5, 2.5), rng.uniform(0, np.pi / 2)])
        elif method == "nov3_D":
            x = np.array(
                [rng.uniform(-2.0, 2.0), rng.uniform(-1.0, 1.0), rng.normal(scale=0.35), rng.normal(scale=0.35)]
            )
        elif method == "drop_c2":
            x = np.array([rng.uniform(-2.0, 2.0), rng.uniform(-1.0, 1.0), rng.normal(scale=0.35)])
        else:
            x = np.array([rng.uniform(-2.0, 2.0), rng.uniform(-1.0, 1.0), rng.normal(scale=0.35)])
        xs.append(x)
        tags.append("screen")
    if qaoa_x is not None:
        g = float(qaoa_x[0])
        b = float(qaoa_x[1])
        if method == "shared_beta":
            xs.append(np.array([g, b]))
            tags.append("qaoa_copy")
        elif method == "nov3_D":
            for c2 in (-0.6, -0.3, -0.1, 0.0, 0.1, 0.3, 0.6):
                for c3 in (-0.6, -0.3, -0.1, 0.0, 0.1, 0.3, 0.6):
                    xs.append(np.array([g, b, c2, c3]))
                    tags.append("qaoa_shell_grid")
        elif method == "drop_c2":
            for c3 in (-0.6, -0.3, -0.1, 0.0, 0.1, 0.3, 0.6):
                xs.append(np.array([g, b, c3]))
                tags.append("qaoa_c3_grid")
        elif method == "drop_c3":
            for c2 in (-0.6, -0.3, -0.1, 0.0, 0.1, 0.3, 0.6):
                xs.append(np.array([g, b, c2]))
                tags.append("qaoa_c2_grid")
    if donor_x is not None and method in ("drop_c2", "drop_c3"):
        g = float(donor_x[0])
        c1, c2, c3 = [float(v) for v in donor_x[1:4]]
        xs.append(np.array([g, c1, c3 if method == "drop_c2" else c2]))
        tags.append("projected_D")
    energies = pool.map_energy(method, 1, xs)
    keep = _greedy(xs, energies, k=conf["l1_keep"], min_dist=0.05)
    # Always polish every structured seed, plus diverse screen hits.
    starts = []
    for i, tag in enumerate(tags):
        if tag != "screen":
            starts.append((tag, xs[i]))
    for i in keep:
        starts.append((tags[i] + f"_{i}", xs[i]))
    polished = polish_starts(pool, method, 1, _dedup(starts), conf["maxfun"])
    polished["screen"] = {"n_points": len(xs), "screen_min_E": float(np.min(energies))}
    return polished


def optimize_higher(pool, method, L, prev, qaoa_x, shared_x, donor_x, B, rng, conf) -> dict:
    cap = conf["higher_cap"]
    if method == "qaoa":
        starts = qaoa_starts(L, prev, B, rng, cap)
    elif method == "shared_beta":
        starts = shared_starts(L, prev, qaoa_x, B, rng, cap)
    else:
        starts = shell_starts(method, L, prev, qaoa_x, shared_x, donor_x, B, rng, cap)
    if not starts:
        width = n_params(method, L)
        starts = [("fallback_random", rng.normal(scale=0.2, size=width)) for _ in range(8)]
    return polish_starts(pool, method, L, starts, conf["maxfun"])
