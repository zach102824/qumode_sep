"""Full-space QAOA for the circulant ring.

Ansatz, layer 1 closest to |+>:
    exp(-i beta sum_i X_i) exp(-i gamma H_cost)
H_cost is the diagonal two-window sum. The objective is <H>, with
H = H_cost - Gamma sum X. Fidelity is recorded, and at L=1 it is also
maximized on its own so a low energy-selected overlap is not mistaken
for a low achievable overlap.
"""

from __future__ import annotations

import math

import numpy as np
from scipy.optimize import minimize

from hamiltonian import apply_sum_x


def plus_state(n: int) -> np.ndarray:
    return np.full(1 << n, 2.0 ** (-0.5 * n), dtype=np.complex128)


def apply_mixer(psi: np.ndarray, beta: float, n: int) -> np.ndarray:
    """exp(-i beta X) on every qubit. Qubit 0 is the fastest index."""
    cb = math.cos(beta)
    sb = math.sin(beta)
    out = np.array(psi, dtype=np.complex128, copy=True)
    for i in range(n):
        v = out.reshape(1 << (n - i - 1), 2, 1 << i)
        a = v[:, 0, :].copy()
        b = v[:, 1, :].copy()
        v[:, 0, :] = cb * a - 1j * sb * b
        v[:, 1, :] = cb * b - 1j * sb * a
        out = v.reshape(-1)
    return out


def evolve(diag: np.ndarray, gammas, betas, n: int, psi0: np.ndarray | None = None) -> np.ndarray:
    psi = plus_state(n) if psi0 is None else np.array(psi0, dtype=np.complex128, copy=True)
    g = np.atleast_1d(np.asarray(gammas, dtype=np.float64))
    b = np.atleast_1d(np.asarray(betas, dtype=np.float64))
    if g.size == 0:
        return psi
    for gv, bv in zip(g, b):
        psi = psi * np.exp(-1j * float(gv) * diag)
        psi = apply_mixer(psi, float(bv), n)
    return psi


def energy_fidelity(psi: np.ndarray, diag: np.ndarray, gamma: float, n: int, gs: np.ndarray) -> tuple[float, float]:
    e = float(np.real(np.vdot(psi, diag * psi))) - gamma * float(np.real(np.vdot(psi, apply_sum_x(psi, n))))
    f = float(abs(np.vdot(gs, psi)) ** 2)
    return e, f


def l1_grid(diag, gamma, n, gs, period: float, n_g: int = 97, n_b: int = 49) -> dict:
    gammas = np.linspace(-0.5 * period, 0.5 * period, n_g, endpoint=False)
    betas = np.linspace(-0.5 * math.pi, 0.5 * math.pi, n_b, endpoint=False)
    plus = plus_state(n)
    best_e, best_e_ang = math.inf, (0.0, 0.0)
    best_f, best_f_ang = -1.0, (0.0, 0.0)
    # Keep a few energy basins.
    kept = []
    for g in gammas:
        phased = plus * np.exp(-1j * float(g) * diag)
        for b in betas:
            psi = apply_mixer(phased, float(b), n)
            e, f = energy_fidelity(psi, diag, gamma, n, gs)
            if e < best_e:
                best_e, best_e_ang = e, (float(g), float(b))
            if f > best_f:
                best_f, best_f_ang = f, (float(g), float(b))
            if len(kept) < 12 or e < kept[-1][0]:
                kept.append((e, float(g), float(b), f))
                kept.sort(key=lambda t: t[0])
                kept = kept[:12]
    return {
        "best_energy": best_e,
        "best_energy_angles": best_e_ang,
        "best_fidelity": best_f,
        "best_fidelity_angles": best_f_ang,
        "basins": kept,
    }


def _opt(diag, gamma, n, gs, g0, b0, objective: str, maxiter: int = 160):
    g0 = np.atleast_1d(np.asarray(g0, dtype=np.float64))
    b0 = np.atleast_1d(np.asarray(b0, dtype=np.float64))
    x0 = np.empty(g0.size + b0.size)
    x0[0::2] = g0
    x0[1::2] = b0

    def fun(x):
        psi = evolve(diag, x[0::2], x[1::2], n)
        e, f = energy_fidelity(psi, diag, gamma, n, gs)
        return e if objective == "energy" else -f

    res = minimize(fun, x0, method="BFGS", options={"maxiter": maxiter, "gtol": 1e-8})
    g, b = res.x[0::2], res.x[1::2]
    e, f = energy_fidelity(evolve(diag, g, b, n), diag, gamma, n, gs)
    return g, b, e, f


def fold_min_abs(values: np.ndarray, period: float) -> np.ndarray:
    half = 0.5 * period
    return np.mod(values + half, period) - half


def interpolate_schedule(gammas: np.ndarray, betas: np.ndarray):
    """Pagano supplement: linear when the new depth is 3, cubic after that."""
    from scipy.interpolate import CubicSpline

    old = np.linspace(0.0, 1.0, gammas.size)
    new = np.linspace(0.0, 1.0, gammas.size + 1)
    if gammas.size + 1 <= 3:
        return np.interp(new, old, gammas), np.interp(new, old, betas)
    return (
        np.asarray(CubicSpline(old, gammas, bc_type="natural")(new), dtype=np.float64),
        np.asarray(CubicSpline(old, betas, bc_type="natural")(new), dtype=np.float64),
    )


def search_ladder(
    diag,
    gamma,
    n,
    gs,
    period: float,
    max_depth: int,
    rng: np.random.Generator,
    n_random: int = 20,
    maxiter: int = 80,
):
    """Energy minimization, L=1..max_depth. Zero-pad keeps the energy from rising.

    Returns one record per depth. Fidelity is whatever the best-energy state has.
    """
    grid = l1_grid(diag, gamma, n, gs, period, n_g=61, n_b=31)
    records = []
    endpoints = []
    # L=1
    pool = []
    for e0, g0, b0, f0 in grid["basins"][:4]:
        psi = evolve(diag, [g0], [b0], n)
        e, f = energy_fidelity(psi, diag, gamma, n, gs)
        pool.append((e, f, np.array([g0]), np.array([b0]), "l1_grid_exact"))
        g, b, e, f = _opt(diag, gamma, n, gs, [g0], [b0], "energy", maxiter=maxiter)
        pool.append((e, f, g, b, "l1_grid_basin"))
        endpoints.append((e, f))
    for i in range(n_random):
        g0 = rng.uniform(-0.5 * period, 0.5 * period, size=1)
        b0 = rng.uniform(-0.5 * math.pi, 0.5 * math.pi, size=1)
        g, b, e, f = _opt(diag, gamma, n, gs, g0, b0, "energy", maxiter=maxiter)
        pool.append((e, f, g, b, f"l1_random_{i}"))
        endpoints.append((e, f))
    gf, bf = grid["best_fidelity_angles"]
    gF, bF, eF, fF = _opt(diag, gamma, n, gs, [gf], [bf], "fidelity", maxiter=150)
    records.append(_pack(pool, endpoints, 1, gamma, n, diag, gs, period, f1max=fF, e_at_fmax=eF, f1_angles=(gF, bF)))
    prev = records[-1]
    for depth in range(2, max_depth + 1):
        pool = []
        endpoints = []
        g_prev = prev["gamma_raw"]
        b_prev = prev["beta_raw"]
        seeds = [
            ("zeropad_end", np.r_[g_prev, 0.0], np.r_[b_prev, 0.0], True),
            ("zeropad_front", np.r_[0.0, g_prev], np.r_[0.0, b_prev], True),
        ]
        if depth == 2:
            for delta in (0.05, 0.1, 0.2):
                seeds.append(
                    (
                        f"pagano_delta_{delta}",
                        np.array([g_prev[0], g_prev[0] + delta]),
                        np.array([b_prev[0], b_prev[0] - delta]),
                        False,
                    )
                )
        else:
            gi, bi = interpolate_schedule(g_prev, b_prev)
            seeds.append(("pagano_interp", gi, bi, True))
            gp, bp = gi.copy(), bi.copy()
            gp[-1] = gp[-1] + 0.2
            bp[-1] = bp[-1] - 0.2
            seeds.append(("interp_end_perturb_0.2", gp, bp, False))
        base_g = np.r_[g_prev, g_prev[-1]]
        base_b = np.r_[b_prev, b_prev[-1]]
        for i in range(4):
            seeds.append(
                (
                    f"perturb_extend_{i}",
                    base_g + rng.normal(0.0, 0.08, size=depth),
                    base_b + rng.normal(0.0, 0.08, size=depth),
                    False,
                )
            )
        for i in range(n_random):
            seeds.append(
                (
                    f"random_{i}",
                    rng.uniform(-0.5 * period, 0.5 * period, size=depth),
                    rng.uniform(-0.5 * math.pi, 0.5 * math.pi, size=depth),
                    False,
                )
            )
        for name, g0, b0, keep_exact in seeds:
            if keep_exact:
                e, f = energy_fidelity(evolve(diag, g0, b0, n), diag, gamma, n, gs)
                pool.append((e, f, np.asarray(g0, dtype=np.float64), np.asarray(b0, dtype=np.float64), name + "_exact"))
            g, b, e, f = _opt(diag, gamma, n, gs, g0, b0, "energy", maxiter=maxiter)
            pool.append((e, f, g, b, name))
            endpoints.append((e, f))
        records.append(_pack(pool, endpoints, depth, gamma, n, diag, gs, period, f1max=None, e_at_fmax=None, f1_angles=None))
        prev = records[-1]
    return records


def _pack(pool, endpoints, depth, gamma, n, diag, gs, period, f1max, e_at_fmax, f1_angles):
    pool = [p for p in pool if np.isfinite(p[0])]
    pool.sort(key=lambda p: (p[0], -p[1]))
    e, f, g, b, source = pool[0]
    gc, bc, notes = canonicalize(g, b, period)
    e2, f2 = energy_fidelity(evolve(diag, gc, bc, n), diag, gamma, n, gs)
    if abs(e2 - e) > 1e-6 or abs(f2 - f) > 1e-6:
        notes = notes + ["canonical_mismatch"]
    # Use the raw state for reported observables; canonical is stored alongside.
    f_ends = np.array([p[1] for p in endpoints]) if endpoints else np.array([f])
    e_ends = np.array([p[0] for p in endpoints]) if endpoints else np.array([e])
    return {
        "L": depth,
        "energy": float(e),
        "F": float(f),
        "source": source,
        "gamma_raw": np.asarray(g, dtype=np.float64),
        "beta_raw": np.asarray(b, dtype=np.float64),
        "gamma_can": gc,
        "beta_can": bc,
        "canon_notes": notes,
        "sum_abs_gamma": float(np.sum(np.abs(g))),
        "sum_abs_beta": float(np.sum(np.abs(b))),
        "n_starts": len(pool),
        "F_min": float(np.min(f_ends)),
        "F_max": float(np.max(f_ends)),
        "F_std": float(np.std(f_ends)),
        "F1max": None if f1max is None else float(f1max),
        "E_at_F1max": None if e_at_fmax is None else float(e_at_fmax),
        "f1_gamma": None if f1_angles is None else float(np.atleast_1d(f1_angles[0])[0]),
        "f1_beta": None if f1_angles is None else float(np.atleast_1d(f1_angles[1])[0]),
        "canon_energy_err": float(abs(e2 - e)),
    }


def canonicalize(gammas, betas, gamma_period: float) -> tuple[np.ndarray, np.ndarray, list[str]]:
    g = np.atleast_1d(np.asarray(gammas, dtype=np.float64)).copy()
    b = np.atleast_1d(np.asarray(betas, dtype=np.float64)).copy()
    notes = []
    if g.size == 0:
        return g, b, notes
    g2 = fold_min_abs(g, gamma_period)
    b2 = fold_min_abs(b, math.pi)
    if np.any(np.abs(g2 - g) > 1e-8):
        notes.append("gamma_period_fold")
    if np.any(np.abs(b2 - b) > 1e-8):
        notes.append("beta_period_pi_fold")
    return g2, b2, notes
