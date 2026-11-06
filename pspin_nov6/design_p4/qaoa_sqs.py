"""Independent-angle QAOA for a signed SQS.

Layer k=1..L acts first. Inside a layer: exp(-i γ H_C) then exp(-i β sum X).
Objective is energy of H = H_C - Gamma_X sum X. Fidelity maximization is a
separate L=1 diagnostic and is never reported as the energy result.

β projective period is π (sum X eigenvalues are all even or all odd with N).
γ period is 2π/gcd(H_C spectrum), computed from the diagonal, not assumed.

Bootstrap follows Pagano arXiv:1906.02700 Suppl. I.1.2 on the RAW angles:
  L=1 grid + local search, including the zero seed
  L=2 seed (β, β-0.2), (γ, γ+0.2)
  L≥3 natural-cubic (linear if only two nodes) resample of the raw schedule
Zero-pad of L-1 is kept, so best energy does not increase with L.
Fidelity of that state is not forced to be monotone.
Shared-γ is optional and tagged separately.
"""
from __future__ import annotations

import math

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize

from ham import energy_of, evolve, f_overlap, plus_state


def _split(x, L):
    v = np.asarray(x, dtype=float).reshape(-1)
    return v[:L].copy(), v[L:].copy()


def _pack(b, g):
    return np.concatenate([np.asarray(b, dtype=float), np.asarray(g, dtype=float)])


def resample(values, new_len: int) -> np.ndarray:
    old = np.asarray(values, dtype=float).reshape(-1)
    if old.size == 1:
        return np.full(new_len, float(old[0]))
    s_old = np.linspace(0.0, 1.0, old.size)
    s_new = np.linspace(0.0, 1.0, new_len)
    if old.size < 3:
        return np.interp(s_new, s_old, old)
    return np.asarray(CubicSpline(s_old, old, bc_type="natural")(s_new), dtype=float)


def canonical(betas, gammas, T: float):
    b = np.array(betas, dtype=float, copy=True)
    g = np.array(gammas, dtype=float, copy=True)
    b = (b + 0.5 * math.pi) % math.pi - 0.5 * math.pi
    if T > 0:
        half = 0.5 * T
        g = (g + half) % T - half
    if g.sum() < -1e-10 or (abs(float(g.sum())) <= 1e-10 and b.sum() < 0):
        b = -b
        g = -g
    return b, g


def duration(betas, gammas) -> float:
    return float(np.sum(np.abs(betas)) + np.sum(np.abs(gammas)))


class Counter:
    def __init__(self):
        self.n = 0


def _minimize(fun, x0, bounds, counter: Counter, maxiter: int, polish: bool):
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])
    x0 = np.clip(np.asarray(x0, dtype=float), lo, hi)

    def wrapped(x):
        counter.n += 1
        return float(fun(x))

    res = minimize(wrapped, x0, method="L-BFGS-B", bounds=bounds, options={"maxiter": maxiter, "ftol": 1e-14, "gtol": 1e-8})
    x_best, f_best = np.array(res.x, dtype=float), float(res.fun)
    if polish:
        def wrapped_nm(x):
            counter.n += 1
            if np.any(x < lo) or np.any(x > hi):
                return 1e6
            return float(fun(x))

        res2 = minimize(wrapped_nm, x_best, method="Nelder-Mead", options={"maxiter": max(40, maxiter // 2), "xatol": 1e-5, "fatol": 1e-12, "adaptive": True})
        if float(res2.fun) < f_best:
            return np.array(res2.x, dtype=float), float(res2.fun)
    return x_best, f_best


def grid_l1(diag, n, gamma_x, T, n_beta=36, n_gamma=72):
    """Vectorized L=1 energies and, if gs is later supplied, fidelities are separate."""
    betas = np.linspace(0.0, math.pi, n_beta, endpoint=False)
    gammas = np.linspace(0.0, T, n_gamma, endpoint=False)
    psi0 = plus_state(n)
    # For each (beta, gamma): phase then mixer. Mixer depends only on beta.
    # Precompute phased states (n_gamma, dim) and mixer by applying to each.
    dim = 1 << n
    phased = psi0[None, :] * np.exp(-1j * gammas[:, None] * diag[None, :])
    energies = np.empty((n_beta, n_gamma))
    states = []
    for ib, beta in enumerate(betas):
        block = np.empty((n_gamma, dim), dtype=np.complex128)
        for ig in range(n_gamma):
            block[ig] = evolve_one_mixer(phased[ig], float(beta), n)
        hc = np.sum(np.abs(block) ** 2 * diag[None, :], axis=1)
        # <sum X> via one bit-flip pass on the batch would be nicer; loop X
        sx = np.empty(n_gamma)
        for ig in range(n_gamma):
            sx[ig] = np.vdot(block[ig], _sumx(block[ig], n)).real
        energies[ib] = hc - gamma_x * sx
        states.append(block)
    return betas, gammas, energies, states


def evolve_one_mixer(psi, beta, n):
    from ham import apply_mixer
    return apply_mixer(psi, beta, n)


def _sumx(psi, n):
    from ham import apply_sum_x
    return apply_sum_x(psi, n)


def optimize_l1(diag, n, gamma_x, gs, T, rng, counter: Counter, mode: str):
    """mode 'energy' minimizes E, mode 'fidelity' maximizes |<gs|ψ>|^2."""
    betas, gammas, energies, states = grid_l1(diag, n, gamma_x, T)
    counter.n += energies.size
    fids = np.empty_like(energies)
    for ib in range(len(betas)):
        # states[ib][ig]
        amps = states[ib] @ np.conj(gs)
        fids[ib] = np.abs(amps) ** 2
    if mode == "energy":
        score = energies
        pick = np.argmin
    else:
        score = -fids
        pick = np.argmin
    # separated seeds
    flat = score.ravel()
    order = np.argsort(flat)
    seeds = []
    for t in order:
        ib, ig = divmod(int(t), score.shape[1])
        b, g = float(betas[ib]), float(gammas[ig])
        if any(abs(b - bb) < 0.2 and abs(g - gg) < max(0.15, T / 30) for bb, gg in seeds):
            continue
        seeds.append((b, g))
        if len(seeds) >= 6:
            break
    seeds.append((0.0, 0.0))
    for _ in range(3):
        seeds.append((float(rng.uniform(0, math.pi)), float(rng.uniform(0, T))))
    bounds = [(-math.pi, math.pi), (-T, T)]

    def objective(x):
        b, g = _split(x, 1)
        psi = evolve(plus_state(n), diag, n, b, g)
        if mode == "energy":
            return energy_of(psi, diag, n, gamma_x)
        return -f_overlap(psi, gs)

    best = None
    for i, (b, g) in enumerate(seeds):
        x0 = _pack([b], [g])
        counter.n += 1
        e0 = float(objective(np.clip(x0, [-math.pi, -T], [math.pi, T])))
        x, val = _minimize(objective, x0, bounds, counter, maxiter=40, polish=(i < 2))
        if e0 < val:
            x, val = np.clip(x0, [-math.pi, -T], [math.pi, T]), e0
        if best is None or val < best[0]:
            best = (val, x, f"seed{i}")
    b, g = _split(best[1], 1)
    psi = evolve(plus_state(n), diag, n, b, g)
    E = energy_of(psi, diag, n, gamma_x)
    F = f_overlap(psi, gs)
    bc, gc = canonical(b, g, T)
    return {
        "E": E, "F": F, "betas_raw": b.tolist(), "gammas_raw": g.tolist(),
        "betas_canonical": bc.tolist(), "gammas_canonical": gc.tolist(),
        "tag": best[2], "duration_raw": duration(b, g), "duration_canonical": duration(bc, gc),
        "psi": psi, "mode": mode,
    }


def optimize_depth(diag, n, gamma_x, gs, T, L, prev, rng, counter: Counter):
    bounds = [(-math.pi, math.pi)] * L + [(-T, T)] * L

    def objective(x):
        b, g = _split(x, L)
        psi = evolve(plus_state(n), diag, n, b, g)
        return energy_of(psi, diag, n, gamma_x)

    seeds = []
    b_prev = np.asarray(prev["betas_raw"], dtype=float)
    g_prev = np.asarray(prev["gammas_raw"], dtype=float)
    if L == 2:
        seeds.append(("pagano_l2", _pack([b_prev[0], b_prev[0] - 0.2], [g_prev[0], g_prev[0] + 0.2])))
    else:
        seeds.append(("pagano_interp", _pack(resample(b_prev, L), resample(g_prev, L))))
    primary = seeds[0][1]
    for sigma in (0.05, 0.15):
        seeds.append((f"perturb_{sigma}", primary + rng.normal(0, sigma, size=primary.size)))
    b_off, g_off = _split(primary, L)
    b_off = b_off.copy(); g_off = g_off.copy()
    b_off[-1] -= 0.2; g_off[-1] += 0.2
    seeds.append(("last_offset", _pack(b_off, g_off)))
    seeds.append(("zero_pad", _pack(np.concatenate([b_prev, [0.0]]), np.concatenate([g_prev, [0.0]]))))
    seeds.append(("signflip", -primary))
    seeds.append(("random", _pack(rng.uniform(-math.pi, math.pi, size=L), rng.uniform(-T, T, size=L))))

    carry = _pack(np.concatenate([b_prev, [0.0]]), np.concatenate([g_prev, [0.0]]))
    counter.n += 1
    carry_E = float(objective(carry))
    best = (carry_E, carry, "carry")
    for tag, x0 in seeds:
        counter.n += 1
        e_seed = float(objective(np.clip(x0, [b[0] for b in bounds], [b[1] for b in bounds])))
        x, val = _minimize(objective, x0, bounds, counter, maxiter=30 + 6 * L, polish=False)
        if e_seed < val:
            x, val = np.clip(x0, [b[0] for b in bounds], [b[1] for b in bounds]), e_seed
        if val < best[0] - 1e-12:
            best = (val, x, tag)
    # polish winner
    x, val = _minimize(objective, best[1], bounds, counter, maxiter=40, polish=True)
    if val < best[0] - 1e-12:
        best = (val, x, best[2] + "_polish")
    if carry_E < best[0] - 1e-12:
        best = (carry_E, carry, "carry")
    b, g = _split(best[1], L)
    psi = evolve(plus_state(n), diag, n, b, g)
    E = energy_of(psi, diag, n, gamma_x)
    F = f_overlap(psi, gs)
    bc, gc = canonical(b, g, T)
    return {
        "E": E, "F": F, "betas_raw": b.tolist(), "gammas_raw": g.tolist(),
        "betas_canonical": bc.tolist(), "gammas_canonical": gc.tolist(),
        "tag": best[2], "duration_raw": duration(b, g), "duration_canonical": duration(bc, gc),
        "psi": psi, "L": L,
    }


def optimize_shared(diag, n, gamma_x, gs, T, L, independent, rng, counter: Counter):
    bounds = [(-T, T)] + [(-math.pi, math.pi)] * L

    def objective(x):
        gamma = float(x[0])
        betas = np.asarray(x[1:], dtype=float)
        psi = evolve(plus_state(n), diag, n, betas, np.full(L, gamma))
        return energy_of(psi, diag, n, gamma_x)

    b = np.asarray(independent["betas_raw"], dtype=float)
    g = np.asarray(independent["gammas_raw"], dtype=float)
    seeds = [
        np.concatenate([[float(np.mean(g))], b]),
        np.concatenate([[float(g[0])], b]),
        np.concatenate([[0.0], np.zeros(L)]),
        np.concatenate([[float(rng.uniform(0, T))], rng.uniform(-math.pi, math.pi, size=L)]),
    ]
    best = None
    for x0 in seeds:
        x, val = _minimize(objective, x0, bounds, counter, maxiter=40, polish=False)
        if best is None or val < best[0]:
            best = (val, x)
    gamma = float(best[1][0])
    betas = np.asarray(best[1][1:], dtype=float)
    gammas = np.full(L, gamma)
    psi = evolve(plus_state(n), diag, n, betas, gammas)
    E = energy_of(psi, diag, n, gamma_x)
    F = f_overlap(psi, gs)
    return {"E": E, "F": F, "gamma_shared": gamma, "betas_raw": betas.tolist(), "gammas_raw": gammas.tolist(), "L": L}


def eta(E, E0, Emax):
    d = E0 - Emax
    if abs(d) < 1e-15:
        return float("nan")
    return float((E - Emax) / d)
