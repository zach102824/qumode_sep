"""QAOA for the Baxter–Wu up/down Hamiltonian.

    |ψ⟩ = ∏_{ℓ=L..1} exp(-i β_ℓ H_X) exp(-i γ_ℓ H_C) |+⟩
    H_X = sum_i X_i
    objective = ⟨H_C⟩ - Gamma ⟨H_X⟩

gamma_period is computed from the cost diagonal (exact step δ, period 2π/δ).
beta_period is π because spec(H_X) lies in one parity class. Both are checked
by replaying the shifted unitary. The L=1 grid reports the energy minimizer
and the maximum fidelity on the same grid; a separate fidelity maximization
checks that a low energy-opt fidelity is not hiding a high-fidelity L=1 point.
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize

BETA_PERIOD = np.pi


def apply_cost(psi, hc, gamma):
    return np.asarray(psi) * np.exp(-1j * float(gamma) * hc)


def apply_mixer(psi, beta, n):
    c = np.cos(float(beta))
    s = np.sin(float(beta))
    out = np.array(psi, dtype=np.complex128, copy=True)
    idx = np.arange(out.shape[0], dtype=np.int64)
    for q in range(n):
        out = c * out - 1j * s * out[idx ^ (1 << q)]
    return out


def apply_qaoa(psi0, hc, n, gammas, betas):
    psi = np.array(psi0, dtype=np.complex128, copy=True)
    for g, b in zip(gammas, betas):
        psi = apply_cost(psi, hc, g)
        psi = apply_mixer(psi, b, n)
    return psi


def energy_of(psi, hc, n, gamma_field):
    amp = np.asarray(psi, dtype=np.complex128)
    ec = float(np.vdot(amp, hc * amp).real)
    ex = 0.0
    idx = np.arange(amp.shape[0], dtype=np.int64)
    for q in range(n):
        ex += float(np.vdot(amp, amp[idx ^ (1 << q)]).real)
    return float(ec - float(gamma_field) * ex), float(ec), float(ex)


def fold_angle(theta, period):
    k = int(np.round(float(theta) / period))
    return float(theta - k * period), k


def fold_list(values, period):
    principals, branches = [], []
    for v in values:
        p, k = fold_angle(v, period)
        principals.append(p)
        branches.append(k)
    return principals, branches


def branch_report(gammas, betas, gamma_period):
    g_p, g_k = fold_list(gammas, gamma_period)
    b_p, b_k = fold_list(betas, BETA_PERIOD)
    jumps = []
    for seq in (g_p, b_p):
        for a, b in zip(seq, seq[1:]):
            jumps.append(abs(b - a))
    return {
        "gamma_principal": g_p,
        "beta_principal": b_p,
        "gamma_branch_k": g_k,
        "beta_branch_k": b_k,
        "n_gamma_wrapped": int(sum(k != 0 for k in g_k)),
        "n_beta_wrapped": int(sum(k != 0 for k in b_k)),
        "max_consecutive_principal_jump": float(max(jumps) if jumps else 0.0),
        "sum_abs_gamma_principal": float(np.sum(np.abs(g_p))) if g_p else 0.0,
        "sum_abs_beta_principal": float(np.sum(np.abs(b_p))) if b_p else 0.0,
        "duration_principal": float(np.sum(np.abs(g_p)) + np.sum(np.abs(b_p))) if g_p else 0.0,
        "gamma_period": float(gamma_period),
        "beta_period": float(BETA_PERIOD),
    }


def _lbfgs(objective, x0, maxiter, bounds):
    res = minimize(
        objective,
        np.asarray(x0, dtype=np.float64),
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": int(maxiter), "ftol": 1e-14, "gtol": 1e-8, "eps": 1e-7},
    )
    return np.asarray(res.x, dtype=np.float64), float(res.fun), int(res.nfev)


def _pack(gammas, betas):
    return np.concatenate([np.asarray(gammas, dtype=np.float64), np.asarray(betas, dtype=np.float64)])


def l1_grid(psi0, hc, n, gamma_field, gs, gamma_period, n_grid):
    gs = np.asarray(gs, dtype=np.complex128)
    betas = np.linspace(0.0, BETA_PERIOD, int(n_grid), endpoint=False)
    gammas = np.linspace(0.0, float(gamma_period), int(n_grid), endpoint=False)
    plus = np.asarray(psi0, dtype=np.complex128)
    Es = np.empty((len(betas), len(gammas)))
    Fs = np.empty_like(Es)
    idx = np.arange(plus.shape[0], dtype=np.int64)
    for ib, beta in enumerate(betas):
        phases = np.exp(-1j * gammas[:, None] * hc[None, :])
        psi = plus[None, :] * phases
        c = np.cos(float(beta))
        s = np.sin(float(beta))
        for q in range(n):
            psi = c * psi - 1j * s * psi[:, idx ^ (1 << q)]
        ec = np.real(np.sum(np.conj(psi) * hc[None, :] * psi, axis=1))
        ex = np.zeros(psi.shape[0])
        for q in range(n):
            ex += np.real(np.sum(np.conj(psi) * psi[:, idx ^ (1 << q)], axis=1))
        Es[ib] = ec - float(gamma_field) * ex
        Fs[ib] = np.abs(psi @ np.conj(gs)) ** 2
    return gammas, betas, Es, Fs


def top_indices(values, k, largest):
    flat = values.ravel()
    kk = min(int(k), flat.size)
    if largest:
        pick = np.argpartition(flat, flat.size - kk)[-kk:]
        pick = pick[np.argsort(-flat[pick])]
    else:
        pick = np.argpartition(flat, kk - 1)[:kk]
        pick = pick[np.argsort(flat[pick])]
    return [divmod(int(t), values.shape[1]) for t in pick]


def interpolate_schedule(prev, length):
    prev = np.asarray(prev, dtype=np.float64)
    q = int(length)
    s_old = np.linspace(0.0, 1.0, prev.size)
    s_new = np.linspace(0.0, 1.0, q)
    if q <= 3 or prev.size < 3:
        return np.interp(s_new, s_old, prev)
    return np.asarray(CubicSpline(s_old, prev, bc_type="natural")(s_new), dtype=np.float64)


def optimize(psi0, hc, n, gamma_field, starts, maxiter, objective, gs, gamma_period):
    gs = np.asarray(gs, dtype=np.complex128)

    def state(x):
        L = len(x) // 2
        return apply_qaoa(psi0, hc, n, x[:L], x[L:])

    def obj(x):
        psi = state(x)
        if objective == "energy":
            return energy_of(psi, hc, n, gamma_field)[0]
        return -float(abs(np.vdot(gs, psi)) ** 2)

    best_x, best_val, winner, nfev = None, np.inf, None, 0
    Lbound = len(np.asarray(starts[0][1])) // 2
    bounds = [(-float(gamma_period), 2 * float(gamma_period))] * Lbound + [(-BETA_PERIOD, 2 * BETA_PERIOD)] * Lbound
    for name, x0 in starts:
        x0 = np.asarray(x0, dtype=np.float64)
        seed_val = float(obj(x0))
        nfev += 1
        x_opt, val, n_loc = _lbfgs(obj, x0, maxiter, bounds)
        nfev += n_loc
        if seed_val < val:
            x_opt, val = x0, seed_val
            name = name + ":seed"
        if val < best_val:
            best_x, best_val, winner = x_opt, val, name
    L = len(best_x) // 2
    return {
        "gammas": [float(v) for v in best_x[:L]],
        "betas": [float(v) for v in best_x[L:]],
        "winner": winner,
        "nfev": int(nfev),
        "objective": objective,
    }


def run_ladder(
    n, hc, gamma_field, gs, E0, Emax, gamma_period,
    max_L=6, n_grid=25, n_random=6, maxiter_l1=60, maxiter_hi=40,
    do_shared_levels=None, seed=0,
):
    from model import plus_state

    psi0 = plus_state(n)
    gs_c = np.asarray(gs, dtype=np.complex128)
    rng = np.random.default_rng(np.random.PCG64(50_000 + int(seed) + 17 * int(round(1000 * gamma_field))))
    rows = []
    e0, _, _ = energy_of(psi0, hc, n, gamma_field)
    f0 = float(abs(np.vdot(gs_c, psi0)) ** 2)
    rows.append({"L": 0, "F": f0, "energy": e0, "eta": _eta(e0, E0, Emax), "winner": "plus", "gammas": [], "betas": [], "nfev": 0,
                 **branch_report([], [], gamma_period), "energy_nested": True, "fidelity_monotone": True})

    g_axis, b_axis, Es, Fs = l1_grid(psi0, hc, n, gamma_field, gs_c, gamma_period, n_grid)
    e_coords = top_indices(Es, 6, largest=False)
    f_coords = top_indices(Fs, 6, largest=True)

    def coord_starts(coords, prefix):
        out = []
        for rank, (ib, ig) in enumerate(coords):
            out.append((f"{prefix}{rank}", _pack([g_axis[ig]], [b_axis[ib]])))
        return out

    e_starts = coord_starts(e_coords, "gridE_")
    for i in range(4):
        e_starts.append((f"random_{i}", _pack(rng.uniform(0, gamma_period, 1), rng.uniform(0, BETA_PERIOD, 1))))
    optE = optimize(psi0, hc, n, gamma_field, e_starts, maxiter_l1, "energy", gs_c, gamma_period)
    f_starts = coord_starts(f_coords, "gridF_")
    for i in range(3):
        f_starts.append((f"frand_{i}", _pack(rng.uniform(0, gamma_period, 1), rng.uniform(0, BETA_PERIOD, 1))))
    optF = optimize(psi0, hc, n, gamma_field, f_starts, max(40, maxiter_l1 // 2), "fidelity", gs_c, gamma_period)
    psiF = apply_qaoa(psi0, hc, n, optF["gammas"], optF["betas"])
    EF = energy_of(psiF, hc, n, gamma_field)[0]
    FF = float(abs(np.vdot(gs_c, psiF)) ** 2)

    prev_g, prev_b = optE["gammas"], optE["betas"]
    prev_E, prev_F = e0, f0
    for L in range(1, max_L + 1):
        if L == 1:
            this = optE
        else:
            starts = _higher_starts(L, prev_g, prev_b, rng, n_random, gamma_period)
            this = optimize(psi0, hc, n, gamma_field, starts, maxiter_hi, "energy", gs_c, gamma_period)
        psi = apply_qaoa(psi0, hc, n, this["gammas"], this["betas"])
        rep = branch_report(this["gammas"], this["betas"], gamma_period)
        psi_p = apply_qaoa(psi0, hc, n, rep["gamma_principal"], rep["beta_principal"])
        prin_ov = float(abs(np.vdot(psi, psi_p)) ** 2)
        E = energy_of(psi, hc, n, gamma_field)[0]
        F = float(abs(np.vdot(gs_c, psi)) ** 2)
        row = {
            "L": L,
            "F": F,
            "energy": E,
            "eta": _eta(E, E0, Emax),
            "winner": this["winner"],
            "nfev": this["nfev"],
            "gammas": this["gammas"],
            "betas": this["betas"],
            "principal_state_overlap": prin_ov,
            "energy_nested": bool(E <= prev_E + 1e-7),
            "fidelity_monotone": bool(F + 1e-7 >= prev_F),
            **rep,
        }
        if L == 1:
            row.update({
                "l1_grid_min_energy": float(Es.min()),
                "l1_grid_max_fidelity": float(Fs.max()),
                "l1_energy_opt_beats_grid": bool(E <= float(Es.min()) + 1e-6),
                "l1_fidelity_max_F": FF,
                "l1_fidelity_max_energy": float(EF),
                "l1_F_gap_fidopt_minus_energyopt": float(FF - F),
                "l1_energy_gap_energyopt_minus_fidopt": float(E - EF),
                # false-hard: energy-opt fidelity much worse than the best L=1 fidelity
                "l1_false_hard": bool(float(Fs.max()) > F + 0.2 and float(Fs.max()) > 0.5),
            })
        if do_shared_levels is None or L in do_shared_levels:
            shared = _shared(psi0, hc, n, gamma_field, gs_c, this["gammas"], this["betas"], rng, gamma_period, E0, Emax)
            row["shared_F"] = shared["F"]
            row["shared_energy"] = shared["energy"]
            row["shared_gamma"] = shared["gamma"]
        g_a, b_a = anneal_angles(L, gamma_field, rep["duration_principal"])
        if g_a:
            psi_a = apply_qaoa(psi0, hc, n, g_a, b_a)
            Ea = energy_of(psi_a, hc, n, gamma_field)[0]
            row["anneal_F"] = float(abs(np.vdot(gs_c, psi_a)) ** 2)
            row["anneal_energy"] = Ea
            row["anneal_duration"] = float(np.sum(np.abs(g_a)) + np.sum(np.abs(b_a)))
        rows.append(row)
        prev_g, prev_b = this["gammas"], this["betas"]
        prev_E, prev_F = E, F
    return rows


def _higher_starts(L, prev_g, prev_b, rng, n_random, gamma_period):
    g_prev = [float(v) for v in prev_g]
    b_prev = [float(v) for v in prev_b]
    starts = []
    if L == 2 and len(g_prev) == 1:
        g1, b1 = g_prev[0], b_prev[0]
        starts.append(("pagano_p2", _pack([g1, g1 + 0.2], [b1, b1 - 0.2])))
        for dg, db, tag in ((0.1, -0.1, "small"), (-0.2, -0.2, "mm"), (0.0, 0.0, "repeat")):
            starts.append((f"p2_{tag}", _pack([g1, g1 + dg], [b1, b1 + db])))
    if L >= 3 and len(g_prev) == L - 1:
        gi = interpolate_schedule(g_prev, L)
        bi = interpolate_schedule(b_prev, L)
        starts.append(("pagano_interp", _pack(gi, bi)))
        starts.append(("interp_pert", _pack(gi + rng.normal(0, 0.05, L), bi + rng.normal(0, 0.08, L))))
    if L >= 2 and len(g_prev) == L - 1:
        starts.append(("zeropad_append", _pack(g_prev + [0.0], b_prev + [0.0])))
        starts.append(("zeropad_prepend", _pack([0.0] + g_prev, [0.0] + b_prev)))
    for i in range(n_random):
        starts.append((
            f"random_{i}",
            _pack(rng.uniform(0.0, gamma_period, L), rng.uniform(0.0, BETA_PERIOD, L)),
        ))
    return starts


def _shared(psi0, hc, n, gamma_field, gs, gammas, betas, rng, gamma_period, E0, Emax):
    L = len(gammas)
    g_mean = float(np.mean(gammas))
    starts = [
        ("mean", np.concatenate([[g_mean], np.asarray(betas, dtype=np.float64)])),
    ]
    for i in range(3):
        starts.append(("r" + str(i), np.concatenate([
            rng.uniform(0.0, gamma_period, 1),
            rng.uniform(0.0, BETA_PERIOD, L),
        ])))

    def obj(x):
        psi = apply_qaoa(psi0, hc, n, [float(x[0])] * L, x[1:])
        return energy_of(psi, hc, n, gamma_field)[0]

    best_x, best_val = None, np.inf
    bounds = [(-float(gamma_period), 2 * float(gamma_period))] + [(-BETA_PERIOD, 2 * BETA_PERIOD)] * L
    for name, x0 in starts:
        seed_val = float(obj(x0))
        x_opt, val, _ = _lbfgs(obj, x0, 30, bounds)
        if seed_val < val:
            x_opt, val = np.asarray(x0, dtype=np.float64), seed_val
        if val < best_val:
            best_x, best_val = x_opt, val
    g = float(best_x[0])
    b = best_x[1:]
    psi = apply_qaoa(psi0, hc, n, [g] * L, b)
    E = energy_of(psi, hc, n, gamma_field)[0]
    F = float(abs(np.vdot(gs, psi)) ** 2)
    return {"gamma": g, "F": F, "energy": E}


def anneal_angles(L, gamma_field, duration):
    """Matched-duration Trotter of H(s) = (1-s)(-H_X) + s (H_C - Gamma H_X).

    gamma = dt * s, beta = -dt * ((1-s) + Gamma * s). One scale matches
    sum |gamma| + sum |beta| to the principal-branch QAOA duration.
    """
    if L <= 0 or not np.isfinite(duration) or duration <= 0:
        return [], []
    s = (np.arange(1, L + 1, dtype=np.float64) - 0.5) / float(L)
    g = s
    b = -((1.0 - s) + float(gamma_field) * s)
    raw = float(np.sum(np.abs(g) + np.abs(b)))
    scale = float(duration) / raw
    return [float(scale * x) for x in g], [float(scale * x) for x in b]


def _eta(E, E0, Emax):
    den = float(Emax - E0)
    if abs(den) < 1e-15:
        return float("nan")
    return float((Emax - E) / den)
