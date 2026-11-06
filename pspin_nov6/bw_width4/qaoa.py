"""Independent-angle QAOA for staggered Baxter–Wu.

    |ψ⟩ = ∏_{ℓ=1..L} exp(-i β_ℓ sum X) exp(-i γ_ℓ H_C) |+⟩
with ℓ=1 acting first. Objective is ⟨H⟩. L=1 fidelity maximization is separate.

γ period comes from the cost spectrum. β period is π. Raw angles are kept;
folding is reporting-only and is not fed back into the bootstrap.
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize

BETA_PERIOD = float(np.pi)


def apply_qaoa(psi0, hc, n, gammas, betas):
    psi = np.array(psi0, dtype=np.complex128, copy=True)
    idx = np.arange(psi.shape[0], dtype=np.int64)
    for g, b in zip(gammas, betas):
        psi *= np.exp(-1j * float(g) * hc)
        c = np.cos(float(b))
        s = np.sin(float(b))
        for q in range(n):
            psi = c * psi - 1j * s * psi[idx ^ (1 << q)]
    return psi


def plus_state(n):
    return np.full(1 << n, 1.0 / np.sqrt(1 << n), dtype=np.complex128)


def fold(theta, period):
    half = 0.5 * period
    return float((theta + half) % period - half)


def canonical(gammas, betas, t_gamma):
    g = np.array([fold(v, t_gamma) for v in gammas], dtype=float)
    b = np.array([fold(v, BETA_PERIOD) for v in betas], dtype=float)
    if g.sum() < 0:
        g = -g
        b = -b
    return g, b


def duration(gammas, betas) -> float:
    return float(np.sum(np.abs(gammas)) + np.sum(np.abs(betas)))


def interpolate(prev, length):
    prev = np.asarray(prev, dtype=float)
    s_old = np.linspace(0.0, 1.0, prev.size)
    s_new = np.linspace(0.0, 1.0, int(length))
    if length <= 3 or prev.size < 3:
        return np.interp(s_new, s_old, prev)
    return np.asarray(CubicSpline(s_old, prev, bc_type="natural")(s_new), dtype=float)


def _lbfgs(fun, x0, maxiter, bounds):
    nfev = {"n": 0}

    def wrapped(x):
        nfev["n"] += 1
        return float(fun(x))

    res = minimize(wrapped, np.asarray(x0, dtype=float), method="L-BFGS-B", bounds=bounds,
                   options={"maxiter": int(maxiter), "ftol": 1e-12, "gtol": 1e-7})
    return np.asarray(res.x, dtype=float), float(res.fun), nfev["n"]


def l1_grid(hc, n, gamma, gs, t_gamma, n_grid):
    betas = np.linspace(0.0, BETA_PERIOD, n_grid, endpoint=False)
    gammas = np.linspace(0.0, t_gamma, n_grid, endpoint=False)
    psi0 = plus_state(n)
    Es = np.empty((n_grid, n_grid))
    Fs = np.empty((n_grid, n_grid))
    for ib, b in enumerate(betas):
        for ig, g in enumerate(gammas):
            psi = apply_qaoa(psi0, hc, n, [g], [b])
            Es[ib, ig] = energy_of(psi, hc, n, gamma)
            Fs[ib, ig] = abs(np.vdot(gs, psi)) ** 2
    return betas, gammas, Es, Fs


def _pack(gammas, betas):
    return np.concatenate([np.asarray(gammas, dtype=float), np.asarray(betas, dtype=float)])


def optimize(hc, n, gamma, starts, maxiter, objective, gs, t_gamma):
    def state(x):
        L = len(x) // 2
        return apply_qaoa(plus_state(n), hc, n, x[:L], x[L:])

    def obj(x):
        psi = state(x)
        if objective == "energy":
            return energy_of(psi, hc, n, gamma)
        return -abs(np.vdot(gs, psi)) ** 2

    L = len(starts[0][1]) // 2
    bounds = [(-t_gamma, 2 * t_gamma)] * L + [(-BETA_PERIOD, 2 * BETA_PERIOD)] * L
    best_x, best_val, winner, nfev = None, np.inf, None, 0
    for name, x0 in starts:
        x0 = np.asarray(x0, dtype=float)
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
    psi = state(best_x)
    E = energy_of(psi, hc, n, gamma)
    F = float(abs(np.vdot(gs, psi)) ** 2)
    return {
        "gammas": best_x[:L].tolist(),
        "betas": best_x[L:].tolist(),
        "energy": E,
        "F": F,
        "winner": winner,
        "nfev": nfev,
        "psi": psi,
    }


def higher_starts(L, prev_g, prev_b, rng, n_random, t_gamma):
    g_prev = [float(v) for v in prev_g]
    b_prev = [float(v) for v in prev_b]
    starts = []
    if L == 2:
        g1, b1 = g_prev[0], b_prev[0]
        starts.append(("pagano_p2", _pack([g1, g1 + 0.2], [b1, b1 - 0.2])))
        starts.append(("p2_small", _pack([g1, g1 + 0.1], [b1, b1 - 0.1])))
        starts.append(("p2_repeat", _pack([g1, g1], [b1, b1])))
    if L >= 3:
        gi = interpolate(g_prev, L)
        bi = interpolate(b_prev, L)
        starts.append(("pagano_interp", _pack(gi, bi)))
        starts.append(("interp_pert", _pack(gi + rng.normal(0, 0.05, L), bi + rng.normal(0, 0.08, L))))
    starts.append(("zeropad_append", _pack(g_prev + [0.0], b_prev + [0.0])))
    starts.append(("zeropad_prepend", _pack([0.0] + g_prev, [0.0] + b_prev)))
    starts.append(("signflip", _pack([-v for v in (interpolate(g_prev, L) if L >= 3 else g_prev + [g_prev[-1]])],
                                     [-v for v in (interpolate(b_prev, L) if L >= 3 else b_prev + [b_prev[-1]])])))
    for i in range(n_random):
        starts.append((f"random_{i}", _pack(rng.uniform(0, t_gamma, L), rng.uniform(0, BETA_PERIOD, L))))
    return starts


def anneal_angles(L, gamma_field, qaoa_duration):
    """H(s)=(1-s)(-sum X)+s(H_C-Gamma sum X). gamma=dt*s, beta=-dt*((1-s)+Gamma*s).

    The single scale matches sum |γ|+|β| to the supplied QAOA duration.
    """
    if L <= 0 or not np.isfinite(qaoa_duration) or qaoa_duration <= 0:
        return [], []
    s = (np.arange(1, L + 1) - 0.5) / float(L)
    g = s
    b = -((1.0 - s) + float(gamma_field) * s)
    raw = float(np.sum(np.abs(g) + np.abs(b)))
    scale = float(qaoa_duration) / raw
    return (scale * g).tolist(), (scale * b).tolist()


def eta(E, E0, Emax):
    den = float(Emax - E0)
    if abs(den) < 1e-15:
        return float("nan")
    return float((Emax - E) / den)


def run_ladder(hc, n, gamma, gs, E0, Emax, t_gamma, max_L, n_grid, n_random, seed):
    rng = np.random.default_rng(seed)
    gs = np.asarray(gs, dtype=np.complex128)
    psi0 = plus_state(n)
    rows = []
    E_plus = energy_of(psi0, hc, n, gamma)
    rows.append({
        "L": 0, "F": float(abs(np.vdot(gs, psi0)) ** 2), "energy": E_plus,
        "eta": eta(E_plus, E0, Emax), "winner": "plus",
        "gammas": [], "betas": [], "gammas_canonical": [], "betas_canonical": [],
        "duration_raw": 0.0, "duration_canonical": 0.0,
        "energy_nested": True, "fidelity_monotone": True, "nfev": 0,
    })
    _, _, Es, Fs = l1_grid(hc, n, gamma, gs, t_gamma, n_grid)
    # top separated energy and fidelity seeds
    def seeds_from(score, largest, prefix, k=6):
        flat = score.ravel()
        order = np.argsort(flat)
        if largest:
            order = order[::-1]
        out = []
        coords = []
        betas = np.linspace(0.0, BETA_PERIOD, n_grid, endpoint=False)
        gammas = np.linspace(0.0, t_gamma, n_grid, endpoint=False)
        for t in order:
            ib, ig = divmod(int(t), n_grid)
            b, g = float(betas[ib]), float(gammas[ig])
            if any(abs(b - bb) < 0.15 and abs(g - gg) < max(0.05, t_gamma / 20) for bb, gg in coords):
                continue
            coords.append((b, g))
            out.append((f"{prefix}{len(out)}", _pack([g], [b])))
            if len(out) >= k:
                break
        return out

    e_starts = seeds_from(Es, False, "gridE_")
    e_starts.append(("zero", _pack([0.0], [0.0])))
    for i in range(3):
        e_starts.append((f"random_{i}", _pack(rng.uniform(0, t_gamma, 1), rng.uniform(0, BETA_PERIOD, 1))))
    optE = optimize(hc, n, gamma, e_starts, 50, "energy", gs, t_gamma)
    f_starts = seeds_from(Fs, True, "gridF_")
    for i in range(2):
        f_starts.append((f"frand_{i}", _pack(rng.uniform(0, t_gamma, 1), rng.uniform(0, BETA_PERIOD, 1))))
    optF = optimize(hc, n, gamma, f_starts, 40, "fidelity", gs, t_gamma)
    # carry zero so energy cannot exceed the |+> energy if the optimizer fails; L=1 need not beat |+> energy? 
    # Energy of |+> may be higher than grid. Nested from L=0 means E(L)>= wait no: lower is better.
    # Include |+> as a feasible point only if angles 0 reproduce it, which they do.
    if E_plus < optE["energy"] - 1e-9:
        optE = {
            "gammas": [0.0], "betas": [0.0], "energy": E_plus, "F": rows[0]["F"],
            "winner": "carry_plus", "nfev": optE["nfev"], "psi": psi0,
        }
    prev = optE
    row1 = _row(1, optE, t_gamma, E0, Emax, True, True, extra={
        "l1_grid_max_fidelity": float(Fs.max()),
        "l1_fidelity_max_F": optF["F"],
        "l1_fidelity_max_energy": optF["energy"],
        "l1_F_gap_fidopt_minus_energyopt": float(optF["F"] - optE["F"]),
        "false_hard": bool(optF["F"] >= optE["F"] + 0.25 and optF["F"] >= 0.4),
        "fid_winner": optF["winner"],
        "fid_gammas": optF["gammas"],
        "fid_betas": optF["betas"],
    })
    g_a, b_a = anneal_angles(1, gamma, row1["duration_raw"])
    if g_a:
        psi_a = apply_qaoa(plus_state(n), hc, n, g_a, b_a)
        row1["anneal_F"] = float(abs(np.vdot(gs, psi_a)) ** 2)
        row1["anneal_energy"] = energy_of(psi_a, hc, n, gamma)
    rows.append(row1)
    for L in range(2, max_L + 1):
        starts = higher_starts(L, prev["gammas"], prev["betas"], rng, n_random, t_gamma)
        this = optimize(hc, n, gamma, starts, 36, "energy", gs, t_gamma)
        # zero-pad carry
        g_pad = list(prev["gammas"]) + [0.0]
        b_pad = list(prev["betas"]) + [0.0]
        psi_pad = apply_qaoa(plus_state(n), hc, n, g_pad, b_pad)
        E_pad = energy_of(psi_pad, hc, n, gamma)
        if E_pad < this["energy"] - 1e-9:
            this = {
                "gammas": g_pad, "betas": b_pad, "energy": E_pad,
                "F": float(abs(np.vdot(gs, psi_pad)) ** 2),
                "winner": "carry", "nfev": this["nfev"], "psi": psi_pad,
            }
        nested = this["energy"] <= prev["energy"] + 1e-7
        mono = this["F"] + 1e-8 >= prev["F"]
        rows.append(_row(L, this, t_gamma, E0, Emax, nested, mono))
        # shared gamma sidecar
        sh = _shared(hc, n, gamma, gs, this["gammas"], this["betas"], rng, t_gamma)
        rows[-1]["shared_F"] = sh["F"]
        rows[-1]["shared_energy"] = sh["energy"]
        rows[-1]["shared_gamma"] = sh["gamma"]
        # matched anneal on the raw duration
        g_a, b_a = anneal_angles(L, gamma, duration(this["gammas"], this["betas"]))
        if g_a:
            psi_a = apply_qaoa(plus_state(n), hc, n, g_a, b_a)
            rows[-1]["anneal_F"] = float(abs(np.vdot(gs, psi_a)) ** 2)
            rows[-1]["anneal_energy"] = energy_of(psi_a, hc, n, gamma)
            rows[-1]["anneal_gammas"] = g_a
            rows[-1]["anneal_betas"] = b_a
        prev = this
    # replay check of the last energy-opt state
    last = rows[-1]
    if last["L"] >= 1 and last.get("gammas"):
        psi = apply_qaoa(plus_state(n), hc, n, last["gammas"], last["betas"])
        E = energy_of(psi, hc, n, gamma)
        F = float(abs(np.vdot(gs, psi)) ** 2)
        last["replay_energy_err"] = abs(E - last["energy"])
        last["replay_F_err"] = abs(F - last["F"])
    return rows, optF


def energy_of(psi, hc, n, gamma):
    # local name used above before definition order... defined below? 
    # Python function body resolves energy_of at call time, and we define it later in this function's module.
    ec = float(np.vdot(psi, hc * psi).real)
    ex = 0.0
    idx = np.arange(psi.shape[0], dtype=np.int64)
    for q in range(n):
        ex += float(np.vdot(psi, psi[idx ^ (1 << q)]).real)
    return ec - float(gamma) * ex


def duration(gammas, betas):
    return float(np.sum(np.abs(gammas)) + np.sum(np.abs(betas)))


def _row(L, rec, t_gamma, E0, Emax, nested, mono, extra=None):
    gc, bc = canonical(rec["gammas"], rec["betas"], t_gamma)
    row = {
        "L": L,
        "F": rec["F"],
        "energy": rec["energy"],
        "eta": eta(rec["energy"], E0, Emax),
        "winner": rec["winner"],
        "gammas": rec["gammas"],
        "betas": rec["betas"],
        "gammas_canonical": gc.tolist(),
        "betas_canonical": bc.tolist(),
        "duration_raw": duration(rec["gammas"], rec["betas"]),
        "duration_canonical": duration(gc, bc),
        "energy_nested": bool(nested),
        "fidelity_monotone": bool(mono),
        "nfev": rec.get("nfev", 0),
    }
    if extra:
        row.update(extra)
    return row


def _shared(hc, n, gamma, gs, gammas, betas, rng, t_gamma):
    L = len(gammas)
    starts = [("mean", np.concatenate([[float(np.mean(gammas))], np.asarray(betas, dtype=float)]))]
    starts.append(("zero", np.zeros(1 + L)))
    starts.append(("rand", np.concatenate([rng.uniform(0, t_gamma, 1), rng.uniform(0, BETA_PERIOD, L)])))

    def obj(x):
        psi = apply_qaoa(plus_state(n), hc, n, [float(x[0])] * L, x[1:])
        return energy_of(psi, hc, n, gamma)

    bounds = [(-t_gamma, 2 * t_gamma)] + [(-BETA_PERIOD, 2 * BETA_PERIOD)] * L
    best_x, best_val = None, np.inf
    for _, x0 in starts:
        seed_val = float(obj(x0))
        x_opt, val, _ = _lbfgs(obj, x0, 30, bounds)
        if seed_val < val:
            x_opt, val = np.asarray(x0, float), seed_val
        if val < best_val:
            best_x, best_val = x_opt, val
    g = float(best_x[0])
    b = best_x[1:]
    psi = apply_qaoa(plus_state(n), hc, n, [g] * L, b)
    return {"gamma": g, "F": float(abs(np.vdot(gs, psi)) ** 2), "energy": energy_of(psi, hc, n, gamma)}
