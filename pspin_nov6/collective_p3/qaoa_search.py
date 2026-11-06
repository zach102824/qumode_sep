"""Pagano-style QAOA search on the symmetric block.

Primary ansatz: independent (gamma_l, beta_l), cost then mixer, layer 1 first.
    U = prod_l exp(-i beta_l * sum_i X_i) exp(-i gamma_l * H_cost)
    H_cost = -N m_z^3 = -(sum Z)^3 / N^2
Objective: <H_target>, H_target = H_cost - Gamma * sum X. Not fidelity.

gamma multiplies H_cost (adjacent Dicke gaps are O(1)). beta multiplies sum X,
so one qubit sees exp(-i beta X) = RX(2 beta) in the Qiskit convention.
A second angle pair (gamma_W, beta_W) is recorded only as a conversion:
    gamma_W = gamma / N^2     (Wauters super-extensive generator -(sum Z)^3)
    beta_W = -beta            (Wauters mixer generator -sum X)
The Pagano +/- 0.2 bootstrap step is applied to (gamma, beta), because those
are the O(1) generators. It is not applied to gamma_W, which is O(gamma/N^2).
"""

from __future__ import annotations

import math

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize

from symmetric import SymmetricModel, canonicalize, observe


def evaluate(model: SymmetricModel, gammas, betas) -> tuple[float, float, np.ndarray]:
    g = np.atleast_1d(np.asarray(gammas, dtype=np.float64))
    b = np.atleast_1d(np.asarray(betas, dtype=np.float64))
    if g.size == 0:
        psi = model.plus.copy()
    else:
        psi = model.evolve(g, b)
    return model.energy_of(psi), model.fidelity(psi), psi


def pack(gammas: np.ndarray, betas: np.ndarray) -> np.ndarray:
    out = np.empty(gammas.size + betas.size, dtype=np.float64)
    out[0::2] = gammas
    out[1::2] = betas
    return out


def unpack(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np.asarray(x[0::2], dtype=np.float64), np.asarray(x[1::2], dtype=np.float64)


def local_minimize(model: SymmetricModel, gammas, betas, maxiter: int = 250) -> tuple[np.ndarray, np.ndarray, float, float]:
    g0 = np.asarray(gammas, dtype=np.float64)
    b0 = np.asarray(betas, dtype=np.float64)
    if g0.size == 0:
        e, f, _ = evaluate(model, g0, b0)
        return g0, b0, e, f

    def fun(x: np.ndarray) -> float:
        g, b = unpack(x)
        psi = model.evolve(g, b)
        return model.energy_of(psi)

    res = minimize(fun, pack(g0, b0), method="BFGS", options={"maxiter": maxiter, "gtol": 1e-8})
    g, b = unpack(res.x)
    e, f, _ = evaluate(model, g, b)
    return g, b, e, f


def local_maximize_fidelity(model: SymmetricModel, gamma: float, beta: float) -> tuple[float, float, float, float]:
    def fun(x: np.ndarray) -> float:
        psi = model.evolve(x[:1], x[1:])
        return -model.fidelity(psi)

    res = minimize(fun, np.array([gamma, beta], dtype=np.float64), method="BFGS", options={"maxiter": 200, "gtol": 1e-9})
    g = float(res.x[0])
    b = float(res.x[1])
    e, f, _ = evaluate(model, [g], [b])
    return g, b, e, f


def l1_grid(model: SymmetricModel, period: float, n_gamma: int, n_beta: int, gamma_focus: float) -> dict:
    """Exhaustive L=1 grid over one gamma period and a finer focus window."""
    betas = np.linspace(0.0, math.pi, n_beta, endpoint=False)
    # Full period. Also the negative half, since the cell is [-T/2, T/2).
    gammas_full = np.linspace(-0.5 * period, 0.5 * period, n_gamma, endpoint=False)
    focus_hi = min(0.5 * period, gamma_focus)
    gammas_focus = np.linspace(-focus_hi, focus_hi, max(n_gamma // 2, 81))
    gammas = np.unique(np.concatenate([gammas_full, gammas_focus]))

    best_e = math.inf
    best_e_angles = (0.0, 0.0)
    best_f = -1.0
    best_f_angles = (0.0, 0.0)
    # Keep a coarse list of good energy cells for polishing: subsample storage.
    kept_g = []
    kept_b = []
    kept_e = []
    chunk = 512
    ec = model.e_cost
    v = model.mixer_v
    lam = model.mixer_lam
    gs = model.gs
    plus = model.plus
    for start in range(0, gammas.size, chunk):
        gsub = gammas[start : start + chunk]
        phased = plus[None, :] * np.exp(-1j * gsub[:, None] * ec[None, :])
        coeff = phased @ v  # (G, D)
        sx_exp = np.sum((np.abs(coeff) ** 2) * lam[None, :], axis=1).real  # (G,)
        phase_b = np.exp(-1j * betas[:, None] * lam[None, :])  # (B, D)
        # c_phase[g,b,k] = coeff[g,k] * phase_b[b,k]
        c_phase = coeff[:, None, :] * phase_b[None, :, :]
        psi = np.einsum("gbk,dk->gbd", c_phase, v, optimize=True)
        hc = np.sum((np.abs(psi) ** 2) * ec[None, None, :], axis=2).real
        energy = hc - model.gamma * sx_exp[:, None]
        amp = np.einsum("d,gbd->gb", gs, psi, optimize=True)
        fid = np.abs(amp) ** 2
        # best in chunk
        flat_e = np.argmin(energy)
        ge, be = np.unravel_index(flat_e, energy.shape)
        if energy[ge, be] < best_e:
            best_e = float(energy[ge, be])
            best_e_angles = (float(gsub[ge]), float(betas[be]))
        flat_f = np.argmax(fid)
        gf, bf = np.unravel_index(flat_f, fid.shape)
        if fid[gf, bf] > best_f:
            best_f = float(fid[gf, bf])
            best_f_angles = (float(gsub[gf]), float(betas[bf]))
        # retain top cells per chunk
        k = min(8, energy.size)
        idx = np.argpartition(energy.ravel(), k - 1)[:k]
        for flat in idx:
            ig, ib = np.unravel_index(int(flat), energy.shape)
            kept_g.append(float(gsub[ig]))
            kept_b.append(float(betas[ib]))
            kept_e.append(float(energy[ig, ib]))
    order = np.argsort(kept_e)
    basins = []
    for i in order:
        g = kept_g[i]
        b = kept_b[i]
        if any(abs(g - g2) < 0.12 and abs(b - b2) < 0.12 for g2, b2, _e in basins):
            continue
        basins.append((g, b, kept_e[i]))
        if len(basins) >= 10:
            break
    return {
        "best_energy": best_e,
        "best_energy_angles": best_e_angles,
        "best_fidelity": best_f,
        "best_fidelity_angles": best_f_angles,
        "basins": basins,
        "n_gamma": int(gammas.size),
        "n_beta": int(betas.size),
    }


def interpolate_schedule(gammas: np.ndarray, betas: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pagano Suppl. resampling from depth q-1 to depth q.

    Old nodes sit at s_i = (i-1)/(q-2). New nodes are evenly spaced by 1/(q-1).
    Linear when the new depth is 3. Cubic for new depth > 3.
    """
    old_l = int(gammas.size)
    new_l = old_l + 1
    old_s = np.linspace(0.0, 1.0, old_l)
    new_s = np.linspace(0.0, 1.0, new_l)
    if new_l <= 3 or old_l < 3:
        return np.interp(new_s, old_s, gammas), np.interp(new_s, old_s, betas)
    gs = CubicSpline(old_s, gammas, bc_type="natural")(new_s)
    bs = CubicSpline(old_s, betas, bc_type="natural")(new_s)
    return np.asarray(gs, dtype=np.float64), np.asarray(bs, dtype=np.float64)


def linear_anneal_angles(depth: int, gamma_field: float, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """Wauters first-order Trotter of H(s) = s H_target + (1-s) (-sum X), constant dt.

    s_m = m/L, m=1..L. Converted to our (gamma, beta) generators.
    gamma = s * dt, beta = -dt * ((1-s) + Gamma * s).
    """
    s = np.arange(1, depth + 1, dtype=np.float64) / float(depth)
    gammas = s * dt
    betas = -dt * ((1.0 - s) + gamma_field * s)
    return gammas, betas


def _consider(pool: list, source: str, energy: float, fid: float, gammas, betas, optimized: bool) -> None:
    pool.append(
        {
            "source": source,
            "optimized": optimized,
            "energy": float(energy),
            "F": float(fid),
            "gamma": np.asarray(gammas, dtype=np.float64).copy(),
            "beta": np.asarray(betas, dtype=np.float64).copy(),
        }
    )


def _opt_and_store(model, pool, source, gammas, betas, maxiter: int) -> None:
    try:
        g, b, e, f = local_minimize(model, gammas, betas, maxiter=maxiter)
    except Exception:
        return
    if not (np.isfinite(e) and np.isfinite(f)):
        return
    _consider(pool, source, e, f, g, b, optimized=True)


def search_depth(
    model: SymmetricModel,
    depth: int,
    rng: np.random.Generator,
    period: float,
    prev: dict | None,
    transfer: tuple[np.ndarray, np.ndarray] | None,
    grid_info: dict | None,
    maxiter: int,
) -> dict:
    """Energy-minimizing search at one depth. Keeps the best energy, not the best fidelity."""
    pool: list[dict] = []
    if depth == 0:
        e, f, psi = evaluate(model, [], [])
        _consider(pool, "zero_layer", e, f, np.zeros(0), np.zeros(0), optimized=False)
        return _finish(model, pool, period, psi_cache={id(pool[0]["gamma"].tobytes()): psi})

    if depth == 1:
        assert grid_info is not None
        for g, b, _e in grid_info["basins"]:
            e_exact, f_exact, _ = evaluate(model, [g], [b])
            _consider(pool, "l1_grid_exact", e_exact, f_exact, [g], [b], optimized=False)
            _opt_and_store(model, pool, "l1_grid_basin", [g], [b], maxiter)
        # literal Pagano-scale random starts inside a few radians, plus a few across the period
        for i in range(6):
            g = float(rng.uniform(-2.5, 2.5))
            b = float(rng.uniform(-math.pi / 2, math.pi / 2))
            _opt_and_store(model, pool, f"l1_random_small_{i}", [g], [b], maxiter)
        for i in range(3):
            g = float(rng.uniform(-0.5 * period, 0.5 * period))
            b = float(rng.uniform(-math.pi / 2, math.pi / 2))
            _opt_and_store(model, pool, f"l1_random_period_{i}", [g], [b], maxiter)
        for dt in (0.4, 1.0, 2.0):
            g, b = linear_anneal_angles(1, model.gamma, dt)
            _opt_and_store(model, pool, f"l1_linear_dt_{dt}", g, b, maxiter)
        fid_diag = _fidelity_diagnostic(model, grid_info)
    else:
        fid_diag = None
        assert prev is not None
        g_prev = prev["gamma_raw"]
        b_prev = prev["beta_raw"]
        # Exact zero-layer extensions. These lock energy monotonicity.
        for tag, g, b in (
            ("zeropad_end", np.r_[g_prev, 0.0], np.r_[b_prev, 0.0]),
            ("zeropad_front", np.r_[0.0, g_prev], np.r_[0.0, b_prev]),
        ):
            e, f, _ = evaluate(model, g, b)
            _consider(pool, tag + "_exact", e, f, g, b, optimized=False)
            _opt_and_store(model, pool, tag, g, b, maxiter)
        if depth == 2:
            for delta in (0.05, 0.1, 0.2, 0.4):
                g = np.array([g_prev[0], g_prev[0] + delta])
                b = np.array([b_prev[0], b_prev[0] - delta])
                _opt_and_store(model, pool, f"pagano_delta_{delta}", g, b, maxiter)
        else:
            g_i, b_i = interpolate_schedule(g_prev, b_prev)
            e, f, _ = evaluate(model, g_i, b_i)
            _consider(pool, "pagano_interp_exact", e, f, g_i, b_i, optimized=False)
            _opt_and_store(model, pool, "pagano_interp", g_i, b_i, maxiter)
            g_p = g_i.copy()
            b_p = b_i.copy()
            g_p[-1] += 0.2
            b_p[-1] -= 0.2
            _opt_and_store(model, pool, "interp_end_perturb_0.2", g_p, b_p, maxiter)
        g_scale = max(2.0, float(np.max(np.abs(g_prev))) * 3.0)
        for i in range(5):
            g = rng.uniform(-g_scale, g_scale, size=depth)
            b = rng.uniform(-math.pi / 2, math.pi / 2, size=depth)
            _opt_and_store(model, pool, f"random_small_{i}", g, b, maxiter)
        for i in range(2):
            g = rng.uniform(-0.5 * period, 0.5 * period, size=depth)
            b = rng.uniform(-0.5 * math.pi, 0.5 * math.pi, size=depth)
            _opt_and_store(model, pool, f"random_period_{i}", g, b, maxiter)
        # perturbations of the previous schedule's zero-pad
        base_g = np.r_[g_prev, g_prev[-1]]
        base_b = np.r_[b_prev, b_prev[-1]]
        for i in range(3):
            g = base_g + rng.normal(0.0, 0.15, size=depth)
            b = base_b + rng.normal(0.0, 0.15, size=depth)
            _opt_and_store(model, pool, f"perturb_extend_{i}", g, b, maxiter)
        for dt in (0.5, 1.0, 2.0, 4.0):
            g, b = linear_anneal_angles(depth, model.gamma, dt)
            _opt_and_store(model, pool, f"linear_dt_{dt}", g, b, maxiter)
        if transfer is not None:
            tg, tb = transfer
            if tg.size == depth:
                _opt_and_store(model, pool, "n_transfer", tg, tb, maxiter)

    return _finish(model, pool, period, fid_diag=fid_diag)


def _fidelity_diagnostic(model: SymmetricModel, grid_info: dict) -> dict:
    starts = [grid_info["best_fidelity_angles"], grid_info["best_energy_angles"]]
    best = None
    for g, b in starts:
        gg, bb, ee, ff = local_maximize_fidelity(model, g, b)
        rec = {"gamma": gg, "beta": bb, "energy": ee, "F": ff}
        if best is None or ff > best["F"]:
            best = rec
    return best


def _finish(model, pool: list, period: float, fid_diag=None, psi_cache=None) -> dict:
    pool = [r for r in pool if np.isfinite(r["energy"]) and np.isfinite(r["F"])]
    if not pool:
        raise RuntimeError("no finite QAOA evaluations")
    # Tie-break: lower energy, then higher fidelity. Fidelity is not the objective.
    pool_sorted = sorted(pool, key=lambda r: (r["energy"], -r["F"]))
    best = pool_sorted[0]
    g_raw = best["gamma"]
    b_raw = best["beta"]
    g_can, b_can, notes = canonicalize(g_raw, b_raw, period)
    # Re-evaluate canonical angles so stored energy matches the folded schedule.
    e_can, f_can, psi = evaluate(model, g_can, b_can)
    e_raw, f_raw, psi_raw = evaluate(model, g_raw, b_raw)
    if abs(e_can - e_raw) > 1e-6 or abs(f_can - f_raw) > 1e-6:
        notes = notes + ["canonical_mismatch"]
    obs = observe(model, psi_raw)
    ends = [r for r in pool if r["optimized"]] or pool
    f_ends = np.array([r["F"] for r in ends], dtype=np.float64)
    e_ends = np.array([r["energy"] for r in ends], dtype=np.float64)
    e_best = float(best["energy"])
    span = max(1e-6, abs(model.energy_of(model.plus) - model.e0))
    near = [r for r in ends if r["energy"] <= e_best + 0.02 * span]
    f_near = np.array([r["F"] for r in near], dtype=np.float64) if near else f_ends
    # conjugation symmetry check on the raw winner
    if g_raw.size:
        e_conj, f_conj, _ = evaluate(model, -g_raw, -b_raw)
        conj_energy_err = abs(e_conj - e_raw)
        conj_f_err = abs(f_conj - f_raw)
    else:
        conj_energy_err = 0.0
        conj_f_err = 0.0
    return {
        "source": best["source"],
        "energy": float(obs["energy"]),
        "F": float(obs["F"]),
        "eta": float(obs["eta"]),
        "mz": float(obs["mz"]),
        "mx": float(obs["mx"]),
        "entropy_bits": float(obs["entropy_bits"]),
        "F_plus": float(obs["F_plus"]),
        "F_coherent": float(obs["F_coherent"]),
        "F_bitstring": float(obs["F_bitstring"]),
        "gamma_raw": np.asarray(g_raw, dtype=np.float64),
        "beta_raw": np.asarray(b_raw, dtype=np.float64),
        "gamma_can": g_can,
        "beta_can": b_can,
        "canon_notes": notes,
        "conj_energy_err": float(conj_energy_err),
        "conj_F_err": float(conj_f_err),
        "sum_abs_gamma": float(np.sum(np.abs(g_raw))) if g_raw.size else 0.0,
        "sum_abs_beta": float(np.sum(np.abs(b_raw))) if b_raw.size else 0.0,
        "n_starts": len(pool),
        "n_optimized": len(ends),
        "n_near": len(near),
        "F_std": float(np.std(f_ends)),
        "F_min": float(np.min(f_ends)),
        "F_max": float(np.max(f_ends)),
        "F_near_std": float(np.std(f_near)) if f_near.size else 0.0,
        "E_std": float(np.std(e_ends)),
        "fid_diag": fid_diag,
        "psi": psi_raw,
    }


def shared_gamma_search(
    model: SymmetricModel,
    depth: int,
    rng: np.random.Generator,
    independent: dict,
    maxiter: int,
) -> dict | None:
    """One shared gamma, independent betas. Diagnostic only."""
    if depth < 1:
        return None
    pool: list[dict] = []

    def expand(gamma: float, betas: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return np.full(depth, float(gamma)), np.asarray(betas, dtype=np.float64)

    def opt_shared(gamma0: float, beta0: np.ndarray, source: str) -> None:
        x0 = np.r_[float(gamma0), np.asarray(beta0, dtype=np.float64)]

        def fun(x: np.ndarray) -> float:
            g, b = expand(x[0], x[1:])
            return model.energy_of(model.evolve(g, b))

        res = minimize(fun, x0, method="BFGS", options={"maxiter": maxiter, "gtol": 1e-8})
        g, b = expand(res.x[0], res.x[1:])
        e, f, _ = evaluate(model, g, b)
        _consider(pool, source, e, f, g, b, optimized=True)

    g_ind = independent["gamma_raw"]
    b_ind = independent["beta_raw"]
    if g_ind.size == depth:
        opt_shared(float(np.mean(g_ind)), b_ind, "shared_from_independent_mean")
    for dt in (0.5, 1.0, 2.0):
        g, b = linear_anneal_angles(depth, model.gamma, dt)
        opt_shared(float(np.mean(g)), b, f"shared_linear_dt_{dt}")
    g_scale = max(2.0, float(np.max(np.abs(g_ind))) * 2.0 if g_ind.size else 2.0)
    for i in range(3):
        opt_shared(
            float(rng.uniform(-g_scale, g_scale)),
            rng.uniform(-math.pi / 2, math.pi / 2, size=depth),
            f"shared_random_{i}",
        )
    best = min(pool, key=lambda r: (r["energy"], -r["F"]))
    return {
        "source": best["source"],
        "energy": best["energy"],
        "F": best["F"],
        "gamma_raw": best["gamma"],
        "beta_raw": best["beta"],
        "sum_abs_gamma": float(np.sum(np.abs(best["gamma"]))),
        "sum_abs_beta": float(np.sum(np.abs(best["beta"]))),
        "F_std": float(np.std([r["F"] for r in pool])),
        "n_starts": len(pool),
    }


def matched_digitized_anneal(model: SymmetricModel, independent: dict) -> dict | None:
    """Linear ramp with the same L and the same sum|gamma| and sum|beta| as the QAOA winner."""
    g = independent["gamma_raw"]
    b = independent["beta_raw"]
    depth = int(g.size)
    if depth == 0:
        return None
    tg = float(np.sum(np.abs(g)))
    tb = float(np.sum(np.abs(b)))
    s = np.arange(1, depth + 1, dtype=np.float64) / float(depth)
    g_shape = s
    b_shape = (1.0 - s) + model.gamma * s
    g_sched = g_shape / g_shape.sum() * tg if tg > 0 else np.zeros(depth)
    b_mag = b_shape / b_shape.sum() * tb if tb > 0 else np.zeros(depth)
    trials = []
    for sign in (1.0, -1.0):
        betas = sign * b_mag
        e, f, _ = evaluate(model, g_sched, betas)
        trials.append((e, f, sign, betas))
    trials.sort(key=lambda t: t[0])
    e, f, sign, betas = trials[0]
    return {
        "energy": e,
        "F": f,
        "beta_sign": sign,
        "gamma": g_sched,
        "beta": betas,
        "sum_abs_gamma": float(np.sum(np.abs(g_sched))),
        "sum_abs_beta": float(np.sum(np.abs(betas))),
        "other_sign_F": trials[1][1],
        "other_sign_energy": trials[1][0],
    }


def trotter_sweep(model: SymmetricModel, depth: int) -> dict | None:
    """Unmatched physical linear Trotter sweep over total time. Not a budget-matched baseline."""
    if depth == 0:
        return None
    best = None
    records = []
    for tau in np.geomspace(0.05, 30.0, 18):
        dt = float(tau) / float(depth)
        g, b = linear_anneal_angles(depth, model.gamma, dt)
        e, f, _ = evaluate(model, g, b)
        rec = {"tau": float(tau), "energy": e, "F": f, "sum_abs_gamma": float(np.sum(np.abs(g))), "sum_abs_beta": float(np.sum(np.abs(b)))}
        records.append(rec)
        if best is None or e < best["energy"]:
            best = rec
    best = dict(best)
    best["n_tau"] = len(records)
    return best


def sign_flip_fidelity(model_f: SymmetricModel, model_af: SymmetricModel, gammas, betas) -> dict:
    """Same angles on +J and -J. Odd p: global X conjugation, fidelities must match."""
    _e_f, f_f, _ = evaluate(model_f, gammas, betas)
    _e_af, f_af, _ = evaluate(model_af, gammas, betas)
    # spectra
    return {
        "F_ferro": f_f,
        "F_af": f_af,
        "F_abs_diff": abs(f_f - f_af),
        "E0_abs_diff": abs(model_f.e0 - model_af.e0),
        "Emax_abs_diff": abs(model_f.emax - model_af.emax),
    }
