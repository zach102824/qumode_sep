"""QAOA for the sparse 3-body target. Energy is the objective, not fidelity.

Ansatz, layer ℓ = 1..L applied in that order (ℓ = 1 first)::

    |ψ⟩ = ∏_{ℓ=L,...,1} exp(-i β_ℓ H_X) exp(-i γ_ℓ H_C) |+⟩

    H_C = - sum_a J_a ZZZ_a     (diagonal cost, J_a = +1 in production)
    H_X = sum_i X_i             (mixer)

Full target whose expectation is minimized::

    H = H_C - Gamma * H_X

Gamma is a Hamiltonian parameter, not an angle rescaling. Both generators
have spectra in a single parity class (all eigenvalues ≡ N mod 2, and the
cost eigenvalues step by 2), so γ and β each have exact period π and not,
in general, period π/2. Adding π to one angle multiplies the state by a
global phase. The map (γ, β) -> (-γ, -β) sends the state to its complex
conjugate and leaves the energy and the fidelity to a real ground state
unchanged.

Pagano et al., arXiv:1906.02700, Suppl. I.1.2 supplies the bootstrap:
grid/local at L = 1, a perturbed copy of the L = 1 schedule at L = 2,
and interpolation of the previous schedule for L >= 3. Zero-layer pads and
extra random starts are additional seeds. They do not replace the energy
objective. A lower energy at higher L does not imply a higher fidelity.
"""

from __future__ import annotations

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize

PERIOD = np.pi


def apply_cost(psi: np.ndarray, hc_diag: np.ndarray, gamma: float) -> np.ndarray:
    phase = np.exp(-1j * float(gamma) * hc_diag)
    return np.asarray(psi) * phase


def apply_mixer(psi: np.ndarray, beta: float, n: int) -> np.ndarray:
    """exp(-i β sum_i X_i) = ⊗_i (cos β I - i sin β X)."""
    c = np.cos(float(beta))
    s = np.sin(float(beta))
    out = np.array(psi, dtype=np.complex128, copy=True)
    idx = np.arange(out.shape[0], dtype=np.int64)
    for q in range(n):
        out = c * out - 1j * s * out[idx ^ (1 << q)]
    return out


def apply_qaoa(psi0, hc_diag, n, gammas, betas) -> np.ndarray:
    if len(gammas) != len(betas):
        raise ValueError("gamma and beta lists must have the same length")
    psi = np.array(psi0, dtype=np.complex128, copy=True)
    for g, b in zip(gammas, betas):
        psi = apply_cost(psi, hc_diag, g)
        psi = apply_mixer(psi, b, n)
    return psi


def expectations(psi, hc_diag, n: int):
    amp = np.asarray(psi, dtype=np.complex128)
    ec = float(np.vdot(amp, hc_diag * amp).real)
    ex = 0.0
    idx = np.arange(amp.shape[0], dtype=np.int64)
    for q in range(n):
        ex += float(np.vdot(amp, amp[idx ^ (1 << q)]).real)
    return ec, ex


def energy_of(psi, hc_diag, n: int, gamma_field: float) -> float:
    ec, ex = expectations(psi, hc_diag, n)
    return float(ec - float(gamma_field) * ex)


def fold_angle(theta: float, period: float = PERIOD):
    """Map θ to (-period/2, period/2] and return the integer branch index k.

    θ = principal + k * period. Numpy round-half-even is used, matching
    np.round, so the representative is reproducible.
    """
    k = int(np.round(float(theta) / period))
    principal = float(theta - k * period)
    return principal, k


def fold_list(values):
    principals = []
    branches = []
    for v in values:
        p, k = fold_angle(float(v))
        principals.append(p)
        branches.append(k)
    return principals, branches


def branch_report(gammas, betas) -> dict:
    g_p, g_k = fold_list(gammas)
    b_p, b_k = fold_list(betas)
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
        "max_abs_branch_k": int(max([0] + [abs(k) for k in g_k + b_k])),
        "max_consecutive_principal_jump": float(max(jumps) if jumps else 0.0),
        "sum_abs_gamma_raw": float(np.sum(np.abs(gammas))) if len(gammas) else 0.0,
        "sum_abs_beta_raw": float(np.sum(np.abs(betas))) if len(betas) else 0.0,
        "sum_abs_gamma_principal": float(np.sum(np.abs(g_p))) if g_p else 0.0,
        "sum_abs_beta_principal": float(np.sum(np.abs(b_p))) if b_p else 0.0,
        "duration_principal": float(np.sum(np.abs(g_p)) + np.sum(np.abs(b_p))) if g_p else 0.0,
        "duration_raw": float(np.sum(np.abs(gammas)) + np.sum(np.abs(betas))) if len(gammas) else 0.0,
    }


def _lbfgs(objective, x0, maxiter: int):
    bounds = [(-PERIOD, 2.0 * PERIOD)] * len(x0)
    res = minimize(
        objective,
        np.asarray(x0, dtype=np.float64),
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": int(maxiter), "ftol": 1e-14, "gtol": 1e-8, "eps": 1e-7},
    )
    return np.asarray(res.x, dtype=np.float64), float(res.fun), int(res.nfev), bool(res.success)


def optimize_angles(
    psi0,
    hc_diag,
    n: int,
    gamma_field: float,
    starts,
    maxiter: int,
    objective: str = "energy",
    gs=None,
):
    """Minimize energy, or independently minimize -fidelity when asked.

    Starts that evaluate worse than the incumbent are kept only if the local
    optimizer improves them. The best start's own value is always a candidate,
    so a failed local search cannot raise the energy above the seed.
    """

    def state(x):
        L = len(x) // 2
        gammas = x[:L]
        betas = x[L:]
        return apply_qaoa(psi0, hc_diag, n, gammas, betas)

    def energy_obj(x):
        return energy_of(state(x), hc_diag, n, gamma_field)

    def fidelity_obj(x):
        psi = state(x)
        return -float(abs(np.vdot(gs, psi)) ** 2)

    obj = energy_obj if objective == "energy" else fidelity_obj
    best_x = None
    best_val = np.inf
    nfev = 0
    winner = None
    for name, x0 in starts:
        x0 = np.asarray(x0, dtype=np.float64)
        seed_val = float(obj(x0))
        nfev += 1
        x_opt, val, n_loc, _ok = _lbfgs(obj, x0, maxiter=maxiter)
        nfev += n_loc
        if seed_val < val:
            # Local search failed to beat its seed. Keep the seed.
            x_opt, val = x0, seed_val
            name = name + ":seed"
        if val < best_val:
            best_val = val
            best_x = x_opt
            winner = name
    L = len(best_x) // 2
    gammas = [float(v) for v in best_x[:L]]
    betas = [float(v) for v in best_x[L:]]
    return {
        "gammas": gammas,
        "betas": betas,
        "objective_value": float(best_val),
        "winner": winner,
        "nfev": int(nfev),
        "objective": objective,
    }


def l1_grid(psi0, hc_diag, n, gamma_field, gs, n_grid: int = 29):
    """Joint energy and fidelity grid on [0, π) x [0, π)."""
    gs = np.asarray(gs, dtype=np.complex128)
    betas = np.linspace(0.0, PERIOD, int(n_grid), endpoint=False)
    gammas = np.linspace(0.0, PERIOD, int(n_grid), endpoint=False)
    # Vectorize over gamma at fixed beta: cost phases, then one mixer angle.
    plus = np.asarray(psi0, dtype=np.complex128)
    Es = np.empty((len(betas), len(gammas)), dtype=np.float64)
    Fs = np.empty_like(Es)
    idx = np.arange(plus.shape[0], dtype=np.int64)
    for ib, beta in enumerate(betas):
        phases = np.exp(-1j * gammas[:, None] * hc_diag[None, :])
        psi = plus[None, :] * phases
        c = np.cos(float(beta))
        s = np.sin(float(beta))
        for q in range(n):
            psi = c * psi - 1j * s * psi[:, idx ^ (1 << q)]
        ec = np.real(np.sum(np.conj(psi) * hc_diag[None, :] * psi, axis=1))
        ex = np.zeros(psi.shape[0], dtype=np.float64)
        for q in range(n):
            ex += np.real(np.sum(np.conj(psi) * psi[:, idx ^ (1 << q)], axis=1))
        Es[ib] = ec - float(gamma_field) * ex
        Fs[ib] = np.abs(psi @ np.conj(gs)) ** 2
    return gammas, betas, Es, Fs


def top_indices(values: np.ndarray, k: int, largest: bool):
    flat = values.ravel()
    kk = min(int(k), flat.size)
    if largest:
        pick = np.argpartition(flat, flat.size - kk)[-kk:]
        pick = pick[np.argsort(flat[pick])[::-1]]
    else:
        pick = np.argpartition(flat, kk - 1)[:kk]
        pick = pick[np.argsort(flat[pick])]
    coords = [divmod(int(t), values.shape[1]) for t in pick]
    return coords


def interpolate_schedule(prev, length: int) -> np.ndarray:
    """Pagano resampling.

    Previous length q-1 is tabulated at s_i = (i)/(q-2) for i = 0..q-2,
    which is linspace(0, 1, q-1). The new schedule is sampled at
    linspace(0, 1, q), i.e. spacing 1/(q-1). Linear for q = 3. For q > 3
    a natural cubic spline is used. With only three previous samples (q = 4)
    the cubic spline is still well-defined (two cubic pieces plus endpoint
    conditions); it is not a global cubic polynomial.
    """
    prev = np.asarray(prev, dtype=np.float64)
    q = int(length)
    if prev.size != q - 1:
        raise ValueError("previous schedule must have length q-1")
    s_old = np.linspace(0.0, 1.0, prev.size)
    s_new = np.linspace(0.0, 1.0, q)
    if q == 3 or prev.size < 3:
        return np.interp(s_new, s_old, prev)
    spline = CubicSpline(s_old, prev, bc_type="natural")
    return np.asarray(spline(s_new), dtype=np.float64)


def fixed_anneal_angles(L: int, gamma_field: float, dt: float = 0.5):
    """Physical Trotter schedule with a fixed slice length, not matched to QAOA.

    Same H(s) and the same sign of β as `anneal_angles`. dt is an absolute
    time per slice, so the total anneal time is dt * L. This baseline does
    not inherit the QAOA optimizer's branch choice.
    """
    if L <= 0:
        return [], []
    s = (np.arange(1, L + 1, dtype=np.float64) - 0.5) / float(L)
    gammas = [float(dt * si) for si in s]
    betas = [float(-dt * ((1.0 - si) + float(gamma_field) * si)) for si in s]
    return gammas, betas


def anneal_angles(L: int, gamma_field: float, duration: float):
    """Digitized linear anneal with total variation matched to `duration`.

    The interpolating Hamiltonian (s from 0 at the mixer to 1 at the target) is

        H(s) = (1-s) (-H_X) + s (H_C - Gamma H_X)
             = s H_C - [(1-s) + s Gamma] H_X.

    A first-order Trotter step exp(-i dt H(s)) ≈ exp(-i β H_X) exp(-i γ H_C)
    with γ = dt * s and β = -dt * [(1-s) + s Gamma]. The minus sign is
    required: H(s) multiplies H_X by a negative coefficient, while the QAOA
    factor is exp(-i β H_X). Slice points are s_ℓ = (ℓ - 1/2)/L.

    dt is chosen so sum |γ| + sum |β| equals `duration` (the principal-branch
    QAOA duration). The γ/β ratio and the sign of β are not refit.
    """
    if L <= 0 or not np.isfinite(duration) or duration <= 0.0:
        return [], []
    s = (np.arange(1, L + 1, dtype=np.float64) - 0.5) / float(L)
    gamma_shape = s
    beta_shape = -((1.0 - s) + float(gamma_field) * s)
    raw = float(np.sum(np.abs(gamma_shape) + np.abs(beta_shape)))
    scale = float(duration) / raw
    gammas = [float(scale * g) for g in gamma_shape]
    betas = [float(scale * b) for b in beta_shape]
    return gammas, betas


def _pack(gammas, betas):
    return np.concatenate([np.asarray(gammas, dtype=np.float64), np.asarray(betas, dtype=np.float64)])


def starts_for_depth(L: int, prev_gammas, prev_betas, rng) -> list:
    """Seeds for depth L >= 2. L = 1 is gridded separately."""
    g_prev = [float(v) for v in prev_gammas]
    b_prev = [float(v) for v in prev_betas]
    starts = []
    if L == 2 and len(g_prev) == 1:
        # Pagano Suppl. I.1.2: β <- (β*, β*-0.2), γ <- (γ*, γ*+0.2).
        # Neighboring shifts are extra deterministic seeds, not a replacement.
        g1, b1 = g_prev[0], b_prev[0]
        starts.append(("pagano_p2", _pack([g1, g1 + 0.2], [b1, b1 - 0.2])))
        for dg, db, tag in (
            (0.2, 0.2, "shift_pp"),
            (-0.2, -0.2, "shift_mm"),
            (0.1, -0.1, "shift_small"),
            (0.4, -0.4, "shift_large"),
            (0.0, 0.0, "repeat"),
        ):
            starts.append((f"p2_{tag}", _pack([g1, g1 + dg], [b1, b1 + db])))
    if len(g_prev) == L - 1 and L >= 3:
        g_i = interpolate_schedule(g_prev, L)
        b_i = interpolate_schedule(b_prev, L)
        starts.append(("pagano_interp", _pack(g_i, b_i)))
        g_f, _gk = fold_list(g_prev)
        b_f, _bk = fold_list(b_prev)
        starts.append(
            (
                "pagano_interp_principal",
                _pack(interpolate_schedule(g_f, L), interpolate_schedule(b_f, L)),
            )
        )
        for sigma, tag in ((0.08, "pert_s"), (0.2, "pert_m")):
            starts.append(
                (
                    f"interp_{tag}",
                    _pack(g_i + rng.normal(0.0, sigma, size=L), b_i + rng.normal(0.0, sigma, size=L)),
                )
            )
    if len(g_prev) == L - 1 and L >= 2:
        # Zero-layer pads reproduce the depth-(L-1) state, so energy cannot
        # rise unless every seed including these pads is dropped. They are not.
        starts.append(("zeropad_append", _pack(g_prev + [0.0], b_prev + [0.0])))
        starts.append(("zeropad_prepend", _pack([0.0] + g_prev, [0.0] + b_prev)))
    n_random = 8
    for i in range(n_random):
        starts.append((f"random_{i}", rng.uniform(0.0, PERIOD, size=2 * L)))
    return starts


def run_depth_ladder(
    n: int,
    hc_diag,
    gamma_field: float,
    gs,
    E0: float,
    Emax: float,
    edges,
    seed: int,
    max_L: int = 5,
    n_grid: int = 29,
    do_shared: bool = True,
):
    """Independent-angle ladder L = 0..max_L, plus secondary shared-γ and anneal."""
    from model import plus_state, state_observables

    psi0 = plus_state(n)
    gs_c = np.asarray(gs, dtype=np.complex128)
    rows = []
    # L = 0
    e0 = energy_of(psi0, hc_diag, n, gamma_field)
    obs0 = state_observables(psi0, n, edges, gs_c, E0, Emax, e0)
    rows.append(
        {
            "L": 0,
            "ansatz": "initial_plus",
            "gammas": [],
            "betas": [],
            "winner": "plus",
            "nfev": 0,
            "energy_nested_vs_previous": True,
            "fidelity_monotone_vs_previous": True,
            **obs0,
            **branch_report([], []),
        }
    )

    gammas_grid, betas_grid, Es, Fs = l1_grid(psi0, hc_diag, n, gamma_field, gs_c, n_grid=n_grid)
    e_coords = top_indices(Es, k=8, largest=False)
    f_coords = top_indices(Fs, k=6, largest=True)
    rng = np.random.default_rng(np.random.PCG64(int(seed) + 333 * int(round(gamma_field * 1000))))

    def coord_start(coords, prefix):
        out = []
        for rank, (ib, ig) in enumerate(coords):
            out.append((f"{prefix}{rank}", _pack([gammas_grid[ig]], [betas_grid[ib]])))
        return out

    energy_starts = coord_start(e_coords, "gridE_")
    for i in range(4):
        energy_starts.append((f"random_{i}", rng.uniform(0.0, PERIOD, size=2)))
    # Explicit small-angle and reflected seeds.
    energy_starts.append(("small", _pack([0.15], [0.35])))
    energy_starts.append(("reflected_seed", _pack([0.8], [-0.4])))

    opt = optimize_angles(
        psi0, hc_diag, n, gamma_field, energy_starts, maxiter=70, objective="energy", gs=gs_c
    )
    # Independent fidelity maximization. This is not the reported optimizer;
    # it is an artifact check against the energy minimizer.
    fid_starts = coord_start(f_coords, "gridF_")
    for i in range(3):
        fid_starts.append((f"frand_{i}", rng.uniform(0.0, PERIOD, size=2)))
    optF = optimize_angles(
        psi0, hc_diag, n, gamma_field, fid_starts, maxiter=50, objective="fidelity", gs=gs_c
    )
    psiF = apply_qaoa(psi0, hc_diag, n, optF["gammas"], optF["betas"])
    EF = energy_of(psiF, hc_diag, n, gamma_field)
    FF = float(abs(np.vdot(gs_c, psiF)) ** 2)

    prev_g = list(opt["gammas"])
    prev_b = list(opt["betas"])
    prev_E = float(e0)
    prev_F = float(obs0["F"])
    for L in range(1, max_L + 1):
        if L == 1:
            this = opt
        else:
            starts = starts_for_depth(L, prev_g, prev_b, rng)
            this = optimize_angles(
                psi0,
                hc_diag,
                n,
                gamma_field,
                starts,
                maxiter=50 if L < 4 else 40,
                objective="energy",
                gs=gs_c,
            )
        psi = apply_qaoa(psi0, hc_diag, n, this["gammas"], this["betas"])
        # Replay from principal angles must reproduce the state.
        rep = branch_report(this["gammas"], this["betas"])
        psi_p = apply_qaoa(psi0, hc_diag, n, rep["gamma_principal"], rep["beta_principal"])
        principal_state_F = float(abs(np.vdot(psi, psi_p)) ** 2)
        E = energy_of(psi, hc_diag, n, gamma_field)
        obs = state_observables(psi, n, edges, gs_c, E0, Emax, E)
        nested = True if prev_E is None else bool(E <= prev_E + 1e-7)
        mono = True if prev_F is None else bool(obs["F"] + 1e-7 >= prev_F)
        row = {
            "L": L,
            "ansatz": "independent",
            "gammas": this["gammas"],
            "betas": this["betas"],
            "winner": this["winner"],
            "nfev": this["nfev"],
            "energy_nested_vs_previous": nested,
            "fidelity_monotone_vs_previous": mono,
            "principal_state_overlap": principal_state_F,
            **obs,
            **rep,
        }
        if L == 1:
            grid_min_E = float(Es.min())
            grid_max_F = float(Fs.max())
            row.update(
                {
                    "l1_grid_min_energy": grid_min_E,
                    "l1_grid_max_fidelity": grid_max_F,
                    "l1_energy_opt_beats_grid": bool(E <= grid_min_E + 1e-6),
                    "l1_fidelity_max_F": FF,
                    "l1_fidelity_max_energy": float(EF),
                    "l1_fidelity_max_gammas": optF["gammas"],
                    "l1_fidelity_max_betas": optF["betas"],
                    "l1_fidelity_max_winner": optF["winner"],
                    "l1_energy_gap_vs_fidelity_opt": float(E - EF),
                    "l1_fidelity_gap_vs_fidelity_opt": float(FF - obs["F"]),
                }
            )
        # Shared-gamma diagnostic at every depth. One shared γ, independent β_ℓ.
        if do_shared:
            shared = _shared_gamma(
                psi0, hc_diag, n, gamma_field, gs_c, this["gammas"], this["betas"], rng, E0, Emax, edges
            )
            row["shared_gamma"] = shared
        # Duration-matched digitized anneal.
        g_a, b_a = anneal_angles(L, gamma_field, rep["duration_principal"])
        if g_a:
            psi_a = apply_qaoa(psi0, hc_diag, n, g_a, b_a)
            Ea = energy_of(psi_a, hc_diag, n, gamma_field)
            obs_a = state_observables(psi_a, n, edges, gs_c, E0, Emax, Ea)
            row["anneal_matched"] = {
                "gammas": g_a,
                "betas": b_a,
                "duration": float(np.sum(np.abs(g_a)) + np.sum(np.abs(b_a))),
                **obs_a,
            }
        g_f, b_f = fixed_anneal_angles(L, gamma_field, dt=0.5)
        if g_f:
            psi_f = apply_qaoa(psi0, hc_diag, n, g_f, b_f)
            Ef = energy_of(psi_f, hc_diag, n, gamma_field)
            obs_f = state_observables(psi_f, n, edges, gs_c, E0, Emax, Ef)
            row["anneal_fixed_dt"] = {
                "dt": 0.5,
                "gammas": g_f,
                "betas": b_f,
                "duration": float(np.sum(np.abs(g_f)) + np.sum(np.abs(b_f))),
                **obs_f,
            }
        rows.append(row)
        prev_g, prev_b = this["gammas"], this["betas"]
        prev_E, prev_F = E, obs["F"]
    return rows


def _shared_gamma(psi0, hc_diag, n, gamma_field, gs, gammas, betas, rng, E0, Emax, edges):
    from model import state_observables

    L = len(gammas)
    g_mean = float(np.mean(gammas))
    starts = []
    starts.append(("from_independent_mean", np.concatenate([[g_mean], np.asarray(betas, dtype=np.float64)])))
    g_p, _ = fold_list(gammas)
    starts.append(
        ("from_principal_mean", np.concatenate([[float(np.mean(g_p))], np.asarray(fold_list(betas)[0], dtype=np.float64)]))
    )
    for i in range(3):
        starts.append((f"random_{i}", rng.uniform(0.0, PERIOD, size=1 + L)))

    def obj(x):
        g = float(x[0])
        b = x[1:]
        psi = apply_qaoa(psi0, hc_diag, n, [g] * L, b)
        return energy_of(psi, hc_diag, n, gamma_field)

    best_x, best_val, winner, nfev = None, np.inf, None, 0
    for name, x0 in starts:
        seed_val = float(obj(x0))
        nfev += 1
        x_opt, val, n_loc, _ok = _lbfgs(obj, x0, maxiter=40)
        nfev += n_loc
        if seed_val < val:
            x_opt, val = np.asarray(x0, dtype=np.float64), seed_val
            name = name + ":seed"
        if val < best_val:
            best_x, best_val, winner = x_opt, val, name
    g = float(best_x[0])
    b = [float(v) for v in best_x[1:]]
    psi = apply_qaoa(psi0, hc_diag, n, [g] * L, b)
    E = energy_of(psi, hc_diag, n, gamma_field)
    obs = state_observables(psi, n, edges, gs, E0, Emax, E)
    return {
        "gamma": g,
        "betas": b,
        "winner": winner,
        "nfev": int(nfev),
        **obs,
        **{f"shared_{k}": v for k, v in branch_report([g] * L, b).items()},
    }
