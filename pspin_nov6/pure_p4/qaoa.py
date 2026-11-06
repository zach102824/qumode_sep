"""QAOA for Family C inside the symmetric irrep, with Pagano bootstrap.

Circuit (primary), layer k = 1..L in time order, k=1 acts first:
    |ψ> = ∏_k exp(-i β_k sum_i X_i) exp(-i γ_k H_C) |+>^{⊗N}
Independent (γ_k, β_k). Objective is the energy of
    H = H_C - Gamma sum_i X_i
not the fidelity. Fidelity is a separate L=1 diagnostic.

Pagano et al. arXiv:1906.02700 Suppl. I.1.2, transcribed as:
  L=1: grid + local optimization (their p=1 grid / greedy local search).
  L=2: seed β=(β*, β*-0.2), γ=(γ*, γ*+0.2), then local optimization.
  L>=3: interpolate the previous raw angles against s_i=(i-1)/(L_prev-1)
        (linear when the previous depth has 2 points; natural cubic spline
        when it has >=3), resample at s=j/(L-1), then local optimization.
        A cubic through exactly 3 points is a natural cubic spline; Pagano's
        note says "cubic" for q>3 without specifying boundary conditions.

Extra seeds, tagged so they are not confused with the paper recipe:
  zero angles, zero-pad of the previous depth, small perturbations,
  random multistarts, a last-layer ±0.2 offset.
The unevolved zero-pad state is always kept as a feasible point, so the
recorded energy is a non-increasing function of L. Fidelity of that
energy-selected state is NOT forced to be monotone.

Shared-γ (one γ, independent β_k) is a diagnostic sidecar only.

Digitized anneal baseline (not variational over 2L angles):
  H(s) = (1-s)(-sum X) + s (H_C - Gamma sum X), s: 0 -> 1,
  first-order Trotter, L steps, midpoint s_k=(k-0.5)/L, total time T,
  same gate order as QAOA. T is chosen on a fixed grid by energy, and
  once by matching sum(|β|+|γ|) to the QAOA energy optimum.
"""
from __future__ import annotations

import math
from typing import Callable

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize

from model import (
    DickeModel,
    canonical_angles,
    eta_score,
    l1_duration,
    subspace_fidelity,
)


def _as1d(x) -> np.ndarray:
    return np.asarray(x, dtype=float).reshape(-1)


def split_angles(x, L: int) -> tuple[np.ndarray, np.ndarray]:
    v = _as1d(x)
    if v.size != 2 * L:
        raise ValueError(f"expected {2*L} params, got {v.size}")
    return v[:L].copy(), v[L:].copy()


def pack_angles(betas, gammas) -> np.ndarray:
    return np.concatenate([_as1d(betas), _as1d(gammas)])


def resample_schedule(values, new_len: int) -> np.ndarray:
    """Pagano normalized-time resample. See module docstring."""
    old = _as1d(values)
    if new_len < 1:
        raise ValueError(new_len)
    if old.size == 1:
        return np.full(new_len, float(old[0]))
    s_old = np.linspace(0.0, 1.0, old.size)
    s_new = np.linspace(0.0, 1.0, new_len)
    if old.size < 3:
        return np.interp(s_new, s_old, old)
    spline = CubicSpline(s_old, old, bc_type="natural")
    return np.asarray(spline(s_new), dtype=float)


def energy_of(model: DickeModel, betas, gammas) -> tuple[float, np.ndarray]:
    c = model.evolve(betas, gammas)
    return model.energy(c), c


def fidelity_of(c: np.ndarray, targets: np.ndarray) -> float:
    return subspace_fidelity(c, targets)


class EvalCounter:
    def __init__(self):
        self.n = 0

    def bump(self, k: int = 1):
        self.n += k


def _local_minimize(
    fun: Callable,
    x0: np.ndarray,
    bounds,
    maxiter: int,
    counter: EvalCounter,
    polish: bool = False,
):
    x0 = np.array(x0, dtype=float)
    lo = hi = None
    if bounds is not None:
        lo = np.array([b[0] for b in bounds], dtype=float)
        hi = np.array([b[1] for b in bounds], dtype=float)
        x0 = np.clip(x0, lo, hi)

    def wrapped(x):
        counter.bump()
        return float(fun(x))

    res = minimize(
        wrapped,
        x0,
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": int(maxiter), "ftol": 1e-14, "gtol": 1e-8},
    )
    x_best = np.array(res.x, dtype=float)
    f_best = float(res.fun)
    if polish:
        def wrapped_nm(x):
            counter.bump()
            if lo is not None and (np.any(x < lo - 1e-8) or np.any(x > hi + 1e-8)):
                pen = np.maximum(0.0, lo - x) ** 2 + np.maximum(0.0, x - hi) ** 2
                return 1e6 + float(np.sum(pen))
            return float(fun(x))

        res2 = minimize(
            wrapped_nm,
            x_best,
            method="Nelder-Mead",
            options={
                "maxiter": int(max(50, maxiter // 2)),
                "xatol": 1e-6,
                "fatol": 1e-12,
                "adaptive": True,
            },
        )
        if float(res2.fun) < f_best:
            return np.array(res2.x, dtype=float), float(res2.fun)
    return x_best, f_best


def bounds_for(L: int, T_gamma: float, kind: str) -> list[tuple[float, float]]:
    """β in [-π, π] (two projective periods) and γ in [-T, T].

    Each β shifts by π and each γ shifts by T as a global phase only, so a
    single period would already be complete. The doubled window keeps a
    minimum off the boundary. Do not cap T: the L=1 grid winner can sit
    anywhere in [0, T), and clipping it would move the bootstrap onto a
    different branch.
    """
    del kind
    if not (T_gamma == T_gamma) or T_gamma <= 0:
        Tg = 4.0 * math.pi
    else:
        Tg = float(T_gamma)
    return [(-math.pi, math.pi)] * L + [(-Tg, Tg)] * L


def grid_l1_energies(model: DickeModel, betas: np.ndarray, gammas: np.ndarray) -> np.ndarray:
    """Vectorized L=1 energy landscape. Returns shape (n_beta, n_gamma)."""
    hc = model.hc
    ev = model.X_evals
    V = model.X_vecs
    c0 = model.c0
    phased = c0[None, :] * np.exp(-1j * gammas[:, None] * hc[None, :])  # (G,D)
    coeff0 = phased @ V  # (G,D)
    G = gammas.shape[0]
    B = betas.shape[0]
    Eb = np.empty((B, G), dtype=np.float64)
    chunk = 32
    for i0 in range(0, B, chunk):
        bb = betas[i0:i0 + chunk]
        # coeff (nb, G, D)
        phase = np.exp(-1j * bb[:, None] * ev[None, :])  # (nb, D)
        coeff = coeff0[None, :, :] * phase[:, None, :]
        psi = np.einsum("bgd,dk->bgk", coeff, V.T)
        hc_e = np.sum(np.abs(psi) ** 2 * hc[None, None, :], axis=2)
        x_e = np.sum(np.abs(coeff) ** 2 * ev[None, None, :], axis=2)
        Eb[i0:i0 + bb.shape[0]] = hc_e - model.gamma * x_e
    return Eb


def _top_grid_minima(Eb: np.ndarray, betas: np.ndarray, gammas: np.ndarray, k: int, sep_b: float, sep_g: float):
    """k lowest grid points separated in angle space. Order is best-first."""
    flat = Eb.ravel()
    order = np.argsort(flat)
    chosen = []
    for t in order:
        ib, ig = divmod(int(t), Eb.shape[1])
        b = float(betas[ib])
        g = float(gammas[ig])
        if any(abs(b - bb) < sep_b and abs(g - gg) < sep_g for bb, gg, _ in chosen):
            # also treat β ~ β+π and γ ~ γ+T as the same basin only via the
            # grid already living in one period. Separation is Euclidean in
            # the fundamental cell, not a smoothing step.
            continue
        chosen.append((b, g, float(flat[t])))
        if len(chosen) >= k:
            break
    return chosen


def optimize_independent(
    model: DickeModel,
    L: int,
    targets_global: np.ndarray,
    targets_reach: np.ndarray,
    prev: dict | None,
    rng: np.random.Generator,
    counter: EvalCounter,
    grid_betas: int = 36,
    grid_gammas: int | None = None,
) -> dict:
    """Energy-optimal independent-angle QAOA at depth L.

    targets_* are Dicke-space orthonormal columns.
    """
    T = float(model.T_gamma)
    bounds = bounds_for(L, T, "energy")

    def objective(x):
        b, g = split_angles(x, L)
        E, _ = energy_of(model, b, g)
        return E

    seeds: list[tuple[str, np.ndarray]] = []
    landscape = None
    grid_info = None

    if L == 1:
        n_g = grid_gammas
        if n_g is None:
            # Spacing 0.06 across one full analytical γ period. The cap is only
            # a runtime guard; pure-family periods at N<=12 stay under it.
            n_g = int(max(64, min(2400, math.ceil(T / 0.06))))
        betas = np.linspace(0.0, math.pi, grid_betas, endpoint=False)
        gammas = np.linspace(0.0, T, n_g, endpoint=False)
        counter.bump(betas.size * gammas.size)
        Eb = grid_l1_energies(model, betas, gammas)
        landscape = Eb
        minima = _top_grid_minima(Eb, betas, gammas, k=8, sep_b=0.15, sep_g=max(0.15, T / 40.0))
        grid_info = {
            "n_beta": int(betas.size),
            "n_gamma": int(gammas.size),
            "beta_max": float(betas[-1]) if betas.size else None,
            "gamma_max_exclusive": float(T),
            "grid_best_energy": float(Eb.min()),
            "separated_minima": [
                {"beta": b, "gamma": g, "E": e} for b, g, e in minima
            ],
        }
        for i, (b, g, _) in enumerate(minima):
            seeds.append((f"grid_{i}", pack_angles([b], [g])))
        seeds.append(("zero", pack_angles([0.0], [0.0])))
        seeds.append(("small", pack_angles([0.2], [0.2])))
        for i in range(4):
            seeds.append((
                "random",
                pack_angles([rng.uniform(0.0, math.pi)], [rng.uniform(0.0, T)]),
            ))
    else:
        assert prev is not None
        b_prev = np.asarray(prev["betas_raw"], dtype=float)
        g_prev = np.asarray(prev["gammas_raw"], dtype=float)
        # exact paper L=2 offset, and the interpolation recipe for L>=3.
        # Both are always available as seeds; interpolation at L=2 (two copies
        # of the L=1 angles) is NOT the paper's L=2 seed, so it is tagged apart.
        if L == 2:
            seeds.append((
                "pagano_l2_offset",
                pack_angles([b_prev[0], b_prev[0] - 0.2], [g_prev[0], g_prev[0] + 0.2]),
            ))
            seeds.append((
                "interp_repeat_not_paper",
                pack_angles(resample_schedule(b_prev, 2), resample_schedule(g_prev, 2)),
            ))
        else:
            seeds.append((
                "pagano_interp",
                pack_angles(resample_schedule(b_prev, L), resample_schedule(g_prev, L)),
            ))
        # perturbations of the primary bootstrap seed
        primary = seeds[0][1]
        for sigma, npert in ((0.05, 2), (0.15, 2)):
            for _ in range(npert):
                noise = rng.normal(0.0, sigma, size=primary.size)
                seeds.append((f"perturb_s{sigma}", primary + noise))
        # last-layer offset in the Pagano direction, extra multistart
        b_off, g_off = split_angles(primary, L)
        b_off = b_off.copy()
        g_off = g_off.copy()
        b_off[-1] = b_off[-1] - 0.2
        g_off[-1] = g_off[-1] + 0.2
        seeds.append(("lastlayer_offset", pack_angles(b_off, g_off)))
        # zero-pad warm start (reproduces depth L-1 exactly before local opt)
        seeds.append((
            "zero_pad",
            pack_angles(np.concatenate([b_prev, [0.0]]), np.concatenate([g_prev, [0.0]])),
        ))
        # global conjugation of the primary seed
        seeds.append(("signflip_seed", -primary))
        for i in range(3):
            br = rng.uniform(-math.pi, math.pi, size=L)
            # concentrate γ near the scale already selected, plus one wide draw
            if i == 0:
                gr = rng.uniform(0.0, min(T, 2.0 * math.pi), size=L)
                tag = "random_wide"
            else:
                scale = max(0.3, float(np.mean(np.abs(g_prev))) + 0.2)
                gr = rng.normal(float(np.mean(g_prev)), scale, size=L)
                tag = "random_near"
            seeds.append((tag, pack_angles(br, gr)))

    # Feasible carry of the previous energy, even if local opt walks away.
    carry_E = None
    carry_x = None
    if prev is not None:
        b_prev = np.asarray(prev["betas_raw"], dtype=float)
        g_prev = np.asarray(prev["gammas_raw"], dtype=float)
        carry_x = pack_angles(np.concatenate([b_prev, [0.0]]), np.concatenate([g_prev, [0.0]]))
        # evaluate exactly, no bound clip
        counter.bump()
        carry_E = float(objective(carry_x))

    best = None
    tried = []
    ranked: list[tuple[float, np.ndarray, str]] = []
    maxiter = 50 if L == 1 else 40 + 8 * L
    for tag, x0 in seeds:
        # also score the raw seed (bound-clipped) so a failed local opt cannot
        # erase a good start
        x_seed = np.array(x0, dtype=float)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        x_seed_clip = np.clip(x_seed, lo, hi)
        counter.bump()
        e_seed = float(objective(x_seed_clip))
        x_opt, e_opt = _local_minimize(
            objective, x_seed_clip, bounds, maxiter=maxiter, counter=counter, polish=False,
        )
        if e_seed < e_opt:
            x_opt, e_opt = x_seed_clip, e_seed
            refined = False
        else:
            refined = True
        tried.append({"tag": tag, "E": e_opt, "refined": refined})
        ranked.append((e_opt, x_opt, tag))
        if best is None or e_opt < best["E"] - 1e-12:
            best = {"E": e_opt, "x": x_opt, "tag": tag}
    # Nelder-Mead polish only on the best two basins. This is local refinement,
    # not a smoother: if it walks onto another symmetry branch we keep the raw
    # angles of whichever point actually has the lower energy.
    ranked.sort(key=lambda t: t[0])
    for e_opt, x_opt, tag in ranked[:2]:
        x_p, e_p = _local_minimize(
            objective, x_opt, bounds, maxiter=maxiter, counter=counter, polish=True,
        )
        tried.append({"tag": tag + "_polish", "E": e_p, "refined": True})
        if e_p < best["E"] - 1e-12:
            best = {"E": e_p, "x": x_p, "tag": tag + "_polish"}

    if carry_E is not None and (best is None or carry_E < best["E"] - 1e-12):
        best = {"E": float(carry_E), "x": carry_x, "tag": "carry_previous"}
        tried.append({"tag": "carry_previous", "E": float(carry_E), "refined": False})

    assert best is not None
    betas, gammas = split_angles(best["x"], L)
    E, c = energy_of(model, betas, gammas)
    # numerical guard: recompute
    E = float(E)
    F_global = fidelity_of(c, targets_global)
    F_reach = fidelity_of(c, targets_reach)
    b_can, g_can = canonical_angles(betas, gammas, T)
    return {
        "L": L,
        "E": E,
        "F_global": F_global,
        "F_reach": F_reach,
        "betas_raw": betas.tolist(),
        "gammas_raw": gammas.tolist(),
        "betas_canonical": b_can.tolist(),
        "gammas_canonical": g_can.tolist(),
        "winner_tag": best["tag"],
        "angle_l1_duration": l1_duration(betas, gammas),
        "tried": tried,
        "grid": grid_info,
        "state": c,
        "evals": counter.n,
    }


def optimize_fidelity_l1(
    model: DickeModel,
    targets: np.ndarray,
    rng: np.random.Generator,
    counter: EvalCounter,
    label: str,
) -> dict:
    """Separate L=1 maximization of subspace fidelity. Not the energy optimum."""
    T = float(model.T_gamma)
    L = 1
    bounds = bounds_for(L, T, "fidelity")
    betas = np.linspace(0.0, math.pi, 28, endpoint=False)
    n_g = int(max(36, min(240, math.ceil(T / 0.12))))
    gammas = np.linspace(0.0, T, n_g, endpoint=False)

    def F_at(b, g):
        c = model.evolve([b], [g])
        return fidelity_of(c, targets), c

    # coarse grid of fidelity
    best_F = -1.0
    best_bg = (0.2, 0.2)
    # vectorize states similarly to energy grid but score fidelity
    hc = model.hc
    ev = model.X_evals
    V = model.X_vecs
    phased = model.c0[None, :] * np.exp(-1j * gammas[:, None] * hc[None, :])
    coeff0 = phased @ V
    # targets in X basis not needed; compute psi and overlap in chunks of beta
    tgt = np.asarray(targets, dtype=np.complex128)
    seeds = []
    flat_best = []
    for ib, b in enumerate(betas):
        coeff = coeff0 * np.exp(-1j * b * ev)[None, :]
        psi = coeff @ V.T  # (G,D)
        amps = psi @ tgt.conj()  # (G, ntarget) if tgt is (D, r) wait
        # tgt columns (D, r), psi (G, D): amps = psi @ tgt.conj() is wrong if we need tgt.conj().T
        amps = psi @ tgt.conj()
        # psi (G,D) @ tgt.conj() (D, r) = (G, r) only if tgt is (D, r) and we conj the matrix elements.
        # subspace fid = sum_r |<t_r|ψ>|^2 = sum_r |t_r† ψ|^2 = || tgt.conj().T @ psi || but psi is row.
        # <t|ψ> = sum_d conj(t_d) ψ_d if t stored as column? vdot(t, psi) = conj(t)·psi.
        # amps[g, r] = sum_d psi[g,d] * conj(tgt[d,r]) = <tgt_r | psi>
        F = np.sum(np.abs(amps) ** 2, axis=1)
        counter.bump(gammas.size)
        j = int(np.argmax(F))
        flat_best.append((float(F[j]), float(b), float(gammas[j])))
        if F[j] > best_F:
            best_F = float(F[j])
            best_bg = (float(b), float(gammas[j]))
    flat_best.sort(reverse=True)
    # separated seeds
    chosen = []
    for F, b, g in flat_best:
        if any(abs(b - bb) < 0.2 and abs(g - gg) < max(0.2, T / 30) for bb, gg in chosen):
            continue
        chosen.append((b, g))
        seeds.append((b, g))
        if len(seeds) >= 6:
            break
    seeds.append((0.0, 0.0))
    for _ in range(3):
        seeds.append((float(rng.uniform(0, math.pi)), float(rng.uniform(0, T))))

    def objective(x):
        b, g = split_angles(x, 1)
        c = model.evolve(b, g)
        return -fidelity_of(c, targets)

    best = {"F": best_F, "x": pack_angles([best_bg[0]], [best_bg[1]]), "tag": "grid"}
    for i, (b, g) in enumerate(seeds):
        x0 = pack_angles([b], [g])
        x_opt, negF = _local_minimize(objective, x0, bounds, maxiter=50, counter=counter)
        F = -negF
        if F > best["F"] + 1e-12:
            best = {"F": float(F), "x": x_opt, "tag": f"local_{i}"}
    betas_o, gammas_o = split_angles(best["x"], 1)
    E, c = energy_of(model, betas_o, gammas_o)
    F = fidelity_of(c, targets)
    return {
        "L": 1,
        "target": label,
        "F_target": float(F),
        "E": float(E),
        "betas_raw": betas_o.tolist(),
        "gammas_raw": gammas_o.tolist(),
        "winner_tag": best["tag"],
        "angle_l1_duration": l1_duration(betas_o, gammas_o),
        "state": c,
    }


def optimize_shared_gamma(
    model: DickeModel,
    L: int,
    targets_global: np.ndarray,
    targets_reach: np.ndarray,
    independent: dict,
    rng: np.random.Generator,
    counter: EvalCounter,
) -> dict:
    """Diagnostic: one shared γ and independent β_k. Not the primary ansatz."""
    T = float(model.T_gamma)
    g_span = T if T == T and T > 0 else 4.0 * math.pi
    bounds = [(-g_span, g_span)] + [(-math.pi, math.pi)] * L

    def unpack(x):
        gamma = float(x[0])
        betas = np.asarray(x[1:], dtype=float)
        gammas = np.full(L, gamma)
        return betas, gammas

    def objective(x):
        b, g = unpack(x)
        return energy_of(model, b, g)[0]

    b_ind = np.asarray(independent["betas_raw"], dtype=float)
    g_ind = np.asarray(independent["gammas_raw"], dtype=float)
    seeds = [
        np.concatenate([[float(np.mean(g_ind))], b_ind]),
        np.concatenate([[float(g_ind[0])], b_ind]),
        np.concatenate([[float(g_ind[-1])], np.full(L, float(np.mean(b_ind)))]),
        np.concatenate([[0.0], np.zeros(L)]),
    ]
    for _ in range(3):
        seeds.append(np.concatenate([
            [float(rng.uniform(0.0, min(T, 2.0 * math.pi)))],
            rng.uniform(-math.pi, math.pi, size=L),
        ]))
    best = None
    for i, x0 in enumerate(seeds):
        counter.bump()
        e0 = float(objective(np.clip(x0, [b[0] for b in bounds], [b[1] for b in bounds])))
        x_opt, e_opt = _local_minimize(objective, x0, bounds, maxiter=70, counter=counter)
        if e0 < e_opt:
            x_opt = np.clip(x0, [b[0] for b in bounds], [b[1] for b in bounds])
            e_opt = e0
        if best is None or e_opt < best["E"] - 1e-12:
            best = {"E": float(e_opt), "x": x_opt, "tag": f"s{i}"}
    betas, gammas = unpack(best["x"])
    E, c = energy_of(model, betas, gammas)
    return {
        "L": L,
        "E": float(E),
        "F_global": fidelity_of(c, targets_global),
        "F_reach": fidelity_of(c, targets_reach),
        "gamma_shared": float(best["x"][0]),
        "betas_raw": betas.tolist(),
        "gammas_raw": gammas.tolist(),
        "winner_tag": best["tag"],
        "angle_l1_duration": l1_duration(betas, gammas),
        "state": c,
    }


def anneal_angles(L: int, T: float, gamma_field: float) -> tuple[np.ndarray, np.ndarray]:
    """Midpoint linear schedule. Returns (betas, gammas) in the QAOA convention.

    H(s) = (1-s)(-sum X) + s (H_C - Gamma sum X)
         = s H_C + [-(1-s) - s Gamma] sum X
    γ_k = Δt * s_k multiplies H_C.
    β_k = Δt * [-(1-s_k) - Gamma s_k] multiplies sum X.
    Δt = T/L, s_k = (k-0.5)/L, k=1..L so s increases with layer index.
    """
    dt = float(T) / float(L)
    betas = np.zeros(L)
    gammas = np.zeros(L)
    for i, k in enumerate(range(1, L + 1)):
        s = (k - 0.5) / float(L)
        gammas[i] = dt * s
        betas[i] = dt * (-(1.0 - s) - float(gamma_field) * s)
    return betas, gammas


def matched_anneal_time(qaoa_duration: float, gamma_field: float) -> float:
    """T such that the anneal's sum(|β|+|γ|) equals qaoa_duration.

    For the midpoint schedule, sum(|β|+|γ|) = T * (1 + Gamma/2).
    Derivation: sum γ = T/2, sum |β| = T*(1+Gamma)/2.
    """
    denom = 1.0 + 0.5 * float(gamma_field)
    if denom <= 1e-12:
        return float(qaoa_duration)
    return float(qaoa_duration) / denom


ANNEAL_T_GRID = (0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 12.0)


def evaluate_anneal(
    model: DickeModel,
    L: int,
    targets_global: np.ndarray,
    targets_reach: np.ndarray,
    qaoa_duration: float | None,
) -> dict:
    """Fixed linear schedules. Best-of-grid by energy, plus one duration-matched T."""
    rows = []
    best = None
    for T in ANNEAL_T_GRID:
        b, g = anneal_angles(L, T, model.gamma)
        E, c = energy_of(model, b, g)
        rec = {
            "T": float(T),
            "E": float(E),
            "F_global": fidelity_of(c, targets_global),
            "F_reach": fidelity_of(c, targets_reach),
            "betas_raw": b.tolist(),
            "gammas_raw": g.tolist(),
            "angle_l1_duration": l1_duration(b, g),
            "matched": False,
        }
        rows.append(rec)
        if best is None or rec["E"] < best["E"] - 1e-12:
            best = rec
    matched = None
    if qaoa_duration is not None:
        Tm = matched_anneal_time(qaoa_duration, model.gamma)
        b, g = anneal_angles(L, Tm, model.gamma)
        E, c = energy_of(model, b, g)
        matched = {
            "T": float(Tm),
            "E": float(E),
            "F_global": fidelity_of(c, targets_global),
            "F_reach": fidelity_of(c, targets_reach),
            "betas_raw": b.tolist(),
            "gammas_raw": g.tolist(),
            "angle_l1_duration": l1_duration(b, g),
            "matched": True,
            "qaoa_duration_target": float(qaoa_duration),
        }
    return {"best_grid": best, "matched": matched, "grid": rows}


def orthonormalize(cols: list[np.ndarray], tol: float = 1e-8) -> np.ndarray:
    if not cols:
        return np.zeros((0, 0), dtype=np.complex128)
    A = np.column_stack([np.asarray(c, dtype=np.complex128) for c in cols])
    # QR
    Q, R = np.linalg.qr(A, mode="reduced")
    keep = np.abs(np.diag(R)) > tol
    if not np.any(keep):
        return np.zeros((A.shape[0], 0), dtype=np.complex128)
    Q = Q[:, keep]
    # drop numerical zero columns already handled
    return Q


def global_targets_in_dicke(full_vecs: np.ndarray, n: int, sym_tol: float = 1e-6) -> tuple[np.ndarray, dict]:
    """Project full-space GS eigenvectors into the Dicke basis and orthonormalize.

    Components orthogonal to the symmetric irrep are discarded by the
    projection. The residual 1 - |P_sym v|^2 is reported per vector so a
    GS that lives outside the ansatz sector is visible (the projected
    columns then do not span that GS).
    """
    cols = []
    weights = []
    for i in range(full_vecs.shape[1]):
        psi = full_vecs[:, i]
        # symmetric weight
        from model import full_to_dicke, symmetric_weight
        w = symmetric_weight(psi, n)
        weights.append(w)
        c = full_to_dicke(psi, n)
        if np.linalg.norm(c) > 1e-8:
            cols.append(c / np.linalg.norm(c) * min(1.0, math.sqrt(max(w, 0.0))))
            # Using the raw projection (not renormalized to 1) would make
            # subspace fidelity equal the true full-space subspace fidelity
            # for states already inside the Dicke sector, and strictly smaller
            # if we renormalized a mostly-orthogonal vector into a fake target.
            cols[-1] = c  # raw projection; orthonormalize() will QR
    Q = orthonormalize(cols)
    info = {
        "sym_weights": weights,
        "n_projected": int(Q.shape[1]) if Q.ndim == 2 else 0,
        "max_sym_weight": float(max(weights) if weights else 0.0),
        "min_sym_weight": float(min(weights) if weights else 0.0),
    }
    return Q, info
