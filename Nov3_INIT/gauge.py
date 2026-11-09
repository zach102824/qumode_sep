"""Exact SK local spin-flip gauges. Classical. Not an ansatz retune."""
from __future__ import annotations

import numpy as np

from Nov3_INIT.bstar import _ground, apply_x_sum, entropy_nats
from Nov3_INIT.circuit import Logical
from Nov9_SK_SPSA.experiments.FT.instance import make_J, pair_diag, z_table

GAUGE_RNG_ID = 4401
N_GAUGES = 4
IDENTITY_GID = 0
BAND = list(range(10552, 10560))
N = 15


def gauge_rng(seed: int, n: int, gid: int) -> np.random.Generator:
    return np.random.default_rng(
        np.random.SeedSequence([int(GAUGE_RNG_ID), int(seed), int(n), int(gid)])
    )


def gauge_vector(seed: int, n: int, gid: int) -> np.ndarray:
    gid = int(gid)
    if gid == IDENTITY_GID:
        return np.ones(int(n), dtype=np.int8)
    if gid < 1 or gid >= N_GAUGES:
        raise ValueError(gid)
    rng = gauge_rng(seed, n, gid)
    return rng.choice(np.array([-1, 1], dtype=np.int8), size=int(n))


def gauged_J(J: np.ndarray, g: np.ndarray) -> np.ndarray:
    g = np.asarray(g, dtype=np.float64)
    J = np.asarray(J, dtype=np.float64)
    return (g[:, None] * J) * g[None, :]


def flip_mask(g: np.ndarray) -> int:
    mask = 0
    for i, gi in enumerate(np.asarray(g).ravel()):
        if int(gi) < 0:
            mask |= 1 << int(i)
    return int(mask)


def apply_X_product(vec: np.ndarray, mask: int) -> np.ndarray:
    """U = ∏_{g_i=-1} X_i. Fixed bit i is (idx >> i) & 1. (Uψ)[b] = ψ[b ⊕ mask]."""
    vec = np.asarray(vec)
    idx = np.arange(vec.size)
    return vec[idx ^ int(mask)]


def map_gs(gs: np.ndarray, g: np.ndarray) -> np.ndarray:
    return apply_X_product(np.asarray(gs, dtype=np.float64), flip_mask(g))


def logical_gauged(n: int, seed: int, b: float, gs: np.ndarray, gid: int) -> Logical:
    J0 = make_J(int(seed), int(n))
    g = gauge_vector(seed, n, gid)
    gs_g = map_gs(gs, g)
    return Logical(int(n), int(seed), float(b), gs_g, J=gauged_J(J0, g))


def apply_H(hc: np.ndarray, b: float, n: int, v: np.ndarray) -> np.ndarray:
    vv = np.asarray(v, dtype=np.float64)
    return hc * vv - float(b) * apply_x_sum(vv, int(n))


def residual(hc: np.ndarray, b: float, n: int, gs: np.ndarray) -> tuple[float, float]:
    gs = np.asarray(gs, dtype=np.float64)
    hv = apply_H(hc, b, n, gs)
    e = float(np.vdot(gs, hv).real)
    return e, float(np.linalg.norm(hv - e * gs))


def frozen_table(n: int = N, seeds=None) -> dict:
    seeds = list(BAND if seeds is None else seeds)
    rows = []
    for seed in seeds:
        for gid in range(N_GAUGES):
            g = gauge_vector(seed, n, gid)
            rows.append({
                "seed": int(seed),
                "gid": int(gid),
                "g": [int(v) for v in g],
                "n_flips": int(np.sum(np.asarray(g) < 0)),
                "mask": flip_mask(g),
                "is_identity": bool(gid == IDENTITY_GID),
            })
    return {
        "GAUGE_RNG_ID": GAUGE_RNG_ID,
        "n": int(n),
        "seeds": [int(s) for s in seeds],
        "n_gauges": N_GAUGES,
        "identity_gid": IDENTITY_GID,
        "recipe": "gid 0 is all +1; gid 1..3 are SeedSequence([4401, seed, N, gid]) Rademacher ±1",
        "not_selected_on_energy": True,
        "rows": rows,
    }


def uniqueness(table: dict) -> dict:
    by_seed = {}
    for row in table["rows"]:
        by_seed.setdefault(int(row["seed"]), []).append(tuple(row["g"]))
    collisions = []
    for seed, gs in by_seed.items():
        signed = [g if g[0] == 1 else tuple(-x for x in g) for g in gs]
        if len(set(signed)) != len(gs):
            collisions.append(int(seed))
    return {"ok": len(collisions) == 0, "global_sign_collisions": collisions}


def algebra_probe(n: int = 6, seed: int = 10018, b: float = 0.4) -> dict:
    J0 = make_J(seed, n)
    z = z_table(n)
    hc0 = pair_diag(J0, z)
    gs = _ground(hc0, b, n)
    g = gauge_vector(seed, n, 1)
    Jp = gauged_J(J0, g)
    hcp = pair_diag(Jp, z)
    gs_p = map_gs(gs, g)
    e0, r0 = residual(hc0, b, n, gs)
    ep, rp = residual(hcp, b, n, gs_p)
    rng = np.random.default_rng(0)
    psi = rng.normal(size=1 << n)
    psi /= np.linalg.norm(psi)
    e_any_0, _ = residual(hc0, b, n, psi)
    e_any_g, _ = residual(hcp, b, n, map_gs(psi, g))
    plus_r = np.full(1 << n, 2.0 ** (-0.5 * n), dtype=np.float64)
    e_plus_id = float(np.vdot(plus_r, apply_H(hc0, b, n, plus_r)))
    e_plus_g = float(np.vdot(plus_r, apply_H(hcp, b, n, plus_r)))
    plus_mapped = apply_X_product(plus_r, flip_mask(g))
    dummy_gs = plus_r.copy()
    log0 = Logical(n, seed, b, dummy_gs, J=J0)
    logg = Logical(n, seed, b, dummy_gs, J=Jp)
    xq = np.array([0.25, 0.25], dtype=np.float64)
    xn = np.array([0.25, 0.0, 0.0, -0.5, 0.0, 0.0], dtype=np.float64)
    psi_q0 = log0.qaoa(xq, 1)
    psi_qg = logg.qaoa(xq, 1)
    e_q0 = log0.energy(psi_q0)
    e_qg = logg.energy(psi_qg)
    # QAOA state maps: U psi_0 = psi_g
    psi_q0_mapped = apply_X_product(psi_q0, flip_mask(g))
    f_q0 = float(np.abs(np.vdot(gs, psi_q0)) ** 2)
    f_qg = float(np.abs(np.vdot(gs_p, psi_qg)) ** 2)
    rng_x = np.random.default_rng(11)
    xq_r = rng_x.uniform(-0.5, 0.5, size=2)
    e_qr0 = log0.energy(log0.qaoa(xq_r, 1))
    e_qrg = logg.energy(logg.qaoa(xq_r, 1))
    e_n0 = log0.energy(log0.nov3(xn, 1))
    e_ng = logg.energy(logg.nov3(xn, 1))
    xn_gen = np.array([0.3, 0.2, -0.1, 0.4, -0.2, 0.15], dtype=np.float64)
    e_n0g = log0.energy(log0.nov3(xn_gen, 1))
    e_ngg = logg.energy(logg.nov3(xn_gen, 1))
    s0 = entropy_nats(gs, n)
    sg = entropy_nats(gs_p, n)
    return {
        "g": [int(v) for v in g],
        "energy_maps": bool(abs(ep - e0) < 1e-10 and abs(e_any_g - e_any_0) < 1e-10),
        "mapped_residual_ok": bool(r0 < 1e-8 and rp < 1e-8),
        "E0_id": e0,
        "E0_g": ep,
        "residual_id": r0,
        "residual_g": rp,
        "any_state_dE": abs(e_any_g - e_any_0),
        "plus_energy_id": e_plus_id,
        "plus_energy_g": e_plus_g,
        "plus_invariant": bool(abs(e_plus_id - e_plus_g) < 1e-12 and np.allclose(plus_mapped, plus_r)),
        "qaoa_center_dE": abs(e_qg - e_q0),
        "qaoa_center_dF": abs(f_qg - f_q0),
        "qaoa_state_mapped": bool(np.allclose(psi_qg, psi_q0_mapped, atol=1e-10)),
        "qaoa_random_dE": abs(e_qrg - e_qr0),
        "qaoa_invariant": bool(abs(e_qg - e_q0) < 1e-10 and abs(f_qg - f_q0) < 1e-12 and abs(e_qrg - e_qr0) < 1e-10),
        "nov3_center_dE": abs(e_ng - e_n0),
        "nov3_generic_dE": abs(e_ngg - e_n0g),
        "nov3_generic_not_covariant": bool(abs(e_ngg - e_n0g) > 1e-8),
        "nov3_not_assumed_covariant": True,
        "entropy_id": s0,
        "entropy_g": sg,
        "entropy_invariant": bool(abs(s0 - sg) < 1e-12),
        "transverse_X_ungauged": True,
    }
