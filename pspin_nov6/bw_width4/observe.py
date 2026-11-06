"""Observables on a saved state. Cuts and momenta are explicit.

Sublattice magnetizations are omitted unless a 3-face-coloring exists.
The 3-sublattice wavevector K=(b1+b2)/3 is reported only when it lies on
the finite-torus momentum grid (both periods divisible by 3).
"""
from __future__ import annotations

import numpy as np

from geometry import k_point_commensurate, momentum_grid, pair_distances, positions


def z_one(psi, n: int) -> np.ndarray:
    p = np.abs(np.asarray(psi)) ** 2
    idx = np.arange(p.shape[0], dtype=np.int64)
    out = np.empty(n)
    for q in range(n):
        z = np.where(((idx >> q) & 1) == 0, 1.0, -1.0)
        out[q] = float(np.dot(p, z))
    return out


def zz_and_connected(psi, lx, ly) -> dict:
    n = lx * ly
    p = np.abs(np.asarray(psi)) ** 2
    m = z_one(psi, n)
    groups = pair_distances(lx, ly)
    idx = np.arange(p.shape[0], dtype=np.int64)
    by_d = {}
    for d, pairs in sorted(groups.items()):
        acc = 0.0
        acc_c = 0.0
        for i, j in pairs:
            same = (((idx >> i) & 1) == ((idx >> j) & 1)).astype(np.float64)
            zz = float(np.dot(p, np.where(same == 1.0, 1.0, -1.0)))
            acc += zz
            acc_c += zz - m[i] * m[j]
        by_d[d] = {
            "n_pairs": len(pairs),
            "mean_zz": acc / len(pairs),
            "mean_zz_connected": acc_c / len(pairs),
        }
    return {"m": m, "by_distance": by_d}


def zzz_means(psi, geom) -> dict:
    n = geom["n"]
    p = np.abs(np.asarray(psi)) ** 2
    idx = np.arange(p.shape[0], dtype=np.uint32)

    def mean(tris):
        acc = 0.0
        for tri in tris:
            bits = np.zeros(p.shape[0], dtype=np.uint8)
            for v in tri:
                bits ^= ((idx >> np.uint32(v)) & np.uint32(1)).astype(np.uint8)
            acc += float(np.dot(p, np.where(bits == 0, 1.0, -1.0)))
        return acc / len(tris)

    up = mean(geom["up"])
    down = mean(geom["down"])
    return {"mean_zzz_up": up, "mean_zzz_down": down, "stagger": up - down}


def half_cut_entropy(psi, keep: list[int]) -> float:
    n = int(round(np.log2(len(psi))))
    keep = [int(q) for q in keep]
    drop = [q for q in range(n) if q not in set(keep)]
    amp = np.asarray(psi, dtype=np.complex128)
    src = np.arange(1 << n, dtype=np.int64)
    new = np.zeros_like(src)
    for b, q in enumerate(keep):
        new |= ((src >> q) & 1) << b
    for b, q in enumerate(drop):
        new |= ((src >> q) & 1) << (len(keep) + b)
    reordered = np.zeros_like(amp)
    reordered[new] = amp
    V = reordered.reshape(1 << len(drop), 1 << len(keep))
    ev = np.clip(np.linalg.eigvalsh(V.conj().T @ V).real, 0.0, None)
    ev = ev[ev > 1e-15]
    if ev.size == 0:
        return 0.0
    return float(-(ev * np.log2(ev)).sum())


def spatial_cuts(lx: int, ly: int) -> list[dict]:
    n = lx * ly
    cuts = []
    if lx % 2 == 0:
        keep = [s for s in range(n) if (s % lx) < lx // 2]
        cuts.append({"name": f"x<{lx // 2}", "keep": keep})
    if ly % 2 == 0:
        keep = [s for s in range(n) if (s // lx) < ly // 2]
        cuts.append({"name": f"y<{ly // 2}", "keep": keep})
    if not cuts:
        cuts.append({"name": "lowest N//2 indices", "keep": list(range(n // 2))})
    return cuts


def entropies(psi, lx, ly) -> dict:
    cuts = spatial_cuts(lx, ly)
    out = {}
    for c in cuts:
        out[c["name"]] = {
            "S_bits": half_cut_entropy(psi, c["keep"]),
            "keep": c["keep"],
        }
    vals = [v["S_bits"] for v in out.values()]
    out["S_min_among_these_cuts"] = float(min(vals))
    out["S_primary_bits"] = vals[0]
    out["primary_cut"] = cuts[0]["name"]
    return out


def structure_factor(psi, lx, ly) -> dict:
    """Connected spin structure factor on the torus momentum grid.

    S_c(q) = (1/N) sum_{ij} exp(i q·(r_i-r_j)) (<Zi Zj> - <Zi><Zj>).
    """
    n = lx * ly
    corr = zz_and_connected(psi, lx, ly)
    m = corr["m"]
    # full <Zi Zj> matrix is small (N<=16)
    p = np.abs(np.asarray(psi)) ** 2
    idx = np.arange(len(p), dtype=np.int64)
    C = np.empty((n, n), dtype=np.float64)
    for i in range(n):
        C[i, i] = 1.0 - m[i] ** 2
        zi = (idx >> i) & 1
        for j in range(i + 1, n):
            same = (zi == ((idx >> j) & 1)).astype(np.float64)
            zz = float(np.dot(p, np.where(same == 1.0, 1.0, -1.0)))
            cij = zz - m[i] * m[j]
            C[i, j] = C[j, i] = cij
    r = positions(lx, ly)
    grid = momentum_grid(lx, ly)
    best = None
    s0 = None
    for item in grid:
        q = item["q"]
        phase = np.exp(1j * (r @ q))
        # (1/N) phase^H C phase
        v = phase
        S = float(np.real(np.conjugate(v) @ (C @ v)) / n)
        rec = {"nx": item["nx"], "ny": item["ny"], "S": S}
        if item["nx"] == 0 and item["ny"] == 0:
            s0 = S
        elif best is None or S > best["S"]:
            best = rec
    kinfo = k_point_commensurate(lx, ly)
    return {
        "S_connected_q0": s0,
        "S_connected_max_nonzero": None if best is None else best,
        **kinfo,
    }


def product_overlap(psi, n: int) -> float:
    """Best real nonnegative product state. A variational number, not a bound theorem."""
    amps = np.clip(np.real(np.asarray(psi)), 0.0, None)
    if float(amps.sum()) <= 1e-15:
        amps = np.abs(np.asarray(psi))
    theta = np.full(n, 0.5 * np.pi)

    def ov(th):
        v = np.array([1.0])
        for t in th:
            v = np.kron(v, np.array([np.cos(t / 2.0), np.sin(t / 2.0)]))
        return float(np.dot(v, amps))

    for _ in range(3):
        for i in range(n):
            grid = np.linspace(0.0, np.pi, 17)
            scores = []
            for g in grid:
                theta[i] = g
                scores.append(ov(theta))
            theta[i] = grid[int(np.argmax(scores))]
    return float(ov(theta) ** 2)


def state_observables(psi, geom, hc, manifold_idx) -> dict:
    lx, ly, n = geom["lx"], geom["ly"], geom["n"]
    corr = zz_and_connected(psi, lx, ly)
    zzz = zzz_means(psi, geom)
    ent = entropies(psi, lx, ly)
    # drop the full keep lists from the compact row; caller can recompute
    ent_compact = {
        "primary_cut": ent["primary_cut"],
        "S_primary_bits": ent["S_primary_bits"],
        "S_min_among_these_cuts": ent["S_min_among_these_cuts"],
        "cuts": {k: v["S_bits"] for k, v in ent.items() if isinstance(v, dict)},
    }
    sf = structure_factor(psi, lx, ly)
    return {
        "mean_zzz_up": zzz["mean_zzz_up"],
        "mean_zzz_down": zzz["mean_zzz_down"],
        "stagger": zzz["stagger"],
        "zz_connected_by_distance": {str(d): v["mean_zz_connected"] for d, v in corr["by_distance"].items()},
        "zz_by_distance": {str(d): v["mean_zz"] for d, v in corr["by_distance"].items()},
        "entropy": ent_compact,
        "structure_factor": {
            "S_q0": sf["S_connected_q0"],
            "S_max_nonzero": sf["S_connected_max_nonzero"],
            "K_on_momentum_grid": sf["K_on_momentum_grid"],
        },
        "F_classical_manifold": float(np.abs(np.asarray(psi))[manifold_idx].sum() ** 2) if False else float(np.abs(np.asarray(psi)) ** 2 @ _mask(len(psi), manifold_idx)),
        "sublattice_m": None,
        "sublattice_note": "no 3-face-coloring on this torus; m_A,m_B,m_C omitted",
    }


def _mask(dim, indices):
    m = np.zeros(dim, dtype=np.float64)
    m[np.asarray(indices, dtype=np.int64)] = 1.0
    return m
