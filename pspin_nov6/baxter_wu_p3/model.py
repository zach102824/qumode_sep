"""Baxter–Wu triangular torus with an up/down sign texture.

Primary Hamiltonian (staggered, lambda = 1), verified non-gauge-equivalent
to the uniform ferromagnet on every torus used here::

    H = - sum_{up ▲} Z_a Z_b Z_c + lambda * sum_{down ▼} Z_a Z_b Z_c
        - Gamma * sum_i X_i

lambda = 1 is the production sign texture. lambda = -1 is the uniform-sign
control (both families ferro). lambda = 0.5 keeps the same frustrated signs
with a weaker down-triangle weight.

The target vector is always the lowest eigenvector of this full 2^N operator.
No sublattice sector is projected out. When a 3-coloring exists, the
double-sublattice X products commute with H; their eigenvalues on the
computed ground state are reported, and the state is kept even if it is a cat.
"""

from __future__ import annotations

import math

import numpy as np

import geometry as geo


def faces_and_coeffs(geom: dict, lam: float):
    """Return (triangles, coefficients) of H_C.

    H_C = -sum_▲ ZZZ + lam * sum_▼ ZZZ.
    """
    tris = []
    coeffs = []
    for tri in geom["up"]:
        tris.append(tri)
        coeffs.append(-1.0)
    for tri in geom["down"]:
        tris.append(tri)
        coeffs.append(float(lam))
    return tris, np.asarray(coeffs, dtype=np.float64)


def zzz_diag(n: int, tri) -> np.ndarray:
    idx = np.arange(1 << n, dtype=np.uint32)
    bits = np.zeros(1 << n, dtype=np.uint8)
    for v in tri:
        bits ^= ((idx >> np.uint32(v)) & np.uint32(1)).astype(np.uint8)
    return np.where(bits == 0, 1.0, -1.0).astype(np.float64)


def cost_diagonal(n: int, tris, coeffs) -> np.ndarray:
    hc = np.zeros(1 << n, dtype=np.float64)
    for tri, c in zip(tris, coeffs):
        hc += float(c) * zzz_diag(n, tri)
    return hc


def kron_zzz_diag(n: int, tri) -> np.ndarray:
    """Independent Pauli diagonal: Kronecker product, qubit of index q is bit q."""
    Z = np.array([[1.0, 0.0], [0.0, -1.0]])
    I = np.eye(2)
    acc = np.array([[1.0]])
    wanted = set(int(v) for v in tri)
    for q in range(n - 1, -1, -1):
        acc = np.kron(acc, Z if q in wanted else I)
    return np.diag(acc).real.copy()


def classical_summary(hc: np.ndarray) -> dict:
    emin = float(hc.min())
    emax = float(hc.max())
    deg = int(np.sum(np.isclose(hc, emin)))
    idx = np.flatnonzero(np.isclose(hc, emin))
    return {
        "Emin": emin,
        "Emax_diag": emax,
        "degeneracy": deg,
        "gs_indices_head": [int(i) for i in idx[:16]],
        "n_listed": int(min(16, deg)),
    }


def energy_step_and_period(hc: np.ndarray) -> dict:
    """Exact gamma period from the cost diagonal.

    If every diagonal entry is an integer multiple of delta, then
    gamma -> gamma + 2π/delta multiplies every amplitude by a global phase.
    A half step is a period only when it also acts as a global phase, which
    is checked by the caller on a state.
    """
    scale = 2.0  # allow half-integers (lambda = 1/2)
    scaled = np.rint(hc * scale)
    if float(np.max(np.abs(hc * scale - scaled))) > 1e-6:
        return {"delta": None, "gamma_period": None, "reason": "diagonal not in (1/2)Z"}
    scaled = scaled.astype(np.int64)
    base = int(scaled.min())
    g = 0
    for v in np.unique(scaled):
        g = math.gcd(g, int(v - base))
    # energies differ by g/scale
    delta = g / scale
    period = (2.0 * np.pi / delta) if delta > 0 else None
    return {
        "delta": float(delta),
        "gamma_period": float(period) if period is not None else None,
        "scaled_gcd": int(g),
        "all_even_integers": bool(np.all(scaled % 2 == 0) and scale == 2.0 and g % 2 == 0),
    }


def apply_H(v: np.ndarray, hc: np.ndarray, gamma: float, n: int) -> np.ndarray:
    out = hc * v
    idx = np.arange(v.shape[0], dtype=np.int64)
    for q in range(n):
        out = out - float(gamma) * v[idx ^ (1 << q)]
    return out


def ground_state(hc: np.ndarray, gamma: float, n: int, k: int = 2):
    """Full-space Lanczos. No sector projection."""
    from scipy.sparse.linalg import LinearOperator, eigsh

    dim = 1 << n
    hc = np.asarray(hc, dtype=np.float64)
    gamma = float(gamma)

    def matvec(v):
        return apply_H(np.asarray(v, dtype=np.float64), hc, gamma, n)

    op = LinearOperator((dim, dim), matvec=matvec, dtype=np.float64)
    evals, evecs = eigsh(op, k=k, which="SA", tol=1e-10, maxiter=max(3000, 8 * n))
    order = np.argsort(evals)
    e0 = float(evals[order[0]])
    e1 = float(evals[order[1]]) if k > 1 else float("nan")
    gs = np.real(evecs[:, order[0]]).astype(np.float64)
    gs /= np.linalg.norm(gs)
    if float(np.sum(gs)) < 0:
        gs *= -1.0
    resid = float(np.linalg.norm(matvec(gs) - e0 * gs))
    # one largest Ritz value checks Emax = -E0 (odd-body + transverse field)
    hi, _ = eigsh(op, k=1, which="LA", tol=1e-8, maxiter=max(2000, 4 * n))
    emax = float(hi[0])
    return {
        "E0": e0,
        "E1": e1,
        "Emax": emax,
        "gap": float(e1 - e0),
        "emax_plus_e0": float(emax + e0),
        "residual": resid,
        "gs": gs,
        "n_negative": int(np.sum(gs < -1e-8)),
        "dim": dim,
        "degenerate": bool((e1 - e0) < 1e-8),
    }


def ground_state_dense(hc: np.ndarray, gamma: float, n: int):
    dim = 1 << n
    H = np.diag(hc.astype(np.float64))
    idx = np.arange(dim)
    for q in range(n):
        partner = idx ^ (1 << q)
        upper = partner > idx
        ii = idx[upper]
        jj = partner[upper]
        H[ii, jj] -= float(gamma)
        H[jj, ii] -= float(gamma)
    evals, evecs = np.linalg.eigh(H)
    gs = np.real(evecs[:, 0]).copy()
    if gs.sum() < 0:
        gs *= -1.0
    gs /= np.linalg.norm(gs)
    return {
        "E0": float(evals[0]),
        "E1": float(evals[1]),
        "Emax": float(evals[-1]),
        "gap": float(evals[1] - evals[0]),
        "gs": gs,
        "residual": float(np.linalg.norm(H @ gs - evals[0] * gs)),
    }


def plus_state(n: int) -> np.ndarray:
    return np.full(1 << n, 1.0 / np.sqrt(1 << n), dtype=np.complex128)


def half_cut_entropy(psi: np.ndarray, keep: np.ndarray) -> dict:
    """Entropy of the qubits listed in `keep` (traced against the complement).

    The computational index uses qubit q as bit q. `keep` is sorted.
    """
    n = int(round(np.log2(psi.shape[0])))
    keep = [int(q) for q in keep]
    drop = [q for q in range(n) if q not in set(keep)]
    # reorder bits so `keep` are the least significant bits
    amp = np.asarray(psi, dtype=np.complex128)
    order = drop + keep  # index bit 0 of the new layout is order[0], which we want as MSB? 
    # We reshape with the last axis = keep. Build the permutation of the flat index.
    # new_index bit position: we'll place keep in the low bits.
    src = np.arange(1 << n, dtype=np.int64)
    new = np.zeros_like(src)
    for new_bit, q in enumerate(keep):
        new |= ((src >> q) & 1) << new_bit
    for new_bit, q in enumerate(drop):
        new |= ((src >> q) & 1) << (len(keep) + new_bit)
    reordered = np.zeros_like(amp)
    reordered[new] = amp
    na = len(keep)
    nb = n - na
    V = reordered.reshape(1 << nb, 1 << na)
    rho = V.conj().T @ V
    ev = np.clip(np.linalg.eigvalsh(rho).real, 0.0, None)
    ev = ev[ev > 1e-15]
    S = float(-(ev * np.log2(ev)).sum()) if ev.size else 0.0
    return {"S_bits": S, "keep": keep, "n_keep": na, "n_drop": nb}


def spatial_cut(lx: int, ly: int) -> dict:
    """A geometric cut: the lower x-half if Lx even, else the lower y-half if Ly even,
    else the lowest N//2 indices. The kept sites are listed explicitly.
    """
    n = lx * ly
    if lx % 2 == 0:
        keep = [site for site in range(n) if (site % lx) < lx // 2]
        kind = f"x < {lx // 2}"
    elif ly % 2 == 0:
        keep = [site for site in range(n) if (site // lx) < ly // 2]
        kind = f"y < {ly // 2}"
    else:
        keep = list(range(n // 2))
        kind = "lowest N//2 indices (both periods odd; not a spatial half)"
    return {"keep": keep, "kind": kind}


def z_expectations(psi, n: int) -> np.ndarray:
    p = np.abs(np.asarray(psi)) ** 2
    idx = np.arange(p.shape[0], dtype=np.int64)
    out = np.empty(n, dtype=np.float64)
    for q in range(n):
        z = np.where(((idx >> q) & 1) == 0, 1.0, -1.0)
        out[q] = float(np.dot(p, z))
    return out


def clause_means(psi, n: int, geom: dict) -> dict:
    p = np.abs(np.asarray(psi)) ** 2
    idx = np.arange(p.shape[0], dtype=np.uint32)

    def mean(tris):
        acc = []
        for tri in tris:
            bits = np.zeros(p.shape[0], dtype=np.uint8)
            for v in tri:
                bits ^= ((idx >> np.uint32(v)) & np.uint32(1)).astype(np.uint8)
            zzz = np.where(bits == 0, 1.0, -1.0)
            acc.append(float(np.dot(p, zzz)))
        return float(np.mean(acc)) if acc else float("nan")

    return {"up": mean(geom["up"]), "down": mean(geom["down"])}


def product_state(theta: np.ndarray, n: int) -> np.ndarray:
    psi = np.array([1.0])
    for q in range(n - 1, -1, -1):
        a0 = np.cos(float(theta[q]) / 2.0)
        a1 = np.sin(float(theta[q]) / 2.0)
        psi = np.kron(psi, np.array([a0, a1]))
    return psi


def best_product_overlap(gs: np.ndarray, n: int, seed: int = 0) -> dict:
    from scipy.optimize import minimize

    gs = np.real(np.asarray(gs, dtype=np.float64))
    gs = gs / np.linalg.norm(gs)

    def loss(theta):
        return -float(np.dot(product_state(theta, n), gs)) ** 2

    z = np.clip(z_expectations(gs, n), -1.0, 1.0)
    marginal = np.arccos(z)
    starts = [np.zeros(n), np.full(n, np.pi / 2), marginal]
    rng = np.random.default_rng(10_000 + int(seed))
    for _ in range(2):
        starts.append(np.clip(marginal + rng.normal(0.0, 0.4, size=n), 0.0, np.pi))
    best = -1.0
    for th0 in starts:
        res = minimize(loss, np.asarray(th0, dtype=np.float64), method="L-BFGS-B", bounds=[(0.0, np.pi)] * n, options={"maxiter": 60, "ftol": 1e-14})
        best = max(best, float(-res.fun))
    f_plus = float(np.dot(gs, np.full(gs.size, 1.0 / np.sqrt(gs.size))) ** 2)
    return {"F_product": best, "F_plus": f_plus, "F_best_bit": float(np.max(gs ** 2))}


def classical_manifold_overlaps(gs: np.ndarray, hc: np.ndarray) -> dict:
    """Overlap with the classical ground manifold and with its equal positive cat.

    The cat is deg^{-1/2} sum_{s in classical GS} |s>, not a hand-picked sector.
    """
    gs = np.real(np.asarray(gs, dtype=np.float64))
    emin = hc.min()
    mask = np.isclose(hc, emin)
    deg = int(mask.sum())
    amp = gs[mask]
    f_sub = float(np.sum(amp ** 2))
    cat_amp = amp.sum() / np.sqrt(deg)
    f_cat = float(cat_amp ** 2)
    return {"degeneracy": deg, "F_classical_subspace": f_sub, "F_equal_positive_cat": f_cat}


def sublattice_eigenvalues(gs: np.ndarray, lx: int, ly: int, geom: dict) -> dict:
    """Eigenvalue of each double-sublattice flip on the full-space GS, if a coloring exists."""
    col = geo.coloring_report(lx, ly, geom)
    if not col["face_coloring_exists"]:
        return {"applicable": False, "reason": "no 3-coloring on this torus"}
    colors = geo.extend_face_coloring(lx, ly, geom)["colors"]
    out = {"applicable": True}
    for a, b in ((0, 1), (0, 2), (1, 2)):
        mask = 0
        for q in range(geom["n"]):
            if int(colors[q]) in (a, b):
                mask |= 1 << q
        flipped = geo_flip(gs, mask)
        # <gs|P|gs> for a real normalized vector that is an eigenvector is ±1
        ev = float(np.dot(gs, flipped))
        out[f"flip_{a}{b}"] = ev
    return out


def geo_flip(psi: np.ndarray, mask: int) -> np.ndarray:
    idx = np.arange(psi.shape[0], dtype=np.int64)
    out = np.empty_like(psi)
    out[idx] = psi[idx ^ int(mask)]
    return out


def term_list(lx: int, ly: int, lam: float, gamma: float) -> list:
    geom = geo.build_triangles(lx, ly)
    terms = []
    for tri in geom["up"]:
        terms.append({"pauli": "ZZZ", "qubits": list(tri), "coeff": -1.0, "face": "up"})
    for tri in geom["down"]:
        terms.append({"pauli": "ZZZ", "qubits": list(tri), "coeff": float(lam), "face": "down"})
    for q in range(lx * ly):
        terms.append({"pauli": "X", "qubits": [q], "coeff": -float(gamma), "face": "field"})
    return terms
