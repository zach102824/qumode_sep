"""Interior max-entropy B*. Classical. Not a quantum-query cost and not a selector."""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.sparse.linalg import LinearOperator, eigsh

from Nov9_SK_SPSA.experiments.FT.instance import make_J, pair_diag, z_table

# Published LE / GR reproduction target. Not an optimization seed of this stream.
PUBLISHED_N9_SEED10000 = 0.5672817157081989


def apply_x_sum(v: np.ndarray, n: int) -> np.ndarray:
    out = np.zeros(v.shape, dtype=np.float64)
    idx = np.arange(v.shape[0])
    vv = np.asarray(v, dtype=np.float64)
    for q in range(n):
        out += vv[idx ^ (1 << q)]
    return out


def entropy_nats(gs: np.ndarray, n: int) -> float:
    n_a = n // 2
    psi = np.asarray(gs, dtype=np.float64).reshape(1 << (n - n_a), 1 << n_a)
    sv = np.linalg.svd(psi, compute_uv=False) ** 2
    sv = sv[sv > 1e-300]
    return float(-(sv * np.log(sv)).sum())


def _ground(hc: np.ndarray, b: float, n: int) -> np.ndarray:
    dim = 1 << n

    def mv(v):
        vv = np.asarray(v, dtype=np.float64)
        return hc * vv - float(b) * apply_x_sum(vv, n)

    op = LinearOperator((dim, dim), matvec=mv, dtype=np.float64)
    _w, vecs = eigsh(op, k=1, which="SA", tol=1e-12)
    g = np.real(np.asarray(vecs[:, 0], dtype=np.float64))
    if float(g.sum()) < 0.0:
        g = -g
    g /= np.linalg.norm(g)
    return g


def spectrum(hc: np.ndarray, b: float, n: int) -> dict:
    dim = 1 << n

    def mv(v):
        vv = np.asarray(v, dtype=np.float64)
        return hc * vv - float(b) * apply_x_sum(vv, n)

    op = LinearOperator((dim, dim), matvec=mv, dtype=np.float64)
    k = min(12, dim - 1)
    w, vecs = eigsh(op, k=k, which="SA", tol=1e-12)
    order = np.argsort(np.real(w))
    w = np.real(w[order])
    vecs = np.real(vecs[:, order])
    idx = np.arange(dim)
    mask = dim - 1
    parity = [float(vecs[:, i] @ vecs[idx ^ mask, i]) for i in range(k)]
    g = np.asarray(vecs[:, 0], dtype=np.float64)
    if float(g.sum()) < 0.0:
        g = -g
    g = g / np.linalg.norm(g)
    e0 = float(w[0])
    odd = [float(w[i]) for i in range(k) if parity[i] < -0.5]
    even_ex = [float(w[i]) for i in range(k) if parity[i] > 0.5 and float(w[i]) > e0 + 1e-8]
    if not odd or not even_ex:
        raise RuntimeError(f"gap sector missing at N={n} b={b}")
    return {
        "E0": e0,
        "E1": float(w[1]),
        "gap": float(w[1] - e0),
        "delta_odd": float(min(odd) - e0),
        "delta_even": float(min(even_ex) - e0),
        "E1_is_odd": bool(parity[1] < -0.5),
        "gs": g,
        "residual": float(np.linalg.norm(mv(g) - e0 * g)),
        "gs_min": float(np.min(g)),
        "gs_parity": float(g @ g[idx ^ mask]),
        "F_plus": float(np.sum(g) ** 2 / dim),
        "S_A": entropy_nats(g, n),
    }


def maximize(J: np.ndarray, n: int) -> dict:
    grid = np.linspace(0.05, 3.0, 60)
    hc = pair_diag(J, z_table(n))
    values = [entropy_nats(_ground(hc, float(b), n), n) for b in grid]
    k = int(np.argmax(values))
    if not (0 < k < len(grid) - 1):
        raise RuntimeError(f"entropy peak on a grid edge at N={n}")
    bracket = (float(grid[k - 1]), float(grid[k + 1]))

    def neg(b):
        return -entropy_nats(_ground(hc, float(b), n), n)

    refined = minimize_scalar(neg, bounds=bracket, method="bounded", options={"xatol": 1e-6})
    spec = spectrum(hc, float(refined.x), n)
    if spec["gap"] <= 1e-8 or spec["gs_parity"] < 1.0 - 1e-6 or spec["gs_min"] <= -1e-8 or spec["residual"] > 1e-8:
        raise RuntimeError(f"B* contract failed at N={n}")
    if spec["delta_odd"] <= 0.0 or spec["delta_even"] <= 0.0:
        raise RuntimeError(f"sector gap failed at N={n}")
    return {
        "Bstar": float(refined.x),
        "S_A": spec["S_A"],
        "grid_argmax_index": k,
        "interior": True,
        "cut": list(range(n // 2)),
        "E0": spec["E0"],
        "E1": spec["E1"],
        "gap": spec["gap"],
        "delta_odd": spec["delta_odd"],
        "delta_even": spec["delta_even"],
        "E1_is_odd": spec["E1_is_odd"],
        "F_plus": spec["F_plus"],
        "E_plus": -float(refined.x) * n,
        "residual": spec["residual"],
        "gs_min": spec["gs_min"],
        "gs_parity": spec["gs_parity"],
        "gs": spec["gs"],
        "hc": hc,
    }


def published_check() -> dict:
    rec = maximize(make_J(10000, 9), 9)
    err = abs(rec["Bstar"] - PUBLISHED_N9_SEED10000)
    if err > 1e-5:
        raise RuntimeError(f"published B* mismatch {rec['Bstar']}")
    return {"N": 9, "seed": 10000, "Bstar": rec["Bstar"], "abs_err": err, "S_A": rec["S_A"]}
