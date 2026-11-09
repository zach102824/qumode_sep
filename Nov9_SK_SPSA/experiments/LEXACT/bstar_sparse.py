"""B* for N>=15. Same checks as IMAX/bstar.py, no dense 2^N matrix.

Classical GS / field prep. Not charged to Q.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.sparse.linalg import LinearOperator, eigsh

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "LE"))
sys.path.insert(0, str(HERE.parent / "IMAX"))
import bstar as BS  # noqa: E402
import core as C  # noqa: E402
import kernel as K  # noqa: E402

B_LO, B_HI, NGRID, XATOL = BS.B_LO, BS.B_HI, BS.NGRID, BS.XATOL


def matvec_factory(d, b, N):
    dim = d.size
    tmp = np.empty(dim)

    def mv(v):
        v = np.asarray(v, dtype=np.float64).reshape(-1)
        K.apply_sumX_real(v, tmp)
        return d * v - b * tmp

    return LinearOperator((dim, dim), matvec=mv, dtype=np.float64)


def gs_one(d, b, N, v0=None, k=1):
    H = matvec_factory(d, b, N)
    kw = dict(k=k, which="SA", tol=1e-12)
    if v0 is not None:
        kw["v0"] = v0
    w, V = eigsh(H, **kw)
    o = np.argsort(w)
    g = V[:, o[0]]
    g = g * np.sign(g.sum())
    g = g / np.linalg.norm(g)
    return w[o], V[:, o], g


def entropy(g, N):
    nA = N // 2
    sv = np.linalg.svd(g.reshape(1 << (N - nA), 1 << nA), compute_uv=False) ** 2
    sv = sv[sv > 1e-300]
    return float(-(sv * np.log(sv)).sum())


def build(N, seed):
    if N <= 12:
        return BS.build(N, seed)
    J = C.make_J(seed, N)
    d = 0.5 * np.sum(C.z_table(N) * (J @ C.z_table(N)), axis=0)
    grid = np.linspace(B_LO, B_HI, NGRID)
    S = np.empty(NGRID)
    v0 = None
    for i, b in enumerate(grid):
        _, _, g = gs_one(d, b, N, v0=v0, k=1)
        v0 = g
        S[i] = entropy(g, N)
    k = int(np.argmax(S))
    nloc = int(sum(1 for i in range(1, NGRID - 1) if S[i] > S[i - 1] and S[i] > S[i + 1]))
    hard = {"grid_argmax_interior": bool(0 < k < NGRID - 1)}
    lo, hi = grid[max(k - 1, 0)], grid[min(k + 1, NGRID - 1)]

    def negS(b):
        _, _, g = gs_one(d, float(b), N, v0=v0, k=1)
        return -entropy(g, N)

    r = minimize_scalar(negS, bounds=(lo, hi), method="bounded", options={"xatol": XATOL})
    bs = float(r.x)
    hard["refined_interior"] = bool(lo + 10 * XATOL < bs < hi - 10 * XATOL)
    w, V, g = gs_one(d, bs, N, v0=v0, k=2)
    dim = 1 << N
    mask = dim - 1
    idx = np.arange(dim)
    par0 = float(g @ g[idx ^ mask])
    hard["unique"] = bool(w[1] - w[0] > 1e-10)
    hard["perron_positive"] = bool(np.min(g) > 0)
    hard["gs_flip_even"] = bool(par0 > 1 - 1e-10)
    Hv = matvec_factory(d, bs, N).matvec(g)
    resid = float(np.linalg.norm(Hv - w[0] * g))
    hard["resid_small"] = bool(resid < 1e-9)
    shift = 20.0 * N

    def mvo(v):
        v = np.asarray(v, dtype=np.float64).reshape(-1)
        return matvec_factory(d, bs, N).matvec(v) + shift * 0.5 * (v + v[idx ^ mask])

    wo, _ = eigsh(LinearOperator((dim, dim), matvec=mvo, dtype=np.float64), k=1, which="SA", tol=1e-10)
    E0_odd = float(wo[0])
    meta = {
        "N": N, "seed": seed, "Bstar": bs, "S_A": entropy(g, N), "bipartition_A": list(range(N // 2)),
        "grid": [B_LO, B_HI, NGRID], "grid_argmax": float(grid[k]), "grid_local_maxima": nloc,
        "S_grid_max": float(S[k]), "J_sha256": C.J_sha(J),
        "E0": float(w[0]), "E1": float(w[1]), "gap_E1_E0": float(w[1] - w[0]),
        "gap_even": None, "gap_odd": float(E0_odd - w[0]),
        "E1_parity": "odd" if abs(E0_odd - w[1]) < 1e-6 else "even_or_unresolved",
        "F_plus": C.plus_fidelity(g), "E_plus": float(-bs * N), "gs_resid": resid,
        "hard_checks": hard, "valid": bool(all(hard.values())),
        "gs_prep": "classical sparse Lanczos, not charged to Q",
        "band": "exclusive_dev_10032_10047",
    }
    return J, meta, g
