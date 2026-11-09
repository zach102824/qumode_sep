"""B* instances: per-instance maximum-GS-entanglement-entropy field (evaluation-side instance definition).

Bipartition (fixed, user-confirmed to match the local stream): A = sites {0..floor(N/2)-1}; site i is
bit i of the basis index (LSB = site 0). S_A = von Neumann entropy (natural log) of the exact GS.
B* = refined argmax of S_A(b) over a coarse grid on [B_LO, B_HI], refined by bounded scalar search.
Checks recorded per instance (an instance that fails a hard check is rejected, never silently used):
  hard: grid argmax interior; refined optimum interior to its bracket; GS unique (E1-E0 > 1e-10) and
        Perron-Frobenius positive; GS even under the global spin flip; residual small.
  soft (reported): number of grid local maxima, E1-E0, even-sector gap, odd-sector gap, S_A, F_plus.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.sparse.linalg import eigsh, LinearOperator

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "LE"))
import core as C  # noqa: E402

B_LO, B_HI, NGRID, XATOL = 0.02, 3.0, 75, 1e-6


def hamiltonian_dense(J, b):
    N = J.shape[0]
    d = C.hzz_diag(J, C.z_table(N))
    dim = 1 << N
    H = np.diag(d)
    idx = np.arange(dim)
    for i in range(N):
        H[idx, idx ^ (1 << i)] -= b
    return H, d


def gs_fast(J, b, d):
    N = J.shape[0]
    dim = 1 << N
    if dim <= 512:
        H, _ = hamiltonian_dense(J, b)
        w, V = np.linalg.eigh(H)
        g = V[:, 0]
    else:
        def mv(v):
            v = np.asarray(v).reshape(-1)
            return d * v - b * C.apply_sumX(v, N)
        w, V = eigsh(LinearOperator((dim, dim), matvec=mv, dtype=np.float64), k=1, which="SA", tol=1e-13)
        g = V[:, 0]
    return g * np.sign(g.sum())


def entropy(g, N):
    nA = N // 2
    sv = np.linalg.svd(g.reshape(1 << (N - nA), 1 << nA), compute_uv=False) ** 2
    sv = sv[sv > 1e-300]
    return float(-(sv * np.log(sv)).sum())


def build(N, seed):
    J = C.make_J(seed, N)
    d = C.hzz_diag(J, C.z_table(N))
    grid = np.linspace(B_LO, B_HI, NGRID)
    S = np.array([entropy(gs_fast(J, b, d), N) for b in grid])
    k = int(np.argmax(S))
    nloc = int(sum(1 for i in range(1, NGRID - 1) if S[i] > S[i - 1] and S[i] > S[i + 1]))
    hard = {"grid_argmax_interior": bool(0 < k < NGRID - 1)}
    lo, hi = grid[max(k - 1, 0)], grid[min(k + 1, NGRID - 1)]
    r = minimize_scalar(lambda b: -entropy(gs_fast(J, b, d), N), bounds=(lo, hi), method="bounded",
                        options={"xatol": XATOL})
    bs = float(r.x)
    hard["refined_interior"] = bool(lo + 10 * XATOL < bs < hi - 10 * XATOL)
    H, _ = hamiltonian_dense(J, bs)
    w, V = np.linalg.eigh(H)
    dim = 1 << N
    flip = np.arange(dim) ^ (dim - 1)
    par = np.einsum("ik,ik->k", V, V[flip, :])
    even = np.where(par > 0)[0]
    odd = np.where(par < 0)[0]
    g = V[:, 0] * np.sign(V[:, 0].sum())
    hard["unique"] = bool(w[1] - w[0] > 1e-10)
    hard["perron_positive"] = bool(np.min(g) > 0)
    hard["gs_flip_even"] = bool(par[0] > 1 - 1e-10)
    resid = float(np.linalg.norm(H @ g - w[0] * g))
    hard["resid_small"] = bool(resid < 1e-9)
    meta = {
        "N": N, "seed": seed, "Bstar": bs, "S_A": entropy(g, N), "bipartition_A": list(range(N // 2)),
        "grid": [B_LO, B_HI, NGRID], "grid_argmax": float(grid[k]), "grid_local_maxima": nloc,
        "S_grid_max": float(S[k]), "J_sha256": C.J_sha(J), "J_upper": C.upper(J).tolist(),
        "E0": float(w[0]), "E1": float(w[1]), "gap_E1_E0": float(w[1] - w[0]),
        "gap_even": float(w[even[1]] - w[0]) if len(even) > 1 else None,
        "gap_odd": float(w[odd[0]] - w[0]) if len(odd) else None,
        "E1_parity": "even" if par[1] > 0 else "odd",
        "F_plus": C.plus_fidelity(g), "E_plus": float(-bs * N), "gs_resid": resid,
        "hard_checks": hard, "valid": bool(all(hard.values())),
    }
    assert meta["E0"] <= -bs * N + 1e-9
    return J, meta, g


if __name__ == "__main__":
    N, seed = int(sys.argv[1]), int(sys.argv[2])
    _, m, _ = build(N, seed)
    print(json.dumps({k: v for k, v in m.items() if k != "J_upper"}, indent=1))
