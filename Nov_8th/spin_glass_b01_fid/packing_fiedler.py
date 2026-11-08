"""Fiedler triple partition of |J|. Classical graph only. No quantum ground state."""
from __future__ import annotations

import numpy as np


def fiedler_packing(J):
    A = np.abs(np.asarray(J, dtype=np.float64))
    N = int(A.shape[0])
    A = 0.5 * (A + A.T)
    np.fill_diagonal(A, 0.0)
    lap = np.diag(A.sum(axis=1)) - A
    w, V = np.linalg.eigh(lap)
    gap = float(w[2] - w[1])
    if not np.isfinite(gap) or gap < 1e-8:
        raise SystemExit(f"Fiedler gap {gap} is below 1e-8")
    v = np.array(V[:, 1], dtype=np.float64)
    mag = np.abs(v)
    imax = int(np.flatnonzero(np.abs(mag - mag.max()) <= 1e-15)[0])
    if v[imax] < 0.0:
        v = -v
    order = sorted(range(N), key=lambda i: (float(v[i]), i))
    triples = [tuple(sorted(order[3 * a: 3 * a + 3])) for a in range(N // 3)]
    part = [list(t) for t in sorted(triples)]
    score = 0.0
    for t in part:
        score += abs(J[t[0], t[1]]) + abs(J[t[0], t[2]]) + abs(J[t[1], t[2]])
    covered = sorted(i for t in part for i in t)
    if covered != list(range(N)):
        raise SystemExit("Fiedler partition is not a partition")
    return part, float(score)
