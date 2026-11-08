"""Classical popcount shells for the original interleaved packing.

Uses only the ZZ diagonal. No quantum ground state, eigenvector, or fidelity.
"""
from __future__ import annotations

import numpy as np

# Rows are cavity popcounts 0, 1, 2, 3. Columns are the three shell weights.
T = np.array([
    [3.0, 3.0, 1.0],
    [1.0, -1.0, -1.0],
    [-1.0, -1.0, 1.0],
    [-3.0, 3.0, -1.0],
], dtype=np.float64)


def interleaved(N):
    m = int(N) // 3
    return [[a, a + m, a + 2 * m] for a in range(m)]


def classical_shell(hc, N):
    """Least squares of a 0/1 popcount target built from argmin of the ZZ diagonal.

    Ties in the diagonal keep the smallest basis index. The target phase is 0 on a
    cavity's popcount and 1 on the other three popcounts. The same three coefficients
    are shared by every cavity.
    """
    hc = np.asarray(hc, dtype=np.float64)
    i_star = int(np.argmin(hc))
    rows, rhs, pops = [], [], []
    for cav in interleaved(N):
        pop = 0
        for q in cav:
            pop += int((i_star >> int(q)) & 1)
        pops.append(int(pop))
        for p in range(4):
            rows.append(T[p])
            rhs.append(0.0 if p == pop else 1.0)
    coef, *_ = np.linalg.lstsq(np.asarray(rows, dtype=np.float64), np.asarray(rhs, dtype=np.float64), rcond=None)
    coef = np.array(coef, dtype=np.float64)
    if coef.shape != (3,) or not np.all(np.isfinite(coef)):
        raise SystemExit("classical shell solve failed")
    return i_star, pops, coef
