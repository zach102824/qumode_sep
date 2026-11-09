"""Ferromagnetic triple partition. Classical couplings only.

H = sum J_ij Z_i Z_j is lowered by aligning Z_i and Z_j when J_ij is negative.
The score of a triple is minus the sum of its three couplings. No eigenvector,
no quantum ground state, and no fidelity.
"""
from __future__ import annotations

import numpy as np


def ferro_packing(J):
    J = np.asarray(J, dtype=np.float64)
    N = int(J.shape[0])
    sites = list(range(N))
    best, best_score = None, -1.0e300

    def rec(rem, acc):
        nonlocal best, best_score
        if not rem:
            part = tuple(sorted(tuple(sorted(t)) for t in acc))
            score = 0.0
            for t in part:
                score -= float(J[t[0], t[1]] + J[t[0], t[2]] + J[t[1], t[2]])
            if score > best_score + 1e-12 or (abs(score - best_score) <= 1e-12 and (best is None or part < best)):
                best, best_score = part, score
            return
        s0, rest = rem[0], rem[1:]
        for i in range(len(rest)):
            for j in range(i + 1, len(rest)):
                rec([rest[k] for k in range(len(rest)) if k != i and k != j], acc + [(s0, rest[i], rest[j])])

    rec(sites, [])
    part = [list(t) for t in best]
    covered = sorted(i for t in part for i in t)
    if covered != list(range(N)) or len(part) != N // 3:
        raise SystemExit("ferromagnetic partition is not a partition")
    return part, float(best_score)
