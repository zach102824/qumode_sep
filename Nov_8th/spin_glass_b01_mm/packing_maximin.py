"""Maximin ferromagnetic triple partition. Classical couplings only.

The score of a triple is minus the sum of its three couplings. The score of a
partition is the minimum of those triple scores. No eigenvector, no quantum
ground state, and no fidelity.
"""
from __future__ import annotations

import numpy as np


def _triple_score(J, t):
    return -float(J[t[0], t[1]] + J[t[0], t[2]] + J[t[1], t[2]])


def maximin_packing(J):
    J = np.asarray(J, dtype=np.float64)
    N = int(J.shape[0])
    sites = list(range(N))
    best, best_score = None, -1.0e300

    def rec(rem, acc):
        nonlocal best, best_score
        if not rem:
            part = tuple(sorted(tuple(sorted(t)) for t in acc))
            score = min(_triple_score(J, t) for t in part)
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
        raise SystemExit("maximin partition is not a partition")
    return part, float(best_score)
