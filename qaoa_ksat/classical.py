"""Classical local search on random k-SAT (same instances as QAOA).

Three solvers, all until E = 0:

* ``walksat_simple`` — paper Sec. IV A 2: random unsat clause, random variable.
* ``walksatlm`` — paper p=0.15, w1=6, w2=5; score = w1*make - w2*break.
* ``satlike`` — unweighted SATLike-style from ``noiseless/strong_classical.py``,
  generalized from 4-SAT to k-SAT.

TTS is **flips** (the scale Table II uses for WalkSATlm). ``lookups`` follows
STRONG_CLASSICAL_BASELINES.md: 1 for the start assignment, then k scored
neighbours per WalkSAT/WalkSATlm step; SATLike BMS samples as in the 4-SAT code.
"""

from __future__ import annotations

import numpy as np

WALKSATLM_P = 0.15
WALKSATLM_W1 = 6
WALKSATLM_W2 = 5
SATLIKE_SWT = 20
SATLIKE_SP = 0.01
SATLIKE_TBMS = 3


def build_occ(n: int, cv: np.ndarray) -> list[list[tuple[int, int]]]:
    """Every literal occurrence, including duplicates, so nsat matches ``_init_nsat``."""
    m, k = cv.shape
    occ: list[list[tuple[int, int]]] = [[] for _ in range(n)]
    for c in range(m):
        for j in range(k):
            occ[int(cv[c, j])].append((c, j))
    return occ


def _init_nsat(x: np.ndarray, cv: np.ndarray, cs: np.ndarray) -> np.ndarray:
    m, k = cv.shape
    nsat = np.zeros(m, dtype=np.int32)
    for c in range(m):
        s = 0
        for j in range(k):
            if x[cv[c, j]] == cs[c, j]:
                s += 1
        nsat[c] = s
    return nsat


def _flip(v: int, x: np.ndarray, nsat: np.ndarray, cv, cs, occ, unsat: set[int]) -> None:
    x[v] = 1 - x[v]
    for c, j in occ[v]:
        if x[v] == cs[c, j]:
            nsat[c] += 1
            if nsat[c] == 1:
                unsat.discard(c)
        else:
            nsat[c] -= 1
            if nsat[c] == 0:
                unsat.add(c)


def _make_break(v: int, x: np.ndarray, nsat: np.ndarray, cv, cs, occ) -> tuple[int, int]:
    mk = 0
    br = 0
    old = int(x[v])
    for c, j in occ[v]:
        lit_true = old == int(cs[c, j])
        if nsat[c] == 0 and not lit_true:
            mk += 1
        elif nsat[c] == 1 and lit_true:
            br += 1
    return mk, br


def _start(n: int, cv, cs, rng: np.random.Generator):
    x = rng.integers(0, 2, size=n, dtype=np.int8)
    nsat = _init_nsat(x, cv, cs)
    unsat = {int(c) for c in np.flatnonzero(nsat == 0)}
    return x, nsat, unsat


def walksat_simple(n, cv, cs, occ, rng, cap: int) -> tuple[int, int]:
    k = cv.shape[1]
    x, nsat, unsat = _start(n, cv, cs, rng)
    lookups = 1
    if not unsat:
        return 0, lookups
    flips = 0
    clist = cv
    while flips < cap:
        c = int(rng.choice(np.fromiter(unsat, dtype=np.int32, count=len(unsat))))
        j = int(rng.integers(0, k))
        _flip(int(clist[c, j]), x, nsat, cv, cs, occ, unsat)
        flips += 1
        lookups += 1
        if not unsat:
            return flips, lookups
    return -1, lookups


def walksatlm(n, cv, cs, occ, rng, cap: int, p=WALKSATLM_P, w1=WALKSATLM_W1, w2=WALKSATLM_W2) -> tuple[int, int]:
    k = cv.shape[1]
    x, nsat, unsat = _start(n, cv, cs, rng)
    lookups = 1
    if not unsat:
        return 0, lookups
    flips = 0
    while flips < cap:
        c = int(rng.choice(np.fromiter(unsat, dtype=np.int32, count=len(unsat))))
        lookups += k
        if rng.random() < p:
            j = int(rng.integers(0, k))
        else:
            best_s = None
            bests = []
            for t in range(k):
                v = int(cv[c, t])
                mk, br = _make_break(v, x, nsat, cv, cs, occ)
                s = w1 * mk - w2 * br
                if best_s is None or s > best_s:
                    best_s = s
                    bests = [t]
                elif s == best_s:
                    bests.append(t)
            j = int(bests[int(rng.integers(0, len(bests)))])
        _flip(int(cv[c, j]), x, nsat, cv, cs, occ, unsat)
        flips += 1
        if not unsat:
            return flips, lookups
    return -1, lookups


def satlike(
    n, cv, cs, occ, rng, cap: int, swt=SATLIKE_SWT, sp=SATLIKE_SP, t_bms=SATLIKE_TBMS
) -> tuple[int, int]:
    m, k = cv.shape
    x, nsat, unsat = _start(n, cv, cs, rng)
    lookups = 1
    if not unsat:
        return 0, lookups
    dw = np.ones(m, dtype=np.int32)
    age = np.zeros(n, dtype=np.int32)
    flips = 0
    step = 0
    while flips < cap:
        step += 1
        sc = np.zeros(n, dtype=np.int32)
        for v in range(n):
            mk = 0
            br = 0
            old = int(x[v])
            for c, j in occ[v]:
                lit_true = old == int(cs[c, j])
                if nsat[c] == 0 and not lit_true:
                    mk += int(dw[c])
                elif nsat[c] == 1 and lit_true:
                    br += int(dw[c])
            sc[v] = mk - br
        good = [v for v in range(n) if sc[v] > 0]
        best = -1
        if good:
            if len(good) <= t_bms:
                cand = good
            else:
                cand = [int(good[i]) for i in rng.integers(0, len(good), size=t_bms)]
            for v in cand:
                lookups += 1
                if best < 0 or sc[v] > sc[best] or (sc[v] == sc[best] and age[v] < age[best]):
                    best = v
        else:
            if rng.random() < sp:
                for c in range(m):
                    if nsat[c] > 0 and dw[c] > 1:
                        dw[c] -= 1
            else:
                for c in unsat:
                    if dw[c] < swt:
                        dw[c] += 1
            c = int(rng.choice(list(unsat)))
            for t in range(k):
                v = int(cv[c, t])
                lookups += 1
                if best < 0 or sc[v] > sc[best] or (sc[v] == sc[best] and age[v] < age[best]):
                    best = v
        _flip(best, x, nsat, cv, cs, occ, unsat)
        flips += 1
        age[best] = step
        if not unsat:
            return flips, lookups
    return -1, lookups


def run_solver(name: str, cv: np.ndarray, cs: np.ndarray, n: int, seed: int, cap: int) -> tuple[int, int]:
    rng = np.random.default_rng(int(seed))
    occ = build_occ(n, cv)
    if name == "walksat_simple":
        return walksat_simple(n, cv, cs, occ, rng, cap)
    if name == "walksatlm":
        return walksatlm(n, cv, cs, occ, rng, cap)
    if name == "satlike":
        return satlike(n, cv, cs, occ, rng, cap)
    raise ValueError(name)
