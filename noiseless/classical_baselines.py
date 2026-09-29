"""WalkSAT and simulated annealing baselines for planted 4-SAT at a matched budget.

Evaluation accounting ("one energy evaluation of a full assignment = one evaluation"):

* the random initial assignment costs 1 evaluation;
* WalkSAT (SKC variant, noise p = 0.5): each step picks a random violated clause and
  computes the break count of each of its 4 variables, i.e. the energy change of 4
  neighbouring full assignments -> 4 evaluations per step (charged even when the
  noisy random-walk branch is taken, since the freebie check needs them);
* SA: single-bit-flip Metropolis, 1 evaluation per proposed flip (the energy of the
  proposed neighbour); geometric temperature T0 = 2.0 -> T1 = 0.05 over the budget.

A run succeeds if it visits the planted ground state (E = 0) within the budget.
Budget unit B0 = 401 = cost evaluations of one 200-step SPSA run (2 per step + final);
budgets B0 x {1, 10, 100, 1000} are reported (x1000 ~ "shot-matched" to 1000 shots per
Gibbs-cost estimate). WalkSAT is anytime, so one run of the largest budget gives the
first-hit evaluation count for every smaller budget; SA's schedule depends on the
budget, so SA is re-run independently for each budget.
"""

from __future__ import annotations

import numpy as np
from numba import njit

B0 = 401
BUDGET_MULTS = (1, 10, 100, 1000)
WALKSAT_NOISE = 0.5
SA_T0 = 2.0
SA_T1 = 0.05


def build_structs(clauses: np.ndarray, polarities: np.ndarray, n: int):
    clauses = np.ascontiguousarray(clauses, dtype=np.int64)
    pats = np.ascontiguousarray(np.where(np.asarray(polarities) > 0, 0, 1), dtype=np.int64)
    occ = [[] for _ in range(n)]
    for c, vs in enumerate(clauses):
        for v in vs:
            occ[int(v)].append(c)
    ptr = np.zeros(n + 1, dtype=np.int64)
    for v in range(n):
        ptr[v + 1] = ptr[v] + len(occ[v])
    lst = np.array([c for v in range(n) for c in occ[v]], dtype=np.int64)
    return clauses, pats, ptr, lst


@njit(cache=True)
def _true_counts(x, clauses, pats):
    m = clauses.shape[0]
    tc = np.zeros(m, dtype=np.int64)
    for c in range(m):
        k = 0
        for j in range(4):
            if x[clauses[c, j]] != pats[c, j]:
                k += 1
        tc[c] = k
    return tc


@njit(cache=True)
def _flip(v, x, tc, clauses, pats, ptr, lst):
    """Flip x[v]; update true-literal counts; return energy change."""
    x[v] ^= 1
    de = 0
    for k in range(ptr[v], ptr[v + 1]):
        c = lst[k]
        for j in range(4):
            if clauses[c, j] == v:
                if x[v] != pats[c, j]:  # literal became true
                    if tc[c] == 0:
                        de -= 1
                    tc[c] += 1
                else:
                    tc[c] -= 1
                    if tc[c] == 0:
                        de += 1
    return de


@njit(cache=True)
def _break_count(v, x, tc, clauses, pats, ptr, lst):
    b = 0
    for k in range(ptr[v], ptr[v + 1]):
        c = lst[k]
        if tc[c] == 1:
            for j in range(4):
                if clauses[c, j] == v and x[v] != pats[c, j]:
                    b += 1
    return b


@njit(cache=True)
def walksat_run(seed, n, clauses, pats, ptr, lst, budget, noise):
    """Returns evaluations used when E=0 first reached, or -1 if not within budget."""
    np.random.seed(seed)
    x = np.zeros(n, dtype=np.int64)
    for i in range(n):
        x[i] = np.random.randint(2)
    tc = _true_counts(x, clauses, pats)
    m = clauses.shape[0]
    evals = 1
    unsat = np.empty(m, dtype=np.int64)
    bc = np.empty(4, dtype=np.int64)
    while True:
        nu = 0
        for c in range(m):
            if tc[c] == 0:
                unsat[nu] = c
                nu += 1
        if nu == 0:
            return evals
        if evals + 4 > budget:
            return -1
        c = unsat[np.random.randint(nu)]
        evals += 4
        best = 1 << 30
        for j in range(4):
            bc[j] = _break_count(clauses[c, j], x, tc, clauses, pats, ptr, lst)
            if bc[j] < best:
                best = bc[j]
        if best > 0 and np.random.random() < noise:
            j = np.random.randint(4)
        else:
            nb = 0
            for jj in range(4):
                if bc[jj] == best:
                    nb += 1
            r = np.random.randint(nb)
            j = -1
            for jj in range(4):
                if bc[jj] == best:
                    if r == 0:
                        j = jj
                        break
                    r -= 1
        _flip(clauses[c, j], x, tc, clauses, pats, ptr, lst)


@njit(cache=True)
def sa_run(seed, n, clauses, pats, ptr, lst, budget, t0, t1):
    """Returns evaluations used when E=0 first reached, or -1."""
    np.random.seed(seed)
    x = np.zeros(n, dtype=np.int64)
    for i in range(n):
        x[i] = np.random.randint(2)
    tc = _true_counts(x, clauses, pats)
    e = 0
    for c in range(clauses.shape[0]):
        if tc[c] == 0:
            e += 1
    evals = 1
    if e == 0:
        return evals
    nsteps = budget - 1
    for s in range(nsteps):
        T = t0 * (t1 / t0) ** (s / max(nsteps - 1, 1))
        v = np.random.randint(n)
        evals += 1
        de = _flip(v, x, tc, clauses, pats, ptr, lst)
        if de <= 0 or np.random.random() < np.exp(-de / T):
            e += de
            if e == 0:
                return evals
        else:
            _flip(v, x, tc, clauses, pats, ptr, lst)
    return -1


def run_baselines(clauses, polarities, n, seed: int, trials: int = 25, mults=BUDGET_MULTS) -> dict:
    cl, pa, ptr, lst = build_structs(clauses, polarities, n)
    budgets = [B0 * int(k) for k in mults]
    bmax = budgets[-1]
    ws_hits = [int(walksat_run((seed * 1000 + t) % 4294967296, n, cl, pa, ptr, lst, bmax, WALKSAT_NOISE)) for t in range(trials)]
    out = {"budgets": budgets, "walksat_first_hit": ws_hits, "walksat_success": {}, "sa_success": {}, "sa_first_hit": {}}
    for b in budgets:
        out["walksat_success"][str(b)] = float(np.mean([(h >= 0 and h <= b) for h in ws_hits]))
        hits = [int(sa_run((seed * 1000 + 500 + t) % 4294967296, n, cl, pa, ptr, lst, b, SA_T0, SA_T1)) for t in range(trials)]
        out["sa_first_hit"][str(b)] = hits
        out["sa_success"][str(b)] = float(np.mean([h >= 0 for h in hits]))
    return out
