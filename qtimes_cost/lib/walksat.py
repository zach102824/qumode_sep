"""Pure-Python SKC WalkSAT (vendored logic from classical/cls.c mode 0).

Pick random unsat clause; if some var has break==0 flip it; else w.p. p random
var in clause, else min-break var. Returns flips until first satisfying assignment.
Optional: compile lib/walksat_cls.c for a faster binary on chaodao.
"""
from __future__ import annotations
import math
import random
from typing import Dict, List, Tuple

import numpy as np


def walksat_one(
    n: int,
    cl: List[Tuple[List[int], List[int]]],
    p: float = 0.5,
    cap: int = 200_000_000,
    rng: random.Random | None = None,
) -> Tuple[int, int]:
    """Run one WalkSAT trajectory. Returns (flips, evals)."""
    rng = rng or random.Random()
    m = len(cl)
    V = [list(v) for v, _ in cl]
    SG = [list(s) for _, s in cl]
    occ = [[] for _ in range(n)]
    for c in range(m):
        for j in range(4):
            occ[V[c][j]].append(c)

    x = [rng.randrange(2) for _ in range(n)]
    tc = [0] * m
    unsat = []
    upos = [-1] * m

    def littrue(c, j):
        return (SG[c][j] > 0) == (x[V[c][j]] == 1)

    def addu(c):
        upos[c] = len(unsat)
        unsat.append(c)

    def delu(c):
        ppos = upos[c]
        last = unsat.pop()
        if ppos < len(unsat):
            unsat[ppos] = last
            upos[last] = ppos
        upos[c] = -1

    for c in range(m):
        tc[c] = sum(1 for j in range(4) if littrue(c, j))
        if tc[c] == 0:
            addu(c)

    def brk(v):
        b = 0
        for c in occ[v]:
            if tc[c] == 1:
                j = V[c].index(v)
                if littrue(c, j):
                    b += 1
        return b

    def flip(v):
        x[v] ^= 1
        for c in occ[v]:
            j = V[c].index(v)
            if littrue(c, j):
                if tc[c] == 0:
                    delu(c)
                tc[c] += 1
            else:
                tc[c] -= 1
                if tc[c] == 0:
                    addu(c)

    fl = 0
    ev = 1
    while unsat and fl < cap:
        c = unsat[rng.randrange(len(unsat))]
        bb = []
        for j in range(4):
            bb.append(brk(V[c][j]))
            ev += 1
        best = min(bb)
        cand = [j for j in range(4) if bb[j] == best]
        if best == 0 or rng.random() >= p:
            j = cand[rng.randrange(len(cand))]
        else:
            j = rng.randrange(4)
        flip(V[c][j])
        fl += 1
    return fl, ev


def walksat_instance(
    n: int,
    cl,
    R: int = 50,
    p: float = 0.5,
    seed: int = 1,
) -> Dict:
    """R independent runs; return per-run flips/evals and summary stats."""
    flips, evals = [], []
    for r in range(R):
        rng = random.Random(seed + 10007 * r + (seed % 1000003))
        fl, ev = walksat_one(n, cl, p=p, rng=rng)
        flips.append(fl)
        evals.append(ev)
    flips = np.asarray(flips, float)
    evals = np.asarray(evals, float)

    def stats(x):
        return {
            "median": float(np.median(x)),
            "B90": float(np.quantile(x, 0.9)),
            "mean": float(x.mean()),
        }

    return {
        "flips_per_run": flips.tolist(),
        "evals_per_run": evals.tolist(),
        "flips": stats(flips),
        "evals": stats(evals),
    }


def aggregate_bars(per_inst_med_flips: List[float]) -> Dict:
    x = np.asarray(per_inst_med_flips, float)
    return {
        "median": float(np.median(x)),
        "B90": float(np.quantile(x, 0.9)),
        "mean": float(x.mean()),
        "n_inst": int(len(x)),
    }
