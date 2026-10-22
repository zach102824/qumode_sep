"""CLEAN 16-level scaling study, 8-SAT version (2026-10-22). Imported as `common` by the clean16 scripts copied here.
Paper ensemble (PRX Quantum 5, 030348, Definition 1 + Table 1): m ~ Poisson(r n), r = 176.54, each of the 8 literals drawn
independently WITH replacement from {x_i, not x_i}; only satisfiable instances kept (no unique-solution filter, as in the paper).
sim.gen_instance / sim.instances are replaced IN MEMORY by this generator (seed = base + 1e7 n + idx, numpy default_rng(seed)).
Cost normalisation nm = 1/r (= n / E[m]) for every formula. Fresh bases 61e9 train / 62e9 test / 63e9 val (never used before)."""
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE)
for p in (ROOT, os.path.join(ROOT, "v3", "native"), os.path.join(ROOT, "v3", "openevolve", "net")):
    if p not in sys.path: sys.path.insert(0, p)
import numpy as np
import compiler as CP, sim
CP.LAYOUTS[20] = [4, 4, 4, 4, 4]
for n in (8, 12, 16, 20): assert CP.LAYOUTS[n] == [4] * (n // 4)
K = 8; R = 176.54; NM = 1.0 / R
TRB, TEB, VAB = 61e9, 62e9, 63e9
TEST_N = [8, 12, 16, 20]; NTEST = 100; VAL_N = 12; NVAL = 200
RES = os.path.join(ROOT, "results", "clean16_8sat"); os.makedirs(RES, exist_ok=True)

def energies(n, cl):
    """unsat-clause count per string, var 0 = MSB; tautological clauses (x and not x) are never violated; repeated literals collapse."""
    T = np.zeros([2] * n, np.int16)
    for v, s in cl:
        req = {}
        for vv, ss in zip(v, s):
            need = 0 if ss > 0 else 1
            if req.get(vv, need) != need: req = None; break
            req[vv] = need
        if req is None: continue
        ix = [slice(None)] * n
        for vv, b in req.items(): ix[vv] = b
        T[tuple(ix)] += 1
    return T.reshape(-1)

def gen_instance(n, seed):
    rng = np.random.default_rng(int(seed)); m = int(rng.poisson(R * n)); L = rng.integers(0, 2 * n, size=(m, K))
    cl = [((L[j] // 2).tolist(), (1 - 2 * (L[j] % 2)).tolist()) for j in range(m)]   # even literal index = positive
    return m, cl, energies(n, cl)

def instances(n, count, base):
    out, k = [], 0
    while len(out) < count:
        seed = int(base) + 10_000_000 * n + k; k += 1
        m, cl, E = gen_instance(n, seed)
        if E.min() == 0: out.append(dict(seed=seed, m=m, E=E, nsol=int((E == 0).sum())))
    return out
sim.gen_instance = gen_instance; sim.instances = instances
