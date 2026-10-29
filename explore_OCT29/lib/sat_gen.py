"""4-SAT TEB instance generator (vendored spirit of oct_explore/sim.py).

- m = round(9.93 * n), 4 distinct sorted vars, random signs
- Keep only satisfiable instances (E.min() == 0) when E is computed
- Seed = base + 10_000_000 * n + k
"""
from __future__ import annotations
import random
from typing import List, Optional, Tuple

import numpy as np

# Historical TEB seed bases (prep_datasets / clean16 spirit)
TRB = 51_000_000_000  # train
TEB = 52_000_000_000  # test
VAB = 53_000_000_000  # val

CLAUSE_RATIO = 9.93


def clause_count(n: int) -> int:
    return int(round(CLAUSE_RATIO * n))


def fast_energies(n: int, cl) -> np.ndarray:
    """Unsat-clause count for every bitstring; index: var0 = MSB."""
    T = np.zeros([2] * n, np.int16)
    for v, s in cl:
        ix = [slice(None)] * n
        for vv, ss in zip(v, s):
            ix[vv] = 0 if ss > 0 else 1
        T[tuple(ix)] += 1
    return T.reshape(-1)


def gen_instance(n: int, seed: int, compute_E: bool = True):
    """Return (m, cl, E_or_None). cl = list of (vars[4], signs[4])."""
    r = random.Random(seed)
    m = clause_count(n)
    cl = []
    for _ in range(m):
        v = sorted(r.sample(range(n), 4))
        s = [r.choice([1, -1]) for _ in v]
        cl.append((v, s))
    E = fast_energies(n, cl) if compute_E else None
    return m, cl, E


def energy_of_assignment(n: int, cl, x_bits) -> int:
    """x_bits: length-n iterable, bit for var i (0/1). Count unsat clauses."""
    e = 0
    for v, s in cl:
        sat = False
        for vv, ss in zip(v, s):
            lit = x_bits[vv] if ss > 0 else 1 - x_bits[vv]
            if lit:
                sat = True
                break
        if not sat:
            e += 1
    return e


def is_satisfiable_via_E(E: np.ndarray) -> bool:
    return bool(E.min() == 0)


def is_satisfiable_bruteforce(n: int, cl) -> Tuple[bool, int]:
    """For n where dense E is skipped: scan all 2^n (ok up to ~21 with patience)."""
    m = len(cl)
    nsol = 0
    for x in range(1 << n):
        bits = [(x >> (n - 1 - i)) & 1 for i in range(n)]
        if energy_of_assignment(n, cl, bits) == 0:
            nsol += 1
            # early-ish: still need full nsol for pack metadata; continue
    return nsol > 0, nsol


def nsol_from_E(E: np.ndarray) -> int:
    return int((E == 0).sum())


def instances(
    n: int,
    count: int,
    base: int,
    compute_E: bool = True,
    max_tries: Optional[int] = None,
):
    """Keep satisfiable ones. Yields dict(seed, m, E|None, nsol, cl)."""
    out, k = [], 0
    limit = max_tries if max_tries is not None else max(count * 200, 10_000)
    while len(out) < count:
        if k >= limit:
            raise RuntimeError(
                f"Gave up after {k} tries collecting {count} sat instances at n={n} base={base}"
            )
        seed = int(base) + 10_000_000 * n + k
        k += 1
        m, cl, E = gen_instance(n, seed, compute_E=compute_E)
        if compute_E:
            if E.min() != 0:
                continue
            nsol = nsol_from_E(E)
        else:
            # For large n without dense E: accept all drawn formulas and set nsol=-1
            # (WalkSAT bars don't need unique-GS filter; optional --verify_sat later).
            nsol = -1
        out.append(dict(seed=seed, m=m, E=E, nsol=nsol, cl=cl))
    return out


def pack_clauses(cl_list: List, m: int, n: int):
    """Stack clause vars/signs into (B, m, 4) arrays."""
    B = len(cl_list)
    cls_v = np.zeros((B, m, 4), dtype=np.uint8)
    cls_s = np.zeros((B, m, 4), dtype=np.int8)
    for i, cl in enumerate(cl_list):
        for c_idx, (v, s) in enumerate(cl):
            cls_v[i, c_idx] = v
            cls_s[i, c_idx] = s
    return cls_v, cls_s
