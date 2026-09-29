"""Family F1: planted, locally rigid, unique-solution random 4-SAT on n variables.

Generalizes ``Hamiltonians/four_sat.py`` (n = 8) to n in {8, 12, 16, 20, 24, 28}
for the RY-only / HEA scaling study. Conventions are unchanged:

* bit x_i in {0, 1}, Z|0> = +1; bitstring character i = site i = MSB-first bit
  (basis index = int(bitstring, 2));
* clause = (variables, polarities), polarity +1 means literal x_v, -1 means ~x_v;
  the clause is violated iff every literal is false, i.e. bits == pattern with
  pattern = 0 for +1 and 1 for -1; H = number of violated clauses.

Recipe per instance (deterministic: rng = default_rng(SEED_BASE + 1000 n + idx)):

1. Plant x*: Hamming weight uniform in [2, n-2], positions uniform; trivial patterns
   (all-0/1, alternating, half blocks) rejected, as for n = 8.
2. Local rigidity (same rule as n = 8): for every site s one clause on s plus three
   random companions in which only the s-literal is true under x*, so every single
   bit-flip of x* violates >= 1 clause.
3. Base density: draw r ~ U[14/8, 23/8] = U[1.75, 2.875] -- the n = 8 generator's
   clause window (MIN_CLAUSES..MAX_CLAUSES)/8 -- and add uniformly random distinct
   4-clauses compatible with x* (random 4-set, random polarity pattern with the one
   pattern that x* violates rejected) until m_base = round(r n) clauses.
4. Greedy top-up (as n = 8): enumerate every satisfying assignment != x*
   (exact brute force over 2^n, numba), then repeatedly add the compatible 4-clause
   that kills the most remaining solutions (random tie-break) until x* is the unique
   solution. When > GREEDY_SAMPLE solutions remain, the kill counts are computed on a
   uniform random subsample of GREEDY_SAMPLE of them (the set itself is still exact
   and filtered exactly after every added clause).
5. Uniqueness is verified twice: exact brute-force model count == 1 (x* only), and a
   CDCL solver (python-sat, Minisat22) proves F AND (x != x*) UNSAT.

Differences from n = 8, documented: the n = 8 script tried 4000 candidates per
instance and kept the one with the smallest greedy-descent basin; here one candidate
per instance is kept (basin scoring needs the full landscape and is not part of F1's
definition). Final m therefore exceeds m_base by the top-up count, which grows with n
(unique-solution planted 4-SAT needs more constraints at larger n); both are recorded.

Usage::

    PYTHONPATH=. python Hamiltonians/four_sat_scaling.py --n 8,12,16,20,24,28 --workers 8
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations
from pathlib import Path

import numpy as np

try:  # numba makes the 2^28 brute force take seconds instead of minutes
    from numba import njit

    HAVE_NUMBA = True
except Exception:  # pragma: no cover
    HAVE_NUMBA = False

    def njit(*a, **k):  # type: ignore
        def deco(f):
            return f

        return deco if not (a and callable(a[0])) else a[0]


CLAUSE_WIDTH = 4
N_VALUES = (8, 12, 16, 20, 24, 28)
NUM_INSTANCES = 20
SEED_BASE = 20260929
BASE_RATIO_RANGE = (14.0 / 8.0, 23.0 / 8.0)
GREEDY_SAMPLE = 2048
OUTPUT_ROOT = Path(__file__).resolve().parent / "four_sat_scaling"
BLOCK_BITS = 22


# ---------------------------------------------------------------------------
# bit helpers
# ---------------------------------------------------------------------------
def bitstring(bits: np.ndarray) -> str:
    return "".join(str(int(b)) for b in np.asarray(bits).reshape(-1))


def index_from_bits(bits: np.ndarray) -> int:
    idx = 0
    for b in np.asarray(bits, dtype=int).reshape(-1):
        idx = (idx << 1) | int(b)
    return int(idx)


def bits_from_index(idx: int, n: int) -> np.ndarray:
    return np.array([(int(idx) >> (n - 1 - k)) & 1 for k in range(n)], dtype=np.int64)


def trivial_bitstrings(n: int) -> set[str]:
    h, q = n // 2, n // 4
    out = {"0" * n, "1" * n, ("01" * n)[:n], ("10" * n)[:n], "0" * h + "1" * (n - h), "1" * h + "0" * (n - h)}
    if q:
        out.add("0" * q + "1" * (n - 2 * q) + "0" * q)
        out.add("1" * q + "0" * (n - 2 * q) + "1" * q)
    return out


def violating_pattern(polarities: np.ndarray) -> np.ndarray:
    return np.where(np.asarray(polarities, dtype=int) > 0, 0, 1)


def clause_masks(clauses: np.ndarray, polarities: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """(M, V) per clause on MSB-first indices: clause violated iff (x & M) == V."""
    clauses = np.asarray(clauses, dtype=np.int64).reshape(-1, CLAUSE_WIDTH)
    polarities = np.asarray(polarities, dtype=np.int64).reshape(-1, CLAUSE_WIDTH)
    masks = np.zeros(len(clauses), dtype=np.int64)
    vals = np.zeros(len(clauses), dtype=np.int64)
    for c, (vs, ps) in enumerate(zip(clauses, polarities, strict=True)):
        for v, bit in zip(vs, violating_pattern(ps), strict=True):
            b = np.int64(1) << np.int64(n - 1 - int(v))
            masks[c] |= b
            if bit:
                vals[c] |= b
    return masks, vals


def unsat_counts(clauses: np.ndarray, polarities: np.ndarray, n: int) -> np.ndarray:
    """Violated-clause count for all 2^n basis states (MSB-first). Use for n <= ~24."""
    masks, vals = clause_masks(clauses, polarities, n)
    idx = np.arange(1 << n, dtype=np.int64)
    e = np.zeros(1 << n, dtype=np.int16)
    for m_, v_ in zip(masks, vals, strict=True):
        e += (idx & m_) == v_
    return e


def energies_of_indices(idx: np.ndarray, masks: np.ndarray, vals: np.ndarray) -> np.ndarray:
    idx = np.asarray(idx, dtype=np.int64)
    e = np.zeros(idx.shape, dtype=np.int64)
    for m_, v_ in zip(masks, vals, strict=True):
        e += (idx & m_) == v_
    return e


# ---------------------------------------------------------------------------
# exact solution enumeration (brute force, blocked)
# ---------------------------------------------------------------------------
@njit(cache=True)
def _block_solutions(start, size, masks, vals, out):  # pragma: no cover - numba
    cnt = 0
    nc = masks.shape[0]
    for x in range(start, start + size):
        ok = True
        for c in range(nc):
            if (x & masks[c]) == vals[c]:
                ok = False
                break
        if ok:
            out[cnt] = x
            cnt += 1
    return cnt


def enumerate_solutions(clauses: np.ndarray, polarities: np.ndarray, n: int) -> np.ndarray:
    """All satisfying assignments (MSB-first indices), exact brute force over 2^n."""
    masks, vals = clause_masks(clauses, polarities, n)
    # most-constraining first is irrelevant for correctness; keep order
    total = 1 << n
    block = min(total, 1 << BLOCK_BITS)
    buf = np.empty(block, dtype=np.int64)
    found: list[np.ndarray] = []
    if not HAVE_NUMBA:  # numpy fallback
        for start in range(0, total, block):
            idx = np.arange(start, start + block, dtype=np.int64)
            ok = np.ones(block, dtype=bool)
            for m_, v_ in zip(masks, vals, strict=True):
                ok &= (idx & m_) != v_
            found.append(idx[ok])
        return np.concatenate(found)
    for start in range(0, total, block):
        c = _block_solutions(np.int64(start), np.int64(block), masks, vals, buf)
        if c:
            found.append(buf[:c].copy())
    return np.concatenate(found) if found else np.zeros(0, dtype=np.int64)


def sat_solver_unique(clauses: np.ndarray, polarities: np.ndarray, planted: np.ndarray) -> bool:
    """CDCL proof that planted is a model and the only one (python-sat Minisat22)."""
    from pysat.solvers import Minisat22

    cnf = [[int(p) * (int(v) + 1) for v, p in zip(vs, ps, strict=True)] for vs, ps in zip(clauses, polarities, strict=True)]
    lits_planted = [(i + 1) if int(b) else -(i + 1) for i, b in enumerate(planted)]
    with Minisat22(bootstrap_with=cnf) as s:
        if not s.solve(assumptions=lits_planted):
            return False
        s.add_clause([-l for l in lits_planted])
        return not s.solve()


# ---------------------------------------------------------------------------
# construction
# ---------------------------------------------------------------------------
def _polarity_for_literal(bit: int, *, want_true: bool) -> int:
    if want_true:
        return 1 if int(bit) == 1 else -1
    return 1 if int(bit) == 0 else -1


def rigidity_clauses(planted: np.ndarray, rng: np.random.Generator) -> tuple[list, list]:
    """Same rule as four_sat._rigidity_clauses, for any n."""
    n = planted.size
    cl, po = [], []
    others = np.arange(n)
    for site in range(n):
        pool = others[others != site]
        comp = rng.choice(pool, size=CLAUSE_WIDTH - 1, replace=False)
        vs = np.sort(np.concatenate(([site], comp))).astype(np.int64)
        ps = np.array([_polarity_for_literal(int(planted[v]), want_true=(int(v) == site)) for v in vs], dtype=np.int64)
        cl.append(vs)
        po.append(ps)
    return cl, po


def _key(vs, ps) -> tuple:
    return (tuple(int(v) for v in vs), tuple(int(p) for p in ps))


def random_compatible_clause(planted: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    n = planted.size
    vs = np.sort(rng.choice(n, size=CLAUSE_WIDTH, replace=False)).astype(np.int64)
    while True:
        ps = rng.choice(np.array([-1, 1], dtype=np.int64), size=CLAUSE_WIDTH)
        if not np.array_equal(violating_pattern(ps), planted[vs]):
            return vs, ps


def greedy_topup(
    planted: np.ndarray,
    clauses: list,
    polarities: list,
    rng: np.random.Generator,
    sample: int = GREEDY_SAMPLE,
) -> tuple[list, list, int]:
    """Add max-kill compatible clauses until planted is the unique model. Returns n_solutions_before."""
    n = planted.size
    planted_idx = index_from_bits(planted)
    sols = enumerate_solutions(np.stack(clauses), np.stack(polarities), n)
    if planted_idx not in set(sols[: min(sols.size, 1)].tolist()) and not np.any(sols == planted_idx):
        raise RuntimeError("planted assignment is not a model")
    n_before = int(sols.size)
    sols = sols[sols != planted_idx]
    combos = np.array(list(combinations(range(n), CLAUSE_WIDTH)), dtype=np.int64)  # (C, 4)
    shifts = (n - 1 - combos).astype(np.int64)
    weights = np.array([8, 4, 2, 1], dtype=np.int64)
    pcode = (planted[combos] * weights).sum(axis=1)  # pattern of x* on each 4-set
    C = combos.shape[0]
    used = {_key(v, p) for v, p in zip(clauses, polarities, strict=True)}
    while sols.size:
        sub = sols if sols.size <= sample else rng.choice(sols, size=sample, replace=False)
        # codes[k, c] = pattern (MSB = first var) of solution k on 4-set c
        # (chunked over 4-sets to bound memory; result identical to the unchunked count)
        counts = np.zeros((C, 16), dtype=float)
        for c0 in range(0, C, 2048):
            c1 = min(C, c0 + 2048)
            codes = np.zeros((sub.size, c1 - c0), dtype=np.int32)
            for j in range(CLAUSE_WIDTH):
                codes += (((sub[:, None] >> shifts[None, c0:c1, j]) & 1) * weights[j]).astype(np.int32)
            codes += 16 * np.arange(c1 - c0, dtype=np.int32)[None, :]
            counts[c0:c1] = np.bincount(codes.ravel(), minlength=16 * (c1 - c0)).reshape(c1 - c0, 16)
        counts[np.arange(C), pcode] = -1.0  # incompatible with x*
        score = counts + 0.5 * rng.random(counts.shape)
        c, code = np.unravel_index(int(np.argmax(score)), score.shape)
        if counts[c, code] <= 0:
            raise RuntimeError("greedy top-up stalled")
        pat = np.array([(code >> (3 - j)) & 1 for j in range(CLAUSE_WIDTH)], dtype=np.int64)
        vs = combos[c].copy()
        ps = np.where(pat == 0, 1, -1).astype(np.int64)
        k = _key(vs, ps)
        if k in used:  # cannot happen: used clauses kill nothing in sols
            raise RuntimeError("duplicate clause chosen")
        used.add(k)
        clauses.append(vs)
        polarities.append(ps)
        m_, v_ = clause_masks(vs[None], ps[None], n)
        sols = sols[(sols & m_[0]) != v_[0]]
    return clauses, polarities, n_before


def generate_instance(n: int, idx: int, seed_base: int = SEED_BASE) -> dict:
    seed = int(seed_base) + 1000 * int(n) + int(idx)
    rng = np.random.default_rng(seed)
    triv = trivial_bitstrings(n)
    t0 = time.perf_counter()
    while True:
        weight = int(rng.integers(2, n - 1))
        planted = np.zeros(n, dtype=np.int64)
        planted[rng.choice(n, size=weight, replace=False)] = 1
        if bitstring(planted) not in triv:
            break
    cl, po = rigidity_clauses(planted, rng)
    n_rigid = len(cl)
    ratio = float(rng.uniform(*BASE_RATIO_RANGE))
    m_base = max(int(round(ratio * n)), n_rigid)
    used = {_key(v, p) for v, p in zip(cl, po, strict=True)}
    while len(cl) < m_base:
        vs, ps = random_compatible_clause(planted, rng)
        k = _key(vs, ps)
        if k in used:
            continue
        used.add(k)
        cl.append(vs)
        po.append(ps)
    cl, po, n_sol_base = greedy_topup(planted, cl, po, rng)
    clauses = np.stack(cl)
    polarities = np.stack(po)
    # --- verification ---
    sols = enumerate_solutions(clauses, polarities, n)
    if not (sols.size == 1 and int(sols[0]) == index_from_bits(planted)):
        raise RuntimeError("brute-force uniqueness check failed")
    if not sat_solver_unique(clauses, polarities, planted):
        raise RuntimeError("SAT-solver uniqueness check failed")
    masks, vals = clause_masks(clauses, polarities, n)
    pidx = index_from_bits(planted)
    for s in range(n):  # local rigidity
        if energies_of_indices(np.array([pidx ^ (1 << (n - 1 - s))]), masks, vals)[0] < 1:
            raise RuntimeError("not locally rigid")
    m = int(len(clauses))
    return {
        "n": int(n),
        "index": int(idx),
        "seed": seed,
        "clauses": clauses,
        "polarities": polarities,
        "ground_bitstring": bitstring(planted),
        "ground_index": int(pidx),
        "weight": int(planted.sum()),
        "base_ratio_draw": ratio,
        "n_rigidity": int(n_rigid),
        "m_base": int(m_base),
        "n_solutions_base": int(n_sol_base),
        "n_topup": int(m - m_base),
        "m": m,
        "m_over_n": m / n,
        "gen_seconds": time.perf_counter() - t0,
    }


def _save(inst: dict, out_dir: Path) -> dict:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from four_sat import combine_z_terms  # n-agnostic Pauli expansion

    n = inst["n"]
    sites, orders, coefficients, identity = combine_z_terms(inst["clauses"], inst["polarities"])
    name = f"four_sat_n{n:02d}_{inst['index']:03d}.npz"
    np.savez_compressed(
        out_dir / name,
        sites=sites,
        orders=orders,
        coefficients=coefficients,
        identity=identity,
        clauses=inst["clauses"],
        polarities=inst["polarities"],
        num_spins=n,
        clause_width=CLAUSE_WIDTH,
        num_clauses=inst["m"],
        ground_bitstring=inst["ground_bitstring"],
    )
    rec = {k: v for k, v in inst.items() if k not in ("clauses", "polarities")}
    rec.update(
        file=name,
        num_spins=n,
        clause_width=CLAUSE_WIDTH,
        num_clauses=inst["m"],
        n_pauli_terms=int(len(coefficients)),
        identity=float(identity),
        energy_min=0,
        n_ground=1,
        clauses=np.asarray(inst["clauses"]).tolist(),
        polarities=np.asarray(inst["polarities"]).tolist(),
    )
    return rec


def _job(args: tuple[int, int, int]) -> dict:
    n, idx, seed_base = args
    return generate_instance(n, idx, seed_base)


def load_instance(path: Path | str) -> dict:
    d = np.load(path, allow_pickle=False)
    return {
        "clauses": np.asarray(d["clauses"], dtype=np.int64),
        "polarities": np.asarray(d["polarities"], dtype=np.int64),
        "n": int(d["num_spins"]),
        "ground_bitstring": str(d["ground_bitstring"]),
        "file": Path(path).name,
    }


def instance_paths(n: int, root: Path = OUTPUT_ROOT) -> list[Path]:
    return sorted((root / f"n{n:02d}").glob(f"four_sat_n{n:02d}_[0-9][0-9][0-9].npz"))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=str, default=",".join(map(str, N_VALUES)))
    p.add_argument("--instances", type=int, default=NUM_INSTANCES)
    p.add_argument("--seed-base", type=int, default=SEED_BASE)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--out", type=str, default=str(OUTPUT_ROOT))
    a = p.parse_args(argv)
    ns = [int(x) for x in a.n.split(",") if x.strip()]
    root = Path(a.out)
    jobs = [(n, i, a.seed_base) for n in ns for i in range(a.instances)]
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        results = list(ex.map(_job, jobs))
    stats = {}
    for n in ns:
        out_dir = root / f"n{n:02d}"
        out_dir.mkdir(parents=True, exist_ok=True)
        recs = [_save(r, out_dir) for r in results if r["n"] == n]
        recs.sort(key=lambda r: r["index"])
        (out_dir / f"four_sat_n{n:02d}_manifest.json").write_text(json.dumps(recs, indent=1) + "\n")
        ms = np.array([r["m"] for r in recs])
        st = {
            "n": n,
            "instances": len(recs),
            "m_mean": float(ms.mean()), "m_min": int(ms.min()), "m_max": int(ms.max()),
            "m_over_n_mean": float(ms.mean() / n), "m_over_n_min": float(ms.min() / n), "m_over_n_max": float(ms.max() / n),
            "m_base_mean": float(np.mean([r["m_base"] for r in recs])),
            "n_topup_mean": float(np.mean([r["n_topup"] for r in recs])),
            "n_solutions_base_median": float(np.median([r["n_solutions_base"] for r in recs])),
            "gen_seconds_mean": float(np.mean([r["gen_seconds"] for r in recs])),
        }
        stats[str(n)] = st
        print(json.dumps(st), flush=True)
    (root / "generation_stats.json").write_text(json.dumps({"seed_base": a.seed_base, "base_ratio_range": BASE_RATIO_RANGE, "greedy_sample": GREEDY_SAMPLE, "per_n": stats}, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
