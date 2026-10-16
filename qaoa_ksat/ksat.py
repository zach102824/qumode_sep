"""Random k-SAT ensemble matching Definition 1 of PRX Quantum 5, 030348.

Each clause is k literals drawn independently with replacement from
{x_0, ¬x_0, …, x_{n-1}, ¬x_{n-1}}. The number of clauses is Poisson(r n).
The cost of an assignment is the number of unsatisfied clauses.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Table I of the paper.
THRESHOLD_R = {
    2: 1.0,
    4: 9.93,
    8: 176.54,
    10: 708.92,
    16: 45425.2,
}


@dataclass(frozen=True)
class Clause:
    """One OR-clause.

    ``variables[j]`` is in ``0 .. n-1``. ``signs[j] == 1`` means the positive
    literal ``x_v``; ``signs[j] == 0`` means ``¬x_v``.
    """

    variables: np.ndarray
    signs: np.ndarray

    def unsat_constraints(self) -> dict[int, int] | None:
        """Bit values that make every literal false, or None if the clause is a tautology."""
        req: dict[int, int] = {}
        for v, s in zip(self.variables.tolist(), self.signs.tolist()):
            need = 0 if int(s) == 1 else 1
            prev = req.get(int(v))
            if prev is not None and prev != need:
                return None
            req[int(v)] = need
        return req


@dataclass
class KSATInstance:
    n: int
    k: int
    r: float
    clauses: tuple[Clause, ...]
    seed: int | None = None

    @property
    def m(self) -> int:
        return len(self.clauses)


def random_ksat(
    n: int,
    k: int,
    r: float | None = None,
    *,
    rng: np.random.Generator | None = None,
    seed: int | None = None,
    m: int | None = None,
) -> KSATInstance:
    """Sample one instance from CNF(n, k, r). If ``m`` is given, skip the Poisson draw."""
    if r is None:
        if k not in THRESHOLD_R:
            raise ValueError(f"no default threshold for k={k}; pass r explicitly")
        r = float(THRESHOLD_R[k])
    rng = rng or np.random.default_rng(seed)
    if m is None:
        m = int(rng.poisson(float(r) * int(n)))
    clauses = tuple(_random_clause(int(n), int(k), rng) for _ in range(int(m)))
    return KSATInstance(n=int(n), k=int(k), r=float(r), clauses=clauses, seed=seed)


def _random_clause(n: int, k: int, rng: np.random.Generator) -> Clause:
    # Literal pool is ordered (x_0, ¬x_0, x_1, ¬x_1, …). Independent draws.
    lits = rng.integers(0, 2 * n, size=int(k), dtype=np.int32)
    variables = lits // 2
    signs = 1 - (lits % 2)  # even index → positive x_v
    return Clause(variables=variables.astype(np.int32), signs=signs.astype(np.int8))


def _unsat_indices(n: int, cons: dict[int, int]) -> np.ndarray:
    """Computational-basis indices (LSB = variable 0) that violate a clause."""
    free = [v for v in range(n) if v not in cons]
    base = 0
    for v, b in cons.items():
        if b:
            base |= 1 << v
    nfree = len(free)
    if nfree == 0:
        return np.array([base], dtype=np.int64)
    out = np.full(1 << nfree, base, dtype=np.int64)
    ar = np.arange(1 << nfree, dtype=np.int64)
    for i, v in enumerate(free):
        out |= ((ar >> i) & 1) << v
    return out


def energy_table(inst: KSATInstance) -> np.ndarray:
    """Integer unsatisfied-clause count for every bitstring, LSB = x_0."""
    n = int(inst.n)
    energy = np.zeros(1 << n, dtype=np.int32)
    for clause in inst.clauses:
        cons = clause.unsat_constraints()
        if cons is None:
            continue
        energy[_unsat_indices(n, cons)] += 1
    return energy


def n_satisfying(energy: np.ndarray) -> int:
    return int(np.count_nonzero(energy == 0))


def is_satisfiable(energy: np.ndarray) -> bool:
    return bool(energy.size) and int(energy.min()) == 0


def clause_unsatisfied_by(clause: Clause, bits: np.ndarray) -> bool:
    """Whether ``bits`` (length n, bit i = x_i) falsifies the clause."""
    for v, s in zip(clause.variables.tolist(), clause.signs.tolist()):
        bit = int(bits[int(v)])
        lit_true = bit == 1 if int(s) == 1 else bit == 0
        if lit_true:
            return False
    return True


def energy_of_index(inst: KSATInstance, index: int) -> int:
    n = int(inst.n)
    bits = np.array([(int(index) >> v) & 1 for v in range(n)], dtype=int)
    return sum(clause_unsatisfied_by(c, bits) for c in inst.clauses)


def sat_instance_from_seed(n: int, k: int, r: float, seed: int, max_tries: int = 2000) -> tuple[KSATInstance, np.ndarray]:
    """Replay ``evaluate.eval_one`` sampling: same RNG, first SAT formula."""
    rng = np.random.default_rng(int(seed))
    for _ in range(int(max_tries)):
        inst = random_ksat(int(n), int(k), float(r), rng=rng, seed=None)
        energy = energy_table(inst)
        if is_satisfiable(energy):
            return inst, energy
    raise RuntimeError(f"no SAT instance after {max_tries} draws (n={n} seed={seed})")


def instance_to_dict(inst: KSATInstance, energy: np.ndarray | None = None, extra: dict | None = None) -> dict:
    sat_idx = None
    n_sat = None
    if energy is not None:
        zeros = np.flatnonzero(energy == 0)
        n_sat = int(zeros.size)
        sat_idx = int(zeros[0]) if zeros.size else None
    out = {
        "n": int(inst.n),
        "k": int(inst.k),
        "r": float(inst.r),
        "m": int(inst.m),
        "seed": inst.seed,
        "n_sat": n_sat,
        "sat_index": sat_idx,  # LSB = x_0
        "clauses": [
            {"vars": c.variables.astype(int).tolist(), "signs": c.signs.astype(int).tolist()}
            for c in inst.clauses
        ],
    }
    if extra:
        out.update(extra)
    return out


def instance_from_dict(d: dict) -> KSATInstance:
    clauses = tuple(
        Clause(
            variables=np.asarray(c["vars"], dtype=np.int32),
            signs=np.asarray(c["signs"], dtype=np.int8),
        )
        for c in d["clauses"]
    )
    return KSATInstance(
        n=int(d["n"]),
        k=int(d["k"]),
        r=float(d["r"]),
        clauses=clauses,
        seed=d.get("seed"),
    )


def clauses_as_arrays(inst: KSATInstance) -> tuple[np.ndarray, np.ndarray]:
    """(m, k) variable indices and signs, padded with the last var if a clause is short."""
    k = int(inst.k)
    m = int(inst.m)
    cv = np.zeros((m, k), dtype=np.int32)
    cs = np.zeros((m, k), dtype=np.int8)
    for i, c in enumerate(inst.clauses):
        L = int(c.variables.size)
        if L == 0:
            continue
        take = min(L, k)
        cv[i, :take] = c.variables[:take]
        cs[i, :take] = c.signs[:take]
        if take < k:
            cv[i, take:] = c.variables[take - 1]
            cs[i, take:] = c.signs[take - 1]
    return cv, cs
