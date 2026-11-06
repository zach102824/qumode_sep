"""Steiner quadruple systems S(3,4,N), constructed and checked in this file.

A block list is an SQS(N) when every 3-subset of an N-set sits in exactly one
4-block. That forces
    b = N(N-1)(N-2)/24 blocks,
    r = (N-1)(N-2)/6 blocks through each point.
Existence for every N≡2 or 4 (mod 6) is Hanani's theorem; this module does
not rely on it. Each design below is either an explicit affine geometry or
the first solution of a fully specified exact-cover search, and verify_sqs
rejects anything that is not an SQS.

Designs used
------------
- N=8:  affine geometry AG(3,2), the 2-flats of F_2^3.
- N=16: affine geometry AG(4,2), the 2-flats of F_2^4.
- N=10 and N=14: smallest-triple exact cover (see exact_cover_sqs). This is
  a reproducible algorithm, not a claim that the output is the lex-least
  SQS under a different branching rule, and not an identification of the
  N=14 isomorphism type among the four that exist in the literature.
"""
from __future__ import annotations

import itertools
import math
from functools import lru_cache

import numpy as np


def sqs_parameters(n: int) -> dict:
    if n < 4 or (n % 6) not in (2, 4):
        raise ValueError(f"S(3,4,N) parameters are integral only for N≡2 or 4 (mod 6); got {n}")
    b = n * (n - 1) * (n - 2) // 24
    r = (n - 1) * (n - 2) // 6
    lam2 = (n - 2) // 2
    return {"N": n, "b": b, "r": r, "lambda2": lam2}


def verify_sqs(n: int, blocks: list[tuple[int, ...]]) -> dict:
    """Exhaustive triple count. Raises if the list is not an SQS(N)."""
    params = sqs_parameters(n)
    if len(blocks) != params["b"]:
        raise AssertionError(f"N={n}: {len(blocks)} blocks, expected {params['b']}")
    seen = {}
    degree = [0] * n
    pair_deg = {}
    for bi, block in enumerate(blocks):
        if len(block) != 4 or len(set(block)) != 4:
            raise AssertionError(f"bad block {block}")
        if any(x < 0 or x >= n for x in block):
            raise AssertionError(f"block out of range {block}")
        b = tuple(sorted(block))
        if blocks[bi] != b:
            raise AssertionError("blocks must be stored sorted")
        for t in itertools.combinations(b, 3):
            if t in seen:
                raise AssertionError(f"triple {t} in {seen[t]} and {b}")
            seen[t] = b
        for x in b:
            degree[x] += 1
        for p in itertools.combinations(b, 2):
            pair_deg[p] = pair_deg.get(p, 0) + 1
    ntrip = math.comb(n, 3)
    if len(seen) != ntrip:
        raise AssertionError(f"covered {len(seen)} triples, expected {ntrip}")
    if degree != [params["r"]] * n:
        raise AssertionError(f"degree {degree}, expected r={params['r']}")
    if any(v != params["lambda2"] for v in pair_deg.values()) or len(pair_deg) != math.comb(n, 2):
        raise AssertionError("pair replication is not λ2")
    return params


def _affine_flats(dim: int) -> list[tuple[int, ...]]:
    """4-point affine 2-flats of F_2^dim, points packed as integers 0..2^dim-1.

    A 2-flat is a coset of a 2-dimensional linear subspace. Three distinct
    points of AG(dim, 2) lie on exactly one such flat because every line has
    two points, so the verification below is the proof for these instances,
    not a citation.
    """
    n = 1 << dim
    # 2-dimensional subspaces: spans of two independent vectors.
    subs = []
    seen = set()
    for a, b in itertools.combinations(range(1, n), 2):
        if a & b == a or a & b == b:
            # one contains the other only if they are equal bits; independence
            # over F_2 fails iff one is zero (excluded) or they are equal.
            pass
        # linearly dependent iff a==b (excluded) or ... two nonzero are
        # dependent iff they are scalar multiples. The only scalar is 1, so
        # any two distinct nonzero vectors are independent. WRONG in higher
        # dim: (a, a) only. In F_2, two vectors are dependent iff one is 0
        # or they are equal. Yes, any two distinct nonzero are independent.
        # But span{a,b}={0,a,b,a^b} has size 4 iff a, b, a^b distinct iff
        # a != b and a != 0 and b != 0 and a^b != 0 which is a!=b.
        # a^b == 0 iff a==b. So EVERY pair of distinct nonzero vectors spans
        # a 4-set. Many pairs give the same subspace.
        s = frozenset((0, a, b, a ^ b))
        if len(s) != 4 or s in seen:
            continue
        seen.add(s)
        subs.append(s)
    blocks = []
    block_seen = set()
    for sub in seen:
        pts = list(sub)
        # cosets: translate by every point, dedup
        for origin in range(n):
            flat = frozenset(origin ^ p for p in pts)
            if flat in block_seen:
                continue
            block_seen.add(flat)
            blocks.append(tuple(sorted(flat)))
    blocks.sort()
    return blocks


def ag_sqs(dim: int) -> list[tuple[int, ...]]:
    """SQS(2^dim) from AG(dim, 2). dim>=2. dim=3 → N=8, dim=4 → N=16."""
    blocks = _affine_flats(dim)
    verify_sqs(1 << dim, blocks)
    return blocks


def exact_cover_sqs(n: int) -> list[tuple[int, ...]]:
    """First SQS found by: cover the smallest uncovered triple, try blocks
    in increasing colex/lex order, forward-check that every open triple still
    has a free block. Deterministic. Not a covering-radius computation.
    """
    params = sqs_parameters(n)
    triples = list(itertools.combinations(range(n), 3))
    t_index = {t: i for i, t in enumerate(triples)}
    blocks = list(itertools.combinations(range(n), 4))
    b_trips = []
    trip_blocks: list[list[int]] = [[] for _ in triples]
    for bi, block in enumerate(blocks):
        ts = [t_index[t] for t in itertools.combinations(block, 3)]
        b_trips.append(ts)
        for t in ts:
            trip_blocks[t].append(bi)
    covered = [False] * len(triples)
    chosen: list[int] = []

    def available(bi: int) -> bool:
        return all(not covered[t] for t in b_trips[bi])

    def place(bi: int) -> None:
        for t in b_trips[bi]:
            covered[t] = True
        chosen.append(bi)

    def undo(bi: int) -> None:
        chosen.pop()
        for t in b_trips[bi]:
            covered[t] = False

    def dead() -> bool:
        for t, is_on in enumerate(covered):
            if is_on:
                continue
            if not any(available(bi) for bi in trip_blocks[t]):
                return True
        return False

    def rec() -> bool:
        t = next((i for i, on in enumerate(covered) if not on), None)
        if t is None:
            return True
        for bi in trip_blocks[t]:
            if not available(bi):
                continue
            place(bi)
            if not dead() and rec():
                return True
            undo(bi)
        return False

    if not rec():
        raise RuntimeError(f"exact cover found no SQS({n})")
    out = [blocks[i] for i in chosen]
    out.sort()
    verify_sqs(n, out)
    if len(out) != params["b"]:
        raise AssertionError("parameter mismatch after search")
    return out


@lru_cache(maxsize=8)
def design_blocks(n: int) -> tuple[tuple[int, ...], ...]:
    """Declared design for this round. Cached. See module docstring."""
    if n == 8:
        blocks = ag_sqs(3)
    elif n == 16:
        blocks = ag_sqs(4)
    elif n in (10, 14):
        blocks = exact_cover_sqs(n)
    else:
        raise ValueError(f"no declared SQS construction for N={n}")
    return tuple(blocks)


def design_name(n: int) -> str:
    return {
        8: "AG(3,2) 2-flats on F_2^3, points 0..7",
        10: "smallest-uncovered-triple exact cover on points 0..9; isomorphism uniqueness NOT certified",
        14: "smallest-uncovered-triple exact cover on points 0..13; which of the four SQS(14) isomorphism types this is was NOT identified",
        16: "AG(4,2) 2-flats on F_2^4, points 0..15",
    }[n]


def incidence_matrix_gf2(n: int, blocks: list[tuple[int, ...]]) -> np.ndarray:
    """b × N matrix over {0,1}, row = characteristic vector of a block."""
    B = np.zeros((len(blocks), n), dtype=np.uint8)
    for i, block in enumerate(blocks):
        for x in block:
            B[i, x] = 1
    return B


def gf2_rank(rows: np.ndarray) -> int:
    """Rank over GF(2) of a uint8 matrix, rows are the generators."""
    A = rows.copy().astype(np.uint8)
    if A.size == 0:
        return 0
    m, n = A.shape
    rank = 0
    col = 0
    for r in range(m):
        while col < n:
            piv = None
            for i in range(r, m):
                if A[i, col]:
                    piv = i
                    break
            if piv is None:
                col += 1
                continue
            if piv != r:
                A[[r, piv]] = A[[piv, r]]
            for i in range(m):
                if i != r and A[i, col]:
                    A[i] ^= A[r]
            rank += 1
            col += 1
            break
    return rank


def code_bitmasks(n: int, blocks: list[tuple[int, ...]]) -> tuple[int, int, np.ndarray]:
    """Image of y |-> (sum_{i in b} y_i)_b as integer bitmasks.

    Returns (rank, code as uint64/object array, mask width b).
    b must be <= 64. True for N<=16 (b=140>64). For b>64 return None code
    and only the rank... N=16 has 140 blocks. Handle generally with bytes.

    For b<=63 we pack into uint64. For larger b, pack into a byte view
    of several uint64 limbs. N=16 needs 3 limbs (140 bits).
    """
    b = len(blocks)
    B = incidence_matrix_gf2(n, blocks)
    rank = gf2_rank(B)
    # Evaluate B y for all y by Gray-code incremental XOR of columns.
    nlimbs = (b + 63) // 64
    code = np.zeros((1 << n, nlimbs), dtype=np.uint64)
    # column bitmasks
    cols = np.zeros((n, nlimbs), dtype=np.uint64)
    for bi, block in enumerate(blocks):
        limb, bit = divmod(bi, 64)
        for x in block:
            cols[x, limb] |= np.uint64(1) << np.uint64(bit)
    acc = np.zeros(nlimbs, dtype=np.uint64)
    code[0] = acc
    for step in range(1, 1 << n):
        # Binary-reflected Gray code: the state after `step` flips is
        # gray = step ^ (step >> 1), and the flipped bit is the trailing
        # one of `step`. Store B*y at index y, not at the step counter.
        pivot = (step & -step).bit_length() - 1
        acc = acc ^ cols[pivot]
        gray = step ^ (step >> 1)
        code[gray] = acc
    return rank, code, b
