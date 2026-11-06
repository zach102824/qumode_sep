"""Clause-aware cavity packing (CAP) — a compilation, not a new ansatz.

Zero new parameters. A permutation π of the n variables puts original
variable π(i) into circuit slot i. Cavity j is the triple of slots
{3j, 3j+1, 3j+2}, i.e. original variables {π(3j), π(3j+1), π(3j+2)}.

Co-occurrence w_ab = number of clauses that contain both a and b.
π_max maximises Σ_j Σ_{a<b in cavity j} w_ab; π_min minimises it.
Search is deterministic greedy assignment plus pairwise-swap local
search (no RNG). Identity and fixed random permutations are controls.

The circuit is unchanged and is applied to f'(x) = f(π^{-1} x), with
the array convention (π x)_i = x_{π(i)} and var0 = MSB, matching
``sat_gen.fast_energies``. Success E'(x) = 0 is the original formula
satisfied under that relabeling (un-permuted E).
"""
from __future__ import annotations

import numpy as np

# Controls. Greedy / local search do not use this. Random perms do.
CAP_RANDOM_SEED = 20261104
N_RANDOM = 8
CAVITY = 3


def cooccurrence(cls_v: np.ndarray, n: int) -> np.ndarray:
    """Pair counts. cls_v is (m, 4) or (B, m, 4) variable indices.

    Returns w with shape (n, n) or (B, n, n). w[a, b] = w[b, a], zero diagonal.
    """
    cls_v = np.asarray(cls_v)
    single = cls_v.ndim == 2
    if single:
        cls_v = cls_v[None, ...]
    if cls_v.ndim != 3 or cls_v.shape[-1] != 4:
        raise ValueError(f"cls_v shape {cls_v.shape}, expected (m, 4) or (B, m, 4)")
    B = int(cls_v.shape[0])
    w = np.zeros((B, n, n), dtype=np.int32)
    rows = np.arange(B, dtype=np.int32)[:, None]
    vv = cls_v.astype(np.int32, copy=False)
    for i in range(4):
        for j in range(i + 1, 4):
            a = vv[:, :, i]
            b = vv[:, :, j]
            np.add.at(w, (rows, a, b), 1)
            np.add.at(w, (rows, b, a), 1)
    return w[0] if single else w


def packing_score(w: np.ndarray, pi: np.ndarray) -> int:
    """Σ over cavities of the three pair-weights. w is (n, n)."""
    pi = np.asarray(pi, dtype=np.int64)
    n = int(pi.shape[0])
    if n % CAVITY != 0:
        raise ValueError(f"n={n} not divisible by {CAVITY}")
    score = 0
    for c in range(n // CAVITY):
        a, b, d = (int(pi[3 * c + t]) for t in range(3))
        score += int(w[a, b]) + int(w[a, d]) + int(w[b, d])
    return int(score)


def _greedy(w: np.ndarray, maximize: bool) -> np.ndarray:
    """Deterministic greedy triples.

    While variables remain, take the unused pair extremising w_ab
    (tie: smaller a, then smaller b), then the third variable extremising
    w_av + w_bv (tie: smaller v). Emit the triple in that order (a < b).
    """
    n = int(w.shape[0])
    remaining = list(range(n))
    pi: list[int] = []
    while remaining:
        best_pair = None
        best_key = None
        for ia, a in enumerate(remaining):
            for b in remaining[ia + 1 :]:
                weight = int(w[a, b])
                key = (weight if maximize else -weight, -a, -b)
                if best_key is None or key > best_key:
                    best_key = key
                    best_pair = (a, b)
        assert best_pair is not None
        a, b = best_pair
        best_v = None
        best_vk = None
        for v in remaining:
            if v == a or v == b:
                continue
            weight = int(w[a, v]) + int(w[b, v])
            key = (weight if maximize else -weight, -v)
            if best_vk is None or key > best_vk:
                best_vk = key
                best_v = v
        assert best_v is not None
        pi.extend((a, b, best_v))
        drop = {a, b, best_v}
        remaining = [v for v in remaining if v not in drop]
    return np.asarray(pi, dtype=np.int64)


def _swap_delta(w: np.ndarray, pi: np.ndarray, i: int, j: int) -> int:
    """Change in packing score from swapping the variables in slots i and j."""
    vi = int(pi[i])
    vj = int(pi[j])
    ci = i // CAVITY
    cj = j // CAVITY
    delta = 0
    for t in range(CAVITY):
        s = 3 * ci + t
        if s == i:
            continue
        o = int(pi[s])
        delta += int(w[vj, o]) - int(w[vi, o])
    for t in range(CAVITY):
        s = 3 * cj + t
        if s == j:
            continue
        o = int(pi[s])
        delta += int(w[vi, o]) - int(w[vj, o])
    return int(delta)


def _local_search(w: np.ndarray, pi: np.ndarray, maximize: bool) -> np.ndarray:
    """Best-improvement pairwise swaps across cavities until a local optimum.

    Ties keep the smallest (i, j). Swaps inside one cavity do not change the
    set score and are skipped. Each accepted swap strictly improves the score.
    """
    pi = np.array(pi, dtype=np.int64, copy=True)
    n = int(pi.shape[0])
    while True:
        best = None  # (improve, i, j)
        for i in range(n):
            ci = i // CAVITY
            for j in range(i + 1, n):
                if j // CAVITY == ci:
                    continue
                delta = _swap_delta(w, pi, i, j)
                improve = delta if maximize else -delta
                if improve <= 0:
                    continue
                if best is None or improve > best[0]:
                    best = (improve, i, j)
        if best is None:
            return pi
        i, j = best[1], best[2]
        pi[i], pi[j] = pi[j], pi[i]


def pack_permutation(w: np.ndarray, maximize: bool) -> np.ndarray:
    """π from greedy assignment followed by pairwise-swap local search."""
    if w.ndim != 2 or w.shape[0] != w.shape[1]:
        raise ValueError(f"w shape {w.shape}")
    n = int(w.shape[0])
    if n % CAVITY != 0:
        raise ValueError(f"n={n} not divisible by {CAVITY}")
    pi = _local_search(w, _greedy(w, maximize), maximize)
    if len(set(int(v) for v in pi.tolist())) != n:
        raise RuntimeError("packer did not return a permutation")
    return pi


def pack_batch(cls_v: np.ndarray, n: int, maximize: bool) -> np.ndarray:
    """Per-instance π, shape (B, n). cls_v is (B, m, 4)."""
    w = cooccurrence(cls_v, n)
    if w.ndim == 2:
        w = w[None, ...]
    out = np.empty((w.shape[0], n), dtype=np.int64)
    for b in range(w.shape[0]):
        out[b] = pack_permutation(w[b], maximize)
    return out


def identity_perm(n: int) -> np.ndarray:
    return np.arange(n, dtype=np.int64)


def random_permutations(n: int, k: int = N_RANDOM, seed: int = CAP_RANDOM_SEED) -> list[np.ndarray]:
    """k fixed permutations of 0..n-1. Stream is PCG64(seed + n), not used by π_max/π_min."""
    rng = np.random.Generator(np.random.PCG64(int(seed) + int(n)))
    return [rng.permutation(n).astype(np.int64) for _ in range(int(k))]


def permute_index(n: int, pi: np.ndarray) -> np.ndarray:
    """y[x] such that E'(x) = E(y). π[slot] = original variable in that slot.

    Bit slot of x is written onto original variable π(slot). var0 = MSB, so
    variable v lives at shift (n - 1 - v). This is f'(x) = f(π^{-1} x) under
    (π x)_i = x_{π(i)}.
    """
    pi = np.asarray(pi, dtype=np.int64)
    if int(pi.shape[0]) != n:
        raise ValueError(f"pi length {pi.shape[0]} != n={n}")
    N = 1 << n
    x = np.arange(N, dtype=np.int64)
    y = np.zeros(N, dtype=np.int64)
    for slot in range(n):
        bit = (x >> (n - 1 - slot)) & np.int64(1)
        y |= bit << np.int64(n - 1 - int(pi[slot]))
    return y


def permute_energy(E: np.ndarray, pi: np.ndarray) -> np.ndarray:
    """Apply one π to a single table (2^n,) or a batch (B, 2^n) with one shared π."""
    E = np.asarray(E)
    n = int(round(np.log2(E.shape[-1])))
    y = permute_index(n, pi)
    return E[..., y]


def permute_energy_instances(E: np.ndarray, pis: np.ndarray) -> np.ndarray:
    """E (B, 2^n), pis (B, n) — a (possibly different) π per instance."""
    E = np.asarray(E)
    pis = np.asarray(pis, dtype=np.int64)
    B, N = E.shape
    n = int(round(np.log2(N)))
    if pis.shape != (B, n):
        raise ValueError(f"pis {pis.shape} != {(B, n)}")
    x = np.arange(N, dtype=np.int64)
    y = np.zeros((B, N), dtype=np.int64)
    for slot in range(n):
        bit = ((x >> (n - 1 - slot)) & np.int64(1)).astype(np.int64)
        shift = (n - 1 - pis[:, slot]).astype(np.int64)
        y |= bit[None, :] << shift[:, None]
    return E[np.arange(B)[:, None], y]


def cavities(pi: np.ndarray) -> list[tuple[int, int, int]]:
    pi = np.asarray(pi, dtype=np.int64)
    return [tuple(int(v) for v in pi[3 * c : 3 * c + 3]) for c in range(len(pi) // 3)]
