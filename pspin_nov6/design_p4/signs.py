"""Sign patterns on a fixed SQS, with certified frustration.

Classical cost. Z_i = (-1)^{y_i}, J_b = (-1)^{ε_b},
    H_C = - ∑_b J_b ∏_{i in b} Z_i.
A block is unsatisfied when (B y + ε)_b = 1. If u is that weight,
    E(y) = -b + 2 u(y).
u_min = min_y u(y) is the frustration. It is constant on cosets of
Im(B) and is computed by enumerating all 2^N configurations, so it is
exact for every pattern we keep (N≤16).

Gauge. ε and ε + B y0 define Hamiltonians conjugated by ∏ X_i^{y0_i}.
That conjugation fixes |+> and ∑ X, so the QAOA family is identical.
We gauge-fix by choosing y0 among the classical minimizers so that y=0
is a minimizer (ε ← ε + B y0), then record y0.

Covering radius. R = max_ε u_min(ε) = covering radius of Im(B) in F_2^b.
Exact enumeration of cosets is possible for SQS(8) (2^{14-rank} = 1024).
It is not attempted for N≥10. No pattern with N≥10 is called optimal.

Structured target rule, set to follow the small-frustration preference
rather than the covering radius:
candidates are uniform, every point-star, every single-block flip, every
pair-star, and, on affine geometries, the off-origin signing.
Among candidates with u_min≥1, minimize u_min, then maximize the exact
automorphism stabilizer, then take the lexicographically smallest canonical
coset representative. Covering-radius signs are computed only for N=8 and
are reported separately; they are not this target.
"""
from __future__ import annotations

import itertools
from collections import defaultdict

import numpy as np

from sqs import code_bitmasks, design_blocks, incidence_matrix_gf2, sqs_parameters


def _pop_limbs(v: np.ndarray) -> int:
    return int(sum(int(x).bit_count() for x in v))


def _xor(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.bitwise_xor(a, b)


def landscape(code: np.ndarray, eps: np.ndarray) -> dict:
    """Exact classical landscape of one sign pattern. code[y] = B y."""
    nconfig = code.shape[0]
    b_bits = None
    weights = np.empty(nconfig, dtype=np.int16)
    for y in range(nconfig):
        weights[y] = _pop_limbs(_xor(code[y], eps))
    u_min = int(weights.min())
    mins = np.flatnonzero(weights == u_min)
    # mean pairwise spin agreement
    if len(mins) <= 1:
        mean_agree = 1.0
    else:
        # sample all pairs if few, else 2000 deterministic pairs
        n = int(round(np.log2(nconfig)))
        acc = 0
        cnt = 0
        mlist = mins.tolist()
        if len(mlist) <= 40:
            pairs = itertools.combinations(mlist, 2)
        else:
            rng = np.random.default_rng(0)
            pairs = (tuple(rng.choice(mins, size=2, replace=False)) for _ in range(2000))
        for a, c in pairs:
            # agreement = n - hamming(a^c)
            ham = int(a ^ c).bit_count()
            acc += n - ham
            cnt += 1
        mean_agree = acc / cnt / n
    return {
        "u_min": u_min,
        "degeneracy": int(len(mins)),
        "minimizers": mins.astype(np.int64),
        "mean_agreement": float(mean_agree),
        "u_max": int(weights.max()),
        "weights": weights,
    }


def gauge_fix(code: np.ndarray, eps: np.ndarray) -> tuple[np.ndarray, int, dict]:
    """Return (ε' , y0, landscape of ε') with y=0 a classical minimizer."""
    raw = landscape(code, eps)
    y0 = int(raw["minimizers"][0])
    eps2 = _xor(eps, code[y0])
    land = landscape(code, eps2)
    if int(land["weights"][0]) != land["u_min"]:
        raise RuntimeError("gauge fix failed")
    return eps2, y0, land


def canonical_coset_rep(code_unique: np.ndarray, eps: np.ndarray) -> bytes:
    """Lex-smallest bitmask in the coset eps+C, as bytes (limb 0 first)."""
    best = None
    for i in range(code_unique.shape[0]):
        rep = _xor(eps, code_unique[i])
        key = rep.tobytes()
        if best is None or key < best[0]:
            best = (key, rep.copy())
    return best[0]


def unique_code(code: np.ndarray) -> np.ndarray:
    # lex-unique rows
    order = np.lexsort(code.T[::-1])
    sorted_code = code[order]
    if len(sorted_code) == 0:
        return sorted_code
    keep = np.empty(len(sorted_code), dtype=bool)
    keep[0] = True
    keep[1:] = np.any(sorted_code[1:] != sorted_code[:-1], axis=1)
    return sorted_code[keep]


def covering_radius_exact(code: np.ndarray, b: int) -> dict | None:
    """Exhaustive covering radius. Only when b≤16 (2^b ≤ 65536)."""
    if b > 16:
        return None
    uniq = unique_code(code)
    nlimbs = code.shape[1]
    if nlimbs != 1:
        raise RuntimeError("b≤16 fits in one limb")
    best_R = -1
    # canonical rep -> (u_min, an eps achieving it)
    winners = []
    # iterate all ε
    C = uniq[:, 0]
    for e in range(1 << b):
        # min weight of e XOR c
        # vectorized
        xs = np.bitwise_xor(C, np.uint64(e))
        # popcount
        w = np.bitwise_count(xs)
        u = int(w.min())
        if u > best_R:
            best_R = u
            winners = [(e, u)]
        elif u == best_R:
            winners.append((e, u))
    # reduce winners to distinct cosets via canonical rep
    cosets = {}
    for e, u in winners:
        eps = np.array([np.uint64(e)], dtype=np.uint64)
        key = canonical_coset_rep(uniq, eps)
        if key not in cosets:
            # store lex-smallest raw integer in the coset, computed from key
            cosets[key] = u
    return {
        "R": int(best_R),
        "n_codewords": int(len(uniq)),
        "n_cosets_total": int((1 << b) // len(uniq)),
        "n_cosets_at_R": len(cosets),
        "certified": True,
        "method": f"exhaustive enumeration of all 2^{b} sign vectors",
    }


def point_star_eps(n: int, blocks, point: int, nlimbs: int) -> np.ndarray:
    eps = np.zeros(nlimbs, dtype=np.uint64)
    for bi, block in enumerate(blocks):
        if point in block:
            limb, bit = divmod(bi, 64)
            eps[limb] |= np.uint64(1) << np.uint64(bit)
    return eps


def single_block_eps(bi: int, nlimbs: int) -> np.ndarray:
    eps = np.zeros(nlimbs, dtype=np.uint64)
    limb, bit = divmod(bi, 64)
    eps[limb] |= np.uint64(1) << np.uint64(bit)
    return eps


def pair_star_eps(blocks, pair: tuple[int, int], nlimbs: int) -> np.ndarray:
    """ε_b = 1 iff both points of the pair lie in the block."""
    a, c = pair
    eps = np.zeros(nlimbs, dtype=np.uint64)
    for bi, block in enumerate(blocks):
        if a in block and c in block:
            limb, bit = divmod(bi, 64)
            eps[limb] |= np.uint64(1) << np.uint64(bit)
    return eps


def origin_flat_eps(blocks, nlimbs: int) -> np.ndarray:
    """ε_b = 1 iff the flat does not contain point 0. Affine designs only."""
    eps = np.zeros(nlimbs, dtype=np.uint64)
    for bi, block in enumerate(blocks):
        if 0 not in block:
            limb, bit = divmod(bi, 64)
            eps[limb] |= np.uint64(1) << np.uint64(bit)
    return eps


def design_automorphisms(n: int, blocks, node_cap: int = 2_000_000) -> dict:
    """Exact Aut of the (unsigned) design by backtrack, or a certified partial
    result if node_cap is hit. Each returned tuple is a verified permutation
    of points sending blocks to blocks.
    """
    bsets = [frozenset(b) for b in blocks]
    blockset = set(bsets)
    # triple -> the unique block containing it, as a frozenset
    triple_block = {}
    for s in bsets:
        for t in itertools.combinations(sorted(s), 3):
            triple_block[t] = s
    nodes = 0
    autos = []
    aborted = False

    def consistent(images: list[int]) -> bool:
        k = len(images)
        # any fully assigned block must hit a block
        assigned = set(range(k))
        for s in bsets:
            if s <= assigned:
                im = frozenset(images[i] for i in s)
                if im not in blockset:
                    return False
        return True

    def bt(images, used):
        nonlocal nodes, aborted
        if aborted:
            return
        nodes += 1
        if nodes > node_cap:
            aborted = True
            return
        if len(images) == n:
            autos.append(tuple(images))
            return
        i = len(images)
        for j in range(n):
            if used[j]:
                continue
            images.append(j)
            used[j] = True
            if consistent(images):
                bt(images, used)
            images.pop()
            used[j] = False
            if aborted:
                return

    bt([], [False] * n)
    # every returned tuple was checked at the leaves by consistent()
    return {
        "order_found": len(autos),
        "exhaustive": not aborted,
        "nodes": nodes,
        "perms": autos,
    }


def stabilizer_of_signs(n: int, blocks, eps: np.ndarray, autos: list[tuple]) -> list[tuple]:
    """Automorphisms with J_{σ(b)} = J_b exactly (not up to gauge)."""
    index = {tuple(b): i for i, b in enumerate(blocks)}
    nlimbs = len(eps)

    def bit(bi: int) -> int:
        limb, b = divmod(bi, 64)
        return int((int(eps[limb]) >> b) & 1)

    kept = []
    for perm in autos:
        ok = True
        for block in blocks:
            im = tuple(sorted(perm[i] for i in block))
            bi = index[tuple(block)]
            bj = index[im]
            if bit(bi) != bit(bj):
                ok = False
                break
        if ok:
            kept.append(perm)
    return kept


def eps_bits(eps: np.ndarray, b: int) -> list[int]:
    out = []
    for bi in range(b):
        limb, bit = divmod(bi, 64)
        out.append(int((int(eps[limb]) >> bit) & 1))
    return out


def apply_signs_to_energy_table(n: int, blocks, eps: np.ndarray) -> np.ndarray:
    """Diagonal of H_C in the MSB computational basis, length 2^N.

    Bit convention matches round 1: qubit 0 is the most significant bit,
    so computational index x has y_i = bit (n-1-i) of x? We store y as the
    integer whose bit i is the spin y_i with bit 0 = point 0 = LSB.
    The statevector index uses MSB = point 0, so index = bitreverse_pack(y).
    Returned array is indexed by the statevector index.
    """
    bcount = len(blocks)
    # chi contribution: for each block, the parity of the n bits.
    # Build E[y_int] = -b + 2 u, y_int LSB = point 0, then scatter to MSB index.
    code_rank, code, b = code_bitmasks(n, blocks)
    if code.shape[0] != (1 << n):
        raise RuntimeError("code length")
    weights = np.empty(1 << n, dtype=np.int16)
    for y in range(1 << n):
        weights[y] = _pop_limbs(_xor(code[y], eps))
    E_y = -bcount + 2 * weights.astype(np.int32)
    # map y (bit i = point i, LSB) to statevector index (point 0 = MSB)
    idx = np.empty(1 << n, dtype=np.int64)
    for y in range(1 << n):
        sv = 0
        for i in range(n):
            if (y >> i) & 1:
                sv |= 1 << (n - 1 - i)
        idx[y] = sv
    diag = np.empty(1 << n, dtype=np.float64)
    diag[idx] = E_y
    return diag


def prepare_catalog(n: int, compute_aut: bool = True, aut_node_cap: int = 2_000_000, short_family: bool = False) -> dict:
    blocks = list(design_blocks(n))
    params = sqs_parameters(n)
    rank, code, b = code_bitmasks(n, blocks)
    uniq = unique_code(code)
    nlimbs = code.shape[1]
    autos = {"order_found": 0, "exhaustive": False, "nodes": 0, "perms": []}
    if compute_aut and n <= 12:
        autos = design_automorphisms(n, blocks, node_cap=aut_node_cap)

    def pack(name: str, eps0: np.ndarray) -> dict:
        eps, y0, land = gauge_fix(code, eps0)
        stab = []
        if autos["perms"]:
            stab = stabilizer_of_signs(n, blocks, eps, autos["perms"])
        can = canonical_coset_rep(uniq, eps)
        return {
            "name": name,
            "eps": eps,
            "y0_gauge": y0,
            "u_min": land["u_min"],
            "degeneracy": land["degeneracy"],
            "mean_agreement": land["mean_agreement"],
            "u_max": land["u_max"],
            "stabilizer_order": len(stab),
            "aut_exhaustive": autos["exhaustive"],
            "canonical": can.hex(),
            "n_neg_blocks": sum(eps_bits(eps, b)),
            "minimizers": land["minimizers"],
        }

    catalog = [pack("uniform", np.zeros(nlimbs, dtype=np.uint64))]
    if short_family:
        # Declared short list for N where the full family is only a frustration
        # certificate, not a QAOA scan. Still deterministic.
        catalog.append(pack("point_star_0", point_star_eps(n, blocks, 0, nlimbs)))
        catalog.append(pack("block_0", single_block_eps(0, nlimbs)))
        catalog.append(pack("pair_0_1", pair_star_eps(blocks, (0, 1), nlimbs)))
    else:
        for p in range(n):
            catalog.append(pack(f"point_star_{p}", point_star_eps(n, blocks, p, nlimbs)))
        for bi in range(b):
            catalog.append(pack(f"block_{bi}", single_block_eps(bi, nlimbs)))
        for a, cpt in itertools.combinations(range(n), 2):
            catalog.append(pack(f"pair_{a}_{cpt}", pair_star_eps(blocks, (a, cpt), nlimbs)))
    if n in (8, 16):
        catalog.append(pack("affine_off_origin", origin_flat_eps(blocks, nlimbs)))

    cover = covering_radius_exact(code, b) if b <= 16 else None
    # pick structured target
    frustrated = [c for c in catalog if c["name"] != "uniform" and c["u_min"] >= 1]
    target = None
    if frustrated:
        frustrated.sort(key=lambda c: (c["u_min"], -c["stabilizer_order"], c["canonical"], c["name"]))
        target = frustrated[0]
    return {
        "N": n,
        "params": params,
        "blocks": blocks,
        "rank": rank,
        "n_codewords": int(len(uniq)),
        "n_cosets": None if (1 << b) < len(uniq) else int((1 << b) // len(uniq)) if b <= 20 else None,
        "b": b,
        "aut_order": autos["order_found"],
        "aut_exhaustive": autos["exhaustive"],
        "aut_nodes": autos["nodes"],
        "covering_radius": cover,
        "catalog": catalog,
        "target_name": None if target is None else target["name"],
        "code": code,
    }
