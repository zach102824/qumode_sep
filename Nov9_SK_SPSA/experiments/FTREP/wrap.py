"""Layout map and FT RNG lists. Does not change alloc.py SPSA."""
from __future__ import annotations

import numpy as np

from alloc import make_rng, x0_center4, x0_qaoa_center4


def to_le(method, x):
    """FT QAOA is interleaved [γ0,β0,...]; LE QAOA is stacked [γ...,β...]. Nov3 is identical."""
    x = np.asarray(x, float)
    if method == "QAOA":
        return np.concatenate([x[0::2], x[1::2]])
    return x.copy()


def rng_lists(seed, nq, b, L, method):
    mid = 0 if method == "QAOA" else 1
    rng_init = [None if s == 0 else make_rng(seed, nq, b, L, mid, s, 2) for s in range(4)]
    rng_spsa = [make_rng(seed, nq, b, L, mid, s, 0) for s in range(4)]
    return rng_init, rng_spsa


def draw_fn(arm):
    def draw(method_name, ell, start_id, rng):
        if arm == "qaoa_center4":
            return x0_qaoa_center4(ell, start_id, rng)
        return x0_center4(method_name, ell, start_id, rng)

    return draw
