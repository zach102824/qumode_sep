"""Layer-tied SNAP: one (s1,s2,s3) repeated L times. Does not change alloc.py.

SPSA geometry is Nov3 L=1 (dim 6). Circuit is original Nov3 L=3 with
SNAP_1 = SNAP_2 = SNAP_3. Q=3072 uses v4 run_equal4 at L=1 so T=378,
matched to JOINT/QAOA center-4. Distinct method_id so RNG does not
collide with JOINT (id 1) or QAOA (id 0).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "FTREP"))
import alloc as A  # noqa: E402

METHOD_ID_TIE = 94
L_SPSA = 1
DIM_SPSA = 6


def expand_tie(x, L):
    """Map [γ, θ_K, θ_M, s1, s2, s3] → L copies of the SNAP triple."""
    x = np.asarray(x, dtype=np.float64)
    if x.size != DIM_SPSA:
        raise ValueError(f"tie SPSA dim {x.size} != {DIM_SPSA}")
    L = int(L)
    out = np.empty(3 + 3 * L, dtype=np.float64)
    out[:3] = x[:3]
    snap = x[3:6]
    for ell in range(L):
        out[3 + 3 * ell : 6 + 3 * ell] = snap
    return out


def rng_lists_tie(seed, nq, b, L_circ):
    rng_init = [
        None if s == 0 else A.make_rng(seed, nq, b, int(L_circ), METHOD_ID_TIE, s, 2)
        for s in range(4)
    ]
    rng_spsa = [
        A.make_rng(seed, nq, b, int(L_circ), METHOD_ID_TIE, s, 0) for s in range(4)
    ]
    return rng_init, rng_spsa


def draw_tie(method_name, ell, start_id, rng):
    del method_name, ell
    return A.x0_center4("Nov3rd", L_SPSA, start_id, rng)
