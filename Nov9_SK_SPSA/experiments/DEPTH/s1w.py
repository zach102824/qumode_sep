"""S1 L1→L pad. Does not modify DEPTH C1 cont.py or frozen S1JOINT."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "N3COND"))
sys.path.insert(0, str(HERE.parent / "STARTS"))
import nov3 as N  # noqa: E402
import starts as ST  # noqa: E402

WARM_MID = 240  # distinct from DEPTH C1 CONT_MID=140
Q1 = 1024


def pad_l1(x1, L):
    """Shared Uc + layer-0 shell from L1; extra shells exact 0."""
    x1 = np.asarray(x1, float)
    if x1.size != 6:
        raise ValueError(x1.size)
    L = int(L)
    if L < 2:
        raise ValueError(L)
    extra = np.zeros(3 * (L - 1))
    out = np.concatenate([x1, extra])
    if out.size != 3 + 3 * L:
        raise ValueError(out.size)
    return out


def rng_cold(name, seed, nq, L):
    return ST.rng_for_arm(name, seed, nq, L)


def rng_l1(seed, nq):
    return ST.rng_for_arm("Nov3", seed, nq, 1)


def rng_warm(seed, nq, L):
    return lambda s: N.rng_for(int(seed), int(nq), int(L), WARM_MID, s, 0)
