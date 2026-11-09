"""Pad L1 parameters into depth L. Ordinary C0 continuation starts."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "N3COND"))
import nov3 as N  # noqa: E402

CONT_MID = {"Nov3": 140, "QAOA": 100}


def pad_l1(name, x1, L):
    x1 = np.asarray(x1, float)
    if L < 1:
        raise ValueError(L)
    if name == "Nov3":
        if x1.size != 6:
            raise ValueError(x1.size)
        extra = np.zeros(3 * (L - 1))
        return np.concatenate([x1, extra])
    if name == "QAOA":
        if x1.size != 2:
            raise ValueError(x1.size)
        g = np.zeros(L)
        b = np.zeros(L)
        g[0] = x1[0]
        b[0] = x1[1]
        return np.concatenate([g, b])
    raise ValueError(name)


def new_layer_idx(name, L):
    if name == "Nov3":
        return list(range(6, 3 + 3 * L))
    if name == "QAOA":
        return list(range(1, L)) + list(range(L + 1, 2 * L))
    raise ValueError(name)


def continuation_starts(name, seed, nq, L, x1):
    x0 = pad_l1(name, x1, L)
    out = [x0.copy()]
    idx = new_layer_idx(name, L)
    mid = CONT_MID[name]
    for s in range(1, 4):
        y = x0.copy()
        if idx:
            y[idx] = N.rng_for(seed, nq, L, mid, s, 2).uniform(-0.6, 0.6, size=len(idx))
        out.append(y)
    return out


def rng_cold(name, seed, nq, L):
    cid = N.circuit_id(name)
    return lambda s: N.rng_for(seed, nq, L, cid, s, 0)


def rng_l1(name, seed, nq):
    cid = N.circuit_id(name)
    return lambda s: N.rng_for(seed, nq, 1, cid, s, 0)


def rng_cont(name, seed, nq, L):
    mid = CONT_MID[name]
    return lambda s: N.rng_for(seed, nq, L, mid, s, 0)
