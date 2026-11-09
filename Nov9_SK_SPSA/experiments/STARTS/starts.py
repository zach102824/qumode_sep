"""Equal-budget multi-start lists. S4 matches historical C0 starts_for."""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "N3COND"))
import nov3 as N  # noqa: E402


def starts_for_n(name, seed, nq, L, nstarts):
    if nstarts < 1:
        raise ValueError(nstarts)
    cid = N.circuit_id(name)
    nparam = 3 + 3 * L if name == "Nov3" else 2 * L
    out = [N.tqa_nov3(L) if name == "Nov3" else N.tqa_qaoa(L)]
    for s in range(1, nstarts):
        out.append(N.rng_for(seed, nq, L, cid, s, 2).uniform(-0.6, 0.6, size=nparam))
    return out


def rng_for_arm(name, seed, nq, L):
    cid = N.circuit_id(name)
    return lambda s: N.rng_for(seed, nq, L, cid, s, 0)
