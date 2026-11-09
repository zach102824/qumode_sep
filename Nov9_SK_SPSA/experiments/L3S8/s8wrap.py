"""8-start RNG lists for v4 run_spread8. Does not change alloc.py or FTREP/wrap.py."""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "FTREP"))
import alloc as A  # noqa: E402

METHOD_ID_S8_NOV3 = 90
METHOD_ID_S8_QAOA = 91


def rng_lists_spread8(seed, nq, b, L, method_id):
    rng_init = [None if s == 0 else A.make_rng(seed, nq, b, L, int(method_id), s, 2)
                for s in range(8)]
    rng_spsa = [A.make_rng(seed, nq, b, L, int(method_id), s, 0) for s in range(8)]
    return rng_init, rng_spsa
