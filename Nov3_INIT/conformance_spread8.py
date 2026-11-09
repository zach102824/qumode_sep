"""Spread8 L=3 freeze. No SPSA on a Hamiltonian."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np

from Nov3_INIT.alloc import (
    METHOD_ID,
    iterations_equal4,
    iterations_spread8,
    run_equal4,
    run_spread8,
    x0_center4,
    x0_qaoa_center4,
    x0_spread8,
)

ROOT = Path(__file__).resolve().parent
PREREG = ROOT / "PREREG_SPREAD8.md"
OUT = ROOT / "results" / "spread8" / "CONFORMANCE.json"
RES = ROOT / "results"
BAND = list(range(10376, 10384))
HELD = list(range(10384, 10392))
RESERVED = {
    "le_10360_10375": set(range(10360, 10376)),
    "local_r25_10344_10359": set(range(10344, 10360)),
    "local_next_ge_10392": set(range(10392, 20000)),
    "ft_and_prior_le_10335": set(range(0, 10336)),
    "holdout_10144_10151": set(range(10144, 10152)),
    "validation_11000_11031": set(range(11000, 11032)),
    "gr_final_20000_20999": set(range(20000, 21000)),
    "window_held_10384_10391": set(HELD),
}


def _collisions() -> dict:
    hits = []
    if RES.exists():
        for path in RES.rglob("*"):
            if path.suffix not in {".json", ".npz", ".md"}:
                continue
            if any(f"s{s}" in path.name for s in BAND):
                hits.append(str(path.relative_to(RES)))
    return {"n": len(hits), "paths": hits}


def _seed_ok() -> dict:
    band = set(BAND)
    overlaps = {name: sorted(band & reserved) for name, reserved in RESERVED.items()}
    return {
        "band": BAND,
        "held_unread": HELD,
        "overlaps": overlaps,
        "ok": all(len(v) == 0 for v in overlaps.values()) and BAND == list(range(10376, 10384)),
    }


def _recipe() -> dict:
    x_n = x0_spread8("Nov3rd", 3, 0, None)
    x_q = x0_spread8("QAOA", 3, 0, None)
    return {
        "T_equal4": iterations_equal4(3072),
        "T_spread8": iterations_spread8(3072),
        "nov3_center": [float(v) for v in x_n],
        "qaoa_center": [float(v) for v in x_q],
        "qaoa_center_matches_v4": bool(np.array_equal(x_q, x0_qaoa_center4(3, 0, None))),
        "nov3_center_matches_v4": bool(np.array_equal(x_n, x0_center4("Nov3rd", 3, 0, None))),
        "method_ids": {"equal4": METHOD_ID["equal4"], "spread8": METHOD_ID["spread8"]},
        "ok": bool(
            iterations_equal4(3072) == 378
            and iterations_spread8(3072) == 186
            and METHOD_ID["spread8"]["Nov3rd"] == 5
            and METHOD_ID["spread8"]["QAOA"] == 6
            and np.array_equal(x_q, np.full(6, 0.25))
            and float(x_n[0]) == 0.25
            and np.all(x_n[3::3] == -0.5)
        ),
    }


def _ledger() -> dict:
    def fake(x):
        return float(np.sum(np.asarray(x) ** 2)), None

    def rngs(n):
        init = [None if s == 0 else np.random.default_rng(7000 + s) for s in range(n)]
        spsa = [np.random.default_rng(8000 + s) for s in range(n)]
        return init, spsa

    ri, rs = rngs(4)
    eq = run_equal4(fake, "Nov3rd", 3, 128, ri, rs, x0_fn=x0_center4)
    ri, rs = rngs(8)
    sp = run_spread8(fake, "Nov3rd", 3, 128, ri, rs)
    return {
        "equal4": eq["queries"],
        "spread8": sp["queries"],
        "ok": eq["queries"]["total"] == 128 and sp["queries"]["total"] == 128 and sp["queries"]["init"] == 8,
    }


def main() -> None:
    seed = _seed_ok()
    coll = _collisions()
    rec = _recipe()
    led = _ledger()
    payload = {
        "ok": bool(seed["ok"] and coll["n"] == 0 and rec["ok"] and led["ok"]),
        "prereg_sha256": hashlib.sha256(PREREG.read_bytes()).hexdigest(),
        "historical_queries_preserved": 2400256,
        "not_l2_center4_resample": True,
        "not_le_n18_depth": True,
        "not_paper_V_M": True,
        "shared_uc_only": True,
        "seeds": seed,
        "collisions_in_nov3_init_results": coll,
        "recipe": rec,
        "ledger_quadratic": led,
        "label_rule": "SPREAD8_L3_EDGE iff nov3_spread8 vs qaoa_spread8 mean dE CI entirely <0 AND nov3_spread8 mean E <= nov3_equal4; else REJECT_SPREAD8_L3",
        "no_gr_patch": True,
        "no_candidacy_file": True,
        "no_spsa_on_hamiltonian_in_this_module": True,
    }

    def _plain(obj):
        if isinstance(obj, np.generic):
            return obj.item()
        raise TypeError(type(obj))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, default=_plain) + "\n")
    print(json.dumps(payload, indent=2, default=_plain))
    if not payload["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
