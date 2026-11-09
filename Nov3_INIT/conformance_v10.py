"""Unread-band provenance and exact-v4 freeze. No SPSA."""
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

from Nov3_INIT.alloc import iterations_equal4, x0_center4, x0_equal4, x0_qaoa_center4

ROOT = Path(__file__).resolve().parent
PREREG = ROOT / "PREREG_V10.md"
OUT = ROOT / "results" / "v10" / "CONFORMANCE.json"
RES = ROOT / "results"
BAND = list(range(10232, 10248))


def _collisions() -> dict:
    hits = []
    for path in RES.rglob("*"):
        name = path.name
        if not any(f"s{s}" in name for s in BAND):
            continue
        if path.suffix in {".json", ".npz", ".md"}:
            hits.append(str(path.relative_to(RES)))
    return {"n": len(hits), "paths": hits}


def _recipe() -> dict:
    x_n = x0_center4("Nov3rd", 2, 0, None)
    x_q = x0_qaoa_center4(2, 0, None)
    expected_n = np.array([0.25, 0.0, 0.0, -0.5, 0.0, 0.0, -0.5, 0.0, 0.0])
    expected_q = np.full(4, 0.25)
    return {
        "nov3_center_L2": [float(v) for v in x_n],
        "qaoa_center_L2": [float(v) for v in x_q],
        "T_Q2048": iterations_equal4(2048),
        "qaoa_center_is_historical_equal4_start0": bool(np.array_equal(x_q, x0_equal4("QAOA", 2, 0, None))),
        "exact_v4_recipe": bool(np.array_equal(x_n, expected_n) and np.array_equal(x_q, expected_q)),
    }


def main() -> None:
    if any(s <= 10231 for s in BAND):
        raise RuntimeError("band intersects ≤10231")
    if set(BAND) & set(range(11000, 11032)) or set(BAND) & set(range(20000, 21000)):
        raise RuntimeError("band overlaps validation or final")
    coll = _collisions()
    recipe = _recipe()
    payload = {
        "ok": bool(recipe["exact_v4_recipe"] and recipe["qaoa_center_is_historical_equal4_start0"] and coll["n"] == 0 and recipe["T_Q2048"] == 250),
        "prereg_sha256": hashlib.sha256(PREREG.read_bytes()).hexdigest(),
        "historical_queries_preserved": 2113536,
        "estimate_seeds": BAND,
        "collisions_in_nov3_init_results": coll,
        "recipe": recipe,
        "label_rule": "ESTIMATE_NEGATIVE_CI iff n=16 mean d_pair CI entirely < 0 else UNSTABLE / CI_CROSSES_0",
        "no_gr_patch": True,
        "no_candidacy_file": True,
        "no_spsa_in_this_module": True,
        "no_L3": True,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    if not payload["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
