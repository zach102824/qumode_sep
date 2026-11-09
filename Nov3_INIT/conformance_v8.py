"""Holdout seed provenance and exact-v4 recipe freeze. No SPSA."""
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

from Nov3_INIT.alloc import x0_center4, x0_equal4, x0_qaoa_center4

ROOT = Path(__file__).resolve().parent
PREREG = ROOT / "PREREG_V8.md"
OUT = ROOT / "results" / "v8" / "CONFORMANCE.json"
RES = ROOT / "results"
HOLDOUT = list(range(10144, 10152))
FORBIDDEN = {
    "ft_dev_10018_10025": set(range(10018, 10026)),
    "local_r15_r16_le_10143": set(range(0, 10144)),
    "le_warm_10128_10135": set(range(10128, 10136)),
    "validation_11000_11031": set(range(11000, 11032)),
    "gr_final_20000_20999": set(range(20000, 21000)),
}


def _collisions() -> dict:
    hits = []
    for path in RES.rglob("*"):
        name = path.name
        if not any(f"s{s}" in name or f"_s{s}_" in name for s in HOLDOUT):
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
        "qaoa_center_is_historical_equal4_start0": bool(np.array_equal(x_q, x0_equal4("QAOA", 2, 0, None))),
        "exact_v4_recipe": bool(np.array_equal(x_n, expected_n) and np.array_equal(x_q, expected_q)),
    }


def main() -> None:
    if set(HOLDOUT) & FORBIDDEN["ft_dev_10018_10025"]:
        raise RuntimeError("holdout overlaps FT development")
    if any(s <= 10143 for s in HOLDOUT):
        raise RuntimeError("holdout intersects local R15/R16 upper bound")
    if set(HOLDOUT) & FORBIDDEN["le_warm_10128_10135"]:
        raise RuntimeError("holdout overlaps LE warm-start")
    if set(HOLDOUT) & FORBIDDEN["validation_11000_11031"]:
        raise RuntimeError("holdout overlaps validation")
    if set(HOLDOUT) & FORBIDDEN["gr_final_20000_20999"]:
        raise RuntimeError("holdout overlaps GR final")
    coll = _collisions()
    recipe = _recipe()
    payload = {
        "ok": bool(recipe["exact_v4_recipe"] and recipe["qaoa_center_is_historical_equal4_start0"] and coll["n"] == 0),
        "prereg_sha256": hashlib.sha256(PREREG.read_bytes()).hexdigest(),
        "historical_queries_preserved": 1949696,
        "holdout_seeds": HOLDOUT,
        "collisions_in_nov3_init_results": coll,
        "recipe": recipe,
        "gates": {
            "H1": "N=15 L=2 Q=2048 mean d_pair CI entirely < 0",
            "H2": "N=18 L=2 Q=2048 mean d_pair CI entirely < 0",
            "confirmed_only_if_both": True,
        },
        "both_q_run": [512, 1024],
        "no_spsa_in_this_module": True,
        "will_not_patch_pr77": True,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    if not payload["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
