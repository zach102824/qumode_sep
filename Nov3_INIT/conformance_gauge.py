"""Gauge-robustness freeze. No SPSA on a Hamiltonian."""
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

from Nov3_INIT.alloc import METHOD_ID, iterations_equal4, make_rng, n_params, run_equal4, x0_center4, x0_qaoa_center4
from Nov3_INIT.gauge import (
    BAND,
    GAUGE_RNG_ID,
    N,
    N_GAUGES,
    algebra_probe,
    frozen_table,
    uniqueness,
)
from Nov9_SK_SPSA.experiments.FT.instance import packing

ROOT = Path(__file__).resolve().parent
PREREG = ROOT / "PREREG_GAUGE.md"
RES = ROOT / "results" / "gauge"
OUT = RES / "CONFORMANCE.json"
GAUGE = RES / "GAUGE.json"
HIST = 2818048
RESERVED = {
    "local_r31_10536_10551": set(range(10536, 10552)),
    "ft_wl_10528_10535": set(range(10528, 10536)),
    "ft_cutoff_10520_10527": set(range(10520, 10528)),
    "local_r30_10504_10519": set(range(10504, 10520)),
    "le_10432_10439": set(range(10432, 10440)),
    "ft_and_prior_le_10463": set(range(0, 10464)),
    "holdout_10144_10151": set(range(10144, 10152)),
    "validation_11000_11031": set(range(11000, 11032)),
    "gr_final_20000_20999": set(range(20000, 21000)),
}


def _plain(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(type(obj))


def _collisions() -> dict:
    hits = []
    parent = ROOT / "results"
    if parent.exists():
        for path in parent.rglob("*"):
            if path.suffix not in {".json", ".npz", ".md"}:
                continue
            if any(f"s{s}" in path.name for s in BAND):
                hits.append(str(path.relative_to(parent)))
    return {"n": len(hits), "paths": hits}


def _seed_ok() -> dict:
    band = set(BAND)
    overlaps = {name: sorted(band & reserved) for name, reserved in RESERVED.items()}
    return {
        "band": BAND,
        "n": len(BAND),
        "overlaps": overlaps,
        "ok": all(len(v) == 0 for v in overlaps.values()) and BAND == list(range(10552, 10560)),
    }


def _ledger() -> dict:
    calls = {"n": 0}

    def fake(x):
        calls["n"] += 1
        return float(np.sum(np.asarray(x) ** 2)), None

    init = [None if s == 0 else np.random.default_rng(9100 + s) for s in range(4)]
    spsa = [np.random.default_rng(9200 + s) for s in range(4)]
    out = run_equal4(fake, "Nov3rd", 1, 128, init, spsa, x0_fn=x0_center4)
    return {"ok": out["queries"]["total"] == 128 and calls["n"] == 128, "ledger": out["queries"]}


def _rng_omits_gauge() -> dict:
    a = make_rng(10552, 15, 0.8, 1, 1, 1, 0)
    b = make_rng(10552, 15, 0.8, 1, 1, 1, 0)
    return {
        "ok": bool(np.allclose(a.normal(size=4), b.normal(size=4))),
        "note": "make_rng has no gauge id; same (seed, method) draws match across gauges",
    }


def main() -> None:
    seed = _seed_ok()
    coll = _collisions()
    table = frozen_table()
    uniq = uniqueness(table)
    alg = algebra_probe()
    led = _ledger()
    rng = _rng_omits_gauge()
    expected_q = len(BAND) * N_GAUGES * 2 * 1024
    payload = {
        "ok": bool(
            seed["ok"]
            and coll["n"] == 0
            and alg["energy_maps"]
            and alg["mapped_residual_ok"]
            and alg["plus_invariant"]
            and alg["qaoa_invariant"]
            and alg["qaoa_state_mapped"]
            and alg["entropy_invariant"]
            and alg["nov3_generic_not_covariant"]
            and led["ok"]
            and rng["ok"]
            and iterations_equal4(1024) == 122
            and n_params("Nov3rd", 1) == 6
            and n_params("QAOA", 1) == 2
            and METHOD_ID["equal4"]["Nov3rd"] == 1
            and METHOD_ID["equal4"]["QAOA"] == 0
            and GAUGE_RNG_ID == 4401
            and packing(N) == [[a, a + 5, a + 10] for a in range(5)]
            and expected_q == 65536
        ),
        "prereg_sha256": hashlib.sha256(PREREG.read_bytes()).hexdigest(),
        "historical_sk_queries": HIST,
        "expected_new_energy_queries": expected_q,
        "expected_total_if_complete": HIST + expected_q,
        "GAUGE_RNG_ID": GAUGE_RNG_ID,
        "n_gauges": N_GAUGES,
        "seeds": seed,
        "collisions_in_nov3_init_results": coll,
        "gauge_uniqueness": uniq,
        "algebra": alg,
        "ledger_quadratic": led,
        "rng_omits_gauge": rng,
        "statistical_unit": "8 original instances; gauges correlated; not 32 iid",
        "primary": "mean over instances of gauge-average Nov3-QAOA E; paired bootstrap 3334",
        "no_adoption_gate": True,
        "qaoa_covariance_is_sanity_not_assumption": True,
        "reuse_original_Bstar": True,
        "no_spsa_on_hamiltonian_in_this_module": True,
        "no_gr_patch": True,
        "compiler_stream_closed": True,
    }
    RES.mkdir(parents=True, exist_ok=True)
    GAUGE.write_text(json.dumps(table, indent=2) + "\n")
    payload["gauge_sha256"] = hashlib.sha256(GAUGE.read_bytes()).hexdigest()
    OUT.write_text(json.dumps(payload, indent=2, default=_plain) + "\n")
    print(json.dumps(payload, indent=2, default=_plain))
    if not payload["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
