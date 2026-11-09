"""L=3 doubled-budget freeze. No SPSA on panel Hamiltonians."""
from __future__ import annotations

import hashlib
import json
import os
import resource
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np

from Nov3_INIT.alloc import METHOD_ID, iterations_equal4, n_params, run_equal4, x0_center4, x0_qaoa_center4
from Nov3_INIT.circuit import Logical
from Nov9_SK_SPSA.experiments.FT.instance import packing

ROOT = Path(__file__).resolve().parent
PREREG = ROOT / "PREREG_L3_2X.md"
RES = ROOT / "results"
OUT = RES / "l3_2x" / "CONFORMANCE.json"
BAND = list(range(10616, 10624))
HOLD = list(range(10624, 10632))
Q = 6144
HIST = 2916352
RESERVED = {
    "le_10432_10439": set(range(10432, 10440)),
    "le_pr115_10552_10559": set(range(10552, 10560)),
    "le_next_10600_10615": set(range(10600, 10616)),
    "local_r31_10536_10551": set(range(10536, 10552)),
    "local_r32_10584_10599": set(range(10584, 10600)),
    "local_r30_10504_10519": set(range(10504, 10520)),
    "ft_10520_10535": set(range(10520, 10536)),
    "gauge_10552_10559": set(range(10552, 10560)),
    "width_10568_10583": set(range(10568, 10584)),
    "ft_vm_10480_10487": set(range(10480, 10488)),
    "ft_anneal_10456_10463": set(range(10456, 10464)),
    "ft_spread8_10376_10383": set(range(10376, 10384)),
    "shellgain_10368_10375": set(range(10368, 10376)),
    "unused_hole_10560_10567": set(range(10560, 10568)),
    "holdout_10144_10151": set(range(10144, 10152)),
    "validation_11000_11031": set(range(11000, 11032)),
    "gr_final_20000_20999": set(range(20000, 21000)),
    "held_unused_10624_10631": set(HOLD),
}


def _plain(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError(type(obj))


def _collisions() -> dict:
    hits = []
    if RES.exists():
        for path in RES.rglob("*"):
            if path.suffix not in {".json", ".npz", ".md"}:
                continue
            if any(f"s{s}" in path.name for s in BAND + HOLD):
                hits.append(str(path.relative_to(RES)))
    return {"n": len(hits), "paths": hits}


def _seed_ok() -> dict:
    band = set(BAND)
    overlaps = {name: sorted(band & reserved) for name, reserved in RESERVED.items()}
    return {
        "band": BAND,
        "hold_unused": HOLD,
        "n": len(BAND),
        "overlaps": overlaps,
        "ok": bool(
            all(len(v) == 0 for v in overlaps.values())
            and BAND == list(range(10616, 10624))
            and HOLD == list(range(10624, 10632))
            and not (set(BAND) & set(HOLD))
        ),
    }


def _rss_kib() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def _probe() -> dict:
    dummy = np.full(1 << 15, 2.0 ** (-0.5 * 15), dtype=np.float64)
    logical = Logical(15, 10018, 0.5, dummy)
    x = x0_center4("Nov3rd", 3, 0, None)
    rss0 = _rss_kib()
    t0 = time.perf_counter()
    e = logical.energy(logical.nov3(x, 3))
    dt = time.perf_counter() - t0
    rss1 = _rss_kib()
    proj = Q * dt
    allowed = bool(dt < 0.6 and proj < 3600.0 and rss1 / (1024.0 * 1024.0) < 1.0 and np.isfinite(e))
    return {
        "seed_not_fitted": 10018,
        "nov3_L3_seconds": dt,
        "nov3_L3_energy": float(e),
        "projected_Q6144_cell_seconds": proj,
        "rss_kib_start": rss0,
        "rss_kib_after": rss1,
        "allowed": allowed,
        "no_spsa": True,
    }


def _ledger() -> dict:
    calls = {"n": 0}

    def fake(x):
        calls["n"] += 1
        return float(np.sum(np.asarray(x) ** 2)), None

    init = [None if s == 0 else np.random.default_rng(5100 + s) for s in range(4)]
    spsa = [np.random.default_rng(5200 + s) for s in range(4)]
    out = run_equal4(fake, "Nov3rd", 3, 128, init, spsa, x0_fn=x0_center4)
    return {"ok": out["queries"]["total"] == 128 and calls["n"] == 128, "ledger": out["queries"]}


def _recipe() -> dict:
    xn = x0_center4("Nov3rd", 3, 0, None)
    xq = x0_qaoa_center4(3, 0, None)
    expected_n = np.array([0.25, 0.0, 0.0] + [-0.5, 0.0, 0.0] * 3)
    expected_q = np.full(6, 0.25)
    return {
        "nov3_start0": [float(v) for v in xn],
        "qaoa_start0": [float(v) for v in xq],
        "n_params_nov3": n_params("Nov3rd", 3),
        "n_params_qaoa": n_params("QAOA", 3),
        "T": iterations_equal4(Q),
        "ok": bool(
            np.allclose(xn, expected_n)
            and np.allclose(xq, expected_q)
            and n_params("Nov3rd", 3) == 12
            and n_params("QAOA", 3) == 6
            and iterations_equal4(Q) == 762
            and METHOD_ID["equal4"]["Nov3rd"] == 1
            and METHOD_ID["equal4"]["QAOA"] == 0
        ),
    }


def main() -> None:
    seed = _seed_ok()
    coll = _collisions()
    rec = _recipe()
    led = _ledger()
    probe = _probe()
    expected = 8 * 2 * Q
    payload = {
        "ok": bool(
            seed["ok"]
            and coll["n"] == 0
            and rec["ok"]
            and led["ok"]
            and probe["allowed"]
            and packing(15) == [[a, a + 5, a + 10] for a in range(5)]
            and expected == 98304
        ),
        "prereg_sha256": hashlib.sha256(PREREG.read_bytes()).hexdigest(),
        "historical_sk_queries": HIST,
        "expected_new_energy_queries": expected,
        "expected_total_if_complete": HIST + expected,
        "Q": Q,
        "q": 2048,
        "L": 3,
        "T": 762,
        "seeds": seed,
        "collisions_in_nov3_init_results": coll,
        "recipe": rec,
        "ledger_quadratic": led,
        "probe": probe,
        "not_spread8": True,
        "not_anneal": True,
        "not_pair_shell": True,
        "not_vm": True,
        "if_fail_stop_no_q4096": True,
        "no_gr_patch": True,
        "no_adoption_gate": True,
        "holdout_l2_fail_stands": True,
        "compiler_stream_closed": True,
        "no_spsa_on_panel_hamiltonian_in_this_module": True,
        "historical_L3_q1024_descriptive": {
            "v7_10018_25": {"mean_dE": 0.0193, "ci": [-0.045, 0.075]},
            "spread8_equal4_10376_83": {"mean_dE": -0.0315, "ci": [-0.106, 0.043]},
            "anneal_v4_10456_63": {"mean_dE": -0.0477, "ci": [-0.121, 0.022]},
            "not_a_paired_test": True,
        },
        "label_rule": "L3_2XBUDGET_ESTIMATE_NEGATIVE_CI iff mean Nov3-QAOA dE CI entirely <0 on n=8; else FAIL or UNSTABLE",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, default=_plain) + "\n")
    print(json.dumps(payload, indent=2, default=_plain))
    if not payload["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
