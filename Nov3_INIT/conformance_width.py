"""N=15/18 L=1 width freeze. N=18 resource gate. No SPSA on panel Hamiltonians."""
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
from Nov3_INIT.bstar import _ground
from Nov3_INIT.circuit import Logical
from Nov9_SK_SPSA.experiments.FT.instance import make_J, packing, pair_diag, z_table

ROOT = Path(__file__).resolve().parent
PREREG = ROOT / "PREREG_WIDTH.md"
RES = ROOT / "results"
OUT = RES / "width" / "CONFORMANCE.json"
BAND = list(range(10568, 10584))
N15 = list(range(10568, 10576))
N18 = list(range(10576, 10584))
HIST = 2883584
RESERVED = {
    "le_10432_10439": set(range(10432, 10440)),
    "gauge_and_buffer_10552_10567": set(range(10552, 10568)),
    "local_r31_10536_10551": set(range(10536, 10552)),
    "local_r30_10504_10519": set(range(10504, 10520)),
    "ft_10520_10535": set(range(10520, 10536)),
    "ft_vm_10480_10487": set(range(10480, 10488)),
    "ft_anneal_10456_10463": set(range(10456, 10464)),
    "ft_spread8_10376_10383": set(range(10376, 10384)),
    "shellgain_10368_10375": set(range(10368, 10376)),
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
        "n15": N15,
        "n18": N18,
        "n": len(BAND),
        "overlaps": overlaps,
        "ok": bool(
            all(len(v) == 0 for v in overlaps.values())
            and BAND == list(range(10568, 10584))
            and N15 == list(range(10568, 10576))
            and N18 == list(range(10576, 10584))
            and len(N15) == 8
            and len(N18) == 8
            and not (set(N15) & set(N18))
        ),
    }


def _rss_kib() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def _n18_gate() -> dict:
    rss0 = _rss_kib()
    J = make_J(10018, 18)
    hc = pair_diag(J, z_table(18))
    t0 = time.perf_counter()
    gs = _ground(hc, 0.5, 18)
    eig_s = time.perf_counter() - t0
    dummy = np.full(1 << 18, 2.0 ** (-0.5 * 18), dtype=np.float64)
    logical = Logical(18, 10018, 0.5, dummy)
    x = x0_center4("Nov3rd", 1, 0, None)
    t1 = time.perf_counter()
    e = logical.energy(logical.nov3(x, 1))
    energy_s = time.perf_counter() - t1
    rss = _rss_kib()
    peak_gib = rss / (1024.0 * 1024.0)
    allowed = bool(
        eig_s < 30.0
        and energy_s < 0.5
        and peak_gib < 4.0
        and gs.size == (1 << 18)
        and hc.size == (1 << 18)
    )
    return {
        "seed_not_fitted": 10018,
        "N": 18,
        "dim": 1 << 18,
        "dense_matrix": False,
        "eigsh_seconds": eig_s,
        "nov3_L1_seconds": energy_s,
        "nov3_L1_energy": float(e),
        "rss_kib_start": rss0,
        "rss_kib_peak": rss,
        "peak_rss_gib": peak_gib,
        "n18_allowed": allowed,
        "no_spsa": True,
    }


def _ledger() -> dict:
    calls = {"n": 0}

    def fake(x):
        calls["n"] += 1
        return float(np.sum(np.asarray(x) ** 2)), None

    init = [None if s == 0 else np.random.default_rng(6100 + s) for s in range(4)]
    spsa = [np.random.default_rng(6200 + s) for s in range(4)]
    out = run_equal4(fake, "Nov3rd", 1, 128, init, spsa, x0_fn=x0_center4)
    return {"ok": out["queries"]["total"] == 128 and calls["n"] == 128, "ledger": out["queries"]}


def _recipe() -> dict:
    xn = x0_center4("Nov3rd", 1, 0, None)
    xq = x0_qaoa_center4(1, 0, None)
    return {
        "nov3_start0": [float(v) for v in xn],
        "qaoa_start0": [float(v) for v in xq],
        "ok": bool(
            abs(float(xn[0]) - 0.25) < 1e-15
            and abs(float(xn[3]) + 0.5) < 1e-15
            and np.allclose(xq, [0.25, 0.25])
            and METHOD_ID["equal4"]["Nov3rd"] == 1
            and METHOD_ID["equal4"]["QAOA"] == 0
            and iterations_equal4(1024) == 122
            and n_params("Nov3rd", 1) == 6
            and n_params("QAOA", 1) == 2
        ),
    }


def main() -> None:
    seed = _seed_ok()
    coll = _collisions()
    rec = _recipe()
    led = _ledger()
    gate = _n18_gate()
    expected = 16 * 2 * 1024
    payload = {
        "ok": bool(
            seed["ok"]
            and coll["n"] == 0
            and rec["ok"]
            and led["ok"]
            and gate["n18_allowed"]
            and packing(15) == [[a, a + 5, a + 10] for a in range(5)]
            and packing(18) == [[a, a + 6, a + 12] for a in range(6)]
            and expected == 32768
        ),
        "prereg_sha256": hashlib.sha256(PREREG.read_bytes()).hexdigest(),
        "historical_sk_queries": HIST,
        "expected_new_energy_queries": expected,
        "expected_total_if_complete": HIST + expected,
        "seeds": seed,
        "collisions_in_nov3_init_results": coll,
        "recipe": rec,
        "ledger_quadratic": led,
        "n18_gate": gate,
        "not_l2_sampling": True,
        "not_l3_schedule_tweak": True,
        "compiler_stream_closed": True,
        "no_gr_patch": True,
        "no_adoption_gate": True,
        "holdout_l2_fail_stands": True,
        "statistical_unit": "8 instances per N; do not pool 16 across N",
        "no_spsa_on_panel_hamiltonian_in_this_module": True,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2, default=_plain) + "\n")
    print(json.dumps(payload, indent=2, default=_plain))
    if not payload["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
