"""N=18 resource gate. No instance B* and no SPSA in this module."""
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

from Nov3_INIT.alloc import x0_center4, x0_equal4, x0_qaoa_center4
from Nov3_INIT.bstar import _ground
from Nov3_INIT.circuit import Logical
from Nov9_SK_SPSA.experiments.FT.instance import make_J, pair_diag, z_table

ROOT = Path(__file__).resolve().parent
PREREG = ROOT / "PREREG_V5.md"
OUT = ROOT / "results" / "v5" / "CONFORMANCE.json"
FORBIDDEN = set(range(11000, 11032)) | set(range(20000, 21000))
DEV = list(range(10018, 10026))


def _rss_kib() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def _probe() -> dict:
    t_cpu0 = time.process_time()
    rss0 = _rss_kib()
    J = make_J(10018, 18)
    hc = pair_diag(J, z_table(18))
    rss_after_diag = _rss_kib()
    t0 = time.perf_counter()
    gs = _ground(hc, 0.5, 18)
    eig_s = time.perf_counter() - t0
    rss_after_eig = _rss_kib()
    dummy = np.full(1 << 18, 2.0 ** (-0.5 * 18), dtype=np.float64)
    logical = Logical(18, 10018, 0.5, dummy)
    x = x0_center4("Nov3rd", 1, 0, None)
    t1 = time.perf_counter()
    psi = logical.nov3(x, 1)
    e = logical.energy(psi)
    energy_s = time.perf_counter() - t1
    rss_after_energy = _rss_kib()
    peak_gib = rss_after_energy / (1024.0 * 1024.0)
    allowed = bool(
        eig_s < 30.0
        and energy_s < 0.5
        and peak_gib < 4.0
        and gs.size == (1 << 18)
        and hc.size == (1 << 18)
    )
    return {
        "seed": 10018,
        "N": 18,
        "dim": 1 << 18,
        "dense_matrix": False,
        "eigsh_seconds": eig_s,
        "nov3_L1_seconds": energy_s,
        "nov3_L1_energy": float(e),
        "rss_kib_start": rss0,
        "rss_kib_after_diag": rss_after_diag,
        "rss_kib_after_eigsh": rss_after_eig,
        "rss_kib_after_energy": rss_after_energy,
        "peak_rss_gib": peak_gib,
        "cpu_seconds": time.process_time() - t_cpu0,
        "n18_allowed": allowed,
        "gate": {"eigsh_lt_30s": eig_s < 30.0, "energy_lt_0.5s": energy_s < 0.5, "rss_lt_4GiB": peak_gib < 4.0},
    }


def _recipe() -> dict:
    x_n = x0_center4("Nov3rd", 1, 0, None)
    x_q = x0_qaoa_center4(1, 0, None)
    same = np.array_equal(x_q, x0_equal4("QAOA", 1, 0, None))
    return {
        "nov3_center": [float(v) for v in x_n],
        "qaoa_center": [float(v) for v in x_q],
        "qaoa_center_is_historical_equal4_start0": bool(same),
        "exact_v4_recipe": bool(np.array_equal(x_n, np.array([0.25, 0.0, 0.0, -0.5, 0.0, 0.0])) and np.array_equal(x_q, np.array([0.25, 0.25]))),
    }


def main() -> None:
    if any(s in FORBIDDEN for s in DEV):
        raise RuntimeError("dev panel overlaps a forbidden band")
    recipe = _recipe()
    t_wall = time.perf_counter()
    probe = _probe()
    probe["wall_seconds"] = time.perf_counter() - t_wall
    payload = {
        "ok": bool(recipe["exact_v4_recipe"] and recipe["qaoa_center_is_historical_equal4_start0"]),
        "prereg_sha256": hashlib.sha256(PREREG.read_bytes()).hexdigest(),
        "historical_queries_preserved": 1679360,
        "forbidden_unread": True,
        "dev_seeds": DEV,
        "recipe": recipe,
        "probe": probe,
        "fidelity_is_posthoc": True,
        "no_spsa_in_this_module": True,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    if not payload["ok"]:
        raise SystemExit(1)
    if not probe["n18_allowed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
