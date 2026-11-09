"""N=18 L=2/3 resource gate. Reuses v5 B*. No SPSA in this module."""
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

from Nov3_INIT.alloc import iterations_equal4, x0_center4, x0_equal4, x0_qaoa_center4
from Nov3_INIT.circuit import Logical

ROOT = Path(__file__).resolve().parent
PREREG = ROOT / "PREREG_V6.md"
OUT = ROOT / "results" / "v6" / "CONFORMANCE.json"
V5_INST = ROOT / "results" / "v5" / "instances"
DEV = list(range(10018, 10026))


def _rss_kib() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)


def _check_instance(seed: int) -> dict:
    meta_path = V5_INST / f"N18_s{seed}.json"
    npz_path = V5_INST / f"N18_s{seed}.npz"
    if not meta_path.exists() or not npz_path.exists():
        raise RuntimeError(f"missing reused B* for seed {seed}")
    meta = json.loads(meta_path.read_text())
    gs = np.load(npz_path)["gs"]
    if int(meta["N"]) != 18 or int(meta["seed"]) != int(seed):
        raise RuntimeError(f"instance identity mismatch seed {seed}")
    if meta.get("cut") != list(range(9)):
        raise RuntimeError(f"cut mismatch seed {seed}")
    if gs.size != (1 << 18):
        raise RuntimeError(f"gs length {gs.size} seed {seed}")
    if not meta.get("interior"):
        raise RuntimeError(f"B* not interior seed {seed}")
    if meta["gap"] <= 1e-8 or meta["gs_parity"] < 1.0 - 1e-6 or meta["gs_min"] <= -1e-8 or meta["residual"] > 1e-8:
        raise RuntimeError(f"B* contract failed on reuse seed {seed}")
    if meta["delta_odd"] <= 0.0 or meta["delta_even"] <= 0.0:
        raise RuntimeError(f"sector gap failed on reuse seed {seed}")
    return {
        "seed": int(seed),
        "Bstar": float(meta["Bstar"]),
        "gap": float(meta["gap"]),
        "reused": True,
        "recomputed": False,
        "source": "results/v5/instances",
    }


def _probe() -> dict:
    reused = [_check_instance(s) for s in DEV]
    meta = json.loads((V5_INST / "N18_s10018.json").read_text())
    gs = np.load(V5_INST / "N18_s10018.npz")["gs"]
    rss0 = _rss_kib()
    cpu0 = time.process_time()
    logical = Logical(18, 10018, float(meta["Bstar"]), gs)
    rss_loaded = _rss_kib()
    times = {}
    energies = {}
    for L in (2, 3):
        x = x0_center4("Nov3rd", L, 0, None)
        t0 = time.perf_counter()
        psi = logical.nov3(x, L)
        e = logical.energy(psi)
        times[f"nov3_L{L}_seconds"] = time.perf_counter() - t0
        energies[f"nov3_L{L}_energy"] = float(e)
        if not np.isfinite(e):
            raise RuntimeError(f"non-finite L={L} energy")
    rss1 = _rss_kib()
    l2 = times["nov3_L2_seconds"]
    l3 = times["nov3_L3_seconds"]
    peak_gib = rss1 / (1024.0 * 1024.0)
    four_gib = 4.0 * peak_gib
    cell_l3_q1024 = 3072 * l3
    allowed = bool(
        l2 < 0.8
        and l3 < 1.2
        and peak_gib < 2.0
        and four_gib < 8.0
        and cell_l3_q1024 < 3600.0
    )
    return {
        "seed": 10018,
        "N": 18,
        "dim": 1 << 18,
        "dense_matrix": False,
        "bstar_reused": True,
        "bstar_recomputed": False,
        "n_instances_valid": len(reused),
        "instances": reused,
        **times,
        **energies,
        "rss_kib_start": rss0,
        "rss_kib_after_load": rss_loaded,
        "rss_kib_after_energy": rss1,
        "peak_rss_gib": peak_gib,
        "projected_four_worker_rss_gib": four_gib,
        "projected_L3_q1024_cell_seconds": cell_l3_q1024,
        "cpu_seconds": time.process_time() - cpu0,
        "T_L2_q512": iterations_equal4(1024),
        "T_L3_q1024": iterations_equal4(3072),
        "n18_l23_allowed": allowed,
        "gate": {
            "L2_energy_lt_0.8s": l2 < 0.8,
            "L3_energy_lt_1.2s": l3 < 1.2,
            "rss_lt_2GiB": peak_gib < 2.0,
            "four_worker_rss_lt_8GiB": four_gib < 8.0,
            "L3_q1024_cell_lt_3600s": cell_l3_q1024 < 3600.0,
        },
    }


def _recipe() -> dict:
    x_n = x0_center4("Nov3rd", 3, 0, None)
    x_q = x0_qaoa_center4(3, 0, None)
    expected_n = np.array([0.25, 0.0, 0.0] + [-0.5, 0.0, 0.0] * 3)
    expected_q = np.full(6, 0.25)
    return {
        "nov3_center_L3": [float(v) for v in x_n],
        "qaoa_center_L3": [float(v) for v in x_q],
        "qaoa_center_is_historical_equal4_start0": bool(np.array_equal(x_q, x0_equal4("QAOA", 3, 0, None))),
        "exact_v4_recipe": bool(np.array_equal(x_n, expected_n) and np.array_equal(x_q, expected_q)),
    }


def main() -> None:
    recipe = _recipe()
    t_wall = time.perf_counter()
    probe = _probe()
    probe["wall_seconds"] = time.perf_counter() - t_wall
    payload = {
        "ok": bool(recipe["exact_v4_recipe"] and recipe["qaoa_center_is_historical_equal4_start0"] and probe["n_instances_valid"] == 8),
        "prereg_sha256": hashlib.sha256(PREREG.read_bytes()).hexdigest(),
        "historical_queries_preserved": 1703936,
        "forbidden_unread": True,
        "dev_seeds": DEV,
        "recipe": recipe,
        "probe": probe,
        "fidelity_is_posthoc": True,
        "no_spsa_in_this_module": True,
        "not_retuned_from_n18_l1": True,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    if not payload["ok"]:
        raise SystemExit(1)
    if not probe["n18_l23_allowed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
