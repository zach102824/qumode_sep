"""Circuit identity for the derived QAOA center, plus the N=15 throughput gate."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np

from Nov3_INIT.alloc import x0_equal4, x0_qaoa_center4
from Nov3_INIT.bstar import _ground
from Nov3_INIT.circuit import Logical
from Nov9_SK_SPSA.experiments.FT.instance import build_instance, make_J, pair_diag, z_table

ROOT = Path(__file__).resolve().parent
PREREG = ROOT / "PREREG_V4.md"
OUT = ROOT / "results" / "v4" / "CONFORMANCE.json"


def _overlap(n: int, L: int) -> dict:
    inst = build_instance(n, 10018, 0.5, role="n3init")
    logical = Logical(n, 10018, 0.5, inst.gs)
    x_nov = np.zeros(3 + 3 * L, dtype=np.float64)
    x_nov[0] = 0.25
    x_nov[3::3] = -0.5
    x_q = x0_qaoa_center4(L, 0, None)
    a = logical.nov3(x_nov, L)
    b = logical.qaoa(x_q, L)
    flipped = np.array(x_q, copy=True)
    flipped[1::2] *= -1.0
    c = logical.qaoa(flipped, L)
    return {
        "N": n,
        "L": L,
        "fidelity": float(abs(np.vdot(a, b)) ** 2),
        "dE": float(logical.energy(a) - logical.energy(b)),
        "phase": float(np.angle(np.vdot(a, b))),
        "flipped_beta_fidelity": float(abs(np.vdot(a, c)) ** 2),
        "qaoa_center": [float(v) for v in x_q],
    }


def _probe() -> dict:
    J = make_J(10018, 15)
    hc = pair_diag(J, z_table(15))
    t0 = time.perf_counter()
    _ground(hc, 0.5, 15)
    eig_s = time.perf_counter() - t0
    logical = Logical(15, 10018, 0.5, np.ones(1 << 15) / np.sqrt(1 << 15))
    x = np.zeros(6, dtype=np.float64)
    x[0] = 0.25
    x[3] = -0.5
    t1 = time.perf_counter()
    logical.nov3(x, 1)
    energy_s = time.perf_counter() - t1
    return {
        "eigsh_seconds": eig_s,
        "nov3_L1_seconds": energy_s,
        "n15_allowed": bool(eig_s < 2.0 and energy_s < 0.05),
        "dense_matrix": False,
    }


def main() -> None:
    rows = [_overlap(6, 1), _overlap(6, 2)]
    identity_ok = all(r["fidelity"] > 1.0 - 1e-8 and abs(r["dE"]) < 1e-8 and r["flipped_beta_fidelity"] < 0.9 for r in rows)
    same = np.array_equal(x0_qaoa_center4(4, 0, None), x0_equal4("QAOA", 4, 0, None))
    probe = _probe()
    payload = {
        "ok": bool(identity_ok and same),
        "prereg_sha256": hashlib.sha256(PREREG.read_bytes()).hexdigest(),
        "overlaps": rows,
        "derived_equals_historical_qaoa_start0": bool(same),
        "probe": probe,
        "fidelity_is_circuit_identity_not_a_selector": True,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    if not payload["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
