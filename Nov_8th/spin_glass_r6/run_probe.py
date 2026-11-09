"""Development probe. Four extra KM starts from the stored round-4 QAOA solution. Energy only."""
from __future__ import annotations

import hashlib
import json
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from km_original import PREREG, assert_original, m2, m3, one_start, qproj_starts

HERE = Path(__file__).resolve().parent
R4 = HERE.parent / "spin_glass_r4" / "jobs"
PREREG_SHA = "933cf6a85e88e64685ba2440b96d3ab8b90fe65353f400137e669c91b4b53352"


def cell_of(N, k, in_gate):
    path = R4 / f"n{N}_k{k}_r0.json"
    job = json.loads(path.read_text())
    if job["status"] != "complete":
        raise SystemExit(f"round-4 job not complete: {path}")
    km = [s for s in job["selected"] if s["family"] == "KM" and s["L"] == 6]
    qaoa = [s for s in job["selected"] if s["family"] == "qaoa" and s["L"] == 6]
    if len(km) != 1 or len(qaoa) != 1:
        raise SystemExit(f"missing L6 selection N={N} k={k}")
    inst = m3.instance(N, k)
    assert_original(inst)
    if inst.doc["J_sha256_f64le"] != job["J_sha256_f64le"]:
        raise SystemExit(f"J drift N={N} k={k}")
    base_e, _, _ = m2.energy_grad(inst, "KM", np.array(km[0]["x_final"], float), 6)
    if abs(base_e - km[0]["E"]) > 1e-6:
        raise SystemExit(f"baseline energy drift N={N} k={k}: {base_e} vs {km[0]['E']}")
    rows = []
    t0 = time.perf_counter()
    for label, x0 in qproj_starts(inst, qaoa[0]["x_final"], 6, 0):
        row = one_start(inst, "KM", x0, 6)
        row.update({"label": label, "x0": [float(v) for v in m2.clip(inst, "KM", x0, 6)]})
        rows.append(row)
    best = min(rows, key=lambda r: r["E"])
    return {
        "N": N, "k": k, "r": 0, "L": 6, "in_gate": in_gate,
        "E_stored": km[0]["E"], "E_baseline": float(base_e),
        "E_qaoa_stored": qaoa[0]["E"],
        "E_best_new": best["E"], "best_label": best["label"],
        "dE": float(best["E"] - base_e),
        "seconds": time.perf_counter() - t0,
        "starts": rows,
        "qaoa_x": qaoa[0]["x_final"],
        "km_x_stored": km[0]["x_final"],
    }


def main():
    digest = hashlib.sha256((HERE / "PREREG.json").read_bytes()).hexdigest()
    if digest != PREREG_SHA:
        raise SystemExit(f"prereg hash {digest} != {PREREG_SHA}")
    spec = PREREG["probe"]
    jobs = [(N, k, True) for N in spec["Ns_gate"] for k in spec["k_gate"]]
    san = spec["sanity_not_in_gate"]
    jobs.append((san["N"], san["k"], False))
    with Pool(4) as pool:
        cells = list(pool.starmap(cell_of, jobs))
    out = {"prereg_sha256": digest, "cells": cells}
    (HERE / "probe_raw.json").write_text(json.dumps(out) + "\n")
    for c in cells:
        print(f"N{c['N']} k{c['k']} dE {c['dE']:+.6f} via {c['best_label']} gate {c['in_gate']}", flush=True)


if __name__ == "__main__":
    main()
