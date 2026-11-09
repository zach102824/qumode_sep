"""Post-selection evaluation with the main simulator. Exact ground data is used here only."""
from __future__ import annotations

import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_sg_r3 as m3  # noqa: E402

m2 = m3.m2


def eval_instance(key):
    N, k = key
    jobs = []
    for r in range(m3.PREREG["repeats"]):
        p = HERE / "jobs" / f"n{N}_k{k}_r{r}.json"
        if p.exists():
            jobs.append(json.loads(p.read_text()))
    if not jobs:
        return None
    inst = m3.instance(N, k)
    gd = m2.ground_data(inst)
    dim = 1 << N
    flip = np.arange(dim) ^ (dim - 1)
    win_par = [float(np.vdot(gd["sub"][:, i], gd["sub"][flip, i]).real) for i in range(gd["n_sub"])]
    out = {"N": N, "k": k, "n_sub": gd["n_sub"], "E0": gd["E0"], "E0_even": gd["E0_even"],
           "even_ground_in_window": gd["even_ground_in_window"], "window_spin_parity": win_par,
           "statuses": {j["r"]: j["status"] for j in jobs}, "cells": []}
    for j in jobs:
        for s in j["selected"]:
            fam, L = s["family"], s["L"]
            _, _, psi = m2.energy_grad(inst, m3.circ(fam), np.array(s["x_final"]), L)
            met = m2.metrics(inst, gd, psi)
            A = 2 * np.pi / inst.sJ
            c = m3.circ(fam)
            nd = L if c == "qaoa" else m2.NHEAD[c]
            out["cells"].append({"r": j["r"], "family": fam, "L": L, "E": s["E"], "x_final": s["x_final"],
                                 "label": s["label"], "root": s["root"], "nested_anc": s["nested_anc"],
                                 "pg_inf": s["pg_inf"], "touch_bound": s["touch_bound"],
                                 "max_diag_frac_of_box": float(np.max(np.abs(s["x_final"][:nd])) / A),
                                 **met})
    (HERE / "eval").mkdir(exist_ok=True)
    (HERE / "eval" / f"n{N}_k{k}.json").write_text(json.dumps(out) + "\n")
    return key


def main():
    (HERE / "eval").mkdir(exist_ok=True)
    ks = m3.PREREG["k_primary"] + m3.PREREG["k_secondary"]
    keys = [(N, k) for N in (12, 9, 6) for k in ks]
    with Pool(int(m3.PREREG["workers"])) as p:
        done = [x for x in p.map(eval_instance, keys, chunksize=1) if x]
    print("evaluated", len(done))


if __name__ == "__main__":
    main()
