"""Fidelity after energy selection. Not imported by the optimizer."""
from __future__ import annotations

import json
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "spin_glass_r4"))
import run_sg_r4 as m4  # noqa: E402

PREREG = json.loads((HERE / "PREREG.json").read_text())
m2 = m4.m2
FORBIDDEN = {"KM_gl", "KM_pack", "KM_fid", "KM_fe", "KM_mm"}


def interleaved(N):
    m = N // 3
    return [[a, a + m, a + 2 * m] for a in range(m)]


def eval_instance(key):
    N, k = key
    if k not in PREREG["k_confirm"]:
        raise SystemExit(f"k={k} is outside the frozen window")
    jobs = []
    for r in range(PREREG["repeats"]):
        p = HERE / "jobs" / f"n{N}_k{k}_r{r}.json"
        if p.exists():
            jobs.append(json.loads(p.read_text()))
    if not jobs:
        return None
    inst = m4.m3.instance(N, k)
    if inst.B != 0.1:
        raise SystemExit("evaluation Hamiltonian is not B=0.1")
    part = interleaved(N)
    if jobs[0]["packing_interleaved"] != part or jobs[0]["packing_abs"] != part:
        raise SystemExit(f"packing is not interleaved at N={N} k={k}")
    gd = m2.ground_data(inst)
    out = {"N": N, "k": k, "B": 0.1, "n_sub": gd["n_sub"], "E0": gd["E0"],
           "even_ground_in_window": bool(gd["even_ground_in_window"]),
           "packing_interleaved": part,
           "statuses": {str(j["r"]): j["status"] for j in jobs}, "cells": []}
    for j in jobs:
        for s in j["selected"]:
            fam, L = s["family"], s["L"]
            if fam in FORBIDDEN or fam not in PREREG["family_order"]:
                raise SystemExit(f"forbidden family {fam} in a window job")
            _, _, psi = m4.energy_of(inst, fam, np.array(s["x_final"]), L)
            met = m2.metrics(inst, gd, psi)
            out["cells"].append({"r": j["r"], "family": fam, "L": L, "E": s["E"],
                                 "x_final": s["x_final"], "label": s["label"], "root": s["root"],
                                 "nested_anc": s["nested_anc"], "pg_inf": s["pg_inf"],
                                 "touch_bound": s["touch_bound"], "n_param": s["n_param"],
                                 "n_eval": s["n_eval"], "capped": s["capped"], **met})
    (HERE / "eval").mkdir(exist_ok=True)
    (HERE / "eval" / f"n{N}_k{k}.json").write_text(json.dumps(out) + "\n")
    return key


def main():
    keys = [(N, k) for N in (12, 9, 6) for k in PREREG["k_confirm"]]
    mp.set_start_method("fork", force=True)
    with mp.Pool(int(PREREG["workers"])) as pool:
        done = [x for x in pool.map(eval_instance, keys, chunksize=1) if x]
    print("evaluated", len(done))


if __name__ == "__main__":
    main()
