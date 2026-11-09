"""Post-selection evaluation. Ground-state data is used only here."""
from __future__ import annotations

import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_sg_r5 as m4  # noqa: E402

m2 = m4.m2
m3 = m4.m3


def eval_instance(key):
    N, k = key
    jobs = []
    for r in range(m4.PREREG["repeats"]):
        p = HERE / "jobs" / f"n{N}_k{k}_r{r}.json"
        if p.exists():
            jobs.append(json.loads(p.read_text()))
    if not jobs:
        return None
    inst = m3.instance(N, k)
    part, score = m4.best_packing(inst.J)
    if part != jobs[0]["packing_abs"] or abs(score - jobs[0]["packing_score_abs"]) > 1e-9:
        raise SystemExit(f"packing drift N={N} k={k}")
    inst_pack = m4.with_packing(inst, part)
    gd = m2.ground_data(inst)
    out = {"N": N, "k": k, "n_sub": gd["n_sub"], "E0": gd["E0"], "E0_even": gd["E0_even"],
           "even_ground_in_window": bool(gd["even_ground_in_window"]),
           "packing_abs": part, "packing_interleaved": jobs[0]["packing_interleaved"],
           "statuses": {j["r"]: j["status"] for j in jobs}, "cells": []}
    for j in jobs:
        for s in j["selected"]:
            fam, L = s["family"], s["L"]
            use = inst_pack if m4.packed(fam) else inst
            _, _, psi = m4.energy_of(use, fam, np.array(s["x_final"]), L)
            met = m2.metrics(inst, gd, psi)
            A = 2 * np.pi / inst.sJ
            nd = L if fam == "qaoa" else (2 + L if fam == "KM_gl" else m2.NHEAD[m4.kind(fam)])
            out["cells"].append({"r": j["r"], "family": fam, "L": L, "E": s["E"], "x_final": s["x_final"],
                                 "label": s["label"], "root": s["root"], "nested_anc": s["nested_anc"],
                                 "pg_inf": s["pg_inf"], "touch_bound": s["touch_bound"], "n_param": s["n_param"],
                                 "max_diag_frac_of_box": float(np.max(np.abs(s["x_final"][:nd])) / A),
                                 **met})
    (HERE / "eval").mkdir(exist_ok=True)
    (HERE / "eval" / f"n{N}_k{k}.json").write_text(json.dumps(out) + "\n")
    return key


def main():
    keys = [(N, k) for N in (12, 9, 6) for k in m4.PREREG["k_confirm"]]
    with Pool(int(m4.PREREG["workers"])) as pool:
        done = [x for x in pool.map(eval_instance, keys, chunksize=1) if x]
    print("evaluated", len(done))


if __name__ == "__main__":
    main()
