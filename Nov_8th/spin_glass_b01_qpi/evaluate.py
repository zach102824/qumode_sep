"""Fidelity after energy selection. Not imported by the optimizer."""
from __future__ import annotations

import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "spin_glass_b01_fid"))
import packing_fiedler  # noqa: E402

sys.path.insert(0, str(HERE.parent / "spin_glass_r4"))
import run_sg_r4 as m4  # noqa: E402

PREREG = json.loads((HERE / "PREREG.json").read_text())
m2 = m4.m2
m4.BASE.update({
    "nodata_pack": "nodata", "K_pack": "K", "KM_pack": "KM",
    "nodata_fid": "nodata", "K_fid": "K", "KM_fid": "KM",
})


def use_for(inst, inst_pack, inst_fid, fam):
    if str(fam).endswith("_fid"):
        return inst_fid
    if str(fam).endswith("_pack"):
        return inst_pack
    return inst


def eval_instance(key):
    N, k = key
    reps = list(range(PREREG["repeats_primary"])) + list(PREREG["qaoa_extra_repeats"])
    jobs = []
    for r in reps:
        p = HERE / "jobs" / f"n{N}_k{k}_r{r}.json"
        if p.exists():
            jobs.append(json.loads(p.read_text()))
    if not jobs:
        return None
    inst = m4.m3.instance(N, k)
    if inst.B != 0.1:
        raise SystemExit("evaluation Hamiltonian is not B=0.1")
    part, score = m4.best_packing(inst.J)
    fid, fscore = packing_fiedler.fiedler_packing(inst.J)
    j0 = jobs[0]
    if part != j0["packing_abs"] or abs(score - j0["packing_score_abs"]) > 1e-9:
        raise SystemExit(f"|J|-max packing drift N={N} k={k}")
    if fid != j0["packing_fiedler"] or abs(fscore - j0["packing_score_fiedler"]) > 1e-9:
        raise SystemExit(f"Fiedler packing drift N={N} k={k}")
    inst_pack = m4.with_packing(inst, part)
    inst_fid = m4.with_packing(inst, fid)
    gd = m2.ground_data(inst)
    out = {"N": N, "k": k, "B": 0.1, "n_sub": gd["n_sub"], "E0": gd["E0"],
           "even_ground_in_window": bool(gd["even_ground_in_window"]),
           "packing_abs": part, "packing_fiedler": fid,
           "statuses": {str(j["r"]): j["status"] for j in jobs}, "cells": []}
    for j in jobs:
        for s in j["selected"]:
            fam, L = s["family"], s["L"]
            if fam == "KM_gl":
                raise SystemExit("per-layer state in a historical-B job")
            use = use_for(inst, inst_pack, inst_fid, fam)
            _, _, psi = m4.energy_of(use, fam, np.array(s["x_final"]), L)
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
    with Pool(int(PREREG["workers"])) as pool:
        done = [x for x in pool.map(eval_instance, keys, chunksize=1) if x]
    print("evaluated", len(done))


if __name__ == "__main__":
    main()
