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

sys.path.insert(0, str(HERE))
import window_rule  # noqa: E402

PREREG = json.loads((HERE / "PREREG.json").read_text())
m2 = m4.m2
TOL = float(PREREG["diagnosis"]["energy_legal_tol"])


def interleaved(N):
    m = int(N) // 3
    return [[a, a + m, a + 2 * m] for a in range(m)]


def add_cell(out, inst, gd, s, fam, r):
    L = s["L"]
    if fam == "KM_gl" or str(fam).endswith("_pack") or str(fam).endswith("_fid"):
        raise SystemExit("forbidden family in a historical-B job")
    circuit = "KM" if fam in ("KM_fsel", "KM_fmax") else fam
    _, _, psi = m4.energy_of(inst, circuit, np.asarray(s["x_final"], float), L)
    met = m2.metrics(inst, gd, psi)
    out["cells"].append({"r": r, "family": fam, "L": L, "E": s["E"],
                         "x_final": s["x_final"], "label": s["label"], "root": s["root"],
                         "nested_anc": s["nested_anc"], "pg_inf": s["pg_inf"],
                         "touch_bound": s["touch_bound"], "n_param": s["n_param"],
                         "n_eval": s["n_eval"], "capped": s["capped"], **met})


def km_l6_rows(N, k, inst, gd):
    rows = []
    for r in (0, 1):
        path = HERE / "logs" / f"n{N}_k{k}_r{r}.jsonl"
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if row["family"] == "KM_gl" or str(row["family"]).endswith(("_pack", "_fid")):
                raise SystemExit("forbidden family in a log")
            if row["family"] != "KM" or int(row["L"]) != 6:
                continue
            _, _, psi = m4.energy_of(inst, "KM", np.asarray(row["x_final"], float), 6)
            met = m2.metrics(inst, gd, psi)
            rows.append({**row, "F": met["F_sub"], "metrics": met})
    return rows


def scored_row(row):
    met = row["metrics"]
    return {"r": int(row["r"]), "family": "KM", "L": 6, "E": row["E"],
            "x_final": row["x_final"], "label": row["label"], "root": row["root"],
            "nested_anc": row["nested_anc"], "pg_inf": row["pg_inf"],
            "touch_bound": row["touch_bound"], "n_param": row["n_param"],
            "n_eval": row["n_eval"], "capped": row["capped"], **met}


def eval_instance(key):
    N, k = key
    if k in (100, 101) or k not in PREREG["k_confirm"]:
        raise SystemExit("held-out or out-of-window index reached evaluation")
    path = HERE / "jobs" / f"n{N}_k{k}.json"
    if not path.exists():
        return None
    job = json.loads(path.read_text())
    inst = m4.m3.instance(N, k)
    if inst.B != 0.1:
        raise SystemExit("evaluation Hamiltonian is not B=0.1")
    if job["packing_interleaved"] != interleaved(N):
        raise SystemExit(f"packing drift N={N} k={k}")
    if abs(float(m2.PREREG["F_sub_tol"]) - 1e-6) > 1e-15:
        raise SystemExit("F_sub tolerance is not 1e-6")
    gd = m2.ground_data(inst)
    statuses = {r: rec["status"] for r, rec in job["repeats"].items()}
    out = {"N": N, "k": k, "B": 0.1, "n_sub": gd["n_sub"], "E0": gd["E0"],
           "even_ground_in_window": bool(gd["even_ground_in_window"]),
           "packing_interleaved": interleaved(N),
           "statuses": statuses, "cells": []}
    for r, rec in job["repeats"].items():
        for s in rec["selected"]:
            add_cell(out, inst, gd, s, s["family"], int(r))
    both = all(statuses.get(str(r)) == "complete" for r in (0, 1))
    if both:
        raw = km_l6_rows(N, k, inst, gd)
        if len(raw) < 7:
            raise SystemExit(f"too few L=6 KM starts N={N} k={k}: {len(raw)}")
        qaoa = [c for c in out["cells"] if c["family"] == "qaoa" and c["L"] == 6 and c["r"] in (0, 1)]
        claim_cells = [c for c in out["cells"] if c["family"] == "KM" and c["L"] == 6 and c["r"] in (0, 1)]
        if len({c["r"] for c in qaoa}) < 2 or len({c["r"] for c in claim_cells}) < 2:
            raise SystemExit(f"L=6 pair is missing N={N} k={k}")
        e_q = min(qaoa, key=lambda c: (c["E"], c["r"], c["label"]))
        claim = min(claim_cells, key=lambda c: (c["E"], c["r"], c["label"]))
        best = window_rule.energy_best(raw)
        if abs(float(claim["E"]) - float(best["E"])) > 1e-8:
            raise SystemExit(f"energy-only state drifted N={N} k={k}")
        chosen = window_rule.pick(raw, e_q["E"], TOL)
        if chosen is None:
            matches = [row for row in raw if int(row["r"]) == int(claim["r"]) and row["label"] == claim["label"]
                       and abs(float(row["E"]) - float(claim["E"])) <= 1e-8]
            if len(matches) != 1:
                raise SystemExit(f"energy-only row is not unique N={N} k={k}")
            chosen = matches[0]
        top = window_rule.unrestricted(raw)
        pool_n = len(window_rule.legal_pool(raw, e_q["E"], TOL))
        fsel = scored_row(chosen)
        fsel["family"] = "KM_fsel"
        fmax = scored_row(top)
        fmax["family"] = "KM_fmax"
        out["cells"].append(fsel)
        out["cells"].append(fmax)
        out["fsel"] = {
            "pool_n": pool_n,
            "n_km_l6": len(raw),
            "E_qaoa": e_q["E"],
            "differs": bool(fsel["F_sub"] > claim["F_sub"] + 1e-6 or abs(fsel["E"] - claim["E"]) > 1e-8),
            "F_energy": claim["F_sub"],
            "F_diag": fsel["F_sub"],
            "E_energy": claim["E"],
            "E_diag": fsel["E"],
            "label_energy": claim["label"],
            "r_energy": int(claim["r"]),
            "label_diag": fsel["label"],
            "r_diag": int(fsel["r"]),
            "unrestricted_differs": bool(fmax["F_sub"] > claim["F_sub"] + 1e-6 or abs(fmax["E"] - claim["E"]) > 1e-8),
            "F_unrestricted": fmax["F_sub"],
            "E_unrestricted": fmax["E"],
        }
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
