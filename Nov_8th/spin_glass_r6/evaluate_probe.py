"""Fidelity of the probe states, after energy selection. Not an input to the gate."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from km_original import m2, m3

HERE = Path(__file__).resolve().parent


def f_of(inst, gd, fam, x, L):
    _, _, psi = m2.energy_grad(inst, fam, np.asarray(x, float), L)
    return m2.metrics(inst, gd, psi)


def main():
    raw = json.loads((HERE / "probe_raw.json").read_text())
    rows = []
    cache = {}
    for c in raw["cells"]:
        key = (c["N"], c["k"])
        if key not in cache:
            inst = m3.instance(*key)
            cache[key] = (inst, m2.ground_data(inst))
        inst, gd = cache[key]
        best = min(c["starts"], key=lambda s: s["E"])
        met_new = f_of(inst, gd, "KM", best["x_final"], 6)
        met_km = f_of(inst, gd, "KM", c["km_x_stored"], 6)
        met_q = f_of(inst, gd, "qaoa", c["qaoa_x"], 6)
        rows.append({
            "N": c["N"], "k": c["k"], "in_gate": c["in_gate"], "label": best["label"],
            "E_new": met_new["E_check"], "E_km": met_km["E_check"], "E_qaoa": met_q["E_check"],
            "F_new": met_new["F_sub"], "F_km": met_km["F_sub"], "F_qaoa": met_q["F_sub"],
            "dE_vs_km": met_new["E_check"] - met_km["E_check"],
            "dF_new": met_new["F_sub"] - met_q["F_sub"],
            "dF_km": met_km["F_sub"] - met_q["F_sub"],
        })
    (HERE / "probe_fidelity.json").write_text(json.dumps(rows, indent=2) + "\n")
    for r in rows:
        print(f"N{r['N']} k{r['k']} dE {r['dE_vs_km']:+.4f} dF {r['dF_km']:+.3f} -> {r['dF_new']:+.3f} {r['label']}")


if __name__ == "__main__":
    main()
