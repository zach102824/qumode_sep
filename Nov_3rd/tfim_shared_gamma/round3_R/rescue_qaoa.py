"""Extra QAOA starts at matched depths after A_sub+M beat them on both metrics.

4x16 starts, Fourier plus perturbations, at p=9,10,11 and a new p=12.
Updates the saved cells and rewrites RESULTS.md.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import circuit
import model
import optimize
import run_round3_r as runner

RESULTS = runner.RESULTS
PERIODS = runner.PERIODS


def load_cell(n, L, name):
    path = RESULTS / "cells" / f"N{n}_L{L}_{name}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


def save_cell(cell):
    runner.save_json(RESULTS / "cells" / f"N{cell['n']}_L{cell['L']}_{cell['name']}.json", cell)


def eval_qaoa(x, L, sys, gs, E0):
    indices = model.precompute(model.packing("contiguous", 12)[0], 12)
    psi, E, g, var = circuit.simulate("qaoa", x, L, sys["hc"], sys["fk"], np.zeros_like(sys["hc"]), sys["B"], 12, indices, True)
    return psi, float(E), g, float(var)


def rescue_depth(L, sys, gs, E0, n_extra, rng):
    cell = load_cell(12, L, "qaoa")
    if cell is None:
        parent = load_cell(12, L - 1, "qaoa")
        if parent is None:
            raise RuntimeError(f"no parent for qaoa L={L}")
        x0 = optimize.continuation("qaoa", parent["x"], L - 1, L, sys["B"], PERIODS, "cubic")
        cell = {
            "name": "qaoa", "kind": "qaoa", "packing": "contiguous",
            "packing_sha256": parent["packing_sha256"], "manifest_sha256": parent["manifest_sha256"],
            "hc_sha256": parent["hc_sha256"], "n": 12, "L": L, "skipped": False,
            "n_params": 2 * L, "matched_p": L, "nfev_sum": 0, "n_starts": 0,
            "E": 1e9, "starts": [],
        }
        base_x = np.asarray(x0, float)
    else:
        base_x = np.asarray(cell["x"], float)
    bounds = circuit.bounds_for("qaoa", L)
    starts = [{"name": "rescue_current", "x0": optimize.clip(base_x, bounds)}]
    ang = circuit.unpack("qaoa", base_x, L)
    xf = np.empty(2 * L)
    xf[0::2] = optimize.fourier_schedule(ang["gammas"])
    xf[1::2] = optimize.fourier_schedule(ang["betas"])
    starts.append({"name": "rescue_fourier", "x0": optimize.clip(xf, bounds)})
    for s, count in ((0.05, 8), (0.15, 8), (0.4, 8)):
        for k in range(count):
            starts.append({"name": f"rescue_{s}_{k}", "x0": optimize.clip(base_x + rng.normal(0, s, size=base_x.size), bounds)})
    while len(starts) < n_extra:
        x = np.zeros(2 * L)
        x[0::2] = rng.uniform(-1.2, 1.2, size=L)
        x[1::2] = rng.uniform(-1.0, 1.0, size=L)
        starts.append({"name": f"rescue_rand{len(starts)}", "x0": x})
    indices = model.precompute(model.packing("contiguous", 12)[0], 12)

    def energy_grad(x):
        _, E, g, _ = circuit.simulate("qaoa", x, L, sys["hc"], sys["fk"], np.zeros_like(sys["hc"]), sys["B"], 12, indices, True)
        return E, g

    print(f"rescue QAOA p={L} starts={len(starts)}", flush=True)
    ran = optimize.run_starts(energy_grad, starts, bounds, 800, 800)
    improved = ran["best"]["E"] < cell.get("E", 1e9) - 1e-12
    cell.setdefault("rescue_rounds", [])
    cell["rescue_rounds"].append({"n_extra": len(starts), "best_E": ran["E_best"] if "E_best" in ran else ran["best"]["E"], "improved_vs_previous_saved": bool(improved)})
    cell["nfev_sum"] = int(cell.get("nfev_sum", 0) + ran["nfev_sum"])
    cell["n_starts"] = int(cell.get("n_starts", 0) + len(starts))
    energies = [r["E"] for r in cell.get("starts", [])] + [r["E"] for r in ran["starts"]]
    cell["starts"] = cell.get("starts", []) + ran["starts"]
    q1, q3 = np.percentile(energies, [25, 75])
    cell["E_median"] = float(np.median(energies))
    cell["E_iqr"] = float(q3 - q1)
    cell["median_error"] = float(cell["E_median"] - E0)
    if improved or cell.get("E", 1e9) > 1e8:
        x = np.asarray(ran["best"]["x"], float)
        psi, E, g, var = eval_qaoa(x, L, sys, gs, E0)
        cell["pre_rescue_E"] = cell.get("E")
        cell["E"] = E
        cell["E0"] = E0
        cell["energy_error"] = float(E - E0)
        cell["F"] = circuit.fidelity(psi, gs)
        cell["var_H"] = var
        cell["x"] = [float(v) for v in x]
        cell["angles"] = circuit.pack_angles(circuit.unpack("qaoa", x, L))
        cell["grad_norm"] = float(np.linalg.norm(g))
        cell["parity"] = circuit.parity_expectation(psi)
        cell["reflection"] = circuit.reflection_expectation(psi, circuit.reflection_indices(12))
        cell["state_hash"] = __import__("hashlib").sha256(np.ascontiguousarray(psi).tobytes()).hexdigest()
        cell["best_start"] = ran["best"]["name"]
        cell["ledger"] = "packing-independent"
        np.savez_compressed(RESULTS / "states" / f"N12_L{L}_qaoa.npz", psi=psi, x=x)
        print(f"  new best E-E0={cell['energy_error']:.4f} F={cell['F']:.4f}", flush=True)
    else:
        print(f"  no improvement; saved E-E0={cell['energy_error']:.4f}", flush=True)
    save_cell(cell)
    return cell


def main():
    sys = model.system(12)
    spec6 = circuit.dense_spectrum(model.system(6)["hc"], model.system(6)["B"], 6)
    spec = circuit.dense_spectrum(sys["hc"], sys["B"], 12)
    # Only need GS and E0; dense_spectrum recomputes. Acceptable.
    rng = np.random.default_rng(424242)
    for L in (9, 10, 11, 12):
        rescue_depth(L, sys, spec["gs"], spec["E0"], 64, rng)
    # Rebuild report inputs from disk.
    cells = {}
    for path in (RESULTS / "cells").glob("*.json"):
        cell = json.loads(path.read_text())
        cells[(cell["n"], cell["L"], cell["name"])] = cell
    diagnostics = json.loads((RESULTS / "packing_diagnostics.json").read_text())
    meta = json.loads((RESULTS / "run_meta.json").read_text())
    q_n = sum(c["nfev_sum"] for (n, L, name), c in cells.items() if name == "qaoa" and n == 12)
    meta["qaoa_nfev"] = q_n
    meta["qaoa_budget_covers_main"] = q_n >= meta["main_A_sub_nfev"]
    meta["rescue"] = "64-start Fourier/perturbation rescue at QAOA p=9,10,11,12 after A_sub_M beat matched QAOA on both metrics at L=5 and L=6"
    runner.save_json(RESULTS / "run_meta.json", meta)
    built = {
        "packing_sha256": meta["packing_sha256"],
        "B12": "-0.3444",
        "B6": "-0.1708",
        "E0_12": spec["E0"],
        "E0_6": spec6["E0"],
        "gap12": spec["full_gap"],
        "gap6": spec6["full_gap"],
    }
    # N6 E0 from the saved cell is exact enough; gap6 is reprinted from the first report.
    runner.write_report(cells, built, [], meta, diagnostics)
    runner.make_figures(cells)
    print("qaoa nfev", q_n, "covers", meta["qaoa_budget_covers_main"])


if __name__ == "__main__":
    main()
