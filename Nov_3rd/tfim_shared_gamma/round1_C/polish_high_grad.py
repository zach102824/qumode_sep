"""Continue saved points whose analytic gradient is still large.

Run from this directory after run_round1.py. Updates the cell json and state
in place when the energy decreases, then rewrites RESULTS.md and the figures.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

import circuit
import hamiltonian
import run_round1

ROOT = Path(__file__).resolve().parent / "results"
GRAD_TRIGGER = 1e-3


def load_ops(n: int, method: str):
    target = json.loads((ROOT / "targets" / f"N{n}.json").read_text())
    arrays = np.load(ROOT / "targets" / f"N{n}_arrays.npz")
    if method == "C_unrestricted":
        extra = np.load(ROOT / "features" / f"N{n}_unrestricted.npz")
        primary = np.load(ROOT / "features" / f"N{n}_primary.npz")
        hc, fk, fs = primary["hc"], extra["fk"], extra["fs"]
    else:
        primary = np.load(ROOT / "features" / f"N{n}_primary.npz")
        hc, fk, fs = primary["hc"], primary["fk"], primary["fs"]
    return target, arrays, hc, fk, fs


def main():
    updated = []
    for path in sorted((ROOT / "cells").glob("*.json")):
        cell = json.loads(path.read_text())
        if cell.get("skipped") or cell.get("grad_norm", 0.0) < GRAD_TRIGGER:
            continue
        n, L, method = int(cell["n"]), int(cell["L"]), cell["method"]
        target, arrays, hc, fk, fs = load_ops(n, method)
        x0 = np.asarray(cell["x"], dtype=np.float64)

        def fun(x, method=method, L=L, hc=hc, fk=fk, fs=fs, B=target["B"], n=n):
            _, energy, grad = circuit.simulate(method, x, L, hc, fk, fs, B, n, True)
            return float(energy), np.asarray(grad, dtype=np.float64)

        res = minimize(
            fun, x0, method="L-BFGS-B", jac=True,
            bounds=circuit.bounds_for(method, L),
            options={"maxiter": 400, "ftol": 1e-15, "gtol": 1e-8, "maxfun": 4000},
        )
        psi, energy, grad = circuit.simulate(method, res.x, L, hc, fk, fs, target["B"], n, True)
        gnorm = float(np.linalg.norm(grad))
        print(f"{method} N{n} L{L}: {cell['energy_error']:.6f} -> {energy - target['E0']:.6f} grad {gnorm:.3e}")
        if energy >= cell["E"] - 1e-8:
            cell["grad_norm"] = gnorm
            cell["polish"] = {"improved": False, "nit": int(res.nit), "nfev": int(res.nfev), "grad_norm": gnorm}
            path.write_text(json.dumps(cell, indent=2) + "\n")
            continue
        can = hamiltonian.canonicalize_vector(psi)
        cell["E"] = float(energy)
        cell["energy_error"] = float(energy - target["E0"])
        cell["F"] = circuit.fidelity(can, arrays["gs"])
        cell["eta"] = circuit.eta_score(energy, target["E0"], target["Emax"])
        cell["x"] = [float(v) for v in res.x]
        cell["angles"] = circuit.pack_angles(circuit.unpack(method, res.x, L))
        cell["grad_norm"] = gnorm
        cell["state_hash"] = hamiltonian.state_hash(can)
        cell["parity_expectation"] = hamiltonian.parity_expectation(can)
        cell["reflection_expectation"] = hamiltonian.reflection_expectation(can, arrays["ridx"])
        cell["E_recompute_minus_optimizer"] = float(energy - float(res.fun))
        cell["nfev_sum"] = int(cell["nfev_sum"] + res.nfev)
        cell["polish"] = {
            "improved": True, "nit": int(res.nit), "nfev": int(res.nfev),
            "success": bool(res.success), "message": str(res.message),
        }
        path.write_text(json.dumps(cell, indent=2) + "\n")
        np.savez_compressed(ROOT / "states" / f"N{n}_L{L}_{method}.npz", psi=can, x=np.asarray(res.x, dtype=np.float64))
        updated.append(path.name)
    cells = {}
    for path in (ROOT / "cells").glob("*.json"):
        cell = json.loads(path.read_text())
        cells[(cell["n"], cell["L"], cell["method"])] = cell
    targets = {n: json.loads((ROOT / "targets" / f"N{n}.json").read_text()) for n in (6, 12)}
    feature_pack = {
        n: {"summary": json.loads((ROOT / "features" / f"N{n}_summary.json").read_text())} for n in (6, 12)
    }
    transfer = json.loads((ROOT / "transfer.json").read_text())
    meta = json.loads((ROOT / "run_meta.json").read_text())
    meta["polish_script"] = "polish_high_grad.py"
    meta["code_hash_at_report"] = run_round1.code_hash()
    (ROOT / "run_meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    run_round1.write_report(cells, targets, feature_pack, transfer, meta, [])
    run_round1.make_figures(cells, targets, feature_pack)
    print("updated", updated)


if __name__ == "__main__":
    main()
