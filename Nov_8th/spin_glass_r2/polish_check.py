"""Post-hoc diagnostic: tight re-polish of the selected L=6 cells. Not used for selection or tables."""
from __future__ import annotations

import json
import sys
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import run_sg_r2 as m  # noqa: E402


def polish(cell):
    N, k, fam, L = cell["N"], cell["k"], cell["family"], cell["L"]
    doc, J = m.r1.load_instance(N, k)
    inst = m.r1.Inst(N, k, J, doc)
    inst.sJ = m.s_J(J)
    x0 = np.array(cell["x_final"]) if cell["coords"] == "r2_linear" else m.r1_to_r2(fam, cell["x_final"], L)
    n = [0]

    def f(x):
        n[0] += 1
        E, g, _ = m.energy_grad(inst, fam, x, L)
        return E, g

    res = minimize(f, x0, jac=True, method="L-BFGS-B", bounds=m.bounds(inst, fam, L),
                   options={"maxiter": 5000, "maxfun": 5000, "ftol": 0.0, "gtol": 1e-10})
    g = f(res.x)[1]
    return {"N": N, "k": k, "family": fam, "L": L, "E_selected": cell["E"], "E_polished": float(res.fun),
            "dE": float(res.fun - cell["E"]), "n_eval": n[0], "message": str(res.message),
            "grad_inf_selected": float(np.max(np.abs(m.energy_grad(inst, fam, x0, L)[1]))),
            "grad_inf_polished": float(np.max(np.abs(g)))}


def main():
    cells = json.loads((HERE / "selected.json").read_text())
    todo = [c for c in cells if c["L"] == 6 and c["family"] in ("KM", "K", "qaoa")]
    with Pool(4) as p:
        rows = p.map(polish, todo, chunksize=1)
    starts = [json.loads(l) for f in sorted((HERE / "logs").glob("*.jsonl")) for l in f.read_text().splitlines()]
    starts = [r for r in starts if "cell_summary" not in r]
    winners = Counter()
    distinct = {}
    for c in cells:
        winners[(c["family"], c["label"].split("_")[0] if not c["label"].startswith("p") else c["label"].split("_", 1)[1])] += 1
    for N in (6, 9, 12):
        for k in range(4):
            for fam in ("KM", "qaoa"):
                Es = sorted(r["E"] for r in starts if r["N"] == N and r["k"] == k and r["family"] == fam and r["L"] == 6)
                levels = [Es[0]]
                for e in Es[1:]:
                    if e - levels[-1] > 1e-6:
                        levels.append(e)
                distinct[f"N{N}_k{k}_{fam}"] = {"n_starts": len(Es), "n_distinct_E_levels": len(levels),
                                                "best": Es[0], "n_at_best_1e-6": sum(1 for e in Es if e - Es[0] <= 1e-6)}
    out = {"polish_L6": rows, "max_abs_dE": max(abs(r["dE"]) for r in rows),
           "winning_start_kinds": {f"{a}|{b}": v for (a, b), v in sorted(winners.items())},
           "L6_start_spread": distinct}
    (HERE / "polish_check.json").write_text(json.dumps(out, indent=2) + "\n")
    for r in rows:
        print(r["N"], r["k"], r["family"], f"{r['dE']:+.2e}", f"{r['grad_inf_selected']:.1e}", f"{r['grad_inf_polished']:.1e}", r["n_eval"])
    print("max |dE|", out["max_abs_dE"])
    print(out["winning_start_kinds"])
    for kk, v in distinct.items():
        print(kk, v)


if __name__ == "__main__":
    main()
