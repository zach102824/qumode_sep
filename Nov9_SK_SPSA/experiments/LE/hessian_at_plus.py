"""DIAGNOSTIC ONLY: finite-difference Hessian of the energy at the |+> point
(gamma=0, theta=0, alpha=0, s=0 for Nov9th; all angles 0 for QAOA). A negative eigenvalue means
an energy-descent direction out of |+> exists at second order."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import core as C  # noqa: E402

OUT = HERE.parents[1] / "results" / "LE"


def hessian(f, n, h=1e-3):
    x0 = np.zeros(n)
    e0 = f(x0)
    H = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            ei, ej = np.eye(n)[i] * h, np.eye(n)[j] * h
            v = (f(x0 + ei + ej) - f(x0 + ei - ej) - f(x0 - ei + ej) + f(x0 - ei - ej)) / (4 * h * h)
            H[i, j] = H[j, i] = v
    g = np.array([(f(x0 + np.eye(n)[i] * h) - f(x0 - np.eye(n)[i] * h)) / (2 * h) for i in range(n)])
    return e0, g, H


def main():
    res = {}
    for N in (6, 9):
        for seed in (10000, 10001, 10002, 10003):
            J = C.make_J(seed, N)
            for b in (1.0, 1.5):
                for L in (1, 2, 3):
                    for model in ("QAOA", "B_strict", "leaky_Uz"):
                        if model == "QAOA":
                            sim = C.QAOA(J, b, L)
                        else:
                            sim = C.Nov9(J, b, L, **({"strict": True} if model == "B_strict" else {"nf": 16}))
                        n = sim.nparam()
                        e0, g, H = hessian(sim.energy, n)
                        w = np.linalg.eigvalsh(H)
                        key = f"N{N}_s{seed}_b{b}_L{L}_{model}"
                        res[key] = {"E_at_point": e0, "max_abs_grad": float(np.max(np.abs(g))),
                                    "min_hess_eig": float(w[0]), "max_hess_eig": float(w[-1])}
                        print(key, {k: f"{v:.4g}" for k, v in res[key].items()}, flush=True)
    (OUT / "hessian_at_plus.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
