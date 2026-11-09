"""DIAGNOSTIC ONLY (not an optimizer, not a method result): uniform random sampling of the bounded
Nov9th parameter box, to test whether any point beats the |+> energy by a QAOA-like margin.
Also probes the first-order energy slope in alpha at alpha=0 for random (gamma, theta, s).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import core as C  # noqa: E402

OUT = HERE.parents[1] / "results" / "LE"


def main(nsamp=20000):
    res = {}
    rng = np.random.default_rng(20261109)
    for seed in (10000, 10001):
        for b in (1.0, 1.5):
            J = C.make_J(seed, 6)
            g = C.exact_gs(J, b)
            for L in (1, 2, 3):
                for model, kw in (("B_strict", {"strict": True}), ("leaky_Uz", {"nf": 16})):
                    sim = C.Nov9(J, b, L, **kw)
                    n = sim.nparam()
                    lo = np.concatenate([[-2.0], [-1.0, -1.0], np.full(L, -3.0), np.full(3 * L, -np.pi)])
                    hi = -lo
                    best = (np.inf, None)
                    for _ in range(nsamp):
                        x = rng.uniform(lo, hi)
                        e = sim.energy(x)
                        if e < best[0]:
                            best = (e, x)
                    d = sim.diagnostics(best[1], g["gs"])
                    # first-order alpha slope at alpha=0 for random other params
                    slopes = []
                    for _ in range(200):
                        x = rng.uniform(lo, hi)
                        x[3 : 3 + L] = 0.0
                        h = 1e-4
                        xp, xm = x.copy(), x.copy()
                        xp[3 : 3 + L] += h
                        xm[3 : 3 + L] -= h
                        slopes.append(abs(sim.energy(xp) - sim.energy(xm)) / (2 * h))
                    key = f"N6_s{seed}_b{b}_L{L}_{model}"
                    res[key] = {"E_plus": -b * 6, "E0": g["E0"], "best_sample_E": best[0],
                                "best_gap_closed_E": (best[0] + b * 6) / (g["E0"] + b * 6),
                                "best_sample_F": d["F"], "best_sample_p_leak": d["p_leak"],
                                "F_plus": C.plus_fidelity(g["gs"]),
                                "median_abs_dE_dalpha_at_0": float(np.median(slopes)),
                                "max_abs_dE_dalpha_at_0": float(np.max(slopes)), "nsamp": nsamp}
                    print(key, {k: round(v, 4) if isinstance(v, float) else v for k, v in res[key].items()}, flush=True)
    (OUT / "landscape_audit.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
