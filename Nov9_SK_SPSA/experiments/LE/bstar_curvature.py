"""DIAGNOSTIC ONLY (no optimization): does the |+> mixing-curvature picture hold at the
maximum-GS-entanglement-entropy field B*?

B* per instance: argmax over b of the half-chain von Neumann entropy S_A of the exact GS, for an
ASSUMED fixed balanced bipartition A = sites {0..floor(N/2)-1}. This may differ from the local
stream's bipartition and is used here only to place the diagnostic. Coarse grid on [0.05, 3.0],
then bounded scalar refinement; the peak must be interior (not on a grid edge). Symmetry and degeneracy
checks: GS even under global spin flip (b>0 makes H stoquastic and irreducible, so the GS is unique and
positive), and the E1-E0 gap is reported.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize_scalar

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import core as C  # noqa: E402
from curvature_escape import DispImag, analyse  # noqa: E402

OUT = HERE.parents[1] / "results" / "LE"


def entropy(gs, N):
    nA = N // 2
    psi = gs.reshape(1 << (N - nA), 1 << nA)  # LSB = site 0 -> low bits are A
    sv = np.linalg.svd(psi, compute_uv=False) ** 2
    sv = sv[sv > 1e-300]
    return float(-(sv * np.log(sv)).sum())


def bstar(J, N):
    grid = np.linspace(0.05, 3.0, 60)
    S = [entropy(C.exact_gs(J, b)["gs"], N) for b in grid]
    k = int(np.argmax(S))
    assert 0 < k < len(grid) - 1, "entropy peak on grid edge"
    r = minimize_scalar(lambda b: -entropy(C.exact_gs(J, b)["gs"], N), bounds=(grid[k - 1], grid[k + 1]),
                        method="bounded", options={"xatol": 1e-6})
    return float(r.x), float(-r.fun)


def main():
    res = {}
    for N in (6, 9):
        for seed in range(10000, 10004):
            J = C.make_J(seed, N)
            b, S = bstar(J, N)
            g = C.exact_gs(J, b)
            gs = g["gs"]
            flip = gs[np.arange(1 << N) ^ ((1 << N) - 1)]
            rec = {"Bstar": b, "S_A": S, "gap_E1_E0": g["E1"] - g["E0"], "flip_parity": float(gs @ flip),
                   "F_plus": C.plus_fidelity(gs)}
            for L in (1, 2):
                q = C.QAOA(J, b, L)
                rec[f"QAOA_L{L}_min_eig"] = analyse(q.energy, np.zeros(q.nparam()), 0)["min_eig"]
                s = C.Nov9(J, b, L, strict=True)
                rec[f"real_L{L}"] = analyse(s.energy, np.zeros(s.nparam()), 3)
                si = C.Nov9(J, b, L, strict=True)
                si.disp = DispImag(8, strict=True)
                rec[f"imag_L{L}_NOT_AUTHORIZED"] = analyse(si.energy, np.zeros(si.nparam()), 3)
            res[f"N{N}_s{seed}"] = rec
            print(f"N{N}_s{seed}", {k: (round(v, 4) if isinstance(v, float) else {kk: round(vv, 3) for kk, vv in v.items()})
                                   for k, v in rec.items()}, flush=True)
    (OUT / "bstar_curvature.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
