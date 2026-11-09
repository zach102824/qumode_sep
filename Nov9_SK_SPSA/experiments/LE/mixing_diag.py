"""DIAGNOSTIC ONLY (no optimization): why real-alpha Nov9th cannot leave |+> from a cold start.

1. Conjugation symmetry. With all SNAP phases s = 0, every gate is real except exp(-i gamma A), so
   psi(-gamma) = conj(psi(gamma)) and E(gamma) = E(-gamma) for ANY real alpha and L. The gamma
   gradient vanishes on the whole s = 0 hyperplane, and the gamma-alpha cross term is a third-order
   (gamma s alpha) effect from a cold start.
2. Mixing curvature. In the 2x2 block (gamma, alpha) at |+>, a cost-coupled escape needs
   H_ga^2 > H_gg H_aa. QAOA has H_aa = 0 (|+> is an eigenstate of its mixer). A displacement
   moves |+> (H_aa > 0), so it needs a large cross term.
Points: QAOA L=1; real alpha L=1 at |+>; real alpha L=2 frame point (s_{1,1}=pi/2, s_{2,1}=-pi/2);
imaginary-axis D(i alpha) L=1 (NOT AUTHORIZED, D7 evidence). Strict logical D8 (isolates ansatz).
alpha is used directly (d alpha/du = ALPHA_MAX at u = 0).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import core as C  # noqa: E402
from curvature_escape import DispImag, hessian  # noqa: E402

OUT = HERE.parents[1] / "results" / "LE"


def block(H, ia, scale):
    hgg, hga, haa = H[0, 0], H[0, ia] / scale, H[ia, ia] / scale**2
    return {"H_gg": float(hgg), "H_ga": float(hga), "H_aa": float(haa),
            "min_eig_2x2": float(np.linalg.eigvalsh(np.array([[hgg, hga], [hga, haa]]))[0])}


def main():
    rng = np.random.default_rng(0)
    res, sym = {}, {}
    for N in (6, 9):
        for seed in range(10000, 10008):
            J = C.make_J(seed, N)
            for b in (1.0, 1.5):
                k = f"N{N}_s{seed}_b{b}"
                q = C.QAOA(J, b, 1)
                Hq = hessian(q.energy, np.zeros(2))
                res[k + "_QAOA_L1"] = block(Hq, 1, 1.0)
                s1 = C.Nov9(J, b, 1, strict=True)
                res[k + "_real_L1_plus"] = block(hessian(s1.energy, np.zeros(s1.nparam())), 3, C.ALPHA_MAX)
                s2 = C.Nov9(J, b, 2, strict=True)
                x = np.zeros(s2.nparam())
                x[5], x[8] = np.pi / 2, -np.pi / 2
                res[k + "_real_L2_frame"] = block(hessian(s2.energy, x), 4, C.ALPHA_MAX)
                si = C.Nov9(J, b, 1, strict=True)
                si.disp = DispImag(8, strict=True)
                res[k + "_imag_L1_plus_NOT_AUTHORIZED"] = block(hessian(si.energy, np.zeros(si.nparam())), 3, C.ALPHA_MAX)
                for L in (1, 2, 3):
                    for strict in (True, False):
                        sim = C.Nov9(J, b, L, strict=strict, **({} if strict else {"nf": 16}))
                        xr = rng.normal(0, 0.5, sim.nparam())
                        x0 = xr.copy()
                        x0[3 + L:] = 0
                        xm, xs = x0.copy(), xr.copy()
                        xm[0] *= -1
                        xs[0] *= -1
                        sym[f"{k}_L{L}_{'strict' if strict else 'leaky'}"] = {
                            "s0_absdiff": abs(sim.energy(x0) - sim.energy(xm)),
                            "s_random_absdiff": abs(sim.energy(xr) - sim.energy(xs))}
    agg = {}
    for k, v in res.items():
        N, _, _, *m = k.split("_")
        a = agg.setdefault(f"{N}_{'_'.join(m)}", {kk: [] for kk in v})
        for kk in v:
            a[kk].append(v[kk])
    summ = {k: {kk: float(np.median(vv)) for kk, vv in v.items()} for k, v in agg.items()}
    symmax = {"max_s0_absdiff": max(v["s0_absdiff"] for v in sym.values()),
              "median_s_random_absdiff": float(np.median([v["s_random_absdiff"] for v in sym.values()]))}
    (OUT / "mixing_diag.json").write_text(json.dumps({"medians_over_8_seeds_2b": summ, "conj_symmetry": symmax,
                                                      "per_instance": res, "symmetry_per_instance": sym}, indent=1))
    for k, v in summ.items():
        print(k, {kk: f"{vv:+.3f}" for kk, vv in v.items()})
    print(symmax)


if __name__ == "__main__":
    main()
