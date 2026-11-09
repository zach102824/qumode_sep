"""DIAGNOSTIC ONLY (no optimization): does a second-order, cost-coupled escape from |+> exist?

For each point, report the minimum Hessian eigenvalue, the gamma component of its eigenvector, and
the minimum eigenvalue with (gamma, theta) frozen. A negative min eigenvalue that turns non-negative
when gamma is frozen means the descent needs the cost unitary (an interacting escape, like QAOA's).
A negative eigenvalue that survives freezing gamma is cavity-local (intra-cavity product physics).

Points (all prepare exactly |+>):
  real_plus   : Nov9th, gamma=theta=alpha=s=0                       (approved family)
  real_frame  : Nov9th, same but s_{1,1}=+pi/2, s_{L,1}=-pi/2        (approved family, L>=2)
  imag_plus   : D(i*alpha) instead of D(alpha), all zero            (NOT AUTHORIZED, D7 evidence only)
Strict logical (8-level) displacement throughout: this isolates the ansatz, not leakage.
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


class DispImag(C.Disp):
    def __call__(self, a):
        U = (self.V * np.exp(-1j * a * self.lam)) @ self.V.conj().T
        ph = np.exp(1j * np.pi / 2 * np.arange(U.shape[0]))
        U = ph[:, None] * U * ph.conj()[None, :]
        return np.ascontiguousarray(U[: self.nf, : self.nf])


def hessian(f, x0, h=1e-3):
    n = x0.size
    E = np.eye(n) * h
    H = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            H[i, j] = H[j, i] = (f(x0 + E[i] + E[j]) - f(x0 + E[i] - E[j]) - f(x0 - E[i] + E[j])
                                 + f(x0 - E[i] - E[j])) / (4 * h * h)
    return H


def analyse(f, x0, nfreeze):
    H = hessian(f, x0)
    w, V = np.linalg.eigh(H)
    keep = list(range(nfreeze, x0.size))
    return {"E": float(f(x0)), "min_eig": float(w[0]), "gamma_comp": float(abs(V[0, 0])),
            "min_eig_gamma_frozen": float(np.linalg.eigvalsh(H[np.ix_(keep, keep)])[0])}


def main():
    res = {}
    for N in (6, 9):
        for seed in range(10000, 10008):
            J = C.make_J(seed, N)
            for b in (1.0, 1.5):
                for L in (1, 2, 3):
                    q = C.QAOA(J, b, L)
                    res[f"N{N}_s{seed}_b{b}_L{L}_QAOA"] = analyse(q.energy, np.zeros(q.nparam()), 0)
                    sim = C.Nov9(J, b, L, strict=True)
                    x = np.zeros(sim.nparam())
                    res[f"N{N}_s{seed}_b{b}_L{L}_real_plus"] = analyse(sim.energy, x, 3)
                    if L >= 2:
                        xf = x.copy()
                        s0 = 3 + L
                        xf[s0], xf[s0 + 3 * (L - 1)] = np.pi / 2, -np.pi / 2
                        res[f"N{N}_s{seed}_b{b}_L{L}_real_frame"] = analyse(sim.energy, xf, 3)
                    si = C.Nov9(J, b, L, strict=True)
                    si.disp = DispImag(8, strict=True)
                    res[f"N{N}_s{seed}_b{b}_L{L}_imag_plus_NOT_AUTHORIZED"] = analyse(si.energy, x, 3)
    agg = {}
    for k, v in res.items():
        N, _, b, L, *m = k.split("_")
        a = agg.setdefault(f"{N}_{b}_{L}_{'_'.join(m)}", {"min_eig": [], "gamma_comp": [], "min_eig_gamma_frozen": []})
        for kk in a:
            a[kk].append(v[kk])
    summ = {k: {kk: float(np.median(vv)) for kk, vv in v.items()} for k, v in agg.items()}
    (OUT / "curvature_escape.json").write_text(json.dumps({"medians_over_8_seeds": summ, "per_instance": res}, indent=1))
    for k, v in summ.items():
        print(k, {kk: f"{vv:+.3f}" for kk, vv in v.items()})


if __name__ == "__main__":
    main()
