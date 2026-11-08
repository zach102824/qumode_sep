"""Original shared-gamma KM helpers. No per-layer gamma, no regrouping, no ground state."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "spin_glass_r3"))
import run_sg_r3 as m3  # noqa: E402

m2 = m3.m2
PREREG = json.loads((HERE / "PREREG.json").read_text())
CAP = int(PREREG["maxfun"])


def seed(*parts):
    return np.random.default_rng(np.random.SeedSequence([int(p) for p in parts]))


def assert_original(inst):
    """Fixed interleaved shells, and w1 equal to the all-X mixer."""
    n = inst.N
    m = n // 3
    expect = [[a, a + m, a + 2 * m] for a in range(m)]
    if m3.r1.packing(n) != expect:
        raise SystemExit(f"packing is not the fixed interleaved grouping: {m3.r1.packing(n)}")
    err = float(np.max(np.abs(inst.w1 - inst.wX)))
    if err > 1e-12:
        raise SystemExit(f"w1 differs from sum X by {err}")
    if inst.dK.shape != inst.hc.shape or inst.dM.shape != inst.hc.shape:
        raise SystemExit("feature diagonal shape")
    if m2.nparam("KM", 6) != 21 or m2.nparam("qaoa", 6) != 12:
        raise SystemExit("parameter count drifted")


def feature_corr(inst):
    """Pearson correlation of the classical diagonals. Not a quantum expectation."""
    def corr(a, b):
        aa = np.asarray(a, float) - float(np.mean(a))
        bb = np.asarray(b, float) - float(np.mean(b))
        na, nb = float(np.linalg.norm(aa)), float(np.linalg.norm(bb))
        if na < 1e-15 or nb < 1e-15:
            return 0.0
        return float(np.dot(aa, bb) / (na * nb))

    return corr(inst.dK, inst.hc), corr(inst.dM, inst.hc)


def qproj_starts(inst, x_qaoa, L, r):
    a, _, _, cs = m2.split("qaoa", np.asarray(x_qaoa, float), L)
    gamma = float(np.mean(a))
    shells = np.column_stack([cs[:, 0], np.zeros(L), np.zeros(L)])
    base = m2.join("KM", np.full(L, gamma), 0.0, 0.0, shells)
    th_k, th_m = feature_corr(inst)
    couple = m2.join("KM", np.full(L, gamma), th_k, th_m, shells)
    out = [("qproj", base), ("couple", couple)]
    for j, sig in enumerate((0.15, 0.40)):
        z = seed(20261150, inst.N, inst.k, L, r, j)
        out.append((f"qproj_perturb_{sig:.2f}", base + sig * z.normal(size=base.size)))
    return out


def one_start(inst, fam, x0, L):
    x0 = m2.clip(inst, fam, x0, L)
    bnds = m2.bounds(inst, fam, L)
    best = {"E": np.inf, "x": np.array(x0, copy=True)}
    cache, used = {}, [0]

    def ev(x):
        if used[0] >= CAP:
            raise StopIteration
        used[0] += 1
        energy, grad, _ = m2.energy_grad(inst, fam, x, L)
        cache["x"] = np.array(x, copy=True)
        cache["g"] = grad
        if energy < best["E"]:
            best.update(E=float(energy), x=np.array(x, copy=True))
        return energy, grad

    reason = "returned"
    try:
        res = minimize(lambda x: ev(x)[0], x0, jac=lambda x: cache["g"] if "x" in cache and np.array_equal(cache["x"], x) else ev(x)[1],
                       method="L-BFGS-B", bounds=bnds,
                       options={"maxiter": CAP, "maxfun": CAP, "ftol": PREREG["ftol"], "gtol": PREREG["gtol"]})
        reason = str(res.message)
    except StopIteration:
        reason = "start_ceiling"
    x = best["x"]
    _, grad, _ = m2.energy_grad(inst, fam, x, L)
    b = np.array(bnds, float)
    pg = np.where((x <= b[:, 0] + 1e-12) & (grad > 0), 0.0, grad)
    pg = np.where((x >= b[:, 1] - 1e-12) & (pg < 0), 0.0, pg)
    touch = bool(np.any((x - b[:, 0] < 1e-6) | (b[:, 1] - x < 1e-6)))
    return {"E": best["E"], "x_final": [float(v) for v in x], "n_eval": used[0], "capped": used[0] >= CAP,
            "stop_reason": reason, "pg_inf": float(np.max(np.abs(pg))), "touch_bound": touch}
