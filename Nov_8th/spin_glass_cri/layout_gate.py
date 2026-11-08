"""Reject parameter layouts that break the Nov 3 shared-gamma circuit.

Packing may differ from the interleaved triples. The gate checked here is the
parameter layout and the layer-independent UcX, not the cavity assignment.
No optimizer is called.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "spin_glass_r1"))
import run_sg as r1  # noqa: E402


def require_original_km(x, L: int):
    """One gamma, one thetaK, one thetaM, and three shell angles per layer."""
    v = np.asarray(x, dtype=np.float64)
    expect = 3 + 3 * int(L)
    if v.shape != (expect,):
        raise ValueError(
            f"KM layout length {v.shape} is not {expect}. "
            "Per-layer gamma, extra heads, and extra shell channels are rejected."
        )
    gamma = float(v[0])
    theta_k = float(v[1])
    theta_m = float(v[2])
    shells = v[3:].reshape(int(L), 3)
    return gamma, theta_k, theta_m, shells


def assert_forward_shares_ucx(inst, L: int = 4):
    """Same diagonal D and the same three shell weights at every layer."""
    gamma, theta_k, theta_m = 0.2, -0.1, 0.05
    shells = np.arange(3 * L, dtype=float).reshape(L, 3) * 0.01
    x = r1.pack("KM", gamma, {"K": theta_k, "M": theta_m}, shells)
    require_original_km(x, L)
    psi, snaps = r1.forward(inst, "KM", x, L, store=True)
    if len(snaps) != L:
        raise RuntimeError("layer count")
    g0 = snaps[0][4]
    w1, w2, w3 = snaps[0][5], snaps[0][6], snaps[0][7]
    gammas = []
    for phase, mid, phi, x2, g, a, b, c, gam in snaps:
        if g.shape != g0.shape or float(np.max(np.abs(g - g0))) > 1e-12:
            raise RuntimeError("D changed between layers")
        if float(np.max(np.abs(a - w1))) > 1e-12:
            raise RuntimeError("shell weight w1 changed between layers")
        if float(np.max(np.abs(b - w2))) > 1e-12 or float(np.max(np.abs(c - w3))) > 1e-12:
            raise RuntimeError("shell weights changed between layers")
        gammas.append(float(gam))
    if max(gammas) - min(gammas) > 1e-12:
        raise RuntimeError("gamma is not shared")
    if abs(gammas[0] - gamma) > 1e-12:
        raise RuntimeError("stored gamma is not the head coefficient")
    if psi.shape != inst.plus.shape:
        raise RuntimeError("state shape")
    return True


def assert_forbidden_layouts_rejected(L: int = 6):
    if r1.nparam("KM", L) != 3 + 3 * L:
        raise RuntimeError("KM parameter count is not 3+3L")
    if r1.nparam("qaoa", L) != 2 * L:
        raise RuntimeError("QAOA parameter count is not 2L")
    # Round-4 KM_gl layout: shared (b, c) plus a per-layer coefficient on Hc.
    km_gl = np.zeros(2 + 4 * L)
    try:
        require_original_km(km_gl, L)
    except ValueError:
        pass
    else:
        raise RuntimeError("per-layer KM_gl layout was accepted")
    too_many_shells = np.zeros(3 + 4 * L)
    try:
        require_original_km(too_many_shells, L)
    except ValueError:
        pass
    else:
        raise RuntimeError("four shell channels were accepted")
    require_original_km(np.zeros(3 + 3 * L), L)


def main():
    assert_forbidden_layouts_rejected()
    rng = np.random.default_rng(np.random.SeedSequence([20261170, 9, 0]))
    n = 9
    J = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            J[i, j] = J[j, i] = float(rng.normal())
    inst = r1.Inst(n, 0, J, {"role": "layout-gate"})
    assert_forward_shares_ucx(inst)
    print("layout gate passed")


if __name__ == "__main__":
    main()
