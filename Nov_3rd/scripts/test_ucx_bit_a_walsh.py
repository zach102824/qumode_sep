#!/usr/bin/env python3
"""Sanity: equal γ_w recovers γ·E; FWHT degree split sums back to E (minus mean)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[2]
_SCR = Path(__file__).resolve().parent
for p in (str(_ROOT), str(_SCR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from Nov_3rd.ansatz import make_sim
from Nov_3rd.graded_cost import (
    WALSH_DEGREES,
    cn_parts_from_E,
    equal_gamma_rel_err,
    fwht,
    walsh_degree_E,
)
from beat_params import phi_from_c, warm_from_qaoa
from Nov_3rd.data_io import load_qaoa


def _one_E(n: int, seed: int = 0) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    E = rng.integers(0, 8, size=(1 << n,)).astype(np.float32)
    E[0] = 0.0
    return torch.as_tensor(E, dtype=torch.float32)


def test_fwht_involution():
    x = torch.randn(64, dtype=torch.float64)
    y = fwht(fwht(x)) / 64.0
    err = float((y - x).abs().max())
    assert err < 1e-10, err
    print(f"PASS fwht involution max|err|={err:.2e}")


def test_degree_sum():
    n = 6
    E = _one_E(n, 1)
    parts = walsh_degree_E(E, n)  # (4, 64)
    recon = parts.sum(0) + E.double().mean()
    # degrees >4 should be ~0 for generic E they are NOT zero — this is a random
    # table, not 4-SAT. Only check w=0 isolation: mean of each E_w ≈ 0.
    for w, p in zip(WALSH_DEGREES, parts):
        mu = float(p.double().mean())
        assert abs(mu) < 1e-6, (w, mu)
    rel = float((recon.float() - E).norm() / (E.norm() + 1e-12))
    # random E has all degrees; recon is only w=1..4 plus mean, so leftover is w>=5
    print(f"PASS degree-w mean0; leftover rel (w>=5 on random E)={rel:.3e}")


def test_sat_degree_leq4():
    """4-SAT E is degree ≤4: sum_{w=1..4} E_w + mean = E."""
    from explore_OCT29.lib.sat_gen import gen_instance

    n = 6
    _, _, E_np = gen_instance(n, seed=123, compute_E=True)
    E = torch.as_tensor(E_np.astype(np.float32))
    parts = walsh_degree_E(E, n)
    recon = parts.sum(0) + float(E.mean())
    rel = float((recon - E).norm() / (E.norm() + 1e-12))
    assert rel < 1e-5, rel
    print(f"PASS 4-SAT Walsh recon rel={rel:.2e}")


def test_equal_gamma_cost_grid():
    from explore_OCT29.lib.sat_gen import gen_instance

    n = 6
    _, _, E_np = gen_instance(n, seed=7, compute_E=True)
    E = torch.as_tensor(E_np.astype(np.float32))
    sim = make_sim(n, 2, torch.device("cpu"))
    rel = equal_gamma_rel_err(E, n, sim.grid_to_x, gamma=0.37)
    assert rel < 1e-5, rel
    print(f"PASS equal-γ cost-grid rel={rel:.2e}")


def test_equal_gamma_forward():
    from explore_OCT29.lib.sat_gen import gen_instance

    n = 6
    L = 2
    _, _, E_np = gen_instance(n, seed=11, compute_E=True)
    E = torch.as_tensor(E_np.astype(np.float32))
    sim = make_sim(n, L, torch.device("cpu"))
    betas, gammas = load_qaoa(L)
    g, c1, c2, c3 = warm_from_qaoa(betas, gammas)
    phi = phi_from_c(c1, c2, c3)
    out0 = sim.forward_nov3rd(E, g, phi)
    parts = cn_parts_from_E(E, n, sim.grid_to_x)
    gw = g.reshape(L, 1).expand(L, 4).contiguous()
    out1 = sim.forward_nov3rd_graded(E, gw, phi, parts)
    inv0 = float(out0["inv_p"])
    inv1 = float(out1["inv_p"])
    rel = abs(inv1 - inv0) / max(abs(inv0), 1e-12)
    ps0, ps1 = float(out0["p_succ"]), float(out1["p_succ"])
    rel_p = abs(ps1 - ps0) / max(abs(ps0), 1e-12)
    assert rel < 1e-5, (rel, inv0, inv1)
    assert rel_p < 1e-5, (rel_p, ps0, ps1)
    print(f"PASS equal-γ forward inv_p rel={rel:.2e} p_succ rel={rel_p:.2e}")


def main():
    test_fwht_involution()
    test_degree_sum()
    test_sat_degree_leq4()
    test_equal_gamma_cost_grid()
    test_equal_gamma_forward()
    print("ALL WALSH-DEGREE SANITY PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
