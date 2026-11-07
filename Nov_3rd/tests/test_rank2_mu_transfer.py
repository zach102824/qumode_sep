#!/usr/bin/env python3
"""Checks for the μ=−0.25 rank-2 transfer. Does not train."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from Nov_3rd import settings as S
from Nov_3rd.quantum_h_rank2 import (
    Rank2Sim,
    best_product_rank2,
    decide_rank2_transfer,
    dense_rank2,
    exact_ground,
    exact_ground_rank2,
    paired_clear,
    sparse_ground_rank2,
    sparse_rank2,
)
from Nov_3rd.quantum_h_tf import FockSim, best_product_energy
from Nov_3rd.quantum_h_transfer import decode, init_fourier, load_split, power_theta_from_fourier, sparse_ground_state


def test_sparse_matches_dense_and_arm_a():
    rng = np.random.default_rng(0)
    for n in (3, 6):
        diag = rng.normal(size=1 << n)
        for lam, mu, kappa in ((0.5, 0.0, 0.0), (0.5, -0.25, 0.0), (0.5, 0.25, -0.25), (0.0, -0.4, 0.2)):
            dense = dense_rank2(diag, n, lam, mu, kappa)
            sparse = sparse_rank2(diag, n, lam, mu, kappa).toarray()
            assert np.allclose(sparse, dense, atol=1e-12), (n, lam, mu, kappa)
    insts, _E = load_split(6, 1, "train", S.OUT / "beat_qaoa_12h" / "train_packs" / "train_n6.npz")
    ins = insts[0]
    for mu, kappa in ((0.0, 0.0), (-0.25, 0.0), (-0.25, -0.25)):
        ref = exact_ground_rank2(ins.diag, ins.clauses, 6, 0.5, mu, kappa, seed=ins.seed, product=False)
        got = sparse_ground_rank2(ins.diag, 6, 0.5, mu, kappa)
        assert abs(got["E0"] - ref["E0"]) < 1e-8, (mu, kappa, got["E0"], ref["E0"])
        assert got["residual"] < 1e-8
    arm = exact_ground(ins.diag, 6, 0.5)
    arm_s = sparse_ground_state(ins.diag, 6, 0.5)
    got0 = sparse_ground_rank2(ins.diag, 6, 0.5, 0.0, 0.0)
    assert abs(got0["E0"] - arm["E0"]) < 1e-8
    assert abs(got0["E0"] - arm_s["E0"]) < 1e-8
    # negative μ moves the ground energy
    got_m = sparse_ground_rank2(ins.diag, 6, 0.5, -0.25, 0.0)
    assert got_m["E0"] < got0["E0"] - 1e-3


def test_mu0_energy_and_grad_match_fock():
    sim_r = Rank2Sim(6, dtype=torch.float64)
    sim_f = FockSim(6, dtype=torch.float64)
    insts, _E = load_split(6, 2, "train", S.OUT / "beat_qaoa_12h" / "train_packs" / "train_n6.npz")
    Hc = torch.as_tensor(np.stack([ins.diag for ins in insts]), dtype=torch.float64)
    theta = init_fourier(4, 1, torch.float64, "cpu")
    g, phi, graded = decode("fourier_qaoa", theta, 4, 6)
    assert graded is False
    e_r = sim_r.energy(Hc, g, phi, 0.5, 0.0, 0.0)
    e_f = sim_f.energy(Hc, g, phi, 0.5)
    assert torch.allclose(e_r, e_f, atol=1e-12), float((e_r - e_f).abs().max())
    e_xx = sim_r.energy(Hc, g, phi, 0.5, -0.25, 0.0)
    assert float((e_xx - e_r).abs().sum()) > 1e-4
    th = torch.nn.Parameter(power_theta_from_fourier(theta, 4).clone())
    th_f = torch.nn.Parameter(th.detach().clone())

    def loss_rank(p):
        gg, pp, _gr = decode("power_p", p, 4, 6)
        return sim_r.energy(Hc, gg, pp, 0.5, 0.0, 0.0).sum()

    def loss_fock(p):
        gg, pp, _gr = decode("power_p", p, 4, 6)
        return sim_f.energy(Hc, gg, pp, 0.5).sum()

    loss_rank(th).backward()
    loss_fock(th_f).backward()
    assert float((th.grad - th_f.grad).abs().max()) < 1e-8
    # p=0 dual matches Fourier at both sizes that identify a_n
    for n in (6, 9):
        sim_n = FockSim(n, dtype=torch.float64)
        rows, _E = load_split(n, 2, "train", S.OUT / "beat_qaoa_12h" / "train_packs" / f"train_n{n}.npz")
        Hn = torch.as_tensor(np.stack([ins.diag for ins in rows]), dtype=torch.float64)
        gf, pf, _ = decode("fourier_qaoa", theta, 4, n)
        gp, pp, _ = decode("power_p", power_theta_from_fourier(theta, 4), 4, n)
        ef = sim_n.energy(Hn, gf, pf, 0.5)
        ep = sim_n.energy(Hn, gp, pp, 0.5)
        assert torch.allclose(ef, ep, atol=1e-10), n
    e_ref = best_product_energy(insts[0].clauses, 6, 0.5, seed=insts[0].seed)
    e_new = best_product_rank2(insts[0].clauses, 6, 0.5, 0.0, 0.0, seed=insts[0].seed)
    assert e_new <= e_ref + 1e-4


def test_clear_and_verdict():
    # Two seeds leave the Fourier point and both drop. Ratio-of-medians drops by 0.044.
    clear = paired_clear(0.880, 0.924, [0.85, 1.0, 0.86], [0.90, 1.0, 0.91])
    assert clear["clear"] and clear["n_active"] == 2 and clear["drop"] > 0.02
    verdict = decide_rank2_transfer(
        R_mu=0.880,
        R_mu0=0.924,
        per_seed_R_mu=[0.85, 1.0, 0.86],
        per_seed_R_mu0=[0.90, 1.0, 0.91],
        peak=0.08,
        prod_gap=0.03,
        R_mu_n15=0.90,
    )
    assert verdict["verdict"] == "PASS", verdict["reason"]

    # Drop below the 0.02 floor is not a clear gain.
    soft = decide_rank2_transfer(
        R_mu=0.915,
        R_mu0=0.924,
        per_seed_R_mu=[0.90, 1.0, 0.905],
        per_seed_R_mu0=[0.91, 1.0, 0.915],
        peak=0.08,
        prod_gap=0.03,
    )
    assert soft["verdict"] == "needs_edit", soft
    assert soft["beat_qaoa"] and not soft["clear"]["clear"]

    # Beats a drifted control but sits above the published 0.924 transfer.
    above = decide_rank2_transfer(
        R_mu=0.950,
        R_mu0=0.990,
        per_seed_R_mu=[0.93, 0.95, 0.94],
        per_seed_R_mu0=[0.98, 0.99, 0.97],
        peak=0.08,
        prod_gap=0.03,
    )
    assert above["verdict"] == "needs_edit"
    assert not above["below_published"]
    assert not above["control_match"]

    killed = decide_rank2_transfer(
        R_mu=1.08,
        R_mu0=0.924,
        per_seed_R_mu=[1.05, 1.10, 1.08],
        per_seed_R_mu0=[0.90, 1.0, 0.91],
        peak=0.08,
        prod_gap=0.03,
    )
    assert killed["verdict"] == "KILL", killed["reason"]

    bitstring = decide_rank2_transfer(
        R_mu=0.80,
        R_mu0=0.924,
        per_seed_R_mu=[0.70, 0.80, 0.75],
        per_seed_R_mu0=[0.90, 1.0, 0.91],
        peak=0.62,
        prod_gap=0.0,
    )
    assert bitstring["verdict"] == "KILL"

    gap = decide_rank2_transfer(
        R_mu=0.880,
        R_mu0=0.924,
        per_seed_R_mu=[0.85, 1.0, 0.86],
        per_seed_R_mu0=[0.90, 1.0, 0.91],
        peak=0.08,
        prod_gap=0.01,
    )
    assert gap["verdict"] == "needs_edit" and not gap["quantum_pass"]

    flip = decide_rank2_transfer(
        R_mu=0.880,
        R_mu0=0.924,
        per_seed_R_mu=[0.85, 1.0, 0.86],
        per_seed_R_mu0=[0.90, 1.0, 0.91],
        peak=0.08,
        prod_gap=0.03,
        R_mu_n15=1.15,
    )
    assert flip["verdict"] == "needs_edit" and flip["n15_flip"]

    # A single seed leaving the warm start is not a multi-seed clear gain.
    one = paired_clear(0.90, 0.96, [0.80, 1.0, 1.0], [0.90, 1.0, 1.0])
    assert one["n_active"] == 1 and not one["clear"]


def main():
    tests = [
        test_sparse_matches_dense_and_arm_a,
        test_mu0_energy_and_grad_match_fock,
        test_clear_and_verdict,
    ]
    for fn in tests:
        fn()
        print("ok", fn.__name__)
    print(f"passed {len(tests)}")


if __name__ == "__main__":
    main()
