#!/usr/bin/env python3
"""Checks for the family-A transfer path. Does not train."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from Nov_3rd import settings as S
from Nov_3rd.quantum_h_tf import FockSim, zzz_diag
from Nov_3rd.quantum_h_transfer import (
    Circuit,
    a_of_n,
    cost_only_matches_one_pulse,
    decode,
    dense_matches_sparse,
    load_split,
    power_theta_from_fourier,
    shell_phi,
    walsh_mass_fractions,
    walsh_pieces,
    walsh_theta_from_fourier,
)


def test_sparse_matches_dense():
    rng = np.random.default_rng(0)
    for n, lam in ((3, 0.4), (4, 1.1)):
        diag = rng.normal(size=1 << n)
        err = dense_matches_sparse(diag, n, lam)
        assert err < 1e-8, (n, err)
    insts, _E = load_split(6, 1, "train", S.OUT / "beat_qaoa_12h" / "train_packs" / "train_n6.npz")
    err = dense_matches_sparse(insts[0].diag, 6, 0.5)
    assert err < 1e-8, err


def test_zzz_is_pure_degree_3():
    insts, _E = load_split(6, 2, "train", S.OUT / "beat_qaoa_12h" / "train_packs" / "train_n6.npz")
    for ins in insts:
        mass = walsh_mass_fractions(ins.diag)
        assert mass["frac_deg_3"] > 0.999, mass
        assert mass["frac_deg_1_to_4"] > 0.999
        pieces = walsh_pieces(torch.as_tensor(ins.diag, dtype=torch.float64).reshape(1, -1))
        # degrees 1,2,4 are the null Walsh knobs on this H_c
        for w in (0, 1, 3):
            assert float(pieces[w].abs().max()) < 1e-8, w
        recon = pieces[2, 0].numpy()
        assert np.allclose(recon, ins.diag, atol=1e-8)
    # one triangle, analytic
    d = zzz_diag(3, [(0, 1, 2, 1)])
    mass = walsh_mass_fractions(d)
    assert abs(mass["frac_deg_3"] - 1.0) < 1e-9


def test_cost_only_is_one_pulse():
    sim = FockSim(6, dtype=torch.float64)
    rng = np.random.default_rng(1)
    pulse = torch.as_tensor(rng.normal(size=(3, 1 << 6)), dtype=torch.float64)
    gammas = torch.tensor([0.2, -0.15, 0.07, 0.01], dtype=torch.float64)
    err = cost_only_matches_one_pulse(sim, pulse, gammas, lam=0.5)
    assert err < 1e-8, err
    # a nonzero shell does not collapse
    circ = Circuit(sim, pulse, pulse, torch.zeros(3, dtype=torch.float64))
    phi = torch.tensor([[0.0, 0.3, -0.1, 0.05]] * 4, dtype=torch.float64)
    e_many = circ.energy(gammas, phi, False, 0.5)
    e_one = circ.energy(gammas.sum().reshape(1), phi[:1], False, 0.5)
    assert float((e_many - e_one).abs().max()) > 1e-3


def test_scored_energy_matches_fock_when_pulse_is_score():
    sim = FockSim(6, dtype=torch.float64)
    insts, _E = load_split(6, 2, "train", S.OUT / "beat_qaoa_12h" / "train_packs" / "train_n6.npz")
    pulse = torch.as_tensor(np.stack([ins.diag for ins in insts]), dtype=torch.float64)
    circ = Circuit(sim, pulse, pulse, torch.zeros(2, dtype=torch.float64))
    gamma = torch.tensor([0.11, -0.07], dtype=torch.float64)
    phi = torch.tensor([[0.0, -0.2, 0.04, 0.01], [0.0, 0.13, -0.02, 0.0]], dtype=torch.float64)
    got = circ.energy(gamma, phi, False, 0.5)
    ref = sim.energy(pulse, gamma, phi, 0.5)
    assert torch.allclose(got, ref, atol=1e-8), (got, ref)


def test_power_scale_and_warm_dual():
    p = torch.tensor(0.4, dtype=torch.float64)
    assert abs(float(a_of_n(9, p)) - 1.0) < 1e-12
    assert abs(float(a_of_n(6, p)) - (8.0 / 5.0) ** 0.4) < 1e-12
    assert abs(float(a_of_n(12, p)) - (8.0 / 11.0) ** 0.4) < 1e-12
    c1 = torch.tensor([0.2, -0.1], dtype=torch.float64)
    c2 = torch.zeros(2, dtype=torch.float64)
    c3 = torch.zeros(2, dtype=torch.float64)
    p0 = torch.tensor(1.7, dtype=torch.float64)
    a = shell_phi(c1, c2, c3, p0, 6)
    b = shell_phi(c1, c2, c3, p0, 12)
    assert torch.allclose(a, b)
    # p=0, c2=c3=0 is the linear mixer at every n
    theta = torch.zeros(4 * 2 + 1, dtype=torch.float64)
    theta[0] = 0.2
    theta[1] = -0.1
    theta[2] = -0.3  # c1 layer 0
    theta[3] = 0.15
    g6, phi6, graded = decode("power_p", theta, 2, 6)
    g12, phi12, _ = decode("power_p", theta, 2, 12)
    assert graded is False
    assert torch.allclose(g6, g12)
    assert torch.allclose(phi6, phi12)
    k = torch.arange(4, dtype=torch.float64)
    expect = torch.stack([-0.3 * k, 0.15 * k], dim=0)
    assert torch.allclose(phi6, expect, atol=1e-12)


def test_walsh_decode_shapes_and_grad():
    L, n = 2, 6
    theta = torch.nn.Parameter(torch.zeros(7 * L + 1, dtype=torch.float64))
    with torch.no_grad():
        theta[: L * 4] = 0.05
        theta[4 * L] = -0.02
        theta[-1] = 0.3
    sim = FockSim(n, dtype=torch.float64)
    rng = np.random.default_rng(2)
    raw = rng.normal(size=(2, 1 << n))
    pulse = torch.as_tensor(raw, dtype=torch.float64)
    pieces = walsh_pieces(pulse)
    circ = Circuit(sim, pulse, pulse, torch.zeros(2, dtype=torch.float64), pieces=pieces)
    gamma, phi, graded = decode("walsh", theta, L, n)
    assert graded and tuple(gamma.shape) == (L, 4) and tuple(phi.shape) == (L, 4)
    e = circ.energy(gamma, phi, True, 0.5)
    e.sum().backward()
    assert theta.grad is not None and float(theta.grad.abs().sum()) > 0


def test_warm_dual_matches_fourier_energy():
    sim = FockSim(6, dtype=torch.float64)
    insts, _E = load_split(6, 2, "train", S.OUT / "beat_qaoa_12h" / "train_packs" / "train_n6.npz")
    pulse = torch.as_tensor(np.stack([ins.diag for ins in insts]), dtype=torch.float64)
    theta_f = torch.zeros(4, dtype=torch.float64)
    theta_f[0] = 0.13
    theta_f[1] = -0.04
    theta_f[2] = 0.02
    theta_f[3] = -0.01
    g_f, phi_f, _ = decode("fourier_qaoa", theta_f, 2, 6)
    g_p, phi_p, _ = decode("power_p", power_theta_from_fourier(theta_f, 2), 2, 6)
    g_w, phi_w, graded = decode("walsh", walsh_theta_from_fourier(theta_f, 2), 2, 6)
    circ = Circuit(sim, pulse, pulse, torch.zeros(2), pieces=walsh_pieces(pulse))
    e_f = circ.energy(g_f, phi_f, False, 0.5)
    e_p = circ.energy(g_p, phi_p, False, 0.5)
    e_w = circ.energy(g_w, phi_w, graded, 0.5)
    assert torch.allclose(e_f, e_p, atol=1e-8), (e_f, e_p)
    # ZZZ is pure degree 3, so copying γ onto every Walsh slot is the same pulse.
    assert torch.allclose(e_f, e_w, atol=1e-8), (e_f, e_w)


def test_unsat_walsh_is_a_real_knob_and_matches_scalar_when_tied():
    insts, E = load_split(6, 4, "train", S.OUT / "beat_qaoa_12h" / "train_packs" / "train_n6.npz")
    mass = walsh_mass_fractions(E)
    # A 4-local unsat count has Walsh degree ≤ 4. Degree 0 is the mean.
    nonconst = 1.0 - mass["by_degree"][0]
    assert mass["frac_other"] < 1e-8, mass
    assert abs(mass["frac_deg_1_to_4"] - nonconst) < 1e-8
    assert mass["frac_deg_3"] / nonconst < 0.5
    sim = FockSim(6, dtype=torch.float64)
    score = torch.as_tensor(E[:2], dtype=torch.float64)
    pulse = score - score.amax(dim=-1, keepdim=True)
    theta_f = torch.tensor([0.13, -0.04, 0.02, -0.01], dtype=torch.float64)
    g_f, phi_f, _ = decode("fourier_qaoa", theta_f, 2, 6)
    g_w, phi_w, graded = decode("walsh", walsh_theta_from_fourier(theta_f, 2), 2, 6)
    circ = Circuit(sim, pulse, score, torch.zeros(2, dtype=torch.float64), pieces=walsh_pieces(pulse))
    e_f = circ.energy(g_f, phi_f, False, 0.75)
    e_w = circ.energy(g_w, phi_w, graded, 0.75)
    assert torch.allclose(e_f, e_w, atol=1e-7), (e_f, e_w)
    del insts


def main():
    tests = [
        test_sparse_matches_dense,
        test_zzz_is_pure_degree_3,
        test_cost_only_is_one_pulse,
        test_scored_energy_matches_fock_when_pulse_is_score,
        test_power_scale_and_warm_dual,
        test_walsh_decode_shapes_and_grad,
        test_warm_dual_matches_fourier_energy,
        test_unsat_walsh_is_a_real_knob_and_matches_scalar_when_tied,
    ]
    for fn in tests:
        fn()
        print("ok", fn.__name__)
    print(f"passed {len(tests)}")


if __name__ == "__main__":
    main()
