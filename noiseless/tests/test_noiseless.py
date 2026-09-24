"""Unit tests for noiseless encoding, unitaries, ECD circuit."""

from __future__ import annotations

import numpy as np
import pytest
import qutip as qt

from noiseless.circuit_local_ecd import (
    apply_circuit,
    build_ansatz_unitary,
    ecd_d_a,
    ecd_e_b,
    n_parameters,
    parallel_ecd_layer,
    random_parameters,
    unpack_params,
)
from noiseless.encoding import (
    DIMS,
    HILBERT_DIM,
    NFOCK,
    N_QUBITS,
    bitstring_from_bits,
    bits_from_bitstring,
    bits_from_denm,
    denm_from_bits,
    denm_from_flat,
    flat_index,
    identity,
    vacuum,
)
from noiseless.spsa_gibbs import (
    NoiselessSimulator,
    beta_regularizer,
    betas_from_x,
    gibbs_objective,
    ground_flat_from_bitstring,
    optimize_trial,
    scale_spsa_a,
)
from noiseless.unitaries import U_NAMES, ab_matrix_elements, build_fixed_u


def test_dims():
    assert DIMS == (2, 2, 8, 8)
    assert HILBERT_DIM == 256
    assert N_QUBITS == 8
    assert vacuum().dims[0] == [2, 2, 8, 8]
    assert vacuum().shape == (256, 1)
    assert identity().shape == (256, 256)


def test_bit_roundtrip():
    for d in range(2):
        for e in range(2):
            for n_a in (0, 3, 7):
                for n_b in (0, 5, 7):
                    bits = bits_from_denm(d, e, n_a, n_b)
                    assert denm_from_bits(bits) == (d, e, n_a, n_b)
                    assert denm_from_flat(flat_index(d, e, n_a, n_b)) == (d, e, n_a, n_b)
                    s = bitstring_from_bits(bits)
                    assert bits_from_bitstring(s).tolist() == bits.tolist()


def test_msb_examples():
    assert bitstring_from_bits(bits_from_denm(1, 0, 6, 0)) == "10110000"
    assert bitstring_from_bits(bits_from_denm(0, 1, 0, 5)) == "01000101"


def test_u_actions():
    u = build_fixed_u("cz_nm")
    for n in range(4):
        for m in range(4):
            assert abs(ab_matrix_elements(u, n, m) - ((-1.0) ** (n * m))) < 1e-10
    ua = build_fixed_u("snap_a_pi")
    for n in range(4):
        for m in range(4):
            assert abs(ab_matrix_elements(ua, n, m) - ((-1.0) ** n)) < 1e-10
    ub = build_fixed_u("snap_b_pi")
    for n in range(4):
        for m in range(4):
            assert abs(ab_matrix_elements(ub, n, m) - ((-1.0) ** m)) < 1e-10


def test_ck_phase_actions():
    u2 = build_fixed_u("ck_pi2")
    u4 = build_fixed_u("ck_pi4")
    for n in range(5):
        for m in range(5):
            want2 = np.exp(-1j * (np.pi / 2.0) * n * m)
            want4 = np.exp(-1j * (np.pi / 4.0) * n * m)
            assert abs(ab_matrix_elements(u2, n, m) - want2) < 1e-10
            assert abs(ab_matrix_elements(u4, n, m) - want4) < 1e-10
    # Unitarity of Fock-diagonal CK family
    assert abs((u2.dag() * u2 - identity()).norm()) < 1e-8
    assert abs((u4.dag() * u4 - identity()).norm()) < 1e-8


def test_cphase_nn_actions():
    u = build_fixed_u("cphase_nn")
    # Diagonal samples
    assert abs(ab_matrix_elements(u, 0, 0) - 1.0) < 1e-10
    assert abs(ab_matrix_elements(u, 1, 1) - (-1.0)) < 1e-10
    assert abs(ab_matrix_elements(u, 2, 2) - (-1.0)) < 1e-10
    assert abs(ab_matrix_elements(u, 3, 3) - (-1.0)) < 1e-10
    assert abs(ab_matrix_elements(u, 1, 2) - 1.0) < 1e-10
    assert abs(ab_matrix_elements(u, 2, 1) - 1.0) < 1e-10
    assert abs(ab_matrix_elements(u, 0, 3) - 1.0) < 1e-10
    assert abs(ab_matrix_elements(u, 4, 0) - 1.0) < 1e-10
    assert abs((u.dag() * u - identity()).norm()) < 1e-8


def test_beamsplitter_unitary():
    for name in ("bs_pi6", "bs_pi4", "bs_pi3", "bs_pi2"):
        u = build_fixed_u(name)
        assert abs((u.dag() * u - identity()).norm()) < 1e-8


def test_beamsplitter_theta_aliases():
    """bs_pi6 / bs_pi4 / bs_pi3 match beamsplitter_ab(π/6, π/4, π/3)."""
    from noiseless.unitaries import beamsplitter_ab

    for name, theta in (
        ("bs_pi6", np.pi / 6.0),
        ("bs_pi4", np.pi / 4.0),
        ("bs_pi3", np.pi / 3.0),
    ):
        u = build_fixed_u(name)
        want = beamsplitter_ab(theta)
        assert abs((u - want).norm()) < 1e-10


def test_ecd_shapes_params():
    assert n_parameters(2) == 16
    assert n_parameters(4) == 32
    assert ecd_d_a(0.5 + 0.3j).shape == (256, 256)
    assert ecd_e_b(0.2j).shape == (256, 256)
    layer = parallel_ecd_layer(0.1, 0.2j, 0.3, 0.4, 0.5, 0.6)
    assert layer.shape == (256, 256)
    x = random_parameters(3, np.random.default_rng(0))
    assert x.shape == (24,)
    assert len(unpack_params(x, 3)) == 3
    u = build_fixed_u("identity")
    ket = apply_circuit(x, 3, u)
    assert abs(ket.norm() - 1.0) < 1e-8
    Uni = build_ansatz_unitary(x, 3, u_fixed=u)
    assert abs((Uni.dag() * Uni - identity()).norm()) < 1e-6


def test_all_u_names():
    for name in U_NAMES:
        assert build_fixed_u(name).shape == (256, 256)


def test_scale_spsa_a():
    assert abs(scale_spsa_a(37) - 0.2) < 1e-12
    assert scale_spsa_a(16) > scale_spsa_a(37)


def test_short_spsa():
    gs = "01011010"
    d, e, na, nb = denm_from_bits(bits_from_bitstring(gs))
    tensor = np.ones(DIMS, dtype=float)
    tensor[d, e, na, nb] = 0.0
    assert np.isfinite(gibbs_objective(np.ones(256) / 256.0, tensor.reshape(-1), 1.0))
    sim = NoiselessSimulator(
        u_fixed=build_fixed_u("identity"),
        energy_tensor=tensor,
        n_layers=2,
        ground_bitstring=gs,
        ground_flat_index=ground_flat_from_bitstring(gs),
    )
    res = optimize_trial(sim, maxiter=3, rng=np.random.default_rng(1))
    assert res.nfev > 0 and np.isfinite(res.fun)


def test_beta_regularizer_defaults():
    """λ1=λ3=0 → regularizer is exactly 0; λ1>0 increases cost when |β| nonzero."""
    x = np.zeros(16, dtype=float)
    x[0], x[2] = 0.6, 0.8  # |β_d|=1.0 for layer 0
    x[1], x[3] = 0.0, 0.0
    betas = betas_from_x(x, 2)
    assert abs(betas[0]) == pytest.approx(1.0)
    assert beta_regularizer(betas, lambda1=0.0, lambda3=0.0) == 0.0
    assert beta_regularizer(betas, lambda1=0.05, lambda3=0.0) == pytest.approx(0.05 * abs(betas).sum())
    # soft cap: |β|=1, β_max=0.5 → excess^2 = 0.25
    reg = beta_regularizer(betas, lambda1=0.0, lambda3=2.0, beta_max=0.5)
    assert reg == pytest.approx(2.0 * 0.25)


def test_cost_lambda0_matches_gibbs_only():
    """Default λ path must match pure Gibbs on a fixed x (bitwise/numerically)."""
    gs = "01011010"
    d, e, na, nb = denm_from_bits(bits_from_bitstring(gs))
    tensor = np.ones(DIMS, dtype=float)
    tensor[d, e, na, nb] = 0.0
    rng = np.random.default_rng(7)
    x = random_parameters(2, rng)
    sim0 = NoiselessSimulator(
        u_fixed=build_fixed_u("identity"),
        energy_tensor=tensor,
        n_layers=2,
        ground_bitstring=gs,
        ground_flat_index=ground_flat_from_bitstring(gs),
        lambda1=0.0,
        lambda3=0.0,
    )
    sim0._current_eta = 1.25
    # Pure Gibbs via objective + probs (same as pre-β-aware cost body)
    probs = sim0.probs_from_x(x)
    want = gibbs_objective(probs, sim0.energies_flat, 1.25)
    got = sim0.cost(x, eta=1.25)
    assert got == want  # exact equality when λ1=λ3=0
    # λ1>0 must increase cost when |β| nonzero
    sim1 = NoiselessSimulator(
        u_fixed=build_fixed_u("identity"),
        energy_tensor=tensor,
        n_layers=2,
        ground_bitstring=gs,
        ground_flat_index=ground_flat_from_bitstring(gs),
        lambda1=0.1,
        lambda3=0.0,
    )
    sim1._current_eta = 1.25
    assert sim1.cost(x, eta=1.25) > got


def test_adapt_off_lam0_matches_gibbs_path():
    """adapt_lambda=False and lam=0 must match pure Gibbs optimize path fields."""
    gs = "01011010"
    d, e, na, nb = denm_from_bits(bits_from_bitstring(gs))
    tensor = np.ones(DIMS, dtype=float)
    tensor[d, e, na, nb] = 0.0
    u = build_fixed_u("identity")
    rng = np.random.default_rng(11)
    x0 = random_parameters(2, rng)
    sim_g = NoiselessSimulator(
        u_fixed=u,
        energy_tensor=tensor,
        n_layers=2,
        ground_bitstring=gs,
        ground_flat_index=ground_flat_from_bitstring(gs),
        lambda1=0.0,
        lam=0.0,
    )
    sim_a = NoiselessSimulator(
        u_fixed=u,
        energy_tensor=tensor,
        n_layers=2,
        ground_bitstring=gs,
        ground_flat_index=ground_flat_from_bitstring(gs),
        lambda1=0.0,
        lam=0.0,
    )
    r0 = optimize_trial(sim_g, maxiter=6, rng=np.random.default_rng(11), x0=x0.copy(), adapt_lambda=False)
    r1 = optimize_trial(sim_a, maxiter=6, rng=np.random.default_rng(11), x0=x0.copy(), adapt_lambda=False, lam=0.0)
    assert r0.fun == r1.fun
    assert np.allclose(r0.x, r1.x)
    assert r1.final_lam == 0.0
    assert r1.mean_lam_post_warmup == 0.0


def test_adapt_lambda_raises_when_betas_large():
    """Smoke: with tiny β_max and forced-large x0, adapt should raise lam above 0."""
    gs = "01011010"
    d, e, na, nb = denm_from_bits(bits_from_bitstring(gs))
    tensor = np.ones(DIMS, dtype=float)
    tensor[d, e, na, nb] = 0.0
    # Large Cartesian β components so |β| >> β_max
    x0 = np.full(n_parameters(2), 3.0, dtype=float)
    sim = NoiselessSimulator(
        u_fixed=build_fixed_u("identity"),
        energy_tensor=tensor,
        n_layers=2,
        ground_bitstring=gs,
        ground_flat_index=ground_flat_from_bitstring(gs),
        lambda1=0.0,
        lam=0.0,
        beta_max=0.1,
    )
    # Short run but enough for warm-up end + at least one adapt tick
    # maxiter=20 → warmup_end=5; adapt every=5 → checks at 10,15,20
    res = optimize_trial(
        sim,
        maxiter=20,
        rng=np.random.default_rng(0),
        x0=x0,
        adapt_lambda=True,
        adapt_warmup_frac=0.25,
        adapt_every=5,
        adapt_f_hi=0.05,
        adapt_f_lo=0.01,
        adapt_lam_min=0.5,
        adapt_lam_max=5.0,
    )
    assert res.final_lam > 0.0
    assert res.mean_lam_post_warmup > 0.0


def test_lambda_cli_alias_sets_lam():
    """beta_regularizer: lam and legacy lambda3 both apply soft-cap."""
    betas = np.array([1.0 + 0j, 0.0])
    a = beta_regularizer(betas, lam=2.0, beta_max=0.5)
    b = beta_regularizer(betas, lambda3=2.0, beta_max=0.5)
    assert a == pytest.approx(b)
    assert a == pytest.approx(2.0 * 0.25)
