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


def test_beamsplitter_unitary():
    for name in ("bs_pi4", "bs_pi2"):
        u = build_fixed_u(name)
        assert abs((u.dag() * u - identity()).norm()) < 1e-8


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
