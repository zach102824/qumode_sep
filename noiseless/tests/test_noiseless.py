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


def test_bit_encode_decode_roundtrip():
    for d in range(2):
        for e in range(2):
            for n_a in range(NFOCK):
                for n_b in range(NFOCK):
                    bits = bits_from_denm(d, e, n_a, n_b)
                    assert bits.shape == (8,)
                    assert denm_from_bits(bits) == (d, e, n_a, n_b)
                    idx = flat_index(d, e, n_a, n_b)
                    assert denm_from_flat(idx) == (d, e, n_a, n_b)
                    s = bitstring_from_bits(bits)
                    assert bits_from_bitstring(s).tolist() == bits.tolist()


def test_msb_bit_map_examples():
    # d=1,e=0,n_A=6=110b,n_B=0 → 10 110 000
    bits = bits_from_denm(1, 0, 6, 0)
    assert bitstring_from_bits(bits) == "10110000"
    # d=0,e=1,n_A=0,n_B=5=101b → 01 000 101
    bits = bits_from_denm(0, 1, 0, 5)
    assert bitstring_from_bits(bits) == "01000101"


def test_u_actions_on_nm():
    u = build_fixed_u("cz_nm")
    # ⟨0,0,n,m|cz|0,0,n,m⟩ = (-1)^{n m}
    for n in range(4):
        for m in range(4):
            amp = ab_matrix_elements(u, n, m)
            expect = (-1.0) ** (n * m)
            assert abs(amp - expect) < 1e-10

    ua = build_fixed_u("snap_a_pi")
    for n in range(4):
        for m in range(4):
            amp = ab_matrix_elements(ua, n, m)
            assert abs(amp - ((-1.0) ** n)) < 1e-10

    ub = build_fixed_u("snap_b_pi")
    for n in range(4):
        for m in range(4):
            amp = ab_matrix_elements(ub, n, m)
            assert abs(amp - ((-1.0) ** m)) < 1e-10

    ui = build_fixed_u("identity")
    assert abs(ab_matrix_elements(ui, 3, 2) - 1.0) < 1e-12


def test_beamsplitter_unitary():
    for name in ("bs_pi4", "bs_pi2"):
        u = build_fixed_u(name)
        assert u.dims == [[2, 2, 8, 8], [2, 2, 8, 8]]
        # Unitary: U† U = I
        prod = u.dag() * u
        assert abs((prod - identity()).norm()) < 1e-8


def test_ecd_shapes_and_params():
    assert n_parameters(2) == 16
    assert n_parameters(3) == 24
    assert n_parameters(4) == 32
    ecd = ecd_d_a(0.5 + 0.3j)
    assert ecd.shape == (256, 256)
    ecd2 = ecd_e_b(0.2j)
    assert ecd2.shape == (256, 256)
    layer = parallel_ecd_layer(0.1, 0.2j, 0.3, 0.4, 0.5, 0.6)
    assert layer.shape == (256, 256)
    x = random_parameters(3, np.random.default_rng(0))
    assert x.shape == (24,)
    layers = unpack_params(x, 3)
    assert len(layers) == 3
    u = build_fixed_u("identity")
    ket = apply_circuit(x, 3, u)
    assert ket.shape == (256, 1)
    assert abs(ket.norm() - 1.0) < 1e-8
    Uni = build_ansatz_unitary(x, 3, u_fixed=u)
    assert abs((Uni.dag() * Uni - identity()).norm()) < 1e-6


def test_all_u_names_build():
    for name in U_NAMES:
        u = build_fixed_u(name)
        assert u.shape == (256, 256)


def test_scale_spsa_a():
    a37 = scale_spsa_a(37)
    assert abs(a37 - 0.2) < 1e-12
    a16 = scale_spsa_a(16)  # L*=2
    assert a16 > a37  # fewer params → larger a


def test_gibbs_and_short_spsa():
    gs = "01011010"
    d, e, na, nb = denm_from_bits(bits_from_bitstring(gs))
    tensor = np.ones(DIMS, dtype=float)
    tensor[d, e, na, nb] = 0.0
    probs = np.ones(256) / 256.0
    f = gibbs_objective(probs, tensor.reshape(-1), eta=1.0)
    assert np.isfinite(f)
    sim = NoiselessSimulator(
        u_fixed=build_fixed_u("identity"),
        energy_tensor=tensor,
        n_layers=2,
        ground_bitstring=gs,
        ground_flat_index=ground_flat_from_bitstring(gs),
    )
    res = optimize_trial(sim, maxiter=3, rng=np.random.default_rng(1))
    assert res.nfev > 0
    assert np.isfinite(res.fun)
    assert res.ground_bitstring == gs
