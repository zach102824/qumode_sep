"""Correctness of the Kraus channels and the noisy density-matrix circuit.

(a) single channels vs closed forms and QuTiP mesolve of their own Lindblad term;
(b) tensor helpers vs dense Kronecker-product matrices, CPTP / Hermiticity of the noisy layer,
    fast JP channel vs the direct q-tensored sequence;
(c) end to end: zero noise vs the noiseless ket simulator, and the whole JP sequence on (q, d, e, A, B)
    vs QuTiP mesolve of the full master equation (eq:lindblad + eq:Htilde).
"""

from __future__ import annotations

import sys
from math import comb
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))

from noiseless.circuit_local_ecd import random_parameters, rotation_np, unpack_params  # noqa: E402
from noiseless.ecd_kbit import KbitLayout, apply_circuit, jp_diag  # noqa: E402
from noiseless.relayout_kbit import XorKbitSim  # noqa: E402
from noisy_simulation import circuit as C  # noqa: E402
from noisy_simulation.device import EICKBUSCH, NOISELESS, Device  # noqa: E402
from noisy_simulation.kraus import (  # noqa: E402
    apply_axis,
    apply_cavity_loss,
    apply_conditional,
    apply_kraus,
    apply_left,
    apply_qubit_channel,
    apply_right_dag,
    apply_two_mode_unitary,
    as_matrix,
    cavity_loss_kraus,
    cavity_loss_weights,
    conditional_left,
    pair_conditional_left,
    qubit_kraus,
    tensor_in_ground,
    trace_out,
)
from noisy_simulation.simulator import NoisyXorKbitSim  # noqa: E402

qt = pytest.importorskip("qutip")

RNG = np.random.default_rng(1234)
SIG_M = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=complex)  # |g><e|, index 0 = g
SIG_Z = np.diag([1.0, -1.0]).astype(complex)
P_E = np.diag([0.0, 1.0]).astype(complex)


def rand_rho(D: int, rng=RNG) -> np.ndarray:
    G = rng.normal(size=(D, D)) + 1j * rng.normal(size=(D, D))
    r = G @ G.conj().T
    return r / np.trace(r)


def rand_rho_t(dims, rng=RNG) -> np.ndarray:
    D = int(np.prod(dims))
    return rand_rho(D, rng).reshape(tuple(dims) * 2)


def embed(M: np.ndarray, k: int, dims) -> np.ndarray:
    out = np.eye(1)
    for i, d in enumerate(dims):
        out = np.kron(out, M if i == k else np.eye(d))
    return out


def kraus_dense(rho: np.ndarray, kraus) -> np.ndarray:
    return sum(K @ rho @ K.conj().T for K in kraus)


def trace_dist(a: np.ndarray, b: np.ndarray) -> float:
    return 0.5 * float(np.abs(np.linalg.eigvalsh(a - b)).sum())


def mesolve_final(rho0: np.ndarray, H: np.ndarray, c_ops, t: float, dims) -> np.ndarray:
    """Final state of drho/dt = -i[H, rho] + sum D[c] rho over time t (dimensionless: rescaled to t = 1)."""
    dd = [list(dims), list(dims)]
    Hq = qt.Qobj(H * t, dims=dd).to("CSR")
    cq = [qt.Qobj(c * np.sqrt(t), dims=dd).to("CSR") for c in c_ops]
    res = qt.mesolve(Hq, qt.Qobj(rho0, dims=dd), [0.0, 1.0], cq,
                     options={"atol": 1e-12, "rtol": 1e-10, "nsteps": 100000})
    return res.states[-1].full()


# ---------------------------------------------------------------------------
# (a) channels
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("n0", [1, 4, 7])
def test_fock_loss_is_binomial(n0):
    nf, kt = 10, 0.3
    rho = np.zeros((nf, nf), dtype=complex)
    rho[n0, n0] = 1.0
    out = kraus_dense(rho, cavity_loss_kraus(kt, nf))
    s = np.exp(-kt)
    expect = np.array([comb(n0, m) * s ** m * (1 - s) ** (n0 - m) if m <= n0 else 0.0 for m in range(nf)])
    assert np.allclose(np.real(np.diag(out)), expect, atol=1e-14)
    assert np.allclose(out, np.diag(np.diag(out)), atol=1e-15)


def test_coherent_state_shrinks_and_stays_pure():
    nf, kt, alpha = 40, 0.4, 2.0
    coh = lambda a: qt.coherent(nf, a).full().ravel()  # noqa: E731
    psi = coh(alpha)
    out = kraus_dense(np.outer(psi, psi.conj()), cavity_loss_kraus(kt, nf))
    tgt = coh(alpha * np.exp(-kt / 2))
    assert abs(np.real(np.trace(out @ out)) - 1.0) < 1e-10
    assert abs(np.real(tgt.conj() @ out @ tgt) - 1.0) < 1e-10


@pytest.mark.parametrize("jmax", [None, 1, 2, 4])
def test_cavity_loss_trace_preserving(jmax):
    nf, kt = 12, 0.05
    S = sum(K.conj().T @ K for K in cavity_loss_kraus(kt, nf, jmax))
    assert np.allclose(S, np.eye(nf), atol=1e-13)


def test_uncompleted_deficit_is_prob_of_extra_jumps():
    nf, kt, jmax = 12, 0.2, 2
    S = sum(K.conj().T @ K for K in cavity_loss_kraus(kt, nf, jmax, complete=False))
    p = 1 - np.exp(-kt)
    n = np.arange(nf)
    keep = np.array([sum(comb(v, j) * p ** j * (1 - p) ** (v - j) for j in range(min(v, jmax) + 1)) for v in n])
    assert np.allclose(np.diag(S).real, keep, atol=1e-14)


def test_qubit_closed_forms():
    gt, gpt = 0.3, 0.2
    rho = np.array([[0.4, 0.3 - 0.2j], [0.3 + 0.2j, 0.6]], dtype=complex)
    out = kraus_dense(rho, qubit_kraus(gt, gpt))
    assert np.isclose(out[1, 1], 0.6 * np.exp(-gt))
    assert np.isclose(out[0, 0], 1 - 0.6 * np.exp(-gt))
    assert np.isclose(out[0, 1], rho[0, 1] * np.exp(-gt / 2 - gpt))
    assert np.allclose(sum(K.conj().T @ K for K in qubit_kraus(gt, gpt)), np.eye(2), atol=1e-15)


def test_device_t2_definition():
    d = EICKBUSCH
    # Ramsey coherence after t decays as exp(-t / T2)
    t = 7e-6
    out = kraus_dense(np.full((2, 2), 0.5, dtype=complex), qubit_kraus(d.gamma * t, d.gamma_phi * t))
    assert np.isclose(abs(out[0, 1]), 0.5 * np.exp(-t / d.t2_q), rtol=1e-12)


def test_cavity_loss_vs_mesolve():
    nf, kt = 8, 0.37
    rho0 = rand_rho(nf)
    a = np.diag(np.sqrt(np.arange(1, nf)), 1).astype(complex)
    ref = mesolve_final(rho0, np.zeros((nf, nf)), [np.sqrt(kt) * a], 1.0, [nf])
    assert np.abs(kraus_dense(rho0, cavity_loss_kraus(kt, nf)) - ref).max() < 1e-8


def test_qubit_channel_vs_mesolve():
    gt, gpt = 0.41, 0.23
    rho0 = rand_rho(2)
    ref = mesolve_final(rho0, np.zeros((2, 2)), [np.sqrt(gt) * SIG_M, np.sqrt(gpt / 2) * SIG_Z], 1.0, [2])
    assert np.abs(kraus_dense(rho0, qubit_kraus(gt, gpt)) - ref).max() < 1e-8


def test_semigroup():
    nf, t1, t2 = 9, 0.13, 0.29
    rho = rand_rho(nf)
    two = kraus_dense(kraus_dense(rho, cavity_loss_kraus(t1, nf)), cavity_loss_kraus(t2, nf))
    assert np.abs(two - kraus_dense(rho, cavity_loss_kraus(t1 + t2, nf))).max() < 1e-14
    q = rand_rho(2)
    two = kraus_dense(kraus_dense(q, qubit_kraus(t1, 2 * t1)), qubit_kraus(t2, 2 * t2))
    assert np.abs(two - kraus_dense(q, qubit_kraus(t1 + t2, 2 * (t1 + t2)))).max() < 1e-14


@pytest.mark.parametrize("jmax", [1, 2, 3])
def test_completed_k0_deviation_order(jmax):
    nf = 16
    devs = [np.abs(cavity_loss_weights(kt, nf, jmax)[0][1] - np.exp(-0.5 * np.arange(nf) * kt)).max()
            for kt in (1e-3, 2e-3)]
    assert devs[0] < (1e-3 * nf) ** (jmax + 1)
    assert 2 ** (jmax + 1) * 0.8 < devs[1] / devs[0] < 2 ** (jmax + 1) * 1.25


def test_jmax2_vs_jmax4_at_device_rates():
    nf = 24
    kt = EICKBUSCH.kappa * (EICKBUSCH.tau_ecd + EICKBUSCH.tau_jp)
    rho = rand_rho(nf)
    err = np.abs(kraus_dense(rho, cavity_loss_kraus(kt, nf, 2)) - kraus_dense(rho, cavity_loss_kraus(kt, nf, 4))).max()
    # first dropped term: 3 losses from the top level, weight (kt n_max)^3 / 3!
    assert err < (kt * (nf - 1)) ** 3 / 6


def test_jmax2_error_at_device_rates():
    nf = 24
    kt = EICKBUSCH.kappa * (EICKBUSCH.tau_ecd + EICKBUSCH.tau_jp)
    rho = rand_rho(nf)
    err = np.abs(kraus_dense(rho, cavity_loss_kraus(kt, nf, 2)) - kraus_dense(rho, cavity_loss_kraus(kt, nf))).max()
    assert err < (kt * nf) ** 3


# ---------------------------------------------------------------------------
# (b) helpers on multi-mode tensors vs dense matrices
# ---------------------------------------------------------------------------
SHAPES = [(2, 2, 3, 3), (2, 2, 2, 3, 3)]


@pytest.mark.parametrize("dims", SHAPES)
def test_left_right_vs_kron(dims):
    rho = rand_rho_t(dims)
    for k, d in enumerate(dims):
        M = RNG.normal(size=(d, d)) + 1j * RNG.normal(size=(d, d))
        F = embed(M, k, dims)
        assert np.allclose(as_matrix(apply_left(rho, M, k)), F @ as_matrix(rho))
        assert np.allclose(as_matrix(apply_right_dag(rho, M, k)), as_matrix(rho) @ F.conj().T)
        assert np.allclose(apply_axis(rho, M, k), apply_left(rho, M, k))


@pytest.mark.parametrize("dims", SHAPES)
def test_cavity_loss_vs_kron(dims):
    rho = rand_rho_t(dims)
    kt = 0.3
    for k in (len(dims) - 2, len(dims) - 1):
        nf = dims[k]
        ref = kraus_dense(as_matrix(rho), [embed(K, k, dims) for K in cavity_loss_kraus(kt, nf, 1)])
        out = apply_cavity_loss(rho, cavity_loss_weights(kt, nf, 1), k)
        assert np.allclose(as_matrix(out), ref, atol=1e-14)
        assert np.allclose(apply_kraus(rho, cavity_loss_kraus(kt, nf, 1), k), out, atol=1e-14)


@pytest.mark.parametrize("dims", SHAPES)
def test_qubit_channel_vs_kron(dims):
    rho = rand_rho_t(dims)
    H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
    channels = {"phase covariant": qubit_kraus(0.3, 0.2),
                "general": [H @ K for K in qubit_kraus(0.3, 0.2)]}
    for name, kr in channels.items():
        for k in range(len(dims) - 2):
            ref = kraus_dense(as_matrix(rho), [embed(K, k, dims) for K in kr])
            assert np.allclose(as_matrix(apply_qubit_channel(rho, kr, k)), ref, atol=1e-14), name


def test_tensor_in_trace_out_roundtrip():
    rho = rand_rho_t((2, 2, 3, 3))
    r5 = tensor_in_ground(rho, 0)
    g = np.zeros((2, 2), dtype=complex)
    g[0, 0] = 1
    assert np.allclose(as_matrix(r5), np.kron(g, as_matrix(rho)))
    assert np.allclose(trace_out(r5, 0), rho)
    r5 = apply_kraus(r5, [embed(U, 0, (2,)) for U in [C.RY_P]], 0)
    assert np.allclose(trace_out(r5, 0), rho)


def test_conditional_and_pair_vs_dense():
    nf = 5
    dims = (2, 2, nf, nf)
    rho = rand_rho_t(dims)
    layer = unpack_params(random_parameters(1, RNG), 1)[0]
    pd = C.ecd_rotation_parts(layer["beta_d"], layer["theta_d"], layer["phi_d"], nf)
    pe = C.ecd_rotation_parts(layer["beta_e"], layer["theta_e"], layer["phi_e"], nf)

    def u_pair(parts, kq, kc):
        coef, ops = parts
        U = np.zeros((int(np.prod(dims)),) * 2, dtype=complex)
        for i in range(2):
            for c in range(2):
                ket = np.zeros((2, 2))
                ket[i, c] = 1
                U += coef[i, c] * embed(ket, kq, dims) @ embed(ops[i], kc, dims)
        return U

    Ud, Ue = u_pair(pd, 0, 2), u_pair(pe, 1, 3)
    for U in (Ud, Ue):
        assert np.allclose(U.conj().T @ U, np.eye(U.shape[0]), atol=1e-12)
    # conditional form == ECD(beta) R built from the explicit gate definition, on (d, A) (x) 1_(e, B)
    R = rotation_np(layer["theta_d"], layer["phi_d"])
    U4 = (C.ecd_u4(layer["beta_d"], nf).reshape(2 * nf, 2 * nf) @ np.kron(R, np.eye(nf))).reshape(2, nf, 2, nf)
    Ud_ref = np.einsum("iajb,xy,uv->ixauj ybv".replace(" ", ""), U4, np.eye(2), np.eye(nf))
    assert np.allclose(Ud, Ud_ref.reshape(Ud.shape), atol=1e-12)

    U = Ue @ Ud
    m = as_matrix(rho)
    ref = U @ m @ U.conj().T
    assert np.allclose(as_matrix(conditional_left(conditional_left(rho, *pd, 0, 2), *pe, 1, 3)), U @ m)
    assert np.allclose(as_matrix(pair_conditional_left(rho, pd, pe)), U @ m)
    assert np.allclose(as_matrix(apply_conditional(apply_conditional(rho, *pd, 0, 2), *pe, 1, 3)), ref)
    assert np.allclose(as_matrix(C.ideal_block(rho, layer, nf)), ref)
    assert np.allclose(as_matrix(C.ideal_block_reference(rho, layer, nf)), ref)
    assert np.allclose(as_matrix(apply_two_mode_unitary(rho, U4, 0, 2)), Ud @ m @ Ud.conj().T)


def _check_state(rho, tol=1e-12):
    m = as_matrix(rho)
    assert np.allclose(m, m.conj().T, atol=tol)
    assert abs(np.trace(m) - 1.0) < 1e-10
    assert np.linalg.eigvalsh(m).min() > -tol


@pytest.mark.parametrize("dims", SHAPES)
def test_every_channel_keeps_a_valid_state(dims):
    dev = EICKBUSCH.with_(noise_scale=40.0)
    rho = rand_rho_t(dims)
    nq = len(dims) - 2
    for k in range(nq):
        _check_state(apply_qubit_channel(rho, qubit_kraus(0.4, 0.3), k))
    for k in (nq, nq + 1):
        for jmax in (None, 1):
            _check_state(apply_cavity_loss(rho, cavity_loss_weights(0.4, dims[k], jmax), k))
    _check_state(C.transmon_idle(rho, dev, 1e-6, axes=tuple(range(nq))))
    _check_state(C.cavity_idle(rho, dev, 1e-6, axes=(nq, nq + 1)))
    if nq == 2:
        _check_state(C.jp_direct(rho, dev))
        _check_state(C.jp_direct(rho, dev, keep_q=True))
        _check_state(C.apply_cavity_channel(rho, C.jp_cavity_channel(dev.with_(jmax=None), dims[2], dims[2] - 1)))


@pytest.mark.parametrize("method", ["direct", "fast"])
def test_noisy_layer_is_cptp(method):
    nf = 6
    dev = EICKBUSCH.with_(noise_scale=30.0, jmax=None)
    rho = C.vacuum_rho(nf)
    for layer in unpack_params(random_parameters(2, RNG), 2):
        rho = C.noisy_layer(rho, layer, dev, method=method, jumps=nf - 1)
        _check_state(rho)


@pytest.mark.parametrize("spec", [dict(), dict(kerr=2e4 * 2 * np.pi, cross_kerr=1e4 * 2 * np.pi)])
def test_fast_channel_matches_direct(spec):
    nf = 6
    dev = EICKBUSCH.with_(noise_scale=30.0, jmax=None, n_slice=2, **spec)
    rho = rand_rho_t((2, 2, nf, nf))
    direct = C.jp_direct(C.cavity_idle(C.transmon_idle(rho, dev, dev.tau_ecd), dev, dev.tau_ecd), dev)
    Cj = C.jp_cavity_channel(dev, nf, nf - 1, dev.tau_ecd)
    fast = C.apply_cavity_channel(C.transmon_idle(rho, dev, dev.tau_ecd + dev.tau_jp), Cj)
    assert np.abs(fast - direct).max() < 1e-13


def test_fast_channel_jump_cap_is_small_at_device_rates():
    nf = 12
    dev = EICKBUSCH.with_(jmax=None)
    rho = C.ideal_block(C.vacuum_rho(nf), unpack_params(random_parameters(1, RNG), 1)[0], nf)
    exact = C.apply_cavity_channel(rho, C.jp_cavity_channel(dev, nf, nf - 1, dev.tau_ecd))
    capped = C.apply_cavity_channel(rho, C.jp_cavity_channel(dev, nf, 3, dev.tau_ecd))
    assert np.abs(exact - capped).max() < 1e-10


def test_apply_cavity_channel_vs_loop():
    nf, J = 5, 2
    dev = EICKBUSCH.with_(noise_scale=50.0)
    Cj = C.jp_cavity_channel(dev, nf, J, dev.tau_ecd)
    rho = rand_rho_t((2, 2, nf, nf))
    ref = np.zeros_like(rho)
    for j in range(J + 1):
        for k in range(J + 1):
            for n in range(j, nf):
                for m in range(k, nf):
                    for n2 in range(j, nf):
                        for m2 in range(k, nf):
                            ref[:, :, n - j, m - k, :, :, n2 - j, m2 - k] += Cj[j, k, n, m, n2, m2] * rho[:, :, n, m, :, :, n2, m2]
    assert np.allclose(C.apply_cavity_channel(rho, Cj), ref, atol=1e-15)


def test_channel_disk_cache(tmp_path, monkeypatch):
    nf = 5
    dev = EICKBUSCH.with_(noise_scale=3.0)
    monkeypatch.setenv(C.CHANNEL_CACHE_ENV, str(tmp_path))
    C.get_cavity_channel.cache_clear()
    a = C.get_cavity_channel(dev, nf, 2, dev.tau_ecd)
    C.get_cavity_channel.cache_clear()
    b = C.get_cavity_channel(dev, nf, 2, dev.tau_ecd)
    C.get_cavity_channel.cache_clear()
    assert isinstance(b, np.memmap) and np.array_equal(np.asarray(b), np.asarray(a))
    assert np.array_equal(np.asarray(b), C.jp_cavity_channel(dev, nf, 2, dev.tau_ecd))


# ---------------------------------------------------------------------------
# (c) end to end
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("method", ["fast", "direct"])
def test_zero_noise_matches_noiseless_ket(method):
    nf, L = 8, 3
    x = random_parameters(L, RNG)
    ket = apply_circuit(x, L, nf)
    rho = C.apply_noisy_circuit(x, L, nf, NOISELESS, method=method)
    assert np.abs(as_matrix(rho) - np.outer(ket, ket.conj())).max() < 1e-12


def test_ideal_jp_sequence_is_jp_diag():
    nf = 6
    rho = rand_rho_t((2, 2, nf, nf))
    out = C.jp_direct(rho, NOISELESS)
    U = np.kron(np.eye(4), np.diag(jp_diag(nf).reshape(-1)))
    assert np.allclose(as_matrix(out), U @ as_matrix(rho) @ U.conj().T, atol=1e-13)


def _full_me_ops(dev: Device, nf: int):
    dims = (2, 2, 2, nf, nf)  # q, d, e, A, B
    n = np.diag(np.arange(nf, dtype=complex))
    a = np.diag(np.sqrt(np.arange(1, nf)), 1).astype(complex)
    I = lambda k, M: embed(M, k, dims)  # noqa: E731
    nA, nB = I(3, n), I(4, n)
    D = int(np.prod(dims))
    H = (-dev.chi_qa * (nA + nB) @ I(0, P_E)
         - dev.chi_spectator * (nA @ I(1, P_E) + nB @ I(2, P_E))
         - 0.5 * dev.kerr * (nA @ (nA - np.eye(D)) + nB @ (nB - np.eye(D)))
         - dev.cross_kerr * nA @ nB)
    c_ops = [np.sqrt(dev.kappa) * I(3, a), np.sqrt(dev.kappa) * I(4, a)]
    for k in (0, 1, 2):
        c_ops += [np.sqrt(dev.gamma) * I(k, SIG_M), np.sqrt(dev.gamma_phi / 2) * I(k, SIG_Z)]
    return dims, H, c_ops


def _me_jp(rho4: np.ndarray, dev: Device, nf: int) -> np.ndarray:
    dims, H, c_ops = _full_me_ops(dev, nf)
    q = lambda U: embed(U, 0, dims)  # noqa: E731
    r = as_matrix(tensor_in_ground(rho4, 0))
    r = q(C.RY_P) @ r @ q(C.RY_P).conj().T
    r = mesolve_final(r, H, c_ops, dev.tau_wait, dims)
    r = q(C.JP_MID) @ r @ q(C.JP_MID).conj().T
    r = mesolve_final(r, H, c_ops, dev.tau_wait, dims)
    r = q(C.RY_M) @ r @ q(C.RY_M).conj().T
    return r


def test_wait_hamiltonian_diagonal_matches_operator_form():
    nf = 4
    dev = EICKBUSCH.with_(chi_spectator=2 * np.pi * 3e4, kerr=2 * np.pi * 5e3, cross_kerr=2 * np.pi * 2e3)
    _, H, _ = _full_me_ops(dev, nf)
    assert np.allclose(np.diag(H), C.wait_hamiltonian(dev, nf, with_de=True).ravel())
    assert np.allclose(H, np.diag(np.diag(H)))


@pytest.mark.parametrize("spec", [dict(), dict(chi_spectator=2 * np.pi * 3e4, kerr=2 * np.pi * 5e3)])
def test_jp_sequence_vs_full_master_equation(spec):
    """All three transmons + both cavities in one Lindblad equation (QuTiP) vs the Strang-Kraus sequence."""
    nf = 6
    rho = rand_rho_t((2, 2, nf, nf))
    dev = EICKBUSCH.with_(jmax=None, **spec)
    assert dev.n_slice == 8
    me = _me_jp(rho, dev, nf)
    noise_effect = trace_dist(me, as_matrix(C.jp_direct(rho, NOISELESS.with_(**spec), keep_q=True)))
    err = trace_dist(as_matrix(C.jp_direct(rho, dev, keep_q=True)), me)
    assert err < 1e-4 and err < 0.005 * noise_effect
    # one slice per wait misplaces the coupler decay relative to the conditional phase (chi t = pi)
    assert trace_dist(as_matrix(C.jp_direct(rho, dev.with_(n_slice=1), keep_q=True)), me) > 10 * err
    # Strang splitting is second order: 4x the slices -> ~16x smaller error
    loud = dev.with_(noise_scale=20.0)
    me = _me_jp(rho, loud, nf)
    errs = [trace_dist(as_matrix(C.jp_direct(rho, loud.with_(n_slice=s), keep_q=True)), me) for s in (4, 16)]
    assert errs[1] < errs[0] / 10


def test_noisy_sim_zero_noise_equals_noiseless_sim():
    lay = KbitLayout(3, 8, encoding="gray")
    E = RNG.normal(size=256)
    gs = format(int(np.argmin(E)), "08b")
    x = random_parameters(2, RNG)
    a = XorKbitSim(lay, E, 2, gs, xa=3, xb=5)
    b = NoisyXorKbitSim(lay, E, 2, gs, xa=3, xb=5, device=NOISELESS)
    assert np.abs(a.probs_from_x(x) - b.probs_from_x(x)).max() < 1e-13
    assert np.isclose(a.cost(x), b.cost(x), rtol=1e-12)
    n = NoisyXorKbitSim(lay, E, 2, gs, device=EICKBUSCH.with_(noise_scale=5.0))
    p = n.probs_from_x(x)
    assert np.isclose(p.sum(), 1.0) and p.min() >= 0
    p[0] = -1.0
    assert n.probs_from_x(x)[0] >= 0  # cached probabilities are returned as a copy


def test_runner_worker_zero_noise_reproduces_noiseless_trial():
    from noiseless.run_relayout_kbit import ham_paths, trial_seed, worker

    path = ham_paths(8, "wmaxsat")[0]
    job = dict(n=8, L=2, s=2, r=3, R=1, lr=0.05, nf=8, inst=0, trial=0, seed=trial_seed(20260917, 0, 0),
               path=str(path), ham_set="wmaxsat", method="fast", nf_check=0, explore=1, topk=1,
               explore_mask="perm", xeta=16.0, reta=2.0, xL=2, xs=None, cool=0, ceta=64.0, clr=0.05, rawk=0,
               polt=1, xbeta=None, xkick=None, code="gray", explore_perm=None)
    a = worker(job)
    b = worker({**job, "sim_cls": NoisyXorKbitSim, "sim_kwargs": {"device": NOISELESS}})
    assert np.abs(np.array(a["x"]) - np.array(b["x"])).max() < 1e-10
    assert np.isclose(a["p_gs"], b["p_gs"], rtol=1e-9)
