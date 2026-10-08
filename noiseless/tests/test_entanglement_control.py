"""Tests for the controlled entanglement comparison (noiseless/run_entanglement_control.py)."""

from __future__ import annotations

import argparse

import numpy as np
import pytest

from noiseless.encoding import NFOCK
from noiseless.run_entanglement_control import (
    build_arm_u,
    build_jobs,
    entropy_dA_eB,
    entropy_profile,
    gamma_gate_ab,
    jp_gate_ab,
    jp_local_gate_ab,
    parse_arm,
    trial_seed,
)
from noiseless.unitaries import build_fixed_u


def test_g0_is_identity():
    u = build_arm_u("g0")
    assert np.allclose(u, np.eye(2 * 2 * NFOCK * NFOCK))


def test_gamma_quarter_equals_jp_up_to_global_phase():
    """U(γ=π/4) · e^{-iπ/4} == jp exactly, so the g0.25 and jp arms are the same physics."""
    g = gamma_gate_ab(np.pi / 4.0) * np.exp(-1j * np.pi / 4.0)
    assert np.max(np.abs(g - jp_gate_ab())) < 1e-12


def test_jp_matches_unitaries_module():
    u_ref = np.asarray(build_fixed_u("jp").full(), dtype=complex)
    assert np.max(np.abs(build_arm_u("jp") - u_ref)) < 1e-12


def test_jp_local_is_a_product_gate_with_jp_phase_profile():
    u = jp_local_gate_ab()
    # Product structure: kron of the same local parity phase on each cavity.
    local = np.diag(np.array([(-1j) ** (n % 2) for n in range(NFOCK)], dtype=complex))
    assert np.max(np.abs(u - np.kron(local, local))) < 1e-12
    # Matches jp exactly on mixed-parity |n,m⟩ (both give -i); differs only on odd/odd.
    jp = jp_gate_ab()
    for n in range(NFOCK):
        for m in range(NFOCK):
            i = n * NFOCK + m
            if (n + m) % 2 == 1:
                assert abs(u[i, i] - jp[i, i]) < 1e-12


def test_parse_arm_validation():
    assert parse_arm("identity") == ("identity", 0.0)
    assert parse_arm("jp") == ("jp", 0.25)
    assert parse_arm("g0.125") == ("g0.125", 0.125)
    assert parse_arm("jp_local")[1] is None
    with pytest.raises(ValueError):
        parse_arm("g0.6")  # outside [0, 0.5]
    with pytest.raises(ValueError):
        parse_arm("bogus")


def _state_dAeB(cav_a: np.ndarray, cav_b: np.ndarray) -> np.ndarray:
    """|g⟩_d |g⟩_e ⊗ |ψ⟩_A ⊗ |φ⟩_B as a flat 256 vector."""
    qd = np.zeros(2, dtype=complex)
    qd[0] = 1.0
    return np.kron(np.kron(qd, qd), np.kron(cav_a, cav_b))


def test_entropy_jp_entangles_jp_local_does_not():
    plus = np.zeros(NFOCK, dtype=complex)
    plus[0] = plus[1] = 1.0 / np.sqrt(2.0)  # (|0⟩+|1⟩)/√2: equal parity superposition
    psi0 = _state_dAeB(plus, plus)
    assert entropy_dA_eB(psi0) < 1e-10  # product state

    psi_jp = build_arm_u("jp") @ psi0
    assert entropy_dA_eB(psi_jp) > 0.9  # jp entangles the parity qubits (~1 bit)

    psi_local = build_arm_u("jp_local") @ psi0
    assert entropy_dA_eB(psi_local) < 1e-10  # the product control does not

    # Dose–response: entropy grows monotonically with γ on [0, π/4].
    ents = [entropy_dA_eB(build_arm_u(f"g{f}") @ psi0) for f in (0.0625, 0.125, 0.1875, 0.25)]
    assert all(b > a for a, b in zip(ents, ents[1:]))


def test_entropy_profile_identity_arm_stays_zero():
    """With U=identity the circuit factorizes across (d,A)|(e,B): entropy 0 after every layer."""
    rng = np.random.default_rng(0)
    L = 3
    x = rng.normal(size=8 * L)
    from noiseless.circuit_local_ecd import extract_u_ab

    u_ab_id = extract_u_ab(build_arm_u("identity"))
    assert max(entropy_profile(x, L, u_ab_id)) < 1e-10

    u_ab_jp = extract_u_ab(build_arm_u("jp"))
    assert len(entropy_profile(x, L, u_ab_jp)) == L


def test_seeds_are_paired_across_arms():
    args = argparse.Namespace(
        ham_dir="Hamiltonians/four_sat",
        arms="identity,jp,g0.125",
        layers="4",
        trials=2,
        steps=8,
        spsa_a=None,
        spsa_c=0.15,
        spsa_A=10.0,
        optimizer="spsa",
        adam_lr=0.05,
        encoding="binary",
        seed=20260917,
        max_h=2,
    )
    jobs = build_jobs(args)
    by_key: dict[tuple, set] = {}
    for j in jobs:
        by_key.setdefault((j["ham_file"], j["n_layers"], j["trial"]), set()).add(j["seed"])
    assert len(by_key) == 2 * 1 * 2  # 2 H × 1 L × 2 trials
    for seeds in by_key.values():
        assert len(seeds) == 1  # every arm gets the SAME seed
    # Distinct (H, L, trial) → distinct seeds; and x0 drawn from equal seeds is identical.
    all_seeds = [next(iter(s)) for s in by_key.values()]
    assert len(set(all_seeds)) == len(all_seeds)
    assert trial_seed(1, 0, 4, 1) != trial_seed(1, 1, 4, 1) != trial_seed(1, 0, 5, 1)
    from noiseless.circuit_local_ecd import random_parameters

    x_a = random_parameters(4, np.random.default_rng(all_seeds[0]))
    x_b = random_parameters(4, np.random.default_rng(all_seeds[0]))
    assert np.array_equal(x_a, x_b)


# --------------------------------------------------------------------------- 2026-10-08 additions


def test_g_half_is_product_i_parity_parity():
    """g0.5 = exp(iπ/2 Π_AΠ_B) = i·Π_A⊗Π_B: a product gate (second product control)."""
    par = np.diag(np.array([(-1.0) ** n for n in range(NFOCK)], dtype=complex))
    u = gamma_gate_ab(np.pi / 2.0)
    assert np.max(np.abs(u - 1j * np.kron(par, par))) < 1e-12
    assert parse_arm("g0.5") == ("g0.5", 0.5)
    plus = np.zeros(NFOCK, dtype=complex)
    plus[0] = plus[1] = 1.0 / np.sqrt(2.0)
    psi0 = _state_dAeB(plus, plus)
    assert entropy_dA_eB(build_arm_u("g0.5") @ psi0) < 1e-10


def test_pre_u_entropy_equals_previous_layer_and_peak_excludes_final_gate():
    from noiseless.circuit_local_ecd import extract_u_ab
    from noiseless.run_entanglement_control import entropy_profiles, entropy_summary

    rng = np.random.default_rng(3)
    L = 4
    x = rng.normal(size=8 * L)
    u_ab = extract_u_ab(build_arm_u("jp"))
    post, pre = entropy_profiles(x, L, u_ab)
    assert pre[0] < 1e-10  # vacuum + local gates: product
    for k in range(1, L):
        assert abs(pre[k] - post[k - 1]) < 1e-9  # local gates leave the cut entropy unchanged
    summ = entropy_summary(x, L, u_ab)
    assert abs(summ["final_entropy_pre_u"] - post[L - 2]) < 1e-9
    assert abs(summ["peak_entropy_effective"] - max(post[: L - 1])) < 1e-9
    assert summ["final_entropy"] >= 0.0 and min(summ["entropy_profile"]) >= 0.0


def test_final_bus_gate_does_not_change_probabilities():
    """The last U is diagonal in the measured basis: p(x) identical with or without it."""
    from noiseless.circuit_local_ecd import apply_layer_factored, extract_u_ab, unpack_params, vacuum_np

    rng = np.random.default_rng(5)
    L = 3
    x = rng.normal(size=8 * L)
    u_ab = extract_u_ab(build_arm_u("jp"))
    eye = np.eye(NFOCK * NFOCK, dtype=complex)
    layers = unpack_params(x, L)
    ket_a = vacuum_np()
    ket_b = vacuum_np()
    for k, ly in enumerate(layers):
        args = (ly["beta_d"], ly["beta_e"], ly["theta_d"], ly["theta_e"], ly["phi_d"], ly["phi_e"])
        ket_a = apply_layer_factored(ket_a, *args, u_ab)
        ket_b = apply_layer_factored(ket_b, *args, u_ab if k < L - 1 else eye)
    assert np.max(np.abs(np.abs(ket_a) ** 2 - np.abs(ket_b) ** 2)) < 1e-12


def test_seeds_paired_across_layouts_and_layout_perms_recorded():
    args = argparse.Namespace(
        ham_dir="Hamiltonians/four_sat", arms="identity,jp", layers="4", trials=2, steps=8,
        spsa_a=None, spsa_c=0.15, spsa_A=10.0, optimizer="spsa", adam_lr=0.05,
        encoding="binary", seed=20260917, max_h=1, layouts="identity,rule_best,rule_bad",
        protocol="tuned",
    )
    jobs = build_jobs(args)
    assert len(jobs) == 1 * 2 * 3 * 2
    by_trial: dict[int, set] = {}
    for j in jobs:
        by_trial.setdefault(j["trial"], set()).add(j["seed"])
        assert j["protocol"] == "tuned"
        assert (j["layout_perm"] is None) == (j["layout"] == "identity")
    assert all(len(s) == 1 for s in by_trial.values())


def test_tuned_protocol_worker_logs_stage_entropies():
    from noiseless.encoding import EncodingSpec, bits_from_bitstring, denm_from_bits, load_four_sat_npz
    from noiseless.run_entanglement_control import _worker

    args = argparse.Namespace(
        ham_dir="Hamiltonians/four_sat", arms="jp", layers="4", trials=1, steps=8,
        spsa_a=None, spsa_c=0.15, spsa_A=10.0, optimizer="spsa", adam_lr=0.05,
        encoding="binary", seed=20260917, max_h=1, layouts="rule_best", protocol="tuned",
    )
    job = build_jobs(args)[0]
    job["steps_per_stage"] = 3
    rec = _worker(job)
    assert rec["ok"], rec.get("traceback")
    assert rec["protocol"] == "tuned" and rec["nfev"] == 4 * (2 * 3 + 1)
    assert [s["n_layers"] for s in rec["stages"]] == [1, 2, 3, 4]
    for s in rec["stages"]:
        assert "x" not in s and len(s["entropy_profile"]) == s["n_layers"]
    gs = load_four_sat_npz(job["ham_path"])["ground_bitstring"]
    _, _, na, nb = denm_from_bits(bits_from_bitstring(gs), EncodingSpec(tuple(job["layout_perm"])))
    assert na in (0, 7) and nb in (0, 7)
