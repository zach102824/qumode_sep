"""Scaling-study family F1: generator, n-qubit HEA / RY-only product, baselines."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "Hamiltonians"))

import four_sat_scaling as gen  # noqa: E402
from noiseless import hea  # noqa: E402
from noiseless.scaling_ansatz import (  # noqa: E402
    HEAnSimulator,
    ProductRYSimulator,
    hea_state_n,
    snake_cz_sublayers,
    snake_lattice_edges,
    spectrum,
)


@pytest.fixture(scope="module", params=[(8, 0), (8, 1), (12, 0), (12, 3)])
def inst(request):
    n, i = request.param
    return gen.generate_instance(n, i)


def test_generator_unique_planted_rigid(inst):
    n = inst["n"]
    e = gen.unsat_counts(inst["clauses"], inst["polarities"], n)
    assert int(e.min()) == 0
    assert int(np.sum(e == 0)) == 1
    assert int(np.argmin(e)) == inst["ground_index"] == int(inst["ground_bitstring"], 2)
    assert 2 <= inst["weight"] <= n - 2
    assert inst["ground_bitstring"] not in gen.trivial_bitstrings(n)
    g = inst["ground_index"]
    for s in range(n):  # local rigidity
        assert e[g ^ (1 << (n - 1 - s))] >= 1
    # clause-wise definition agrees with the mask evaluation
    bits = gen.bits_from_index(g ^ 5, n)
    direct = sum(int(np.all(bits[v] == gen.violating_pattern(p))) for v, p in zip(inst["clauses"], inst["polarities"]))
    assert direct == e[g ^ 5]
    # m accounting and no duplicate clauses
    assert inst["m"] == inst["m_base"] + inst["n_topup"] == len(inst["clauses"])
    keys = {gen._key(v, p) for v, p in zip(inst["clauses"], inst["polarities"])}
    assert len(keys) == inst["m"]
    lo, hi = gen.BASE_RATIO_RANGE
    assert lo * n - 1 <= inst["m_base"] <= hi * n + 1
    assert gen.sat_solver_unique(inst["clauses"], inst["polarities"], gen.bits_from_index(g, n))


def test_generator_deterministic():
    a = gen.generate_instance(12, 5)
    b = gen.generate_instance(12, 5)
    assert np.array_equal(a["clauses"], b["clauses"]) and np.array_equal(a["polarities"], b["polarities"])


def test_brute_force_enumeration_matches_spectrum():
    rng = np.random.default_rng(3)
    n = 12
    cl = np.stack([np.sort(rng.choice(n, 4, replace=False)) for _ in range(25)])
    po = rng.choice([-1, 1], size=cl.shape)
    e = gen.unsat_counts(cl, po, n)
    assert np.array_equal(gen.enumerate_solutions(cl, po, n), np.flatnonzero(e == 0))


def test_snake_lattice_n8_matches_hea():
    assert snake_cz_sublayers(8) == hea.CZ_SUBLAYERS
    assert snake_lattice_edges(8) == hea.lattice_edges_1idx()
    for n in (8, 12, 16, 20):
        pairs = [tuple(sorted(e)) for sub in snake_cz_sublayers(n) for e in sub]
        assert len(pairs) == len(set(pairs)) == 3 * n // 2 - 2
        assert set(pairs) == snake_lattice_edges(n)


@pytest.mark.parametrize("L", [0, 1, 2, 3])
def test_hea_n8_state_matches_existing(L):
    rng = np.random.default_rng(10 + L)
    for _ in range(3):
        x = rng.uniform(-4, 4, 8 * (L + 1))
        assert np.allclose(hea_state_n(x, 8), hea.hea_state(x), atol=1e-12)


@pytest.mark.parametrize("L", [1, 2])
def test_hea_n8_energy_and_cost_match_existing_arm(L):
    # existing n=8 family instance + existing HEASimulator
    from noiseless.encoding import load_four_sat_npz, z_terms_from_npz
    from noiseless.qaoa import diagonal_spectrum_from_terms

    path = sorted((ROOT / "Hamiltonians" / "four_sat").glob("four_sat_0*.npz"))[3]
    inst8 = load_four_sat_npz(path)
    terms, meta = z_terms_from_npz(path)
    e_full = diagonal_spectrum_from_terms(terms, float(meta["identity"]))
    d = np.load(path)
    assert np.allclose(spectrum(d["clauses"], d["polarities"], 8), e_full)
    old = hea.HEASimulator(energies=e_full, ground_bitstring=inst8["ground_bitstring"])
    new = HEAnSimulator(e_full, inst8["ground_bitstring"], 8, L)
    rng = np.random.default_rng(L)
    for _ in range(3):
        x = rng.uniform(0, np.pi, 8 * (L + 1))
        eo, en = old.evaluate(x), new.evaluate(x)
        assert np.isclose(eo["energy_mean"], en["energy_mean"], atol=1e-12)
        assert np.isclose(eo["p_gs"], en["p_gs"], atol=1e-12)
        assert np.isclose(old.cost(x, 1.7), new.cost(x, 1.7), atol=1e-12)
        assert np.isclose(old.refresh_eta(x, 1), new.refresh_eta(x, 1), atol=1e-12)
        old.eta_ctrl = type(old.eta_ctrl)()
        new.eta_ctrl = type(new.eta_ctrl)()


def test_hea_n8_full_trial_matches_existing():
    from noiseless.scaling_ansatz import optimize_trial

    path = sorted((ROOT / "Hamiltonians" / "four_sat").glob("four_sat_0*.npz"))[0]
    d = np.load(path)
    e = spectrum(d["clauses"], d["polarities"], 8)
    gs = format(int(np.argmin(e)), "08b")
    ro = hea.optimize_hea_trial(hea.HEASimulator(energies=e, ground_bitstring=gs), 1, maxiter=30, rng=np.random.default_rng(5))
    rn = optimize_trial(HEAnSimulator(e, gs, 8, 1), 16, maxiter=30, rng=np.random.default_rng(5))
    assert np.allclose(ro.x, rn.x, atol=1e-9)
    assert np.isclose(ro.p_gs, rn.p_gs, atol=1e-9)


def test_ry0_analytic_matches_statevector_n8(inst):
    if inst["n"] != 8:
        pytest.skip("n=8 only")
    e = spectrum(inst["clauses"], inst["polarities"], 8)
    sim = ProductRYSimulator(inst["clauses"], inst["polarities"], inst["ground_bitstring"], 8, energies=e)
    rng = np.random.default_rng(0)
    for _ in range(5):
        x = rng.uniform(-4, 4, 8)
        p = hea.born_probs(hea.hea_state(x)) if hasattr(hea, "born_probs") else np.abs(hea.hea_state(x)) ** 2
        e_mean, p_gs, ml = sim.analytic(x)
        assert np.isclose(e_mean, float(p @ e), atol=1e-12)
        assert np.isclose(p_gs, p[inst["ground_index"]], atol=1e-12)
        assert ml == format(int(np.argmax(p)), "08b")
        assert np.allclose(sim.probs_from_x(x), p, atol=1e-12)
        # exact Gibbs cost equals the existing hea_ry0 arm's cost
        old = hea.HEASimulator(energies=e, ground_bitstring=inst["ground_bitstring"])
        assert np.isclose(sim.cost(x, 2.3), old.cost(x, 2.3), atol=1e-12)


def test_ry0_sampled_cost_converges_to_exact():
    inst = gen.generate_instance(12, 1)
    e = spectrum(inst["clauses"], inst["polarities"], 12)
    ex = ProductRYSimulator(inst["clauses"], inst["polarities"], inst["ground_bitstring"], 12, energies=e)
    sa = ProductRYSimulator(inst["clauses"], inst["polarities"], inst["ground_bitstring"], 12, cost_mode="sampled", n_samples=400_000)
    sa.resample(np.random.default_rng(1))
    x = np.random.default_rng(2).uniform(0, np.pi, 12)
    for eta in (0.3, 1.0, 3.0):
        assert abs(sa.cost(x, eta) - ex.cost(x, eta)) < 0.02
    # sampled energies are the true energies of the sampled bitstrings
    bits = (sa._U[:50] < sa.p1(x)).astype(int)
    idx = [int("".join(map(str, b)), 2) for b in bits]
    assert np.array_equal(sa.sample_energies(x)[:50], e[idx])


def test_classical_baselines_small():
    from noiseless.classical_baselines import build_structs, run_baselines, sa_run, walksat_run

    inst = gen.generate_instance(12, 2)
    out = run_baselines(inst["clauses"], inst["polarities"], 12, seed=123, trials=10)
    assert out["walksat_success"][str(out["budgets"][-1])] == 1.0
    assert all(0.0 <= v <= 1.0 for v in out["sa_success"].values())
    cl, pa, ptr, lst = build_structs(inst["clauses"], inst["polarities"], 12)
    for s in range(5):
        h = walksat_run(s, 12, cl, pa, ptr, lst, 401, 0.5)
        assert h == -1 or 1 <= h <= 401
        h = sa_run(s, 12, cl, pa, ptr, lst, 4010, 2.0, 0.05)
        assert h == -1 or 1 <= h <= 4010
