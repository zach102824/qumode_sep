"""μ=0 transverse field: operator checks and the shallow decision rule."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from Nov_3rd.hva_safe import (
    dense_H,
    expectation_H,
    fourier_state,
    ground_bundle,
    hamiltonian_matvec,
    lambda_of_state,
    plus_state,
    product_energy_xz,
    snap_state,
    snap_state_from_components,
    walsh_degree_components,
    walsh_degree_diag,
)
from Nov_3rd.tf_only import decision_self_test, lambda_coeff


def test_decision_rule():
    decision_self_test()


def test_mu0_has_no_xx_and_matches_histogram():
    n = 6
    lam = 0.75
    rng = np.random.default_rng(0)
    unsat = rng.integers(0, 5, size=1 << n).astype(np.float64)
    lx = lambda_coeff(lam)
    H = dense_H(unsat, n, lambda_x=lx, mu_xx=0.0)
    # Only the diagonal and single-bit X flips are nonzero.
    places = [1 << (n - 1 - q) for q in range(n)]
    allowed = set(places)
    idx = np.arange(1 << n)
    for i in idx:
        row = H[i].copy()
        row[i] = 0.0
        nz = np.flatnonzero(np.abs(row) > 1e-12)
        for j in nz:
            assert int(i ^ int(j)) in allowed
            assert abs(row[j] - lx) < 1e-12
        assert abs(H[i, i] - unsat[i]) < 1e-12
    v = rng.normal(size=1 << n)
    mv = hamiltonian_matvec(v, unsat, n, lambda_x=lx, mu_xx=0.0)
    assert np.max(np.abs(mv - H @ v)) < 1e-8

    bundle = ground_bundle(unsat, n, lambda_x=lx, mu_xx=0.0)
    e0 = float(np.linalg.eigvalsh(H)[0])
    assert abs(bundle["E0"] - e0) < 1e-8
    assert bundle["E0"] <= float(unsat.min()) + 1e-8
    lam_c = np.asarray(lambda_of_state(bundle["psi"], n), dtype=np.float64).reshape(-1)
    assert lam_c.shape == (n // 3,)
    assert np.all(lam_c >= 1.0 / 3.0 - 1e-8)
    assert np.all(lam_c <= 1.0 + 1e-8)

    Et = torch.tensor(unsat, dtype=torch.float64).reshape(1, -1)
    psi = plus_state(1, n, torch.float64)
    h = float(expectation_H(psi, Et, n, lambda_x=lx, mu_xx=0.0))
    assert abs(h - (float(unsat.mean()) - lam * n)) < 1e-8
    e_prod = product_energy_xz(np.full(n, 0.5 * np.pi), unsat, n, lam=lam, mu=0.0)
    assert abs(e_prod - h) < 1e-8
    # A computational-basis state has no X energy when μ=0.
    x = int(np.argmin(unsat))
    theta = np.array([((x >> (n - 1 - i)) & 1) * np.pi for i in range(n)])
    assert abs(product_energy_xz(theta, unsat, n, lam=lam, mu=0.0) - float(unsat[x])) < 1e-8


def test_cached_walsh_matches_snap_state():
    rng = np.random.default_rng(1)
    n = 6
    B = 3
    L = 2
    Cn = torch.tensor(rng.normal(size=(B, 1 << n)), dtype=torch.float64)
    gamma = torch.tensor(rng.normal(scale=0.2, size=(L, 4)), dtype=torch.float64)
    phi = torch.tensor(rng.normal(scale=0.3, size=(L, 4)), dtype=torch.float64)
    diag = walsh_degree_diag(Cn, gamma[0])
    comps = walsh_degree_components(Cn)
    diag2 = torch.tensordot(gamma[0], comps, dims=([0], [0]))
    assert torch.max(torch.abs(diag - diag2)) < 1e-8
    a = snap_state(Cn, gamma, phi)
    b = snap_state_from_components(comps, gamma, phi)
    overlap = torch.sum(a.conj() * b, dim=-1).abs()
    assert torch.min(overlap) > 1 - 1e-8


def test_hva_zero_xx_is_fourier_and_mu0_still_feels_the_mixer():
    """β^XX=0 is Fourier. A nonzero XX mixer still changes ⟨H⟩ at μ=0, because
    it changes the state. That is why this seal does not train HVA: the extra
    angle is not the XX term of H, and the user treats it as degenerate with
    Fourier for the comparison.
    """
    from Nov_3rd.hva_safe import cn_from_unsat, hva_state, state_fidelity

    n = 6
    rng = np.random.default_rng(2)
    E = torch.tensor(rng.integers(0, 4, size=(2, 1 << n)), dtype=torch.float64)
    Cn = cn_from_unsat(E, n)
    g = torch.tensor([0.2, -0.1], dtype=torch.float64)
    b = torch.tensor([0.3, 0.15], dtype=torch.float64)
    z = torch.zeros(2, dtype=torch.float64)
    sf = fourier_state(Cn, b, g)
    sh = hva_state(Cn, b, z, g)
    assert float(torch.min(state_fidelity(sf, sh))) > 1 - 1e-8
    bxx = torch.tensor([0.4, -0.25], dtype=torch.float64)
    sh2 = hva_state(Cn, b, bxx, g)
    h0 = expectation_H(sh, E, n, lambda_x=-0.5, mu_xx=0.0)
    h1 = expectation_H(sh2, E, n, lambda_x=-0.5, mu_xx=0.0)
    assert float(torch.max(torch.abs(h0 - h1))) > 1e-6
