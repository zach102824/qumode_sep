#!/usr/bin/env python3
"""Bars and (λ, μ) threading for the cavity-Λ grid. No training."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from Nov_3rd.hva_safe import (
    LAMBDA_X,
    MU_XX,
    dense_H,
    expectation_H,
    ground_bundle,
    hamiltonian_matvec,
    plus_state,
    self_check,
)
from Nov_3rd.scripts.run_lambda_mu_grid_lambda import (
    BAR_GAP,
    BAR_LAMBDA,
    BAR_R,
    SAFE_CELL,
    cells,
    coeffs,
    is_safe,
    _self_test_gate,
)


def main() -> None:
    _self_test_gate()
    assert cells()[0] == (0.50, 0.0)
    assert SAFE_CELL in cells()
    assert len(cells()) == 15
    assert abs(BAR_LAMBDA - 0.80) < 1e-12
    assert abs(BAR_GAP - 0.03) < 1e-12
    assert abs(BAR_R - 0.85) < 1e-12
    assert is_safe(*SAFE_CELL)

    # SAFE defaults are unchanged.
    assert abs(LAMBDA_X + 0.75) < 1e-12
    assert abs(MU_XX + 0.50) < 1e-12

    rng = np.random.default_rng(0)
    n = 6
    E = rng.integers(0, 5, size=1 << n).astype(np.float64)
    Et = torch.tensor(E, dtype=torch.float64).reshape(1, -1)
    psi = plus_state(1, n, torch.float64)

    for lam, mu in ((0.50, 0.0), (0.75, -0.50), (1.00, -0.35)):
        lx, mx = coeffs(lam, mu)
        H = dense_H(E, n, lambda_x=lx, mu_xx=mx)
        v = rng.normal(size=1 << n)
        if np.max(np.abs(hamiltonian_matvec(v, E, n, lambda_x=lx, mu_xx=mx) - H @ v)) > 1e-8:
            raise AssertionError(f"matvec mismatch at λ={lam} μ={mu}")
        bundle = ground_bundle(E, n, lambda_x=lx, mu_xx=mx)
        e_dense = float(np.linalg.eigvalsh(H)[0])
        if abs(bundle["E0"] - e_dense) > 1e-8:
            raise AssertionError(f"E0 {bundle['E0']} != {e_dense}")
        h_plus = float(expectation_H(psi, Et, n, lambda_x=lx, mu_xx=mx))
        # |+⟩: ⟨ΣX⟩=n, ⟨ΣXX⟩=n (one pair-sum per qubit, three pairs per cavity).
        h_formula = float(E.mean() + lx * n + mx * n)
        if abs(h_plus - h_formula) > 1e-8:
            raise AssertionError(f"|+⟩ {h_plus} != {h_formula} at λ={lam} μ={mu}")

    # μ=0 drops the XX piece relative to μ=−0.50 at the same λ.
    h_xx = float(expectation_H(psi, Et, n, lambda_x=-0.75, mu_xx=-0.50))
    h_0 = float(expectation_H(psi, Et, n, lambda_x=-0.75, mu_xx=0.0))
    if abs((h_xx - h_0) - (-0.50 * n)) > 1e-8:
        raise AssertionError(f"XX shift {h_xx - h_0}")

    # A non-SAFE point must not silently reuse the SAFE matrix.
    e_safe = ground_bundle(E, n)["E0"]
    e_mu0 = ground_bundle(E, n, lambda_x=-0.75, mu_xx=0.0)["E0"]
    if abs(e_safe - e_mu0) < 1e-6:
        raise AssertionError("μ did not enter the ground state")

    self_check()
    print("PASS test_lambda_mu_grid")


if __name__ == "__main__":
    main()
