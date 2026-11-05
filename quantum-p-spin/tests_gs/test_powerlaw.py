"""GS tests for power-law p=3 TFIM on a ring."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG.parent))
sys.path.insert(0, str(PKG))

from pspin import (  # noqa: E402
    apply_H,
    make_powerlaw_instance,
    ring_diameter,
    powerlaw_weights,
    ground_state_dense,
)


def test_ring_diameter():
    assert ring_diameter(0, 1, 2, 6) == 2
    assert ring_diameter(0, 1, 3, 6) == 3
    assert ring_diameter(0, 2, 4, 6) == 4  # equally spaced on n=6 → diam=4? gaps all 2, max=2, diam=4
    # n=9 consecutive
    assert ring_diameter(0, 1, 2, 9) == 2
    print("PASS ring_diameter")


def test_ferro_classical_allplus():
    """Ferro: Hc(|0…0⟩) = -n (Kac norm)."""
    for n, alpha in [(6, 0.0), (6, 1.0), (9, 1.5)]:
        inst = make_powerlaw_instance(n, Gamma=0.0, alpha=alpha, seed=0, signed=False)
        assert abs(inst.Hc_diag[0] + float(n)) < 1e-8, (n, alpha, inst.Hc_diag[0])
        assert abs(inst.E0 - inst.Hc_diag.min()) < 1e-8
    print("PASS ferro_classical_allplus")


def test_ferro_residual():
    for n, G, a in [(6, 0.5, 1.0), (9, 0.75, 1.0), (6, 1.0, 0.5)]:
        inst = make_powerlaw_instance(n, G, a, seed=0, signed=False)
        Hv = apply_H(inst.gs, inst.Hc_diag, G, n)
        res = np.linalg.norm(Hv - inst.E0 * inst.gs)
        assert res < 1e-5, res
    print("PASS ferro_residual")


def test_signed_seed_diversity():
    a = make_powerlaw_instance(6, 0.5, 1.0, seed=0, signed=True)
    b = make_powerlaw_instance(6, 0.5, 1.0, seed=1, signed=True)
    assert not np.allclose(a.J, b.J)
    assert abs(a.E0 - b.E0) > 1e-6 or not np.allclose(a.gs, b.gs)
    # replay
    a2 = make_powerlaw_instance(6, 0.5, 1.0, seed=0, signed=True)
    assert np.allclose(a.J, a2.J) and abs(a.E0 - a2.E0) < 1e-10
    print("PASS signed_seed_diversity")


def test_alpha0_flat_weights():
    triples, w, N = powerlaw_weights(6, 0.0)
    assert np.allclose(w, 1.0)
    # N = C(6,3)/6 = 20/6
    assert abs(N - 20.0 / 6.0) < 1e-12
    print("PASS alpha0_flat_weights")


def test_signed_hermitian_eigh():
    inst = make_powerlaw_instance(6, 0.5, 1.0, seed=3, signed=True)
    Ed, _ = ground_state_dense(inst.Hc_diag, 0.5, 6)
    assert abs(Ed - inst.E0) < 1e-4
    print(f"PASS signed_hermitian_eigh E0={inst.E0:.6f}")


def test_Gamma_infty_plus():
    inst = make_powerlaw_instance(6, 20.0, 1.0, seed=0, signed=True)
    plus = np.ones(64, dtype=np.complex128) / 8.0
    k = int(np.argmax(np.abs(inst.gs)))
    gs = inst.gs * (np.abs(inst.gs[k]) / (inst.gs[k] + 1e-30))
    F = abs(np.vdot(plus, gs)) ** 2
    assert F > 0.99, F
    print(f"PASS Gamma_infty_plus F={F:.6f}")


def main():
    test_ring_diameter()
    test_ferro_classical_allplus()
    test_ferro_residual()
    test_signed_seed_diversity()
    test_alpha0_flat_weights()
    test_signed_hermitian_eigh()
    test_Gamma_infty_plus()
    print("ALL POWERLAW GS TESTS PASSED")


if __name__ == "__main__":
    main()
