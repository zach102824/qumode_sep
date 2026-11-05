"""Ground-state confirmation tests for quantum p-spin H = Hc + Hx."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PKG = Path(__file__).resolve().parents[1]
ROOT = PKG.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(PKG))

from pspin import (  # noqa: E402
    apply_H,
    ground_state_dense,
    ground_state_lanczos,
    hc_diagonal,
    make_instance,
    scale_p3,
    all_triples,
)


TOL_RES = 1e-5
TOL_E = 1e-5


def _rayleigh(gs, Hc, Gamma, n):
    return float(np.vdot(gs, apply_H(gs, Hc, Gamma, n)).real)


def test_hermiticity_real_E0():
    for n, G, seed in [(6, 0.5, 0), (9, 0.5, 1)]:
        inst = make_instance(n, G, seed)
        assert abs(inst.E0.imag if hasattr(inst.E0, "imag") else 0.0) < 1e-12
        # build dense H and check Hermitian
        dim = 1 << n
        H = np.diag(inst.Hc_diag.astype(np.complex128))
        idx = np.arange(dim)
        for i in range(n):
            shift = n - 1 - i
            partner = idx ^ (1 << shift)
            mask = partner > idx
            H[idx[mask], partner[mask]] -= G
            H[partner[mask], idx[mask]] -= G
        assert np.allclose(H, H.conj().T), f"H not Hermitian n={n}"
        evals = np.linalg.eigvalsh(H)
        assert abs(evals[0] - inst.E0) < 1e-4, (evals[0], inst.E0)
    print("PASS hermiticity_real_E0")


def test_residual():
    for n, G, seed in [(6, 0.5, 0), (9, 0.5, 2), (12, 0.5, 0)]:
        inst = make_instance(n, G, seed)
        Hv = apply_H(inst.gs, inst.Hc_diag, inst.Gamma, inst.n)
        res = np.linalg.norm(Hv - inst.E0 * inst.gs) / np.linalg.norm(inst.gs)
        assert res < TOL_RES, f"residual n={n}: {res}"
    print("PASS residual")


def test_variational_bound():
    rng = np.random.default_rng(0)
    for n, G in [(6, 0.5), (9, 0.5)]:
        inst = make_instance(n, G, seed=3)
        for _ in range(8):
            v = rng.normal(size=1 << n) + 1j * rng.normal(size=1 << n)
            v = v / np.linalg.norm(v)
            Er = float(np.vdot(v, apply_H(v, inst.Hc_diag, G, n)).real)
            assert Er >= inst.E0 - 1e-6, (Er, inst.E0)
    print("PASS variational_bound")


def test_Gamma_infty_plus_state():
    """Large Γ → GS ≈ |+⟩^⊗n."""
    n, G, seed = 6, 20.0, 0
    inst = make_instance(n, G, seed)
    plus = np.ones(1 << n, dtype=np.complex128) / np.sqrt(1 << n)
    # fix phase
    k = int(np.argmax(np.abs(inst.gs)))
    phase = np.abs(inst.gs[k]) / (inst.gs[k] + 1e-30)
    gs = inst.gs * phase
    F = abs(np.vdot(plus, gs)) ** 2
    assert F > 0.99, f"Γ→∞ fidelity to |+ⁿ⟩ = {F}"
    print(f"PASS Gamma_infty F={F:.6f}")


def test_Gamma_zero_classical():
    n, G, seed = 6, 0.0, 7
    inst = make_instance(n, G, seed)
    E_min = float(inst.Hc_diag.min())
    assert abs(inst.E0 - E_min) < 1e-8, (inst.E0, E_min)
    # GS should be concentrated on argmin bitstring(s)
    idxs = np.where(np.isclose(inst.Hc_diag, E_min))[0]
    mass = float(np.sum(np.abs(inst.gs[idxs]) ** 2))
    assert mass > 0.999, mass
    print(f"PASS Gamma0 classical mass={mass:.6f}")


def test_dense_vs_lanczos():
    for n in (6, 9):
        inst_seed = 11
        # rebuild Hc
        rng = np.random.default_rng(inst_seed)
        triples = all_triples(n)
        J = rng.normal(0.0, 1.0, size=len(triples))
        s = scale_p3(n)
        Hc = hc_diagonal(n, triples, J, s)
        G = 0.5
        Ed, gsd = ground_state_dense(Hc, G, n)
        El, gsl = ground_state_lanczos(Hc, G, n, seed=inst_seed + 17)
        # phase align
        k = int(np.argmax(np.abs(gsd)))
        gsd = gsd * (np.abs(gsd[k]) / (gsd[k] + 1e-30))
        gsl = gsl * (np.abs(gsl[k]) / (gsl[k] + 1e-30))
        ov = abs(np.vdot(gsd, gsl)) ** 2
        assert abs(Ed - El) < 1e-3, (Ed, El)
        assert ov > 0.999, ov
        print(f"  n={n} Ed={Ed:.6f} El={El:.6f} ov={ov:.8f}")
    print("PASS dense_vs_lanczos")


def test_n6_full_eigh():
    inst = make_instance(6, 0.5, seed=0)
    dim = 64
    H = np.diag(inst.Hc_diag.astype(np.complex128))
    idx = np.arange(dim)
    for i in range(6):
        shift = 5 - i
        partner = idx ^ (1 << shift)
        mask = partner > idx
        H[idx[mask], partner[mask]] -= 0.5
        H[partner[mask], idx[mask]] -= 0.5
    evals, evecs = np.linalg.eigh(H)
    assert abs(evals[0] - inst.E0) < 1e-6
    ov = abs(np.vdot(evecs[:, 0], inst.gs)) ** 2
    assert ov > 0.9999, ov
    print(f"PASS n6_full_eigh ov={ov:.10f}")


def test_seed_replay():
    a = make_instance(6, 0.5, seed=42)
    b = make_instance(6, 0.5, seed=42)
    assert abs(a.E0 - b.E0) < 1e-12
    assert np.allclose(a.J, b.J)
    assert np.allclose(a.gs, b.gs)
    c = make_instance(6, 0.5, seed=43)
    assert abs(c.E0 - a.E0) > 1e-8 or not np.allclose(c.J, a.J)
    print("PASS seed_replay")


def test_rayleigh_equals_E0():
    for n, G, seed in [(6, 0.5, 0), (9, 0.5, 1), (12, 0.5, 0)]:
        inst = make_instance(n, G, seed)
        Er = _rayleigh(inst.gs, inst.Hc_diag, G, n)
        assert abs(Er - inst.E0) < TOL_E, (Er, inst.E0)
    print("PASS rayleigh_equals_E0")


def main():
    test_hermiticity_real_E0()
    test_residual()
    test_variational_bound()
    test_Gamma_infty_plus_state()
    test_Gamma_zero_classical()
    test_dense_vs_lanczos()
    test_n6_full_eigh()
    test_seed_replay()
    test_rayleigh_equals_E0()
    print("ALL GS TESTS PASSED")


if __name__ == "__main__":
    main()
