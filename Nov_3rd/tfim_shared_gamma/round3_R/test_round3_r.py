"""Packing, relabeling, and gradient tests. python3 test_round3_r.py"""

from __future__ import annotations

import numpy as np

import circuit
import model


def check(name, cond, detail=""):
    if not cond:
        raise AssertionError(f"FAIL {name} {detail}")
    print("ok", name)


def test_manifest_packings_and_neel():
    check("manifest", model.sha256_file(model.MANIFEST) == model.EXPECTED_MANIFEST)
    raw, digest = model.load_packings()
    check("packing hash nonempty", len(digest) == 64)
    sub, _ = model.packing("sublattice", 12)
    check("sublattice frozen", sub == [[0, 2, 4], [6, 8, 10], [1, 3, 5], [7, 9, 11]])
    check("sublattice reflection closed", model.reflection_closed(sub, 12))
    check("stride reflection closed", model.reflection_closed(model.packing("stride4", 12)[0], 12))
    check("contiguous reflection closed", model.reflection_closed(model.packing("contiguous", 12)[0], 12))
    # Neel 0101... on sublattice cavities is Fock 0 or 7.
    neel = [i % 2 for i in range(12)]
    for cav in sub:
        bits = [neel[q] for q in cav]
        check(f"neel cavity {cav} uniform", len(set(bits)) == 1, str(bits))
    cont = model.packing("contiguous", 12)[0]
    # one basis state, all weight-1 cavities: symmetric weight 1/81
    idx = 0
    for q, b in enumerate([0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]):
        idx |= b << q
    psi = np.zeros(1 << 12, dtype=np.complex128)
    psi[idx] = 1
    w = model.sym_weight(psi, cont)
    check("bare neel cat 1/81", abs(w - 1 / 81) < 1e-12, str(w))
    check("plus is symmetric", abs(model.sym_weight(circuit.plus_state(12), cont) - 1) < 1e-10)
    check("plus sublattice symmetric", abs(model.sym_weight(circuit.plus_state(12), sub) - 1) < 1e-10)


def test_relabel_and_grads():
    sys = model.system(6)
    B = sys["B"]
    trips, _ = model.packing("sublattice", 6)
    idx = model.precompute(trips, 6)
    cont, _ = model.packing("contiguous", 6)
    idx_c = model.precompute(cont, 6)
    rng = np.random.default_rng(3)
    x = rng.normal(scale=0.3, size=circuit.n_params("A", 2))
    psi_d, e_d, _, _ = circuit.simulate("A", x, 2, sys["hc"], sys["fk"], np.zeros_like(sys["hc"]), B, 6, idx, False)
    # Relabel into contiguous slots and permute H. Do not compare the permuted state to the original GS.
    src, new_of_old, inv = model.relabel_maps(trips, 6)
    J_p = model.permute_J(sys["J"], src)
    z = model.z_table(6)
    hc_p = model.pair_diag(J_p, z)
    K_p = model.pair_K(J_p)
    scale = model.rms(model.upper(J_p)) / model.rms(model.upper(K_p))
    fk_p = model.pair_diag(K_p * scale, z)
    psi_c, e_c, _, _ = circuit.simulate("A", x, 2, hc_p, fk_p, np.zeros_like(hc_p), B, 6, idx_c, False)
    back = np.empty_like(psi_c)
    back[inv] = psi_c  # inv[new] = old, so back[old] = psi_c[new]
    # new_of_old[old] = new. back[old] should equal psi_c[new_of_old[old]]
    back = np.empty_like(psi_d)
    back[np.arange(64)] = psi_c[new_of_old]
    check("relabel energy", abs(e_d - e_c) < 1e-8, f"{e_d} {e_c}")
    check("relabel state", np.max(np.abs(psi_d - back)) < 1e-8, str(np.max(np.abs(psi_d - back))))
    # Original H must be the one in the direct energy.
    e_check, _ = circuit.energy_var(psi_d, sys["hc"], B, 6)
    check("direct energy uses logical H", abs(e_check - e_d) < 1e-8)
    # Gradients
    for kind, packing_name, L in (("A", "sublattice", 1), ("k0", "sublattice", 1), ("M", "contiguous", 1), ("qaoa", "contiguous", 1), ("sameG", "sublattice", 1)):
        trips_k, _ = model.packing(packing_name, 6)
        indices = model.precompute(trips_k, 6)
        fm, _, _ = model.block_mean_diag(sys["J"], trips_k, sys["z"])
        xx = rng.normal(scale=0.25, size=circuit.n_params(kind, L))
        _, _, g, _ = circuit.simulate(kind, xx, L, sys["hc"], sys["fk"], fm, B, 6, indices, True)
        gn = circuit.numerical_grad(kind, xx, L, sys["hc"], sys["fk"], fm, B, 6, indices)
        err = np.max(np.abs(g - gn))
        check(f"grad {kind}", err < 2e-5, str(err))
    # Parity preserved; reflection preserved on sublattice.
    _, _, _, _ = None, None, None, None
    psi, _, _, _ = circuit.simulate("A", x, 2, sys["hc"], sys["fk"], np.zeros_like(sys["hc"]), B, 6, idx, False)
    check("parity", abs(circuit.parity_expectation(psi) - 1) < 1e-8)
    ridx = circuit.reflection_indices(6)
    check("sublattice reflection", abs(circuit.reflection_expectation(psi, ridx) - 1) < 1e-8, str(circuit.reflection_expectation(psi, ridx)))
    # A random packing need not preserve reflection.
    rand, _ = model.packing("random0", 12)
    closed = [model.reflection_closed(model.packing(f"random{i}", 12)[0], 12) for i in range(6)]
    print("info random reflection-closed", closed)


if __name__ == "__main__":
    test_manifest_packings_and_neel()
    test_relabel_and_grads()
    print("ALL TESTS PASSED")
