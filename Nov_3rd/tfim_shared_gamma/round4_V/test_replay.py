"""Replay source L5/L6 states and check packing identities."""
from __future__ import annotations

import hashlib
import json

import numpy as np

from sim import (
    DATA,
    System,
    load_manifest,
    load_packings,
    logical_state,
    overlap,
    parse_J,
    physical_state,
    rfull_energy_grad,
)


def check(name, cond, detail=""):
    if not cond:
        raise AssertionError(f"FAIL {name}: {detail}")
    print(f"PASS {name} {detail}")


def main():
    manifest, _ = load_manifest()
    packings, _ = load_packings()
    J, B = parse_J(manifest, 12)
    packing = packings["sublattice"]["12"]
    sys = System(J, B, packing)
    # M projection: intra coeffs 0, each block constant and equal to the mean of J.
    for a, ca in enumerate(packing):
        for i in ca:
            for j in ca:
                if i < j:
                    check(f"intra M {i}{j}", abs(sys.M[i, j]) < 1e-15)
    for a, ca in enumerate(packing):
        for b, cb in enumerate(packing):
            if a >= b:
                continue
            block = J[np.ix_(ca, cb)]
            vals = sys.M[np.ix_(ca, cb)]
            check(
                f"block mean {a}{b}",
                np.allclose(vals, block.mean()) and np.allclose(vals, sys.jbar[a, b]),
                f"jbar={sys.jbar[a,b]:.6f}",
            )
    # Gradients.
    rng = np.random.default_rng(0)
    x = rng.normal(scale=0.15, size=3 + 6)
    E, g = rfull_energy_grad(sys, 2, x)

    def fun(z):
        return rfull_energy_grad(sys, 2, z)[0]

    fd = np.zeros_like(x)
    eps = 1e-7
    for i in range(x.size):
        step = np.zeros_like(x)
        step[i] = eps
        fd[i] = (fun(x + step) - fun(x - step)) / (2 * eps)
    err = float(np.max(np.abs(g - fd)))
    check("grad", err < 5e-6, f"max {err:.3e}")

    # Physical vs logical, both within-cavity orders.
    psi_l = logical_state(sys, 2, x)
    psi_p = physical_state(sys, 2, x)
    check("physical-logical", overlap(psi_l, psi_p) > 1 - 1e-10, f"{overlap(psi_l, psi_p):.12f}")
    # Reversed cavity bit order must not change the logical energy.
    rev = [list(reversed(c)) for c in packing]
    sys_r = System(J, B, rev)
    psi_r = logical_state(sys_r, 2, x)
    check("cavity order energy", abs(sys.energy(psi_l) - sys_r.energy(psi_r)) < 1e-10)
    # The reversed local-bit map is a different embedding. Record the overlap; do not assume it.
    print(f"reversed-bit overlap {overlap(psi_l, psi_r):.6f}")
    # pi/2 shell shifts and full angle reversal. Energies of the reversed vector match; the state is the conjugate.
    base_E = sys.energy(psi_l)
    for name, idx in (("c1", 3), ("c2", 4), ("c3", 5)):
        y = x.copy()
        y[idx] += 0.5 * np.pi
        py = logical_state(sys, 2, y)
        print(f"{name} +pi/2 overlap {overlap(psi_l, py):.6e} dE {abs(sys.energy(py)-base_E):.3e}")
    # gamma and the shell flip; theta_K and theta_M stay inside G.
    y = x.copy()
    y[0] *= -1.0
    y[3:] *= -1.0
    psi_tr = logical_state(sys, 2, y)
    check("time-reversal energy", abs(sys.energy(psi_tr) - base_E) < 1e-9)
    check("time-reversal conjugate", overlap(psi_tr, np.conj(psi_l)) > 1 - 1e-8, f"{overlap(psi_tr, np.conj(psi_l)):.12f}")
    print(f"time-reversal overlap with the original ray {overlap(psi_tr, psi_l):.6f}")

    # Joint relabel: swap sites 0 and 1.
    pi = np.arange(12)
    pi[0], pi[1] = 1, 0
    J2 = np.zeros_like(J)
    for i in range(12):
        for j in range(12):
            J2[pi[i], pi[j]] = J[i, j]
    pack2 = [[int(pi[s]) for s in c] for c in packing]
    sys2 = System(J2, B, pack2)
    # Permute a logical basis state: bit i moves to bit pi[i].
    dim = 1 << 12
    image = np.zeros(dim, dtype=np.int64)
    for i, dest in enumerate(pi):
        image |= ((np.arange(dim) >> i) & 1) << int(dest)
    psi2 = np.zeros_like(psi_l)
    psi2[image] = psi_l
    check("relabel energy", abs(sys.energy(psi_l) - sys2.energy(psi2)) < 1e-9, f"{sys.energy(psi_l)} {sys2.energy(psi2)}")

    report = {}
    for L, name in ((5, "N12_L5_A_sub_M"), (6, "N12_L6_A_sub_M")):
        cell = json.loads((DATA / f"{name}.json").read_text())
        src = np.load(DATA / f"{name}.npz")["psi"]
        x = np.asarray(cell["x"], dtype=np.float64)
        psi = logical_state(sys, L, x)
        E = sys.energy(psi)
        # GS not yet; fidelity after spectrum. Store psi.
        ov = overlap(psi, src)
        ov_c = overlap(psi, np.conj(src))
        # bit-reverse index
        rev_idx = sys._basis_map(np.arange(12)[::-1])
        src_br = np.zeros_like(src)
        src_br[rev_idx] = src
        ov_br = overlap(psi, src_br)
        digest = hashlib.sha256(np.ascontiguousarray(psi).tobytes()).hexdigest()
        src_digest = hashlib.sha256(np.ascontiguousarray(src).tobytes()).hexdigest()
        check(f"L{L} source artifact hash", src_digest == cell["state_hash"], src_digest)
        check(f"L{L} energy", abs(E - cell["E"]) < 1e-10, f"{E-cell['E']:.3e}")
        check(f"L{L} overlap", ov >= 1 - 1e-12, f"{ov:.16f}")
        aligned = psi * np.exp(-1j * np.angle(np.vdot(src, psi)))
        residual = float(np.max(np.abs(aligned - src)))
        check(f"L{L} residual", residual < 1e-12, f"{residual:.3e}")
        print(f"L{L} replay hash {digest} source hash {src_digest} residual {residual:.3e}")
        # Opposite within-cavity bit significance, recorded rather than assumed.
        rev_map = sys._basis_map(np.array([sys.packing[p // 3][2 - (p % 3)] for p in range(12)]))
        phys = physical_state(sys, L, cell["x"])
        physical_amps = phys[sys.logical_of]
        wrong = np.zeros_like(psi)
        wrong[rev_map] = physical_amps
        print(f"L{L} opposite-bit-map overlap {overlap(wrong, src):.6f}")
        print(f"L{L} E {E:.10f} src {cell['E']:.10f} dE {E-cell['E']:.3e} ov {ov:.16f} ovconj {ov_c:.6f} ovbr {ov_br:.6f}")
        report[L] = {"E": E, "srcE": cell["E"], "ov": ov, "ovconj": ov_c, "ovbr": ov_br, "psi": psi, "src": src, "cell": cell, "state_hash": digest}
    return sys, report


if __name__ == "__main__":
    main()
