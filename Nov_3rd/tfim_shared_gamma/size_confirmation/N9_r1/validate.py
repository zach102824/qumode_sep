"""N=9 checks before any optimizer call. Writes the target manifest."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import harness as h

ROOT = Path(__file__).resolve().parent
B = float("-0.3444")


def check(name, cond, detail=""):
    if not cond:
        raise SystemExit(f"FAIL {name} {detail}")
    print("ok", name)


def fd(fun, x, eps=1e-6):
    g = np.zeros_like(x)
    for i in range(x.size):
        e = np.zeros_like(x)
        e[i] = eps
        g[i] = (fun(x + e) - fun(x - e)) / (2 * eps)
    return g


def physical_state(target, gamma, thk, thm, shells):
    """Vacuum, Hadamards around the diagonal, popcount shell, final Hadamards, map back."""
    n = target.n
    dim = 1 << n
    phys_to_log = np.empty(n, dtype=np.int64)
    for a, cav in enumerate(target.packing):
        for r, site in enumerate(cav):
            phys_to_log[3 * a + r] = site
    logical_of = np.zeros(dim, dtype=np.int64)
    for p, site in enumerate(phys_to_log):
        logical_of |= ((np.arange(dim) >> p) & 1).astype(np.int64) << int(site)
    g_log = target.g_diag(thk, thm)
    phase = np.exp(-1j * float(gamma) * g_log[logical_of])
    psi = np.zeros(dim, dtype=np.complex128)
    psi[0] = 1.0
    had = np.array([[1, 1], [1, -1]], dtype=np.complex128) / np.sqrt(2)
    singles = [h.cavity_index(n, [q]) for q in range(n)]

    def apply_h(vec):
        out = np.array(vec, copy=True)
        for index in singles:
            out[index] = out[index] @ had.T
        return out

    for row in shells:
        psi = apply_h(psi)
        psi = psi * phase
        psi = apply_h(psi)
        phi = np.empty(4)
        for k in range(4):
            s2 = ((3 - 2 * k) ** 2 - 3) / 2.0
            phi[k] = row[0] * (3 - 2 * k) + row[1] * s2 + row[2] * ((-1) ** k)
        bits = np.zeros(dim, dtype=np.int64)
        idx = np.arange(dim)
        ph = np.zeros(dim)
        for a in range(len(target.packing)):
            pop = np.zeros(dim, dtype=np.int64)
            for r in range(3):
                pop += (idx >> (3 * a + r)) & 1
            ph += phi[pop]
        psi = psi * np.exp(-1j * ph)
    psi = apply_h(psi)
    logical = np.empty_like(psi)
    logical[logical_of] = psi
    return logical


def main():
    target = h.Target(9, B)
    check("Jnn", abs(target.J[0, 1] - 1.0) < 1e-15, str(target.J[0, 1]))
    check("packing", target.packing == [[0, 3, 6], [1, 4, 7], [2, 5, 8]])
    check("features kept", target.feature_rank >= 1)
    spec = h.ground_state(target)
    check("gs residual", spec["residual"] < 1e-10, str(spec["residual"]))
    excited = spec["evals"][1]
    # parity of the first excited eigenvector is measured, not assumed
    H = h.dense_h(target)
    # ground_state already diagonalized; recompute the vector from the returned evals is not stored.
    # A second eigh is a validation cost, not training.
    evals, evecs = np.linalg.eigh(H)
    e1 = evecs[:, 1]
    e1_parity = float(np.vdot(e1, e1[::-1]).real)
    same_parity_gap = None
    for k in range(1, evals.size):
        pk = float(np.vdot(evecs[:, k], evecs[:, k][::-1]).real)
        if pk > 0.5:
            same_parity_gap = float(evals[k] - evals[0])
            break
    check("gap measured", spec["full_gap"] > 0.0, str(spec["full_gap"]))
    print("info gs parity", spec["parity"], "e1 parity", e1_parity, "gap", spec["full_gap"])
    rng = np.random.default_rng(6)
    x = rng.normal(scale=0.2, size=9)
    _, _, g, _ = h.logical_grad(target, x)
    gn = fd(lambda z: h.logical_grad(target, z)[1], x)
    check("candidate grad", np.max(np.abs(g - gn)) < 2e-5, str(np.max(np.abs(g - gn))))
    xq = rng.normal(scale=0.3, size=4)
    _, _, gq, _ = h.qaoa_grad(target, xq)
    gqn = fd(lambda z: h.qaoa_grad(target, z)[1], xq)
    check("qaoa grad", np.max(np.abs(gq - gqn)) < 2e-5, str(np.max(np.abs(gq - gqn))))
    psi_s = h.qaoa_state(target, xq[0::2], xq[1::2])
    psi_d = h.dense_qaoa_state(target, xq[0::2], xq[1::2])
    check("dense qaoa", abs(np.vdot(psi_s, psi_d)) > 1 - 1e-10, str(abs(np.vdot(psi_s, psi_d))))
    shells = x[3:].reshape(2, 3)
    psi_l = h.logical_candidate(target, x[0], x[1], x[2], shells)
    psi_p = physical_state(target, x[0], x[1], x[2], shells)
    check("packed map", abs(np.vdot(psi_l, psi_p)) > 1 - 1e-10, str(abs(np.vdot(psi_l, psi_p))))
    H = h.dense_h(target)
    E_dense = float(np.vdot(psi_l, H @ psi_l).real)
    E_struct = h.logical_grad(target, x)[1]
    check("dense energy", abs(E_dense - E_struct) < 1e-10, f"{E_dense} {E_struct}")
    manifest = {
        "family": "fixed-B power-law extension, not the paper ion fit and not B=-0.1708",
        "formula": "H = sum_{i<j} |i-j|^{-1.1} Zi Zj + B sum_i Xi",
        "B": "-0.3444",
        "alpha": 1.1,
        "n": 9,
        "boundary": "open",
        "Jnn": 1.0,
        "kac": False,
        "Jij": [[i, j, format(float(target.J[i, j]), ".17g")] for i in range(9) for j in range(i + 1, 9)],
        "historical_N6_B_not_this_target": "-0.1708",
    }
    raw = json.dumps(manifest, indent=2).encode()
    (ROOT / "target_manifest.json").write_bytes(raw)
    packing = {"n": 6, "rule": "[a, a+n/3, a+2n/3]", "cavities": target.packing, "low_bit": "first"}
    praw = json.dumps(packing, indent=2).encode()
    (ROOT / "packing.json").write_bytes(praw)
    code = (ROOT.parents[1] / "size_confirmation" / "harness.py").read_bytes()
    report = {
        "manifest_sha256": hashlib.sha256(raw).hexdigest(),
        "packing_sha256": hashlib.sha256(praw).hexdigest(),
        "harness_sha256": hashlib.sha256(code).hexdigest(),
        "J01": float(target.J[0, 1]),
        "rK": target.rK,
        "rM": target.rM,
        "feature_rank_of_J_K_M": target.feature_rank,
        "pair_slots": 36,
        "first_excited_gap_is_measured": True,
        "E0": spec["E0"],
        "Emax": spec["Emax"],
        "full_gap": spec["full_gap"],
        "gs_residual": spec["residual"],
        "gs_parity": spec["parity"],
        "first_excited_parity": e1_parity,
        "same_parity_gap": same_parity_gap,
        "degeneracy_assumed": False,
        "features_not_dropped": True,
        "P_formula": "3L+3",
    }
    (ROOT / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
