"""IMAX conformance tests (must all pass before any IMAX fit). Output: results/IMAX/TESTS.json."""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "LE"))
sys.path.insert(0, str(HERE))
import core as C  # noqa: E402
import bstar as BS  # noqa: E402

OUT = HERE.parents[1] / "results" / "IMAX"
R = []


def check(name, val, tol):
    ok = bool(val <= tol)
    R.append({"test": name, "value": float(val), "tol": tol, "pass": ok})
    print(f"{name:44s} {val:.3e}  tol {tol:g}  {'PASS' if ok else 'FAIL'}")


def coherent(beta, n):
    k = np.arange(n)
    lf = np.array([math.lgamma(i + 1) for i in k])
    amp = np.exp(-abs(beta) ** 2 / 2 - 0.5 * lf) * np.power(complex(beta), k)
    return amp


def main():
    t0 = time.time()
    nf = 48
    for axis, ph in (("real", 1.0), ("imag", 1j)):
        D = C.Disp(nf, axis=axis)
        err = max(np.max(np.abs(D(a)[:, 0] - coherent(ph * a, nf))) for a in (0.05, 0.3, -0.3, 1.0))
        check(f"T17_coherent_state_analytic_{axis}", err, 1e-12)
    Dr, Di = C.Disp(nf, axis="real"), C.Disp(nf, axis="imag")
    check("T18_dagger_imag_D(ia)^H=D(-ia)", max(np.max(np.abs(Di(a).conj().T - Di(-a))) for a in (0.1, 0.3)), 1e-13)
    check("T18_dagger_real_D(a)^T=D(-a)", max(np.max(np.abs(Dr(a).T - Dr(-a))) for a in (0.1, 0.3)), 1e-13)
    Rn = np.diag(1j ** np.arange(nf))
    check("T19_axis_relation_D(ia)=R D(a) R^H",
          max(np.max(np.abs(Di(a) - Rn @ Dr(a) @ Rn.conj().T)) for a in (0.1, 0.3, 1.0)), 1e-12)
    blk = slice(0, 8)
    check("T20_composition_imag_low_block",
          np.max(np.abs((Di(0.2) @ Di(0.1))[blk, blk] - Di(0.3)[blk, blk])), 1e-12)
    for nfx in (16, 24, 32):
        D = C.Disp(nfx, axis="imag")(C.ALPHA_MAX)
        check(f"T21_trace_preservation_code_inputs_nf{nfx}",
              float(np.max(np.abs(1 - np.linalg.norm(D[:, :8], axis=0)))), 1e-10 if nfx >= 24 else 1e-6)
    D8r, D8i = C.Disp(8, strict=True, axis="real"), C.Disp(8, strict=True, axis="imag")
    R8 = np.diag(1j ** np.arange(8))
    check("T22_strict_D8_axis_relation", np.max(np.abs(D8i(0.3) - R8 @ D8r(0.3) @ R8.conj().T)), 1e-12)
    check("T22_strict_D8_unitary", np.max(np.abs(D8i(0.3).conj().T @ D8i(0.3) - np.eye(8))), 1e-12)

    # independent full-circuit path: real D conjugated by R (curvature_escape.DispImag)
    sys.path.insert(0, str(HERE.parent / "LE"))
    from curvature_escape import DispImag  # noqa: E402
    J, meta, gs = BS.build(6, 10000)
    b = meta["Bstar"]
    rng = np.random.default_rng(1)
    err_e = err_f = 0.0
    for strict in (True, False):
        for L in (1, 2, 3):
            a = C.Nov9(J, b, L, nf=16, strict=strict, axis="imag")
            ref = C.Nov9(J, b, L, nf=16, strict=strict, axis="real")
            ref.disp = DispImag(ref.nf, strict=strict)
            for _ in range(5):
                x = rng.normal(0, 0.5, a.nparam())
                err_e = max(err_e, abs(a.energy(x) - ref.energy(x)))
                err_f = max(err_f, abs(a.diagnostics(x, gs)["F"] - ref.diagnostics(x, gs)["F"]))
    check("T23_imag_circuit_vs_independent_conjugated_path_E", err_e, 1e-11)
    check("T23_imag_circuit_vs_independent_conjugated_path_F", err_f, 1e-11)

    # conjugation symmetry: real axis exact, imag axis broken (mechanism documentation)
    sr = C.Nov9(J, b, 2, nf=16, axis="real")
    si = C.Nov9(J, b, 2, nf=16, axis="imag")
    x = rng.normal(0, 0.5, sr.nparam())
    x[3 + 2:] = 0
    xm = x.copy()
    xm[0] *= -1
    check("T24_real_axis_E(g)=E(-g)_at_s0", abs(sr.energy(x) - sr.energy(xm)), 1e-12)
    brk = abs(si.energy(x) - si.energy(xm))
    R.append({"test": "T24_imag_axis_breaks_symmetry_at_s0 (info)", "value": float(brk), "tol": None, "pass": bool(brk > 1e-6)})
    print(f"T24_imag_axis_breaks_symmetry_at_s0 (info)   {brk:.3e}  {'PASS' if brk > 1e-6 else 'FAIL'}")

    # H_ext bound with the imaginary axis (no exploitation of leakage), random parameters incl. large alpha
    worst = np.inf
    for L in (1, 2, 3):
        s = C.Nov9(J, b, L, nf=16, axis="imag")
        for _ in range(100):
            x = rng.normal(0, 1.5, s.nparam())
            worst = min(worst, s.energy(x) - meta["E0"])
    check("T25_imag_Hext_energy_ge_E0 (neg margin)", max(0.0, -worst), 1e-10)
    q = C.QAOA(J, b, 2)
    wq = min(q.energy(rng.normal(0, 1, 4)) - meta["E0"] for _ in range(100))
    check("T25_qaoa_energy_ge_E0 (neg margin)", max(0.0, -wq), 1e-10)
    check("T26_Bstar_instance_valid_N6_s10000", 0.0 if meta["valid"] else 1.0, 0.0)
    allpass = all(r["pass"] for r in R)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "TESTS.json").write_text(json.dumps({"all_pass": allpass, "tests": R, "seconds": time.time() - t0}, indent=1))
    print("ALL PASS" if allpass else "FAILURES", f"{time.time() - t0:.1f}s")
    return allpass


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
