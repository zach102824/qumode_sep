"""N3COND conformance. Must pass before the scale block and before any fit."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "LE"))
sys.path.insert(0, str(HERE.parent / "IMAX"))
import core as C  # noqa: E402
import bstar as BS  # noqa: E402
import nov3 as N  # noqa: E402
import spsa as S  # noqa: E402

OUT = HERE.parents[1] / "results" / "N3COND"
R = []


def check(name, val, tol):
    ok = bool(val <= tol)
    R.append({"test": name, "value": float(val), "tol": tol, "pass": ok})
    print(f"{name:52s} {val:.3e}  tol {tol:g}  {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    prereg = json.loads((HERE / "PREREG_N3COND_B1.json").read_text())
    assert prereg["seeds"]["dev_scale"] == [10010, 10011, 10012, 10013]
    assert prereg["seeds"]["holdout"] == [10014, 10015, 10016, 10017]
    assert set(prereg["seeds"]["dev_scale"]).isdisjoint(prereg["seeds"]["holdout"])
    for bad in list(range(10000, 10010)) + list(range(11000, 11032)) + [20000, 20999]:
        assert bad not in prereg["seeds"]["dev_scale"] + prereg["seeds"]["holdout"]

    J = C.make_J(10010, 6)
    check("T1_J_sha_stable", 0.0 if C.J_sha(J) == C.J_sha(C.make_J(10010, 6)) else 1.0, 0.0)
    b = 1.0
    nov = N.Nov3(J, b, 2)
    q = N.QAOA(J, b, 2)
    check("T2_nparam_shared_gamma", 0.0 if nov.nparam() == 3 + 3 * 2 else 1.0, 0.0)
    check("T2_W_on_zero_is_plus", float(np.max(np.abs(N.fwht(nov.zero) - nov.plus))), 1e-12)
    check("T2_W_squared", float(np.max(np.abs(N.fwht(N.fwht(nov.plus)) - nov.plus))), 1e-12)

    rng = np.random.default_rng(1)
    err_st = 0.0
    for L in (1, 2, 3):
        a = N.Nov3(J, b, L)
        for _ in range(6):
            x = rng.normal(0, 0.4, a.nparam())
            err_st = max(err_st, float(np.max(np.abs(a.state(x) - a.state_plus_rewrite(x)))))
    check("T3_zero_circuit_equals_plus_rewrite", err_st, 1e-12)

    err_e = err_s = 0.0
    for L in (1, 2, 3):
        a = N.Nov3(J, b, L)
        qq = N.QAOA(J, b, L)
        for _ in range(6):
            gamma = float(rng.normal(0, 0.3))
            s1 = rng.normal(0, 0.4, size=L)
            x = np.concatenate([[gamma], [0.0, 0.0], np.column_stack([s1, np.zeros(L), np.zeros(L)]).reshape(-1)])
            xq = np.concatenate([np.full(L, gamma), -0.5 * s1])
            ps, pq = a.state(x), qq.state(xq)
            err_e = max(err_e, abs(a.energy(x) - qq.energy(xq)))
            # shell s1*p = s1*sum(1-Z)/2 contributes a global phase exp(-i s1 N/2)
            phase = np.exp(-1j * np.sum(s1) * a.N / 2.0)
            err_s = max(err_s, float(np.max(np.abs(ps - pq * phase))))
            err_s = max(err_s, abs(abs(np.vdot(pq, ps)) - 1.0))
    check("T4_s2s3_0_matches_QAOA_energy", err_e, 1e-10)
    check("T4_s2s3_0_matches_QAOA_state_up_to_global_phase", err_s, 1e-10)

    # one cavity, explicit exp(-i beta sum X)
    J3 = C.make_J(1, 3)
    a3 = N.Nov3(J3, 0.7, 1)
    beta = 0.37
    x = np.array([0.0, 0.0, 0.0, -2 * beta, 0.0, 0.0])
    psi = a3.state(x)
    q3 = N.QAOA(J3, 0.7, 1)
    phase = np.exp(-1j * (-2 * beta) * 3 / 2.0)
    check("T4b_one_cavity_beta", float(np.max(np.abs(psi - q3.state([0.0, beta]) * phase))), 1e-12)

    gs = C.exact_gs(J, b)["gs"]
    e0 = C.exact_gs(J, b)["E0"]
    worst = min(nov.energy(rng.normal(0, 0.5, nov.nparam())) - e0 for _ in range(20))
    check("T5_energy_ge_E0", max(0.0, -worst), 1e-8)
    x = rng.normal(0, 0.3, nov.nparam())
    d = nov.diagnostics(x, gs)
    check("T5_norm", abs(d["norm"] - 1.0), 1e-12)
    check("T5_even_parity", abs(d["parity"] - 1.0), 1e-10)
    # |+> energy is -b N
    check("T5_plus_energy", abs(nov.energy_of_state(nov.plus) - (-b * nov.N)), 1e-12)

    # shared gamma nesting: s=0, L layers of gamma equals 1 layer of L*gamma
    a1 = N.Nov3(J, b, 1)
    a3 = N.Nov3(J, b, 3)
    g = 0.21
    x1 = np.array([3 * g, 0.11, -0.07, 0.0, 0.0, 0.0])
    x3 = np.concatenate([[g], [0.11, -0.07], np.zeros(9)])
    check("T6_shared_gamma_nest_s0", float(np.max(np.abs(a1.state(x1) - a3.state(x3)))), 1e-12)

    # s2, s3 are not redundant
    x_lin = np.array([0.2, 0.0, 0.0, 0.3, 0.0, 0.0])
    x_s2 = np.array([0.2, 0.0, 0.0, 0.3, 0.4, 0.0])
    x_s3 = np.array([0.2, 0.0, 0.0, 0.3, 0.0, 0.5])
    a = N.Nov3(J, b, 1)
    gap = float(np.max(np.abs(a.state(x_lin) - a.state(x_s2))))
    gap3 = float(np.max(np.abs(a.state(x_lin) - a.state(x_s3))))
    check("T7_s2_moves_state", 0.0 if gap > 1e-6 else 1.0, 0.0)
    check("T7_s3_moves_state", 0.0 if gap3 > 1e-6 else 1.0, 0.0)

    # query counter, both conditioners, fake energy
    def f(x):
        return float(np.dot(x, x))

    for cond, a_over in (("C0", None), ("C1", 0.2)):
        starts = [np.zeros(5), np.ones(5) * 0.1, -np.ones(5) * 0.1, np.full(5, 0.05)]
        sig = np.full(5, 0.1)
        rec = S.spsa_run(f, starts, sig, lambda s: np.random.default_rng(s), 128, a_override=a_over)
        check(f"T8_queries_128_{cond}", abs(rec["queries"] - 128), 0.0)
        check(f"T8_K_{cond}", abs(rec["K_per_start"] - 11), 0.0)
        if cond == "C1":
            check("T8_gain_frozen", 0.0 if all(m["gain_frozen"] and m["a_used"] == 0.2 for m in rec["start_meta"]) else 1.0, 0.0)
        else:
            check("T8_gain_live", 0.0 if all(not m["gain_frozen"] for m in rec["start_meta"]) else 1.0, 0.0)

    # lambda formula
    rows = np.array([[1.0, 2.0, 4.0], [1.0, 2.0, 4.0]])
    lam, a0, g, gref = S.freeze_lambda(rows)
    check("T9_lambda_equalizes", float(np.max(np.abs(lam * g - gref))), 1e-12)
    rows2 = np.array([[1e-15, 1.0], [1e-15, 1.0]])
    lam2, _, g2, gref2 = S.freeze_lambda(rows2)
    check("T9_lambda_clip", abs(lam2[0] - 4.0), 0.0)
    # simultaneous |component| is constant; axial probes are not
    def quad(x):
        return float(x[0] ** 2 + 100.0 * x[1] ** 2)

    row = S.probes_at(quad, np.array([1.0, 1.0]), np.ones(2), np.random.default_rng(0), 1)[0]
    check("T13_simultaneous_components_equal", abs(row[0] - row[1]), 0.0)
    gax = S.axial_probe(quad, np.array([1.0, 1.0]), np.ones(2))
    lam_ax, _, _, _ = S.freeze_lambda(gax.reshape(1, -1))
    check("T13_axial_lambda_not_flat", 0.0 if abs(lam_ax[0] - lam_ax[1]) > 0.5 else 1.0, 0.0)
    check("T13_axial_g_ratio", abs(gax[1] / gax[0] - 100.0), 1e-9)

    # B* acceptance machinery on N=6, seed 10010 (not an optimization)
    Jb, meta, gsb = BS.build(6, 10010)
    okb = meta["valid"] and meta["E1_parity"] == "odd" and meta["gap_even"] > 0 and meta["gs_resid"] < 1e-9
    check("T10_Bstar_N6_s10010_accept", 0.0 if okb else 1.0, 0.0)
    check("T10_Bstar_A_cut", 0.0 if meta["bipartition_A"] == [0, 1, 2] else 1.0, 0.0)
    check("T10_resid", meta["gs_resid"], 1e-9)
    # F of the GS against itself through the circuit at a point is post-hoc only; just the overlap identity
    sim = N.Nov3(Jb, meta["Bstar"], 1)
    check("T10_gs_norm", abs(float(np.dot(gsb, gsb)) - 1.0), 1e-12)

    # TQA shared gamma value
    x0 = N.tqa_nov3(3)
    check("T11_tqa_shared_gamma", abs(x0[0] - 0.375), 0.0)
    check("T11_tqa_theta0", abs(x0[1]) + abs(x0[2]), 0.0)
    t = (np.arange(3) + 0.5) / 3
    beta = -(1 - t) * 0.75
    s1 = x0[3::3]
    check("T11_tqa_s1", float(np.max(np.abs(s1 - (-2 * beta)))), 1e-15)

    # cap arithmetic for every preregistered (q, L)
    bad = 0
    for q in prereg["batch_B1"]["q"]:
        for L in prereg["batch_B1"]["L"]:
            try:
                S.split_cap(q * L)
            except ValueError:
                bad += 1
    check("T12_caps_divide", float(bad), 0.0)

    allpass = all(r["pass"] for r in R)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "TESTS.json").write_text(json.dumps({"all_pass": allpass, "n": len(R), "tests": R}, indent=1))
    print("ALL PASS" if allpass else "FAILURES", f"n={len(R)}")
    return allpass


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
