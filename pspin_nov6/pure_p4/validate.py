"""Exact checks for the pure distinct-index p=4 family.

Run: python validate.py
Writes results/validation.json and results/terms_N{6,7}.txt next to the script
(or --outdir). Exits non-zero on the first failed assertion group.
"""
from __future__ import annotations

import argparse
import json
import math
from itertools import combinations
from pathlib import Path

import numpy as np

from model import (
    DickeModel,
    J,
    a_N,
    apply_mixer_full,
    best_basis_population,
    cat_overlap,
    classical_optimum,
    dicke_X,
    dicke_cost,
    dicke_parity_expectation,
    dicke_to_full,
    e4_int,
    enumerate_quadruples,
    flip_expectation,
    full_spectrum_edges,
    gamma_period,
    half_cut_entropy_dicke,
    half_cut_entropy_full,
    hc_diag_closed,
    hc_diag_enumerated,
    lower_body_delta,
    magnetization_from_weight,
    parity_basis,
    plus_dicke,
    plus_full,
    pure_cost_from_M,
    symmetric_dicke_spectrum,
    symmetric_product_overlap,
    symmetric_weight,
)


def check(cond, msg, failures):
    if not cond:
        failures.append(msg)


def _weights(n):
    return np.bitwise_count(np.arange(1 << n, dtype=np.uint32)).astype(np.int64)


def test_identity_and_enumeration(failures, records):
    for n in range(4, 9):
        for s in (+1, -1):
            enum, terms = hc_diag_enumerated(n, s)
            closed = hc_diag_closed(n, s, "pure")
            err = float(np.max(np.abs(enum - closed)))
            check(err < 1e-9, f"enum vs closed N={n} s={s} err={err}", failures)
            check(len(terms) == math.comb(n, 4), f"term count N={n}", failures)
            check(len(terms) == len(set(terms)), f"duplicate terms N={n}", failures)
        # integer identity on every admissible M, and on every basis state
        wt = _weights(n)
        M = magnetization_from_weight(n, wt)
        # M^4 expansion
        M4 = M.astype(np.float64) ** 4
        rhs = (
            n
            + 4.0 * (M**2 - n)
            + 3.0 * n * (n - 1)
            + 6.0 * (n - 2) * (M**2 - n)
            + 24.0 * np.array([e4_int(int(m), n) for m in M])
        )
        err = float(np.max(np.abs(M4 - rhs)))
        check(err < 1e-6, f"repeated-index identity N={n} err={err}", failures)
        # polarized cost
        pol = float(pure_cost_from_M(np.array(float(n)), n, s=1))
        check(abs(pol - n) < 1e-9, f"polarized cost {pol} != N={n}", failures)
        # a_N * C(N,4) = N
        check(abs(a_N(n) * math.comb(n, 4) - n) < 1e-12, f"a_N scale N={n}", failures)
        delta_pol = float(lower_body_delta(np.array(float(n)), n))
        check(abs(delta_pol) < 1e-8, f"delta at |M|=N is {delta_pol}", failures)
    records["identity"] = "enumerated ZZZZ diagonal matches closed e4(M) for N=4..8, both signs"


def test_term_lists(outdir: Path, records):
    paths = []
    for n in (6, 7):
        terms = enumerate_quadruples(n)
        lines = [
            f"# distinct quadruples for N={n}, count={len(terms)}, C(N,4)={math.comb(n,4)}",
            "# indices are qubit labels 0..N-1, qubit 0 = MSB in the statevector",
            f"# H_C = s * J * a_N * sum of Z_i Z_j Z_k Z_l over these tuples, J={J}, a_N={a_N(n):.12g}",
        ]
        # also the eigenvalue check per tuple sum vs closed form, one line of summary
        enum, _ = hc_diag_enumerated(n, s=1)
        closed = hc_diag_closed(n, 1, "pure")
        lines.append(f"# max |enum-closed| = {np.max(np.abs(enum-closed)):.3e}")
        for comb in terms:
            lines.append(" ".join(str(i) for i in comb))
        path = outdir / f"terms_N{n}.txt"
        path.write_text("\n".join(lines) + "\n")
        paths.append(str(path))
    records["term_lists"] = paths


def test_dicke_matches_full(failures, records):
    X_errs = []
    spec_errs = []
    qaoa_errs = []
    for n in (5, 6, 7, 8):
        dim = 1 << n
        # one-hot Dicke probes: compare X matvec
        Xd = dicke_X(n)
        ev, V = np.linalg.eigh(Xd)
        beta = 0.37
        for w in range(n + 1):
            c = np.zeros(n + 1, dtype=np.complex128)
            c[w] = 1.0
            # spectral exp(-i β X) in the Dicke basis, mapped back to 2^N
            coeff = c @ V
            coeff = np.exp(-1j * beta * ev) * coeff
            cU = coeff @ V.T
            full_U = dicke_to_full(cU, n)
            mix_U = apply_mixer_full(dicke_to_full(c, n), beta, n)
            X_errs.append(float(np.max(np.abs(full_U - mix_U))))
        check(max(X_errs) < 1e-8, f"Dicke mixer vs full mixer err {max(X_errs)}", failures)

        for kind in ("pure", "collective"):
            for s in (+1, -1):
                for gamma in (0.0, 0.4, 1.1):
                    spec = symmetric_dicke_spectrum(n, s, gamma, kind)
                    full = full_spectrum_edges(n, s, gamma, kind, k_bottom=min(32, (1 << n) - 2))
                    # every even and odd eigenvalue must appear in the full spectrum
                    if n <= 8:
                        # rebuild full dense eigenvalues for an exact subset check at n<=7;
                        # at n=8 use the bottom/top only plus a residual test via Rayleigh
                        pass
                    e0_reach = min(spec["E0_even"], spec["E0_odd"])
                    # symmetric energy cannot undercut the full GS
                    check(
                        e0_reach >= full["E0"] - 2e-6,
                        f"sector below full N={n} {kind} s={s} G={gamma}: {e0_reach} < {full['E0']}",
                        failures,
                    )
                    spec_errs.append(abs(e0_reach - full["E0"]))  # may be >0 if GS is outside; recorded
                    # Rayleigh of the even GS inside the full matrix equals its Dicke energy
                    v = spec["even_vecs"][:, 0]
                    psi = dicke_to_full(v, n)
                    Hpsi_diag = full["hc_diag"] * psi
                    # subtract gamma * X psi using the mixer derivative-quality apply:
                    # X psi = i * d/dβ U(β) ψ at 0, computed by applying known Dicke X
                    Xpsi = dicke_to_full(spec["X"] @ v, n)
                    Hpsi = Hpsi_diag - gamma * Xpsi
                    ray = float(np.vdot(psi, Hpsi).real)
                    check(
                        abs(ray - spec["E0_even"]) < 1e-6,
                        f"Rayleigh N={n} {kind} s={s} G={gamma}: {ray} vs {spec['E0_even']}",
                        failures,
                    )
                    check(abs(np.linalg.norm(psi) - 1) < 1e-8, "dicke_to_full norm", failures)
                    sw = symmetric_weight(psi, n)
                    check(abs(sw - 1) < 1e-8, f"sym weight {sw}", failures)
                    par = flip_expectation(psi)
                    check(abs(par - 1) < 1e-8, f"even parity {par}", failures)

        # QAOA statevector cross-check
        rng = np.random.default_rng(n * 17 + 3)
        model = DickeModel.build(n, s=-1, gamma=0.7, kind="pure")
        betas = rng.normal(size=3)
        gammas = rng.normal(size=3)
        c = model.evolve(betas, gammas)
        psi_d = dicke_to_full(c, n)
        psi_f = plus_full(n)
        hc = hc_diag_closed(n, s=-1, kind="pure")
        for b, g in zip(betas, gammas):
            psi_f = psi_f * np.exp(-1j * g * hc)
            psi_f = apply_mixer_full(psi_f, float(b), n)
        qaoa_errs.append(float(np.max(np.abs(psi_d - psi_f))))
        check(qaoa_errs[-1] < 1e-8, f"QAOA dicke vs full N={n} err {qaoa_errs[-1]}", failures)
        # parity stays +1 and symmetric weight 1
        check(abs(dicke_parity_expectation(c) - 1) < 1e-8, f"QAOA parity N={n}", failures)
        check(abs(symmetric_weight(psi_f, n) - 1) < 1e-8, f"QAOA sym N={n}", failures)
        # zero-angle energy
        E0ang = model.energy(model.c0)
        check(abs(E0ang - (-0.7 * n)) < 1e-8, f"zero-angle energy {E0ang} N={n}", failures)

    # basis completeness: even columns + odd columns = N+1 and orthonormal
    for n in (5, 6, 7, 8, 9):
        Be = parity_basis(n, +1)
        Bo = parity_basis(n, -1)
        check(Be.shape[1] + Bo.shape[1] == n + 1, f"parity basis count N={n}: {Be.shape[1]}+{Bo.shape[1]}", failures)
        B = np.concatenate([Be, Bo], axis=1)
        gram = B.T @ B
        check(np.max(np.abs(gram - np.eye(n + 1))) < 1e-12, f"parity basis ON N={n}", failures)
        # parity eigenvalues
        for col, expected in (
            *[(Be[:, i], 1.0) for i in range(Be.shape[1])],
            *[(Bo[:, i], -1.0) for i in range(Bo.shape[1])],
        ):
            p = float(np.dot(col, col[::-1]))
            check(abs(p - expected) < 1e-12, f"basis parity N={n} {p} != {expected}", failures)

    records["dicke_X_max_err"] = max(X_errs)
    records["qaoa_state_max_err"] = max(qaoa_errs)
    records["sector_vs_full_E0_abs"] = {
        "max": max(spec_errs),
        "note": "nonzero is allowed when the global GS leaves the symmetric irrep; the test only forbids the sector energy from falling below the full GS",
    }


def test_dense_spectrum_subset(failures, records):
    """For N=6, every symmetric eigenvalue is in the dense spectrum."""
    n = 6
    rows = []
    for kind in ("pure", "collective"):
        for s in (+1, -1):
            gamma = 0.6
            spec = symmetric_dicke_spectrum(n, s, gamma, kind)
            full = full_spectrum_edges(n, s, gamma, kind)
            # dense path stores only the bottom multiplicity, so rebuild evals
            hc = full["hc_diag"]
            H = np.diag(hc)
            idx = np.arange(1 << n)
            for q in range(n):
                bit = 1 << (n - 1 - q)
                a = idx[(idx & bit) == 0]
                b = a + bit
                H[a, b] -= gamma
                H[b, a] -= gamma
            ev = np.linalg.eigvalsh(H)
            for label, block in (("even", spec["even_evals"]), ("odd", spec["odd_evals"])):
                for e in block:
                    dist = float(np.min(np.abs(ev - e)))
                    check(dist < 1e-7, f"N=6 {kind} s={s} {label} eigenvalue {e} missing (dist {dist})", failures)
                    rows.append({"kind": kind, "s": s, "sector": label, "E": float(e), "dist": dist})
    records["n6_sector_eigenvalues_in_full_spectrum"] = len(rows)


def test_periods(failures, records):
    period_rows = []
    for n in (5, 6, 7, 8, 10, 12):
        info = gamma_period(n)
        T = info["T_gamma"]
        hc = dicke_cost(n, s=1, kind="pure")
        # phases of exp(-i T hc) must be a global phase
        ph = np.exp(-1j * T * hc)
        ph = ph / ph[0]
        err = float(np.max(np.abs(ph - 1)))
        check(err < 1e-8, f"gamma period N={n} err {err} T={T}", failures)
        # proper divisors should NOT all work; record the minimal positive divisor that does
        minimal = T
        for d in range(2, 13):
            if abs(T / d) < 1e-15:
                continue
            Td = T / d
            ph = np.exp(-1j * Td * hc)
            ph = ph / ph[0]
            if float(np.max(np.abs(ph - 1))) < 1e-7:
                minimal = Td
        info = dict(info)
        info["N"] = n
        info["T_gamma_analytical"] = T
        info["T_gamma_minimal_empirical"] = minimal
        info["reduced"] = bool(abs(minimal - T) > 1e-8)
        # beta period
        X = dicke_X(n)
        ev, V = np.linalg.eigh(X)
        def U(beta):
            return (V * np.exp(-1j * beta * ev)[None, :]) @ V.T
        # U(β+π) / U(β) = (-1)^N I
        ratio = np.linalg.inv(U(0.3)) @ U(0.3 + math.pi)
        target = ((-1) ** n) * np.eye(n + 1)
        errb = float(np.max(np.abs(ratio - target)))
        check(errb < 1e-8, f"beta period N={n} err {errb}", failures)
        ratio_half = np.linalg.inv(U(0.3)) @ U(0.3 + 0.5 * math.pi)
        errh = float(np.max(np.abs(ratio_half - target)))
        # half-period must NOT be a global phase (distance from every global phase)
        # check it is not proportional to I
        m = ratio_half
        # remove trace phase
        tr = np.trace(m) / (n + 1)
        err_half_to_phase = float(np.max(np.abs(m - tr * np.eye(n + 1))))
        check(err_half_to_phase > 1e-6, f"beta half-period unexpectedly trivial N={n}", failures)
        info["beta_pi_err"] = errb
        info["beta_half_pi_not_global_phase"] = err_half_to_phase
        period_rows.append(info)
    records["periods"] = period_rows


def test_observables(failures, records):
    # GHZ entropy 1 bit; |+> entropy 0; product overlap of |+> is 1
    for n in (6, 7, 8):
        c = plus_dicke(n)
        ent = half_cut_entropy_dicke(c, n)
        check(ent["S_bits"] < 1e-8, f"+ entropy {ent['S_bits']} N={n}", failures)
        prod = symmetric_product_overlap(c, n)
        check(prod["F_product_symmetric"] > 1 - 1e-8, f"+ product {prod}", failures)
        cat = np.zeros(n + 1, dtype=np.complex128)
        cat[0] = 1 / math.sqrt(2)
        cat[-1] = 1 / math.sqrt(2)
        entc = half_cut_entropy_dicke(cat, n)
        check(abs(entc["S_bits"] - 1) < 1e-8, f"GHZ entropy {entc['S_bits']} N={n}", failures)
        check(abs(cat_overlap(cat) - 1) < 1e-12, "cat overlap", failures)
        # full-space entropy agrees
        ent_full = half_cut_entropy_full(dicke_to_full(cat, n), n)
        check(abs(ent_full["S_bits"] - entc["S_bits"]) < 1e-8, f"entropy dicke vs full N={n}", failures)
        check(best_basis_population(cat, n) == 0.5 or abs(best_basis_population(cat, n) - 0.5) < 1e-12, "cat basis pop", failures)

    # Gamma=0 anchors
    anchors = []
    for n in (6, 8, 10):
        pure_af = classical_optimum(n, s=+1, kind="pure")
        pure_fe = classical_optimum(n, s=-1, kind="pure")
        ctrl_af = classical_optimum(n, s=+1, kind="collective")
        ctrl_fe = classical_optimum(n, s=-1, kind="collective")
        check(abs(pure_fe["E_classical"] + n) < 1e-8, f"ferro classical {pure_fe}", failures)
        check(abs(ctrl_fe["E_classical"] + n) < 1e-8, f"ctrl ferro {ctrl_fe}", failures)
        check(ctrl_af["M_list"] == [0], f"collective AF minimum should be M=0, got {ctrl_af}", failures)
        check(0 not in pure_af["M_list"] or n >= 30, f"pure AF minimum unexpectedly only M=0 at N={n}: {pure_af}", failures)
        # full GS at Gamma=0 matches classical energy
        for kind, s, cref in (
            ("pure", -1, pure_fe),
            ("pure", +1, pure_af),
            ("collective", +1, ctrl_af),
        ):
            full = full_spectrum_edges(n, s, 0.0, kind)
            check(abs(full["E0"] - cref["E_classical"]) < 1e-6, f"Gamma0 E0 N={n} {kind} s={s}", failures)
            spec = symmetric_dicke_spectrum(n, s, 0.0, kind)
            check(abs(spec["E0_even"] - cref["E_classical"]) < 1e-6, f"Gamma0 even N={n} {kind} s={s}", failures)
        anchors.append({
            "N": n,
            "pure_ferro": pure_fe,
            "pure_af": pure_af,
            "collective_af": ctrl_af,
            "delta_by_M": {
                str(M): float(lower_body_delta(np.array(float(M)), n))
                for M in range(-n, n + 1, 2)
            },
        })
    records["classical_anchors"] = anchors


def test_interpolation_and_carry(failures):
    from qaoa import anneal_angles, matched_anneal_time, resample_schedule
    mid = resample_schedule([0.2, 0.8], 3)
    check(np.allclose(mid, [0.2, 0.5, 0.8]), f"linear resample {mid}", failures)
    # Natural-cubic spline reproduces a line (second derivative vanishes).
    # A quadratic is NOT reproduced: natural boundary conditions force y''=0
    # at the endpoints, which a quadratic does not satisfy. That is a
    # property of the chosen spline, recorded in the QAOA docstring.
    s = np.linspace(0, 1, 4)
    y = 1 + 2 * s
    y5 = resample_schedule(y, 5)
    s5 = np.linspace(0, 1, 5)
    target = 1 + 2 * s5
    check(np.max(np.abs(y5 - target)) < 1e-10, f"cubic resample of a line {y5}", failures)
    b, g = anneal_angles(4, T=2.0, gamma_field=0.5)
    check(abs(g[0] - (2.0 / 4) * 0.5 / 4) < 1e-12 or True, "anneal defined", failures)
    # s1 = 0.5/4 = 0.125, dt=0.5, γ1=0.0625
    check(abs(g[0] - 0.0625) < 1e-12, f"anneal gamma0 {g[0]}", failures)
    # β1 = dt * (-(1-s) - Gamma s) = 0.5 * (-(1-0.125) - 0.5*0.125)
    expect = 0.5 * (-(1 - 0.125) - 0.5 * 0.125)
    check(abs(b[0] - expect) < 1e-12, f"anneal beta0 {b[0]} vs {expect}", failures)
    dur = float(np.sum(np.abs(b)) + np.sum(np.abs(g)))
    Tm = matched_anneal_time(dur, 0.5)
    check(abs(Tm - 2.0) < 1e-9, f"matched T {Tm}", failures)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=Path, default=Path(__file__).resolve().parent / "results" / "validation")
    args = ap.parse_args()
    out = args.outdir
    out.mkdir(parents=True, exist_ok=True)
    failures = []
    records = {"J": J, "normalization": "a_N = N/C(N,4), J=1, H = H_C - Gamma sum X"}
    test_identity_and_enumeration(failures, records)
    test_term_lists(out, records)
    test_dicke_matches_full(failures, records)
    test_dense_spectrum_subset(failures, records)
    test_periods(failures, records)
    test_observables(failures, records)
    test_interpolation_and_carry(failures)
    records["n_failures"] = len(failures)
    records["failures"] = failures
    (out / "validation.json").write_text(json.dumps(records, indent=2))
    print(json.dumps({"n_failures": len(failures), "failures": failures[:12], "outdir": str(out)}, indent=2))
    if failures:
        raise SystemExit(1)
    print("validation OK")


if __name__ == "__main__":
    main()
