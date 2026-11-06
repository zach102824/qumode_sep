"""Construction checks. Physics scans should not run if these fail.

Checks:
- distinct-index bookkeeping identity for (sum Z)^3
- bit-parity cost diagonal versus an independent Kronecker Pauli build
- matrix-free matvec versus dense H
- exact γ, β period π, and the failure of period π/2
- (γ, β) -> (-γ, -β) conjugation
- statevector QAOA versus dense expm replay
- unique classical all-+ ground state of a full-rank uniform instance at zero field
- gauge equivalence (subset spin flip) and the odd-p global sign flip
- half-cut entropy of a product state and of a Bell pair
- stoquastic ground-state sign
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import expm

import model
import qaoa


def _check(name, ok, detail, failures):
    row = {"name": name, "pass": bool(ok), "detail": detail}
    if not ok:
        failures.append(row)
    return row


def run_validation(graph: dict | None = None) -> dict:
    failures = []
    rows = []

    resid = model.distinct_index_identity_residual(6)
    rows.append(
        _check(
            "identity_(sumZ)^3_N6",
            resid < 1e-9,
            {"max_abs_residual": resid},
            failures,
        )
    )

    # Small fixed full-rank graph if the caller did not pass one.
    if graph is None:
        found = model.select_graphs(6, k=1, max_seed=5000)
        if not found["graphs"]:
            rows.append(_check("graph_available", False, found, failures))
            return {"pass": False, "checks": rows}
        graph = found["graphs"][0]
    n = int(graph["n"])
    edges = [tuple(e) for e in graph["edges"]]
    J = model.uniform_couplings(len(edges), +1.0)
    hc = model.cost_diagonal(n, edges, J)

    # Independent Pauli diagonal for every edge.
    pauli_mismatch = 0.0
    for edge in edges:
        bit = model.triple_eigenvalues(n, edge)
        kron = model.pauli_zzz_diagonal_kron(n, edge)
        pauli_mismatch = max(pauli_mismatch, float(np.max(np.abs(bit - kron))))
    rows.append(
        _check(
            "pauli_kron_matches_bit_diag",
            pauli_mismatch < 1e-10,
            {"max_abs": pauli_mismatch, "n": n, "n_edges": len(edges)},
            failures,
        )
    )

    classical = model.count_classical_ground_states(hc, len(edges))
    rows.append(
        _check(
            "unique_classical_allplus_ground_state",
            classical["n_ground"] == 1 and classical["all_plus_index_is_ground"],
            classical,
            failures,
        )
    )

    rank = model.gf2_rank(model.incidence_matrix(edges, n))
    rows.append(_check("gf2_full_rank", rank == n, {"rank": rank, "n": n}, failures))

    # Zero-field dense GS is the classical product.
    zero = model.ground_state(hc, 0.0, n)
    f_prod = float(zero["gs"][0] ** 2)
    rows.append(
        _check(
            "gamma0_dense_gs_is_allplus_product",
            (not zero["degenerate"]) and f_prod > 1.0 - 1e-8 and abs(zero["evals0"] + len(edges)) < 1e-8,
            {"E0": zero["evals0"], "F_allplus": f_prod, "gap": zero["gap"], "degenerate": zero["degenerate"]},
            failures,
        )
    )

    Gamma = 0.7
    spec = model.ground_state(hc, Gamma, n)
    dense = model.ground_state_dense(hc, Gamma, n)
    gs_ov = float(np.dot(spec["gs"], dense["gs"]) ** 2)
    rows.append(
        _check(
            "lanczos_matches_dense_ground_state",
            gs_ov > 1.0 - 1e-8
            and abs(spec["evals0"] - dense["evals0"]) < 1e-8
            and abs(spec["gap"] - dense["gap"]) < 1e-7
            and abs(spec["evals_max"] - dense["evals_max"]) < 1e-7
            and spec["residual"] < 1e-8,
            {
                "overlap": gs_ov,
                "E0_diff": abs(spec["evals0"] - dense["evals0"]),
                "gap_diff": abs(spec["gap"] - dense["gap"]),
                "Emax_diff": abs(spec["evals_max"] - dense["evals_max"]),
                "lanczos_residual": spec["residual"],
                "method": spec["method"],
            },
            failures,
        )
    )
    rows.append(
        _check(
            "spectrum_symmetric_Emax_equals_minus_E0",
            abs(spec["emax_plus_e0"]) < 1e-8 and abs(dense["evals_max"] + dense["evals0"]) < 1e-8,
            {
                "emax_plus_e0": spec["emax_plus_e0"],
                "reason": "Conjugation by prod_i Y_i sends Z->-Z and X->-X. Odd-body ZZZ and the transverse field both flip, so H -> -H.",
            },
            failures,
        )
    )
    rows.append(
        _check(
            "full_space_dimension",
            spec["dim"] == (1 << n),
            {"dim": spec["dim"]},
            failures,
        )
    )
    rows.append(
        _check(
            "stoquastic_gs_nonnegative",
            spec["n_negative_amps"] == 0 and not spec["degenerate"],
            {"n_negative": spec["n_negative_amps"], "gap": spec["gap"], "degenerate": spec["degenerate"]},
            failures,
        )
    )

    # Matvec versus dense.
    H = model.dense_hamiltonian(hc, Gamma, n)
    rng = np.random.default_rng(0)
    v = rng.normal(size=1 << n) + 1j * rng.normal(size=1 << n)
    mv = model.apply_H_matvec(v, hc, Gamma, n)
    dense_mv = H @ v
    matvec_err = float(np.max(np.abs(mv - dense_mv)))
    rows.append(_check("matvec_matches_dense", matvec_err < 1e-9, {"max_abs": matvec_err}, failures))

    # Spectrum of the cost: all values ≡ M (mod 2) and step by 2, so period is π.
    M = len(edges)
    # Every diagonal entry must be an integer congruent to M mod 2.
    rounded = np.rint(hc)
    int_err = float(np.max(np.abs(hc - rounded)))
    parity_ok = bool(np.all((np.rint(rounded).astype(int) - M) % 2 == 0))
    rows.append(
        _check(
            "cost_spectrum_same_parity",
            int_err < 1e-9 and parity_ok,
            {"max_abs_int_err": int_err, "M": M, "min": float(hc.min()), "max": float(hc.max())},
            failures,
        )
    )

    psi_rand = rng.normal(size=1 << n) + 1j * rng.normal(size=1 << n)
    psi_rand = psi_rand / np.linalg.norm(psi_rand)
    g0 = 0.37
    psi_g = qaoa.apply_cost(psi_rand, hc, g0)
    psi_g_pi = qaoa.apply_cost(psi_rand, hc, g0 + np.pi)
    psi_g_half = qaoa.apply_cost(psi_rand, hc, g0 + 0.5 * np.pi)
    per_pi = float(abs(np.vdot(psi_g, psi_g_pi)) ** 2)
    per_half = float(abs(np.vdot(psi_g, psi_g_half)) ** 2)
    rows.append(
        _check(
            "gamma_period_pi_exact",
            per_pi > 1.0 - 1e-10,
            {"overlap": per_pi},
            failures,
        )
    )
    rows.append(
        _check(
            "gamma_period_not_pi_over_2",
            per_half < 0.999,
            {"overlap_at_half_period": per_half},
            failures,
        )
    )

    # Mixer period on |0>, which is not an X eigenstate. |+> would hide the period.
    ket0 = np.zeros(1 << n, dtype=np.complex128)
    ket0[0] = 1.0
    b0 = 0.41
    psi_b = qaoa.apply_mixer(ket0, b0, n)
    psi_b_pi = qaoa.apply_mixer(ket0, b0 + np.pi, n)
    psi_b_half = qaoa.apply_mixer(ket0, b0 + 0.5 * np.pi, n)
    b_pi = float(abs(np.vdot(psi_b, psi_b_pi)) ** 2)
    b_half = float(abs(np.vdot(psi_b, psi_b_half)) ** 2)
    rows.append(_check("beta_period_pi_exact", b_pi > 1.0 - 1e-10, {"overlap": b_pi}, failures))
    rows.append(
        _check(
            "beta_period_not_pi_over_2",
            b_half < 0.999,
            {"overlap_at_half_period": b_half, "note": "even-p π/2 claim does not hold for this odd-p sum-X mixer"},
            failures,
        )
    )

    # Conjugation symmetry.
    plus = model.plus_state(n)
    gammas = [0.31, -0.22]
    betas = [0.17, 0.55]
    psi = qaoa.apply_qaoa(plus, hc, n, gammas, betas)
    psi_c = qaoa.apply_qaoa(plus, hc, n, [-g for g in gammas], [-b for b in betas])
    conj_ov = float(abs(np.vdot(np.conj(psi), psi_c)) ** 2)
    e1 = qaoa.energy_of(psi, hc, n, Gamma)
    e2 = qaoa.energy_of(psi_c, hc, n, Gamma)
    f1 = float(abs(np.vdot(spec["gs"], psi)) ** 2)
    f2 = float(abs(np.vdot(spec["gs"], psi_c)) ** 2)
    rows.append(
        _check(
            "angle_conjugation_symmetry",
            conj_ov > 1.0 - 1e-10 and abs(e1 - e2) < 1e-9 and abs(f1 - f2) < 1e-10,
            {"state_overlap_with_conjugate": conj_ov, "energy_abs_diff": abs(e1 - e2), "F_abs_diff": abs(f1 - f2)},
            failures,
        )
    )

    # Dense expm replay of one layer and of two layers.
    HX = model.build_HX(n)
    HC = np.diag(hc)
    def expm_qaoa(gs_list, bs_list):
        psi = plus.copy()
        for g, b in zip(gs_list, bs_list):
            psi = expm(-1j * b * HX) @ (expm(-1j * g * HC) @ psi)
        return psi

    replay_err = 0.0
    for schedule in ([(0.2, -0.35)], [(0.15, 0.4), (0.55, -0.1)]):
        gv = [g for g, _b in schedule]
        bv = [b for _g, b in schedule]
        a = qaoa.apply_qaoa(plus, hc, n, gv, bv)
        b = expm_qaoa(gv, bv)
        replay_err = max(replay_err, float(np.max(np.abs(a - b))))
    rows.append(_check("circuit_expm_replay", replay_err < 1e-9, {"max_abs": replay_err}, failures))

    # Principal-branch replay.
    raw_g, raw_b = [0.2 + np.pi, -1.7], [0.4 - 2 * np.pi, 3.3]
    psi_raw = qaoa.apply_qaoa(plus, hc, n, raw_g, raw_b)
    rep = qaoa.branch_report(raw_g, raw_b)
    psi_prin = qaoa.apply_qaoa(plus, hc, n, rep["gamma_principal"], rep["beta_principal"])
    prin_ov = float(abs(np.vdot(psi_raw, psi_prin)) ** 2)
    rows.append(
        _check(
            "principal_branch_same_state",
            prin_ov > 1.0 - 1e-10 and rep["n_gamma_wrapped"] + rep["n_beta_wrapped"] >= 1,
            {"overlap": prin_ov, "branches_gamma": rep["gamma_branch_k"], "branches_beta": rep["beta_branch_k"]},
            failures,
        )
    )

    # Gauge: flip qubits {0, 2} when n > 2.
    flipped = [0, 2] if n > 2 else [0]
    Jg = model.gauge_couplings(edges, J, flipped)
    hc_g = model.cost_diagonal(n, edges, Jg)
    spec_g = model.ground_state(hc_g, Gamma, n)
    mask = 0
    for q in flipped:
        mask |= 1 << q
    mapped = model.flip_state(spec["gs"].astype(np.complex128), mask)
    ov = np.vdot(mapped, spec_g["gs"].astype(np.complex128))
    gauge_gs = float(abs(ov) ** 2)
    # Same angles, fidelity to each ground state.
    psi_g = qaoa.apply_qaoa(plus, hc_g, n, [0.33, 0.12], [0.2, -0.18])
    psi_u = qaoa.apply_qaoa(plus, hc, n, [0.33, 0.12], [0.2, -0.18])
    Fg = float(abs(np.vdot(spec_g["gs"], psi_g)) ** 2)
    Fu = float(abs(np.vdot(spec["gs"], psi_u)) ** 2)
    # The gauge-transformed QAOA state should match.
    state_map = float(abs(np.vdot(model.flip_state(psi_u, mask), psi_g)) ** 2)
    rows.append(
        _check(
            "gauge_spin_flip_equivalence",
            gauge_gs > 1.0 - 1e-8 and abs(Fg - Fu) < 1e-8 and state_map > 1.0 - 1e-8,
            {
                "flipped_qubits": flipped,
                "gs_overlap": gauge_gs,
                "F_uniform": Fu,
                "F_gauge": Fg,
                "qaoa_state_overlap": state_map,
                "spectrum_E0_diff": abs(spec["evals0"] - spec_g["evals0"]),
            },
            failures,
        )
    )

    # Global sign flip J -> -J, implemented by P = X^{\otimes n}. Odd p.
    Jm = model.uniform_couplings(len(edges), -1.0)
    hc_m = model.cost_diagonal(n, edges, Jm)
    spec_m = model.ground_state(hc_m, Gamma, n)
    mask_all = (1 << n) - 1
    mapped_all = model.flip_state(spec["gs"].astype(np.complex128), mask_all)
    ov_all = float(abs(np.vdot(mapped_all, spec_m["gs"].astype(np.complex128))) ** 2)
    ang_g = [0.27, -0.41, 0.15]
    ang_b = [0.33, 0.08, -0.22]
    psi_m = qaoa.apply_qaoa(plus, hc_m, n, ang_g, ang_b)
    psi_p = qaoa.apply_qaoa(plus, hc, n, ang_g, ang_b)
    Fm = float(abs(np.vdot(spec_m["gs"], psi_m)) ** 2)
    Fp = float(abs(np.vdot(spec["gs"], psi_p)) ** 2)
    state_all = float(abs(np.vdot(model.flip_state(psi_p, mask_all), psi_m)) ** 2)
    rows.append(
        _check(
            "odd_p_global_sign_flip",
            ov_all > 1.0 - 1e-8 and abs(Fm - Fp) < 1e-8 and state_all > 1.0 - 1e-8,
            {
                "gs_overlap": ov_all,
                "F_plusJ": Fp,
                "F_minusJ": Fm,
                "F_abs_diff": abs(Fm - Fp),
                "qaoa_state_overlap": state_all,
                "note": "Same angles, fidelities to each model's own full-space ground state.",
            },
            failures,
        )
    )

    # Global flip is the gauge that flips every edge, because every edge has odd size.
    J_all = model.gauge_couplings(edges, J, list(range(n)))
    rows.append(
        _check(
            "global_flip_matches_all_minus_gauge",
            bool(np.allclose(J_all, -1.0)),
            {"unique_couplings": sorted(set(float(x) for x in J_all))},
            failures,
        )
    )

    # Entropy unit tests, independent of the Hamiltonian.
    prod = np.zeros(1 << n, dtype=np.complex128)
    prod[0] = 1.0
    S_prod = model.half_cut_entropy(prod, n)["S_half_bits"]
    # Bell pair on the two sides of the cut: |00...0> + |1 on qubit 0 and qubit na>.
    na = n // 2
    bell = np.zeros(1 << n, dtype=np.complex128)
    bell[0] = 1.0 / np.sqrt(2.0)
    bell[(1 << 0) | (1 << na)] = 1.0 / np.sqrt(2.0)
    S_bell = model.half_cut_entropy(bell, n)["S_half_bits"]
    rows.append(
        _check(
            "entropy_product_and_bell",
            abs(S_prod) < 1e-10 and abs(S_bell - 1.0) < 1e-8,
            {"S_product": S_prod, "S_bell": S_bell, "cut": model.half_cut_entropy(prod, n)["cut_definition"]},
            failures,
        )
    )

    # Vectorized L=1 grid agrees with the scalar circuit at the sampled nodes.
    g_grid, b_grid, Es, Fs = qaoa.l1_grid(plus, hc, n, Gamma, spec["gs"], n_grid=5)
    grid_err = 0.0
    for ib in (0, 2, 4):
        for ig in (1, 3):
            psi_s = qaoa.apply_qaoa(plus, hc, n, [float(g_grid[ig])], [float(b_grid[ib])])
            e_s = qaoa.energy_of(psi_s, hc, n, Gamma)
            f_s = float(abs(np.vdot(spec["gs"], psi_s)) ** 2)
            grid_err = max(grid_err, abs(e_s - float(Es[ib, ig])), abs(f_s - float(Fs[ib, ig])))
    rows.append(
        _check(
            "l1_grid_matches_scalar_circuit",
            grid_err < 1e-9,
            {"max_abs_E_or_F": grid_err},
            failures,
        )
    )

    # |+> and |0> energies.
    e_plus = qaoa.energy_of(plus, hc, n, Gamma)
    e_zero = qaoa.energy_of(ket0, hc, n, Gamma)
    rows.append(
        _check(
            "analytic_endpoint_energies",
            abs(e_plus + Gamma * n) < 1e-9 and abs(e_zero + len(edges)) < 1e-9,
            {"E_plus": e_plus, "expected_plus": -Gamma * n, "E_zero": e_zero, "expected_zero": -len(edges)},
            failures,
        )
    )

    # Collective operator is a different diagonal: many more triples plus a field.
    # Compare our H_C with - (sum Z)^3 / N^2 on this N (the collective ferromagnet cost).
    s = np.zeros(1 << n, dtype=np.float64)
    for q in range(n):
        s += model.z_eigenvalues(n, q)
    h_coll = -((s**3) / float(n) ** 2)
    op_diff = float(np.max(np.abs(hc - h_coll)))
    rows.append(
        _check(
            "sparse_differs_from_collective_p3",
            op_diff > 0.5,
            {
                "max_abs_diagonal_diff": op_diff,
                "sparse_terms": len(edges),
                "collective_distinct_triples": int(n * (n - 1) * (n - 2) / 6),
                "note": "Collective -(sum Z)^3/N^2 also generates a longitudinal field. This check only records that the diagonals differ.",
            },
            failures,
        )
    )

    # Best product overlap at Gamma=0 is 1, and it is the classical product.
    prod_fit = model.best_product_overlap(zero["gs"], n, seed=0)
    rows.append(
        _check(
            "product_overlap_at_zero_field",
            prod_fit["F_product"] > 1.0 - 1e-6 and prod_fit["F_classical_allplusZ"] > 1.0 - 1e-8,
            {"F_product": prod_fit["F_product"], "F_classical": prod_fit["F_classical_allplusZ"]},
            failures,
        )
    )

    return {
        "pass": len(failures) == 0,
        "n_checks": len(rows),
        "n_failed": len(failures),
        "graph_id": graph["graph_id"],
        "failures": failures,
        "checks": rows,
    }
