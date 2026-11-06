"""Small-N checks for the Baxter–Wu up/down construction."""

from __future__ import annotations

import numpy as np
from scipy.linalg import expm

import geometry as geo
import model as M
import qaoa


def run_validation() -> dict:
    failures = []
    checks = []

    def add(name, ok, detail):
        checks.append({"name": name, "pass": bool(ok), "detail": detail})
        if not ok:
            failures.append(checks[-1])

    geom = geo.build_triangles(3, 3)
    n = 9
    add(
        "no_duplicate_or_coincident_faces_3x3",
        geom["n_duplicate_up"] == 0 and geom["n_duplicate_down"] == 0 and geom["n_up_down_coincidences"] == 0 and geom["M"] == 2 * n,
        {"M": geom["M"], "dup_up": geom["n_duplicate_up"], "coinc": geom["n_up_down_coincidences"]},
    )
    for lx, ly in ((3, 4), (3, 6), (3, 7)):
        g = geo.build_triangles(lx, ly)
        add(
            f"faces_{lx}x{ly}",
            g["n_duplicate_up"] == 0 and g["n_duplicate_down"] == 0 and g["n_up_down_coincidences"] == 0 and g["M"] == 2 * g["n"],
            {"M": g["M"], "N": g["n"]},
        )

    tris, coeffs = M.faces_and_coeffs(geom, 1.0)
    hc = M.cost_diagonal(n, tris, coeffs)
    kron_err = 0.0
    for tri, c in zip(tris, coeffs):
        kron_err = max(kron_err, float(np.max(np.abs(c * M.kron_zzz_diag(n, tri) - c * M.zzz_diag(n, tri)))))
    add("kron_matches_bit_diagonal", kron_err < 1e-10, {"max_abs": kron_err})

    # Period from the diagonal, confirmed on a generic state.
    per = M.energy_step_and_period(hc)
    T = per["gamma_period"]
    rng = np.random.default_rng(0)
    psi = rng.normal(size=1 << n) + 1j * rng.normal(size=1 << n)
    psi /= np.linalg.norm(psi)
    a = qaoa.apply_cost(psi, hc, 0.21)
    b = qaoa.apply_cost(psi, hc, 0.21 + T)
    half = qaoa.apply_cost(psi, hc, 0.21 + 0.5 * T)
    add("gamma_period_exact", abs(np.vdot(a, b)) ** 2 > 1 - 1e-10, {"period": T, "overlap": float(abs(np.vdot(a, b)) ** 2)})
    add("gamma_half_period_not", abs(np.vdot(a, half)) ** 2 < 0.999, {"overlap": float(abs(np.vdot(a, half)) ** 2)})

    ket0 = np.zeros(1 << n, dtype=np.complex128)
    ket0[0] = 1.0
    mb = qaoa.apply_mixer(ket0, 0.37, n)
    mb2 = qaoa.apply_mixer(ket0, 0.37 + np.pi, n)
    mbh = qaoa.apply_mixer(ket0, 0.37 + 0.5 * np.pi, n)
    add("beta_period_pi", abs(np.vdot(mb, mb2)) ** 2 > 1 - 1e-10, {"overlap": float(abs(np.vdot(mb, mb2)) ** 2)})
    add("beta_half_not", abs(np.vdot(mb, mbh)) ** 2 < 0.999, {"overlap": float(abs(np.vdot(mb, mbh)) ** 2)})

    # expm replay
    HX = np.zeros((1 << n, 1 << n))
    idx = np.arange(1 << n)
    for q in range(n):
        partner = idx ^ (1 << q)
        upper = partner > idx
        HX[idx[upper], partner[upper]] = 1
        HX[partner[upper], idx[upper]] = 1
    HC = np.diag(hc)
    plus = M.plus_state(n)
    err = 0.0
    for sched in ([(0.11, 0.22)], [(0.07, -0.15), (0.19, 0.4)]):
        psi_s = plus.copy()
        for g, bta in sched:
            psi_s = expm(-1j * bta * HX) @ (expm(-1j * g * HC) @ psi_s)
        psi_q = qaoa.apply_qaoa(plus, hc, n, [g for g, _ in sched], [b for _, b in sched])
        err = max(err, float(np.max(np.abs(psi_s - psi_q))))
    add("expm_replay", err < 1e-8, {"max_abs": err})

    dense = M.ground_state_dense(hc, 1.0, n)
    lanc = M.ground_state(hc, 1.0, n)
    ov = float(np.dot(dense["gs"], lanc["gs"]) ** 2)
    add(
        "lanczos_matches_dense_full_space",
        ov > 1 - 1e-8 and abs(dense["E0"] - lanc["E0"]) < 1e-8 and lanc["dim"] == 512,
        {"overlap": ov, "dE": abs(dense["E0"] - lanc["E0"]), "dim": lanc["dim"]},
    )
    add("spectrum_symmetric", abs(lanc["emax_plus_e0"]) < 1e-8, {"emax_plus_e0": lanc["emax_plus_e0"]})
    add("gs_nonnegative", lanc["n_negative"] == 0 and not lanc["degenerate"], {"n_negative": lanc["n_negative"], "gap": lanc["gap"]})

    # Entropy unit tests.
    prod = np.zeros(1 << n, dtype=np.complex128)
    prod[0] = 1.0
    Sprod = M.half_cut_entropy(prod, list(range(4)))["S_bits"]
    bell = np.zeros(1 << n, dtype=np.complex128)
    bell[0] = 1 / np.sqrt(2)
    bell[(1 << 0) | (1 << 4)] = 1 / np.sqrt(2)
    Sbell = M.half_cut_entropy(bell, [0])["S_bits"]
    add("entropy_product_and_bell", abs(Sprod) < 1e-8 and abs(Sbell - 1) < 1e-6, {"S_product": Sprod, "S_bell": Sbell})

    # Gauge: staggered is not uniform. Brute on 3x3.
    # Reuse the already-trusted GF2 plus a spectrum comparison against uniform.
    tris_u, coeff_u = M.faces_and_coeffs(geom, -1.0)
    hc_u = M.cost_diagonal(n, tris_u, coeff_u)
    # multisets differ
    same_spectrum = np.allclose(np.sort(hc), np.sort(hc_u))
    add("staggered_spectrum_differs_from_uniform", not same_spectrum, {"stag_min": float(hc.min()), "uniform_min": float(hc_u.min())})

    # Z2xZ2 commutes on 3x3, and the full-space GS eigenvalue is recorded not imposed.
    colors = geo.extend_face_coloring(3, 3, geom)["colors"]
    v = rng.normal(size=1 << n)
    Hv = M.apply_H(v, hc, 0.7, n)
    mask = 0
    for q in range(n):
        if int(colors[q]) in (0, 1):
            mask |= 1 << q
    PHv = M.geo_flip(Hv, mask)
    HPv = M.apply_H(M.geo_flip(v, mask), hc, 0.7, n)
    add("z2xz2_commutes_on_3x3", float(np.max(np.abs(PHv - HPv))) < 1e-8, {"max_abs": float(np.max(np.abs(PHv - HPv)))})

    # 3x4 has no face coloring.
    g12 = geo.build_triangles(3, 4)
    col12 = geo.coloring_report(3, 4, g12)
    add("n12_coloring_broken", col12["face_coloring_exists"] is False, {"mono_edges": col12["n_monochrome_edges"]})
    g21 = geo.build_triangles(3, 7)
    col21 = geo.coloring_report(3, 7, g21)
    add("n21_coloring_broken", col21["face_coloring_exists"] is False, {"mono_edges": col21["n_monochrome_edges"]})
    g18 = geo.build_triangles(3, 6)
    col18 = geo.coloring_report(3, 6, g18)
    add("n18_coloring_holds", col18["face_coloring_exists"] is True, {"counts": col18["face_color_counts"]})

    gauge = geo.gauge_report(geom)
    add("n9_staggered_not_gauge_equivalent", gauge["gauge_equivalent_to_uniform_ferro"] is False, {"rank": gauge["rank_unique"], "left_kernel": gauge["left_kernel_dim"]})

    return {"pass": len(failures) == 0, "n_checks": len(checks), "n_failed": len(failures), "failures": failures, "checks": checks}
