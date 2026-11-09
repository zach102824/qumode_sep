"""T1–T14 for the provisional N9-FT simulator. Tolerances are 1e-12 unless noted."""
from __future__ import annotations

import numpy as np
from scipy.special import eval_genlaguerre, factorial

from Nov9_SK_SPSA.experiments.FT.instance import (
    build_instance,
    make_J,
    pair_diag,
    sha256_f64,
    upper,
    z_table,
)
from Nov9_SK_SPSA.experiments.FT.params import (
    METHOD_ID,
    initial_x,
    make_rng,
    n_iter_for_start_budget,
    spec_for,
    unpack,
)
from Nov9_SK_SPSA.experiments.FT.sim import (
    _WALSH_V,
    apply_Hext,
    boson_generator,
    displacement_matrix,
    fwht,
    generator,
    get_tables,
    h0_energy,
    logical_shell_phase,
    metrics_from_fock,
    qaoa_dense_state,
    run_nov3,
    run_nov3_vacuum,
    run_nov9,
    run_qaoa,
    snap_level_phase,
    t_from_s,
    walsh_phase,
)
from Nov9_SK_SPSA.experiments.FT.spsa import spsa_start


def _check(name, ok, detail):
    return {"name": name, "ok": bool(ok), "detail": detail}


def _s_phase(s1, s2, s3):
    out = np.zeros(4)
    for p in range(4):
        out[p] = s1 * p + s2 * (p * (p - 1) / 2.0) + (s3 if p == 3 else 0.0)
    return out


def run_conformance() -> list[dict]:
    out = []
    inst = build_instance(6, 10000, 1.0, role="pilot")
    inst_b = build_instance(6, 10000, 1.5, role="pilot")

    J2 = make_J(10000, 6)
    same = np.array_equal(J2, inst.J) and sha256_f64(upper(J2)) == inst.J_sha256
    same_b = np.array_equal(inst.J, inst_b.J)
    out.append(_check("T1_J_sha256", same and same_b, {"sha": inst.J_sha256, "shared_across_b": same_b}))

    out.append(_check("T2_gs_residual", inst.residual <= 1e-12 and inst_b.residual <= 1e-12 and inst.gs.min() > 0,
                       {"residual_b1": inst.residual, "residual_b1.5": inst_b.residual, "gs_min": float(inst.gs.min())}))

    spec = spec_for("nov3_km", 2)
    x = initial_x(spec, 2, 1, make_rng(10000, 6, 1.0, 2, METHOD_ID["nov3_km"], 1, 2), inst.theta_K_prior)
    un = unpack(spec, x)
    vac = run_nov3_vacuum(inst, un)
    g = inst.zz_cost(un["theta"])
    psi = np.full(1 << 6, 2.0 ** (-3.0), dtype=np.complex128)
    eph = np.exp(-1j * un["gamma"] * g)
    for ell in range(2):
        shell = np.exp(-1j * logical_shell_phase(6, inst.groups, *un["shells"][ell]))
        psi *= eph
        psi = fwht(psi)
        psi *= shell
        psi = fwht(psi)
    ov = np.vdot(vac, psi)
    phase = ov / abs(ov)
    diff = float(np.max(np.abs(vac - psi * phase)))
    out.append(_check("T3_nov3_rewrite", abs(ov) > 1 - 1e-12 and diff <= 1e-12, {"overlap": abs(ov), "max_abs": diff}))

    alpha = 0.37
    D = generator(48).exp(alpha)
    Dref = generator(48).expm_reference(alpha)
    n = np.arange(48)
    coh = np.exp(-0.5 * alpha**2) * alpha**n / np.sqrt(factorial(n))
    coh_err = float(np.max(np.abs(D[:, 0] - coh)))
    expm_err = float(np.max(np.abs(D - Dref)))
    lag_err = 0.0
    for m in (0, 1, 4, 7, 15):
        for n_ in (0, 2, 5, 11):
            if m >= n_:
                pred = np.exp(-0.5 * alpha**2) * np.sqrt(factorial(n_) / factorial(m)) * alpha ** (m - n_) * eval_genlaguerre(n_, m - n_, alpha**2)
            else:
                pred = np.exp(-0.5 * alpha**2) * np.sqrt(factorial(m) / factorial(n_)) * (-alpha) ** (n_ - m) * eval_genlaguerre(m, n_ - m, alpha**2)
            lag_err = max(lag_err, abs(D[m, n_] - pred))
    out.append(_check("T4_displacement", max(coh_err, expm_err, lag_err) <= 1e-12,
                       {"coherent": coh_err, "expm": expm_err, "laguerre": lag_err}))

    DD = generator(48).exp(0.25) @ generator(48).exp(-0.25)
    t5 = float(np.max(np.abs(DD - np.eye(48))))
    out.append(_check("T5_displacement_inverse", t5 <= 1e-12, {"max_abs": t5}))

    s1, s2, s3 = 0.31, -0.17, 0.44
    rel = (walsh_phase(t_from_s(s1, s2, s3)) - walsh_phase(t_from_s(s1, s2, s3))[0]) - (_s_phase(s1, s2, s3) - _s_phase(s1, s2, s3)[0])
    # direct levels vs logical popcount on one cavity
    ph = snap_level_phase(8, s1, s2, s3)
    pop_ok = np.allclose(ph[:4], _s_phase(s1, s2, s3)[:4]) or True
    pops = [(f & 1) + ((f >> 1) & 1) + ((f >> 2) & 1) for f in range(8)]
    level_err = max(abs(ph[f] - _s_phase(s1, s2, s3)[pops[f]]) for f in range(8))
    out.append(_check("T6_shell_map", np.max(np.abs(rel)) <= 1e-12 and level_err <= 1e-12 and abs(np.linalg.det(_WALSH_V[1:] - _WALSH_V[0])) > 1,
                       {"relative_phase_err": float(np.max(np.abs(rel))), "level_err": level_err}))

    tables = get_tables(6, 10000, 1.0, 10, "pilot")
    unp = {"gamma": 0.2, "theta": {"K": 0.3, "M": -0.4}, "alphas": [0.0], "shells": np.array([[0.2, -0.1, 0.05]])}
    # Diagonal operators commute. Also Uz and U0 differ on a partial leak.
    from Nov9_SK_SPSA.experiments.FT.sim import _ucz_phase
    pz = _ucz_phase(tables, 0.2, {"K": 0.3, "M": -0.2}, "uz")
    p0 = _ucz_phase(tables, 0.2, {"K": 0.3, "M": -0.2}, "u0")
    snap = snap_level_phase(10, 0.2, -0.1, 0.05)
    # product of diagonals is commutative by construction; check a partially leaked basis vector.
    differ = float(np.max(np.abs(pz - p0)))
    # commute test: phase multiply order on a random vector
    rng = np.random.default_rng(1)
    v = rng.normal(size=tables.dim) + 1j * rng.normal(size=tables.dim)
    e1 = np.exp(-1j * pz) * v
    # snap is a product over cavities; build full snap phase
    full_snap = np.zeros(tables.dim)
    for c in range(tables.C):
        full_snap += snap[tables.fs[:, c]]
    a = np.exp(-1j * full_snap) * (np.exp(-1j * pz) * v)
    b = np.exp(-1j * pz) * (np.exp(-1j * full_snap) * v)
    commute = float(np.max(np.abs(a - b)))
    out.append(_check("T7_ucz_snap_commute", commute <= 1e-12 and differ > 1e-8,
                       {"commute_max_abs": commute, "Uz_vs_U0_max_abs_phase": differ}))

    # T8 dense spectrum at NF=10.
    dim = tables.dim
    H = np.zeros((dim, dim), dtype=np.float64)
    for i in range(dim):
        e = np.zeros(dim)
        e[i] = 1.0
        H[:, i] = np.real(apply_Hext(e.astype(np.complex128), tables))
    H = 0.5 * (H + H.T)
    evals = np.linalg.eigvalsh(H)
    gap = float(evals[0] - inst.E0)
    psi_gs = tables.embed_logical(inst.gs.astype(np.complex128))
    e_gs = metrics_from_fock(psi_gs, tables)["E"]
    out.append(_check("T8_Hext_bound", evals[0] >= inst.E0 - 1e-8 and abs(e_gs - inst.E0) <= 1e-8,
                       {"min_Hext": float(evals[0]), "E0": inst.E0, "min_minus_E0": gap, "embedded_GS": e_gs}))

    G8 = boson_generator(8)
    expect = {
        (0, 1): (1.0, "bit0"),
        (1, 2): (np.sqrt(2), "bit0_and_bit1"),
        (2, 3): (np.sqrt(3), "bit0"),
        (3, 4): (2.0, "bit0_bit1_bit2"),
        (4, 5): (np.sqrt(5), "bit0"),
        (5, 6): (np.sqrt(6), "bit0_and_bit1"),
        (6, 7): (np.sqrt(7), "bit0"),
    }
    edge_err = 0.0
    flips_ok = True
    for (i, j), (w, tag) in expect.items():
        edge_err = max(edge_err, abs(G8[j, i] - w))
        bits = i ^ j
        if tag == "bit0":
            flips_ok &= bits == 0b001
        elif tag == "bit0_and_bit1":
            flips_ok &= bits == 0b011
        else:
            flips_ok &= bits == 0b111
    # no other code edges
    off = 0
    for i in range(8):
        for j in range(i + 1, 8):
            if abs(G8[j, i]) > 1e-12 and (i, j) not in expect:
                off += 1
    out.append(_check("T9_generator_edges", edge_err <= 1e-12 and flips_ok and off == 0,
                       {"weight_err": edge_err, "unexpected_edges": off}))

    tables16 = get_tables(6, 10000, 1.0, 16, "pilot")
    un = {"gamma": -0.37, "theta": {"K": 0.4, "M": -0.2}, "alphas": [0.0, 0.0], "shells": np.array([[0.3, -0.2, 0.1], [-0.1, 0.2, -0.4]])}
    m10 = run_nov9(tables16, un, "uz", "restricted")
    out.append(_check("T10_alpha0", abs(m10["E_zz"]) <= 1e-12 and m10["p_leak"] <= 1e-12,
                       {"E_zz": m10["E_zz"], "p_leak": m10["p_leak"], "E": m10["E"]}))

    def zz(gamma):
        un1 = {"gamma": gamma, "theta": {"K": 0.25, "M": -0.15}, "alphas": [0.08], "shells": np.array([[0.2, -0.3, 0.1]])}
        return run_nov9(tables16, un1, "uz", "restricted")["E_zz"]
    d11 = abs(zz(0.33) - zz(-0.33))
    out.append(_check("T11_gamma_even_L1", d11 <= 1e-12, {"abs_diff": d11}))

    psiL = np.zeros(tables16.dim, dtype=np.complex128)
    psiL.reshape((16, 16))[9, 1] = 1.0
    e_ext = metrics_from_fock(psiL, tables16)["E"]
    e_h0 = h0_energy(psiL, tables16)
    out.append(_check("T12_H0_lower", e_h0 < e_ext - 1e-8, {"Hext": e_ext, "H0": e_h0}))

    specq = spec_for("qaoa", 1)
    n_calls = {"n": 0}

    def obj(x, ctx):
        n_calls["n"] += 1
        return {"E": float(np.sum(x**2)), "F": 1.0, "ok": True, "n_queries": 1}

    total_q = 0
    for s in range(4):
        rngs = make_rng(10000, 6, 1.0, 1, METHOD_ID["qaoa"], s, 0)
        x0 = np.zeros(specq.n)
        r = spsa_start(obj, x0, specq, rngs, n_iter_for_start_budget(256))
        total_q += r["queries"]
    out.append(_check("T13_query_counter", total_q == 1024 and n_calls["n"] == 1024,
                       {"queries": total_q, "calls": n_calls["n"]}))

    xq = initial_x(specq, 1, 1, make_rng(10000, 6, 1.0, 1, 0, 1, 2), 0.0)
    # L=2 independent implementation
    specq2 = spec_for("qaoa", 2)
    xq2 = initial_x(specq2, 2, 3, make_rng(3, 6, 1.5, 2, 0, 3, 2), 0.0)
    unq = unpack(specq2, xq2)
    fast = run_qaoa(inst_b, unq)
    dense = qaoa_dense_state(inst_b, unq)
    qerr = float(np.max(np.abs(fast["_psi"] - dense)))
    out.append(_check("T14_qaoa_dense", qerr <= 1e-12, {"max_abs": qerr, "E": fast["E"]}))

    # Feature definitions, not a numbered gate, kept with the suite.
    rms_ok = True
    detail_f = {}
    for name, raw, scale in (
        ("K", inst.K_raw, inst.scale_K),
        ("M", inst.M_raw, inst.scale_M),
        ("R1", inst.R1_raw, inst.scale_R1),
    ):
        got = float(np.sqrt(np.mean(upper(raw * scale) ** 2)))
        detail_f[name] = {"rms": got, "scale": scale}
        rms_ok &= abs(got - inst.rms_J) <= 1e-12
    R, lam, v = inst.R1_raw, inst.lam_min_J, inst.v_min
    Rref = lam * np.outer(v, v)
    np.fill_diagonal(Rref, 0.0)
    # v sign was fixed by sum>0 inside the builder; compare absolute outer.
    rerr = float(np.max(np.abs(np.abs(R) - np.abs(Rref)))) if False else float(np.max(np.abs(R - Rref)))
    zt = z_table(6)
    h = inst.J @ zt
    const = float(np.sum(np.diag(inst.J @ inst.J)))
    kerr = float(np.max(np.abs(np.sum(h**2, axis=0) - (const + 2 * pair_diag(inst.K_raw, zt)))))
    out.append(_check("feature_rms_rank_K_identity", rms_ok and rerr <= 1e-10 and kerr <= 1e-10,
                       {"rms": detail_f, "R1_err": rerr, "K_identity": kerr,
                        "R1_residual_vs_JKM": inst.feature_diag["R1_residual_vs_span_JKM"],
                        "singular_values": inst.feature_diag["feature_singular_values"]}))
    return out


def main():
    rows = run_conformance()
    failed = [r for r in rows if not r["ok"]]
    for r in rows:
        print(("PASS " if r["ok"] else "FAIL ") + r["name"] + " " + str(r["detail"]))
    if failed:
        raise SystemExit(f"{len(failed)} conformance tests failed")
    print(f"all {len(rows)} checks passed")


if __name__ == "__main__":
    main()
