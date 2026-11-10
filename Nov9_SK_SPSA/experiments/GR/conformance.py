"""T1–T14 plus permutation tests for N9-GR. No optimizer sees fidelity."""
from __future__ import annotations

import inspect
import json
import math
import sys
import traceback
from pathlib import Path

import numpy as np
from scipy.special import eval_genlaguerre

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import RESULT_DIR, load_prereg
from encoding import build_named_groups, feature_M, m_correlation, packing_from
from gates import (
    W_binH,
    displacement_matrix,
    flip_all_table,
    generator,
    hadamard3,
    s_to_walsh,
    shell_phase_of_pop,
    walsh_shell_phase,
)
from hamiltonian import (
    apply_X_sum,
    build_instance,
    dense_H,
    logical_energy,
    make_J,
    pair_K,
    pair_diag,
    plus_state,
    sha256_f64,
    upper,
)
from model import (
    H0_energy,
    _observe,
    build_fock_model,
    dense_hext,
    evaluate_nov3,
    evaluate_nov9,
    evaluate_qaoa,
    nov3_x0,
    nov9_x0,
    plus_fock,
    unpack_nov9,
)
from spsa import iterations_for_cap, run_spsa


class Suite:
    def __init__(self):
        self.rows = []

    def check(self, name: str, ok: bool, detail: str) -> None:
        self.rows.append({"name": name, "pass": bool(ok), "detail": detail})
        print(("PASS " if ok else "FAIL ") + name + " | " + detail)

    @property
    def ok(self) -> bool:
        return all(r["pass"] for r in self.rows)


def _max(a) -> float:
    return float(np.max(np.abs(a)))


def _embed_gs(model, gs):
    psi = np.zeros((model.nf,) * model.C, dtype=np.complex128)
    inv = {}
    flat = model.logical_map.reshape(-1)
    for f, x in enumerate(flat):
        if int(x) >= 0:
            inv[int(x)] = f
    for x, amp in enumerate(gs):
        psi.reshape(-1)[inv[x]] = amp
    return psi


def _fock_nov3(model, x, L: int, motif: str):
    gamma, thK, thM, s = (float(x[0]), float(x[1]), float(x[2]), np.asarray(x[3:], dtype=np.float64).reshape(L, 3))
    ang = gamma * (model.dJ_phase + thK * model.dK_phase + thM * model.dM_phase)
    phase = np.exp(-1j * ang)
    W = W_binH(model.nf)
    C = model.C
    if motif == "vacuum":
        psi = np.zeros((model.nf,) * C, dtype=np.complex128)
        psi[(0,) * C] = 1.0
        for ell in range(L):
            for c in range(C):
                psi = _apply(psi, W, c)
            psi = psi * phase
            for c in range(C):
                psi = _apply(psi, W, c)
            sph = _snap(model, s[ell])
            for c in range(C):
                psi = psi * sph.reshape(_shape(C, c, model.nf))
        for c in range(C):
            psi = _apply(psi, W, c)
    elif motif == "rewrite":
        psi = plus_fock(model.nf, C)
        for ell in range(L):
            psi = psi * phase
            for c in range(C):
                psi = _apply(psi, W, c)
            sph = _snap(model, s[ell])
            for c in range(C):
                psi = psi * sph.reshape(_shape(C, c, model.nf))
            for c in range(C):
                psi = _apply(psi, W, c)
    else:
        raise ValueError(motif)
    return psi


def _apply(psi, U, c):
    p = np.moveaxis(psi, c, 0)
    sh = p.shape
    p = U @ p.reshape(U.shape[0], -1)
    return np.moveaxis(p.reshape(sh), 0, c)


def _shape(C, c, nf):
    sh = [1] * C
    sh[c] = nf
    return sh


def _snap(model, srow):
    from gates import snap_diagonal

    return snap_diagonal(model.nf, float(srow[0]), float(srow[1]), float(srow[2]), model.packing.encoding)


def _logical_amp(model, psi):
    amp = np.zeros(1 << model.inst.n, dtype=np.complex128)
    logical = model.logical_map
    mask = logical >= 0
    np.add.at(amp, logical[mask], psi[mask])
    return amp


def run() -> Suite:
    S = Suite()
    pre = load_prereg()
    inst = build_instance(10000, 6, 1.0, pilot=True)
    inst_b = build_instance(10000, 6, 1.5, pilot=True)
    J2 = make_J(10000, 6)
    S.check("T1_J_sha", sha256_f64(upper(J2)) == inst.j_upper_sha and np.allclose(J2, inst.J), inst.j_upper_sha[:16])
    S.check(
        "T1_same_J_both_fields",
        inst.j_upper_sha == inst_b.j_upper_sha and np.allclose(inst.J, inst_b.J),
        "shared J",
    )
    S.check("T2_residual", inst.residual <= 1e-10 and inst_b.residual <= 1e-10, f"{inst.residual:.2e}")
    S.check("T2_positive_even", inst.gs_min > 0.0 and inst.gs_parity > 0.99, f"min {inst.gs_min:.3e} P {inst.gs_parity:.6f}")
    plus = plus_state(6)
    S.check("P_plus_energy", abs(logical_energy(inst, plus) + inst.b * inst.n) < 1e-12, f"{logical_energy(inst, plus):.6f}")
    e_basis = np.zeros(64, dtype=np.complex128)
    e_basis[0] = 1.0
    S.check("P_basis_energy", abs(logical_energy(inst, e_basis) - inst.hc[0]) < 1e-12, "Z basis")

    # K identity and M blocks
    h = inst.J @ inst.z
    delta = np.sum(h * h, axis=0) - 2.0 * pair_diag(inst.K_raw, inst.z)
    S.check("P_K_identity", float(np.max(np.abs(delta - delta[0]))) < 1e-10, f"span {float(np.max(delta)-np.min(delta)):.2e}")
    groups = build_named_groups(inst.J, 10000)
    inter = groups["inter_id"][0]
    rawM, _Mh, scaleM, dM = feature_M(inst.J, inter, inst.rms_J, inst.z)
    intra_ok = True
    for cav in inter:
        for a in range(3):
            for b in range(a + 1, 3):
                if rawM[cav[a], cav[b]] != 0.0:
                    intra_ok = False
    block_ok = abs(rawM[0, 1] - rawM[0, 3]) < 1e-15 and abs(np.sqrt(np.mean(upper(_Mh) ** 2)) - inst.rms_J) < 1e-12
    S.check("P_M_blocks", intra_ok and block_ok, f"scale {scaleM:.6f}")
    S.check(
        "P_feature_rms",
        abs(np.sqrt(np.mean(upper(inst.K_hat) ** 2)) - inst.rms_J) < 1e-12,
        f"rmsJ {inst.rms_J:.6e}",
    )
    corr = m_correlation(inst.J, inter, inst.rms_J, inst.z, inst.hc)
    S.check("P_pearson_finite", np.isfinite(corr), f"{corr:.6f}")

    greedy = __import__("encoding").greedy_groups
    S.check(
        "P_no_gs_in_assignment",
        "gs" not in inspect.signature(greedy).parameters
        and "J" in inspect.signature(greedy).parameters
        and all(v[1]["gs_reads"] == 0 for v in groups.values()),
        "signature is J only; gs_reads=0",
    )
    S.check(
        "P_antiR_is_reversal",
        groups["G_antiR"][0] == [[g[2], g[1], g[0]] for g in groups["G_R"][0]],
        str(groups["G_antiR"][0]),
    )
    r0, _ = __import__("encoding").random_groups(10000, 6, 0)
    r0b, _ = __import__("encoding").random_groups(10000, 6, 0)
    S.check("P_random_deterministic", r0 == r0b and r0 != __import__("encoding").random_groups(10000, 6, 1)[0], str(r0))

    pk = packing_from("inter_id", inter, "binary", groups["inter_id"][1])
    m8 = build_fock_model(inst, pk, 8, "Uz", False)
    m10 = build_fock_model(inst, pk, 10, "Uz", False)
    m16 = build_fock_model(inst, pk, 16, "Uz", False)

    # T3
    rng = np.random.default_rng(1)
    x3 = np.array([0.27, -0.15, 0.4, 0.2, -0.1, 0.05, -0.3, 0.1, 0.2], dtype=np.float64)
    psi_a = _fock_nov3(m8, x3, 2, "vacuum")
    psi_b = _fock_nov3(m8, x3, 2, "rewrite")
    amp_a = _logical_amp(m8, psi_a)
    amp_b = _logical_amp(m8, psi_b)
    out3 = evaluate_nov3(inst, inter, dM, x3, 2, keep_state=True)
    err_ab = float(np.linalg.norm(psi_a - psi_b))
    err_logic = float(np.linalg.norm(amp_a - out3["psi"]))
    S.check("T3_nov3_rewrite", err_ab < 1e-12 and err_logic < 1e-12, f"fock {err_ab:.2e} logical {err_logic:.2e}")
    x3_0 = nov3_x0(1, "E", None)
    x3_0[0] = 0.0
    out30 = evaluate_nov3(inst, inter, dM, x3_0, 1)
    S.check("T3_zero_is_plus", abs(out30["F"] - inst.F_plus) < 1e-12 and abs(out30["E"] + 6.0) < 1e-10, f"F {out30['F']:.6f}")

    # T4 T5
    alpha = 0.4
    D = displacement_matrix(alpha, 48, False)
    vac = np.zeros(48, dtype=np.complex128)
    vac[0] = 1.0
    coh = D @ vac
    analytic = np.array([np.exp(-0.5 * alpha**2) * alpha**k / math.sqrt(math.factorial(k)) for k in range(20)])
    pois = np.abs(analytic) ** 2
    lag_err = 0.0
    for m in range(10):
        for n in range(10):
            if m >= n:
                pred = (
                    np.exp(-0.5 * alpha**2)
                    * math.sqrt(math.factorial(n) / math.factorial(m))
                    * alpha ** (m - n)
                    * float(eval_genlaguerre(n, m - n, alpha**2))
                )
            else:
                pred = np.conjugate(
                    np.exp(-0.5 * alpha**2)
                    * math.sqrt(math.factorial(m) / math.factorial(n))
                    * (-alpha) ** (n - m)
                    * float(eval_genlaguerre(m, n - m, alpha**2))
                )
            lag_err = max(lag_err, abs(D[m, n] - pred))
    S.check(
        "T4_displacement",
        _max(coh[:20] - analytic) < 1e-12 and _max(np.abs(coh[:20]) ** 2 - pois) < 1e-12 and lag_err < 1e-10,
        f"coh {_max(coh[:20]-analytic):.2e} lag {lag_err:.2e}",
    )
    D7 = displacement_matrix(0.7, 48)
    Dm = displacement_matrix(-0.7, 48)
    D8 = displacement_matrix(0.5, 8, strict8=True)
    crop = displacement_matrix(0.5, 48)[:8, :8]
    S.check(
        "T5_inverse_and_not_crop",
        _max(D7 @ Dm - np.eye(48)) < 1e-10 and _max(D8 - crop) > 1e-3 and _max(D8.conj().T @ D8 - np.eye(8)) < 1e-10,
        f"crop gap {_max(D8-crop):.3e}",
    )
    # shell map
    map_err = 0.0
    for s in [(0.2, -0.4, 0.7), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0), (-0.3, 0.2, -0.1)]:
        c1, c2, c3, lam = s_to_walsh(*s)
        for p in range(4):
            map_err = max(map_err, abs(shell_phase_of_pop(p, *s) - (walsh_shell_phase(p, c1, c2, c3) + lam)))
    M = np.array([[-0.5, -0.5, -0.125], [0.0, 0.25, 0.125], [0.0, 0.0, -0.125]])
    S.check("T6_shell_map", map_err < 1e-12 and abs(np.linalg.det(M)) > 1e-6, f"err {map_err:.2e} det {np.linalg.det(M):.4f}")

    # T7 commute
    psi = (np.random.default_rng(2).normal(size=(8, 8)) + 1j * np.random.default_rng(3).normal(size=(8, 8))).astype(np.complex128)
    psi /= np.linalg.norm(psi)
    Uc = np.exp(-1j * 0.37 * (m8.dJ_phase + 0.2 * m8.dK_phase - 0.1 * m8.dM_phase))
    sph = _snap(m8, np.array([0.3, -0.2, 0.4]))
    Sdiag = sph.reshape(8, 1) * sph.reshape(1, 8)
    a = Sdiag * (Uc * psi)
    b = Uc * (Sdiag * psi)
    S.check("T7_commute", float(np.linalg.norm(a - b)) < 1e-12, f"{float(np.linalg.norm(a-b)):.2e}")

    # T8
    Hext = dense_hext(m10)
    ev = np.linalg.eigvalsh(Hext)
    Hlog = dense_H(inst.hc, inst.b, 6)
    perm = np.full(64, -1, dtype=np.int64)
    for flat, x in enumerate(m10.logical_map.reshape(-1)):
        if int(x) >= 0:
            perm[int(x)] = flat
    block_err = 0.0
    for i in range(64):
        for j in range(64):
            block_err = max(block_err, abs(Hext[perm[i], perm[j]] - Hlog[i, j]))
    pattern_ok = True
    pattern_min = []
    for mask0 in (0, 1):
        for mask1 in (0, 1):
            idx = []
            for flat in range(100):
                f0, f1 = divmod_fock(flat, 10)
                if (f0 >= 8) != bool(mask0) or (f1 >= 8) != bool(mask1):
                    continue
                idx.append(flat)
            mn = float(np.linalg.eigvalsh(Hext[np.ix_(idx, idx)])[0])
            pattern_min.append(mn)
            if mn < inst.E0 - 1e-8:
                pattern_ok = False
    S.check(
        "T8_Hext_bound",
        pattern_ok and ev[0] >= inst.E0 - 1e-8 and abs(ev[0] - inst.E0) < 1e-8 and block_err < 1e-8,
        f"min {ev[0]:.6f} E0 {inst.E0:.6f} block {block_err:.2e} patterns {min(pattern_min):.4f}",
    )

    # gray and G_R code blocks
    for rule, enc in (("G_R", "binary"), ("G_R", "gray"), ("inter_id", "gray")):
        groups_r = groups["G_R"][0] if rule == "G_R" else inter
        ops = groups["G_R"][1] if rule == "G_R" else groups["inter_id"][1]
        pk_r = packing_from(rule + enc, groups_r, enc, ops)
        mr = build_fock_model(inst, pk_r, 8, "Uz", False)
        Hr = dense_hext(mr)
        perm = np.full(64, -1, dtype=np.int64)
        for flat, x in enumerate(mr.logical_map.reshape(-1)):
            if int(x) >= 0:
                perm[int(x)] = flat
        err = 0.0
        for i in range(64):
            row = Hr[perm[i], perm] 
            err = max(err, float(np.max(np.abs(row - Hlog[i]))))
        embedded = _embed_gs(mr, inst.gs)
        ob = _observe(mr, embedded, np.zeros(1))
        S.check(f"P_perm_{rule}_{enc}", err < 1e-8 and abs(ob["F"] - 1.0) < 1e-10, f"block {err:.2e} F {ob['F']:.12f}")

    # wrong bit order must not look like the ground state
    pk_bad_groups = [[g[1], g[0], g[2]] for g in inter]
    pk_bad = packing_from("swapped", pk_bad_groups, "binary", {})
    mbad = build_fock_model(inst, pk_bad, 8, "Uz", False)
    wrong = _embed_gs(m16, inst.gs)  # embedded with inter map
    # overlap of inter-embedded state read with swapped logical map
    amp_wrong = _logical_amp(mbad, _embed_gs(m16, inst.gs)[:8, :8] if False else _embed_gs(build_fock_model(inst, pk, 8, "Uz", False), inst.gs))
    # Rebuild: take the correctly embedded fock state and decode with swapped roles.
    psi_right = _embed_gs(m8, inst.gs)
    amp_sw = _logical_amp(mbad, psi_right)
    F_sw = float(np.abs(np.vdot(inst.gs, amp_sw)) ** 2)
    S.check("P_swapped_bits_not_gs", F_sw < 0.95, f"F_swapped {F_sw:.4f}")

    # T9 edges
    G = generator(12, strict8=False)
    expect = {
        0: (1, 1.0),
        1: (3, np.sqrt(2)),
        2: (1, np.sqrt(3)),
        3: (7, 2.0),
        4: (1, np.sqrt(5)),
        5: (3, np.sqrt(6)),
        6: (1, np.sqrt(7)),
        7: (None, np.sqrt(8)),
    }
    edge_ok = True
    for f, (xor, weight) in expect.items():
        if abs(G[f + 1, f] - weight) > 1e-12:
            edge_ok = False
        if f < 7 and (f ^ (f + 1)) != xor:
            edge_ok = False
    # gray edge counts
    from gates import gray_bits

    counts = [0, 0, 0]
    for f in range(7):
        xor = gray_bits(f) ^ gray_bits(f + 1)
        for r in range(3):
            if (xor >> r) & 1:
                counts[r] += 1
    level7_pop = int(gray_bits(7) == 0b100)
    S.check("T9_edges", edge_ok and counts == [4, 2, 1] and level7_pop == 1, f"gray counts {counts}")

    # T10 T11
    x = nov9_x0(1, "E", None)
    x[3] = 0.0
    x[0] = 0.8
    x[4:] = 0.3
    z = evaluate_nov9(m16, x, 1)
    S.check("T10_alpha0", abs(z["E_diag"]) < 1e-10 and z["p_leak"] < 1e-12, f"diag {z['E_diag']:.2e} pleak {z['p_leak']:.2e}")
    xp = nov9_x0(1, "E", None)
    xp[0] = 0.55
    xp[4:] = 0.25
    xm = xp.copy()
    xm[0] = -0.55
    op = evaluate_nov9(m16, xp, 1)
    om = evaluate_nov9(m16, xm, 1)
    S.check("T11_gamma_even", abs(op["E_diag"] - om["E_diag"]) < 1e-10, f"diff {abs(op['E_diag']-om['E_diag']):.2e}")
    alphas = evaluate_nov9(m16, nov9_x0(3, "E", None), 3)["alpha"]
    S.check("P_echo_signs", np.allclose(alphas, [0.06, -0.06, 0.06]), str(alphas))

    # T12 H0
    psi = np.zeros((10, 10), dtype=np.complex128)
    psi[8, 0] = 1.0
    ob = _observe(m10, psi, np.zeros(1))
    h0 = H0_energy(m10, psi)
    S.check("T12_H0_lower", h0 < ob["E"] - 1e-8, f"H0 {h0:.4f} Hext {ob['E']:.4f}")

    # U0 vs Uz differ off code and agree on code
    mU0 = build_fock_model(inst, pk, 10, "U0", False)
    code_agree = _max(mU0.dJ_phase[:8, :8] - m10.dJ_phase[:8, :8])
    partial = abs(mU0.dJ_phase[8, 0] - m10.dJ_phase[8, 0])
    S.check("P_U0_vs_Uz", code_agree < 1e-12 and partial > 1e-8, f"code {code_agree:.2e} partial {partial:.3e}")

    # T14 QAOA
    xq = np.array([0.31, -0.22, 0.11, 0.42], dtype=np.float64)
    oq = evaluate_qaoa(inst, xq, 2, keep_state=True)

    def rx(beta):
        c, s = np.cos(beta), np.sin(beta)
        return np.array([[c, -1j * s], [-1j * s, c]], dtype=np.complex128)

    def mixer(beta, n=6):
        U = rx(beta)
        for _ in range(n - 1):
            U = np.kron(U, rx(beta))
        return U

    psi = plus_state(6)
    for ell in range(2):
        psi = psi * np.exp(-1j * float(xq[2 * ell]) * inst.hc)
        psi = mixer(float(xq[2 * ell + 1])) @ psi
    Hpsi = inst.hc * psi - inst.b * apply_X_sum(psi, 6)
    E_ref = float(np.vdot(psi, Hpsi).real)
    S.check(
        "T14_qaoa",
        float(np.linalg.norm(psi - oq["psi"])) < 1e-12 and abs(E_ref - oq["E"]) < 1e-12,
        f"state {float(np.linalg.norm(psi-oq['psi'])):.2e}",
    )
    xq0 = np.zeros(2)
    oq0 = evaluate_qaoa(inst, xq0, 1)
    S.check("T14_zero", abs(oq0["F"] - inst.F_plus) < 1e-12 and abs(oq0["E"] + 6) < 1e-10, f"{oq0['E']:.6f}")

    # W bit order
    W = hadamard3()
    Z0 = np.diag([1.0 - 2.0 * (i & 1) for i in range(8)])
    X0 = np.zeros((8, 8))
    for i in range(8):
        X0[i ^ 1, i] = 1.0
    S.check("P_W_lsb", _max(W @ Z0 @ W - X0) < 1e-12 and _max(W @ W - np.eye(8)) < 1e-12, "BinH")

    # T13 SPSA accounting and energy-only selection
    def bowl(x):
        e = float(np.sum((np.asarray(x) - 0.2) ** 2))
        return {"E": e, "norm": 1.0, "F": e}

    rngs = [np.random.default_rng(np.random.SeedSequence([7, 6, 10, 1, 2, s, 0])) for s in range(4)]
    x0s = [np.zeros(2) for _ in range(4)]
    res = run_spsa(bowl, x0s, np.full(2, 0.1), 1024, rngs)
    caps_ok = all(iterations_for_cap(c) == n for c, n in ((64, 3), (128, 11), (256, 27), (512, 59), (1024, 123)))
    S.check(
        "T13_queries",
        (not res.failed) and res.counter.total == 1024 and res.energy < 1e-3 and caps_ok and res.payload["F"] < 1e-3,
        f"E {res.energy:.3e} queries {res.counter.as_dict()} nit {res.n_iter}",
    )

    def bad(_x):
        return {"E": 0.0, "norm": 2.0, "F": 1.0}

    bad_res = run_spsa(bad, x0s, np.full(2, 0.1), 128, rngs)
    S.check("P_failure_scores_x0", bad_res.failed and bad_res.candidate == "failure_x0", bad_res.failure)

    # blind guard
    blocked = False
    try:
        build_instance(11000, 6, 1.0, pilot=True)
    except RuntimeError:
        blocked = True
    blocked_f = False
    try:
        build_instance(20000, 6, 1.0, pilot=False)
    except RuntimeError:
        blocked_f = True
    S.check("P_blind_seeds_blocked", blocked and blocked_f, "validation and final")
    approved = (
        pre["idea"] == "N9-GR"
        and pre["d2_d5_user_confirmed"] is True
        and pre["d2_d5_scope"] == "approved_simulation_assumptions_not_hardware"
        and pre["hardware_claim"] is False
        and pre["freeze"] is False
        and pre["validation_touched"] is False
        and pre["final_touched"] is False
        and pre["approval"]["when"] == "2026-11-09 18:41 Asia/Shanghai"
    )
    S.check("P_prereg_loaded", approved, pre["_sha256"][:16])
    # strict logical has no leak
    ms = build_fock_model(inst, pk, 16, "Uz", True)
    xs = nov9_x0(2, "E", None)
    os_ = evaluate_nov9(ms, xs, 2)
    S.check(
        "P_strict_noleak",
        ms.nf == 8 and ms.gate_name.startswith("D8") and os_["p_leak"] < 1e-12,
        ms.gate_name,
    )
    return S


def divmod_fock(flat: int, nf: int):
    f1 = flat % nf
    f0 = flat // nf
    return f0, f1


def main():
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    try:
        suite = run()
    except Exception:
        traceback.print_exc()
        raise
    payload = {"all_pass": suite.ok, "tests": suite.rows, "prereg_sha256": load_prereg()["_sha256"]}
    (RESULT_DIR / "conformance.json").write_text(json.dumps(payload, indent=2) + "\n")
    print("ALL PASS" if suite.ok else "CONFORMANCE FAILED")
    return 0 if suite.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
