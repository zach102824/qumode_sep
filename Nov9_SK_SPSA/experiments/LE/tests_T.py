"""Conformance tests T1-T14 (+T15 independent qubit-space strict-logical check, T16 bit order).

Run: python -m Nov9_SK_SPSA.experiments.LE.tests_T  (from repo root)
Writes results/LE/TESTS.json.
"""
from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from scipy.linalg import expm
from scipy.special import eval_genlaguerre

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import core as C  # noqa: E402

OUT = HERE.parents[1] / "results" / "LE"
RES = {}


def rec(name, val, tol, ok=None):
    ok = (val <= tol) if ok is None else ok
    RES[name] = {"value": float(val), "tol": tol, "pass": bool(ok)}
    print(f"{name:40s} {val:.3e}  tol {tol:g}  {'PASS' if ok else 'FAIL'}", flush=True)


def T1():
    J1, J2 = C.make_J(10000, 9), C.make_J(10000, 9)
    rec("T1_J_regen_sha", 0.0 if C.J_sha(J1) == C.J_sha(J2) else 1.0, 0.0)
    v = np.concatenate([C.upper(C.make_J(s, 12)) for s in range(10000, 10032)])
    rec("T1_J_variance_x_N_minus_1", abs(np.var(v) * 12 - 1.0), 0.15)


def T2():
    for N in (6, 9):
        g = C.exact_gs(C.make_J(10000, N), 1.0)
        rec(f"T2_gs_resid_N{N}", g["resid"], 1e-9)
        rec(f"T2_gs_positive_N{N}", float(-g["gs"].min()), 1e-12)


def T3():
    """Nov3rd zero->[W e^{-i g E} W -> S]^L -> W  ==  plus->[UcZ -> W S W]^L (qubit space, N=6)."""
    N, L = 6, 3
    rng = np.random.default_rng(1)
    J = C.make_J(10000, N)
    hzz = C.hzz_diag(J, C.z_table(N))
    m = N // 3
    dim = 1 << N
    x = np.arange(dim)
    pc = np.zeros(dim, dtype=int)
    cav_pop = []
    for c in range(m):
        f = np.zeros(dim, dtype=int)
        for r in range(3):
            f |= ((x >> (c + r * m)) & 1) << r
        cav_pop.append(np.array([bin(v).count("1") for v in f]))
    H1 = np.array([[1, 1], [1, -1]]) / math.sqrt(2)
    W = np.array([[1.0]])
    for _ in range(N):
        W = np.kron(H1, W)
    gam = rng.normal(size=L)
    ph = rng.normal(size=(L, 4))
    psi = np.zeros(dim, complex)
    psi[0] = 1
    for ell in range(L):
        psi = W @ (np.exp(-1j * gam[ell] * hzz) * (W @ psi))
        psi = psi * np.exp(-1j * sum(ph[ell][p] for p in cav_pop))
    psi = W @ psi
    phi = np.full(dim, 2 ** (-N / 2), complex)
    for ell in range(L):
        phi = np.exp(-1j * gam[ell] * hzz) * phi
        phi = W @ (np.exp(-1j * sum(ph[ell][p] for p in cav_pop)) * (W @ phi))
    rec("T3_nov3rd_basis_rewrite", np.max(np.abs(psi - phi)), 1e-12)


def T4():
    nf = 48
    Dm = C.Disp(nf)
    for a in (0.1, 0.3, 0.9):
        D = Dm(a)
        n = np.arange(nf)
        coh = np.exp(-a * a / 2) * np.array([a ** k / math.sqrt(math.factorial(k)) for k in n])
        rec(f"T4_coherent_a{a}", np.max(np.abs(D[:, 0] - coh)), 1e-12)
        err = 0.0
        for mm in range(8):
            for nn in range(8):
                if mm >= nn:
                    v = math.sqrt(math.factorial(nn) / math.factorial(mm)) * a ** (mm - nn) * math.exp(-a * a / 2) * eval_genlaguerre(nn, mm - nn, a * a)
                else:
                    v = math.sqrt(math.factorial(mm) / math.factorial(nn)) * (-a) ** (nn - mm) * math.exp(-a * a / 2) * eval_genlaguerre(mm, nn - mm, a * a)
                err = max(err, abs(D[mm, nn] - v))
        rec(f"T4_laguerre8x8_a{a}", err, 1e-12)
    # cropped-8x8 exponential is NOT the displacement
    D8 = C.Disp(8, strict=True)(0.3)
    rec("T4_strict_D8_differs_from_trueD", -np.max(np.abs(D8 - Dm(0.3)[:8, :8])), -1e-3)


def T5():
    Dm = C.Disp(48)
    P = Dm(0.3) @ Dm(-0.3)
    rec("T5_D_Dminus_identity_on_code", np.max(np.abs(P[:8, :8] - np.eye(8))), 1e-12)


def T6():
    p = np.arange(4)
    t = np.stack([3 - 2 * p, np.array([3, -1, -1, 3]), np.array([1, -1, 1, -1])])
    rng = np.random.default_rng(2)
    cvec = rng.normal(size=3)
    target = cvec @ t
    target = target - target[0]
    Amat = np.stack([p, p * (p - 1) / 2.0, (p == 3).astype(float)], axis=1)
    s = np.linalg.solve(Amat[1:], target[1:])
    rec("T6_shell_s_to_t_map", np.max(np.abs(Amat @ s - target)), 1e-12)
    rec("T6_map_invertible_cond", np.linalg.cond(Amat[1:]), 1e3)


def T7_T10_T11_T16():
    J = C.make_J(10001, 6)
    rng = np.random.default_rng(3)
    for ext in ("Uz", "U0"):
        sim = C.Nov9(J, 1.0, 2, nf=16, ext=ext)
        psi = rng.normal(size=sim.shape) + 1j * rng.normal(size=sim.shape)
        diag = sim.G[0] + 0.3 * sim.G[1]
        U = np.exp(-1j * 0.4 * diag)
        S = sim._outer_prod([np.exp(-1j * C.shell_phase([0.2, -0.3, 0.5], sim.nf))] * sim.C)
        rec(f"T7_commute_UcZ_S_{ext}", np.max(np.abs(U * (S * psi) - S * (U * psi))), 1e-14)
    sim = C.Nov9(J, 1.0, 2, nf=24)
    x = rng.normal(size=sim.nparam()) * 0.3
    x[1 + 2 : 1 + 2 + 2] = 0.0  # u=0 -> alpha=0
    psi = sim.state(x)
    prob = np.abs(psi) ** 2
    rec("T10_alpha0_ZZ_zero", abs(float(np.sum(prob * sim.Ezz))), 1e-12)
    rec("T10_alpha0_noleak", 1 - float(np.sum(prob[sim.sl])), 1e-12)
    sim1 = C.Nov9(J, 1.0, 1, nf=24)
    x = np.array([0.37, 0.2, -0.4, 0.5, 0.3, -0.2, 0.4])
    xm = x.copy()
    xm[0] = -x[0]
    e1 = float(np.sum(np.abs(sim1.state(x)) ** 2 * sim1.Ezz))
    e2 = float(np.sum(np.abs(sim1.state(xm)) ** 2 * sim1.Ezz))
    rec("T11_L1_ZZ_even_in_gamma", abs(e1 - e2), 1e-12)
    # T16 bit order / code projector: code-block Ezz equals qubit hzz under code_to_x; plus F
    g = C.exact_gs(J, 1.0)
    rec("T16_code_bitorder_Ezz", np.max(np.abs(sim.Ezz[sim.sl] - g["hzz"][sim.code_to_x])), 1e-12)
    z = np.zeros(sim.nparam())
    d = sim.diagnostics(z, g["gs"])
    rec("T16_plus_F_matches", abs(d["F"] - C.plus_fidelity(g["gs"])), 1e-12)
    rec("T16_code_to_x_bijective", 0.0 if np.unique(sim.code_to_x).size == 64 else 1.0, 0.0)


def _hext_matrix(sim):
    dim = int(np.prod(sim.shape))
    H = np.zeros((dim, dim))
    for k in range(dim):
        e = np.zeros(dim)
        e[k] = 1.0
        psi = e.reshape(sim.shape)
        out = sim.Ezz * psi
        lk = np.zeros(sim.shape)
        for c in range(sim.C):
            lk = lk + 3.0 * sim._bcast(1.0 - sim.cm, c)
        out = out + sim.b * lk * psi
        for c in range(sim.C):
            for r in range(3):
                perm = np.arange(sim.nf)
                perm[:8] = np.arange(8) ^ (1 << r)
                Xp = np.take(psi, perm, axis=c) * sim._bcast(sim.cm, c)
                out = out - sim.b * Xp
        H[:, k] = out.reshape(-1)
    return H


def T8_T12():
    for seed in (10000, 10001, 10002):
        J = C.make_J(seed, 6)
        for b in (1.0, 1.5):
            g = C.exact_gs(J, b)
            sim = C.Nov9(J, b, 1, nf=10)
            H = _hext_matrix(sim)
            rec(f"T8_Hext_sym_s{seed}_b{b}", np.max(np.abs(H - H.T)), 1e-12)
            w = np.linalg.eigvalsh(H)
            rec(f"T8_Hext_min_eq_E0_s{seed}_b{b}", abs(w[0] - g["E0"]), 1e-10)
            # leaked blocks strictly above E0 (b>0): min over states with any leak
            leakmask = (1 - sim.allcode).reshape(-1) > 0
            Hl = H[np.ix_(leakmask, leakmask)]
            wl = np.linalg.eigvalsh(Hl)
            rec(f"T8_leak_block_gap_s{seed}_b{b}", -(wl[0] - g["E0"]), -1e-6)
    # T12: H0 (leak scored 0) gives lower energy than H_ext at a leaky point
    J = C.make_J(10000, 6)
    sim = C.Nov9(J, 1.0, 3, nf=24)
    g = C.exact_gs(J, 1.0)
    x = np.zeros(sim.nparam())
    x[3:6] = np.arctanh(0.29 / C.ALPHA_MAX)
    d = sim.diagnostics(x, g["gs"])
    rec("T12_H0_below_Hext_at_leaky_point", -(d["E"] - d["E_H0"]), -1e-3)
    RES["T12_detail"] = {"E_ext": d["E"], "E_H0": d["E_H0"], "p_leak": d["p_leak"]}


def T9():
    edges = {}
    for f in range(8):
        flip = f ^ (f + 1)
        edges[f"{f}-{f+1}"] = [r for r in range(4) if (flip >> r) & 1]
    expect = {"0-1": [0], "1-2": [0, 1], "2-3": [0], "3-4": [0, 1, 2], "4-5": [0], "5-6": [0, 1], "6-7": [0], "7-8": [0, 1, 2, 3]}
    rec("T9_binary_edge_table", 0.0 if edges == expect else 1.0, 0.0)
    RES["T9_edges"] = edges
    G = C.ladder_gen(48)
    nz = [(i, j) for i in range(8) for j in range(8) if abs(G[i, j]) > 0]
    rec("T9_generator_tridiagonal_on_code", 0.0 if all(abs(i - j) == 1 for i, j in nz) else 1.0, 0.0)


def T13():
    J = C.make_J(10000, 6)
    sim = C.Nov9(J, 1.0, 2, nf=16)
    rng = np.random.default_rng(5)
    starts = [rng.normal(size=sim.nparam()) * 0.1 for _ in range(4)]
    r = C.spsa_run(sim.energy, starts, sim.sigma(), lambda s: np.random.default_rng(s), 1024)
    rec("T13_query_count_1024", abs(r["queries"] - 1024), 0)
    r = C.spsa_run(sim.energy, starts, sim.sigma(), lambda s: np.random.default_rng(s), 128)
    rec("T13_query_count_128", abs(r["queries"] - 128), 0)


def T14():
    N, p = 6, 3
    J = C.make_J(10002, N)
    q = C.QAOA(J, 1.0, p)
    rng = np.random.default_rng(6)
    x = rng.normal(size=2 * p)
    dim = 1 << N
    X1 = np.array([[0, 1], [1, 0]])
    SX = np.zeros((dim, dim))
    for i in range(N):
        op = np.array([[1.0]])
        for k in range(N):
            op = np.kron(X1 if k == i else np.eye(2), op)
        SX += op
    Hzz = np.diag(q.hzz)
    psi = np.full(dim, 2 ** (-N / 2), complex)
    for ell in range(p):
        psi = expm(-1j * x[p + ell] * SX) @ (expm(-1j * x[ell] * Hzz) @ psi)
    H = Hzz - 1.0 * SX
    e_ref = float(np.real(psi.conj() @ H @ psi))
    rec("T14_qaoa_state", np.max(np.abs(psi - q.state(x))), 1e-12)
    rec("T14_qaoa_energy", abs(e_ref - q.energy(x)), 1e-12)


def T15():
    """Strict-logical Nov9 tensor sim vs independent qubit-index construction (N=6, L=2)."""
    N, L = 6, 2
    J = C.make_J(10003, N)
    sim = C.Nov9(J, 1.0, L, strict=True)
    rng = np.random.default_rng(7)
    x = rng.normal(size=sim.nparam()) * 0.4
    gamma, theta, alpha, s = sim.unpack(x)
    A = gamma * (J + sum(t * F for t, F in zip(theta, sim.Fhat)))
    z = C.z_table(N)
    dA = C.hzz_diag(A, z)
    dim = 1 << N
    xs = np.arange(dim)
    m = N // 3
    psi = np.full(dim, 2 ** (-N / 2), complex)
    Dm = C.Disp(8, strict=True)
    for ell in range(L):
        psi = np.exp(-1j * dA) * psi
        D = Dm(alpha[ell])
        for c in range(m):
            f = np.zeros(dim, dtype=int)
            for r in range(3):
                f |= ((xs >> (c + r * m)) & 1) << r
            base = xs.copy()
            for r in range(3):
                base &= ~(1 << (c + r * m))
            new = np.zeros(dim, complex)
            for fp in range(8):
                src = base.copy()
                for r in range(3):
                    src |= ((fp >> r) & 1) << (c + r * m)
                new += D[f, fp] * psi[src]
            psi = new
        th = C.shell_phase(s[ell], 8)
        for c in range(m):
            f = np.zeros(dim, dtype=int)
            for r in range(3):
                f |= ((xs >> (c + r * m)) & 1) << r
            psi = psi * np.exp(-1j * th[f])
    t = sim.state(x)[sim.sl]
    ref = psi[sim.code_to_x]
    rec("T15_strict_vs_qubit_state", np.max(np.abs(t - ref)), 1e-12)
    q = C.QAOA(J, 1.0, 1)
    rec("T15_strict_vs_qubit_energy", abs(sim.energy(x) - q.energy_of_state(psi)), 1e-12)


def main():
    t0 = time.time()
    for fn in (T1, T2, T3, T4, T5, T6, T7_T10_T11_T16, T8_T12, T9, T13, T14, T15):
        fn()
    ok = all(v.get("pass", True) for v in RES.values() if isinstance(v, dict) and "pass" in v)
    RES["_all_pass"] = ok
    RES["_seconds"] = time.time() - t0
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "TESTS.json").write_text(json.dumps(RES, indent=1))
    print("ALL PASS" if ok else "SOME FAIL", f"{time.time()-t0:.1f}s")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
