"""Bounded ED reproduction of Gupta et al. arXiv:2606.18339v1 Fig. 2 at L=12, Gamma=1.

H = -sum_{i<j} J_ij Z_i Z_j - Gamma sum_i X_i, J_ij ~ N(0, 1/L). See PREREG_GUPTA_FIG2.md.
Construction A: Kronecker-product dense + full eigh. Construction B: bitwise parity-block eigsh.
No ADAPT, no SPSA, no scan.
"""
from __future__ import annotations

import csv
import hashlib
import json
import platform
import resource
import time
from pathlib import Path

import numpy as np
import scipy
import scipy.sparse as sp
from scipy import stats
from scipy.sparse.linalg import eigsh

from Nov3_INIT.gapdiag import block, clusters, embed, full_sparse, parity_sparse
from Nov9_SK_SPSA.experiments.FT.instance import make_J, pair_diag, z_table

ROOT = Path(__file__).resolve().parent
PREREG = ROOT / "PREREG_GUPTA_FIG2.md"
RES = ROOT / "results" / "gupta_fig2"
L = 12
GAMMA = 1.0
SEEDS = tuple(range(10920, 10950))
TAU = 1e-8
K = 8

I2 = np.eye(2)
X2 = np.array([[0.0, 1.0], [1.0, 0.0]])
Z2 = np.array([1.0, -1.0])


def site_vec(q: int, v: np.ndarray) -> np.ndarray:
    """Diagonal of a single-site operator at qubit q; qubit q is bit q of the basis index."""
    left = np.ones(1 << (L - 1 - q))
    right = np.ones(1 << q)
    return np.kron(np.kron(left, v), right)


def site_mat(q: int, m: np.ndarray) -> sp.csr_matrix:
    return sp.kron(sp.kron(sp.identity(1 << (L - 1 - q)), sp.csr_matrix(m)), sp.identity(1 << q)).tocsr()


def H_kron(J: np.ndarray, gamma: float) -> np.ndarray:
    zs = [site_vec(q, Z2) for q in range(L)]
    diag = np.zeros(1 << L)
    for i in range(L):
        for j in range(i + 1, L):
            diag -= J[i, j] * zs[i] * zs[j]
    H = np.diag(diag)
    for q in range(L):
        H -= gamma * site_mat(q, X2).toarray()
    return H


def P_kron() -> np.ndarray:
    P = sp.identity(1, format="csr")
    for _ in range(L):
        P = sp.kron(P, sp.csr_matrix(X2)).tocsr()
    return P.toarray()


def sumX_kron() -> np.ndarray:
    S = np.zeros((1 << L, 1 << L))
    for q in range(L):
        S += site_mat(q, X2).toarray()
    return S


def dense_A(J: np.ndarray, P: np.ndarray, SX: np.ndarray) -> dict:
    H = H_kron(J, GAMMA)
    w, U = np.linalg.eigh(H)
    for cl in clusters(w):
        if len(cl) > 1:
            sub = U[:, cl]
            _pw, pv = np.linalg.eigh(sub.T @ P @ sub)
            U[:, cl] = sub @ pv
    nlow = 64
    Ul = U[:, :nlow]
    par = np.einsum("ij,ij->j", Ul, P @ Ul)
    res = float(np.max(np.linalg.norm(H @ Ul - Ul * w[None, :nlow], axis=0)))
    comm = float(np.linalg.norm(H @ P - P @ H))
    g0 = Ul[:, 0]
    sx0 = SX @ g0
    coup = np.abs(Ul.T @ sx0)
    s0 = 1 if par[0] > 0 else -1
    same = [i for i in range(nlow) if par[i] * s0 > 0.5]
    opp = [i for i in range(nlow) if par[i] * s0 < -0.5]
    plus = [i for i in range(nlow) if par[i] > 0.5]
    minus = [i for i in range(nlow) if par[i] < -0.5]
    coupled = next((i for i in same[1:] if coup[i] > TAU), None)
    return {
        "E0": float(w[0]), "E1": float(w[1]), "E2": float(w[2]),
        "full_gap": float(w[1] - w[0]),
        "par0": float(par[0]), "par1": float(par[1]),
        "E0_plus": float(w[plus[0]]), "E1_plus": float(w[plus[1]]),
        "E0_minus": float(w[minus[0]]), "E1_minus": float(w[minus[1]]),
        "Delta_plus": float(w[plus[1]] - w[plus[0]]),
        "Delta_minus": float(w[minus[1]] - w[minus[0]]),
        "cross_split": float(w[minus[0]] - w[plus[0]]),
        "c1_same": float(coup[same[1]]),
        "coupled_gap": float(w[coupled] - w[0]) if coupled is not None else None,
        "coupled_is_first_same": bool(coupled == same[1]),
        "max_opp_coupling": float(max(coup[i] for i in opp)),
        "residual_A": res, "commutator_A": comm,
        "low8": [(float(w[i]), "+" if par[i] > 0 else "-") for i in range(8)],
        "gs_min_amp_signfixed": float(np.min(g0 * np.sign(g0.sum()))),
    }


def sparse_B(J: np.ndarray, Jscale: float = 1.0, gamma: float = GAMMA) -> dict:
    hc = pair_diag(-Jscale * J, z_table(L))
    out = {}
    res = []
    Hf = full_sparse(hc, gamma, L)
    for s, key in ((1, "plus"), (-1, "minus")):
        Bm = block(hc, gamma, L, s)
        v0 = np.ones(Bm.shape[0]) / np.sqrt(Bm.shape[0])
        w, u = eigsh(Bm, k=K, which="SA", tol=1e-12, v0=v0, maxiter=200000)
        o = np.argsort(w)
        w, u = w[o], u[:, o]
        res.append(float(np.max(np.linalg.norm(Hf @ embed(u, L, s) - embed(u, L, s) * w[None, :], axis=0))))
        out[f"w_{key}"] = w
    v0 = np.ones(Hf.shape[0]) / np.sqrt(Hf.shape[0])
    wf = np.sort(eigsh(Hf, k=K, which="SA", tol=1e-12, v0=v0, maxiter=200000)[0])
    P = parity_sparse(L)
    union = np.sort(np.concatenate([out["w_plus"], out["w_minus"]]))[:K]
    E0 = float(min(out["w_plus"][0], out["w_minus"][0]))
    return {
        "E0": E0, "E1": float(union[1]), "full_gap": float(union[1] - union[0]),
        "Delta_plus": float(out["w_plus"][1] - out["w_plus"][0]),
        "E1_parity": "+" if np.isclose(union[1], out["w_plus"][1]) and not np.isclose(union[1], out["w_minus"][0]) else "-",
        "residual_B": float(max(res)),
        "commutator_B": float(sp.linalg.norm(Hf @ P - P @ Hf)),
        "full_eigsh_vs_union": float(np.max(np.abs(wf - union))),
    }


def welch(a: np.ndarray, b: np.ndarray) -> dict:
    a, b = np.asarray(a, float), np.asarray(b, float)
    va, vb = a.var(ddof=1) / a.size, b.var(ddof=1) / b.size
    d = a.mean() - b.mean()
    se = np.sqrt(va + vb)
    df = (va + vb) ** 2 / (va ** 2 / (a.size - 1) + vb ** 2 / (b.size - 1))
    t = stats.t.ppf(0.975, df)
    p = float(stats.ttest_ind(a, b, equal_var=False).pvalue)
    ks = stats.ks_2samp(a, b)
    return {"diff": float(d), "se": float(se), "df": float(df), "ci95": [float(d - t * se), float(d + t * se)],
            "includes_0": bool(d - t * se <= 0.0 <= d + t * se), "welch_p": p,
            "ks_stat": float(ks.statistic), "ks_p": float(ks.pvalue)}


def desc(v) -> dict:
    v = np.asarray(v, float)
    sem = v.std(ddof=1) / np.sqrt(v.size)
    t = stats.t.ppf(0.975, v.size - 1)
    return {"n": int(v.size), "mean": float(v.mean()), "sd": float(v.std(ddof=1)), "sem": float(sem),
            "ci95": [float(v.mean() - t * sem), float(v.mean() + t * sem)],
            "median": float(np.median(v)), "min": float(v.min()), "max": float(v.max())}


def main() -> None:
    RES.mkdir(parents=True, exist_ok=True)
    prereg_sha = hashlib.sha256(PREREG.read_bytes()).hexdigest()
    ref = json.loads((RES / "fig2_L12_digitized.json").read_text())
    t0 = time.perf_counter()
    P = P_kron()
    SX = sumX_kron()
    rows = []
    for s in SEEDS:
        ts = time.perf_counter()
        J = make_J(s, L)
        a = dense_A(J, P, SX)
        b = sparse_B(J)
        s1 = sparse_B(J, Jscale=0.25, gamma=0.5 * GAMMA)
        s2 = sparse_B(J, Jscale=2.0, gamma=GAMMA)
        u = J[np.triu_indices(L, 1)]
        row = {
            "seed": s, "L": L, "Gamma": GAMMA,
            "J_var_times_L": float(np.var(u, ddof=0) * L),
            "E0_over_L": a["E0"] / L, **{k: v for k, v in a.items() if k != "low8"},
            "low8": " ".join(f"{e:.10f}{p}" for e, p in a["low8"]),
            "E1_parity": "+" if a["par1"] > 0 else "-",
            "ground_parity": "+" if a["par0"] > 0 else "-",
            "B_E0": b["E0"], "B_full_gap": b["full_gap"], "B_Delta_plus": b["Delta_plus"],
            "A_vs_B_E0": abs(a["E0"] - b["E0"]), "A_vs_B_full_gap": abs(a["full_gap"] - b["full_gap"]),
            "A_vs_B_Delta_plus": abs(a["Delta_plus"] - b["Delta_plus"]),
            "residual_B": b["residual_B"], "commutator_B": b["commutator_B"],
            "B_full_eigsh_vs_union": b["full_eigsh_vs_union"],
            "S1_spinhalf_E0_over_L": s1["E0"] / L, "S1_full_gap": s1["full_gap"], "S1_Delta_plus": s1["Delta_plus"],
            "S2_doublesum_E0_over_L": s2["E0"] / L, "S2_full_gap": s2["full_gap"], "S2_Delta_plus": s2["Delta_plus"],
            "seconds": time.perf_counter() - ts,
        }
        rows.append(row)
        print(f"s={s} E0/L={row['E0_over_L']:.5f} gap={row['full_gap']:.4f} P0={row['ground_parity']} "
              f"P1={row['E1_parity']} D+={row['Delta_plus']:.4f} c1={row['c1_same']:.3f} "
              f"res={row['residual_A']:.1e}/{row['residual_B']:.1e} AvsB={row['A_vs_B_E0']:.1e} {row['seconds']:.1f}s",
              flush=True)
    wall = time.perf_counter() - t0
    keys = list(rows[0].keys())
    with (RES / "instances.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    col = lambda k: np.array([r[k] for r in rows], float)
    fig_E = np.array(ref["E_per_L"])
    fig_G = np.array(ref["gap"])
    comp = {
        "E0_over_L": {"ours": desc(col("E0_over_L")), "fig2": desc(fig_E), "welch": welch(col("E0_over_L"), fig_E)},
        "full_gap": {"ours": desc(col("full_gap")), "fig2": desc(fig_G), "welch": welch(col("full_gap"), fig_G)},
        "Delta_plus_vs_fig_gap": {"ours": desc(col("Delta_plus")), "fig2": desc(fig_G), "welch": welch(col("Delta_plus"), fig_G)},
        "S1_spinhalf_E0_over_L": {"ours": desc(col("S1_spinhalf_E0_over_L")), "welch": welch(col("S1_spinhalf_E0_over_L"), fig_E)},
        "S1_spinhalf_full_gap": {"ours": desc(col("S1_full_gap")), "welch": welch(col("S1_full_gap"), fig_G)},
        "S2_doublesum_E0_over_L": {"ours": desc(col("S2_doublesum_E0_over_L")), "welch": welch(col("S2_doublesum_E0_over_L"), fig_E)},
        "S2_doublesum_full_gap": {"ours": desc(col("S2_full_gap")), "welch": welch(col("S2_full_gap"), fig_G)},
    }
    verdict = {
        "E0_over_L": "AGREE" if comp["E0_over_L"]["welch"]["includes_0"] else "TENSION",
        "full_gap": "AGREE" if comp["full_gap"]["welch"]["includes_0"] else "TENSION",
        "Delta_plus_vs_fig_gap": "AGREE" if comp["Delta_plus_vs_fig_gap"]["welch"]["includes_0"] else "TENSION",
    }
    g_full, g_plus = verdict["full_gap"] == "AGREE", verdict["Delta_plus_vs_fig_gap"] == "AGREE"
    verdict["gap_definition"] = ("consistent with full gap E1-E0 only" if g_full and not g_plus else
                                 "consistent with Delta_plus only" if g_plus and not g_full else "not resolved")
    payload = {
        "paper": "Gupta et al., arXiv:2606.18339v1, Fig. 2 (sk.pdf), L=12, Gamma=1",
        "hamiltonian_primary": "H = -sum_{i<j} J_ij Z_i Z_j - Gamma sum_i X_i; J_ij ~ N(0, 1/L); Pauli; single sum",
        "prereg_sha256": prereg_sha, "seeds": list(SEEDS), "tau": TAU,
        "n_ground_plus": int(sum(r["ground_parity"] == "+" for r in rows)),
        "n_E1_odd": int(sum(r["E1_parity"] == "-" for r in rows)),
        "n_coupled_is_first_same": int(sum(r["coupled_is_first_same"] for r in rows)),
        "max_residual_A": float(col("residual_A").max()), "max_residual_B": float(col("residual_B").max()),
        "max_commutator_A": float(col("commutator_A").max()), "max_commutator_B": float(col("commutator_B").max()),
        "max_A_vs_B": float(max(col("A_vs_B_E0").max(), col("A_vs_B_full_gap").max(), col("A_vs_B_Delta_plus").max())),
        "max_B_full_vs_union": float(col("B_full_eigsh_vs_union").max()),
        "max_opp_coupling": float(col("max_opp_coupling").max()),
        "min_c1_same": float(col("c1_same").min()),
        "Delta_plus": desc(col("Delta_plus")), "Delta_minus": desc(col("Delta_minus")),
        "cross_split": desc(col("cross_split")), "coupled_gap": desc(col("coupled_gap")),
        "c1_same": desc(col("c1_same")),
        "comparison": comp, "verdict": verdict,
        "wall_seconds": wall, "peak_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "env": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
        "quantum_energy_queries": 0, "no_adapt": True, "no_scan": True,
    }
    (RES / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({k: payload[k] for k in ("verdict", "n_ground_plus", "n_E1_odd", "n_coupled_is_first_same",
                                               "max_residual_A", "max_residual_B", "max_commutator_A",
                                               "max_commutator_B", "max_A_vs_B", "max_B_full_vs_union",
                                               "max_opp_coupling", "min_c1_same", "wall_seconds")}, indent=1))
    for k, v in comp.items():
        o = v["ours"]
        wv = v["welch"]
        print(f"{k}: ours {o['mean']:.5f} ± {o['sem']:.5f} [{o['min']:.4f},{o['max']:.4f}]  "
              f"diff {wv['diff']:+.5f} CI [{wv['ci95'][0]:+.5f},{wv['ci95'][1]:+.5f}] p={wv['welch_p']:.3g} KSp={wv['ks_p']:.3g}")


if __name__ == "__main__":
    main()
