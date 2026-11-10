"""Bounded spectral-gap diagnostic. N=6 dense, N=12 parity-block sparse. No SPSA, no ansatz.

See PREREG_GAPDIAG.md. H = sum_{i<j} J_ij Z_i Z_j - g sum_i X_i, J_ij ~ N(0, 1/N) (J=1).
Drive assumed: transverse ramp, dH/dg = V = -sum_i X_i, which commutes with P = prod_i X_i.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import resource
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np
import scipy
import scipy.sparse as sp
from scipy.optimize import minimize_scalar
from scipy.sparse.linalg import eigsh

from Nov3_INIT.bstar import _ground, apply_x_sum, entropy_nats
from Nov9_SK_SPSA.experiments.FT.instance import make_J, pair_diag, z_table

ROOT = Path(__file__).resolve().parent
PREREG = ROOT / "PREREG_GAPDIAG.md"
RES = ROOT / "results" / "gapdiag"
PANEL = {6: tuple(range(10904, 10912)), 12: tuple(range(10912, 10920))}
G_FIXED = 1.5
TAUS = (1e-10, 1e-8, 1e-6, 1e-4)
TAU_PRIMARY = 1e-8
K_LADDER = (8, 16, 32, 64)
CLUSTER_TOL = 1e-9
N_LOW = 8


def peak_only(J: np.ndarray, n: int) -> dict:
    """bstar.maximize entropy recipe without its gap/parity contract (no gap-based selection)."""
    grid = np.linspace(0.05, 3.0, 60)
    hc = pair_diag(J, z_table(n))
    values = [entropy_nats(_ground(hc, float(b), n), n) for b in grid]
    k = int(np.argmax(values))
    if not (0 < k < len(grid) - 1):
        return {"interior": False, "grid_argmax_index": k, "Bstar": None}
    bracket = (float(grid[k - 1]), float(grid[k + 1]))

    def neg(b):
        return -entropy_nats(_ground(hc, float(b), n), n)

    refined = minimize_scalar(neg, bounds=bracket, method="bounded", options={"xatol": 1e-6})
    return {
        "interior": True,
        "grid_argmax_index": k,
        "Bstar": float(refined.x),
        "S_A_at_Bstar": float(-refined.fun),
        "S_A_grid_max": float(values[k]),
        "cut": list(range(n // 2)),
    }


def full_sparse(hc: np.ndarray, g: float, n: int) -> sp.csr_matrix:
    dim = 1 << n
    idx = np.arange(dim)
    rows, cols = [], []
    for q in range(n):
        rows.append(idx)
        cols.append(idx ^ (1 << q))
    rows = np.concatenate(rows)
    cols = np.concatenate(cols)
    X = sp.csr_matrix((np.ones(rows.size), (rows, cols)), shape=(dim, dim))
    return (sp.diags(hc) - g * X).tocsr()


def parity_sparse(n: int) -> sp.csr_matrix:
    dim = 1 << n
    idx = np.arange(dim)
    return sp.csr_matrix((np.ones(dim), (idx, idx ^ (dim - 1))), shape=(dim, dim))


def block(hc: np.ndarray, g: float, n: int, sign: int) -> sp.csr_matrix:
    half = 1 << (n - 1)
    low = half - 1
    idx = np.arange(half)
    rows, cols, vals = [], [], []
    for q in range(n - 1):
        rows.append(idx)
        cols.append(idx ^ (1 << q))
        vals.append(np.ones(half))
    rows.append(idx)
    cols.append(idx ^ low)
    vals.append(np.full(half, float(sign)))
    A = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(half, half))
    return (sp.diags(hc[:half]) - g * A).tocsr()


def embed(u: np.ndarray, n: int, sign: int) -> np.ndarray:
    dim = 1 << n
    half = dim >> 1
    v = np.zeros((dim,) + u.shape[1:], dtype=np.float64)
    v[:half] = u / np.sqrt(2.0)
    v[(dim - 1) ^ np.arange(half)] = sign * u / np.sqrt(2.0)
    return v


def V_apply(v: np.ndarray, n: int) -> np.ndarray:
    if v.ndim == 1:
        return -apply_x_sum(v, n)
    return np.stack([-apply_x_sum(v[:, i], n) for i in range(v.shape[1])], axis=1)


def clusters(w: np.ndarray, tol: float = CLUSTER_TOL) -> list[list[int]]:
    out = [[0]]
    for i in range(1, len(w)):
        if w[i] - w[out[-1][-1]] < tol:
            out[-1].append(i)
        else:
            out.append([i])
    return out


def sector_list(w: np.ndarray, vecs_full: np.ndarray, sign: int, ground_vec: np.ndarray, n: int, e_ref: float) -> list[dict]:
    """Cluster-level couplings |Pi_cluster V|0>| computed in full space."""
    vg = V_apply(ground_vec, n)
    rows = []
    for cl in clusters(w):
        sub = vecs_full[:, cl]
        amp = sub.T @ vg
        rows.append({
            "sector": "+" if sign > 0 else "-",
            "E": float(w[cl[0]]),
            "E_minus_E0": float(w[cl[0]] - e_ref),
            "degeneracy": len(cl),
            "coupling": float(np.linalg.norm(amp)),
        })
    return rows


def residual(H, vecs: np.ndarray, w: np.ndarray) -> float:
    r = H @ vecs - vecs * w[None, :]
    return float(np.max(np.linalg.norm(r, axis=0)))


def solve_dense(hc: np.ndarray, g: float, n: int) -> dict:
    H = full_sparse(hc, g, n).toarray()
    P = parity_sparse(n).toarray()
    w, U = np.linalg.eigh(H)
    for cl in clusters(w):
        if len(cl) > 1:
            sub = U[:, cl]
            pw, pv = np.linalg.eigh(sub.T @ P @ sub)
            U[:, cl] = sub @ pv
    par = np.einsum("ij,ij->j", U, P @ U)
    res = residual(H, U, w)
    comm = float(np.linalg.norm(H @ P - P @ H))
    plus = np.where(par > 0.5)[0]
    minus = np.where(par < -0.5)[0]
    if len(plus) + len(minus) != len(w):
        raise RuntimeError("parity not resolved")
    out = {
        "w_plus": w[plus], "V_plus": U[:, plus],
        "w_minus": w[minus], "V_minus": U[:, minus],
        "w_all": w, "par_all": par, "residual_max": res, "commutator_fro": comm,
        "k_used": int(len(plus)), "method": "dense_full_eigh",
    }
    audit = {}
    for s, key in ((1, "plus"), (-1, "minus")):
        bw = np.linalg.eigvalsh(block(hc, g, n, s).toarray())
        audit[f"block_dense_vs_full_{key}"] = float(np.max(np.abs(bw - out[f"w_{key}"])))
    out["audit"] = audit
    return out


def solve_sparse(hc: np.ndarray, g: float, n: int, k: int) -> dict:
    Hf = full_sparse(hc, g, n)
    P = parity_sparse(n)
    comm = float(sp.linalg.norm(Hf @ P - P @ Hf))
    out = {"commutator_fro": comm, "method": "parity_block_eigsh", "k_used": k}
    res = []
    for s, key in ((1, "plus"), (-1, "minus")):
        B = block(hc, g, n, s)
        v0 = np.ones(B.shape[0]) / np.sqrt(B.shape[0])
        w, u = eigsh(B, k=k, which="SA", tol=1e-12, v0=v0, maxiter=200000)
        o = np.argsort(w)
        w, u = w[o], u[:, o]
        vf = embed(u, n, s)
        res.append(residual(Hf, vf, w))
        out[f"w_{key}"] = w
        out[f"V_{key}"] = vf
        out[f"block_residual_{key}"] = residual(B, u, w)
    out["residual_max"] = float(max(res))
    pv = P @ np.hstack([out["V_plus"], out["V_minus"]])
    ref = np.hstack([out["V_plus"], -out["V_minus"]])
    out["parity_eigvec_err"] = float(np.max(np.abs(pv - ref)))
    return out


def audit_sparse(hc: np.ndarray, g: float, n: int, sol: dict) -> dict:
    audit = {}
    dense_all = []
    for s, key in ((1, "plus"), (-1, "minus")):
        bw = np.linalg.eigvalsh(block(hc, g, n, s).toarray())
        dense_all.append(bw)
        k = len(sol[f"w_{key}"])
        audit[f"block_dense_vs_eigsh_{key}"] = float(np.max(np.abs(bw[:k] - sol[f"w_{key}"])))
    Hf = full_sparse(hc, g, n)
    v0 = np.ones(Hf.shape[0]) / np.sqrt(Hf.shape[0])
    wf = np.sort(eigsh(Hf, k=12, which="SA", tol=1e-12, v0=v0, maxiter=200000)[0])
    merged_dense = np.sort(np.concatenate(dense_all))[:12]
    audit["full_eigsh12_vs_dense_block_union"] = float(np.max(np.abs(wf - merged_dense)))
    kept = np.sort(np.concatenate([sol["w_plus"], sol["w_minus"]]))[:N_LOW]
    audit["low8_kept_vs_dense_block_union"] = float(np.max(np.abs(kept - merged_dense[:N_LOW])))
    return audit


def coupled_gap(rows: list[dict], tau: float) -> dict | None:
    for r in rows[1:]:
        if r["coupling"] > tau:
            return r
    return None


def analyze(n: int, seed: int, J: np.ndarray, hc: np.ndarray, label: str, g: float) -> tuple[dict, list[dict]]:
    t0 = time.perf_counter()
    if n <= 6:
        sol = solve_dense(hc, g, n)
        k_hist = [sol["k_used"]]
    else:
        k_hist = []
        for k in K_LADDER:
            sol = solve_sparse(hc, g, n, k)
            k_hist.append(k)
            gs_sign = 1 if sol["w_plus"][0] <= sol["w_minus"][0] else -1
            key = "plus" if gs_sign > 0 else "minus"
            g0 = sol[f"V_{key}"][:, 0]
            rows_same = sector_list(sol[f"w_{key}"], sol[f"V_{key}"], gs_sign, g0, n, float(sol[f"w_{key}"][0]))
            if coupled_gap(rows_same, TAU_PRIMARY) is not None:
                break
        sol["audit"] = audit_sparse(hc, g, n, sol)
    e0p, e0m = float(sol["w_plus"][0]), float(sol["w_minus"][0])
    gs_sign = 1 if e0p <= e0m else -1
    key_same = "plus" if gs_sign > 0 else "minus"
    key_opp = "minus" if gs_sign > 0 else "plus"
    ground = sol[f"V_{key_same}"][:, 0]
    if float(ground.sum()) < 0:
        ground = -ground
    E0 = min(e0p, e0m)
    rows_same = sector_list(sol[f"w_{key_same}"], sol[f"V_{key_same}"], gs_sign, ground, n, E0)
    rows_opp = sector_list(sol[f"w_{key_opp}"], sol[f"V_{key_opp}"], -gs_sign, ground, n, E0)
    P = parity_sparse(n)
    gs_par = float(ground @ (P @ ground))
    all_levels = sorted(
        [(float(e), "+") for e in sol["w_plus"]] + [(float(e), "-") for e in sol["w_minus"]]
    )
    low = all_levels[:N_LOW]
    first_same = rows_same[1] if len(rows_same) > 1 else None
    rec = {
        "N": n, "seed": seed, "field": label, "g": g, "g_over_J": g,
        "E0": E0, "E1": low[1][0], "full_gap": low[1][0] - E0, "E1_parity": low[1][1],
        "ground_parity_expect": gs_par, "ground_sector": "+" if gs_sign > 0 else "-",
        "ground_min_amp": float(np.min(ground)),
        "E0_plus": e0p, "E1_plus": float(sol["w_plus"][1]),
        "E0_minus": e0m, "E1_minus": float(sol["w_minus"][1]),
        "Delta_plus": float(sol["w_plus"][1] - e0p),
        "Delta_minus": float(sol["w_minus"][1] - e0m),
        "cross_split": e0m - e0p,
        "c_first_same_excited": first_same["coupling"] if first_same else None,
        "max_opposite_coupling": float(max(r["coupling"] for r in rows_opp)),
        "V0_norm": float(np.linalg.norm(V_apply(ground, n))),
        "residual_max": sol["residual_max"],
        "commutator_fro": sol["commutator_fro"],
        "k_ladder": k_hist, "k_used": sol["k_used"], "method": sol["method"],
        "audit": sol["audit"],
        "parity_eigvec_err": sol.get("parity_eigvec_err"),
        "low_levels": [{"E": e, "parity": p} for e, p in low],
        "seconds": time.perf_counter() - t0,
    }
    for tau in TAUS:
        cg = coupled_gap(rows_same, tau)
        tag = f"{tau:.0e}"
        if cg is None:
            rec[f"coupled_gap_tau{tag}"] = None
            rec[f"coupled_level_tau{tag}"] = None
            rec[f"coupled_c_tau{tag}"] = None
        else:
            rec[f"coupled_gap_tau{tag}"] = cg["E_minus_E0"]
            rec[f"coupled_level_tau{tag}"] = rows_same.index(cg)
            rec[f"coupled_c_tau{tag}"] = cg["coupling"]
    cg = coupled_gap(rows_same, TAU_PRIMARY)
    rec["coupled_gap"] = cg["E_minus_E0"] if cg else None
    rec["coupled_is_first_same_excited"] = bool(cg is not None and rows_same.index(cg) == 1)
    rec["adiabatic_ratio_c_over_gap2"] = (cg["coupling"] / cg["E_minus_E0"] ** 2) if cg else None
    rec["n_same_zero_coupled_below_coupled"] = (rows_same.index(cg) - 1) if cg else None
    levels = []
    for r in rows_same + rows_opp:
        levels.append({"N": n, "seed": seed, "field": label, "g": g, **r,
                       "same_sector_as_ground": r["sector"] == rec["ground_sector"]})
    levels.sort(key=lambda z: z["E"])
    return rec, levels


def _stats(vals):
    v = np.array([x for x in vals if x is not None], dtype=np.float64)
    if v.size == 0:
        return {"n": 0}
    return {"n": int(v.size), "mean": float(v.mean()), "min": float(v.min()), "max": float(v.max()),
            "median": float(np.median(v))}


def main() -> None:
    RES.mkdir(parents=True, exist_ok=True)
    prereg_sha = hashlib.sha256(PREREG.read_bytes()).hexdigest()
    panel_doc = {"panel": {str(k): list(v) for k, v in PANEL.items()}, "prereg_sha256": prereg_sha,
                 "frozen_before_gaps": True}
    bstar_rows = {}
    t_pre = time.perf_counter()
    for n, seeds in PANEL.items():
        for s in seeds:
            J = make_J(s, n)
            t0 = time.perf_counter()
            pk = peak_only(J, n)
            pk["seconds"] = time.perf_counter() - t0
            u = J[np.triu_indices(n, 1)]
            pk["J_rms_times_sqrtN"] = float(np.sqrt(np.mean(u ** 2)) * np.sqrt(n))
            bstar_rows[(n, s)] = pk
            print(f"B* N={n} s={s} interior={pk['interior']} B*={pk['Bstar']} {pk['seconds']:.2f}s", flush=True)
    pre_wall = time.perf_counter() - t_pre
    panel_doc["bstar"] = {f"N{n}_s{s}": v for (n, s), v in bstar_rows.items()}
    panel_doc["bstar_preprocessing_wall_seconds"] = pre_wall
    (RES / "PANEL_FROZEN.json").write_text(json.dumps(panel_doc, indent=2) + "\n")

    recs, levels = [], []
    t_spec = time.perf_counter()
    for n, seeds in PANEL.items():
        z = z_table(n)
        for s in seeds:
            J = make_J(s, n)
            hc = pair_diag(J, z)
            pk = bstar_rows[(n, s)]
            fields = [("g1.5J", G_FIXED)]
            if pk["interior"]:
                fields.insert(0, ("Bstar", pk["Bstar"]))
            for label, g in fields:
                rec, lv = analyze(n, s, J, hc, label, float(g))
                rec["Bstar_interior"] = pk["interior"]
                recs.append(rec)
                levels.extend(lv)
                print(f"N={n} s={s} {label} g={g:.4f} gap={rec['full_gap']:.4g} E1par={rec['E1_parity']} "
                      f"D+={rec['Delta_plus']:.4g} D-={rec['Delta_minus']:.4g} split={rec['cross_split']:.3g} "
                      f"coupled={rec['coupled_gap']} k={rec['k_used']} res={rec['residual_max']:.1e} "
                      f"opp={rec['max_opposite_coupling']:.1e}", flush=True)
    spec_wall = time.perf_counter() - t_spec

    scalar_keys = ["N", "seed", "field", "g", "Bstar_interior", "E0", "E1", "full_gap", "E1_parity",
                   "ground_sector", "ground_parity_expect", "ground_min_amp", "E0_plus", "E1_plus",
                   "E0_minus", "E1_minus", "Delta_plus", "Delta_minus", "cross_split",
                   "c_first_same_excited", "coupled_gap", "coupled_is_first_same_excited",
                   "n_same_zero_coupled_below_coupled", "adiabatic_ratio_c_over_gap2",
                   *[f"coupled_gap_tau{t:.0e}" for t in TAUS], *[f"coupled_c_tau{t:.0e}" for t in TAUS],
                   "max_opposite_coupling", "V0_norm", "residual_max", "commutator_fro", "k_used", "method",
                   "seconds"]
    with (RES / "instances.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=scalar_keys + ["audit_max_abs", "low8"])
        w.writeheader()
        for r in recs:
            row = {k: r.get(k) for k in scalar_keys}
            row["audit_max_abs"] = max(r["audit"].values())
            row["low8"] = " ".join(f"{x['E']:.10f}{x['parity']}" for x in r["low_levels"])
            w.writerow(row)
    with (RES / "levels.csv").open("w", newline="") as f:
        keys = ["N", "seed", "field", "g", "sector", "same_sector_as_ground", "E", "E_minus_E0", "degeneracy", "coupling"]
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in levels:
            w.writerow({k: r[k] for k in keys})

    summary = {}
    for n in PANEL:
        for label in ("Bstar", "g1.5J"):
            rs = [r for r in recs if r["N"] == n and r["field"] == label]
            summary[f"N{n}_{label}"] = {
                "n": len(rs),
                "g": _stats([r["g"] for r in rs]),
                "full_gap": _stats([r["full_gap"] for r in rs]),
                "Delta_plus": _stats([r["Delta_plus"] for r in rs]),
                "Delta_minus": _stats([r["Delta_minus"] for r in rs]),
                "cross_split": _stats([r["cross_split"] for r in rs]),
                "coupled_gap": _stats([r["coupled_gap"] for r in rs]),
                "c_first_same_excited": _stats([r["c_first_same_excited"] for r in rs]),
                "adiabatic_ratio": _stats([r["adiabatic_ratio_c_over_gap2"] for r in rs]),
                "n_E1_odd": int(sum(1 for r in rs if r["E1_parity"] == "-")),
                "n_ground_plus": int(sum(1 for r in rs if r["ground_sector"] == "+")),
                "n_coupled_found": int(sum(1 for r in rs if r["coupled_gap"] is not None)),
                "n_coupled_is_first_same_excited": int(sum(1 for r in rs if r["coupled_is_first_same_excited"])),
                "n_tau_sensitive": int(sum(1 for r in rs if len({r[f'coupled_gap_tau{t:.0e}'] for t in TAUS}) > 1)),
                "max_opposite_coupling": float(max(r["max_opposite_coupling"] for r in rs)) if rs else None,
                "max_residual": float(max(r["residual_max"] for r in rs)) if rs else None,
                "max_commutator_fro": float(max(r["commutator_fro"] for r in rs)) if rs else None,
                "max_audit": float(max(max(r["audit"].values()) for r in rs)) if rs else None,
                "min_ground_parity_expect": float(min(r["ground_parity_expect"] for r in rs)) if rs else None,
                "k_used": sorted({r["k_used"] for r in rs}),
            }
    payload = {
        "prereg_sha256": prereg_sha,
        "hamiltonian": "H = sum_{i<j} J_ij Z_i Z_j - g sum_i X_i; J_ij ~ N(0, 1/N) i.e. J=1; g dimensionless g/J",
        "drive": "transverse ramp; dH/dg = -sum_i X_i commutes with P = prod X_i",
        "panel": {str(k): list(v) for k, v in PANEL.items()},
        "taus": list(TAUS), "tau_primary": TAU_PRIMARY, "k_ladder": list(K_LADDER), "cluster_tol": CLUSTER_TOL,
        "bstar_preprocessing_wall_seconds": pre_wall,
        "spectral_wall_seconds": spec_wall,
        "quantum_energy_queries": 0,
        "historical_sk_queries_unchanged": 3252512,
        "no_spsa": True, "no_ansatz": True, "no_min_gap_claim": True,
        "entropy_peak_not_assumed_critical": True,
        "env": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
        "peak_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "summary": summary,
        "instances": recs,
    }
    (RES / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(summary, indent=1), flush=True)


if __name__ == "__main__":
    main()
