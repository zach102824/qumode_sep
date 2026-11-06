"""Round-1 QAOA scan for the sparse uniform-sign 3-body model.

Run from this directory:

    python run_round1.py
    python run_round1.py --self-test

Writes graphs, Pauli term lists, validation, spectra, angles, CSV/JSON,
and plots under runs/round1/. Every selected graph is reported. Gammas are
not filtered before the ladder; a post-hoc depth-separation ranking is
stored separately and labeled as post-hoc.
"""

from __future__ import annotations

import os

# One BLAS thread per process. The scan uses a process pool; extra BLAS
# threads oversubscribe the machine and make dense audits very slow.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
import csv
import json
import sys
import time
import traceback
from pathlib import Path

import numpy as np

THIS = Path(__file__).resolve().parent
if str(THIS) not in sys.path:
    sys.path.insert(0, str(THIS))

import model
import qaoa
import validate

# Prescribed window from the round-1 brief. Adaptive points were chosen from
# a ground-state survey (saved as gamma_survey.json), not from QAOA fidelities:
# 0.95 is where the half-cut entropy becomes appreciable while the overlap
# with |+> is still modest; 1.0 is the finite-size minimum of the gap.
PRESCRIBED_GAMMAS = (0.3, 0.5, 0.7, 0.9, 1.1, 1.3)
ADAPTIVE_GAMMAS = (0.95, 1.0)
GAMMAS = PRESCRIBED_GAMMAS + ADAPTIVE_GAMMAS
GAMMA_ROLE = {g: "prescribed" for g in PRESCRIBED_GAMMAS}
GAMMA_ROLE.update({g: "adaptive_from_gs_survey" for g in ADAPTIVE_GAMMAS})
NS = (6, 8, 10, 12)


def json_dump(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=False) + "\n")


def survey_one(task: dict) -> list:
    n = int(task["n"])
    edges = [tuple(e) for e in task["edges"]]
    hc = model.cost_diagonal(n, edges, model.uniform_couplings(len(edges), 1.0))
    rows = []
    for gamma in task["gammas"]:
        gamma = float(gamma)
        spec = model.ground_state(hc, gamma, n)
        ent = model.half_cut_entropy(spec["gs"], n)
        prod = model.best_product_overlap(spec["gs"], n, seed=int(task["seed"]) + int(round(gamma * 1000)))
        obs = model.state_observables(spec["gs"], n, edges, spec["gs"], spec["evals0"], spec["evals_max"], spec["evals0"])
        rows.append(
            {
                "graph_id": task["graph_id"],
                "n": n,
                "seed": int(task["seed"]),
                "linear": bool(task["linear"]),
                "gamma": gamma,
                "gamma_role": GAMMA_ROLE.get(gamma, "survey_only"),
                "E0": spec["evals0"],
                "E1": spec["evals1"],
                "Emax": spec["evals_max"],
                "gap": spec["gap"],
                "degenerate": spec["degenerate"],
                "n_negative_amps": spec["n_negative_amps"],
                "gs_residual": spec["residual"],
                "emax_plus_e0": spec["emax_plus_e0"],
                "S_gs": ent["S_half_bits"],
                "cut": ent["cut_definition"],
                "m_z_gs": obs["m_z"],
                "m_x_gs": obs["m_x"],
                "q_z_gs": obs["q_z"],
                "clause_mean_gs": obs["clause_mean"],
                "F_plus_gs": prod["F_plus"],
                "F_classical_gs": prod["F_classical_allplusZ"],
                "F_best_bitstring_gs": prod["F_best_bitstring"],
                "F_product_gs": prod["F_product"],
            }
        )
    return rows


def run_cell(task: dict) -> dict:
    t0 = time.time()
    try:
        n = int(task["n"])
        gamma = float(task["gamma"])
        edges = [tuple(e) for e in task["edges"]]
        couplings = model.uniform_couplings(len(edges), +1.0)
        hc = model.cost_diagonal(n, edges, couplings)
        spec = model.ground_state(hc, gamma, n)
        prod = model.best_product_overlap(spec["gs"], n, seed=int(task["seed"]) + int(round(gamma * 1000)))
        ent = model.half_cut_entropy(spec["gs"], n)
        gs_obs = model.state_observables(
            spec["gs"], n, edges, spec["gs"], spec["evals0"], spec["evals_max"], spec["evals0"]
        )
        ladder = qaoa.run_depth_ladder(
            n=n,
            hc_diag=hc,
            gamma_field=gamma,
            gs=spec["gs"],
            E0=spec["evals0"],
            Emax=spec["evals_max"],
            edges=edges,
            seed=int(task["seed"]),
            max_L=int(task["max_L"]),
            n_grid=int(task["n_grid"]),
            do_shared=bool(task["do_shared"]),
        )
        # Replay the independent schedules once more from the saved angles.
        plus = model.plus_state(n)
        replay_err = 0.0
        for row in ladder:
            if row["L"] == 0:
                continue
            psi = qaoa.apply_qaoa(plus, hc, n, row["gammas"], row["betas"])
            e = qaoa.energy_of(psi, hc, n, gamma)
            replay_err = max(replay_err, abs(e - row["energy"]))
        spectrum = {
            "graph_id": task["graph_id"],
            "n": n,
            "seed": int(task["seed"]),
            "linear": bool(task["linear"]),
            "gamma": gamma,
            "gamma_role": GAMMA_ROLE.get(gamma, "unspecified"),
            "E0": spec["evals0"],
            "E1": spec["evals1"],
            "Emax": spec["evals_max"],
            "gap": spec["gap"],
            "degenerate": spec["degenerate"],
            "n_negative_amps": spec["n_negative_amps"],
            "dim": spec["dim"],
            "gs_method": spec["method"],
            "gs_residual": spec["residual"],
            "emax_plus_e0": spec["emax_plus_e0"],
            "S_gs": ent["S_half_bits"],
            "cut": ent["cut_definition"],
            "cut_low_qubits": ent["cut_low_qubits"],
            "m_z_gs": gs_obs["m_z"],
            "m_x_gs": gs_obs["m_x"],
            "q_z_gs": gs_obs["q_z"],
            "clause_mean_gs": gs_obs["clause_mean"],
            "F_plus_gs": prod["F_plus"],
            "F_classical_gs": prod["F_classical_allplusZ"],
            "F_best_bitstring_gs": prod["F_best_bitstring"],
            "F_product_gs": prod["F_product"],
            "product_fit_nfev": prod["n_fev"],
        }
        return {
            "ok": True,
            "seconds": time.time() - t0,
            "replay_energy_abs_err": float(replay_err),
            "spectrum": spectrum,
            "ladder": ladder,
            "task": {
                "graph_id": task["graph_id"],
                "n": n,
                "seed": int(task["seed"]),
                "linear": bool(task["linear"]),
                "gamma": gamma,
                "gamma_role": GAMMA_ROLE.get(gamma, "unspecified"),
            },
        }
    except Exception:
        return {
            "ok": False,
            "seconds": time.time() - t0,
            "error": traceback.format_exc(),
            "task": {
                "graph_id": task.get("graph_id"),
                "n": task.get("n"),
                "gamma": task.get("gamma"),
            },
        }


def _median(xs):
    if not xs:
        return None
    return float(np.median(np.asarray(xs, dtype=np.float64)))


def _spread(xs):
    if not xs:
        return {}
    a = np.asarray(xs, dtype=np.float64)
    return {
        "n": int(a.size),
        "min": float(a.min()),
        "median": float(np.median(a)),
        "max": float(a.max()),
        "mean": float(a.mean()),
        "std": float(a.std(ddof=0)),
    }


def flatten_independent(cells: list) -> list:
    rows = []
    for cell in cells:
        if not cell.get("ok"):
            continue
        spec = cell["spectrum"]
        for row in cell["ladder"]:
            if row["ansatz"] not in ("independent", "initial_plus"):
                continue
            flat = {
                "graph_id": spec["graph_id"],
                "n": spec["n"],
                "seed": spec["seed"],
                "linear": spec["linear"],
                "gamma": spec["gamma"],
                "gamma_role": spec.get("gamma_role"),
                "L": row["L"],
                "F": row["F"],
                "energy": row["energy"],
                "eta": row["eta"],
                "S_half_bits": row["S_half_bits"],
                "m_z": row["m_z"],
                "m_x": row["m_x"],
                "q_z": row["q_z"],
                "clause_mean": row["clause_mean"],
                "E0": spec["E0"],
                "E1": spec["E1"],
                "Emax": spec["Emax"],
                "gap": spec["gap"],
                "S_gs": spec["S_gs"],
                "m_z_gs": spec["m_z_gs"],
                "m_x_gs": spec["m_x_gs"],
                "clause_mean_gs": spec["clause_mean_gs"],
                "F_plus_gs": spec["F_plus_gs"],
                "F_classical_gs": spec["F_classical_gs"],
                "F_product_gs": spec["F_product_gs"],
                "F_best_bitstring_gs": spec["F_best_bitstring_gs"],
                "winner": row.get("winner"),
                "nfev": row.get("nfev"),
                "energy_nested_vs_previous": row.get("energy_nested_vs_previous"),
                "fidelity_monotone_vs_previous": row.get("fidelity_monotone_vs_previous"),
                "sum_abs_gamma_principal": row.get("sum_abs_gamma_principal"),
                "sum_abs_beta_principal": row.get("sum_abs_beta_principal"),
                "duration_principal": row.get("duration_principal"),
                "duration_raw": row.get("duration_raw"),
                "n_gamma_wrapped": row.get("n_gamma_wrapped"),
                "n_beta_wrapped": row.get("n_beta_wrapped"),
                "max_consecutive_principal_jump": row.get("max_consecutive_principal_jump"),
                "principal_state_overlap": row.get("principal_state_overlap"),
                "cell_seconds": cell["seconds"],
                "replay_energy_abs_err": cell["replay_energy_abs_err"],
            }
            if row["L"] == 1:
                for key in (
                    "l1_grid_min_energy",
                    "l1_grid_max_fidelity",
                    "l1_energy_opt_beats_grid",
                    "l1_fidelity_max_F",
                    "l1_fidelity_max_energy",
                    "l1_energy_gap_vs_fidelity_opt",
                    "l1_fidelity_gap_vs_fidelity_opt",
                ):
                    flat[key] = row.get(key)
            shared = row.get("shared_gamma")
            if shared:
                flat["shared_F"] = shared["F"]
                flat["shared_energy"] = shared["energy"]
                flat["shared_eta"] = shared["eta"]
                flat["shared_gamma_value"] = shared["gamma"]
            anneal = row.get("anneal_matched")
            if anneal:
                flat["anneal_F"] = anneal["F"]
                flat["anneal_energy"] = anneal["energy"]
                flat["anneal_eta"] = anneal["eta"]
                flat["anneal_duration"] = anneal["duration"]
            fixed = row.get("anneal_fixed_dt")
            if fixed:
                flat["anneal_fixed_F"] = fixed["F"]
                flat["anneal_fixed_energy"] = fixed["energy"]
                flat["anneal_fixed_duration"] = fixed["duration"]
            rows.append(flat)
    return rows


def write_csv(path: Path, rows: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    # Union of keys, stable order with a preferred prefix.
    preferred = [
        "graph_id", "n", "seed", "linear", "gamma", "L", "F", "energy", "eta",
        "S_half_bits", "gap", "E0", "Emax",
    ]
    keys = []
    seen = set()
    for k in preferred:
        if any(k in r for r in rows):
            keys.append(k)
            seen.add(k)
    for r in rows:
        for k in r.keys():
            if k not in seen:
                keys.append(k)
                seen.add(k)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def group_stats(flat: list) -> list:
    buckets = {}
    for r in flat:
        buckets.setdefault((r["gamma"], r["n"], r["L"]), []).append(r)
    out = []
    for (gamma, n, L), rows in sorted(buckets.items()):
        out.append(
            {
                "gamma": gamma,
                "n": n,
                "L": L,
                "F": _spread([r["F"] for r in rows]),
                "eta": _spread([r["eta"] for r in rows]),
                "energy": _spread([r["energy"] for r in rows]),
                "S_qaoa": _spread([r["S_half_bits"] for r in rows]),
                "anneal_F": _spread([r["anneal_F"] for r in rows if "anneal_F" in r]),
                "shared_F": _spread([r["shared_F"] for r in rows if "shared_F" in r]),
            }
        )
    return out


def depth_separation(flat: list) -> list:
    """Per (graph, gamma) fidelity ladder and per (n, gamma) median ladder."""
    by = {}
    for r in flat:
        by.setdefault((r["graph_id"], r["gamma"]), {})[r["L"]] = r
    per_graph = []
    for (gid, gamma), depths in sorted(by.items()):
        if 1 not in depths or 5 not in depths:
            continue
        series = {str(L): depths[L]["F"] for L in sorted(depths)}
        f1 = depths[1]["F"]
        f5 = depths[5]["F"]
        mono = all(
            depths[L]["F"] + 1e-7 >= depths[L - 1]["F"]
            for L in range(1, 6)
            if L in depths and (L - 1) in depths
        )
        per_graph.append(
            {
                "graph_id": gid,
                "n": depths[1]["n"],
                "seed": depths[1]["seed"],
                "gamma": gamma,
                "F_by_L": series,
                "F1": f1,
                "F5": f5,
                "dF_5_minus_1": f5 - f1,
                "monotone_L1_to_L5": mono,
                "F_plus_gs": depths[1]["F_plus_gs"],
                "F_product_gs": depths[1]["F_product_gs"],
                "S_gs": depths[1]["S_gs"],
                "gap": depths[1]["gap"],
                "m_z_gs": depths[1]["m_z_gs"],
                "clause_mean_gs": depths[1]["clause_mean_gs"],
            }
        )
    # Medians across graphs.
    med = []
    keys = sorted({(g["n"], g["gamma"]) for g in per_graph})
    for n, gamma in keys:
        subset = [g for g in per_graph if g["n"] == n and g["gamma"] == gamma]
        def col(name):
            return [g[name] for g in subset]
        f_by_L = {}
        for L in range(0, 6):
            vals = []
            for g in subset:
                if str(L) in g["F_by_L"]:
                    vals.append(g["F_by_L"][str(L)])
            f_by_L[str(L)] = _spread(vals)
        med.append(
            {
                "n": n,
                "gamma": gamma,
                "F_by_L": f_by_L,
                "dF_5_minus_1": _spread(col("dF_5_minus_1")),
                "F1": _spread(col("F1")),
                "F5": _spread(col("F5")),
                "fraction_monotone": float(np.mean([g["monotone_L1_to_L5"] for g in subset])),
                "F_plus_gs": _spread(col("F_plus_gs")),
                "F_product_gs": _spread(col("F_product_gs")),
                "S_gs": _spread(col("S_gs")),
                "gap": _spread(col("gap")),
                "m_z_gs": _spread(col("m_z_gs")),
                "clause_mean_gs": _spread(col("clause_mean_gs")),
                "n_graphs": len(subset),
            }
        )
    return per_graph, med


def scaling_diagnostic(med_rows: list) -> dict:
    """Descriptive fits only. Four sizes cannot establish an asymptotic law."""
    # Use median F at each (n, L, gamma). Compare L/N against L * 2^(-N/2).
    out = {"by_gamma": [], "note": "N in {6,8,10,12} only. Fits are descriptive, not asymptotic."}
    gammas = sorted({r["gamma"] for r in med_rows})
    for gamma in gammas:
        rows = [r for r in med_rows if r["gamma"] == gamma]
        # Per-layer gain (F5-F1)/4 versus 1/N and versus 2^{-N/2}.
        ns, gain, inv, grover = [], [], [], []
        xs_ln, ys = [], []
        for r in rows:
            if r["dF_5_minus_1"]["n"] == 0:
                continue
            n = r["n"]
            g = r["dF_5_minus_1"]["median"] / 4.0
            ns.append(n)
            gain.append(g)
            inv.append(1.0 / n)
            grover.append(2.0 ** (-n / 2.0))
            for L in range(0, 6):
                block = r["F_by_L"].get(str(L)) or {}
                if block.get("n", 0):
                    xs_ln.append(L / n)
                    ys.append(block["median"])

        def lin_fit(x, y):
            x = np.asarray(x, dtype=np.float64)
            y = np.asarray(y, dtype=np.float64)
            if x.size < 2:
                return None
            A = np.column_stack([np.ones_like(x), x])
            coef, *_ = np.linalg.lstsq(A, y, rcond=None)
            pred = A @ coef
            ss_res = float(np.sum((y - pred) ** 2))
            ss_tot = float(np.sum((y - y.mean()) ** 2))
            r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 1e-15 else float("nan")
            rmse = float(np.sqrt(np.mean((y - pred) ** 2)))
            return {"intercept": float(coef[0]), "slope": float(coef[1]), "r2": r2, "rmse": rmse, "n_points": int(x.size)}

        out["by_gamma"].append(
            {
                "gamma": gamma,
                "per_layer_gain_vs_1_over_N": lin_fit(inv, gain),
                "per_layer_gain_vs_2_to_minus_N_over_2": lin_fit(grover, gain),
                "median_F_vs_L_over_N": lin_fit(xs_ln, ys),
                "points": [
                    {"n": int(n), "per_layer_gain_(F5-F1)/4": float(g), "1/N": float(iv), "2^(-N/2)": float(gr)}
                    for n, g, iv, gr in zip(ns, gain, inv, grover)
                ],
            }
        )
    return out


def posthoc_gamma(med_rows: list) -> dict:
    """Largest median F5-F1 with median F1 < 0.5. Labeled post-hoc."""
    cands = []
    for r in med_rows:
        if r["F1"]["n"] == 0 or r["n"] < 8:
            continue
        cands.append(r)
    if not cands:
        return {"available": False}
    # Rank by median depth separation, prefer F1 not in the easy 0.8-0.9 band.
    def key(r):
        f1 = r["F1"]["median"]
        easy_penalty = 1.0 if f1 >= 0.5 else 0.0
        return (-easy_penalty, r["dF_5_minus_1"]["median"])

    best = max(cands, key=key)
    return {
        "available": True,
        "selection": "post-hoc, after seeing fidelities; not a pre-registered Gamma",
        "rule": "among N>=8, maximize median (F5-F1), after penalizing median F1>=0.5",
        "n": best["n"],
        "gamma": best["gamma"],
        "median_F1": best["F1"]["median"],
        "median_F5": best["F5"]["median"],
        "median_dF": best["dF_5_minus_1"]["median"],
    }


def make_plots(flat, spectra, med_rows, out: Path) -> list:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out.mkdir(parents=True, exist_ok=True)
    written = []
    colors = {0: "#111111", 1: "#d62728", 2: "#1f77b4", 3: "#2ca02c", 4: "#17becf", 5: "#c51b8a"}
    markers = {0: "o", 1: "D", 2: "x", 3: "^", 4: "v", 5: "s"}

    # Spectrum diagnostics.
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 7.2), sharex=True)
    panels = [
        (axes[0, 0], "gap", "gap"),
        (axes[0, 1], "S_gs", "half-cut entropy (bits)"),
        (axes[1, 0], "F_product_gs", "best product overlap"),
        (axes[1, 1], "F_plus_gs", "overlap with |+>"),
    ]
    for ax, key, title in panels:
        for n in NS:
            pts = [s for s in spectra if s["n"] == n]
            # median and individuals per gamma
            gammas = sorted({p["gamma"] for p in pts})
            meds = []
            for g in gammas:
                vals = [p[key] for p in pts if p["gamma"] == g]
                ax.scatter([g] * len(vals), vals, s=18, alpha=0.35)
                meds.append(float(np.median(vals)))
            ax.plot(gammas, meds, marker="o", label=f"N={n}")
        ax.set_title(title)
        ax.set_xlabel("Gamma")
        ax.grid(True, alpha=0.3)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Sparse uniform-sign 3-regular 3-body — full-space ground state")
    fig.tight_layout()
    p = out / "spectra_vs_gamma.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    written.append(p.name)

    # F vs 1/N faceted by Gamma, one curve per depth. All graphs as faint points.
    gammas = list(GAMMAS)
    ncols = 4
    nrows = int(np.ceil(len(gammas) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(14.5, 3.4 * nrows), sharey=True)
    axes = np.atleast_1d(axes).ravel()
    for ax in axes[len(gammas):]:
        ax.axis("off")
    for ax, gamma in zip(axes, gammas):
        for L in range(0, 6):
            pts = [r for r in flat if r["gamma"] == gamma and r["L"] == L]
            if not pts:
                continue
            ax.scatter(
                [1.0 / r["n"] for r in pts],
                [r["F"] for r in pts],
                s=16,
                alpha=0.35,
                color=colors[L],
                marker=markers[L],
            )
            xs, ys = [], []
            for n in NS:
                vals = [r["F"] for r in pts if r["n"] == n]
                if vals:
                    xs.append(1.0 / n)
                    ys.append(float(np.median(vals)))
            ax.plot(xs, ys, color=colors[L], marker=markers[L], label=f"L={L}")
        role = "prescribed" if gamma in PRESCRIBED_GAMMAS else "adaptive"
        ax.set_title(f"Gamma={gamma:g} ({role})")
        ax.set_xlabel("1/N")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("F = |overlap|^2")
    if len(axes) > ncols:
        axes[ncols].set_ylabel("F = |overlap|^2")
    axes[0].legend(fontsize=7, loc="best")
    fig.suptitle("Independent-angle QAOA fidelity vs 1/N (all graphs; lines are medians)")
    fig.tight_layout()
    p = out / "F_vs_invN_by_depth.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    written.append(p.name)

    # F vs L/N, one panel per Gamma.
    fig, axes = plt.subplots(nrows, ncols, figsize=(14.5, 3.4 * nrows), sharey=True)
    axes = np.atleast_1d(axes).ravel()
    for ax in axes[len(gammas):]:
        ax.axis("off")
    n_marker = {6: "o", 8: "s", 10: "D", 12: "^"}
    for ax, gamma in zip(axes, gammas):
        for n in NS:
            pts = [r for r in flat if r["gamma"] == gamma and r["n"] == n]
            # median curve
            xs, ys = [], []
            for L in range(0, 6):
                vals = [r["F"] for r in pts if r["L"] == L]
                ax.scatter(
                    [L / n] * len(vals),
                    vals,
                    s=14,
                    alpha=0.3,
                    marker=n_marker[n],
                    color=colors[L],
                )
                if vals:
                    xs.append(L / n)
                    ys.append(float(np.median(vals)))
            ax.plot(xs, ys, marker=n_marker[n], label=f"N={n}")
        role = "prescribed" if gamma in PRESCRIBED_GAMMAS else "adaptive"
        ax.set_title(f"Gamma={gamma:g} ({role})")
        ax.set_xlabel("L/N")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("F")
    if len(axes) > ncols:
        axes[ncols].set_ylabel("F")
    axes[0].legend(fontsize=7)
    fig.suptitle("F vs L/N — sparse model, all instances (lines = medians)")
    fig.tight_layout()
    p = out / "F_vs_L_over_N.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    written.append(p.name)

    # Heatmap of median F5-F1.
    fig, ax = plt.subplots(figsize=(9.2, 3.6))
    mat = np.full((len(NS), len(gammas)), np.nan)
    for i, n in enumerate(NS):
        for j, gamma in enumerate(gammas):
            hit = [r for r in med_rows if r["n"] == n and r["gamma"] == gamma]
            if hit and hit[0]["dF_5_minus_1"]["n"]:
                mat[i, j] = hit[0]["dF_5_minus_1"]["median"]
    im = ax.imshow(mat, cmap="coolwarm", aspect="auto", vmin=-0.05, vmax=1.0)
    ax.set_xticks(range(len(gammas)), [str(g) for g in gammas])
    ax.set_yticks(range(len(NS)), [str(n) for n in NS])
    ax.set_xlabel("Gamma")
    ax.set_ylabel("N")
    for i in range(len(NS)):
        for j in range(len(gammas)):
            if np.isfinite(mat[i, j]):
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, label="median F(L=5)-F(L=1)")
    ax.set_title("Depth separation (post-hoc view of the full grid)")
    fig.tight_layout()
    p = out / "depth_separation_heatmap.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    written.append(p.name)

    # QAOA vs matched anneal at L=5.
    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    for r in flat:
        if r["L"] != 5 or "anneal_F" not in r:
            continue
        ax.scatter(r["anneal_F"], r["F"], s=18, alpha=0.7)
    lim = [0, 1]
    ax.plot(lim, lim, color="gray", lw=1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("digitized anneal F (matched sum|γ|+sum|β|)")
    ax.set_ylabel("independent QAOA F")
    ax.set_title("L=5 fidelity: QAOA vs duration-matched anneal")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    p = out / "qaoa_vs_anneal_L5.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    written.append(p.name)
    return written


def tables_markdown(med_rows, per_graph, scaling, posthoc, spectra_summary) -> str:
    lines = []
    lines.append("# Sparse uniform-sign 3-body QAOA — round 1 tables")
    lines.append("")
    lines.append("Model: H = -sum_{triples} Z_i Z_j Z_k - Gamma sum_i X_i")
    lines.append("on full-rank GF(2) 3-regular simple hypergraphs, couplings all +1.")
    lines.append("This is not the collective p-spin and not a random-sign glass.")
    lines.append("F is the squared overlap with the full-space ground state.")
    lines.append("eta = (Emax - E) / (Emax - E0) of that same full Hamiltonian.")
    lines.append("Half-cut entropy is the low n/2 bits versus the high bits, in bits.")
    lines.append("")
    lines.append("## Median fidelity by Gamma, N, L")
    lines.append("")
    lines.append("| Gamma | N | L | F median | F min | F max | eta median |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|")
    for r in med_rows:
        for L in range(0, 6):
            block = r["F_by_L"].get(str(L)) or {}
            if not block.get("n"):
                continue
            eta = r["eta"] if False else None
        # eta lives in group stats; F_by_L is enough here. Print F.
        for L in range(0, 6):
            block = r["F_by_L"].get(str(L)) or {}
            if not block.get("n"):
                continue
            lines.append(
                f"| {r['gamma']} | {r['n']} | {L} | {block['median']:.4f} | {block['min']:.4f} | {block['max']:.4f} | |"
            )
    lines.append("")
    lines.append("## Ground state at each Gamma (median over the reported graphs)")
    lines.append("")
    lines.append("| Gamma | N | gap | S_gs | F_plus | F_product | F_classical | m_z | clause |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in med_rows:
        # GS quantities repeat for every L; the med row already aggregated them once.
        lines.append(
            "| {g} | {n} | {gap:.4f} | {S:.3f} | {fp:.4f} | {fpr:.3f} | {fc:.3f} | {mz:.3f} | {cl:.3f} |".format(
                g=r["gamma"],
                n=r["n"],
                gap=r["gap"]["median"],
                S=r["S_gs"]["median"],
                fp=r["F_plus_gs"]["median"],
                fpr=r["F_product_gs"]["median"],
                fc=r["F_classical_gs"]["median"] if "F_classical_gs" in r else float("nan"),
                mz=r["m_z_gs"]["median"],
                cl=r["clause_mean_gs"]["median"],
            )
        )
    # The table above repeats  because med_rows is per (n, gamma) once. Good.
    # Wait, I'm iterating med_rows which is one per (n, gamma), but the loop doesn't repeat L. Good.
    # But I had a useless first loop. It's OK, only one row per (n, gamma). Let me look... 
    # I only have one for r in med_rows for the GS table. Good.
    lines.append("")
    lines.append("## Post-hoc depth separation")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(posthoc, indent=2))
    lines.append("```")
    lines.append("")
    lines.append("## Descriptive scaling fits (not asymptotics)")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(scaling, indent=2))
    lines.append("```")
    lines.append("")
    lines.append(f"Per-graph ladders stored: {len(per_graph)}")
    lines.append("")
    return "\n".join(lines)


def collect_failures(cells: list) -> list:
    bad = []
    for cell in cells:
        if not cell.get("ok"):
            bad.append({"kind": "exception", "task": cell.get("task"), "error": cell.get("error")})
            continue
        if cell["replay_energy_abs_err"] > 1e-6:
            bad.append({"kind": "replay", "task": cell["task"], "err": cell["replay_energy_abs_err"]})
        spec = cell["spectrum"]
        if abs(spec.get("emax_plus_e0", 0.0)) > 1e-6 or spec.get("gs_residual", 0.0) > 1e-6:
            bad.append({"kind": "spectrum", "task": cell["task"], "emax_plus_e0": spec.get("emax_plus_e0"), "residual": spec.get("gs_residual")})
        for row in cell["ladder"]:
            if row["L"] == 0:
                continue
            ov = row.get("principal_state_overlap")
            if ov is not None and ov < 1.0 - 1e-6:
                bad.append({"kind": "principal_branch", "task": cell["task"], "L": row["L"], "overlap": ov})
            if row.get("energy_nested_vs_previous") is False:
                bad.append({"kind": "energy_not_nested", "task": cell["task"], "L": row["L"], "energy": row["energy"]})
            if row["L"] == 1:
                if row.get("l1_energy_opt_beats_grid") is False:
                    bad.append({"kind": "l1_worse_than_grid", "task": cell["task"], "energy": row["energy"], "grid": row.get("l1_grid_min_energy")})
                gap = row.get("l1_energy_gap_vs_fidelity_opt")
                if gap is not None and gap > 1e-4:
                    bad.append(
                        {
                            "kind": "l1_fidelity_opt_found_lower_energy",
                            "task": cell["task"],
                            "energy_gap": gap,
                            "F_energy_opt": row["F"],
                            "F_fidelity_opt": row.get("l1_fidelity_max_F"),
                        }
                    )
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(THIS / "runs" / "round1"))
    ap.add_argument("--graphs-per-n", type=int, default=4)
    ap.add_argument("--max-L", type=int, default=5)
    ap.add_argument("--grid", type=int, default=29)
    ap.add_argument("--workers", type=int, default=max(1, min(4, os.cpu_count() or 1)))
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--no-shared", action="store_true")
    ap.add_argument("--max-seed", type=int, default=50000)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "run.log"

    def say(msg):
        line = time.strftime("%H:%M:%S") + " " + msg
        print(line, flush=True)
        with log_path.open("a") as f:
            f.write(line + "\n")

    say("validation")
    report = validate.run_validation()
    json_dump(out / "validation.json", report)
    say(f"validation pass={report['pass']} checks={report['n_checks']} failed={report['n_failed']}")
    if not report["pass"]:
        say("validation failed; not starting the scan")
        for row in report["failures"]:
            say(f"FAIL {row['name']} {row['detail']}")
        sys.exit(1)

    say("select graphs")
    catalogs = []
    graphs = []
    for n in NS:
        cat = model.select_graphs(n, k=args.graphs_per_n, max_seed=args.max_seed)
        catalogs.append({k: v for k, v in cat.items() if k != "graphs"} | {"n_chosen": len(cat["graphs"]), "chosen_ids": [g["graph_id"] for g in cat["graphs"]]})
        graphs.extend(cat["graphs"])
        say(
            f"N={n} chosen={len(cat['graphs'])} linear_impossible={cat['linear_impossible_by_pair_count']} "
            f"ids={[g['graph_id'] for g in cat['graphs']]}"
        )
    if args.self_test:
        graphs = [g for g in graphs if g["n"] == 6][:1]
    json_dump(out / "graphs.json", {"catalogs": catalogs, "graphs": graphs})

    # Explicit term lists at each Gamma (coefficients of the full target).
    terms = []
    for g in graphs:
        edges = g["edges"]
        gammas = (0.7,) if args.self_test else GAMMAS
        for gamma in gammas:
            terms.append(
                {
                    "graph_id": g["graph_id"],
                    "gamma": gamma,
                    "terms": model.term_list(g["n"], edges, gamma),
                }
            )
    json_dump(out / "term_lists.json", terms)

    # Classical uniqueness on every selected graph.
    classical_rows = []
    for g in graphs:
        edges = [tuple(e) for e in g["edges"]]
        hc = model.cost_diagonal(g["n"], edges, model.uniform_couplings(len(edges), 1.0))
        info = model.count_classical_ground_states(hc, len(edges))
        info.update({"graph_id": g["graph_id"], "gf2_rank": g["gf2_rank"], "linear": g["linear"]})
        classical_rows.append(info)
        if info["n_ground"] != 1 or not info["all_plus_index_is_ground"]:
            say(f"classical GS check failed for {g['graph_id']}: {info}")
            sys.exit(1)
    json_dump(out / "classical_ground_states.json", classical_rows)

    largest = max(graphs, key=lambda g: g["n"])
    if largest["n"] >= 12 and not args.self_test:
        say(f"dense audit {largest['graph_id']} Gamma=1")
        edges = [tuple(e) for e in largest["edges"]]
        hc = model.cost_diagonal(largest["n"], edges, model.uniform_couplings(len(edges), 1.0))
        lan = model.ground_state(hc, 1.0, largest["n"])
        den = model.ground_state_dense(hc, 1.0, largest["n"])
        audit = {
            "graph_id": largest["graph_id"],
            "gamma": 1.0,
            "overlap": float(np.dot(lan["gs"], den["gs"]) ** 2),
            "E0_diff": abs(lan["evals0"] - den["evals0"]),
            "gap_diff": abs(lan["gap"] - den["gap"]),
            "Emax_diff": abs(lan["evals_max"] - den["evals_max"]),
            "lanczos_residual": lan["residual"],
            "emax_plus_e0": lan["emax_plus_e0"],
        }
        json_dump(out / "dense_audit_n12.json", audit)
        say(f"dense audit overlap={audit['overlap']:.12f} dE={audit['E0_diff']:.3e}")
        if audit["overlap"] < 1.0 - 1e-8 or audit["E0_diff"] > 1e-7:
            say("dense audit failed")
            sys.exit(1)

    survey_gammas = sorted(set([round(float(x), 2) for x in np.arange(0.2, 3.21, 0.2)] + [float(g) for g in GAMMAS]))
    if args.self_test:
        survey_gammas = [0.7, 1.0]
    say(f"ground-state survey gammas={survey_gammas}")
    survey_tasks = [{**g, "gammas": survey_gammas} for g in graphs]
    survey_rows = []
    if args.workers <= 1 or len(survey_tasks) == 1:
        for part in map(survey_one, survey_tasks):
            survey_rows.extend(part)
    else:
        import multiprocessing as mp
        ctx = mp.get_context("fork")
        with ctx.Pool(processes=args.workers) as pool:
            for part in pool.imap_unordered(survey_one, survey_tasks):
                survey_rows.extend(part)
    json_dump(out / "gamma_survey.json", survey_rows)
    write_csv(out / "gamma_survey.csv", survey_rows)
    say(f"survey rows={len(survey_rows)}")

    gammas = (0.7,) if args.self_test else GAMMAS
    max_L = 2 if args.self_test else args.max_L
    tasks = []
    for g in graphs:
        for gamma in gammas:
            tasks.append(
                {
                    **g,
                    "gamma": float(gamma),
                    "max_L": max_L,
                    "n_grid": 11 if args.self_test else args.grid,
                    "do_shared": (not args.no_shared) and (not args.self_test or True),
                }
            )
    # Prioritize N=10,12,8,6 so a partial log is useful, but submit all.
    priority = {10: 0, 12: 1, 8: 2, 6: 3}
    tasks.sort(key=lambda t: (priority.get(t["n"], 9), t["gamma"], t["seed"]))
    say(f"tasks={len(tasks)} workers={args.workers} max_L={max_L}")

    cells = []
    jsonl = out / "cells.jsonl"
    if jsonl.exists():
        jsonl.unlink()
    t_start = time.time()
    if args.workers <= 1 or len(tasks) == 1:
        iterator = map(run_cell, tasks)
        for i, cell in enumerate(iterator, start=1):
            cells.append(cell)
            with jsonl.open("a") as f:
                f.write(json.dumps(cell) + "\n")
            say(f"cell {i}/{len(tasks)} ok={cell.get('ok')} {cell.get('task')} {cell.get('seconds', 0):.1f}s")
    else:
        import multiprocessing as mp
        ctx = mp.get_context("fork")
        with ctx.Pool(processes=args.workers) as pool:
            for i, cell in enumerate(pool.imap_unordered(run_cell, tasks), start=1):
                cells.append(cell)
                with jsonl.open("a") as f:
                    f.write(json.dumps(cell) + "\n")
                say(f"cell {i}/{len(tasks)} ok={cell.get('ok')} {cell.get('task')} {cell.get('seconds', 0):.1f}s")
    say(f"sweep done in {time.time() - t_start:.1f}s")

    failures = collect_failures(cells)
    json_dump(out / "optimizer_flags.json", failures)
    spectra = [c["spectrum"] for c in cells if c.get("ok")]
    json_dump(out / "spectra.json", spectra)
    write_csv(out / "spectra.csv", spectra)
    flat = flatten_independent(cells)
    write_csv(out / "qaoa_independent.csv", flat)
    per_graph, med_rows = depth_separation(flat)
    # Attach classical F into med rows. depth_separation already has F_product etc.
    # F_classical is inside per_graph; add spread here.
    for row in med_rows:
        subset = [g for g in per_graph if g["n"] == row["n"] and g["gamma"] == row["gamma"]]
        # F_classical was not stored in per_graph. Pull from flat L=1.
        vals = [r["F_classical_gs"] for r in flat if r["n"] == row["n"] and r["gamma"] == row["gamma"] and r["L"] == 1]
        row["F_classical_gs"] = _spread(vals)
    scaling = scaling_diagnostic(med_rows)
    posthoc = posthoc_gamma(med_rows)
    summary = {
        "model": "sparse_uniform_sign_3regular_3body",
        "model_note": "Distinct-index uniform +1 couplings on 3-regular simple hypergraphs. Not collective p-spin. Not a spin glass.",
        "hamiltonian": "H = -sum_a Z_i Z_j Z_k - Gamma * sum_i X_i",
        "qaoa": {
            "form": "prod_l exp(-i beta_l H_X) exp(-i gamma_l H_C) |+>",
            "H_C": "-sum_a ZZZ_a",
            "H_X": "sum_i X_i",
            "objective": "full target energy <H_C - Gamma H_X>",
            "gamma_period": "pi",
            "beta_period": "pi",
            "beta_period_not": "pi/2",
            "duration": "sum |principal gamma| + sum |principal beta|, period pi folded into (-pi/2, pi/2]",
        },
        "n_tasks": len(tasks),
        "n_ok": sum(1 for c in cells if c.get("ok")),
        "n_failed_tasks": sum(1 for c in cells if not c.get("ok")),
        "wall_seconds": time.time() - t_start,
        "graphs": catalogs,
        "validation_pass": True,
        "n_optimizer_flags": len(failures),
        "median_rows": med_rows,
        "scaling": scaling,
        "posthoc_depth_separation": posthoc,
        "group_stats": group_stats(flat),
    }
    json_dump(out / "summary.json", summary)
    json_dump(out / "per_graph_fidelity.json", per_graph)
    # Angles, separate so a reader can replay without parsing the full ladder.
    angles = []
    for cell in cells:
        if not cell.get("ok"):
            continue
        for row in cell["ladder"]:
            if row["L"] == 0:
                continue
            if row["ansatz"] != "independent":
                continue
            angles.append(
                {
                    "graph_id": cell["task"]["graph_id"],
                    "n": cell["task"]["n"],
                    "gamma_field": cell["task"]["gamma"],
                    "L": row["L"],
                    "gammas_raw": row["gammas"],
                    "betas_raw": row["betas"],
                    "gammas_principal": row["gamma_principal"],
                    "betas_principal": row["beta_principal"],
                    "gamma_branch_k": row["gamma_branch_k"],
                    "beta_branch_k": row["beta_branch_k"],
                    "winner": row["winner"],
                    "duration_principal": row["duration_principal"],
                    "duration_raw": row["duration_raw"],
                    "F": row["F"],
                    "energy": row["energy"],
                    "eta": row["eta"],
                }
            )
    json_dump(out / "angles.json", angles)
    (out / "TABLES.md").write_text(tables_markdown(med_rows, per_graph, scaling, posthoc, spectra))
    if not args.self_test:
        plots = make_plots(flat, survey_rows, med_rows, out)
        say(f"plots {plots}")
    say(f"flags {len(failures)}")
    say(f"wrote {out}")


if __name__ == "__main__":
    main()
