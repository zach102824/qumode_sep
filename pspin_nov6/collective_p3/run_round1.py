"""Bounded round-1 QAOA search: collective ferromagnetic p=3.

Primary objective is the energy of H = -N m_z^3 - Gamma sum X.
Fidelity is recorded and, at L=1 only, maximized separately as an optimizer diagnostic.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import pauli_enum as pe
import symmetric as sym
from qaoa_search import (
    l1_grid,
    matched_digitized_anneal,
    search_depth,
    shared_gamma_search,
    sign_flip_fidelity,
    trotter_sweep,
)
from validate import run_validation

# Gates fixed before the search, taken from the task brief.
# Low L1: squared GS overlap in [0.02, 0.25] at N=8,10,12.
# Material gain: F(L=5)-F(L=1) >= 0.10 with F increasing from L=1 to L=3 and L=5.
LOW_L1 = (0.02, 0.25)
TOO_EASY_F1 = 0.50
MATERIAL_DELTA = 0.10
BASE_GAMMAS = (0.6, 0.8, 1.0, 1.15, 1.25, 1.35)


def physics_row(n: int, gamma: float) -> dict:
    model = sym.SymmetricModel(n, gamma, cost_sign=-1.0)
    obs = sym.observe(model, model.gs)
    c1 = float(pe.pauli_coefficients(n)[1])
    field_energy = c1 * float(np.sum((model.gs ** 2) * model.mz))
    # Distinct-index diagonal inside the same symmetric irrep.
    e_dist = model.e_cost - c1 * model.mz
    h_dist = np.diag(e_dist) - gamma * model.sx
    ev_dist = np.linalg.eigvalsh(h_dist)
    return {
        "N": n,
        "Gamma": gamma,
        "E0": model.e0,
        "Emax": model.emax,
        "E0_sym": model.e0_sym,
        "Emax_sym": model.emax_sym,
        "gap_sym": model.gap_sym,
        "gap_full": model.gap_full,
        "gs_sector": model.e0_sector,
        "gs_in_max_spin": bool(model.gs_in_max_spin),
        "mz": obs["mz"],
        "mx": obs["mx"],
        "entropy_bits": obs["entropy_bits"],
        "F_plus": obs["F_plus"],
        "F_coherent": obs["F_coherent"],
        "F_bitstring": obs["F_bitstring"],
        "field_energy": field_energy,
        "E0_distinct_sym": float(ev_dist[0]),
        "delta_E0_sym_coll_minus_dist": model.e0_sym - float(ev_dist[0]),
    }


def choose_gammas(rows: list[dict]) -> list[float]:
    chosen = set(BASE_GAMMAS)
    for n in (8, 12, 20):
        sub = [r for r in rows if r["N"] == n]
        if not sub:
            continue
        gap_row = min(sub, key=lambda r: r["gap_full"])
        chosen.add(round(float(gap_row["Gamma"]), 3))
        ordered = [r for r in sub if r["mz"] >= 0.5]
        if ordered:
            edge = max(ordered, key=lambda r: r["Gamma"])
            chosen.add(round(float(edge["Gamma"]), 3))
    # Drop near-duplicates (within 0.025) in favor of the base grid.
    base = set(BASE_GAMMAS)
    extras = sorted(g for g in chosen if g not in base)
    kept = list(BASE_GAMMAS)
    for g in extras:
        if all(abs(g - h) > 0.025 for h in kept):
            kept.append(g)
    return sorted(kept)


def _jsonable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items() if k != "psi"}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    return obj


def run_one(n: int, gamma: float, seed: int, do_shared: bool, max_depth: int = 5) -> tuple[list[dict], list[dict], list[str]]:
    t0 = time.perf_counter()
    model = sym.SymmetricModel(n, gamma, cost_sign=-1.0)
    model_af = sym.SymmetricModel(n, gamma, cost_sign=+1.0)
    _g, period = sym.gamma_period(n)
    rng = np.random.default_rng(seed)
    # Grid density: about 0.05 in gamma across one full period, plus a focus window.
    n_gamma = int(min(12000, max(401, round(period / 0.05))))
    n_beta = 31
    grid = l1_grid(model, period, n_gamma=n_gamma, n_beta=n_beta, gamma_focus=6.0) if max_depth >= 1 else None
    rows: list[dict] = []
    angles: list[dict] = []
    notes: list[str] = []
    prev = None
    best_by_depth: dict[int, dict] = {}
    for depth in range(0, max_depth + 1):
        out = search_depth(
            model,
            depth,
            rng,
            period,
            prev=prev,
            transfer=None,
            grid_info=grid,
            maxiter=180,
        )
        prev = out
        best_by_depth[depth] = out
        sf = sign_flip_fidelity(model, model_af, out["gamma_raw"], out["beta_raw"])
        anneal = matched_digitized_anneal(model, out) if depth >= 1 else None
        trotter = trotter_sweep(model, depth) if depth >= 1 else None
        shared = shared_gamma_search(model, depth, rng, out, maxiter=140) if do_shared and depth >= 1 else None
        if sf["F_abs_diff"] > 1e-8 or sf["E0_abs_diff"] > 1e-8:
            notes.append(f"SIGN FLIP FAIL N={n} G={gamma} L={depth} dF={sf['F_abs_diff']}")
        replay_ov = None
        if n <= 12 and depth in (1, max_depth) and depth >= 1:
            cost = pe.enumerated_cost_diagonal(n)
            psi0 = np.full(1 << n, 2.0 ** (-0.5 * n), dtype=np.complex128)
            psi_full = pe.evolve_statevector(psi0, n, out["gamma_raw"], out["beta_raw"], cost)
            emb = pe.embed_dicke(out["psi"], n)
            replay_ov = float(abs(np.vdot(psi_full, emb)))
            if abs(replay_ov - 1.0) > 1e-8:
                notes.append(f"REPLAY FAIL N={n} G={gamma} L={depth} ov={replay_ov}")
        fid_diag = out["fid_diag"]
        row = {
            "N": n,
            "Gamma": gamma,
            "L": depth,
            "ansatz": "independent",
            "energy": out["energy"],
            "E0": model.e0,
            "Emax": model.emax,
            "eta": out["eta"],
            "F": out["F"],
            "gap_sym": model.gap_sym,
            "gap_full": model.gap_full,
            "mz": out["mz"],
            "mx": out["mx"],
            "entropy_bits": out["entropy_bits"],
            "entropy_cut": f"qubits[0,{n // 2})",
            "F_plus": out["F_plus"],
            "F_coherent": out["F_coherent"],
            "F_bitstring": out["F_bitstring"],
            "sum_abs_gamma": out["sum_abs_gamma"],
            "sum_abs_beta": out["sum_abs_beta"],
            "sum_abs_gamma_W": out["sum_abs_gamma"] / float(n * n),
            "sum_abs_beta_W": out["sum_abs_beta"],
            "winner_source": out["source"],
            "canon_notes": "|".join(out["canon_notes"]) if out["canon_notes"] else "",
            "conj_energy_err": out["conj_energy_err"],
            "conj_F_err": out["conj_F_err"],
            "n_starts": out["n_starts"],
            "n_near": out["n_near"],
            "F_std": out["F_std"],
            "F_min": out["F_min"],
            "F_max": out["F_max"],
            "F_near_std": out["F_near_std"],
            "E_std": out["E_std"],
            "signflip_dF": sf["F_abs_diff"],
            "replay_overlap": replay_ov if replay_ov is not None else "",
            "F_L1_fidelity_opt": fid_diag["F"] if fid_diag else "",
            "E_L1_fidelity_opt": fid_diag["energy"] if fid_diag else "",
            "anneal_matched_F": None if anneal is None else anneal["F"],
            "anneal_matched_energy": None if anneal is None else anneal["energy"],
            "anneal_matched_beta_sign": None if anneal is None else anneal["beta_sign"],
            "trotter_best_F": None if trotter is None else trotter["F"],
            "trotter_best_energy": None if trotter is None else trotter["energy"],
            "trotter_best_tau": None if trotter is None else trotter["tau"],
            "shared_F": None if shared is None else shared["F"],
            "shared_energy": None if shared is None else shared["energy"],
            "shared_source": None if shared is None else shared["source"],
            "shared_F_std": None if shared is None else shared["F_std"],
            "gamma_period": period,
            "seconds": time.perf_counter() - t0,
        }
        rows.append(row)
        ang = {
            "N": n,
            "Gamma": gamma,
            "L": depth,
            "ansatz": "independent",
            "winner_source": out["source"],
            "canon_notes": out["canon_notes"],
            "gamma_raw": out["gamma_raw"],
            "beta_raw": out["beta_raw"],
            "gamma_can": out["gamma_can"],
            "beta_can": out["beta_can"],
            "gamma_W_raw": out["gamma_raw"] / float(n * n),
            "beta_W_raw": -out["beta_raw"],
            "mixer": "exp(-i beta sum_i X_i); beta_W = -beta uses generator -sum X",
            "cost": "exp(-i gamma H_cost), H_cost=-(sum Z)^3/N^2; gamma_W=gamma/N^2 uses -(sum Z)^3",
            "sum_abs_gamma": out["sum_abs_gamma"],
            "sum_abs_beta": out["sum_abs_beta"],
        }
        if fid_diag and depth == 1:
            ang["fidelity_opt_diagnostic"] = fid_diag
        if anneal is not None:
            ang["matched_anneal_gamma"] = anneal["gamma"]
            ang["matched_anneal_beta"] = anneal["beta"]
        if shared is not None:
            ang["shared_gamma_raw"] = shared["gamma_raw"]
            ang["shared_beta_raw"] = shared["beta_raw"]
        angles.append(_jsonable(ang))
        # Drop the state before the next depth to limit memory.
        out.pop("psi", None)
        prev = out
    notes.append(f"done N={n} Gamma={gamma} in {time.perf_counter() - t0:.1f}s")
    return rows, angles, notes


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    keys: list[str] = []
    for row in rows:
        for k in row:
            if k not in keys:
                keys.append(k)
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def plot_results(rows: list[dict], physics: list[dict], out: Path) -> list[str]:
    made = []
    ind = [r for r in rows if r["ansatz"] == "independent"]
    gammas = sorted({float(r["Gamma"]) for r in ind})
    # Finite-size phase diagram.
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2))
    for n in sorted({int(r["N"]) for r in physics}):
        sub = sorted((r for r in physics if int(r["N"]) == n), key=lambda r: r["Gamma"])
        g = [r["Gamma"] for r in sub]
        axes[0].plot(g, [r["mz"] for r in sub], label=f"N={n}")
        axes[1].semilogy(g, [max(r["gap_full"], 1e-16) for r in sub], label=f"N={n}")
        axes[2].plot(g, [r["F_plus"] for r in sub], label=f"N={n}")
    axes[0].set_ylabel(r"GS $\langle m_z\rangle$")
    axes[1].set_ylabel("full-H gap")
    axes[2].set_ylabel(r"GS $|\langle+|GS\rangle|^2$")
    for ax in axes:
        ax.set_xlabel(r"$\Gamma$")
        ax.axvline(sym.mean_field_gamma_c(), color="k", ls="--", lw=0.8)
    axes[0].legend(fontsize=7, ncol=2)
    fig.tight_layout()
    p = out / "gs_phase_diagram.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    made.append(str(p))

    # F vs 1/N, one panel per Gamma that was searched.
    n_g = len(gammas)
    ncols = min(3, n_g)
    nrows = int(math.ceil(n_g / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.4 * ncols, 2.8 * nrows), squeeze=False)
    for i, g in enumerate(gammas):
        ax = axes[i // ncols][i % ncols]
        sub = [r for r in ind if abs(float(r["Gamma"]) - g) < 1e-9]
        for depth in range(0, 6):
            pts = sorted((r for r in sub if int(r["L"]) == depth), key=lambda r: r["N"])
            if not pts:
                continue
            x = [1.0 / r["N"] for r in pts]
            y = [r["F"] for r in pts]
            # Marker is the fidelity at the best energy. The bar is the full range of
            # local-search fidelities, including worse-energy minima.
            lo = [max(0.0, r["F"] - r["F_min"]) for r in pts]
            hi = [max(0.0, r["F_max"] - r["F"]) for r in pts]
            ax.errorbar(x, y, yerr=[lo, hi], marker="o", ms=3.5, lw=1.0, capsize=2, label=f"L={depth}")
        ax.set_title(rf"$\Gamma$={g:g}")
        ax.set_xlabel("1/N")
        ax.set_ylabel(r"$F=|\langle\psi|GS\rangle|^2$")
        ax.set_ylim(-0.02, 1.02)
    for j in range(n_g, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", fontsize=8)
    fig.suptitle("Marker: F at the best energy. Bar: min to max F over local searches.", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    p = out / "F_vs_invN_by_depth.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    made.append(str(p))

    fig, axes = plt.subplots(nrows, ncols, figsize=(3.4 * ncols, 2.8 * nrows), squeeze=False)
    for i, g in enumerate(gammas):
        ax = axes[i // ncols][i % ncols]
        sub = [r for r in ind if abs(float(r["Gamma"]) - g) < 1e-9]
        for n in sorted({int(r["N"]) for r in sub}):
            pts = sorted((r for r in sub if int(r["N"]) == n), key=lambda r: r["L"])
            x = [r["L"] / r["N"] for r in pts]
            y = [r["F"] for r in pts]
            lo = [max(0.0, r["F"] - r["F_min"]) for r in pts]
            hi = [max(0.0, r["F_max"] - r["F"]) for r in pts]
            ax.errorbar(x, y, yerr=[lo, hi], marker="o", ms=3.5, lw=1.0, capsize=2, label=f"N={n}")
        ax.set_title(rf"$\Gamma$={g:g}")
        ax.set_xlabel("L/N")
        ax.set_ylabel(r"$F$")
        ax.set_ylim(-0.02, 1.02)
    for j in range(n_g, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", fontsize=8)
    fig.suptitle("Marker: F at the best energy. Bar: min to max F over local searches.", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    p = out / "F_vs_L_over_N.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    made.append(str(p))
    return made


def score_gates(rows: list[dict]) -> dict:
    ind = [r for r in rows if r["ansatz"] == "independent"]
    points = []
    for g in sorted({float(r["Gamma"]) for r in ind}):
        for n in (8, 10, 12):
            by_l = {int(r["L"]): r for r in ind if int(r["N"]) == n and abs(float(r["Gamma"]) - g) < 1e-9}
            if not {1, 3, 5}.issubset(by_l):
                continue
            f1 = float(by_l[1]["F"])
            f3 = float(by_l[3]["F"])
            f5 = float(by_l[5]["F"])
            low = LOW_L1[0] <= f1 <= LOW_L1[1]
            material = (f5 - f1) >= MATERIAL_DELTA and f3 > f1 and f5 > f1
            points.append(
                {
                    "N": n,
                    "Gamma": g,
                    "F1": f1,
                    "F3": f3,
                    "F5": f5,
                    "dF_5_1": f5 - f1,
                    "low_L1": low,
                    "too_easy": f1 > TOO_EASY_F1,
                    "material_gain": material,
                    "pass_point": low and material,
                    "F_not_monotone": any(float(by_l[L]["F"]) + 1e-9 < float(by_l[L - 1]["F"]) for L in range(1, 6) if L in by_l and (L - 1) in by_l),
                    "energy_rose": any(float(by_l[L]["energy"]) > float(by_l[L - 1]["energy"]) + 1e-6 for L in range(1, 6) if L in by_l and (L - 1) in by_l),
                }
            )
    # Overall: a single Gamma passes at every N in {8,10,12}.
    winners = []
    for g in sorted({p["Gamma"] for p in points}):
        ok = {p["N"] for p in points if p["Gamma"] == g and p["pass_point"]}
        if ok == {8, 10, 12}:
            winners.append(g)
    if winners:
        verdict = "PASS"
        reason = f"low L1 and material L=3..5 gain at N=8,10,12 for Gamma in {winners}"
    else:
        # Partial if each size has some Gamma that passes, or two sizes share one.
        per_n = {n: [p for p in points if p["N"] == n and p["pass_point"]] for n in (8, 10, 12)}
        if all(per_n.values()):
            verdict = "PARTIAL"
            reason = "each of N=8,10,12 has some Gamma passing, but no common Gamma passes all three"
        elif sum(bool(v) for v in per_n.values()) >= 2:
            verdict = "PARTIAL"
            reason = "the low-L1 plus material-gain gate holds at two of N=8,10,12, not all three on one Gamma"
        else:
            verdict = "FAIL"
            reason = "low L1 plus material L3-L5 gain does not hold across N=8,10,12"
    return {"verdict": verdict, "reason": reason, "points": points, "passing_gammas": winners}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "results")
    ap.add_argument("--pilot", action="store_true", help="one (N,Gamma) timing run")
    ap.add_argument("--skip-validate", action="store_true")
    ap.add_argument("--max-depth", type=int, default=5)
    args = ap.parse_args()
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)
    log_lines: list[str] = []
    if not args.skip_validate:
        log_lines.extend(run_validation())
    if args.pilot:
        t0 = time.perf_counter()
        rows, angles, notes = run_one(8, 1.0, seed=1, do_shared=True, max_depth=args.max_depth)
        print(f"PILOT {time.perf_counter() - t0:.1f}s", flush=True)
        for r in rows:
            print(
                f"L={r['L']} F={r['F']:.4f} E={r['energy']:.4f} eta={r['eta']:.3f} "
                f"src={r['winner_source']} dFsig={r['signflip_dF']:.1e} replay={r['replay_overlap']}",
                flush=True,
            )
        print("\n".join(notes))
        return

    sizes = [6, 8, 10, 12, 16, 20, 32]
    fine = [round(x, 3) for x in np.linspace(0.4, 1.7, 27)]
    physics = []
    for n in sizes:
        for g in fine:
            physics.append(physics_row(n, g))
            if not physics[-1]["gs_in_max_spin"]:
                log_lines.append(f"WARNING GS left max spin N={n} Gamma={g} sector={physics[-1]['gs_sector']}")
    write_csv(out / "gs_scan.csv", physics)
    (out / "gs_scan.json").write_text(json.dumps(physics, indent=2), encoding="utf-8")
    gammas = choose_gammas(physics)
    log_lines.append("QAOA Gammas: " + ", ".join(f"{g:.3f}" for g in gammas))
    print(log_lines[-1], flush=True)

    for n in (6, 8, 10, 12):
        pe.write_term_list(str(out / f"hamiltonian_terms_N{n}.csv"), n)

    all_rows: list[dict] = []
    all_angles: list[dict] = []
    # Sequential over N so a crash keeps earlier sizes; Gamma loop is the inner cost.
    # Four processes, one per Gamma batch.
    import multiprocessing as mp

    for n in sizes:
        # Shared-gamma diagnostic on sizes where the GS is not already |+>, and always for N<=12.
        tasks = []
        for g in gammas:
            nearest = min((r for r in physics if r["N"] == n), key=lambda r: abs(r["Gamma"] - g))
            do_shared = n <= 12 or float(nearest["F_plus"]) <= 0.3
            seed = 20261106 + 1000 * n + int(round(g * 1000))
            tasks.append((n, g, seed, do_shared, args.max_depth))
        print(f"starting N={n} ({len(tasks)} Gammas)", flush=True)
        ctx = mp.get_context("fork")
        with ctx.Pool(processes=min(4, len(tasks))) as pool:
            results = pool.starmap(run_one, tasks)
        for rows, angles, notes in results:
            all_rows.extend(rows)
            all_angles.extend(angles)
            log_lines.extend(notes)
            print(notes[-1], flush=True)
        write_csv(out / "qaoa_round1.csv", all_rows)
        (out / "angles.json").write_text(json.dumps(all_angles, indent=2), encoding="utf-8")
        (out / "verification.log").write_text("\n".join(log_lines) + "\n", encoding="utf-8")

    made = plot_results(all_rows, physics, out)
    gate = score_gates(all_rows)
    (out / "gate.json").write_text(json.dumps(gate, indent=2), encoding="utf-8")
    log_lines.append("plots: " + ", ".join(made))
    log_lines.append(f"VERDICT {gate['verdict']}: {gate['reason']}")
    (out / "verification.log").write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    print(log_lines[-1], flush=True)


if __name__ == "__main__":
    main()
