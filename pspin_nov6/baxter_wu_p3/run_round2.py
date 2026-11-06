"""Round-2 QAOA for the staggered Baxter–Wu torus.

Run from this directory:

    OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python3 run_round2.py

The target is the full-space ground state. Geometry, gauge rank, and
coloring are written before the ladders.
"""

from __future__ import annotations

import csv
import json
import os
import sys
import time
import traceback
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np

THIS = Path(__file__).resolve().parent
if str(THIS) not in sys.path:
    sys.path.insert(0, str(THIS))

import geometry as geo
import model as M
import qaoa
import validate

# lambda = 1 staggered sign texture. Gammas sit where the N=12 ground state
# is entangled and not a computational product, with overlap to |+> still modest.
# N=18 is included at the two fields that bracket that window.
SMALL_CELLS = [
    {"lx": 3, "ly": 3, "lam": 1.0, "gamma": g, "max_L": 6, "n_grid": 25, "n_random": 6, "maxiter_hi": 40}
    for g in (0.5, 0.8, 1.0)
] + [
    {"lx": 3, "ly": 4, "lam": 1.0, "gamma": g, "max_L": 6, "n_grid": 25, "n_random": 6, "maxiter_hi": 40}
    for g in (0.7, 0.9, 1.1, 1.3)
]
LARGE_CELLS = [
    {"lx": 3, "ly": 6, "lam": 1.0, "gamma": g, "max_L": 5, "n_grid": 17, "n_random": 4, "maxiter_hi": 25}
    for g in (1.0, 1.3)
]


def json_dump(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2) + "\n")


def manifest(lx, ly, lam):
    geom = geo.build_triangles(lx, ly)
    col = geo.coloring_report(lx, ly, geom)
    gauge = geo.gauge_report(geom)
    sym = geo.symmetry_operators_report(lx, ly, geom, col)
    # drop non-serializable color arrays
    col = {k: v for k, v in col.items() if k != "face_coloring_detail" or isinstance(v, (str, int, float, bool, list, dict, type(None)))}
    if isinstance(col.get("face_coloring_detail"), dict):
        col["face_coloring_detail"] = {kk: vv for kk, vv in col["face_coloring_detail"].items() if kk != "colors"}
    tris, coeffs = M.faces_and_coeffs(geom, lam)
    hc = M.cost_diagonal(geom["n"], tris, coeffs)
    clas = M.classical_summary(hc)
    per = M.energy_step_and_period(hc)
    cut = M.spatial_cut(lx, ly)
    return {
        "lx": lx,
        "ly": ly,
        "n": geom["n"],
        "lam": lam,
        "up": [list(t) for t in geom["up"]],
        "down": [list(t) for t in geom["down"]],
        "M": geom["M"],
        "n_duplicate_up": geom["n_duplicate_up"],
        "n_duplicate_down": geom["n_duplicate_down"],
        "n_up_down_coincidences": geom["n_up_down_coincidences"],
        "both_periods_divisible_by_3": geom["both_divisible_by_3"],
        "coloring": col,
        "gauge": gauge,
        "symmetry": sym,
        "classical": clas,
        "period": per,
        "cut": cut,
        "model": "H = -sum_up ZZZ + lam * sum_down ZZZ - Gamma * sum X",
    }


def run_cell(task: dict) -> dict:
    t0 = time.time()
    try:
        lx, ly, lam = int(task["lx"]), int(task["ly"]), float(task["lam"])
        gamma = float(task["gamma"])
        geom = geo.build_triangles(lx, ly)
        n = geom["n"]
        tris, coeffs = M.faces_and_coeffs(geom, lam)
        hc = M.cost_diagonal(n, tris, coeffs)
        per = M.energy_step_and_period(hc)
        spec = M.ground_state(hc, gamma, n)
        cut = M.spatial_cut(lx, ly)
        ent = M.half_cut_entropy(spec["gs"], cut["keep"])
        clauses = M.clause_means(spec["gs"], n, geom)
        prod = M.best_product_overlap(spec["gs"], n, seed=lx * 100 + ly + int(gamma * 100))
        man = M.classical_manifold_overlaps(spec["gs"], hc)
        sym_ev = M.sublattice_eigenvalues(spec["gs"], lx, ly, geom)
        # symmetry eigenvalues may contain numpy arrays; keep floats only
        sym_ev = {k: (float(v) if isinstance(v, (float, np.floating)) else v) for k, v in sym_ev.items() if k != "colors"}
        mz = float(np.mean(M.z_expectations(spec["gs"], n)))
        ladder = qaoa.run_ladder(
            n=n,
            hc=hc,
            gamma_field=gamma,
            gs=spec["gs"],
            E0=spec["E0"],
            Emax=spec["Emax"],
            gamma_period=per["gamma_period"],
            max_L=int(task["max_L"]),
            n_grid=int(task["n_grid"]),
            n_random=int(task["n_random"]),
            maxiter_hi=int(task["maxiter_hi"]),
            do_shared_levels=None if n <= 12 else {1, 3, 5},
            seed=lx * 1000 + ly * 10 + int(round(gamma * 100)),
        )
        # replay energies
        plus = M.plus_state(n)
        replay = 0.0
        for row in ladder:
            if row["L"] == 0:
                continue
            psi = qaoa.apply_qaoa(plus, hc, n, row["gammas"], row["betas"])
            e = qaoa.energy_of(psi, hc, n, gamma)[0]
            replay = max(replay, abs(e - row["energy"]))
        return {
            "ok": True,
            "seconds": time.time() - t0,
            "replay_energy_abs_err": replay,
            "task": {"lx": lx, "ly": ly, "n": n, "lam": lam, "gamma": gamma},
            "spectrum": {
                "E0": spec["E0"],
                "E1": spec["E1"],
                "Emax": spec["Emax"],
                "gap": spec["gap"],
                "emax_plus_e0": spec["emax_plus_e0"],
                "residual": spec["residual"],
                "n_negative": spec["n_negative"],
                "degenerate": spec["degenerate"],
                "dim": spec["dim"],
                "S_bits": ent["S_bits"],
                "cut_kind": cut["kind"],
                "cut_keep": cut["keep"],
                "clause_up": clauses["up"],
                "clause_down": clauses["down"],
                "m_z": mz,
                "F_product": prod["F_product"],
                "F_plus": prod["F_plus"],
                "F_best_bit": prod["F_best_bit"],
                "F_classical_subspace": man["F_classical_subspace"],
                "F_equal_positive_cat": man["F_equal_positive_cat"],
                "classical_degeneracy": man["degeneracy"],
                "symmetry_eigenvalues": sym_ev,
                "gamma_period": per["gamma_period"],
                "cost_delta": per["delta"],
            },
            "ladder": ladder,
        }
    except Exception:
        return {"ok": False, "seconds": time.time() - t0, "error": traceback.format_exc(), "task": task}


def write_csv(path: Path, rows: list) -> None:
    if not rows:
        path.write_text("")
        return
    keys = []
    seen = set()
    for r in rows:
        for k in r:
            if k not in seen:
                keys.append(k)
                seen.add(k)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def flatten(cells):
    rows = []
    for cell in cells:
        if not cell.get("ok"):
            continue
        spec = cell["spectrum"]
        task = cell["task"]
        for row in cell["ladder"]:
            flat = {
                "n": task["n"],
                "lx": task["lx"],
                "ly": task["ly"],
                "lam": task["lam"],
                "gamma": task["gamma"],
                "L": row["L"],
                "F": row["F"],
                "energy": row["energy"],
                "eta": row["eta"],
                "gap": spec["gap"],
                "E0": spec["E0"],
                "Emax": spec["Emax"],
                "S_gs": spec["S_bits"],
                "F_product_gs": spec["F_product"],
                "F_plus_gs": spec["F_plus"],
                "F_best_bit_gs": spec["F_best_bit"],
                "F_subspace": spec["F_classical_subspace"],
                "F_cat": spec["F_equal_positive_cat"],
                "degeneracy": spec["classical_degeneracy"],
                "m_z_gs": spec["m_z"],
                "clause_up_gs": spec["clause_up"],
                "clause_down_gs": spec["clause_down"],
                "winner": row.get("winner"),
                "energy_nested": row.get("energy_nested"),
                "fidelity_monotone": row.get("fidelity_monotone"),
                "duration_principal": row.get("duration_principal"),
                "sum_abs_gamma_principal": row.get("sum_abs_gamma_principal"),
                "sum_abs_beta_principal": row.get("sum_abs_beta_principal"),
                "n_gamma_wrapped": row.get("n_gamma_wrapped"),
                "n_beta_wrapped": row.get("n_beta_wrapped"),
                "principal_state_overlap": row.get("principal_state_overlap"),
                "gamma_period": row.get("gamma_period"),
            }
            for k in (
                "l1_grid_max_fidelity",
                "l1_fidelity_max_F",
                "l1_F_gap_fidopt_minus_energyopt",
                "l1_energy_gap_energyopt_minus_fidopt",
                "l1_false_hard",
                "l1_energy_opt_beats_grid",
                "shared_F",
                "anneal_F",
                "anneal_energy",
            ):
                if k in row:
                    flat[k] = row[k]
            rows.append(flat)
    return rows


def make_plots(flat, out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {0: "#222222", 1: "#d62728", 2: "#1f77b4", 3: "#2ca02c", 4: "#17becf", 5: "#c51b8a", 6: "#ff7f0e"}
    markers = {0: "o", 1: "D", 2: "x", 3: "^", 4: "v", 5: "s", 6: "P"}
    # group by (n, gamma)
    keys = sorted({(r["n"], r["gamma"]) for r in flat})
    ncols = 4
    nrows = int(np.ceil(len(keys) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.4 * ncols, 3.0 * nrows), sharey=True)
    axes = np.atleast_1d(axes).ravel()
    for ax in axes[len(keys):]:
        ax.axis("off")
    for ax, (n, gamma) in zip(axes, keys):
        pts = [r for r in flat if r["n"] == n and r["gamma"] == gamma]
        Ls = sorted({int(r["L"]) for r in pts})
        ax.plot(Ls, [next(r["F"] for r in pts if int(r["L"]) == L) for L in Ls], color="#333333")
        for L in Ls:
            r = next(r for r in pts if int(r["L"]) == L)
            ax.scatter([L], [r["F"]], color=colors.get(L, "k"), marker=markers.get(L, "o"), s=30, zorder=3)
            if "anneal_F" in r:
                ax.scatter([L], [r["anneal_F"]], facecolors="none", edgecolors=colors.get(L, "k"), marker="o", s=40)
        l1 = next((r for r in pts if int(r["L"]) == 1), None)
        title = f"N={n} Γ={gamma:g}"
        if l1 and "l1_grid_max_fidelity" in l1:
            title += f"\nL1max={l1['l1_grid_max_fidelity']:.2f} F_E={l1['F']:.2f}"
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("L")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("F = |overlap|^2")
    fig.suptitle("Staggered Baxter–Wu λ=1, full-space GS. Open circles: matched anneal")
    fig.tight_layout()
    fig.savefig(out / "F_vs_L.png", dpi=140)
    plt.close(fig)

    # F vs 1/N at the two fields shared by N=9,12,18 if present: use gamma 1.0 and nearest.
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    gammas = sorted({r["gamma"] for r in flat})
    for gamma in gammas:
        for L, color in ((1, "#d62728"), (5, "#c51b8a")):
            pts = [r for r in flat if r["gamma"] == gamma and int(r["L"]) == L]
            if len(pts) < 2:
                continue
            pts.sort(key=lambda r: r["n"])
            ax.plot([1 / r["n"] for r in pts], [r["F"] for r in pts], marker="o", color=color, alpha=0.85, label=f"Γ={gamma:g} L={L}")
    ax.set_xlabel("1/N")
    ax.set_ylabel("F")
    ax.set_title("F vs 1/N by field and depth (one geometry per N)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / "F_vs_invN.png", dpi=140)
    plt.close(fig)


def main():
    out = THIS / "runs" / "round2"
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
        for row in report["failures"]:
            say(f"FAIL {row['name']} {row['detail']}")
        sys.exit(1)

    say("geometry manifest")
    geos = []
    for lx, ly in ((3, 3), (3, 4), (4, 3), (3, 6), (6, 3), (3, 7), (7, 3)):
        geos.append(manifest(lx, ly, 1.0))
        g = geos[-1]
        say(
            f"  {lx}x{ly} gauge_eq={g['gauge']['gauge_equivalent_to_uniform_ferro']} "
            f"coloring={g['coloring']['face_coloring_exists']} Z2xZ2={g['symmetry']['z2xz2_from_sublattice_flips']} "
            f"deg={g['classical']['degeneracy']} Emin={g['classical']['Emin']} period={g['period']['gamma_period']:.6f}"
        )
    json_dump(out / "geometries.json", geos)
    # explicit terms at Gamma=1 for the three QAOA geometries
    terms = []
    for lx, ly in ((3, 3), (3, 4), (3, 6)):
        terms.append({"lx": lx, "ly": ly, "lam": 1.0, "gamma_for_X": 1.0, "terms": M.term_list(lx, ly, 1.0, 1.0)})
    json_dump(out / "term_lists.json", terms)

    tasks = list(SMALL_CELLS)
    say(f"small cells {len(tasks)}")
    cells = []
    jsonl = out / "cells.jsonl"
    if jsonl.exists():
        jsonl.unlink()

    import multiprocessing as mp
    ctx = mp.get_context("fork")
    with ctx.Pool(processes=4) as pool:
        for i, cell in enumerate(pool.imap_unordered(run_cell, tasks), start=1):
            cells.append(cell)
            with jsonl.open("a") as f:
                f.write(json.dumps(cell) + "\n")
            if cell.get("ok"):
                l1 = next(r for r in cell["ladder"] if r["L"] == 1)
                ltop = cell["ladder"][-1]
                say(
                    f"cell {i}/{len(tasks)} N={cell['task']['n']} G={cell['task']['gamma']} "
                    f"L1 F={l1['F']:.3f} L1max={l1['l1_grid_max_fidelity']:.3f} Ffid={l1['l1_fidelity_max_F']:.3f} "
                    f"false_hard={l1['l1_false_hard']} L{ltop['L']} F={ltop['F']:.3f} "
                    f"S={cell['spectrum']['S_bits']:.2f} Fprod={cell['spectrum']['F_product']:.3f} "
                    f"{cell['seconds']:.1f}s"
                )
            else:
                say(f"cell {i} FAIL {cell.get('error')[-400:]}")

    # N=18 only after small cells succeeded. One geometry, two fields.
    if all(c.get("ok") for c in cells):
        say("N=18 cells")
        for task in LARGE_CELLS:
            cell = run_cell(task)
            cells.append(cell)
            with jsonl.open("a") as f:
                f.write(json.dumps(cell) + "\n")
            if cell.get("ok"):
                l1 = next(r for r in cell["ladder"] if r["L"] == 1)
                ltop = cell["ladder"][-1]
                say(
                    f"N=18 G={task['gamma']} L1 F={l1['F']:.3f} L1max={l1['l1_grid_max_fidelity']:.3f} "
                    f"Ffid={l1['l1_fidelity_max_F']:.3f} false_hard={l1['l1_false_hard']} "
                    f"L{ltop['L']} F={ltop['F']:.3f} {cell['seconds']:.1f}s"
                )
            else:
                say("N=18 FAIL " + cell.get("error", "")[-500:])

    flat = flatten(cells)
    write_csv(out / "qaoa.csv", flat)
    json_dump(out / "summary_cells.json", [{k: v for k, v in c.items() if k != "ladder"} | {"ladder_F": [(r["L"], r["F"], r["energy"]) for r in c.get("ladder", [])]} for c in cells])
    angles = []
    for c in cells:
        if not c.get("ok"):
            continue
        for r in c["ladder"]:
            if r["L"] == 0:
                continue
            angles.append({
                "n": c["task"]["n"], "lx": c["task"]["lx"], "ly": c["task"]["ly"],
                "gamma_field": c["task"]["gamma"], "L": r["L"],
                "gammas_raw": r["gammas"], "betas_raw": r["betas"],
                "gammas_principal": r["gamma_principal"], "betas_principal": r["beta_principal"],
                "gamma_branch_k": r["gamma_branch_k"], "beta_branch_k": r["beta_branch_k"],
                "F": r["F"], "energy": r["energy"], "winner": r["winner"],
                "duration_principal": r["duration_principal"],
            })
    json_dump(out / "angles.json", angles)
    if flat:
        make_plots(flat, out)
    say(f"wrote {out} ok={sum(1 for c in cells if c.get('ok'))}/{len(cells)}")


if __name__ == "__main__":
    main()
