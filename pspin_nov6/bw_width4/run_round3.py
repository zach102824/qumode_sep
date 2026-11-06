"""Width-4 Baxter–Wu stress test.

Geometries: 4×3 (orientation check against 3×4, not a second size) and 4×4.
Gamma grid {0.6, 0.8, 1.0, 1.2}. λ=1 staggered signs.
SQS / p=4 are not touched.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from geometry import (
    build_triangles,
    coloring_report,
    k_point_commensurate,
    lattice_isomorphism,
    validate_geometry,
)
from model import (
    classical_manifold,
    cost_diagonal,
    explicit_cost,
    gamma_period,
    ground_dense,
    ground_state,
    manifold_weight,
)
from observe import product_overlap, state_observables
from qaoa import apply_qaoa, energy_of, plus_state, run_ladder


GAMMAS = (0.6, 0.8, 1.0, 1.2)


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    if isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def write_terms(path: Path, geom: dict):
    lines = [
        f"# {geom['lx']}x{geom['ly']} N={geom['n']} staggered Baxter-Wu",
        "# up triangles, coefficient -1 in H_C",
    ]
    for t in geom["up"]:
        lines.append("up " + " ".join(map(str, t)))
    lines.append("# down triangles, coefficient +1 in H_C (lambda=1)")
    for t in geom["down"]:
        lines.append("down " + " ".join(map(str, t)))
    path.write_text("\n".join(lines) + "\n")


def spectrum_distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(np.sort(a) - np.sort(b))))


def orientation_report() -> dict:
    g34 = build_triangles(3, 4)
    g43 = build_triangles(4, 3)
    v34 = validate_geometry(g34)
    v43 = validate_geometry(g43)
    iso = lattice_isomorphism(g43, g34, lam=1.0)
    hc34 = cost_diagonal(g34, 1.0)
    hc43 = cost_diagonal(g43, 1.0)
    # full H at Gamma=1
    s34 = ground_dense(hc34, 1.0, 12)
    s43 = ground_dense(hc43, 1.0, 12)
    spec_h = spectrum_distance(s34["evals"], s43["evals"])
    spec_c = spectrum_distance(hc34, hc43)
    # explicit kronecker cross-check on 4x3
    err = float(np.max(np.abs(hc43 - explicit_cost(g43, 1.0))))
    return {
        "validate_3x4": v34,
        "validate_4x3": v43,
        "coloring_3x4": coloring_report(3, 4, g34),
        "coloring_4x3": coloring_report(4, 3, g43),
        "isomorphism_search": {k: v for k, v in iso.items()},
        "cost_spectrum_max_abs_diff": spec_c,
        "full_H_Gamma1_spectrum_max_abs_diff": spec_h,
        "E0_3x4": s34["E0"],
        "E0_4x3": s43["E0"],
        "explicit_vs_bit_diag_4x3": err,
        "same_physics": bool(spec_h < 1e-8),
        "note": (
            "4x3 is an orientation/implementation test of the 3x4 torus, not an independent size, "
            "when the full spectra agree."
        ),
    }


def corr_distance(a: dict, b: dict) -> float:
    keys = set(a) | set(b)
    if not keys:
        return 0.0
    return float(max(abs(a.get(k, 0.0) - b.get(k, 0.0)) for k in keys))


def enrich(row, psi, geom, hc, man_idx, gs_obs, E0, gap):
    obs = state_observables(psi, geom, hc, man_idx)
    r = (row["energy"] - E0) / gap if gap > 1e-12 else float("nan")
    row = dict(row)
    row.update({
        "mean_zzz_up": obs["mean_zzz_up"],
        "mean_zzz_down": obs["mean_zzz_down"],
        "stagger": obs["stagger"],
        "F_classical_manifold": obs["F_classical_manifold"],
        "S_primary_bits": obs["entropy"]["S_primary_bits"],
        "primary_cut": obs["entropy"]["primary_cut"],
        "S_min_cuts": obs["entropy"]["S_min_among_these_cuts"],
        "cut_entropies": obs["entropy"]["cuts"],
        "zz_connected": obs["zz_connected_by_distance"],
        "S_q0": obs["structure_factor"]["S_q0"],
        "S_max_q": obs["structure_factor"]["S_max_nonzero"],
        "K_on_grid": obs["structure_factor"]["K_on_momentum_grid"],
        "r_L": r,
        "gap_bound_1_minus_r": (1.0 - r) if r == r else float("nan"),
        "err_energy": row["energy"] - E0,
        "err_zzz_up": abs(obs["mean_zzz_up"] - gs_obs["mean_zzz_up"]),
        "err_zzz_down": abs(obs["mean_zzz_down"] - gs_obs["mean_zzz_down"]),
        "err_zz_connected": corr_distance(obs["zz_connected_by_distance"], gs_obs["zz_connected_by_distance"]),
        "err_manifold": abs(obs["F_classical_manifold"] - gs_obs["F_classical_manifold"]),
        "err_entropy": abs(obs["entropy"]["S_primary_bits"] - gs_obs["entropy"]["S_primary_bits"]),
    })
    # angles are lists; csv writer will skip them if we keep them only in json
    return row


def run_geometry(geom, gammas, max_L, n_grid, n_random, seed0):
    n = geom["n"]
    hc = cost_diagonal(geom, 1.0)
    per = gamma_period(hc)
    # period check
    phase = np.exp(-1j * per["T_gamma"] * hc)
    phase = phase / phase[0]
    if float(np.max(np.abs(phase - 1))) > 1e-8:
        raise RuntimeError(f"gamma period failed {geom['lx']}x{geom['ly']}")
    half = np.exp(-1j * 0.5 * per["T_gamma"] * hc)
    half = half / half[0]
    half_is_global = float(np.max(np.abs(half - 1))) < 1e-8
    man = classical_manifold(hc)
    rows = []
    angle_blob = []
    gs_cache = {}
    for i, G in enumerate(gammas):
        print(f"  GS {geom['lx']}x{geom['ly']} Gamma={G}", flush=True)
        gs = ground_state(hc, G, n)
        gs_cache[G] = gs
        gs_obs = state_observables(gs["gs"], geom, hc, man["indices"])
        gs_obs["F_product"] = product_overlap(gs["gs"], n)
        print(
            f"    E0={gs['E0']:.4f} gap={gs['gap']:.4f} Emax+E0={gs['Emax']+gs['E0']:.2e} "
            f"resid={gs['residual']:.1e} Fman={gs_obs['F_classical_manifold']:.3f} "
            f"S={gs_obs['entropy']['S_primary_bits']:.2f} prod={gs_obs['F_product']:.3f}",
            flush=True,
        )
        print(f"  QAOA L<={max_L}", flush=True)
        t0 = time.perf_counter()
        ladder, _fid = run_ladder(
            hc, n, G, gs["gs"], gs["E0"], gs["Emax"], per["T_gamma"],
            max_L=max_L, n_grid=n_grid, n_random=n_random, seed=seed0 + i * 17,
        )
        print(f"    ladder {time.perf_counter()-t0:.1f}s", flush=True)
        for rec in ladder:
            if rec["L"] == 0:
                psi = plus_state(n)
            else:
                psi = apply_qaoa(plus_state(n), hc, n, rec["gammas"], rec["betas"])
            enriched = enrich(rec, psi, geom, hc, man["indices"], gs_obs, gs["E0"], gs["gap"])
            enriched.update({
                "lx": geom["lx"], "ly": geom["ly"], "n": n, "Gamma": G,
                "Gamma_over_Gamma_star": None,  # filled after the grid
                "E0": gs["E0"], "E1": gs["E1"], "Emax": gs["Emax"], "gap": gs["gap"],
                "gs_residual": gs["residual"], "ed_method": gs["method"],
                "T_gamma": per["T_gamma"], "half_period_is_global_phase": half_is_global,
                "classical_degeneracy": man["degeneracy"], "classical_Emin": man["Emin"],
                "gs_F_product": gs_obs["F_product"],
                "gs_F_manifold": gs_obs["F_classical_manifold"],
                "gs_zzz_up": gs_obs["mean_zzz_up"],
                "gs_zzz_down": gs_obs["mean_zzz_down"],
                "gs_S_primary": gs_obs["entropy"]["S_primary_bits"],
                "gs_primary_cut": gs_obs["entropy"]["primary_cut"],
                "K_commensurate": k_point_commensurate(geom["lx"], geom["ly"])["K_on_momentum_grid"],
            })
            rows.append(enriched)
            angle_blob.append({
                "lx": geom["lx"], "ly": geom["ly"], "Gamma": G, "L": rec["L"],
                "winner": rec.get("winner"),
                "gammas": rec.get("gammas"), "betas": rec.get("betas"),
                "gammas_canonical": rec.get("gammas_canonical"),
                "betas_canonical": rec.get("betas_canonical"),
                "F": rec["F"], "energy": rec["energy"],
                "l1_fidelity_max_F": rec.get("l1_fidelity_max_F"),
                "false_hard": rec.get("false_hard"),
                "anneal_F": rec.get("anneal_F"),
                "shared_F": rec.get("shared_F"),
                "replay_energy_err": rec.get("replay_energy_err"),
                "replay_F_err": rec.get("replay_F_err"),
            })
    # Gamma* = grid point with the smallest gap. Not a critical-point claim.
    gaps = {}
    for r in rows:
        gaps[r["Gamma"]] = r["gap"]
    g_star = min(gaps, key=gaps.get)
    for r in rows:
        r["Gamma_star"] = g_star
        r["Gamma_over_Gamma_star"] = r["Gamma"] / g_star
    return rows, angle_blob, {"Gamma_star": g_star, "gaps": gaps, "T_gamma": per["T_gamma"],
                              "half_period_is_global_phase": half_is_global,
                              "classical_degeneracy": man["degeneracy"], "classical_Emin": man["Emin"]}


def write_csv(path, rows):
    import csv
    skip_types = (list, dict)
    keys = []
    for r in rows:
        for k, v in r.items():
            if k not in keys and not isinstance(v, skip_types):
                keys.append(k)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            out = {}
            for k in keys:
                v = r.get(k, "")
                if isinstance(v, float):
                    out[k] = f"{v:.8g}"
                elif isinstance(v, (bool, np.bool_)):
                    out[k] = "1" if v else "0"
                else:
                    out[k] = v
            w.writerow(out)


def make_plots(out: Path, rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figdir = out / "figures"
    figdir.mkdir(exist_ok=True)

    def series(lx, ly):
        return [r for r in rows if r["lx"] == lx and r["ly"] == ly and "F" in r]

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.0), sharey=True)
    for ax, (lx, ly) in zip(axes, ((4, 3), (4, 4))):
        sub = series(lx, ly)
        gammas = sorted({r["Gamma"] for r in sub})
        for G in gammas:
            rr = sorted([r for r in sub if r["Gamma"] == G], key=lambda r: r["L"])
            ax.plot([r["L"] for r in rr], [r["F"] for r in rr], marker="o", label=f"Γ={G:g}")
        ax.set_title(f"{lx}×{ly}")
        ax.set_xlabel("L")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, frameon=False)
    axes[0].set_ylabel("energy-optimal F")
    fig.suptitle("Staggered BW, independent-angle QAOA")
    fig.tight_layout()
    fig.savefig(figdir / "F_vs_L.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for lx, ly in ((4, 3), (4, 4)):
        sub = [r for r in series(lx, ly) if r["L"] >= 1]
        # one point per gamma at max L and at L=1
        for G in sorted({r["Gamma"] for r in sub}):
            r1 = next(r for r in sub if r["Gamma"] == G and r["L"] == 1)
            ax.scatter([G], [r1.get("l1_fidelity_max_F", np.nan)], marker="x", color="C0")
            ax.scatter([G], [r1["F"]], marker="o", color="C1")
        ax.plot([], [], "x", color="C0", label=f"{lx}×{ly} F1max" if lx == 4 and ly == 3 else None)
    # clearer grouped plot
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8), sharey=True)
    for ax, (lx, ly) in zip(axes, ((4, 3), (4, 4))):
        sub = [r for r in series(lx, ly) if r["L"] == 1]
        sub = sorted(sub, key=lambda r: r["Gamma"])
        g = [r["Gamma"] for r in sub]
        ax.plot(g, [r["F"] for r in sub], "o-", label="F1 energy")
        ax.plot(g, [r.get("l1_fidelity_max_F", np.nan) for r in sub], "x--", label="F1 max")
        deep = []
        for G in g:
            rr = [r for r in series(lx, ly) if r["Gamma"] == G]
            deep.append(max(r["F"] for r in rr))
        ax.plot(g, deep, "s-", label="best energy-opt F")
        ax.set_title(f"{lx}×{ly}")
        ax.set_xlabel("Γ")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, frameon=False)
    axes[0].set_ylabel("F")
    fig.tight_layout()
    fig.savefig(figdir / "F1_vs_deep.png", dpi=140)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8), sharey=False)
    for ax, (lx, ly) in zip(axes, ((4, 3), (4, 4))):
        for G in GAMMAS:
            rr = sorted([r for r in series(lx, ly) if r["Gamma"] == G and r["L"] >= 1], key=lambda r: r["L"])
            if not rr:
                continue
            ax.plot([r["L"] for r in rr], [r["err_energy"] for r in rr], marker="o", label=f"Γ={G:g}")
        ax.set_title(f"{lx}×{ly} energy residual E−E0")
        ax.set_xlabel("L")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(figdir / "energy_residual_vs_L.png", dpi=140)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8), sharey=True)
    for ax, (lx, ly) in zip(axes, ((4, 3), (4, 4))):
        for G in GAMMAS:
            rr = sorted([r for r in series(lx, ly) if r["Gamma"] == G and r["L"] >= 1 and "anneal_F" in r], key=lambda r: r["L"])
            if not rr:
                continue
            ax.plot([r["L"] for r in rr], [r["F"] for r in rr], marker="o", label=f"QAOA Γ={G:g}")
            ax.plot([r["L"] for r in rr], [r["anneal_F"] for r in rr], marker="x", ls="--", label=f"anneal Γ={G:g}")
        ax.set_title(f"{lx}×{ly}")
        ax.set_xlabel("L")
        ax.set_ylim(-0.02, 1.02)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=6.5, frameon=False, ncol=2)
    axes[0].set_ylabel("F")
    fig.suptitle("Duration-matched anneal vs energy-QAOA (comparison only)")
    fig.tight_layout()
    fig.savefig(figdir / "anneal_vs_qaoa.png", dpi=140)
    plt.close(fig)


def main():
    t0 = time.perf_counter()
    out = Path(__file__).resolve().parent / "results" / "round3"
    out.mkdir(parents=True, exist_ok=True)
    print("orientation 4x3 vs 3x4", flush=True)
    ori = orientation_report()
    print(json.dumps({k: ori[k] for k in (
        "same_physics", "cost_spectrum_max_abs_diff", "full_H_Gamma1_spectrum_max_abs_diff",
        "E0_3x4", "E0_4x3", "explicit_vs_bit_diag_4x3",
    )}, indent=2), flush=True)
    print("iso", ori["isomorphism_search"], flush=True)
    g44 = build_triangles(4, 4)
    v44 = validate_geometry(g44)
    c44 = coloring_report(4, 4, g44)
    if not v44["ok"]:
        raise SystemExit(f"4x4 geometry failed {v44}")
    for geom, name in ((build_triangles(4, 3), "4x3"), (g44, "4x4"), (build_triangles(3, 4), "3x4")):
        write_terms(out / f"terms_{name}.txt", geom)
    (out / "orientation.json").write_text(json.dumps(_jsonable({
        **ori,
        "validate_4x4": v44,
        "coloring_4x4": c44,
        "K_4x3": k_point_commensurate(4, 3),
        "K_4x4": k_point_commensurate(4, 4),
    }), indent=2))

    all_rows = []
    angles = []
    summaries = {}
    # 4x3 first (N=12), then 4x4
    jobs = [
        (build_triangles(4, 3), 6, 25, 4, 11),
        (build_triangles(4, 4), 5, 21, 3, 29),
    ]
    for geom, max_L, n_grid, n_random, seed in jobs:
        print(f"=== {geom['lx']}x{geom['ly']} L<={max_L}", flush=True)
        rows, ang, summary = run_geometry(geom, GAMMAS, max_L, n_grid, n_random, seed)
        all_rows.extend(rows)
        angles.extend(ang)
        summaries[f"{geom['lx']}x{geom['ly']}"] = summary
        write_csv(out / "qaoa.csv", all_rows)
        (out / "angles.json").write_text(json.dumps(_jsonable(angles), indent=2))
    (out / "summaries.json").write_text(json.dumps(_jsonable(summaries), indent=2))
    # strip bulky nested dicts already flattened
    write_csv(out / "qaoa.csv", all_rows)
    (out / "rows.json").write_text(json.dumps(_jsonable(all_rows), indent=2))
    make_plots(out, all_rows)
    meta = {"wall_s": time.perf_counter() - t0, "gammas": list(GAMMAS), "numpy": np.__version__}
    (out / "run_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"done in {meta['wall_s']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
