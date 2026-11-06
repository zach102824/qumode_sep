"""Round-2 SQS sign scan. Reproducible. Does not claim covering-radius
optimality except for the exhaustive N=8 certificate.

Gamma list and which points get L=1..6 are fixed in this file from the
design (small frustration vs uniform vs certified N=8 covering radius),
not from a fidelity cutoff. N=14 and N=16 are built and their frustration
is certified; they are not given QAOA because the N=8 and N=10 screens
are cat or L=2..4 saturated. That decision is recorded after the screen,
in the output, and is not a scaling claim.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from ham import (
    dense_spectrum,
    energy_of,
    explicit_hc_from_pauli,
    f_overlap,
    flip_expectation,
    gamma_period_from_diag,
    ground_pair,
    hc_diag_for_pattern,
    observables,
    plus_state,
    signs_pm_from_eps,
)
from qaoa_sqs import Counter, eta, optimize_depth, optimize_l1, optimize_shared
from signs import prepare_catalog
from sqs import design_blocks, design_name, sqs_parameters, verify_sqs

# Per-degree Gamma. Gamma_X = r * Gamma is the coefficient of -sum X.
GAMMAS = (0.02, 0.05, 0.10, 0.15, 0.25, 0.40, 0.80, 1.50)
SCAN_N8 = ("uniform", "block_0", "pair_0_1", "affine_off_origin")
SCAN_N10 = ("uniform", "block_0", "pair_0_1")
# Deep curves: ordered side of each family, where a cat or a code-superposition
# might still be the ground state. Chosen from the sign list, not from F.
DEEP = (
    (8, "uniform", 0.05),
    (8, "block_0", 0.05),
    (8, "pair_0_1", 0.02),
    (8, "pair_0_1", 0.05),
    (10, "uniform", 0.10),
    (10, "block_0", 0.10),
    (10, "pair_0_1", 0.10),
)


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (bytes, bytearray)):
        return obj.hex()
    return obj


def save_blocks(out: Path, n: int, blocks):
    lines = [
        f"# SQS({n}) {design_name(n)}",
        f"# b={len(blocks)} r={sqs_parameters(n)['r']}",
        "# one block per line, point labels 0..N-1",
    ]
    for b in blocks:
        lines.append(" ".join(str(x) for x in b))
    (out / f"blocks_N{n}.txt").write_text("\n".join(lines) + "\n")


def pattern_row(cat, name: str) -> dict:
    hits = [c for c in cat["catalog"] if c["name"] == name]
    if not hits:
        # block_0 always exists; pair and uniform too
        raise KeyError(name)
    c = hits[0]
    return c


def screen_point(cat, name, gamma, rng_seed: int) -> dict:
    c = pattern_row(cat, name)
    n = cat["N"]
    blocks = cat["blocks"]
    diag = hc_diag_for_pattern(n, blocks, c["eps"])
    # independent Pauli check once per pattern is done outside
    per = gamma_period_from_diag(diag)
    r = cat["params"]["r"]
    gx = r * gamma
    t0 = time.perf_counter()
    gs = ground_pair(diag, n, gx)
    obs = observables(gs["gs"], n, c["minimizers"], diag, gx)
    ctr = Counter()
    rng = np.random.default_rng(rng_seed)
    eopt = optimize_l1(diag, n, gx, gs["gs"], per["T_gamma"], rng, ctr, "energy")
    fopt = optimize_l1(diag, n, gx, gs["gs"], per["T_gamma"], rng, ctr, "fidelity")
    false_hard = bool(fopt["F"] >= eopt["F"] + 0.25 and fopt["F"] >= 0.4)
    return {
        "N": n,
        "pattern": name,
        "u_min": c["u_min"],
        "degeneracy": c["degeneracy"],
        "stabilizer_order": c["stabilizer_order"],
        "mean_agreement": c["mean_agreement"],
        "Gamma_per_degree": gamma,
        "r": r,
        "Gamma_X": gx,
        "E0": gs["E0"],
        "gap": gs["gap"],
        "Emax": gs["Emax"],
        "multiplicity": gs["multiplicity"],
        "parity_even_weight": gs["parity_even_weight"],
        "E0_Pplus": gs["E0_Pplus"],
        "E0_Pminus": gs["E0_Pminus"],
        "residual": gs["residual"],
        "ed_method": gs["method"],
        "F0": obs["F_plus"],
        "F_polarized_cat": obs["F_polarized_cat"],
        "F_manifold": obs["F_manifold"],
        "F_product": obs["F_product"],
        "S_half_bits": obs["S_half_bits"],
        "subsystem": obs["subsystem"],
        "F1_energy": eopt["F"],
        "E1_energy": eopt["E"],
        "F1_max": fopt["F"],
        "E_at_F1_max": fopt["E"],
        "false_hard_gap": float(fopt["F"] - eopt["F"]),
        "false_hard_flag": false_hard,
        "T_gamma": per["T_gamma"],
        "gcd_Hc": per["gcd"],
        "eta_L1_energy": eta(eopt["E"], gs["E0"], gs["Emax"]),
        "eta_init": eta(energy_of(plus_state(n), diag, n, gx), gs["E0"], gs["Emax"]),
        "wall_s": time.perf_counter() - t0,
        "n_evals": ctr.n,
    }


def deep_curve(cat, name, gamma, Lmax=6) -> list[dict]:
    c = pattern_row(cat, name)
    n = cat["N"]
    blocks = cat["blocks"]
    diag = hc_diag_for_pattern(n, blocks, c["eps"])
    per = gamma_period_from_diag(diag)
    T = per["T_gamma"]
    r = cat["params"]["r"]
    gx = r * gamma
    spec = dense_spectrum(diag, n, gx)
    ev = spec["evals"]
    e0 = float(ev[0])
    emax = float(ev[-1])
    # unique GS vector; if a 1e-6 multiplet, fidelity is the subspace weight
    mask = ev <= e0 + 1e-6
    tgt = spec["evecs"][:, mask]

    def subF(psi):
        a = tgt.conj().T @ psi
        return float(np.vdot(a, a).real)

    # optimize against the even ground vector (the returned GS is even when
    # parity weight is 1; pick the most-even column)
    scores = [flip_expectation(tgt[:, i]) for i in range(tgt.shape[1])]
    gs_vec = tgt[:, int(np.argmax(scores))]
    rng = np.random.default_rng(20261106 + 100 * n + int(gamma * 1000))
    ctr = Counter()
    rows = []
    psi0 = plus_state(n)
    rows.append({
        "N": n, "pattern": name, "Gamma_per_degree": gamma, "Gamma_X": gx, "r": r,
        "L": 0, "method": "init", "E": energy_of(psi0, diag, n, gx),
        "F": f_overlap(psi0, gs_vec), "F_sub": subF(psi0),
        "eta": eta(energy_of(psi0, diag, n, gx), e0, emax),
        "tag": "plus", "E0": e0, "Emax": emax, "gap": float(ev[1] - e0),
        "mult": int(mask.sum()),
        "T_gamma": T,
    })
    prev = optimize_l1(diag, n, gx, gs_vec, T, rng, ctr, "energy")
    rows.append(_pack_deep(n, name, gamma, gx, r, 1, "qaoa_energy", prev, subF, e0, emax, T, ev))
    fmax = optimize_l1(diag, n, gx, gs_vec, T, rng, ctr, "fidelity")
    rows.append(_pack_deep(n, name, gamma, gx, r, 1, "qaoa_fidelity_L1", fmax, subF, e0, emax, T, ev))
    for L in range(2, Lmax + 1):
        prev = optimize_depth(diag, n, gx, gs_vec, T, L, prev, rng, ctr)
        rows.append(_pack_deep(n, name, gamma, gx, r, L, "qaoa_energy", prev, subF, e0, emax, T, ev))
    sh = optimize_shared(diag, n, gx, gs_vec, T, Lmax, prev, rng, ctr)
    rows.append({
        "N": n, "pattern": name, "Gamma_per_degree": gamma, "Gamma_X": gx, "r": r,
        "L": Lmax, "method": "shared_gamma", "E": sh["E"], "F": sh["F"],
        "F_sub": subF(evolve_shared(diag, n, sh)),
        "eta": eta(sh["E"], e0, emax), "tag": "shared",
        "gamma_shared": sh["gamma_shared"], "E0": e0, "Emax": emax,
        "gap": float(ev[1] - e0), "mult": int(mask.sum()), "T_gamma": T,
        "betas_raw": sh["betas_raw"], "gammas_raw": sh["gammas_raw"],
    })
    rows[0]["n_evals_total"] = ctr.n
    return rows


def evolve_shared(diag, n, sh):
    from ham import evolve
    L = len(sh["betas_raw"])
    return evolve(plus_state(n), diag, n, sh["betas_raw"], np.full(L, sh["gamma_shared"]))


def _pack_deep(n, name, gamma, gx, r, L, method, rec, subF, e0, emax, T, ev):
    return {
        "N": n, "pattern": name, "Gamma_per_degree": gamma, "Gamma_X": gx, "r": r,
        "L": L, "method": method, "E": rec["E"], "F": rec["F"], "F_sub": subF(rec["psi"]),
        "eta": eta(rec["E"], e0, emax), "tag": rec["tag"],
        "E0": e0, "Emax": emax, "gap": float(ev[1] - e0), "T_gamma": T,
        "betas_raw": rec["betas_raw"], "gammas_raw": rec["gammas_raw"],
        "betas_canonical": rec["betas_canonical"], "gammas_canonical": rec["gammas_canonical"],
        "duration_raw": rec["duration_raw"], "duration_canonical": rec["duration_canonical"],
    }


def catalog_summary(cat) -> dict:
    # drop heavy arrays
    rows = []
    seen = set()
    for c in cat["catalog"]:
        key = (c["canonical"], c["u_min"], c["stabilizer_order"], c["degeneracy"])
        rec = {
            "name": c["name"],
            "u_min": c["u_min"],
            "degeneracy": c["degeneracy"],
            "stabilizer_order": c["stabilizer_order"],
            "aut_exhaustive": c["aut_exhaustive"],
            "mean_agreement": c["mean_agreement"],
            "n_neg_blocks": c["n_neg_blocks"],
            "canonical": c["canonical"],
            "y0_gauge": c["y0_gauge"],
        }
        rec["duplicate_coset"] = key in seen
        seen.add(key)
        rows.append(rec)
    return {
        "N": cat["N"],
        "design": design_name(cat["N"]),
        "b": cat["b"],
        "r": cat["params"]["r"],
        "rank_B": cat["rank"],
        "n_codewords": cat["n_codewords"],
        "aut_order": cat["aut_order"],
        "aut_exhaustive": cat["aut_exhaustive"],
        "aut_nodes": cat["aut_nodes"],
        "covering_radius": cat["covering_radius"],
        "target_name_min_frustration": cat["target_name"],
        "patterns": rows,
    }


def write_csv(path: Path, rows: list[dict]):
    import csv
    if not rows:
        return
    keys = []
    for r in rows:
        for k in r:
            if k not in keys and not isinstance(r[k], (list, dict)):
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


def make_plots(out: Path, screen, deep):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    figdir = out / "figures"
    figdir.mkdir(exist_ok=True)
    # F1 energy vs F1 max
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.0), sharey=True)
    for ax, n in zip(axes, (8, 10)):
        sub = [r for r in screen if r["N"] == n]
        patterns = []
        for r in sub:
            if r["pattern"] not in patterns:
                patterns.append(r["pattern"])
        for p in patterns:
            rows = sorted([r for r in sub if r["pattern"] == p], key=lambda r: r["Gamma_per_degree"])
            g = [r["Gamma_per_degree"] for r in rows]
            ax.plot(g, [r["F1_energy"] for r in rows], marker="o", ls="--", label=f"{p} F1E")
            ax.plot(g, [r["F1_max"] for r in rows], marker="x", ls=":", label=f"{p} F1max")
        ax.set_xscale("log")
        ax.set_xlabel(r"per-degree $\Gamma$  (field coeff. is $r\Gamma$)")
        ax.set_ylim(-0.02, 1.02)
        ax.set_title(f"N={n}")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel(r"$L=1$ squared overlap")
    axes[0].legend(fontsize=7, frameon=False, ncol=1)
    fig.suptitle(r"Energy-selected $F$ vs fidelity-maximized $F$ at $L=1$")
    fig.tight_layout()
    fig.savefig(figdir / "F1_energy_vs_max.png", dpi=140)
    plt.close(fig)

    # deep F vs L
    fig, ax = plt.subplots(figsize=(6.6, 4.2))
    keys = []
    for r in deep:
        if r["method"] != "qaoa_energy" and r["method"] != "init":
            continue
        k = (r["N"], r["pattern"], r["Gamma_per_degree"])
        if k not in keys:
            keys.append(k)
    for k in keys:
        rows = sorted([r for r in deep if (r["N"], r["pattern"], r["Gamma_per_degree"]) == k and r["method"] in ("init", "qaoa_energy")], key=lambda r: r["L"])
        ax.plot([r["L"] for r in rows], [r["F"] for r in rows], marker="o", label=f"N={k[0]} {k[1]} Γ={k[2]:g}")
    ax.set_xlabel("depth L")
    ax.set_ylabel("energy-optimal F")
    ax.set_ylim(-0.02, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7, frameon=False)
    ax.set_title("Energy-optimal F vs L (ordered-side Γ)")
    fig.tight_layout()
    fig.savefig(figdir / "F_vs_L_deep.png", dpi=140)
    plt.close(fig)

    # regime: Fcat and F0
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8), sharey=True)
    for ax, n in zip(axes, (8, 10)):
        sub = [r for r in screen if r["N"] == n and r["pattern"] in ("uniform", "block_0", "affine_off_origin")]
        for p in sorted({r["pattern"] for r in sub}):
            rows = sorted([r for r in sub if r["pattern"] == p], key=lambda r: r["Gamma_per_degree"])
            ax.plot([r["Gamma_per_degree"] for r in rows], [r["F_polarized_cat"] for r in rows], marker="o", label=f"{p} cat")
            ax.plot([r["Gamma_per_degree"] for r in rows], [r["F0"] for r in rows], marker="x", ls="--", label=f"{p} F0")
        ax.set_xscale("log")
        ax.set_xlabel(r"per-degree $\Gamma$")
        ax.set_title(f"N={n}")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.02, 1.02)
    axes[0].set_ylabel("overlap")
    axes[0].legend(fontsize=6.5, frameon=False)
    fig.suptitle("Polarized-cat overlap vs |<+||GS>|^2")
    fig.tight_layout()
    fig.savefig(figdir / "regime_cat_F0.png", dpi=140)
    plt.close(fig)


def main():
    t_run = time.perf_counter()
    out = Path(__file__).resolve().parent / "results" / "round2"
    out.mkdir(parents=True, exist_ok=True)
    # designs
    design_meta = []
    for n in (8, 10, 14, 16):
        blocks = list(design_blocks(n))
        params = verify_sqs(n, blocks)
        save_blocks(out, n, blocks)
        design_meta.append({"N": n, "name": design_name(n), **params, "n_blocks_written": len(blocks)})
        print(f"design N={n} OK b={params['b']} r={params['r']}", flush=True)

    catalogs = {}
    summaries = []
    for n in (8, 10):
        print(f"catalog N={n}", flush=True)
        cat = prepare_catalog(n, compute_aut=True)
        catalogs[n] = cat
        summaries.append(catalog_summary(cat))
        # Pauli cross-check on the target and uniform
        for name in ("uniform", cat["target_name"]):
            c = pattern_row(cat, name)
            diag = hc_diag_for_pattern(n, cat["blocks"], c["eps"])
            ref = explicit_hc_from_pauli(n, cat["blocks"], signs_pm_from_eps(c["eps"], cat["b"]))
            err = float(np.max(np.abs(diag - ref)))
            if err > 1e-8:
                raise SystemExit(f"diagonal mismatch {name} N={n} {err}")
            print(f"  diag check {name} err={err:.1e} target_rule={cat['target_name']}", flush=True)
    for n in (14, 16):
        print(f"short catalog N={n}", flush=True)
        cat = prepare_catalog(n, compute_aut=False, short_family=True)
        catalogs[n] = cat
        summaries.append(catalog_summary(cat))
        c = pattern_row(cat, "uniform")
        diag = hc_diag_for_pattern(n, cat["blocks"], c["eps"])
        ref = explicit_hc_from_pauli(n, cat["blocks"], signs_pm_from_eps(c["eps"], cat["b"]))
        err = float(np.max(np.abs(diag - ref)))
        if err > 1e-8:
            raise SystemExit(f"diagonal mismatch uniform N={n} {err}")
        print(f"  N={n} uniform u={c['u_min']} deg={c['degeneracy']} diag err={err:.1e}", flush=True)

    (out / "designs_signs.json").write_text(json.dumps(_jsonable({"designs": design_meta, "catalogs": summaries}), indent=2))

    screen = []
    seed = 1
    for n, names in ((8, SCAN_N8), (10, SCAN_N10)):
        for name in names:
            for g in GAMMAS:
                print(f"screen N={n} {name} G={g}", flush=True)
                screen.append(screen_point(catalogs[n], name, g, seed))
                seed += 1
                row = screen[-1]
                print(
                    f"  F0={row['F0']:.3f} Fcat={row['F_polarized_cat']:.3f} Fman={row['F_manifold']:.3f} "
                    f"F1E={row['F1_energy']:.3f} F1max={row['F1_max']:.3f} gap={row['gap']:.4f} "
                    f"P+={row['parity_even_weight']:.3f}",
                    flush=True,
                )
    write_csv(out / "screen.csv", screen)

    deep = []
    for n, name, g in DEEP:
        print(f"deep N={n} {name} G={g}", flush=True)
        rows = deep_curve(catalogs[n], name, g, Lmax=6)
        deep.extend(rows)
        q = [r for r in rows if r["method"] in ("init", "qaoa_energy")]
        print("  " + " ".join(f"L{r['L']}={r['F']:.3f}" for r in q), flush=True)
    # drop non-scalar for csv but keep angles in json
    write_csv(out / "deep.csv", deep)
    (out / "deep_angles.json").write_text(json.dumps(_jsonable(deep), indent=2))
    make_plots(out, screen, deep)
    meta = {
        "wall_s": time.perf_counter() - t_run,
        "gammas_per_degree": list(GAMMAS),
        "deep": [list(x) for x in DEEP],
        "note": "N=14 and N=16 were constructed and frustration-certified, not QAOA-scanned.",
        "numpy": np.__version__,
    }
    (out / "run_meta.json").write_text(json.dumps(meta, indent=2))
    print(f"done in {meta['wall_s']:.1f}s -> {out}", flush=True)


if __name__ == "__main__":
    main()
