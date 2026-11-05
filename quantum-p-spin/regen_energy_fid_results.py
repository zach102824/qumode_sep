"""Regenerate summary.csv + RESULTS.md from per_instance.csv with energy columns.

Columns: n|L|QAOA F|QAOA E|QAOA E0|QAOA ΔE|QAOA i95|shared F|shared E|shared E0|shared ΔE|shared i95
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

PKG = Path(__file__).resolve().parent
DEFAULT_OUT = PKG / "runs/compare/energy_fid_grid"

DEFAULT_NS = (6, 9, 12, 15, 18)
LS = (2, 4, 6, 8, 10, 12, 14, 16)


def med(xs):
    return float(np.median(xs)) if xs else float("nan")


def load_per_instance(path: Path):
    bucket = defaultdict(list)
    with path.open() as f:
        for row in csv.DictReader(f):
            key = (int(row["n"]), int(row["L"]), row["method"])
            F = float(row["final_F"])
            E = float(row["final_E"])
            E0 = float(row["E0"])
            bucket[key].append(
                {
                    "F": F,
                    "E": E,
                    "E0": E0,
                    "dE": E - E0,
                    "i95": float(row["i95"]),
                    "n_params": int(float(row["n_params"])),
                }
            )
    return bucket


def aggregate(rows):
    return {
        "F": med([r["F"] for r in rows]),
        "E": med([r["E"] for r in rows]),
        "E0": med([r["E0"] for r in rows]),
        "dE": med([r["dE"] for r in rows]),
        "i95": med([r["i95"] for r in rows]),
        "n_params": rows[0]["n_params"] if rows else "",
        "N": len(rows),
    }


def write_outputs(out_dir: Path, bucket, ns):
    sfields = [
        "n", "L",
        "QAOA_F", "QAOA_E", "QAOA_E0", "QAOA_dE", "QAOA_i95",
        "shared_F", "shared_E", "shared_E0", "shared_dE", "shared_i95",
        "QAOA_np", "shared_np", "N",
    ]
    srows = []
    for n in ns:
        for L in LS:
            fq = bucket.get((n, L, "free_qaoa"))
            sg = bucket.get((n, L, "shared_gb"))
            if not fq and not sg:
                continue
            row = {"n": n, "L": L}
            if fq:
                a = aggregate(fq)
                row.update(
                    {
                        "QAOA_F": a["F"],
                        "QAOA_E": a["E"],
                        "QAOA_E0": a["E0"],
                        "QAOA_dE": a["dE"],
                        "QAOA_i95": a["i95"],
                        "QAOA_np": a["n_params"],
                        "N": a["N"],
                    }
                )
            else:
                row.update(
                    {
                        "QAOA_F": "", "QAOA_E": "", "QAOA_E0": "", "QAOA_dE": "",
                        "QAOA_i95": "", "QAOA_np": "",
                    }
                )
            if sg:
                a = aggregate(sg)
                row.update(
                    {
                        "shared_F": a["F"],
                        "shared_E": a["E"],
                        "shared_E0": a["E0"],
                        "shared_dE": a["dE"],
                        "shared_i95": a["i95"],
                        "shared_np": a["n_params"],
                        "N": a["N"],
                    }
                )
            else:
                row.update(
                    {
                        "shared_F": "", "shared_E": "", "shared_E0": "",
                        "shared_dE": "", "shared_i95": "", "shared_np": "",
                    }
                )
            if "N" not in row or row.get("N") in ("", None):
                row["N"] = len(fq or sg or [])
            srows.append(row)

    summary_path = out_dir / "summary.csv"
    with summary_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=sfields)
        w.writeheader()
        for r in srows:
            w.writerow(r)

    md = out_dir / "RESULTS.md"
    lines = [
        "# Energy-fid grid — free QAOA vs shared γ+β_ℓ",
        "",
        "## Protocol",
        "",
        r"- $H = H_c + H_x$, $H_c = -(\sqrt{3}/n)\sum J_{ijk} Z_i Z_j Z_k$, $H_x = -\Gamma\sum X_i$",
        "- Γ = 0.5, J ~ N(0,1)",
        "- Ansatz Nov_3rd: `|0⟩ → [U_c^X(γ) → SNAP(φ)]^L → W`, φ_k = -β k",
        "- Methods: free_qaoa (2L) vs shared_gb (L+1)",
        "- Per instance: cost = ⟨H⟩, one LBFGS, init β=γ=+0.01",
        "- LBFGS: 40 outer steps × max_iter=5 (cap ≤200 iters), lr=0.25, strong_wolfe",
        "- N = 32 instances (seeds 0..31); cell = **median** over instances",
        "- i95 = first outer step where F ≥ 0.95 × that run's final F (step 0 = init)",
        "- ΔE = ⟨H⟩ − E0 (energy error above true ground state)",
        "- Parallelism used: ProcessPool ≤8 workers (BLAS/torch threads=1); n12 throttled to 4",
        "",
        f"**Table locked to n ∈ {{{', '.join(str(n) for n in ns)}}}** (n15/n18 stopped; not reported).",
        "",
        f"Completed (n,L) rows: {len(srows)} / {len(ns)*len(LS)}",
        "",
        "## Table (medians)",
        "",
        "| n | L | QAOA F | QAOA E | QAOA E0 | QAOA ΔE | QAOA i95 | shared F | shared E | shared E0 | shared ΔE | shared i95 |",
        "|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|",
    ]

    def fmt(x, nd=4):
        if x == "" or x is None:
            return "—"
        try:
            v = float(x)
        except (TypeError, ValueError):
            return "—"
        if nd == 1:
            return f"{v:.1f}"
        return f"{v:.{nd}f}"

    for r in srows:
        lines.append(
            f"| {r['n']} | {r['L']} | {fmt(r['QAOA_F'])} | {fmt(r['QAOA_E'])} | "
            f"{fmt(r['QAOA_E0'])} | {fmt(r['QAOA_dE'])} | {fmt(r['QAOA_i95'], 1)} | "
            f"{fmt(r['shared_F'])} | {fmt(r['shared_E'])} | {fmt(r['shared_E0'])} | "
            f"{fmt(r['shared_dE'])} | {fmt(r['shared_i95'], 1)} |"
        )

    # takeaways for n9/n12 focus
    lines += [
        "",
        "## Takeaways (n=9,12 focus)",
        "",
    ]
    for n in ns:
        if n < 9:
            continue
        qF = [r for r in srows if r["n"] == n]
        if not qF:
            continue
        best_q = max(qF, key=lambda r: float(r["QAOA_F"]) if r["QAOA_F"] != "" else -1)
        best_s = max(qF, key=lambda r: float(r["shared_F"]) if r["shared_F"] != "" else -1)
        lines.append(
            f"- **n={n}**: free QAOA best medF={float(best_q['QAOA_F']):.4f} at L={best_q['L']} "
            f"(ΔE={float(best_q['QAOA_dE']):.4f}, i95={float(best_q['QAOA_i95']):.1f}); "
            f"shared γ+β best medF={float(best_s['shared_F']):.4f} at L={best_s['L']} "
            f"(ΔE={float(best_s['shared_dE']):.4f}). Shared lags free and often **worsens** with depth."
        )

    lines += [
        "",
        "## Artifacts",
        "",
        "- `summary.csv` — pivoted medians (F, E, E0, ΔE, i95) for locked n",
        "- `per_instance.csv` — all instances (may still contain n15/n18 rows from partial run; table ignores them)",
        "- `grid.log` — run log (stopped before finishing n15/n18)",
        "- Tests: `tests.log` (GS + ansatz) — all PASSED before grid",
        "- Regenerate: `python regen_energy_fid_results.py --out runs/compare/energy_fid_grid --ns 6 9 12`",
        "",
        "## Status",
        "",
        "- Grid **STOPPED** on request; n15/n18 not finished and excluded from table.",
        "- 8-core ProcessPool parallelism was used (throttled on larger n).",
        "",
    ]
    md.write_text("\n".join(lines) + "\n")
    print(f"wrote {summary_path} ({len(srows)} rows)")
    print(f"wrote {md}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default=str(DEFAULT_OUT))
    ap.add_argument(
        "--ns",
        type=int,
        nargs="+",
        default=[6, 9, 12],
        help="n values to include (default: 6 9 12 locked)",
    )
    args = ap.parse_args()
    out_dir = Path(args.out)
    inst = out_dir / "per_instance.csv"
    if not inst.exists():
        raise SystemExit(f"missing {inst}")
    bucket = load_per_instance(inst)
    ns = tuple(int(x) for x in args.ns)
    allow = set(ns)
    bucket = {k: v for k, v in bucket.items() if k[0] in allow}
    write_outputs(out_dir, bucket, ns)


if __name__ == "__main__":
    main()
