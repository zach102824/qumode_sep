"""Summarize N3COND runs. Dev and holdout stay in separate columns."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[1] / "results" / "N3COND"
PREREG = json.loads((HERE / "PREREG_N3COND_B1.json").read_text())


def load():
    return [json.loads(p.read_text()) for p in sorted((OUT / "runs").glob("N*/*.json"))]


def boot_ci(d, n=10000):
    d = np.asarray(d, float)
    if len(d) == 0:
        return [float("nan"), float("nan")]
    r = np.random.default_rng(0)
    m = d[r.integers(0, len(d), size=(n, len(d)))].mean(axis=1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def cell_stats(rs):
    F = np.array([r["diag"]["F"] for r in rs])
    E = np.array([r["diag"]["E"] for r in rs])
    return {"n": len(rs), "medF": float(np.median(F)), "meanF": float(F.mean()),
            "minF": float(F.min()), "q25F": float(np.percentile(F, 25)),
            "frac_F_ge_0.5": float(np.mean(F >= 0.5)),
            "medE": float(np.median(E)), "meanE": float(E.mean()),
            "failures": int(sum(r["status"] != "ok" for r in rs)),
            "queries": int(sum(r.get("queries") or 0 for r in rs)),
            "wall_s": float(sum(r.get("wall_s") or 0 for r in rs)),
            "seeds": [r["seed"] for r in rs]}


def main():
    R = load()
    bands = ("dev", "holdout")
    cells = {}
    contrasts = {}
    for r in R:
        key = (r["N"], r["L"], r["q"], r["circuit"], r["conditioner"], r["band"])
        cells.setdefault(key, []).append(r)
    summary_cells = {}
    for key, rs in cells.items():
        N, L, q, circuit, cond, band = key
        summary_cells[f"N{N}_L{L}_q{q}_{circuit}_{cond}_{band}"] = cell_stats(rs)
    contrasts = {}
    for N in sorted({r["N"] for r in R}):
        for L in sorted({r["L"] for r in R if r["N"] == N}):
            for q in sorted({r["q"] for r in R if r["N"] == N and r["L"] == L}):
                for band in bands:
                    def rows(circuit, cond, band=band, N=N, L=L, q=q):
                        return cells.get((N, L, q, circuit, cond, band), [])

                    def add(name, A, B):
                        if not A or not B:
                            return
                        contrasts[f"N{N}_L{L}_q{q}_{band}_{name}"] = {
                            "F": _paired_fn(A, B, lambda z: z["diag"]["F"]),
                            "E": _paired_fn(A, B, lambda z: z["diag"]["E"]),
                        }

                    add("Nov3_C1_minus_C0", rows("Nov3", "C1"), rows("Nov3", "C0"))
                    add("QAOA_C1_minus_C0", rows("QAOA", "C1"), rows("QAOA", "C0"))
                    add("Nov3_minus_QAOA_C0", rows("Nov3", "C0"), rows("QAOA", "C0"))
                    add("Nov3_minus_QAOA_C1", rows("Nov3", "C1"), rows("QAOA", "C1"))

    primary = contrasts.get("N9_L3_q1024_holdout_Nov3_C1_minus_C0")
    reading = None
    if primary and primary["F"] and primary["E"]:
        helps = primary["F"]["median"] >= 0.02 and primary["E"]["median"] < 0
        reading = {"helps_nov3": bool(helps), "rule": "holdout median dF(C1-C0)>=0.02 and median dE<0",
                   "median_dF": primary["F"]["median"], "median_dE": primary["E"]["median"],
                   "mean_dF": primary["F"]["mean"], "mean_dF_ci95": primary["F"]["ci95_mean"],
                   "mean_dE": primary["E"]["mean"], "mean_dE_ci95": primary["E"]["ci95_mean"]}
    S = {"n_runs": len(R), "failures": int(sum(r["status"] != "ok" for r in R)),
         "queries_charged_in_Q": int(sum(r.get("queries") or 0 for r in R)),
         "wall_s": float(sum(r.get("wall_s") or 0 for r in R)),
         "prereg_sha256": sorted({r["prereg_sha256"] for r in R}),
         "cells": summary_cells, "contrasts": contrasts, "primary_N9_L3_q1024_holdout": reading}
    (OUT / "summary_B1.json").write_text(json.dumps(S, indent=1))
    md(S)
    print(f"runs {S['n_runs']} failures {S['failures']} queries {S['queries_charged_in_Q']}")
    if reading:
        print("primary", json.dumps(reading))


def _paired_fn(rows_a, rows_b, fn):
    db = {r["seed"]: r for r in rows_b}
    xs = []
    for a in rows_a:
        b = db.get(a["seed"])
        if b is not None:
            xs.append(fn(a) - fn(b))
    xs = np.asarray(xs, float)
    if len(xs) == 0:
        return None
    return {"n": int(len(xs)), "mean": float(xs.mean()), "median": float(np.median(xs)),
            "ci95_mean": boot_ci(xs), "n_pos": int(np.sum(xs > 0)), "n_neg": int(np.sum(xs < 0))}


def fmt(p, nd=3):
    if not p:
        return "—"
    return f"{p['median']:+.{nd}f} / {p['mean']:+.{nd}f} [{p['ci95_mean'][0]:+.{nd}f},{p['ci95_mean'][1]:+.{nd}f}] {p['n_pos']}/{p['n']}"


def md(S):
    lines = ["# N3COND development results", "",
             "Original Nov3rd, shared gamma and theta, full s1/s2/s3. Scalar SPSA (C0) versus frozen diagonal scale (C1). QAOA p=L gets the same conditioner and the same Q=qL. B* only. Fidelity is post hoc.",
             "Dev seeds 10010–10013 supplied the frozen scale. Holdout seeds 10014–10017 did not. Dev deltas are descriptive. The predictive cell is N9 L3 q=1024 holdout.",
             ""]
    if S["primary_N9_L3_q1024_holdout"]:
        p = S["primary_N9_L3_q1024_holdout"]
        lines.append(f"Primary reading: helps_nov3={p['helps_nov3']}. Holdout median dF={p['median_dF']:+.4f}, median dE={p['median_dE']:+.4f}, mean dF={p['mean_dF']:+.4f} CI {p['mean_dF_ci95']}.")
        lines.append("")
    lines.append(f"Runs {S['n_runs']}, failures {S['failures']}, charged comparison queries {S['queries_charged_in_Q']}, summed worker wall {S['wall_s']:.1f}s.")
    lines.append("")
    lines.append("The trailing count is how many paired differences are positive. For fidelity, positive means C1 is higher. For energy, positive means C1 is worse.")
    lines.append("")
    lines.append("| cell | band | med F C0 | med F C1 | med E C0 | med E C1 | dF C1−C0 med/mean [CI] n(dF>0) | dE C1−C0 med/mean [CI] n(dE>0) |")
    lines.append("|---|---|---|---|---|---|---|---|")
    keys = sorted(S["contrasts"])
    for k in keys:
        if not k.endswith("Nov3_C1_minus_C0") and "Nov3_C1_minus_C0" not in k:
            continue
        # key N{N}_L{L}_q{q}_{band}_Nov3_C1_minus_C0
        parts = k.split("_")
        # N9 L3 q1024 holdout
        N = parts[0][1:]
        L = parts[1][1:]
        q = parts[2][1:]
        band = parts[3]
        c0 = S["cells"].get(f"N{N}_L{L}_q{q}_Nov3_C0_{band}")
        c1 = S["cells"].get(f"N{N}_L{L}_q{q}_Nov3_C1_{band}")
        if not c0 or not c1:
            continue
        ct = S["contrasts"][k]
        lines.append(f"| N{N} L{L} q{q} Nov3 | {band} | {c0['medF']:.4f} | {c1['medF']:.4f} | {c0['medE']:.4f} | {c1['medE']:.4f} | {fmt(ct['F'])} | {fmt(ct['E'])} |")
    lines.append("")
    lines.append("QAOA conditioner and Nov3−QAOA, same layout (dF is the first member minus the second).")
    lines.append("")
    lines.append("| contrast | dF med/mean [CI] n(dF>0) | dE med/mean [CI] n(dE>0) |")
    lines.append("|---|---|---|")
    for k in keys:
        ct = S["contrasts"][k]
        lines.append(f"| {k} | {fmt(ct['F'])} | {fmt(ct['E'])} |")
    (OUT / "RESULTS.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
