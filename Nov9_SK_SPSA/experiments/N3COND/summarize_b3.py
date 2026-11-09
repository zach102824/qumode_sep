"""Summarize B3 only. Does not rewrite RESULTS.md or summary_B1.json."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[1] / "results" / "N3COND"


def boot_ci(d, n=10000):
    d = np.asarray(d, float)
    r = np.random.default_rng(0)
    m = d[r.integers(0, len(d), size=(n, len(d)))].mean(axis=1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def paired(a_rows, b_rows, fn):
    db = {r["seed"]: r for r in b_rows}
    xs = [fn(a) - fn(db[a["seed"]]) for a in a_rows if a["seed"] in db]
    xs = np.asarray(xs, float)
    return {"n": int(len(xs)), "mean": float(xs.mean()), "median": float(np.median(xs)),
            "ci95_mean": boot_ci(xs), "n_pos": int(np.sum(xs > 0)), "n_neg": int(np.sum(xs < 0))}


def main():
    rows = []
    for p in sorted((OUT / "runs").glob("N9/*_M[02]_q*.json")):
        rows.append(json.loads(p.read_text()))
    if not rows:
        raise SystemExit("no B3 rows")
    if any(r["band"] != "read_dev" or r["seeds_are_fresh_holdout"] for r in rows):
        raise SystemExit("a B3 row is mislabeled as holdout")
    cells = {}
    for r in rows:
        cells.setdefault((r["L"], r["q"], r["circuit"], r["conditioner"]), []).append(r)
    contrasts = {}
    for L, q, circuit in sorted({(L, q, c) for L, q, c, _ in cells}):
        A = cells.get((L, q, circuit, "M2"), [])
        B = cells.get((L, q, circuit, "M0"), [])
        if A and B:
            contrasts[f"N9_L{L}_q{q}_{circuit}_M2_minus_M0"] = {
                "E": paired(A, B, lambda z: z["diag"]["E"]),
                "F": paired(A, B, lambda z: z["diag"]["F"]),
            }
    gate_key = "N9_L3_q1024_Nov3_M2_minus_M0"
    g = contrasts[gate_key]["E"]
    passed = g["median"] < 0 and g["ci95_mean"][1] < 0
    summary = {
        "n_runs": len(rows), "failures": int(sum(r["status"] != "ok" for r in rows)),
        "queries_inside_Q": int(sum(r.get("queries") or 0 for r in rows)),
        "outside_cap_queries": int(sum(r.get("outside_cap_queries") or 0 for r in rows)),
        "calibration_queries": int(sum(r["query_cat"]["calibration"] for r in rows if r["status"] == "ok")),
        "wall_s": float(sum(r.get("wall_s") or 0 for r in rows)),
        "seeds_role": "read_dev",
        "contrasts": contrasts,
        "gate": {"cell": gate_key, "passed": bool(passed), "energy": g,
                 "rule": "median dE<0 and mean CI entirely below 0; F is not in the gate"},
    }
    (OUT / "summary_B3.json").write_text(json.dumps(summary, indent=1))
    lines = ["# N3COND B3 read-dev results", "",
             "Seeds 10010–10017 are read development, not a fresh holdout. M2−M0. Negative ΔE means M2 is lower. Fidelity is post hoc and is not the gate.",
             "",
             f"Runs {summary['n_runs']}, failures {summary['failures']}, queries inside Q {summary['queries_inside_Q']}, outside-cap queries {summary['outside_cap_queries']}, calibration inside Q {summary['calibration_queries']}, summed worker wall {summary['wall_s']:.1f}s.",
             "",
             f"Energy gate passed: {passed}.",
             "",
             "| contrast | ΔE med/mean [CI] n(ΔE>0) | ΔF med/mean [CI] n(ΔF>0) |",
             "|---|---|---|"]
    for k, ct in contrasts.items():
        e, f = ct["E"], ct["F"]
        lines.append(
            f"| {k} | {e['median']:+.4f} / {e['mean']:+.4f} [{e['ci95_mean'][0]:+.4f},{e['ci95_mean'][1]:+.4f}] {e['n_pos']}/{e['n']} "
            f"| {f['median']:+.4f} / {f['mean']:+.4f} [{f['ci95_mean'][0]:+.4f},{f['ci95_mean'][1]:+.4f}] {f['n_pos']}/{f['n']} |"
        )
    (OUT / "RESULTS_B3.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(summary["gate"], indent=1))
    print("queries", summary["queries_inside_Q"], "outside", summary["outside_cap_queries"], "fail", summary["failures"])


if __name__ == "__main__":
    main()
