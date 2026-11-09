"""Summarize B4. Reads reused C0 files. Does not rewrite earlier summaries."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[1] / "results" / "N3COND"
RUNS = OUT / "runs"


def boot_ci(d, n=10000):
    d = np.asarray(d, float)
    r = np.random.default_rng(0)
    m = d[r.integers(0, len(d), size=(n, len(d)))].mean(axis=1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def paired(a_rows, b_rows, fn):
    db = {r["seed"]: r for r in b_rows}
    xs = np.asarray([fn(a) - fn(db[a["seed"]]) for a in a_rows if a["seed"] in db], float)
    return {"n": int(len(xs)), "mean": float(xs.mean()), "median": float(np.median(xs)),
            "ci95_mean": boot_ci(xs), "n_pos": int(np.sum(xs > 0)), "n_neg": int(np.sum(xs < 0)),
            "values": [float(v) for v in xs]}


def load_c0(N, circuit):
    rows = []
    for seed in range(10010, 10018):
        path = RUNS / f"N{N}" / f"s{seed}_L3_{circuit}_C0_q1024.json"
        r = json.loads(path.read_text())
        if not (r["queries"] == 3072 and r["Q"] == 3072 and r["query_cat"]["calibration"] == 32
                and r["K_per_start"] == 379 and r["status"] == "ok"
                and all(not m["gain_frozen"] for m in r["start_meta"])):
            raise SystemExit(f"C0 reuse audit failed: {path.name}")
        rows.append(r)
    return rows


def cell(rows, arm):
    return [r for r in rows if r["conditioner"] == arm]


def main():
    new = []
    for p in sorted(RUNS.glob("N*/s*_L3_*_S[02]_q1024.json")):
        new.append(json.loads(p.read_text()))
    if len(new) != 64:
        raise SystemExit(f"expected 64 new fits, found {len(new)}")
    if any(r["band"] != "read_dev" or r["seeds_are_fresh_holdout"] or r["queries"] != 3072 for r in new):
        raise SystemExit("a B4 row is mislabeled or not at Q=3072")
    contrasts = {}
    gates = {}
    for N in (9, 12):
        for circuit in ("Nov3", "QAOA"):
            rows = [r for r in new if r["N"] == N and r["circuit"] == circuit]
            c0 = load_c0(N, circuit)
            s2, s0 = cell(rows, "S2"), cell(rows, "S0")
            contrasts[f"N{N}_{circuit}_S2_minus_C0"] = {
                "E": paired(s2, c0, lambda z: z["diag"]["E"]),
                "F": paired(s2, c0, lambda z: z["diag"]["F"]),
            }
            contrasts[f"N{N}_{circuit}_S2_minus_S0"] = {
                "E": paired(s2, s0, lambda z: z["diag"]["E"]),
                "F": paired(s2, s0, lambda z: z["diag"]["F"]),
            }
            contrasts[f"N{N}_{circuit}_S0_minus_C0"] = {
                "E": paired(s0, c0, lambda z: z["diag"]["E"]),
                "F": paired(s0, c0, lambda z: z["diag"]["F"]),
            }
    for N in (9, 12):
        e_c0 = contrasts[f"N{N}_Nov3_S2_minus_C0"]["E"]
        e_s0 = contrasts[f"N{N}_Nov3_S2_minus_S0"]["E"]
        ok = e_c0["median"] < 0 and e_c0["ci95_mean"][1] < 0 and e_s0["median"] < 0
        gates[f"N{N}"] = {"passed": bool(ok), "dE_C0": {k: e_c0[k] for k in e_c0 if k != "values"},
                          "dE_S0_median": e_s0["median"]}
    survives = bool(gates["N9"]["passed"] and gates["N12"]["passed"])
    summary = {
        "n_new_runs": len(new), "failures": int(sum(r["status"] != "ok" for r in new)),
        "queries_newly_charged": int(sum(r["queries"] for r in new)),
        "outside_cap_queries": int(sum(r["outside_cap_queries"] for r in new)),
        "calibration_queries_inside_Q": int(sum(r["query_cat"]["calibration"] for r in new)),
        "c0_reused_runs": 32, "c0_queries_not_added_again": 32 * 3072,
        "prior_ledger_preserved": 575616,
        "wall_s": float(sum(r["wall_s"] for r in new)),
        "seeds_role": "read_dev",
        "contrasts": contrasts,
        "gates": gates,
        "branch_survives": survives,
        "rule": "both N9 and N12 Nov3 must beat C0 on the mean CI and beat the sham on the median; F is not in the gate",
    }
    # drop per-seed lists from the on-disk summary? Keep them; they are the paired values. Fine.
    (OUT / "summary_B4.json").write_text(json.dumps(summary, indent=1))
    lines = ["# N3COND B4 read-dev results", "",
             "Seeds 10010–10017 are read development. Negative ΔE means the first arm is lower. Fidelity is post hoc.",
             "",
             f"New fits {summary['n_new_runs']}, failures {summary['failures']}, newly charged {summary['queries_newly_charged']}, outside cap {summary['outside_cap_queries']}. C0 reuse is 32 existing runs, not charged again. Branch survives: {survives}.",
             "",
             "| contrast | ΔE med/mean [CI] n(ΔE>0) | ΔF med/mean [CI] n(ΔF>0) |",
             "|---|---|---|"]
    for k, ct in contrasts.items():
        e, f = ct["E"], ct["F"]
        lines.append(
            f"| {k} | {e['median']:+.4f} / {e['mean']:+.4f} [{e['ci95_mean'][0]:+.4f},{e['ci95_mean'][1]:+.4f}] {e['n_pos']}/{e['n']} "
            f"| {f['median']:+.4f} / {f['mean']:+.4f} [{f['ci95_mean'][0]:+.4f},{f['ci95_mean'][1]:+.4f}] {f['n_pos']}/{f['n']} |"
        )
    (OUT / "RESULTS_B4.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"branch_survives": survives, "gates": gates}, indent=1))


if __name__ == "__main__":
    main()
