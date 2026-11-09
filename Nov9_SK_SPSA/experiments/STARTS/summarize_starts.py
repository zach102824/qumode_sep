"""Paired STARTS S8 summary. Energy-only selection. F post hoc. Gate at N15 L3 Nov3."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[1] / "results" / "STARTS"
RUNS = OUT / "runs"
PREREG = json.loads((HERE / "PREREG_STARTS_S8.json").read_text())
SEEDS = list(PREREG["seeds"]["ids"])
FORBIDDEN = set(range(10010, 10048)) | set(range(11000, 11032)) | set(range(20000, 21000))


def boot_ci(d, n=10000):
    d = np.asarray(d, float)
    if d.size == 0:
        return [float("nan"), float("nan")]
    r = np.random.default_rng(0)
    m = d[r.integers(0, len(d), size=(n, len(d)))].mean(axis=1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def paired_vals(xs):
    xs = np.asarray(xs, float)
    if len(xs) == 0:
        return {"n": 0, "mean": float("nan"), "median": float("nan"),
                "ci95_mean": [float("nan"), float("nan")],
                "n_pos": 0, "n_neg": 0, "n_zero": 0,
                "min": float("nan"), "max": float("nan"),
                "p10": float("nan"), "p90": float("nan"), "values": []}
    return {
        "n": int(len(xs)),
        "mean": float(xs.mean()),
        "median": float(np.median(xs)),
        "ci95_mean": boot_ci(xs),
        "n_pos": int(np.sum(xs > 0)),
        "n_neg": int(np.sum(xs < 0)),
        "n_zero": int(np.sum(xs == 0)),
        "min": float(xs.min()),
        "max": float(xs.max()),
        "p10": float(np.percentile(xs, 10)),
        "p90": float(np.percentile(xs, 90)),
        "values": [float(v) for v in xs],
    }


def load_rows():
    return [json.loads(p.read_text()) for p in sorted(RUNS.glob("N*/s*_q1024.json"))]


def audit_row(r):
    problems = []
    if r["seed"] in FORBIDDEN:
        problems.append("forbidden_seed")
    if r["seed"] not in SEEDS:
        problems.append("seed_outside_prereg")
    if r.get("band") != "exclusive_unread_dev":
        problems.append("band")
    if r.get("q") != 1024 or r.get("Q") != 3072 or r.get("L") != 3 or r.get("N") != 15:
        problems.append("batch")
    if r.get("status") == "ok":
        if r["queries"] != 3072:
            problems.append("queries_ne_Q")
        if r.get("outside_cap_queries", 0) != 0:
            problems.append("outside_cap")
        if r["arm"] == "S4" and (r.get("nstarts") != 4 or r.get("K_per_start") != 379):
            problems.append("S4_split")
        if r["arm"] == "S8" and (r.get("nstarts") != 8 or r.get("K_per_start") != 187):
            problems.append("S8_split")
        cats = r.get("query_cat") or {}
        if sum(cats.values()) != r["queries"]:
            problems.append("cat_sum")
        if r["arm"] == "S4" and cats.get("calibration") != 32:
            problems.append("S4_cal")
        if r["arm"] == "S8" and cats.get("calibration") != 64:
            problems.append("S8_cal")
        if any(m.get("gain_frozen") for m in r.get("start_meta") or []):
            problems.append("gain_frozen")
        if r.get("optimizer") != "C0":
            problems.append("optimizer")
    return problems


def index(rows, circ, arm):
    out = {}
    for r in rows:
        if r["circuit"] == circ and r["arm"] == arm and r.get("status") == "ok":
            out[r["seed"]] = r
    return out


def contrast(a, b):
    seeds = sorted(set(a) & set(b))
    rec = {"n": len(seeds), "seeds": seeds}
    for key in ("E_err_over_N", "E", "F"):
        xs = []
        for s in seeds:
            ra, rb = a[s], b[s]
            va = ra["E_err_over_N"] if key == "E_err_over_N" else (ra["diag"]["E"] if key == "E" else ra["diag"]["F"])
            vb = rb["E_err_over_N"] if key == "E_err_over_N" else (rb["diag"]["E"] if key == "E" else rb["diag"]["F"])
            xs.append(va - vb)
        rec[key] = paired_vals(xs)
    rec["n_energy_wins_a"] = rec["E"]["n_neg"]
    rec["n_F_wins_a"] = rec["F"]["n_pos"]
    return rec


def cell_means(idx):
    seeds = sorted(idx)
    e = np.array([idx[s]["E_err_over_N"] for s in seeds], float)
    f = np.array([idx[s]["diag"]["F"] for s in seeds], float)
    sel = [idx[s]["sel_start"] for s in seeds]
    return {
        "n": len(seeds), "seeds": seeds,
        "E_err_over_N": paired_vals(e),
        "F": paired_vals(f),
        "sel_start": sel,
    }


def fmt(ct, key):
    z = ct[key]
    return (f"{z['mean']:+.5f} med {z['median']:+.5f} "
            f"[{z['ci95_mean'][0]:+.5f},{z['ci95_mean'][1]:+.5f}] "
            f"tails [{z['min']:+.5f},{z['max']:+.5f}]")


def main():
    inst = json.loads((OUT / "INSTANCES.json").read_text()) if (OUT / "INSTANCES.json").exists() else {"rows": []}
    accepted = {(r["N"], r["seed"]) for r in inst["rows"] if r["accepted"]}
    rejected = [r for r in inst["rows"] if not r["accepted"]]
    rows = load_rows()
    audits = {f"{r['seed']}_{r['circuit']}_{r['arm']}": audit_row(r) for r in rows}
    bad = {k: v for k, v in audits.items() if v}
    failures = [r for r in rows if r.get("status") != "ok"]
    ok = [r for r in rows if r.get("status") == "ok"]
    charged = int(sum(r.get("queries", 0) for r in rows))
    expected, missing = [], []
    for seed in SEEDS:
        if (15, seed) not in accepted:
            continue
        for circ in PREREG["batch"]["circuits"]:
            for arm in PREREG["batch"]["arms"]:
                expected.append((seed, circ, arm))
    have = {(r["seed"], r["circuit"], r["arm"]) for r in rows}
    for t in expected:
        if t not in have:
            missing.append(t)
    cells, contrasts, nvq = {}, {}, {}
    for circ in ("Nov3", "QAOA"):
        s4, s8 = index(ok, circ, "S4"), index(ok, circ, "S8")
        cells[f"{circ}_S4"] = cell_means(s4)
        cells[f"{circ}_S8"] = cell_means(s8)
        contrasts[f"{circ}_S8_minus_S4"] = contrast(s8, s4)
    for arm in ("S4", "S8"):
        nvq[f"Nov3_minus_QAOA_{arm}"] = contrast(index(ok, "Nov3", arm), index(ok, "QAOA", arm))
    gate_ct = contrasts["Nov3_S8_minus_S4"]["E"]
    lo, hi = gate_ct["ci95_mean"]
    adopt = bool(gate_ct["n"] > 0 and hi < 0 and gate_ct["median"] < 0)
    summary = {
        "experiment": "STARTS_S8",
        "n_runs": len(rows), "n_ok": len(ok), "n_failures": len(failures),
        "n_expected_from_accepted_instances": len(expected),
        "missing": missing, "rejected_instances": rejected,
        "queries_charged": charged,
        "outside_cap_queries": int(sum(r.get("outside_cap_queries", 0) for r in rows)),
        "audit_failures": bad,
        "prior_n3cond_ledger_preserved": 772224,
        "lexact_spsa_ledger_preserved": 2560,
        "depth_c1_ledger_preserved": 327680,
        "this_stream": "STARTS_S8",
        "cells": cells, "contrasts": contrasts, "nov3_vs_qaoa": nvq,
        "gate": {
            "cell": "N15 L3 Nov3",
            "metric": "mean CI of E_S8 - E_S4 entirely < 0 AND median ΔE < 0",
            "n": gate_ct["n"], "mean": gate_ct["mean"], "median": gate_ct["median"],
            "ci95": gate_ct["ci95_mean"], "n_S8_lower_energy": gate_ct["n_neg"],
            "adopt_S8_as_candidate": adopt,
            "decision": "adopt_candidate" if adopt else "reject",
            "gr": "not patched",
        },
        "seeds": SEEDS,
        "forbidden_seeds_used": [r["seed"] for r in rows if r["seed"] in FORBIDDEN],
    }
    (OUT / "summary_S8.json").write_text(json.dumps(summary, indent=1))
    lines = [
        "# STARTS S8: equal-budget 8 starts vs C0 4 starts at N15 L3",
        "",
        "Exclusive unread development seeds 10048–10055. Ordinary IID C0 SPSA. Energy-only selection. F post hoc.",
        "Q = 3072 inclusive. S4 K=379×4. S8 K=187×8. No continuation, no free prefix.",
        "",
        f"Runs {summary['n_runs']} ok {summary['n_ok']} failures {summary['n_failures']}. "
        f"Charged {summary['queries_charged']}. Outside cap {summary['outside_cap_queries']}. "
        f"Rejected instances {len(rejected)}. Missing {len(missing)}.",
        "",
        f"Gate N15 L3 Nov3 mean ΔE (S8−S4) = {gate_ct['mean']:+.6f} med {gate_ct['median']:+.6f} "
        f"CI [{lo:+.6f}, {hi:+.6f}] n={gate_ct['n']}: **{summary['gate']['decision']}**. GR not patched.",
        "",
        "N3COND 772224, LEXACT SPSA 2560, DEPTH C1 327680 are not revised. This stream is STARTS_S8.",
        "",
        "## S8 vs S4",
        "",
        "| circuit | Δ(E−E0)/N mean/med [CI] tails wins(S8) | ΔE | ΔF wins(S8) |",
        "|---|---|---|---|",
    ]
    for circ in ("Nov3", "QAOA"):
        ct = contrasts[f"{circ}_S8_minus_S4"]
        eN, e, f = ct["E_err_over_N"], ct["E"], ct["F"]
        lines.append(
            f"| {circ} | {fmt(ct, 'E_err_over_N')} {e['n_neg']}/{e['n']} "
            f"| {fmt(ct, 'E')} | {fmt(ct, 'F')} {f['n_pos']}/{f['n']} |"
        )
    lines += [
        "",
        "## Nov3 vs QAOA under each start count",
        "",
        "| arm | Δ(E−E0)/N Nov3−QAOA mean/med [CI] tails Nov3 lower | ΔF Nov3 higher |",
        "|---|---|---|",
    ]
    for arm in ("S4", "S8"):
        ct = nvq[f"Nov3_minus_QAOA_{arm}"]
        lines.append(
            f"| {arm} | {fmt(ct, 'E_err_over_N')} {ct['E']['n_neg']}/{ct['E']['n']} "
            f"| {fmt(ct, 'F')} {ct['F']['n_pos']}/{ct['F']['n']} |"
        )
    (OUT / "RESULTS_S8.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"gate": summary["gate"], "charged": charged, "ok": len(ok),
                      "missing": missing, "audit_failures": bad}, indent=1))
    if bad:
        raise SystemExit("audit failed")
    if missing:
        raise SystemExit(f"missing {len(missing)} expected runs")


if __name__ == "__main__":
    main()
