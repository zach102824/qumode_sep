"""Paired STARTS S1 summary. Energy-only. F post hoc. Gate Nov3 S1 vs reused S4."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[1] / "results" / "STARTS"
RUNS = OUT / "runs" / "N15"
PREREG = json.loads((HERE / "PREREG_STARTS_S1.json").read_text())
SEEDS = list(PREREG["seeds"]["ids"])
FORBIDDEN = set(range(11000, 11032)) | set(range(20000, 21000))
KEXP = {"S1": 1531, "S2": 763, "S4": 379}
NST = {"S1": 1, "S2": 2, "S4": 4}
CAL = {"S1": 8, "S2": 16, "S4": 32}


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
        "n": int(len(xs)), "mean": float(xs.mean()), "median": float(np.median(xs)),
        "ci95_mean": boot_ci(xs), "n_pos": int(np.sum(xs > 0)),
        "n_neg": int(np.sum(xs < 0)), "n_zero": int(np.sum(xs == 0)),
        "min": float(xs.min()), "max": float(xs.max()),
        "p10": float(np.percentile(xs, 10)), "p90": float(np.percentile(xs, 90)),
        "values": [float(v) for v in xs],
    }


def load_arm(arm):
    rows = []
    for p in sorted(RUNS.glob(f"s*_L3_*_{arm}_q1024.json")):
        rows.append(json.loads(p.read_text()))
    return rows


def audit_row(r, arm):
    problems = []
    if r["seed"] in FORBIDDEN:
        problems.append("forbidden_seed")
    if r["seed"] not in SEEDS:
        problems.append("seed")
    if r.get("N") != 15 or r.get("L") != 3 or r.get("Q") != 3072:
        problems.append("batch")
    if r.get("status") == "ok":
        if r["queries"] != 3072 or r.get("outside_cap_queries", 0) != 0:
            problems.append("Q")
        if r.get("nstarts") != NST[arm] or r.get("K_per_start") != KEXP[arm]:
            problems.append("split")
        if (r.get("query_cat") or {}).get("calibration") != CAL[arm]:
            problems.append("cal")
        if any(m.get("gain_frozen") for m in r.get("start_meta") or []):
            problems.append("frozen")
        if r.get("optimizer") != "C0":
            problems.append("optimizer")
        if arm != "S4" and r.get("band") != "read_dev":
            problems.append("band")
    return problems


def index(rows, circ):
    return {r["seed"]: r for r in rows if r["circuit"] == circ and r.get("status") == "ok"}


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
    return {"n": len(seeds), "seeds": seeds, "E_err_over_N": paired_vals(e),
            "F": paired_vals(f), "sel_start": [idx[s]["sel_start"] for s in seeds]}


def fmt(ct, key):
    z = ct[key]
    return (f"{z['mean']:+.5f} med {z['median']:+.5f} "
            f"[{z['ci95_mean'][0]:+.5f},{z['ci95_mean'][1]:+.5f}] "
            f"tails [{z['min']:+.5f},{z['max']:+.5f}]")


def main():
    reuse = json.loads((OUT / "S4_REUSE_AUDIT.json").read_text()) if (OUT / "S4_REUSE_AUDIT.json").exists() else {}
    by = {arm: load_arm(arm) for arm in ("S1", "S2", "S4")}
    audits = {}
    for arm, rows in by.items():
        for r in rows:
            bad = audit_row(r, arm)
            if bad:
                audits[f"{r['seed']}_{r['circuit']}_{arm}"] = bad
    new = [r for arm in ("S1", "S2") for r in by[arm]]
    s4_reused = [r for r in by["S4"] if r.get("status") == "ok"]
    charged = int(sum(r.get("queries", 0) for r in new if r.get("status") == "ok"))
    ok = {arm: [r for r in rows if r.get("status") == "ok"] for arm, rows in by.items()}
    cells, contrasts, nvq = {}, {}, {}
    for circ in ("Nov3", "QAOA"):
        for arm in ("S1", "S2", "S4"):
            cells[f"{circ}_{arm}"] = cell_means(index(ok[arm], circ))
        contrasts[f"{circ}_S1_minus_S4"] = contrast(index(ok["S1"], circ), index(ok["S4"], circ))
        contrasts[f"{circ}_S2_minus_S4"] = contrast(index(ok["S2"], circ), index(ok["S4"], circ))
    for arm in ("S1", "S4"):
        nvq[f"Nov3_minus_QAOA_{arm}"] = contrast(index(ok[arm], "Nov3"), index(ok[arm], "QAOA"))
    gate_ct = contrasts["Nov3_S1_minus_S4"]["E"]
    lo, hi = gate_ct["ci95_mean"]
    adopt = bool(gate_ct["n"] > 0 and hi < 0 and gate_ct["median"] < 0)
    missing = []
    for seed in SEEDS:
        for circ in ("Nov3", "QAOA"):
            for arm in ("S1", "S2", "S4"):
                if seed not in index(ok[arm], circ):
                    missing.append((seed, circ, arm))
    summary = {
        "experiment": "STARTS_S1",
        "s4_reused": bool(reuse.get("all_ok")),
        "s4_not_recharged": len(s4_reused),
        "n_new_ok": len([r for r in new if r.get("status") == "ok"]),
        "n_new_fail": len([r for r in new if r.get("status") != "ok"]),
        "queries_newly_charged": charged,
        "outside_cap_queries": int(sum(r.get("outside_cap_queries", 0) for r in new)),
        "s4_queries_not_added_again": 16 * 3072,
        "audit_failures": audits,
        "missing": missing,
        "prior_n3cond_ledger_preserved": 772224,
        "lexact_spsa_ledger_preserved": 2560,
        "depth_c1_ledger_preserved": 327680,
        "starts_s8_ledger_preserved": 98304,
        "this_stream": "STARTS_S1",
        "cells": cells, "contrasts": contrasts, "nov3_vs_qaoa": nvq,
        "gate": {
            "cell": "N15 L3 Nov3",
            "metric": "mean CI of E_S1 - E_S4 entirely < 0 AND median ΔE < 0",
            "n": gate_ct["n"], "mean": gate_ct["mean"], "median": gate_ct["median"],
            "ci95": gate_ct["ci95_mean"], "n_S1_lower_energy": gate_ct["n_neg"],
            "adopt_S1_as_candidate": adopt,
            "decision": "adopt_candidate" if adopt else "reject",
            "gr": "not patched",
        },
        "seeds": SEEDS, "band": "read_dev",
    }
    (OUT / "summary_S1.json").write_text(json.dumps(summary, indent=1))
    lines = [
        "# STARTS S1: concentrate Q on TQA vs historical four-start C0",
        "",
        "Read-dev seeds 10048–10055. S4 reused from PR82 after recipe audit. Ordinary C0 SPSA. Energy-only. F post hoc.",
        "Q=3072 inclusive. S1 K=1531×1. S2 K=763×2 (not in the gate). S4 K=379×4 reused, not recharged.",
        "",
        f"New ok {summary['n_new_ok']} fail {summary['n_new_fail']}. Newly charged {charged}. "
        f"Outside cap {summary['outside_cap_queries']}. S4 reused {summary['s4_reused']} "
        f"({summary['s4_not_recharged']} files, {summary['s4_queries_not_added_again']} not added again).",
        "",
        f"Gate N15 L3 Nov3 mean ΔE (S1−S4) = {gate_ct['mean']:+.6f} med {gate_ct['median']:+.6f} "
        f"CI [{lo:+.6f}, {hi:+.6f}] n={gate_ct['n']}: **{summary['gate']['decision']}**. GR not patched.",
        "",
        "N3COND 772224, LEXACT 2560, DEPTH C1 327680, STARTS S8 98304 are not revised.",
        "",
        "## S1/S2 vs reused S4",
        "",
        "| contrast | Δ(E−E0)/N mean/med [CI] tails wins | ΔE | ΔF wins |",
        "|---|---|---|---|",
    ]
    for circ in ("Nov3", "QAOA"):
        for arm in ("S1", "S2"):
            ct = contrasts[f"{circ}_{arm}_minus_S4"]
            e, f = ct["E"], ct["F"]
            lines.append(
                f"| {circ} {arm}−S4 | {fmt(ct, 'E_err_over_N')} {e['n_neg']}/{e['n']} "
                f"| {fmt(ct, 'E')} | {fmt(ct, 'F')} {f['n_pos']}/{f['n']} |"
            )
    lines += [
        "",
        "## Nov3 vs QAOA under S1 and S4",
        "",
        "| arm | Δ(E−E0)/N Nov3−QAOA mean/med [CI] tails Nov3 lower | ΔF Nov3 higher |",
        "|---|---|---|",
    ]
    for arm in ("S1", "S4"):
        ct = nvq[f"Nov3_minus_QAOA_{arm}"]
        lines.append(
            f"| {arm} | {fmt(ct, 'E_err_over_N')} {ct['E']['n_neg']}/{ct['E']['n']} "
            f"| {fmt(ct, 'F')} {ct['F']['n_pos']}/{ct['F']['n']} |"
        )
    (OUT / "RESULTS_S1.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"gate": summary["gate"], "charged": charged, "s4_reused": summary["s4_reused"],
                      "missing": missing, "audit_failures": audits}, indent=1))
    if audits:
        raise SystemExit("audit failed")
    if missing:
        raise SystemExit(f"missing {missing}")


if __name__ == "__main__":
    main()
