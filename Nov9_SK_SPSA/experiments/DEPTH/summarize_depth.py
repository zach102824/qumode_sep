"""Paired DEPTH C1 summary. Energy-only selection. F post hoc. Gate at N15 L3 Nov3."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[1] / "results" / "DEPTH"
RUNS = OUT / "runs"
PREREG = json.loads((HERE / "PREREG_DEPTH_C1.json").read_text())
SEEDS = list(PREREG["seeds"]["ids"])
FORBIDDEN = set(range(10010, 10036)) | set(range(11000, 11032)) | set(range(20000, 21000))


def boot_ci(d, n=10000):
    d = np.asarray(d, float)
    if d.size == 0:
        return [float("nan"), float("nan")]
    r = np.random.default_rng(0)
    m = d[r.integers(0, len(d), size=(n, len(d)))].mean(axis=1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def paired_vals(xs):
    xs = np.asarray(xs, float)
    return {
        "n": int(len(xs)),
        "mean": float(xs.mean()) if len(xs) else float("nan"),
        "median": float(np.median(xs)) if len(xs) else float("nan"),
        "ci95_mean": boot_ci(xs),
        "n_pos": int(np.sum(xs > 0)) if len(xs) else 0,
        "n_neg": int(np.sum(xs < 0)) if len(xs) else 0,
        "n_zero": int(np.sum(xs == 0)) if len(xs) else 0,
        "values": [float(v) for v in xs],
    }


def load_ok():
    rows = []
    for p in sorted(RUNS.glob("N*/s*_q1024.json")):
        r = json.loads(p.read_text())
        rows.append(r)
    return rows


def audit_row(r):
    problems = []
    if r["seed"] in FORBIDDEN:
        problems.append("forbidden_seed")
    if r["seed"] not in SEEDS:
        problems.append("seed_outside_prereg")
    if r.get("band") != "exclusive_unread_dev":
        problems.append("band")
    if r.get("q") != 1024 or r.get("Q") != 1024 * r["L"]:
        problems.append("Q")
    if r.get("status") == "ok":
        if r["queries"] != r["Q"]:
            problems.append("queries_ne_Q")
        if r.get("outside_cap_queries", 0) != 0:
            problems.append("outside_cap")
        if r["arm"] == "cold":
            if r.get("l1_queries", 0) != 0:
                problems.append("cold_l1")
        if r["arm"] == "cont":
            if r.get("l1_queries") != 1024:
                problems.append("cont_l1")
            if r.get("cont_queries") != r["Q"] - 1024:
                problems.append("cont_rem")
            if r["l1_queries"] + r["cont_queries"] != r["Q"]:
                problems.append("cont_sum")
        cats = r.get("query_cat") or {}
        if sum(cats.values()) != r["queries"]:
            problems.append("cat_sum")
        if any(m.get("gain_frozen") for m in r.get("start_meta") or []):
            problems.append("gain_frozen")
        if r.get("optimizer") != "C0":
            problems.append("optimizer")
    return problems


def index(rows, N, L, circ, arm):
    out = {}
    for r in rows:
        if r["N"] == N and r["L"] == L and r["circuit"] == circ and r["arm"] == arm and r.get("status") == "ok":
            out[r["seed"]] = r
    return out


def contrast(a, b, keys=("E_err_over_N", "E", "F")):
    seeds = sorted(set(a) & set(b))
    rec = {"n": len(seeds), "seeds": seeds}
    for key in keys:
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


def better_of(cold, cont):
    out = {}
    seeds = set(cold) | set(cont)
    for s in seeds:
        cands = []
        if s in cold:
            cands.append(("cold", cold[s]))
        if s in cont:
            cands.append(("cont", cont[s]))
        if not cands:
            continue
        arm, row = min(cands, key=lambda t: t[1]["diag"]["E"])
        rec = dict(row)
        rec["better_arm"] = arm
        out[s] = rec
    return out


def cell_means(idx):
    seeds = sorted(idx)
    e = np.array([idx[s]["E_err_over_N"] for s in seeds], float)
    f = np.array([idx[s]["diag"]["F"] for s in seeds], float)
    return {
        "n": len(seeds), "seeds": seeds,
        "E_err_over_N": paired_vals(e),
        "F": paired_vals(f),
    }


def main():
    inst = json.loads((OUT / "INSTANCES.json").read_text()) if (OUT / "INSTANCES.json").exists() else {"rows": []}
    accepted = {(r["N"], r["seed"]) for r in inst["rows"] if r["accepted"]}
    rejected = [r for r in inst["rows"] if not r["accepted"]]
    rows = load_ok()
    audits = {f"{r['N']}_{r['seed']}_{r['L']}_{r['circuit']}_{r['arm']}": audit_row(r) for r in rows}
    bad = {k: v for k, v in audits.items() if v}
    failures = [r for r in rows if r.get("status") != "ok"]
    ok = [r for r in rows if r.get("status") == "ok"]
    charged = int(sum(r.get("queries", 0) for r in rows))
    expected = []
    missing = []
    for nq in PREREG["batch"]["N"]:
        for seed in SEEDS:
            if (nq, seed) not in accepted:
                continue
            for L in PREREG["batch"]["L"]:
                for circ in PREREG["batch"]["circuits"]:
                    for arm in PREREG["batch"]["arms"]:
                        expected.append((nq, seed, L, circ, arm))
    have = {(r["N"], r["seed"], r["L"], r["circuit"], r["arm"]) for r in rows}
    for t in expected:
        if t not in have:
            missing.append(t)
    contrasts = {}
    cells = {}
    better = {}
    nvq = {}
    for nq in PREREG["batch"]["N"]:
        for L in PREREG["batch"]["L"]:
            for circ in PREREG["batch"]["circuits"]:
                cold = index(ok, nq, L, circ, "cold")
                cont = index(ok, nq, L, circ, "cont")
                cells[f"N{nq}_L{L}_{circ}_cold"] = cell_means(cold)
                cells[f"N{nq}_L{L}_{circ}_cont"] = cell_means(cont)
                contrasts[f"N{nq}_L{L}_{circ}_cont_minus_cold"] = contrast(cont, cold)
            bn = better_of(index(ok, nq, L, "Nov3", "cold"), index(ok, nq, L, "Nov3", "cont"))
            bq = better_of(index(ok, nq, L, "QAOA", "cold"), index(ok, nq, L, "QAOA", "cont"))
            better[f"N{nq}_L{L}_Nov3"] = {
                "n": len(bn),
                "arms": {str(s): bn[s]["better_arm"] for s in sorted(bn)},
            }
            better[f"N{nq}_L{L}_QAOA"] = {
                "n": len(bq),
                "arms": {str(s): bq[s]["better_arm"] for s in sorted(bq)},
            }
            nvq[f"N{nq}_L{L}_Nov3_minus_QAOA_better"] = contrast(bn, bq)
    gate_ct = contrasts["N15_L3_Nov3_cont_minus_cold"]["E"]
    lo, hi = gate_ct["ci95_mean"]
    adopt = bool(gate_ct["n"] > 0 and hi < 0)
    summary = {
        "experiment": "DEPTH_C1",
        "n_runs": len(rows),
        "n_ok": len(ok),
        "n_failures": len(failures),
        "n_expected_from_accepted_instances": len(expected),
        "missing": missing,
        "rejected_instances": rejected,
        "queries_charged": charged,
        "outside_cap_queries": int(sum(r.get("outside_cap_queries", 0) for r in rows)),
        "audit_failures": bad,
        "prior_n3cond_ledger_preserved": 772224,
        "lexact_spsa_ledger_preserved": 2560,
        "this_stream": "DEPTH_C1",
        "cells": cells,
        "contrasts": contrasts,
        "better_arm": better,
        "nov3_vs_qaoa_better": nvq,
        "gate": {
            "cell": "N15 L3 Nov3",
            "metric": "mean CI of E_cont - E_cold",
            "n": gate_ct["n"],
            "mean": gate_ct["mean"],
            "ci95": gate_ct["ci95_mean"],
            "n_cont_lower_energy": gate_ct["n_neg"],
            "adopt_continuation": adopt,
            "decision": "adopt" if adopt else "reject",
        },
        "seeds": SEEDS,
        "forbidden_seeds_used": [r["seed"] for r in rows if r["seed"] in FORBIDDEN],
    }
    (OUT / "summary_C1.json").write_text(json.dumps(summary, indent=1))
    lines = [
        "# DEPTH C1: charged L1-continuation vs cold L",
        "",
        "Exclusive unread development seeds 10036–10043. Ordinary IID SPSA. Energy-only selection. F post hoc.",
        "Q = 1024·L inclusive of L1 prefix, continuation, calibration, candidates, and counted failures.",
        "",
        f"Runs {summary['n_runs']} ok {summary['n_ok']} failures {summary['n_failures']}. "
        f"Charged {summary['queries_charged']}. Outside cap {summary['outside_cap_queries']}. "
        f"Rejected instances {len(rejected)}. Missing {len(missing)}.",
        "",
        f"Gate N15 L3 Nov3 mean ΔE (cont−cold) = {gate_ct['mean']:+.6f} "
        f"CI [{lo:+.6f}, {hi:+.6f}] n={gate_ct['n']}: **{summary['gate']['decision']}**.",
        "",
        "N3COND 772224 and LEXACT SPSA 2560 are not revised. This stream is DEPTH_C1.",
        "",
        "## Continuation vs cold",
        "",
        "| cell | Δ(E−E0)/N mean [CI] wins(cont) | ΔE mean [CI] | ΔF mean [CI] wins(cont) |",
        "|---|---|---|---|",
    ]
    for nq in PREREG["batch"]["N"]:
        for L in PREREG["batch"]["L"]:
            for circ in PREREG["batch"]["circuits"]:
                ct = contrasts[f"N{nq}_L{L}_{circ}_cont_minus_cold"]
                eN, e, f = ct["E_err_over_N"], ct["E"], ct["F"]
                lines.append(
                    f"| N{nq} L{L} {circ} | {eN['mean']:+.5f} [{eN['ci95_mean'][0]:+.5f},{eN['ci95_mean'][1]:+.5f}] "
                    f"{e['n_neg']}/{e['n']} | {e['mean']:+.4f} [{e['ci95_mean'][0]:+.4f},{e['ci95_mean'][1]:+.4f}] "
                    f"| {f['mean']:+.4f} [{f['ci95_mean'][0]:+.4f},{f['ci95_mean'][1]:+.4f}] {f['n_pos']}/{f['n']} |"
                )
    lines += [
        "",
        "## Nov3 vs QAOA (per-seed energy-better of cold/continuation)",
        "",
        "| cell | Δ(E−E0)/N mean [CI] Nov3 lower | ΔF mean [CI] Nov3 higher |",
        "|---|---|---|",
    ]
    for nq in PREREG["batch"]["N"]:
        for L in PREREG["batch"]["L"]:
            ct = nvq[f"N{nq}_L{L}_Nov3_minus_QAOA_better"]
            eN, f = ct["E_err_over_N"], ct["F"]
            lines.append(
                f"| N{nq} L{L} | {eN['mean']:+.5f} [{eN['ci95_mean'][0]:+.5f},{eN['ci95_mean'][1]:+.5f}] "
                f"{ct['E']['n_neg']}/{ct['E']['n']} | {f['mean']:+.4f} [{f['ci95_mean'][0]:+.4f},{f['ci95_mean'][1]:+.4f}] "
                f"{f['n_pos']}/{f['n']} |"
            )
    (OUT / "RESULTS_C1.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"gate": summary["gate"], "charged": charged, "ok": len(ok),
                      "missing": missing, "audit_failures": bad}, indent=1))
    if bad:
        raise SystemExit("audit failed")
    if missing:
        raise SystemExit(f"missing {len(missing)} expected runs")


if __name__ == "__main__":
    main()
