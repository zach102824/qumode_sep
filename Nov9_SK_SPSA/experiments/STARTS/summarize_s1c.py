"""S1C confirmation. Gate Nov3 S1-S4 mean CI < 0 at N12 AND N15. F diagnostic."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[1] / "results" / "STARTS"
RUNS = OUT / "runs"
PREREG = json.loads((HERE / "PREREG_STARTS_S1C.json").read_text())
SEEDS = list(PREREG["seeds"]["ids"])
FORBIDDEN = set(range(10010, 10056)) | set(range(11000, 11032)) | set(range(20000, 21000))
KEXP = {"S1": 1531, "S4": 379}
NST = {"S1": 1, "S4": 4}
CAL = {"S1": 8, "S4": 32}


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


def load_rows():
    rows = []
    for p in sorted(RUNS.glob("N*/s*_L3_*_S[14]_q1024.json")):
        r = json.loads(p.read_text())
        if r.get("seed") in SEEDS:
            rows.append(r)
    return rows


def audit_row(r):
    problems = []
    arm = r.get("arm")
    if r["seed"] in FORBIDDEN:
        problems.append("forbidden_seed")
    if r["seed"] not in SEEDS:
        problems.append("seed")
    if r.get("L") != 3 or r.get("Q") != 3072 or r.get("N") not in (12, 15):
        problems.append("batch")
    if r.get("band") != "fresh_dev":
        problems.append("band")
    if r.get("status") == "ok":
        if r["queries"] != 3072 or r.get("outside_cap_queries", 0) != 0:
            problems.append("Q")
        if r.get("nstarts") != NST.get(arm) or r.get("K_per_start") != KEXP.get(arm):
            problems.append("split")
        if (r.get("query_cat") or {}).get("calibration") != CAL.get(arm):
            problems.append("cal")
        if any(m.get("gain_frozen") for m in r.get("start_meta") or []):
            problems.append("frozen")
        if r.get("optimizer") != "C0":
            problems.append("optimizer")
        if r.get("reused"):
            problems.append("reused")
    return problems


def index(rows, nq, circ, arm):
    return {r["seed"]: r for r in rows
            if r["N"] == nq and r["circuit"] == circ and r["arm"] == arm and r.get("status") == "ok"}


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
    e = np.array([idx[s]["E_err_over_N"] for s in seeds], float) if seeds else np.array([])
    f = np.array([idx[s]["diag"]["F"] for s in seeds], float) if seeds else np.array([])
    return {"n": len(seeds), "seeds": seeds, "E_err_over_N": paired_vals(e), "F": paired_vals(f)}


def fmt(ct, key):
    z = ct[key]
    return (f"{z['mean']:+.5f} med {z['median']:+.5f} "
            f"[{z['ci95_mean'][0]:+.5f},{z['ci95_mean'][1]:+.5f}] "
            f"tails [{z['min']:+.5f},{z['max']:+.5f}]")


def ci_entirely_below_zero(ct):
    lo, hi = ct["ci95_mean"]
    return bool(ct["n"] > 0 and hi < 0)


def main():
    inst = json.loads((OUT / "INSTANCES_S1C.json").read_text()) if (OUT / "INSTANCES_S1C.json").exists() else {"rows": []}
    rejected = [r for r in inst.get("rows", []) if not r.get("accepted")]
    rows = load_rows()
    audits = {f"N{r['N']}_{r['seed']}_{r['circuit']}_{r['arm']}": audit_row(r) for r in rows}
    bad = {k: v for k, v in audits.items() if v}
    ok = [r for r in rows if r.get("status") == "ok"]
    fail = [r for r in rows if r.get("status") != "ok"]
    charged = int(sum(r.get("queries", 0) for r in rows))
    expected, missing = [], []
    accepted = {(r["N"], r["seed"]) for r in inst.get("rows", []) if r.get("accepted")}
    for nq in (12, 15):
        for seed in SEEDS:
            if (nq, seed) not in accepted:
                continue
            for circ in ("Nov3", "QAOA"):
                for arm in ("S1", "S4"):
                    expected.append((nq, seed, circ, arm))
    have = {(r["N"], r["seed"], r["circuit"], r["arm"]) for r in rows}
    for t in expected:
        if t not in have:
            missing.append(t)
    cells, contrasts, nvq, gates = {}, {}, {}, {}
    for nq in (12, 15):
        for circ in ("Nov3", "QAOA"):
            for arm in ("S1", "S4"):
                cells[f"N{nq}_{circ}_{arm}"] = cell_means(index(ok, nq, circ, arm))
            contrasts[f"N{nq}_{circ}_S1_minus_S4"] = contrast(index(ok, nq, circ, "S1"), index(ok, nq, circ, "S4"))
        for arm in ("S1", "S4"):
            nvq[f"N{nq}_Nov3_minus_QAOA_{arm}"] = contrast(index(ok, nq, "Nov3", arm), index(ok, nq, "QAOA", arm))
        g = contrasts[f"N{nq}_Nov3_S1_minus_S4"]["E"]
        gates[f"N{nq}"] = {
            "n": g["n"], "mean": g["mean"], "median": g["median"],
            "ci95": g["ci95_mean"], "n_S1_lower": g["n_neg"],
            "ci_entirely_below_0": ci_entirely_below_zero(g),
        }
    confirm = bool(gates["N12"]["ci_entirely_below_0"] and gates["N15"]["ci_entirely_below_0"])
    summary = {
        "experiment": "STARTS_S1C",
        "n_runs": len(rows), "n_ok": len(ok), "n_failures": len(fail),
        "failures": [{"N": r["N"], "seed": r["seed"], "circuit": r["circuit"], "arm": r["arm"],
                      "error": r.get("error")} for r in fail],
        "rejected_instances": rejected,
        "missing": missing,
        "queries_charged": charged,
        "outside_cap_queries": int(sum(r.get("outside_cap_queries", 0) for r in rows)),
        "audit_failures": bad,
        "prior_n3cond_ledger_preserved": 772224,
        "lexact_spsa_ledger_preserved": 2560,
        "depth_c1_ledger_preserved": 327680,
        "starts_s8_ledger_preserved": 98304,
        "starts_s1_ledger_preserved": 98304,
        "this_stream": "STARTS_S1C",
        "cells": cells, "contrasts": contrasts, "nov3_vs_qaoa": nvq,
        "gates": gates,
        "confirmation": {
            "pass": confirm,
            "decision": "confirms_on_fresh_dev" if confirm else "does_not_generalize",
            "gr": "not patched",
            "note": "Fair S1 Nov3 vs QAOA energy is reported separately and is not this gate.",
        },
        "seeds": SEEDS, "band": "fresh_dev",
    }
    (OUT / "summary_S1C.json").write_text(json.dumps(summary, indent=1))
    lines = [
        "# STARTS S1C: frozen confirmation of S1 vs S4 on fresh-dev 10056–10063",
        "",
        "Recipes frozen from PR83. No fit changes. Energy selects. F diagnostic only.",
        "Q=3072 inclusive. S1 K=1531×1. S4 K=379×4, fitted and charged on these seeds.",
        "",
        f"Runs {len(rows)} ok {len(ok)} failures {len(fail)}. Charged {charged}. "
        f"Outside cap {summary['outside_cap_queries']}. Rejected B* {len(rejected)}. Missing {len(missing)}.",
        "",
        f"Gate Nov3 S1−S4 mean ΔE CI entirely < 0 at N12 AND N15: **{summary['confirmation']['decision']}**.",
        f"N12 mean {gates['N12']['mean']:+.6f} CI {gates['N12']['ci95']} n={gates['N12']['n']} "
        f"{'pass' if gates['N12']['ci_entirely_below_0'] else 'fail'}.",
        f"N15 mean {gates['N15']['mean']:+.6f} CI {gates['N15']['ci95']} n={gates['N15']['n']} "
        f"{'pass' if gates['N15']['ci_entirely_below_0'] else 'fail'}.",
        "GR not patched. Fair Nov3 vs QAOA is not the S1-vs-S4 gate.",
        "",
        "## S1 vs S4",
        "",
        "| cell | Δ(E−E0)/N mean/med [CI] tails wins(S1) | ΔE | ΔF wins(S1) |",
        "|---|---|---|---|",
    ]
    for nq in (12, 15):
        for circ in ("Nov3", "QAOA"):
            ct = contrasts[f"N{nq}_{circ}_S1_minus_S4"]
            e, f = ct["E"], ct["F"]
            lines.append(
                f"| N{nq} {circ} | {fmt(ct, 'E_err_over_N')} {e['n_neg']}/{e['n']} "
                f"| {fmt(ct, 'E')} | {fmt(ct, 'F')} {f['n_pos']}/{f['n']} |"
            )
    lines += [
        "",
        "## Fair Nov3 vs QAOA (energy and F; not the confirmation gate)",
        "",
        "| cell | Δ(E−E0)/N Nov3−QAOA mean/med [CI] tails Nov3 lower | ΔF Nov3 higher |",
        "|---|---|---|",
    ]
    for nq in (12, 15):
        for arm in ("S1", "S4"):
            ct = nvq[f"N{nq}_Nov3_minus_QAOA_{arm}"]
            lines.append(
                f"| N{nq} {arm} | {fmt(ct, 'E_err_over_N')} {ct['E']['n_neg']}/{ct['E']['n']} "
                f"| {fmt(ct, 'F')} {ct['F']['n_pos']}/{ct['F']['n']} |"
            )
    (OUT / "RESULTS_S1C.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"confirmation": summary["confirmation"], "gates": gates,
                      "charged": charged, "ok": len(ok), "fail": len(fail),
                      "rejected": rejected, "missing": missing, "audit_failures": bad}, indent=1))
    if bad:
        raise SystemExit("audit failed")
    if missing:
        raise SystemExit(f"missing {missing}")


if __name__ == "__main__":
    main()
