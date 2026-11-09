"""S1SH: N15 L=1,2 frozen S1 Nov3 vs QAOA. Soft energy-win note, not GR."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[1] / "results" / "STARTS"
RUNS = OUT / "runs" / "N15"
PREREG = json.loads((HERE / "PREREG_STARTS_S1SH.json").read_text())
SEEDS = list(PREREG["seeds"]["ids"])
LS = [int(L) for L in PREREG["batch"]["L"]]
Q_OF = {1: 1024, 2: 2048}
KEXP = {1024: 507, 2048: 1019}
FORBIDDEN = set(range(10010, 10080)) | set(range(11000, 11032)) | set(range(20000, 21000))


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
    for p in sorted(RUNS.glob("s*_L[12]_*_S1_Q*.json")):
        r = json.loads(p.read_text())
        if r.get("seed") in SEEDS and r.get("stage") == "S1SH":
            rows.append(r)
    return rows


def audit_row(r):
    problems = []
    Q = r.get("Q")
    L = r.get("L")
    if r["seed"] in FORBIDDEN:
        problems.append("forbidden_seed")
    if r["seed"] not in SEEDS:
        problems.append("seed")
    if r.get("N") != 15 or L not in LS or r.get("arm") != "S1":
        problems.append("batch")
    if Q != Q_OF.get(L):
        problems.append("Q_of_L")
    if r.get("band") != "exclusive_unread_dev":
        problems.append("band")
    if r.get("continuation"):
        problems.append("continuation")
    if r.get("status") == "ok":
        if r["queries"] != Q or r.get("outside_cap_queries", 0) != 0:
            problems.append("Q")
        if r.get("nstarts") != 1 or r.get("K_per_start") != KEXP.get(Q):
            problems.append("split")
        if (r.get("query_cat") or {}).get("calibration") != 8:
            problems.append("cal")
        if (r.get("query_cat") or {}).get("candidate") != 2:
            problems.append("cand")
        if any(m.get("gain_frozen") for m in r.get("start_meta") or []):
            problems.append("frozen")
        if r.get("optimizer") != "C0" or r.get("reused"):
            problems.append("optimizer")
    return problems


def index(rows, circ, L):
    return {r["seed"]: r for r in rows
            if r["circuit"] == circ and r.get("L") == L and r.get("status") == "ok"}


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


def energy_nov3_win(ct):
    hi = ct["E_err_over_N"]["ci95_mean"][1]
    return bool(ct["n"] > 0 and hi < 0)


def main():
    inst = json.loads((OUT / "INSTANCES_S1SH.json").read_text()) if (OUT / "INSTANCES_S1SH.json").exists() else {"rows": []}
    rejected = [r for r in inst.get("rows", []) if not r.get("accepted")]
    rows = load_rows()
    audits = {f"{r['seed']}_{r['circuit']}_L{r['L']}": audit_row(r) for r in rows}
    bad = {k: v for k, v in audits.items() if v}
    ok = [r for r in rows if r.get("status") == "ok"]
    fail = [r for r in rows if r.get("status") != "ok"]
    charged = int(sum(r.get("queries", 0) for r in rows))
    accepted = {r["seed"] for r in inst.get("rows", []) if r.get("accepted")}
    expected, missing = [], []
    for seed in SEEDS:
        if seed not in accepted:
            continue
        for L in LS:
            for circ in ("Nov3", "QAOA"):
                expected.append((seed, circ, L))
    have = {(r["seed"], r["circuit"], r["L"]) for r in rows}
    for t in expected:
        if t not in have:
            missing.append(t)
    cells, nvq = {}, {}
    candidates = []
    for L in LS:
        for circ in ("Nov3", "QAOA"):
            cells[f"{circ}_L{L}"] = cell_means(index(ok, circ, L))
        nvq[L] = contrast(index(ok, "Nov3", L), index(ok, "QAOA", L))
        if energy_nov3_win(nvq[L]):
            candidates.append(L)
    honesty = "gap_already_at_shallow_depth" if not candidates else "development_energy_win_candidate"
    if candidates and set(candidates) != set(LS) and any(not energy_nov3_win(nvq[L]) for L in LS):
        honesty = "mixed_by_depth"
    summary = {
        "experiment": "STARTS_S1SH",
        "n_runs": len(rows), "n_ok": len(ok), "n_failures": len(fail),
        "failures": [{"seed": r["seed"], "circuit": r["circuit"], "L": r["L"],
                      "error": r.get("error")} for r in fail],
        "rejected_instances": rejected, "missing": missing,
        "queries_charged": charged,
        "outside_cap_queries": int(sum(r.get("outside_cap_queries", 0) for r in rows)),
        "audit_failures": bad,
        "prior_n3cond_ledger_preserved": 772224,
        "lexact_spsa_ledger_preserved": 2560,
        "depth_c1_ledger_preserved": 327680,
        "starts_s8_ledger_preserved": 98304,
        "starts_s1_ledger_preserved": 98304,
        "starts_s1c_ledger_preserved": 196608,
        "starts_s1l6_ledger_preserved": 196608,
        "starts_s1q_ledger_preserved": 344064,
        "this_stream": "STARTS_S1SH",
        "cells": cells, "nov3_vs_qaoa": {str(L): nvq[L] for L in LS},
        "soft_note": {
            "development_energy_win_Ls": candidates,
            "rule": "Nov3−QAOA mean Δ(E-E0)/N CI entirely below 0",
            "gr": "not patched",
            "s1": "remains a candidate",
            "honesty": honesty,
        },
        "seeds": SEEDS, "band": "exclusive_unread_dev",
        "continuation": False,
    }
    (OUT / "summary_S1SH.json").write_text(json.dumps(summary, indent=1))
    lines = [
        "# STARTS S1SH: frozen S1 Nov3 vs QAOA at N15 L=1,2 Q=qL (cold)",
        "",
        "Exclusive unread seeds 10080–10087. Frozen PR83/84/85 S1 (TQA-only C0). Energy selects. F post hoc.",
        "L∈{1,2}, Q=1024·L. S1 K=507 / 1019. Cold independent L. Not continuation. GR not patched.",
        "",
        f"Runs {len(rows)} ok {len(ok)} failures {len(fail)}. Charged {charged}. "
        f"Outside cap {summary['outside_cap_queries']}. Rejected B* {len(rejected)}. Missing {len(missing)}.",
        "",
        f"Soft note: **{honesty}**. Development energy-win L: {candidates or 'none'}. Not a GR adoption.",
        "",
        "## Primary: Nov3 − QAOA under S1",
        "",
        "| L | Q | Δ(E−E0)/N mean/med [CI] tails Nov3 lower | ΔE | ΔF | energy-win candidate? |",
        "|---|---|---|---|---|---|",
    ]
    for L in LS:
        ct = nvq[L]
        win = "yes" if energy_nov3_win(ct) else "no"
        lines.append(
            f"| {L} | {Q_OF[L]} | {fmt(ct, 'E_err_over_N')} {ct['E']['n_neg']}/{ct['E']['n']} "
            f"| {fmt(ct, 'E')} | {fmt(ct, 'F')} {ct['F']['n_pos']}/{ct['F']['n']} | {win} |"
        )
    (OUT / "RESULTS_S1SH.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"soft_note": summary["soft_note"], "charged": charged, "ok": len(ok),
                      "fail": len(fail), "rejected": rejected, "missing": missing,
                      "audit_failures": bad}, indent=1))
    if bad:
        raise SystemExit("audit failed")
    if missing:
        raise SystemExit(f"missing {missing}")


if __name__ == "__main__":
    main()
