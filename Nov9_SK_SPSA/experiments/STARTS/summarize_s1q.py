"""S1Q: N15 L3 frozen S1 Nov3 vs QAOA across Q. Soft energy-win note, not GR."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[1] / "results" / "STARTS"
RUNS = OUT / "runs" / "N15"
PREREG = json.loads((HERE / "PREREG_STARTS_S1Q.json").read_text())
SEEDS = list(PREREG["seeds"]["ids"])
QS = [int(q) for q in PREREG["batch"]["Q"]]
FORBIDDEN = set(range(10010, 10072)) | set(range(11000, 11032)) | set(range(20000, 21000))
KEXP = {3072: 1531, 6144: 3067, 12288: 6139}


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
    for p in sorted(RUNS.glob("s*_L3_*_S1_Q*.json")):
        r = json.loads(p.read_text())
        if r.get("seed") in SEEDS and r.get("stage") == "S1Q":
            rows.append(r)
    return rows


def audit_row(r):
    problems = []
    Q = r.get("Q")
    if r["seed"] in FORBIDDEN:
        problems.append("forbidden_seed")
    if r["seed"] not in SEEDS:
        problems.append("seed")
    if r.get("N") != 15 or r.get("L") != 3 or r.get("arm") != "S1":
        problems.append("batch")
    if Q not in KEXP:
        problems.append("Q_unknown")
    if r.get("band") != "exclusive_unread_dev":
        problems.append("band")
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


def index(rows, circ, Q):
    return {r["seed"]: r for r in rows
            if r["circuit"] == circ and r.get("Q") == Q and r.get("status") == "ok"}


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
    inst = json.loads((OUT / "INSTANCES_S1Q.json").read_text()) if (OUT / "INSTANCES_S1Q.json").exists() else {"rows": []}
    rejected = [r for r in inst.get("rows", []) if not r.get("accepted")]
    status = json.loads((OUT / "STATUS_S1Q.json").read_text()) if (OUT / "STATUS_S1Q.json").exists() else {}
    rows = load_rows()
    audits = {f"{r['seed']}_{r['circuit']}_Q{r['Q']}": audit_row(r) for r in rows}
    bad = {k: v for k, v in audits.items() if v}
    ok = [r for r in rows if r.get("status") == "ok"]
    fail = [r for r in rows if r.get("status") != "ok"]
    charged = int(sum(r.get("queries", 0) for r in rows))
    accepted = {r["seed"] for r in inst.get("rows", []) if r.get("accepted")}
    ran_q = set(QS)
    if status.get("q12288") == "skipped_heavy":
        ran_q = {3072, 6144}
    expected, missing = [], []
    for seed in SEEDS:
        if seed not in accepted:
            continue
        for circ in ("Nov3", "QAOA"):
            for Q in ran_q:
                expected.append((seed, circ, Q))
    have = {(r["seed"], r["circuit"], r["Q"]) for r in rows}
    for t in expected:
        if t not in have:
            missing.append(t)
    cells, nvq, vs3072 = {}, {}, {}
    candidates = []
    for Q in QS:
        for circ in ("Nov3", "QAOA"):
            cells[f"{circ}_Q{Q}"] = cell_means(index(ok, circ, Q))
        nvq[Q] = contrast(index(ok, "Nov3", Q), index(ok, "QAOA", Q))
        if energy_nov3_win(nvq[Q]):
            candidates.append(Q)
        if Q != 3072:
            for circ in ("Nov3", "QAOA"):
                vs3072[f"{circ}_Q{Q}_minus_Q3072"] = contrast(index(ok, circ, Q), index(ok, circ, 3072))
    honesty = "gap_persists_vs_budget" if not candidates else "development_energy_win_candidate"
    summary = {
        "experiment": "STARTS_S1Q",
        "n_runs": len(rows), "n_ok": len(ok), "n_failures": len(fail),
        "failures": [{"seed": r["seed"], "circuit": r["circuit"], "Q": r["Q"],
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
        "this_stream": "STARTS_S1Q",
        "cells": cells, "nov3_vs_qaoa": {str(Q): nvq[Q] for Q in QS},
        "vs_own_q3072": vs3072,
        "soft_note": {
            "development_energy_win_Qs": candidates,
            "rule": "Nov3−QAOA mean Δ(E-E0)/N CI entirely below 0",
            "gr": "not patched",
            "honesty": honesty,
        },
        "resource": {
            "q12288": status.get("q12288"),
            "blocker": status.get("blocker"),
            "stage_wall_s": status.get("stage_wall_s"),
            "estimated_q12288_s": status.get("estimated_q12288_s"),
        },
        "seeds": SEEDS, "band": "exclusive_unread_dev",
    }
    (OUT / "summary_S1Q.json").write_text(json.dumps(summary, indent=1))
    lines = [
        "# STARTS S1Q: frozen S1 Nov3 vs QAOA at N15 L=3 vs budget",
        "",
        "Exclusive unread seeds 10072–10079. Frozen PR83/84/85 S1 (TQA-only C0). Energy selects. F post hoc.",
        "Q ∈ {3072, 6144, 12288} inclusive. S1 K=1531 / 3067 / 6139. No S4/S8. GR not patched.",
        "",
        f"Runs {len(rows)} ok {len(ok)} failures {len(fail)}. Charged {charged}. "
        f"Outside cap {summary['outside_cap_queries']}. Rejected B* {len(rejected)}. Missing {len(missing)}.",
        f"Q=12288: {status.get('q12288')}. Blocker: {status.get('blocker')}.",
        "",
        f"Soft note: **{honesty}**. Development energy-win Qs: {candidates or 'none'}. Not a GR adoption.",
        "",
        "## Primary: Nov3 − QAOA under S1",
        "",
        "| Q | Δ(E−E0)/N mean/med [CI] tails Nov3 lower | ΔE | ΔF | energy-win candidate? |",
        "|---|---|---|---|---|",
    ]
    for Q in QS:
        ct = nvq[Q]
        if ct["n"] == 0:
            lines.append(f"| {Q} | not run | | | |")
            continue
        win = "yes" if energy_nov3_win(ct) else "no"
        lines.append(
            f"| {Q} | {fmt(ct, 'E_err_over_N')} {ct['E']['n_neg']}/{ct['E']['n']} "
            f"| {fmt(ct, 'E')} | {fmt(ct, 'F')} {ct['F']['n_pos']}/{ct['F']['n']} | {win} |"
        )
    lines += [
        "",
        "## Secondary: within-method vs own Q=3072 (negative energy = improved)",
        "",
        "| method | Q | Δ(E−E0)/N vs Q=3072 mean/med [CI] tails | ΔF |",
        "|---|---|---|---|",
    ]
    for Q in (6144, 12288):
        for circ in ("Nov3", "QAOA"):
            key = f"{circ}_Q{Q}_minus_Q3072"
            if key not in vs3072 or vs3072[key]["n"] == 0:
                lines.append(f"| {circ} | {Q} | not run | |")
                continue
            ct = vs3072[key]
            lines.append(
                f"| {circ} | {Q} | {fmt(ct, 'E_err_over_N')} {ct['E']['n_neg']}/{ct['E']['n']} "
                f"| {fmt(ct, 'F')} {ct['F']['n_pos']}/{ct['F']['n']} |"
            )
    (OUT / "RESULTS_S1Q.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"soft_note": summary["soft_note"], "resource": summary["resource"],
                      "charged": charged, "ok": len(ok), "fail": len(fail),
                      "rejected": rejected, "missing": missing, "audit_failures": bad}, indent=1))
    if bad:
        raise SystemExit("audit failed")
    if missing:
        raise SystemExit(f"missing {missing}")


if __name__ == "__main__":
    main()
