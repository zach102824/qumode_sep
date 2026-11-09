"""DEPTH S1W: Gate A WARM<COLD and Gate B WARM<QAOA at N15 L2. Report all cells."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[1] / "results" / "DEPTH"
PREREG = json.loads((HERE / "PREREG_DEPTH_S1W.json").read_text())
SEEDS = list(PREREG["seeds"]["ids"])
NS = [12, 15]
LS = [2, 3]
Q_OF = {2: 2048, 3: 3072}
KEXP = {1024: 507, 2048: 1019, 3072: 1531}
FORBIDDEN = set(range(10010, 10128)) | set(range(11000, 11032)) | set(range(20000, 21000))


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
    for nq in NS:
        d = OUT / "runs" / f"N{nq}"
        if not d.exists():
            continue
        for p in sorted(d.glob("s*_L[23]_*_Q*.json")):
            r = json.loads(p.read_text())
            if r.get("seed") in SEEDS and r.get("stage") == "S1W":
                rows.append(r)
    return rows


def audit_row(r):
    problems = []
    arm, L, Q = r.get("arm"), r.get("L"), r.get("Q")
    if r["seed"] in FORBIDDEN:
        problems.append("forbidden_seed")
    if r["seed"] not in SEEDS:
        problems.append("seed")
    if r.get("N") not in NS or L not in LS or Q != Q_OF.get(L):
        problems.append("batch")
    if r.get("band") != "exclusive_unread_dev":
        problems.append("band")
    if arm == "WARM" and not r.get("continuation"):
        problems.append("continuation_flag")
    if arm != "WARM" and r.get("continuation"):
        problems.append("cold_continuation")
    if r.get("status") == "ok":
        if r["queries"] != Q or r.get("outside_cap_queries", 0) != 0:
            problems.append("Q")
        if r.get("nstarts") != 1:
            problems.append("nstarts")
        if r.get("reused"):
            problems.append("reused")
        if any(m.get("gain_frozen") for m in r.get("start_meta") or []):
            problems.append("frozen")
        if arm in ("COLD", "QAOA"):
            if r.get("K_per_start") != KEXP[Q]:
                problems.append("split")
            if (r.get("query_cat") or {}).get("calibration") != 8:
                problems.append("cal")
            if (r.get("query_cat") or {}).get("candidate") != 2:
                problems.append("cand")
            if r.get("l1_queries", 0) != 0:
                problems.append("l1")
        if arm == "WARM":
            rem = Q - 1024
            if r.get("l1_queries") != 1024 or r.get("cont_queries") != rem:
                problems.append("share")
            if r.get("K_per_start") != KEXP[rem]:
                problems.append("split")
            qc = r.get("query_cat") or {}
            if qc.get("calibration") != 16:
                problems.append("cal")
            if qc.get("candidate") != 4:
                problems.append("cand")
            pad = np.asarray(r.get("x_pad") or [], float)
            xl1 = np.asarray(r.get("x_l1") or [], float)
            if pad.size != 3 + 3 * L or xl1.size != 6:
                problems.append("pad_size")
            elif float(np.max(np.abs(pad[:6] - xl1))) > 0 or float(np.max(np.abs(pad[6:]))) > 0:
                problems.append("pad")
            want_circ = "QAOA" if arm == "QAOA" else "Nov3"
            if r.get("circuit") != want_circ:
                problems.append("circuit")
        if arm == "QAOA" and r.get("circuit") != "QAOA":
            problems.append("circuit")
        if arm in ("COLD", "WARM") and r.get("circuit") != "Nov3":
            problems.append("circuit")
    return problems


def index(rows, nq, L, arm):
    return {r["seed"]: r for r in rows
            if r["N"] == nq and r["L"] == L and r["arm"] == arm and r.get("status") == "ok"}


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


def pass_below(ct):
    return bool(ct["n"] > 0 and ct["E"]["ci95_mean"][1] < 0)


def main():
    inst = json.loads((OUT / "INSTANCES_S1W.json").read_text()) if (OUT / "INSTANCES_S1W.json").exists() else {"rows": []}
    rejected = [r for r in inst.get("rows", []) if not r.get("accepted")]
    rows = load_rows()
    audits = {f"N{r['N']}_s{r['seed']}_L{r['L']}_{r['arm']}": audit_row(r) for r in rows}
    bad = {k: v for k, v in audits.items() if v}
    ok = [r for r in rows if r.get("status") == "ok"]
    fail = [r for r in rows if r.get("status") != "ok"]
    charged = int(sum(r.get("queries", 0) for r in rows))
    accepted = {(r["N"], r["seed"]) for r in inst.get("rows", []) if r.get("accepted")}
    expected, missing = [], []
    for nq in NS:
        for seed in SEEDS:
            if (nq, seed) not in accepted:
                continue
            for L in LS:
                for arm in ("COLD", "WARM", "QAOA"):
                    expected.append((nq, seed, L, arm))
    have = {(r["N"], r["seed"], r["L"], r["arm"]) for r in rows}
    for t in expected:
        if t not in have:
            missing.append(t)
    cells, contrasts, gates = {}, {}, {}
    for nq in NS:
        for L in LS:
            for arm in ("COLD", "WARM", "QAOA"):
                cells[f"N{nq}_L{L}_{arm}"] = cell_means(index(ok, nq, L, arm))
            contrasts[f"N{nq}_L{L}_WARM_minus_COLD"] = contrast(
                index(ok, nq, L, "WARM"), index(ok, nq, L, "COLD"))
            contrasts[f"N{nq}_L{L}_WARM_minus_QAOA"] = contrast(
                index(ok, nq, L, "WARM"), index(ok, nq, L, "QAOA"))
            contrasts[f"N{nq}_L{L}_COLD_minus_QAOA"] = contrast(
                index(ok, nq, L, "COLD"), index(ok, nq, L, "QAOA"))
            ga = pass_below(contrasts[f"N{nq}_L{L}_WARM_minus_COLD"])
            gb = pass_below(contrasts[f"N{nq}_L{L}_WARM_minus_QAOA"])
            gates[f"N{nq}_L{L}"] = {"A": ga, "B": gb, "both": ga and gb}
    adopt = bool(gates["N15_L2"]["both"])
    summary = {
        "experiment": "DEPTH_S1W",
        "n_runs": len(rows), "n_ok": len(ok), "n_failures": len(fail),
        "failures": [{"N": r["N"], "seed": r["seed"], "L": r["L"], "arm": r["arm"],
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
        "starts_s1sh_ledger_preserved": 49152,
        "starts_s1alt_ledger_preserved": 65536,
        "starts_s1alt23_ledger_preserved": 163840,
        "starts_s1alt13_ledger_preserved": 98304,
        "this_stream": "DEPTH_S1W",
        "cells": cells, "contrasts": contrasts, "gates_by_cell": gates,
        "gate": {
            "adopt_cell": "N15 L2",
            "A": gates["N15_L2"]["A"],
            "B": gates["N15_L2"]["B"],
            "both_pass": adopt,
            "decision": "adopt_for_n18_consideration" if adopt else "stop_no_n18",
            "gr": "not patched",
        },
        "seeds": SEEDS, "band": "exclusive_unread_dev",
    }
    (OUT / "summary_S1W.json").write_text(json.dumps(summary, indent=1))
    gA = contrasts["N15_L2_WARM_minus_COLD"]["E"]
    gB = contrasts["N15_L2_WARM_minus_QAOA"]["E"]
    lines = [
        "# DEPTH S1W: S1JOINT L1→L warm-start vs cold S1JOINT and QAOA p=L",
        "",
        "Exclusive unread seeds 10128–10135. Q=1024·L. COLD/QAOA = frozen S1JOINT. "
        "WARM: Q1=1024 at L=1, extra shells 0, full-vector C0 continue. Energy selects. F post hoc. GR not patched.",
        "",
        f"Runs {len(rows)} ok {len(ok)} failures {len(fail)}. Charged {charged}. "
        f"Outside cap {summary['outside_cap_queries']}. Rejected B* {len(rejected)}. Missing {len(missing)}.",
        "",
        f"Adopt cell N15 L2 Gate A (WARM−COLD) mean ΔE {gA['mean']:+.6f} CI [{gA['ci95_mean'][0]:+.6f}, {gA['ci95_mean'][1]:+.6f}] "
        f"{gA['n_neg']}/{gA['n']}: **{'pass' if gates['N15_L2']['A'] else 'fail'}**.",
        f"Adopt cell N15 L2 Gate B (WARM−QAOA) mean ΔE {gB['mean']:+.6f} CI [{gB['ci95_mean'][0]:+.6f}, {gB['ci95_mean'][1]:+.6f}] "
        f"{gB['n_neg']}/{gB['n']}: **{'pass' if gates['N15_L2']['B'] else 'fail'}**.",
        f"Decision: **{summary['gate']['decision']}**.",
        "",
        "## Gate A: WARM − COLD",
        "",
        "| N | L | Q | mean ΔE med [CI] tails wins(WARM) | Δ(E−E0)/N | ΔF wins | A |",
        "|---|---|---|---|---|---|---|",
    ]
    for nq in NS:
        for L in LS:
            ct = contrasts[f"N{nq}_L{L}_WARM_minus_COLD"]
            lines.append(
                f"| {nq} | {L} | {Q_OF[L]} | {fmt(ct, 'E')} {ct['E']['n_neg']}/{ct['E']['n']} "
                f"| {fmt(ct, 'E_err_over_N')} | {fmt(ct, 'F')} {ct['F']['n_pos']}/{ct['F']['n']} "
                f"| {'yes' if gates[f'N{nq}_L{L}']['A'] else 'no'} |"
            )
    lines += [
        "",
        "## Gate B: WARM − QAOA",
        "",
        "| N | L | Q | mean ΔE med [CI] tails wins(WARM) | Δ(E−E0)/N | ΔF wins | B |",
        "|---|---|---|---|---|---|---|",
    ]
    for nq in NS:
        for L in LS:
            ct = contrasts[f"N{nq}_L{L}_WARM_minus_QAOA"]
            lines.append(
                f"| {nq} | {L} | {Q_OF[L]} | {fmt(ct, 'E')} {ct['E']['n_neg']}/{ct['E']['n']} "
                f"| {fmt(ct, 'E_err_over_N')} | {fmt(ct, 'F')} {ct['F']['n_pos']}/{ct['F']['n']} "
                f"| {'yes' if gates[f'N{nq}_L{L}']['B'] else 'no'} |"
            )
    lines += [
        "",
        "## Fair COLD Nov3 vs QAOA (not a gate)",
        "",
        "| N | L | Δ(E−E0)/N COLD−QAOA mean/med [CI] tails Nov3 lower | ΔF |",
        "|---|---|---|---|",
    ]
    for nq in NS:
        for L in LS:
            ct = contrasts[f"N{nq}_L{L}_COLD_minus_QAOA"]
            lines.append(
                f"| {nq} | {L} | {fmt(ct, 'E_err_over_N')} {ct['E']['n_neg']}/{ct['E']['n']} "
                f"| {fmt(ct, 'F')} {ct['F']['n_pos']}/{ct['F']['n']} |"
            )
    (OUT / "RESULTS_S1W.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"gate": summary["gate"], "gates_by_cell": gates, "charged": charged,
                      "ok": len(ok), "fail": len(fail), "rejected": rejected,
                      "missing": missing, "audit_failures": bad}, indent=1))
    if bad:
        raise SystemExit("audit failed")
    if missing:
        raise SystemExit(f"missing {missing}")


if __name__ == "__main__":
    main()
