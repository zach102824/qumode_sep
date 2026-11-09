"""FTREP gates R1/R2: N15 and N12 Nov3−QAOA mean ΔE CI entirely <0. FT bootstrap."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[1] / "results" / "FTREP"
PREREG = json.loads((HERE / "PREREG_FTREP.json").read_text())
SEEDS = list(PREREG["seeds"]["ids"])
NS = [12, 15]
ARMS = ("nov3_center4", "qaoa_center4")
FORBIDDEN = set(range(10010, 10152)) | set(range(11000, 11032)) | set(range(20000, 21000))
N_BOOT = 4000


def boot_ci(d, rng):
    d = np.asarray(d, float)
    if d.size == 0:
        return [float("nan"), float("nan")]
    m = d[rng.integers(0, len(d), size=(N_BOOT, len(d)))].mean(axis=1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def paired_vals(xs, rng):
    xs = np.asarray(xs, float)
    if len(xs) == 0:
        return {"n": 0, "mean": float("nan"), "median": float("nan"),
                "ci95_mean": [float("nan"), float("nan")],
                "n_pos": 0, "n_neg": 0, "n_zero": 0,
                "min": float("nan"), "max": float("nan"),
                "p10": float("nan"), "p90": float("nan"), "values": []}
    return {
        "n": int(len(xs)), "mean": float(xs.mean()), "median": float(np.median(xs)),
        "ci95_mean": boot_ci(xs, rng), "n_pos": int(np.sum(xs > 0)),
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
        for p in sorted(d.glob("s*_L2_*_Q2048.json")):
            r = json.loads(p.read_text())
            if r.get("seed") in SEEDS and r.get("stage") == "FTREP":
                rows.append(r)
    return rows


def audit_row(r):
    problems = []
    if r["seed"] in FORBIDDEN:
        problems.append("forbidden_seed")
    if r["seed"] not in SEEDS:
        problems.append("seed")
    if r.get("N") not in NS or r.get("L") != 2 or r.get("Q") != 2048:
        problems.append("batch")
    if r.get("band") != "exclusive_unread_dev":
        problems.append("band")
    if r.get("continuation"):
        problems.append("continuation")
    if r.get("optimizer") != "FT_v4_center4":
        problems.append("optimizer")
    if r.get("status") == "ok":
        if r["queries"] != 2048 or r.get("outside_cap_queries", 0) != 0:
            problems.append("Q")
        if r.get("nstarts") != 4 or r.get("K_per_start") != 250:
            problems.append("split")
        qc = r.get("query_cat") or {}
        if qc.get("init") != 4 or qc.get("calibration") != 32:
            problems.append("cal")
        if qc.get("perturb") != 2000 or qc.get("candidate") != 12:
            problems.append("cand")
        if qc.get("monitor", 0) != 0 or qc.get("screen", 0) != 0:
            problems.append("extra")
        if r.get("reused"):
            problems.append("reused")
        want = "QAOA" if r.get("arm") == "qaoa_center4" else "Nov3"
        if r.get("circuit") != want:
            problems.append("circuit")
    return problems


def index(rows, nq, arm):
    return {r["seed"]: r for r in rows
            if r["N"] == nq and r["arm"] == arm and r.get("status") == "ok"}


def contrast(a, b, rng):
    seeds = sorted(set(a) & set(b))
    rec = {"n": len(seeds), "seeds": seeds}
    for key in ("E_err_over_N", "E", "F"):
        xs = []
        for s in seeds:
            ra, rb = a[s], b[s]
            va = ra["E_err_over_N"] if key == "E_err_over_N" else (ra["diag"]["E"] if key == "E" else ra["diag"]["F"])
            vb = rb["E_err_over_N"] if key == "E_err_over_N" else (rb["diag"]["E"] if key == "E" else rb["diag"]["F"])
            xs.append(va - vb)
        rec[key] = paired_vals(xs, rng)
    rec["n_energy_wins_a"] = rec["E"]["n_neg"]
    rec["n_F_wins_a"] = rec["F"]["n_pos"]
    rec["n_start0_nov3"] = int(sum(a[s].get("sel_start") == 0 for s in seeds)) if a else 0
    return rec


def cell_means(idx, rng):
    seeds = sorted(idx)
    e = np.array([idx[s]["E_err_over_N"] for s in seeds], float) if seeds else np.array([])
    f = np.array([idx[s]["diag"]["F"] for s in seeds], float) if seeds else np.array([])
    return {"n": len(seeds), "seeds": seeds, "E_err_over_N": paired_vals(e, rng), "F": paired_vals(f, rng),
            "n_start0": int(sum(idx[s].get("sel_start") == 0 for s in seeds))}


def fmt(ct, key):
    z = ct[key]
    return (f"{z['mean']:+.5f} med {z['median']:+.5f} "
            f"[{z['ci95_mean'][0]:+.5f},{z['ci95_mean'][1]:+.5f}] "
            f"tails [{z['min']:+.5f},{z['max']:+.5f}]")


def pass_below(ct):
    return bool(ct["n"] > 0 and ct["E"]["ci95_mean"][1] < 0)


def main():
    inst = json.loads((OUT / "INSTANCES_FTREP.json").read_text()) if (OUT / "INSTANCES_FTREP.json").exists() else {"rows": []}
    rejected = [r for r in inst.get("rows", []) if not r.get("accepted")]
    rows = load_rows()
    audits = {f"N{r['N']}_s{r['seed']}_{r['arm']}": audit_row(r) for r in rows}
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
            for arm in ARMS:
                expected.append((nq, seed, arm))
    have = {(r["N"], r["seed"], r["arm"]) for r in rows}
    for t in expected:
        if t not in have:
            missing.append(t)
    cells, contrasts, gates = {}, {}, {}
    rng_gate = np.random.default_rng(3334)
    for nq in NS:
        contrasts[f"N{nq}"] = contrast(index(ok, nq, "nov3_center4"), index(ok, nq, "qaoa_center4"), rng_gate)
        gates[nq] = pass_below(contrasts[f"N{nq}"])
    rng_cells = np.random.default_rng(3334)
    for nq in NS:
        cells[f"N{nq}_Nov3"] = cell_means(index(ok, nq, "nov3_center4"), rng_cells)
        cells[f"N{nq}_QAOA"] = cell_means(index(ok, nq, "qaoa_center4"), rng_cells)
    both = bool(gates[15] and gates[12])
    summary = {
        "experiment": "FTREP",
        "n_runs": len(rows), "n_ok": len(ok), "n_failures": len(fail),
        "failures": [{"N": r["N"], "seed": r["seed"], "arm": r["arm"], "error": r.get("error")} for r in fail],
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
        "depth_s1w_ledger_preserved": 245760,
        "this_stream": "FTREP",
        "cells": cells, "contrasts": contrasts,
        "gate": {
            "R1_N15": gates[15], "R2_N12": gates[12], "both_pass": both,
            "decision": "FT-L2 REPLICATED on LE" if both else "FAIL_not_replicated",
            "gr": "not patched",
        },
        "seeds": SEEDS, "band": "exclusive_unread_dev",
        "ft_alloc_sha256": PREREG["ft_source"]["alloc_py_sha256"],
        "ft_head_commit": PREREG["ft_source"]["head_commit"],
    }
    (OUT / "summary_FTREP.json").write_text(json.dumps(summary, indent=1))
    lines = [
        "# FTREP: LE replicate of FT PR79 v4 center-4 at L=2 Q=2048",
        "",
        f"FT alloc.py sha256 `{PREREG['ft_source']['alloc_py_sha256']}` from PR79 `{PREREG['ft_source']['head_commit'][:12]}`.",
        "Exclusive unread seeds 10152–10159. N∈{12,15} L=2 Q=2048. Four equal SPSA blocks, frozen gains, center-4 starts.",
        "Energy selects (init/last/Polyak/mid). F post hoc. GR not patched. FT holdout 10144–10151 unused.",
        "",
        f"Runs {len(rows)} ok {len(ok)} failures {len(fail)}. Charged {charged}. "
        f"Outside cap {summary['outside_cap_queries']}. Rejected B* {len(rejected)}. Missing {len(missing)}.",
        "",
        f"Gate R1 N15 mean ΔE {fmt(contrasts['N15'], 'E')} {contrasts['N15']['E']['n_neg']}/{contrasts['N15']['E']['n']}: "
        f"**{'pass' if gates[15] else 'fail'}**.",
        f"Gate R2 N12 mean ΔE {fmt(contrasts['N12'], 'E')} {contrasts['N12']['E']['n_neg']}/{contrasts['N12']['E']['n']}: "
        f"**{'pass' if gates[12] else 'fail'}**.",
        f"Decision: **{summary['gate']['decision']}**.",
        "",
        "## Nov3 − QAOA (center4)",
        "",
        "| N | mean ΔE med [CI] tails wins(Nov3) | Δ(E−E0)/N | ΔF wins | start0 Nov3 |",
        "|---|---|---|---|---|",
    ]
    for nq in NS:
        ct = contrasts[f"N{nq}"]
        n0 = cells[f"N{nq}_Nov3"]["n_start0"]
        lines.append(
            f"| {nq} | {fmt(ct, 'E')} {ct['E']['n_neg']}/{ct['E']['n']} "
            f"| {fmt(ct, 'E_err_over_N')} | {fmt(ct, 'F')} {ct['F']['n_pos']}/{ct['F']['n']} "
            f"| {n0}/{ct['n']} |"
        )
    (OUT / "RESULTS_FTREP.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"gate": summary["gate"], "charged": charged, "ok": len(ok), "fail": len(fail),
                      "rejected": rejected, "missing": missing, "audit_failures": bad}, indent=1))
    if bad:
        raise SystemExit("audit failed")
    if missing:
        raise SystemExit(f"missing {missing}")


if __name__ == "__main__":
    main()
