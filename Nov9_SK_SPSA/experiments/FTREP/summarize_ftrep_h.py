"""FTREP H Gate R: N15 holdout replay. Compare to FT bd9effe3f. No GR."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[1] / "results" / "FTREP"
PREREG = json.loads((HERE / "PREREG_FTREP_H.json").read_text())
FT = json.loads((OUT / "FT_HOLDOUT_BD9EFFE3F.json").read_text())
SEEDS = list(PREREG["seeds"]["ids"])
ARMS = ("nov3_center4", "qaoa_center4")
FORBIDDEN = set(range(10010, 10144)) | set(range(10152, 10200)) | set(range(11000, 11032)) | set(range(20000, 21000))
N_BOOT = 4000
BAND = "ft_holdout_replay"


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
    for nq in (15, 18):
        d = OUT / "runs" / f"N{nq}"
        if not d.exists():
            continue
        for p in sorted(d.glob("s*_L2_*_Q2048.json")):
            r = json.loads(p.read_text())
            if r.get("seed") in SEEDS and r.get("stage") == "FTREP_H":
                rows.append(r)
    return rows


def audit_row(r):
    problems = []
    if r["seed"] in FORBIDDEN:
        problems.append("forbidden_seed")
    if r["seed"] not in SEEDS:
        problems.append("seed")
    if r.get("N") not in (15, 18) or r.get("L") != 2 or r.get("Q") != 2048:
        problems.append("batch")
    if r.get("band") != BAND:
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
    rec["n_start0_nov3"] = int(sum(a[s].get("sel_start") == 0 for s in seeds))
    rec["n_start0_qaoa"] = int(sum(b[s].get("sel_start") == 0 for s in seeds))
    rec["per_seed"] = []
    for s in seeds:
        rec["per_seed"].append({
            "seed": s,
            "E_Nov3": a[s]["diag"]["E"], "E_QAOA": b[s]["diag"]["E"],
            "dE": a[s]["diag"]["E"] - b[s]["diag"]["E"],
            "F_Nov3": a[s]["diag"]["F"], "F_QAOA": b[s]["diag"]["F"],
            "start_Nov3": a[s].get("sel_start"), "start_QAOA": b[s].get("sel_start"),
            "Bstar": a[s].get("Bstar"),
        })
    return rec


def fmt(ct, key):
    z = ct[key]
    return (f"{z['mean']:+.5f} med {z['median']:+.5f} "
            f"[{z['ci95_mean'][0]:+.5f},{z['ci95_mean'][1]:+.5f}] "
            f"tails [{z['min']:+.5f},{z['max']:+.5f}]")


def side_by_side(nq, ct):
    ft = {r["seed"]: r for r in FT["cells"][f"N{nq}"]}
    rows = []
    for rec in ct.get("per_seed", []):
        f = ft[rec["seed"]]
        rows.append({
            "seed": rec["seed"],
            "LE_dE": rec["dE"], "FT_dE": f["dE"], "dE_LE_minus_FT": rec["dE"] - f["dE"],
            "LE_E_Nov3": rec["E_Nov3"], "FT_E_Nov3": f["E_Nov3"],
            "LE_E_QAOA": rec["E_QAOA"], "FT_E_QAOA": f["E_QAOA"],
            "LE_start_Nov3": rec["start_Nov3"], "FT_start_Nov3": f["start_Nov3"],
            "LE_Bstar": rec["Bstar"], "FT_Bstar": f["Bstar"],
        })
    return rows


def main():
    inst = json.loads((OUT / "INSTANCES_FTREP_H.json").read_text()) if (OUT / "INSTANCES_FTREP_H.json").exists() else {"rows": []}
    rejected = [r for r in inst.get("rows", []) if not r.get("accepted")]
    rows = load_rows()
    audits = {f"N{r['N']}_s{r['seed']}_{r['arm']}": audit_row(r) for r in rows}
    bad = {k: v for k, v in audits.items() if v}
    ok = [r for r in rows if r.get("status") == "ok"]
    fail = [r for r in rows if r.get("status") != "ok"]
    charged = int(sum(r.get("queries", 0) for r in rows))
    accepted = {(r["N"], r["seed"]) for r in inst.get("rows", []) if r.get("accepted")}
    ns_present = sorted({r["N"] for r in rows})
    expected, missing = [], []
    for nq in ns_present:
        for seed in SEEDS:
            if (nq, seed) not in accepted:
                continue
            for arm in ARMS:
                expected.append((nq, seed, arm))
    have = {(r["N"], r["seed"], r["arm"]) for r in rows}
    for t in expected:
        if t not in have:
            missing.append(t)
    for seed in SEEDS:
        for arm in ARMS:
            if (15, seed, arm) not in have and (15, seed) in accepted:
                missing.append((15, seed, arm))
            elif (15, seed) not in accepted:
                missing.append((15, seed, arm))
    missing = sorted(set(missing))
    rng = np.random.default_rng(3334)
    contrasts = {}
    for nq in (15, 18):
        a, b = index(ok, nq, "nov3_center4"), index(ok, nq, "qaoa_center4")
        if a and b:
            contrasts[f"N{nq}"] = contrast(a, b, rng)
    ct15 = contrasts.get("N15", {"n": 0, "E": {"mean": float("nan"), "median": float("nan"),
                                               "ci95_mean": [float("nan"), float("nan")],
                                               "n_neg": 0, "min": float("nan"), "max": float("nan")}})
    gate_pass = bool(ct15["n"] > 0 and ct15["E"]["ci95_mean"][1] < 0)
    decision = "HOLDOUT_DISCREPANCY" if gate_pass else "HOLDOUT_INSTABILITY CONFIRMED on LE"
    status = json.loads((OUT / "STATUS_FTREP_H.json").read_text()) if (OUT / "STATUS_FTREP_H.json").exists() else {}
    wall_fit = float(sum(r.get("wall_s", 0) for r in rows))
    cpu_fit = float(sum(r.get("cpu_s", 0) for r in rows if r.get("cpu_s") is not None))
    rss = [r.get("peak_rss_kib") for r in rows if r.get("peak_rss_kib") is not None]
    cmp15 = side_by_side(15, contrasts["N15"]) if "N15" in contrasts else []
    cmp18 = side_by_side(18, contrasts["N18"]) if "N18" in contrasts else []
    summary = {
        "experiment": "FTREP_H",
        "n_runs": len(rows), "n_ok": len(ok), "n_failures": len(fail),
        "failures": [{"N": r["N"], "seed": r["seed"], "arm": r["arm"], "error": r.get("error")} for r in fail],
        "rejected_instances": rejected, "missing": missing,
        "queries_charged": charged,
        "outside_cap_queries": int(sum(r.get("outside_cap_queries", 0) for r in rows)),
        "audit_failures": bad,
        "n15_bstar_wall_s": status.get("n15_bstar_wall_s"),
        "n15_fit_wall_s": status.get("n15_fit_wall_s"),
        "n18_bstar_wall_s": status.get("n18_bstar_wall_s"),
        "n18_fit_wall_s": status.get("n18_fit_wall_s"),
        "sum_cell_wall_s": wall_fit,
        "sum_cell_cpu_s": cpu_fit,
        "peak_rss_kib_max": int(max(rss)) if rss else None,
        "this_stream": "FTREP_H",
        "contrasts": contrasts,
        "vs_ft_N15": cmp15,
        "vs_ft_N18": cmp18,
        "gate": {
            "R": "N15 L2 Q=2048",
            "n": ct15["n"], "mean": ct15["E"]["mean"], "median": ct15["E"]["median"],
            "ci95": ct15["E"]["ci95_mean"], "n_Nov3_lower": ct15["E"]["n_neg"],
            "pass": gate_pass, "decision": decision, "gr": "not patched",
            "ft_H1_mean": PREREG["ft_published_h1"]["mean_dE"],
            "ft_H1_ci": PREREG["ft_published_h1"]["ci"],
            "ft_H1_pass": False,
        },
        "seeds": SEEDS, "band": BAND,
        "ft_alloc_sha256": PREREG["ft_source"]["alloc_py_sha256"],
        "ft_holdout_commit": PREREG["ft_source"]["holdout_commit"],
        "n18_ran": "N18" in contrasts,
    }
    (OUT / "summary_FTREP_H.json").write_text(json.dumps(summary, indent=1))
    g = summary["gate"]
    lines = [
        "# FTREP H: LE replay of FT L=2 holdout 10144–10151",
        "",
        f"Same alloc.py `{PREREG['ft_source']['alloc_py_sha256']}` / recipe `{PREREG['ft_source']['recipe_head_commit'][:12]}`.",
        f"FT holdout table from `{PREREG['ft_source']['holdout_commit'][:12]}`. Seeds 10144–10151 intentional replay.",
        "N=15 L=2 Q=2048 Gate R. N=18 optional report. Energy selects. F post hoc. GR not patched. No candidacy revival.",
        "",
        f"Runs {len(rows)} ok {len(ok)} failures {len(fail)}. Charged {charged}. "
        f"Outside cap {summary['outside_cap_queries']}. Rejected B* {len(rejected)}. Missing {len(missing)}.",
        "",
        f"Gate R N15 mean ΔE {fmt(ct15, 'E')} {ct15['E']['n_neg']}/{ct15['n']}: **{'pass' if gate_pass else 'fail'}**.",
        f"FT H1 (bd9effe3f) mean ΔE {PREREG['ft_published_h1']['mean_dE']:+.4f} CI {PREREG['ft_published_h1']['ci']} 4/8: fail.",
        f"Decision: **{decision}**.",
        "",
    ]
    if "N18" in contrasts:
        lines.append(f"N18 report mean ΔE {fmt(contrasts['N18'], 'E')} {contrasts['N18']['E']['n_neg']}/{contrasts['N18']['n']}.")
        lines.append(f"FT H2 mean ΔE {PREREG['ft_published_h2']['mean_dE']:+.4f} CI {PREREG['ft_published_h2']['ci']} 5/8: fail.")
        lines.append("")
    lines.extend([
        "## Side-by-side vs FT holdout N15 Q=2048 (`bd9effe3f`)",
        "",
        "| seed | LE ΔE | FT ΔE | LE−FT | LE start0 | FT start0 |",
        "|---|---|---|---|---|---|",
    ])
    for r in cmp15:
        lines.append(
            f"| {r['seed']} | {r['LE_dE']:+.5f} | {r['FT_dE']:+.5f} | {r['dE_LE_minus_FT']:+.5f} "
            f"| {r['LE_start_Nov3']} | {r['FT_start_Nov3']} |"
        )
    if cmp18:
        lines.extend([
            "",
            "## Side-by-side vs FT holdout N18 Q=2048 (`bd9effe3f`)",
            "",
            "| seed | LE ΔE | FT ΔE | LE−FT | LE start0 | FT start0 |",
            "|---|---|---|---|---|---|",
        ])
        for r in cmp18:
            lines.append(
                f"| {r['seed']} | {r['LE_dE']:+.5f} | {r['FT_dE']:+.5f} | {r['dE_LE_minus_FT']:+.5f} "
                f"| {r['LE_start_Nov3']} | {r['FT_start_Nov3']} |"
            )
    lines.extend([
        "",
        f"N15 B* wall {status.get('n15_bstar_wall_s')}. N15 fit wall {status.get('n15_fit_wall_s')}. "
        f"N18 B* wall {status.get('n18_bstar_wall_s')}. N18 fit wall {status.get('n18_fit_wall_s')}. "
        f"Sum cell wall {wall_fit:.1f}s. Sum cell CPU {cpu_fit:.1f}s. Peak RSS KiB {summary['peak_rss_kib_max']}.",
        "",
    ])
    (OUT / "RESULTS_FTREP_H.md").write_text("\n".join(lines))
    print(json.dumps({"gate": summary["gate"], "charged": charged, "ok": len(ok), "fail": len(fail),
                      "rejected": rejected, "missing": missing, "audit_failures": bad,
                      "n18_ran": summary["n18_ran"]}, indent=1))
    if bad:
        raise SystemExit("audit failed")
    if missing:
        raise SystemExit(f"missing {missing}")


if __name__ == "__main__":
    main()
