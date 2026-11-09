"""FTREP D: per-L present/absent labels. Characterization, not adoption. No GR."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[1] / "results" / "FTREP"
PREREG = json.loads((HERE / "PREREG_FTREP_D.json").read_text())
SEEDS = list(PREREG["seeds"]["ids"])
LS = [2, 3, 4]
Q_OF = {2: 2048, 3: 3072, 4: 4096}
T_OF = {2: 250, 3: 378, 4: 506}
PERTURB = {2: 2000, 3: 3024, 4: 4048}
ARMS = ("nov3_center4", "qaoa_center4")
FORBIDDEN = set(range(10010, 10264)) | set(range(11000, 11032)) | set(range(20000, 21000))
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
    d = OUT / "runs" / "N15"
    if not d.exists():
        return rows
    for ell in LS:
        Q = Q_OF[ell]
        for p in sorted(d.glob(f"s*_L{ell}_*_Q{Q}.json")):
            r = json.loads(p.read_text())
            if r.get("seed") in SEEDS and r.get("stage") == "FTREP_D":
                rows.append(r)
    return rows


def audit_row(r):
    problems = []
    if r["seed"] in FORBIDDEN:
        problems.append("forbidden_seed")
    if r["seed"] not in SEEDS:
        problems.append("seed")
    ell = r.get("L")
    if r.get("N") != 15 or ell not in LS or r.get("Q") != Q_OF.get(ell):
        problems.append("batch")
    if r.get("band") != "exclusive_unread_dev":
        problems.append("band")
    if r.get("continuation"):
        problems.append("continuation")
    if r.get("optimizer") != "FT_v4_center4":
        problems.append("optimizer")
    if r.get("status") == "ok":
        if r["queries"] != Q_OF[ell] or r.get("outside_cap_queries", 0) != 0:
            problems.append("Q")
        if r.get("nstarts") != 4 or r.get("K_per_start") != T_OF[ell]:
            problems.append("split")
        qc = r.get("query_cat") or {}
        if qc.get("init") != 4 or qc.get("calibration") != 32:
            problems.append("cal")
        if qc.get("perturb") != PERTURB[ell] or qc.get("candidate") != 12:
            problems.append("cand")
        if qc.get("monitor", 0) != 0 or qc.get("screen", 0) != 0:
            problems.append("extra")
        want = "QAOA" if r.get("arm") == "qaoa_center4" else "Nov3"
        if r.get("circuit") != want:
            problems.append("circuit")
    return problems


def index(rows, ell, arm):
    return {r["seed"]: r for r in rows
            if r["L"] == ell and r["arm"] == arm and r.get("status") == "ok"}


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
    rec["mean_F_nov3"] = float(np.mean([a[s]["diag"]["F"] for s in seeds])) if seeds else float("nan")
    rec["mean_F_qaoa"] = float(np.mean([b[s]["diag"]["F"] for s in seeds])) if seeds else float("nan")
    rec["label"] = ("present" if rec["n"] > 0 and rec["E"]["ci95_mean"][1] < 0
                    else "absent/inconclusive")
    return rec


def fmt(ct, key):
    z = ct[key]
    return (f"{z['mean']:+.5f} med {z['median']:+.5f} "
            f"[{z['ci95_mean'][0]:+.5f},{z['ci95_mean'][1]:+.5f}] "
            f"tails [{z['min']:+.5f},{z['max']:+.5f}]")


def main():
    inst = json.loads((OUT / "INSTANCES_FTREP_D.json").read_text()) if (OUT / "INSTANCES_FTREP_D.json").exists() else {"rows": []}
    rejected = [r for r in inst.get("rows", []) if not r.get("accepted")]
    rows = load_rows()
    audits = {f"s{r['seed']}_L{r['L']}_{r['arm']}": audit_row(r) for r in rows}
    bad = {k: v for k, v in audits.items() if v}
    ok = [r for r in rows if r.get("status") == "ok"]
    fail = [r for r in rows if r.get("status") != "ok"]
    charged = int(sum(r.get("queries", 0) for r in rows))
    accepted = {r["seed"] for r in inst.get("rows", []) if r.get("accepted")}
    expected, missing = [], []
    for seed in SEEDS:
        if seed not in accepted:
            continue
        for ell in LS:
            for arm in ARMS:
                expected.append((seed, ell, arm))
    have = {(r["seed"], r["L"], r["arm"]) for r in rows}
    for t in expected:
        if t not in have:
            missing.append(t)
    rng = np.random.default_rng(3334)
    contrasts, labels = {}, {}
    for ell in LS:
        contrasts[f"L{ell}"] = contrast(index(ok, ell, "nov3_center4"),
                                        index(ok, ell, "qaoa_center4"), rng)
        labels[ell] = contrasts[f"L{ell}"]["label"]
    status = json.loads((OUT / "STATUS_FTREP_D.json").read_text()) if (OUT / "STATUS_FTREP_D.json").exists() else {}
    wall_fit = float(sum(r.get("wall_s", 0) for r in rows))
    cpu_fit = float(sum(r.get("cpu_s", 0) for r in rows if r.get("cpu_s") is not None))
    rss = [r.get("peak_rss_kib") for r in rows if r.get("peak_rss_kib") is not None]
    per_L = {}
    for ell in LS:
        ct = contrasts[f"L{ell}"]
        per_L[ell] = {
            "Q": Q_OF[ell], "n": ct["n"], "mean_dE": ct["E"]["mean"], "median_dE": ct["E"]["median"],
            "ci95": ct["E"]["ci95_mean"], "n_Nov3_lower": ct["E"]["n_neg"],
            "mean_F_nov3": ct["mean_F_nov3"], "mean_F_qaoa": ct["mean_F_qaoa"],
            "n_start0_nov3": ct["n_start0_nov3"], "label": ct["label"],
        }
    summary = {
        "experiment": "FTREP_D",
        "n_runs": len(rows), "n_ok": len(ok), "n_failures": len(fail),
        "failures": [{"seed": r["seed"], "L": r["L"], "arm": r["arm"], "error": r.get("error")} for r in fail],
        "rejected_instances": rejected, "missing": missing,
        "queries_charged": charged,
        "outside_cap_queries": int(sum(r.get("outside_cap_queries", 0) for r in rows)),
        "audit_failures": bad,
        "bstar_wall_s": status.get("bstar_wall_s"),
        "fit_wall_s": status.get("fit_wall_s"),
        "sum_cell_wall_s": wall_fit,
        "sum_cell_cpu_s": cpu_fit,
        "peak_rss_kib_max": int(max(rss)) if rss else None,
        "this_stream": "FTREP_D",
        "contrasts": contrasts,
        "per_L": per_L,
        "labels": {str(k): v for k, v in labels.items()},
        "adoption": "none — characterization only",
        "gr": "not patched",
        "seeds": SEEDS, "band": "exclusive_unread_dev",
        "ft_alloc_sha256": PREREG["ft_source"]["alloc_py_sha256"],
        "ft_head_commit": PREREG["ft_source"]["head_commit"],
    }
    (OUT / "summary_FTREP_D.json").write_text(json.dumps(summary, indent=1))
    lines = [
        "# FTREP D: N=15 center-4 depth curve L∈{2,3,4}",
        "",
        f"Same alloc.py `{PREREG['ft_source']['alloc_py_sha256']}` / PR79 `{PREREG['ft_source']['head_commit'][:12]}`.",
        "Seeds 10264–10271. N=15 only. Q=1024·L. Four equal blocks, frozen gains, center-4. Energy selects. F post hoc.",
        "Characterization, not adoption. GR not patched regardless of L=2.",
        "",
        f"Runs {len(rows)} ok {len(ok)} failures {len(fail)}. Charged {charged}. "
        f"Outside cap {summary['outside_cap_queries']}. Rejected B* {len(rejected)}. Missing {len(missing)}.",
        "",
        "| L | Q | mean ΔE med [CI] tails wins | mean F Nov3 | mean F QAOA | label |",
        "|---|---|---|---|---|---|",
    ]
    for ell in LS:
        ct = contrasts[f"L{ell}"]
        lines.append(
            f"| {ell} | {Q_OF[ell]} | {fmt(ct, 'E')} {ct['E']['n_neg']}/{ct['n']} "
            f"| {ct['mean_F_nov3']:.4f} | {ct['mean_F_qaoa']:.4f} | **{ct['label']}** |"
        )
    lines.extend([
        "",
        f"B* wall {status.get('bstar_wall_s')}. Fit wall {status.get('fit_wall_s')}. "
        f"Sum cell wall {wall_fit:.1f}s. Sum cell CPU {cpu_fit:.1f}s. Peak RSS KiB {summary['peak_rss_kib_max']}.",
        "",
    ])
    (OUT / "RESULTS_FTREP_D.md").write_text("\n".join(lines))
    print(json.dumps({"labels": summary["labels"], "per_L": per_L, "charged": charged,
                      "ok": len(ok), "fail": len(fail), "rejected": rejected, "missing": missing,
                      "audit_failures": bad}, indent=1))
    if bad:
        raise SystemExit("audit failed")
    if missing:
        raise SystemExit(f"missing {missing}")


if __name__ == "__main__":
    main()
