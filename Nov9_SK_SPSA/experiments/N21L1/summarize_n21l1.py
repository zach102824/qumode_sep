"""N21L1: ESTIMATE_NEGATIVE_CI iff mean ΔE CI entirely <0; else UNSTABLE. No GR."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[1] / "results" / "N21L1"
PREREG = json.loads((HERE / "PREREG_N21L1.json").read_text())
SEEDS = list(PREREG["seeds"]["ids"])
ARMS = ("nov3_center4", "qaoa_center4")
FORBIDDEN = (
    set(range(10010, 10416)) | set(range(10416, 10432))
    | set(range(11000, 11032)) | set(range(20000, 21000))
)
N_BOOT = 4000
QFIX = 1024
TFIX = 122
R25 = -0.0229
R16 = -0.0184


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
    d = OUT / "runs" / "N21"
    if not d.exists():
        return rows
    for p in sorted(d.glob(f"s*_L1_*_Q{QFIX}.json")):
        r = json.loads(p.read_text())
        if r.get("seed") in SEEDS and r.get("stage") == "N21L1":
            rows.append(r)
    return rows


def audit_row(r):
    problems = []
    if r["seed"] in FORBIDDEN or r["seed"] <= 10415:
        problems.append("forbidden_seed")
    if r["seed"] not in SEEDS:
        problems.append("seed")
    if r.get("N") != 21 or r.get("L") != 1 or r.get("Q") != QFIX:
        problems.append("batch")
    if r.get("band") != "exclusive_unread_dev":
        problems.append("band")
    if r.get("continuation"):
        problems.append("continuation")
    if r.get("optimizer") != "FT_v4_center4":
        problems.append("optimizer")
    if r.get("status") == "ok":
        if r["queries"] != QFIX or r.get("outside_cap_queries", 0) != 0:
            problems.append("Q")
        if r.get("nstarts") != 4 or r.get("K_per_start") != TFIX:
            problems.append("split")
        qc = r.get("query_cat") or {}
        if qc.get("init") != 4 or qc.get("calibration") != 32:
            problems.append("cal")
        if qc.get("perturb") != 976 or qc.get("candidate") != 12:
            problems.append("cand")
        if qc.get("monitor", 0) != 0 or qc.get("screen", 0) != 0:
            problems.append("extra")
        want = "QAOA" if r.get("arm") == "qaoa_center4" else "Nov3"
        if r.get("circuit") != want:
            problems.append("circuit")
    return problems


def index(rows, arm):
    return {r["seed"]: r for r in rows if r["arm"] == arm and r.get("status") == "ok"}


def contrast(a, b, rng):
    seeds = sorted(set(a) & set(b))
    rec = {"n": len(seeds), "seeds": seeds}
    for key in ("E_err_over_N", "E", "F"):
        xs = []
        for s in seeds:
            ra, rb = a[s], b[s]
            va = ra["E_err_over_N"] if key == "E_err_over_N" else (
                ra["diag"]["E"] if key == "E" else ra["diag"]["F"]
            )
            vb = rb["E_err_over_N"] if key == "E_err_over_N" else (
                rb["diag"]["E"] if key == "E" else rb["diag"]["F"]
            )
            xs.append(va - vb)
        rec[key] = paired_vals(xs, rng)
    rec["n_energy_wins_a"] = rec["E"]["n_neg"]
    rec["mean_F_nov3"] = float(np.mean([a[s]["diag"]["F"] for s in seeds])) if seeds else float("nan")
    rec["mean_F_qaoa"] = float(np.mean([b[s]["diag"]["F"] for s in seeds])) if seeds else float("nan")
    rec["mean_dE_over_N"] = rec["E"]["mean"] / 21.0 if seeds else float("nan")
    return rec


def fmt(ct, key):
    z = ct[key]
    return (f"{z['mean']:+.5f} med {z['median']:+.5f} "
            f"[{z['ci95_mean'][0]:+.5f},{z['ci95_mean'][1]:+.5f}] "
            f"tails [{z['min']:+.5f},{z['max']:+.5f}]")


def main():
    inst = json.loads((OUT / "INSTANCES_N21L1.json").read_text()) if (OUT / "INSTANCES_N21L1.json").exists() else {"rows": []}
    rejected = [r for r in inst.get("rows", []) if not r.get("accepted")]
    rows = load_rows()
    audits = {f"s{r['seed']}_{r['arm']}": audit_row(r) for r in rows}
    bad = {k: v for k, v in audits.items() if v}
    ok = [r for r in rows if r.get("status") == "ok"]
    fail = [r for r in rows if r.get("status") != "ok"]
    charged = int(sum(r.get("queries", 0) for r in rows))
    accepted = {r["seed"] for r in inst.get("rows", []) if r.get("accepted")}
    expected, missing = [], []
    for seed in SEEDS:
        if seed not in accepted:
            continue
        for arm in ARMS:
            expected.append((seed, arm))
    have = {(r["seed"], r["arm"]) for r in rows}
    for t in expected:
        if t not in have:
            missing.append(t)
    rng = np.random.default_rng(3334)
    ct = contrast(index(ok, "nov3_center4"), index(ok, "qaoa_center4"), rng)
    label = ("N21-L1 ESTIMATE_NEGATIVE_CI"
              if ct["n"] == 8 and ct["E"]["ci95_mean"][1] < 0
              else "UNSTABLE")
    status = json.loads((OUT / "STATUS_N21L1.json").read_text()) if (OUT / "STATUS_N21L1.json").exists() else {}
    wall_fit = float(sum(r.get("wall_s", 0) for r in rows))
    cpu_fit = float(sum(r.get("cpu_s", 0) for r in rows if r.get("cpu_s") is not None))
    rss = [r.get("peak_rss_kib") for r in rows if r.get("peak_rss_kib") is not None]
    rss += [r.get("peak_rss_kib") for r in inst.get("rows", []) if r.get("peak_rss_kib") is not None]
    summary = {
        "experiment": "N21L1",
        "n_runs": len(rows), "n_ok": len(ok), "n_failures": len(fail),
        "failures": [{"seed": r["seed"], "arm": r["arm"], "error": r.get("error")} for r in fail],
        "rejected_instances": rejected, "missing": missing,
        "queries_charged": charged,
        "outside_cap_queries": int(sum(r.get("outside_cap_queries", 0) for r in rows)),
        "audit_failures": bad,
        "bstar_wall_s": status.get("bstar_wall_s"),
        "fit_wall_s": status.get("fit_wall_s"),
        "nproc_bstar": status.get("nproc_bstar"),
        "nproc_fit": status.get("nproc_fit"),
        "bstar_fallback": status.get("bstar_fallback"),
        "fit_fallback": status.get("fit_fallback"),
        "sum_cell_wall_s": wall_fit,
        "sum_cell_cpu_s": cpu_fit,
        "peak_rss_kib_max": int(max(rss)) if rss else None,
        "this_stream": "N21L1",
        "contrast": ct,
        "label": label,
        "mean_dE": ct["E"]["mean"],
        "ci95": ct["E"]["ci95_mean"],
        "wins": ct["E"]["n_neg"],
        "mean_dE_over_N": ct["mean_dE_over_N"],
        "vs_local": {
            "R25_N18_dE_over_N": R25,
            "R16_dE_over_N": R16,
            "this_mean_dE_over_N": ct["mean_dE_over_N"],
        },
        "mean_F_nov3": ct["mean_F_nov3"],
        "mean_F_qaoa": ct["mean_F_qaoa"],
        "adoption": "none",
        "gr": "not patched",
        "seeds": SEEDS, "band": "exclusive_unread_dev",
        "ft_alloc_sha256": PREREG["ft_source"]["alloc_py_sha256"],
        "ft_head_commit": PREREG["ft_source"]["head_commit"],
    }
    (OUT / "summary_N21L1.json").write_text(json.dumps(summary, indent=1))
    lines = [
        "# N21L1: N=21 L=1 Q=1024 frozen v4 center-4",
        "",
        f"Unmodified v4 `{PREREG['ft_source']['alloc_py_sha256']}` / PR79 `{PREREG['ft_source']['head_commit'][:12]}`.",
        "Seeds 10432–10439. N=21 L=1 Q=1024 T=122. Four equal blocks, center-4. Energy selects. F post hoc.",
        "≤2 workers, BLAS1. No L2+. No GR.",
        "",
        f"Runs {len(rows)} ok {len(ok)} failures {len(fail)}. Charged {charged}. "
        f"Outside cap {summary['outside_cap_queries']}. Rejected B* {len(rejected)}. Missing {len(missing)}.",
        f"nproc B* {status.get('nproc_bstar')} fit {status.get('nproc_fit')}. "
        f"fallback B* {status.get('bstar_fallback')} fit {status.get('fit_fallback')}.",
        "",
        f"**Label {label}.** Nov3−QAOA: {fmt(ct, 'E')} {ct['E']['n_neg']}/{ct['n']}",
        f"mean ΔE/N {ct['mean_dE_over_N']:+.5f} vs local R25 N18 {R25:+.4f} and R16 {R16:+.4f}",
        f"mean F Nov3 {ct['mean_F_nov3']:.4f} QAOA {ct['mean_F_qaoa']:.4f}",
        "",
        f"B* wall {status.get('bstar_wall_s')}. Fit wall {status.get('fit_wall_s')}. "
        f"Sum cell wall {wall_fit:.1f}s. Sum cell CPU {cpu_fit:.1f}s. Peak RSS KiB {summary['peak_rss_kib_max']}.",
        "",
    ]
    (OUT / "RESULTS_N21L1.md").write_text("\n".join(lines))
    print(json.dumps({
        "label": label, "mean_dE": ct["E"]["mean"], "ci95": ct["E"]["ci95_mean"],
        "wins": f"{ct['E']['n_neg']}/{ct['n']}",
        "mean_dE_over_N": ct["mean_dE_over_N"],
        "vs_local": summary["vs_local"],
        "mean_F_nov3": ct["mean_F_nov3"], "mean_F_qaoa": ct["mean_F_qaoa"],
        "charged": charged, "ok": len(ok), "fail": len(fail),
        "nproc_bstar": status.get("nproc_bstar"), "nproc_fit": status.get("nproc_fit"),
        "rejected": rejected, "missing": missing, "audit_failures": bad,
    }, indent=1))
    if bad:
        raise SystemExit("audit failed")
    if missing:
        raise SystemExit(f"missing {missing}")


if __name__ == "__main__":
    main()
