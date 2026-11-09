"""FTREP N18 gate: mean Nov3−QAOA ΔE CI entirely <0. FT bootstrap. Wall/CPU."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[1] / "results" / "FTREP"
RUNS = OUT / "runs" / "N18"
PREREG = json.loads((HERE / "PREREG_FTREP_N18.json").read_text())
SEEDS = list(PREREG["seeds"]["ids"])
ARMS = ("nov3_center4", "qaoa_center4")
FORBIDDEN = set(range(10010, 10168)) | set(range(11000, 11032)) | set(range(20000, 21000))
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
    if not RUNS.exists():
        return rows
    for p in sorted(RUNS.glob("s*_L2_*_Q2048.json")):
        r = json.loads(p.read_text())
        if r.get("seed") in SEEDS and r.get("stage") == "FTREP_N18":
            rows.append(r)
    return rows


def audit_row(r):
    problems = []
    if r["seed"] in FORBIDDEN:
        problems.append("forbidden_seed")
    if r["seed"] not in SEEDS:
        problems.append("seed")
    if r.get("N") != 18 or r.get("L") != 2 or r.get("Q") != 2048:
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
            va = ra["E_err_over_N"] if key == "E_err_over_N" else (ra["diag"]["E"] if key == "E" else ra["diag"]["F"])
            vb = rb["E_err_over_N"] if key == "E_err_over_N" else (rb["diag"]["E"] if key == "E" else rb["diag"]["F"])
            xs.append(va - vb)
        rec[key] = paired_vals(xs, rng)
    rec["n_energy_wins_a"] = rec["E"]["n_neg"]
    rec["n_F_wins_a"] = rec["F"]["n_pos"]
    rec["n_start0_nov3"] = int(sum(a[s].get("sel_start") == 0 for s in seeds))
    rec["n_start0_qaoa"] = int(sum(b[s].get("sel_start") == 0 for s in seeds))
    return rec


def fmt(ct, key):
    z = ct[key]
    return (f"{z['mean']:+.5f} med {z['median']:+.5f} "
            f"[{z['ci95_mean'][0]:+.5f},{z['ci95_mean'][1]:+.5f}] "
            f"tails [{z['min']:+.5f},{z['max']:+.5f}]")


def main():
    inst = json.loads((OUT / "INSTANCES_FTREP_N18.json").read_text()) if (OUT / "INSTANCES_FTREP_N18.json").exists() else {"rows": []}
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
    adopt = bool(ct["n"] > 0 and ct["E"]["ci95_mean"][1] < 0)
    status = json.loads((OUT / "STATUS_FTREP_N18.json").read_text()) if (OUT / "STATUS_FTREP_N18.json").exists() else {}
    wall_fit = float(sum(r.get("wall_s", 0) for r in rows))
    cpu_fit = float(sum(r.get("cpu_s", 0) for r in rows))
    rss = [r.get("peak_rss_kib") for r in rows if r.get("peak_rss_kib") is not None]
    summary = {
        "experiment": "FTREP_N18",
        "n_runs": len(rows), "n_ok": len(ok), "n_failures": len(fail),
        "failures": [{"seed": r["seed"], "arm": r["arm"], "error": r.get("error")} for r in fail],
        "rejected_instances": rejected, "missing": missing,
        "queries_charged": charged,
        "outside_cap_queries": int(sum(r.get("outside_cap_queries", 0) for r in rows)),
        "audit_failures": bad,
        "bstar_wall_s": status.get("bstar_wall_s"),
        "fit_wall_s": status.get("fit_wall_s"),
        "sum_cell_wall_s": wall_fit,
        "sum_cell_cpu_s": cpu_fit,
        "peak_rss_kib_max": int(max(rss)) if rss else None,
        "prior_n3cond_ledger_preserved": 772224,
        "ftrep_n12n15_ledger_preserved": 65536,
        "this_stream": "FTREP_N18",
        "contrast": ct,
        "gate": {
            "cell": "N18 L2 Q=2048",
            "n": ct["n"], "mean": ct["E"]["mean"], "median": ct["E"]["median"],
            "ci95": ct["E"]["ci95_mean"], "n_Nov3_lower": ct["E"]["n_neg"],
            "pass": adopt,
            "decision": "N18-L2 SCALED" if adopt else "FAIL_n18_did_not_scale",
            "gr": "not patched",
        },
        "seeds": SEEDS, "band": "exclusive_unread_dev",
        "ft_alloc_sha256": PREREG["ft_source"]["alloc_py_sha256"],
        "ft_head_commit": PREREG["ft_source"]["head_commit"],
    }
    (OUT / "summary_FTREP_N18.json").write_text(json.dumps(summary, indent=1))
    g = summary["gate"]
    lo, hi = g["ci95"]
    lines = [
        "# FTREP N18: FT v4 center-4 at N=18 L=2 Q=2048",
        "",
        f"Same alloc.py `{PREREG['ft_source']['alloc_py_sha256']}` / PR79 `{PREREG['ft_source']['head_commit'][:12]}`.",
        "Seeds 10168–10175. Four equal blocks, T=250, frozen gains, center-4. Energy selects. F post hoc. No L3. GR not patched.",
        "",
        f"Runs {len(rows)} ok {len(ok)} failures {len(fail)}. Charged {charged}. "
        f"Outside cap {summary['outside_cap_queries']}. Rejected B* {len(rejected)}. Missing {len(missing)}.",
        "",
        f"Gate N18 mean ΔE (Nov3−QAOA) = {g['mean']:+.6f} med {g['median']:+.6f} "
        f"CI [{lo:+.6f}, {hi:+.6f}] {g['n_Nov3_lower']}/{g['n']}: **{g['decision']}**.",
        "",
        f"Δ(E−E0)/N {fmt(ct, 'E_err_over_N')}. ΔF {fmt(ct, 'F')} {ct['F']['n_pos']}/{ct['n']}. "
        f"Nov3 start0 {ct['n_start0_nov3']}/{ct['n']}. QAOA start0 {ct['n_start0_qaoa']}/{ct['n']}.",
        "",
        f"B* wall {status.get('bstar_wall_s')}. Fit wall {status.get('fit_wall_s')}. "
        f"Sum cell wall {wall_fit:.1f}s. Sum cell CPU {cpu_fit:.1f}s. Peak RSS KiB {summary['peak_rss_kib_max']}.",
        "",
    ]
    (OUT / "RESULTS_FTREP_N18.md").write_text("\n".join(lines))
    print(json.dumps({"gate": summary["gate"], "charged": charged, "ok": len(ok), "fail": len(fail),
                      "rejected": rejected, "missing": missing, "audit_failures": bad,
                      "bstar_wall_s": status.get("bstar_wall_s"), "fit_wall_s": status.get("fit_wall_s"),
                      "sum_cell_cpu_s": cpu_fit, "peak_rss_kib_max": summary["peak_rss_kib_max"]}, indent=1))
    if bad:
        raise SystemExit("audit failed")
    if missing:
        raise SystemExit(f"missing {missing}")


if __name__ == "__main__":
    main()
