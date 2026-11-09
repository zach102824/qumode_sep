"""L3TIE gates: TIE vs QAOA CI entirely <0, and TIE mean E ≤ JOINT mean E."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[1] / "results" / "L3TIE"
PREREG = json.loads((HERE / "PREREG_L3TIE.json").read_text())
SEEDS = list(PREREG["seeds"]["ids"])
ARMS = ("JOINT", "TIE", "QAOA")
FORBIDDEN = (
    set(range(10504, 10552)) | set(range(10568, 10600))
    | set(range(10400, 10440)) | set(range(10144, 10152))
    | set(range(11000, 11032)) | set(range(20000, 21000))
)
N_BOOT = 4000
QFIX = 3072


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
    for p in sorted(d.glob(f"s*_L3_*_Q{QFIX}.json")):
        r = json.loads(p.read_text())
        if r.get("seed") in SEEDS and r.get("stage") == "L3TIE":
            rows.append(r)
    return rows


def audit_row(r):
    problems = []
    if r["seed"] in FORBIDDEN or r["seed"] in range(10560, 10568):
        problems.append("forbidden_seed")
    if r["seed"] not in SEEDS:
        problems.append("seed")
    if r.get("N") != 15 or r.get("L") != 3 or r.get("Q") != QFIX:
        problems.append("batch")
    if r.get("band") != "exclusive_unread_dev":
        problems.append("band")
    if r.get("continuation"):
        problems.append("continuation")
    if r.get("pair_phase") or not r.get("no_pair_phase", True):
        problems.append("pair_phase")
    if not r.get("not_a_start_count_variant", True):
        problems.append("start_count")
    if not r.get("not_a_two_stage_freeze_order", True):
        problems.append("two_stage")
    if not r.get("not_a_shell_unshare", True):
        problems.append("shell_unshare")
    arm = r.get("arm")
    if r.get("status") == "ok":
        if r["queries"] != QFIX or r.get("outside_cap_queries", 0) != 0:
            problems.append("Q")
        if r.get("nstarts") != 4 or r.get("K_per_start") != 378:
            problems.append("split")
        qc = r.get("query_cat") or {}
        if qc.get("init") != 4 or qc.get("calibration") != 32:
            problems.append("cal")
        if qc.get("perturb") != 3024 or qc.get("candidate") != 12:
            problems.append("cand")
        if qc.get("monitor", 0) != 0 or qc.get("screen", 0) != 0:
            problems.append("extra")
        if arm == "TIE":
            if r.get("optimizer") != "FT_v4_tie_snap" or r.get("circuit") != "Nov3":
                problems.append("optimizer")
            if r.get("dim_spsa") != 6 or r.get("dim_circuit") != 12:
                problems.append("dim")
        elif arm == "JOINT":
            if r.get("optimizer") != "FT_v4_center4" or r.get("circuit") != "Nov3":
                problems.append("optimizer")
        elif arm == "QAOA":
            if r.get("optimizer") != "FT_v4_center4" or r.get("circuit") != "QAOA":
                problems.append("optimizer")
        else:
            problems.append("arm")
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
    rec["n_F_wins_a"] = rec["F"]["n_pos"]
    rec["mean_E_a"] = float(np.mean([a[s]["diag"]["E"] for s in seeds])) if seeds else float("nan")
    rec["mean_E_b"] = float(np.mean([b[s]["diag"]["E"] for s in seeds])) if seeds else float("nan")
    rec["mean_F_a"] = float(np.mean([a[s]["diag"]["F"] for s in seeds])) if seeds else float("nan")
    rec["mean_F_b"] = float(np.mean([b[s]["diag"]["F"] for s in seeds])) if seeds else float("nan")
    rec["n_start0_a"] = int(sum(a[s].get("sel_start") == 0 for s in seeds))
    rec["n_start0_b"] = int(sum(b[s].get("sel_start") == 0 for s in seeds))
    return rec


def fmt(ct, key):
    z = ct[key]
    return (f"{z['mean']:+.5f} med {z['median']:+.5f} "
            f"[{z['ci95_mean'][0]:+.5f},{z['ci95_mean'][1]:+.5f}] "
            f"tails [{z['min']:+.5f},{z['max']:+.5f}]")


def main():
    inst = json.loads((OUT / "INSTANCES_L3TIE.json").read_text()) if (OUT / "INSTANCES_L3TIE.json").exists() else {"rows": []}
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
    tie = index(ok, "TIE")
    joint = index(ok, "JOINT")
    qaoa = index(ok, "QAOA")
    ct_tq = contrast(tie, qaoa, rng)
    ct_tj = contrast(tie, joint, rng)
    ct_jq = contrast(joint, qaoa, rng)
    primary_pass = bool(ct_tq["n"] == 8 and ct_tq["E"]["ci95_mean"][1] < 0)
    secondary_pass = bool(ct_tj["n"] == 8 and ct_tj["mean_E_a"] <= ct_tj["mean_E_b"])
    gate = "PASS" if primary_pass and secondary_pass else "FAIL"
    status = json.loads((OUT / "STATUS_L3TIE.json").read_text()) if (OUT / "STATUS_L3TIE.json").exists() else {}
    wall_fit = float(sum(r.get("wall_s", 0) for r in rows))
    cpu_fit = float(sum(r.get("cpu_s", 0) for r in rows if r.get("cpu_s") is not None))
    rss = [r.get("peak_rss_kib") for r in rows if r.get("peak_rss_kib") is not None]
    summary = {
        "experiment": "L3TIE",
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
        "this_stream": "L3TIE",
        "contrast_TIE_QAOA": ct_tq,
        "contrast_TIE_JOINT": ct_tj,
        "contrast_JOINT_QAOA": ct_jq,
        "primary_pass": primary_pass,
        "secondary_pass": secondary_pass,
        "gate": gate,
        "adoption": "none",
        "gr": "not patched",
        "seeds": SEEDS, "band": "exclusive_unread_dev",
        "ft_alloc_sha256": PREREG["ft_source"]["alloc_py_sha256"],
        "ft_head_commit": PREREG["ft_source"]["head_commit"],
    }
    (OUT / "summary_L3TIE.json").write_text(json.dumps(summary, indent=1))
    lines = [
        "# L3TIE: N=15 L=3 layer-tied SNAP vs JOINT vs QAOA p=3",
        "",
        f"Unmodified v4 `{PREREG['ft_source']['alloc_py_sha256']}` / PR79 `{PREREG['ft_source']['head_commit'][:12]}`.",
        "Seeds 10552–10559. N=15 L=3 Q=3072 T=378. Four equal blocks, center-4. Energy selects. F post hoc.",
        "TIE: one SNAP triple repeated L=3, SPSA dim 6. JOINT: per-layer SNAP, dim 12. QAOA: 4-start matched.",
        "Not start-count / freeze-order / shell-unshare / two-stage. No GR.",
        "",
        f"Runs {len(rows)} ok {len(ok)} failures {len(fail)}. Charged {charged}. "
        f"Outside cap {summary['outside_cap_queries']}. Rejected B* {len(rejected)}. Missing {len(missing)}.",
        "",
        f"**Primary** TIE−QAOA: {fmt(ct_tq, 'E')} {ct_tq['E']['n_neg']}/{ct_tq['n']} "
        f"{'PASS' if primary_pass else 'FAIL'}",
        f"**Secondary** TIE mean E {ct_tj['mean_E_a']:.6f} vs JOINT {ct_tj['mean_E_b']:.6f} "
        f"{'PASS' if secondary_pass else 'FAIL'}",
        f"**Gate {gate}.** GR not patched.",
        "",
        f"mean F TIE {ct_tq['mean_F_a']:.4f} JOINT {ct_tj['mean_F_b']:.4f} QAOA {ct_tq['mean_F_b']:.4f}",
        f"JOINT−QAOA (report): {fmt(ct_jq, 'E')} {ct_jq['E']['n_neg']}/{ct_jq['n']}",
        "",
        f"B* wall {status.get('bstar_wall_s')}. Fit wall {status.get('fit_wall_s')}. "
        f"Sum cell wall {wall_fit:.1f}s. Sum cell CPU {cpu_fit:.1f}s. Peak RSS KiB {summary['peak_rss_kib_max']}.",
        "",
    ]
    (OUT / "RESULTS_L3TIE.md").write_text("\n".join(lines))
    print(json.dumps({
        "gate": gate, "primary_pass": primary_pass, "secondary_pass": secondary_pass,
        "TIE_QAOA": {"mean": ct_tq["E"]["mean"], "ci95": ct_tq["E"]["ci95_mean"],
                     "wins": f"{ct_tq['E']['n_neg']}/{ct_tq['n']}"},
        "TIE_E": ct_tj["mean_E_a"], "JOINT_E": ct_tj["mean_E_b"],
        "mean_F_TIE": ct_tq["mean_F_a"], "mean_F_JOINT": ct_tj["mean_F_b"],
        "mean_F_QAOA": ct_tq["mean_F_b"],
        "charged": charged, "ok": len(ok), "fail": len(fail),
        "rejected": rejected, "missing": missing, "audit_failures": bad,
    }, indent=1))
    if bad:
        raise SystemExit("audit failed")
    if missing:
        raise SystemExit(f"missing {missing}")


if __name__ == "__main__":
    main()
