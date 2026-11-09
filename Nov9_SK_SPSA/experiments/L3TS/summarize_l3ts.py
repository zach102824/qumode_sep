"""L3TS gates: TWO_STAGE vs QAOA CI entirely <0, and TWO_STAGE mean E ≤ JOINT mean E."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[1] / "results" / "L3TS"
PREREG = json.loads((HERE / "PREREG_L3TS.json").read_text())
SEEDS = list(PREREG["seeds"]["ids"])
ARMS = ("JOINT", "TWO_STAGE", "QAOA")
FORBIDDEN = (
    set(range(10010, 10360)) | set(range(10328, 10336)) | set(range(10344, 10360))
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
        if r.get("seed") in SEEDS and r.get("stage") == "L3TS":
            rows.append(r)
    return rows


def audit_row(r):
    problems = []
    if r["seed"] in FORBIDDEN:
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
    arm = r.get("arm")
    if r.get("status") == "ok":
        if r["queries"] != QFIX or r.get("outside_cap_queries", 0) != 0:
            problems.append("Q")
        qc = r.get("query_cat") or {}
        if qc.get("monitor", 0) != 0 or qc.get("screen", 0) != 0:
            problems.append("extra")
        if arm == "TWO_STAGE":
            if r.get("optimizer") != "FT_v4_two_stage":
                problems.append("optimizer")
            if r.get("circuit") != "Nov3":
                problems.append("circuit")
            if qc.get("init") != 6 or qc.get("calibration") != 48:
                problems.append("cal")
            if qc.get("perturb") != 3000 or qc.get("candidate") != 18:
                problems.append("cand")
            if r.get("K_per_start") != 186 or r.get("K_shell") != 378:
                problems.append("split")
            qj, qs = r.get("query_cat_joint") or {}, r.get("query_cat_shell") or {}
            if qj.get("init") != 4 or qs.get("init") != 2:
                problems.append("stage_init")
            uc = np.asarray(r.get("uc_frozen") or [], float)
            x = np.asarray(r.get("x") or [], float)
            if uc.size != 3 or x.size < 3 or not np.allclose(x[:3], uc):
                problems.append("uc_moved")
        else:
            if r.get("optimizer") != "FT_v4_center4":
                problems.append("optimizer")
            if r.get("nstarts") != 4 or r.get("K_per_start") != 378:
                problems.append("split")
            if qc.get("init") != 4 or qc.get("calibration") != 32:
                problems.append("cal")
            if qc.get("perturb") != 3024 or qc.get("candidate") != 12:
                problems.append("cand")
            want = "QAOA" if arm == "QAOA" else "Nov3"
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
    rec["n_F_wins_a"] = rec["F"]["n_pos"]
    rec["mean_E_a"] = float(np.mean([a[s]["diag"]["E"] for s in seeds])) if seeds else float("nan")
    rec["mean_E_b"] = float(np.mean([b[s]["diag"]["E"] for s in seeds])) if seeds else float("nan")
    rec["mean_F_a"] = float(np.mean([a[s]["diag"]["F"] for s in seeds])) if seeds else float("nan")
    rec["mean_F_b"] = float(np.mean([b[s]["diag"]["F"] for s in seeds])) if seeds else float("nan")
    return rec


def fmt(ct, key):
    z = ct[key]
    return (f"{z['mean']:+.5f} med {z['median']:+.5f} "
            f"[{z['ci95_mean'][0]:+.5f},{z['ci95_mean'][1]:+.5f}] "
            f"tails [{z['min']:+.5f},{z['max']:+.5f}]")


def main():
    inst = json.loads((OUT / "INSTANCES_L3TS.json").read_text()) if (OUT / "INSTANCES_L3TS.json").exists() else {"rows": []}
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
    two = index(ok, "TWO_STAGE")
    joint = index(ok, "JOINT")
    qaoa = index(ok, "QAOA")
    ct_tq = contrast(two, qaoa, rng)
    ct_tj = contrast(two, joint, rng)
    ct_jq = contrast(joint, qaoa, rng)
    primary_pass = bool(ct_tq["n"] > 0 and ct_tq["E"]["ci95_mean"][1] < 0)
    secondary_pass = bool(ct_tj["n"] > 0 and ct_tj["mean_E_a"] <= ct_tj["mean_E_b"])
    gate = "PASS" if primary_pass and secondary_pass else "FAIL"
    status = json.loads((OUT / "STATUS_L3TS.json").read_text()) if (OUT / "STATUS_L3TS.json").exists() else {}
    wall_fit = float(sum(r.get("wall_s", 0) for r in rows))
    cpu_fit = float(sum(r.get("cpu_s", 0) for r in rows if r.get("cpu_s") is not None))
    rss = [r.get("peak_rss_kib") for r in rows if r.get("peak_rss_kib") is not None]
    summary = {
        "experiment": "L3TS",
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
        "this_stream": "L3TS",
        "contrasts": {
            "TWO_STAGE_minus_QAOA": ct_tq,
            "TWO_STAGE_minus_JOINT": ct_tj,
            "JOINT_minus_QAOA": ct_jq,
        },
        "primary": {
            "rule": "TWO_STAGE vs QAOA mean ΔE CI entirely <0",
            "mean_dE": ct_tq["E"]["mean"],
            "ci95": ct_tq["E"]["ci95_mean"],
            "wins": ct_tq["E"]["n_neg"],
            "n": ct_tq["n"],
            "pass": primary_pass,
        },
        "secondary": {
            "rule": "TWO_STAGE mean E ≤ JOINT mean E",
            "mean_E_TWO_STAGE": ct_tj["mean_E_a"],
            "mean_E_JOINT": ct_tj["mean_E_b"],
            "pass": secondary_pass,
        },
        "gate": gate,
        "mean_F": {
            "TWO_STAGE": ct_tq["mean_F_a"],
            "QAOA": ct_tq["mean_F_b"],
            "JOINT": ct_tj["mean_F_b"],
        },
        "adoption": "none",
        "gr": "not patched",
        "seeds": SEEDS, "band": "exclusive_unread_dev",
        "ft_alloc_sha256": PREREG["ft_source"]["alloc_py_sha256"],
        "ft_head_commit": PREREG["ft_source"]["head_commit"],
        "split": PREREG["split_documented_before_fits"],
    }
    (OUT / "summary_L3TS.json").write_text(json.dumps(summary, indent=1))
    lines = [
        "# L3TS: N=15 L=3 two-stage SPSA vs JOINT vs QAOA p=3",
        "",
        f"JOINT/QAOA: unmodified v4 center-4 `{PREREG['ft_source']['alloc_py_sha256']}` / PR79 `{PREREG['ft_source']['head_commit'][:12]}`.",
        "TWO_STAGE: Q=1536 joint center-4 (T=186, 4 starts) then freeze Uc; Q=1536 shell-only (T=378, 2 blocks).",
        "Seeds 10360–10367. N=15 L=3 Q=3072. Energy selects. F post hoc. No pair-phase. No GR.",
        "",
        f"Runs {len(rows)} ok {len(ok)} failures {len(fail)}. Charged {charged}. "
        f"Outside cap {summary['outside_cap_queries']}. Rejected B* {len(rejected)}. Missing {len(missing)}.",
        "",
        f"**Primary** TWO_STAGE−QAOA: {fmt(ct_tq, 'E')} {ct_tq['E']['n_neg']}/{ct_tq['n']} "
        f"{'PASS' if primary_pass else 'FAIL'}",
        f"**Secondary** TWO_STAGE mean E {ct_tj['mean_E_a']:+.6f} vs JOINT {ct_tj['mean_E_b']:+.6f} "
        f"{'PASS' if secondary_pass else 'FAIL'}",
        f"**Gate {gate}.** GR not patched.",
        "",
        f"mean F TWO_STAGE {ct_tq['mean_F_a']:.4f} JOINT {ct_tj['mean_F_b']:.4f} QAOA {ct_tq['mean_F_b']:.4f}",
        f"JOINT−QAOA (report): {fmt(ct_jq, 'E')} {ct_jq['E']['n_neg']}/{ct_jq['n']}",
        "",
        f"B* wall {status.get('bstar_wall_s')}. Fit wall {status.get('fit_wall_s')}. "
        f"Sum cell wall {wall_fit:.1f}s. Sum cell CPU {cpu_fit:.1f}s. Peak RSS KiB {summary['peak_rss_kib_max']}.",
        "",
    ]
    (OUT / "RESULTS_L3TS.md").write_text("\n".join(lines))
    print(json.dumps({
        "gate": gate, "primary_pass": primary_pass, "secondary_pass": secondary_pass,
        "primary": summary["primary"], "secondary": summary["secondary"],
        "mean_F": summary["mean_F"], "charged": charged,
        "ok": len(ok), "fail": len(fail), "rejected": rejected, "missing": missing,
        "audit_failures": bad,
    }, indent=1))
    if bad:
        raise SystemExit("audit failed")
    if missing:
        raise SystemExit(f"missing {missing}")


if __name__ == "__main__":
    main()
