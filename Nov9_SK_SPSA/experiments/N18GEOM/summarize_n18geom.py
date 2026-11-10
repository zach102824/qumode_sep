"""N18GEOM: GEOM_TRANSFER_NEGATIVE_CI iff geom-Nov3 − QAOA CI entirely <0."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[1] / "results" / "N18GEOM"
PREREG = json.loads((HERE / "PREREG_N18GEOM.json").read_text())
TRAIN = list(PREREG["split_documented_before_fits"]["train"])
TEST = list(PREREG["split_documented_before_fits"]["test"])
SEEDS = list(PREREG["seeds"]["ids"])
FORBIDDEN = (
    set(range(10560, 10568)) | set(range(10600, 10616)) | set(range(10624, 10632))
    | set(range(10648, 10664)) | set(range(10680, 10696)) | set(range(10712, 10728))
    | set(range(10728, 10744)) | set(range(10744, 10760)) | set(range(10760, 10776))
    | set(range(10776, 10792)) | set(range(10792, 10808)) | set(range(10808, 10824))
    | set(range(10824, 10840)) | set(range(10840, 10856)) | set(range(10856, 10872))
    | set(range(10888, 10904))
    | set(range(10144, 10152)) | set(range(11000, 11032)) | set(range(20000, 21000))
)
N_BOOT = 4000
QFIX = 1024


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


def audit_fit(r):
    problems = []
    if r["seed"] in FORBIDDEN:
        problems.append("forbidden_seed")
    if r["seed"] not in SEEDS:
        problems.append("seed")
    if r.get("N") != 18 or r.get("L") != 1 or r.get("Q") != QFIX:
        problems.append("batch")
    if r.get("continuation") or r.get("L", 1) != 1:
        problems.append("l2l3")
    if r.get("status") == "ok":
        if r["queries"] != QFIX or r.get("outside_cap_queries", 0) != 0:
            problems.append("Q")
        if r.get("nstarts") != 4 or r.get("K_per_start") != 122:
            problems.append("split")
        qc = r.get("query_cat") or {}
        if qc.get("init") != 4 or qc.get("calibration") != 32:
            problems.append("cal")
        if qc.get("perturb") != 976 or qc.get("candidate") != 12:
            problems.append("cand")
        if qc.get("transfer", 0) != 0 or qc.get("monitor", 0) != 0 or qc.get("screen", 0) != 0:
            problems.append("extra")
        if r.get("arm") == "nov3_geom_tk_tm":
            if r.get("circuit") != "Nov3" or r.get("role") != "geom_spsa":
                problems.append("circuit")
            x = r.get("x") or []
            fm = r.get("frozen_mean") or {}
            if len(x) != 6:
                problems.append("dim")
            elif abs(x[1] - float(fm.get("theta_K", 1e9))) > 1e-10:
                problems.append("tk_not_frozen")
            elif abs(x[2] - float(fm.get("theta_M", 1e9))) > 1e-10:
                problems.append("tm_not_frozen")
        else:
            want = "QAOA" if r.get("arm") == "qaoa_center4" else "Nov3"
            if r.get("circuit") != want or r.get("optimizer") != "FT_v4_center4":
                problems.append("circuit")
    return problems


def index_fit(rows, split, arm):
    return {r["seed"]: r for r in rows
            if r.get("split") == split and r.get("arm") == arm and r.get("status") == "ok"}


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
    rec["mean_E_a"] = float(np.mean([a[s]["diag"]["E"] for s in seeds])) if seeds else float("nan")
    rec["mean_E_b"] = float(np.mean([b[s]["diag"]["E"] for s in seeds])) if seeds else float("nan")
    rec["mean_F_a"] = float(np.mean([a[s]["diag"]["F"] for s in seeds])) if seeds else float("nan")
    rec["mean_F_b"] = float(np.mean([b[s]["diag"]["F"] for s in seeds])) if seeds else float("nan")
    return rec


def maybe_contrast(a, b, rng):
    if not a or not b or not (set(a) & set(b)):
        return None
    return contrast(a, b, rng)


def fmt(ct, key):
    z = ct[key]
    return (f"{z['mean']:+.5f} med {z['median']:+.5f} "
            f"[{z['ci95_mean'][0]:+.5f},{z['ci95_mean'][1]:+.5f}] "
            f"tails [{z['min']:+.5f},{z['max']:+.5f}]")


def fmt_or(ct, key):
    if ct is None:
        return "n/a"
    return f"{fmt(ct, key)} {ct['E']['n_neg']}/{ct['n']}"


def conc_line(block, title):
    names = block["names"]
    parts = [f"{title} n={block['n']}"]
    for i, name in enumerate(names):
        parts.append(
            f"{name} mean {block['mean'][i]:+.4f} std {block['std'][i]:.4f} "
            f"range [{block['min'][i]:+.4f},{block['max'][i]:+.4f}]"
        )
    return "; ".join(parts)


def main():
    vec_path = OUT / "TRANSFER_VECTORS_N18GEOM.json"
    if not vec_path.exists():
        raise SystemExit("freeze missing")
    freeze = json.loads(vec_path.read_text())
    inst = json.loads((OUT / "INSTANCES_N18GEOM.json").read_text()) if (OUT / "INSTANCES_N18GEOM.json").exists() else {"rows": []}
    rejected = [r for r in inst.get("rows", []) if not r.get("accepted")]
    d = OUT / "runs" / "N18"
    fits = []
    if d.exists():
        for p in sorted(d.glob("s*_L1_*_Q1024.json")):
            r = json.loads(p.read_text())
            if r.get("stage") == "N18GEOM":
                fits.append(r)
    bad = {}
    for r in fits:
        pbl = audit_fit(r)
        if pbl:
            bad[f"fit_s{r['seed']}_{r['arm']}"] = pbl
    ok_fits = [r for r in fits if r.get("status") == "ok"]
    fail_fits = [r for r in fits if r.get("status") != "ok"]
    charged = int(sum(r.get("queries", 0) for r in fits))
    rng = np.random.default_rng(3334)
    train_n = index_fit(ok_fits, "train", "nov3_center4")
    test_n = index_fit(ok_fits, "test", "nov3_center4")
    test_q = index_fit(ok_fits, "test", "qaoa_center4")
    geom_n = index_fit(ok_fits, "test", "nov3_geom_tk_tm")
    ct_primary = maybe_contrast(geom_n, test_q, rng)
    ct_ceil = maybe_contrast(test_n, test_q, rng)
    ct_gap = maybe_contrast(geom_n, test_n, rng)
    label = ("GEOM_TRANSFER_NEGATIVE_CI"
             if ct_primary is not None and ct_primary["n"] == 8
             and ct_primary["E"]["ci95_mean"][1] < 0
             else "UNSTABLE")
    gap_ci = None if ct_gap is None else ct_gap["E"]["ci95_mean"]
    transfer_gap_ci_includes_0 = bool(gap_ci is not None and gap_ci[0] <= 0 <= gap_ci[1])
    transfer_gap_ci_entirely_gt0 = bool(gap_ci is not None and gap_ci[0] > 0)
    status = json.loads((OUT / "STATUS_N18GEOM.json").read_text()) if (OUT / "STATUS_N18GEOM.json").exists() else {}
    wall = float(sum(r.get("wall_s", 0) for r in fits))
    cpu = float(sum(r.get("cpu_s", 0) or 0 for r in fits))
    rss = [r.get("peak_rss_kib") for r in fits if r.get("peak_rss_kib") is not None]
    summary = {
        "experiment": "N18GEOM",
        "n_fits": len(fits), "n_ok_fits": len(ok_fits), "n_fail_fits": len(fail_fits),
        "failures": [{"seed": r["seed"], "arm": r.get("arm"), "role": r.get("role"),
                      "error": r.get("error")} for r in fail_fits],
        "rejected_instances": rejected,
        "queries_charged": charged,
        "outside_cap_queries": int(sum(r.get("outside_cap_queries", 0) for r in fits)),
        "audit_failures": bad,
        "label": label,
        "primary_rule": "geom_spsa_tk_tm_frozen",
        "frozen_mean": freeze.get("frozen_mean"),
        "frozen_std": freeze.get("frozen_std"),
        "contrast_primary_geom_vs_qaoa": ct_primary,
        "contrast_test_ceiling": ct_ceil,
        "contrast_transfer_gap": ct_gap,
        "transfer_gap_ci_includes_0": transfer_gap_ci_includes_0,
        "transfer_gap_ci_entirely_gt0": transfer_gap_ci_entirely_gt0,
        "transfer_gap_does_not_switch_primary": True,
        "n_train_nov3": len(train_n),
        "concentration_train_nov3_unsigned": freeze.get("nov3_unsigned"),
        "adoption": "none",
        "gr": "not patched",
        "does_not_revive_l2_l3": True,
        "does_not_reopen_pr119": True,
        "not_a_two_template_copy": True,
        "bstar_wall_s": status.get("bstar_wall_s"),
        "train_fit_wall_s": status.get("train_fit_wall_s"),
        "test_wall_s": status.get("test_wall_s"),
        "sum_cell_wall_s": wall,
        "sum_cell_cpu_s": cpu,
        "peak_rss_kib_max": int(max(rss)) if rss else None,
        "seeds": SEEDS, "train": TRAIN, "test": TEST,
        "ft_alloc_sha256": PREREG["ft_source"]["alloc_py_sha256"],
        "ft_head_commit": PREREG["ft_source"]["head_commit"],
    }
    (OUT / "summary_N18GEOM.json").write_text(json.dumps(summary, indent=1))
    lines = [
        "# N18GEOM: N=18 L=1 SK-geometry freeze (θ_K, θ_M) plus per-instance SPSA",
        "",
        f"Unmodified v4 `{PREREG['ft_source']['alloc_py_sha256']}` / PR79 `{PREREG['ft_source']['head_commit'][:12]}`.",
        "Train 10872–10879, test 10880–10887. N=18 L=1 Q=1024 T=122 center-4. Energy selects. F post hoc.",
        "θ_K and θ_M frozen from train mean before test. Geom SPSA overwrites those coordinates in every energy call.",
        "Not two-template. Not unsigned-mean primary. No GR. Does not reopen PR119.",
        "",
        f"Fits {len(fits)} ok {len(ok_fits)}. Charged {charged}. Rejected B* {len(rejected)}.",
        f"Frozen θ_K={freeze.get('frozen_mean', {}).get('theta_K')} θ_M={freeze.get('frozen_mean', {}).get('theta_M')}.",
        "",
        f"**Label {label}.**",
        f"primary geom Nov3 − QAOA: {fmt_or(ct_primary, 'E')}",
        f"1 test ceiling independent Nov3−QAOA: {fmt_or(ct_ceil, 'E')}",
        f"2 transfer gap geom Nov3 vs independent Nov3: {fmt_or(ct_gap, 'E')} "
        f"(CI includes 0: {transfer_gap_ci_includes_0}; entirely >0: {transfer_gap_ci_entirely_gt0}; descriptive only)",
        "",
        conc_line(freeze.get("nov3_unsigned"), "Train Nov3 unsigned"),
        "",
        f"B* wall {status.get('bstar_wall_s')}. Train fit wall {status.get('train_fit_wall_s')}. "
        f"Test wall {status.get('test_wall_s')}. Sum cell wall {wall:.1f}s. Peak RSS KiB {summary['peak_rss_kib_max']}.",
        "",
    ]
    (OUT / "RESULTS_N18GEOM.md").write_text("\n".join(lines))

    def slim(ct):
        if ct is None:
            return None
        return {"mean": ct["E"]["mean"], "ci95": ct["E"]["ci95_mean"],
                "wins": f"{ct['E']['n_neg']}/{ct['n']}"}

    print(json.dumps({
        "label": label,
        "primary": slim(ct_primary),
        "ceiling": slim(ct_ceil),
        "transfer_gap": slim(ct_gap),
        "transfer_gap_ci_includes_0": transfer_gap_ci_includes_0,
        "transfer_gap_ci_entirely_gt0": transfer_gap_ci_entirely_gt0,
        "frozen_mean": freeze.get("frozen_mean"),
        "charged": charged, "ok_fits": len(ok_fits),
        "rejected": rejected, "audit_failures": bad,
    }, indent=1))
    if bad:
        raise SystemExit("audit failed")


if __name__ == "__main__":
    main()
