"""N18XFER: TRANSFER_MEAN_NEGATIVE_CI iff test shared-mean Nov3−QAOA CI entirely <0."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[1] / "results" / "N18XFER"
PREREG = json.loads((HERE / "PREREG_N18XFER.json").read_text())
TRAIN = list(PREREG["split_documented_before_fits"]["train"])
TEST = list(PREREG["split_documented_before_fits"]["test"])
SEEDS = list(PREREG["seeds"]["ids"])
FORBIDDEN = (
    set(range(10504, 10536)) | set(range(10536, 10552)) | set(range(10552, 10568))
    | set(range(10568, 10584)) | set(range(10584, 10600)) | set(range(10600, 10616))
    | set(range(10616, 10632)) | set(range(10632, 10648)) | set(range(10664, 10680))
    | set(range(10432, 10440)) | set(range(10144, 10152))
    | set(range(11000, 11032)) | set(range(20000, 21000))
)
PR116_STD = {
    "nov3": {"gamma": 0.2520, "theta_K": 0.0511, "theta_M": 0.0233,
             "s1": 0.8058, "s2": 0.1043, "s3": 0.2341},
    "qaoa": {"gamma": 0.2613, "beta": 0.4015},
}
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
        want = "QAOA" if r.get("arm") == "qaoa_center4" else "Nov3"
        if r.get("circuit") != want or r.get("optimizer") != "FT_v4_center4":
            problems.append("circuit")
    return problems


def audit_xfer(r):
    problems = []
    if r["seed"] not in TEST:
        problems.append("not_test")
    if r["seed"] in TRAIN or r["seed"] in FORBIDDEN:
        problems.append("forbidden_seed")
    if r.get("N") != 18 or r.get("L") != 1 or r.get("Q") != 1:
        problems.append("batch")
    if r.get("status") == "ok":
        if r["queries"] != 1:
            problems.append("Q")
        qc = r.get("query_cat") or {}
        if qc.get("transfer") != 1:
            problems.append("transfer_cat")
        if (qc.get("init", 0) or qc.get("calibration", 0) or qc.get("perturb", 0)
                or qc.get("candidate", 0) or qc.get("monitor", 0) or qc.get("screen", 0)):
            problems.append("extra")
    return problems


def index_fit(rows, split, arm):
    return {r["seed"]: r for r in rows
            if r.get("role") == "independent_fit" and r.get("split") == split
            and r.get("arm") == arm and r.get("status") == "ok"}


def index_xfer(rows, arm, agg):
    return {r["seed"]: r for r in rows
            if r.get("role") == "transfer_eval" and r.get("arm") == arm
            and r.get("aggregator") == agg and r.get("status") == "ok"}


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


def fmt(ct, key):
    z = ct[key]
    return (f"{z['mean']:+.5f} med {z['median']:+.5f} "
            f"[{z['ci95_mean'][0]:+.5f},{z['ci95_mean'][1]:+.5f}] "
            f"tails [{z['min']:+.5f},{z['max']:+.5f}]")


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
    vec_path = OUT / "TRANSFER_VECTORS_N18XFER.json"
    if not vec_path.exists():
        raise SystemExit("freeze missing")
    freeze = json.loads(vec_path.read_text())
    inst = json.loads((OUT / "INSTANCES_N18XFER.json").read_text()) if (OUT / "INSTANCES_N18XFER.json").exists() else {"rows": []}
    rejected = [r for r in inst.get("rows", []) if not r.get("accepted")]
    d = OUT / "runs" / "N18"
    fits, xfers = [], []
    if d.exists():
        for p in sorted(d.glob("s*_L1_*_Q1024.json")):
            r = json.loads(p.read_text())
            if r.get("stage") == "N18XFER":
                fits.append(r)
        for p in sorted(d.glob("s*_L1_*_xfer_*_Q1.json")):
            r = json.loads(p.read_text())
            if r.get("stage") == "N18XFER":
                xfers.append(r)
    bad = {}
    for r in fits:
        pbl = audit_fit(r)
        if pbl:
            bad[f"fit_s{r['seed']}_{r['arm']}"] = pbl
    for r in xfers:
        pbl = audit_xfer(r)
        if pbl:
            bad[f"xfer_s{r['seed']}_{r['arm']}_{r.get('aggregator')}"] = pbl
    ok_fits = [r for r in fits if r.get("status") == "ok"]
    fail_fits = [r for r in fits if r.get("status") != "ok"]
    ok_xfers = [r for r in xfers if r.get("status") == "ok"]
    fail_xfers = [r for r in xfers if r.get("status") != "ok"]
    charged = int(sum(r.get("queries", 0) for r in fits + xfers))
    rng = np.random.default_rng(3334)
    train_n = index_fit(ok_fits, "train", "nov3_center4")
    train_q = index_fit(ok_fits, "train", "qaoa_center4")
    test_n = index_fit(ok_fits, "test", "nov3_center4")
    test_q = index_fit(ok_fits, "test", "qaoa_center4")
    mean_n = index_xfer(ok_xfers, "nov3_center4", "mean")
    mean_q = index_xfer(ok_xfers, "qaoa_center4", "mean")
    med_n = index_xfer(ok_xfers, "nov3_center4", "median")
    med_q = index_xfer(ok_xfers, "qaoa_center4", "median")
    ct_train = contrast(train_n, train_q, rng)
    ct_mean = contrast(mean_n, mean_q, rng)
    ct_med = contrast(med_n, med_q, rng)
    ct_ceil = contrast(test_n, test_q, rng)
    ct_gap = contrast(mean_n, test_n, rng)
    label = ("TRANSFER_MEAN_NEGATIVE_CI"
             if ct_mean["n"] == 8 and ct_mean["E"]["ci95_mean"][1] < 0
             else "UNSTABLE")
    vs_pr116 = {}
    for arm, key in (("nov3", "nov3"), ("qaoa", "qaoa")):
        block = freeze[arm]
        for i, name in enumerate(block["names"]):
            vs_pr116[f"{arm}_{name}"] = {
                "this": float(block["std"][i]),
                "pr116": float(PR116_STD[key][name]),
            }
    status = json.loads((OUT / "STATUS_N18XFER.json").read_text()) if (OUT / "STATUS_N18XFER.json").exists() else {}
    wall = float(sum(r.get("wall_s", 0) for r in fits + xfers))
    cpu = float(sum(r.get("cpu_s", 0) or 0 for r in fits + xfers))
    rss = [r.get("peak_rss_kib") for r in fits + xfers if r.get("peak_rss_kib") is not None]
    summary = {
        "experiment": "N18XFER",
        "n_fits": len(fits), "n_ok_fits": len(ok_fits), "n_fail_fits": len(fail_fits),
        "n_xfers": len(xfers), "n_ok_xfers": len(ok_xfers), "n_fail_xfers": len(fail_xfers),
        "failures": [{"seed": r["seed"], "arm": r.get("arm"), "role": r.get("role"),
                      "error": r.get("error")} for r in fail_fits + fail_xfers],
        "rejected_instances": rejected,
        "queries_charged": charged,
        "outside_cap_queries": int(sum(r.get("outside_cap_queries", 0) for r in fits + xfers)),
        "audit_failures": bad,
        "label": label,
        "contrast_train": ct_train,
        "contrast_test_mean": ct_mean,
        "contrast_test_median": ct_med,
        "contrast_test_ceiling": ct_ceil,
        "contrast_transfer_gap": ct_gap,
        "concentration_train_nov3": freeze.get("nov3"),
        "concentration_train_qaoa": freeze.get("qaoa"),
        "vs_pr116_train_std_descriptive": vs_pr116,
        "adoption": "none",
        "gr": "not patched",
        "does_not_revive_l2_l3": True,
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
    (OUT / "summary_N18XFER.json").write_text(json.dumps(summary, indent=1))
    lines = [
        "# N18XFER: N=18 L=1 parameter concentration and transfer",
        "",
        f"Unmodified v4 `{PREREG['ft_source']['alloc_py_sha256']}` / PR79 `{PREREG['ft_source']['head_commit'][:12]}`.",
        "Train 10648–10655, test 10656–10663. N=18 L=1 Q=1024 T=122 center-4. Energy selects. F post hoc.",
        "Mean and median frozen from train before test. Primary aggregator remains mean (not switched to median from PR116 test). Not L2/L3. Not architecture. No GR.",
        "",
        f"Fits {len(fits)} ok {len(ok_fits)}. Transfer evals {len(xfers)} ok {len(ok_xfers)}. "
        f"Charged {charged}. Rejected B* {len(rejected)}.",
        "",
        f"**Label {label}.**",
        f"1 train Nov3−QAOA: {fmt(ct_train, 'E')} {ct_train['E']['n_neg']}/{ct_train['n']}",
        f"2a test mean Nov3−QAOA: {fmt(ct_mean, 'E')} {ct_mean['E']['n_neg']}/{ct_mean['n']}",
        f"2b test median Nov3−QAOA: {fmt(ct_med, 'E')} {ct_med['E']['n_neg']}/{ct_med['n']}",
        f"3 test ceiling Nov3−QAOA: {fmt(ct_ceil, 'E')} {ct_ceil['E']['n_neg']}/{ct_ceil['n']}",
        f"4 transfer gap mean vs independent Nov3: {fmt(ct_gap, 'E')} (descriptive)",
        "",
        conc_line(freeze["nov3"], "Train Nov3"),
        conc_line(freeze["qaoa"], "Train QAOA"),
        "vs PR116 N=15 train std (descriptive, different seeds, not paired): "
        + "; ".join(f"{k} {v['this']:.4f} vs {v['pr116']:.4f}" for k, v in vs_pr116.items()),
        "",
        f"B* wall {status.get('bstar_wall_s')}. Train fit wall {status.get('train_fit_wall_s')}. "
        f"Test wall {status.get('test_wall_s')}. Sum cell wall {wall:.1f}s. Peak RSS KiB {summary['peak_rss_kib_max']}.",
        "",
    ]
    (OUT / "RESULTS_N18XFER.md").write_text("\n".join(lines))
    print(json.dumps({
        "label": label,
        "train": {"mean": ct_train["E"]["mean"], "ci95": ct_train["E"]["ci95_mean"],
                  "wins": f"{ct_train['E']['n_neg']}/{ct_train['n']}"},
        "test_mean": {"mean": ct_mean["E"]["mean"], "ci95": ct_mean["E"]["ci95_mean"],
                      "wins": f"{ct_mean['E']['n_neg']}/{ct_mean['n']}"},
        "test_median": {"mean": ct_med["E"]["mean"], "ci95": ct_med["E"]["ci95_mean"],
                        "wins": f"{ct_med['E']['n_neg']}/{ct_med['n']}"},
        "test_ceiling": {"mean": ct_ceil["E"]["mean"], "ci95": ct_ceil["E"]["ci95_mean"],
                         "wins": f"{ct_ceil['E']['n_neg']}/{ct_ceil['n']}"},
        "transfer_gap": {"mean": ct_gap["E"]["mean"], "ci95": ct_gap["E"]["ci95_mean"]},
        "charged": charged, "ok_fits": len(ok_fits), "ok_xfers": len(ok_xfers),
        "rejected": rejected, "audit_failures": bad,
    }, indent=1))
    if bad:
        raise SystemExit("audit failed")


if __name__ == "__main__":
    main()
