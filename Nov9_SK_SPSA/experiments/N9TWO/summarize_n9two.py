"""N9TWO: TWO_TEMPLATE_TRANSFER_NEGATIVE_CI iff better-of-two Nov3−QAOA CI entirely <0."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[1] / "results" / "N9TWO"
PREREG = json.loads((HERE / "PREREG_N9TWO.json").read_text())
TRAIN = list(PREREG["split_documented_before_fits"]["train"])
TEST = list(PREREG["split_documented_before_fits"]["test"])
SEEDS = list(PREREG["seeds"]["ids"])
FORBIDDEN = (
    set(range(10504, 10536)) | set(range(10536, 10552)) | set(range(10552, 10568))
    | set(range(10568, 10584)) | set(range(10584, 10600)) | set(range(10600, 10616))
    | set(range(10616, 10632)) | set(range(10632, 10648)) | set(range(10648, 10664))
    | set(range(10664, 10680)) | set(range(10680, 10696)) | set(range(10696, 10712))
    | set(range(10712, 10728)) | set(range(10728, 10744)) | set(range(10744, 10760))
    | set(range(10760, 10776)) | set(range(10776, 10792)) | set(range(10792, 10808))
    | set(range(10808, 10824)) | set(range(10824, 10840)) | set(range(10856, 10872))
    | set(range(10432, 10440)) | set(range(10144, 10152))
    | set(range(11000, 11032)) | set(range(20000, 21000))
)
N_BOOT = 4000
QFIX = 1024
ALLOWED_XFER = {
    ("nov3_center4", "cluster_pos"), ("nov3_center4", "cluster_neg"),
    ("qaoa_center4", "cluster_pos"), ("qaoa_center4", "cluster_neg"),
    ("nov3_center4", "unsigned_mean"), ("nov3_center4", "sign_aligned_mean"),
    ("qaoa_center4", "unsigned_mean"),
}


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
    if r.get("N") != 9 or r.get("L") != 1 or r.get("Q") != QFIX:
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
    if r.get("N") != 9 or r.get("L") != 1 or r.get("Q") != 1:
        problems.append("batch")
    if (r.get("arm"), r.get("aggregator")) not in ALLOWED_XFER:
        problems.append("aggregator")
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


def better_of(pos, neg):
    """Energy-min of nonempty cluster templates. No extra query."""
    out = {}
    seeds = sorted(set(pos) | set(neg))
    which = {}
    for s in seeds:
        cands = []
        if s in pos:
            cands.append(("cluster_pos", pos[s]))
        if s in neg:
            cands.append(("cluster_neg", neg[s]))
        if not cands:
            continue
        chosen_name, chosen = min(cands, key=lambda z: z[1]["diag"]["E"])
        out[s] = chosen
        which[s] = chosen_name
    return out, which


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
    if block is None or block.get("empty"):
        return f"{title} EMPTY (no template invented)"
    names = block["names"]
    parts = [f"{title} n={block['n']}"]
    for i, name in enumerate(names):
        parts.append(
            f"{name} mean {block['mean'][i]:+.4f} std {block['std'][i]:.4f} "
            f"range [{block['min'][i]:+.4f},{block['max'][i]:+.4f}]"
        )
    return "; ".join(parts)


def maybe_contrast(a, b, rng):
    if not a or not b or not (set(a) & set(b)):
        return None
    return contrast(a, b, rng)


def fmt_or(ct, key):
    if ct is None:
        return "n/a (empty cluster)"
    return f"{fmt(ct, key)} {ct['E']['n_neg']}/{ct['n']}"


def main():
    vec_path = OUT / "TRANSFER_VECTORS_N9TWO.json"
    if not vec_path.exists():
        raise SystemExit("freeze missing")
    freeze = json.loads(vec_path.read_text())
    inst = json.loads((OUT / "INSTANCES_N9TWO.json").read_text()) if (OUT / "INSTANCES_N9TWO.json").exists() else {"rows": []}
    rejected = [r for r in inst.get("rows", []) if not r.get("accepted")]
    d = OUT / "runs" / "N9"
    fits, xfers = [], []
    if d.exists():
        for p in sorted(d.glob("s*_L1_*_Q1024.json")):
            r = json.loads(p.read_text())
            if r.get("stage") == "N9TWO":
                fits.append(r)
        for p in sorted(d.glob("s*_L1_*_xfer_*_Q1.json")):
            r = json.loads(p.read_text())
            if r.get("stage") == "N9TWO":
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
    n_pos = index_xfer(ok_xfers, "nov3_center4", "cluster_pos")
    n_neg = index_xfer(ok_xfers, "nov3_center4", "cluster_neg")
    q_pos = index_xfer(ok_xfers, "qaoa_center4", "cluster_pos")
    q_neg = index_xfer(ok_xfers, "qaoa_center4", "cluster_neg")
    unsigned_n = index_xfer(ok_xfers, "nov3_center4", "unsigned_mean")
    aligned_n = index_xfer(ok_xfers, "nov3_center4", "sign_aligned_mean")
    unsigned_q = index_xfer(ok_xfers, "qaoa_center4", "unsigned_mean")
    two_n, which_n = better_of(n_pos, n_neg)
    two_q, which_q = better_of(q_pos, q_neg)
    ct_primary = maybe_contrast(two_n, two_q, rng)
    ct_train = maybe_contrast(train_n, train_q, rng)
    ct_ceil = maybe_contrast(test_n, test_q, rng)
    ct_two_vs_ind_q = maybe_contrast(two_n, test_q, rng)
    ct_gap = maybe_contrast(two_n, test_n, rng)
    ct_npos_u = maybe_contrast(n_pos, unsigned_q, rng)
    ct_nneg_u = maybe_contrast(n_neg, unsigned_q, rng)
    ct_npos_indq = maybe_contrast(n_pos, test_q, rng)
    ct_nneg_indq = maybe_contrast(n_neg, test_q, rng)
    ct_unsigned = maybe_contrast(unsigned_n, unsigned_q, rng)
    ct_aligned = maybe_contrast(aligned_n, unsigned_q, rng)
    label = ("TWO_TEMPLATE_TRANSFER_NEGATIVE_CI"
             if ct_primary is not None and ct_primary["n"] == 8
             and ct_primary["E"]["ci95_mean"][1] < 0
             else "UNSTABLE")
    gap_ci = None if ct_gap is None else ct_gap["E"]["ci95_mean"]
    transfer_gap_ci_entirely_gt0 = bool(
        gap_ci is not None and gap_ci[0] > 0
    )
    transfer_gap_ci_includes_0 = bool(
        gap_ci is not None and gap_ci[0] <= 0 <= gap_ci[1]
    )
    status = json.loads((OUT / "STATUS_N9TWO.json").read_text()) if (OUT / "STATUS_N9TWO.json").exists() else {}
    wall = float(sum(r.get("wall_s", 0) for r in fits + xfers))
    cpu = float(sum(r.get("cpu_s", 0) or 0 for r in fits + xfers))
    rss = [r.get("peak_rss_kib") for r in fits + xfers if r.get("peak_rss_kib") is not None]
    nc = freeze.get("nov3_clusters") or {}
    qc = freeze.get("qaoa_clusters") or {}
    summary = {
        "experiment": "N9TWO",
        "n_fits": len(fits), "n_ok_fits": len(ok_fits), "n_fail_fits": len(fail_fits),
        "n_xfers": len(xfers), "n_ok_xfers": len(ok_xfers), "n_fail_xfers": len(fail_xfers),
        "failures": [{"seed": r["seed"], "arm": r.get("arm"), "role": r.get("role"),
                      "error": r.get("error")} for r in fail_fits + fail_xfers],
        "rejected_instances": rejected,
        "queries_charged": charged,
        "outside_cap_queries": int(sum(r.get("outside_cap_queries", 0) for r in fits + xfers)),
        "audit_failures": bad,
        "label": label,
        "primary_rule": "better_of_two_cluster_means",
        "nov3_n_templates": nc.get("n_templates"),
        "qaoa_n_templates": qc.get("n_templates"),
        "nov3_n_pos": nc.get("n_pos"), "nov3_n_neg": nc.get("n_neg"),
        "qaoa_n_pos": qc.get("n_pos"), "qaoa_n_neg": qc.get("n_neg"),
        "two_template_choice_nov3": which_n,
        "two_template_choice_qaoa": which_q,
        "contrast_primary_two_template": ct_primary,
        "contrast_train": ct_train,
        "contrast_test_ceiling": ct_ceil,
        "contrast_two_nov3_vs_independent_qaoa": ct_two_vs_ind_q,
        "contrast_transfer_gap": ct_gap,
        "transfer_gap_ci_entirely_gt0": transfer_gap_ci_entirely_gt0,
        "transfer_gap_ci_includes_0": transfer_gap_ci_includes_0,
        "transfer_gap_does_not_switch_primary": True,
        "contrast_nov3_pos_vs_unsigned_qaoa": ct_npos_u,
        "contrast_nov3_neg_vs_unsigned_qaoa": ct_nneg_u,
        "contrast_nov3_pos_vs_independent_qaoa": ct_npos_indq,
        "contrast_nov3_neg_vs_independent_qaoa": ct_nneg_indq,
        "contrast_unsigned_replication": ct_unsigned,
        "contrast_sign_aligned_replication": ct_aligned,
        "concentration_train_nov3_unsigned": freeze.get("nov3_unsigned"),
        "concentration_train_nov3_sign_aligned": freeze.get("nov3_sign_aligned"),
        "concentration_train_nov3_clusters": freeze.get("nov3_clusters"),
        "concentration_train_qaoa_unsigned": freeze.get("qaoa_unsigned"),
        "concentration_train_qaoa_clusters": freeze.get("qaoa_clusters"),
        "gauge": freeze.get("gauge"),
        "adoption": "none",
        "gr": "not patched",
        "does_not_revive_l2_l3": True,
        "does_not_reopen_pr119": True,
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
    (OUT / "summary_N9TWO.json").write_text(json.dumps(summary, indent=1))
    lines = [
        "# N9TWO: N=9 L=1 two-template bimodal cluster-mean transfer",
        "",
        f"Unmodified v4 `{PREREG['ft_source']['alloc_py_sha256']}` / PR79 `{PREREG['ft_source']['head_commit'][:12]}`.",
        "Train 10840–10847, test 10848–10855. N=9 L=1 Q=1024 T=122 center-4. Energy selects. F post hoc.",
        "Two Nov3 cluster means (sign(s1)) and two QAOA cluster means (sign(beta)) frozen from train before test.",
        "Two-template energy = min of nonempty cluster-template energies. Unsigned/sign-aligned secondary only. No GR. Does not reopen PR119.",
        "",
        f"Fits {len(fits)} ok {len(ok_fits)}. Transfer evals {len(xfers)} ok {len(ok_xfers)}. "
        f"Charged {charged}. Rejected B* {len(rejected)}.",
        f"Nov3 templates {nc.get('n_templates')} (n_pos={nc.get('n_pos')} n_neg={nc.get('n_neg')}); "
        f"QAOA templates {qc.get('n_templates')} (n_pos={qc.get('n_pos')} n_neg={qc.get('n_neg')}).",
        "",
        f"**Label {label}.**",
        f"primary two-template Nov3 − two-template QAOA: {fmt_or(ct_primary, 'E')}",
        f"1 train independent Nov3−QAOA: {fmt_or(ct_train, 'E')}",
        f"2 test ceiling independent Nov3−QAOA: {fmt_or(ct_ceil, 'E')}",
        f"3 two-template Nov3 vs independent QAOA: {fmt_or(ct_two_vs_ind_q, 'E')}",
        f"4 transfer gap two-template Nov3 vs independent Nov3: {fmt_or(ct_gap, 'E')} "
        f"(CI entirely >0: {transfer_gap_ci_entirely_gt0}; CI includes 0: {transfer_gap_ci_includes_0}; descriptive only; does not switch primary)",
        f"5a Nov3 cluster+ vs unsigned QAOA: {fmt_or(ct_npos_u, 'E')}",
        f"5b Nov3 cluster− vs unsigned QAOA: {fmt_or(ct_nneg_u, 'E')}",
        f"5c Nov3 cluster+ vs independent QAOA: {fmt_or(ct_npos_indq, 'E')}",
        f"5d Nov3 cluster− vs independent QAOA: {fmt_or(ct_nneg_indq, 'E')}",
        f"6a unsigned-mean Nov3 − unsigned QAOA (replication): {fmt_or(ct_unsigned, 'E')}",
        f"6b sign-aligned Nov3 − unsigned QAOA (replication): {fmt_or(ct_aligned, 'E')}",
        "",
        conc_line(freeze.get("nov3_unsigned"), "Train Nov3 unsigned"),
        conc_line(freeze.get("nov3_sign_aligned"), "Train Nov3 sign-aligned"),
        conc_line(nc.get("pos"), "Train Nov3 cluster+"),
        conc_line(nc.get("neg"), "Train Nov3 cluster-"),
        conc_line(freeze.get("qaoa_unsigned"), "Train QAOA unsigned"),
        conc_line(qc.get("pos"), "Train QAOA cluster+"),
        conc_line(qc.get("neg"), "Train QAOA cluster-"),
        "",
        f"B* wall {status.get('bstar_wall_s')}. Train fit wall {status.get('train_fit_wall_s')}. "
        f"Test wall {status.get('test_wall_s')}. Sum cell wall {wall:.1f}s. Peak RSS KiB {summary['peak_rss_kib_max']}.",
        "",
    ]
    (OUT / "RESULTS_N9TWO.md").write_text("\n".join(lines))

    def slim(ct):
        if ct is None:
            return None
        return {"mean": ct["E"]["mean"], "ci95": ct["E"]["ci95_mean"],
                "wins": f"{ct['E']['n_neg']}/{ct['n']}"}

    print(json.dumps({
        "label": label,
        "primary": slim(ct_primary),
        "train": slim(ct_train),
        "ceiling": slim(ct_ceil),
        "two_vs_ind_qaoa": slim(ct_two_vs_ind_q),
        "transfer_gap": slim(ct_gap),
        "transfer_gap_ci_entirely_gt0": transfer_gap_ci_entirely_gt0,
        "transfer_gap_ci_includes_0": transfer_gap_ci_includes_0,
        "charged": charged, "ok_fits": len(ok_fits), "ok_xfers": len(ok_xfers),
        "rejected": rejected, "audit_failures": bad,
        "nov3_n_pos": nc.get("n_pos"), "nov3_n_neg": nc.get("n_neg"),
        "qaoa_n_pos": qc.get("n_pos"), "qaoa_n_neg": qc.get("n_neg"),
        "n_xfers": len(xfers),
    }, indent=1))
    if bad:
        raise SystemExit("audit failed")


if __name__ == "__main__":
    main()
