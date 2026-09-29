"""Phase 3 analysis of the F1 scaling sweep -> noiseless/SCALING_SUMMARY.md tables.

Reads the per-trial checkpoints noiseless/results/scaling_runs/<tag>/*.jsonl and
Hamiltonians/four_sat_scaling/generation_stats.json; writes
noiseless/results/scaling_<tag>_analysis_summary.json and prints markdown tables.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
RUNS = REPO / "noiseless" / "results" / "scaling_runs"


def load(tag):
    cells = {}
    for p in sorted((RUNS / tag).glob("*.jsonl")):
        recs = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
        if recs:
            cells[(recs[0]["arm"], int(recs[0]["n"]))] = recs
    return cells


def fit_log(ns, ys):
    """log10 y = a + b n  -> returns b, c = 10^b per qubit, (n used)."""
    ns, ys = np.asarray(ns, float), np.asarray(ys, float)
    ok = ys > 0
    if ok.sum() < 2:
        return None
    b, a = np.polyfit(ns[ok], np.log10(ys[ok]), 1)
    return {"slope_log10_per_qubit": float(b), "c": float(10**b), "intercept": float(a), "n_used": ns[ok].astype(int).tolist()}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="main")
    a = ap.parse_args(argv)
    cells = load(a.tag)
    out = {"quantum": {}, "classical": {}, "fits": {}, "validation": {}}
    md = []
    gen = json.loads((REPO / "Hamiltonians" / "four_sat_scaling" / "generation_stats.json").read_text())["per_n"]
    md.append("| n | m mean [min,max] | m/n mean [min,max] | m_base mean | top-up mean | #sol after base (median) |")
    md.append("|---|---|---|---|---|---|")
    for n, s in sorted(gen.items(), key=lambda kv: int(kv[0])):
        md.append(f"| {n} | {s['m_mean']:.1f} [{s['m_min']},{s['m_max']}] | {s['m_over_n_mean']:.2f} [{s['m_over_n_min']:.2f},{s['m_over_n_max']:.2f}] | {s['m_base_mean']:.1f} | {s['n_topup_mean']:.1f} | {s['n_solutions_base_median']:.0f} |")
    md.append("")
    arms_q = ["ry0", "ry0_split", "ry0_sampled", "hea1", "hea2"]
    md.append("| arm | n | params | cost | inst | trials | success | mean p(GS) | median p(GS) | mean <H> | s/trial |")
    md.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for arm in arms_q:
        for n in sorted(n for (ar, n) in cells if ar == arm):
            r = cells[(arm, n)]
            pg = np.array([x["p_gs"] for x in r])
            s = {
                "n_params": r[0]["n_params"], "cost_mode": r[0].get("cost_mode", "exact"),
                "instances": len({x["inst"] for x in r}), "trials": len(r),
                "success": float(np.mean([x["success"] for x in r])),
                "mean_p_gs": float(pg.mean()), "median_p_gs": float(np.median(pg)),
                "mean_energy": float(np.mean([x["energy_mean"] for x in r])),
                "wall_s": float(np.mean([x["wall_s"] for x in r])),
            }
            out["quantum"][f"{arm}_n{n:02d}"] = s
            md.append(f"| {arm} | {n} | {s['n_params']} | {s['cost_mode']} | {s['instances']} | {s['trials']} | {s['success']:.3f} | {s['mean_p_gs']:.3g} | {s['median_p_gs']:.3g} | {s['mean_energy']:.3f} | {s['wall_s']:.1f} |")
    md.append("")
    # classical
    cl = {n: r for (ar, n), r in cells.items() if ar == "classical"}
    if cl:
        budgets = cl[min(cl)][0]["budgets"]
        md.append("| n | inst | " + " | ".join(f"WalkSAT@{b}" for b in budgets) + " | " + " | ".join(f"SA@{b}" for b in budgets) + " |")
        md.append("|---|---|" + "---|" * (2 * len(budgets)))
        for n in sorted(cl):
            r = cl[n]
            ws = {b: float(np.mean([x["walksat_success"][str(b)] for x in r])) for b in budgets}
            sa = {b: float(np.mean([x["sa_success"][str(b)] for x in r])) for b in budgets}
            hits = [h for x in r for h in x["walksat_first_hit"] if h >= 0]
            out["classical"][str(n)] = {"instances": len(r), "walksat": ws, "sa": sa,
                                        "walksat_median_first_hit": float(np.median(hits)) if hits else None}
            md.append(f"| {n} | {len(r)} | " + " | ".join(f"{ws[b]:.3f}" for b in budgets) + " | " + " | ".join(f"{sa[b]:.3f}" for b in budgets) + " |")
        md.append("")
    # fits
    md.append("| series | n range | slope log10/qubit | c (per-qubit factor) |")
    md.append("|---|---|---|---|")
    for arm in ["ry0", "hea1", "hea2"]:
        ks = sorted(n for (ar, n) in cells if ar == arm)
        for stat in ("mean_p_gs", "median_p_gs", "success"):
            ys = [out["quantum"][f"{arm}_n{n:02d}"][stat] for n in ks]
            f = fit_log(ks, ys)
            if f:
                out["fits"][f"{arm}:{stat}"] = f
                md.append(f"| {arm} {stat} | {f['n_used'][0]}-{f['n_used'][-1]} ({len(f['n_used'])} pts) | {f['slope_log10_per_qubit']:.4f} | {f['c']:.4f} |")
    if cl:
        for alg in ("walksat", "sa"):
            for b in budgets:
                ks = sorted(cl)
                ys = [out["classical"][str(n)][alg][b] for n in ks]
                f = fit_log(ks, ys)
                if f and min(ys) < 1.0:
                    out["fits"][f"{alg}@{b}:success"] = f
                    md.append(f"| {alg} success @{b} | {f['n_used'][0]}-{f['n_used'][-1]} ({len(f['n_used'])} pts) | {f['slope_log10_per_qubit']:.4f} | {f['c']:.4f} |")
    md.append("")
    # validation: paired exact vs sampled
    md.append("| n | trials | exact success | sampled success | exact mean p(GS) | sampled mean p(GS) | exact median | sampled median | same final ML bitstring | Pearson r(log10 p) |")
    md.append("|---|---|---|---|---|---|---|---|---|---|")
    for n in sorted(n for (ar, n) in cells if ar == "ry0_sampled"):
        if ("ry0", n) not in cells:
            continue
        ex = {(x["inst"], x["trial"]): x for x in cells[("ry0", n)]}
        sa = {(x["inst"], x["trial"]): x for x in cells[("ry0_sampled", n)]}
        keys = sorted(set(ex) & set(sa))
        pe = np.array([ex[k]["p_gs"] for k in keys]); ps = np.array([sa[k]["p_gs"] for k in keys])
        v = {
            "trials": len(keys),
            "exact_success": float(np.mean([ex[k]["success"] for k in keys])),
            "sampled_success": float(np.mean([sa[k]["success"] for k in keys])),
            "exact_mean_p_gs": float(pe.mean()), "sampled_mean_p_gs": float(ps.mean()),
            "exact_median_p_gs": float(np.median(pe)), "sampled_median_p_gs": float(np.median(ps)),
            "same_ml_frac": float(np.mean([ex[k]["most_likely_bitstring"] == sa[k]["most_likely_bitstring"] for k in keys])),
            "pearson_log10": float(np.corrcoef(np.log10(np.maximum(pe, 1e-300)), np.log10(np.maximum(ps, 1e-300)))[0, 1]),
        }
        out["validation"][str(n)] = v
        md.append(f"| {n} | {v['trials']} | {v['exact_success']:.3f} | {v['sampled_success']:.3f} | {v['exact_mean_p_gs']:.3g} | {v['sampled_mean_p_gs']:.3g} | {v['exact_median_p_gs']:.3g} | {v['sampled_median_p_gs']:.3g} | {v['same_ml_frac']:.3f} | {v['pearson_log10']:.3f} |")
    (REPO / "noiseless" / "results" / f"scaling_{a.tag}_analysis_summary.json").write_text(json.dumps(out, indent=1) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
