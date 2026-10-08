#!/usr/bin/env python3
"""Analysis for RELAYOUT_SUMMARY.md: layout-finding methods A–F and the D/E resource sweep.

Reads the latest full JSON per tag from noiseless/results/relayout_runs/ (gitignored) and
writes noiseless/results/relayout_analysis_summary.json + figures under
noiseless/results/relayout_figs/.

Methods (all tuned growth L=1→4 + SPSA-Adam, jp, 20 four_sat H × 25 trials, paired seeds):
  A  identity layout, 1604 evals
  B  A's round-0 most-likely string + radius-1 classical polish (9 energy lookups, 0 extra
     circuit evals); scored as success only (B is a classical decode, not a state)
  C  --relayout as committed in 70f1f73 (lr 0.5, random init, last round, xor_vacuum)
  D  relayout fixed: lr 0.05, small init, best round, xor_vacuum
  E  as D with --relayout-target rule
  F1 oracle: tier-rule layout from the TRUE GS, single tuned run (1604)
  F2 oracle: per-H screen-best class (H0–H9 only), single tuned run (1604)
"""

from __future__ import annotations

import glob
import json
import re
import sys
from pathlib import Path

import numpy as np
from scipy import stats

_REPO = Path(__file__).resolve().parents[1]
RUNS = _REPO / "noiseless" / "results" / "relayout_runs"
OUT_JSON = _REPO / "noiseless" / "results" / "relayout_analysis_summary.json"
FIG_DIR = _REPO / "noiseless" / "results" / "relayout_figs"


def load(tag: str) -> list[dict] | None:
    files = sorted(f for f in glob.glob(str(RUNS / f"{tag}_*Z.json")) if not f.endswith("_summary.json"))
    files = [f for f in files if re.fullmatch(rf".*/{re.escape(tag)}_\d{{8}}T\d{{6}}Z\.json", f)]
    if not files:
        return None
    recs = json.load(open(files[-1]))["records"]
    bad = [r for r in recs if not r.get("ok")]
    if bad:
        print(f"WARNING {tag}: {len(bad)} failed records", file=sys.stderr)
    return [r for r in recs if r.get("ok")]


def key(r: dict) -> tuple:
    return (r["ham_file"], int(r["trial"]))


def sign_test(wins: int, losses: int) -> float | None:
    n = wins + losses
    if n == 0:
        return None
    return float(stats.binomtest(wins, n, 0.5).pvalue)


def wilcoxon(d: np.ndarray) -> float | None:
    d = np.asarray(d, dtype=float)
    if np.all(np.abs(d) < 1e-12):
        return None
    try:
        return float(stats.wilcoxon(d, zero_method="wilcox").pvalue)
    except ValueError:
        return None


def method_row(name: str, recs: list[dict], *, success_field=None, p_field="p_gs",
               evals=None) -> dict:
    succ = [bool(success_field(r)) if success_field else bool(r["success"]) for r in recs]
    pg = [float(p_field(r)) if callable(p_field) else float(r[p_field]) for r in recs]
    ev = [float(evals(r)) if callable(evals) else float(r["nfev"]) for r in recs]
    per_h = {}
    for r, s, p in zip(recs, succ, pg):
        per_h.setdefault(r["ham_file"], []).append((s, p))
    worst_h = min(np.mean([s for s, _ in v]) for v in per_h.values())
    return {
        "method": name,
        "n_trials": len(recs),
        "n_h": len(per_h),
        "success_rate": float(np.mean(succ)),
        "mean_p_gs": float(np.mean(pg)),
        "median_p_gs": float(np.median(pg)),
        "frac_p_gs_ge_0.5": float(np.mean(np.asarray(pg) >= 0.5)),
        "worst_h_success": float(worst_h),
        "mean_evals": float(np.mean(ev)),
    }


def paired(name: str, recs: list[dict], ref: dict[tuple, dict], *, p_of=None, s_of=None,
           ref_p=None, ref_s=None) -> dict:
    p_of = p_of or (lambda r: float(r["p_gs"]))
    s_of = s_of or (lambda r: bool(r["success"]))
    ref_p = ref_p or (lambda r: float(r["p_gs"]))
    ref_s = ref_s or (lambda r: bool(r["success"]))
    d, sw, sl = [], 0, 0
    for r in recs:
        k = key(r)
        if k not in ref:
            continue
        d.append(p_of(r) - ref_p(ref[k]))
        a, b = s_of(r), ref_s(ref[k])
        sw += int(a and not b)
        sl += int(b and not a)
    d = np.asarray(d)
    w, l = int(np.sum(d > 1e-12)), int(np.sum(d < -1e-12))
    return {
        "method": name, "n_pairs": int(d.size), "mean_delta_p_gs": float(np.mean(d)) if d.size else None,
        "median_delta_p_gs": float(np.median(d)) if d.size else None,
        "p_gs_win_tie_loss": [w, int(d.size - w - l), l],
        "sign_test_p": sign_test(w, l), "wilcoxon_p": wilcoxon(d),
        "success_gained_lost": [sw, sl], "mcnemar_exact_p": sign_test(sw, sl),
    }


def main() -> int:
    out: dict = {"methods": [], "paired_vs_A": [], "sweep": [], "per_h": {}, "notes": []}
    A = load("rl_A")
    C, D, E = load("rl_C"), load("rl_D"), load("rl_E")
    F1, F2 = load("rl_F1"), load("rl_F2")
    if A is None:
        raise SystemExit("rl_A missing")
    refA = {key(r): r for r in A}
    rows = [method_row("A identity tuned", A)]
    # B from D's round 0 (bit-for-bit = A); fall back to C.
    srcB = D or C
    if srcB:
        r0_p = {key(r): r["rounds"][0]["p_gs"] for r in srcB}
        same = np.mean([abs(r0_p[k] - refA[k]["p_gs"]) < 1e-12 for k in r0_p if k in refA])
        out["notes"].append(f"round-0 of relayout runs == A bit-for-bit on {same:.3f} of trials")
        rows.append(method_row("B A + radius-1 polish (0 extra evals)", srcB,
                               success_field=lambda r: r["rounds"][0]["polished_is_ground"],
                               p_field=lambda r: r["rounds"][0]["p_gs"],
                               evals=lambda r: r["rounds"][0]["nfev"]))
    for name, R in (("C relayout as committed (lr0.5,random,last,xor)", C),
                    ("D relayout fixed (lr0.05,small,best,xor_vacuum)", D),
                    ("E as D, target=rule", E)):
        if R:
            row = method_row(name, R)
            row["stop_fixed_point"] = float(np.mean([r["relayout_stop"] == "fixed_point" for r in R]))
            row["selected_round_hist"] = np.bincount(
                [next((k for k, rr in enumerate(r["rounds"]) if rr.get("selected", k == len(r["rounds"]) - 1)), 0)
                 for r in R], minlength=3).tolist()
            row["final_below_round0"] = float(np.mean([r["p_gs"] < r["rounds"][0]["p_gs"] - 1e-12 for r in R]))
            rows.append(row)
    if F1:
        rows.append(method_row("F1 oracle: rule layout on TRUE GS", F1))
    if F2:
        rows.append(method_row("F2 oracle: screen-best layout (H0-H9)", F2))
        A10 = [r for r in A if r["ham_file"] in {x["ham_file"] for x in F2}]
        rows.append(method_row("A restricted to H0-H9", A10))
        F2x = [r for r in F2 if int(r["trial"]) not in (1, 2, 3)]
        rows.append(method_row("F2 excl. trials 1-3 (screen inits)", F2x))
        A10x = [r for r in A10 if int(r["trial"]) not in (1, 2, 3)]
        rows.append(method_row("A H0-H9 excl. trials 1-3", A10x))
        if F1:
            F1_10 = [r for r in F1 if r["ham_file"] in {x["ham_file"] for x in F2}]
            rows.append(method_row("F1 restricted to H0-H9", F1_10))
    EG = load("rl_EG_g200_e200")
    if EG:
        row = method_row("EG rule + regrow (g200 e200)", EG)
        row["stop_fixed_point"] = float(np.mean([r["relayout_stop"] == "fixed_point" for r in EG]))
        row["final_below_round0"] = float(np.mean([r["p_gs"] < r["rounds"][0]["p_gs"] - 1e-12 for r in EG]))
        rows.append(row)
    out["methods"] = rows
    pv = []
    if srcB:
        pv.append(paired("B vs A (success only)", srcB, refA,
                         p_of=lambda r: r["rounds"][0]["p_gs"],
                         s_of=lambda r: r["rounds"][0]["polished_is_ground"]))
    for name, R in (("C", C), ("D", D), ("E", E), ("EG", EG), ("F1", F1), ("F2", F2)):
        if R:
            pv.append(paired(name, R, refA))
    out["paired_vs_A"] = pv
    # per-H
    for name, R in (("A", A), ("D", D), ("E", E), ("EG", EG), ("F1", F1), ("F2", F2), ("C", C)):
        if not R:
            continue
        ph = {}
        for r in R:
            ph.setdefault(r["ham_file"], []).append(r)
        out["per_h"][name] = {h: {"success": float(np.mean([x["success"] for x in v])),
                                  "mean_p_gs": float(np.mean([x["p_gs"] for x in v])),
                                  "gs": v[0]["ground_bitstring"],
                                  "encoding": v[0].get("encoding")}
                              for h, v in sorted(ph.items())}
    if srcB:
        out["per_h"]["B_success"] = {}
        for r in srcB:
            out["per_h"]["B_success"].setdefault(r["ham_file"], []).append(r["rounds"][0]["polished_is_ground"])
        out["per_h"]["B_success"] = {h: float(np.mean(v)) for h, v in sorted(out["per_h"]["B_success"].items())}
    # sweep
    sw = []
    for meth, base in (("D", D), ("E", E), ("EG", None)):
        for g in (10, 25, 50, 100, 200):
            for e in (10, 25, 50, 100, 200):
                if meth == "EG":
                    R = load(f"rl_EG_g{g}_e{e}")
                else:
                    R = base if (g == 200 and e == 200) else load(f"rl_sweep{meth}_g{g}_e{e}")
                if not R:
                    continue
                row = method_row(f"{meth} g{g} e{e}", R)
                row.update({"method_family": meth, "grow_steps": g, "extra_steps": e,
                            "round0_success": float(np.mean([r["rounds"][0]["success"] for r in R])),
                            "round0_mean_p_gs": float(np.mean([r["rounds"][0]["p_gs"] for r in R])),
                            "round0_evals": float(np.mean([r["rounds"][0]["nfev"] for r in R])),
                            "B_polished_is_ground": float(np.mean([r["rounds"][0]["polished_is_ground"] for r in R])),
                            "raw_candidate_is_ground": float(np.mean([r["rounds"][0]["most_likely_bitstring"] == r["ground_bitstring"] for r in R])),
                            "stop_fixed_point": float(np.mean([r["relayout_stop"] == "fixed_point" for r in R]))})
                pr = paired(row["method"], R, refA)
                row.update({"vs_A_mean_delta_p": pr["mean_delta_p_gs"], "vs_A_wilcoxon_p": pr["wilcoxon_p"],
                            "vs_A_success_gained_lost": pr["success_gained_lost"],
                            "vs_A_mcnemar_p": pr["mcnemar_exact_p"]})
                sw.append(row)
    out["sweep"] = sw
    # growth-only references at other budgets (A) and oracle F1 at other budgets
    refs = []
    for tag, lab in (("rl_A_g250", "A g250"), ("rl_A_g400", "A g400"),
                     ("rl_F1_g50", "F1 g50"), ("rl_F1_g100", "F1 g100")):
        R = load(tag)
        if R:
            row = method_row(lab, R)
            pr = paired(lab, R, refA)
            row.update({"vs_A_mean_delta_p": pr["mean_delta_p_gs"], "vs_A_wilcoxon_p": pr["wilcoxon_p"],
                        "vs_A_success_gained_lost": pr["success_gained_lost"]})
            refs.append(row)
    out["budget_refs"] = refs
    for r in refs:
        print(f"{r['method']:>10} evals={r['mean_evals']:7.0f} succ={r['success_rate']:.3f} p={r['mean_p_gs']:.3f}")

    # jp_local product-gate combinations (same layouts/methods, U = jp_local instead of jp)
    jl_tags = [
        ("A_jl", "rl_A_jl"),
        ("F1_jl", "rl_F1_jl"),
        ("D_jl g25 e25", "rl_D_jl_g25_e25"),
        ("D_jl g50 e50", "rl_D_jl_g50_e50"),
        ("D_jl g50 e100", "rl_D_jl_g50_e100"),
        ("D_jl g200 e200", "rl_D_jl_g200_e200"),
        ("EG_jl g200 e200", "rl_EG_jl_g200_e200"),
    ]
    jl_loaded: dict[str, list] = {}
    jl_rows = []
    for lab, tag in jl_tags:
        R = load(tag)
        if not R:
            continue
        jl_loaded[lab] = R
        row = method_row(lab, R)
        if lab.startswith("D_jl") or lab.startswith("EG_jl"):
            row["stop_fixed_point"] = float(np.mean([r["relayout_stop"] == "fixed_point" for r in R]))
            row["round0_success"] = float(np.mean([r["rounds"][0]["success"] for r in R]))
            row["round0_mean_p_gs"] = float(np.mean([r["rounds"][0]["p_gs"] for r in R]))
            row["B_polished_is_ground"] = float(np.mean([r["rounds"][0]["polished_is_ground"] for r in R]))
        jl_rows.append(row)
        print(f"JL {lab:>16} evals={row['mean_evals']:7.0f} succ={row['success_rate']:.3f} p={row['mean_p_gs']:.3f}")
    out["jp_local_combos"] = {"methods": jl_rows, "paired": []}
    refA_jl = {key(r): r for r in jl_loaded["A_jl"]} if "A_jl" in jl_loaded else {}
    jl_pairs = []
    # A_jl vs A (jp baseline identity layout)
    if "A_jl" in jl_loaded:
        jl_pairs.append(paired("A_jl vs A", jl_loaded["A_jl"], refA))
    # F1_jl vs F1 and vs A_jl
    if "F1_jl" in jl_loaded and F1:
        jl_pairs.append(paired("F1_jl vs F1", jl_loaded["F1_jl"], {key(r): r for r in F1}))
    if "F1_jl" in jl_loaded and refA_jl:
        jl_pairs.append(paired("F1_jl vs A_jl", jl_loaded["F1_jl"], refA_jl))
    # D_jl vs A_jl and vs matching D (jp); EG_jl vs A_jl / EG
    d_pairs = [
        ("D_jl g25 e25", "D g25 e25", load("rl_sweepD_g25_e25")),
        ("D_jl g50 e50", "D g50 e50", load("rl_sweepD_g50_e50")),
        ("D_jl g50 e100", "D g50 e100", load("rl_sweepD_g50_e100")),
        ("D_jl g200 e200", "D g200 e200", D),
    ]
    for jplab, jplab_jp, Rjp in d_pairs:
        if jplab not in jl_loaded:
            continue
        if refA_jl:
            jl_pairs.append(paired(f"{jplab} vs A_jl", jl_loaded[jplab], refA_jl))
        if Rjp:
            jl_pairs.append(paired(f"{jplab} vs {jplab_jp}", jl_loaded[jplab], {key(r): r for r in Rjp}))
    if "EG_jl g200 e200" in jl_loaded and refA_jl:
        jl_pairs.append(paired("EG_jl vs A_jl", jl_loaded["EG_jl g200 e200"], refA_jl))
    if "EG_jl g200 e200" in jl_loaded and EG:
        jl_pairs.append(paired("EG_jl vs EG", jl_loaded["EG_jl g200 e200"], {key(r): r for r in EG}))
    out["jp_local_combos"]["paired"] = jl_pairs
    for p in jl_pairs:
        print(f"JLPAIR {p['method']}: d={p['mean_delta_p_gs']} wil={p['wilcoxon_p']} "
              f"succ+/-={p['success_gained_lost']} n={p['n_pairs']}")

    a_row = rows[0]
    beat = [r for r in sw if r["success_rate"] >= a_row["success_rate"] and r["mean_p_gs"] > a_row["mean_p_gs"]
            and (r["vs_A_wilcoxon_p"] or 1) < 0.05]
    out["cheapest_beating_A"] = sorted(beat, key=lambda r: r["mean_evals"])[:5]
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(json.dumps({"methods": rows, "paired_vs_A": pv}, indent=1))
    for r in sw:
        print(f"{r['method']:>12} evals={r['mean_evals']:7.0f} succ={r['success_rate']:.3f} p={r['mean_p_gs']:.3f} "
              f"r0succ={r['round0_success']:.3f} r0p={r['round0_mean_p_gs']:.3f} B={r['B_polished_is_ground']:.3f} "
              f"raw={r['raw_candidate_is_ground']:.3f} r0ev={r['round0_evals']:.0f}")
    # figure
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        FIG_DIR.mkdir(parents=True, exist_ok=True)
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
        for meth, mk in (("D", "o"), ("E", "s"), ("EG", "D")):
            pts = [r for r in sw if r["method_family"] == meth]
            if not pts:
                continue
            axes[0].scatter([r["mean_evals"] for r in pts], [r["mean_p_gs"] for r in pts], marker=mk, label=f"{meth} (relayout)")
            for r in pts:
                axes[0].annotate(f"{r['grow_steps']}/{r['extra_steps']}", (r["mean_evals"], r["mean_p_gs"]), fontsize=7)
        r0 = {}
        for r in sw:
            if r["method_family"] != "D":
                continue
            r0.setdefault(r["grow_steps"], (r["round0_evals"], r["round0_mean_p_gs"], r["B_polished_is_ground"], r["round0_success"]))
        xs = sorted(r0)
        axes[0].plot([r0[g][0] for g in xs], [r0[g][1] for g in xs], "k--", marker="^", label="A (growth only)")
        for rr in rows:
            if rr["method"].startswith("F1"):
                axes[0].axhline(rr["mean_p_gs"], color="g", ls=":", label="F1 oracle (rule on true GS)")
        for rr in out.get("budget_refs", []):
            if rr["method"].startswith("A"):
                axes[0].scatter([rr["mean_evals"]], [rr["mean_p_gs"]], marker="^", color="k")
        axes[0].set_xscale("log")
        axes[1].set_xscale("log")
        axes[0].set_xlabel("circuit cost evaluations per trial")
        axes[0].set_ylabel("mean p(GS)")
        axes[0].legend(fontsize=7)
        axes[0].set_title("p(GS) vs budget (20 H x 25 trials)")
        axes[1].plot([r0[g][0] for g in xs], [r0[g][3] for g in xs], "k--", marker="^", label="A success (argmax=GS)")
        axes[1].plot([r0[g][0] for g in xs], [r0[g][2] for g in xs], "b-", marker="v", label="B polished guess = GS")
        for meth, mk in (("D", "o"), ("E", "s"), ("EG", "D")):
            pts = [r for r in sw if r["method_family"] == meth]
            if not pts:
                continue
            axes[1].scatter([r["mean_evals"] for r in pts], [r["success_rate"] for r in pts], marker=mk, label=f"{meth} success")
        axes[1].set_xlabel("circuit cost evaluations per trial")
        axes[1].set_ylabel("fraction of trials")
        axes[1].legend(fontsize=7)
        axes[1].set_title("success vs budget")
        fig.tight_layout()
        fig.savefig(FIG_DIR / "relayout_budget.png", dpi=130)
        print("fig", FIG_DIR / "relayout_budget.png")
    except Exception as exc:  # noqa: BLE001
        print("figure failed:", exc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
