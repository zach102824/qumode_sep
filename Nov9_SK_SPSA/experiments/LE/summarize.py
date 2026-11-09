"""Summarize LE milestone runs -> results/LE/summary_M1.json, RESULTS_M1.md, plots."""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[1] / "results" / "LE"
PREREG = json.loads((HERE / "PREREG_LE_M1.json").read_text())
PREREG["arms"].update(json.loads((HERE / "PREREG_LE_M1b.json").read_text())["new_arms"])
ARMS = list(PREREG["arms"])


def load():
    R = []
    for p in sorted((OUT / "runs").glob("N*/*.json")):
        R.append(json.loads(p.read_text()))
    return R


def med(v):
    v = [x for x in v if x is not None and np.isfinite(x)]
    return float(np.median(v)) if v else float("nan")


def main():
    R = load()
    idx = {(r["N"], r["seed"], r["b"], r["L"], r["arm"], r["cap"]): r for r in R}
    Ns = sorted({r["N"] for r in R})
    cells = {}
    for N in Ns:
        for b in PREREG["fields"]:
            for L in PREREG["L"]:
                for cap in PREREG["caps"]:
                    for arm in ARMS:
                        rs = [idx.get((N, s, b, L, arm, cap)) for s in PREREG["seeds"]]
                        rs = [r for r in rs if r]
                        if not rs:
                            continue
                        q = [idx.get((N, r["seed"], b, L, "QAOA", cap)) for r in rs]
                        dF = [r["diag"]["F"] - qq["diag"]["F"] for r, qq in zip(rs, q) if qq]
                        dE = [r["diag"]["E"] - qq["diag"]["E"] for r, qq in zip(rs, q) if qq]
                        cells[f"N{N}_b{b}_L{L}_cap{cap}_{arm}"] = {
                            "N": N, "b": b, "L": L, "cap": cap, "arm": arm, "n": len(rs),
                            "medF": med([r["diag"]["F"] for r in rs]),
                            "frac_F_ge_0.7": float(np.mean([r["diag"]["F"] >= 0.7 for r in rs])),
                            "medF_plus": med([r["F_plus"] for r in rs]),
                            "med_eta": med([r["eta"] for r in rs]),
                            "med_p_leak": med([r["diag"]["p_leak"] for r in rs]),
                            "max_p_leak": float(np.max([r["diag"]["p_leak"] for r in rs])),
                            "med_resid_rel": med([r["resid_rel"] for r in rs]),
                            "med_parity": med([r["diag"]["parity"] for r in rs]),
                            "med_dF_vs_QAOA": med(dF), "wins_F_vs_QAOA": int(sum(x > 0 for x in dF)),
                            "med_dE_vs_QAOA": med(dE), "wins_E_vs_QAOA": int(sum(x < 0 for x in dE)),
                            "med_E_H0_minus_E": med([r["diag"].get("E_H0", np.nan) - r["diag"]["E"] for r in rs]),
                            "failures": int(sum(r["status"] != "ok" for r in rs)),
                            "queries": sorted({r.get("queries") for r in rs if r.get("queries")}),
                        }
    # cutoff convergence over all leaky cap-1024 runs
    cut = defaultdict(lambda: {"dE": 0.0, "dF": 0.0, "dleak": 0.0, "n": 0})
    for r in R:
        if "cutoff" not in r:
            continue
        ref = r["cutoff"]["48"]
        for nf, v in r["cutoff"].items():
            c = cut[f"N{r['N']}_nf{nf}"]
            c["dE"] = max(c["dE"], abs(v["E"] - ref["E"]))
            c["dF"] = max(c["dF"], abs(v["F"] - ref["F"]))
            c["dleak"] = max(c["dleak"], abs(v["p_leak"] - ref["p_leak"]))
            c["n"] += 1
    # Uz vs U0 at same params, and alpha sign patterns
    u0 = defaultdict(list)
    signs = defaultdict(lambda: [0, 0])
    alpha_abs = defaultdict(list)
    for r in R:
        if r.get("cross_U0_same_params") and r["status"] == "ok":
            u0[f"N{r['N']}_b{r['b']}_L{r['L']}_{r['arm']}"].append(abs(r["diag"]["F"] - r["cross_U0_same_params"]["F"]))
        if r.get("params") and r["L"] >= 2 and r["cap"] == 1024:
            a = np.array(r["params"]["alpha"])
            alt = bool(np.all(np.sign(a[1:]) == -np.sign(a[:-1])))
            k = f"N{r['N']}_L{r['L']}_{r['arm']}"
            signs[k][0] += int(alt)
            signs[k][1] += 1
        if r.get("params") and r["cap"] == 1024:
            alpha_abs[f"N{r['N']}_L{r['L']}_{r['arm']}"].append(float(np.max(np.abs(r["params"]["alpha"]))))
    summ = {"prereg_sha256_seen": sorted({r["prereg_sha256"] for r in R}), "n_runs": len(R),
            "failures": int(sum(r["status"] != "ok" for r in R)), "cells": cells,
            "cutoff_max_dev_vs_nf48": dict(cut),
            "Uz_minus_U0_same_params_medabsdF": {k: med(v) for k, v in u0.items()},
            "alpha_alternating_fraction": {k: f"{v[0]}/{v[1]}" for k, v in signs.items()},
            "median_max_abs_alpha": {k: med(v) for k, v in alpha_abs.items()},
            "wall_s_total": float(sum(r["wall_s"] for r in R))}
    (OUT / "summary_M1.json").write_text(json.dumps(summ, indent=1))
    write_md(summ, Ns)
    plots(R, cells, Ns)
    print(f"runs {len(R)} failures {summ['failures']}")


def write_md(S, Ns):
    C = S["cells"]
    L_ = []
    L_.append("# LE milestone 1 results (approved Nov9 simulation benchmark; 8 dev seeds; descriptive only)\n")
    L_.append("Off-code physics = approved simulation definitions (Zach, Nov 9 18:41 Shanghai): D2 SNAP phase 0 above level 7, D3 Uz (U0 = sensitivity arm), D4 per-term H_ext, D5 ideal logical readout. These are identical to the definitions frozen in PREREG_LE_M1 (written before the approval as 'provisional'). NOT a hardware-feasibility result, NOT validation/final. Unconditioned F, no postselection. Exact-energy SPSA, no shots. 8 samples: no target claim. B_alt = strict-logical diagnostic, not a true D. LE_frame = post-M1 init variant (PREREG_LE_M1b).\n")
    L_.append(f"Runs {S['n_runs']}, failures {S['failures']}, prereg sha {', '.join(s[:12] for s in S['prereg_sha256_seen'])}.\n")
    for cap in PREREG["caps"]:
        L_.append(f"\n## cap {cap} queries\n")
        L_.append("| N | b | L | arm | medF | F>=.7 | medF+ | med eta | med p_leak | max p_leak | med dF vs QAOA | F wins | med dE vs QAOA | E wins | med parity |")
        L_.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for N in Ns:
            for b in PREREG["fields"]:
                for Lv in PREREG["L"]:
                    for arm in ARMS:
                        c = C.get(f"N{N}_b{b}_L{Lv}_cap{cap}_{arm}")
                        if not c:
                            continue
                        L_.append(f"| {N} | {b} | {Lv} | {arm} | {c['medF']:.4f} | {c['frac_F_ge_0.7']:.2f} | {c['medF_plus']:.4f} | {c['med_eta']:.3f} | {c['med_p_leak']:.2e} | {c['max_p_leak']:.2e} | {c['med_dF_vs_QAOA']:+.4f} | {c['wins_F_vs_QAOA']}/{c['n']} | {c['med_dE_vs_QAOA']:+.4f} | {c['wins_E_vs_QAOA']}/{c['n']} | {c['med_parity']:+.3f} |")
    L_.append("\n## cutoff convergence (max over leaky cap-1024 selected points, vs NF=48)\n")
    L_.append("| key | max dE | max dF | max dp_leak | n |")
    L_.append("|---|---|---|---|---|")
    for k, v in sorted(S["cutoff_max_dev_vs_nf48"].items()):
        L_.append(f"| {k} | {v['dE']:.2e} | {v['dF']:.2e} | {v['dleak']:.2e} | {v['n']} |")
    L_.append("\n## Uz vs U0 at identical parameters (median |dF|)\n")
    for k, v in sorted(S["Uz_minus_U0_same_params_medabsdF"].items()):
        L_.append(f"- {k}: {v:.2e}")
    L_.append("\n## optimized alpha sign pattern (fraction alternating, cap 1024)\n")
    for k, v in sorted(S["alpha_alternating_fraction"].items()):
        L_.append(f"- {k}: {v}  (median max|alpha| {S['median_max_abs_alpha'].get(k, float('nan')):.3f})")
    (OUT / "RESULTS_M1.md").write_text("\n".join(L_) + "\n")


def plots(R, C, Ns):
    colors = {"QAOA": "k", "LE_alt": "tab:blue", "LE_same": "tab:cyan", "LE_alt_U0": "tab:purple",
              "LE_alt_s3off": "tab:orange", "B_alt": "tab:red", "LE_frame": "tab:green"}
    fig, axes = plt.subplots(len(Ns), 2, figsize=(11, 4 * len(Ns)), squeeze=False)
    for i, N in enumerate(Ns):
        for j, b in enumerate(PREREG["fields"]):
            ax = axes[i, j]
            for arm in ARMS:
                for cap, ls in ((1024, "-"), (128, ":")):
                    ys = [C.get(f"N{N}_b{b}_L{L}_cap{cap}_{arm}", {}).get("medF", np.nan) for L in PREREG["L"]]
                    ax.plot(PREREG["L"], ys, ls, marker="o", color=colors[arm], label=f"{arm} cap{cap}")
            fp = C.get(f"N{N}_b{b}_L1_cap1024_QAOA", {}).get("medF_plus", np.nan)
            ax.axhline(fp, color="gray", ls="--", label="|+> baseline")
            ax.set_title(f"N={N} b={b} median F (8 dev seeds)")
            ax.set_xlabel("L")
            ax.set_ylabel("median F (unconditioned)")
    axes[0, 0].legend(fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(OUT / "medF_vs_L.png", dpi=110)
    plt.close(fig)
    # leakage vs NF
    fig, ax = plt.subplots(figsize=(6, 4))
    for r in R:
        if "cutoff" in r and r["arm"] == "LE_alt" and r["L"] == 3:
            nfs = sorted(int(k) for k in r["cutoff"])
            ax.semilogy(nfs, [max(abs(r["cutoff"][str(n)]["p_leak"] - r["cutoff"]["48"]["p_leak"]), 1e-18) for n in nfs],
                        color={6: "tab:green", 9: "tab:blue", 12: "tab:red"}[r["N"]], alpha=0.4)
    ax.set_xlabel("NF_sim")
    ax.set_ylabel("|p_leak(NF) - p_leak(48)|")
    ax.set_title("cutoff convergence, LE_alt L=3 selected points (green N6, blue N9, red N12)")
    fig.tight_layout()
    fig.savefig(OUT / "leak_vs_NF.png", dpi=110)
    plt.close(fig)
    # paired F scatter
    fig, ax = plt.subplots(figsize=(5, 5))
    for r in R:
        if r["arm"] in ("LE_alt", "B_alt", "LE_frame") and r["cap"] == 1024:
            q = [x for x in R if x["arm"] == "QAOA" and x["cap"] == 1024 and (x["N"], x["seed"], x["b"], x["L"]) == (r["N"], r["seed"], r["b"], r["L"])]
            if q:
                ax.scatter(q[0]["diag"]["F"], r["diag"]["F"], s=10, color=colors[r["arm"]], alpha=0.6,
                           marker={6: "^", 9: "o", 12: "s"}[r["N"]])
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("QAOA p=L F")
    ax.set_ylabel("Nov9th F (blue LE_alt, red strict B, green LE_frame)")
    ax.set_title("paired per instance, cap 1024")
    fig.tight_layout()
    fig.savefig(OUT / "paired_F_scatter.png", dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
