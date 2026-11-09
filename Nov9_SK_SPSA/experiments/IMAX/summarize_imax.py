"""Summarize IMAX M1 -> results/IMAX/summary_M1.json, RESULTS_M1.md, plots."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[1] / "results" / "IMAX"
PREREG = json.loads((HERE / "PREREG_IMAX_M1.json").read_text())
ARMS = list(PREREG["arms"])
COL = {"QAOA": "k", "IM": "tab:red", "IM_U0": "tab:orange", "IM_B": "tab:purple", "RE": "tab:blue"}


def load():
    return [json.loads(p.read_text()) for p in sorted((OUT / "runs").glob("N*/*.json"))]


def boot_ci(d, n=10000):
    d = np.asarray(d, float)
    if len(d) == 0:
        return [float("nan")] * 2
    r = np.random.default_rng(0)
    m = d[r.integers(0, len(d), size=(n, len(d)))].mean(axis=1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def main():
    R = load()
    idx = {(r["N"], r["seed"], r["L"], r["arm"], r["cap"]): r for r in R}
    Ns = sorted({r["N"] for r in R})
    cells = {}
    for N in Ns:
        for L in PREREG["L"]:
            for cap in PREREG["caps"]:
                for arm in ARMS:
                    rs = [idx[(N, s, L, arm, cap)] for s in PREREG["seeds"] if (N, s, L, arm, cap) in idx]
                    if not rs:
                        continue
                    F = np.array([r["diag"]["F"] for r in rs])
                    c = {"N": N, "L": L, "cap": cap, "arm": arm, "n": len(rs),
                         "medF": float(np.median(F)), "meanF": float(F.mean()), "minF": float(F.min()),
                         "q25F": float(np.percentile(F, 25)), "frac_F_ge_0.5": float(np.mean(F >= 0.5)),
                         "medF_plus": float(np.median([r["F_plus"] for r in rs])),
                         "med_eta": float(np.median([r["eta"] for r in rs])),
                         "med_closure_E": float(np.median([r["closure_E"] for r in rs])),
                         "med_resid_rel": float(np.median([r["resid_rel"] for r in rs])),
                         "med_p_leak": float(np.median([r["diag"]["p_leak"] for r in rs])),
                         "max_p_leak": float(np.max([r["diag"]["p_leak"] for r in rs])),
                         "failures": int(sum(r["status"] != "ok" for r in rs)),
                         "queries": sorted({r.get("queries") for r in rs}),
                         "uncertified": int(sum(1 for r in rs if r.get("cutoff_certified") is False))}
                    for ref in ("QAOA", "RE"):
                        if arm == ref:
                            continue
                        pr = [(r, idx.get((N, r["seed"], L, ref, cap))) for r in rs]
                        pr = [(a, b) for a, b in pr if b]
                        dF = [a["diag"]["F"] - b["diag"]["F"] for a, b in pr]
                        dE = [a["diag"]["E"] - b["diag"]["E"] for a, b in pr]
                        c[f"vs_{ref}"] = {"med_dF": float(np.median(dF)) if dF else None, "mean_dF": float(np.mean(dF)) if dF else None,
                                          "ci95_mean_dF": boot_ci(dF), "wins_F": int(sum(x > 0 for x in dF)),
                                          "med_dE": float(np.median(dE)) if dE else None, "ci95_mean_dE": boot_ci(dE),
                                          "wins_E": int(sum(x < 0 for x in dE)), "n": len(pr)}
                    if arm == "IM" and cap == max(PREREG["caps"]):
                        u0 = [abs(r["diag"]["F"] - r["cross_U0_same_params"]["F"]) for r in rs if r.get("cross_U0_same_params")]
                        c["med_absdF_U0_same_params"] = float(np.median(u0)) if u0 else None
                    cells[f"N{N}_L{L}_cap{cap}_{arm}"] = c
    cut = {}
    for r in R:
        if "cutoff" not in r:
            continue
        ref = r["cutoff"]["48"]
        for nf, v in r["cutoff"].items():
            k = f"N{r['N']}_{r['arm']}_nf{nf}"
            c = cut.setdefault(k, {"dE": 0.0, "dF": 0.0, "dleak": 0.0, "max_abs_norm_minus_1": 0.0, "n": 0})
            c["dE"] = max(c["dE"], abs(v["E"] - ref["E"]))
            c["dF"] = max(c["dF"], abs(v["F"] - ref["F"]))
            c["dleak"] = max(c["dleak"], abs(v["p_leak"] - ref["p_leak"]))
            c["max_abs_norm_minus_1"] = max(c["max_abs_norm_minus_1"], abs(v["norm"] - 1))
            c["n"] += 1
    inst = {}
    for p in sorted((OUT / "instances").glob("*.json")):
        m = json.loads(p.read_text())
        inst[p.stem] = {k: m[k] for k in ("Bstar", "S_A", "gap_E1_E0", "gap_even", "gap_odd", "E1_parity", "F_plus", "grid_local_maxima", "valid")}
    S = {"prereg_sha256_seen": sorted({r["prereg_sha256"] for r in R}), "n_runs": len(R),
         "failures": int(sum(r["status"] != "ok" for r in R)),
         "uncertified": int(sum(1 for r in R if r.get("cutoff_certified") is False)),
         "cells": cells, "cutoff_max_dev_vs_nf48": cut, "instances": inst,
         "wall_s_total": float(sum(r["wall_s"] for r in R))}
    (OUT / "summary_M1.json").write_text(json.dumps(S, indent=1))
    md(S, Ns)
    plots(R, cells, Ns)
    print(f"runs {len(R)} failures {S['failures']} uncertified {S['uncertified']}")


def md(S, Ns):
    C = S["cells"]
    o = ["# IMAX milestone 1 (B* fields; 8 dev seeds; descriptive only)\n",
         "User-approved fixed imaginary-axis true displacement D(i alpha) vs the real-axis control vs QAOA p=L, at the per-instance maximum-GS-entropy field B* (A = sites 0..floor(N/2)-1). Approved D2-D5 simulation definitions; NOT a hardware result; unconditioned F, no postselection; exact-energy SPSA, all queries counted. IM_B = strict-logical diagnostic, NOT true D. 8 samples: no target claim; negative curvature at |+> was a prediction, not proof.\n",
         f"Runs {S['n_runs']}, failures {S['failures']}, uncertified cutoff {S['uncertified']}, prereg {', '.join(s[:12] for s in S['prereg_sha256_seen'])}.\n"]
    for cap in PREREG["caps"]:
        o.append(f"\n## cap {cap}\n")
        o.append("| N | L | arm | medF | meanF | minF | q25F | F>=.5 | medF+ | med eta | med E-closure | med p_leak | dF vs QAOA mean [95% CI] | F wins | dE vs QAOA med | E wins | dF vs RE mean [95% CI] | wins vs RE |")
        o.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for N in Ns:
            for L in PREREG["L"]:
                for arm in ARMS:
                    c = C.get(f"N{N}_L{L}_cap{cap}_{arm}")
                    if not c:
                        continue
                    q = c.get("vs_QAOA")
                    rr = c.get("vs_RE")
                    qs = f"{q['mean_dF']:+.3f} [{q['ci95_mean_dF'][0]:+.3f},{q['ci95_mean_dF'][1]:+.3f}] | {q['wins_F']}/{q['n']} | {q['med_dE']:+.3f} | {q['wins_E']}/{q['n']}" if q else "- | - | - | -"
                    rs = f"{rr['mean_dF']:+.3f} [{rr['ci95_mean_dF'][0]:+.3f},{rr['ci95_mean_dF'][1]:+.3f}] | {rr['wins_F']}/{rr['n']}" if rr else "- | -"
                    o.append(f"| {N} | {L} | {arm} | {c['medF']:.4f} | {c['meanF']:.4f} | {c['minF']:.4f} | {c['q25F']:.4f} | {c['frac_F_ge_0.5']:.2f} | {c['medF_plus']:.4f} | {c['med_eta']:+.3f} | {c['med_closure_E']:+.3f} | {c['med_p_leak']:.1e} | {qs} | {rs} |")
    o.append("\n## IM vs U0 at identical parameters (median |dF|, cap 1024)\n")
    for k, c in sorted(C.items()):
        if c.get("med_absdF_U0_same_params") is not None:
            o.append(f"- {k}: {c['med_absdF_U0_same_params']:.2e}")
    o.append("\n## cutoff certification (max over selected points vs NF=48)\n")
    o.append("| key | max dE | max dF | max dp_leak | max abs(norm-1) | n |")
    o.append("|---|---|---|---|---|---|")
    for k, v in sorted(S["cutoff_max_dev_vs_nf48"].items()):
        o.append(f"| {k} | {v['dE']:.2e} | {v['dF']:.2e} | {v['dleak']:.2e} | {v['max_abs_norm_minus_1']:.2e} | {v['n']} |")
    o.append("\n## instances (B*, checks)\n")
    o.append("| instance | B* | S_A | E1-E0 | E1 parity | even gap | F+ | grid maxima | valid |")
    o.append("|---|---|---|---|---|---|---|---|---|")
    for k, m in S["instances"].items():
        o.append(f"| {k} | {m['Bstar']:.6f} | {m['S_A']:.4f} | {m['gap_E1_E0']:.5f} | {m['E1_parity']} | {m['gap_even']:.4f} | {m['F_plus']:.4f} | {m['grid_local_maxima']} | {m['valid']} |")
    (OUT / "RESULTS_M1.md").write_text("\n".join(o) + "\n")


def plots(R, C, Ns):
    fig, axes = plt.subplots(1, len(Ns), figsize=(5 * len(Ns), 4), squeeze=False)
    for i, N in enumerate(Ns):
        ax = axes[0, i]
        for arm in ARMS:
            for cap, ls in ((1024, "-"), (128, ":")):
                ys = [C.get(f"N{N}_L{L}_cap{cap}_{arm}", {}).get("medF", np.nan) for L in PREREG["L"]]
                ax.plot(PREREG["L"], ys, ls, marker="o", color=COL[arm], label=f"{arm} cap{cap}")
        ax.axhline(C.get(f"N{N}_L1_cap1024_QAOA", {}).get("medF_plus", np.nan), color="gray", ls="--", label="|+>")
        ax.set_title(f"N={N}, B*: median F (8 dev seeds)")
        ax.set_xlabel("L")
        ax.set_ylabel("median unconditioned F")
    axes[0, 0].legend(fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(OUT / "medF_vs_L.png", dpi=110)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(5, 5))
    for r in R:
        if r["arm"] in ("IM", "RE") and r["cap"] == 1024:
            q = [x for x in R if x["arm"] == "QAOA" and x["cap"] == 1024 and (x["N"], x["seed"], x["L"]) == (r["N"], r["seed"], r["L"])]
            if q:
                ax.scatter(q[0]["diag"]["F"], r["diag"]["F"], s=12, color=COL[r["arm"]], alpha=0.7,
                           marker={6: "^", 9: "o", 12: "s"}[r["N"]])
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("QAOA p=L F")
    ax.set_ylabel("Nov9 F (red IM = D(i alpha), blue RE = D(alpha))")
    ax.set_title("paired per instance, cap 1024 (^N6 oN9 sN12)")
    fig.tight_layout()
    fig.savefig(OUT / "paired_F_scatter.png", dpi=110)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 4))
    for r in R:
        if "cutoff" in r and r["arm"] == "IM" and r["cap"] == 1024:
            nfs = sorted(int(k) for k in r["cutoff"])
            ax.semilogy(nfs, [max(abs(r["cutoff"][str(n)]["p_leak"] - r["cutoff"]["48"]["p_leak"]), 1e-18) for n in nfs],
                        color={6: "tab:green", 9: "tab:blue", 12: "tab:red"}[r["N"]], alpha=0.4)
    ax.set_xlabel("NF_sim")
    ax.set_ylabel("|p_leak(NF) - p_leak(48)|")
    ax.set_title("IM selected points, cap 1024 (green N6, blue N9, red N12)")
    fig.tight_layout()
    fig.savefig(OUT / "leak_vs_NF.png", dpi=110)
    plt.close(fig)


if __name__ == "__main__":
    main()
