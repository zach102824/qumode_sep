"""Statistics over independent disorder instances, reliability across repeats, replay verification."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binomtest, wilcoxon

HERE = Path(__file__).resolve().parent
P = json.loads((HERE / "PREREG.json").read_text())
FAMS = ("KM", "K", "nodata", "qaoa", "qaoa_heavy")
R = P["repeats"]
CR = P["claim"]


def load():
    ev, rp = {}, {}
    for f in (HERE / "eval").glob("n*_k*.json"):
        d = json.loads(f.read_text()); ev[(d["N"], d["k"])] = d
    for f in (HERE / "replay").glob("n*_k*.json"):
        d = json.loads(f.read_text()); rp[(d["N"], d["k"])] = d
    return ev, rp


def complete(d):
    return len(d["statuses"]) == R and all(v == "complete" for v in d["statuses"].values())


def cells_of(d, mode):
    """mode: 'best' (lowest E over repeats) or an int repeat."""
    out = {}
    for c in d["cells"]:
        if mode != "best" and c["r"] != mode:
            continue
        key = (c["family"], c["L"])
        if key not in out or c["E"] < out[key]["E"]:
            out[key] = c
    return out


def outcome(d, a, q):
    FK, FQ = a["F_sub"], q["F_sub"]
    cl = 1 - CR["clamp"]
    rho = float(np.log((1 - min(FQ, cl)) / (1 - min(FK, cl))))
    dF = FK - FQ
    joint = a["E"] <= q["E"] + 1e-6
    if not d["even_ground_in_window"]:
        res = "tie_unreachable"
    elif max(FK, FQ) < CR["floor"]:
        res = "tie_both_failed"
    elif rho > np.log(CR["ratio"]) and dF > CR["dF_min"]:
        res = "win_joint" if joint else "win_fidelity_only"
    elif rho < -np.log(CR["ratio"]) and dF < -CR["dF_min"]:
        res = "loss_joint" if q["E"] <= a["E"] + 1e-6 else "loss_fidelity_only"
    else:
        res = "tie"
    return {"rho": rho, "dF": dF, "res": res, "dE": a["E"] - q["E"]}


def boot(v, seed=20261125):
    v = np.asarray(v, float)
    if v.size == 0:
        return (np.nan, np.nan)
    g = np.random.default_rng(seed)
    meds = np.median(v[g.integers(0, v.size, size=(CR["bootstrap"], v.size))], axis=1)
    return (float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5)))


def paired(ev, keys, L, comp, mode, fam="KM"):
    rows = []
    for key in keys:
        d = ev[key]
        C = cells_of(d, mode)
        if (fam, L) in C and (comp, L) in C:
            rows.append((key, outcome(d, C[(fam, L)], C[(comp, L)])))
    w = sum(1 for _, o in rows if o["res"].startswith("win"))
    lo = sum(1 for _, o in rows if o["res"].startswith("loss"))
    p_sign = binomtest(w, w + lo, 0.5, alternative="greater").pvalue if w + lo else 1.0
    rhos = [o["rho"] for _, o in rows if o["res"] != "tie_unreachable"]
    dFs = [o["dF"] for _, o in rows]
    try:
        p_w = float(wilcoxon(rhos, alternative="greater").pvalue) if len(rhos) > 0 and np.any(np.asarray(rhos) != 0) else 1.0
    except ValueError:
        p_w = 1.0
    return {"n": len(rows), "wins": w, "losses": lo,
            "win_joint": sum(1 for _, o in rows if o["res"] == "win_joint"),
            "win_fid_only": sum(1 for _, o in rows if o["res"] == "win_fidelity_only"),
            "loss_joint": sum(1 for _, o in rows if o["res"] == "loss_joint"),
            "loss_fid_only": sum(1 for _, o in rows if o["res"] == "loss_fidelity_only"),
            "tie": sum(1 for _, o in rows if o["res"] == "tie"),
            "tie_floor": sum(1 for _, o in rows if o["res"] == "tie_both_failed"),
            "tie_unreach": sum(1 for _, o in rows if o["res"] == "tie_unreachable"),
            "p_sign": float(p_sign), "p_wilcoxon": p_w,
            "median_rho": float(np.median(rhos)) if rhos else np.nan, "rho_ci": boot(rhos),
            "median_dF": float(np.median(dFs)) if dFs else np.nan, "dF_ci": boot(dFs),
            "mean_dF": float(np.mean(dFs)) if dFs else np.nan, "rows": rows}


def holm(ps):
    order = np.argsort(ps)
    adj = np.zeros(len(ps))
    run = 0.0
    for i, idx in enumerate(order):
        run = max(run, (len(ps) - i) * ps[idx])
        adj[idx] = min(1.0, run)
    return adj


def ab_pass(ev, keysN, comp, mode):
    st = {N: paired(ev, keysN[N], 6, comp, mode) for N in P["Ns"]}
    adj = holm([st[N]["p_sign"] for N in P["Ns"]])
    return {N: (bool(adj[i] < 0.05 and st[N]["rho_ci"][0] > 0), float(adj[i]), st[N]) for i, N in enumerate(P["Ns"])}


def fmt_ci(c):
    return f"[{c[0]:+.3f}, {c[1]:+.3f}]"


def main():
    ev, rp = load()
    out = ["# Spin glass round 3: robustness over fresh disorder instances", ""]
    keys_all = {N: sorted([k for (n, k) in ev if n == N]) for N in P["Ns"]}
    prim = {N: [(N, k) for k in P["k_primary"] if (N, k) in ev and complete(ev[(N, k)])] for N in P["Ns"]}
    sec = {N: [(N, k) for k in P["k_secondary"] if (N, k) in ev and complete(ev[(N, k)])] for N in P["Ns"]}
    incomplete = [(n, k, ev[(n, k)]["statuses"]) for (n, k) in sorted(ev) if not complete(ev[(n, k)])]
    missing = [(N, k) for N in P["Ns"] for k in P["k_primary"] + P["k_secondary"] if (N, k) not in ev]
    out.append("## Sample")
    out.append("")
    out.append("| N | fresh instances complete (k=4..27) | secondary complete (k=0..3) | repeats each |")
    out.append("|---|---|---|---|")
    for N in P["Ns"]:
        out.append(f"| {N} | {len(prim[N])}/24 | {len(sec[N])}/4 | {R} |")
    out.append("")
    out.append(f"Incomplete instances (excluded from statistics, listed): {incomplete}. Never started: {missing}.")
    out.append("")
    # replay gate
    gate = {}
    out.append("## Independent replay")
    out.append("")
    out.append("| N | instances | states | max abs dE | max abs dF_sub | max abs dE0 | n_sub mismatches | max norm err | max gs residual | pass |")
    out.append("|---|---|---|---|---|---|---|---|---|---|")
    tolsens = []
    for N in P["Ns"]:
        ks = [key for key in rp if key[0] == N]
        cs = [c for key in ks for c in rp[key]["cells"]]
        if not cs:
            gate[N] = False
            out.append(f"| {N} | 0 | 0 | | | | | | | False |")
            continue
        mdE = max(abs(c["dE"]) / max(1, abs(c["E_replay"])) for c in cs)
        mdF = max(abs(c["dF_sub"]) for c in cs)
        mdE0 = max(abs(rp[key]["dE0"]) for key in ks)
        mism = sum(1 for key in ks if rp[key]["n_sub_replay"] != rp[key]["n_sub_main"])
        mn = max(c["norm_err"] for c in cs)
        gr = max(rp[key]["gs_residual"] for key in ks)
        t = P["replay_tol"]
        ok = mdE <= t["E_rel"] and mdF <= t["F_sub"] and mdE0 <= t["E0"] and mism == 0 and mn <= t["norm"]
        gate[N] = bool(ok)
        out.append(f"| {N} | {len(ks)} | {len(cs)} | {mdE:.1e} | {mdF:.1e} | {mdE0:.1e} | {mism} | {mn:.1e} | {gr:.1e} | {ok} |")
        for key in ks:
            for c in rp[key]["cells"]:
                if c["family"] in ("KM", "qaoa") and abs(c["F_sub_1e-4"] - c["F_sub_1e-8"]) > 1e-3:
                    tolsens.append((key[0], key[1], c["family"], c["L"], c["r"], round(c["F_sub_1e-8"], 4), round(c["F_sub_replay"], 4), round(c["F_sub_1e-4"], 4)))
    out.append("")
    out.append("Relative energy error is |E_replay - E_logged| / max(1, |E|). The replay rebuilds J from the seed, the K and M "
               "features from their definitions, all operators as sparse Pauli strings, evolves with expm_multiply, and "
               "takes the ground space from eigsh. It imports no runner code.")
    out.append("")
    tsk = sorted({(a, b) for a, b, *_ in tolsens})
    out.append(f"KM/QAOA selected states whose F_sub differs by more than 1e-3 between window tolerances 1e-8 and 1e-4: "
               f"{len(tolsens)} states on instances {tsk}.")
    out.append("")
    # fidelity tables
    out.append("## F_sub of the energy-selected state (best over repeats), fresh instances")
    out.append("")
    for N in P["Ns"]:
        out.append(f"### N={N} (n={len(prim[N])} instances)")
        out.append("")
        out.append("| L | " + " | ".join(f"{f} mean / median" for f in FAMS) + " |")
        out.append("|---|" + "---|" * len(FAMS))
        for L in range(1, 7):
            vals = []
            for f in FAMS:
                xs = [cells_of(ev[key], "best")[(f, L)]["F_sub"] for key in prim[N]]
                vals.append(f"{np.mean(xs):.3f} / {np.median(xs):.3f}" if xs else "")
            out.append(f"| {L} | " + " | ".join(vals) + " |")
        out.append("")
    # paired
    out.append("## Paired KM vs QAOA over instances (best over repeats, fresh set)")
    out.append("")
    out.append("Win: log[(1-F_Q)/(1-F_KM)] > log 1.1 and dF > 0.005. Joint win also has E_KM <= E_Q + 1e-6. "
               "Ties: inside the band, both F < 0.02, or the 1e-6 window has no even-sector state.")
    out.append("")
    out.append("| N | L | n | wins (joint / F-only) | losses (joint / F-only) | ties (band / floor / unreachable) | median dF [95% CI] | mean dF | median rho [95% CI] | sign p | Wilcoxon p |")
    out.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for comp in ("qaoa", "qaoa_heavy"):
        if comp == "qaoa_heavy":
            out.append("| | | | | | | | | | | |")
            out.append("| **vs QAOA-heavy** | | | | | | | | | | |")
        for N in P["Ns"]:
            for L in range(1, 7):
                if comp == "qaoa_heavy" and L < 5:
                    continue
                s = paired(ev, prim[N], L, comp, "best")
                out.append(f"| {N} | {L} | {s['n']} | {s['wins']} ({s['win_joint']} / {s['win_fid_only']}) | "
                           f"{s['losses']} ({s['loss_joint']} / {s['loss_fid_only']}) | {s['tie']} / {s['tie_floor']} / {s['tie_unreach']} | "
                           f"{s['median_dF']:+.3f} {fmt_ci(s['dF_ci'])} | {s['mean_dF']:+.3f} | {s['median_rho']:+.3f} {fmt_ci(s['rho_ci'])} | "
                           f"{s['p_sign']:.2g} | {s['p_wilcoxon']:.2g} |")
    out.append("")
    # claim
    out.append("## Frozen claim criteria")
    out.append("")
    main_ab = ab_pass(ev, prim, "qaoa", "best")
    heavy_ab = ab_pass(ev, prim, "qaoa_heavy", "best")
    rep_ab = [ab_pass(ev, prim, "qaoa", r) for r in range(R)]
    verdict = {}
    for N in P["Ns"]:
        a_b, padj, st = main_ab[N]
        s5 = paired(ev, prim[N], 5, "qaoa", "best")
        c_ok = s5["p_sign"] < 0.05 and s5["rho_ci"][0] > 0
        reps = sum(1 for r in range(R) if rep_ab[r][N][0])
        d_ok = reps >= 2 and heavy_ab[N][0]
        ok = bool(a_b and c_ok and d_ok and gate[N] and len(prim[N]) > 0)
        verdict[N] = ok
        out.append(f"- N={N}: (a) Holm-adjusted sign p at L6 = {padj:.3g}; (b) median rho CI lower = {st['rho_ci'][0]:+.3f}; "
                   f"(a)&(b) {'pass' if a_b else 'fail'}. (c) L5 sign p = {s5['p_sign']:.3g}, rho CI lower = {s5['rho_ci'][0]:+.3f}: "
                   f"{'pass' if c_ok else 'fail'}. (d) per-repeat (a)&(b) pass in {reps}/{R}; vs QAOA-heavy "
                   f"{'pass' if heavy_ab[N][0] else 'fail'} (Holm p {heavy_ab[N][1]:.3g}). Replay gate {'pass' if gate[N] else 'fail'}. "
                   f"**{'generalizes at this N' if ok else 'not established at this N'}**")
    n_ok = sum(verdict.values())
    overall = "generalizes" if n_ok == len(P["Ns"]) else ("partial" if n_ok else "not supported")
    out.append("")
    out.append(f"Overall verdict under the frozen rule: **{overall}** ({[N for N in P['Ns'] if verdict[N]]} pass).")
    out.append("")
    # per-instance L6 and L5
    for title, keyset in (("fresh", prim), ("secondary k=0..3", sec)):
        out.append(f"## Per-instance F_sub at L=6 and L=5 ({title}), best over repeats")
        out.append("")
        out.append("| N | k | L | KM | K | nodata | QAOA | QAOA-heavy | dF KM-QAOA | E_KM - E_QAOA | outcome | KM F per repeat | QAOA F per repeat |")
        out.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for N in P["Ns"]:
            for key in keyset[N]:
                d = ev[key]
                C = cells_of(d, "best")
                for L in (6, 5):
                    o = outcome(d, C[("KM", L)], C[("qaoa", L)])
                    per = {f: "/".join(f"{cells_of(d, r)[(f, L)]['F_sub']:.3f}" for r in range(R)) for f in ("KM", "qaoa")}
                    out.append(f"| {N} | {key[1]} | {L} | " + " | ".join(f"{C[(f, L)]['F_sub']:.3f}" for f in FAMS)
                               + f" | {o['dF']:+.3f} | {o['dE']:+.4f} | {o['res']} | {per['KM']} | {per['qaoa']} |")
        out.append("")
    # secondary summary vs R2
    out.append("## Secondary instances k=0..3: R3 best-over-repeats vs R2 at L=6 (F_sub)")
    out.append("")
    r2 = {(c["N"], c["k"], c["family"]): c["F_sub"] for c in json.loads((HERE.parent / "spin_glass_r2" / "selected.json").read_text()) if c["L"] == 6}
    out.append("| N | k | KM R2 | KM R3 | QAOA R2 | QAOA R3 |")
    out.append("|---|---|---|---|---|---|")
    for N in P["Ns"]:
        for key in sec[N]:
            C = cells_of(ev[key], "best")
            out.append(f"| {N} | {key[1]} | {r2.get((N, key[1], 'KM'), float('nan')):.3f} | {C[('KM', 6)]['F_sub']:.3f} | "
                       f"{r2.get((N, key[1], 'qaoa'), float('nan')):.3f} | {C[('qaoa', 6)]['F_sub']:.3f} |")
    out.append("")
    # reliability
    out.append("## Optimizer reliability across the three repeats (fresh + secondary, complete instances)")
    out.append("")
    out.append("| N | family | L | repeats at best E (1e-6) | mean across-repeat F SD | max F range | instances with E spread > 1e-4 | E within 1e-6 but F differs > 0.01 |")
    out.append("|---|---|---|---|---|---|---|---|")
    allk = {N: prim[N] + sec[N] for N in P["Ns"]}
    for N in P["Ns"]:
        for f in ("KM", "qaoa", "qaoa_heavy"):
            for L in (1, 3, 5, 6):
                at, sds, rng_, spread, amb = 0, [], 0.0, 0, 0
                for key in allk[N]:
                    reps = [cells_of(ev[key], r)[(f, L)] for r in range(R)]
                    Es = np.array([c["E"] for c in reps]); Fs = np.array([c["F_sub"] for c in reps])
                    at += int(np.sum(Es - Es.min() <= 1e-6))
                    sds.append(float(np.std(Fs))); rng_ = max(rng_, float(Fs.max() - Fs.min()))
                    spread += int(Es.max() - Es.min() > 1e-4)
                    for i in range(R):
                        for j in range(i + 1, R):
                            if abs(Es[i] - Es[j]) <= 1e-6 and abs(Fs[i] - Fs[j]) > 0.01:
                                amb += 1
                n = len(allk[N])
                out.append(f"| {N} | {f} | {L} | {at}/{R * n} | {np.mean(sds):.4f} | {rng_:.3f} | {spread}/{n} | {amb} |")
    out.append("")
    hv = []
    for N in P["Ns"]:
        for key in allk[N]:
            C = cells_of(ev[key], "best")
            for L in range(1, 7):
                hv.append((N, L, C[("qaoa_heavy", L)]["E"] - C[("qaoa", L)]["E"]))
    out.append("QAOA-heavy minus QAOA best energy (negative means the heavy budget found lower energy):")
    out.append("")
    out.append("| N | cells | heavy lower by > 1e-6 | standard lower by > 1e-6 | most negative |")
    out.append("|---|---|---|---|---|")
    for N in P["Ns"]:
        v = [x for n, _, x in hv if n == N]
        if v:
            out.append(f"| {N} | {len(v)} | {sum(1 for x in v if x < -1e-6)} | {sum(1 for x in v if x > 1e-6)} | {min(v):+.2e} |")
    out.append("")
    viol_mono, viol_nest, nested, pgs = [], [], [0, 0], {f: [] for f in FAMS}
    for (N, k), d in ev.items():
        for r in range(R):
            C = cells_of(d, r)
            for (f, L), c in C.items():
                pgs[f].append(c["pg_inf"])
                if (f, L - 1) in C and c["E"] > C[(f, L - 1)]["E"] + 1e-9:
                    viol_mono.append((N, k, r, f, L))
                if f == "KM" and ("K", L) in C and c["E"] > C[("K", L)]["E"] + 1e-9:
                    viol_nest.append((N, k, r, L))
                if f == "KM":
                    nested[0] += int(c["nested_anc"]); nested[1] += 1
    out.append(f"Energy rose from L-1 to L in a repeat: {len(viol_mono)} {viol_mono[:20]}. KM above K in a repeat: {len(viol_nest)} {viol_nest[:20]}.")
    out.append(f"Selected KM cells with a nested-seed ancestor: {nested[0]}/{nested[1]}.")
    out.append("Projected-gradient infinity norm at the selected point (median / max): "
               + ", ".join(f"{f} {np.median(v):.1e} / {np.max(v):.1e}" for f, v in pgs.items() if v) + ".")
    out.append("")
    # bounds
    out.append("## Bound coverage")
    out.append("")
    touch_sel = [(N, k, c["r"], c["family"], c["L"]) for (N, k), d in ev.items() for c in d["cells"] if c["touch_bound"]]
    fr = {f: max((c["max_diag_frac_of_box"] for d in ev.values() for c in d["cells"] if c["family"] == f), default=0) for f in FAMS}
    starts_touch, starts_tot = 0, 0
    for f in (HERE / "logs").glob("*.jsonl"):
        for line in f.read_text().splitlines():
            r_ = json.loads(line); starts_tot += 1; starts_touch += int(r_["touch_bound"])
    out.append(f"Selected cells with any coordinate within 1e-6 of a bound: {len(touch_sel)} {touch_sel[:20]}.")
    out.append("Largest |diagonal coefficient| / box half-width over selected cells: " + ", ".join(f"{f} {v:.3f}" for f, v in fr.items()) + ".")
    out.append(f"Starts whose final point touched a bound: {starts_touch}/{starts_tot}.")
    out.append("")
    # equal-parameter descriptive
    out.append("## Equal-parameter view (descriptive only, fresh set, mean F_sub)")
    out.append("")
    out.append("| N | KM L1 (6) vs QAOA L3 (6) | KM L3 (12) vs QAOA L6 (12) |")
    out.append("|---|---|---|")
    for N in P["Ns"]:
        mf = lambda f, L: np.mean([cells_of(ev[key], "best")[(f, L)]["F_sub"] for key in prim[N]]) if prim[N] else np.nan
        out.append(f"| {N} | {mf('KM', 1):.3f} vs {mf('qaoa', 3):.3f} | {mf('KM', 3):.3f} vs {mf('qaoa', 6):.3f} |")
    out.append("")
    # accounting
    out.append("## Accounting")
    out.append("")
    out.append("| N | family | starts | evals | capped |")
    out.append("|---|---|---|---|---|")
    acc = {}
    for f in (HERE / "logs").glob("*.jsonl"):
        for line in f.read_text().splitlines():
            r_ = json.loads(line)
            a = acc.setdefault((r_["N"], r_["family"]), [0, 0, 0])
            a[0] += 1; a[1] += r_["n_eval"]; a[2] += int(r_["capped"])
    for (N, f), a in sorted(acc.items()):
        out.append(f"| {N} | {f} | {a[0]} | {a[1]} | {a[2]} |")
    meta = json.loads((HERE / "run_meta.json").read_text()) if (HERE / "run_meta.json").exists() else {}
    out.append("")
    out.append(f"Optimization wall time {meta.get('wall_s', float('nan')):.0f} s of {P['optimize_wall_seconds']} s allowed.")
    (HERE / "RESULTS.md").write_text("\n".join(out) + "\n")
    # figure
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.3))
    for ax, N in zip(axes[:3], P["Ns"]):
        for f, mk in (("KM", "o"), ("K", "s"), ("qaoa", "x"), ("qaoa_heavy", "^"), ("nodata", "+")):
            ys, lo, hi = [], [], []
            for L in range(1, 7):
                xs = [cells_of(ev[key], "best")[(f, L)]["F_sub"] for key in prim[N]]
                if not xs:
                    break
                ys.append(np.mean(xs))
                g = np.random.default_rng(1)
                bs = np.mean(np.asarray(xs)[g.integers(0, len(xs), (2000, len(xs)))], axis=1)
                lo.append(ys[-1] - np.percentile(bs, 2.5)); hi.append(np.percentile(bs, 97.5) - ys[-1])
            if ys:
                ax.errorbar(range(1, len(ys) + 1), ys, yerr=[lo, hi], marker=mk, label=f, capsize=2)
        ax.set_title(f"N={N}, n={len(prim[N])} fresh"); ax.set_xlabel("L")
    axes[0].set_ylabel("mean F_sub, 95% bootstrap CI"); axes[0].legend(frameon=False, fontsize=7)
    ax = axes[3]
    for N, mk in zip(P["Ns"], ("o", "s", "^")):
        pts = [(cells_of(ev[key], "best")[("qaoa", 6)]["F_sub"], cells_of(ev[key], "best")[("KM", 6)]["F_sub"]) for key in prim[N]]
        if pts:
            ax.scatter(*zip(*pts), marker=mk, label=f"N={N}", s=18)
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("QAOA F_sub, L=6"); ax.set_ylabel("KM F_sub, L=6"); ax.legend(frameon=False, fontsize=7)
    fig.tight_layout(); fig.savefig(HERE / "r3_fsub.png", dpi=120); plt.close(fig)
    (HERE / "verdict.json").write_text(json.dumps({"per_N": {str(k): v for k, v in verdict.items()}, "overall": overall,
                                                    "replay_gate": {str(k): v for k, v in gate.items()}}, indent=2) + "\n")
    print("\n".join(out[:12])); print(verdict, overall)


if __name__ == "__main__":
    main()
