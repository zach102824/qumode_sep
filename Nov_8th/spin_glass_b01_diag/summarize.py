"""Tables for the historical B=0.1 QAOA-depth diagnosis."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

HERE = Path(__file__).resolve().parent
P = json.loads((HERE / "PREREG.json").read_text())
W = P["win"]
C = P["confirmation"]
D = P["diagnosis"]
PRIMARY = tuple(C["comparator_repeats"])


def load(dirname):
    out = {}
    d = HERE / dirname
    if not d.exists():
        return out
    for f in d.glob("n*_k*.json"):
        doc = json.loads(f.read_text())
        out[(doc["N"], doc["k"])] = doc
    return out


def complete(d):
    return all(d["statuses"].get(str(r)) == "complete" for r in PRIMARY) and d["statuses"].get("deep") == "complete"


def best_over(d, fam, L, repeats):
    cs = [c for c in d["cells"] if c["family"] == fam and c["L"] == L and c["r"] in repeats]
    if len({c["r"] for c in cs}) < len(tuple(repeats)):
        return None
    return min(cs, key=lambda c: (c["E"], c["r"]))


def deep_cell(d):
    cs = [c for c in d["cells"] if c["family"] == "qaoa_deep" and c["L"] == 6]
    if len(cs) != 1:
        return None
    return cs[0]


def outcome(d, a, q):
    fk, fq = a["F_sub"], q["F_sub"]
    cl = 1 - W["clamp"]
    rho = float(np.log((1 - min(fq, cl)) / (1 - min(fk, cl))))
    dF = fk - fq
    if not d["even_ground_in_window"]:
        res = "tie_unreachable"
    elif max(fk, fq) < W["floor"]:
        res = "tie_both_failed"
    elif rho > np.log(W["ratio"]) and dF > W["dF_min"]:
        res = "win_joint" if a["E"] <= q["E"] + 1e-6 else "win_fidelity_only"
    elif rho < -np.log(W["ratio"]) and dF < -W["dF_min"]:
        res = "loss_joint" if q["E"] <= a["E"] + 1e-6 else "loss_fidelity_only"
    else:
        res = "tie"
    return res, dF, a["E"] - q["E"]


def boot_median(v):
    v = np.asarray(v, float)
    g = np.random.default_rng(C["bootstrap"]["seed"])
    meds = np.median(v[g.integers(0, v.size, size=(C["bootstrap"]["resamples"], v.size))], axis=1)
    return float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))


def line_for(rows):
    dF = [r[2] for r in rows]
    res = [r[1] for r in rows]
    wins = sum(r.startswith("win") for r in res)
    joint = sum(r == "win_joint" for r in res)
    fonly = sum(r == "win_fidelity_only" for r in res)
    losses = sum(r.startswith("loss") for r in res)
    ties = sum(r.startswith("tie") for r in res)
    ci = boot_median(dF)
    pos = sum(v > 0 for v in dF)
    neg = sum(v < 0 for v in dF)
    p = float(binomtest(pos, pos + neg, 0.5).pvalue) if pos + neg else float("nan")
    return wins, joint, fonly, losses, ties, ci, float(np.median(dF)), float(np.mean(dF)), p


def overall_of(scored, keys, incomplete):
    if incomplete:
        return "incomplete"
    if all(scored[N] for N in P["Ns"]):
        return "confirmed"
    if scored[6]:
        return "partial"
    return "not_confirmed"


def reading(base_pass, deep_pass, under):
    if base_pass:
        return D["reading_base_passes"]
    if not under:
        return D["reading_not_underoptimized_and_base_fails"]
    if base_pass != deep_pass:
        return D["reading_underoptimized_and_pass_changes"]
    return D["reading_underoptimized_and_pass_same"]


def main():
    ev, rp = load("eval"), load("replay")
    lines = ["# Historical B=0.1 QAOA-depth diagnosis", "",
             "Benchmark `B = 0.1`. Not a critical-field result. `B_cri` stays withheld. "
             "The packing rounds, the union, the quarter-pi restarts, the classical popcount initialization, "
             "and the repulsion escape are not this result.", "",
             "KM is the interleaved shared-gamma circuit with the round-4 starts and two repeats. "
             "The historical comparator is QAOA with that same start list and the same two repeats. "
             "qaoa_deep keeps that L=6 vector and adds eight box starts and eight N(0, 0.75) kicks, "
             "then selects the lowest true energy. Equal search is not claimed for the deep column.", ""]
    keys = {}
    for N in P["Ns"]:
        keys[N] = [(N, k) for k in P["k_confirm"] if (N, k) in ev and complete(ev[(N, k)])]
    bad = [(n, k, ev[(n, k)]["statuses"]) for (n, k) in sorted(ev) if not complete(ev[(n, k)])]
    missing = [(N, k) for N in P["Ns"] for k in P["k_confirm"] if (N, k) not in ev]
    lines.append(f"Incomplete: {bad}. Never started: {missing}.")
    lines.append("")
    lines.append("| N | complete |")
    lines.append("|---|---|")
    for N in P["Ns"]:
        lines.append(f"| {N} | {len(keys[N])}/{len(P['k_confirm'])} |")
    lines.append("")
    lines.append("KM has 3+3L parameters (21 at L=6). QAOA has 2L (12 at L=6).")
    lines.append("")
    tol = P["replay_tol"]
    lines.append("## Independent replay")
    lines.append("")
    lines.append("| N | states | max rel dE | max abs dF | max abs dE0 | n_sub mismatches | max norm err | pass |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for N in P["Ns"]:
        ks = [key for key in rp if key[0] == N]
        cs = [c for key in ks for c in rp[key]["cells"]]
        if not cs:
            lines.append(f"| {N} | 0 | | | | | | False |")
            continue
        mdE = max(abs(c["dE_rel"]) for c in cs)
        mdF = max(abs(c["dF_sub"]) for c in cs)
        md0 = max(abs(rp[key]["dE0"]) for key in ks)
        mism = sum(rp[key]["n_sub_replay"] != rp[key]["n_sub_main"] for key in ks)
        mn = max(c["norm_err"] for c in cs)
        ok = mdE <= tol["E_rel"] and mdF <= tol["F_sub"] and md0 <= tol["E0"] and mism == 0 and mn <= tol["norm"]
        lines.append(f"| {N} | {len(cs)} | {mdE:.1e} | {mdF:.1e} | {md0:.1e} | {mism} | {mn:.1e} | {ok} |")
    lines.append("")
    lines.append("## Equal depth, historical QAOA repeats 0 and 1")
    lines.append("")
    lines.append("| family | N | L | n | wins (joint / fidelity-only) | losses | ties | median dF [95% CI] | mean dF | sign p |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    detail = {}
    qreps = tuple(C["comparator_repeats"])
    for N in P["Ns"]:
        for L in (6, 3):
            rows = []
            for key in keys[N]:
                d = ev[key]
                a, q = best_over(d, "KM", L, PRIMARY), best_over(d, "qaoa", L, qreps)
                if not (a and q):
                    continue
                res, dF, dE = outcome(d, a, q)
                rows.append((d["k"], res, dF, dE, a["F_sub"], q["F_sub"], a["E"], q["E"], a.get("label", "")))
            if not rows:
                continue
            wins, joint, fonly, losses, ties, ci, med, mean, p = line_for(rows)
            detail[(N, L, "base")] = {"wins": wins, "losses": losses, "ci": ci, "n": len(rows)}
            lines.append(f"| KM | {N} | {L} | {len(rows)} | {wins} ({joint} / {fonly}) | {losses} | {ties} | "
                         f"{med:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}] | {mean:+.3f} | {p:.3g} |")
    lines.append("")
    lines.append("## Equal depth against qaoa_deep at L=6, unequal search")
    lines.append("")
    lines.append("| family | N | L | n | wins (joint / fidelity-only) | losses | ties | median dF [95% CI] | mean dF | sign p |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for N in P["Ns"]:
        rows = []
        for key in keys[N]:
            d = ev[key]
            a, q = best_over(d, "KM", 6, PRIMARY), deep_cell(d)
            if not (a and q):
                continue
            res, dF, dE = outcome(d, a, q)
            rows.append((d["k"], res, dF, dE, a["F_sub"], q["F_sub"], a["E"], q["E"], a.get("label", "")))
        if not rows:
            continue
        wins, joint, fonly, losses, ties, ci, med, mean, p = line_for(rows)
        detail[(N, 6, "deep")] = {"wins": wins, "losses": losses, "ci": ci, "n": len(rows)}
        lines.append(f"| KM | {N} | 6 | {len(rows)} | {wins} ({joint} / {fonly}) | {losses} | {ties} | "
                     f"{med:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}] | {mean:+.3f} | {p:.3g} |")
    lines.append("")
    lines.append("## Diagnosis")
    lines.append("")
    lines.append(f"An N is under-optimized when the median energy drop exceeds {D['drop_min']} "
                 f"or at least {D['drop_count']} instances drop by more than {D['drop_min']}. "
                 "The drop is historical QAOA energy minus qaoa_deep energy.")
    lines.append("")
    lines.append("| N | n | median drop | n drop > 0.01 | underoptimized | base pass | deep pass |")
    lines.append("|---|---|---|---|---|---|---|")
    scored_base, scored_deep = {}, {}
    under = {}
    drops_out = {}
    incomplete = False
    for N in P["Ns"]:
        if len(keys[N]) < C["min_complete"] or (N, C["L"], "base") not in detail or (N, 6, "deep") not in detail:
            incomplete = True
            scored_base[N] = False
            scored_deep[N] = False
            under[N] = False
            lines.append(f"| {N} | {len(keys[N])} | | | | | |")
            continue
        st = detail[(N, C["L"], "base")]
        sd = detail[(N, 6, "deep")]
        scored_base[N] = bool(st["wins"] > st["losses"] and st["ci"][0] > 0)
        scored_deep[N] = bool(sd["wins"] > sd["losses"] and sd["ci"][0] > 0)
        drops = []
        for key in keys[N]:
            d = ev[key]
            base, deep = best_over(d, "qaoa", 6, qreps), deep_cell(d)
            drops.append(base["E"] - deep["E"])
        med = float(np.median(drops))
        nbig = sum(v > float(D["drop_min"]) for v in drops)
        under[N] = bool(med > float(D["drop_min"]) or nbig >= int(D["drop_count"]))
        drops_out[str(N)] = {"median": med, "n_above": int(nbig), "drops": [float(v) for v in drops]}
        lines.append(f"| {N} | {len(drops)} | {med:.4f} | {nbig} | {under[N]} | {scored_base[N]} | {scored_deep[N]} |")
    base_overall = overall_of(scored_base, keys, incomplete)
    deep_overall = overall_of(scored_deep, keys, incomplete)
    lines.append("")
    for N in P["Ns"]:
        if incomplete and N not in under:
            continue
        if len(keys[N]) < C["min_complete"]:
            continue
        lines.append(f"N={N}: {reading(scored_base[N], scored_deep[N], under[N])}")
    lines.append("")
    pe = P["secondary_equal_param"]
    lines.append(f"## Secondary equal parameter count, {pe['n_param']} parameters, not the verdict")
    lines.append("")
    lines.append("| N | comparator | wins (joint) | losses | median dF [95% CI] |")
    lines.append("|---|---|---|---|---|")
    for N in P["Ns"]:
        rows = []
        for key in keys[N]:
            d = ev[key]
            a = best_over(d, "KM", pe["km_L"], PRIMARY)
            q = best_over(d, "qaoa", pe["qaoa_L"], tuple(pe["qaoa_repeats"]))
            if a and q:
                res, dF, _ = outcome(d, a, q)
                rows.append((d["k"], res, dF))
        if rows:
            wins, joint, _, losses, _, ci, med, _, _ = line_for(rows)
            lines.append(f"| {N} | historical QAOA | {wins} ({joint}) | {losses} | {med:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}] |")
        rows = []
        for key in keys[N]:
            d = ev[key]
            a, q = best_over(d, "KM", pe["km_L"], PRIMARY), deep_cell(d)
            if a and q:
                res, dF, _ = outcome(d, a, q)
                rows.append((d["k"], res, dF))
        if rows:
            wins, joint, _, losses, _, ci, med, _, _ = line_for(rows)
            lines.append(f"| {N} | qaoa_deep | {wins} ({joint}) | {losses} | {med:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}] |")
    lines.append("")
    lines.append("## Verdict")
    lines.append("")
    lines.append(f"Historical comparator: **{base_overall}**.")
    lines.append(f"Heavier QAOA search: **{deep_overall}**.")
    if base_overall == "confirmed" and deep_overall != "confirmed":
        lines.append("The historical pass does not survive the heavier comparator. This round is not a confirmed advantage.")
    for N in P["Ns"]:
        lines.append(f"- N={N}: historical {'pass' if scored_base[N] else 'does not pass'}, "
                     f"deep {'pass' if scored_deep[N] else 'does not pass'}, "
                     f"underoptimized {under.get(N, False)}, on {len(keys[N])} complete instances.")
    lines.append("")
    lines.append("## L=6 instances")
    lines.append("")
    lines.append("| N | k | KM label | deep label | F KM | F base | F deep | dF base | dF deep | E KM-base | energy drop | base | deep |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    n_deep_label = 0
    for N in P["Ns"]:
        for key in keys[N]:
            d = ev[key]
            a = best_over(d, "KM", 6, PRIMARY)
            q = best_over(d, "qaoa", 6, qreps)
            z = deep_cell(d)
            if not (a and q and z):
                continue
            rb, dFb, dEb = outcome(d, a, q)
            rd, dFd, _ = outcome(d, a, z)
            if str(z.get("label", "")).startswith("deep_"):
                n_deep_label += 1
            lines.append(f"| {N} | {d['k']} | {a.get('label', '')} | {z.get('label', '')} | {a['F_sub']:.3f} | "
                         f"{q['F_sub']:.3f} | {z['F_sub']:.3f} | {dFb:+.3f} | {dFd:+.3f} | {dEb:+.4f} | "
                         f"{q['E'] - z['E']:+.4f} | {rb} | {rd} |")
    evals = capped = starts = deep_starts = 0
    for f in (HERE / "logs").glob("*.jsonl"):
        for line in f.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            evals += int(row["n_eval"])
            capped += int(bool(row["capped"]))
            starts += 1
            if str(row.get("label", "")).startswith("deep_"):
                deep_starts += 1
    grid = 0
    for f in (HERE / "jobs").glob("n*_k*.json"):
        grid += int(json.loads(f.read_text()).get("grid_energy_evals", 0))
    meta = json.loads((HERE / "run_meta.json").read_text()) if (HERE / "run_meta.json").exists() else {}
    lines.append("")
    lines.append("## Cost")
    lines.append("")
    lines.append(f"Optimizer wall {meta.get('wall_s', float('nan')):.1f} s of {P['optimize_wall_seconds']}. "
                 f"Starts {starts}, of which {deep_starts} are the extra L=6 QAOA starts. "
                 f"L=6 qaoa_deep states whose label is one of those starts: {n_deep_label}. "
                 f"Logged energy evaluations {evals}, L=1 grid evaluations {grid}, capped starts {capped}.")
    text = "\n".join(lines) + "\n"
    (HERE / "RESULTS.md").write_text(text)
    (HERE / "verdict.json").write_text(json.dumps({
        "benchmark": "historical B=0.1",
        "rule": "standard interleaved KM against historical QAOA, with a heavier L=6 QAOA search as the diagnosis",
        "base_overall": base_overall,
        "deep_overall": deep_overall,
        "pass_base": {str(N): bool(scored_base[N]) for N in P["Ns"]},
        "pass_deep": {str(N): bool(scored_deep[N]) for N in P["Ns"]},
        "underoptimized": {str(N): bool(under.get(N, False)) for N in P["Ns"]},
        "energy_drop": drops_out,
        "n_complete": {str(N): len(keys[N]) for N in P["Ns"]},
    }, indent=2) + "\n")
    print(base_overall, deep_overall)


if __name__ == "__main__":
    main()
