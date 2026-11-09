"""Tables for the historical B=0.1 diverse-lineage round."""
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
    return all(d["statuses"].get(str(r)) == "complete" for r in PRIMARY) and d["statuses"].get("div") == "complete"


def best_over(d, fam, L, repeats):
    cs = [c for c in d["cells"] if c["family"] == fam and c["L"] == L and c["r"] in repeats]
    if len({c["r"] for c in cs}) < len(tuple(repeats)):
        return None
    return min(cs, key=lambda c: (c["E"], c["r"]))


def div_cell(d):
    cs = [c for c in d["cells"] if c["family"] == "KM_div" and c["L"] == 6]
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


def overall_of(scored, incomplete):
    if incomplete:
        return "incomplete"
    if all(scored[N] for N in P["Ns"]):
        return "confirmed"
    if scored[6]:
        return "partial"
    return "not_confirmed"


def reading(base_pass, div_pass, deeper):
    if base_pass:
        return D["reading_base_passes"]
    if not deeper:
        return D["reading_not_deeper_and_base_fails"]
    if base_pass != div_pass:
        return D["reading_deeper_and_pass_changes"]
    return D["reading_deeper_and_pass_same"]


def main():
    ev, rp = load("eval"), load("replay")
    lines = ["# Historical B=0.1 KM diverse L=1 lineages", "",
             "Benchmark `B = 0.1`. Not a critical-field result. `B_cri` stays withheld. "
             "The packing rounds, the union, the quarter-pi restarts, the classical popcount initialization, "
             "the repulsion escape, the QAOA-depth diagnosis, the energy-legal selection diagnostic, "
             "the uniform box initialization, and the scale-normalized restarts are not this result.", "",
             "KM is the interleaved shared-gamma circuit with the round-4 starts and two repeats. "
             "KM_div keeps that L=6 vector and adds lineages grown from optimized L=1 states the energy beam dropped, "
             "then selects the lowest true energy. The continuation column is equal search. The diverse column is not.", ""]
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
    lines.append("## Equal depth, equal search, continuation KM versus QAOA repeats 0 and 1")
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
            detail[(N, L, "base")] = {"wins": wins, "joint": joint, "fonly": fonly, "losses": losses,
                                      "ties": ties, "ci": ci, "med": med, "mean": mean, "p": p, "n": len(rows)}
            lines.append(f"| KM | {N} | {L} | {len(rows)} | {wins} ({joint} / {fonly}) | {losses} | {ties} | "
                         f"{med:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}] | {mean:+.3f} | {p:.3g} |")
    lines.append("")
    lines.append("## Equal depth, diverse-inclusive KM at L=6, unequal search")
    lines.append("")
    lines.append("| family | N | L | n | wins (joint / fidelity-only) | losses | ties | median dF [95% CI] | mean dF | sign p |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for N in P["Ns"]:
        rows = []
        for key in keys[N]:
            d = ev[key]
            a, q = div_cell(d), best_over(d, "qaoa", 6, qreps)
            if not (a and q):
                continue
            res, dF, dE = outcome(d, a, q)
            rows.append((d["k"], res, dF, dE, a["F_sub"], q["F_sub"], a["E"], q["E"], a.get("label", "")))
        if not rows:
            continue
        wins, joint, fonly, losses, ties, ci, med, mean, p = line_for(rows)
        detail[(N, 6, "div")] = {"wins": wins, "joint": joint, "fonly": fonly, "losses": losses,
                                 "ties": ties, "ci": ci, "med": med, "mean": mean, "p": p, "n": len(rows)}
        lines.append(f"| KM_div | {N} | 6 | {len(rows)} | {wins} ({joint} / {fonly}) | {losses} | {ties} | "
                     f"{med:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}] | {mean:+.3f} | {p:.3g} |")
    lines.append("")
    lines.append("## Energy drop")
    lines.append("")
    lines.append(f"An N is deeper when the median energy drop exceeds {D['drop_min']} "
                 f"or at least {D['drop_count']} instances drop by more than {D['drop_min']}. "
                 "The drop is continuation KM energy minus diverse-inclusive energy.")
    lines.append("")
    lines.append("| N | n | median drop | n drop > 0.01 | deeper | continuation pass | diverse pass |")
    lines.append("|---|---|---|---|---|---|---|")
    scored_base, scored_div = {}, {}
    deeper = {}
    drops_out = {}
    incomplete = False
    for N in P["Ns"]:
        if len(keys[N]) < C["min_complete"] or (N, C["L"], "base") not in detail or (N, 6, "div") not in detail:
            incomplete = True
            scored_base[N] = False
            scored_div[N] = False
            deeper[N] = False
            lines.append(f"| {N} | {len(keys[N])} | | | | | |")
            continue
        st = detail[(N, C["L"], "base")]
        sr = detail[(N, 6, "div")]
        scored_base[N] = bool(st["wins"] > st["losses"] and st["ci"][0] > 0)
        scored_div[N] = bool(sr["wins"] > sr["losses"] and sr["ci"][0] > 0)
        drops = []
        for key in keys[N]:
            d = ev[key]
            base, div = best_over(d, "KM", 6, PRIMARY), div_cell(d)
            drop = base["E"] - div["E"]
            if drop < -1e-9:
                raise SystemExit(f"diverse energy is above the continuation energy at N={N} k={d['k']}")
            drops.append(drop)
        med = float(np.median(drops))
        nbig = sum(v > float(D["drop_min"]) for v in drops)
        deeper[N] = bool(med > float(D["drop_min"]) or nbig >= int(D["drop_count"]))
        drops_out[str(N)] = {"median": med, "n_above": int(nbig), "drops": [float(v) for v in drops]}
        lines.append(f"| {N} | {len(drops)} | {med:.4f} | {nbig} | {deeper[N]} | {scored_base[N]} | {scored_div[N]} |")
    base_overall = overall_of(scored_base, incomplete)
    div_overall = overall_of(scored_div, incomplete)
    lines.append("")
    for N in P["Ns"]:
        if len(keys[N]) < C["min_complete"]:
            continue
        lines.append(f"N={N}: {reading(scored_base[N], scored_div[N], deeper[N])}")
    lines.append("")
    pe = P["secondary_equal_param"]
    lines.append(f"## Secondary equal parameter count, {pe['n_param']} parameters, not the verdict")
    lines.append("")
    lines.append("| N | wins (joint) | losses | median dF [95% CI] |")
    lines.append("|---|---|---|---|")
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
            lines.append(f"| {N} | {wins} ({joint}) | {losses} | {med:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}] |")
    lines.append("")
    lines.append("## Verdict")
    lines.append("")
    lines.append(f"Continuation column: **{base_overall}**.")
    lines.append(f"Diverse-inclusive column: **{div_overall}**.")
    lines.append("The verdict of this round is the diverse-inclusive column. Equal depth is the comparison. Equal search is the continuation column only.")
    for N in P["Ns"]:
        lines.append(f"- N={N}: continuation {'pass' if scored_base[N] else 'does not pass'}, "
                     f"diverse {'pass' if scored_div[N] else 'does not pass'}, "
                     f"deeper {deeper.get(N, False)}, on {len(keys[N])} complete instances.")
    lines.append("")
    lines.append("## L=6 instances")
    lines.append("")
    lines.append("| N | k | continuation label | diverse label | F continuation | F diverse | F QAOA | dF continuation | dF diverse | E continuation-QAOA | energy drop | continuation | diverse |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    n_div_label = 0
    for N in P["Ns"]:
        for key in keys[N]:
            d = ev[key]
            a = best_over(d, "KM", 6, PRIMARY)
            q = best_over(d, "qaoa", 6, qreps)
            z = div_cell(d)
            if not (a and q and z):
                continue
            rb, dFb, dEb = outcome(d, a, q)
            rd, dFd, _ = outcome(d, z, q)
            if str(z.get("label", "")).startswith("div"):
                n_div_label += 1
            lines.append(f"| {N} | {d['k']} | {a.get('label', '')} | {z.get('label', '')} | {a['F_sub']:.3f} | "
                         f"{z['F_sub']:.3f} | {q['F_sub']:.3f} | {dFb:+.3f} | {dFd:+.3f} | {dEb:+.4f} | "
                         f"{a['E'] - z['E']:+.4f} | {rb} | {rd} |")
    lines.append("")
    lines.append("## Lineage gaps, descriptive")
    lines.append("")
    lines.append("A start recovers the continuation energy when it is at most 1e-8 above it. "
                 "The nearest other gap is descriptive. It is not a second threshold.")
    lines.append("")
    lines.append("| N | L=6 lineage starts | strictly lower | recovered | nearest gap above 1e-8 |")
    lines.append("|---|---|---|---|---|")
    gap_out = {}
    for N in P["Ns"]:
        n_l6 = n_low = n_rec = 0
        above = []
        for k in P["k_confirm"]:
            job_path = HERE / "jobs" / f"n{N}_k{k}.json"
            if not job_path.exists():
                continue
            job = json.loads(job_path.read_text())
            if not job.get("km_base_L6"):
                continue
            base = float(job["km_base_L6"]["E"])
            for r in PRIMARY:
                path = HERE / "logs" / f"n{N}_k{k}_div_r{r}.jsonl"
                if not path.exists():
                    continue
                for line in path.read_text().splitlines():
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    if int(row["L"]) != 6:
                        continue
                    n_l6 += 1
                    gap = float(row["E"]) - base
                    if gap < -1e-9:
                        n_low += 1
                    elif gap <= 1e-8:
                        n_rec += 1
                    else:
                        above.append(gap)
        nearest = min(above) if above else None
        gap_out[str(N)] = {"n_l6": n_l6, "n_lower": n_low, "n_recovered": n_rec,
                           "nearest_above": None if nearest is None else float(nearest)}
        near_s = " " if nearest is None else f"{nearest:.4f}"
        lines.append(f"| {N} | {n_l6} | {n_low} | {n_rec} | {near_s} |")
    evals = capped = starts = div_starts = 0
    for f in (HERE / "logs").glob("*.jsonl"):
        for line in f.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            evals += int(row["n_eval"])
            capped += int(bool(row["capped"]))
            starts += 1
            if str(row.get("family", "")) == "KM_div":
                div_starts += 1
    grid = 0
    n_lineages = 0
    for f in (HERE / "jobs").glob("n*_k*.json"):
        job = json.loads(f.read_text())
        grid += int(job.get("grid_energy_evals", 0))
        for rec in job.get("repeats", {}).values():
            n_lineages += int(rec.get("n_diverse", 0))
    meta = json.loads((HERE / "run_meta.json").read_text()) if (HERE / "run_meta.json").exists() else {}
    lines.append("")
    lines.append("## Cost")
    lines.append("")
    lines.append(f"Optimizer wall {meta.get('wall_s', float('nan')):.1f} s of {P['optimize_wall_seconds']}. "
                 f"Starts {starts}, of which {div_starts} are diverse-lineage starts. "
                 f"Lineages opened {n_lineages}. "
                 f"L=6 KM_div states whose label is a lineage child: {n_div_label}. "
                 f"Logged energy evaluations {evals}, L=1 grid evaluations {grid}, capped starts {capped}.")
    text = "\n".join(lines) + "\n"
    (HERE / "RESULTS.md").write_text(text)
    (HERE / "verdict.json").write_text(json.dumps({
        "benchmark": "historical B=0.1",
        "rule": "interleaved shared-gamma KM; continuation column plus diverse L=1 lineages",
        "base_overall": base_overall,
        "div_overall": div_overall,
        "verdict": div_overall,
        "pass_base": {str(N): bool(scored_base[N]) for N in P["Ns"]},
        "pass_div": {str(N): bool(scored_div[N]) for N in P["Ns"]},
        "deeper": {str(N): bool(deeper.get(N, False)) for N in P["Ns"]},
        "energy_drop": drops_out,
        "lineage_gaps": gap_out,
        "n_div_selected": n_div_label,
        "n_lineages": n_lineages,
        "n_complete": {str(N): len(keys[N]) for N in P["Ns"]},
        "detail": {f"{N}|{L}|{col}": rec for (N, L, col), rec in detail.items()},
    }, indent=2) + "\n")
    print(base_overall, div_overall)


if __name__ == "__main__":
    main()
