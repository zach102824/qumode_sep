"""Tables for the historical B=0.1 energy-only union. Reads energy-selected cells."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

HERE = Path(__file__).resolve().parent
P = json.loads((HERE / "PREREG.json").read_text())
W = P["win"]
C = P["confirmation"]
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


def complete_primary(d):
    return all(d["statuses"].get(str(r)) == "complete" for r in PRIMARY)


def best_over(d, fam, L, repeats):
    cs = [c for c in d["cells"] if c["family"] == fam and c["L"] == L and c["r"] in repeats]
    if len({c["r"] for c in cs}) < len(tuple(repeats)):
        return None
    return min(cs, key=lambda c: (c["E"], c["r"]))


def union_over(d, L, repeats):
    a = best_over(d, "KM_pack", L, repeats)
    b = best_over(d, "KM_fid", L, repeats)
    if a is None or b is None:
        return None
    if a["E"] <= b["E"]:
        return a
    return b


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


def pick(d, fam, L, repeats):
    if fam == "KM_union":
        return union_over(d, L, repeats)
    return best_over(d, fam, L, repeats)


def paired(ev, keys, fam, L, qreps):
    rows = []
    for key in keys:
        d = ev[key]
        a, q = pick(d, fam, L, PRIMARY), pick(d, "qaoa", L, qreps)
        if not (a and q):
            continue
        res, dF, dE = outcome(d, a, q)
        rows.append((d["k"], res, dF, dE, a["F_sub"], q["F_sub"], a["E"], q["E"], a.get("family", fam)))
    return rows


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


def main():
    ev, rp = load("eval"), load("replay")
    lines = ["# Historical B=0.1 energy-only union", "",
             "Benchmark `B = 0.1`. Not a critical-field result. `B_cri` stays withheld. "
             "The earlier |J|-max and Fiedler verdicts are not this result.", ""]
    keys = {}
    for N in P["Ns"]:
        keys[N] = [(N, k) for k in P["k_confirm"] if (N, k) in ev and complete_primary(ev[(N, k)])]
    bad = [(n, k, ev[(n, k)]["statuses"]) for (n, k) in sorted(ev) if not complete_primary(ev[(n, k)])]
    missing = [(N, k) for N in P["Ns"] for k in P["k_confirm"] if (N, k) not in ev]
    lines.append(f"Primary repeats incomplete: {bad}. Never started: {missing}.")
    lines.append("")
    lines.append("| N | primary complete |")
    lines.append("|---|---|")
    for N in P["Ns"]:
        lines.append(f"| {N} | {len(keys[N])}/{len(P['k_confirm'])} |")
    lines.append("")
    lines.append("Each component has 3+3L parameters (21 at L=6). The union keeps one of those vectors. QAOA has 2L (12 at L=6).")
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
    lines.append("## Equal depth, primary QAOA repeats 0 and 1")
    lines.append("")
    lines.append("| family | N | L | n | wins (joint / fidelity-only) | losses | ties | median dF [95% CI] | mean dF | sign p |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    detail = {}
    qreps = tuple(C["comparator_repeats"])
    for fam in ("KM_union", "KM_pack", "KM_fid"):
        for N in P["Ns"]:
            for L in (6, 3):
                rows = paired(ev, keys[N], fam, L, qreps)
                if not rows:
                    continue
                wins, joint, fonly, losses, ties, ci, med, mean, p = line_for(rows)
                detail[(fam, N, L)] = {"wins": wins, "losses": losses, "ci": ci, "n": len(rows)}
                lines.append(f"| {fam} | {N} | {L} | {len(rows)} | {wins} ({joint} / {fonly}) | {losses} | {ties} | "
                             f"{med:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}] | {mean:+.3f} | {p:.3g} |")
        lines.append("| | | | | | | | | | |")
    lines.append("")
    sec = P["secondary_equal_search"]
    lines.append("## Secondary equal search, not the verdict")
    lines.append("")
    lines.append("QAOA uses four repeats. The union still uses two repeats of each packing.")
    lines.append("")
    lines.append("| N | n | wins (joint) | losses | median dF [95% CI] |")
    lines.append("|---|---|---|---|---|")
    four = tuple(sec["qaoa_repeats"])
    for N in P["Ns"]:
        ready = [key for key in keys[N] if all(ev[key]["statuses"].get(str(r)) == "complete" for r in four)]
        rows = paired(ev, ready, "KM_union", 6, four)
        if len(ready) < C["min_complete"] or not rows:
            lines.append(f"| {N} | {len(ready)} | not scored | | |")
            continue
        wins, joint, _, losses, _, ci, med, _, _ = line_for(rows)
        lines.append(f"| {N} | {len(rows)} | {wins} ({joint}) | {losses} | {med:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}] |")
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
            a = union_over(d, pe["union_L"], PRIMARY)
            q = best_over(d, "qaoa", pe["qaoa_L"], tuple(pe["qaoa_repeats"]))
            if a and q:
                res, dF, _ = outcome(d, a, q)
                rows.append((d["k"], res, dF))
        if not rows:
            continue
        wins, joint, _, losses, _, ci, med, _, _ = line_for(rows)
        lines.append(f"| {N} | {wins} ({joint}) | {losses} | {med:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}] |")
    lines.append("")
    scored = {}
    incomplete = False
    for N in P["Ns"]:
        if len(keys[N]) < C["min_complete"] or ("KM_union", N, C["L"]) not in detail:
            incomplete = True
            scored[N] = False
            continue
        st = detail[("KM_union", N, C["L"])]
        scored[N] = bool(st["wins"] > st["losses"] and st["ci"][0] > 0)
    if incomplete:
        overall = "incomplete"
    elif all(scored[N] for N in P["Ns"]):
        overall = "confirmed"
    elif scored[6]:
        overall = "partial"
    else:
        overall = "not_confirmed"
    lines.append("## Verdict")
    lines.append("")
    lines.append(f"Overall: **{overall}**.")
    for N in P["Ns"]:
        lines.append(f"- N={N}: {'pass' if scored[N] else 'does not pass'} on {len(keys[N])} complete instances.")
    lines.append("")
    lines.append("## L=6 union instances")
    lines.append("")
    lines.append("| N | k | chosen | F pack | F Fiedler | F union | F QAOA | dF union | E union - E QAOA | outcome |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for N in P["Ns"]:
        for key in keys[N]:
            d = ev[key]
            u = union_over(d, 6, PRIMARY)
            pk = best_over(d, "KM_pack", 6, PRIMARY)
            fd = best_over(d, "KM_fid", 6, PRIMARY)
            q = best_over(d, "qaoa", 6, qreps)
            if not (u and pk and fd and q):
                continue
            res, dF, dE = outcome(d, u, q)
            lines.append(f"| {N} | {d['k']} | {u['family']} | {pk['F_sub']:.3f} | {fd['F_sub']:.3f} | {u['F_sub']:.3f} | "
                         f"{q['F_sub']:.3f} | {dF:+.3f} | {dE:+.4f} | {res} |")
    evals = capped = starts = 0
    for f in (HERE / "logs").glob("*.jsonl"):
        for line in f.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            evals += int(row["n_eval"])
            capped += int(bool(row["capped"]))
            starts += 1
    grid = 0
    for f in (HERE / "jobs").glob("n*_k*_r*.json"):
        grid += int(json.loads(f.read_text()).get("grid_energy_evals", 0))
    meta = json.loads((HERE / "run_meta.json").read_text()) if (HERE / "run_meta.json").exists() else {}
    lines.append("")
    lines.append("## Cost")
    lines.append("")
    lines.append(f"Optimizer wall {meta.get('wall_s', float('nan')):.1f} s of {P['optimize_wall_seconds']}. "
                 f"Starts {starts}, logged energy evaluations {evals}, L=1 grid evaluations {grid}, capped starts {capped}.")
    text = "\n".join(lines) + "\n"
    (HERE / "RESULTS.md").write_text(text)
    (HERE / "verdict.json").write_text(json.dumps({
        "benchmark": "historical B=0.1",
        "rule": "energy-only union",
        "overall": overall,
        "pass": {str(N): bool(scored[N]) for N in P["Ns"]},
        "n_complete": {str(N): len(keys[N]) for N in P["Ns"]},
    }, indent=2) + "\n")
    print(overall)


if __name__ == "__main__":
    main()
