"""Tables for the historical B=0.1 energy-legal selection diagnostic."""
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
    return all(d["statuses"].get(str(r)) == "complete" for r in PRIMARY) and "fsel" in d


def best_over(d, fam, L, repeats):
    cs = [c for c in d["cells"] if c["family"] == fam and c["L"] == L and c["r"] in repeats]
    if len({c["r"] for c in cs}) < len(tuple(repeats)):
        return None
    return min(cs, key=lambda c: (c["E"], c["r"], c["label"]))


def fsel_cell(d):
    cs = [c for c in d["cells"] if c["family"] == "KM_fsel" and c["L"] == 6]
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


def reading(claim_pass, diag_pass, n_differ):
    if claim_pass and not diag_pass:
        raise SystemExit("diagnostic pass bit fell below the energy-only pass bit")
    if claim_pass:
        return D["reading_claim_passes"]
    if diag_pass:
        return D["reading_selection_limited"]
    if n_differ == 0:
        return D["reading_no_alternative"]
    return D["reading_alternative_still_fails"]


def rows_for(keys, ev, fam):
    out = []
    qreps = tuple(C["comparator_repeats"])
    for key in keys:
        d = ev[key]
        q = best_over(d, "qaoa", 6, qreps)
        a = best_over(d, "KM", 6, PRIMARY) if fam == "KM" else fsel_cell(d)
        if not (a and q):
            continue
        res, dF, dE = outcome(d, a, q)
        out.append((d["k"], res, dF, dE, a["F_sub"], q["F_sub"], a["E"], q["E"], a.get("label", ""), a.get("r")))
    return out


def main():
    ev, rp = load("eval"), load("replay")
    lines = ["# Historical B=0.1 energy-legal selection diagnostic", "",
             "Benchmark `B = 0.1`. Not a critical-field result. `B_cri` stays withheld. "
             "The packing rounds, the union, the quarter-pi restarts, the classical popcount initialization, "
             "the repulsion escape, and the QAOA-depth diagnosis are not this result.", "",
             "The claim is energy-only interleaved shared-gamma KM against equal-depth QAOA, "
             "both on the round-4 starts and repeats 0 and 1. "
             "The diagnostic state is the highest subspace overlap among L=6 KM starts whose energy "
             f"is at most the QAOA energy plus {D['energy_legal_tol']}. "
             "That column is not the verdict. The unrestricted maximum overlap has no pass rule.", ""]
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
    lines.append("## Equal depth, energy-only, the claim")
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
            detail[(N, L, "claim")] = {"wins": wins, "losses": losses, "ci": ci, "n": len(rows),
                                       "joint": joint, "fonly": fonly, "ties": ties, "median": med, "mean": mean, "p": p}
            lines.append(f"| KM | {N} | {L} | {len(rows)} | {wins} ({joint} / {fonly}) | {losses} | {ties} | "
                         f"{med:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}] | {mean:+.3f} | {p:.3g} |")
    lines.append("")
    lines.append("## Diagnostic column at L=6, same starts, not the verdict")
    lines.append("")
    lines.append("| family | N | L | n | wins (joint / fidelity-only) | losses | ties | median dF [95% CI] | mean dF | sign p | n differ |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for N in P["Ns"]:
        rows = rows_for(keys[N], ev, "KM_fsel")
        if not rows:
            continue
        wins, joint, fonly, losses, ties, ci, med, mean, p = line_for(rows)
        n_differ = sum(bool(ev[key]["fsel"]["differs"]) for key in keys[N])
        detail[(N, 6, "diag")] = {"wins": wins, "losses": losses, "ci": ci, "n": len(rows),
                                  "joint": joint, "fonly": fonly, "ties": ties, "median": med, "mean": mean,
                                  "p": p, "n_differ": n_differ}
        lines.append(f"| KM_fsel | {N} | 6 | {len(rows)} | {wins} ({joint} / {fonly}) | {losses} | {ties} | "
                     f"{med:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}] | {mean:+.3f} | {p:.3g} | {n_differ} |")
    lines.append("")
    lines.append("## Diagnosis")
    lines.append("")
    lines.append("A size is selection-limited when the energy-only column does not pass and the diagnostic column does. "
                 "The diagnostic state differs when its subspace overlap exceeds the energy-only overlap by more than 1e-6 "
                 "or its energy differs by more than 1e-8.")
    lines.append("")
    lines.append("| N | n | claim pass | diagnostic pass | selection_limited | n differ | median F gain | median pool |")
    lines.append("|---|---|---|---|---|---|---|---|")
    scored_claim, scored_diag = {}, {}
    limited = {}
    differ_n = {}
    incomplete = False
    for N in P["Ns"]:
        if len(keys[N]) < C["min_complete"] or (N, C["L"], "claim") not in detail or (N, 6, "diag") not in detail:
            incomplete = True
            scored_claim[N] = False
            scored_diag[N] = False
            limited[N] = False
            differ_n[N] = 0
            lines.append(f"| {N} | {len(keys[N])} | | | | | | |")
            continue
        st = detail[(N, C["L"], "claim")]
        sd = detail[(N, 6, "diag")]
        scored_claim[N] = bool(st["wins"] > st["losses"] and st["ci"][0] > 0)
        scored_diag[N] = bool(sd["wins"] > sd["losses"] and sd["ci"][0] > 0)
        limited[N] = bool((not scored_claim[N]) and scored_diag[N])
        differ_n[N] = int(sd["n_differ"])
        gains = [ev[key]["fsel"]["F_diag"] - ev[key]["fsel"]["F_energy"] for key in keys[N]]
        pools = [ev[key]["fsel"]["pool_n"] for key in keys[N]]
        lines.append(f"| {N} | {len(keys[N])} | {scored_claim[N]} | {scored_diag[N]} | {limited[N]} | "
                     f"{differ_n[N]} | {float(np.median(gains)):+.4f} | {float(np.median(pools)):.1f} |")
    claim_overall = overall_of(scored_claim, incomplete)
    diag_overall = overall_of(scored_diag, incomplete)
    lines.append("")
    for N in P["Ns"]:
        if len(keys[N]) < C["min_complete"]:
            continue
        lines.append(f"N={N}: {reading(scored_claim[N], scored_diag[N], differ_n[N])}")
    lines.append("")
    lines.append("## Unrestricted maximum overlap, not a comparator")
    lines.append("")
    lines.append("| N | n differ | median F gain | median energy gap |")
    lines.append("|---|---|---|---|")
    unrestricted = {}
    for N in P["Ns"]:
        if not keys[N]:
            continue
        gains = [ev[key]["fsel"]["F_unrestricted"] - ev[key]["fsel"]["F_energy"] for key in keys[N]]
        gaps = [ev[key]["fsel"]["E_unrestricted"] - ev[key]["fsel"]["E_energy"] for key in keys[N]]
        n_diff = sum(bool(ev[key]["fsel"]["unrestricted_differs"]) for key in keys[N])
        unrestricted[str(N)] = {"n_differ": int(n_diff), "median_F_gain": float(np.median(gains)),
                                "median_E_gap": float(np.median(gaps))}
        lines.append(f"| {N} | {n_diff} | {float(np.median(gains)):+.4f} | {float(np.median(gaps)):+.4f} |")
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
    lines.append(f"Energy-only claim: **{claim_overall}**.")
    lines.append(f"Energy-legal diagnostic, not the verdict: **{diag_overall}**.")
    for N in P["Ns"]:
        lines.append(f"- N={N}: energy-only {'pass' if scored_claim[N] else 'does not pass'}, "
                     f"diagnostic {'pass' if scored_diag[N] else 'does not pass'}, "
                     f"selection_limited {limited.get(N, False)}, on {len(keys[N])} complete instances.")
    lines.append("")
    lines.append("## L=6 instances")
    lines.append("")
    lines.append("| N | k | KM label | fsel label | F KM | F fsel | F QAOA | dF claim | dF fsel | E KM-QAOA | pool | differ | claim | fsel | F max | E gap |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for N in P["Ns"]:
        for key in keys[N]:
            d = ev[key]
            a = best_over(d, "KM", 6, PRIMARY)
            q = best_over(d, "qaoa", 6, qreps)
            z = fsel_cell(d)
            info = d["fsel"]
            if not (a and q and z):
                continue
            rb, dFb, dEb = outcome(d, a, q)
            rd, dFd, _ = outcome(d, z, q)
            lines.append(f"| {N} | {d['k']} | {a.get('label', '')} | {z.get('label', '')} | {a['F_sub']:.3f} | "
                         f"{z['F_sub']:.3f} | {q['F_sub']:.3f} | {dFb:+.3f} | {dFd:+.3f} | {dEb:+.4f} | "
                         f"{info['pool_n']} | {info['differs']} | {rb} | {rd} | "
                         f"{info['F_unrestricted']:.3f} | {info['E_unrestricted'] - info['E_energy']:+.4f} |")
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
    for f in (HERE / "jobs").glob("n*_k*.json"):
        grid += int(json.loads(f.read_text()).get("grid_energy_evals", 0))
    meta = json.loads((HERE / "run_meta.json").read_text()) if (HERE / "run_meta.json").exists() else {}
    lines.append("")
    lines.append("## Cost")
    lines.append("")
    lines.append(f"Optimizer wall {meta.get('wall_s', float('nan')):.1f} s of {P['optimize_wall_seconds']}. "
                 f"Starts {starts}. Logged energy evaluations {evals}, L=1 grid evaluations {grid}, capped starts {capped}.")
    text = "\n".join(lines) + "\n"
    (HERE / "RESULTS.md").write_text(text)
    (HERE / "verdict.json").write_text(json.dumps({
        "benchmark": "historical B=0.1",
        "rule": "energy-only interleaved KM against equal-depth QAOA; energy-legal fidelity column is not the verdict",
        "claim_overall": claim_overall,
        "diagnostic_overall": diag_overall,
        "pass_claim": {str(N): bool(scored_claim[N]) for N in P["Ns"]},
        "pass_diagnostic": {str(N): bool(scored_diag[N]) for N in P["Ns"]},
        "selection_limited": {str(N): bool(limited.get(N, False)) for N in P["Ns"]},
        "n_differ": {str(N): int(differ_n.get(N, 0)) for N in P["Ns"]},
        "unrestricted": unrestricted,
        "n_complete": {str(N): len(keys[N]) for N in P["Ns"]},
        "detail": {f"{N}:{kind}": detail[(N, 6, kind)] for N in P["Ns"] for kind in ("claim", "diag")
                   if (N, 6, kind) in detail},
    }, indent=2) + "\n")
    print(claim_overall, diag_overall)


if __name__ == "__main__":
    main()
