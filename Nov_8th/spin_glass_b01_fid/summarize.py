"""Tables for the historical B=0.1 packing confirmation. Reads energy-selected cells."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

HERE = Path(__file__).resolve().parent
P = json.loads((HERE / "PREREG.json").read_text())
W = P["win"]
C = P["confirmation"]
R = P["repeats"]


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
    return len(d["statuses"]) == R and all(v == "complete" for v in d["statuses"].values())


def best_cell(d, fam, L):
    cs = [c for c in d["cells"] if c["family"] == fam and c["L"] == L]
    return min(cs, key=lambda c: (c["E"], c["r"])) if cs else None


def outcome(d, a, q):
    fk, fq = a["F_sub"], q["F_sub"]
    cl = 1 - W["clamp"]
    rho = float(np.log((1 - min(fq, cl)) / (1 - min(fk, cl))))
    dF = fk - fq
    joint = a["E"] <= q["E"] + 1e-6
    if not d["even_ground_in_window"]:
        res = "tie_unreachable"
    elif max(fk, fq) < W["floor"]:
        res = "tie_both_failed"
    elif rho > np.log(W["ratio"]) and dF > W["dF_min"]:
        res = "win_joint" if joint else "win_fidelity_only"
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


def paired(ev, keys, fam, L):
    rows = []
    for key in keys:
        d = ev[key]
        a, q = best_cell(d, fam, L), best_cell(d, "qaoa", L)
        if not (a and q):
            continue
        res, dF, dE = outcome(d, a, q)
        rows.append((d["k"], res, dF, dE, a["F_sub"], q["F_sub"], a["E"], q["E"], a["n_param"], q["n_param"]))
    return rows


def main():
    ev, rp = load("eval"), load("replay")
    lines = ["# Historical B=0.1 Fiedler packing", "",
             "Benchmark `B = 0.1`. Not a critical-field result. `B_cri` stays withheld. "
             "The |J|-max confirmation is not this result.", ""]
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
    lines.append("KM and KM_fid have 3+3L parameters (21 at L=6). QAOA has 2L (12 at L=6).")
    lines.append("")
    # replay
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
    lines.append("## Equal depth, best of two repeats")
    lines.append("")
    lines.append("| family | N | L | n | wins (joint / fidelity-only) | losses | ties | median dF [95% CI] | mean dF | sign p |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    detail = {}
    for fam in ("KM_fid", "KM"):
        for N in P["Ns"]:
            for L in (6, 5, 3, 1):
                rows = paired(ev, keys[N], fam, L)
                if not rows:
                    continue
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
                detail[(fam, N, L)] = {"wins": wins, "losses": losses, "joint": joint, "ci": ci, "n": len(rows),
                                       "median": float(np.median(dF)), "p": p}
                lines.append(f"| {fam} | {N} | {L} | {len(rows)} | {wins} ({joint} / {fonly}) | {losses} | {ties} | "
                             f"{np.median(dF):+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}] | {np.mean(dF):+.3f} | {p:.3g} |")
        lines.append("| | | | | | | | | | |")
    lines.append("")
    sec = P["secondary_equal_param"]
    lines.append(f"## Secondary equal parameter count, {sec['n_param']} parameters, not the verdict")
    lines.append("")
    lines.append("| family | N | wins (joint) | losses | median dF [95% CI] |")
    lines.append("|---|---|---|---|---|")
    for N in P["Ns"]:
        rows = []
        for key in keys[N]:
            d = ev[key]
            a, q = best_cell(d, "KM_fid", sec["KM_fid_L"]), best_cell(d, "qaoa", sec["qaoa_L"])
            if a and q:
                res, dF, _ = outcome(d, a, q)
                rows.append((res, dF))
        if not rows:
            continue
        dF = [r[1] for r in rows]
        res = [r[0] for r in rows]
        ci = boot_median(dF)
        wins = sum(r.startswith("win") for r in res)
        joint = sum(r == "win_joint" for r in res)
        losses = sum(r.startswith("loss") for r in res)
        lines.append(f"| KM_fid L=3 vs QAOA L=6 | {N} | {wins} ({joint}) | {losses} | {np.median(dF):+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}] |")
    lines.append("")
    # verdict
    scored = {}
    incomplete = False
    for N in P["Ns"]:
        if len(keys[N]) < C["min_complete"]:
            incomplete = True
            scored[N] = False
            continue
        st = detail[("KM_fid", N, C["L"])]
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
    # costs
    evals = 0
    capped = 0
    starts = 0
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
    lines.append("## Cost")
    lines.append("")
    lines.append(f"Optimizer wall {meta.get('wall_s', float('nan')):.1f} s of {P['optimize_wall_seconds']}. "
                 f"Starts {starts}, logged energy evaluations {evals}, L=1 grid evaluations {grid}, "
                 f"capped starts {capped}.")
    lines.append("")
    # per instance L=6
    lines.append("## L=6 instances")
    lines.append("")
    lines.append("| N | k | F KM | F Fiedler | F QAOA | dF Fiedler | E Fiedler - E QAOA | Fiedler outcome |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for N in P["Ns"]:
        for key in keys[N]:
            d = ev[key]
            km, pk, q = best_cell(d, "KM", 6), best_cell(d, "KM_fid", 6), best_cell(d, "qaoa", 6)
            if not (km and pk and q):
                continue
            res, dF, dE = outcome(d, pk, q)
            lines.append(f"| {N} | {d['k']} | {km['F_sub']:.3f} | {pk['F_sub']:.3f} | {q['F_sub']:.3f} | {dF:+.3f} | {dE:+.4f} | {res} |")
    text = "\n".join(lines) + "\n"
    (HERE / "RESULTS.md").write_text(text)
    (HERE / "verdict.json").write_text(json.dumps({
        "benchmark": "historical B=0.1, Fiedler packing",
        "overall": overall,
        "pass": {str(N): bool(scored[N]) for N in P["Ns"]},
        "n_complete": {str(N): len(keys[N]) for N in P["Ns"]},
    }, indent=2) + "\n")
    print(overall)


if __name__ == "__main__":
    main()
