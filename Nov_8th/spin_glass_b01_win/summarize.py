"""Tables for the historical B=0.1 window confirmation. Reads energy-selected cells.

A block interval is printed and is not a pass. The verdict is the frozen
overall rule on the pool plus both blocks.
"""
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
        rows.append((d["k"], res, dF, dE, a["F_sub"], q["F_sub"], a["E"], q["E"], a["n_param"], q["n_param"], a["capped"]))
    return rows


def tally(rows):
    dF = [r[2] for r in rows]
    res = [r[1] for r in rows]
    wins = sum(r.startswith("win") for r in res)
    joint = sum(r == "win_joint" for r in res)
    fonly = sum(r == "win_fidelity_only" for r in res)
    losses = sum(r.startswith("loss") for r in res)
    ties = sum(r.startswith("tie") for r in res)
    ci = boot_median(dF) if dF else (float("nan"), float("nan"))
    pos = sum(v > 0 for v in dF)
    neg = sum(v < 0 for v in dF)
    p = float(binomtest(pos, pos + neg, 0.5).pvalue) if pos + neg else float("nan")
    return {"n": len(rows), "wins": wins, "joint": joint, "fonly": fonly, "losses": losses, "ties": ties,
            "ci": ci, "median": float(np.median(dF)) if dF else float("nan"),
            "mean": float(np.mean(dF)) if dF else float("nan"), "p": p}


def fmt(st):
    return (f"{st['wins']} ({st['joint']} / {st['fonly']}) | {st['losses']} | {st['ties']} | "
            f"{st['median']:+.3f} [{st['ci'][0]:+.3f}, {st['ci'][1]:+.3f}] | {st['mean']:+.3f} | {st['p']:.3g}")


def main():
    ev, rp = load("eval"), load("replay")
    lines = ["# Historical B=0.1 window confirmation of interleaved KM", "",
             "Benchmark `B = 0.1`. Not a critical-field result. `B_cri` stays withheld. "
             "The circuit is the original interleaved shared-gamma KM. Block rows are controls. "
             "A block is not a result.", ""]
    groups = {
        "pool": list(P["k_confirm"]),
        "block_A": list(P["block_A"]),
        "block_B": list(P["block_B"]),
    }
    keys = {name: {} for name in groups}
    for name, ks in groups.items():
        for N in P["Ns"]:
            keys[name][N] = [(N, k) for k in ks if (N, k) in ev and complete(ev[(N, k)])]
    bad = [(n, k, ev[(n, k)]["statuses"]) for (n, k) in sorted(ev) if not complete(ev[(n, k)])]
    missing = [(N, k) for N in P["Ns"] for k in P["k_confirm"] if (N, k) not in ev]
    lines.append(f"Incomplete: {bad}. Never started: {missing}.")
    lines.append("")
    lines.append("| sample | N | complete |")
    lines.append("|---|---|---|")
    for name in groups:
        for N in P["Ns"]:
            lines.append(f"| {name} | {N} | {len(keys[name][N])}/{len(groups[name])} |")
    lines.append("")
    lines.append("KM has 3+3L parameters (21 at L=6). QAOA has 2L (12 at L=6).")
    lines.append("Both use the round-4 start list. KM has no extra starts. There is no second packing.")
    lines.append("")
    tol = P["replay_tol"]
    lines.append("## Independent replay")
    lines.append("")
    lines.append("| N | states | max rel dE | max abs dF | max abs dE0 | n_sub mismatches | max norm err | pass |")
    lines.append("|---|---|---|---|---|---|---|---|")
    replay_ok = True
    for N in P["Ns"]:
        ks = [key for key in rp if key[0] == N]
        cs = [c for key in ks for c in rp[key]["cells"]]
        if not cs:
            replay_ok = False
            lines.append(f"| {N} | 0 | | | | | | False |")
            continue
        mdE = max(abs(c["dE_rel"]) for c in cs)
        mdF = max(abs(c["dF_sub"]) for c in cs)
        md0 = max(abs(rp[key]["dE0"]) for key in ks)
        mism = sum(rp[key]["n_sub_replay"] != rp[key]["n_sub_main"] for key in ks)
        mn = max(c["norm_err"] for c in cs)
        ok = mdE <= tol["E_rel"] and mdF <= tol["F_sub"] and md0 <= tol["E0"] and mism == 0 and mn <= tol["norm"]
        replay_ok = replay_ok and ok
        lines.append(f"| {N} | {len(cs)} | {mdE:.1e} | {mdF:.1e} | {md0:.1e} | {mism} | {mn:.1e} | {ok} |")
    lines.append("")
    lines.append("## Equal depth, best of two repeats")
    lines.append("")
    lines.append("Block intervals are descriptive. They are not passes.")
    lines.append("")
    lines.append("| sample | N | L | n | wins (joint / fidelity-only) | losses | ties | median dF [95% CI] | mean dF | sign p |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    detail = {}
    for name in ("pool", "block_A", "block_B"):
        for N in P["Ns"]:
            for L in (6, 5, 3, 1):
                rows = paired(ev, keys[name][N], "KM", L)
                if not rows:
                    continue
                st = tally(rows)
                detail[(name, N, L)] = st
                lines.append(f"| {name} | {N} | {L} | {st['n']} | {fmt(st)} |")
        lines.append("| | | | | | | | | | |")
    lines.append("")
    sec = P["secondary_equal_param"]
    lines.append(f"## Secondary equal parameter count, {sec['n_param']} parameters, pool only, not the verdict")
    lines.append("")
    lines.append("| N | wins (joint) | losses | median dF [95% CI] |")
    lines.append("|---|---|---|---|")
    for N in P["Ns"]:
        rows = []
        for key in keys["pool"][N]:
            d = ev[key]
            a, q = best_cell(d, "KM", sec["KM_L"]), best_cell(d, "qaoa", sec["qaoa_L"])
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
        lines.append(f"| {N} | {wins} ({joint}) | {losses} | {np.median(dF):+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}] |")
    lines.append("")
    incomplete = False
    pool_pass, stable, clears = {}, {}, {}
    for N in P["Ns"]:
        enough = (len(keys["pool"][N]) >= C["min_complete_pool"]
                  and len(keys["block_A"][N]) >= C["min_complete_block"]
                  and len(keys["block_B"][N]) >= C["min_complete_block"])
        if not enough or ("pool", N, C["L"]) not in detail:
            incomplete = True
            pool_pass[N] = False
            stable[N] = False
            clears[N] = False
            continue
        pool = detail[("pool", N, C["L"])]
        ba = detail[("block_A", N, C["L"])]
        bb = detail[("block_B", N, C["L"])]
        pool_pass[N] = bool(pool["wins"] > pool["losses"] and pool["ci"][0] > 0)
        stable[N] = bool(ba["median"] > 0 and bb["median"] > 0 and ba["wins"] >= ba["losses"] and bb["wins"] >= bb["losses"])
        clears[N] = bool(pool_pass[N] and stable[N])
    if incomplete:
        overall = "incomplete"
    elif all(clears[N] for N in P["Ns"]):
        overall = "confirmed"
    elif clears[6]:
        overall = "partial"
    else:
        overall = "not_confirmed"
    lines.append("## Verdict")
    lines.append("")
    lines.append(f"Overall: **{overall}**.")
    lines.append("A size clears only when the pool passes and both blocks are window-stable. "
                 "Pool-pass alone is not the claim. A block is not the claim.")
    for N in P["Ns"]:
        lines.append(f"- N={N}: complete pool {len(keys['pool'][N])}/24, "
                     f"block A {len(keys['block_A'][N])}/12, block B {len(keys['block_B'][N])}/12, "
                     f"pool pass {pool_pass[N]}, window-stable {stable[N]}, clears {clears[N]}.")
    lines.append("")
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
    l6_capped = []
    for f in (HERE / "jobs").glob("n*_k*_r*.json"):
        doc = json.loads(f.read_text())
        grid += int(doc.get("grid_energy_evals", 0))
    for N in P["Ns"]:
        for key in keys["pool"][N]:
            cell = best_cell(ev[key], "KM", 6)
            if cell and cell["capped"]:
                l6_capped.append({"N": N, "k": ev[key]["k"], "F_sub": cell["F_sub"]})
    meta = json.loads((HERE / "run_meta.json").read_text()) if (HERE / "run_meta.json").exists() else {}
    lines.append("## Cost")
    lines.append("")
    lines.append(f"Optimizer wall {meta.get('wall_s', float('nan')):.1f} s of {P['optimize_wall_seconds']}. "
                 f"Starts {starts}, logged energy evaluations {evals}, L=1 grid evaluations {grid}, "
                 f"capped starts {capped}. Selected L=6 KM states that hit the evaluation cap: {l6_capped}.")
    lines.append("")
    lines.append("## L=6 instances")
    lines.append("")
    lines.append("| block | N | k | F KM | F QAOA | dF | E KM - E QAOA | outcome | KM capped |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for name, ks in (("A", P["block_A"]), ("B", P["block_B"])):
        for N in P["Ns"]:
            for k in ks:
                key = (N, k)
                if key not in keys["pool"][N]:
                    continue
                d = ev[key]
                km, q = best_cell(d, "KM", 6), best_cell(d, "qaoa", 6)
                if not (km and q):
                    continue
                res, dF, dE = outcome(d, km, q)
                lines.append(f"| {name} | {N} | {k} | {km['F_sub']:.3f} | {q['F_sub']:.3f} | "
                             f"{dF:+.3f} | {dE:+.4f} | {res} | {km['capped']} |")
    text = "\n".join(lines) + "\n"
    (HERE / "RESULTS.md").write_text(text)
    (HERE / "verdict.json").write_text(json.dumps({
        "benchmark": "historical B=0.1, interleaved shared-gamma KM, window confirmation",
        "overall": overall,
        "replay_ok": bool(replay_ok),
        "clears": {str(N): bool(clears[N]) for N in P["Ns"]},
        "pool_pass_not_the_claim": {str(N): bool(pool_pass[N]) for N in P["Ns"]},
        "window_stable": {str(N): bool(stable[N]) for N in P["Ns"]},
        "n_complete": {name: {str(N): len(keys[name][N]) for N in P["Ns"]} for name in groups},
        "L6": {f"{name}_N{N}": detail.get((name, N, 6), {}) for name in groups for N in P["Ns"]},
        "selected_L6_capped": l6_capped,
    }, indent=2) + "\n")
    print(overall)


if __name__ == "__main__":
    main()
