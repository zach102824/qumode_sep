"""Confirmation tables for KM_gl. The claim rule is read from PREREG.json."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binomtest

HERE = Path(__file__).resolve().parent
P = json.loads((HERE / "PREREG.json").read_text())
W, C, R = P["win"], P["confirmation"], P["repeats"]
FOCUS = ("KM", "KM_gl", "qaoa")


def load(name):
    out = {}
    d = HERE / name
    if d.exists():
        for f in d.glob("n*_k*.json"):
            doc = json.loads(f.read_text())
            out[(doc["N"], doc["k"])] = doc
    return out


def complete(d):
    return len(d["statuses"]) == R and all(v == "complete" for v in d["statuses"].values())


def best(d, fam, L, mode):
    cs = [c for c in d["cells"] if c["family"] == fam and c["L"] == L and (mode == "best" or c["r"] == mode)]
    return min(cs, key=lambda c: c["E"]) if cs else None


def outcome(d, a, q):
    FK, FQ = a["F_sub"], q["F_sub"]
    cl = 1 - W["clamp"]
    rho = float(np.log((1 - min(FQ, cl)) / (1 - min(FK, cl))))
    dF, joint = FK - FQ, a["E"] <= q["E"] + 1e-6
    if not d["even_ground_in_window"]:
        res = "tie_unreachable"
    elif max(FK, FQ) < W["floor"]:
        res = "tie_both_failed"
    elif rho > np.log(W["ratio"]) and dF > W["dF_min"]:
        res = "win_joint" if joint else "win_fidelity_only"
    elif rho < -np.log(W["ratio"]) and dF < -W["dF_min"]:
        res = "loss_joint" if q["E"] <= a["E"] + 1e-6 else "loss_fidelity_only"
    else:
        res = "tie"
    return res, dF, a["E"] - q["E"]


def holm(ps):
    order = np.argsort(ps)
    adj, run = np.zeros(len(ps)), 0.0
    for i, idx in enumerate(order):
        run = max(run, (len(ps) - i) * ps[idx])
        adj[idx] = min(1.0, run)
    return adj


def boot(v):
    v = np.asarray(v, float)
    if v.size == 0:
        return (float("nan"), float("nan"))
    g = np.random.default_rng(C["bootstrap_seed"])
    meds = np.median(v[g.integers(0, v.size, size=(C["bootstrap"], v.size))], axis=1)
    return float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))


def paired(ev, keys, fam, L, mode="best"):
    rows = []
    for key in keys:
        a, q = best(ev[key], fam, L, mode), best(ev[key], "qaoa", L, mode)
        if a and q:
            rows.append(outcome(ev[key], a, q))
    res = [r[0] for r in rows]
    dF = np.array([r[1] for r in rows])
    wj = sum(r == "win_joint" for r in res)
    wf = sum(r == "win_fidelity_only" for r in res)
    lo = sum(r.startswith("loss") for r in res)
    p = binomtest(wj + wf, wj + wf + lo, 0.5, alternative="greater").pvalue if wj + wf + lo else 1.0
    return {"n": len(rows), "wins": wj + wf, "joint": wj, "fid_only": wf, "losses": lo,
            "ties": len(rows) - (wj + wf) - lo, "p": float(p),
            "median": float(np.median(dF)) if len(dF) else float("nan"),
            "mean": float(np.mean(dF)) if len(dF) else float("nan"), "ci": boot(dF), "dF": dF}


def main():
    ev, rp = load("eval"), load("replay")
    keys = {N: [(N, k) for k in P["k_confirm"] if (N, k) in ev and complete(ev[(N, k)])] for N in P["Ns"]}
    bad = [(n, k, ev[(n, k)]["statuses"]) for (n, k) in sorted(ev) if not complete(ev[(n, k)])]
    missing = [(N, k) for N in P["Ns"] for k in P["k_confirm"] if (N, k) not in ev]
    L = ["# Spin glass round 5: confirmation of KM_gl", "",
         "Instances k=28..43 were not fit in round 4. KM_gl is the per-layer family, not the retained KM.", ""]
    L.append("## Sample")
    L.append("")
    L.append("| N | complete |")
    L.append("|---|---|")
    for N in P["Ns"]:
        L.append(f"| {N} | {len(keys[N])}/16 |")
    L.append("")
    L.append(f"Incomplete: {bad}. Never started: {missing}.")
    L.append("")
    tol, gate = P["replay_tol"], {}
    L += ["## Independent replay", "",
          "| N | states | max rel dE | max abs dF | max abs dE0 | n_sub mismatches | max norm err | pass |",
          "|---|---|---|---|---|---|---|---|"]
    for N in P["Ns"]:
        ks = [key for key in rp if key[0] == N]
        cs = [c for key in ks for c in rp[key]["cells"]]
        if not cs:
            gate[N] = False
            L.append(f"| {N} | 0 | | | | | | False |")
            continue
        mdE, mdF = max(abs(c["dE_rel"]) for c in cs), max(abs(c["dF_sub"]) for c in cs)
        md0 = max(abs(rp[key]["dE0"]) for key in ks)
        mism = sum(rp[key]["n_sub_replay"] != rp[key]["n_sub_main"] for key in ks)
        mn = max(c["norm_err"] for c in cs)
        ok = mdE <= tol["E_rel"] and mdF <= tol["F_sub"] and md0 <= tol["E0"] and mism == 0 and mn <= tol["norm"]
        gate[N] = bool(ok)
        L.append(f"| {N} | {len(cs)} | {mdE:.1e} | {mdF:.1e} | {md0:.1e} | {mism} | {mn:.1e} | {ok} |")
    L.append("")
    L.append("## Mean / median F_sub, best over repeats")
    L.append("")
    for N in P["Ns"]:
        L += [f"### N={N} (n={len(keys[N])})", "", "| L | " + " | ".join(FOCUS) + " |", "|---|" + "---|" * len(FOCUS)]
        for Lv in range(1, 7):
            vals = []
            for f in FOCUS:
                xs = [best(ev[key], f, Lv, "best")["F_sub"] for key in keys[N]]
                vals.append(f"{np.mean(xs):.3f} / {np.median(xs):.3f}" if xs else "")
            L.append(f"| {Lv} | " + " | ".join(vals) + " |")
        L.append("")
    L += ["## Paired differences against QAOA", "",
          "| family | N | L | n | wins (joint / F-only) | losses | ties | median dF [95% CI] | mean dF | sign p |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    cache = {}
    for fam in ("KM", "KM_gl"):
        for N in P["Ns"]:
            for Lv in (6, 5, 3, 1):
                s = paired(ev, keys[N], fam, Lv)
                if not s["n"]:
                    continue
                L.append(f"| {fam} | {N} | {Lv} | {s['n']} | {s['wins']} ({s['joint']} / {s['fid_only']}) | {s['losses']} | "
                         f"{s['ties']} | {s['median']:+.3f} [{s['ci'][0]:+.3f}, {s['ci'][1]:+.3f}] | {s['mean']:+.3f} | {s['p']:.3g} |")
                cache[(fam, N, Lv, "best")] = s
            for r in range(R):
                cache[(fam, N, 6, r)] = paired(ev, keys[N], fam, 6, r)
        L.append("| | | | | | | | | | |")
    ps = [cache[("KM_gl", N, 6, "best")]["p"] for N in P["Ns"]]
    adj = holm(ps) if all(("KM_gl", N, 6, "best") in cache for N in P["Ns"]) else [1, 1, 1]
    L += ["", "## Frozen confirmation rule", ""]
    verdict = {}
    for i, N in enumerate(P["Ns"]):
        g, km = cache.get(("KM_gl", N, 6, "best")), cache.get(("KM", N, 6, "best"))
        g5 = cache.get(("KM_gl", N, 5, "best"))
        reps = [cache.get(("KM_gl", N, 6, r)) for r in range(R)]
        if not g or not km or not g5:
            verdict[N] = False
            L.append(f"- N={N}: incomplete, not confirmed")
            continue
        checks = {
            "holm_p": bool(adj[i] < 0.05),
            "ci": bool(g["ci"][0] > 0),
            "beats_KM_median": bool(g["median"] >= km["median"]),
            "joint_ge_losses": bool(g["joint"] >= g["losses"]),
            "L5": bool(g5["median"] > 0 and g5["wins"] >= g5["losses"]),
            "repeats": bool(all(r and r["median"] > 0 for r in reps)),
            "replay": bool(gate.get(N, False)),
        }
        verdict[N] = bool(all(checks.values()))
        L.append(f"- N={N}: Holm p {adj[i]:.3g}; median dF {g['median']:+.3f} [{g['ci'][0]:+.3f}, {g['ci'][1]:+.3f}] "
                 f"vs KM {km['median']:+.3f}; joint {g['joint']} losses {g['losses']} F-only {g['fid_only']}; "
                 f"L5 median {g5['median']:+.3f} wins {g5['wins']} losses {g5['losses']}; "
                 f"repeat medians {[None if r is None else round(r['median'], 3) for r in reps]}; "
                 f"checks {checks}; **{'confirmed' if verdict[N] else 'not confirmed'}**")
    n_ok = sum(verdict.values())
    big = verdict.get(9, False) or verdict.get(12, False)
    overall = "confirmed" if n_ok == 3 else ("partial" if verdict.get(6) and big else "not confirmed")
    L += ["", f"Overall: **{overall}**.", ""]
    L += ["## Per instance at L=6", "",
          "| N | k | KM | KM_gl | QAOA | dF gl | E gl - E qaoa | E gl - E KM | gl outcome | gl F per repeat |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    for N in P["Ns"]:
        for key in keys[N]:
            d = ev[key]
            c = {f: best(d, f, 6, "best") for f in FOCUS}
            if not all(c.values()):
                continue
            res, dF, dE = outcome(d, c["KM_gl"], c["qaoa"])
            per = "/".join(f"{best(d, 'KM_gl', 6, r)['F_sub']:.3f}" for r in range(R))
            L.append(f"| {N} | {key[1]} | {c['KM']['F_sub']:.3f} | {c['KM_gl']['F_sub']:.3f} | {c['qaoa']['F_sub']:.3f} | "
                     f"{dF:+.3f} | {dE:+.3f} | {c['KM_gl']['E'] - c['KM']['E']:+.4f} | {res} | {per} |")
    L.append("")
    acc = {}
    for f in (HERE / "logs").glob("*.jsonl"):
        for line in f.read_text().splitlines():
            row = json.loads(line)
            a = acc.setdefault((row["N"], row["family"]), [0, 0, 0])
            a[0] += 1; a[1] += row["n_eval"]; a[2] += int(row["capped"])
    L += ["## Accounting", "", "| N | family | starts | optimizer evals | capped |", "|---|---|---|---|---|"]
    for key, a in sorted(acc.items()):
        L.append(f"| {key[0]} | {key[1]} | {a[0]} | {a[1]} | {a[2]} |")
    meta = json.loads((HERE / "run_meta.json").read_text()) if (HERE / "run_meta.json").exists() else {}
    L.append("")
    L.append(f"Optimization wall {meta.get('wall_s', float('nan')):.0f} s of {P['optimize_wall_seconds']} s.")
    (HERE / "RESULTS.md").write_text("\n".join(L) + "\n")
    (HERE / "verdict.json").write_text(json.dumps({"per_N": {str(k): v for k, v in verdict.items()}, "overall": overall,
                                                   "replay_gate": {str(k): v for k, v in gate.items()}}, indent=2) + "\n")
    fig, axes = plt.subplots(1, 4, figsize=(12, 3.2))
    for ax, N in zip(axes[:3], P["Ns"]):
        for f, mk in (("KM", "o"), ("KM_gl", "^"), ("qaoa", "x")):
            ys = [np.mean([best(ev[key], f, Lv, "best")["F_sub"] for key in keys[N]]) if keys[N] else np.nan for Lv in range(1, 7)]
            ax.plot(range(1, 7), ys, marker=mk, label=f)
        ax.set_title(f"N={N}, n={len(keys[N])}"); ax.set_xlabel("L")
    axes[0].set_ylabel("mean F_sub"); axes[0].legend(frameon=False, fontsize=7)
    ax = axes[3]
    for N, mk in zip(P["Ns"], ("o", "s", "^")):
        pts = [(best(ev[key], "qaoa", 6, "best")["F_sub"], best(ev[key], "KM_gl", 6, "best")["F_sub"]) for key in keys[N]]
        if pts:
            ax.scatter(*zip(*pts), marker=mk, s=16, label=f"N={N}")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("QAOA F_sub"); ax.set_ylabel("KM_gl F_sub"); ax.legend(frameon=False, fontsize=7)
    fig.tight_layout(); fig.savefig(HERE / "r5_fsub.png", dpi=120); plt.close(fig)
    print(overall, verdict)


if __name__ == "__main__":
    main()
