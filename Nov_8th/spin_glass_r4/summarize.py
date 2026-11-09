"""Development-set tables and the frozen advancement rule. Reads energy-selected cells only."""
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
W = P["win"]
A = P["advancement"]
R = P["repeats"]
FOCUS = ("KM", "KM_pack", "KM_gl", "qaoa")


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


def best_cell(d, fam, L, mode):
    cs = [c for c in d["cells"] if c["family"] == fam and c["L"] == L and (mode == "best" or c["r"] == mode)]
    return min(cs, key=lambda c: c["E"]) if cs else None


def outcome(d, a, q):
    FK, FQ = a["F_sub"], q["F_sub"]
    cl = 1 - W["clamp"]
    rho = float(np.log((1 - min(FQ, cl)) / (1 - min(FK, cl))))
    dF = FK - FQ
    joint = a["E"] <= q["E"] + 1e-6
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


def stats(ev, keys, fam, L):
    dF, res, joint_w, wins = [], [], 0, 0
    km_joint = km_wins = 0
    dF_km, mean_f, mean_km = [], [], []
    for key in keys:
        d = ev[key]
        a, q, km = best_cell(d, fam, L, "best"), best_cell(d, "qaoa", L, "best"), best_cell(d, "KM", L, "best")
        if not (a and q and km):
            continue
        r, df, _ = outcome(d, a, q)
        rk, dfk, _ = outcome(d, km, q)
        dF.append(df); dF_km.append(dfk); res.append(r)
        mean_f.append(a["F_sub"]); mean_km.append(km["F_sub"])
        wins += int(r.startswith("win")); km_wins += int(rk.startswith("win"))
        joint_w += int(r == "win_joint"); km_joint += int(rk == "win_joint")
    return {"n": len(dF), "median_dF": float(np.median(dF)) if dF else np.nan,
            "median_dF_KM": float(np.median(dF_km)) if dF_km else np.nan,
            "mean_F": float(np.mean(mean_f)) if mean_f else np.nan,
            "mean_F_KM": float(np.mean(mean_km)) if mean_km else np.nan,
            "wins": wins, "km_wins": km_wins, "joint": joint_w, "km_joint": km_joint,
            "losses": sum(r.startswith("loss") for r in res),
            "res": res, "dF": dF}


def advances(st, N):
    gain = st["median_dF"] - st["median_dF_KM"]
    if N in (9, 12):
        return bool(gain >= A["med_gain_min"] and st["mean_F"] >= st["mean_F_KM"] - A["mean_drop_max"]
                    and st["joint"] >= st["km_joint"] and st["wins"] >= st["km_wins"])
    return bool(gain >= -A["n6_drop_max"] and st["joint"] >= st["km_joint"])


def boot_median(v, seed=20261135):
    v = np.asarray(v, float)
    if v.size == 0:
        return (np.nan, np.nan)
    g = np.random.default_rng(seed)
    meds = np.median(v[g.integers(0, v.size, size=(10000, v.size))], axis=1)
    return (float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5)))


def main():
    ev, rp = load("eval"), load("replay")
    keys = {N: [(N, k) for k in P["k_dev"] if (N, k) in ev and complete(ev[(N, k)])] for N in P["Ns"]}
    bad = [(n, k, ev[(n, k)]["statuses"]) for (n, k) in sorted(ev) if not complete(ev[(n, k)])]
    missing = [(N, k) for N in P["Ns"] for k in P["k_dev"] if (N, k) not in ev]
    lines = ["# Spin glass round 4 (development set)", "",
             "Fixed instances k=4..15, two repeats. Advancement is decided by the frozen rule, not by inspection.", ""]
    lines.append("## Sample")
    lines.append("")
    lines.append("| N | complete dev instances |")
    lines.append("|---|---|")
    for N in P["Ns"]:
        lines.append(f"| {N} | {len(keys[N])}/12 |")
    lines.append("")
    lines.append(f"Incomplete: {bad}. Never started: {missing}.")
    lines.append("")
    # replay
    tol = P["replay_tol"]
    lines.append("## Independent replay")
    lines.append("")
    lines.append("| N | states | max rel dE | max abs dF | max abs dE0 | n_sub mismatches | max norm err | pass |")
    lines.append("|---|---|---|---|---|---|---|---|")
    gate = {}
    for N in P["Ns"]:
        ks = [key for key in rp if key[0] == N]
        cs = [c for key in ks for c in rp[key]["cells"]]
        if not cs:
            gate[N] = False
            lines.append(f"| {N} | 0 | | | | | | False |")
            continue
        mdE = max(abs(c["dE_rel"]) for c in cs)
        mdF = max(abs(c["dF_sub"]) for c in cs)
        md0 = max(abs(rp[key]["dE0"]) for key in ks)
        mism = sum(rp[key]["n_sub_replay"] != rp[key]["n_sub_main"] for key in ks)
        mn = max(c["norm_err"] for c in cs)
        ok = mdE <= tol["E_rel"] and mdF <= tol["F_sub"] and md0 <= tol["E0"] and mism == 0 and mn <= tol["norm"]
        gate[N] = bool(ok)
        lines.append(f"| {N} | {len(cs)} | {mdE:.1e} | {mdF:.1e} | {md0:.1e} | {mism} | {mn:.1e} | {ok} |")
    lines.append("")
    lines.append("## Mean / median F_sub, best over the two repeats")
    lines.append("")
    for N in P["Ns"]:
        lines.append(f"### N={N} (n={len(keys[N])})")
        lines.append("")
        lines.append("| L | " + " | ".join(FOCUS) + " |")
        lines.append("|---|" + "---|" * len(FOCUS))
        for L in range(1, 7):
            vals = []
            for f in FOCUS:
                xs = [best_cell(ev[key], f, L, "best")["F_sub"] for key in keys[N]]
                vals.append(f"{np.mean(xs):.3f} / {np.median(xs):.3f}" if xs else "")
            lines.append(f"| {L} | " + " | ".join(vals) + " |")
        lines.append("")
    lines.append("## Paired against QAOA at L=6 and L=5")
    lines.append("")
    lines.append("Win: log infidelity ratio above log 1.1 and fidelity difference above 0.005. Joint win also has energy at least as low.")
    lines.append("")
    lines.append("| family | N | L | n | wins (joint / F-only) | losses | ties | median dF [95% CI] | mean dF |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    detail = {}
    for fam in ("KM", "KM_pack", "KM_gl"):
        for N in P["Ns"]:
            for L in (6, 5, 3, 1):
                rows = []
                for key in keys[N]:
                    a, q = best_cell(ev[key], fam, L, "best"), best_cell(ev[key], "qaoa", L, "best")
                    if a and q:
                        rows.append(outcome(ev[key], a, q))
                if not rows:
                    continue
                res = [r[0] for r in rows]
                dF = [r[1] for r in rows]
                wj = sum(r == "win_joint" for r in res)
                wf = sum(r == "win_fidelity_only" for r in res)
                lo, nwin = sum(r.startswith("loss") for r in res), wj + wf
                p = binomtest(nwin, nwin + lo, 0.5, alternative="greater").pvalue if nwin + lo else 1.0
                ci = boot_median(dF)
                lines.append(f"| {fam} | {N} | {L} | {len(rows)} | {nwin} ({wj} / {wf}) | {lo} | {len(rows) - nwin - lo} | "
                             f"{np.median(dF):+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}] | {np.mean(dF):+.3f} |")
                if L == 6:
                    detail[(fam, N)] = {"p_sign": float(p), "median_dF": float(np.median(dF)), "ci": ci,
                                        "joint": wj, "fid_only": wf, "losses": lo}
        lines.append("| | | | | | | | | |")
    lines.append("")
    lines.append("## Frozen advancement rule, L=6")
    lines.append("")
    decision = {}
    for fam in A["variants"]:
        per = {}
        ok_big, ok6 = False, True
        for N in P["Ns"]:
            st = stats(ev, keys[N], fam, 6)
            ok = advances(st, N) if st["n"] else False
            per[N] = {**st, "pass": ok, "gain": st["median_dF"] - st["median_dF_KM"]}
            if N in (9, 12):
                ok_big = ok_big or ok
            else:
                ok6 = ok
            lines.append(f"- {fam} N={N}: median dF {st['median_dF']:+.3f} vs KM {st['median_dF_KM']:+.3f} "
                         f"(gain {per[N]['gain']:+.3f}); mean F {st['mean_F']:.3f} vs {st['mean_F_KM']:.3f}; "
                         f"wins {st['wins']} vs {st['km_wins']}; joint {st['joint']} vs {st['km_joint']}; "
                         f"{'pass' if ok else 'fail'}")
        decision[fam] = {"advance": bool(ok_big and ok6 and all(gate.values())), "per_N": {str(n): {"pass": v["pass"], "gain": v["gain"]} for n, v in per.items()}}
        lines.append(f"- **{fam}: {'advances' if decision[fam]['advance'] else 'does not advance'}** (replay gate {'pass' if all(gate.values()) else 'fail'})")
        lines.append("")
    adv = [f for f, v in decision.items() if v["advance"]]
    if len(adv) == 2:
        g = {f: sum(decision[f]["per_N"][str(N)]["gain"] for N in P["Ns"]) for f in adv}
        chosen = "KM_pack" if g["KM_pack"] >= g["KM_gl"] else "KM_gl"
    else:
        chosen = adv[0] if adv else None
    lines.append(f"Chosen for a confirmation round: **{chosen if chosen else 'none'}**.")
    lines.append("")
    lines.append("## Per instance at L=6, best over repeats")
    lines.append("")
    lines.append("| N | k | KM | KM_pack | KM_gl | QAOA | dF pack | dF gl | E pack - E qaoa | E gl - E qaoa | KM outcome | pack outcome | gl outcome |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for N in P["Ns"]:
        for key in keys[N]:
            d = ev[key]
            c = {f: best_cell(d, f, 6, "best") for f in FOCUS}
            if not all(c.values()):
                continue
            op = outcome(d, c["KM_pack"], c["qaoa"])
            og = outcome(d, c["KM_gl"], c["qaoa"])
            okm = outcome(d, c["KM"], c["qaoa"])
            lines.append(f"| {N} | {key[1]} | {c['KM']['F_sub']:.3f} | {c['KM_pack']['F_sub']:.3f} | {c['KM_gl']['F_sub']:.3f} | "
                         f"{c['qaoa']['F_sub']:.3f} | {op[1]:+.3f} | {og[1]:+.3f} | {op[2]:+.3f} | {og[2]:+.3f} | {okm[0]} | {op[0]} | {og[0]} |")
    lines.append("")
    # reliability and bounds
    lines.append("## Repeats and bounds")
    lines.append("")
    lines.append("| N | family | L6 repeats at best energy | instances with energy spread > 1e-4 | selected cells on a bound, any L | max diagonal fraction of the box |")
    lines.append("|---|---|---|---|---|---|")
    for N in P["Ns"]:
        for f in FOCUS:
            at = spread = touch = 0
            frac = 0.0
            n = len(keys[N])
            for key in keys[N]:
                reps = [best_cell(ev[key], f, 6, r) for r in range(R)]
                if any(x is None for x in reps):
                    continue
                Es = np.array([c["E"] for c in reps])
                at += int(np.sum(Es - Es.min() <= 1e-6))
                spread += int(Es.max() - Es.min() > 1e-4)
            for key in keys[N]:
                for c in ev[key]["cells"]:
                    if c["family"] == f:
                        frac = max(frac, c["max_diag_frac_of_box"])
                        touch += int(c["touch_bound"])
            lines.append(f"| {N} | {f} | {at}/{R * n} | {spread}/{n} | {touch} | {frac:.3f} |")
    lines.append("")
    viol = []
    for key in [k for N in P["Ns"] for k in keys[N]]:
        for r in range(R):
            for f in ("KM", "KM_pack", "KM_gl"):
                for L in range(2, 7):
                    a, b = best_cell(ev[key], f, L, r), best_cell(ev[key], f, L - 1, r)
                    if a and b and a["E"] > b["E"] + 1e-9:
                        viol.append((key[0], key[1], r, f, L))
    lines.append(f"Energy rose with depth inside a repeat: {len(viol)} {viol[:12]}.")
    lines.append("")
    # accounting from logs
    acc = {}
    for f in (HERE / "logs").glob("*.jsonl"):
        for line in f.read_text().splitlines():
            row = json.loads(line)
            a = acc.setdefault((row["N"], row["family"]), [0, 0, 0])
            a[0] += 1; a[1] += row["n_eval"]; a[2] += int(row["capped"])
    grid = 0
    for f in (HERE / "jobs").glob("*.json"):
        grid += json.loads(f.read_text()).get("grid_energy_evals", 0)
    lines.append("## Accounting")
    lines.append("")
    lines.append("| N | family | starts | optimizer evals | capped |")
    lines.append("|---|---|---|---|---|")
    for (N, f), a in sorted(acc.items()):
        lines.append(f"| {N} | {f} | {a[0]} | {a[1]} | {a[2]} |")
    lines.append("")
    meta = json.loads((HERE / "run_meta.json").read_text()) if (HERE / "run_meta.json").exists() else {}
    lines.append(f"Grid energy evaluations, not inside the optimizer count: {grid}.")
    lines.append(f"Optimization wall {meta.get('wall_s', float('nan')):.0f} s of {P['optimize_wall_seconds']} s.")
    lines.append(f"Gradient check: {meta.get('grad_check', {})}.")
    (HERE / "RESULTS.md").write_text("\n".join(lines) + "\n")
    (HERE / "advancement.json").write_text(json.dumps({"decision": decision, "chosen": chosen,
                                                       "replay_gate": {str(k): v for k, v in gate.items()}}, indent=2) + "\n")
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.2), sharey=True)
    for ax, N in zip(axes, P["Ns"]):
        for f, mk in (("KM", "o"), ("KM_pack", "s"), ("KM_gl", "^"), ("qaoa", "x")):
            ys = []
            for L in range(1, 7):
                xs = [best_cell(ev[key], f, L, "best")["F_sub"] for key in keys[N]]
                ys.append(np.mean(xs) if xs else np.nan)
            ax.plot(range(1, 7), ys, marker=mk, label=f)
        ax.set_title(f"N={N}, n={len(keys[N])}"); ax.set_xlabel("L")
    axes[0].set_ylabel("mean F_sub"); axes[0].legend(frameon=False, fontsize=7)
    fig.tight_layout(); fig.savefig(HERE / "r4_fsub.png", dpi=120); plt.close(fig)
    print(chosen, {f: decision[f]["advance"] for f in decision})


if __name__ == "__main__":
    main()
