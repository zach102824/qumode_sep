"""S1ALT23: N15 L=2,3 S1ALT vs S1JOINT. Gate per L. Fair S1ALT Nov3 vs QAOA separate."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[1] / "results" / "STARTS"
RUNS = OUT / "runs" / "N15"
PREREG = json.loads((HERE / "PREREG_STARTS_S1ALT23.json").read_text())
SEEDS = list(PREREG["seeds"]["ids"])
LS = [2, 3]
Q_OF = {2: 2048, 3: 3072}
KEXP = {2048: 1019, 3072: 1531}
AB = {1019: (510, 509), 1531: (766, 765)}
FORBIDDEN = set(range(10010, 10096)) | set(range(11000, 11032)) | set(range(20000, 21000))


def boot_ci(d, n=10000):
    d = np.asarray(d, float)
    if d.size == 0:
        return [float("nan"), float("nan")]
    r = np.random.default_rng(0)
    m = d[r.integers(0, len(d), size=(n, len(d)))].mean(axis=1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def paired_vals(xs):
    xs = np.asarray(xs, float)
    if len(xs) == 0:
        return {"n": 0, "mean": float("nan"), "median": float("nan"),
                "ci95_mean": [float("nan"), float("nan")],
                "n_pos": 0, "n_neg": 0, "n_zero": 0,
                "min": float("nan"), "max": float("nan"),
                "p10": float("nan"), "p90": float("nan"), "values": []}
    return {
        "n": int(len(xs)), "mean": float(xs.mean()), "median": float(np.median(xs)),
        "ci95_mean": boot_ci(xs), "n_pos": int(np.sum(xs > 0)),
        "n_neg": int(np.sum(xs < 0)), "n_zero": int(np.sum(xs == 0)),
        "min": float(xs.min()), "max": float(xs.max()),
        "p10": float(np.percentile(xs, 10)), "p90": float(np.percentile(xs, 90)),
        "values": [float(v) for v in xs],
    }


def load_rows():
    rows = []
    for p in sorted(RUNS.glob("s*_L[23]_*_S1*_Q*.json")):
        r = json.loads(p.read_text())
        if r.get("seed") in SEEDS and r.get("stage") == "S1ALT23":
            rows.append(r)
    return rows


def audit_row(r):
    problems = []
    arm, L, Q = r.get("arm"), r.get("L"), r.get("Q")
    if r["seed"] in FORBIDDEN:
        problems.append("forbidden_seed")
    if r["seed"] not in SEEDS:
        problems.append("seed")
    if r.get("N") != 15 or L not in LS or Q != Q_OF.get(L):
        problems.append("batch")
    if r.get("band") != "exclusive_unread_dev":
        problems.append("band")
    if r.get("continuation"):
        problems.append("continuation")
    if r.get("status") == "ok":
        if r["queries"] != Q or r.get("outside_cap_queries", 0) != 0:
            problems.append("Q")
        if r.get("nstarts") != 1 or r.get("K_per_start") != KEXP.get(Q):
            problems.append("split")
        if (r.get("query_cat") or {}).get("calibration") != 8:
            problems.append("cal")
        if (r.get("query_cat") or {}).get("candidate") != 2:
            problems.append("cand")
        if any(m.get("gain_frozen") for m in r.get("start_meta") or []):
            problems.append("frozen")
        if r.get("reused"):
            problems.append("reused")
        if arm == "S1JOINT" and r.get("optimizer") != "C0":
            problems.append("optimizer")
        if arm == "S1ALT":
            if r.get("optimizer") != "C0_ALT":
                problems.append("optimizer")
            m0 = (r.get("start_meta") or [{}])[0]
            na, nb = AB[KEXP[Q]]
            if m0.get("joint_prefix_K", 1) != 0:
                problems.append("prefix")
            if m0.get("n_block_A_steps") != na or m0.get("n_block_B_steps") != nb:
                problems.append("blocks")
    return problems


def index(rows, circ, arm, L):
    return {r["seed"]: r for r in rows
            if r["circuit"] == circ and r["arm"] == arm and r.get("L") == L and r.get("status") == "ok"}


def contrast(a, b):
    seeds = sorted(set(a) & set(b))
    rec = {"n": len(seeds), "seeds": seeds}
    for key in ("E_err_over_N", "E", "F"):
        xs = []
        for s in seeds:
            ra, rb = a[s], b[s]
            va = ra["E_err_over_N"] if key == "E_err_over_N" else (ra["diag"]["E"] if key == "E" else ra["diag"]["F"])
            vb = rb["E_err_over_N"] if key == "E_err_over_N" else (rb["diag"]["E"] if key == "E" else rb["diag"]["F"])
            xs.append(va - vb)
        rec[key] = paired_vals(xs)
    rec["n_energy_wins_a"] = rec["E"]["n_neg"]
    rec["n_F_wins_a"] = rec["F"]["n_pos"]
    return rec


def cell_means(idx):
    seeds = sorted(idx)
    e = np.array([idx[s]["E_err_over_N"] for s in seeds], float) if seeds else np.array([])
    f = np.array([idx[s]["diag"]["F"] for s in seeds], float) if seeds else np.array([])
    return {"n": len(seeds), "seeds": seeds, "E_err_over_N": paired_vals(e), "F": paired_vals(f)}


def fmt(ct, key):
    z = ct[key]
    return (f"{z['mean']:+.5f} med {z['median']:+.5f} "
            f"[{z['ci95_mean'][0]:+.5f},{z['ci95_mean'][1]:+.5f}] "
            f"tails [{z['min']:+.5f},{z['max']:+.5f}]")


def energy_nov3_win(ct):
    hi = ct["E_err_over_N"]["ci95_mean"][1]
    return bool(ct["n"] > 0 and hi < 0)


def main():
    inst = json.loads((OUT / "INSTANCES_S1ALT23.json").read_text()) if (OUT / "INSTANCES_S1ALT23.json").exists() else {"rows": []}
    rejected = [r for r in inst.get("rows", []) if not r.get("accepted")]
    rows = load_rows()
    audits = {f"{r['seed']}_{r['circuit']}_{r['arm']}_L{r['L']}": audit_row(r) for r in rows}
    bad = {k: v for k, v in audits.items() if v}
    ok = [r for r in rows if r.get("status") == "ok"]
    fail = [r for r in rows if r.get("status") != "ok"]
    charged = int(sum(r.get("queries", 0) for r in rows))
    accepted = {r["seed"] for r in inst.get("rows", []) if r.get("accepted")}
    expected, missing = [], []
    for seed in SEEDS:
        if seed not in accepted:
            continue
        for L in LS:
            for circ in ("Nov3", "QAOA"):
                for arm in ("S1JOINT", "S1ALT"):
                    expected.append((seed, circ, arm, L))
    have = {(r["seed"], r["circuit"], r["arm"], r["L"]) for r in rows}
    for t in expected:
        if t not in have:
            missing.append(t)
    cells, contrasts, nvq, gates = {}, {}, {}, {}
    for L in LS:
        for circ in ("Nov3", "QAOA"):
            for arm in ("S1JOINT", "S1ALT"):
                cells[f"{circ}_{arm}_L{L}"] = cell_means(index(ok, circ, arm, L))
            contrasts[f"{circ}_S1ALT_minus_S1JOINT_L{L}"] = contrast(
                index(ok, circ, "S1ALT", L), index(ok, circ, "S1JOINT", L))
        for arm in ("S1JOINT", "S1ALT"):
            nvq[f"Nov3_minus_QAOA_{arm}_L{L}"] = contrast(
                index(ok, "Nov3", arm, L), index(ok, "QAOA", arm, L))
        gct = contrasts[f"Nov3_S1ALT_minus_S1JOINT_L{L}"]["E"]
        lo, hi = gct["ci95_mean"]
        gates[L] = {
            "cell": f"N15 L{L} Nov3", "n": gct["n"], "mean": gct["mean"],
            "median": gct["median"], "ci95": gct["ci95_mean"],
            "n_S1ALT_lower_energy": gct["n_neg"],
            "pass": bool(gct["n"] > 0 and hi < 0),
        }
    both = all(gates[L]["pass"] for L in LS)
    summary = {
        "experiment": "STARTS_S1ALT23",
        "n_runs": len(rows), "n_ok": len(ok), "n_failures": len(fail),
        "failures": [{"seed": r["seed"], "circuit": r["circuit"], "arm": r["arm"], "L": r["L"],
                      "error": r.get("error")} for r in fail],
        "rejected_instances": rejected, "missing": missing,
        "queries_charged": charged,
        "outside_cap_queries": int(sum(r.get("outside_cap_queries", 0) for r in rows)),
        "audit_failures": bad,
        "prior_n3cond_ledger_preserved": 772224,
        "lexact_spsa_ledger_preserved": 2560,
        "depth_c1_ledger_preserved": 327680,
        "starts_s8_ledger_preserved": 98304,
        "starts_s1_ledger_preserved": 98304,
        "starts_s1c_ledger_preserved": 196608,
        "starts_s1l6_ledger_preserved": 196608,
        "starts_s1q_ledger_preserved": 344064,
        "starts_s1sh_ledger_preserved": 49152,
        "starts_s1alt_ledger_preserved": 65536,
        "this_stream": "STARTS_S1ALT23",
        "cells": cells, "contrasts": contrasts, "nov3_vs_qaoa": nvq,
        "gate": {
            "by_L": {str(L): gates[L] for L in LS},
            "both_pass": both,
            "decision": "s1alt_holds_at_L2_and_L3" if both else "s1alt_does_not_hold_at_both_L",
            "gr": "not patched",
        },
        "nov3_energy_superiority": {
            f"S1ALT_L{L}": energy_nov3_win(nvq[f"Nov3_minus_QAOA_S1ALT_L{L}"]) for L in LS
        } | {"rule": "claim only if mean Δ(E-E0)/N of Nov3-QAOA under S1ALT CI entirely below 0"},
        "seeds": SEEDS, "band": "exclusive_unread_dev",
    }
    (OUT / "summary_S1ALT23.json").write_text(json.dumps(summary, indent=1))
    lines = [
        "# STARTS S1ALT23: stacked S1 + PR88 ALT at N15 L=2,3 Q=qL",
        "",
        "Exclusive unread seeds 10096–10103. Frozen S1JOINT and exact PR88 S1ALT. Energy selects. F post hoc.",
        "Q=1024·L. K=1019 / 1531. Odd=Uc/γ even=SNAP/β. No JOINT prefix. GR not patched.",
        "",
        f"Runs {len(rows)} ok {len(ok)} failures {len(fail)}. Charged {charged}. "
        f"Outside cap {summary['outside_cap_queries']}. Rejected B* {len(rejected)}. Missing {len(missing)}.",
        "",
        f"Overall: **{summary['gate']['decision']}**.",
        "",
        "## Gate: Nov3 S1ALT − S1JOINT",
        "",
        "| L | Q | mean ΔE med [CI] tails wins(S1ALT) | Δ(E−E0)/N | pass |",
        "|---|---|---|---|---|",
    ]
    for L in LS:
        ct = contrasts[f"Nov3_S1ALT_minus_S1JOINT_L{L}"]
        g = gates[L]
        lines.append(
            f"| {L} | {Q_OF[L]} | {fmt(ct, 'E')} {ct['E']['n_neg']}/{ct['E']['n']} "
            f"| {fmt(ct, 'E_err_over_N')} | {'yes' if g['pass'] else 'no'} |"
        )
    lines += [
        "",
        "## S1ALT vs S1JOINT (both circuits)",
        "",
        "| L | circuit | Δ(E−E0)/N mean/med [CI] tails wins(S1ALT) | ΔF wins(S1ALT) |",
        "|---|---|---|---|",
    ]
    for L in LS:
        for circ in ("Nov3", "QAOA"):
            ct = contrasts[f"{circ}_S1ALT_minus_S1JOINT_L{L}"]
            lines.append(
                f"| {L} | {circ} | {fmt(ct, 'E_err_over_N')} {ct['E']['n_neg']}/{ct['E']['n']} "
                f"| {fmt(ct, 'F')} {ct['F']['n_pos']}/{ct['F']['n']} |"
            )
    lines += [
        "",
        "## Fair Nov3 vs QAOA under S1ALT (not the S1ALT-vs-S1JOINT gate)",
        "",
        "Energy-win candidate only if that CI lies entirely below 0.",
        "",
        "| L | arm | Δ(E−E0)/N Nov3−QAOA mean/med [CI] tails Nov3 lower | ΔF | energy-win candidate? |",
        "|---|---|---|---|---|",
    ]
    for L in LS:
        for arm in ("S1JOINT", "S1ALT"):
            ct = nvq[f"Nov3_minus_QAOA_{arm}_L{L}"]
            sup = "yes" if energy_nov3_win(ct) else "no"
            lines.append(
                f"| {L} | {arm} | {fmt(ct, 'E_err_over_N')} {ct['E']['n_neg']}/{ct['E']['n']} "
                f"| {fmt(ct, 'F')} {ct['F']['n_pos']}/{ct['F']['n']} | {sup} |"
            )
    (OUT / "RESULTS_S1ALT23.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"gate": summary["gate"], "nov3_energy_superiority": summary["nov3_energy_superiority"],
                      "charged": charged, "ok": len(ok), "fail": len(fail),
                      "rejected": rejected, "missing": missing, "audit_failures": bad}, indent=1))
    if bad:
        raise SystemExit("audit failed")
    if missing:
        raise SystemExit(f"missing {missing}")


if __name__ == "__main__":
    main()
