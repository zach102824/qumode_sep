"""S1ALT13: N15 L=2 1:3 vs PR88 1:1. Gate Nov3 S1ALT13−S1ALT11. Fair S1ALT13 Nov3 vs QAOA separate."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[1] / "results" / "STARTS"
RUNS = OUT / "runs" / "N15"
PREREG = json.loads((HERE / "PREREG_STARTS_S1ALT13.json").read_text())
SEEDS = list(PREREG["seeds"]["ids"])
ARMS = ("S1JOINT", "S1ALT11", "S1ALT13")
AB = {"S1ALT11": (510, 509), "S1ALT13": (255, 764)}
FORBIDDEN = set(range(10010, 10104)) | set(range(11000, 11032)) | set(range(20000, 21000))


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
    for p in sorted(RUNS.glob("s*_L2_*_S1*_Q2048.json")):
        r = json.loads(p.read_text())
        if r.get("seed") in SEEDS and r.get("stage") == "S1ALT13":
            rows.append(r)
    return rows


def audit_row(r):
    problems = []
    arm = r.get("arm")
    if r["seed"] in FORBIDDEN:
        problems.append("forbidden_seed")
    if r["seed"] not in SEEDS:
        problems.append("seed")
    if r.get("N") != 15 or r.get("L") != 2 or r.get("Q") != 2048:
        problems.append("batch")
    if r.get("band") != "exclusive_unread_dev":
        problems.append("band")
    if r.get("continuation"):
        problems.append("continuation")
    if r.get("status") == "ok":
        if r["queries"] != 2048 or r.get("outside_cap_queries", 0) != 0:
            problems.append("Q")
        if r.get("nstarts") != 1 or r.get("K_per_start") != 1019:
            problems.append("split")
        if (r.get("query_cat") or {}).get("calibration") != 8:
            problems.append("cal")
        if (r.get("query_cat") or {}).get("candidate") != 2:
            problems.append("cand")
        if any(m.get("gain_frozen") for m in r.get("start_meta") or []):
            problems.append("frozen")
        if r.get("reused"):
            problems.append("reused")
        want_opt = {"S1JOINT": "C0", "S1ALT11": "C0_ALT", "S1ALT13": "C0_ALT13"}
        if r.get("optimizer") != want_opt.get(arm):
            problems.append("optimizer")
        if arm in AB:
            m0 = (r.get("start_meta") or [{}])[0]
            na, nb = AB[arm]
            if m0.get("joint_prefix_K", 1) != 0:
                problems.append("prefix")
            if m0.get("n_block_A_steps") != na or m0.get("n_block_B_steps") != nb:
                problems.append("blocks")
            if arm == "S1ALT13" and m0.get("leftover_blocks") != ["A", "B", "B"]:
                problems.append("leftover")
    return problems


def index(rows, circ, arm):
    return {r["seed"]: r for r in rows
            if r["circuit"] == circ and r["arm"] == arm and r.get("status") == "ok"}


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
    inst = json.loads((OUT / "INSTANCES_S1ALT13.json").read_text()) if (OUT / "INSTANCES_S1ALT13.json").exists() else {"rows": []}
    rejected = [r for r in inst.get("rows", []) if not r.get("accepted")]
    rows = load_rows()
    audits = {f"{r['seed']}_{r['circuit']}_{r['arm']}": audit_row(r) for r in rows}
    bad = {k: v for k, v in audits.items() if v}
    ok = [r for r in rows if r.get("status") == "ok"]
    fail = [r for r in rows if r.get("status") != "ok"]
    charged = int(sum(r.get("queries", 0) for r in rows))
    accepted = {r["seed"] for r in inst.get("rows", []) if r.get("accepted")}
    expected, missing = [], []
    for seed in SEEDS:
        if seed not in accepted:
            continue
        for circ in ("Nov3", "QAOA"):
            for arm in ARMS:
                expected.append((seed, circ, arm))
    have = {(r["seed"], r["circuit"], r["arm"]) for r in rows}
    for t in expected:
        if t not in have:
            missing.append(t)
    cells, contrasts, nvq = {}, {}, {}
    for circ in ("Nov3", "QAOA"):
        for arm in ARMS:
            cells[f"{circ}_{arm}"] = cell_means(index(ok, circ, arm))
        contrasts[f"{circ}_S1ALT13_minus_S1ALT11"] = contrast(index(ok, circ, "S1ALT13"), index(ok, circ, "S1ALT11"))
        contrasts[f"{circ}_S1ALT13_minus_S1JOINT"] = contrast(index(ok, circ, "S1ALT13"), index(ok, circ, "S1JOINT"))
        contrasts[f"{circ}_S1ALT11_minus_S1JOINT"] = contrast(index(ok, circ, "S1ALT11"), index(ok, circ, "S1JOINT"))
    for arm in ARMS:
        nvq[f"Nov3_minus_QAOA_{arm}"] = contrast(index(ok, "Nov3", arm), index(ok, "QAOA", arm))
    gate_ct = contrasts["Nov3_S1ALT13_minus_S1ALT11"]["E"]
    lo, hi = gate_ct["ci95_mean"]
    adopt = bool(gate_ct["n"] > 0 and hi < 0)
    alt13_win = energy_nov3_win(nvq["Nov3_minus_QAOA_S1ALT13"])
    summary = {
        "experiment": "STARTS_S1ALT13",
        "n_runs": len(rows), "n_ok": len(ok), "n_failures": len(fail),
        "failures": [{"seed": r["seed"], "circuit": r["circuit"], "arm": r["arm"],
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
        "starts_s1alt23_ledger_preserved": 163840,
        "this_stream": "STARTS_S1ALT13",
        "cells": cells, "contrasts": contrasts, "nov3_vs_qaoa": nvq,
        "gate": {
            "cell": "N15 L2 Nov3",
            "metric": "mean CI of E_S1ALT13 - E_S1ALT11 entirely < 0",
            "n": gate_ct["n"], "mean": gate_ct["mean"], "median": gate_ct["median"],
            "ci95": gate_ct["ci95_mean"], "n_S1ALT13_lower_energy": gate_ct["n_neg"],
            "pass": adopt,
            "decision": "alt13_beats_alt11_at_L2" if adopt else "alt13_does_not_clear_alt11",
            "gr": "not patched",
        },
        "nov3_energy_superiority": {
            "S1JOINT": energy_nov3_win(nvq["Nov3_minus_QAOA_S1JOINT"]),
            "S1ALT11": energy_nov3_win(nvq["Nov3_minus_QAOA_S1ALT11"]),
            "S1ALT13": alt13_win,
            "rule": "claim only if mean Δ(E-E0)/N of Nov3-QAOA CI entirely below 0",
        },
        "seeds": SEEDS, "band": "exclusive_unread_dev",
    }
    (OUT / "summary_S1ALT13.json").write_text(json.dumps(summary, indent=1))
    lines = [
        "# STARTS S1ALT13: 1:3 shell-heavy vs PR88 1:1 at N15 L=2 Q=2048",
        "",
        "Exclusive unread seeds 10104–10111. Frozen S1JOINT and exact PR88 S1ALT11. Energy selects. F post hoc.",
        "S1ALT13: k%4==0 shared Uc/γ, else SNAP/β. Leftover continue_phase ABB. K=1019 → 255/764. GR not patched.",
        "",
        f"Runs {len(rows)} ok {len(ok)} failures {len(fail)}. Charged {charged}. "
        f"Outside cap {summary['outside_cap_queries']}. Rejected B* {len(rejected)}. Missing {len(missing)}.",
        "",
        f"Gate N15 L2 Nov3 mean ΔE (S1ALT13−S1ALT11) = {gate_ct['mean']:+.6f} med {gate_ct['median']:+.6f} "
        f"CI [{lo:+.6f}, {hi:+.6f}] n={gate_ct['n']}: **{summary['gate']['decision']}**.",
        "",
        "## Pairwise energy (first minus second)",
        "",
        "| circuit | contrast | Δ(E−E0)/N mean/med [CI] tails wins | ΔE | ΔF wins |",
        "|---|---|---|---|---|",
    ]
    for circ in ("Nov3", "QAOA"):
        for name in ("S1ALT13_minus_S1ALT11", "S1ALT13_minus_S1JOINT", "S1ALT11_minus_S1JOINT"):
            ct = contrasts[f"{circ}_{name}"]
            e, f = ct["E"], ct["F"]
            lines.append(
                f"| {circ} | {name} | {fmt(ct, 'E_err_over_N')} {e['n_neg']}/{e['n']} "
                f"| {fmt(ct, 'E')} | {fmt(ct, 'F')} {f['n_pos']}/{f['n']} |"
            )
    lines += [
        "",
        "## Fair Nov3 vs QAOA (not the 1:3-vs-1:1 gate)",
        "",
        "Energy-win candidate only if that CI lies entirely below 0.",
        "",
        "| arm | Δ(E−E0)/N Nov3−QAOA mean/med [CI] tails Nov3 lower | ΔF | energy-win candidate? |",
        "|---|---|---|---|",
    ]
    for arm in ARMS:
        ct = nvq[f"Nov3_minus_QAOA_{arm}"]
        sup = "yes" if energy_nov3_win(ct) else "no"
        lines.append(
            f"| {arm} | {fmt(ct, 'E_err_over_N')} {ct['E']['n_neg']}/{ct['E']['n']} "
            f"| {fmt(ct, 'F')} {ct['F']['n_pos']}/{ct['F']['n']} | {sup} |"
        )
    (OUT / "RESULTS_S1ALT13.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"gate": summary["gate"], "nov3_energy_superiority": summary["nov3_energy_superiority"],
                      "charged": charged, "ok": len(ok), "fail": len(fail),
                      "rejected": rejected, "missing": missing, "audit_failures": bad}, indent=1))
    if bad:
        raise SystemExit("audit failed")
    if missing:
        raise SystemExit(f"missing {missing}")


if __name__ == "__main__":
    main()
