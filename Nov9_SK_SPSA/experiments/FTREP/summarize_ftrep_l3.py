"""FTREP L3 gates: N15 and N12 Nov3−QAOA mean ΔE CI entirely <0. FT bootstrap."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE.parents[1] / "results" / "FTREP"
PREREG = json.loads((HERE / "PREREG_FTREP_L3.json").read_text())
SEEDS = list(PREREG["seeds"]["ids"])
NS = [12, 15]
ARMS = ("nov3_center4", "qaoa_center4")
FORBIDDEN = set(range(10010, 10192)) | set(range(11000, 11032)) | set(range(20000, 21000))
N_BOOT = 4000


def boot_ci(d, rng):
    d = np.asarray(d, float)
    if d.size == 0:
        return [float("nan"), float("nan")]
    m = d[rng.integers(0, len(d), size=(N_BOOT, len(d)))].mean(axis=1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def paired_vals(xs, rng):
    xs = np.asarray(xs, float)
    if len(xs) == 0:
        return {"n": 0, "mean": float("nan"), "median": float("nan"),
                "ci95_mean": [float("nan"), float("nan")],
                "n_pos": 0, "n_neg": 0, "n_zero": 0,
                "min": float("nan"), "max": float("nan"),
                "p10": float("nan"), "p90": float("nan"), "values": []}
    return {
        "n": int(len(xs)), "mean": float(xs.mean()), "median": float(np.median(xs)),
        "ci95_mean": boot_ci(xs, rng), "n_pos": int(np.sum(xs > 0)),
        "n_neg": int(np.sum(xs < 0)), "n_zero": int(np.sum(xs == 0)),
        "min": float(xs.min()), "max": float(xs.max()),
        "p10": float(np.percentile(xs, 10)), "p90": float(np.percentile(xs, 90)),
        "values": [float(v) for v in xs],
    }


def load_rows():
    rows = []
    for nq in NS:
        d = OUT / "runs" / f"N{nq}"
        if not d.exists():
            continue
        for p in sorted(d.glob("s*_L3_*_Q3072.json")):
            r = json.loads(p.read_text())
            if r.get("seed") in SEEDS and r.get("stage") == "FTREP_L3":
                rows.append(r)
    return rows


def audit_row(r):
    problems = []
    if r["seed"] in FORBIDDEN:
        problems.append("forbidden_seed")
    if r["seed"] not in SEEDS:
        problems.append("seed")
    if r.get("N") not in NS or r.get("L") != 3 or r.get("Q") != 3072:
        problems.append("batch")
    if r.get("band") != "exclusive_unread_dev":
        problems.append("band")
    if r.get("continuation"):
        problems.append("continuation")
    if r.get("optimizer") != "FT_v4_center4":
        problems.append("optimizer")
    if r.get("status") == "ok":
        if r["queries"] != 3072 or r.get("outside_cap_queries", 0) != 0:
            problems.append("Q")
        if r.get("nstarts") != 4 or r.get("K_per_start") != 378:
            problems.append("split")
        qc = r.get("query_cat") or {}
        if qc.get("init") != 4 or qc.get("calibration") != 32:
            problems.append("cal")
        if qc.get("perturb") != 3024 or qc.get("candidate") != 12:
            problems.append("cand")
        if qc.get("monitor", 0) != 0 or qc.get("screen", 0) != 0:
            problems.append("extra")
        if r.get("reused"):
            problems.append("reused")
        want = "QAOA" if r.get("arm") == "qaoa_center4" else "Nov3"
        if r.get("circuit") != want:
            problems.append("circuit")
    return problems


def index(rows, nq, arm):
    return {r["seed"]: r for r in rows
            if r["N"] == nq and r["arm"] == arm and r.get("status") == "ok"}


def contrast(a, b, rng):
    seeds = sorted(set(a) & set(b))
    rec = {"n": len(seeds), "seeds": seeds}
    for key in ("E_err_over_N", "E", "F"):
        xs = []
        for s in seeds:
            ra, rb = a[s], b[s]
            va = ra["E_err_over_N"] if key == "E_err_over_N" else (ra["diag"]["E"] if key == "E" else ra["diag"]["F"])
            vb = rb["E_err_over_N"] if key == "E_err_over_N" else (rb["diag"]["E"] if key == "E" else rb["diag"]["F"])
            xs.append(va - vb)
        rec[key] = paired_vals(xs, rng)
    rec["n_energy_wins_a"] = rec["E"]["n_neg"]
    rec["n_F_wins_a"] = rec["F"]["n_pos"]
    rec["n_start0_nov3"] = int(sum(a[s].get("sel_start") == 0 for s in seeds))
    rec["n_start0_qaoa"] = int(sum(b[s].get("sel_start") == 0 for s in seeds))
    return rec


def fmt(ct, key):
    z = ct[key]
    return (f"{z['mean']:+.5f} med {z['median']:+.5f} "
            f"[{z['ci95_mean'][0]:+.5f},{z['ci95_mean'][1]:+.5f}] "
            f"tails [{z['min']:+.5f},{z['max']:+.5f}]")


def pass_below(ct):
    return bool(ct["n"] > 0 and ct["E"]["ci95_mean"][1] < 0)


def main():
    inst = json.loads((OUT / "INSTANCES_FTREP_L3.json").read_text()) if (OUT / "INSTANCES_FTREP_L3.json").exists() else {"rows": []}
    rejected = [r for r in inst.get("rows", []) if not r.get("accepted")]
    rows = load_rows()
    audits = {f"N{r['N']}_s{r['seed']}_{r['arm']}": audit_row(r) for r in rows}
    bad = {k: v for k, v in audits.items() if v}
    ok = [r for r in rows if r.get("status") == "ok"]
    fail = [r for r in rows if r.get("status") != "ok"]
    charged = int(sum(r.get("queries", 0) for r in rows))
    accepted = {(r["N"], r["seed"]) for r in inst.get("rows", []) if r.get("accepted")}
    expected, missing = [], []
    for nq in NS:
        for seed in SEEDS:
            if (nq, seed) not in accepted:
                continue
            for arm in ARMS:
                expected.append((nq, seed, arm))
    have = {(r["N"], r["seed"], r["arm"]) for r in rows}
    for t in expected:
        if t not in have:
            missing.append(t)
    contrasts, gates = {}, {}
    rng_gate = np.random.default_rng(3334)
    for nq in NS:
        contrasts[f"N{nq}"] = contrast(index(ok, nq, "nov3_center4"), index(ok, nq, "qaoa_center4"), rng_gate)
        gates[nq] = pass_below(contrasts[f"N{nq}"])
    both = bool(gates[15] and gates[12])
    status = json.loads((OUT / "STATUS_FTREP_L3.json").read_text()) if (OUT / "STATUS_FTREP_L3.json").exists() else {}
    wall_fit = float(sum(r.get("wall_s", 0) for r in rows))
    cpu_fit = float(sum(r.get("cpu_s", 0) for r in rows if r.get("cpu_s") is not None))
    rss = [r.get("peak_rss_kib") for r in rows if r.get("peak_rss_kib") is not None]
    summary = {
        "experiment": "FTREP_L3",
        "n_runs": len(rows), "n_ok": len(ok), "n_failures": len(fail),
        "failures": [{"N": r["N"], "seed": r["seed"], "arm": r["arm"], "error": r.get("error")} for r in fail],
        "rejected_instances": rejected, "missing": missing,
        "queries_charged": charged,
        "outside_cap_queries": int(sum(r.get("outside_cap_queries", 0) for r in rows)),
        "audit_failures": bad,
        "bstar_wall_s": status.get("bstar_wall_s"),
        "fit_wall_s": status.get("fit_wall_s"),
        "sum_cell_wall_s": wall_fit,
        "sum_cell_cpu_s": cpu_fit,
        "peak_rss_kib_max": int(max(rss)) if rss else None,
        "prior_n3cond_ledger_preserved": 772224,
        "ftrep_n12n15_ledger_preserved": 65536,
        "ftrep_n18_ledger_preserved": 32768,
        "this_stream": "FTREP_L3",
        "contrasts": contrasts,
        "gate": {
            "L3-15": gates[15], "L3-12": gates[12], "both_pass": both,
            "N15": {
                "n": contrasts["N15"]["n"], "mean": contrasts["N15"]["E"]["mean"],
                "median": contrasts["N15"]["E"]["median"],
                "ci95": contrasts["N15"]["E"]["ci95_mean"],
                "n_Nov3_lower": contrasts["N15"]["E"]["n_neg"], "pass": gates[15],
            },
            "N12": {
                "n": contrasts["N12"]["n"], "mean": contrasts["N12"]["E"]["mean"],
                "median": contrasts["N12"]["E"]["median"],
                "ci95": contrasts["N12"]["E"]["ci95_mean"],
                "n_Nov3_lower": contrasts["N12"]["E"]["n_neg"], "pass": gates[12],
            },
            "decision": "center4-L3 EDGE" if both else "FAIL_l3_inconclusive",
            "gr": "not patched",
        },
        "seeds": SEEDS, "band": "exclusive_unread_dev",
        "ft_alloc_sha256": PREREG["ft_source"]["alloc_py_sha256"],
        "ft_head_commit": PREREG["ft_source"]["head_commit"],
    }
    (OUT / "summary_FTREP_L3.json").write_text(json.dumps(summary, indent=1))
    lines = [
        "# FTREP L3: FT v4 center-4 at N∈{12,15} L=3 Q=3072",
        "",
        f"Same alloc.py `{PREREG['ft_source']['alloc_py_sha256']}` / PR79 `{PREREG['ft_source']['head_commit'][:12]}`.",
        "Seeds 10192–10199. Four equal blocks, T=378, frozen gains, center-4. Energy selects. F post hoc. No N18 L3. GR not patched.",
        "",
        f"Runs {len(rows)} ok {len(ok)} failures {len(fail)}. Charged {charged}. "
        f"Outside cap {summary['outside_cap_queries']}. Rejected B* {len(rejected)}. Missing {len(missing)}.",
        "",
        f"Gate L3-15 mean ΔE {fmt(contrasts['N15'], 'E')} {contrasts['N15']['E']['n_neg']}/{contrasts['N15']['E']['n']}: "
        f"**{'pass' if gates[15] else 'fail'}**.",
        f"Gate L3-12 mean ΔE {fmt(contrasts['N12'], 'E')} {contrasts['N12']['E']['n_neg']}/{contrasts['N12']['E']['n']}: "
        f"**{'pass' if gates[12] else 'fail'}**.",
        f"Decision: **{summary['gate']['decision']}**.",
        "",
        "## Nov3 − QAOA (center4 L=3)",
        "",
        "| N | mean ΔE med [CI] tails wins(Nov3) | Δ(E−E0)/N | ΔF wins | start0 Nov3 |",
        "|---|---|---|---|---|",
    ]
    for nq in NS:
        ct = contrasts[f"N{nq}"]
        lines.append(
            f"| {nq} | {fmt(ct, 'E')} {ct['E']['n_neg']}/{ct['E']['n']} "
            f"| {fmt(ct, 'E_err_over_N')} | {fmt(ct, 'F')} {ct['F']['n_pos']}/{ct['F']['n']} "
            f"| {ct['n_start0_nov3']}/{ct['n']} |"
        )
    lines.extend([
        "",
        f"B* wall {status.get('bstar_wall_s')}. Fit wall {status.get('fit_wall_s')}. "
        f"Sum cell wall {wall_fit:.1f}s. Sum cell CPU {cpu_fit:.1f}s. Peak RSS KiB {summary['peak_rss_kib_max']}.",
        "",
    ])
    (OUT / "RESULTS_FTREP_L3.md").write_text("\n".join(lines))
    print(json.dumps({"gate": summary["gate"], "charged": charged, "ok": len(ok), "fail": len(fail),
                      "rejected": rejected, "missing": missing, "audit_failures": bad,
                      "bstar_wall_s": status.get("bstar_wall_s"), "fit_wall_s": status.get("fit_wall_s"),
                      "sum_cell_cpu_s": cpu_fit, "peak_rss_kib_max": summary["peak_rss_kib_max"]}, indent=1))
    if bad:
        raise SystemExit("audit failed")
    if missing:
        raise SystemExit(f"missing {missing}")


if __name__ == "__main__":
    main()
