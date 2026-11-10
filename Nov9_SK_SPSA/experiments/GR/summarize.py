"""Summarize the GR pilot. Fidelity was not used to select parameters."""
from __future__ import annotations

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from common import RESULT_DIR, load_prereg

RUN_DIR = RESULT_DIR / "runs"


def _finite(v):
    return isinstance(v, (int, float)) and math.isfinite(float(v))


def _median(xs):
    ys = [float(v) for v in xs if _finite(v)]
    if not ys:
        return None
    return float(np.median(ys))


def _boot_median(values, rng):
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return None
    draws = rng.integers(0, arr.size, size=(10000, arr.size))
    meds = np.median(arr[draws], axis=1)
    return {
        "median": float(np.median(arr)),
        "ci95": [float(np.percentile(meds, 2.5)), float(np.percentile(meds, 97.5))],
        "n": int(arr.size),
    }


def _load():
    rows = []
    for path in sorted(RUN_DIR.glob("*.json")):
        doc = json.loads(path.read_text())
        doc["_file"] = path.name
        rows.append(doc)
    return rows


def _key(row):
    return (row["N"], row["b"], row["L"], row["cap"], row["arm"])


def _cell(rows, n, b, L, cap, arm):
    hit = [r for r in rows if r.get("status") == "ok" and (r["N"], float(r["b"]), r["L"], r["cap"], r["arm"]) == (n, float(b), L, cap, arm)]
    hit.sort(key=lambda r: r["seed"])
    return hit


def _paired(rows, n, b, L, cap, arm, rng):
    a = _cell(rows, n, b, L, cap, arm)
    q = _cell(rows, n, b, L, cap, "QAOA")
    qm = {r["seed"]: r for r in q}
    dF, dE, both = [], [], []
    for r in a:
        if r["seed"] not in qm:
            continue
        dF.append(r["selected"]["F"] - qm[r["seed"]]["selected"]["F"])
        dE.append(r["selected"]["E"] - qm[r["seed"]]["selected"]["E"])
        both.append(r["seed"])
    wins = sum(1 for v in dF if v > 0.0)
    energy_wins = sum(1 for v in dE if v < 0.0)
    return {
        "arm": arm,
        "n_pairs": len(dF),
        "dF": _boot_median(dF, rng),
        "dE": _boot_median(dE, rng),
        "F_win_rate": (wins / len(dF)) if dF else None,
        "E_win_rate_lower_is_better": (energy_wins / len(dE)) if dE else None,
    }


def run():
    pre = load_prereg()
    rng = np.random.default_rng(int(pre["bootstrap"]["seed"]))
    rows = _load()
    ok = [r for r in rows if r.get("status") == "ok"]
    bad = [r for r in rows if r.get("status") != "ok"]
    arms = [a["name"] for a in pre["arms"]]
    tables = []
    for n in pre["pilot"]["N"]:
        for b in pre["fields"]:
            for L in pre["pilot"]["L"]:
                for cap in pre["pilot"]["caps"]:
                    methods = []
                    for arm in arms:
                        cell = _cell(rows, n, b, L, cap, arm)
                        methods.append(
                            {
                                "arm": arm,
                                "n": len(cell),
                                "median_F": _median([r["selected"]["F"] for r in cell]),
                                "median_E": _median([r["selected"]["E"] for r in cell]),
                                "median_eta": _median([r["selected"]["eta"] for r in cell]),
                                "median_eta_E": _median([r["selected"]["eta_E"] for r in cell]),
                                "median_E_resid": _median([r["selected"]["E_resid"] for r in cell]),
                                "median_p_leak": _median([r["selected"]["p_leak"] for r in cell]),
                                "median_F_plus": _median([r["selected"]["F_plus"] for r in cell]),
                                "median_init_F": _median([r["init_start_E_oracle"]["F"] for r in cell]),
                                "median_init_E": _median([r["init_start_E_oracle"]["E"] for r in cell]),
                                "queries_total": sum(int(r["queries"]["total"]) for r in cell),
                                "failures_in_cell": sum(1 for r in cell if r.get("failure")),
                            }
                        )
                    pairs = [_paired(rows, n, b, L, cap, arm, rng) for arm in arms if arm != "QAOA"]
                    tables.append({"N": n, "b": b, "L": L, "cap": cap, "methods": methods, "paired_vs_QAOA": pairs})

    def assignment(n, b, L, cap):
        base = _cell(rows, n, b, L, cap, "G_R")
        out = {"n": len(base)}
        for other in ("inter_id", "G_id", "G_antiR", "inter_R", "Gray_G_R", "B_G_R", "G_R_U0"):
            om = {r["seed"]: r for r in _cell(rows, n, b, L, cap, other)}
            dF = [r["selected"]["F"] - om[r["seed"]]["selected"]["F"] for r in base if r["seed"] in om]
            dE = [r["selected"]["E"] - om[r["seed"]]["selected"]["E"] for r in base if r["seed"] in om]
            out[f"G_R_minus_{other}"] = {"dF": _boot_median(dF, rng), "dE": _boot_median(dE, rng)}
        rand_iqr = []
        d_rand = []
        for r in base:
            vals = []
            for name in ("rand0", "rand1", "rand2"):
                hit = [q for q in _cell(rows, n, b, L, cap, name) if q["seed"] == r["seed"]]
                if hit:
                    vals.append(hit[0]["selected"]["F"])
            if len(vals) == 3:
                rand_iqr.append(float(np.percentile(vals, 75) - np.percentile(vals, 25)))
                d_rand.append(r["selected"]["F"] - float(np.median(vals)))
        out["G_R_minus_median_random_F"] = _boot_median(d_rand, rng)
        out["random_encoding_IQR_F"] = _boot_median(rand_iqr, rng)
        return out

    assignments = []
    for n in pre["pilot"]["N"]:
        for b in pre["fields"]:
            for L in pre["pilot"]["L"]:
                for cap in pre["pilot"]["caps"]:
                    assignments.append({"N": n, "b": b, "L": L, "cap": cap, **assignment(n, b, L, cap)})

    u0 = []
    for n in pre["pilot"]["N"]:
        for b in pre["fields"]:
            for L in pre["pilot"]["L"]:
                for cap in pre["pilot"]["caps"]:
                    for primary, sens in (("inter_id", "inter_id_U0"), ("G_R", "G_R_U0")):
                        a = {r["seed"]: r for r in _cell(rows, n, b, L, cap, primary)}
                        s = {r["seed"]: r for r in _cell(rows, n, b, L, cap, sens)}
                        dF = [abs(a[k]["selected"]["F"] - s[k]["selected"]["F"]) for k in a if k in s]
                        u0.append({"N": n, "b": b, "L": L, "cap": cap, "pair": primary, "median_abs_dF": _median(dF), "n": len(dF)})

    query_total = sum(int(r["queries"]["total"]) for r in ok)
    by_cat = defaultdict(int)
    for r in ok:
        for k, v in r["queries"].items():
            by_cat[k] += int(v)
    seconds = sum(float(r.get("seconds") or 0.0) for r in rows)
    inst = {}
    inst_path = RESULT_DIR / "instance_diagnostics.json"
    if inst_path.exists():
        inst_doc = json.loads(inst_path.read_text())
        for row in inst_doc["rows"]:
            for b, fields in row["fields"].items():
                inst.setdefault(f"N{row['N']}_b{b}", []).append(fields["F_plus"])
    fplus = {k: _median(v) for k, v in inst.items()}
    post = {}
    post_path = RESULT_DIR / "postcheck.json"
    if post_path.exists():
        post = json.loads(post_path.read_text())
    cutoff = {}
    cut_path = RESULT_DIR / "cutoff.json"
    if cut_path.exists():
        cutoff = json.loads(cut_path.read_text())
        cutoff.pop("rows", None)
    summary = {
        "idea": "N9-GR",
        "approval": pre["approval"],
        "hardware_claim": False,
        "shot_claim": False,
        "freeze": False,
        "validation_touched": False,
        "final_touched": False,
        "fidelity": "unconditioned code overlap; postselected fidelity is stored only as F_postselected_not_physical and was not used to select",
        "pilot": pre["pilot"],
        "n_rows": len(rows),
        "n_ok": len(ok),
        "n_not_ok": len(bad),
        "not_ok_files": [r["_file"] for r in bad],
        "query_total_counted": query_total,
        "queries_by_category": dict(by_cat),
        "cpu_seconds_sum": seconds,
        "median_F_plus_dev": fplus,
        "nf_sim": cutoff.get("nf_sim"),
        "c_spsa": cutoff.get("c_spsa"),
        "cutoff_worst": cutoff.get("worst"),
        "cutoff_timings": cutoff.get("timings"),
        "postcheck": post,
        "parent_reported_not_reproduced": {
            "source": "parent message, local Nov3 stream, not run in GR",
            "instance": "N=18 b=1 L=1 seed=10000 Nov3rd",
            "F": 0.61289,
            "F_plus": 0.21317,
            "queries": 1024,
            "note": "one instance and no paired QAOA yet, so the scaling target stays open",
        },
        "tables": tables,
        "assignments": assignments,
        "u0_abs_dF": u0,
        "prereg_sha256": pre["_sha256"],
    }
    (RESULT_DIR / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    _write_md(summary)
    print(f"wrote summary n_ok={len(ok)} n_bad={len(bad)} queries={query_total}")


def _fmt(v):
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.4f}"
    return str(v)


def _write_md(summary: dict) -> None:
    lines = []
    lines.append("# N9-GR bounded development results")
    lines.append("")
    lines.append("Zach approved D2–D5 at 18:41 Asia/Shanghai on 2026-11-09 as **simulation assumptions**.")
    lines.append("This is not a hardware-feasibility claim. The ideal logical readout uses an unmodelled `W_L`.")
    lines.append("Validation seeds 11000–11031 and final seeds 20000–20999 were not opened. The recipe is not frozen.")
    lines.append("Fidelity is the unconditioned code overlap. It was not a selector and there is no postselection.")
    lines.append("Every optimizer is SPSA. QAOA uses p = L independent angles. Displacement, when present, sits between `UcZ` and the bare shell SNAP, and `UcZ` is shared across layers.")
    lines.append("")
    lines.append(f"Counted energy queries: {summary['query_total_counted']}. OK rows: {summary['n_ok']}. Other rows: {summary['n_not_ok']}.")
    lines.append(f"NF_sim = {summary.get('nf_sim')}. SPSA c = {summary.get('c_spsa')}. Sum of job CPU seconds = {summary['cpu_seconds_sum']:.1f}.")
    lines.append("")
    lines.append("## Do not read a small-N fidelity of 0.8 as the result")
    lines.append("")
    lines.append("The dev-panel median overlap of the uniform plus state with the ground state is already high at N = 6 and 9.")
    lines.append("The parent also reported one local Nov3rd point, not reproduced here: N = 18, b = 1, L = 1, seed 10000, F = 0.61289 from F+ = 0.21317 at 1024 queries, with no paired QAOA yet. Scaling stays open.")
    lines.append("")
    lines.append("| panel | median F+ |")
    lines.append("|---|---:|")
    for k, v in summary["median_F_plus_dev"].items():
        lines.append(f"| {k} | {_fmt(v)} |")
    lines.append("")
    lines.append("## Paired QAOA, cap 1024 and cap 128")
    lines.append("")
    lines.append("ΔF is method minus QAOA. ΔE is method minus QAOA, so a negative ΔE is a lower energy. Bootstrap intervals use 8 seeds and are wide.")
    lines.append("")
    focus = ("G_R", "inter_id", "G_antiR", "Gray_G_R", "Nov3rd", "QAOA", "B_G_R")
    for cap in (128, 1024):
        lines.append(f"### Cap {cap}")
        lines.append("")
        lines.append("| N | b | L | arm | med F | med F+ | med η | med E resid | med leak | med ΔF vs QAOA | ΔF 95% CI | F wins | med ΔE |")
        lines.append("|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---|---:|---:|")
        for table in summary["tables"]:
            if table["cap"] != cap:
                continue
            mmap = {m["arm"]: m for m in table["methods"]}
            pmap = {p["arm"]: p for p in table["paired_vs_QAOA"]}
            for arm in focus:
                m = mmap.get(arm)
                if not m:
                    continue
                p = pmap.get(arm, {})
                dF = (p.get("dF") or {}) if p else {}
                dE = (p.get("dE") or {}) if p else {}
                ci = dF.get("ci95")
                ci_s = "" if not ci else f"[{ci[0]:.3f}, {ci[1]:.3f}]"
                lines.append(
                    "| {N} | {b} | {L} | {arm} | {F} | {Fp} | {eta} | {er} | {lk} | {dF} | {ci} | {wr} | {dE} |".format(
                        N=table["N"],
                        b=table["b"],
                        L=table["L"],
                        arm=arm,
                        F=_fmt(m["median_F"]),
                        Fp=_fmt(m["median_F_plus"]),
                        eta=_fmt(m["median_eta"]),
                        er=_fmt(m["median_E_resid"]),
                        lk=_fmt(m["median_p_leak"]),
                        dF=_fmt(dF.get("median")) if dF else "",
                        ci=ci_s,
                        wr=_fmt(p.get("F_win_rate")) if p else "",
                        dE=_fmt(dE.get("median")) if dE else "",
                    )
                )
        lines.append("")
    lines.append("## Encoding comparisons at cap 1024")
    lines.append("")
    lines.append("G+R minus another arm. The N = 12 falsifier in the consultation was not run. These N = 6 and 9 numbers are the pilot analogue only.")
    lines.append("")
    lines.append("| N | b | L | contrast | med ΔF | ΔF 95% CI |")
    lines.append("|---:|---:|---:|---|---:|---|")
    for row in summary["assignments"]:
        if row["cap"] != 1024:
            continue
        for name in ("G_R_minus_median_random_F", "G_R_minus_G_antiR", "G_R_minus_inter_id", "G_R_minus_Gray_G_R", "random_encoding_IQR_F"):
            block = row.get(name)
            if name.startswith("G_R_minus_") and name != "G_R_minus_median_random_F":
                block = (block or {}).get("dF")
            if not block:
                continue
            ci = block.get("ci95")
            ci_s = "" if not ci else f"[{ci[0]:.3f}, {ci[1]:.3f}]"
            lines.append(f"| {row['N']} | {row['b']} | {row['L']} | {name} | {_fmt(block.get('median'))} | {ci_s} |")
    lines.append("")
    lines.append("## U0 sensitivity, median absolute ΔF")
    lines.append("")
    lines.append("| N | b | L | cap | pair | median abs ΔF |")
    lines.append("|---:|---:|---:|---:|---|---:|")
    for row in summary["u0_abs_dF"]:
        if row["cap"] != 1024:
            continue
        lines.append(f"| {row['N']} | {row['b']} | {row['L']} | {row['cap']} | {row['pair']} | {_fmt(row['median_abs_dF'])} |")
    lines.append("")
    lines.append("## Cutoff and leakage")
    lines.append("")
    lines.append(f"Post-check against NF = 48: {json.dumps(summary.get('postcheck'))}")
    lines.append("")
    lines.append("Prestudy worst-case gaps are in `cutoff.json`. Population outside the chosen cutoff and changes in E, F, and leak probability were required to be at most 1e-8 on the preregistered stress grid before SPSA.")
    lines.append("")
    lines.append("## What this does not claim")
    lines.append("")
    lines.append("No hardware feasibility. No shot-noise result. No freeze. N = 12 and N = 15 were timed and not optimized. The absolute gate F ≥ 0.8 at N ≥ 12 was not tested.")
    lines.append("")
    lines.append("Artifacts: `results/GR/summary.json`, `runs/*.json`, `instance_diagnostics.json`, `cutoff.json`, `postcheck.json`, `conformance.json`.")
    (RESULT_DIR / "SUMMARY.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    run()
