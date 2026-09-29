#!/usr/bin/env python3
"""Summarize jp encoding / layer-growth arms against the random-init baselines.

Arms (all jp, λ=0 Gibbs, 20 four_sat × 25 trials, seed 20260917):
  A      binary, random init L, 200 steps (existing)
  A400/A800  binary random init, 400/800 steps (existing step scan)
  B      gray, random init L, 200 steps
  C      binary --grow, 200 steps PER STAGE (total 200·L)
  D      gray   --grow, 200 steps PER STAGE
  Ceq    binary --grow, equal-total 200 steps (aux run, stopped plan)

Reads full JSONs in noiseless/results (gitignored); writes encoding_grow_stats_summary.json
and prints a markdown table. Pass --spec label:L:file (repeatable) to override defaults.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

_RES = Path(__file__).resolve().parent / "results"

DEFAULT_SPECS = [
    ("A", 2, "jp_L2_20260929T041311Z.json"),
    ("A", 3, "jp_L3_20260929T041643Z.json"),
    ("A", 4, "jp_B0_20260928T042134Z.json"),
    ("A400", 2, "jp_L2_s400_20260929T062619Z.json"),
    ("A400", 3, "jp_L3_s400_20260929T063319Z.json"),
    ("A400", 4, "jp_L4_s400_20260929T064228Z.json"),
    ("A800", 2, "jp_L2_s800_20260929T065200Z.json"),
    ("A800", 3, "jp_L3_s800_20260929T070550Z.json"),
    ("A800", 4, "jp_L4_s800_20260929T072352Z.json"),
]


def _latest(prefix: str) -> str | None:
    c = sorted(p.name for p in _RES.glob(f"{prefix}_2026*Z.json"))
    return c[-1] if c else None


def _stats(recs: list[dict]) -> dict:
    pg = np.array([float(r["p_gs"]) for r in recs])
    succ = np.array([bool(r["success"]) for r in recs])
    by_h: dict[str, list[float]] = {}
    for r in recs:
        by_h.setdefault(r["ham_file"], []).append(float(r["p_gs"]))
    out = {
        "n": int(pg.size),
        "success": float(succ.mean()),
        "mean_p_gs": float(pg.mean()),
        "median_p_gs": float(np.median(pg)),
        "frac_p_gs_gt_0p5": float(np.mean(pg > 0.5)),
        "best_of_25_mean_p_gs": float(np.mean([max(v) for v in by_h.values()])),
        "mean_wall_s": float(np.mean([float(r["wall_s"]) for r in recs])),
        "total_steps": int(recs[0].get("nit", 0)),
    }
    if recs and recs[0].get("stages"):
        out["stages"] = []
        for si in range(len(recs[0]["stages"])):
            st = [r["stages"][si] for r in recs]
            row = {
                "n_layers": st[0]["n_layers"],
                "steps": st[0]["steps"],
                "success": float(np.mean([s["success"] for s in st])),
                "mean_p_gs": float(np.mean([s["p_gs"] for s in st])),
                "median_p_gs": float(np.median([s["p_gs"] for s in st])),
            }
            if si > 0:
                prev = [r["stages"][si - 1] for r in recs]
                row["max_abs_insert_dev"] = float(
                    max(abs(s["p_gs_after_insert"] - q["p_gs"]) for s, q in zip(st, prev))
                )
                row["mean_kick_dp_gs"] = float(
                    np.mean([s["p_gs_after_kick"] - s["p_gs_after_insert"] for s in st])
                )
            out["stages"].append(row)
    return out


def _paired(a: list[dict], b: list[dict]) -> dict:
    ka = {(r["ham_file"], r["trial"]): r for r in a}
    pairs = [(ka[(r["ham_file"], r["trial"])], r) for r in b if (r["ham_file"], r["trial"]) in ka]
    d = np.array([q["p_gs"] - p["p_gs"] for p, q in pairs])
    return {
        "n": len(pairs),
        "frac_b_higher": float(np.mean(d > 0)),
        "mean_dp": float(d.mean()),
        "median_dp": float(np.median(d)),
        "fail_to_succ": int(sum((not p["success"]) and q["success"] for p, q in pairs)),
        "succ_to_fail": int(sum(p["success"] and not q["success"] for p, q in pairs)),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--spec", action="append", default=[])
    ap.add_argument("--out-json", default=str(_RES / "encoding_grow_stats_summary.json"))
    args = ap.parse_args()
    specs = list(DEFAULT_SPECS)
    for label, prefix in (
        ("B", "jp_enc_B_gray"),
        ("C", "jp_enc_C_bin_grow_ps200"),
        ("D", "jp_enc_D_gray_grow_ps200"),
        ("Ceq", "jp_enc_C_bin_grow"),
    ):
        f = _latest(prefix)
        if f:
            specs += [(label, L, f) for L in (2, 3, 4)]
    for s in args.spec:
        label, L, f = s.split(":", 2)
        specs.append((label, int(L), f))
    cache: dict[str, dict] = {}
    recs_by: dict[tuple, list[dict]] = {}
    table: dict = {"files": {}}
    for label, L, f in specs:
        if f not in cache:
            cache[f] = json.loads((_RES / f).read_text())
        recs = [r for r in cache[f]["records"] if r.get("ok") and int(r["n_layers"]) == L]
        recs_by[(label, L)] = recs
        table[f"{label}_L{L}"] = _stats(recs)
        table["files"][f"{label}_L{L}"] = f
    table["paired"] = {}
    for (label, L), recs in recs_by.items():
        if label != "A" and ("A", L) in recs_by:
            table["paired"][f"{label}_vs_A_L{L}"] = _paired(recs_by[("A", L)], recs)
    for lab, ref in (("C", "A400"), ("D", "A400"), ("C", "A800"), ("D", "A800"), ("D", "C"), ("B", "A")):
        for L in (2, 3, 4):
            if (lab, L) in recs_by and (ref, L) in recs_by:
                table["paired"][f"{lab}_vs_{ref}_L{L}"] = _paired(recs_by[(ref, L)], recs_by[(lab, L)])
    Path(args.out_json).write_text(json.dumps(table, indent=2))
    print("| arm | L | total steps | success | mean p(GS) | median p(GS) | frac p(GS)>0.5 | best-of-25 | wall/trial s |")
    print("|---|---|---|---|---|---|---|---|---|")
    for label, L, _ in specs:
        s = table[f"{label}_L{L}"]
        print(
            f"| {label} | {L} | {s['total_steps']} | {s['success']:.3f} | {s['mean_p_gs']:.4f} "
            f"| {s['median_p_gs']:.4f} | {s['frac_p_gs_gt_0p5']:.3f} | {s['best_of_25_mean_p_gs']:.4f} "
            f"| {s['mean_wall_s']:.2f} |"
        )
    print()
    for k, v in table["paired"].items():
        print(k, v)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
