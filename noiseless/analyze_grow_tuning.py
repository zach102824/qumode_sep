#!/usr/bin/env python3
"""Grow+Adam hyper-parameter tuning analysis (jp, binary, λ=0 Gibbs, final L=4, 800 SPSA steps).

Reads the latest full (gitignored) JSON per tag:
  screen: grow_tune_S_<name>   (four_sat_000..004 × 10 trials; baseline = grow_tune_S_base)
  combo:  grow_tune_X_<name>   (combos on the same screen subset)
  full:   grow_tune_F_<name>   (20 four_sat × 25 trials; baseline = C_adam jp_adam_C_grow_ps200, L=4)
Stats: success (± binomial stderr), mean p(GS) (± stderr over trials), median, frac>0.5, paired
Δp(GS) vs the baseline on identical (instance, trial, seed, x0) with stderr, success flips,
eval count, wall; failure modes (argmax energy level) for full runs.
Writes grow_tune_stats_summary.json and prints markdown tables.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from noiseless.analyze_adam import _failure_modes
from noiseless.analyze_encoding_grow import _latest

_REPO = Path(__file__).resolve().parents[1]
_RES = _REPO / "noiseless" / "results"


def _load(f: str, L: int = 4) -> tuple[list[dict], dict]:
    d = json.loads((_RES / f).read_text())
    return [r for r in d["records"] if r.get("ok") and int(r["n_layers"]) == L], d


def _row(recs: list[dict], base: list[dict] | None) -> dict:
    pg = np.array([r["p_gs"] for r in recs])
    s = np.array([r["success"] for r in recs], dtype=float)
    n = pg.size
    out = {
        "n": int(n),
        "success": float(s.mean()),
        "success_se": float(np.sqrt(s.mean() * (1 - s.mean()) / n)),
        "mean_p_gs": float(pg.mean()),
        "p_gs_se": float(pg.std(ddof=1) / np.sqrt(n)),
        "median_p_gs": float(np.median(pg)),
        "frac_gt_0p5": float(np.mean(pg > 0.5)),
        "nfev": sorted({int(r["nfev"]) for r in recs}),
        "wall_per_trial_s": float(np.mean([r["wall_s"] for r in recs])),
        "stage_success": None,
        "stage_mean_p_gs": None,
    }
    st = [r.get("stages") or [] for r in recs]
    if st and all(len(x) == len(st[0]) for x in st):
        out["stage_success"] = [float(np.mean([x[i]["success"] for x in st])) for i in range(len(st[0]))]
        out["stage_mean_p_gs"] = [float(np.mean([x[i]["p_gs"] for x in st])) for i in range(len(st[0]))]
    if base is not None:
        B = {(r["ham_file"], r["trial"]): r for r in base}
        pairs = [(r, B[(r["ham_file"], r["trial"])]) for r in recs if (r["ham_file"], r["trial"]) in B]
        d = np.array([a["p_gs"] - b["p_gs"] for a, b in pairs])
        out["paired_n"] = len(pairs)
        out["paired_dp"] = float(d.mean())
        out["paired_dp_se"] = float(d.std(ddof=1) / np.sqrt(d.size)) if d.size > 1 else 0.0
        out["frac_higher"] = float(np.mean(d > 0))
        out["fail_to_succ"] = int(sum(a["success"] and not b["success"] for a, b in pairs))
        out["succ_to_fail"] = int(sum(b["success"] and not a["success"] for a, b in pairs))
    return out


def main() -> int:
    table: dict = {"screen": {}, "combo": {}, "full": {}, "files": {}}
    fb = _latest("grow_tune_S_base")
    base_s = _load(fb)[0] if fb else None
    for kind, pref in (("screen", "grow_tune_S_"), ("combo", "grow_tune_X_")):
        names = sorted({p.name[len(pref):].rsplit("_2026", 1)[0]
                        for p in _RES.glob(f"{pref}*_2026*Z.json")})
        for nm in names:
            f = _latest(pref + nm)
            recs, d = _load(f)
            table["files"][pref + nm] = f
            table[kind][nm] = {"args": {k: d["args"].get(k) for k in (
                "adam_lr", "grow_start", "grow_kick_sigma", "grow_steps_schedule", "grow_lr_schedule",
                "grow_eta_scale", "grow_c_schedule", "steps")} | {"spsa_c": recs[0]["stages"][-1].get("spsa_c")},
                **_row(recs, base_s)}
    base_full = _load("jp_adam_C_grow_ps200_20260929T090938Z.json")[0]
    table["full"]["C_adam (baseline)"] = {**_row(base_full, None), **_failure_modes(base_full)}
    for p in sorted({p.name[len("grow_tune_F_"):].rsplit("_2026", 1)[0]
                     for p in _RES.glob("grow_tune_F_*_2026*Z.json")}):
        f = _latest("grow_tune_F_" + p)
        recs, _ = _load(f)
        table["files"]["grow_tune_F_" + p] = f
        table["full"][p] = {**_row(recs, base_full), **_failure_modes(recs)}
    (_RES / "grow_tune_stats_summary.json").write_text(json.dumps(table, indent=2))

    def fmt(nm, r, paired=True):
        s = (f"| {nm} | {r['success']:.3f} ± {r['success_se']:.3f} | {r['mean_p_gs']:.4f} ± {r['p_gs_se']:.4f} "
             f"| {r['median_p_gs']:.4f} | {r['frac_gt_0p5']:.3f} |")
        if paired and "paired_dp" in r:
            s += (f" {r['paired_dp']:+.4f} ± {r['paired_dp_se']:.4f} | {r['frac_higher']:.2f} "
                  f"| {r['fail_to_succ']} / {r['succ_to_fail']} |")
        return s + f" {','.join(map(str, r['nfev']))} |"

    for kind in ("screen", "combo"):
        print(f"\n## {kind}\n| setting | success ± se | mean p(GS) ± se | median | frac>0.5 | paired Δp ± se | "
              "frac higher | flips f→s / s→f | evals |")
        print("|---|---|---|---|---|---|---|---|---|")
        for nm, r in sorted(table[kind].items(), key=lambda kv: -kv[1]["mean_p_gs"]):
            print(fmt(nm, r))
    print("\n## full")
    for nm, r in table["full"].items():
        print(fmt(nm, r), "stages", r["stage_success"], r["stage_mean_p_gs"],
              "fail", {k: r.get(k) for k in ("n_fail", "frac_first_excited", "mean_p_gs_fail", "median_p_gs_fail")},
              f"wall/trial {r['wall_per_trial_s']:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
