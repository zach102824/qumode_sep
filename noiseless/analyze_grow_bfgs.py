#!/usr/bin/env python3
"""Growth + BFGS (n=8, jp, binary, λ=0 Gibbs, L 1→4, restart η mode) vs references.

Reads the latest full (gitignored) fleet JSON per tag and writes grow_bfgs_n8_stats.json +
prints markdown tables. Arms (all 20 four_sat × 25 trials, seed 20260917, L=4 records):
  grow_bfgs_k005  jp_grow_bfgs_restart         (kick σ 0.05)
  grow_bfgs_k05   jp_grow_bfgs_restart_kick05  (kick σ 0.5)
  adam_tuned      grow_tune_F_lr_sched         (growth + SPSA-Adam, lr 0.5,0.2,0.05,0.02)
  adam_untuned    jp_adam_C_grow_ps200         (growth + SPSA-Adam, lr 0.05; C_adam)
  bfgs_fixedL4    jp_bfgs_B0                   (random-init BFGS at L=4, legacy η callback)
  spsa200_L4      jp_B0                        (random-init SPSA 200 steps at L=4)
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
_RES = _REPO / "noiseless" / "results"
STUCK = 1e-6

ARMS = {
    "grow_bfgs_k005": "jp_grow_bfgs_restart",
    "grow_bfgs_k05": "jp_grow_bfgs_restart_kick05",
    "adam_tuned": "grow_tune_F_lr_sched",
    "adam_untuned": "jp_adam_C_grow_ps200",
    "bfgs_fixedL4": "jp_bfgs_B0",
    "spsa200_L4": "jp_B0",
}


def _latest(prefix: str) -> str | None:
    c = sorted(p.name for p in _RES.glob(f"{prefix}_2026*Z.json"))
    return c[-1] if c else None


def _key(r: dict) -> tuple:
    return (r["ham_file"], int(r["trial"]), int(r["seed"]))


def _stats(recs: list[dict]) -> dict:
    pg = np.array([float(r["p_gs"]) for r in recs])
    s = np.array([bool(r["success"]) for r in recs], dtype=float)
    n = pg.size
    by_h: dict[str, list[float]] = {}
    for r in recs:
        by_h.setdefault(r["ham_file"], []).append(float(r["p_gs"]))
    best = {h: max(v) for h, v in sorted(by_h.items())}
    out = {
        "n": int(n),
        "success": float(s.mean()),
        "mean_p_gs": float(pg.mean()),
        "p_gs_se": float(pg.std(ddof=1) / np.sqrt(n)),
        "median_p_gs": float(np.median(pg)),
        "frac_gt_0p5": float(np.mean(pg > 0.5)),
        "frac_stuck": float(np.mean(pg < STUCK)),
        "max_p_gs": float(pg.max()),
        "mean_nfev": float(np.mean([r["nfev"] for r in recs])),
        "wall_per_trial_s": float(np.mean([r["wall_s"] for r in recs])),
        "best_of_25_per_instance": best,
        "mean_best_of_25": float(np.mean(list(best.values()))),
        "min_best_of_25": float(min(best.values())),
        "instances_all_stuck": int(sum(1 for v in by_h.values() if max(v) < STUCK)),
        "mean_p_gs_not_stuck": float(pg[pg >= STUCK].mean()),
        "median_p_gs_not_stuck": float(np.median(pg[pg >= STUCK])),
        "success_not_stuck": float(s[pg >= STUCK].mean()),
        "frac_failures_stuck": float(np.mean(pg[s == 0] < STUCK)) if (s == 0).any() else 0.0,
        "success_per_instance": {h: int(sum(r["success"] for r in recs if r["ham_file"] == h))
                                 for h in sorted(by_h)},
    }
    if recs[0].get("stages"):
        st = [r["stages"] for r in recs]
        ns = len(st[0])
        out["stage_mean_p_gs"] = [float(np.mean([t[i]["p_gs"] for t in st])) for i in range(ns)]
        out["stage_success"] = [float(np.mean([t[i]["success"] for t in st])) for i in range(ns)]
        out["stage_frac_stuck"] = [float(np.mean([t[i]["p_gs"] < STUCK for t in st])) for i in range(ns)]
        out["stage_mean_nfev"] = [float(np.mean([t[i]["nfev"] for t in st])) for i in range(ns)]
        if "wall_s" in st[0][0]:
            out["stage_mean_wall_s"] = [float(np.mean([t[i]["wall_s"] for t in st])) for i in range(ns)]
            out["stage_mean_nit"] = [float(np.mean([t[i]["nit"] for t in st])) for i in range(ns)]
            out["stage_max_nit"] = [int(max(t[i]["nit"] for t in st)) for i in range(ns)]
            out["stage_termination"] = [dict(Counter(t[i].get("termination") for t in st))
                                        for i in range(ns)]
            out["stage_mean_restarts"] = [float(np.mean([t[i].get("n_restarts", 0) for t in st]))
                                          for i in range(ns)]
    return out


def _paired(a: list[dict], b: list[dict]) -> dict:
    mb = {_key(r): r for r in b}
    pairs = [(r, mb[_key(r)]) for r in a if _key(r) in mb]
    d = np.array([x["p_gs"] - y["p_gs"] for x, y in pairs])
    return {
        "n": len(pairs),
        "mean_dp": float(d.mean()),
        "dp_se": float(d.std(ddof=1) / np.sqrt(d.size)),
        "median_dp": float(np.median(d)),
        "frac_higher": float(np.mean(d > 1e-6)),
        "fail_to_succ": int(sum(1 for x, y in pairs if x["success"] and not y["success"])),
        "succ_to_fail": int(sum(1 for x, y in pairs if y["success"] and not x["success"])),
    }


def main() -> int:
    recs: dict[str, list[dict]] = {}
    files: dict[str, str] = {}
    for arm, tag in ARMS.items():
        f = _latest(tag)
        if f is None:
            print(f"missing {tag}")
            continue
        d = json.loads((_RES / f).read_text())
        rs = [r for r in d["records"] if r.get("ok") and int(r["n_layers"]) == 4]
        recs[arm], files[arm] = rs, f
    stats = {a: _stats(r) for a, r in recs.items()}
    paired = {}
    for a in ("grow_bfgs_k005", "grow_bfgs_k05"):
        for b in ("adam_tuned", "bfgs_fixedL4", "grow_bfgs_k005"):
            if a in recs and b in recs and a != b:
                paired[f"{a}_vs_{b}"] = _paired(recs[a], recs[b])
    hams = sorted(stats[next(iter(stats))]["best_of_25_per_instance"])
    union_best = {h: max(stats[a]["best_of_25_per_instance"][h] for a in stats) for h in hams}
    out = {"files": files, "stats": stats, "paired": paired, "stuck_threshold": STUCK,
           "best_over_all_arms_per_instance": union_best}
    (_RES / "grow_bfgs_n8_stats.json").write_text(json.dumps(out, indent=2))

    print("| arm | n | success | mean p(GS) ± se | median | frac>0.5 | frac stuck | max p | mean best-of-25 | min best-of-25 | evals/trial | wall/trial s | stage mean p(GS) L1..L4 |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for a, s in stats.items():
        sp = " / ".join(f"{v:.3f}" for v in s.get("stage_mean_p_gs", [])) or "—"
        print(f"| {a} | {s['n']} | {s['success']:.3f} | {s['mean_p_gs']:.4f} ± {s['p_gs_se']:.4f} | "
              f"{s['median_p_gs']:.4f} | {s['frac_gt_0p5']:.3f} | {s['frac_stuck']:.3f} | {s['max_p_gs']:.4f} | "
              f"{s['mean_best_of_25']:.4f} | {s['min_best_of_25']:.4f} | {s['mean_nfev']:.0f} | "
              f"{s['wall_per_trial_s']:.2f} | {sp} |")
    print()
    for k, v in paired.items():
        print(k, json.dumps(v))
    print("not-stuck:", {a: (round(v["mean_p_gs_not_stuck"], 4), round(v["median_p_gs_not_stuck"], 4),
                             round(v["success_not_stuck"], 3), round(v["frac_failures_stuck"], 3))
                         for a, v in stats.items()})
    print("best over all arms:", {h[-7:-4]: round(v, 3) for h, v in union_best.items()},
          "mean", round(float(np.mean(list(union_best.values()))), 4),
          "min", round(min(union_best.values()), 4))
    for a in ("grow_bfgs_k005", "grow_bfgs_k05"):
        if a in stats:
            s = stats[a]
            print(a, {k: s[k] for k in s if k.startswith("stage_")})
            print(a, "best-of-25", {h[-7:-4]: round(v, 3) for h, v in s["best_of_25_per_instance"].items()})
            print(a, "succ/inst", {h[-7:-4]: v for h, v in s["success_per_instance"].items()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
