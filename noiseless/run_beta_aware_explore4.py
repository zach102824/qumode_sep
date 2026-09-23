#!/usr/bin/env python3
"""Explore4: refine soft-cap around current best (≥10% cut at success≥0.90).

Best so far:
  λ1=0.05 λ3=0.5 β_max=2.5 steps=400 → succ=0.906 cut=12.2%
  λ1=0    λ3=1.0 β_max=2.2 steps=400 → succ=0.900 cut=12.1% (best max shrink)
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))
spec = importlib.util.spec_from_file_location(
    "campaign", _REPO / "noiseless" / "run_beta_aware_campaign.py"
)
camp = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(camp)


def has_tag(tag: str) -> bool:
    return any(camp._RESULTS.glob(f"{tag}_*_summary.json"))


def load_all() -> list[dict]:
    by = {}
    for p in camp._RESULTS.glob("fleet_beta_aware_*_summary.json"):
        try:
            m = camp.load_metrics(p)
            if m["mean_abs_beta"] < 50:
                by[m["tag"]] = m
        except Exception:
            pass
    return list(by.values())


def main() -> int:
    os.chdir(_REPO)
    rows = load_all()
    notes = [
        "Explore4: soft-cap refinement aiming for success≥0.90 and |β| cut≥15%.",
        "Prior best ~12% cut at success≥0.90 (soft-cap / mild L1+cap).",
    ]

    jobs = []
    for l1 in (0.0, 0.03, 0.05):
        for l3 in (0.75, 1.0, 1.5, 2.0, 3.0):
            for bm in (2.0, 2.1, 2.2, 2.3, 2.4):
                for steps in (400, 800):
                    jobs.append((l1, l3, bm, steps))
    # Extra long SPSA on previous champs
    for l1, l3, bm in ((0.0, 1.0, 2.2), (0.05, 0.5, 2.5), (0.0, 1.0, 2.1), (0.05, 1.0, 2.2)):
        for steps in (800, 1200):
            jobs.append((l1, l3, bm, steps))

    # de-dupe
    seen = set()
    uniq = []
    for j in jobs:
        if j not in seen:
            seen.add(j)
            uniq.append(j)

    camp.log(f"explore4 queued {len(uniq)} fleets")
    for i, (l1, l3, bm, steps) in enumerate(uniq):
        tag = (
            f"fleet_beta_aware_A4_l1_{str(l1).replace('.', 'p')}"
            f"_l3_{str(l3).replace('.', 'p')}_bmax_{str(bm).replace('.', 'p')}_steps{steps}"
        )
        if has_tag(tag):
            continue
        p = camp.run_fleet(tag, lambda1=l1, lambda3=l3, beta_max=bm, steps=steps)
        if not p:
            notes.append(f"FAIL {tag}")
            continue
        m = camp.load_metrics(p)
        rows.append(m)
        notes.append(
            f"{tag}: success={m['success_rate']:.3f}, mean|β|={m['mean_abs_beta']:.3f}, "
            f"cut={100*m['beta_cut_frac']:.1f}%, mean_max={m['mean_max_abs_beta']:.3f}."
        )
        if (i + 1) % 4 == 0 or m["success_rate"] >= 0.90 and m["beta_cut_frac"] >= 0.15:
            camp.write_summary(
                sorted(rows, key=lambda r: (-r["success_rate"], -r["beta_cut_frac"])),
                notes[-80:],
            )
            camp.try_git_push(f"β-aware explore4 progress ({i+1}/{len(uniq)})")
            if m["success_rate"] >= 0.90 and m["beta_cut_frac"] >= 0.20:
                notes.append(f"EARLY HIT ≥20%: {tag}")
                camp.write_summary(sorted(rows, key=lambda r: (-r["success_rate"], -r["beta_cut_frac"])), notes[-80:])
                camp.try_git_push("β-aware explore4 EARLY HIT ≥20%")

    good20 = [r for r in rows if r["success_rate"] >= 0.90 and r["beta_cut_frac"] >= 0.20]
    good15 = [r for r in rows if r["success_rate"] >= 0.90 and r["beta_cut_frac"] >= 0.15]
    good10 = [r for r in rows if r["success_rate"] >= 0.90 and r["beta_cut_frac"] >= 0.10]
    pool = good20 or good15 or good10 or [r for r in rows if r["success_rate"] >= 0.90]
    pool.sort(key=lambda r: (r["beta_cut_frac"], r["success_rate"]), reverse=True)
    if pool:
        b = pool[0]
        notes.append(
            f"RECOMMEND: λ1={b['lambda1']} λ3={b['lambda3']} β_max={b['beta_max']} steps={b.get('steps')} "
            f"→ success={b['success_rate']:.3f} cut={100*b['beta_cut_frac']:.1f}% mean|β|={b['mean_abs_beta']:.3f} "
            f"mean_max={b['mean_max_abs_beta']:.3f} (tag={b['tag']})."
        )
    hit = "YES ≥20%" if good20 else ("YES ≥15%" if good15 else ("partial ≥10%" if good10 else "NO"))
    notes.append(f"Hit target? {hit} (n≥10% @succ≥0.90: {len(good10)}; ≥15%: {len(good15)}; ≥20%: {len(good20)})")
    camp.write_summary(sorted(rows, key=lambda r: (-r["success_rate"], -r["beta_cut_frac"])), notes[-100:])
    camp.try_git_push("β-aware explore4 final")
    camp.log("explore4 complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
