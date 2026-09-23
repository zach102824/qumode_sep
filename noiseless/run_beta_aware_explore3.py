#!/usr/bin/env python3
"""Explore3: recover success on frontier β-aware settings via more SPSA + tight grid.

Frontier from explore2: ~15% |β| cut at success≈0.85 (λ1=0.12, λ3=0.1, β_max=2).
Try steps∈{400,800} on those, plus a mild L1×soft-cap grid at 400 steps.
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


def existing_tags() -> set[str]:
    return {p.name.rsplit("_20", 1)[0] for p in camp._RESULTS.glob("fleet_beta_aware_*_summary.json")}
    # fragile; better:
    

def has_tag(tag: str) -> bool:
    return any(camp._RESULTS.glob(f"{tag}_*_summary.json"))


def load_all() -> list[dict]:
    by = {}
    for p in camp._RESULTS.glob("fleet_beta_aware_*_summary.json"):
        try:
            m = camp.load_metrics(p)
            by[m["tag"]] = m
        except Exception:
            pass
    return list(by.values())


def main() -> int:
    os.chdir(_REPO)
    rows = load_all()
    notes = [
        "Explore3: longer SPSA on frontier + mild L1×soft-cap grid.",
        "Goal: success≥0.90 with |β| cut ≥15–20%.",
    ]

    frontier = [
        # (tag_prefix, λ1, λ3, β_max)
        ("A3_f_l1_0p12_l3_0p1_bmax_2p0", 0.12, 0.1, 2.0),
        ("A3_f_l1_0p07_l3_1p0_bmax_2p5", 0.07, 1.0, 2.5),
        ("A3_f_l1_0p07_l3_5p0_bmax_3p0", 0.07, 5.0, 3.0),
        ("A3_f_l1_0p12_l3_1p0_bmax_3p0", 0.12, 1.0, 3.0),
        ("A3_f_l1_0p0_l3_1p0_bmax_2p5", 0.0, 1.0, 2.5),
        ("A3_f_l1_0p1_l3_0", 0.1, 0.0, None),
        ("A3_f_l1_0p15_l3_0", 0.15, 0.0, None),
    ]
    for prefix, l1, l3, bm in frontier:
        for steps in (400, 800):
            tag = f"fleet_beta_aware_{prefix}_steps{steps}"
            if has_tag(tag):
                continue
            p = camp.run_fleet(tag, lambda1=l1, lambda3=l3, beta_max=bm, steps=steps)
            if p:
                m = camp.load_metrics(p)
                rows.append(m)
                notes.append(
                    f"{tag}: success={m['success_rate']:.3f}, mean|β|={m['mean_abs_beta']:.3f}, "
                    f"cut={100*m['beta_cut_frac']:.1f}%, mean_max={m['mean_max_abs_beta']:.3f}."
                )
                camp.write_summary(
                    sorted(rows, key=lambda r: (float(r["lambda1"]), float(r["lambda3"]), str(r["beta_max"]), int(r.get("steps") or 0))),
                    notes,
                )
        camp.try_git_push(f"β-aware explore3 frontier {prefix}")

    # Mild grid at 400 steps: λ1 × λ3 × β_max
    mild = []
    for l1 in (0.0, 0.03, 0.05, 0.08):
        for l3 in (0.0, 0.2, 0.5, 1.0):
            for bm in (None, 2.0, 2.2, 2.5):
                if l3 == 0.0 and bm is not None:
                    continue  # no soft-cap term
                if l3 > 0.0 and bm is None:
                    continue
                mild.append((l1, l3, bm))
    for l1, l3, bm in mild:
        bm_s = "none" if bm is None else str(bm).replace(".", "p")
        tag = (
            f"fleet_beta_aware_A3g_l1_{str(l1).replace('.', 'p')}"
            f"_l3_{str(l3).replace('.', 'p')}_bmax_{bm_s}_steps400"
        )
        if has_tag(tag):
            continue
        p = camp.run_fleet(tag, lambda1=l1, lambda3=l3, beta_max=bm, steps=400)
        if not p:
            continue
        m = camp.load_metrics(p)
        rows.append(m)
        notes.append(
            f"{tag}: success={m['success_rate']:.3f}, mean|β|={m['mean_abs_beta']:.3f}, "
            f"cut={100*m['beta_cut_frac']:.1f}%."
        )
        # early-ish logging
        if len(rows) % 3 == 0:
            camp.write_summary(
                sorted(rows, key=lambda r: (float(r["lambda1"]), float(r["lambda3"]), str(r["beta_max"]), int(r.get("steps") or 0))),
                notes,
            )
            camp.try_git_push("β-aware explore3 mild grid progress")

    camp.write_summary(
        sorted(rows, key=lambda r: (float(r["lambda1"]), float(r["lambda3"]), str(r["beta_max"]), int(r.get("steps") or 0))),
        notes,
    )

    good20 = [r for r in rows if r["success_rate"] >= 0.90 and r["beta_cut_frac"] >= 0.20]
    good15 = [r for r in rows if r["success_rate"] >= 0.90 and r["beta_cut_frac"] >= 0.15]
    good10 = [r for r in rows if r["success_rate"] >= 0.90 and r["beta_cut_frac"] >= 0.10]
    ok90 = [r for r in rows if r["success_rate"] >= 0.90]
    pool = good20 or good15 or good10 or ok90
    pool.sort(key=lambda r: (r["beta_cut_frac"], r["success_rate"]), reverse=True)
    if pool:
        b = pool[0]
        notes.append(
            f"RECOMMEND: λ1={b['lambda1']} λ3={b['lambda3']} β_max={b['beta_max']} steps={b.get('steps')} "
            f"→ success={b['success_rate']:.3f} cut={100*b['beta_cut_frac']:.1f}% mean|β|={b['mean_abs_beta']:.3f} "
            f"(tag={b['tag']})."
        )
    hit = "YES ≥20%" if good20 else ("YES ≥15%" if good15 else ("partial ≥10%" if good10 else "NO — no ≥10% cut at success≥0.90"))
    notes.append(f"Hit target success≥0.90 with meaningful |β| cut? {hit}")
    camp.write_summary(
        sorted(rows, key=lambda r: (float(r["lambda1"]), float(r["lambda3"]), str(r["beta_max"]), int(r.get("steps") or 0))),
        notes,
    )
    camp.try_git_push("β-aware explore3 final")
    camp.log("explore3 complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
