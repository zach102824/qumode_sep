#!/usr/bin/env python3
"""Extended β-aware exploration after A1 showed L1-only tradeoff.

Plan:
1. Finer λ1 in [0.04, 0.12] where success is near 0.90.
2. Soft-cap A2 around best mild λ1 (and λ1=0): λ3∈{0.1,1,5} × β_max∈{2.0,2.5,3.0}.
3. steps=400 on top candidates.
4. L*=3 on best 1–2.
Updates BETA_AWARE_SUMMARY.md and pushes periodically.
"""
from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))
# Reuse helpers from the main campaign module
spec = importlib.util.spec_from_file_location(
    "campaign", _REPO / "noiseless" / "run_beta_aware_campaign.py"
)
camp = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(camp)

_RESULTS = camp._RESULTS
_SUMMARY = camp._SUMMARY


def existing_rows() -> list[dict]:
    rows = []
    for p in sorted(_RESULTS.glob("fleet_beta_aware_*_summary.json")):
        # skip L3 / steps400 duplicates handled later; include all
        try:
            rows.append(camp.load_metrics(p))
        except Exception as exc:  # noqa: BLE001
            camp.log(f"skip {p.name}: {exc}")
    # de-dupe by tag keeping newest (sorted by path time via glob mtime already unsorted)
    by_tag: dict[str, dict] = {}
    for r in rows:
        by_tag[r["tag"]] = r
    return list(by_tag.values())


def tag_l1(lam: float) -> str:
    s = f"{lam:g}".replace(".", "p")
    return f"fleet_beta_aware_A1x_l1_{s}"


def main() -> int:
    os.chdir(_REPO)
    rows = existing_rows()
    notes = [
        "Phase-1 A0/A1 complete: pure L1 never hit success≥0.90 AND |β| cut≥15%.",
        "Phase-2: finer λ1, soft-cap A2, longer SPSA, L*=3 follow-ups.",
    ]
    camp.write_summary(sorted(rows, key=lambda r: (float(r["lambda1"]), float(r["lambda3"]), str(r["beta_max"]))), notes)

    # --- finer λ1 ---
    fine = [0.04, 0.06, 0.07, 0.08, 0.09, 0.12]
    for lam in fine:
        tag = tag_l1(lam)
        if any(r["tag"] == tag for r in rows):
            continue
        p = camp.run_fleet(tag, lambda1=lam, lambda3=0.0)
        if not p:
            notes.append(f"fine λ1={lam} FAILED")
            continue
        m = camp.load_metrics(p)
        rows.append(m)
        notes.append(
            f"fine λ1={lam}: success={m['success_rate']:.3f}, mean|β|={m['mean_abs_beta']:.3f}, "
            f"cut={100*m['beta_cut_frac']:.1f}%."
        )
        camp.write_summary(sorted(rows, key=lambda r: (float(r["lambda1"]), float(r["lambda3"]), str(r["beta_max"]))), notes)
        if m["success_rate"] < 0.80:
            notes.append(f"stop fine grid early at λ1={lam} (success<{0.80}).")
            break
    camp.try_git_push("β-aware fine λ1 grid results")

    # pick mild λ1 with success≥0.90 maximizing cut; also try λ1=0.05 and nearest below 0.90
    mild = [r for r in rows if r["lambda3"] == 0.0 and r.get("beta_max") is None and str(r.get("layers")) in ("4", "4")]
    mild_ok = [r for r in mild if r["success_rate"] >= 0.90]
    mild_ok.sort(key=lambda r: r["beta_cut_frac"], reverse=True)
    near = [r for r in mild if 0.85 <= r["success_rate"] < 0.90]
    near.sort(key=lambda r: r["beta_cut_frac"], reverse=True)
    l1_for_a2 = []
    if mild_ok:
        l1_for_a2.append(float(mild_ok[0]["lambda1"]))
    l1_for_a2.append(0.0)
    if near:
        l1_for_a2.append(float(near[0]["lambda1"]))
    # unique preserve order
    seen = set()
    l1_for_a2 = [x for x in l1_for_a2 if not (x in seen or seen.add(x))]
    notes.append(f"A2 soft-cap λ1 set: {l1_for_a2}")
    camp.write_summary(sorted(rows, key=lambda r: (float(r["lambda1"]), float(r["lambda3"]), str(r["beta_max"]))), notes)

    for lam in l1_for_a2:
        for l3 in (0.1, 1.0, 5.0):
            for bm in (2.0, 2.5, 3.0):
                tag = (
                    f"fleet_beta_aware_A2x_l1_{str(lam).replace('.', 'p')}"
                    f"_l3_{str(l3).replace('.', 'p')}_bmax_{str(bm).replace('.', 'p')}"
                )
                if any(r["tag"] == tag for r in rows):
                    continue
                p = camp.run_fleet(tag, lambda1=lam, lambda3=l3, beta_max=bm)
                if not p:
                    notes.append(f"A2x {tag} FAILED")
                    continue
                m = camp.load_metrics(p)
                rows.append(m)
                notes.append(
                    f"A2x λ1={lam} λ3={l3} βmax={bm}: success={m['success_rate']:.3f}, "
                    f"mean|β|={m['mean_abs_beta']:.3f}, mean_max|β|={m['mean_max_abs_beta']:.3f}, "
                    f"cut={100*m['beta_cut_frac']:.1f}%."
                )
                camp.write_summary(
                    sorted(rows, key=lambda r: (float(r["lambda1"]), float(r["lambda3"]), str(r["beta_max"]))),
                    notes,
                )
        camp.try_git_push(f"β-aware A2x soft-cap λ1={lam}")

    # Best candidates for steps=400 and L*=3
    scored = [
        r for r in rows
        if str(r.get("layers")) in ("4",) and int(r.get("steps") or 200) == 200
    ]
    # prefer success≥0.90, then larger cut, then lower mean_max
    scored.sort(
        key=lambda r: (
            r["success_rate"] >= 0.90,
            r["success_rate"] >= 0.85,
            r["beta_cut_frac"] if r["beta_cut_frac"] == r["beta_cut_frac"] else -1,
            -r["mean_max_abs_beta"],
        ),
        reverse=True,
    )
    top = scored[:3]
    notes.append("Top candidates for follow-up: " + ", ".join(t["tag"] for t in top))
    for t in top:
        # 400 steps
        tag400 = t["tag"] + "_steps400"
        if not any(r["tag"] == tag400 for r in rows):
            p = camp.run_fleet(
                tag400,
                lambda1=float(t["lambda1"]),
                lambda3=float(t["lambda3"]),
                beta_max=t["beta_max"],
                steps=400,
            )
            if p:
                rows.append(camp.load_metrics(p))
                camp.write_summary(
                    sorted(rows, key=lambda r: (float(r["lambda1"]), float(r["lambda3"]), str(r["beta_max"]))),
                    notes,
                )
        # L*=3
        tag3 = t["tag"] + "_L3"
        if not any(r["tag"] == tag3 for r in rows):
            p = camp.run_fleet(
                tag3,
                lambda1=float(t["lambda1"]),
                lambda3=float(t["lambda3"]),
                beta_max=t["beta_max"],
                layers="3",
            )
            if p:
                rows.append(camp.load_metrics(p))
                camp.write_summary(
                    sorted(rows, key=lambda r: (float(r["lambda1"]), float(r["lambda3"]), str(r["beta_max"]))),
                    notes,
                )
    camp.try_git_push("β-aware steps400 / L*=3 follow-ups")

    # Final recommendation
    good20 = [r for r in rows if r["success_rate"] >= 0.90 and r["beta_cut_frac"] >= 0.20]
    good15 = [r for r in rows if r["success_rate"] >= 0.90 and r["beta_cut_frac"] >= 0.15]
    good10 = [r for r in rows if r["success_rate"] >= 0.90 and r["beta_cut_frac"] >= 0.10]
    pool = good20 or good15 or good10 or [r for r in rows if r["success_rate"] >= 0.90]
    pool.sort(key=lambda r: (r["beta_cut_frac"], r["success_rate"]), reverse=True)
    if pool:
        b = pool[0]
        notes.append(
            f"RECOMMEND: λ1={b['lambda1']}, λ3={b['lambda3']}, β_max={b['beta_max']} "
            f"(tag={b['tag']}) → success={b['success_rate']:.3f}, mean|β|={b['mean_abs_beta']:.3f}, "
            f"cut={100*b['beta_cut_frac']:.1f}%, mean_max|β|={b['mean_max_abs_beta']:.3f}."
        )
    else:
        notes.append("No success≥0.90 setting improved |β|; prefer Gibbs-only defaults.")
    # also note best mean_max reduction with success≥0.90
    ok = [r for r in rows if r["success_rate"] >= 0.90]
    if ok:
        by_max = sorted(ok, key=lambda r: r["mean_max_abs_beta"])
        b = by_max[0]
        notes.append(
            f"Lowest mean trial-max|β| at success≥0.90: {b['tag']} → "
            f"mean_max|β|={b['mean_max_abs_beta']:.3f} (baseline ~3.38)."
        )
    camp.write_summary(
        sorted(rows, key=lambda r: (float(r["lambda1"]), float(r["lambda3"]), str(r["beta_max"]))),
        notes,
    )
    camp.try_git_push("β-aware explore2 final recommendation")
    camp.log("explore2 complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
