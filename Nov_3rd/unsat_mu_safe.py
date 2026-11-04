"""Quantumness-safe μ on the fine unsat Walsh grid.

    H = diag(unsat) − 0.75 Σ_i X_i + μ Σ_cav Σ_{i<j} X_i X_j
    μ ∈ {−1.00, −0.95, −0.90, −0.85, −0.80, −0.75, −0.60, −0.50}

λ = 0.75 is fixed. R is the frozen n=12 L=16 Walsh ratio of medians against
Fourier QAOA on the same μ. Lower R is the improvement direction from PR #26
and PR #28. The SAFE champ minimizes that R subject to a median n=12 product
gap of at least 0.05 and at least 95% of the n=12 transfer instances at
gap ≥ 0.02. The EDGE champ is the unconstrained argmin of R. n=21 is out of
scope.
"""
from __future__ import annotations

import math

import numpy as np

from Nov_3rd.seal_unsat import TIE_R, mark_of
from Nov_3rd.unsat_mu_sweep import cache_tag, file_tag, mu_label

# Decimal grid. −1.00, −0.85, −0.75, and −0.50 are resumed checkpoints.
MU_GRID = (-1.00, -0.95, -0.90, -0.85, -0.80, -0.75, -0.60, -0.50)
RESUME_MUS = (-1.00, -0.85, -0.75, -0.50)
ANCHOR_MU = -0.50
# Exact n=12 Walsh ratio of medians from the PR #26 L=16 checkpoint.
PUBLISHED_ANCHOR_R = 0.15015595428732964
# Exact resumed ratios from PR #28 (ratio of medians, n=12 Walsh).
RESUME_R = {
    -1.00: 0.09675474400205866,
    -0.85: 0.10185718116703826,
    -0.75: 0.1237824099740523,
    -0.50: 0.15015595428732964,
}
ANCHOR_TOL = 5e-4
MED_GAP_MIN = 0.05
FRAC_MIN = 0.95
GAP_FLOOR = 0.02
PEAK_SPLIT = 0.5
N_TRANSFER = 12


def gap_stats(gap) -> dict:
    """Product-gap tail on one instance list. Equality at the floor counts as a pass."""
    arr = np.asarray(gap, dtype=np.float64).reshape(-1)
    if arr.size == 0 or not np.all(np.isfinite(arr)):
        raise ValueError("product gap has an empty or non-finite entry")
    n_ge = int(np.sum(arr >= GAP_FLOOR - 1e-15))
    n = int(arr.size)
    return {
        "n_inst": n,
        "gap_med": float(np.median(arr)),
        "gap_min": float(arr.min()),
        "n_gap_ge": n_ge,
        "n_gap_below": n - n_ge,
        "frac_gap_ge": float(n_ge / n),
    }


def _walsh12(row: dict) -> dict:
    try:
        return row["walsh"][N_TRANSFER]
    except KeyError:
        return row["walsh"][str(N_TRANSFER)]


def _q(row: dict, n: int) -> dict:
    quantum = row.get("quantum") or {}
    if n in quantum:
        return quantum[n]
    return quantum[str(n)]


def _gap_failures(q12: dict) -> list[str]:
    """Why an n=12 ground state misses the quantumness-safe bar. Empty means it clears."""
    bad = []
    if "prod_gap_med" not in q12 or "frac_gap_ge" not in q12 or "peak_med" not in q12:
        bad.append("n=12 gap statistics are missing")
        return bad
    gap = float(q12["prod_gap_med"])
    frac = float(q12["frac_gap_ge"])
    peak = float(q12["peak_med"])
    n_inst = int(q12.get("n_inst") or 0)
    n_ge = q12.get("n_gap_ge")
    if not math.isfinite(gap) or gap + 1e-12 < MED_GAP_MIN:
        bad.append(f"median product gap {gap:.3f} is under {MED_GAP_MIN:.2f}")
    if not math.isfinite(frac) or frac + 1e-12 < FRAC_MIN:
        if n_ge is None:
            bad.append(f"fraction of instances with gap ≥ {GAP_FLOOR:.2f} is {frac:.3f}")
        else:
            bad.append(
                f"{int(n_ge)}/{n_inst} instances have gap ≥ {GAP_FLOOR:.2f} "
                f"({frac:.3f}, bar {FRAC_MIN:.2f})"
            )
    if not math.isfinite(peak) or peak >= PEAK_SPLIT:
        bad.append(f"n=12 ground state is a bitstring (peak med {peak:.3f})")
    return bad


def _bitstring_ns(row: dict) -> list[str]:
    bad = []
    quantum = row.get("quantum") or {}
    for n in (6, 9, 12):
        if n not in quantum and str(n) not in quantum:
            bad.append(f"n={n} quantumness missing")
            continue
        peak = float(_q(row, n)["peak_med"])
        if peak >= PEAK_SPLIT:
            bad.append(f"n={n} ground state is a bitstring (peak med {peak:.3f})")
    return bad


def _pack(mu: float, row: dict) -> dict:
    cell = _walsh12(row)
    seeds = [float(x) for x in cell["seed_Rs"]]
    R = float(cell["R"])
    q12 = _q(row, 12)
    return {
        "mu": float(mu),
        "R": R,
        "seed_Rs": seeds,
        "seed_med_R": float(np.median(seeds)) if seeds else float("nan"),
        "mark": cell.get("mark") or mark_of(R),
        "gap": float(q12.get("prod_gap_med", float("nan"))),
        "frac": float(q12.get("frac_gap_ge", float("nan"))),
        "n_inst": int(q12.get("n_inst") or 0),
        "n_gap_ge": int(q12.get("n_gap_ge") or 0),
        "gap_min": float(q12.get("gap_min", float("nan"))),
        "peak": float(q12.get("peak_med", float("nan"))),
        "gap_fail": _gap_failures(q12),
        "bitstring": _bitstring_ns(row),
    }


def _better(a: dict, b: dict) -> bool:
    """True when a should win a tie-break against b. Lower R, then a thicker gap."""
    key_a = (float(a["R"]), -float(a["gap"]), -float(a["frac"]), abs(float(a["mu"])))
    key_b = (float(b["R"]), -float(b["gap"]), -float(b["frac"]), abs(float(b["mu"])))
    return key_a < key_b


def _pick(cands: list[dict]) -> dict:
    win = cands[0]
    for c in cands[1:]:
        if _better(c, win):
            win = c
    return win


def decide_mu_safe(by_mu: dict) -> dict:
    """PASS when the grid names a SAFE champ and an EDGE champ.

    SAFE minimizes frozen n=12 Walsh R among points with median product gap
    ≥ 0.05, ≥95% of instances at gap ≥ 0.02, and a non-bitstring ground state.
    EDGE is the unconstrained argmin of that R. The operating point is SAFE.
    """
    labels = [mu_label(m) for m in MU_GRID]
    missing = [lb for lb in labels if lb not in by_mu]
    if missing:
        return {
            "verdict": "needs_edit",
            "operating_mu": None,
            "safe_mu": None,
            "edge_mu": None,
            "reason": "μ grid is missing " + ", ".join(missing) + ".",
        }

    packed = []
    incomplete = []
    for mu in MU_GRID:
        row = by_mu[mu_label(mu)]
        try:
            packed.append(_pack(mu, row))
        except (KeyError, ValueError, TypeError) as exc:
            incomplete.append(f"{mu_label(mu)} ({exc})")
    if incomplete:
        return {
            "verdict": "needs_edit",
            "operating_mu": None,
            "safe_mu": None,
            "edge_mu": None,
            "reason": "Sweep cells are incomplete: " + "; ".join(incomplete) + ".",
        }

    for c in packed:
        if len(c["seed_Rs"]) < 3 or not all(math.isfinite(r) for r in c["seed_Rs"]):
            return {
                "verdict": "needs_edit",
                "operating_mu": float(c["mu"]),
                "safe_mu": None,
                "edge_mu": None,
                "reason": f"μ={c['mu']:+.2f} n=12 Walsh ratio is missing a 3-seed baseline.",
            }
        if not math.isfinite(c["R"]):
            return {
                "verdict": "needs_edit",
                "operating_mu": float(c["mu"]),
                "safe_mu": None,
                "edge_mu": None,
                "reason": f"μ={c['mu']:+.2f} n=12 Walsh R is not finite.",
            }
        published = None
        for known, val in RESUME_R.items():
            if abs(float(c["mu"]) - float(known)) < 1e-9:
                published = float(val)
                break
        if published is not None and abs(c["R"] - published) > ANCHOR_TOL:
            return {
                "verdict": "needs_edit",
                "operating_mu": float(c["mu"]),
                "safe_mu": None,
                "edge_mu": None,
                "reason": (
                    f"μ={c['mu']:+.2f} n=12 Walsh R={c['R']:.4f} does not match the "
                    f"resumed checkpoint {published:.4f}."
                ),
            }

    edge = _pick(packed)
    feasible = [c for c in packed if not c["gap_fail"] and not c["bitstring"]]
    infeasible = [c for c in packed if c["gap_fail"] or c["bitstring"]]
    if not feasible:
        return {
            "verdict": "needs_edit",
            "operating_mu": None,
            "safe_mu": None,
            "edge_mu": float(edge["mu"]),
            "edge_R": float(edge["R"]),
            "reason": (
                "No μ clears median product gap ≥ 0.05 and ≥95% of n=12 instances "
                f"with gap ≥ 0.02. Unconstrained argmin is μ={edge['mu']:+.2f} "
                f"at R={edge['R']:.3f}."
            ),
            "infeasible": [_brief(c) for c in infeasible],
        }

    safe = _pick(feasible)
    if float(safe["R"]) > TIE_R:
        return {
            "verdict": "KILL",
            "operating_mu": float(safe["mu"]),
            "safe_mu": float(safe["mu"]),
            "safe_R": float(safe["R"]),
            "edge_mu": float(edge["mu"]),
            "edge_R": float(edge["R"]),
            "reason": (
                f"SAFE champ μ={safe['mu']:+.2f} n=12 Walsh R={safe['R']:.3f} "
                "loses to Fourier QAOA on this H."
            ),
        }

    same = abs(float(safe["mu"]) - float(edge["mu"])) < 1e-9
    if same:
        reason = (
            f"Operate at μ={safe['mu']:+.2f}. SAFE and EDGE are the same champ: "
            f"frozen n=12 L=16 Walsh R={safe['R']:.3f} is the unconstrained minimum "
            f"on this grid, the median product gap is {safe['gap']:.3f}, and "
            f"{safe['n_gap_ge']}/{safe['n_inst']} instances have gap ≥ {GAP_FLOOR:.2f}."
        )
    else:
        why = edge["gap_fail"][0] if edge["gap_fail"] else (
            edge["bitstring"][0] if edge["bitstring"] else "misses the gap bar"
        )
        reason = (
            f"Operate at μ={safe['mu']:+.2f}. That is the SAFE champ: frozen n=12 "
            f"L=16 Walsh R={safe['R']:.3f} (seed-median R={safe['seed_med_R']:.3f}) "
            f"is the lowest ratio whose median product gap is at least {MED_GAP_MIN:.2f} "
            f"({safe['gap']:.3f}) and whose instance fraction at gap ≥ {GAP_FLOOR:.2f} "
            f"is at least {FRAC_MIN:.0%} ({safe['n_gap_ge']}/{safe['n_inst']}). "
            f"The unconstrained argmin is the EDGE champ μ={edge['mu']:+.2f} at "
            f"R={edge['R']:.3f} (median gap {edge['gap']:.3f}, "
            f"{edge['n_gap_ge']}/{edge['n_inst']} instances ≥ {GAP_FLOOR:.2f}). "
            f"The EDGE champ misses the safe bar: {why}."
        )
    return {
        "verdict": "PASS",
        "operating_mu": float(safe["mu"]),
        "safe_mu": float(safe["mu"]),
        "safe_R": float(safe["R"]),
        "edge_mu": float(edge["mu"]),
        "edge_R": float(edge["R"]),
        "same_champ": same,
        "reason": reason,
        "feasible": [_brief(c) for c in feasible],
        "infeasible": [_brief(c) for c in infeasible],
        "packed": [_brief(c) for c in packed],
    }


def _brief(c: dict) -> dict:
    return {
        "mu": float(c["mu"]),
        "R": float(c["R"]),
        "seed_med_R": float(c["seed_med_R"]),
        "gap": float(c["gap"]),
        "frac": float(c["frac"]),
        "n_gap_ge": int(c["n_gap_ge"]),
        "n_inst": int(c["n_inst"]),
        "gap_min": float(c["gap_min"]),
        "peak": float(c["peak"]),
        "mark": c["mark"],
        "gap_fail": list(c["gap_fail"]),
        "bitstring": list(c["bitstring"]),
    }


__all__ = [
    "ANCHOR_MU",
    "ANCHOR_TOL",
    "FRAC_MIN",
    "GAP_FLOOR",
    "MED_GAP_MIN",
    "MU_GRID",
    "PUBLISHED_ANCHOR_R",
    "RESUME_MUS",
    "RESUME_R",
    "cache_tag",
    "decide_mu_safe",
    "file_tag",
    "gap_stats",
    "mu_label",
]
