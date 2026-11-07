"""Verdict for the unsat Walsh λ sweep at fixed μ=−0.50.

    H = diag(unsat) − λ Σ_i X_i + μ Σ_cav Σ_{i<j} X_i X_j
    μ = −0.50
    λ ∈ {0.50, 0.65, 0.75, 0.90, 1.00}

The gate is unchanged. λ = 0.75 is the PR #26 operating point (frozen n=12
L=16 Walsh R = 0.150). A different λ replaces it only when that ratio drops
by at least 0.02 on the same bar as the μ sweep, and the ground state stays
quantum. n=21 is out of scope.
"""
from __future__ import annotations

import math

from Nov_3rd.seal_unsat import BEAT_R, FLIP_R, REVERSE_R, TIE_R, mark_of
from Nov_3rd.unsat_rank2_xx import MIN_DROP, cache_filename, clearly_below

# Decimal grid. 0.75 is the PR #26 champ, not a new train.
LAM_GRID = (0.50, 0.65, 0.75, 0.90, 1.00)
ANCHOR_LAM = 0.75
FIXED_MU = -0.50
# Exact ratio of medians from the PR #26 L=16 Walsh checkpoints, n=12, μ=−0.50.
PUBLISHED_ANCHOR_R = 0.15015595428732964
ANCHOR_TOL = 5e-4
CLEAR_DROP = float(MIN_DROP)
PROD_GAP_MIN = 0.02


def lam_label(lam: float) -> str:
    return f"{float(lam):.2f}"


def file_tag(lam: float) -> str:
    """Stem fragment. λ=0.75 is ``0p75``."""
    return f"{float(lam):.2f}".replace(".", "p")


def _walsh(row: dict, n: int) -> dict:
    try:
        return row["walsh"][n]
    except KeyError as exc:
        raise KeyError(n) from exc


def _quantum_failures(row: dict) -> list[str]:
    """Bitstring peaks and a product-like n=12 ground state."""
    q = row.get("quantum") or {}
    bad = []
    for n in (6, 9, 12):
        if n not in q:
            bad.append(f"n={n} quantumness missing")
            continue
        peak = float(q[n]["peak_med"])
        if peak >= 0.5:
            bad.append(f"n={n} ground state is a bitstring (peak med {peak:.3f})")
    if 12 in q and float(q[12]["prod_gap_med"]) + 1e-12 < PROD_GAP_MIN:
        gap = float(q[12]["prod_gap_med"])
        bad.append(f"n=12 product gap {gap:.3f} is under {PROD_GAP_MIN:.2f}")
    if 15 in q and float(q[15]["peak_med"]) >= 0.5:
        peak = float(q[15]["peak_med"])
        bad.append(f"n=15 ground state is a bitstring (peak med {peak:.3f})")
    return bad


def _reverses(row: dict) -> bool:
    r6 = float(_walsh(row, 6)["R"])
    r9 = float(_walsh(row, 9)["R"])
    r12 = float(_walsh(row, 12)["R"])
    mark6 = _walsh(row, 6).get("mark") or mark_of(r6)
    mark9 = _walsh(row, 9).get("mark") or mark_of(r9)
    return mark6 == "BEAT" and mark9 == "UNDER" and r12 + 1e-12 >= REVERSE_R


def _n15_flip(row: dict) -> bool:
    if 15 not in row.get("walsh", {}):
        return False
    return float(_walsh(row, 15)["R"]) + 1e-12 >= FLIP_R


def _seed_beats(cell: dict) -> int:
    return sum(float(r) <= BEAT_R for r in cell["seed_Rs"])


def _pack_challenger(lam: float, row: dict, anchor_R: float) -> dict:
    cell = _walsh(row, 12)
    seeds = [float(x) for x in cell["seed_Rs"]]
    R = float(cell["R"])
    return {
        "lam": float(lam),
        "R": R,
        "seed_Rs": seeds,
        "clear": clearly_below(R, anchor_R, seeds, CLEAR_DROP),
        "q_fail": _quantum_failures(row),
        "reverses": _reverses(row),
        "n15_flip": _n15_flip(row),
        "n_seed_beat": _seed_beats(cell),
        "mark": cell.get("mark") or mark_of(R),
        "R15": float(_walsh(row, 15)["R"]) if 15 in row["walsh"] else None,
        "peak12": float(row["quantum"][12]["peak_med"]),
        "gap12": float(row["quantum"][12]["prod_gap_med"]),
    }


def decide_lambda_sweep(by_lam: dict) -> dict:
    """PASS / needs_edit / KILL for the λ grid at μ=−0.50.

    PASS when some quantum λ is clearly below the in-run λ=0.75 n=12 Walsh
    ratio, or when none is and λ=0.75 remains the operating point.

    needs_edit when the grid, the anchor checkpoint, or a clear challenger's
    transfer gate is incomplete.

    KILL when the operating point's ground state is a bitstring, Walsh loses
    to Fourier there, or the L=16 advantage reverses.
    """
    labels = [lam_label(lam) for lam in LAM_GRID]
    missing = [lb for lb in labels if lb not in by_lam]
    if missing:
        return {
            "verdict": "needs_edit",
            "null": False,
            "reason": "λ grid is missing " + ", ".join(missing) + ".",
            "operating_lam": None,
        }

    anchor_label = lam_label(ANCHOR_LAM)
    anchor = by_lam[anchor_label]
    try:
        a_cell = _walsh(anchor, 12)
        for n in (6, 9, 15):
            _walsh(anchor, n)
        aR = float(a_cell["R"])
        a_seeds = [float(x) for x in a_cell["seed_Rs"]]
    except KeyError as exc:
        return {
            "verdict": "needs_edit",
            "null": False,
            "reason": f"λ=0.75 Walsh cell is missing n={exc}.",
            "operating_lam": None,
        }
    if len(a_seeds) < 3 or not all(math.isfinite(r) for r in a_seeds) or not math.isfinite(aR):
        return {
            "verdict": "needs_edit",
            "null": False,
            "reason": "λ=0.75 n=12 Walsh ratio is missing a 3-seed baseline.",
            "operating_lam": ANCHOR_LAM,
        }
    if abs(aR - PUBLISHED_ANCHOR_R) > ANCHOR_TOL:
        return {
            "verdict": "needs_edit",
            "null": False,
            "reason": (
                f"λ=0.75 n=12 Walsh R={aR:.4f} does not match the PR #26 checkpoint "
                f"{PUBLISHED_ANCHOR_R:.4f}."
            ),
            "operating_lam": ANCHOR_LAM,
            "R_anchor": aR,
        }

    anchor_q = _quantum_failures(anchor)
    bitstring = [msg for msg in anchor_q if "bitstring" in msg and "n=15" not in msg]
    if bitstring:
        return {
            "verdict": "KILL",
            "null": False,
            "reason": "λ=0.75 " + "; ".join(bitstring) + ".",
            "operating_lam": ANCHOR_LAM,
            "R_anchor": aR,
        }

    packed = []
    incomplete = []
    for lam in LAM_GRID:
        if abs(float(lam) - ANCHOR_LAM) < 1e-12:
            continue
        row = by_lam[lam_label(lam)]
        try:
            packed.append(_pack_challenger(lam, row, aR))
        except KeyError as exc:
            incomplete.append(f"{lam_label(lam)} missing {exc}")
    if incomplete:
        return {
            "verdict": "needs_edit",
            "null": False,
            "reason": "Sweep cells are incomplete: " + "; ".join(incomplete) + ".",
            "operating_lam": ANCHOR_LAM,
            "R_anchor": aR,
        }

    eligible = [c for c in packed if not c["q_fail"] and not c["reverses"]]
    disqualified = [c for c in packed if c["q_fail"] or c["reverses"]]
    clear = [c for c in eligible if c["clear"]["clear"] and c["R"] + 1e-12 < aR]
    clear.sort(key=lambda c: (c["R"], abs(c["lam"] - ANCHOR_LAM)))
    eligible_sorted = sorted(eligible, key=lambda c: c["R"])

    info = {
        "R_anchor": aR,
        "anchor_seed_Rs": a_seeds,
        "n_clear": len(clear),
        "n_disqualified": len(disqualified),
        "null": False,
        "fixed_mu": FIXED_MU,
    }
    if eligible_sorted:
        info["best_quantum_lam"] = eligible_sorted[0]["lam"]
        info["best_quantum_R"] = eligible_sorted[0]["R"]
    else:
        info["best_quantum_lam"] = ANCHOR_LAM
        info["best_quantum_R"] = aR

    def _near_sentence() -> str:
        if not eligible_sorted:
            return "No other λ kept a quantum ground state."
        near = eligible_sorted[0]
        drop = float(near["clear"]["drop"])
        return (
            f"Lowest other quantum n=12 R is λ={near['lam']:.2f} at {near['R']:.3f} "
            f"(drop {drop:+.3f} against {aR:.3f}, bar {CLEAR_DROP:.2f})."
        )

    def _disq_sentence() -> str:
        if not disqualified:
            return ""
        bits = []
        for c in sorted(disqualified, key=lambda x: x["R"]):
            why = c["q_fail"][0] if c["q_fail"] else "L=16 advantage reverses"
            bits.append(f"λ={c['lam']:.2f} R={c['R']:.3f} ({why})")
        return " Not eligible: " + "; ".join(bits) + "."

    if clear:
        win = clear[0]
        row = by_lam[lam_label(win["lam"])]
        if _reverses(row) or float(win["R"]) > TIE_R:
            info["verdict"] = "KILL"
            info["operating_lam"] = float(win["lam"])
            info["reason"] = (
                f"λ={win['lam']:.2f} n=12 Walsh R={win['R']:.3f} loses to Fourier "
                f"or the L=16 advantage reverses."
            )
            return info
        missed = []
        if float(win["R"]) > BEAT_R:
            missed.append(f"n=12 R={win['R']:.3f} is above {BEAT_R}")
        if win["n_seed_beat"] < 2:
            missed.append(f"only {win['n_seed_beat']}/3 seeds beat Fourier")
        if win["n15_flip"]:
            missed.append(f"n=15 R={win['R15']:.3f} flips past {FLIP_R}")
        if win["q_fail"]:
            missed.extend(win["q_fail"])
        if missed:
            info["verdict"] = "needs_edit"
            info["operating_lam"] = float(win["lam"])
            info["reason"] = (
                f"λ={win['lam']:.2f} n=12 R={win['R']:.3f} is clearly below λ=0.75 "
                f"but does not clear the transfer gates: " + "; ".join(missed) + "."
            )
            return info
        info["verdict"] = "PASS"
        info["operating_lam"] = float(win["lam"])
        info["R12"] = float(win["R"])
        info["reason"] = (
            f"λ={win['lam']:.2f} frozen n=12 L=16 Walsh R={win['R']:.3f} is a BEAT and clearly "
            f"below the λ=0.75 ratio {aR:.3f} "
            f"(drop {win['clear']['drop']:.3f}, seed-median R={win['clear']['seed_median']:.3f}). "
            f"{win['n_seed_beat']}/3 seeds beat. "
            f"n=12 peak {win['peak12']:.3f}, product gap {win['gap12']:.3f}. "
            f"n=15 R={win['R15']:.3f}."
            + _disq_sentence()
        )
        return info

    if anchor_q:
        info["verdict"] = "needs_edit"
        info["operating_lam"] = ANCHOR_LAM
        info["reason"] = "λ=0.75 quantumness fails: " + "; ".join(anchor_q) + "."
        return info
    if _reverses(anchor) or aR > TIE_R:
        info["verdict"] = "KILL"
        info["operating_lam"] = ANCHOR_LAM
        info["reason"] = (
            f"λ=0.75 n=12 Walsh R={aR:.3f} loses to Fourier or the L=16 advantage reverses."
        )
        return info
    if aR > BEAT_R or _seed_beats(a_cell) < 2 or _n15_flip(anchor):
        info["verdict"] = "needs_edit"
        info["operating_lam"] = ANCHOR_LAM
        info["reason"] = (
            f"λ=0.75 remains the lowest clear ratio (R={aR:.3f}) but no longer clears "
            f"the transfer gates (seed beats {_seed_beats(a_cell)}/3, "
            f"n=15 flip={_n15_flip(anchor)})."
        )
        return info

    info["verdict"] = "PASS"
    info["operating_lam"] = ANCHOR_LAM
    info["R12"] = aR
    info["still_optimal"] = True
    info["reason"] = (
        f"λ=0.75 remains the operating point. Frozen n=12 L=16 Walsh R={aR:.3f} "
        f"matches the PR #26 checkpoint and no other quantum λ on the grid drops it "
        f"by {CLEAR_DROP:.2f}. {_near_sentence()}{_disq_sentence()}"
    )
    return info


__all__ = [
    "ANCHOR_LAM",
    "ANCHOR_TOL",
    "CLEAR_DROP",
    "FIXED_MU",
    "LAM_GRID",
    "PUBLISHED_ANCHOR_R",
    "cache_filename",
    "decide_lambda_sweep",
    "file_tag",
    "lam_label",
]
