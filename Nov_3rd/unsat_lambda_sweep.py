"""Verdict for the unsat Walsh λ sweep at fixed μ=−1.00.

    H = diag(unsat) − λ Σ_i X_i + μ Σ_cav Σ_{i<j} X_i X_j
    μ = −1.00
    λ ∈ {0.50, 0.65, 0.75, 0.90, 1.00, 1.25}

The gate is unchanged. λ = 0.75 is the PR #28 checkpoint at this μ (frozen
n=12 L=16 Walsh R = 0.097), not a new train. The operating λ is the lowest
frozen n=12 Walsh ratio among cells whose median product gap is at least
0.02. If the lowest ratio on the whole grid misses that gap, the report
names that unconstrained winner and the safe winner. The other transfer
gates stay. n=21 is out of scope.
"""
from __future__ import annotations

import math

import numpy as np

from Nov_3rd.seal_unsat import BEAT_R, FLIP_R, REVERSE_R, TIE_R, mark_of
from Nov_3rd.unsat_rank2_xx import cache_filename

# Decimal grid. 0.75 is the PR #28 champ at μ=−1.00, not a new train.
LAM_GRID = (0.50, 0.65, 0.75, 0.90, 1.00, 1.25)
ANCHOR_LAM = 0.75
FIXED_MU = -1.00
# Exact ratio of medians from the PR #28 L=16 Walsh checkpoints, n=12, μ=−1.00, λ=0.75.
PUBLISHED_ANCHOR_R = 0.09675474400205866
ANCHOR_TOL = 5e-4
PROD_GAP_MIN = 0.02


def lam_label(lam: float) -> str:
    return f"{float(lam):.2f}"


def file_tag(lam: float) -> str:
    """Stem fragment. λ=0.75 is ``0p75``. λ=1.25 is ``1p25``."""
    return f"{float(lam):.2f}".replace(".", "p")


def mu_ckpt_tag(mu: float) -> str:
    """Checkpoint stem. μ=−1.00 is ``mu-1p00``."""
    sign = "-" if float(mu) < 0 else ""
    return f"mu{sign}{abs(float(mu)):.2f}".replace(".", "p")


def gap_tail(gap, peak) -> dict:
    """Instance product-gap tail. The gate uses the median; the fraction under 0.02 is reported."""
    gaps = np.asarray(list(gap), dtype=np.float64)
    peaks = np.asarray(list(peak), dtype=np.float64)
    if gaps.size == 0 or gaps.shape != peaks.shape:
        raise ValueError("gap and peak lengths")
    n_below = int(np.sum(gaps < PROD_GAP_MIN))
    n_inst = int(gaps.shape[0])
    return {
        "frac_gap_below": float(n_below) / float(n_inst),
        "n_gap_below": n_below,
        "n_inst": n_inst,
        "gap_min": float(gaps.min()),
        "peak_max": float(peaks.max()),
        "peak_med": float(np.median(peaks)),
        "prod_gap_med": float(np.median(gaps)),
    }


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


def _pack_cell(lam: float, row: dict) -> dict:
    cell = _walsh(row, 12)
    seeds = [float(x) for x in cell["seed_Rs"]]
    R = float(cell["R"])
    gap12 = float(row["quantum"][12]["prod_gap_med"])
    return {
        "lam": float(lam),
        "R": R,
        "seed_Rs": seeds,
        "q_fail": _quantum_failures(row),
        "reverses": _reverses(row),
        "n15_flip": _n15_flip(row),
        "n_seed_beat": _seed_beats(cell),
        "mark": cell.get("mark") or mark_of(R),
        "R15": float(_walsh(row, 15)["R"]) if 15 in row["walsh"] else None,
        "peak12": float(row["quantum"][12]["peak_med"]),
        "gap12": gap12,
        "gap_ok": bool(gap12 + 1e-12 >= PROD_GAP_MIN),
    }


def _best(cands: list[dict]) -> dict:
    return sorted(cands, key=lambda c: (c["R"], abs(c["lam"] - ANCHOR_LAM), c["lam"]))[0]


def _bitstring_kill(c: dict) -> list[str]:
    return [msg for msg in c["q_fail"] if "bitstring" in msg and not msg.startswith("n=15")]


def _transfer_misses(c: dict) -> list[str]:
    missed = []
    if float(c["R"]) > BEAT_R:
        missed.append(f"n=12 R={c['R']:.3f} is above {BEAT_R}")
    if c["n_seed_beat"] < 2:
        missed.append(f"only {c['n_seed_beat']}/3 seeds beat Fourier")
    if c["n15_flip"]:
        missed.append(f"n=15 R={c['R15']:.3f} flips past {FLIP_R}")
    for msg in c["q_fail"]:
        if msg.startswith("n=15") and "bitstring" in msg:
            missed.append(msg)
    return missed


def _loses(c: dict) -> bool:
    return bool(c["reverses"] or float(c["R"]) > TIE_R)


def decide_lambda_sweep(by_lam: dict) -> dict:
    """PASS / needs_edit / KILL for the λ grid at μ=−1.00.

    The operating λ is the lowest frozen n=12 Walsh ratio among cells with
    median product gap ≥ 0.02, provided the transfer gates hold. If the
    lowest ratio on the grid misses that gap, both that unconstrained winner
    and the safe winner are reported.

    needs_edit when the grid, the PR #28 anchor, or the chosen cell's
    transfer gate is incomplete.

    KILL when the chosen cell's ground state is a bitstring, Walsh loses to
    Fourier there, or the L=16 advantage reverses.
    """
    labels = [lam_label(lam) for lam in LAM_GRID]
    missing = [lb for lb in labels if lb not in by_lam]
    if missing:
        return {
            "verdict": "needs_edit",
            "null": False,
            "reason": "λ grid is missing " + ", ".join(missing) + ".",
            "operating_lam": None,
            "fixed_mu": FIXED_MU,
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
            "fixed_mu": FIXED_MU,
        }
    if len(a_seeds) < 3 or not all(math.isfinite(r) for r in a_seeds) or not math.isfinite(aR):
        return {
            "verdict": "needs_edit",
            "null": False,
            "reason": "λ=0.75 n=12 Walsh ratio is missing a 3-seed baseline.",
            "operating_lam": ANCHOR_LAM,
            "fixed_mu": FIXED_MU,
        }
    if abs(aR - PUBLISHED_ANCHOR_R) > ANCHOR_TOL:
        return {
            "verdict": "needs_edit",
            "null": False,
            "reason": (
                f"λ=0.75 n=12 Walsh R={aR:.4f} does not match the PR #28 checkpoint "
                f"{PUBLISHED_ANCHOR_R:.4f}."
            ),
            "operating_lam": ANCHOR_LAM,
            "R_anchor": aR,
            "fixed_mu": FIXED_MU,
        }

    packed = []
    incomplete = []
    for lam in LAM_GRID:
        row = by_lam[lam_label(lam)]
        try:
            packed.append(_pack_cell(lam, row))
        except KeyError as exc:
            incomplete.append(f"{lam_label(lam)} missing {exc}")
    if incomplete:
        return {
            "verdict": "needs_edit",
            "null": False,
            "reason": "Sweep cells are incomplete: " + "; ".join(incomplete) + ".",
            "operating_lam": ANCHOR_LAM,
            "R_anchor": aR,
            "fixed_mu": FIXED_MU,
        }

    unconstrained = _best(packed)
    safe_pool = [c for c in packed if c["gap_ok"]]
    safe = _best(safe_pool) if safe_pool else None
    split = not bool(unconstrained["gap_ok"])
    info = {
        "R_anchor": aR,
        "anchor_seed_Rs": a_seeds,
        "null": False,
        "fixed_mu": FIXED_MU,
        "split": split,
        "unconstrained_lam": float(unconstrained["lam"]),
        "unconstrained_R": float(unconstrained["R"]),
        "unconstrained_gap": float(unconstrained["gap12"]),
        "safe_lam": None if safe is None else float(safe["lam"]),
        "safe_R": None if safe is None else float(safe["R"]),
        "safe_gap": None if safe is None else float(safe["gap12"]),
        "n_safe": len(safe_pool),
    }

    def _both() -> str:
        if not split:
            return ""
        if safe is None:
            return (
                f" Unconstrained winner is λ={unconstrained['lam']:.2f} "
                f"(n=12 R={unconstrained['R']:.3f}, median gap {unconstrained['gap12']:.3f}). "
                "No cell has median n=12 product gap ≥ 0.02."
            )
        return (
            f" Unconstrained winner is λ={unconstrained['lam']:.2f} "
            f"(n=12 R={unconstrained['R']:.3f}, median gap {unconstrained['gap12']:.3f}). "
            f"Safe winner is λ={safe['lam']:.2f} "
            f"(n=12 R={safe['R']:.3f}, median gap {safe['gap12']:.3f})."
        )

    if safe is None:
        bits = _bitstring_kill(unconstrained)
        if bits or _loses(unconstrained):
            info["verdict"] = "KILL"
            info["operating_lam"] = float(unconstrained["lam"])
            why = "; ".join(bits) if bits else "Walsh loses to Fourier or the L=16 advantage reverses"
            info["reason"] = (
                f"No λ keeps a median n=12 product gap ≥ {PROD_GAP_MIN:.2f}. "
                f"λ={unconstrained['lam']:.2f} n=12 R={unconstrained['R']:.3f}: {why}."
                + _both()
            )
            return info
        info["verdict"] = "needs_edit"
        info["operating_lam"] = None
        info["reason"] = (
            f"No λ keeps a median n=12 product gap ≥ {PROD_GAP_MIN:.2f}."
            + _both()
        )
        return info

    win = safe
    row = by_lam[lam_label(win["lam"])]
    bits = _bitstring_kill(win)
    if bits:
        info["verdict"] = "KILL"
        info["operating_lam"] = float(win["lam"])
        info["R12"] = float(win["R"])
        info["reason"] = (
            f"λ={win['lam']:.2f} has the lowest n=12 Walsh R among cells with median gap "
            f"≥ {PROD_GAP_MIN:.2f} (R={win['R']:.3f}) but "
            + "; ".join(bits)
            + "."
            + _both()
        )
        return info
    if _loses(win) or _reverses(row):
        info["verdict"] = "KILL"
        info["operating_lam"] = float(win["lam"])
        info["R12"] = float(win["R"])
        info["reason"] = (
            f"λ={win['lam']:.2f} n=12 Walsh R={win['R']:.3f} loses to Fourier "
            f"or the L=16 advantage reverses."
            + _both()
        )
        return info
    missed = _transfer_misses(win)
    if missed:
        info["verdict"] = "needs_edit"
        info["operating_lam"] = float(win["lam"])
        info["R12"] = float(win["R"])
        info["reason"] = (
            f"λ={win['lam']:.2f} has the lowest n=12 Walsh R among cells with median gap "
            f"≥ {PROD_GAP_MIN:.2f} (R={win['R']:.3f}) but does not clear the transfer gates: "
            + "; ".join(missed)
            + "."
            + _both()
        )
        return info

    info["verdict"] = "PASS"
    info["operating_lam"] = float(win["lam"])
    info["R12"] = float(win["R"])
    info["still_optimal"] = abs(float(win["lam"]) - ANCHOR_LAM) < 1e-12
    if info["still_optimal"] and not split:
        info["reason"] = (
            f"λ=0.75 remains the operating point. It has the lowest frozen n=12 L=16 Walsh "
            f"R on the gap-safe grid (R={aR:.3f}, median gap {win['gap12']:.3f}). "
            f"{win['n_seed_beat']}/3 seeds beat. "
            f"n=12 peak {win['peak12']:.3f}. n=15 R={win['R15']:.3f}. "
            f"The checkpoint matches PR #28."
        )
    elif info["still_optimal"] and split:
        info["reason"] = (
            f"λ=0.75 remains the operating point among cells with median product gap "
            f"≥ {PROD_GAP_MIN:.2f} (R={aR:.3f}, median gap {win['gap12']:.3f}). "
            f"{win['n_seed_beat']}/3 seeds beat. "
            f"n=12 peak {win['peak12']:.3f}. n=15 R={win['R15']:.3f}."
            + _both()
        )
    else:
        info["reason"] = (
            f"λ={win['lam']:.2f} has the lowest frozen n=12 L=16 Walsh R among cells with "
            f"median product gap ≥ {PROD_GAP_MIN:.2f} "
            f"(R={win['R']:.3f}, median gap {win['gap12']:.3f}, "
            f"λ=0.75 is {aR:.3f}). "
            f"{win['n_seed_beat']}/3 seeds beat. "
            f"n=12 peak {win['peak12']:.3f}. n=15 R={win['R15']:.3f}."
            + _both()
        )
    return info


__all__ = [
    "ANCHOR_LAM",
    "ANCHOR_TOL",
    "FIXED_MU",
    "LAM_GRID",
    "PROD_GAP_MIN",
    "PUBLISHED_ANCHOR_R",
    "cache_filename",
    "decide_lambda_sweep",
    "file_tag",
    "gap_tail",
    "lam_label",
    "mu_ckpt_tag",
]
