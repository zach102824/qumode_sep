"""2D quantumness-safe grid for unsat Walsh.

    H = diag(unsat) − λ Σ_i X_i + μ Σ_cav Σ_{i<j} X_i X_j
    μ ∈ {−0.90, −0.75, −0.60, −0.50}
    λ ∈ {0.75, 0.90, 1.00}

Train n=6 and n=9 only, L=16, the PR #26 budget. Freeze n=12. n=21 is out
of scope.

R is the frozen n=12 L=16 Walsh ratio of medians against Fourier QAOA on the
same (λ, μ). Lower R is the improvement direction from PR #26.

SAFE minimizes that R among cells with median product gap ≥ 0.05 and at
least 95% of the n=12 instances at gap ≥ 0.02. EDGE minimizes that R among
cells with at least 95% of the n=12 instances at gap ≥ 0.02. The median-gap
bar is the only extra SAFE constraint. The operating point is SAFE.
"""
from __future__ import annotations

import math

import numpy as np

from Nov_3rd.seal_unsat import BEAT_R, FLIP_R, REVERSE_R, TIE_R, mark_of

# Decimal grid. Rows are μ, then λ.
MU_GRID = (-0.90, -0.75, -0.60, -0.50)
LAM_GRID = (0.75, 0.90, 1.00)
CELLS = tuple((float(lam), float(mu)) for mu in MU_GRID for lam in LAM_GRID)

# Identical to a finished PR #26-budget run. Not a retrain.
# λ=0.75 is the PR #31 μ grid. λ∈{0.90, 1.00} at μ=−0.50 is the PR #29 λ sweep.
RESUME_CELLS = (
    (0.75, -0.90),
    (0.75, -0.75),
    (0.75, -0.60),
    (0.75, -0.50),
    (0.90, -0.50),
    (1.00, -0.50),
)
# Exact n=12 Walsh ratio of medians from those checkpoints.
PUBLISHED_R = {
    (0.75, -0.90): 0.0918444208231456,
    (0.75, -0.75): 0.1237824099740523,
    (0.75, -0.60): 0.1550306674441098,
    (0.75, -0.50): 0.15015595428732964,
    (0.90, -0.50): 0.1775630745000315,
    (1.00, -0.50): 0.17093849009163756,
}
ANCHOR = (0.75, -0.50)
ANCHOR_TOL = 5e-4
MED_GAP_MIN = 0.05
FRAC_MIN = 0.95
GAP_FLOOR = 0.02
PEAK_SPLIT = 0.5
N_TRANSFER = 12


def cell_key(lam: float, mu: float) -> str:
    return f"l{float(lam):.2f}_m{float(mu):+.2f}"


def file_stem(lam: float, mu: float) -> str:
    """Checkpoint stem. λ=0.75, μ=−0.90 is ``l0p75_m-0p90``."""
    ltag = f"{float(lam):.2f}".replace(".", "p")
    mtag = f"{float(mu):+.2f}".replace("+", "").replace(".", "p")
    return f"l{ltag}_m{mtag}"


def is_resume(lam: float, mu: float) -> bool:
    return any(
        abs(float(lam) - float(a)) < 1e-12 and abs(float(mu) - float(b)) < 1e-12
        for a, b in RESUME_CELLS
    )


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


def _walsh(row: dict, n: int) -> dict:
    try:
        return row["walsh"][n]
    except KeyError:
        return row["walsh"][str(n)]


def _q(row: dict, n: int) -> dict:
    quantum = row.get("quantum") or {}
    if n in quantum:
        return quantum[n]
    return quantum[str(n)]


def _bitstring_ns(row: dict) -> list[str]:
    bad = []
    quantum = row.get("quantum") or {}
    for n in (6, 9, 12):
        if n not in quantum and str(n) not in quantum:
            bad.append(f"n={n} quantumness missing")
            continue
        peak = float(_q(row, n)["peak_med"])
        if not math.isfinite(peak) or peak >= PEAK_SPLIT:
            bad.append(f"n={n} ground state is a bitstring (peak med {peak:.3f})")
    return bad


def _safe_gap_failures(q12: dict) -> list[str]:
    """Why an n=12 ground state misses the SAFE bar. Empty means it clears."""
    bad = []
    if "prod_gap_med" not in q12 or "frac_gap_ge" not in q12 or "peak_med" not in q12:
        bad.append("n=12 gap statistics are missing")
        return bad
    gap = float(q12["prod_gap_med"])
    frac = float(q12["frac_gap_ge"])
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
    return bad


def _edge_gap_failures(q12: dict) -> list[str]:
    """EDGE only requires the instance bar. The median-gap bar is SAFE-only."""
    bad = []
    if "frac_gap_ge" not in q12:
        bad.append("n=12 gap statistics are missing")
        return bad
    frac = float(q12["frac_gap_ge"])
    n_inst = int(q12.get("n_inst") or 0)
    n_ge = q12.get("n_gap_ge")
    if not math.isfinite(frac) or frac + 1e-12 < FRAC_MIN:
        if n_ge is None:
            bad.append(f"fraction of instances with gap ≥ {GAP_FLOOR:.2f} is {frac:.3f}")
        else:
            bad.append(
                f"{int(n_ge)}/{n_inst} instances have gap ≥ {GAP_FLOOR:.2f} "
                f"({frac:.3f}, bar {FRAC_MIN:.2f})"
            )
    return bad


def _reverses(row: dict) -> bool:
    r6 = float(_walsh(row, 6)["R"])
    r9 = float(_walsh(row, 9)["R"])
    r12 = float(_walsh(row, 12)["R"])
    mark6 = _walsh(row, 6).get("mark") or mark_of(r6)
    mark9 = _walsh(row, 9).get("mark") or mark_of(r9)
    return mark6 == "BEAT" and mark9 == "UNDER" and r12 + 1e-12 >= REVERSE_R


def _pack(lam: float, mu: float, row: dict) -> dict:
    cell = _walsh(row, N_TRANSFER)
    seeds = [float(x) for x in cell["seed_Rs"]]
    R = float(cell["R"])
    q12 = _q(row, 12)
    return {
        "lam": float(lam),
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
        "safe_fail": _safe_gap_failures(q12),
        "edge_fail": _edge_gap_failures(q12),
        "bitstring": _bitstring_ns(row),
        "reverses": _reverses(row),
        "R15": float(_walsh(row, 15)["R"]),
        "n_seed_beat": int(sum(float(r) <= BEAT_R for r in seeds)),
    }


def _better(a: dict, b: dict) -> bool:
    """True when a should win. Lower R, then a thicker gap, then a higher fraction."""
    key_a = (
        float(a["R"]),
        -float(a["gap"]),
        -float(a["frac"]),
        float(a["lam"]),
        abs(float(a["mu"])),
    )
    key_b = (
        float(b["R"]),
        -float(b["gap"]),
        -float(b["frac"]),
        float(b["lam"]),
        abs(float(b["mu"])),
    )
    return key_a < key_b


def _pick(cands: list[dict]) -> dict:
    win = cands[0]
    for c in cands[1:]:
        if _better(c, win):
            win = c
    return win


def _published(lam: float, mu: float):
    for (a, b), val in PUBLISHED_R.items():
        if abs(float(lam) - float(a)) < 1e-12 and abs(float(mu) - float(b)) < 1e-12:
            return float(val)
    return None


def _transfer_misses(c: dict) -> list[str]:
    missed = []
    if float(c["R"]) > BEAT_R:
        missed.append(f"n=12 R={c['R']:.3f} is above {BEAT_R}")
    if int(c["n_seed_beat"]) < 2:
        missed.append(f"only {int(c['n_seed_beat'])}/3 seeds beat Fourier")
    if float(c["R15"]) + 1e-12 >= FLIP_R:
        missed.append(f"n=15 R={c['R15']:.3f} flips past {FLIP_R}")
    return missed


def _where(c: dict) -> str:
    return f"λ={c['lam']:.2f}, μ={c['mu']:+.2f}"


def decide_lambda_mu_grid(by_cell: dict) -> dict:
    """PASS when the grid names a SAFE champ and an EDGE champ and SAFE clears the gates.

    SAFE minimizes frozen n=12 Walsh R among cells with median product gap
    ≥ 0.05, ≥95% of instances at gap ≥ 0.02, and a non-bitstring ground state
    on n=6, n=9, and n=12.

    EDGE minimizes that R among cells with ≥95% of instances at gap ≥ 0.02
    and the same non-bitstring gate. A thin median does not remove a cell
    from EDGE.

    The operating point is SAFE. power_p does not choose the cell. n=15 does
    not choose the cell. n=21 is out of scope.
    """
    labels = [cell_key(lam, mu) for lam, mu in CELLS]
    missing = [lb for lb in labels if lb not in by_cell]
    if missing:
        return {
            "verdict": "needs_edit",
            "operating_lam": None,
            "operating_mu": None,
            "safe_lam": None,
            "safe_mu": None,
            "edge_lam": None,
            "edge_mu": None,
            "reason": "λ×μ grid is missing " + ", ".join(missing) + ".",
        }

    packed = []
    incomplete = []
    for lam, mu in CELLS:
        row = by_cell[cell_key(lam, mu)]
        try:
            packed.append(_pack(lam, mu, row))
        except (KeyError, ValueError, TypeError) as exc:
            incomplete.append(f"{cell_key(lam, mu)} ({exc})")
    if incomplete:
        return {
            "verdict": "needs_edit",
            "operating_lam": None,
            "operating_mu": None,
            "safe_lam": None,
            "safe_mu": None,
            "edge_lam": None,
            "edge_mu": None,
            "reason": "Sweep cells are incomplete: " + "; ".join(incomplete) + ".",
        }

    for c in packed:
        if len(c["seed_Rs"]) < 3 or not all(math.isfinite(r) for r in c["seed_Rs"]):
            return _edit(c, f"{_where(c)} n=12 Walsh ratio is missing a 3-seed baseline.")
        if not math.isfinite(c["R"]):
            return _edit(c, f"{_where(c)} n=12 Walsh R is not finite.")
        published = _published(c["lam"], c["mu"])
        if published is not None and abs(c["R"] - published) > ANCHOR_TOL:
            return _edit(
                c,
                f"{_where(c)} n=12 Walsh R={c['R']:.4f} does not match the "
                f"resumed checkpoint {published:.4f}.",
            )

    safe_pool = [c for c in packed if not c["safe_fail"] and not c["bitstring"]]
    edge_pool = [c for c in packed if not c["edge_fail"] and not c["bitstring"]]
    if not edge_pool:
        return {
            "verdict": "needs_edit",
            "operating_lam": None,
            "operating_mu": None,
            "safe_lam": None,
            "safe_mu": None,
            "edge_lam": None,
            "edge_mu": None,
            "reason": (
                "No cell has ≥95% of n=12 instances at gap ≥ 0.02 "
                "with a non-bitstring ground state."
            ),
            "packed": [_brief(c) for c in packed],
        }
    edge = _pick(edge_pool)
    if not safe_pool:
        return {
            "verdict": "needs_edit",
            "operating_lam": None,
            "operating_mu": None,
            "safe_lam": None,
            "safe_mu": None,
            "edge_lam": float(edge["lam"]),
            "edge_mu": float(edge["mu"]),
            "edge_R": float(edge["R"]),
            "reason": (
                "No cell clears median product gap ≥ 0.05 and ≥95% of n=12 instances "
                f"with gap ≥ 0.02. EDGE champ is {_where(edge)} at R={edge['R']:.3f}."
            ),
            "edge_pool": [_brief(c) for c in edge_pool],
            "packed": [_brief(c) for c in packed],
        }
    safe = _pick(safe_pool)
    if c_reverses_or_loses(safe):
        return {
            "verdict": "KILL",
            "operating_lam": float(safe["lam"]),
            "operating_mu": float(safe["mu"]),
            "safe_lam": float(safe["lam"]),
            "safe_mu": float(safe["mu"]),
            "safe_R": float(safe["R"]),
            "edge_lam": float(edge["lam"]),
            "edge_mu": float(edge["mu"]),
            "edge_R": float(edge["R"]),
            "reason": (
                f"SAFE champ {_where(safe)} n=12 Walsh R={safe['R']:.3f} "
                "loses to Fourier QAOA on this H or the L=16 advantage reverses."
            ),
            "safe_pool": [_brief(c) for c in safe_pool],
            "edge_pool": [_brief(c) for c in edge_pool],
        }
    missed = _transfer_misses(safe)
    if missed:
        return {
            "verdict": "needs_edit",
            "operating_lam": float(safe["lam"]),
            "operating_mu": float(safe["mu"]),
            "safe_lam": float(safe["lam"]),
            "safe_mu": float(safe["mu"]),
            "safe_R": float(safe["R"]),
            "edge_lam": float(edge["lam"]),
            "edge_mu": float(edge["mu"]),
            "edge_R": float(edge["R"]),
            "reason": (
                f"SAFE champ {_where(safe)} n=12 Walsh R={safe['R']:.3f} "
                "does not clear the transfer gates: " + "; ".join(missed) + "."
            ),
            "safe_pool": [_brief(c) for c in safe_pool],
            "edge_pool": [_brief(c) for c in edge_pool],
        }

    same = (
        abs(float(safe["lam"]) - float(edge["lam"])) < 1e-12
        and abs(float(safe["mu"]) - float(edge["mu"])) < 1e-12
    )
    if same:
        reason = (
            f"Operate at {_where(safe)}. SAFE and EDGE are the same champ: "
            f"frozen n=12 L=16 Walsh R={safe['R']:.3f} is the lowest ratio with "
            f"median product gap ≥ {MED_GAP_MIN:.2f} ({safe['gap']:.3f}) and "
            f"{safe['n_gap_ge']}/{safe['n_inst']} instances at gap ≥ {GAP_FLOOR:.2f}."
        )
    else:
        why = ""
        if edge["safe_fail"]:
            why = edge["safe_fail"][0]
        reason = (
            f"Operate at {_where(safe)}. That is the SAFE champ: frozen n=12 "
            f"L=16 Walsh R={safe['R']:.3f} (seed-median R={safe['seed_med_R']:.3f}) "
            f"is the lowest ratio whose median product gap is at least {MED_GAP_MIN:.2f} "
            f"({safe['gap']:.3f}) and whose instance fraction at gap ≥ {GAP_FLOOR:.2f} "
            f"is at least {FRAC_MIN:.0%} ({safe['n_gap_ge']}/{safe['n_inst']}). "
            f"The EDGE champ is {_where(edge)} at R={edge['R']:.3f} "
            f"(median gap {edge['gap']:.3f}, {edge['n_gap_ge']}/{edge['n_inst']} "
            f"instances ≥ {GAP_FLOOR:.2f})."
        )
        if why:
            reason += f" The EDGE champ misses the SAFE bar: {why}."
    return {
        "verdict": "PASS",
        "operating_lam": float(safe["lam"]),
        "operating_mu": float(safe["mu"]),
        "safe_lam": float(safe["lam"]),
        "safe_mu": float(safe["mu"]),
        "safe_R": float(safe["R"]),
        "edge_lam": float(edge["lam"]),
        "edge_mu": float(edge["mu"]),
        "edge_R": float(edge["R"]),
        "same_champ": same,
        "reason": reason,
        "safe_pool": [_brief(c) for c in safe_pool],
        "edge_pool": [_brief(c) for c in edge_pool],
        "packed": [_brief(c) for c in packed],
    }


def c_reverses_or_loses(c: dict) -> bool:
    return bool(c["reverses"] or float(c["R"]) > TIE_R)


def _edit(c: dict, reason: str) -> dict:
    return {
        "verdict": "needs_edit",
        "operating_lam": float(c["lam"]),
        "operating_mu": float(c["mu"]),
        "safe_lam": None,
        "safe_mu": None,
        "edge_lam": None,
        "edge_mu": None,
        "reason": reason,
    }


def _brief(c: dict) -> dict:
    return {
        "lam": float(c["lam"]),
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
        "safe_fail": list(c["safe_fail"]),
        "edge_fail": list(c["edge_fail"]),
        "bitstring": list(c["bitstring"]),
        "n_seed_beat": int(c["n_seed_beat"]),
        "R15": float(c["R15"]),
    }


__all__ = [
    "ANCHOR",
    "ANCHOR_TOL",
    "CELLS",
    "FRAC_MIN",
    "GAP_FLOOR",
    "LAM_GRID",
    "MED_GAP_MIN",
    "MU_GRID",
    "PUBLISHED_R",
    "RESUME_CELLS",
    "cell_key",
    "decide_lambda_mu_grid",
    "file_stem",
    "gap_stats",
    "is_resume",
]
