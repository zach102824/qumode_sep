"""Verdict for the unsat Walsh sweep below μ=−0.50.

    H = diag(unsat) − 0.75 Σ_i X_i + μ Σ_cav Σ_{i<j} X_i X_j
    μ ∈ {−1.00, −0.85, −0.75, −0.65, −0.50}

μ=−0.50 is the PR #26 operating point (frozen n=12 L=16 Walsh R = 0.150).
A more negative μ replaces it only when that ratio drops by at least 0.02
on both the ratio of medians and the median per-seed ratio, and the ground
state stays quantum (peak < 0.5, n=12 product gap ≥ 0.02). If μ=−1.00 wins,
the grid is still on its negative edge. n=21 is out of scope.
"""
from __future__ import annotations

from Nov_3rd.unsat_mu_sweep import cache_tag, decide_mu_sweep, file_tag, mu_label

# Decimal grid. −0.50 is the PR #26 champ, not a new train.
MU_GRID = (-1.00, -0.85, -0.75, -0.65, -0.50)
ANCHOR_MU = -0.50
# Exact ratio of medians from the PR #26 L=16 Walsh checkpoints, n=12.
PUBLISHED_ANCHOR_R = 0.15015595428732964
EDGE_MU = -1.00
ANCHOR_TEXT = "μ=−0.50"
CHECKPOINT_TEXT = "the PR #26 checkpoint"


def decide_mu_neg(by_mu: dict) -> dict:
    """PASS / needs_edit / KILL for μ ∈ {−1.00, −0.85, −0.75, −0.65, −0.50}.

    PASS when a quantum μ is clearly below the in-run μ=−0.50 n=12 Walsh
    ratio, or when none is and μ=−0.50 remains optimal. A challenger whose
    ground state is a bitstring or whose n=12 product gap is under 0.02 is a
    quantumness break: it does not replace μ=−0.50.
    """
    out = decide_mu_sweep(
        by_mu,
        grid=MU_GRID,
        anchor_mu=ANCHOR_MU,
        published_R=PUBLISHED_ANCHOR_R,
        anchor_text=ANCHOR_TEXT,
        checkpoint_text=CHECKPOINT_TEXT,
    )
    qbreak = [d for d in out.get("disqualified") or [] if d.get("q_fail")]
    if qbreak:
        bits = []
        for d in sorted(qbreak, key=lambda x: float(x["R"])):
            bits.append(f"μ={float(d['mu']):+.2f} R={float(d['R']):.3f} ({d['q_fail'][0]})")
        out["quantumness_break"] = True
        out["reason"] = out["reason"].rstrip() + " Quantumness break: " + "; ".join(bits) + "."
    op = out.get("operating_mu")
    if (
        out.get("verdict") == "PASS"
        and not out.get("still_optimal")
        and op is not None
        and abs(float(op) - EDGE_MU) < 1e-9
    ):
        out["further_edge"] = True
        out["reason"] = (
            out["reason"].rstrip()
            + " μ=−1.00 is the negative edge of this grid (further edge). "
            "A μ below −1.00 was not trained."
        )
    return out


__all__ = [
    "ANCHOR_MU",
    "ANCHOR_TEXT",
    "CHECKPOINT_TEXT",
    "EDGE_MU",
    "MU_GRID",
    "PUBLISHED_ANCHOR_R",
    "cache_tag",
    "decide_mu_neg",
    "file_tag",
    "mu_label",
]
