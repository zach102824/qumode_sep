"""Decision rule for the matched Λ≤0.80 bakeoff.

Three nomination cells from the (λ, μ) grid (PR #44) and the μ=0 seal (PR #45):

    A  λ=0.50  μ=0      prior n12 Λ=0.559  L=2 R≈0.75
    B  λ=0.50  μ=−0.15  prior n12 Λ=0.741  L=2 R≈0.635
    C  λ=0.75  μ=0      prior n12 Λ=0.748  L=2 R≈0.62

The fair baseline is Fourier QAOA when μ=0 and HVA when μ≠0. Rank is

    (1) median cavity Λ ≤ 0.80 at both n=12 and n=18
    (2) lowest n=12 L=2 R against that baseline
    (3) lowest n=12 Λ, then lowest n=18 Λ

A cell is PASS only when that Λ pair holds, the product gap stays ≥ 0.03
at both sizes, the median Z-peak stays below 1/2, and all 5 seeds have
n=12 L=2 R ≤ 0.85 (and the ratio of seed-median excesses does too).
The family is PASS when at least one cell does. Otherwise it is needs_edit.
NEW_SAFE is the first PASS cell in that rank order.
"""
from __future__ import annotations

import math
from typing import Iterable, Sequence

BAR_LAMBDA = 0.80
BAR_GAP = 0.03
BAR_PEAK = 0.5
BAR_R = 0.85
BAR_R_SOFT = 0.98
N_SEEDS = 5
# Nomination numbers. A/C ground-state figures are the five-seed μ=0 seal
# (PR #45); B is the three-seed grid (PR #44), which did not print four digits.
PRIORS = {
    "A": {"lambda": 0.50, "mu": 0.0, "lam12": 0.5585, "gap12": 0.1102, "peak12": 0.1569, "r": 0.751},
    "B": {"lambda": 0.50, "mu": -0.15, "lam12": 0.741, "gap12": 0.188, "peak12": 0.0497, "r": 0.635},
    "C": {"lambda": 0.75, "mu": 0.0, "lam12": 0.7477, "gap12": 0.1265, "peak12": 0.0379, "r": 0.621},
}

CANDIDATES = (
    {"id": "A", "lambda": 0.50, "mu": 0.0, "baseline": "fourier"},
    {"id": "B", "lambda": 0.50, "mu": -0.15, "baseline": "hva"},
    {"id": "C", "lambda": 0.75, "mu": 0.0, "baseline": "fourier"},
)


def assert_n_in_scope(n: int) -> None:
    if int(n) >= 21:
        raise RuntimeError("n=21 is out of scope for this bakeoff")


def baseline_for(mu: float) -> str:
    """Fourier when the Hamiltonian has no XX term, otherwise HVA."""
    return "fourier" if abs(float(mu)) < 1e-15 else "hva"


def ratio(num: float, den: float) -> float:
    if den == 0 or not math.isfinite(den) or not math.isfinite(num):
        return float("inf")
    return float(num) / float(den)


def official_r(snap_meds: Sequence[float], base_meds: Sequence[float]) -> float:
    """R = median_s(Med ε_SNAP) / median_s(Med ε_baseline)."""
    if len(snap_meds) == 0 or len(snap_meds) != len(base_meds):
        raise ValueError("snap and baseline seed medians must be the same length")
    snap = sorted(float(x) for x in snap_meds)
    base = sorted(float(x) for x in base_meds)
    mid = len(snap) // 2
    # Odd count (the five-seed protocol). Even counts use the upper median.
    return ratio(snap[mid], base[mid])


def seed_ratios(snap_meds: Sequence[float], base_meds: Sequence[float]) -> list[float]:
    return [ratio(float(s), float(b)) for s, b in zip(snap_meds, base_meds)]


def _le(value: float, bar: float) -> bool:
    return math.isfinite(value) and float(value) <= float(bar) + 1e-12


def _ge(value: float, bar: float) -> bool:
    return math.isfinite(value) and float(value) + 1e-12 >= float(bar)


def classify_candidate(
    *,
    lam12: float,
    lam18: float,
    gap12: float,
    gap18: float,
    peak12: float,
    peak18: float,
    r_official: float,
    r_seeds: Sequence[float],
) -> dict:
    """PASS / soft / FAIL for one cell. Soft means the quantumness bars hold and the lead does not."""
    fails: list[str] = []
    lam_pass = _le(lam12, BAR_LAMBDA) and _le(lam18, BAR_LAMBDA)
    if not _le(lam12, BAR_LAMBDA):
        fails.append(f"n12 median Λ {lam12:.3f} > {BAR_LAMBDA:.2f}")
    if not _le(lam18, BAR_LAMBDA):
        fails.append(f"n18 median Λ {lam18:.3f} > {BAR_LAMBDA:.2f}")
    if not _ge(gap12, BAR_GAP):
        fails.append(f"n12 product gap {gap12:.3f} < {BAR_GAP:.2f}")
    if not _ge(gap18, BAR_GAP):
        fails.append(f"n18 product gap {gap18:.3f} < {BAR_GAP:.2f}")
    if not (math.isfinite(peak12) and peak12 < BAR_PEAK):
        fails.append(f"n12 Z-peak {peak12:.3f} is classical (≥ {BAR_PEAK:.1f})")
    if not (math.isfinite(peak18) and peak18 < BAR_PEAK):
        fails.append(f"n18 Z-peak {peak18:.3f} is classical (≥ {BAR_PEAK:.1f})")
    seeds = [float(x) for x in r_seeds]
    n_clear = sum(1 for r in seeds if _le(r, BAR_R))
    if len(seeds) != N_SEEDS:
        fails.append(f"need {N_SEEDS} seeds, got {len(seeds)}")
    if not _le(r_official, BAR_R):
        fails.append(f"L=2 R {r_official:.3f} > {BAR_R:.2f}")
    if len(seeds) == N_SEEDS and n_clear < N_SEEDS:
        fails.append(f"L=2 R clears {n_clear}/{N_SEEDS} seeds at {BAR_R:.2f}")
    quantum_ok = not any(
        msg.startswith("n12 median")
        or msg.startswith("n18 median")
        or msg.startswith("n12 product")
        or msg.startswith("n18 product")
        or msg.startswith("n12 Z-peak")
        or msg.startswith("n18 Z-peak")
        for msg in fails
    )
    r_hard = (
        len(seeds) == N_SEEDS
        and n_clear == N_SEEDS
        and _le(r_official, BAR_R)
    )
    r_soft_band = len(seeds) == N_SEEDS and _le(r_official, BAR_R_SOFT)
    if quantum_ok and r_hard:
        call = "PASS"
    elif quantum_ok and r_soft_band:
        call = "soft"
    else:
        call = "FAIL"
    return {
        "call": call,
        "fails": fails,
        "lambda_pass": bool(lam_pass),
        "quantum_ok": bool(quantum_ok),
        "n_clear": int(n_clear),
        "r": float(r_official),
        "lam12": float(lam12),
        "lam18": float(lam18),
    }


def rank_key(cell: dict) -> tuple:
    """Lower is better. Λ-passing cells first, then low R, then low Λ."""
    r = float(cell["r"])
    return (
        0 if cell["lambda_pass"] else 1,
        r if math.isfinite(r) else float("inf"),
        float(cell["lam12"]),
        float(cell["lam18"]),
        str(cell["id"]),
    )


def rank_cells(cells: Sequence[dict]) -> list[dict]:
    return sorted(cells, key=rank_key)


def family_verdict(calls: Iterable[str]) -> str:
    calls = list(calls)
    if any(c == "PASS" for c in calls):
        return "PASS"
    return "needs_edit"


def recommend(ranked: Sequence[dict]) -> dict:
    """NEW_SAFE is the best PASS cell. A soft rank-1 cell is not locked."""
    passes = [c for c in ranked if c["call"] == "PASS"]
    top = ranked[0]
    if not passes:
        return {
            "lock": False,
            "id": None,
            "lambda": None,
            "mu": None,
            "baseline": None,
            "nearest_id": top["id"],
            "nearest_call": top["call"],
        }
    win = passes[0]
    return {
        "lock": True,
        "id": win["id"],
        "lambda": float(win["lambda"]),
        "mu": float(win["mu"]),
        "baseline": win["baseline"],
        "nearest_id": top["id"],
        "nearest_call": top["call"],
    }


def decision_self_test() -> None:
    """The rank and the 5/5 bar, including the nomination order if n=18 Λ holds."""
    for spec in CANDIDATES:
        if baseline_for(spec["mu"]) != spec["baseline"]:
            raise AssertionError(spec)
    assert_n_in_scope(18)
    try:
        assert_n_in_scope(21)
    except RuntimeError:
        pass
    else:
        raise AssertionError("n=21 must be refused")

    def cell(cid: str, lam18: float, r: float, seeds: list[float], gap18: float = 0.08, peak18: float = 0.05) -> dict:
        spec = next(c for c in CANDIDATES if c["id"] == cid)
        prior = PRIORS[cid]
        decision = classify_candidate(
            lam12=prior["lam12"],
            lam18=lam18,
            gap12=prior["gap12"],
            gap18=gap18,
            peak12=prior["peak12"],
            peak18=peak18,
            r_official=r,
            r_seeds=seeds,
        )
        decision.update(
            {
                "id": cid,
                "lambda": spec["lambda"],
                "mu": spec["mu"],
                "baseline": spec["baseline"],
            }
        )
        return decision

    hard = [0.70, 0.71, 0.72, 0.73, 0.74]
    all_pass = rank_cells(
        [
            cell("A", 0.56, 0.751, hard),
            cell("B", 0.74, 0.635, [0.63, 0.64, 0.63, 0.64, 0.63]),
            cell("C", 0.75, 0.621, [0.62, 0.62, 0.62, 0.62, 0.62]),
        ]
    )
    if [c["id"] for c in all_pass] != ["C", "B", "A"]:
        raise AssertionError([c["id"] for c in all_pass])
    if family_verdict(c["call"] for c in all_pass) != "PASS":
        raise AssertionError(all_pass)
    locked = recommend(all_pass)
    if not locked["lock"] or locked["id"] != "C" or abs(locked["mu"]) > 1e-15:
        raise AssertionError(locked)

    # Same R: lower n12 Λ wins. A and a clone are not both in the alphabet, so
    # compare A (Λ 0.559) with C forced to the same R and a passing n18 Λ.
    tie = rank_cells(
        [
            cell("A", 0.56, 0.70, [0.70] * 5),
            cell("C", 0.70, 0.70, [0.70] * 5),
        ]
    )
    if [c["id"] for c in tie] != ["A", "C"]:
        raise AssertionError([c["id"] for c in tie])

    # Λ miss outranks nothing that passed, even with a tiny R.
    mixed = rank_cells(
        [
            cell("A", 0.90, 0.30, [0.30] * 5),
            cell("B", 0.74, 0.635, [0.63] * 5),
            cell("C", 0.81, 0.40, [0.40] * 5),
        ]
    )
    if mixed[0]["id"] != "B" or mixed[0]["call"] != "PASS":
        raise AssertionError([(c["id"], c["call"]) for c in mixed])
    if mixed[1]["lambda_pass"] or mixed[2]["lambda_pass"]:
        raise AssertionError(mixed)
    if recommend(mixed)["id"] != "B":
        raise AssertionError(recommend(mixed))

    soft = cell("C", 0.75, 0.90, [0.90, 0.91, 0.88, 0.92, 0.89])
    if soft["call"] != "soft":
        raise AssertionError(soft)
    four = cell("C", 0.75, 0.80, [0.80, 0.81, 0.82, 0.83, 0.90])
    if four["call"] != "soft" or four["n_clear"] != 4:
        raise AssertionError(four)
    edge = cell("A", 0.80, 0.85, [0.85] * 5)
    if edge["call"] != "PASS" or edge["fails"]:
        raise AssertionError(edge)
    gap = cell("A", 0.56, 0.50, [0.50] * 5, gap18=0.029)
    if gap["call"] != "FAIL" or gap["lambda_pass"] is not True:
        raise AssertionError(gap)
    classical = cell("A", 0.56, 0.50, [0.50] * 5, peak18=0.5)
    if classical["call"] != "FAIL":
        raise AssertionError(classical)
    all_soft = [cell(cid, PRIORS[cid]["lam12"], 0.90, [0.90] * 5) for cid in ("A", "B", "C")]
    if family_verdict(c["call"] for c in all_soft) != "needs_edit":
        raise AssertionError(all_soft)
    if recommend(rank_cells(all_soft))["lock"]:
        raise AssertionError("soft cells must not lock")
    # Official R is the ratio of medians, not the median of ratios.
    r = official_r([1.0, 3.0, 5.0], [2.0, 4.0, 10.0])
    if abs(r - 3.0 / 4.0) > 1e-12:
        raise AssertionError(r)


def _fmt(x, nd: int = 3) -> str:
    if x is None or not isinstance(x, (int, float)) or not math.isfinite(float(x)):
        return "—"
    return f"{float(x):.{nd}f}"


def _fmt_mu(mu: float) -> str:
    return f"{float(mu):.2f}"


def render(summary: dict) -> str:
    """Markdown report. `summary` is the aggregate written next to RESULTS.md."""
    verdict = summary["verdict"]
    rec = summary["recommendation"]
    if rec["lock"]:
        headline = (
            f"**Verdict: PASS. NEW_SAFE paper-lock is {rec['id']}, "
            f"λ={rec['lambda']:.2f}, μ={_fmt_mu(rec['mu'])} "
            f"(baseline {rec['baseline']}).**"
        )
    else:
        headline = "**Verdict: needs_edit. No NEW_SAFE paper-lock.**"
    lines = [
        "# Matched bakeoff — three Λ≤0.80 candidates",
        "",
        headline,
        "",
        "Round 12 trains the three cells that cleared Λ≤0.80 on the n=12 grid "
        "(PR #44 / PR #45) under one protocol. Walsh Hamming-shell SNAP is compared "
        "with the fair baseline: Fourier QAOA when μ=0, HVA when μ≠0. "
        "The killed SAFE point (λ=0.75, μ=−0.50, n=12 Λ≈0.941) is not retrained.",
        "",
        "## Hamiltonian",
        "",
        r"\[",
        r"H = \mathrm{diag}(\mathrm{unsat}) - \lambda \sum_i X_i + \mu \sum_c \sum_{i<j\in c} X_i X_j",
        r"\]",
        "",
        "Cavity c owns qubits [3c, 3c+3), qubit 0 = MSB. "
        "Λ is the median of ⟨(Σ_{i∈c} X_i / 3)^2⟩ over cavities and instances of the true ground state. "
        "The product gap is (E_prod − E_0) / |E_0| for the best real XZ product state found by the SAFE start set. "
        "Z-peak is the median over instances of max_x |⟨x|ψ_0⟩|^2. "
        "R = Medε_SNAP / Medε_baseline at n=12, L=2. "
        "Medε is the median over seeds of each seed's median-over-instances excess ⟨H⟩−E_0.",
        "",
        "## Gate",
        "",
        f"- **Λ pass** when median Λ ≤ {BAR_LAMBDA:.2f} at n=12 (B={summary['freeze_b']}) "
        f"and at n=18 (B={summary['n18_b']}).",
        f"- **PASS** a cell when it also has product gap ≥ {BAR_GAP:.2f} and Z-peak < {BAR_PEAK:.1f} "
        f"at both sizes, L=2 R ≤ {BAR_R:.2f}, and all {N_SEEDS} seeds have per-seed R ≤ {BAR_R:.2f}.",
        f"- **soft** when those quantumness bars hold and L=2 R ≤ {BAR_R_SOFT:.2f}, but the 5/5 bar does not.",
        "- **FAIL** a cell when a quantumness bar misses, or the L=2 lead is above the soft band.",
        "- **Family PASS** when at least one cell is PASS. Otherwise **needs_edit**.",
        "- **NEW_SAFE** is the first PASS cell in the rank. Rank is Λ-pass first, then lowest L=2 R, "
        "then lowest n=12 Λ, then lowest n=18 Λ. A soft cell is not locked.",
        "- L=4 is the same budget at one extra depth. It is reported and is not the lock.",
        "",
        "## Protocol",
        "",
        "Packs: `beat_qaoa_12h/train_packs`. Train indices [0:200) on n=6 and n=9. "
        "Checkpoint on the n=9 holdout [200:250). Freeze n=12, the first "
        f"{summary['freeze_b']} instances. "
        f"The n=18 probe is the first {summary['n18_b']} satisfiable formulas from the same generator "
        f"and base ({summary['n18_base']}), seed = base + 10_000_000·n + k. "
        "It is stored in this run directory. `train_packs/` was not rewritten.",
        "",
        "Seeds {0,1,2,3,4}. L∈{2,4}. "
        "One batch stream per seed: NumPy Generator(seed) and torch.manual_seed(seed), "
        "shared by every cell, both methods, and both depths. "
        "L-BFGS matches `Nov_3rd.settings`: "
        f"{summary['rounds']} rounds × max_iter {summary['lbfgs_iter']}, "
        f"lr {summary['lbfgs_lr']}, history 16, strong Wolfe, grad clip 5, "
        f"batch {summary['batch']} (half n=6 / half n=9). New optimizer each round. "
        "Loss = mean ε/n. Each seed restores the round with the best n=9-holdout Med ε. "
        "Paper init β=β^X=0.01, β^XX=0, γ=γ_w=−0.01, φ_k=−β k.",
        "",
        "Ground states: dense eigh at n=6 and n=9, ARPACK at n=12 and n=18. "
        "n=21 was not loaded. `qaoa_angles/`, `train69_ladder_fresh250/`, and `settings.py` were not written.",
        "",
        "## Rank",
        "",
        "| rank | id | λ | μ | baseline | Λ12 | Λ18 | gap12 | gap18 | Z-peak12 | Z-peak18 | L=2 R | 5/5 | call |",
        "|--:|:--|---:|---:|:--|---:|---:|---:|---:|---:|---:|---:|:--|---|",
    ]
    for i, cell in enumerate(summary["ranked"], start=1):
        g12 = cell["n12"]
        g18 = cell["n18"]
        lines.append(
            "| {rank} | {id} | {lam:.2f} | {mu:.2f} | {base} | {l12} | {l18} | {g12} | {g18} | {p12} | {p18} | {r} | {nclear}/5 | {call} |".format(
                rank=i,
                id=cell["id"],
                lam=cell["lambda"],
                mu=cell["mu"],
                base=cell["baseline"],
                l12=_fmt(g12["lambda_med"]),
                l18=_fmt(g18["lambda_med"]),
                g12=_fmt(g12["gap_med"]),
                g18=_fmt(g18["gap_med"]),
                p12=_fmt(g12["peak_med"], 4),
                p18=_fmt(g18["peak_med"], 4),
                r=_fmt(cell["L2"]["by_n"]["12"]["r"]),
                nclear=cell["decision"]["n_clear"],
                call=cell["decision"]["call"],
            )
        )
    lines += [
        "",
        "## Ground state",
        "",
        "| id | n | B | median Λ | min Λ | max Λ | frac Λ>0.80 | frac Λ>0.85 | median gap | min gap | median Z-peak | frac peak≥0.5 |",
        "|:--|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for cell in summary["ranked"]:
        for n, block in (("12", cell["n12"]), ("18", cell["n18"])):
            lines.append(
                "| {id} | {n} | {B} | {med} | {mn} | {mx} | {f80} | {f85} | {gap} | {gmin} | {peak} | {fp} |".format(
                    id=cell["id"],
                    n=n,
                    B=block["B"],
                    med=_fmt(block["lambda_med"], 4),
                    mn=_fmt(block["lambda_min"], 4),
                    mx=_fmt(block["lambda_max"], 4),
                    f80=_fmt(block["frac_gt_080"], 3),
                    f85=_fmt(block["frac_gt_085"], 3),
                    gap=_fmt(block["gap_med"], 4),
                    gmin=_fmt(block["gap_min"], 4),
                    peak=_fmt(block["peak_med"], 4),
                    fp=_fmt(block["frac_peak_ge_05"], 3),
                )
            )
    lines += [
        "",
        "n=12 Λ on this pack, against the nomination value:",
        "",
        "| id | Λ12 now | Λ12 prior | gap12 now | gap12 prior | Z-peak12 now | Z-peak12 prior |",
        "|:--|---:|---:|---:|---:|---:|---:|",
    ]
    for cell in summary["ranked"]:
        prior = PRIORS[cell["id"]]
        lines.append(
            "| {id} | {now} | {old} | {gnow} | {gold} | {pnow} | {pold} |".format(
                id=cell["id"],
                now=_fmt(cell["n12"]["lambda_med"], 4),
                old=_fmt(prior["lam12"], 4),
                gnow=_fmt(cell["n12"]["gap_med"], 4),
                gold=_fmt(prior["gap12"], 4),
                pnow=_fmt(cell["n12"]["peak_med"], 4),
                pold=_fmt(prior["peak12"], 4),
            )
        )
    lines += [
        "",
        "## Ratios",
        "",
        "Holdout at n=6 and n=9 is the checkpoint distribution. The decision row is n=12, L=2.",
        "",
        "| id | L | n | Medε baseline | Medε SNAP | R | paired SNAP<baseline | seeds ≤0.85 |",
        "|:--|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for cell in summary["ranked"]:
        for depth in ("L2", "L4"):
            if depth not in cell:
                continue
            block = cell[depth]
            L = 2 if depth == "L2" else 4
            for n in ("6", "9", "12"):
                row = block["by_n"][n]
                nclear = ""
                if n == "12":
                    nclear = f"{block['n_clear']}/5"
                lines.append(
                    "| {id} | {L} | {n} | {b} | {s} | {r} | {pw}/{pn} | {nc} |".format(
                        id=cell["id"],
                        L=L,
                        n=n,
                        b=_fmt(row["med_base"], 4),
                        s=_fmt(row["med_snap"], 4),
                        r=_fmt(row["r"], 4),
                        pw=row["paired_win"],
                        pn=row["paired_n"],
                        nc=nclear,
                    )
                )
    lines += [
        "",
        "### Seeds at n=12, L=2",
        "",
        "| id | seed | SNAP | baseline | R | best round | wall s |",
        "|:--|---:|---:|---:|---:|---:|---:|",
    ]
    for cell in summary["ranked"]:
        for row in cell["L2"]["per_seed"]:
            lines.append(
                "| {id} | {seed} | {s} | {b} | {r} | {rnd} | {wall} |".format(
                    id=cell["id"],
                    seed=row["seed"],
                    s=_fmt(row["snap"], 4),
                    b=_fmt(row["base"], 4),
                    r=_fmt(row["r"], 4),
                    rnd=row["best_round"],
                    wall=_fmt(row["wall_s"], 1),
                )
            )
    if any("L4" in c for c in summary["ranked"]):
        lines += [
            "",
            "### Seeds at n=12, L=4",
            "",
            "| id | seed | SNAP | baseline | R | best round | wall s |",
            "|:--|---:|---:|---:|---:|---:|---:|",
        ]
        for cell in summary["ranked"]:
            for row in cell["L4"]["per_seed"]:
                lines.append(
                    "| {id} | {seed} | {s} | {b} | {r} | {rnd} | {wall} |".format(
                        id=cell["id"],
                        seed=row["seed"],
                        s=_fmt(row["snap"], 4),
                        b=_fmt(row["base"], 4),
                        r=_fmt(row["r"], 4),
                        rnd=row["best_round"],
                        wall=_fmt(row["wall_s"], 1),
                    )
                )
    lines += ["", "## Recommendation", ""]
    lines.extend(_recommendation_lines(summary))
    lines += ["", "## Reading", ""]
    lines.extend(_reading(summary))
    lines += [
        "",
        "Per-seed JSON is `runs/L{L}_{method}_s{seed}_{id}.json`. "
        "Ground-state arrays are `cache/diag_{id}_n{n}.npz`. "
        "The n=18 formulas are `probe_n18.npz`.",
        "",
    ]
    return "\n".join(lines)


def _recommendation_lines(summary: dict) -> list[str]:
    rec = summary["recommendation"]
    ranked = summary["ranked"]
    if rec["lock"]:
        win = next(c for c in ranked if c["id"] == rec["id"])
        return [
            f"Paper-lock **{win['id']}** as NEW_SAFE: λ={win['lambda']:.2f}, μ={_fmt_mu(win['mu'])}, "
            f"baseline {win['baseline']}.",
            "",
            f"n=12 Λ={_fmt(win['n12']['lambda_med'], 3)}, n=18 Λ={_fmt(win['n18']['lambda_med'], 3)}, "
            f"n=12 gap={_fmt(win['n12']['gap_med'], 3)}, n=18 gap={_fmt(win['n18']['gap_med'], 3)}, "
            f"n=12 Z-peak={_fmt(win['n12']['peak_med'], 4)}, n=18 Z-peak={_fmt(win['n18']['peak_med'], 4)}, "
            f"L=2 R={_fmt(win['L2']['by_n']['12']['r'], 3)} on {win['decision']['n_clear']}/5 seeds.",
            "",
            "It is the Λ-passing cell with the lowest L=2 R among cells that also clear the 5/5 bar. "
            + (
                "It is rank 1."
                if ranked[0]["id"] == win["id"]
                else f"Rank 1 is {ranked[0]['id']} ({ranked[0]['decision']['call']}); the lock is the best PASS cell."
            ),
            "",
            _lock_context(summary, win),
        ]
    top = ranked[0]
    why = "; ".join(top["decision"]["fails"]) if top["decision"]["fails"] else top["decision"]["call"]
    return [
        "Do not paper-lock a NEW_SAFE point from this shortlist.",
        "",
        f"The leading ranked cell is {top['id']} (λ={top['lambda']:.2f}, μ={_fmt_mu(top['mu'])}, "
        f"call {top['decision']['call']}): {why}.",
        "",
        "Family call is needs_edit: no cell clears Λ≤0.80 at n=12 and n=18 together with the L=2 R bar on 5/5 seeds.",
    ]


def _lock_context(summary: dict, win: dict) -> str:
    """Why this cell, and what L=4 did not decide."""
    bits = []
    bits.append(
        f"{win['id']} clears the median bar with "
        f"{_fmt(100.0 * win['n12']['frac_gt_080'], 1)}% of n=12 cavity-instance pairs "
        f"and {_fmt(100.0 * win['n18']['frac_gt_080'], 1)}% at n=18 still above 0.80."
    )
    lowest = min(summary["ranked"], key=lambda c: (c["n12"]["lambda_med"], c["id"]))
    if lowest["id"] != win["id"]:
        bits.append(
            f"{lowest['id']} has the lowest cavity Λ "
            f"(n12 {_fmt(lowest['n12']['lambda_med'], 3)}, n18 {_fmt(lowest['n18']['lambda_med'], 3)}) "
            f"and a higher L=2 R ({_fmt(lowest['L2']['by_n']['12']['r'], 3)}). "
            "Among cells that clear Λ≤0.80, the rank uses the lowest R."
        )
    l4 = []
    for cell in summary["ranked"]:
        if "L4" not in cell:
            continue
        l4.append(
            f"{cell['id']} {_fmt(cell['L4']['by_n']['12']['r'], 3)} on {cell['L4']['n_clear']}/5"
        )
    if l4:
        bits.append(
            "L=4 stays in the soft band (" + ", ".join(l4) + "). The paper-lock uses L=2."
        )
    n6 = []
    for cell in summary["ranked"]:
        block = cell.get("L4", {}).get("by_n", {}).get("6")
        if block is not None and float(block["r"]) > 1.0:
            n6.append(f"{cell['id']} (R={_fmt(block['r'], 3)})")
    if n6:
        bits.append("On the n=6 holdout at L=4, SNAP is above the baseline for " + "; ".join(n6) + ".")
    return " ".join(bits)


def _reading(summary: dict) -> list[str]:
    out = []
    for cell in summary["ranked"]:
        d = cell["decision"]
        extra = ""
        if d["fails"]:
            extra = " Misses: " + "; ".join(d["fails"]) + "."
        beta = cell["L2"].get("mean_abs_beta_xx")
        beta_s = ""
        if beta is not None:
            beta_s = f" HVA mean |β^XX|={_fmt(beta, 3)}."
        l4 = ""
        if "L4" in cell:
            l4 = (
                f", L=4 R={_fmt(cell['L4']['by_n']['12']['r'], 3)} "
                f"({cell['L4']['n_clear']}/5)"
            )
        out.append(
            f"- **{cell['id']}** (λ={cell['lambda']:.2f}, μ={_fmt_mu(cell['mu'])}, {cell['baseline']}): "
            f"{d['call']}. Λ12={_fmt(cell['n12']['lambda_med'], 3)}, "
            f"Λ18={_fmt(cell['n18']['lambda_med'], 3)}, "
            f"gap12={_fmt(cell['n12']['gap_med'], 3)}, gap18={_fmt(cell['n18']['gap_med'], 3)}, "
            f"Z-peak12={_fmt(cell['n12']['peak_med'], 4)}, Z-peak18={_fmt(cell['n18']['peak_med'], 4)}, "
            f"L=2 R={_fmt(cell['L2']['by_n']['12']['r'], 3)} "
            f"({d['n_clear']}/5){l4}.{beta_s}{extra}"
        )
    drift = 0.0
    for cell in summary["ranked"]:
        prior = PRIORS[cell["id"]]
        drift = max(
            drift,
            abs(float(cell["n12"]["lambda_med"]) - float(prior["lam12"])),
            abs(float(cell["n12"]["gap_med"]) - float(prior["gap12"])),
            abs(float(cell["n12"]["peak_med"]) - float(prior["peak12"])),
        )
    out.append("")
    if drift < 0.002:
        out.append(
            "n=12 Λ, the product gap, and the Z-peak reproduce the nomination "
            "(A and C from the μ=0 seal, B from the three-seed grid) to 0.001."
        )
        out.append("")
    out.append(f"The family call is **{summary['verdict']}**.")
    return out
