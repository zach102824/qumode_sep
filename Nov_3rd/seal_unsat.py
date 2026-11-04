"""Verdict rules for the unsat λ=0.75 Walsh / power_p seal.

Pure functions. The training script fills the cells; this module decides
PASS / needs_edit / KILL and which diagonal wins the transfer story.

Comparison target is the published ZZZ champ (PR #13 smoke, PR #15 transfer),
not a retrain. Those runs used B_train=32. This seal uses B_train≥128.
Ratios are comparable; absolute residuals are not, because H differs.
"""
from __future__ import annotations

import math

BEAT_R = 0.98
TIE_R = 1.02
# n=15 may soften. A flip is R at or above this after an n=12 win.
FLIP_R = 1.10
# n=6 beat, n=9 loss, n=12 this high: the advantage reversed.
REVERSE_R = 1.05
# Cost-only (φ=0) at or below this multiple of Walsh means the diagonal
# pulse already matches the full motif, so GRAPE of H_c found the state.
GRAPE_MATCH = 1.05

# PR #15 transfer_A, L-BFGS, seeds 0/1/2, B_train=32, B_hold=48,
# B_n12=32, B_n15=16. Head is power_p. Walsh is a null knob on this diagonal.
# R is the ratio of median-over-seeds of median ε.
ZZZ_CHAMP = {
    (2, 6): {"R": 1.000, "mark": "TIE", "med": 0.0435, "qaoa": 0.0435, "role": "hold"},
    (2, 9): {"R": 0.999, "mark": "TIE", "med": 0.0971, "qaoa": 0.0971, "role": "hold"},
    (2, 12): {"R": 1.000, "mark": "TIE", "med": 0.1425, "qaoa": 0.1425, "role": "transfer"},
    (2, 15): {"R": 1.000, "mark": "TIE", "med": 0.1538, "qaoa": 0.1538, "role": "transfer"},
    (16, 6): {"R": 0.654, "mark": "BEAT", "med": 0.0026, "qaoa": 0.0040, "role": "hold"},
    (16, 9): {"R": 0.853, "mark": "BEAT", "med": 0.0149, "qaoa": 0.0174, "role": "hold"},
    (16, 12): {"R": 0.924, "mark": "BEAT", "med": 0.0274, "qaoa": 0.0297, "role": "transfer"},
    (16, 15): {"R": 0.905, "mark": "BEAT", "med": 0.0393, "qaoa": 0.0435, "role": "transfer"},
}

# Same PR #15 run, unsat Walsh at B_train=32. Reference only. Not the seal.
UNSAT_WALSH_B32 = {
    (2, 6): 0.735,
    (2, 9): 0.756,
    (2, 12): 0.831,
    (2, 15): 0.768,
    (16, 6): 0.807,
    (16, 9): 0.794,
    (16, 12): 0.886,
    (16, 15): 0.848,
}

TRANSFER_NS = (12, 15)
DEPTHS = (2, 16)
SELECT_NS = (6, 9)


def mark_of(R: float) -> str:
    if not math.isfinite(R):
        return "UNDEF"
    if R <= BEAT_R:
        return "BEAT"
    if R <= TIE_R:
        return "TIE"
    return "UNDER"


def _cell(cells: dict, method: str, L: int, n: int) -> dict:
    key = (method, int(L), int(n))
    if key not in cells:
        raise KeyError(f"missing cell {key}")
    return cells[key]


def transfer_beats(table: dict, ns=TRANSFER_NS, Ls=DEPTHS) -> int:
    """How many frozen (L, n) cells are BEAT. ``table`` is keyed by (L, n)."""
    n_beat = 0
    for L in Ls:
        for n in ns:
            if table[(int(L), int(n))]["mark"] == "BEAT":
                n_beat += 1
    return n_beat


def pick_hc_winner(cells: dict) -> dict:
    """Which diagonal wins the freeze-to-n=12/15 story.

    Unsat's candidates are the Walsh head and the scalar power_p head from
    this seal. ZZZ's candidate is the published power_p champ (Walsh is
    identically the scalar pulse on a pure degree-3 diagonal).

    Eligible if L=16 n=12 is BEAT. Among eligible heads, more BEAT cells on
    {12,15}×{2,16} wins, then a lower n=12 L=16 R. A gap under 0.01 with
    the same beat count is a tie.
    """
    heads = []
    for method, label, source in (
        ("walsh", "unsat", "this seal"),
        ("power_p", "unsat", "this seal"),
    ):
        table = {(L, n): _cell(cells, method, L, n) for L in DEPTHS for n in (6, 9, 12, 15)}
        r12 = float(table[(16, 12)]["R"])
        heads.append({
            "hc": "unsat",
            "head": method,
            "source": source,
            "eligible": table[(16, 12)]["mark"] == "BEAT",
            "transfer_beats": transfer_beats(table),
            "R12": r12,
            "mark12": table[(16, 12)]["mark"],
        })
    ztable = ZZZ_CHAMP
    heads.append({
        "hc": "zzz",
        "head": "power_p",
        "source": "PR #15 published, B_train=32",
        "eligible": ztable[(16, 12)]["mark"] == "BEAT",
        "transfer_beats": transfer_beats(ztable),
        "R12": float(ztable[(16, 12)]["R"]),
        "mark12": ztable[(16, 12)]["mark"],
    })
    eligible = [h for h in heads if h["eligible"]]
    if not eligible:
        return {"winner": "none", "head": None, "reason": "no head beats Fourier QAOA at n=12 L=16", "heads": heads}

    def sort_key(h):
        return (-int(h["transfer_beats"]), float(h["R12"]))

    ordered = sorted(eligible, key=sort_key)
    best = ordered[0]
    reason = (
        f"{best['hc']} via {best['head']}: {best['transfer_beats']}/4 transfer BEATs, "
        f"n=12 L=16 R={best['R12']:.3f}"
    )
    if len(ordered) > 1:
        nxt = ordered[1]
        if (
            int(best["transfer_beats"]) == int(nxt["transfer_beats"])
            and abs(float(best["R12"]) - float(nxt["R12"])) < 0.01
        ):
            return {
                "winner": "tie",
                "head": None,
                "reason": (
                    f"{best['hc']}/{best['head']} and {nxt['hc']}/{nxt['head']} "
                    f"tie on transfer BEATs ({best['transfer_beats']}) and n=12 L=16 R "
                    f"({best['R12']:.3f} vs {nxt['R12']:.3f})"
                ),
                "heads": heads,
            }
    return {"winner": best["hc"], "head": best["head"], "reason": reason, "heads": heads}


def _grape_matches(grape: dict, cells: dict) -> bool:
    """True when the optimized φ=0 pulse is as good as Walsh at n=12 L=16."""
    if not grape or "cost_only_med" not in grape:
        return False
    g = float(grape["cost_only_med"][16][12])
    w = float(_cell(cells, "walsh", 16, 12)["med"])
    if w <= 0:
        return g <= w
    return g <= GRAPE_MATCH * w


def decide(cells: dict, quantum: dict, grape: dict) -> dict:
    """PASS / needs_edit / KILL for the unsat Walsh seal.

    PASS: Walsh beats Fourier QAOA at L=2 and L=16 on the {6,9} holdout and
    on frozen n=12; n=15 stays below the flip line at both depths; at least
    two of three n=12 L=16 seeds beat and the median of those seed ratios
    agrees with the ratio of medians; the n=12 ground state is not a
    bitstring; the cost-only pulse does not match Walsh.

    power_p is reported and does not gate PASS. It is the scalar control
    PR #11 already found short at L=2.

    KILL: a gated ground state is a bitstring; neither head beats on n=6 and
    n=9; the L=16 advantage reverses from n=6 to n=12; or GRAPE of the
    diagonal already matches the full motif.
    """
    for n in (6, 9, 12, 15):
        if n not in quantum:
            return {"verdict": "needs_edit", "reason": f"quantumness missing at n={n}", "winner": pick_hc_winner(cells)}
        peak = float(quantum[n]["peak_med"])
        if n in (6, 9, 12) and peak >= 0.5:
            return {
                "verdict": "KILL",
                "reason": f"n={n} ground state is a bitstring (peak med {peak:.3f} ≥ 0.5).",
                "winner": {"winner": "none", "head": None, "reason": "classical ground state", "heads": []},
            }

    if _grape_matches(grape, cells):
        g = float(grape["cost_only_med"][16][12])
        w = float(_cell(cells, "walsh", 16, 12)["med"])
        return {
            "verdict": "KILL",
            "reason": (
                f"cost-only Med ε at n=12 L=16 is {g:.4f}, within {GRAPE_MATCH:.2f}× of Walsh {w:.4f}. "
                "GRAPE of the diagonal H_c already matches the full gate."
            ),
            "winner": {"winner": "none", "head": None, "reason": "diagonal pulse is enough", "heads": []},
        }

    def beats_both(method: str, L: int, ns) -> bool:
        return all(_cell(cells, method, L, n)["mark"] == "BEAT" for n in ns)

    walsh_sel = {L: beats_both("walsh", L, SELECT_NS) for L in DEPTHS}
    power_sel = {L: beats_both("power_p", L, SELECT_NS) for L in DEPTHS}
    any_both = any(walsh_sel.values()) or any(power_sel.values())
    if not any_both:
        return {
            "verdict": "KILL",
            "reason": (
                "Neither Walsh nor power_p reaches Med ε ≤ 0.98× Fourier QAOA on both "
                "n=6 and n=9 at either smoked depth. SNAP does not beat QAOA at this B."
            ),
            "winner": pick_hc_winner(cells),
        }

    r6 = float(_cell(cells, "walsh", 16, 6)["R"])
    r9 = float(_cell(cells, "walsh", 16, 9)["R"])
    r12 = float(_cell(cells, "walsh", 16, 12)["R"])
    if r6 <= BEAT_R and r9 > TIE_R and r12 >= REVERSE_R:
        return {
            "verdict": "KILL",
            "reason": (
                f"L=16 Walsh advantage reverses: n=6 R={r6:.3f}, n=9 R={r9:.3f}, "
                f"n=12 R={r12:.3f} ≥ {REVERSE_R}."
            ),
            "winner": pick_hc_winner(cells),
        }

    walsh_n12 = {L: _cell(cells, "walsh", L, 12)["mark"] == "BEAT" for L in DEPTHS}
    n15_hold = all(float(_cell(cells, "walsh", L, 15)["R"]) < FLIP_R for L in DEPTHS)
    seed_Rs = [float(x) for x in _cell(cells, "walsh", 16, 12)["seed_Rs"]]
    if len(seed_Rs) < 3:
        return {
            "verdict": "needs_edit",
            "reason": f"n=12 L=16 Walsh has {len(seed_Rs)} seeds; the seal needs at least 3.",
            "winner": pick_hc_winner(cells),
        }
    seed_med = float(sorted(seed_Rs)[len(seed_Rs) // 2]) if len(seed_Rs) % 2 == 1 else (
        0.5 * (sorted(seed_Rs)[len(seed_Rs) // 2 - 1] + sorted(seed_Rs)[len(seed_Rs) // 2])
    )
    n_seed_beat = sum(r <= BEAT_R for r in seed_Rs)
    seeds_agree = seed_med <= BEAT_R and n_seed_beat >= 2 and r12 <= BEAT_R

    winner = pick_hc_winner(cells)
    n15_quantum = float(quantum[15]["peak_med"]) < 0.5
    both_depths = all(walsh_sel[L] and walsh_n12[L] for L in DEPTHS)
    if both_depths and n15_hold and seeds_agree and n15_quantum:
        return {
            "verdict": "PASS",
            "reason": (
                f"Walsh beats Fourier QAOA at L=2 and L=16 on n=6, n=9, and frozen n=12 "
                f"(n=12 L=16 R={r12:.3f}). n=15 stays under {FLIP_R} "
                f"(L=16 R={float(_cell(cells, 'walsh', 16, 15)['R']):.3f}). "
                f"{n_seed_beat}/{len(seed_Rs)} seeds beat at n=12 L=16 "
                f"(median seed R={seed_med:.3f}). "
                f"Transfer winner: {winner['winner']} ({winner['reason']})."
            ),
            "winner": winner,
            "seed_median_R12": seed_med,
            "n_seed_beat": n_seed_beat,
        }

    bits = []
    if not all(walsh_sel.values()):
        missed = [str(L) for L in DEPTHS if not walsh_sel[L]]
        bits.append(f"Walsh misses both-n BEAT on the {{6,9}} holdout at L={','.join(missed)}")
    if not all(walsh_n12.values()):
        missed = [str(L) for L in DEPTHS if not walsh_n12[L]]
        bits.append(f"frozen n=12 is not BEAT at L={','.join(missed)} (L=16 R={r12:.3f})")
    if not n15_hold:
        bits.append(
            f"n=15 flips a depth "
            f"(L=2 R={float(_cell(cells, 'walsh', 2, 15)['R']):.3f}, "
            f"L=16 R={float(_cell(cells, 'walsh', 16, 15)['R']):.3f})"
        )
    if not seeds_agree:
        bits.append(
            f"seeds disagree at n=12 L=16 "
            f"(per-seed R={', '.join(f'{r:.3f}' for r in seed_Rs)}, "
            f"median {seed_med:.3f}, ratio-of-medians {r12:.3f}, {n_seed_beat} beats)"
        )
    if not n15_quantum:
        bits.append(f"n=15 ground state is a bitstring (peak med {float(quantum[15]['peak_med']):.3f})")
    return {
        "verdict": "needs_edit",
        "reason": "; ".join(bits) if bits else "seal incomplete",
        "winner": winner,
        "seed_median_R12": seed_med,
        "n_seed_beat": n_seed_beat,
    }
