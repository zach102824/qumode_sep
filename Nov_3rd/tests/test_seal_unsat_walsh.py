#!/usr/bin/env python3
"""Verdict rules for the unsat Walsh seal. No training."""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from Nov_3rd.seal_unsat import decide, pick_hc_winner


def _quantum(peak=0.08):
    return {n: {"peak_med": peak} for n in (6, 9, 12, 15)}


def _cells(walsh_R, power_R=None, seed_Rs=None):
    """Build a full unsat cell table. ``walsh_R[(L, n)]`` is the ratio."""
    if power_R is None:
        power_R = {(L, n): 0.99 for L in (2, 16) for n in (6, 9, 12, 15)}
    cells = {}
    for method, table in (("walsh", walsh_R), ("power_p", power_R)):
        for (L, n), R in table.items():
            R = float(R)
            if R <= 0.98:
                mark = "BEAT"
            elif R <= 1.02:
                mark = "TIE"
            else:
                mark = "UNDER"
            srs = seed_Rs if (method == "walsh" and L == 16 and n == 12 and seed_Rs is not None) else [R, R, R]
            cells[(method, L, n)] = {
                "R": R,
                "mark": mark,
                "med": 0.01 * R,
                "seed_Rs": list(srs),
            }
    return cells


def _grape(cost=0.40):
    return {"cost_only_med": {16: {12: cost}}}


def _all_beat_walsh():
    return {(L, n): (0.70 if L == 2 else 0.85) for L in (2, 16) for n in (6, 9, 12, 15)}


def test_pass_when_walsh_transfers_both_depths():
    cells = _cells(_all_beat_walsh(), seed_Rs=[0.86, 0.84, 0.90])
    out = decide(cells, _quantum(), _grape())
    assert out["verdict"] == "PASS", out
    assert out["winner"]["winner"] == "unsat"
    assert out["winner"]["head"] == "walsh"
    assert out["n_seed_beat"] == 3


def test_power_p_miss_does_not_block_walsh_pass():
    """Scalar power_p may tie. PASS is the Walsh head."""
    power = {(L, n): 0.99 for L in (2, 16) for n in (6, 9, 12, 15)}
    cells = _cells(_all_beat_walsh(), power_R=power, seed_Rs=[0.80, 0.88, 0.91])
    out = decide(cells, _quantum(), _grape())
    assert out["verdict"] == "PASS", out


def test_kill_classical_ground_state():
    cells = _cells(_all_beat_walsh())
    q = _quantum(0.04)
    q[12] = {"peak_med": 0.62}
    out = decide(cells, q, _grape())
    assert out["verdict"] == "KILL", out
    assert out["winner"]["winner"] == "none"


def test_kill_when_neither_head_beats_both_n():
    weak = {(L, n): 1.05 for L in (2, 16) for n in (6, 9, 12, 15)}
    # one-sided beats must not count as both-n
    weak[(16, 6)] = 0.90
    cells = _cells(weak, power_R=dict(weak))
    out = decide(cells, _quantum(), _grape())
    assert out["verdict"] == "KILL", out


def test_kill_when_advantage_reverses():
    table = _all_beat_walsh()
    table[(16, 6)] = 0.80
    table[(16, 9)] = 1.10
    table[(16, 12)] = 1.20
    table[(16, 15)] = 1.20
    cells = _cells(table, seed_Rs=[1.2, 1.2, 1.2])
    out = decide(cells, _quantum(), _grape())
    assert out["verdict"] == "KILL", out


def test_kill_when_cost_only_matches_walsh():
    cells = _cells(_all_beat_walsh())
    # walsh med = 0.01 * R = 0.0085 at L=16; cost-only equal to that
    out = decide(cells, _quantum(), _grape(cost=0.008))
    assert out["verdict"] == "KILL", out


def test_needs_edit_when_only_deep_depth_transfers():
    table = _all_beat_walsh()
    table[(2, 12)] = 1.00
    table[(2, 15)] = 1.00
    cells = _cells(table, seed_Rs=[0.86, 0.88, 0.90])
    out = decide(cells, _quantum(), _grape())
    assert out["verdict"] == "needs_edit", out
    assert out["winner"]["winner"] == "unsat"


def test_needs_edit_when_seeds_split_the_line():
    table = _all_beat_walsh()
    cells = _cells(table, seed_Rs=[0.70, 1.05, 1.10])
    # ratio of medians still a beat, but two seeds are over the line
    out = decide(cells, _quantum(), _grape())
    assert out["verdict"] == "needs_edit", out
    assert out["n_seed_beat"] == 1


def test_zzz_wins_when_unsat_only_ties_transfer():
    table = {(L, n): 0.99 for L in (2, 16) for n in (6, 9, 12, 15)}
    # selection still beats so this is needs_edit, not KILL
    for n in (6, 9):
        table[(16, n)] = 0.90
        table[(2, n)] = 0.90
    cells = _cells(table, power_R=dict(table), seed_Rs=[0.99, 0.99, 0.99])
    out = decide(cells, _quantum(), _grape())
    assert out["verdict"] == "needs_edit", out
    assert out["winner"]["winner"] == "zzz", out["winner"]


def test_needs_edit_when_n15_ground_state_is_classical():
    cells = _cells(_all_beat_walsh(), seed_Rs=[0.86, 0.84, 0.90])
    q = _quantum(0.08)
    q[15] = {"peak_med": 0.71}
    out = decide(cells, q, _grape())
    assert out["verdict"] == "needs_edit", out


def test_tie_when_transfer_profiles_match():
    # 2 transfer beats (L=16 only) and R within 0.01 of ZZZ's 0.924
    table = {(L, n): (0.920 if L == 16 else 1.00) for L in (2, 16) for n in (6, 9, 12, 15)}
    cells = _cells(table, power_R={(L, n): 1.05 for L in (2, 16) for n in (6, 9, 12, 15)})
    win = pick_hc_winner(cells)
    assert win["winner"] == "tie", win


def main():
    tests = [
        test_pass_when_walsh_transfers_both_depths,
        test_power_p_miss_does_not_block_walsh_pass,
        test_kill_classical_ground_state,
        test_kill_when_neither_head_beats_both_n,
        test_kill_when_advantage_reverses,
        test_kill_when_cost_only_matches_walsh,
        test_needs_edit_when_only_deep_depth_transfers,
        test_needs_edit_when_seeds_split_the_line,
        test_zzz_wins_when_unsat_only_ties_transfer,
        test_needs_edit_when_n15_ground_state_is_classical,
        test_tie_when_transfer_profiles_match,
    ]
    for fn in tests:
        fn()
        print("ok", fn.__name__)
    print(f"passed {len(tests)}")


if __name__ == "__main__":
    main()
