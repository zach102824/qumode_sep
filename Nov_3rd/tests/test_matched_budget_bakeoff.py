"""Protocol locks for the matched-budget champ bakeoff. No full train."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from Nov_3rd.scripts.matched_budget_bakeoff import (
    B_HOLD,
    B_N12,
    B_N15,
    B_TRAIN,
    LAM_A,
    LAM_C,
    LR,
    MAX_ITER,
    MU_B,
    ROUNDS_L16,
    ROUNDS_L2,
    SEEDS,
    apply_gaussian_J,
    diag_expectation,
    fit_matched,
    mark_of,
    rank_transfer,
)


class _Toy:
    def __init__(self):
        self.n = 6
        self.e0 = torch.zeros(2, dtype=torch.float64)

    def energy(self, gamma, phi, graded, lam):
        return self.e0 + gamma.square().sum() + phi.square().sum()


class _Split:
    def __init__(self):
        self.n = 6
        self.role = "hold"
        self.circuit = _Toy()


def test_protocol_is_pr19_full_budget():
    assert (ROUNDS_L2, ROUNDS_L16, MAX_ITER, LR) == (18, 48, 8, 0.35)
    assert SEEDS == (0, 1, 2)
    assert (B_TRAIN, B_HOLD, B_N12, B_N15) == (128, 128, 64, 32)
    assert LAM_A == 0.5 and LAM_C == 0.75 and MU_B == -0.25


def test_gaussian_J_is_deterministic_and_not_pm1():
    clauses = [(0, 1, 3, 1), (0, 4, 7, -1)]
    a = apply_gaussian_J(clauses, seed=51060000016, n=6)
    b = apply_gaussian_J(clauses, seed=51060000016, n=6)
    c = apply_gaussian_J(clauses, seed=51060000024, n=6)
    assert a == b
    assert [abs(J) for *_, J in a] != [1.0, 1.0]
    assert a != c
    assert all(abs(J) > 1e-6 for *_, J in a)


def test_diag_expectation_matches_uniform_plus():
    diag = np.arange(8, dtype=np.float64)
    theta = np.full(3, 0.5 * np.pi)
    # |+⟩ is uniform, mean of 0..7 is 3.5, and sin(π/2)=1 so the field is separate.
    assert abs(diag_expectation(diag, theta) - 3.5) < 1e-12


def test_rank_orders_n12_then_n15():
    cells = []
    for fid, r12, r15 in (("gauss", 0.90, 0.50), ("rank2", 0.80, 0.99), ("unsat", 0.80, 0.70)):
        cells.append({"id": fid, "champ": fid, "head": "h", "L": 16, "n": 12, "R": r12, "mark": mark_of(r12), "med_eps": r12})
        cells.append({"id": fid, "champ": fid, "head": "h", "L": 16, "n": 15, "R": r15, "mark": mark_of(r15), "med_eps": r15})
    order = [row["id"] for row in rank_transfer(cells)]
    assert order == ["unsat", "rank2", "gauss"]


def test_fit_runs_every_round_when_the_score_stalls():
    hold = [_Split(), _Split()]
    # Two copies with n=6 and n=9. The toy ignores n except through decode's shell scale.
    hold[1].n = 9
    train = [_Split(), _Split()]
    train[1].n = 9
    rec = fit_matched(
        "fourier_qaoa", 2, 0, train, hold, 0.5,
        rounds=3, max_iter=2, lr=0.2, theta0=None, ref_med=None, family="toy",
    )
    assert rec["rounds_ran"] == 3
    assert rec["early_stopped"] is False
    assert rec["rounds_budget"] == 3
