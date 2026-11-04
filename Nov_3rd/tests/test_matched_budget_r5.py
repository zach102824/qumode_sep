"""Protocol locks for the Round-5 μ=−0.25 champ bakeoff. No full train."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from Nov_3rd.scripts.matched_budget_r5 import (
    B_HOLD,
    B_N12,
    B_N15,
    B_TRAIN,
    FAMILIES,
    KAPPA,
    LAM_A,
    LAM_B,
    LAM_C,
    LR,
    MAX_ITER,
    MU,
    ROUNDS_L16,
    ROUNDS_L2,
    SEEDS,
    ZXScore,
    apply_gaussian_J,
    fit_matched,
    mark_of,
    protocol_ok,
    rank_transfer,
    unsat_histogram_check,
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
    assert KAPPA == 0.0 and MU == -0.25
    assert (LAM_A, LAM_B, LAM_C) == (0.5, 0.5, 0.75)
    heads = [f["head"] for f in FAMILIES]
    assert heads == ["power_p", "power_p", "walsh"]
    assert [f["id"] for f in FAMILIES] == ["gauss", "rank2", "unsat"]
    assert all(f["mu"] == -0.25 for f in FAMILIES)


def test_gaussian_J_is_deterministic_and_not_pm1():
    clauses = [(0, 1, 3, 1), (0, 4, 7, -1)]
    a = apply_gaussian_J(clauses, seed=51060000016, n=6)
    b = apply_gaussian_J(clauses, seed=51060000016, n=6)
    c = apply_gaussian_J(clauses, seed=51060000024, n=6)
    assert a == b
    assert [abs(J) for *_, J in a] != [1.0, 1.0]
    assert a != c
    assert all(abs(J) > 1e-6 for *_, J in a)


def test_rank_orders_n12_then_n15():
    cells = []
    for fid, r12, r15 in (("gauss", 0.90, 0.50), ("rank2", 0.80, 0.99), ("unsat", 0.80, 0.70)):
        cells.append({"id": fid, "champ": fid, "head": "h", "L": 16, "n": 12, "R": r12, "mark": mark_of(r12), "med_eps": r12})
        cells.append({"id": fid, "champ": fid, "head": "h", "L": 16, "n": 15, "R": r15, "mark": mark_of(r15), "med_eps": r15})
    order = [row["id"] for row in rank_transfer(cells)]
    assert order == ["unsat", "rank2", "gauss"]


def test_fit_runs_every_round_when_the_score_stalls():
    hold = [_Split(), _Split()]
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
    assert rec["lbfgs_failures"] == 0


def test_protocol_ok_rejects_a_short_budget():
    rec = {
        "family": "gauss",
        "method": "power_p",
        "L": 16,
        "seed": 0,
        "rounds_ran": 24,
        "rounds_budget": 24,
        "early_stopped": False,
        "lbfgs_failures": 0,
        "lr": LR,
    }
    ok, why = protocol_ok([rec])
    assert ok is False
    assert "stopped early" in why or "expected" in why


def test_unsat_zx_histogram_matches_statevector():
    got = unsat_histogram_check()
    assert got["mu0_abs"] < 1e-8
    assert got["statevector_abs"] < 1e-7
    assert ZXScore is not None


def test_no_n21_in_the_split_list():
    text = Path(__file__).resolve().parents[1].joinpath("scripts", "matched_budget_r5.py").read_text()
    assert "n=21 is out of this bakeoff" in text
    assert "B_N21" not in text
