"""Gate tests for the Λ≤0.80 matched bakeoff. No training."""
from __future__ import annotations

import unittest

import torch

from Nov_3rd.bakeoff_lambda_safe import (
    CANDIDATES,
    assert_n_in_scope,
    decision_self_test,
    family_verdict,
    recommend,
    render,
    rank_cells,
)
from Nov_3rd.hva_safe import cn_from_unsat, snap_state, snap_state_from_components, walsh_degree_components


class BakeoffGateTest(unittest.TestCase):
    def test_decision_self_test(self):
        decision_self_test()

    def test_n21_refused(self):
        with self.assertRaises(RuntimeError):
            assert_n_in_scope(21)

    def test_family_words(self):
        self.assertEqual(family_verdict(["soft", "soft", "FAIL"]), "needs_edit")
        self.assertEqual(family_verdict(["FAIL", "PASS", "soft"]), "PASS")
        ranked = rank_cells(
            [
                {"id": "A", "lambda": 0.5, "mu": 0.0, "baseline": "fourier", "call": "soft",
                 "lambda_pass": True, "r": 0.9, "lam12": 0.56, "lam18": 0.57},
                {"id": "C", "lambda": 0.75, "mu": 0.0, "baseline": "fourier", "call": "PASS",
                 "lambda_pass": True, "r": 0.62, "lam12": 0.75, "lam18": 0.76},
            ]
        )
        rec = recommend(ranked)
        self.assertTrue(rec["lock"])
        self.assertEqual(rec["id"], "C")
        self.assertEqual(ranked[0]["id"], "C")

    def test_render_smoke(self):
        cell = {
            "id": "A",
            "lambda": 0.50,
            "mu": 0.0,
            "baseline": "fourier",
            "n12": _block(0.559, 0.11, 0.16),
            "n18": _block(0.57, 0.08, 0.04),
            "decision": {"call": "PASS", "fails": [], "n_clear": 5, "lambda_pass": True},
            "L2": _depth(0.75, 5),
            "L4": _depth(0.90, 2),
            "lambda_pass": True,
            "call": "PASS",
            "r": 0.75,
            "lam12": 0.559,
            "lam18": 0.57,
        }
        summary = {
            "verdict": "PASS",
            "recommendation": {
                "lock": True, "id": "A", "lambda": 0.50, "mu": 0.0, "baseline": "fourier",
                "nearest_id": "A", "nearest_call": "PASS",
            },
            "ranked": [cell],
            "freeze_b": 128,
            "n18_b": 32,
            "n18_base": 61000000000,
            "depths": [2, 4],
            "rounds": 40,
            "lbfgs_iter": 8,
            "lbfgs_lr": 0.25,
            "batch": 64,
        }
        text = render(summary)
        self.assertIn("NEW_SAFE paper-lock is A", text)
        self.assertIn("n=21 was not loaded", text)
        self.assertEqual(len(CANDIDATES), 3)

    def test_walsh_components_match_snap(self):
        torch.manual_seed(0)
        n, B, L = 6, 3, 2
        E = torch.randint(0, 5, (B, 1 << n), dtype=torch.float64)
        Cn = cn_from_unsat(E, n)
        gamma = torch.randn(L, 4, dtype=torch.float64)
        phi = torch.randn(L, 4, dtype=torch.float64)
        comps = walsh_degree_components(Cn)
        a = snap_state(Cn, gamma, phi)
        b = snap_state_from_components(comps, gamma, phi)
        err = torch.max(torch.abs(a - b)).item()
        self.assertLess(err, 1e-10)


def _block(lam: float, gap: float, peak: float) -> dict:
    return {
        "B": 32,
        "lambda_med": lam,
        "lambda_min": lam - 0.1,
        "lambda_max": lam + 0.1,
        "frac_gt_080": 0.01,
        "frac_gt_085": 0.0,
        "gap_med": gap,
        "gap_min": gap / 2,
        "peak_med": peak,
        "frac_peak_ge_05": 0.0,
    }


def _depth(r: float, n_clear: int) -> dict:
    by = {"med_snap": r, "med_base": 1.0, "r": r, "paired_win": 100, "paired_n": 128}
    return {
        "by_n": {"6": dict(by), "9": dict(by), "12": dict(by)},
        "per_seed": [
            {"seed": s, "snap": r, "base": 1.0, "r": r, "best_round": 10, "wall_s": 1.0}
            for s in range(5)
        ],
        "n_clear": n_clear,
        "r_seeds": [r] * 5,
    }


if __name__ == "__main__":
    unittest.main()
