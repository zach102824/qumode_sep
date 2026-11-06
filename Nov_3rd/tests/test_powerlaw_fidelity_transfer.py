"""Verdict rules and B=2 wiring for the λ∈{0.25, 0.50} fidelity transfer."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from Nov_3rd.scripts.run_powerlaw_fidelity_lam025_050 import (
    call_one,
    call_study,
    library_check,
)


class VerdictTest(unittest.TestCase):
    def test_pass_when_freeze_holds(self):
        self.assertEqual(call_one(True, 0.96, 0.80), "pass")
        self.assertEqual(call_one(True, 0.96, 0.71), "pass")

    def test_drop_budget_is_0_25(self):
        # 0.96 - 0.70 = 0.26, which is past the drop budget even though F12 hits 0.70.
        self.assertEqual(call_one(True, 0.96, 0.70), "collapse")
        self.assertEqual(call_one(True, 0.95, 0.70), "pass")

    def test_collapse_and_miss(self):
        self.assertEqual(call_one(True, 0.97, 0.40), "collapse")
        self.assertEqual(call_one(False, 0.80, 0.80), "no95")
        self.assertEqual(call_one(True, 0.96, None), "no95")

    def test_slice_priority(self):
        self.assertEqual(call_study(["pass", "no95"]), "PASS")
        self.assertEqual(call_study(["collapse", "no95"]), "needs_edit")
        self.assertEqual(call_study(["no95", "no95"]), "KILL")
        self.assertEqual(call_study(["pass", "collapse"]), "PASS")


class LibraryTest(unittest.TestCase):
    def test_b2_matches_fourier_init_and_graded_cost(self):
        library_check()


if __name__ == "__main__":
    unittest.main()
