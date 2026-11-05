"""Decision bars and the power-law Hamiltonian identity."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from Nov_3rd.powerlaw_tfim import decide, self_check, transfer_collapses


def _ratios(r6, r9, r12, hva=None):
    out = {"fourier": {"6": r6, "9": r9, "12": r12}}
    if hva is not None:
        out["hva"] = {"6": hva[0], "9": hva[1], "12": hva[2]}
    return out


def _cell(lam, mu, L, r6, r9, r12, quantum=True, hva=None, per_seed=None):
    return {
        "lam": lam,
        "mu": mu,
        "L": L,
        "quantum_pass": quantum,
        "ratios": _ratios(r6, r9, r12, hva=hva),
        "per_seed": per_seed or {},
    }


class DecisionTest(unittest.TestCase):
    def test_driver_pass(self):
        cells = [
            _cell(0.5, -0.5, 2, 0.70, 0.75, 0.80, hva=(0.72, 0.78, 0.84)),
            _cell(0.5, -0.5, 4, 1.10, 1.10, 1.10, hva=(1.10, 1.10, 1.10)),
            _cell(0.5, 0.0, 4, 1.05, 1.05, 1.05),
            _cell(1.0, 0.0, 4, 1.05, 1.05, 1.05),
            _cell(1.0, -0.5, 4, 1.05, 1.05, 1.05, hva=(1.05, 1.05, 1.05)),
        ]
        out = decide(cells)
        self.assertEqual(out["verdict"], "PASS")
        self.assertTrue(out["driver_passes"])
        self.assertTrue(out["seal_warranted"]["warranted"])

    def test_quantumness_blocks_pass(self):
        cells = [
            _cell(0.5, -0.5, 2, 0.40, 0.42, 0.45, quantum=False, hva=(0.50, 0.52, 0.55)),
            _cell(0.5, -0.5, 4, 0.90, 0.92, 0.95, quantum=False, hva=(0.90, 0.92, 0.95)),
            _cell(0.5, 0.0, 4, 0.90, 0.92, 0.95, quantum=False),
            _cell(1.0, 0.0, 4, 0.90, 0.92, 0.95, quantum=False),
            _cell(1.0, -0.5, 4, 0.90, 0.92, 0.95, quantum=False, hva=(0.90, 0.92, 0.95)),
        ]
        out = decide(cells)
        self.assertEqual(out["verdict"], "needs_edit")
        self.assertFalse(out["kill"])

    def test_hva_can_clear_the_driver_bar(self):
        cells = [
            _cell(1.0, -0.5, 4, 0.95, 0.94, 0.96, hva=(0.70, 0.72, 0.80)),
            _cell(0.5, 0.0, 4, 1.10, 1.10, 1.10),
            _cell(1.0, 0.0, 4, 1.10, 1.10, 1.10),
            _cell(0.5, -0.5, 4, 1.10, 1.10, 1.10, hva=(1.10, 1.10, 1.10)),
        ]
        out = decide(cells)
        self.assertEqual(out["verdict"], "PASS")
        self.assertIn("hva", out["driver_passes"][0])

    def test_transfer_collapse_is_not_a_pass(self):
        # n12 is under 0.90 but it ran away from the train ratios.
        cells = [
            _cell(0.5, -0.5, 2, 0.40, 0.42, 0.88, hva=(0.40, 0.42, 0.88)),
            _cell(0.5, -0.5, 4, 0.97, 0.96, 1.05, hva=(0.97, 0.96, 1.05)),
            _cell(0.5, 0.0, 4, 0.97, 0.96, 1.05),
            _cell(1.0, 0.0, 4, 0.97, 0.96, 1.05),
            _cell(1.0, -0.5, 4, 0.97, 0.96, 1.05, hva=(0.97, 0.96, 1.05)),
        ]
        out = decide(cells)
        self.assertEqual(out["verdict"], "needs_edit")
        self.assertTrue(transfer_collapses(0.88, 0.40, 0.42))

    def test_pure_family_pass(self):
        cells = [
            _cell(0.5, 0.0, 4, 0.90, 0.91, 0.96),
            _cell(1.0, 0.0, 2, 1.10, 1.10, 1.10),
            _cell(0.5, -0.5, 2, 1.05, 1.04, 1.02, hva=(1.05, 1.04, 1.02)),
            _cell(0.5, -0.5, 4, 1.05, 1.04, 1.02, hva=(1.05, 1.04, 1.02)),
            _cell(1.0, -0.5, 4, 1.05, 1.04, 1.02, hva=(1.05, 1.04, 1.02)),
        ]
        out = decide(cells)
        self.assertEqual(out["verdict"], "PASS")
        self.assertTrue(out["pure_passes"])
        self.assertFalse(out["driver_passes"])

    def test_kill_when_every_l4_train_ratio_misses(self):
        cells = [
            _cell(0.5, 0.0, 2, 1.01, 1.02, 1.20),
            _cell(0.5, 0.0, 4, 1.01, 1.02, 1.20),
            _cell(1.0, 0.0, 4, 0.99, 1.05, 1.20),
            _cell(0.5, -0.5, 4, 1.10, 1.10, 1.20, hva=(1.10, 1.10, 1.20)),
            _cell(1.0, -0.5, 4, 1.10, 1.10, 1.20, hva=(1.10, 1.10, 1.20)),
        ]
        out = decide(cells)
        self.assertEqual(out["verdict"], "KILL")
        self.assertFalse(out["seal_warranted"]["warranted"])

    def test_near_miss_asks_for_a_seal(self):
        cells = [
            _cell(1.0, -0.5, 4, 0.90, 0.91, 0.92, hva=(0.90, 0.91, 0.92)),
            _cell(0.5, 0.0, 4, 1.10, 1.10, 1.10),
            _cell(1.0, 0.0, 4, 1.10, 1.10, 1.10),
            _cell(0.5, -0.5, 4, 1.10, 1.10, 1.10, hva=(1.10, 1.10, 1.10)),
        ]
        out = decide(cells)
        self.assertEqual(out["verdict"], "needs_edit")
        self.assertTrue(out["seal_warranted"]["warranted"])

    def test_reversed_depth_is_not_a_seal(self):
        cells = [
            _cell(1.0, 0.0, 2, 0.97, 0.96, 0.99),
            _cell(1.0, 0.0, 4, 0.99, 1.02, 1.03),
            _cell(0.5, -0.5, 4, 0.64, 0.78, 0.94, hva=(0.90, 0.95, 0.99)),
            _cell(0.5, 0.0, 4, 0.80, 0.85, 1.02, quantum=False),
            _cell(1.0, -0.5, 4, 0.72, 0.83, 0.96, hva=(0.95, 0.99, 1.00)),
        ]
        out = decide(cells)
        self.assertEqual(out["verdict"], "needs_edit")
        self.assertFalse(out["seal_warranted"]["warranted"])


class HamiltonianTest(unittest.TestCase):
    def test_self_check(self):
        self_check()


if __name__ == "__main__":
    unittest.main()
