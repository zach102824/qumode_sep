"""Gates for the λ=1.5 fidelity-transfer verdict and the B=2 shell."""
from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

_PATH = Path(__file__).resolve().parents[1] / "scripts" / "run_powerlaw_fidelity_lam150.py"
_SPEC = importlib.util.spec_from_file_location("fidlam150", _PATH)
mod = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(mod)

from Nov_3rd.hva_safe import state_fidelity  # noqa: E402
from Nov_3rd.powerlaw_tfim import cost_diag, ground_state, make_instance  # noqa: E402


def _cell(L, tr6, tr9, ho6, ho9, f12, f15):
    def box(f):
        return {"med_F": float(f)}

    return {
        "L": L,
        "method": "snap_b2",
        "train": {"6": box(tr6), "9": box(tr9)},
        "hold": {"6": box(ho6), "9": box(ho9)},
        "freeze": {"12": box(f12), "15": box(f15)},
    }


class VerdictTest(unittest.TestCase):
    def test_pass_when_freeze_holds(self):
        cells = [
            _cell(1, 0.90, 0.88, 0.89, 0.87, 0.80, 0.70),
            _cell(4, 0.97, 0.96, 0.96, 0.95, 0.93, 0.90),
            _cell(8, 0.99, 0.99, 0.98, 0.98, 0.97, 0.96),
        ]
        out = mod.decide(cells)
        self.assertEqual(out["verdict"], "PASS")
        self.assertEqual(out["Lstar"], 4)

    def test_needs_edit_when_freeze_collapses(self):
        cells = [_cell(8, 0.97, 0.96, 0.96, 0.95, 0.70, 0.40)]
        out = mod.decide(cells)
        self.assertEqual(out["verdict"], "needs_edit")
        self.assertEqual(out["Lstar"], 8)

    def test_needs_edit_when_only_train_clears(self):
        cells = [_cell(16, 0.96, 0.95, 0.90, 0.80, 0.50, 0.40)]
        out = mod.decide(cells)
        self.assertEqual(out["verdict"], "needs_edit")
        self.assertIsNone(out["Lstar"])

    def test_kill_when_train_never_clears(self):
        cells = [
            _cell(1, 0.40, 0.30, 0.40, 0.30, 0.20, 0.10),
            _cell(16, 0.94, 0.90, 0.93, 0.88, 0.80, 0.70),
        ]
        out = mod.decide(cells)
        self.assertEqual(out["verdict"], "KILL")
        self.assertIsNone(out["Lstar"])


class AnsatzTest(unittest.TestCase):
    def test_b2_contains_paper_shell(self):
        phi = -float(mod.INIT_BETA) * np.arange(4)
        err = np.max(np.abs(mod.B2_INIT @ mod.B2_BASIS - phi))
        self.assertLess(err, 1e-12)

    def test_b2_init_matches_fourier(self):
        inst = make_instance(6, 0)
        Cn = cost_diag(torch.tensor(inst["diag"], dtype=torch.float64).reshape(1, -1), 6)
        L = 3
        snap = mod.init_local("snap_b2", L)
        four = mod.init_local("fourier", L)
        a = mod.prepare_local("snap_b2", Cn, snap)
        b = mod.prepare_local("fourier", Cn, four)
        fid = float(state_fidelity(a, b)[0])
        self.assertGreater(fid, 1.0 - 1e-10)

    def test_mean_x_of_x_ground_state(self):
        diag = np.zeros(8, dtype=np.float64)
        gs = ground_state(diag, 3, 1.5, 0.0)
        self.assertAlmostEqual(mod.mean_x(gs["psi"], 3), 1.0, places=8)
        self.assertAlmostEqual(mod.f_plus(gs["psi"]), 1.0, places=8)
        z = np.zeros(8)
        z[0] = 1.0
        self.assertAlmostEqual(mod.mean_x(z, 3), 0.0, places=12)

    def test_b2_gamma_has_a_fidelity_gradient(self):
        inst = make_instance(6, 1)
        gs = ground_state(inst["diag"], 6, 1.5, 0.0)
        Cn = cost_diag(torch.tensor(inst["diag"], dtype=torch.float64).reshape(1, -1), 6)
        angles = mod.init_local("snap_b2", 2)
        params = mod.bind_local("snap_b2", angles)
        pool = {
            "diag": torch.tensor(inst["diag"], dtype=torch.float64).reshape(1, -1),
            "psi": torch.tensor(gs["psi"], dtype=torch.float64).reshape(1, -1),
        }
        loss = mod._infidelity("snap_b2", angles, pool, torch.tensor([0]), 6)
        loss.backward()
        grads = [p.grad for p in params]
        self.assertTrue(all(g is not None for g in grads))
        self.assertTrue(any(float(g.abs().sum()) > 0 for g in grads))


if __name__ == "__main__":
    unittest.main()
