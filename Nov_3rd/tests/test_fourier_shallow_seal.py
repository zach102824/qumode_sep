"""Gate tests for the shallow Fourier seal. No training."""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from Nov_3rd.fourier_shallow_gate import self_test
from Nov_3rd.hva_safe import init_angles


def test_gate_rule():
    self_test()


def test_fourier_init_has_no_xx_mixer():
    ang = init_angles("fourier", 4, dtype=__import__("torch").float64)
    assert set(ang) == {"betas", "gammas"}
    assert "beta_xx" not in ang
    assert ang["betas"].shape == (4,)
    assert ang["gammas"].shape == (4,)


def test_script_refuses_deep_and_hva():
    import importlib.util

    path = _ROOT / "Nov_3rd" / "scripts" / "run_fourier_shallow_seal.py"
    spec = importlib.util.spec_from_file_location("run_fourier_shallow_seal", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    try:
        mod._reject_scope([2, 8], ["fourier", "snap"])
    except SystemExit:
        pass
    else:
        raise AssertionError("L=8 must be refused")
    try:
        mod._reject_scope([2, 4], ["fourier", "hva"])
    except SystemExit:
        pass
    else:
        raise AssertionError("HVA training must be refused")
    mod._reject_scope([2, 4], ["fourier", "snap"])


if __name__ == "__main__":
    test_gate_rule()
    test_fourier_init_has_no_xx_mixer()
    test_script_refuses_deep_and_hva()
    print("ok")
