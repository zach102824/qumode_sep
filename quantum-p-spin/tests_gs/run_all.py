"""Run all GS + ansatz tests."""
from __future__ import annotations
import sys
from pathlib import Path

PKG = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PKG.parent))
sys.path.insert(0, str(PKG))
sys.path.insert(0, str(PKG / "tests_gs"))
sys.path.insert(0, str(PKG / "tests_ansatz"))

import test_gs  # noqa: E402
import test_ansatz  # noqa: E402


def main():
    print("=== GS TESTS ===")
    test_gs.main()
    print("\n=== ANSATZ TESTS ===")
    test_ansatz.main()
    print("\n=== ALL TESTS PASSED ===")


if __name__ == "__main__":
    main()
