"""Run all GS + ansatz tests."""
from __future__ import annotations
import importlib.util
import sys
from pathlib import Path

PKG = Path(__file__).resolve().parent
sys.path.insert(0, str(PKG.parent))
sys.path.insert(0, str(PKG))


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    print("=== GS TESTS ===", flush=True)
    gs = _load("test_gs", PKG / "tests_gs" / "test_gs.py")
    gs.main()
    print("\n=== ANSATZ TESTS ===", flush=True)
    an = _load("test_ansatz", PKG / "tests_ansatz" / "test_ansatz.py")
    an.main()
    print("\n=== ALL TESTS PASSED ===", flush=True)


if __name__ == "__main__":
    main()
