"""Shared paths, prereg load, and seed guards for N9-GR.

D2–D5 were approved by Zach at 18:41 Asia/Shanghai on 2026-11-09 as simulation
assumptions. This package still makes no hardware-feasibility claim, no physical
shot claim, and it does not touch validation, final, or a recipe freeze.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

GR_DIR = Path(__file__).resolve().parent
RESULT_DIR = GR_DIR.parents[1] / "results" / "GR"
PREREG_PATH = GR_DIR / "prereg.json"

ALPHA_MAX = 0.3
CODE_LEVELS = 8


def load_prereg() -> dict:
    raw = PREREG_PATH.read_bytes()
    doc = json.loads(raw.decode("utf-8"))
    doc["_sha256"] = hashlib.sha256(raw).hexdigest()
    doc["_bytes"] = len(raw)
    return doc


def assert_dev_seed(seed: int, *, pilot: bool) -> None:
    seed = int(seed)
    if 11000 <= seed <= 11031 or 20000 <= seed <= 20999 or seed >= 11000:
        raise RuntimeError(f"blind seed {seed} is blocked (validation/final)")
    if not (10000 <= seed <= 10031):
        raise RuntimeError(f"seed {seed} is outside the dev window 10000..10031")
    if pilot and seed > 10007:
        raise RuntimeError(f"pilot seed {seed} is outside the first-8 window 10000..10007")


def assert_pilot_N(n: int) -> None:
    if int(n) not in (6, 9):
        raise RuntimeError(f"N={n} is outside the GR pilot (6, 9). N=12/15 need a feasibility pass first.")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def code_fingerprint() -> dict:
    out = {}
    for p in sorted(GR_DIR.glob("*.py")) + sorted(GR_DIR.glob("*.json")) + sorted(GR_DIR.glob("*.md")):
        out[p.name] = sha256_file(p)
    return out
