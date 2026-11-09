"""Shared paths and the frozen v1 budget rules. Imported before any fit."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

HERE = Path(__file__).resolve().parent
PREREG_PATH = HERE / "prereg.json"
OUT = HERE.parents[2] / "results" / "GR" / "campaign_q"
RUNS = OUT / "runs"
INSTANCES = OUT / "instances"
CACHE = OUT / "cache"

RECIPE = "NOV3_QAOA_QL_v1"


def load_prereg() -> dict:
    raw = PREREG_PATH.read_bytes()
    doc = json.loads(raw.decode("utf-8"))
    if doc.get("recipe_id") != RECIPE:
        raise RuntimeError(f"unexpected recipe {doc.get('recipe_id')}")
    doc["_sha256"] = hashlib.sha256(raw).hexdigest()
    return doc


def prereg_sha() -> str:
    return hashlib.sha256(PREREG_PATH.read_bytes()).hexdigest()


def seed_span(spec: dict) -> list[int]:
    return list(range(int(spec["start"]), int(spec["stop"]) + 1))


def allocation(q: int, L: int) -> dict | None:
    """Inclusive 4-start split. None means the cell is below the accounting floor."""
    Q = int(q) * int(L)
    if Q % 4 != 0:
        return None
    per = Q // 4
    if per < 16 or (per - 12) % 2 != 0:
        return None
    return {"Q": Q, "per_start": per, "T": (per - 12) // 2}


def runnable_qs(L: int, candidates: list[int]) -> list[int]:
    return [int(q) for q in candidates if allocation(int(q), int(L)) is not None]


def assert_allowed_seed(seed: int, stage: str) -> None:
    seed = int(seed)
    if 11000 <= seed <= 11031:
        raise RuntimeError(f"validation seed {seed} is blocked")
    if 10010 <= seed <= 10031:
        raise RuntimeError(f"seed {seed} belongs to another worker")
    if stage == "dev":
        if not 10000 <= seed <= 10009:
            raise RuntimeError(f"dev seed {seed} is outside 10000-10009")
    elif stage == "final":
        if not 20000 <= seed <= 20999:
            raise RuntimeError(f"final seed {seed} is outside 20000-20999")
    else:
        raise RuntimeError(f"unknown stage {stage}")


def job_id(job: dict) -> str:
    return f"n{int(job['N'])}_s{int(job['seed'])}_L{int(job['L'])}_q{int(job['q'])}_{job['method']}"


def result_path(job: dict) -> Path:
    return RUNS / f"n{int(job['N']):02d}" / f"s{int(job['seed'])}" / f"L{int(job['L'])}_q{int(job['q'])}_{job['method']}.json"


def atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    os.replace(tmp, path)
