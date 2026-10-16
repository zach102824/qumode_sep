#!/usr/bin/env python3
"""Write the SAT formulas used in QAOA evals to qaoa_ksat/hamiltonians/."""

from __future__ import annotations

import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from qaoa_ksat.ksat import THRESHOLD_R, instance_to_dict, sat_instance_from_seed

HAM_DIR = Path(__file__).resolve().parent / "hamiltonians"
RESULTS = Path(__file__).resolve().parent / "results"


def _jobs_from_eval_json(path: Path) -> list[dict]:
    d = json.loads(path.read_text())
    jobs = []
    for row in d.get("rows", []):
        if not row.get("ok"):
            continue
        jobs.append(
            {
                "n": int(row["n"]),
                "k": int(row["k"]),
                "r": float(row["r"]),
                "p": int(row["p"]),
                "trial": int(row["trial"]),
                "seed": int(row["seed"]),
                "source": path.name,
            }
        )
    return jobs


def _dump_one(job: dict) -> str:
    inst, energy = sat_instance_from_seed(job["n"], job["k"], job["r"], job["seed"])
    rec = instance_to_dict(
        inst,
        energy,
        extra={
            "eval_p": job["p"],
            "eval_trial": job["trial"],
            "eval_seed": job["seed"],
            "source": job["source"],
        },
    )
    rec["seed"] = job["seed"]
    ndir = HAM_DIR / f"n{job['n']:02d}"
    ndir.mkdir(parents=True, exist_ok=True)
    out = ndir / f"seed_{job['seed']}.json"
    out.write_text(json.dumps(rec))
    return str(out)


def dump_all(workers: int = 6) -> list[Path]:
    HAM_DIR.mkdir(parents=True, exist_ok=True)
    jobs = []
    for path in sorted(RESULTS.glob("eval_n*_k8_*.json")):
        jobs.extend(_jobs_from_eval_json(path))
    # Unique by (n, seed); keep first p tag.
    seen: set[tuple[int, int]] = set()
    uniq = []
    for job in jobs:
        key = (job["n"], job["seed"])
        if key in seen:
            continue
        seen.add(key)
        uniq.append(job)
    written: list[Path] = []
    if workers <= 1:
        for job in uniq:
            written.append(Path(_dump_one(job)))
            if len(written) % 20 == 0:
                print(f"wrote {len(written)}/{len(uniq)}", flush=True)
    else:
        with ProcessPoolExecutor(max_workers=int(workers)) as ex:
            futs = [ex.submit(_dump_one, job) for job in uniq]
            for i, fut in enumerate(as_completed(futs), 1):
                written.append(Path(fut.result()))
                if i % 20 == 0:
                    print(f"wrote {i}/{len(uniq)}", flush=True)
    man = {
        "k": 8,
        "r": THRESHOLD_R[8],
        "n_files": len(written),
        "by_n": {},
    }
    for p in written:
        n = int(p.parent.name[1:])
        man["by_n"].setdefault(str(n), 0)
        man["by_n"][str(n)] += 1
    (HAM_DIR / "manifest.json").write_text(json.dumps(man, indent=2))
    print(f"wrote {len(written)} instances -> {HAM_DIR}")
    print(man["by_n"])
    return written


if __name__ == "__main__":
    dump_all()
