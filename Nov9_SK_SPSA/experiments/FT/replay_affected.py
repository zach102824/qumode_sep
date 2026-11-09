"""Re-simulate only the dev records whose SPSA start aborted on restricted-D norm.

NF=32 is the trace-preserving cutoff: on every captured abort state the
oscillator at NF=48 puts ~1e-30 outside level 32, and NF=32 matches NF=48
to ~1e-14 in energy, fidelity, and leakage. The 1e-9 norm gate is unchanged.
States are not renormalized. Original pilot JSON is not modified.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from Nov9_SK_SPSA.experiments.FT.params import TRAIN_SEEDS
from Nov9_SK_SPSA.experiments.FT.run_ft import RESULT, _pool, load

REPLAY = RESULT / "replay_nf32"
NF = 32


def _tasks():
    cat = load(RESULT / "audit_followup" / "ledger_and_counts.json")
    tasks = []
    for row in cat["rows"]:
        doc = load(RESULT / row["path"])
        kind = doc["kind"]
        if kind == "cold":
            tasks.append({
                "kind": "cold",
                "method": doc["method"],
                "N": doc["N"],
                "b": float(doc["b"]),
                "L": int(doc["L"]),
                "seed": int(doc["seed"]),
                "cap": int(doc["cap"]),
                "n_iter": int(doc["n_iter_per_start"]),
                "nf": NF,
                "alpha_sigma_scale": float(doc["alpha_sigma_scale"]),
                "rng_seed": int(doc["seed"]),
                "path": str(REPLAY / row["path"]),
                "original": row["path"],
            })
        elif kind == "train":
            tasks.append({
                "kind": "train",
                "method": doc["method"],
                "N": doc["N"],
                "b": float(doc["b"]),
                "L": int(doc["L"]),
                "cap": int(doc["cap"]),
                "n_iter": int(doc["n_iter_per_start"]),
                "nf": NF,
                "alpha_sigma_scale": float(doc["alpha_sigma_scale"]),
                "rng_seed": TRAIN_SEEDS[0],
                "path": str(REPLAY / row["path"]),
                "original": row["path"],
            })
        elif kind in ("finetune", "zeroshot"):
            tasks.append({
                "kind": kind,
                "method": doc["method"],
                "train_method": doc["train_method"],
                "N": doc["N"],
                "b": float(doc["b"]),
                "L": int(doc["L"]),
                "seed": int(doc["seed"]),
                "cap": int(doc["cap"]),
                "n_iter": int(doc.get("n_iter_per_start") or 0),
                "nf": NF,
                "alpha_sigma_scale": float(doc["alpha_sigma_scale"]),
                "rng_seed": int(doc["seed"]),
                "x_star": doc["x_star"],
                "path": str(REPLAY / row["path"]),
                "original": row["path"],
            })
        else:
            raise RuntimeError(kind)
    return tasks


def _conv(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    raise TypeError(type(o))


def archive(tasks):
    changes = []
    for task in tasks:
        new = load(Path(task["path"]))
        old = load(RESULT / task["original"])
        x_changed = None
        if "x_star" in old and "x_star" in new:
            a = np.asarray(old["x_star"], dtype=np.float64)
            b = np.asarray(new["x_star"], dtype=np.float64)
            x_changed = float(np.max(np.abs(a - b)))
        changes.append({
            "path": task["original"],
            "reason": "restricted-D norm abort was the contractive NF=16 tail; replayed at trace-preserving NF=32 with the same 1e-9 gate, same seeds, same budget, no renormalization",
            "original_nf": old.get("nf"),
            "replay_nf": new.get("nf"),
            "original_failed_starts": old.get("failed_starts"),
            "replay_failed_starts": new.get("failed_starts"),
            "original_queries": old.get("queries_counted"),
            "replay_queries": new.get("queries_counted"),
            "original_selected_start": old.get("selected_start"),
            "replay_selected_start": new.get("selected_start"),
            "original_E": old.get("E"),
            "replay_E": new.get("E"),
            "original_F": old.get("F"),
            "replay_F": new.get("F"),
            "delta_E": None if old.get("E") is None else float(new["E"]) - float(old["E"]),
            "delta_F": None if old.get("F") is None else float(new["F"]) - float(old["F"]),
            "original_p_leak": old.get("p_leak"),
            "replay_p_leak": new.get("p_leak"),
            "original_eta": old.get("eta"),
            "replay_eta": new.get("eta"),
            "x_star_max_abs_change": x_changed,
        })
    path = REPLAY / "CHANGES.json"
    path.write_text(json.dumps(changes, default=_conv, indent=2))
    return changes


def main():
    tasks = _tasks()
    print(f"replay tasks {len(tasks)}", flush=True)
    # Drop the bookkeeping key before the worker sees it.
    worker_tasks = []
    for t in tasks:
        w = dict(t)
        w.pop("original", None)
        worker_tasks.append(w)
    results = _pool(worker_tasks, 4)
    errors = [r for r in results if r.get("error")]
    print("errors", len(errors), flush=True)
    if errors:
        (REPLAY / "ERRORS.txt").write_text("\n\n".join(r["error"] for r in errors))
        raise SystemExit("replay errors")
    changes = archive(tasks)
    dF = [c["delta_F"] for c in changes if c["delta_F"] is not None]
    dE = [c["delta_E"] for c in changes if c["delta_E"] is not None]
    still = sum(1 for c in changes if c["replay_failed_starts"])
    print(
        f"changed {len(changes)} still_failed {still} "
        f"median dF {np.median(dF):.6e} max |dF| {np.max(np.abs(dF)):.6e} "
        f"max |dE| {np.max(np.abs(dE)):.6e}",
        flush=True,
    )


if __name__ == "__main__":
    main()
