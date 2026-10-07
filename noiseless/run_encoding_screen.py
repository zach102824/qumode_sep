#!/usr/bin/env python3
"""Encoding screen: variable→physical-slot assignments × K paired random inits on one Hamiltonian.

Trial = tuned growth + SPSA-Adam (the run_u_sweep default: jp, L 1→4, 200 steps/stage, Adam lr
0.5/0.2/0.05/0.02, kick 0.05, infinite-shot Gibbs cost with the sampled-tail η controller,
binary cavity codewords), with the Hamiltonian's variables placed on the chip by
``EncodingSpec(perm)``. Assignments are the 8!/2 = 20160 symmetry classes of
:func:`noiseless.encoding.distinct_assignments` (class index 0 = identity/legacy layout).

Init seeds are shared across assignments (paired) and equal run_u_sweep's jp L=4 seeds:
  seed = --seed + 100_000*ham_index + 1_000*U_NAMES.index('jp') + 10*4 + init
so the identity assignment reproduces the run_u_sweep tuned-default trials bit-for-bit.

Assignment lists (--assignments):
  smoke        identity + (--n-random) random distinct non-identity classes (rng --assign-seed)
  all          every class 0..20159
  range:a:b    classes a..b-1
  file:PATH    whitespace/comma-separated class indices
  i,j,k        explicit class indices

Checkpoint: one JSON line per finished trial in --out (default
noiseless/results/encoding_screen/<tag>/<ham stem>.jsonl, gitignored); finished (class, init)
pairs are skipped on restart. At the end (or with --summarize only) a per-assignment summary
<jsonl stem>_summary.json is written next to it.
"""

from __future__ import annotations

import os as _os

_os.environ.setdefault("OMP_NUM_THREADS", "1")
_os.environ.setdefault("MKL_NUM_THREADS", "1")
_os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from noiseless.encoding import (  # noqa: E402
    EncodingSpec,
    distinct_assignments,
    energy_tensor_for_spec,
    list_four_sat_npz,
    load_four_sat_npz,
    logical_energies_from_terms,
)
from noiseless.run_u_sweep import TUNED_GROW_LR_SCHEDULE, TUNED_STEPS_PER_STAGE  # noqa: E402
from noiseless.spsa_gibbs import NoiselessSimulator, ground_flat_from_bitstring, grow_trial  # noqa: E402
from noiseless.unitaries import U_NAMES, build_fixed_u  # noqa: E402

FINAL_L = 4
U_NAME = "jp"
LR_SCHEDULE = [float(v) for v in TUNED_GROW_LR_SCHEDULE.split(",")]
RUN_ROOT = _REPO / "noiseless" / "results" / "encoding_screen"


def init_seed(seed0: int, ham_index: int, init: int) -> int:
    return int(seed0) + 100_000 * int(ham_index) + 1_000 * list(U_NAMES).index(U_NAME) + 10 * FINAL_L + int(init)


@lru_cache(maxsize=None)
def _classes() -> list[tuple[int, ...]]:
    return distinct_assignments()


@lru_cache(maxsize=4)
def _ham(path: str) -> dict:
    inst = load_four_sat_npz(path)  # binary/identity; logical data is encoding-free
    le = logical_energies_from_terms(inst["terms"], inst["identity"])
    return {"logical_energies": le, "ground_bitstring": inst["ground_bitstring"],
            "ground_energy": float(inst["ground_energy"])}


@lru_cache(maxsize=1)
def _u():
    return build_fixed_u(U_NAME)


def run_trial(job: dict) -> dict:
    t0 = time.perf_counter()
    h = _ham(job["ham_path"])
    spec = EncodingSpec(tuple(job["perm"]))
    sim = NoiselessSimulator(
        u_fixed=_u(),
        energy_tensor=energy_tensor_for_spec(h["logical_energies"], spec),
        n_layers=1,
        ground_bitstring=h["ground_bitstring"],
        ground_flat_index=ground_flat_from_bitstring(h["ground_bitstring"], spec),
        encoding=spec,
    )
    res = grow_trial(sim, final_layers=FINAL_L, rng=np.random.default_rng(int(job["seed"])),
                     optimizer="spsa_adam", steps_per_stage=TUNED_STEPS_PER_STAGE,
                     lr_schedule=LR_SCHEDULE)
    return {
        "ham_file": job["ham_file"], "ham_index": job["ham_index"], "class_idx": job["class_idx"],
        "perm": list(job["perm"]), "init": job["init"], "seed": job["seed"],
        "p_gs": float(res.p_gs), "success": bool(res.success), "energy_mean": float(res.energy_mean),
        "fun": float(res.fun), "nfev": int(res.nfev), "most_likely_bitstring": res.most_likely_bitstring,
        "ground_bitstring": res.ground_bitstring, "ground_energy": h["ground_energy"],
        "stage_p_gs": [float(s["p_gs"]) for s in res.stages],
        "wall_s": float(time.perf_counter() - t0),
        "t_done": float(time.time()),  # unix time (status script throughput / ETA)
    }


def select_assignments(spec: str, n_random: int, assign_seed: int) -> list[int]:
    n = len(_classes())
    if spec == "smoke":
        rng = np.random.default_rng(int(assign_seed))
        rest = rng.choice(np.arange(1, n), size=int(n_random), replace=False)
        return [0] + sorted(int(v) for v in rest)
    if spec == "all":
        return list(range(n))
    if spec.startswith("range:"):
        a, b = (int(v) for v in spec.split(":")[1:3])
        return list(range(a, min(b, n)))
    if spec.startswith("file:"):
        txt = Path(spec[5:]).read_text().replace(",", " ").split()
        return [int(v) for v in txt]
    return [int(v) for v in spec.split(",") if v.strip()]


def _load_done(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass  # a partially written last line from an interrupted run
    return out


def summarize(records: list[dict]) -> dict:
    by: dict[int, list[dict]] = {}
    for r in records:
        by.setdefault(int(r["class_idx"]), []).append(r)
    rows = []
    for c, rs in by.items():
        pg = np.array([r["p_gs"] for r in rs])
        rows.append({
            "class_idx": c, "perm": rs[0]["perm"],
            "slots": EncodingSpec(tuple(rs[0]["perm"])).slot_of_variable(),
            "n": len(rs), "mean_p_gs": float(pg.mean()), "median_p_gs": float(np.median(pg)),
            "min_p_gs": float(pg.min()), "max_p_gs": float(pg.max()),
            "success": float(np.mean([r["success"] for r in rs])),
            "mean_energy": float(np.mean([r["energy_mean"] for r in rs])),
            "mean_wall_s": float(np.mean([r["wall_s"] for r in rs])),
            "total_wall_s": float(np.sum([r["wall_s"] for r in rs])),
        })
    rows.sort(key=lambda r: (-r["mean_p_gs"], r["class_idx"]))
    for i, r in enumerate(rows):
        r["rank"] = i + 1
    means = np.array([r["mean_p_gs"] for r in rows]) if rows else np.array([np.nan])
    ident = next((r for r in rows if r["class_idx"] == 0), None)
    return {
        "n_assignments": len(rows), "n_trials": len(records),
        "spread_mean_p_gs": {"min": float(means.min()), "median": float(np.median(means)),
                             "max": float(means.max()), "std": float(means.std())},
        "identity": ident,
        "ranking": rows,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ham-dir", default=str(_REPO / "Hamiltonians" / "four_sat"))
    p.add_argument("--ham", default="0", help="Hamiltonian index into the sorted four_sat list, or file name")
    p.add_argument("--assignments", default="smoke")
    p.add_argument("--n-random", type=int, default=99)
    p.add_argument("--assign-seed", type=int, default=20260917)
    p.add_argument("--inits", type=int, default=3)
    p.add_argument("--seed", type=int, default=20260917)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--tag", default="screen")
    p.add_argument("--out", default=None, help="JSONL checkpoint path")
    p.add_argument("--summarize", action="store_true", help="only (re)write the summary")
    args = p.parse_args(argv)

    paths = list_four_sat_npz(args.ham_dir)
    if args.ham.isdigit():
        hi = int(args.ham)
    else:
        hi = [q.name for q in paths].index(args.ham)
    ham_path = paths[hi]
    out = Path(args.out) if args.out else RUN_ROOT / args.tag / f"{ham_path.stem}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    summ_path = out.with_name(out.stem + "_summary.json")

    classes = _classes()
    todo_classes = select_assignments(args.assignments, args.n_random, args.assign_seed)
    done = _load_done(out)
    done_keys = {(int(r["class_idx"]), int(r["init"])) for r in done}
    jobs = []
    for c in todo_classes:
        for t in range(int(args.inits)):
            if (c, t) in done_keys:
                continue
            jobs.append({"ham_path": str(ham_path), "ham_file": ham_path.name, "ham_index": hi,
                         "class_idx": int(c), "perm": list(classes[c]), "init": t,
                         "seed": init_seed(args.seed, hi, t)})
    stamp = lambda: datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")  # noqa: E731
    if not args.summarize and not jobs and summ_path.exists():
        try:
            prev = json.loads(summ_path.read_text())
        except (OSError, json.JSONDecodeError):
            prev = {}
        if prev.get("n_trials") == len(todo_classes) * int(args.inits):
            # resume of a finished Hamiltonian: keep the existing summary (and its wall-time metadata)
            print(f"[{stamp()}] {ham_path.name}: complete ({len(done_keys)} trials), summary kept", flush=True)
            return 0
    t_start = time.perf_counter()
    if not args.summarize and jobs:
        print(f"[{stamp()}] {ham_path.name}: {len(todo_classes)} assignments x {args.inits} inits, "
              f"{len(done_keys)} done, {len(jobs)} to run, workers={args.workers} -> {out}", flush=True)
        if out.exists() and out.stat().st_size and not out.read_bytes().endswith(b"\n"):
            with out.open("a") as fh:  # interrupted mid-line: terminate the partial line
                fh.write("\n")
        with out.open("a") as fh:
            if args.workers <= 1:
                it = (run_trial(j) for j in jobs)
                for k, rec in enumerate(it, 1):
                    fh.write(json.dumps(rec) + "\n")
                    fh.flush()
            else:
                with ProcessPoolExecutor(max_workers=int(args.workers)) as ex:
                    futs = {ex.submit(run_trial, j) for j in jobs}
                    for k, fut in enumerate(as_completed(futs), 1):
                        fh.write(json.dumps(fut.result()) + "\n")
                        fh.flush()
                        futs.discard(fut)  # do not retain finished results (60k jobs/H)
                        if k % 1000 == 0 or k == len(jobs):
                            print(f"  [{k}/{len(jobs)}] {time.perf_counter() - t_start:.1f}s", flush=True)
    wall = time.perf_counter() - t_start
    wanted = set(todo_classes)
    recs = [r for r in _load_done(out) if int(r["class_idx"]) in wanted and int(r["init"]) < args.inits]
    summ = summarize(recs)
    summ.update({"created_utc": stamp(), "ham_file": ham_path.name, "ham_index": hi,
                 "assignments": args.assignments, "inits": args.inits, "seed": args.seed,
                 "this_invocation_wall_s": wall, "this_invocation_trials": len(jobs),
                 "workers": args.workers, "jsonl": str(out)})
    summ_path.write_text(json.dumps(summ, indent=2))
    sp = summ["spread_mean_p_gs"]
    ident = summ["identity"]
    print(f"[{stamp()}] wall {wall:.1f}s for {len(jobs)} trials; mean p(GS) over assignments "
          f"min {sp['min']:.4f} median {sp['median']:.4f} max {sp['max']:.4f}"
          + (f"; identity {ident['mean_p_gs']:.4f} rank {ident['rank']}/{summ['n_assignments']}" if ident else ""),
          flush=True)
    print(f"summary {summ_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
