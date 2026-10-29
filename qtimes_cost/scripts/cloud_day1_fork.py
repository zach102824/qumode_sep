#!/usr/bin/env python3
"""Fork from the cloud L=12 BEST. No L≈10 box hot-band.

Grid: L∈{12,14,16} × lr∈{3e-5,5e-5,1e-4} × leakpen∈{0.05,0.1,0.2},
warm-started from the saved L=12 weights. The exact BEST cell is not
retrained from scratch: it gets a longer Adam run and a longer early-stop
patience. Learned α0 only.

A trial is promoted only after val-40 beats 12.01/16.98 on both medians
and on S_comb, and test medians stay within 20% of the cloud L=12 test
(15.12 / 16.79). Search-val does not promote.

  python -m qtimes_cost.scripts.cloud_day1_fork --workers 4
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
import csv
import json
import math
import multiprocessing as mp
import shutil
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from qtimes_cost.lib.promote import CLOUD_L12, fork_promote, ratio_9_6
from qtimes_cost.lib.train_loop import PKG
from qtimes_cost.scripts.cloud_day1_explore import (
    CSV_FIELDS,
    SEARCH,
    _append_csv,
    _row_from_result,
    _save_best,
    reval_states,
)
from qtimes_cost.lib.cloud_worker import run_trial as _worker

FORK_STEPS = 640
CONTINUE_STEPS = 1600
TEST_SLACK = 1.20


def _next_id(csv_path: Path) -> int:
    if not csv_path.exists():
        return 0
    last = 0
    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            try:
                last = max(last, int(row["trial_id"]) + 1)
            except (KeyError, ValueError):
                continue
    return last


def build_fork_trials(start_id: int, init_dir: str) -> list[dict]:
    """27 cells. The parent hyperparams get the long continuation, not a short retry."""
    common = {
        "seed": 0, "H": 128, "rounds": 4, "alpha0": "learned",
        "init_dir": init_dir, "src_L": 12,
    }
    rows = [{
        **common,
        "L": 12, "lr": 5e-5, "leakpen": 0.1, "tag": "best_continue",
        "steps": CONTINUE_STEPS,
        "patience_evals": 10, "min_steps": 640, "eval_every": 40,
        "hopeless_after": 10**9,
    }]
    for L in (16, 14, 12):
        for lr in (5e-5, 1e-4, 3e-5):
            for lp in (0.1, 0.05, 0.2):
                if L == 12 and abs(lr - 5e-5) < 1e-15 and abs(lp - 0.1) < 1e-15:
                    continue
                rows.append({
                    **common,
                    "L": L, "lr": lr, "leakpen": lp, "tag": "fork",
                    "steps": FORK_STEPS,
                    "patience_evals": 6, "min_steps": 280, "eval_every": 40,
                    "hopeless_after": 320, "hopeless_med9": 80.0, "hopeless_S": 5.2,
                })
    for i, row in enumerate(rows):
        row["trial_id"] = start_id + i
    return rows


def _caps() -> dict:
    return {
        "med6": CLOUD_L12["test_med6"] * TEST_SLACK,
        "med9": CLOUD_L12["test_med9"] * TEST_SLACK,
    }


def _champ0() -> dict:
    return {
        "med6": CLOUD_L12["val40_med6"],
        "med9": CLOUD_L12["val40_med9"],
        "s_comb": CLOUD_L12["val40_S"],
        "trial_id": 0,
    }


def _append_report(root: Path, rows: list[dict], promoted: list[dict], champ: dict) -> None:
    lines = []
    lines.append("")
    lines.append("## Fork from L=12 BEST")
    lines.append("")
    lines.append(
        "Warm-started from trial 0 (L=12, leakpen=0.1, lr=5e-5, learned α0). "
        "Grid is L∈{12,14,16} × lr∈{3e-5,5e-5,1e-4} × leakpen∈{0.05,0.1,0.2}. "
        "No L≈10. The parent cell is `best_continue` (more Adam steps, patience 10 evals). "
        "Other cells are shorter forks. Promote only if val-40 beats "
        f"**{CLOUD_L12['val40_med6']:.2f} / {CLOUD_L12['val40_med9']:.2f}** on both medians and on S_comb, "
        f"and test medians stay ≤ {TEST_SLACK:.2f}× the cloud test "
        f"({CLOUD_L12['test_med6']:.2f} / {CLOUD_L12['test_med9']:.2f})."
    )
    lines.append("")
    lines.append("| id | tag | L | lp | lr | steps | status | search n6/n9 | val40 n6/n9 | val40 S | test n6/n9 | decision |")
    lines.append("|---:|---|---:|---:|---:|---:|---|---|---|---:|---|---|")
    for r in rows:
        lines.append(
            "| {tid} | {tag} | {L} | {lp} | {lr} | {steps} | {status} | {s6}/{s9} | {v6}/{v9} | {vs} | {t6}/{t9} | {why} |".format(
                tid=r["trial_id"], tag=r["tag"], L=r["L"], lp=r["leakpen"], lr=r["lr"],
                steps=r["steps_ran"], status=r["status"],
                s6=_f(r.get("search_med6")), s9=_f(r.get("search_med9")),
                v6=_f(r.get("val40_med6")), v9=_f(r.get("val40_med9")),
                vs=_f(r.get("val40_S"), 3),
                t6=_f(r.get("test_med6")), t9=_f(r.get("test_med9")),
                why=r.get("decision") or "",
            )
        )
    lines.append("")
    if promoted:
        last = promoted[-1]
        lines.append(
            f"Promoted trial **{last['trial_id']}** "
            f"(L={last['L']}, leakpen={last['leakpen']}, lr={last['lr']}): "
            f"val-40 {last['val40_med6']:.2f} / {last['val40_med9']:.2f}, "
            f"test {last['test_med6']:.2f} / {last['test_med9']:.2f}."
        )
    else:
        lines.append(
            f"No fork beat val-40 {champ['med6']:.2f} / {champ['med9']:.2f} "
            "without pushing the test past the guard. Trial 0 stays BEST."
        )
    lines.append("")
    path = root / "REPORT.md"
    prev = path.read_text() if path.exists() else ""
    if "## Fork from L=12 BEST" in prev:
        prev = prev.split("## Fork from L=12 BEST")[0].rstrip() + "\n"
    path.write_text(prev + "\n".join(lines))


def _f(x, nd=1):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return ""
    if not math.isfinite(v):
        return ""
    return f"{v:.{nd}f}"


def _run(args) -> None:
    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=True)
    # Freeze the parent weights. Later promotes overwrite BEST/ and must not
    # change the warm-start the other workers load.
    round1 = root / "BEST_round1"
    if not round1.exists():
        shutil.copytree(Path(args.init), round1)
    init_dir = str(round1.resolve())
    csv_path = root / "trials.csv"
    start = _next_id(csv_path)
    trials = build_fork_trials(start, init_dir)
    if args.limit:
        trials = trials[: args.limit]

    cpus = os.cpu_count() or 1
    workers = args.workers or min(8, cpus)
    workers = max(1, min(workers, len(trials), cpus))
    search = dict(SEARCH)
    search["kill_after"] = 10**9
    print(
        f"fork: {len(trials)} trials from id {start}, workers={workers}, init={init_dir}",
        flush=True,
    )
    champ = _champ0()
    caps = _caps()
    promoted = []
    logged = []
    ctx = mp.get_context("spawn")
    with ctx.Manager() as manager, ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
        best_s = manager.dict()
        best_s["s"] = champ["s_comb"]
        best_lock = manager.Lock()
        payloads = []
        for trial in trials:
            payloads.append({
                "trial": trial,
                "best_s": best_s,
                "best_lock": best_lock,
                "steps": int(trial["steps"]),
                "search": search,
            })
        futs = {pool.submit(_worker, p): p["trial"] for p in payloads}
        for fut in as_completed(futs):
            trial = futs[fut]
            try:
                result = fut.result()
            except Exception:
                result = {
                    "trial_id": trial["trial_id"], "cfg": trial, "arch": {},
                    "status": "error", "steps_ran": 0, "best_step": -1, "wall_s": 0.0,
                    "metrics": {}, "s_comb": float("nan"), "states": {},
                    "error": traceback.format_exc(limit=6),
                }
            decision = result.get("status") or ""
            val40 = test = None
            if result.get("metrics") and math.isfinite(float(result.get("s_comb", float("nan")))):
                print(f"reval t{result['trial_id']} val=40 test=100", flush=True)
                rev = reval_states(
                    result["arch"], int(result["cfg"]["L"]), result["states"],
                    int(args.max_val), int(args.max_test),
                )
                val40 = rev["val"]["gate"]
                test = rev["test"]["gate"]
                keep, why = fork_promote(val40, test, champ, test_caps=caps)
                decision = why
                if keep:
                    champ = {
                        "med6": val40["med6"], "med9": val40["med9"],
                        "s_comb": val40["s_comb"], "trial_id": result["trial_id"],
                    }
                    _save_best(root, result, why, kind="fork_reval")
                    promoted.append({
                        "trial_id": result["trial_id"],
                        "L": result["cfg"]["L"], "leakpen": result["cfg"]["leakpen"],
                        "lr": result["cfg"]["lr"],
                        "val40_med6": val40["med6"], "val40_med9": val40["med9"],
                        "test_med6": test["med6"], "test_med9": test["med9"],
                    })
                    print(
                        f"PROMOTE t{result['trial_id']} val40 {val40['med6']:.2f}/{val40['med9']:.2f} "
                        f"test {test['med6']:.2f}/{test['med9']:.2f}",
                        flush=True,
                    )
            elif result.get("error"):
                decision = "error"
                err_path = root / "errors" / f"t{trial['trial_id']}.txt"
                err_path.parent.mkdir(parents=True, exist_ok=True)
                err_path.write_text(result["error"])
            metrics = result.get("metrics") or {}
            m6 = metrics.get(6) or {}
            m9 = metrics.get(9) or {}
            row_log = {
                "trial_id": result["trial_id"],
                "tag": trial.get("tag"),
                "L": trial.get("L"),
                "leakpen": trial.get("leakpen"),
                "lr": trial.get("lr"),
                "steps_ran": result.get("steps_ran"),
                "status": result.get("status"),
                "search_med6": m6.get("median_inv_p", ""),
                "search_med9": m9.get("median_inv_p", ""),
                "val40_med6": None if val40 is None else val40["med6"],
                "val40_med9": None if val40 is None else val40["med9"],
                "val40_S": None if val40 is None else val40["s_comb"],
                "test_med6": None if test is None else test["med6"],
                "test_med9": None if test is None else test["med9"],
                "test_ratio": None if test is None else ratio_9_6(test["med6"], test["med9"]),
                "decision": decision,
            }
            logged.append(row_log)
            csv_row = _row_from_result(
                result, decision == "promote", decision,
                float(champ["s_comb"]),
            )
            _append_csv(csv_path, csv_row)
            print(
                f"done t{result['trial_id']} {decision} wall={float(result.get('wall_s') or 0):.1f}s",
                flush=True,
            )
    logged.sort(key=lambda r: int(r["trial_id"]))
    (root / "fork_reval.json").write_text(json.dumps({
        "parent": CLOUD_L12,
        "test_caps": caps,
        "promoted": promoted,
        "trials": logged,
        "alpha0": "learned",
    }, indent=2))
    _append_report(root, logged, promoted, _champ0() if not promoted else champ)
    print(
        f"fork finished {len(logged)} trials, promotions={len(promoted)}",
        flush=True,
    )


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", type=str, default=str(PKG / "runs" / "cloud_day1"))
    ap.add_argument("--init", type=str, default=str(PKG / "runs" / "cloud_day1" / "BEST"))
    ap.add_argument("--max-val", type=int, default=40)
    ap.add_argument("--max-test", type=int, default=100)
    args = ap.parse_args(argv)
    t0 = time.time()
    _run(args)
    print(f"wall={time.time() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
