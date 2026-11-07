"""Unbounded random local search, aimed at QAOA where the shell looks better.

Differential evolution was confined to one fundamental domain and often missed
the bootstrap basin. These starts are unbounded.
"""
from __future__ import annotations

import json
import time

import thread_env  # noqa: F401
import numpy as np

from ansatz import Model, n_params
from common import dump_json
from optimize import Pool, polish_starts
from report import build_verdict, write_figures, write_results_md, write_summary_csv
from rescue import adopt
from run_round1 import RESULTS, zero_shot


def main():
    log_path = RESULTS / "blitz.log"

    def log(msg):
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        print(line, flush=True)
        with log_path.open("a") as fh:
            fh.write(line + "\n")

    targets = {}
    models = {}
    gs = {}
    for n in (6, 12):
        arrays = np.load(RESULTS / f"target_n{n}_arrays.npz")
        public = json.loads((RESULTS / f"target_n{n}.json").read_text())
        public["J"] = arrays["J"]
        public["hc_diag"] = arrays["hc"]
        public["gs"] = arrays["gs"]
        targets[n] = public
        models[n] = Model(n, public["J"], float(public["B"]), public["hc_diag"])
        gs[n] = np.asarray(public["gs"], dtype=np.complex128)

    records = {}
    for path in (RESULTS / "opt").glob("n*_L*_*.json"):
        row = json.loads(path.read_text())
        records[(row["n"], row["L"], row["method"])] = row

    jobs = []
    for L in range(2, 7):
        jobs.append((6, L, "qaoa", 180, 350))
        jobs.append((6, L, "nov3_D", 80, 350))
    for L in range(2, 7):
        jobs.append((12, L, "qaoa", 40, 200))
        jobs.append((12, L, "nov3_D", 40, 200))

    for n, L, method, n_random, maxfun in jobs:
        current = records[(n, L, method)]
        log(f"blitz N={n} L={L} {method} E={current['E']:.6f}")
        rng = np.random.default_rng(50_000 + n * 100 + L * 10 + (1 if method == "qaoa" else 2))
        width = n_params(method, L)
        base = np.asarray(current["x_raw"], dtype=np.float64)
        starts = [("keep", base)]
        for i in range(n_random):
            starts.append((f"wide_{i}", rng.normal(scale=0.6, size=width)))
        for i in range(max(10, n_random // 5)):
            starts.append((f"near_{i}", base + rng.normal(scale=0.25, size=width)))
        pool = Pool(models[n], 4)
        try:
            t0 = time.perf_counter()
            polished = polish_starts(pool, method, L, starts, maxfun=maxfun)
            polished["seconds"] = time.perf_counter() - t0
        finally:
            pool.close()
        updated = adopt(models[n], gs[n], targets[n], method, L, current, polished, log, "unbounded_blitz")
        records[(n, L, method)] = updated

    rows = [records[k] for k in sorted(records)]
    scale = float(targets[6]["kappa_mean_degree"]) / float(targets[12]["kappa_mean_degree"])
    transfer = []
    for L in range(1, 9):
        for method in ("qaoa", "shared_beta", "nov3_D", "drop_c2", "drop_c3"):
            src = records.get((6, L, method))
            if src is None or src.get("status") != "complete":
                continue
            transfer.extend(zero_shot(models[12], gs[12], targets[12], src["x_raw"], method, L, scale))
    old_notes = []
    if (RESULTS / "summary.json").exists():
        old_notes = list(json.loads((RESULTS / "summary.json").read_text()).get("notes") or [])
    notes = old_notes + [
        "Unbounded blitz: wide Gaussian starts and perturbations of the incumbent, accepted only on lower energy."
    ]
    verdict = build_verdict([r for r in rows if r.get("status") == "complete"])
    dump_json(RESULTS / "summary.json", {"rows": rows, "verdict": verdict, "notes": notes})
    dump_json(RESULTS / "verdict.json", verdict)
    dump_json(RESULTS / "transfer.json", transfer)
    write_summary_csv(RESULTS / "summary.csv", rows)
    loaded = {n: json.loads((RESULTS / f"target_n{n}.json").read_text()) for n in (6, 12)}
    write_figures(RESULTS / "figures", rows, loaded)
    write_results_md(RESULTS / "RESULTS.md", rows, loaded, verdict, transfer, notes)
    log(f"verdict program_success={verdict['program_success']}")
    log(verdict["headline"])


if __name__ == "__main__":
    main()
