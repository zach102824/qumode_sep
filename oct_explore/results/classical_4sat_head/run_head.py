#!/usr/bin/env python3
"""Classical head-to-head on TEB 4-SAT test formulas (n=6,8,10).

Reuses sim.instances(..., TEB=52e9) — same Part1/2 test set.
Solvers: WalkSATlm (audited C), UBCSAT (walksat/novelty+/adaptnovelty+/saps/rsaps),
         CDCL via PySAT (Glucose4, Cadical195).
Units: tries (max_flips=n, restart) and lookups (= starts + flips).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]  # oct_explore
REPO = ROOT.parent
sys.path.insert(0, str(ROOT))
import sim

TEB = 52_000_000_000
NTEST = 100
NS = (6, 8, 10)
RUNS_WS = 500          # until-SAT meta-runs for WalkSATlm (match Part tables)
UBC_TRY_POOL = 2000    # independent cutoff=n tries per formula for UBCSAT
WSLM = str(REPO / "qaoa_ksat/walksatlm_c/wslm")
UBCSAT = str(HERE / "bin/ubcsat")
CNF_DIR = HERE / "cnf"
RAW_DIR = HERE / "raw"
CNF_DIR.mkdir(parents=True, exist_ok=True)
RAW_DIR.mkdir(parents=True, exist_ok=True)

# WalkSATlm paper-matched defaults (Zach-approved p=0.30)
WSLM_P, WSLM_W1, WSLM_W2 = "0.30", "6", "5"

# UBCSAT algorithms + light default / tuned knobs
# (wp / novnoise tuned lightly on VAB below if TUNED_PARAMS present)
UBC_ALGS = [
    ("walksat", []),
    ("novelty+", []),
    ("adaptnovelty+", []),
    ("saps", []),
    ("rsaps", []),
]


def mb(x):
    x = np.asarray(x, float)
    return [float(np.median(x)), float(np.quantile(x, 0.9))]


def export_cnfs(n: int):
    """Write DIMACS for each TEB test formula; return list of (seed, path, m)."""
    ins = sim.instances(n, NTEST, TEB)
    out = []
    for idx, i in enumerate(ins):
        seed = int(i["seed"])
        m, cl, E = sim.gen_instance(n, seed)
        assert E.min() == 0 and m == int(round(9.93 * n))
        path = CNF_DIR / f"n{n}_i{idx:03d}_s{seed}.cnf"
        if not path.exists():
            lines = [f"c TEB 4-SAT n={n} seed={seed} m={m}", f"p cnf {n} {m}"]
            for v, s in cl:
                lits = [int((vv + 1) * ss) for vv, ss in zip(v, s)]
                lines.append(" ".join(str(L) for L in lits) + " 0")
            path.write_text("\n".join(lines) + "\n")
        out.append((seed, path, m, cl))
    return out


def wslm_inst_line(n, cl, m):
    # same encoding as clean16/wsat.py
    return f"{n} 4 {m} " + " ".join(
        str((v + 1) * s) for vs, ss in zip(*zip(*cl)) for v, s in zip(vs, ss)
        # wait: zip(*cl) gives wrong structure — match wsat.py exactly
    )


def wslm_line_from_cl(n, m, cl):
    """Match clean16/wsat.py literal order."""
    parts = []
    for v, s in cl:
        for vv, ss in zip(v, s):
            parts.append(str((vv + 1) * ss))
    return f"{n} 4 {m} " + " ".join(parts)


def run_walksatlm(n: int, formulas, runs: int = RUNS_WS) -> dict:
    lines = []
    seeds = []
    for seed, path, m, cl in formulas:
        seeds.append(seed)
        line = wslm_line_from_cl(n, m, cl)
        for _ in range(runs):
            lines.append(line)
    inp = "\n".join(lines) + "\n"
    t0 = time.time()
    out = subprocess.run(
        [WSLM, "run", str(n), "1", str(12345 + n), WSLM_P, WSLM_W1, WSLM_W2],
        input=inp, capture_output=True, text=True, check=True,
    ).stdout
    A = np.loadtxt(out.splitlines(), ndmin=2).reshape(len(formulas), runs, 3)
    tries = A[..., 0].mean(1)
    flips = A[..., 1].mean(1)
    # wslm prints tries, flips, evals — lookups = tries + flips (start counted in evals;
    # clean16/wsat.py uses look = tries + flips)
    look = tries + flips
    res = {
        "algo": "WalkSATlm",
        "source": "wslm.c",
        "params": {"p": float(WSLM_P), "w1": int(WSLM_W1), "w2": int(WSLM_W2), "max_flips": n, "restart": 1},
        "n": n,
        "runs_per_formula": runs,
        "seeds": seeds,
        "tries": tries.tolist(),
        "lookups": look.tolist(),
        "flips": flips.tolist(),
        "tries_mb": mb(tries),
        "lookups_mb": mb(look),
        "wall_s": time.time() - t0,
    }
    return res


def _ubcsat_one(args):
    """Run UBCSAT on one formula: pool of cutoff=n tries → mean tries/lookups."""
    algo, extra, n, seed, cnf_path, pool, base_seed = args
    out_path = RAW_DIR / f"ubc_{algo.replace('+','p')}_n{n}_s{seed}.txt"
    cmd = [
        UBCSAT, "-alg", algo, *extra,
        "-i", str(cnf_path),
        "-runs", str(pool),
        "-cutoff", str(n),
        "-seed", str(base_seed + seed % 1000003),
        "-r", "out", str(out_path), "run,found,steps",
        "-r", "stats", "null",
    ]
    # suppress default stats noise
    t0 = time.time()
    proc = subprocess.run(cmd, capture_output=True, text=True)
    wall = time.time() - t0
    if proc.returncode != 0 and not out_path.exists():
        return {
            "seed": seed, "ok": False, "err": proc.stderr[-500:], "wall_s": wall,
            "mean_tries": float("nan"), "mean_lookups": float("nan"),
            "p_success": 0.0, "n_success": 0,
        }
    found, steps = [], []
    for line in out_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 3:
            continue
        try:
            found.append(int(parts[1]))
            steps.append(int(parts[2]))
        except ValueError:
            continue
    found = np.asarray(found, int)
    steps = np.asarray(steps, int)
    if len(found) == 0:
        return {
            "seed": seed, "ok": False, "err": "no rows", "wall_s": wall,
            "mean_tries": float("nan"), "mean_lookups": float("nan"),
            "p_success": 0.0, "n_success": 0,
        }
    succ = found == 1
    n_s = int(succ.sum())
    n_f = int((~succ).sum())
    p = n_s / len(found)
    if n_s == 0:
        # never solved in pool — lower-bound tries by pool size
        mean_tries = float(pool)  # conservative lower bound proxy
        mean_lookups = float(pool) * (1 + n)  # all failed full cutoff
        return {
            "seed": seed, "ok": True, "wall_s": wall,
            "mean_tries": mean_tries, "mean_lookups": mean_lookups,
            "p_success": 0.0, "n_success": 0, "censored": True,
        }
    # E[tries] = 1/p ; E[lookups] = ((1-p)/p)*(1+E[steps|fail]) + (1+E[steps|ok])
    e_fail = float(steps[~succ].mean()) if n_f else float(n)
    e_ok = float(steps[succ].mean())
    mean_tries = 1.0 / p
    mean_lookups = ((1 - p) / p) * (1.0 + e_fail) + (1.0 + e_ok)
    return {
        "seed": seed, "ok": True, "wall_s": wall,
        "mean_tries": mean_tries, "mean_lookups": mean_lookups,
        "p_success": float(p), "n_success": n_s,
        "e_steps_fail": e_fail, "e_steps_ok": e_ok, "censored": False,
    }


def run_ubcsat(n, formulas, algo, extra, pool=UBC_TRY_POOL, workers=4) -> dict:
    jobs = []
    for seed, path, m, cl in formulas:
        jobs.append((algo, extra, n, seed, path, pool, 900000 + n * 1000))
    rows = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_ubcsat_one, j) for j in jobs]
        for f in as_completed(futs):
            rows.append(f.result())
    rows.sort(key=lambda r: r["seed"])
    # align to formula seed order
    by = {r["seed"]: r for r in rows}
    ordered = [by[s] for s, *_ in formulas]
    tries = np.array([r["mean_tries"] for r in ordered], float)
    looks = np.array([r["mean_lookups"] for r in ordered], float)
    return {
        "algo": f"UBCSAT:{algo}",
        "source": "ubcsat (dtompkins, LP64-patched)",
        "params": {"alg": algo, "extra": extra, "cutoff": n, "try_pool": pool},
        "n": n,
        "seeds": [s for s, *_ in formulas],
        "tries": tries.tolist(),
        "lookups": looks.tolist(),
        "per_formula": ordered,
        "tries_mb": mb(tries),
        "lookups_mb": mb(looks),
        "n_censored": int(sum(1 for r in ordered if r.get("censored"))),
        "wall_s": time.time() - t0,
    }


def run_cdcl(n, formulas, solver_name="glucose4") -> dict:
    from pysat.formula import CNF
    from pysat.solvers import Glucose4, Cadical195

    Solver = {"glucose4": Glucose4, "cadical": Cadical195}[solver_name]
    conflicts, decisions, propagations, walls, statuses = [], [], [], [], []
    t_all = time.time()
    for seed, path, m, cl in formulas:
        cnf = CNF(from_file=str(path))
        t0 = time.time()
        with Solver(bootstrap_with=cnf.clauses, use_timer=True) as s:
            sat = s.solve()
            stats = s.accum_stats() or {}
            conflicts.append(int(stats.get("conflicts", -1)))
            decisions.append(int(stats.get("decisions", -1)))
            propagations.append(int(stats.get("propagations", -1)))
            statuses.append(bool(sat))
        walls.append(time.time() - t0)
    conflicts = np.asarray(conflicts, float)
    decisions = np.asarray(decisions, float)
    propagations = np.asarray(propagations, float)
    walls = np.asarray(walls, float)
    return {
        "algo": f"CDCL:{solver_name}",
        "source": "pysat",
        "n": n,
        "unit_note": "CDCL conflicts/decisions are NOT SLS tries; listed for completeness (paper compared Glucose/MapleSAT).",
        "seeds": [s for s, *_ in formulas],
        "sat_all": all(statuses),
        "conflicts": conflicts.tolist(),
        "decisions": decisions.tolist(),
        "propagations": propagations.tolist(),
        "wall_s_per": walls.tolist(),
        "conflicts_mb": mb(conflicts),
        "decisions_mb": mb(decisions),
        "propagations_mb": mb(propagations),
        "wall_ms_mb": mb(walls * 1000),
        "wall_s": time.time() - t_all,
    }


def light_tune_ubc(n_tune=8, n_val=30, pool=400):
    """Tiny VAB grid for walksat -wp and novelty+ -novnoise; return extra-arg map."""
    print("[tune] building small VAB set...", flush=True)
    # use val stream via sim.instances with VAB
    VAB = 53_000_000_000
    ins = sim.instances(n_tune, n_val, VAB)
    formulas = []
    for idx, i in enumerate(ins):
        seed = int(i["seed"])
        m, cl, E = sim.gen_instance(n_tune, seed)
        assert E.min() == 0
        path = CNF_DIR / f"val_n{n_tune}_i{idx:03d}_s{seed}.cnf"
        if not path.exists():
            lines = [f"c VAB n={n_tune} seed={seed}", f"p cnf {n_tune} {m}"]
            for v, s in cl:
                lits = [int((vv + 1) * ss) for vv, ss in zip(v, s)]
                lines.append(" ".join(str(L) for L in lits) + " 0")
            path.write_text("\n".join(lines) + "\n")
        formulas.append((seed, path, m, cl))

    tuned = {}
    # walksat wp grid
    best, best_med = None, 1e99
    for wp in (0.3, 0.4, 0.5, 0.6):
        r = run_ubcsat(n_tune, formulas, "walksat", ["-wp", str(wp)], pool=pool, workers=4)
        med = r["tries_mb"][0]
        print(f"  walksat wp={wp}: tries med={med:.3f}", flush=True)
        if med < best_med:
            best_med, best = med, ["-wp", str(wp)]
    tuned["walksat"] = best
    # novelty+ noise grid
    best, best_med = None, 1e99
    for nv in (0.3, 0.5, 0.7):
        r = run_ubcsat(n_tune, formulas, "novelty+", ["-novnoise", str(nv)], pool=pool, workers=4)
        med = r["tries_mb"][0]
        print(f"  novelty+ novnoise={nv}: tries med={med:.3f}", flush=True)
        if med < best_med:
            best_med, best = med, ["-novnoise", str(nv)]
    tuned["novelty+"] = best
    # adaptnovelty+ / saps / rsaps: keep defaults (empty extras)
    tuned["adaptnovelty+"] = []
    tuned["saps"] = []
    tuned["rsaps"] = []
    (HERE / "tuned_params.json").write_text(json.dumps(tuned, indent=2))
    print("[tune] chose", tuned, flush=True)
    return tuned


def main():
    assert Path(WSLM).is_file(), WSLM
    assert Path(UBCSAT).is_file(), UBCSAT

    # export all CNFs first
    all_f = {}
    for n in NS:
        print(f"[export] n={n}", flush=True)
        all_f[n] = export_cnfs(n)
        print(f"  {len(all_f[n])} formulas, first seeds {[s for s, *_ in all_f[n][:3]]}", flush=True)

    # light tune
    tuned_path = HERE / "tuned_params.json"
    if tuned_path.exists():
        tuned = json.load(open(tuned_path))
        print("[tune] loaded", tuned, flush=True)
    else:
        tuned = light_tune_ubc()

    results = []

    # WalkSATlm
    for n in NS:
        print(f"[WalkSATlm] n={n}", flush=True)
        r = run_walksatlm(n, all_f[n])
        print(f"  tries {r['tries_mb']} lookups {r['lookups_mb']} ({r['wall_s']:.1f}s)", flush=True)
        results.append(r)
        json.dump(r, open(RAW_DIR / f"wslm_n{n}.json", "w"))

    # UBCSAT
    for algo, _ in UBC_ALGS:
        extra = tuned.get(algo, [])
        for n in NS:
            print(f"[UBCSAT {algo} {extra}] n={n}", flush=True)
            r = run_ubcsat(n, all_f[n], algo, extra, workers=4)
            print(f"  tries {r['tries_mb']} lookups {r['lookups_mb']} censored={r['n_censored']} ({r['wall_s']:.1f}s)", flush=True)
            results.append(r)
            tag = algo.replace("+", "p")
            json.dump(r, open(RAW_DIR / f"ubc_{tag}_n{n}.json", "w"))

    # CDCL
    for solver in ("glucose4", "cadical"):
        for n in NS:
            print(f"[CDCL {solver}] n={n}", flush=True)
            r = run_cdcl(n, all_f[n], solver)
            print(f"  conflicts {r['conflicts_mb']} decisions {r['decisions_mb']} ({r['wall_s']:.1f}s)", flush=True)
            results.append(r)
            json.dump(r, open(RAW_DIR / f"cdcl_{solver}_n{n}.json", "w"))

    json.dump(results, open(HERE / "all_results.json", "w"))
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
