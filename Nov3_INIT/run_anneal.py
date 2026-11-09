"""N=15 L=3 V4 Spall vs asymptotically optimal anneal vs QAOA. Q=3072."""
from __future__ import annotations

import hashlib
import json
import os
import resource
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import numpy as np

from Nov3_INIT.alloc import (
    ANNEAL_ALPHA,
    ANNEAL_GAMMA,
    METHOD_ID,
    make_rng,
    run_equal4,
    x0_center4,
    x0_qaoa_center4,
)
from Nov3_INIT.bstar import maximize
from Nov3_INIT.circuit import Logical
from Nov3_INIT.run_v1 import _dump
from Nov9_SK_SPSA.experiments.FT.instance import make_J

ROOT = Path(__file__).resolve().parent
PREREG = ROOT / "PREREG_ANNEAL.md"
RES = ROOT / "results" / "anneal"
CELLS = RES / "cells"
INST = RES / "instances"
CONF = RES / "CONFORMANCE.json"
BAND = list(range(10456, 10464))
N = 15
L = 3
Q = 3072
N_BOOT = 4000
HISTORICAL = 2498560
CELL_WALL_CAP = 1800.0
CUT = list(range(7))
ARMS = (
    ("V4", "Nov3rd", METHOD_ID["equal4"]["Nov3rd"], None, None),
    ("ANNEAL", "Nov3rd", METHOD_ID["equal4"]["ANNEAL"], ANNEAL_ALPHA, ANNEAL_GAMMA),
    ("qaoa_center4", "QAOA", METHOD_ID["equal4"]["QAOA"], None, None),
)


def prereg_sha() -> str:
    return hashlib.sha256(PREREG.read_bytes()).hexdigest()


def allow(seed: int) -> None:
    seed = int(seed)
    if seed not in BAND:
        raise RuntimeError(f"seed {seed} is not in the anneal band 10456..10463")
    if 10432 <= seed <= 10439:
        raise RuntimeError(f"LE seed {seed}")
    if 10440 <= seed <= 10455:
        raise RuntimeError(f"local R27 seed {seed}")
    if 10416 <= seed <= 10431:
        raise RuntimeError(f"local R26 seed {seed}")
    if seed <= 10415:
        raise RuntimeError(f"prior seed {seed}")
    if 10144 <= seed <= 10151:
        raise RuntimeError(f"holdout seed {seed}")
    if 11000 <= seed <= 11031 or 20000 <= seed <= 20999:
        raise RuntimeError(f"blind seed {seed}")


def _conf() -> dict:
    doc = json.loads(CONF.read_text())
    if doc.get("prereg_sha256") != prereg_sha() or not doc.get("ok"):
        raise RuntimeError("anneal conformance does not match the current preregistration")
    if doc.get("collisions_in_nov3_init_results", {}).get("n", 1) != 0:
        raise RuntimeError("anneal seeds already appear under Nov3_INIT/results")
    return doc


def _bstar_worker(seed: int) -> dict:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    allow(int(seed))
    t0 = time.perf_counter()
    cpu0 = time.process_time()
    rec = maximize(make_J(int(seed), N), N)
    gs = rec.pop("gs")
    rec.pop("hc")
    rec.update({
        "N": N,
        "seed": int(seed),
        "seconds": time.perf_counter() - t0,
        "cpu_seconds": time.process_time() - cpu0,
        "peak_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "prereg_sha256": prereg_sha(),
        "classical_not_a_quantum_query": True,
        "anneal_band": True,
    })
    if rec.get("cut") != CUT:
        raise RuntimeError(f"unexpected cut {rec.get('cut')}")
    return {"rec": rec, "gs": gs}


def build_instances() -> None:
    _conf()
    pending = [s for s in BAND if not (INST / f"N{N}_s{s}.json").exists()]
    for s in BAND:
        allow(s)
    if not pending:
        return
    import multiprocessing as mp

    INST.mkdir(parents=True, exist_ok=True)
    with mp.get_context("fork").Pool(4) as pool:
        for got in pool.imap_unordered(_bstar_worker, pending, chunksize=1):
            seed = got["rec"]["seed"]
            np.savez(INST / f"N{N}_s{seed}.npz", gs=got["gs"])
            _dump(INST / f"N{N}_s{seed}.json", got["rec"])
            r = got["rec"]
            print(f"B* N=15 seed={seed} b={r['Bstar']:.6f} gap={r['gap']:.4g} {r['seconds']:.1f}s", flush=True)


def _logical(seed: int):
    meta = json.loads((INST / f"N{N}_s{seed}.json").read_text())
    gs = np.load(INST / f"N{N}_s{seed}.npz")["gs"]
    if int(meta["N"]) != N or int(meta["seed"]) != int(seed) or gs.size != (1 << N) or meta.get("cut") != CUT:
        raise RuntimeError(f"B* invalid seed {seed}")
    return meta, Logical(N, seed, float(meta["Bstar"]), gs)


def _path(arm: str, method: str, seed: int) -> Path:
    return CELLS / f"{arm}_{method}_N{N}_s{seed}_L{L}_q1024.json"


def _one(task: dict) -> dict:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    arm, method = task["arm"], task["method"]
    seed, mid = int(task["seed"]), int(task["method_id"])
    alpha, gamma = task["alpha"], task["gamma"]
    allow(seed)
    path = _path(arm, method, seed)
    if path.exists():
        return {"skipped": True}
    t0 = time.perf_counter()
    cpu0 = time.process_time()
    try:
        meta, logical = _logical(seed)
        b = float(meta["Bstar"])
        rng_init = [None if s == 0 else make_rng(seed, N, b, L, mid, s, 2) for s in range(4)]
        rng_spsa = [make_rng(seed, N, b, L, mid, s, 0) for s in range(4)]

        def draw(method_name, ell, start_id, rng, _method=method):
            if _method == "QAOA":
                return x0_qaoa_center4(ell, start_id, rng)
            return x0_center4(method_name, ell, start_id, rng)

        def propagate(x, _logical=logical, _method=method, _t0=t0):
            if time.perf_counter() - _t0 > CELL_WALL_CAP:
                raise RuntimeError(f"cell wall exceeded {CELL_WALL_CAP}s")
            return _logical.propagate(_method, L, x)

        out = run_equal4(
            propagate, method, L, Q, rng_init, rng_spsa, x0_fn=draw, alpha=alpha, gamma=gamma
        )
        psi = out.pop("aux")
        f = logical.fidelity(psi)
        f_plus = float(meta["F_plus"])
        e0 = float(meta["E0"])
        record = {
            "ok": True,
            "failure": None,
            "blocker": None,
            "version": "N3INIT_anneal",
            "arm": arm,
            "method": method,
            "method_id": mid,
            "alpha": out.get("alpha"),
            "gamma": out.get("gamma"),
            "A": out.get("A"),
            "N": N,
            "seed": seed,
            "L": L,
            "q": 1024,
            "Q": Q,
            "procedural_cap": Q,
            "Bstar": b,
            "E0": e0,
            "E_plus": meta["E_plus"],
            "F_plus": f_plus,
            "E": out["E"],
            "E_error_over_N": float((out["E"] - e0) / float(N)),
            "F": f,
            "eta": float((f - f_plus) / (1.0 - f_plus)),
            "escaped": bool(out["E"] < meta["E_plus"] - 1e-3 * N),
            "which": out["which"],
            "start_id": int(out["start_id"]),
            "n_iter": int(out["n_iter"]),
            "tails": out["tails"],
            "queries": out["queries"],
            "x": [float(v) for v in out["x"]],
            "seconds": time.perf_counter() - t0,
            "cpu_seconds": time.process_time() - cpu0,
            "peak_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
            "prereg_sha256": task["prereg_sha256"],
            "fidelity_posthoc": True,
            "shared_uc_only": True,
        }
    except Exception as exc:
        kind = "wall" if "wall exceeded" in repr(exc) else "cell_infeasible"
        record = {
            "ok": False,
            "failure": repr(exc),
            "blocker": kind,
            "version": "N3INIT_anneal",
            "arm": arm,
            "method": method,
            "N": N,
            "seed": seed,
            "L": L,
            "q": 1024,
            "Q": Q,
            "seconds": time.perf_counter() - t0,
            "cpu_seconds": time.process_time() - cpu0,
            "peak_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
            "prereg_sha256": task["prereg_sha256"],
            "physics_not_approximated": True,
        }
    _dump(path, record)
    extra = f" E={record['E']:.4f} F={record['F']:.3f}" if record["ok"] else " " + record["failure"]
    print(f"{'ok' if record['ok'] else 'FAIL'} anneal {arm} N=15 s={seed} L=3 q=1024{extra} {record['seconds']:.1f}s", flush=True)
    return record


def run_fits() -> None:
    import multiprocessing as mp

    _conf()
    sha = prereg_sha()
    pending = []
    for seed in BAND:
        for arm, method, mid, alpha, gamma in ARMS:
            if not _path(arm, method, seed).exists():
                pending.append({
                    "arm": arm,
                    "method": method,
                    "seed": seed,
                    "method_id": mid,
                    "alpha": alpha,
                    "gamma": gamma,
                    "prereg_sha256": sha,
                })
    print(f"anneal pending {len(pending)}", flush=True)
    wall0 = time.perf_counter()
    if pending:
        with mp.get_context("fork").Pool(4) as pool:
            list(pool.imap_unordered(_one, pending, chunksize=1))
    (RES / "wall.json").write_text(json.dumps({"fit_seconds": time.perf_counter() - wall0, "workers": 4, "blas_threads": 1}) + "\n")


def _boot(values: np.ndarray, rng: np.random.Generator) -> dict:
    values = np.asarray(values, dtype=np.float64)
    n = values.shape[0]
    draws = rng.integers(0, n, size=(N_BOOT, n))
    means = values[draws].mean(axis=1)
    return {
        "n": int(n),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "min": float(values.min()),
        "max": float(values.max()),
        "ci95": [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))],
    }


def _arm(ok, name):
    return sorted([r for r in ok if r["arm"] == name], key=lambda r: r["seed"])


def summarize() -> None:
    rows = [json.loads(p.read_text()) for p in CELLS.glob("*.json")]
    fails = [r for r in rows if not r.get("ok")]
    ok = [r for r in rows if r.get("ok")]
    inst = [json.loads(p.read_text()) for p in INST.glob("N15_s*.json")]
    rng = np.random.default_rng(3334)
    v4, an, qa = _arm(ok, "V4"), _arm(ok, "ANNEAL"), _arm(ok, "qaoa_center4")
    complete = len(v4) == len(an) == len(qa) == 8
    stats = None
    primary_ok = secondary_ok = False
    if complete:
        e_v = np.array([r["E"] for r in v4])
        e_a = np.array([r["E"] for r in an])
        e_q = np.array([r["E"] for r in qa])
        d_aq = _boot(e_a - e_q, rng)
        primary_ok = bool(d_aq["mean"] < 0.0 and d_aq["ci95"][1] < 0.0)
        secondary_ok = bool(float(e_a.mean()) <= float(e_v.mean()))
        stats = {
            "d_anneal_qaoa": d_aq,
            "d_v4_qaoa": _boot(e_v - e_q, rng),
            "d_anneal_v4": _boot(e_a - e_v, rng),
            "wins_anneal_qaoa": int(np.sum(e_a < e_q)),
            "E_V4": _boot(e_v, rng),
            "E_ANNEAL": _boot(e_a, rng),
            "E_QAOA": _boot(e_q, rng),
            "F_V4": _boot(np.array([r["F"] for r in v4]), rng),
            "F_ANNEAL": _boot(np.array([r["F"] for r in an]), rng),
            "F_QAOA": _boot(np.array([r["F"] for r in qa]), rng),
            "start0": {
                "V4": int(sum(r["start_id"] == 0 for r in v4)),
                "ANNEAL": int(sum(r["start_id"] == 0 for r in an)),
                "QAOA": int(sum(r["start_id"] == 0 for r in qa)),
            },
            "seeds": [int(r["seed"]) for r in an],
            "primary_ci_lt0": primary_ok,
            "secondary_anneal_le_v4": secondary_ok,
        }
    if fails or not complete:
        label = "REJECT_ANNEAL_L3"
    elif primary_ok and secondary_ok:
        label = "ANNEAL_L3_EDGE"
    else:
        label = "REJECT_ANNEAL_L3"
    new_q = int(sum(r["queries"]["total"] for r in ok))
    payload = {
        "version": "N3INIT_anneal",
        "no_gr_patch": True,
        "no_candidacy_file": True,
        "holdout_fail_stands": True,
        "no_refit_10368_10375": True,
        "n_files": len(rows),
        "n_fail": len(fails),
        "failures": [{"arm": r["arm"], "seed": r["seed"], "failure": r["failure"]} for r in fails],
        "new_queries": new_q,
        "historical_queries_spent_once": HISTORICAL,
        "total_research_queries": HISTORICAL + new_q,
        "classical_bstar": {
            "n": len(inst),
            "wall_seconds": float(sum(r["seconds"] for r in inst)),
            "cpu_seconds": float(sum(r.get("cpu_seconds", 0.0) for r in inst)),
            "peak_rss_kib_max": int(max((r.get("peak_rss_kib", 0) for r in inst), default=0)),
            "not_a_quantum_query": True,
        },
        "label": label,
        "complete": complete,
        "stats": stats,
    }
    _dump(RES / "summary.json", payload)
    lines = [
        "# N3INIT L=3 annealed SPSA N=15",
        "",
        "Original shared-Uc Nov3. V4 is Spall 0.602/0.101. ANNEAL is α=1, γ=1/6. Seeds 10456–10463. Q=3072. No GR patch. No re-fit of 10368–75.",
        "",
        f"**{label}.**",
        "",
        f"New quantum queries {new_q}. Historical {HISTORICAL}. Total research queries {HISTORICAL + new_q}. Failures {len(fails)}.",
        f"Classical B* wall {payload['classical_bstar']['wall_seconds']:.1f}s, CPU {payload['classical_bstar']['cpu_seconds']:.1f}s, peak RSS {payload['classical_bstar']['peak_rss_kib_max']} KiB.",
        "",
    ]
    if stats:
        d = stats["d_anneal_qaoa"]
        dv = stats["d_v4_qaoa"]
        da = stats["d_anneal_v4"]
        lines += [
            f"Primary ANNEAL vs QAOA mean ΔE {d['mean']:+.4f} CI [{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}] wins {stats['wins_anneal_qaoa']}/8.",
            f"Context V4 vs QAOA mean ΔE {dv['mean']:+.4f} CI [{dv['ci95'][0]:+.3f}, {dv['ci95'][1]:+.3f}].",
            f"ANNEAL vs V4 mean ΔE {da['mean']:+.4f} CI [{da['ci95'][0]:+.3f}, {da['ci95'][1]:+.3f}].",
            f"mean E V4 {stats['E_V4']['mean']:.4f} ANNEAL {stats['E_ANNEAL']['mean']:.4f} QAOA {stats['E_QAOA']['mean']:.4f}.",
            f"mean F V4 {stats['F_V4']['mean']:.3f} ANNEAL {stats['F_ANNEAL']['mean']:.3f} QAOA {stats['F_QAOA']['mean']:.3f}.",
            f"start0 V4 {stats['start0']['V4']}/8 ANNEAL {stats['start0']['ANNEAL']}/8 QAOA {stats['start0']['QAOA']}/8.",
            f"Primary CI<0 {stats['primary_ci_lt0']}. Secondary ANNEAL≤V4 {stats['secondary_anneal_le_v4']}.",
            "",
        ]
    if fails:
        lines += ["## Blockers", ""]
        for r in fails:
            lines.append(f"- {r['arm']} seed={r['seed']}: {r['failure']}")
        lines.append("")
    lines.append("Fit workers 4, BLAS 1. Each completed ledger equals Q=3072.")
    (RES / "RESULTS.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", required=True, choices=("bstar", "run", "summary"))
    args = parser.parse_args()
    if not PREREG.exists():
        raise RuntimeError("anneal preregistration missing")
    _conf()
    if args.phase == "bstar":
        build_instances()
    elif args.phase == "run":
        run_fits()
    else:
        summarize()


if __name__ == "__main__":
    main()
