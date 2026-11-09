"""N=18 L1 paired confirmation. Exact v4 four-start recipes. Gate first."""
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

from Nov3_INIT.alloc import make_rng, run_equal4, x0_center4, x0_qaoa_center4
from Nov3_INIT.bstar import maximize
from Nov3_INIT.circuit import Logical
from Nov3_INIT.run_v1 import TEST, TRAIN, _dump, allow
from Nov9_SK_SPSA.experiments.FT.instance import make_J

ROOT = Path(__file__).resolve().parent
PREREG = ROOT / "PREREG_V5.md"
RES = ROOT / "results" / "v5"
CELLS = RES / "cells"
INST = RES / "instances"
CONF = RES / "CONFORMANCE.json"
SEEDS = TRAIN + TEST
QS = (512, 1024)
N_BOOT = 4000
HISTORICAL = 1679360
PRIOR_N18 = ROOT / "results"


def prereg_sha() -> str:
    return hashlib.sha256(PREREG.read_bytes()).hexdigest()


def _conf() -> dict:
    doc = json.loads(CONF.read_text())
    if doc.get("prereg_sha256") != prereg_sha() or not doc.get("ok"):
        raise RuntimeError("v5 conformance does not match the current preregistration")
    if not doc.get("probe", {}).get("n18_allowed"):
        raise RuntimeError("N=18 resource gate closed")
    return doc


def _prior_n18(seed: int) -> None:
    hits = []
    for path in PRIOR_N18.rglob("N18_s*.json"):
        if path.parent.name != "instances":
            continue
        if path.name == f"N18_s{seed}.json" and "v5" not in path.parts:
            hits.append(str(path))
    if hits:
        raise RuntimeError(f"prior N=18 instance already exists: {hits}")


def _logical(seed: int):
    meta = json.loads((INST / f"N18_s{seed}.json").read_text())
    gs = np.load(INST / f"N18_s{seed}.npz")["gs"]
    return meta, Logical(18, seed, float(meta["Bstar"]), gs)


def _n18_worker(seed: int) -> dict:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    allow(int(seed))
    _prior_n18(int(seed))
    t0 = time.perf_counter()
    cpu0 = time.process_time()
    rec = maximize(make_J(int(seed), 18), 18)
    gs = rec.pop("gs")
    rec.pop("hc")
    rec.update({
        "N": 18,
        "seed": int(seed),
        "seconds": time.perf_counter() - t0,
        "cpu_seconds": time.process_time() - cpu0,
        "peak_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "prereg_sha256": prereg_sha(),
        "dense_matrix": False,
        "classical_not_a_quantum_query": True,
        "cut": rec.get("cut"),
    })
    if rec.get("cut") != list(range(9)):
        raise RuntimeError(f"unexpected N=18 cut {rec.get('cut')}")
    return {"rec": rec, "gs": gs}


def build_instances() -> None:
    _conf()
    pending = []
    for seed in SEEDS:
        allow(seed)
        _prior_n18(seed)
        if not (INST / f"N18_s{seed}.json").exists():
            pending.append(seed)
    if not pending:
        return
    import multiprocessing as mp

    INST.mkdir(parents=True, exist_ok=True)
    with mp.get_context("fork").Pool(4) as pool:
        for got in pool.imap_unordered(_n18_worker, pending, chunksize=1):
            seed = got["rec"]["seed"]
            np.savez(INST / f"N18_s{seed}.npz", gs=got["gs"])
            _dump(INST / f"N18_s{seed}.json", got["rec"])
            r = got["rec"]
            print(
                f"B* N=18 seed={seed} b={r['Bstar']:.6f} gap={r['gap']:.4g} "
                f"{r['seconds']:.1f}s rss={r['peak_rss_kib']}KiB",
                flush=True,
            )


def _path(arm: str, method: str, seed: int, q: int) -> Path:
    return CELLS / f"{arm}_{method}_N18_s{seed}_L1_q{q}.json"


def _one(task: dict) -> dict:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    arm, method = task["arm"], task["method"]
    seed, q = int(task["seed"]), int(task["q"])
    allow(seed)
    path = _path(arm, method, seed, q)
    if path.exists():
        return {"skipped": True}
    t0 = time.perf_counter()
    cpu0 = time.process_time()
    meta, logical = _logical(seed)
    b = float(meta["Bstar"])
    mid = 0 if method == "QAOA" else 1
    Q = q
    rng_init = [None if s == 0 else make_rng(seed, 18, b, 1, mid, s, 2) for s in range(4)]
    rng_spsa = [make_rng(seed, 18, b, 1, mid, s, 0) for s in range(4)]

    def draw(method_name, ell, start_id, rng, _arm=arm):
        if _arm == "qaoa_center4":
            return x0_qaoa_center4(ell, start_id, rng)
        return x0_center4(method_name, ell, start_id, rng)

    def propagate(x, _logical=logical, _method=method):
        return _logical.propagate(_method, 1, x)

    try:
        out = run_equal4(propagate, method, 1, Q, rng_init, rng_spsa, x0_fn=draw)
        psi = out.pop("aux")
        f = logical.fidelity(psi)
        f_plus = float(meta["F_plus"])
        e0 = float(meta["E0"])
        record = {
            "ok": True,
            "failure": None,
            "version": "N3INIT_v5",
            "candidate_not_benchmark_adoption": True,
            "arm": arm,
            "method": method,
            "method_id": mid,
            "N": 18,
            "seed": seed,
            "L": 1,
            "q": q,
            "Q": Q,
            "procedural_cap": Q,
            "Bstar": b,
            "E0": e0,
            "E_plus": meta["E_plus"],
            "F_plus": f_plus,
            "E": out["E"],
            "E_error_over_N": float((out["E"] - e0) / 18.0),
            "F": f,
            "eta": float((f - f_plus) / (1.0 - f_plus)),
            "escaped": bool(out["E"] < meta["E_plus"] - 1e-3 * 18),
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
            "development_not_population": True,
            "qaoa_center_is_derived": arm == "qaoa_center4",
            "fidelity_posthoc": True,
            "exact_v4_recipe": True,
        }
    except Exception as exc:
        record = {
            "ok": False,
            "failure": repr(exc),
            "version": "N3INIT_v5",
            "arm": arm,
            "method": method,
            "N": 18,
            "seed": seed,
            "L": 1,
            "q": q,
            "Q": q,
            "seconds": time.perf_counter() - t0,
            "cpu_seconds": time.process_time() - cpu0,
            "peak_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
            "prereg_sha256": task["prereg_sha256"],
        }
    _dump(path, record)
    extra = f" E={record['E']:.4f} F={record['F']:.3f}" if record["ok"] else " " + record["failure"]
    print(f"{'ok' if record['ok'] else 'FAIL'} v5 {arm} N=18 s={seed} q={q}{extra} {record['seconds']:.1f}s", flush=True)
    return record


def _pending() -> list[dict]:
    sha = prereg_sha()
    rows = []
    for seed in SEEDS:
        for q in QS:
            for arm, method in (("nov3_center4", "Nov3rd"), ("qaoa_center4", "QAOA")):
                if not _path(arm, method, seed, q).exists():
                    rows.append({"arm": arm, "method": method, "seed": seed, "q": q, "prereg_sha256": sha})
    return rows


def run_fits() -> None:
    import multiprocessing as mp

    _conf()
    wall0 = time.perf_counter()
    pending = _pending()
    print(f"v5 pending {len(pending)}", flush=True)
    if pending:
        with mp.get_context("fork").Pool(4) as pool:
            list(pool.imap_unordered(_one, pending, chunksize=1))
    (RES / "wall.json").write_text(
        json.dumps({"fit_seconds": time.perf_counter() - wall0, "workers": 4, "blas_threads": 1}) + "\n"
    )


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


def _flag_pair(stat: dict) -> str:
    lo, hi = stat["ci95"]
    if stat["mean"] < 0.0 and hi < 0.0:
        return "persists_ci_excludes_0"
    if stat["mean"] < 0.0:
        return "negative_point_ci_includes_0"
    if lo > 0.0:
        return "does_not_persist_ci_excludes_0"
    return "positive_point_ci_includes_0"


def summarize() -> None:
    rows = [json.loads(p.read_text()) for p in CELLS.glob("*.json")]
    fails = [r for r in rows if not r.get("ok")]
    ok = [r for r in rows if r.get("ok")]
    inst = [json.loads(p.read_text()) for p in INST.glob("N18_s*.json")]
    rng = np.random.default_rng(3334)
    groups = []
    for q in QS:
        nov = sorted([r for r in ok if r["arm"] == "nov3_center4" and r["q"] == q], key=lambda r: r["seed"])
        qa = sorted([r for r in ok if r["arm"] == "qaoa_center4" and r["q"] == q], key=lambda r: r["seed"])
        if not nov or not qa or len(nov) != len(qa):
            continue
        e_n = np.array([r["E"] for r in nov])
        e_q = np.array([r["E"] for r in qa])
        err_n = np.array([r["E_error_over_N"] for r in nov])
        err_q = np.array([r["E_error_over_N"] for r in qa])
        f_n = np.array([r["F"] for r in nov])
        f_q = np.array([r["F"] for r in qa])
        d_pair = _boot(e_n - e_q, rng)
        groups.append({
            "N": 18,
            "L": 1,
            "q": q,
            "Q": q,
            "n": len(nov),
            "d_pair": d_pair,
            "persists": _flag_pair(d_pair),
            "energy_wins": int(np.sum(e_n < e_q)),
            "energy_ties": int(np.sum(np.abs(e_n - e_q) <= 1e-8)),
            "E_nov3": _boot(e_n, rng),
            "E_qaoa": _boot(e_q, rng),
            "E_error_over_N_nov3": _boot(err_n, rng),
            "E_error_over_N_qaoa": _boot(err_q, rng),
            "F_nov3": _boot(f_n, rng),
            "F_qaoa": _boot(f_q, rng),
            "center_start0_selected": int(sum(r["start_id"] == 0 for r in nov)),
            "qaoa_start0_selected": int(sum(r["start_id"] == 0 for r in qa)),
            "queries": int(sum(r["queries"]["total"] for r in nov + qa)),
            "seconds": float(sum(r["seconds"] for r in nov + qa)),
            "cpu_seconds": float(sum(r["cpu_seconds"] for r in nov + qa)),
            "peak_rss_kib_max": int(max(r["peak_rss_kib"] for r in nov + qa)),
        })
    new_q = int(sum(r["queries"]["total"] for r in ok))
    both = [g["persists"] == "persists_ci_excludes_0" for g in groups]
    payload = {
        "version": "N3INIT_v5",
        "candidate_not_benchmark_adoption": True,
        "n_files": len(rows),
        "n_fail": len(fails),
        "failures": [{"arm": r["arm"], "seed": r["seed"], "q": r["q"], "failure": r["failure"]} for r in fails],
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
        "n15_shallow_edge_persists_at_both_q": bool(len(groups) == 2 and all(both)),
        "groups": groups,
        "not_a_population_or_1000_claim": True,
    }
    _dump(RES / "summary.json", payload)
    lines = [
        "# N3INIT v5 N=18 L1",
        "",
        "Exact v4 four-start recipes. Development candidate only. n=8, seeds 10018–10025, not a population.",
        "QAOA start 0 is the derived center `(γ, β) = (0.25, 0.25)`. Energy selects. Fidelity is post hoc.",
        "",
        f"New quantum queries {new_q}. Historical {HISTORICAL}. Total research queries {HISTORICAL + new_q}. Failures {len(fails)}.",
        f"Classical B* wall {payload['classical_bstar']['wall_seconds']:.1f}s, CPU {payload['classical_bstar']['cpu_seconds']:.1f}s, peak RSS {payload['classical_bstar']['peak_rss_kib_max']} KiB. Not in the quantum ledger.",
        "",
        "| q | Q | n | mean E Nov3 | mean E QAOA | mean (E−E0)/N Nov3 | mean (E−E0)/N QAOA | mean F Nov3 | mean F QAOA | mean ΔE | CI | median ΔE | min | max | wins | start0 Nov3 | start0 QAOA |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for g in groups:
        lines.append(
            f"| {g['q']} | {g['Q']} | {g['n']} | {g['E_nov3']['mean']:.4f} | {g['E_qaoa']['mean']:.4f} | "
            f"{g['E_error_over_N_nov3']['mean']:.4f} | {g['E_error_over_N_qaoa']['mean']:.4f} | "
            f"{g['F_nov3']['mean']:.3f} | {g['F_qaoa']['mean']:.3f} | {g['d_pair']['mean']:+.4f} | "
            f"[{g['d_pair']['ci95'][0]:+.3f}, {g['d_pair']['ci95'][1]:+.3f}] | {g['d_pair']['median']:+.4f} | "
            f"{g['d_pair']['min']:+.4f} | {g['d_pair']['max']:+.4f} | {g['energy_wins']}/{g['n']} | "
            f"{g['center_start0_selected']}/{g['n']} | {g['qaoa_start0_selected']}/{g['n']} |"
        )
    lines += ["", "## Read flags", ""]
    for g in groups:
        lines.append(f"- q={g['q']}: `{g['persists']}`.")
    lines.append(f"- both budgets persist: `{payload['n15_shallow_edge_persists_at_both_q']}`.")
    lines.append("")
    lines.append("Do not attribute an energy edge to the coherent center unless start 0 is actually selected.")
    lines.append(f"Fit cell time {sum(r['seconds'] for r in ok):.1f}s. Workers 4, BLAS 1. Each ledger equals its Q.")
    (RES / "RESULTS.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", required=True, choices=("bstar", "run", "summary"))
    args = parser.parse_args()
    if not PREREG.exists():
        raise RuntimeError("v5 preregistration missing")
    _conf()
    if args.phase == "bstar":
        build_instances()
    elif args.phase == "run":
        run_fits()
    else:
        summarize()


if __name__ == "__main__":
    main()
