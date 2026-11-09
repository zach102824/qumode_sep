"""N=15 L=2 Q=2048 n=16 effect-size panel. Estimation only. Exact v4 recipe."""
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
from Nov3_INIT.run_v1 import _dump
from Nov9_SK_SPSA.experiments.FT.instance import make_J

ROOT = Path(__file__).resolve().parent
PREREG = ROOT / "PREREG_V9.md"
RES = ROOT / "results" / "v9"
CELLS = RES / "cells"
INST = RES / "instances"
CONF = RES / "CONFORMANCE.json"
BAND = list(range(10200, 10216))
N, L, Q = 15, 2, 2048
N_BOOT = 4000
HISTORICAL = 2048000
CELL_WALL_CAP = 1800.0
CUT = list(range(7))
TRAIN_CELLS = ROOT / "results" / "v7" / "cells"
HOLDOUT_CELLS = ROOT / "results" / "v8" / "cells"


def prereg_sha() -> str:
    return hashlib.sha256(PREREG.read_bytes()).hexdigest()


def allow(seed: int) -> None:
    seed = int(seed)
    if seed not in BAND:
        raise RuntimeError(f"seed {seed} is not in the estimate band 10200..10215")
    if 10018 <= seed <= 10025:
        raise RuntimeError(f"train seed {seed}")
    if 10144 <= seed <= 10151:
        raise RuntimeError(f"holdout seed {seed}")
    if seed <= 10199:
        raise RuntimeError(f"LE band seed {seed}")
    if 10184 <= seed <= 10191:
        raise RuntimeError(f"local R19 seed {seed}")
    if 11000 <= seed <= 11031 or 20000 <= seed <= 20999:
        raise RuntimeError(f"blind seed {seed}")


def _conf() -> dict:
    doc = json.loads(CONF.read_text())
    if doc.get("prereg_sha256") != prereg_sha() or not doc.get("ok"):
        raise RuntimeError("v9 conformance does not match the current preregistration")
    if doc.get("collisions_in_nov3_init_results", {}).get("n", 1) != 0:
        raise RuntimeError("estimate seeds already appear under Nov3_INIT/results")
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
        "dense_matrix": False,
        "classical_not_a_quantum_query": True,
        "estimate_band": True,
    })
    if rec.get("cut") != CUT:
        raise RuntimeError(f"unexpected cut {rec.get('cut')}")
    return {"rec": rec, "gs": gs}


def build_instances() -> None:
    _conf()
    pending = []
    for s in BAND:
        allow(s)
        if not (INST / f"N{N}_s{s}.json").exists():
            pending.append(s)
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
    arm, method, seed = task["arm"], task["method"], int(task["seed"])
    allow(seed)
    path = _path(arm, method, seed)
    if path.exists():
        return {"skipped": True}
    t0 = time.perf_counter()
    cpu0 = time.process_time()
    try:
        meta, logical = _logical(seed)
        b = float(meta["Bstar"])
        mid = 0 if method == "QAOA" else 1
        rng_init = [None if s == 0 else make_rng(seed, N, b, L, mid, s, 2) for s in range(4)]
        rng_spsa = [make_rng(seed, N, b, L, mid, s, 0) for s in range(4)]

        def draw(method_name, ell, start_id, rng, _arm=arm):
            if _arm == "qaoa_center4":
                return x0_qaoa_center4(ell, start_id, rng)
            return x0_center4(method_name, ell, start_id, rng)

        def propagate(x, _logical=logical, _method=method, _t0=t0):
            if time.perf_counter() - _t0 > CELL_WALL_CAP:
                raise RuntimeError(f"cell wall exceeded {CELL_WALL_CAP}s")
            return _logical.propagate(_method, L, x)

        out = run_equal4(propagate, method, L, Q, rng_init, rng_spsa, x0_fn=draw)
        psi = out.pop("aux")
        f = logical.fidelity(psi)
        f_plus = float(meta["F_plus"])
        e0 = float(meta["E0"])
        record = {
            "ok": True,
            "failure": None,
            "blocker": None,
            "version": "N3INIT_v9",
            "estimate_only": True,
            "arm": arm,
            "method": method,
            "method_id": mid,
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
            "exact_v4_recipe": True,
        }
    except Exception as exc:
        kind = "wall" if "wall exceeded" in repr(exc) else "cell_infeasible"
        record = {
            "ok": False,
            "failure": repr(exc),
            "blocker": kind,
            "version": "N3INIT_v9",
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
    print(f"{'ok' if record['ok'] else 'FAIL'} v9 {arm} N=15 s={seed} L=2 q=1024{extra} {record['seconds']:.1f}s", flush=True)
    return record


def run_fits() -> None:
    import multiprocessing as mp

    _conf()
    sha = prereg_sha()
    pending = []
    for seed in BAND:
        for arm, method in (("nov3_center4", "Nov3rd"), ("qaoa_center4", "QAOA")):
            if not _path(arm, method, seed).exists():
                pending.append({"arm": arm, "method": method, "seed": seed, "prereg_sha256": sha})
    print(f"v9 pending {len(pending)}", flush=True)
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


def _paired(nov, qa, rng):
    e_n = np.array([r["E"] for r in nov])
    e_q = np.array([r["E"] for r in qa])
    f_n = np.array([r["F"] for r in nov])
    f_q = np.array([r["F"] for r in qa])
    d = _boot(e_n - e_q, rng)
    return {
        "n": len(nov),
        "d_pair": d,
        "energy_wins": int(np.sum(e_n < e_q)),
        "energy_ties": int(np.sum(np.abs(e_n - e_q) <= 1e-8)),
        "E_nov3": _boot(e_n, rng),
        "E_qaoa": _boot(e_q, rng),
        "F_nov3": _boot(f_n, rng),
        "F_qaoa": _boot(f_q, rng),
        "center_start0_selected": int(sum(r["start_id"] == 0 for r in nov)),
        "qaoa_start0_selected": int(sum(r["start_id"] == 0 for r in qa)),
        "seeds": [int(r["seed"]) for r in nov],
    }


def _load_band(folder: Path, seeds: list[int]):
    nov, qa = [], []
    for seed in seeds:
        p_n = folder / f"nov3_center4_Nov3rd_N15_s{seed}_L2_q1024.json"
        p_q = folder / f"qaoa_center4_QAOA_N15_s{seed}_L2_q1024.json"
        if not p_n.exists() or not p_q.exists():
            raise RuntimeError(f"missing historical N15 L2 Q2048 cell seed={seed}")
        a, b = json.loads(p_n.read_text()), json.loads(p_q.read_text())
        if not a.get("ok") or not b.get("ok") or a["Q"] != 2048 or b["Q"] != 2048:
            raise RuntimeError(f"historical cell unusable seed={seed}")
        nov.append(a)
        qa.append(b)
    return nov, qa


def summarize() -> None:
    rows = [json.loads(p.read_text()) for p in CELLS.glob("*.json")]
    fails = [r for r in rows if not r.get("ok")]
    ok = [r for r in rows if r.get("ok")]
    inst = [json.loads(p.read_text()) for p in INST.glob("N15_s*.json")]
    rng = np.random.default_rng(3334)
    nov = sorted([r for r in ok if r["arm"] == "nov3_center4"], key=lambda r: r["seed"])
    qa = sorted([r for r in ok if r["arm"] == "qaoa_center4"], key=lambda r: r["seed"])
    primary = _paired(nov, qa, rng) if nov and qa and len(nov) == len(qa) else None
    label = "INCOMPLETE"
    if primary:
        lo, hi = primary["d_pair"]["ci95"]
        label = "ESTIMATE_NEGATIVE_CI" if primary["d_pair"]["mean"] < 0.0 and hi < 0.0 else "UNSTABLE / CI_CROSSES_0"
    train_n, train_q = _load_band(TRAIN_CELLS, list(range(10018, 10026)))
    hold_n, hold_q = _load_band(HOLDOUT_CELLS, list(range(10144, 10152)))
    pool = _paired(train_n + hold_n + nov, train_q + hold_q + qa, rng) if primary else None
    new_q = int(sum(r["queries"]["total"] for r in ok))
    payload = {
        "version": "N3INIT_v9",
        "estimate_only": True,
        "no_gr_patch": True,
        "no_candidacy_file": True,
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
        "primary_n16": primary,
        "exploratory_pool_n32_not_a_gate": pool,
    }
    _dump(RES / "summary.json", payload)
    lines = [
        "# N3INIT v9 N=15 L=2 Q=2048 effect-size panel",
        "",
        "Estimation only. Exact v4 recipe. Seeds 10200–10215. No GR patch. No candidacy file.",
        "",
        f"**{label}.**",
        "",
        f"New quantum queries {new_q}. Historical {HISTORICAL}. Total research queries {HISTORICAL + new_q}. Failures {len(fails)}.",
        f"Classical B* wall {payload['classical_bstar']['wall_seconds']:.1f}s, CPU {payload['classical_bstar']['cpu_seconds']:.1f}s, peak RSS {payload['classical_bstar']['peak_rss_kib_max']} KiB.",
        "",
    ]
    if primary:
        d = primary["d_pair"]
        lines += [
            f"n=16 mean ΔE {d['mean']:+.4f} CI [{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}] median {d['median']:+.4f} min {d['min']:+.4f} max {d['max']:+.4f} wins {primary['energy_wins']}/{primary['n']} ties {primary['energy_ties']}.",
            f"mean F Nov3 {primary['F_nov3']['mean']:.3f} QAOA {primary['F_qaoa']['mean']:.3f}. start0 Nov3 {primary['center_start0_selected']}/{primary['n']} QAOA {primary['qaoa_start0_selected']}/{primary['n']}.",
            "",
            "## Exploratory pool n=32 (not a gate)",
            "",
            "Train 10018–25 (v7) + holdout 10144–51 (v8) + this band 10200–15. Queries already charged. Not a confirmation and not an adoption.",
        ]
        p = pool["d_pair"]
        lines.append(
            f"n=32 mean ΔE {p['mean']:+.4f} CI [{p['ci95'][0]:+.3f}, {p['ci95'][1]:+.3f}] median {p['median']:+.4f} min {p['min']:+.4f} max {p['max']:+.4f} wins {pool['energy_wins']}/{pool['n']}."
        )
    if fails:
        lines += ["", "## Blockers", ""]
        for r in fails:
            lines.append(f"- {r['arm']} seed={r['seed']}: {r['failure']}")
    lines.append("")
    lines.append(f"Fit cell time {sum(r['seconds'] for r in ok + fails):.1f}s. Workers 4, BLAS 1. Each completed ledger equals Q=2048.")
    (RES / "RESULTS.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", required=True, choices=("bstar", "run", "summary"))
    args = parser.parse_args()
    if not PREREG.exists():
        raise RuntimeError("v9 preregistration missing")
    _conf()
    if args.phase == "bstar":
        build_instances()
    elif args.phase == "run":
        run_fits()
    else:
        summarize()


if __name__ == "__main__":
    main()
