"""Unread-seed N=15 and N=18 L=1 width panel. Frozen v4 center-4. Energy selects."""
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

from Nov3_INIT.alloc import METHOD_ID, make_rng, n_params, run_equal4, x0_center4, x0_qaoa_center4
from Nov3_INIT.bstar import maximize
from Nov3_INIT.circuit import Logical
from Nov3_INIT.run_v1 import _dump
from Nov9_SK_SPSA.experiments.FT.instance import make_J, packing

ROOT = Path(__file__).resolve().parent
PREREG = ROOT / "PREREG_WIDTH.md"
RES = ROOT / "results" / "width"
CELLS = RES / "cells"
INST = RES / "instances"
CONF = RES / "CONFORMANCE.json"
N15 = list(range(10568, 10576))
N18 = list(range(10576, 10584))
BAND = N15 + N18
L = 1
Q = 1024
N_BOOT = 4000
HISTORICAL = 2883584
CELL_WALL = {15: 600.0, 18: 1800.0}
CUT = {15: list(range(7)), 18: list(range(9))}
METHODS = (
    ("nov3_center4", "Nov3rd", METHOD_ID["equal4"]["Nov3rd"]),
    ("qaoa_center4", "QAOA", METHOD_ID["equal4"]["QAOA"]),
)


def prereg_sha() -> str:
    return hashlib.sha256(PREREG.read_bytes()).hexdigest()


def n_of(seed: int) -> int:
    seed = int(seed)
    if seed in N15:
        return 15
    if seed in N18:
        return 18
    raise RuntimeError(f"seed {seed} is not assigned to N=15 or N=18")


def allow(seed: int) -> None:
    seed = int(seed)
    if seed not in BAND:
        raise RuntimeError(f"seed {seed} is not in the width band 10568..10583")
    if 10552 <= seed <= 10567:
        raise RuntimeError(f"gauge/buffer seed {seed}")
    if 10536 <= seed <= 10551:
        raise RuntimeError(f"local R31 seed {seed}")
    if 10520 <= seed <= 10535:
        raise RuntimeError(f"FT cutoff/WL seed {seed}")
    if 10504 <= seed <= 10519:
        raise RuntimeError(f"local R30 seed {seed}")
    if 10432 <= seed <= 10439:
        raise RuntimeError(f"LE N21 seed {seed}")
    if 10480 <= seed <= 10487:
        raise RuntimeError(f"FT V(M) seed {seed}")
    if seed <= 10463:
        raise RuntimeError(f"FT/prior seed {seed}")
    if 10144 <= seed <= 10151:
        raise RuntimeError(f"holdout seed {seed}")
    if 11000 <= seed <= 11031 or 20000 <= seed <= 20999:
        raise RuntimeError(f"blind seed {seed}")


def _conf() -> dict:
    doc = json.loads(CONF.read_text())
    if doc.get("prereg_sha256") != prereg_sha() or not doc.get("ok"):
        raise RuntimeError("width conformance does not match the current preregistration")
    if not doc.get("n18_gate", {}).get("n18_allowed"):
        raise RuntimeError("N=18 resource gate closed")
    if doc.get("collisions_in_nov3_init_results", {}).get("n", 1) != 0:
        raise RuntimeError("width seeds already appear under Nov3_INIT/results")
    if packing(15) != [[a, a + 5, a + 10] for a in range(5)]:
        raise RuntimeError("N=15 packing")
    if packing(18) != [[a, a + 6, a + 12] for a in range(6)]:
        raise RuntimeError("N=18 packing")
    return doc


def _bstar_worker(seed: int) -> dict:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    allow(int(seed))
    n = n_of(seed)
    t0 = time.perf_counter()
    cpu0 = time.process_time()
    rec = maximize(make_J(int(seed), n), n)
    gs = rec.pop("gs")
    rec.pop("hc")
    rec.update({
        "N": n,
        "seed": int(seed),
        "seconds": time.perf_counter() - t0,
        "cpu_seconds": time.process_time() - cpu0,
        "peak_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "prereg_sha256": prereg_sha(),
        "dense_matrix": False,
        "classical_not_a_quantum_query": True,
        "width_band": True,
    })
    if rec.get("cut") != CUT[n]:
        raise RuntimeError(f"unexpected cut N={n} {rec.get('cut')}")
    return {"rec": rec, "gs": gs}


def build_instances() -> None:
    _conf()
    pending = []
    for s in BAND:
        allow(s)
        n = n_of(s)
        if not (INST / f"N{n}_s{s}.json").exists():
            pending.append(s)
    if not pending:
        return
    import multiprocessing as mp

    INST.mkdir(parents=True, exist_ok=True)
    with mp.get_context("fork").Pool(4) as pool:
        for got in pool.imap_unordered(_bstar_worker, pending, chunksize=1):
            seed = got["rec"]["seed"]
            n = got["rec"]["N"]
            np.savez(INST / f"N{n}_s{seed}.npz", gs=got["gs"])
            _dump(INST / f"N{n}_s{seed}.json", got["rec"])
            r = got["rec"]
            print(f"B* N={n} seed={seed} b={r['Bstar']:.6f} gap={r['gap']:.4g} {r['seconds']:.1f}s", flush=True)


def _logical(n: int, seed: int):
    meta = json.loads((INST / f"N{n}_s{seed}.json").read_text())
    gs = np.load(INST / f"N{n}_s{seed}.npz")["gs"]
    if int(meta["N"]) != n or int(meta["seed"]) != int(seed) or gs.size != (1 << n) or meta.get("cut") != CUT[n]:
        raise RuntimeError(f"B* invalid N={n} seed={seed}")
    return meta, Logical(n, seed, float(meta["Bstar"]), gs)


def _path(arm: str, method: str, n: int, seed: int) -> Path:
    return CELLS / f"{arm}_{method}_N{n}_s{seed}_L{L}_q{Q}.json"


def _one(task: dict) -> dict:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    arm, method, mid = task["arm"], task["method"], int(task["method_id"])
    seed = int(task["seed"])
    n = n_of(seed)
    allow(seed)
    path = _path(arm, method, n, seed)
    if path.exists():
        return {"skipped": True}
    t0 = time.perf_counter()
    cpu0 = time.process_time()
    try:
        meta, logical = _logical(n, seed)
        b = float(meta["Bstar"])
        rng_init = [None if s == 0 else make_rng(seed, n, b, L, mid, s, 2) for s in range(4)]
        rng_spsa = [make_rng(seed, n, b, L, mid, s, 0) for s in range(4)]

        def draw(method_name, ell, start_id, rng, _method=method):
            if _method == "QAOA":
                return x0_qaoa_center4(ell, start_id, rng)
            return x0_center4(method_name, ell, start_id, rng)

        def propagate(x, _log=logical, _method=method, _t0=t0, _n=n):
            if time.perf_counter() - _t0 > CELL_WALL[_n]:
                raise RuntimeError(f"cell wall exceeded {CELL_WALL[_n]}s")
            return _log.propagate(_method, L, x)

        out = run_equal4(propagate, method, L, Q, rng_init, rng_spsa, x0_fn=draw)
        psi = out.pop("aux")
        f = logical.fidelity(psi)
        e0 = float(meta["E0"])
        f_plus = float(meta["F_plus"])
        record = {
            "ok": True,
            "failure": None,
            "version": "N3INIT_width",
            "arm": arm,
            "method": method,
            "method_id": mid,
            "n_params": n_params(method, L),
            "N": n,
            "seed": seed,
            "L": L,
            "q": Q,
            "Q": Q,
            "procedural_cap": Q,
            "Bstar": b,
            "E0": e0,
            "E_plus": meta["E_plus"],
            "F_plus": f_plus,
            "E": out["E"],
            "E_error_over_N": float((out["E"] - e0) / float(n)),
            "F": f,
            "eta": float((f - f_plus) / (1.0 - f_plus)),
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
            "no_gr_patch": True,
            "not_l2": True,
        }
    except Exception as exc:
        record = {
            "ok": False,
            "failure": repr(exc),
            "version": "N3INIT_width",
            "arm": arm,
            "method": method,
            "N": n,
            "seed": seed,
            "L": L,
            "q": Q,
            "Q": Q,
            "seconds": time.perf_counter() - t0,
            "cpu_seconds": time.process_time() - cpu0,
            "peak_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
            "prereg_sha256": task["prereg_sha256"],
        }
    _dump(path, record)
    extra = f" E={record['E']:.4f} F={record['F']:.3f}" if record["ok"] else " " + record["failure"]
    print(f"{'ok' if record['ok'] else 'FAIL'} width {arm} N={n} s={seed}{extra} {record['seconds']:.1f}s", flush=True)
    return record


def run_fits() -> None:
    import multiprocessing as mp

    _conf()
    sha = prereg_sha()
    pending = []
    for seed in BAND:
        n = n_of(seed)
        for arm, method, mid in METHODS:
            if not _path(arm, method, n, seed).exists():
                pending.append({
                    "arm": arm,
                    "method": method,
                    "method_id": mid,
                    "seed": seed,
                    "prereg_sha256": sha,
                })
    print(f"width pending {len(pending)}", flush=True)
    wall0 = time.perf_counter()
    if pending:
        with mp.get_context("fork").Pool(4) as pool:
            list(pool.imap_unordered(_one, pending, chunksize=1))
    (RES / "wall.json").write_text(json.dumps({"fit_seconds": time.perf_counter() - wall0, "workers": 4, "blas_threads": 1}) + "\n")


def _boot(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=np.float64)
    n = values.shape[0]
    rng = np.random.default_rng(3334)
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


def _pair(ok: list[dict], n: int) -> dict:
    nov = sorted([r for r in ok if r["arm"] == "nov3_center4" and r["N"] == n], key=lambda r: r["seed"])
    qa = sorted([r for r in ok if r["arm"] == "qaoa_center4" and r["N"] == n], key=lambda r: r["seed"])
    if len(nov) != 8 or len(qa) != 8:
        return {"N": n, "incomplete": True, "n_nov": len(nov), "n_qaoa": len(qa)}
    e_n = np.array([r["E"] for r in nov])
    e_q = np.array([r["E"] for r in qa])
    d = _boot(e_n - e_q)
    return {
        "N": n,
        "L": L,
        "Q": Q,
        "incomplete": False,
        "n": 8,
        "seeds": [int(r["seed"]) for r in nov],
        "dE": d,
        "wins": int(np.sum(e_n < e_q)),
        "ci_entirely_lt0": bool(d["mean"] < 0.0 and d["ci95"][1] < 0.0),
        "E_nov3": _boot(e_n),
        "E_qaoa": _boot(e_q),
        "errN_nov3": _boot(np.array([r["E_error_over_N"] for r in nov])),
        "errN_qaoa": _boot(np.array([r["E_error_over_N"] for r in qa])),
        "F_nov3": _boot(np.array([r["F"] for r in nov])),
        "F_qaoa": _boot(np.array([r["F"] for r in qa])),
        "start0_nov3": int(sum(r["start_id"] == 0 for r in nov)),
        "start0_qaoa": int(sum(r["start_id"] == 0 for r in qa)),
    }


def summarize() -> None:
    rows = [json.loads(p.read_text()) for p in CELLS.glob("*.json")]
    fails = [r for r in rows if not r.get("ok")]
    ok = [r for r in rows if r.get("ok")]
    inst = [json.loads(p.read_text()) for p in INST.glob("N*_s*.json")]
    g15 = _pair(ok, 15)
    g18 = _pair(ok, 18)
    complete = (not g15.get("incomplete")) and (not g18.get("incomplete")) and not fails
    if not complete:
        label = "WIDTH_L1_MISS"
    elif g15["ci_entirely_lt0"] and g18["ci_entirely_lt0"]:
        label = "WIDTH_L1_BOTH"
    elif g15["ci_entirely_lt0"]:
        label = "WIDTH_L1_N15_ONLY"
    elif g18["ci_entirely_lt0"]:
        label = "WIDTH_L1_N18_ONLY"
    else:
        label = "WIDTH_L1_MISS"
    new_q = int(sum(r["queries"]["total"] for r in ok))
    payload = {
        "version": "N3INIT_width",
        "no_gr_patch": True,
        "no_adoption_gate": True,
        "not_l2_sampling": True,
        "compiler_stream_closed": True,
        "holdout_l2_fail_stands": True,
        "n_files": len(rows),
        "n_fail": len(fails),
        "failures": [{"arm": r.get("arm"), "N": r.get("N"), "seed": r.get("seed"), "failure": r.get("failure")} for r in fails],
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
        "N15": g15,
        "N18": g18,
        "not_pooled_across_N": True,
    }
    _dump(RES / "summary.json", payload)

    def blk(tag: str, g: dict) -> list[str]:
        if g.get("incomplete"):
            return [f"## {tag}", "", f"incomplete {g}", ""]
        d = g["dE"]
        return [
            f"## {tag}",
            "",
            f"Nov3−QAOA mean ΔE {d['mean']:+.4f} CI [{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}] wins {g['wins']}/{g['n']}. CI<0 {g['ci_entirely_lt0']}.",
            f"mean E Nov3 {g['E_nov3']['mean']:.4f} QAOA {g['E_qaoa']['mean']:.4f}. mean (E−E0)/N Nov3 {g['errN_nov3']['mean']:.4f} QAOA {g['errN_qaoa']['mean']:.4f}.",
            f"mean F Nov3 {g['F_nov3']['mean']:.3f} QAOA {g['F_qaoa']['mean']:.3f} (post hoc). start0 Nov3 {g['start0_nov3']}/{g['n']} QAOA {g['start0_qaoa']}/{g['n']}.",
            f"seeds {g['seeds']}.",
            "",
        ]

    lines = [
        "# Unread-seed N=15 and N=18 L=1 width panel",
        "",
        "Original shared-Uc Nov3 vs QAOA p=1. Frozen v4 center-4. Exclusive seeds 10568–10575 at N=15 and 10576–10583 at N=18. Not L=2. Not pooled across N. Exploratory; no adoption; no GR. Compiler stream closed. L=2 holdout FAIL stands.",
        "",
        f"**{label}.**",
        "",
        f"New energy queries {new_q}. Historical {HISTORICAL}. Total {HISTORICAL + new_q}. Failures {len(fails)}.",
        f"Classical B* wall {payload['classical_bstar']['wall_seconds']:.1f}s.",
        "",
    ]
    lines += blk("N=15 L=1 Q=1024", g15)
    lines += blk("N=18 L=1 Q=1024", g18)
    lines += [
        "v5 N=18 L=1 used development seeds 10018–10025 (confounded with N=15). This panel uses exclusive unread instances at each width.",
        "Do not read WIDTH_L1_BOTH as undoing L=2 holdout FAIL or as a hardware claim.",
        "",
    ]
    if fails:
        lines += ["## Blockers", ""]
        for r in fails:
            lines.append(f"- {r.get('arm')} N={r.get('N')} s={r.get('seed')}: {r.get('failure')}")
        lines.append("")
    lines.append("Fit workers 4, BLAS 1. Each completed ledger equals Q=1024.")
    (RES / "RESULTS.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", required=True, choices=("bstar", "run", "summary"))
    args = parser.parse_args()
    if not PREREG.exists():
        raise RuntimeError("width preregistration missing")
    _conf()
    if args.phase == "bstar":
        build_instances()
    elif args.phase == "run":
        run_fits()
    else:
        summarize()


if __name__ == "__main__":
    main()
