"""N=15 L=3 q=2048 doubled-budget panel. Frozen v4 center-4. Energy selects."""
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
PREREG = ROOT / "PREREG_L3_2X.md"
RES = ROOT / "results" / "l3_2x"
CELLS = RES / "cells"
INST = RES / "instances"
CONF = RES / "CONFORMANCE.json"
BAND = list(range(10616, 10624))
HOLD = list(range(10624, 10632))
N = 15
L = 3
Q_PER_LAYER = 2048
Q = 6144
N_BOOT = 4000
HISTORICAL = 2916352
CELL_WALL_CAP = 3600.0
CUT = list(range(7))
METHODS = (
    ("nov3_center4", "Nov3rd", METHOD_ID["equal4"]["Nov3rd"]),
    ("qaoa_center4", "QAOA", METHOD_ID["equal4"]["QAOA"]),
)
HIST_L3 = {
    "v7_10018_25": {"mean_dE": 0.0193, "ci": [-0.045, 0.075], "Q": 3072},
    "spread8_equal4_10376_83": {"mean_dE": -0.0315, "ci": [-0.106, 0.043], "Q": 3072},
    "anneal_v4_10456_63": {"mean_dE": -0.0477, "ci": [-0.121, 0.022], "Q": 3072},
}


def prereg_sha() -> str:
    return hashlib.sha256(PREREG.read_bytes()).hexdigest()


def allow(seed: int) -> None:
    seed = int(seed)
    if seed in HOLD:
        raise RuntimeError(f"held-unused seed {seed}")
    if seed not in BAND:
        raise RuntimeError(f"seed {seed} is not in the L3-2x band 10616..10623")
    if 10600 <= seed <= 10615:
        raise RuntimeError(f"LE next seed {seed}")
    if 10584 <= seed <= 10599:
        raise RuntimeError(f"local R32 seed {seed}")
    if 10568 <= seed <= 10583:
        raise RuntimeError(f"width seed {seed}")
    if 10560 <= seed <= 10567:
        raise RuntimeError(f"unused-hole seed {seed}")
    if 10552 <= seed <= 10559:
        raise RuntimeError(f"gauge/LE-PR115 seed {seed}")
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
        raise RuntimeError("l3_2x conformance does not match the current preregistration")
    if doc.get("collisions_in_nov3_init_results", {}).get("n", 1) != 0:
        raise RuntimeError("l3_2x seeds already appear under Nov3_INIT/results")
    if packing(N) != [[a, a + 5, a + 10] for a in range(5)]:
        raise RuntimeError("N=15 packing")
    if int(doc.get("T", -1)) != 762 or int(doc.get("Q", -1)) != Q:
        raise RuntimeError("Q/T mismatch")
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
        "l3_2x_band": True,
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
    return CELLS / f"{arm}_{method}_N{N}_s{seed}_L{L}_q{Q_PER_LAYER}.json"


def _one(task: dict) -> dict:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    arm, method, mid = task["arm"], task["method"], int(task["method_id"])
    seed = int(task["seed"])
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

        def propagate(x, _log=logical, _method=method, _t0=t0):
            if time.perf_counter() - _t0 > CELL_WALL_CAP:
                raise RuntimeError(f"cell wall exceeded {CELL_WALL_CAP}s")
            return _log.propagate(_method, L, x)

        out = run_equal4(propagate, method, L, Q, rng_init, rng_spsa, x0_fn=draw)
        psi = out.pop("aux")
        f = logical.fidelity(psi)
        e0 = float(meta["E0"])
        f_plus = float(meta["F_plus"])
        record = {
            "ok": True,
            "failure": None,
            "version": "N3INIT_l3_2x",
            "arm": arm,
            "method": method,
            "method_id": mid,
            "n_params": n_params(method, L),
            "N": N,
            "seed": seed,
            "L": L,
            "q": Q_PER_LAYER,
            "Q": Q,
            "T": int(out["n_iter"]),
            "procedural_cap": Q,
            "same_Q_both_methods": True,
            "Bstar": b,
            "E0": e0,
            "E_plus": meta["E_plus"],
            "F_plus": f_plus,
            "E": out["E"],
            "E_error_over_N": float((out["E"] - e0) / float(N)),
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
            "not_spread8": True,
            "not_anneal": True,
        }
        if int(out["n_iter"]) != 762 or int(out["queries"]["total"]) != Q:
            raise RuntimeError(f"ledger/T mismatch {out['n_iter']} {out['queries']}")
    except Exception as exc:
        record = {
            "ok": False,
            "failure": repr(exc),
            "version": "N3INIT_l3_2x",
            "arm": arm,
            "method": method,
            "N": N,
            "seed": seed,
            "L": L,
            "q": Q_PER_LAYER,
            "Q": Q,
            "seconds": time.perf_counter() - t0,
            "cpu_seconds": time.process_time() - cpu0,
            "peak_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
            "prereg_sha256": task["prereg_sha256"],
        }
    _dump(path, record)
    extra = f" E={record['E']:.4f} F={record['F']:.3f}" if record["ok"] else " " + record["failure"]
    print(f"{'ok' if record['ok'] else 'FAIL'} l3_2x {arm} N=15 s={seed}{extra} {record['seconds']:.1f}s", flush=True)
    return record


def run_fits() -> None:
    import multiprocessing as mp

    _conf()
    sha = prereg_sha()
    pending = []
    for seed in BAND:
        for arm, method, mid in METHODS:
            if not _path(arm, method, seed).exists():
                pending.append({
                    "arm": arm,
                    "method": method,
                    "method_id": mid,
                    "seed": seed,
                    "prereg_sha256": sha,
                })
    print(f"l3_2x pending {len(pending)}", flush=True)
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


def summarize() -> None:
    rows = [json.loads(p.read_text()) for p in CELLS.glob("*.json")]
    fails = [r for r in rows if not r.get("ok")]
    ok = [r for r in rows if r.get("ok")]
    inst = [json.loads(p.read_text()) for p in INST.glob("N15_s*.json")]
    nov = sorted([r for r in ok if r["arm"] == "nov3_center4"], key=lambda r: r["seed"])
    qa = sorted([r for r in ok if r["arm"] == "qaoa_center4"], key=lambda r: r["seed"])
    complete = len(nov) == 8 and len(qa) == 8 and not fails
    primary = None
    if complete:
        e_n = np.array([r["E"] for r in nov])
        e_q = np.array([r["E"] for r in qa])
        d = _boot(e_n - e_q)
        primary = {
            "dE": d,
            "wins": int(np.sum(e_n < e_q)),
            "ci_entirely_lt0": bool(d["mean"] < 0.0 and d["ci95"][1] < 0.0),
            "E_nov3": _boot(e_n),
            "E_qaoa": _boot(e_q),
            "F_nov3": _boot(np.array([r["F"] for r in nov])),
            "F_qaoa": _boot(np.array([r["F"] for r in qa])),
            "errN_nov3": _boot(np.array([r["E_error_over_N"] for r in nov])),
            "errN_qaoa": _boot(np.array([r["E_error_over_N"] for r in qa])),
            "start0_nov3": int(sum(r["start_id"] == 0 for r in nov)),
            "start0_qaoa": int(sum(r["start_id"] == 0 for r in qa)),
            "seeds": [int(r["seed"]) for r in nov],
            "T": 762,
            "Q": Q,
        }
        label = "L3_2XBUDGET_ESTIMATE_NEGATIVE_CI" if primary["ci_entirely_lt0"] else "L3_2XBUDGET_FAIL"
    else:
        label = "L3_2XBUDGET_UNSTABLE"
    new_q = int(sum(r["queries"]["total"] for r in ok))
    payload = {
        "version": "N3INIT_l3_2x",
        "no_gr_patch": True,
        "no_adoption_gate": True,
        "if_fail_stop_no_q4096": True,
        "not_spread8": True,
        "not_anneal": True,
        "holdout_l2_fail_stands": True,
        "compiler_stream_closed": True,
        "n_files": len(rows),
        "n_fail": len(fails),
        "failures": [{"arm": r.get("arm"), "seed": r.get("seed"), "failure": r.get("failure")} for r in fails],
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
        "primary": primary,
        "secondary_historical_L3_q1024_not_paired": HIST_L3,
        "held_unused": HOLD,
    }
    _dump(RES / "summary.json", payload)
    lines = [
        "# N=15 L=3 doubled budget q=2048 Q=6144",
        "",
        "Original shared-Uc Nov3 (12 params) vs QAOA p=3 (6 params). Frozen v4 center-4. Same Q both methods. Seeds 10616–10623. 10624–10631 unused. Not spread8/anneal/pair-shell/V(M). No GR. If FAIL, stop; no q=4096.",
        "",
        f"**{label}.**",
        "",
        f"New energy queries {new_q}. Historical {HISTORICAL}. Total {HISTORICAL + new_q}. Failures {len(fails)}.",
        f"Classical B* wall {payload['classical_bstar']['wall_seconds']:.1f}s.",
        "",
    ]
    if primary is None:
        lines.append("Panel incomplete.")
    else:
        d = primary["dE"]
        lines += [
            "## Primary: Nov3−QAOA at Q=6144 (n=8)",
            "",
            f"mean ΔE {d['mean']:+.4f} CI [{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}] wins {primary['wins']}/8. CI<0 {primary['ci_entirely_lt0']}.",
            f"mean E Nov3 {primary['E_nov3']['mean']:.4f} QAOA {primary['E_qaoa']['mean']:.4f}. mean (E−E0)/N Nov3 {primary['errN_nov3']['mean']:.4f} QAOA {primary['errN_qaoa']['mean']:.4f}.",
            f"mean F Nov3 {primary['F_nov3']['mean']:.3f} QAOA {primary['F_qaoa']['mean']:.3f} (post hoc). start0 Nov3 {primary['start0_nov3']}/8 QAOA {primary['start0_qaoa']}/8.",
            f"T={primary['T']}. seeds {primary['seeds']}.",
            "",
            "## Secondary: historical L=3 q=1024 Q=3072 (different seeds, not paired)",
            "",
            "- v7 10018–25: mean ΔE +0.0193 CI [-0.045, +0.075]",
            "- spread8 equal4 10376–83: mean ΔE -0.0315 CI [-0.106, +0.043]",
            "- anneal V4 10456–63: mean ΔE -0.0477 CI [-0.121, +0.022]",
            "",
            "Not a paired test. Not an adoption. L=2 holdout FAIL stands. Compiler stream closed.",
            "",
        ]
        if label == "L3_2XBUDGET_FAIL":
            lines.append("FAIL: stop further L=3 budget-increase retries. Do not climb to q=4096.")
            lines.append("")
    if fails:
        lines += ["## Blockers", ""]
        for r in fails:
            lines.append(f"- {r.get('arm')} s={r.get('seed')}: {r.get('failure')}")
        lines.append("")
    lines.append("Fit workers 4, BLAS 1. Each completed ledger equals Q=6144.")
    (RES / "RESULTS.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", required=True, choices=("bstar", "run", "summary"))
    args = parser.parse_args()
    if not PREREG.exists():
        raise RuntimeError("l3_2x preregistration missing")
    _conf()
    if args.phase == "bstar":
        build_instances()
    elif args.phase == "run":
        run_fits()
    else:
        summarize()


if __name__ == "__main__":
    main()
