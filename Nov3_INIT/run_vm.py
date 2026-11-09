"""N=15 L=2,3 paper V(M) vs shared-shell Nov3 vs QAOA. Fair Q=1024·L."""
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
    METHOD_ID,
    make_rng,
    n_params,
    run_equal4,
    x0_center4,
    x0_qaoa_center4,
    x0_vm_center4,
)
from Nov3_INIT.bstar import maximize
from Nov3_INIT.circuit import Logical
from Nov3_INIT.run_v1 import _dump
from Nov9_SK_SPSA.experiments.FT.instance import make_J, packing

ROOT = Path(__file__).resolve().parent
PREREG = ROOT / "PREREG_VM.md"
RES = ROOT / "results" / "vm"
CELLS = RES / "cells"
INST = RES / "instances"
CONF = RES / "CONFORMANCE.json"
BAND = list(range(10480, 10488))
N = 15
LS = (2, 3)
Q_PER_LAYER = 1024
N_BOOT = 4000
HISTORICAL = 2572288
CELL_WALL_CAP = 1800.0
CUT = list(range(7))
ARMS = (("nov3_center4", "Nov3rd"), ("paper_vm", "VM"), ("qaoa_center4", "QAOA"))


def prereg_sha() -> str:
    return hashlib.sha256(PREREG.read_bytes()).hexdigest()


def allow(seed: int) -> None:
    seed = int(seed)
    if seed not in BAND:
        raise RuntimeError(f"seed {seed} is not in the V(M) band 10480..10487")
    if 10464 <= seed <= 10479:
        raise RuntimeError(f"local R28 seed {seed}")
    if seed <= 10463:
        raise RuntimeError(f"FT/prior seed {seed}")
    if 10432 <= seed <= 10439:
        raise RuntimeError(f"LE seed {seed}")
    if 10144 <= seed <= 10151:
        raise RuntimeError(f"holdout seed {seed}")
    if 11000 <= seed <= 11031 or 20000 <= seed <= 20999:
        raise RuntimeError(f"blind seed {seed}")


def _conf() -> dict:
    doc = json.loads(CONF.read_text())
    if doc.get("prereg_sha256") != prereg_sha() or not doc.get("ok"):
        raise RuntimeError("V(M) conformance does not match the current preregistration")
    if doc.get("collisions_in_nov3_init_results", {}).get("n", 1) != 0:
        raise RuntimeError("V(M) seeds already appear under Nov3_INIT/results")
    if packing(N) != [[a, a + 5, a + 10] for a in range(5)]:
        raise RuntimeError("N=15 packing is not interleaved m=5")
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
        "vm_band": True,
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


def _path(arm: str, method: str, seed: int, L: int) -> Path:
    return CELLS / f"{arm}_{method}_N{N}_s{seed}_L{L}_q{Q_PER_LAYER}.json"


def _one(task: dict) -> dict:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    arm, method = task["arm"], task["method"]
    seed, L = int(task["seed"]), int(task["L"])
    allow(seed)
    path = _path(arm, method, seed, L)
    if path.exists():
        return {"skipped": True}
    Q = Q_PER_LAYER * L
    t0 = time.perf_counter()
    cpu0 = time.process_time()
    try:
        meta, logical = _logical(seed)
        b = float(meta["Bstar"])
        mid = METHOD_ID["equal4"][method]
        rng_init = [None if s == 0 else make_rng(seed, N, b, L, mid, s, 2) for s in range(4)]
        rng_spsa = [make_rng(seed, N, b, L, mid, s, 0) for s in range(4)]
        ju = logical.j_upper

        def draw(method_name, ell, start_id, rng, _method=method, _ju=ju):
            if _method == "QAOA":
                return x0_qaoa_center4(ell, start_id, rng)
            if _method == "VM":
                return x0_vm_center4(N, ell, start_id, rng, _ju)
            return x0_center4(method_name, ell, start_id, rng)

        def propagate(x, _logical=logical, _method=method, _t0=t0):
            if time.perf_counter() - _t0 > CELL_WALL_CAP:
                raise RuntimeError(f"cell wall exceeded {CELL_WALL_CAP}s")
            return _logical.propagate(_method, L, x)

        out = run_equal4(
            propagate, method, L, Q, rng_init, rng_spsa, x0_fn=draw, n=N if method == "VM" else None
        )
        psi = out.pop("aux")
        f = logical.fidelity(psi)
        f_plus = float(meta["F_plus"])
        e0 = float(meta["E0"])
        record = {
            "ok": True,
            "failure": None,
            "blocker": None,
            "version": "N3INIT_vm",
            "arm": arm,
            "method": method,
            "method_id": mid,
            "n_params": n_params(method, L, N),
            "N": N,
            "seed": seed,
            "L": L,
            "q": Q_PER_LAYER,
            "Q": Q,
            "procedural_cap": Q,
            "same_query_budget_across_arms": True,
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
            "architecture": "paper_V_M" if method == "VM" else method,
            "not_nov3_hybrid": True,
        }
    except Exception as exc:
        kind = "wall" if "wall exceeded" in repr(exc) else "cell_infeasible"
        record = {
            "ok": False,
            "failure": repr(exc),
            "blocker": kind,
            "version": "N3INIT_vm",
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
            "physics_not_approximated": True,
        }
    _dump(path, record)
    extra = f" E={record['E']:.4f} F={record['F']:.3f}" if record["ok"] else " " + record["failure"]
    print(f"{'ok' if record['ok'] else 'FAIL'} vm {arm} N=15 s={seed} L={L} q=1024{extra} {record['seconds']:.1f}s", flush=True)
    return record


def run_fits() -> None:
    import multiprocessing as mp

    _conf()
    sha = prereg_sha()
    pending = []
    for L in LS:
        for seed in BAND:
            for arm, method in ARMS:
                if not _path(arm, method, seed, L).exists():
                    pending.append({"arm": arm, "method": method, "seed": seed, "L": L, "prereg_sha256": sha})
    print(f"vm pending {len(pending)}", flush=True)
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


def _group(ok: list[dict], L: int, rng: np.random.Generator) -> dict:
    nov3 = sorted([r for r in ok if r["arm"] == "nov3_center4" and r["L"] == L], key=lambda r: r["seed"])
    vm = sorted([r for r in ok if r["arm"] == "paper_vm" and r["L"] == L], key=lambda r: r["seed"])
    qa = sorted([r for r in ok if r["arm"] == "qaoa_center4" and r["L"] == L], key=lambda r: r["seed"])
    if not (len(nov3) == len(vm) == len(qa) == 8):
        return {"L": L, "incomplete": True, "n_nov3": len(nov3), "n_vm": len(vm), "n_qaoa": len(qa)}
    e_n = np.array([r["E"] for r in nov3])
    e_v = np.array([r["E"] for r in vm])
    e_q = np.array([r["E"] for r in qa])
    d_vq = _boot(e_v - e_q, rng)
    d_vn = _boot(e_v - e_n, rng)
    return {
        "L": L,
        "incomplete": False,
        "n": 8,
        "Q": Q_PER_LAYER * L,
        "n_params": {"Nov3rd": n_params("Nov3rd", L), "VM": n_params("VM", L, N), "QAOA": n_params("QAOA", L)},
        "d_vmQ": d_vq,
        "d_vm_minus_nov3": d_vn,
        "vm_vs_qaoa_wins": int(np.sum(e_v < e_q)),
        "vm_vs_nov3_wins": int(np.sum(e_v < e_n)),
        "E_nov3": _boot(e_n, rng),
        "E_vm": _boot(e_v, rng),
        "E_qaoa": _boot(e_q, rng),
        "F_nov3": _boot(np.array([r["F"] for r in nov3]), rng),
        "F_vm": _boot(np.array([r["F"] for r in vm]), rng),
        "F_qaoa": _boot(np.array([r["F"] for r in qa]), rng),
        "start0_nov3": int(sum(r["start_id"] == 0 for r in nov3)),
        "start0_vm": int(sum(r["start_id"] == 0 for r in vm)),
        "start0_qaoa": int(sum(r["start_id"] == 0 for r in qa)),
        "seeds": [int(r["seed"]) for r in vm],
        "l3_vm_qaoa_ci_lt0": bool(d_vq["mean"] < 0.0 and d_vq["ci95"][1] < 0.0),
        "vm_mean_E_le_nov3": bool(float(e_v.mean()) <= float(e_n.mean())),
        "l2_vm_qaoa_ci_not_gt0": bool(not (d_vq["mean"] > 0.0 and d_vq["ci95"][0] > 0.0)),
    }


def _label(g2: dict, g3: dict, fails: list) -> str:
    if fails or g2.get("incomplete") or g3.get("incomplete"):
        return "REJECT_VM_L3"
    l3 = bool(g3["l3_vm_qaoa_ci_lt0"] and g3["vm_mean_E_le_nov3"])
    l2 = bool(g2["vm_mean_E_le_nov3"] and g2["l2_vm_qaoa_ci_not_gt0"])
    return "VM_L3_EDGE" if l3 and l2 else "REJECT_VM_L3"


def summarize() -> None:
    rows = [json.loads(p.read_text()) for p in CELLS.glob("*.json")]
    fails = [r for r in rows if not r.get("ok")]
    ok = [r for r in rows if r.get("ok")]
    inst = [json.loads(p.read_text()) for p in INST.glob("N15_s*.json")]
    rng = np.random.default_rng(3334)
    g2 = _group(ok, 2, rng)
    g3 = _group(ok, 3, rng)
    label = _label(g2, g3, fails)
    new_q = int(sum(r["queries"]["total"] for r in ok))
    payload = {
        "version": "N3INIT_vm",
        "no_gr_patch": True,
        "no_candidacy_file": True,
        "holdout_fail_stands": True,
        "paper_V_M_implemented": True,
        "architecture": "p=L independent all-to-all V(M) then shared Rx; not Nov3 shells",
        "not_nov3_retune": True,
        "no_refit_10368_10375": True,
        "n_files": len(rows),
        "n_fail": len(fails),
        "failures": [{"arm": r["arm"], "seed": r["seed"], "L": r["L"], "failure": r["failure"]} for r in fails],
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
        "L2_sanity": g2,
        "L3_primary": g3,
        "gates": {
            "L3_vm_qaoa_ci_entirely_lt0": g3.get("l3_vm_qaoa_ci_lt0"),
            "L3_vm_mean_E_le_nov3": g3.get("vm_mean_E_le_nov3"),
            "L2_vm_mean_E_le_nov3": g2.get("vm_mean_E_le_nov3"),
            "L2_vm_qaoa_ci_not_entirely_gt0": g2.get("l2_vm_qaoa_ci_not_gt0"),
        },
    }
    _dump(RES / "summary.json", payload)
    lines = [
        "# N3INIT paper V(M) architecture N=15 L=2,3",
        "",
        "Separate architecture arm. Paper V(M) p=L with shared Rx mixer. Not Nov3 shells. Seeds 10480–10487. Same Q=1024·L across arms. No GR patch. No candidacy file. Holdout FAIL stands.",
        "",
        f"**{label}.**",
        "",
        f"New quantum queries {new_q}. Historical {HISTORICAL}. Total research queries {HISTORICAL + new_q}. Failures {len(fails)}.",
        f"Classical B* wall {payload['classical_bstar']['wall_seconds']:.1f}s, CPU {payload['classical_bstar']['cpu_seconds']:.1f}s, peak RSS {payload['classical_bstar']['peak_rss_kib_max']} KiB.",
        "",
        "Param counts N=15: QAOA 2L, shared Nov3 3+3L, V(M) 106L. L=2: 4 / 9 / 212. L=3: 6 / 12 / 318.",
        "",
    ]
    for tag, g in (("L=3 primary", g3), ("L=2 sanity", g2)):
        if g.get("incomplete"):
            lines.append(f"{tag}: incomplete {g}")
            continue
        d = g["d_vmQ"]
        ds = g["d_vm_minus_nov3"]
        lines += [
            f"## {tag}",
            "",
            f"V(M) vs QAOA mean ΔE {d['mean']:+.4f} CI [{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}] wins {g['vm_vs_qaoa_wins']}/{g['n']}.",
            f"V(M) vs Nov3 mean ΔE {ds['mean']:+.4f} CI [{ds['ci95'][0]:+.3f}, {ds['ci95'][1]:+.3f}] V(M) mean E {g['E_vm']['mean']:.4f} Nov3 {g['E_nov3']['mean']:.4f} QAOA {g['E_qaoa']['mean']:.4f}.",
            f"mean F V(M) {g['F_vm']['mean']:.3f} Nov3 {g['F_nov3']['mean']:.3f} QAOA {g['F_qaoa']['mean']:.3f}. start0 V(M) {g['start0_vm']}/{g['n']} Nov3 {g['start0_nov3']}/{g['n']} QAOA {g['start0_qaoa']}/{g['n']}.",
            "",
        ]
    if fails:
        lines += ["## Blockers", ""]
        for r in fails:
            lines.append(f"- {r['arm']} L={r['L']} seed={r['seed']}: {r['failure']}")
        lines.append("")
    lines.append("Fit workers 4, BLAS 1. Each completed ledger equals Q=1024·L.")
    (RES / "RESULTS.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", required=True, choices=("bstar", "run", "summary"))
    args = parser.parse_args()
    if not PREREG.exists():
        raise RuntimeError("V(M) preregistration missing")
    _conf()
    if args.phase == "bstar":
        build_instances()
    elif args.phase == "run":
        run_fits()
    else:
        summarize()


if __name__ == "__main__":
    main()
