"""N=15 L=1,2 Fock-cutoff Nov3 vs QAOA. D0=8 and D0+2=10. Frozen v4 center-4."""
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
from Nov3_INIT.fock import D0, D0_PLUS_2, FockNov3, leakage_logical
from Nov3_INIT.run_v1 import _dump
from Nov9_SK_SPSA.experiments.FT.instance import make_J, packing

ROOT = Path(__file__).resolve().parent
PREREG = ROOT / "PREREG_CUTOFF.md"
RES = ROOT / "results" / "cutoff"
CELLS = RES / "cells"
INST = RES / "instances"
CONF = RES / "CONFORMANCE.json"
BAND = list(range(10520, 10528))
N = 15
LS = (1, 2)
Q_PER_LAYER = 1024
N_BOOT = 4000
HISTORICAL = 2695168
CELL_WALL_CAP = 1800.0
CUT = list(range(7))
ARMS = (
    ("nov3_D0", "Nov3rd", D0),
    ("nov3_D2", "Nov3rd", D0_PLUS_2),
    ("qaoa_D0", "QAOA", D0),
    ("qaoa_D2", "QAOA", D0_PLUS_2),
)


def prereg_sha() -> str:
    return hashlib.sha256(PREREG.read_bytes()).hexdigest()


def allow(seed: int) -> None:
    seed = int(seed)
    if seed not in BAND:
        raise RuntimeError(f"seed {seed} is not in the cutoff band 10520..10527")
    if 10504 <= seed <= 10519:
        raise RuntimeError(f"local R30 seed {seed}")
    if 10480 <= seed <= 10487:
        raise RuntimeError(f"FT V(M) seed {seed}")
    if 10432 <= seed <= 10439:
        raise RuntimeError(f"LE N21 seed {seed}")
    if seed <= 10463:
        raise RuntimeError(f"FT/prior seed {seed}")
    if 10144 <= seed <= 10151:
        raise RuntimeError(f"holdout seed {seed}")
    if 11000 <= seed <= 11031 or 20000 <= seed <= 20999:
        raise RuntimeError(f"blind seed {seed}")


def _conf() -> dict:
    doc = json.loads(CONF.read_text())
    if doc.get("prereg_sha256") != prereg_sha() or not doc.get("ok"):
        raise RuntimeError("cutoff conformance does not match the current preregistration")
    if doc.get("collisions_in_nov3_init_results", {}).get("n", 1) != 0:
        raise RuntimeError("cutoff seeds already appear under Nov3_INIT/results")
    if packing(N) != [[a, a + 5, a + 10] for a in range(5)]:
        raise RuntimeError("N=15 packing is not interleaved m=5")
    if int(doc.get("D0", -1)) != D0 or int(doc.get("D0_plus_2", -1)) != D0_PLUS_2:
        raise RuntimeError("cutoff D0 mismatch")
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
        "cutoff_band": True,
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


def _path(arm: str, method: str, seed: int, L: int, nf: int) -> Path:
    return CELLS / f"{arm}_{method}_N{N}_s{seed}_L{L}_q{Q_PER_LAYER}_nf{nf}.json"


def _one(task: dict) -> dict:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    arm, method, nf = task["arm"], task["method"], int(task["nf"])
    seed, L = int(task["seed"]), int(task["L"])
    allow(seed)
    path = _path(arm, method, seed, L, nf)
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
        fock = FockNov3(logical, nf) if method == "Nov3rd" else None

        def draw(method_name, ell, start_id, rng, _method=method):
            if _method == "QAOA":
                return x0_qaoa_center4(ell, start_id, rng)
            return x0_center4(method_name, ell, start_id, rng)

        def propagate(x, _logical=logical, _method=method, _fock=fock, _t0=t0, _L=L):
            if time.perf_counter() - _t0 > CELL_WALL_CAP:
                raise RuntimeError(f"cell wall exceeded {CELL_WALL_CAP}s")
            if _method == "QAOA":
                return _logical.propagate("QAOA", _L, x)
            return _fock.propagate(x, _L)

        out = run_equal4(propagate, method, L, Q, rng_init, rng_spsa, x0_fn=draw)
        psi = out.pop("aux")
        if method == "Nov3rd":
            f = fock.fidelity(psi)
            leak = fock.leakage(psi)
        else:
            f = logical.fidelity(psi)
            leak = leakage_logical(psi, logical.groups)
            leak["nf"] = nf
        f_plus = float(meta["F_plus"])
        e0 = float(meta["E0"])
        record = {
            "ok": True,
            "failure": None,
            "blocker": None,
            "version": "N3INIT_cutoff",
            "arm": arm,
            "method": method,
            "method_id": mid,
            "nf": nf,
            "D0": D0,
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
            "leakage": leak,
            "seconds": time.perf_counter() - t0,
            "cpu_seconds": time.process_time() - cpu0,
            "peak_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
            "prereg_sha256": task["prereg_sha256"],
            "fidelity_posthoc": True,
            "ideal_W_L_only": True,
            "compiled_W_L_not_modelled": True,
            "rng_omits_nf": True,
        }
    except Exception as exc:
        kind = "wall" if "wall exceeded" in repr(exc) else "cell_infeasible"
        record = {
            "ok": False,
            "failure": repr(exc),
            "blocker": kind,
            "version": "N3INIT_cutoff",
            "arm": arm,
            "method": method,
            "nf": nf,
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
    extra = f" E={record['E']:.4f} F={record['F']:.3f} p_leak={record['leakage']['p_leak']:.2e}" if record["ok"] else " " + record["failure"]
    print(f"{'ok' if record['ok'] else 'FAIL'} cutoff {arm} N=15 s={seed} L={L} nf={nf}{extra} {record['seconds']:.1f}s", flush=True)
    return record


def run_fits() -> None:
    import multiprocessing as mp

    _conf()
    sha = prereg_sha()
    pending = []
    for L in LS:
        for seed in BAND:
            for arm, method, nf in ARMS:
                if not _path(arm, method, seed, L, nf).exists():
                    pending.append({"arm": arm, "method": method, "nf": nf, "seed": seed, "L": L, "prereg_sha256": sha})
    print(f"cutoff pending {len(pending)}", flush=True)
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


def _pair(ok: list[dict], L: int, nf: int, rng: np.random.Generator) -> dict:
    nov = f"nov3_D0" if nf == D0 else "nov3_D2"
    qa = f"qaoa_D0" if nf == D0 else "qaoa_D2"
    n_rows = sorted([r for r in ok if r["arm"] == nov and r["L"] == L], key=lambda r: r["seed"])
    q_rows = sorted([r for r in ok if r["arm"] == qa and r["L"] == L], key=lambda r: r["seed"])
    if not (len(n_rows) == len(q_rows) == 8):
        return {"L": L, "nf": nf, "incomplete": True, "n_nov3": len(n_rows), "n_qaoa": len(q_rows)}
    e_n = np.array([r["E"] for r in n_rows])
    e_q = np.array([r["E"] for r in q_rows])
    d = _boot(e_n - e_q, rng)
    leaks = [r["leakage"]["p_leak"] for r in n_rows]
    mean_ns = [r["leakage"]["mean_n_per_cavity"] for r in n_rows]
    p_high = [r["leakage"]["p_highest_any_cavity"] for r in n_rows]
    return {
        "L": L,
        "nf": nf,
        "incomplete": False,
        "n": 8,
        "Q": Q_PER_LAYER * L,
        "d_nov3Q": d,
        "nov3_vs_qaoa_wins": int(np.sum(e_n < e_q)),
        "E_nov3": _boot(e_n, rng),
        "E_qaoa": _boot(e_q, rng),
        "F_nov3": _boot(np.array([r["F"] for r in n_rows]), rng),
        "F_qaoa": _boot(np.array([r["F"] for r in q_rows]), rng),
        "p_leak_nov3": _boot(np.array(leaks), rng),
        "mean_n_nov3": _boot(np.array(mean_ns), rng),
        "p_highest_nov3": _boot(np.array(p_high), rng),
        "start0_nov3": int(sum(r["start_id"] == 0 for r in n_rows)),
        "start0_qaoa": int(sum(r["start_id"] == 0 for r in q_rows)),
        "seeds": [int(r["seed"]) for r in n_rows],
        "ci_entirely_lt0": bool(d["mean"] < 0.0 and d["ci95"][1] < 0.0),
        "mean_dE": d["mean"],
    }


def summarize() -> None:
    rows = [json.loads(p.read_text()) for p in CELLS.glob("*.json")]
    fails = [r for r in rows if not r.get("ok")]
    ok = [r for r in rows if r.get("ok")]
    inst = [json.loads(p.read_text()) for p in INST.glob("N15_s*.json")]
    rng = np.random.default_rng(3334)
    g1_d0 = _pair(ok, 1, D0, rng)
    g1_d2 = _pair(ok, 1, D0_PLUS_2, rng)
    g2_d0 = _pair(ok, 2, D0, rng)
    g2_d2 = _pair(ok, 2, D0_PLUS_2, rng)
    complete = all(not g.get("incomplete") for g in (g1_d0, g1_d2, g2_d0, g2_d2))
    primary = bool(
        complete
        and not fails
        and g1_d0.get("ci_entirely_lt0")
        and g1_d2.get("ci_entirely_lt0")
    )
    d_shift = None
    if complete:
        d_shift = abs(float(g1_d2["mean_dE"]) - float(g1_d0["mean_dE"]))
    l2_sign = None
    if complete:
        s0 = np.sign(g2_d0["mean_dE"]) if abs(g2_d0["mean_dE"]) > 1e-15 else 0.0
        s2 = np.sign(g2_d2["mean_dE"]) if abs(g2_d2["mean_dE"]) > 1e-15 else 0.0
        l2_sign = bool(s0 == s2)
    label = "CUTOFF_ROBUST_L1" if primary else "CUTOFF_SENSITIVE"
    new_q = int(sum(r["queries"]["total"] for r in ok))
    payload = {
        "version": "N3INIT_cutoff",
        "no_gr_patch": True,
        "no_candidacy_file": True,
        "holdout_fail_stands": True,
        "ideal_W_L_only": True,
        "compiled_W_L_not_modelled": True,
        "D0": D0,
        "D0_plus_2": D0_PLUS_2,
        "D0_plus_4_not_run": True,
        "no_refit_10368_10375": True,
        "n_files": len(rows),
        "n_fail": len(fails),
        "failures": [{"arm": r["arm"], "seed": r["seed"], "L": r["L"], "nf": r.get("nf"), "failure": r["failure"]} for r in fails],
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
        "L1_D0": g1_d0,
        "L1_D2": g1_d2,
        "L2_D0": g2_d0,
        "L2_D2": g2_d2,
        "gates": {
            "L1_D0_ci_entirely_lt0": g1_d0.get("ci_entirely_lt0"),
            "L1_D2_ci_entirely_lt0": g1_d2.get("ci_entirely_lt0"),
            "L1_abs_mean_dE_shift": d_shift,
            "L2_sign_concordance": l2_sign,
        },
    }
    _dump(RES / "summary.json", payload)
    lines = [
        "# N3INIT Fock-cutoff validation N=15 L=1,2",
        "",
        f"Frozen v4 center-4. D0=NF=8, D0+2=NF=10, ideal W_L pad. Compiled W_L not modelled. Seeds 10520–10527. No GR patch.",
        "",
        f"**{label}.**",
        "",
        f"New quantum queries {new_q}. Historical {HISTORICAL}. Total research queries {HISTORICAL + new_q}. Failures {len(fails)}.",
        f"Classical B* wall {payload['classical_bstar']['wall_seconds']:.1f}s, CPU {payload['classical_bstar']['cpu_seconds']:.1f}s, peak RSS {payload['classical_bstar']['peak_rss_kib_max']} KiB.",
        "",
        f"Secondary |mean ΔE(D0+2)−mean ΔE(D0)| at L=1: {d_shift if d_shift is not None else 'n/a'}.",
        f"L=2 sign concordance (exploratory): {l2_sign}.",
        "",
        "p_leak is 1−P(all cavities in Fock 0..7). At D0=8, p_highest is occupancy of code level 7, not pad leakage. At D0+2, p_highest is occupancy of Fock 9 (the pad edge). Compiled W_L is not modelled.",
        "",
    ]
    for tag, g in (("L=1 D0=8 primary", g1_d0), ("L=1 D0+2=10 primary", g1_d2), ("L=2 D0=8 exploratory", g2_d0), ("L=2 D0+2=10 exploratory", g2_d2)):
        if g.get("incomplete"):
            lines.append(f"{tag}: incomplete {g}")
            continue
        d = g["d_nov3Q"]
        lines += [
            f"## {tag}",
            "",
            f"Nov3 vs QAOA mean ΔE {d['mean']:+.4f} CI [{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}] wins {g['nov3_vs_qaoa_wins']}/{g['n']}. CI<0 {g['ci_entirely_lt0']}.",
            f"mean E Nov3 {g['E_nov3']['mean']:.4f} QAOA {g['E_qaoa']['mean']:.4f}. mean F Nov3 {g['F_nov3']['mean']:.3f} QAOA {g['F_qaoa']['mean']:.3f}.",
            f"Nov3 p_leak {g['p_leak_nov3']['mean']:.3e} mean n/cavity {g['mean_n_nov3']['mean']:.4f} p_highest {g['p_highest_nov3']['mean']:.3e}. start0 Nov3 {g['start0_nov3']}/{g['n']} QAOA {g['start0_qaoa']}/{g['n']}.",
            "",
        ]
    if fails:
        lines += ["## Blockers", ""]
        for r in fails:
            lines.append(f"- {r['arm']} L={r['L']} nf={r.get('nf')} seed={r['seed']}: {r['failure']}")
        lines.append("")
    lines.append("Fit workers 4, BLAS 1. Each completed ledger equals Q=1024·L. Ideal W_L only; no hardware claim.")
    (RES / "RESULTS.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", required=True, choices=("bstar", "run", "summary"))
    args = parser.parse_args()
    if not PREREG.exists():
        raise RuntimeError("cutoff preregistration missing")
    _conf()
    if args.phase == "bstar":
        build_instances()
    elif args.phase == "run":
        run_fits()
    else:
        summarize()


if __name__ == "__main__":
    main()
