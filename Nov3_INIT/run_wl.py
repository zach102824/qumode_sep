"""N=15 L=1 non-ideal W_L leakage probe. Frozen v4 center-4. NF=10."""
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
from Nov3_INIT.fock import D0_PLUS_2, EPS_WL, FockNov3
from Nov3_INIT.run_v1 import _dump
from Nov9_SK_SPSA.experiments.FT.instance import make_J, packing

ROOT = Path(__file__).resolve().parent
PREREG = ROOT / "PREREG_WL.md"
RES = ROOT / "results" / "wl"
CELLS = RES / "cells"
INST = RES / "instances"
CONF = RES / "CONFORMANCE.json"
BAND = list(range(10528, 10536))
N = 15
L = 1
Q_PER_LAYER = 1024
N_BOOT = 4000
HISTORICAL = 2793472
CELL_WALL_CAP = 1800.0
CUT = list(range(7))
NF = D0_PLUS_2
ARMS = (
    ("nov3_ideal", "Nov3rd", METHOD_ID["equal4"]["Nov3rd"], 0.0, "rewrite"),
    ("nov3_wl", "Nov3rd", METHOD_ID["equal4"]["WL"], EPS_WL, "hardware"),
    ("qaoa_wl", "QAOA", METHOD_ID["equal4"]["QAOA"], EPS_WL, "hardware"),
)


def prereg_sha() -> str:
    return hashlib.sha256(PREREG.read_bytes()).hexdigest()


def allow(seed: int) -> None:
    seed = int(seed)
    if seed not in BAND:
        raise RuntimeError(f"seed {seed} is not in the W_L band 10528..10535")
    if 10504 <= seed <= 10519:
        raise RuntimeError(f"local R30 seed {seed}")
    if 10520 <= seed <= 10527:
        raise RuntimeError(f"FT cutoff seed {seed}")
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
        raise RuntimeError("W_L conformance does not match the current preregistration")
    if doc.get("collisions_in_nov3_init_results", {}).get("n", 1) != 0:
        raise RuntimeError("W_L seeds already appear under Nov3_INIT/results")
    if packing(N) != [[a, a + 5, a + 10] for a in range(5)]:
        raise RuntimeError("N=15 packing is not interleaved m=5")
    if abs(float(doc.get("eps", -1.0)) - EPS_WL) > 1e-15 or int(doc.get("nf", -1)) != NF:
        raise RuntimeError("W_L ε/NF mismatch")
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
        "wl_band": True,
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
    return CELLS / f"{arm}_{method}_N{N}_s{seed}_L{L}_q{Q_PER_LAYER}_nf{NF}.json"


def _one(task: dict) -> dict:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    arm, method = task["arm"], task["method"]
    mid, eps, order = int(task["method_id"]), float(task["eps"]), task["order"]
    seed = int(task["seed"])
    allow(seed)
    path = _path(arm, method, seed)
    if path.exists():
        return {"skipped": True}
    Q = Q_PER_LAYER * L
    t0 = time.perf_counter()
    cpu0 = time.process_time()
    try:
        meta, logical = _logical(seed)
        b = float(meta["Bstar"])
        rng_init = [None if s == 0 else make_rng(seed, N, b, L, mid, s, 2) for s in range(4)]
        rng_spsa = [make_rng(seed, N, b, L, mid, s, 0) for s in range(4)]
        fock = FockNov3(logical, NF, eps=eps, order=order)

        def draw(method_name, ell, start_id, rng, _method=method):
            if _method == "QAOA":
                return x0_qaoa_center4(ell, start_id, rng)
            return x0_center4(method_name, ell, start_id, rng)

        def propagate(x, _fock=fock, _method=method, _t0=t0):
            if time.perf_counter() - _t0 > CELL_WALL_CAP:
                raise RuntimeError(f"cell wall exceeded {CELL_WALL_CAP}s")
            if _method == "QAOA":
                return _fock.propagate_qaoa(x, L)
            return _fock.propagate(x, L)

        out = run_equal4(propagate, method, L, Q, rng_init, rng_spsa, x0_fn=draw)
        psi = out.pop("aux")
        f = fock.fidelity(psi)
        leak = fock.leakage(psi)
        f_plus = float(meta["F_plus"])
        e0 = float(meta["E0"])
        record = {
            "ok": True,
            "failure": None,
            "blocker": None,
            "version": "N3INIT_wl",
            "arm": arm,
            "method": method,
            "method_id": mid,
            "eps": eps,
            "order": order,
            "nf": NF,
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
            "channel": "real Givens 7-8 after ideal W_BinH; SNAP phase 0 on f>=8",
            "compiled_snap_d_still_unmodelled": True,
            "rng_omits_eps": True,
        }
    except Exception as exc:
        kind = "wall" if "wall exceeded" in repr(exc) else "cell_infeasible"
        record = {
            "ok": False,
            "failure": repr(exc),
            "blocker": kind,
            "version": "N3INIT_wl",
            "arm": arm,
            "method": method,
            "eps": eps,
            "order": order,
            "nf": NF,
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
    print(f"{'ok' if record['ok'] else 'FAIL'} wl {arm} N=15 s={seed} L={L}{extra} {record['seconds']:.1f}s", flush=True)
    return record


def run_fits() -> None:
    import multiprocessing as mp

    _conf()
    sha = prereg_sha()
    pending = []
    for seed in BAND:
        for arm, method, mid, eps, order in ARMS:
            if not _path(arm, method, seed).exists():
                pending.append({
                    "arm": arm,
                    "method": method,
                    "method_id": mid,
                    "eps": eps,
                    "order": order,
                    "seed": seed,
                    "prereg_sha256": sha,
                })
    print(f"wl pending {len(pending)}", flush=True)
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


def _arm(ok: list[dict], arm: str) -> list[dict]:
    return sorted([r for r in ok if r["arm"] == arm], key=lambda r: r["seed"])


def _pair(left: list[dict], right: list[dict], rng: np.random.Generator, tag: str) -> dict:
    if not (len(left) == len(right) == 8):
        return {"tag": tag, "incomplete": True, "n_left": len(left), "n_right": len(right)}
    e_l = np.array([r["E"] for r in left])
    e_r = np.array([r["E"] for r in right])
    d = _boot(e_l - e_r, rng)
    return {
        "tag": tag,
        "incomplete": False,
        "n": 8,
        "Q": Q_PER_LAYER * L,
        "dE": d,
        "left_vs_right_wins": int(np.sum(e_l < e_r)),
        "E_left": _boot(e_l, rng),
        "E_right": _boot(e_r, rng),
        "F_left": _boot(np.array([r["F"] for r in left]), rng),
        "F_right": _boot(np.array([r["F"] for r in right]), rng),
        "p_leak_left": _boot(np.array([r["leakage"]["p_leak"] for r in left]), rng),
        "p_leak_right": _boot(np.array([r["leakage"]["p_leak"] for r in right]), rng),
        "start0_left": int(sum(r["start_id"] == 0 for r in left)),
        "start0_right": int(sum(r["start_id"] == 0 for r in right)),
        "seeds": [int(r["seed"]) for r in left],
        "ci_entirely_lt0": bool(d["mean"] < 0.0 and d["ci95"][1] < 0.0),
        "mean_dE": d["mean"],
    }


def summarize() -> None:
    rows = [json.loads(p.read_text()) for p in CELLS.glob("*.json")]
    fails = [r for r in rows if not r.get("ok")]
    ok = [r for r in rows if r.get("ok")]
    inst = [json.loads(p.read_text()) for p in INST.glob("N15_s*.json")]
    rng = np.random.default_rng(3334)
    a = _arm(ok, "nov3_ideal")
    b = _arm(ok, "nov3_wl")
    c = _arm(ok, "qaoa_wl")
    primary = _pair(b, c, rng, "nov3_wl_minus_qaoa_wl")
    secondary = _pair(b, a, rng, "nov3_wl_minus_nov3_ideal")
    complete = not primary.get("incomplete") and not secondary.get("incomplete")
    label = "WL_ROBUST_L1" if complete and not fails and primary.get("ci_entirely_lt0") else "WL_SENSITIVE"
    new_q = int(sum(r["queries"]["total"] for r in ok))
    payload = {
        "version": "N3INIT_wl",
        "no_gr_patch": True,
        "no_candidacy_file": True,
        "holdout_fail_stands": True,
        "cutoff_robust_l1_stands": True,
        "not_a_hardware_physical_edge_claim": True,
        "compiled_snap_d_still_unmodelled": True,
        "eps": EPS_WL,
        "nf": NF,
        "channel": "real Givens 7-8 after ideal W_BinH; SNAP phase 0 on f>=8",
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
        "primary_nov3_wl_vs_qaoa_wl": primary,
        "secondary_nov3_wl_vs_nov3_ideal": secondary,
        "p_leak_nov3_ideal": _boot(np.array([r["leakage"]["p_leak"] for r in a]), rng) if len(a) == 8 else None,
        "p_leak_nov3_wl": _boot(np.array([r["leakage"]["p_leak"] for r in b]), rng) if len(b) == 8 else None,
        "p_leak_qaoa_wl": _boot(np.array([r["leakage"]["p_leak"] for r in c]), rng) if len(c) == 8 else None,
        "gates": {
            "primary_ci_entirely_lt0": primary.get("ci_entirely_lt0"),
            "complete": complete,
        },
    }
    _dump(RES / "summary.json", payload)
    lines = [
        "# N3INIT non-ideal W_L leakage probe N=15 L=1",
        "",
        "Frozen v4 center-4. NF=10. Phenomenological W_ε = G_{7,8}(ε=0.10) W_BinH, not compiled-W. SNAP phase 0 on f≥8. Seeds 10528–10535. No GR patch. Not a hardware claim.",
        "",
        f"**{label}.**",
        "",
        f"New quantum queries {new_q}. Historical {HISTORICAL}. Total research queries {HISTORICAL + new_q}. Failures {len(fails)}.",
        f"Classical B* wall {payload['classical_bstar']['wall_seconds']:.1f}s, CPU {payload['classical_bstar']['cpu_seconds']:.1f}s, peak RSS {payload['classical_bstar']['peak_rss_kib_max']} KiB.",
        "",
    ]
    if primary.get("incomplete"):
        lines.append(f"Primary incomplete {primary}")
    else:
        d = primary["dE"]
        lines += [
            "## Primary: G_7,8(0.10) Nov3 vs QAOA (same illustrative embedding)",
            "",
            f"Nov3−QAOA mean ΔE {d['mean']:+.4f} CI [{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}] wins {primary['left_vs_right_wins']}/{primary['n']}. CI<0 {primary['ci_entirely_lt0']}.",
            f"mean E Nov3 {primary['E_left']['mean']:.4f} QAOA {primary['E_right']['mean']:.4f}. mean F Nov3 {primary['F_left']['mean']:.3f} QAOA {primary['F_right']['mean']:.3f}.",
            f"p_leak Nov3 {primary['p_leak_left']['mean']:.3e} QAOA {primary['p_leak_right']['mean']:.3e}. start0 Nov3 {primary['start0_left']}/{primary['n']} QAOA {primary['start0_right']}/{primary['n']}.",
            "",
        ]
    if secondary.get("incomplete"):
        lines.append(f"Secondary incomplete {secondary}")
    else:
        d = secondary["dE"]
        lines += [
            "## Secondary: non-ideal Nov3 vs ideal Nov3",
            "",
            f"non-ideal−ideal mean ΔE {d['mean']:+.4f} CI [{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}] (Nov3-only; no ideal QAOA arm; equal p_leak is not equal energy bias).",
            f"mean E non-ideal {secondary['E_left']['mean']:.4f} ideal {secondary['E_right']['mean']:.4f}.",
            f"p_leak non-ideal {secondary['p_leak_left']['mean']:.3e} ideal {secondary['p_leak_right']['mean']:.3e}.",
            "",
        ]
    pl_a = payload["p_leak_nov3_ideal"]
    pl_b = payload["p_leak_nov3_wl"]
    pl_c = payload["p_leak_qaoa_wl"]
    if pl_a and pl_b and pl_c:
        lines += [
            "## p_leak",
            "",
            f"ideal Nov3 {pl_a['mean']:.3e} [{pl_a['ci95'][0]:.3e}, {pl_a['ci95'][1]:.3e}].",
            f"non-ideal Nov3 {pl_b['mean']:.3e} [{pl_b['ci95'][0]:.3e}, {pl_b['ci95'][1]:.3e}].",
            f"non-ideal QAOA {pl_c['mean']:.3e} [{pl_c['ci95'][0]:.3e}, {pl_c['ci95'][1]:.3e}].",
            "",
        ]
    if fails:
        lines += ["## Blockers", ""]
        for r in fails:
            lines.append(f"- {r['arm']} seed={r['seed']}: {r['failure']}")
        lines.append("")
    lines.append("Fit workers 4, BLAS 1. Each completed ledger equals Q=1024. Phenomenological G_7,8(0.10) only. SNAP–D compile of W_L remains unmodelled. No hardware physical-edge claim.")
    (RES / "RESULTS.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", required=True, choices=("bstar", "run", "summary"))
    args = parser.parse_args()
    if not PREREG.exists():
        raise RuntimeError("W_L preregistration missing")
    _conf()
    if args.phase == "bstar":
        build_instances()
    elif args.phase == "run":
        run_fits()
    else:
        summarize()


if __name__ == "__main__":
    main()
