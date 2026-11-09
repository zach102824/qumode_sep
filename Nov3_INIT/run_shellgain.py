"""N=15 L=3 shell-emphasized SPSA vs joint Nov3 vs QAOA. Q=3072 each."""
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

from Nov3_INIT.alloc import METHOD_ID, make_rng, run_equal4, shell_a_mult, x0_center4, x0_qaoa_center4
from Nov3_INIT.bstar import maximize
from Nov3_INIT.circuit import Logical
from Nov3_INIT.run_v1 import _dump
from Nov9_SK_SPSA.experiments.FT.instance import make_J

ROOT = Path(__file__).resolve().parent
PREREG = ROOT / "PREREG_SHELLGAIN.md"
RES = ROOT / "results" / "shellgain"
CELLS = RES / "cells"
INST = RES / "instances"
CONF = RES / "CONFORMANCE.json"
BAND = list(range(10368, 10376))
N = 15
L = 3
Q = 3072
N_BOOT = 4000
HISTORICAL = 2301952
CELL_WALL_CAP = 1800.0
CUT = list(range(7))
ARMS = (
    ("JOINT", "Nov3rd", 1.0, METHOD_ID["equal4"]["Nov3rd"]),
    ("SHELL2", "Nov3rd", 2.0, METHOD_ID["equal4"]["SHELL2"]),
    ("SHELL4", "Nov3rd", 4.0, METHOD_ID["equal4"]["SHELL4"]),
    ("qaoa_center4", "QAOA", 1.0, METHOD_ID["equal4"]["QAOA"]),
)


def prereg_sha() -> str:
    return hashlib.sha256(PREREG.read_bytes()).hexdigest()


def allow(seed: int) -> None:
    seed = int(seed)
    if seed not in BAND:
        raise RuntimeError(f"seed {seed} is not in the shell-gain band 10368..10375")
    if 10360 <= seed <= 10367:
        raise RuntimeError(f"LE two-stage seed {seed}")
    if 10344 <= seed <= 10359:
        raise RuntimeError(f"local R25 seed {seed}")
    if seed <= 10335:
        raise RuntimeError(f"prior FT/Nov3_INIT seed {seed}")
    if 10144 <= seed <= 10151:
        raise RuntimeError(f"holdout seed {seed}")
    if 11000 <= seed <= 11031 or 20000 <= seed <= 20999:
        raise RuntimeError(f"blind seed {seed}")


def _conf() -> dict:
    doc = json.loads(CONF.read_text())
    if doc.get("prereg_sha256") != prereg_sha() or not doc.get("ok"):
        raise RuntimeError("shell-gain conformance does not match the current preregistration")
    if doc.get("collisions_in_nov3_init_results", {}).get("n", 1) != 0:
        raise RuntimeError("shell-gain seeds already appear under Nov3_INIT/results")
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
        "shellgain_band": True,
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
    seed = int(task["seed"])
    ratio = float(task["ratio"])
    mid = int(task["method_id"])
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
        a_mult = None if method == "QAOA" else shell_a_mult(L, ratio)

        def draw(method_name, ell, start_id, rng, _method=method):
            if _method == "QAOA":
                return x0_qaoa_center4(ell, start_id, rng)
            return x0_center4(method_name, ell, start_id, rng)

        def propagate(x, _logical=logical, _method=method, _t0=t0):
            if time.perf_counter() - _t0 > CELL_WALL_CAP:
                raise RuntimeError(f"cell wall exceeded {CELL_WALL_CAP}s")
            return _logical.propagate(_method, L, x)

        out = run_equal4(propagate, method, L, Q, rng_init, rng_spsa, x0_fn=draw, a_mult=a_mult)
        psi = out.pop("aux")
        f = logical.fidelity(psi)
        f_plus = float(meta["F_plus"])
        e0 = float(meta["E0"])
        record = {
            "ok": True,
            "failure": None,
            "blocker": None,
            "version": "N3INIT_shellgain",
            "arm": arm,
            "method": method,
            "method_id": mid,
            "shell_a_ratio": ratio if method == "Nov3rd" else None,
            "a_mult": out.get("a_mult"),
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
            "not_le_two_stage": True,
            "shared_uc_only": True,
        }
    except Exception as exc:
        kind = "wall" if "wall exceeded" in repr(exc) else "cell_infeasible"
        record = {
            "ok": False,
            "failure": repr(exc),
            "blocker": kind,
            "version": "N3INIT_shellgain",
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
    print(f"{'ok' if record['ok'] else 'FAIL'} shellgain {arm} N=15 s={seed} L=3 q=1024{extra} {record['seconds']:.1f}s", flush=True)
    return record


def run_fits() -> None:
    import multiprocessing as mp

    _conf()
    sha = prereg_sha()
    pending = []
    for seed in BAND:
        for arm, method, ratio, mid in ARMS:
            if not _path(arm, method, seed).exists():
                pending.append({"arm": arm, "method": method, "seed": seed, "ratio": ratio, "method_id": mid, "prereg_sha256": sha})
    print(f"shellgain pending {len(pending)}", flush=True)
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


def _arm(ok: list[dict], name: str) -> list[dict]:
    return sorted([r for r in ok if r["arm"] == name], key=lambda r: r["seed"])


def summarize() -> None:
    rows = [json.loads(p.read_text()) for p in CELLS.glob("*.json")]
    fails = [r for r in rows if not r.get("ok")]
    ok = [r for r in rows if r.get("ok")]
    inst = [json.loads(p.read_text()) for p in INST.glob("N15_s*.json")]
    rng = np.random.default_rng(3334)
    joint = _arm(ok, "JOINT")
    s2 = _arm(ok, "SHELL2")
    s4 = _arm(ok, "SHELL4")
    qa = _arm(ok, "qaoa_center4")
    complete = len(joint) == len(s2) == len(s4) == len(qa) == 8
    winner = None
    primary_ok = False
    secondary_ok = False
    stats = None
    if complete:
        e_j = np.array([r["E"] for r in joint])
        e2 = np.array([r["E"] for r in s2])
        e4 = np.array([r["E"] for r in s4])
        e_q = np.array([r["E"] for r in qa])
        mean2, mean4 = float(e2.mean()), float(e4.mean())
        winner = "SHELL2" if mean2 <= mean4 else "SHELL4"
        e_w = e2 if winner == "SHELL2" else e4
        d_wq = _boot(e_w - e_q, rng)
        primary_ok = bool(d_wq["mean"] < 0.0 and d_wq["ci95"][1] < 0.0)
        secondary_ok = bool(float(e_w.mean()) <= float(e_j.mean()))
        stats = {
            "winner": winner,
            "d_winner_qaoa": d_wq,
            "d_shell2_qaoa": _boot(e2 - e_q, rng),
            "d_shell4_qaoa": _boot(e4 - e_q, rng),
            "d_winner_joint": _boot(e_w - e_j, rng),
            "wins_winner_qaoa": int(np.sum(e_w < e_q)),
            "E_JOINT": _boot(e_j, rng),
            "E_SHELL2": _boot(e2, rng),
            "E_SHELL4": _boot(e4, rng),
            "E_QAOA": _boot(e_q, rng),
            "F_JOINT": _boot(np.array([r["F"] for r in joint]), rng),
            "F_SHELL2": _boot(np.array([r["F"] for r in s2]), rng),
            "F_SHELL4": _boot(np.array([r["F"] for r in s4]), rng),
            "F_QAOA": _boot(np.array([r["F"] for r in qa]), rng),
            "start0": {
                "JOINT": int(sum(r["start_id"] == 0 for r in joint)),
                "SHELL2": int(sum(r["start_id"] == 0 for r in s2)),
                "SHELL4": int(sum(r["start_id"] == 0 for r in s4)),
                "QAOA": int(sum(r["start_id"] == 0 for r in qa)),
            },
            "seeds": [int(r["seed"]) for r in joint],
            "primary_ci_lt0": primary_ok,
            "secondary_winner_le_joint": secondary_ok,
        }
    if fails or not complete:
        label = "REJECT_SHELL_GAIN"
    elif primary_ok and secondary_ok:
        label = "SHELL_GAIN_L3_EDGE"
    else:
        label = "REJECT_SHELL_GAIN"
    new_q = int(sum(r["queries"]["total"] for r in ok))
    payload = {
        "version": "N3INIT_shellgain",
        "no_gr_patch": True,
        "no_candidacy_file": True,
        "holdout_fail_stands": True,
        "pair_shell_reject_stands": True,
        "not_le_two_stage": True,
        "not_paper_V_M": True,
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
        "# N3INIT shell-emphasized SPSA N=15 L=3",
        "",
        "Shared-Uc Nov3 only. Not LE two-stage. Not paper V(M). Seeds 10368–10375. Q=3072 each. No GR patch. Holdout FAIL stands.",
        "",
        f"**{label}.**",
        "",
        f"New quantum queries {new_q}. Historical {HISTORICAL}. Total research queries {HISTORICAL + new_q}. Failures {len(fails)}.",
        f"Classical B* wall {payload['classical_bstar']['wall_seconds']:.1f}s, CPU {payload['classical_bstar']['cpu_seconds']:.1f}s, peak RSS {payload['classical_bstar']['peak_rss_kib_max']} KiB.",
        "",
    ]
    if stats:
        d = stats["d_winner_qaoa"]
        lines += [
            f"Winner {stats['winner']}. vs QAOA mean ΔE {d['mean']:+.4f} CI [{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}] wins {stats['wins_winner_qaoa']}/8.",
            f"mean E JOINT {stats['E_JOINT']['mean']:.4f} SHELL2 {stats['E_SHELL2']['mean']:.4f} SHELL4 {stats['E_SHELL4']['mean']:.4f} QAOA {stats['E_QAOA']['mean']:.4f}.",
            f"mean F JOINT {stats['F_JOINT']['mean']:.3f} SHELL2 {stats['F_SHELL2']['mean']:.3f} SHELL4 {stats['F_SHELL4']['mean']:.3f} QAOA {stats['F_QAOA']['mean']:.3f}.",
            f"start0 JOINT {stats['start0']['JOINT']}/8 SHELL2 {stats['start0']['SHELL2']}/8 SHELL4 {stats['start0']['SHELL4']}/8 QAOA {stats['start0']['QAOA']}/8.",
            f"Primary CI<0 {stats['primary_ci_lt0']}. Secondary winner≤JOINT {stats['secondary_winner_le_joint']}.",
            "",
        ]
        d2, d4 = stats["d_shell2_qaoa"], stats["d_shell4_qaoa"]
        lines.append(f"SHELL2 vs QAOA {d2['mean']:+.4f} [{d2['ci95'][0]:+.3f}, {d2['ci95'][1]:+.3f}]. SHELL4 vs QAOA {d4['mean']:+.4f} [{d4['ci95'][0]:+.3f}, {d4['ci95'][1]:+.3f}].")
        lines.append("")
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
        raise RuntimeError("shell-gain preregistration missing")
    _conf()
    if args.phase == "bstar":
        build_instances()
    elif args.phase == "run":
        run_fits()
    else:
        summarize()


if __name__ == "__main__":
    main()
