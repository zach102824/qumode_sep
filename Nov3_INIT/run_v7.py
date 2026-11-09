"""N=15 L=2,3 paired development. Exact v4 recipes. Reuses v4 B*."""
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
from Nov3_INIT.circuit import Logical
from Nov3_INIT.run_v1 import TEST, TRAIN, _dump, allow

ROOT = Path(__file__).resolve().parent
PREREG = ROOT / "PREREG_V7.md"
RES = ROOT / "results" / "v7"
CELLS = RES / "cells"
CONF = RES / "CONFORMANCE.json"
V4_INST = ROOT / "results" / "v4" / "instances"
SEEDS = TRAIN + TEST
LS = (2, 3)
QS = (512, 1024)
N_BOOT = 4000
HISTORICAL = 1826816
CELL_WALL_CAP = 1800.0
CUT = list(range(7))


def prereg_sha() -> str:
    return hashlib.sha256(PREREG.read_bytes()).hexdigest()


def _conf() -> dict:
    doc = json.loads(CONF.read_text())
    if doc.get("prereg_sha256") != prereg_sha() or not doc.get("ok"):
        raise RuntimeError("v7 conformance does not match the current preregistration")
    if not doc.get("probe", {}).get("n15_l23_allowed"):
        raise RuntimeError("N=15 L=2/3 resource gate closed")
    if doc.get("probe", {}).get("bstar_recomputed"):
        raise RuntimeError("v7 probe recomputed B*")
    if int(doc.get("probe", {}).get("n_instances_valid", 0)) != 8:
        raise RuntimeError("v7 probe did not validate all eight reused B* files")
    return doc


def _logical(seed: int):
    meta = json.loads((V4_INST / f"N15_s{seed}.json").read_text())
    gs = np.load(V4_INST / f"N15_s{seed}.npz")["gs"]
    if int(meta["N"]) != 15 or int(meta["seed"]) != int(seed):
        raise RuntimeError(f"instance identity mismatch seed {seed}")
    if meta.get("cut") != CUT or gs.size != (1 << 15):
        raise RuntimeError(f"reused B* invalid seed {seed}")
    return meta, Logical(15, seed, float(meta["Bstar"]), gs)


def _path(arm: str, method: str, seed: int, L: int, q: int) -> Path:
    return CELLS / f"{arm}_{method}_N15_s{seed}_L{L}_q{q}.json"


def _one(task: dict) -> dict:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    arm, method = task["arm"], task["method"]
    seed, L, q = int(task["seed"]), int(task["L"]), int(task["q"])
    allow(seed)
    path = _path(arm, method, seed, L, q)
    if path.exists():
        return {"skipped": True}
    t0 = time.perf_counter()
    cpu0 = time.process_time()
    try:
        meta, logical = _logical(seed)
        b = float(meta["Bstar"])
        mid = 0 if method == "QAOA" else 1
        Q = q * L
        rng_init = [None if s == 0 else make_rng(seed, 15, b, L, mid, s, 2) for s in range(4)]
        rng_spsa = [make_rng(seed, 15, b, L, mid, s, 0) for s in range(4)]

        def draw(method_name, ell, start_id, rng, _arm=arm):
            if _arm == "qaoa_center4":
                return x0_qaoa_center4(ell, start_id, rng)
            return x0_center4(method_name, ell, start_id, rng)

        def propagate(x, _logical=logical, _method=method, _L=L, _t0=t0):
            if time.perf_counter() - _t0 > CELL_WALL_CAP:
                raise RuntimeError(f"cell wall exceeded {CELL_WALL_CAP}s")
            return _logical.propagate(_method, _L, x)

        out = run_equal4(propagate, method, L, Q, rng_init, rng_spsa, x0_fn=draw)
        psi = out.pop("aux")
        f = logical.fidelity(psi)
        f_plus = float(meta["F_plus"])
        e0 = float(meta["E0"])
        record = {
            "ok": True,
            "failure": None,
            "blocker": None,
            "version": "N3INIT_v7",
            "candidate_not_benchmark_adoption": True,
            "arm": arm,
            "method": method,
            "method_id": mid,
            "N": 15,
            "seed": seed,
            "L": L,
            "q": q,
            "Q": Q,
            "procedural_cap": Q,
            "Bstar": b,
            "Bstar_reused_from_v4": True,
            "E0": e0,
            "E_plus": meta["E_plus"],
            "F_plus": f_plus,
            "E": out["E"],
            "E_error_over_N": float((out["E"] - e0) / 15.0),
            "F": f,
            "eta": float((f - f_plus) / (1.0 - f_plus)),
            "escaped": bool(out["E"] < meta["E_plus"] - 1e-3 * 15),
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
            "not_le_s1_or_local_r10": True,
        }
    except Exception as exc:
        kind = "wall" if "wall exceeded" in repr(exc) else "cell_infeasible"
        record = {
            "ok": False,
            "failure": repr(exc),
            "blocker": kind,
            "version": "N3INIT_v7",
            "arm": arm,
            "method": method,
            "N": 15,
            "seed": seed,
            "L": L,
            "q": q,
            "Q": q * L,
            "seconds": time.perf_counter() - t0,
            "cpu_seconds": time.process_time() - cpu0,
            "peak_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
            "prereg_sha256": task["prereg_sha256"],
            "physics_not_approximated": True,
        }
    _dump(path, record)
    extra = f" E={record['E']:.4f} F={record['F']:.3f}" if record["ok"] else " " + record["failure"]
    print(f"{'ok' if record['ok'] else 'FAIL'} v7 {arm} N=15 s={seed} L={L} q={q}{extra} {record['seconds']:.1f}s", flush=True)
    return record


def _pending() -> list[dict]:
    sha = prereg_sha()
    rows = []
    for L in LS:
        for q in QS:
            for seed in SEEDS:
                for arm, method in (("nov3_center4", "Nov3rd"), ("qaoa_center4", "QAOA")):
                    if not _path(arm, method, seed, L, q).exists():
                        rows.append({"arm": arm, "method": method, "seed": seed, "L": L, "q": q, "prereg_sha256": sha})
    return rows


def run_fits() -> None:
    import multiprocessing as mp

    _conf()
    wall0 = time.perf_counter()
    pending = _pending()
    print(f"v7 pending {len(pending)}", flush=True)
    if pending:
        with mp.get_context("fork").Pool(4) as pool:
            list(pool.imap_unordered(_one, pending, chunksize=1))
    (RES / "wall.json").write_text(
        json.dumps({"fit_seconds": time.perf_counter() - wall0, "workers": 4, "blas_threads": 1, "bstar_reused": True}) + "\n"
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
        return "nov3_lower_ci_excludes_0"
    if stat["mean"] < 0.0:
        return "negative_point_ci_includes_0"
    if lo > 0.0:
        return "qaoa_lower_ci_excludes_0"
    return "positive_point_ci_includes_0"


def summarize() -> None:
    rows = [json.loads(p.read_text()) for p in CELLS.glob("*.json")]
    fails = [r for r in rows if not r.get("ok")]
    ok = [r for r in rows if r.get("ok")]
    rng = np.random.default_rng(3334)
    groups = []
    for L in LS:
        for q in QS:
            nov = sorted([r for r in ok if r["arm"] == "nov3_center4" and r["L"] == L and r["q"] == q], key=lambda r: r["seed"])
            qa = sorted([r for r in ok if r["arm"] == "qaoa_center4" and r["L"] == L and r["q"] == q], key=lambda r: r["seed"])
            if not nov or not qa or len(nov) != len(qa):
                groups.append({
                    "N": 15,
                    "L": L,
                    "q": q,
                    "Q": q * L,
                    "n": min(len(nov), len(qa)),
                    "incomplete": True,
                    "n_nov3": len(nov),
                    "n_qaoa": len(qa),
                })
                continue
            e_n = np.array([r["E"] for r in nov])
            e_q = np.array([r["E"] for r in qa])
            err_n = np.array([r["E_error_over_N"] for r in nov])
            err_q = np.array([r["E_error_over_N"] for r in qa])
            f_n = np.array([r["F"] for r in nov])
            f_q = np.array([r["F"] for r in qa])
            d_pair = _boot(e_n - e_q, rng)
            groups.append({
                "N": 15,
                "L": L,
                "q": q,
                "Q": q * L,
                "n": len(nov),
                "incomplete": False,
                "d_pair": d_pair,
                "d_error_over_N": _boot(err_n - err_q, rng),
                "nov3_lower": _flag_pair(d_pair),
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
    payload = {
        "version": "N3INIT_v7",
        "candidate_not_benchmark_adoption": True,
        "n_files": len(rows),
        "n_fail": len(fails),
        "failures": [
            {"arm": r["arm"], "seed": r["seed"], "L": r["L"], "q": r["q"], "blocker": r.get("blocker"), "failure": r["failure"]}
            for r in fails
        ],
        "new_queries": new_q,
        "historical_queries_spent_once": HISTORICAL,
        "total_research_queries": HISTORICAL + new_q,
        "bstar_reused_from_v4": True,
        "bstar_recomputed": False,
        "groups": groups,
        "not_a_population_or_1000_claim": True,
        "not_le_s1_or_local_r10": True,
    }
    _dump(RES / "summary.json", payload)
    lines = [
        "# N3INIT v7 N=15 L=2,3",
        "",
        "Exact v4 four-start recipes, not retuned from N=18. Development candidate only. n=8, seeds 10018–10025.",
        "QAOA start 0 is the derived center `(γ, β) = (0.25, 0.25)`. Energy selects. Fidelity is post hoc. B* reused from v4.",
        "Not LE S1 and not local R10.",
        "",
        f"New quantum queries {new_q}. Historical {HISTORICAL}. Total research queries {HISTORICAL + new_q}. Failures {len(fails)}.",
        "",
        "| L | q | Q | n | mean (E−E0)/N Nov3 | QAOA | mean F Nov3 | QAOA | mean ΔE | CI | median ΔE | min | max | wins | start0 Nov3 | start0 QAOA | flag |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for g in groups:
        if g.get("incomplete"):
            lines.append(f"| {g['L']} | {g['q']} | {g['Q']} | incomplete nov3={g['n_nov3']} qaoa={g['n_qaoa']} |")
            continue
        lines.append(
            f"| {g['L']} | {g['q']} | {g['Q']} | {g['n']} | {g['E_error_over_N_nov3']['mean']:.4f} | {g['E_error_over_N_qaoa']['mean']:.4f} | "
            f"{g['F_nov3']['mean']:.3f} | {g['F_qaoa']['mean']:.3f} | {g['d_pair']['mean']:+.4f} | "
            f"[{g['d_pair']['ci95'][0]:+.3f}, {g['d_pair']['ci95'][1]:+.3f}] | {g['d_pair']['median']:+.4f} | "
            f"{g['d_pair']['min']:+.4f} | {g['d_pair']['max']:+.4f} | {g['energy_wins']}/{g['n']} | "
            f"{g['center_start0_selected']}/{g['n']} | {g['qaoa_start0_selected']}/{g['n']} | `{g['nov3_lower']}` |"
        )
    lines += ["", "## Read flags", ""]
    for g in groups:
        if g.get("incomplete"):
            lines.append(f"- L={g['L']} q={g['q']}: incomplete.")
        else:
            lines.append(f"- L={g['L']} q={g['q']}: `{g['nov3_lower']}`.")
    if fails:
        lines += ["", "## Blockers (physics not approximated)", ""]
        for r in fails:
            lines.append(f"- {r['arm']} seed={r['seed']} L={r['L']} q={r['q']}: {r.get('blocker')} {r['failure']}")
    lines.append("")
    lines.append("Do not attribute an energy edge to the coherent center unless start 0 is actually selected.")
    lines.append(f"Fit cell time {sum(r['seconds'] for r in ok + fails):.1f}s. Workers 4, BLAS 1. Each completed ledger equals its Q.")
    (RES / "RESULTS.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", required=True, choices=("run", "summary"))
    args = parser.parse_args()
    if not PREREG.exists():
        raise RuntimeError("v7 preregistration missing")
    _conf()
    if args.phase == "run":
        run_fits()
    else:
        summarize()


if __name__ == "__main__":
    main()
