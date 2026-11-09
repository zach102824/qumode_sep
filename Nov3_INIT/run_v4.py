"""Depth pilot and optional N=15 L1. QAOA uses the derived center. Prior cells stay put."""
from __future__ import annotations

import hashlib
import json
import os
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
PREREG = ROOT / "PREREG_V4.md"
RES = ROOT / "results" / "v4"
CELLS = RES / "cells"
INST = RES / "instances"
V1_INST = ROOT / "results" / "v1" / "instances"
CONF = RES / "CONFORMANCE.json"
SEEDS = TRAIN + TEST
QS = (512, 1024)
DEPTH = ((9, 4), (9, 6), (12, 4), (12, 6))
N_BOOT = 4000


def prereg_sha() -> str:
    return hashlib.sha256(PREREG.read_bytes()).hexdigest()


def _conf() -> dict:
    doc = json.loads(CONF.read_text())
    if doc.get("prereg_sha256") != prereg_sha() or not doc.get("ok"):
        raise RuntimeError("v4 conformance does not match the current preregistration")
    return doc


def _logical(n: int, seed: int):
    for folder in (INST, V1_INST):
        meta_path = folder / f"N{n}_s{seed}.json"
        npz_path = folder / f"N{n}_s{seed}.npz"
        if meta_path.exists() and npz_path.exists():
            meta = json.loads(meta_path.read_text())
            gs = np.load(npz_path)["gs"]
            return meta, Logical(n, seed, float(meta["Bstar"]), gs)
    raise RuntimeError(f"missing B* N={n} seed={seed}")


def _n15_worker(seed: int) -> dict:
    os.environ["OMP_NUM_THREADS"] = "1"
    t0 = time.perf_counter()
    rec = maximize(make_J(int(seed), 15), 15)
    gs = rec.pop("gs")
    rec.pop("hc")
    rec.update({
        "N": 15,
        "seed": int(seed),
        "seconds": time.perf_counter() - t0,
        "prereg_sha256": prereg_sha(),
        "dense_matrix": False,
    })
    return {"rec": rec, "gs": gs}


def build_n15() -> None:
    if not _conf()["probe"]["n15_allowed"]:
        print("N15 probe gate closed", flush=True)
        return
    pending = []
    for seed in SEEDS:
        allow(seed)
        if not (INST / f"N15_s{seed}.json").exists():
            pending.append(seed)
    if not pending:
        return
    import multiprocessing as mp

    with mp.get_context("fork").Pool(4) as pool:
        for got in pool.imap_unordered(_n15_worker, pending):
            seed = got["rec"]["seed"]
            INST.mkdir(parents=True, exist_ok=True)
            np.savez(INST / f"N15_s{seed}.npz", gs=got["gs"])
            _dump(INST / f"N15_s{seed}.json", got["rec"])
            print(f"B* N=15 seed={seed} b={got['rec']['Bstar']:.6f} {got['rec']['seconds']:.1f}s", flush=True)


def _path(arm: str, method: str, n: int, seed: int, L: int, q: int) -> Path:
    return CELLS / f"{arm}_{method}_N{n}_s{seed}_L{L}_q{q}.json"


def _jobs() -> list[tuple]:
    doc = _conf()
    rows = [(n, L) for n, L in DEPTH]
    if doc["probe"]["n15_allowed"]:
        rows.append((15, 1))
    out = []
    for n, L in rows:
        for seed in SEEDS:
            for q in QS:
                out.append((n, seed, L, q))
    return out


def _one(task: dict) -> dict:
    os.environ["OMP_NUM_THREADS"] = "1"
    arm, method = task["arm"], task["method"]
    n, seed, L, q = int(task["N"]), int(task["seed"]), int(task["L"]), int(task["q"])
    allow(seed)
    path = _path(arm, method, n, seed, L, q)
    if path.exists():
        return {"skipped": True}
    t0 = time.perf_counter()
    meta, logical = _logical(n, seed)
    b = float(meta["Bstar"])
    mid = 0 if method == "QAOA" else 1
    Q = q * L
    rng_init = [None if s == 0 else make_rng(seed, n, b, L, mid, s, 2) for s in range(4)]
    rng_spsa = [make_rng(seed, n, b, L, mid, s, 0) for s in range(4)]

    def draw(method_name, ell, start_id, rng, _arm=arm):
        if _arm == "qaoa_center4":
            return x0_qaoa_center4(ell, start_id, rng)
        if _arm == "nov3_center4":
            return x0_center4(method_name, ell, start_id, rng)
        return None

    def propagate(x, _logical=logical, _method=method, _L=L):
        return _logical.propagate(_method, _L, x)

    try:
        out = run_equal4(propagate, method, L, Q, rng_init, rng_spsa, x0_fn=None if arm == "nov3_equal4" else draw)
        if arm == "nov3_center4":
            ref = json.loads(_path("nov3_equal4", "Nov3rd", n, seed, L, q).read_text())
            for row in out["tails"]:
                if row["start_id"] == 0:
                    continue
                prev = ref["tails"][row["start_id"]]
                if abs(row["E"] - prev["E"]) > 1e-8 or abs(row["e_init"] - prev["e_init"]) > 1e-8:
                    raise RuntimeError(f"start {row['start_id']} diverged")
        psi = out.pop("aux")
        f = logical.fidelity(psi)
        f_plus = float(meta["F_plus"])
        record = {
            "ok": True,
            "failure": None,
            "version": "N3INIT_v4",
            "candidate_not_benchmark_adoption": True,
            "arm": arm,
            "method": method,
            "method_id": mid,
            "N": n,
            "seed": seed,
            "L": L,
            "q": q,
            "Q": Q,
            "procedural_cap": Q,
            "Bstar": b,
            "E0": meta["E0"],
            "E_plus": meta["E_plus"],
            "F_plus": f_plus,
            "E": out["E"],
            "F": f,
            "eta": float((f - f_plus) / (1.0 - f_plus)),
            "escaped": bool(out["E"] < meta["E_plus"] - 1e-3 * n),
            "which": out["which"],
            "start_id": int(out["start_id"]),
            "n_iter": int(out["n_iter"]),
            "tails": out["tails"],
            "queries": out["queries"],
            "x": [float(v) for v in out["x"]],
            "seconds": time.perf_counter() - t0,
            "prereg_sha256": task["prereg_sha256"],
            "development_not_fresh_holdout": True,
            "qaoa_center_is_derived": arm == "qaoa_center4",
            "fidelity_posthoc": True,
        }
    except Exception as exc:
        record = {
            "ok": False,
            "failure": repr(exc),
            "version": "N3INIT_v4",
            "arm": arm,
            "method": method,
            "N": n,
            "seed": seed,
            "L": L,
            "q": q,
            "Q": q * L,
            "seconds": time.perf_counter() - t0,
            "prereg_sha256": task["prereg_sha256"],
        }
    _dump(path, record)
    extra = f" E={record['E']:.4f} F={record['F']:.3f}" if record["ok"] else " " + record["failure"]
    print(f"{'ok' if record['ok'] else 'FAIL'} v4 {arm} N={n} s={seed} L={L} q={q}{extra} {record['seconds']:.1f}s", flush=True)
    return record


def _pending(arms: tuple[str, ...]) -> list[dict]:
    sha = prereg_sha()
    rows = []
    for n, seed, L, q in _jobs():
        for arm in arms:
            method = "QAOA" if arm == "qaoa_center4" else "Nov3rd"
            if not _path(arm, method, n, seed, L, q).exists():
                rows.append({"arm": arm, "method": method, "N": n, "seed": seed, "L": L, "q": q, "prereg_sha256": sha})
    return rows


def run_fits() -> None:
    import multiprocessing as mp

    wall0 = time.perf_counter()
    for arms in (("nov3_equal4", "qaoa_center4"), ("nov3_center4",)):
        pending = _pending(arms)
        print(f"v4 pending {len(pending)}", flush=True)
        if pending:
            with mp.get_context("fork").Pool(4) as pool:
                list(pool.imap_unordered(_one, pending, chunksize=1))
    (RES / "wall.json").write_text(json.dumps({"fit_seconds": time.perf_counter() - wall0, "workers": 4}) + "\n")


def _historical() -> int:
    total = 0
    root = ROOT / "results"
    for path in root.rglob("*.json"):
        if "v4" in path.parts or path.parent.name != "cells":
            continue
        doc = json.loads(path.read_text())
        if doc.get("ok") and "queries" in doc:
            total += int(doc["queries"]["total"])
    return total


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


def _flag(stat: dict) -> str:
    lo, hi = stat["ci95"]
    if stat["mean"] < 0.0 and hi < 0.0:
        return "negative_ci_excludes_0"
    if stat["mean"] < 0.0:
        return "negative_point_ci_includes_0"
    if lo > 0.0:
        return "positive_ci_excludes_0"
    return "positive_point_ci_includes_0"


def summarize() -> None:
    rows = [json.loads(p.read_text()) for p in CELLS.glob("*.json")]
    fails = [r for r in rows if not r.get("ok")]
    ok = [r for r in rows if r.get("ok")]
    rng = np.random.default_rng(3334)
    groups = []
    levels = sorted({(r["N"], r["L"], r["q"]) for r in ok})
    # Prereg order is N, L, q ascending. sorted tuples already do that.
    for n, L, q in levels:
        def take(arm, n=n, L=L, q=q):
            hit = [r for r in ok if r["N"] == n and r["L"] == L and r["q"] == q and r["arm"] == arm]
            hit.sort(key=lambda r: r["seed"])
            return hit
        cen, eq, qa = take("nov3_center4"), take("nov3_equal4"), take("qaoa_center4")
        if not (cen and eq and qa) or not (len(cen) == len(eq) == len(qa)):
            continue
        e_c = np.array([r["E"] for r in cen])
        e_e = np.array([r["E"] for r in eq])
        e_q = np.array([r["E"] for r in qa])
        f_c = np.array([r["F"] for r in cen])
        f_e = np.array([r["F"] for r in eq])
        f_q = np.array([r["F"] for r in qa])
        d_self = _boot(e_c - e_e, rng)
        d_pair = _boot(e_c - e_q, rng)
        groups.append({
            "N": n,
            "L": L,
            "q": q,
            "Q": q * L,
            "d_self": d_self,
            "d_pair": d_pair,
            "center_helps": _flag(d_self),
            "beats_fair_qaoa": _flag(d_pair),
            "energy_wins_vs_equal": int(np.sum(e_c < e_e)),
            "energy_wins_vs_qaoa": int(np.sum(e_c < e_q)),
            "E_center": _boot(e_c, rng),
            "E_equal": _boot(e_e, rng),
            "E_qaoa": _boot(e_q, rng),
            "F_center": _boot(f_c, rng),
            "F_equal": _boot(f_e, rng),
            "F_qaoa": _boot(f_q, rng),
            "center_start0_selected": int(sum(r["start_id"] == 0 for r in cen)),
            "qaoa_start0_selected": int(sum(r["start_id"] == 0 for r in qa)),
            "queries": int(sum(r["queries"]["total"] for r in cen + eq + qa)),
            "seconds": float(sum(r["seconds"] for r in cen + eq + qa)),
        })
    new_q = int(sum(r["queries"]["total"] for r in ok))
    hist = _historical()
    payload = {
        "version": "N3INIT_v4",
        "candidate_not_benchmark_adoption": True,
        "n_files": len(rows),
        "n_fail": len(fails),
        "failures": [{"arm": r["arm"], "N": r["N"], "seed": r["seed"], "L": r["L"], "q": r["q"], "failure": r["failure"]} for r in fails],
        "new_queries": new_q,
        "historical_queries_spent_once": hist,
        "total_research_queries": hist + new_q,
        "groups": groups,
    }
    _dump(RES / "summary.json", payload)
    lines = [
        "# N3INIT v4",
        "",
        "Development candidate only. Not a benchmark adoption. QAOA start 0 is the derived center `(γ, β) = (0.25, 0.25)` on every layer.",
        "Seeds 10018–10025 are development at N=9/12. N=15 uses the same band because the width is new.",
        "",
        f"New queries {new_q}. Historical queries counted once {hist}. Total research queries {hist + new_q}. Failures {len(fails)}.",
        "",
        "| N | L | q | Q | n | mean E center | median | min | max | mean E equal | mean E QAOA | mean Δ self | self CI | mean Δ pair | pair CI | wins vs equal | wins vs QAOA | mean F center | mean F QAOA | start0 Nov3 | start0 QAOA |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for g in groups:
        lines.append(
            f"| {g['N']} | {g['L']} | {g['q']} | {g['Q']} | {g['d_pair']['n']} | "
            f"{g['E_center']['mean']:.4f} | {g['E_center']['median']:.4f} | {g['E_center']['min']:.4f} | {g['E_center']['max']:.4f} | "
            f"{g['E_equal']['mean']:.4f} | {g['E_qaoa']['mean']:.4f} | {g['d_self']['mean']:+.4f} | "
            f"[{g['d_self']['ci95'][0]:+.3f}, {g['d_self']['ci95'][1]:+.3f}] | {g['d_pair']['mean']:+.4f} | "
            f"[{g['d_pair']['ci95'][0]:+.3f}, {g['d_pair']['ci95'][1]:+.3f}] | "
            f"{g['energy_wins_vs_equal']}/{g['d_self']['n']} | {g['energy_wins_vs_qaoa']}/{g['d_pair']['n']} | "
            f"{g['F_center']['mean']:.3f} | {g['F_qaoa']['mean']:.3f} | {g['center_start0_selected']}/{g['d_self']['n']} | {g['qaoa_start0_selected']}/{g['d_pair']['n']} |"
        )
    lines += ["", "## Read flags", ""]
    for g in groups:
        lines.append(f"- N={g['N']} L={g['L']} q={g['q']}: center_helps `{g['center_helps']}`; beats_fair_qaoa `{g['beats_fair_qaoa']}`.")
    lines.append("")
    lines.append(f"Cell time {sum(r['seconds'] for r in ok):.1f}s on the recorded cells. Workers 4.")
    (RES / "RESULTS.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", required=True, choices=("n15", "run", "summary"))
    args = parser.parse_args()
    if not PREREG.exists():
        raise RuntimeError("v4 preregistration missing")
    _conf()
    if args.phase == "n15":
        build_n15()
    elif args.phase == "run":
        run_fits()
    else:
        summarize()


if __name__ == "__main__":
    main()
