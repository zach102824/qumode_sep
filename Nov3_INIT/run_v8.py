"""Fresh-holdout L=2 confirmation on seeds 10144–10151. Exact v4 recipe."""
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
PREREG = ROOT / "PREREG_V8.md"
RES = ROOT / "results" / "v8"
CELLS = RES / "cells"
INST = RES / "instances"
CONF = RES / "CONFORMANCE.json"
HOLDOUT = list(range(10144, 10152))
NS = (15, 18)
L = 2
QS = (512, 1024)
N_BOOT = 4000
HISTORICAL = 1949696
CELL_WALL_CAP = 1800.0


def prereg_sha() -> str:
    return hashlib.sha256(PREREG.read_bytes()).hexdigest()


def allow(seed: int) -> None:
    seed = int(seed)
    if seed not in HOLDOUT:
        raise RuntimeError(f"seed {seed} is not in the unread holdout 10144..10151")
    if 10018 <= seed <= 10025:
        raise RuntimeError(f"development seed {seed}")
    if seed <= 10143:
        raise RuntimeError(f"local/LE band seed {seed}")
    if 10128 <= seed <= 10135:
        raise RuntimeError(f"LE warm-start seed {seed}")
    if 11000 <= seed <= 11031 or 20000 <= seed <= 20999:
        raise RuntimeError(f"blind seed {seed}")


def _conf() -> dict:
    doc = json.loads(CONF.read_text())
    if doc.get("prereg_sha256") != prereg_sha() or not doc.get("ok"):
        raise RuntimeError("v8 conformance does not match the current preregistration")
    if doc.get("collisions_in_nov3_init_results", {}).get("n", 1) != 0:
        raise RuntimeError("holdout seeds already appear under Nov3_INIT/results")
    return doc


def _cut(n: int) -> list[int]:
    return list(range(n // 2))


def _bstar_worker(task: tuple[int, int]) -> dict:
    n, seed = task
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    allow(int(seed))
    t0 = time.perf_counter()
    cpu0 = time.process_time()
    rec = maximize(make_J(int(seed), int(n)), int(n))
    gs = rec.pop("gs")
    rec.pop("hc")
    rec.update({
        "N": int(n),
        "seed": int(seed),
        "seconds": time.perf_counter() - t0,
        "cpu_seconds": time.process_time() - cpu0,
        "peak_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "prereg_sha256": prereg_sha(),
        "dense_matrix": False,
        "classical_not_a_quantum_query": True,
        "holdout": True,
    })
    if rec.get("cut") != _cut(n):
        raise RuntimeError(f"unexpected cut N={n} {rec.get('cut')}")
    return {"rec": rec, "gs": gs}


def build_instances() -> None:
    _conf()
    pending = []
    for n in NS:
        for seed in HOLDOUT:
            allow(seed)
            if not (INST / f"N{n}_s{seed}.json").exists():
                pending.append((n, seed))
    if not pending:
        return
    import multiprocessing as mp

    INST.mkdir(parents=True, exist_ok=True)
    with mp.get_context("fork").Pool(4) as pool:
        for got in pool.imap_unordered(_bstar_worker, pending, chunksize=1):
            n = got["rec"]["N"]
            seed = got["rec"]["seed"]
            np.savez(INST / f"N{n}_s{seed}.npz", gs=got["gs"])
            _dump(INST / f"N{n}_s{seed}.json", got["rec"])
            r = got["rec"]
            print(
                f"B* N={n} seed={seed} b={r['Bstar']:.6f} gap={r['gap']:.4g} "
                f"{r['seconds']:.1f}s rss={r['peak_rss_kib']}KiB",
                flush=True,
            )


def _logical(n: int, seed: int):
    meta = json.loads((INST / f"N{n}_s{seed}.json").read_text())
    gs = np.load(INST / f"N{n}_s{seed}.npz")["gs"]
    if int(meta["N"]) != int(n) or int(meta["seed"]) != int(seed):
        raise RuntimeError(f"instance identity mismatch N={n} seed={seed}")
    if gs.size != (1 << n) or meta.get("cut") != _cut(n):
        raise RuntimeError(f"B* invalid N={n} seed={seed}")
    return meta, Logical(n, seed, float(meta["Bstar"]), gs)


def _path(arm: str, method: str, n: int, seed: int, q: int) -> Path:
    return CELLS / f"{arm}_{method}_N{n}_s{seed}_L{L}_q{q}.json"


def _one(task: dict) -> dict:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    arm, method = task["arm"], task["method"]
    n, seed, q = int(task["N"]), int(task["seed"]), int(task["q"])
    allow(seed)
    path = _path(arm, method, n, seed, q)
    if path.exists():
        return {"skipped": True}
    t0 = time.perf_counter()
    cpu0 = time.process_time()
    try:
        meta, logical = _logical(n, seed)
        b = float(meta["Bstar"])
        mid = 0 if method == "QAOA" else 1
        Q = q * L
        rng_init = [None if s == 0 else make_rng(seed, n, b, L, mid, s, 2) for s in range(4)]
        rng_spsa = [make_rng(seed, n, b, L, mid, s, 0) for s in range(4)]

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
            "version": "N3INIT_v8",
            "candidate_not_benchmark_adoption": True,
            "holdout": True,
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
            "E0": e0,
            "E_plus": meta["E_plus"],
            "F_plus": f_plus,
            "E": out["E"],
            "E_error_over_N": float((out["E"] - e0) / float(n)),
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
            "version": "N3INIT_v8",
            "arm": arm,
            "method": method,
            "N": n,
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
    print(f"{'ok' if record['ok'] else 'FAIL'} v8 {arm} N={n} s={seed} L=2 q={q}{extra} {record['seconds']:.1f}s", flush=True)
    return record


def _pending() -> list[dict]:
    sha = prereg_sha()
    rows = []
    for n in NS:
        for q in QS:
            for seed in HOLDOUT:
                for arm, method in (("nov3_center4", "Nov3rd"), ("qaoa_center4", "QAOA")):
                    if not _path(arm, method, n, seed, q).exists():
                        rows.append({"arm": arm, "method": method, "N": n, "seed": seed, "q": q, "prereg_sha256": sha})
    return rows


def run_fits() -> None:
    import multiprocessing as mp

    _conf()
    wall0 = time.perf_counter()
    pending = _pending()
    print(f"v8 pending {len(pending)}", flush=True)
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


def _h_pass(stat: dict) -> bool:
    return bool(stat["mean"] < 0.0 and stat["ci95"][1] < 0.0)


def summarize() -> None:
    rows = [json.loads(p.read_text()) for p in CELLS.glob("*.json")]
    fails = [r for r in rows if not r.get("ok")]
    ok = [r for r in rows if r.get("ok")]
    inst = [json.loads(p.read_text()) for p in INST.glob("N*_s*.json")]
    rng = np.random.default_rng(3334)
    groups = []
    for n in NS:
        for q in QS:
            nov = sorted([r for r in ok if r["arm"] == "nov3_center4" and r["N"] == n and r["q"] == q], key=lambda r: r["seed"])
            qa = sorted([r for r in ok if r["arm"] == "qaoa_center4" and r["N"] == n and r["q"] == q], key=lambda r: r["seed"])
            if not nov or not qa or len(nov) != len(qa):
                groups.append({"N": n, "L": L, "q": q, "Q": q * L, "incomplete": True, "n_nov3": len(nov), "n_qaoa": len(qa)})
                continue
            e_n = np.array([r["E"] for r in nov])
            e_q = np.array([r["E"] for r in qa])
            err_n = np.array([r["E_error_over_N"] for r in nov])
            err_q = np.array([r["E_error_over_N"] for r in qa])
            f_n = np.array([r["F"] for r in nov])
            f_q = np.array([r["F"] for r in qa])
            d_pair = _boot(e_n - e_q, rng)
            groups.append({
                "N": n,
                "L": L,
                "q": q,
                "Q": q * L,
                "n": len(nov),
                "incomplete": False,
                "d_pair": d_pair,
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
                "is_gate_row": q == 1024,
            })
    h1 = next((g for g in groups if g.get("N") == 15 and g.get("q") == 1024 and not g.get("incomplete")), None)
    h2 = next((g for g in groups if g.get("N") == 18 and g.get("q") == 1024 and not g.get("incomplete")), None)
    h1_pass = bool(h1 and _h_pass(h1["d_pair"]))
    h2_pass = bool(h2 and _h_pass(h2["d_pair"]))
    confirmed = bool(h1_pass and h2_pass)
    new_q = int(sum(r["queries"]["total"] for r in ok))
    payload = {
        "version": "N3INIT_v8",
        "holdout_seeds": HOLDOUT,
        "n_files": len(rows),
        "n_fail": len(fails),
        "failures": [{"arm": r["arm"], "N": r["N"], "seed": r["seed"], "q": r["q"], "blocker": r.get("blocker"), "failure": r["failure"]} for r in fails],
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
        "H1_N15_Q2048": h1_pass,
        "H2_N18_Q2048": h2_pass,
        "L2_holdout": "CONFIRMED" if confirmed else "FAIL",
        "will_not_patch_pr77": True,
        "groups": groups,
    }
    _dump(RES / "summary.json", payload)
    lines = [
        "# N3INIT v8 L=2 holdout",
        "",
        "Exact v4 four-start recipes. Fresh holdout seeds 10144–10151. Not GR. Not development 10018–10025.",
        "Gates H1/H2 use Q=2048 only. Both q=512 and q=1024 were run.",
        "",
        f"New quantum queries {new_q}. Historical {HISTORICAL}. Total research queries {HISTORICAL + new_q}. Failures {len(fails)}.",
        f"Classical B* wall {payload['classical_bstar']['wall_seconds']:.1f}s, CPU {payload['classical_bstar']['cpu_seconds']:.1f}s, peak RSS {payload['classical_bstar']['peak_rss_kib_max']} KiB.",
        "",
        f"**L2-holdout {payload['L2_holdout']}.** H1 N15 Q2048 `{h1_pass}`. H2 N18 Q2048 `{h2_pass}`.",
        "",
        "| N | q | Q | n | mean (E−E0)/N Nov3 | QAOA | mean F Nov3 | QAOA | mean ΔE | CI | median | min | max | wins | start0 Nov3 | gate |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|---|",
    ]
    for g in groups:
        if g.get("incomplete"):
            lines.append(f"| {g['N']} | {g['q']} | {g['Q']} | incomplete |")
            continue
        gate = "H1" if g["N"] == 15 and g["q"] == 1024 else "H2" if g["N"] == 18 and g["q"] == 1024 else "report"
        lines.append(
            f"| {g['N']} | {g['q']} | {g['Q']} | {g['n']} | {g['E_error_over_N_nov3']['mean']:.4f} | {g['E_error_over_N_qaoa']['mean']:.4f} | "
            f"{g['F_nov3']['mean']:.3f} | {g['F_qaoa']['mean']:.3f} | {g['d_pair']['mean']:+.4f} | "
            f"[{g['d_pair']['ci95'][0]:+.3f}, {g['d_pair']['ci95'][1]:+.3f}] | {g['d_pair']['median']:+.4f} | "
            f"{g['d_pair']['min']:+.4f} | {g['d_pair']['max']:+.4f} | {g['energy_wins']}/{g['n']} | "
            f"{g['center_start0_selected']}/{g['n']} | {gate} |"
        )
    if fails:
        lines += ["", "## Blockers", ""]
        for r in fails:
            lines.append(f"- {r['arm']} N={r['N']} seed={r['seed']} q={r['q']}: {r.get('blocker')} {r['failure']}")
    lines.append("")
    lines.append("Do not patch PR #77 from this file.")
    lines.append(f"Fit cell time {sum(r['seconds'] for r in ok + fails):.1f}s. Workers 4, BLAS 1.")
    (RES / "RESULTS.md").write_text("\n".join(lines) + "\n")
    if confirmed:
        note = [
            "# L2-holdout CONFIRMED — candidacy note for GR",
            "",
            "Do not apply this patch to PR #77. Owner decision only.",
            "",
            f"Recipe prereg SHA256 `{prereg_sha()}`.",
            "Exact v4 four-start center-4 / equal-SPSA-block. Original Nov3, shared Uc, full 3-shell, QAOA p=L, SPSA only, entropy-peak B*.",
            "Development seeds 10018–10025 (used, not this confirmation). Holdout seeds 10144–10151. Validation and final unread.",
            "",
            f"H1 N=15 Q=2048 mean ΔE {h1['d_pair']['mean']:+.4f} CI [{h1['d_pair']['ci95'][0]:+.3f}, {h1['d_pair']['ci95'][1]:+.3f}] wins {h1['energy_wins']}/{h1['n']}.",
            f"H2 N=18 Q=2048 mean ΔE {h2['d_pair']['mean']:+.4f} CI [{h2['d_pair']['ci95'][0]:+.3f}, {h2['d_pair']['ci95'][1]:+.3f}] wins {h2['energy_wins']}/{h2['n']}.",
            "",
            f"New quantum queries {new_q}. Cumulative research queries {HISTORICAL + new_q}.",
        ]
        (RES / "CANDIDACY_NOTE.md").write_text("\n".join(note) + "\n")
    print("\n".join(lines), flush=True)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", required=True, choices=("bstar", "run", "summary"))
    args = parser.parse_args()
    if not PREREG.exists():
        raise RuntimeError("v8 preregistration missing")
    _conf()
    if args.phase == "bstar":
        build_instances()
    elif args.phase == "run":
        run_fits()
    else:
        summarize()


if __name__ == "__main__":
    main()
