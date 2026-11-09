"""N=15 L=1 SK spin-flip gauge panel. Frozen v4 center-4. Energy selects."""
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
from Nov3_INIT.bstar import entropy_nats, maximize
from Nov3_INIT.gauge import (
    BAND,
    N,
    N_GAUGES,
    apply_X_product,
    flip_mask,
    frozen_table,
    gauge_vector,
    gauged_J,
    logical_gauged,
    map_gs,
    residual,
)
from Nov3_INIT.run_v1 import _dump
from Nov9_SK_SPSA.experiments.FT.instance import make_J, packing, pair_diag, z_table

ROOT = Path(__file__).resolve().parent
PREREG = ROOT / "PREREG_GAUGE.md"
RES = ROOT / "results" / "gauge"
CELLS = RES / "cells"
INST = RES / "instances"
CONF = RES / "CONFORMANCE.json"
GAUGE = RES / "GAUGE.json"
N_BOOT = 4000
HISTORICAL = 2818048
CELL_WALL_CAP = 600.0
CUT = list(range(7))
L = 1
Q_PER_LAYER = 1024
METHODS = (
    ("nov3", "Nov3rd", METHOD_ID["equal4"]["Nov3rd"]),
    ("qaoa", "QAOA", METHOD_ID["equal4"]["QAOA"]),
)
QAOA_SANITY = 1e-8


def prereg_sha() -> str:
    return hashlib.sha256(PREREG.read_bytes()).hexdigest()


def allow(seed: int) -> None:
    seed = int(seed)
    if seed not in BAND:
        raise RuntimeError(f"seed {seed} is not in the gauge band 10552..10559")
    if 10536 <= seed <= 10551:
        raise RuntimeError(f"local R31 seed {seed}")
    if 10528 <= seed <= 10535:
        raise RuntimeError(f"FT WL seed {seed}")
    if 10520 <= seed <= 10527:
        raise RuntimeError(f"FT cutoff seed {seed}")
    if 10504 <= seed <= 10519:
        raise RuntimeError(f"local R30 seed {seed}")
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
        raise RuntimeError("gauge conformance does not match the current preregistration")
    if doc.get("collisions_in_nov3_init_results", {}).get("n", 1) != 0:
        raise RuntimeError("gauge seeds already appear under Nov3_INIT/results")
    if packing(N) != [[a, a + 5, a + 10] for a in range(5)]:
        raise RuntimeError("N=15 packing is not interleaved m=5")
    frozen = json.loads(GAUGE.read_text())
    if hashlib.sha256(GAUGE.read_bytes()).hexdigest() != doc.get("gauge_sha256"):
        raise RuntimeError("GAUGE.json hash mismatch")
    if frozen != frozen_table():
        raise RuntimeError("GAUGE.json does not match the preregistered RNG recipe")
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
        "gauge_band": True,
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


def _meta_gs(seed: int):
    meta = json.loads((INST / f"N{N}_s{seed}.json").read_text())
    gs = np.load(INST / f"N{N}_s{seed}.npz")["gs"]
    if int(meta["N"]) != N or int(meta["seed"]) != int(seed) or gs.size != (1 << N) or meta.get("cut") != CUT:
        raise RuntimeError(f"B* invalid seed {seed}")
    return meta, gs


def _path(method_tag: str, seed: int, gid: int) -> Path:
    return CELLS / f"{method_tag}_g{gid}_N{N}_s{seed}_L{L}_q{Q_PER_LAYER}.json"


def check_gauges() -> None:
    """Classical: mapped gs is an eigenvector of H' at the original B*. No SPSA."""
    _conf()
    rows = []
    z = z_table(N)
    for seed in BAND:
        allow(seed)
        meta, gs = _meta_gs(seed)
        b = float(meta["Bstar"])
        e0 = float(meta["E0"])
        J0 = make_J(seed, N)
        s0 = entropy_nats(gs, N)
        for gid in range(N_GAUGES):
            g = gauge_vector(seed, N, gid)
            hc = pair_diag(gauged_J(J0, g), z)
            gs_g = map_gs(gs, g)
            e, resid = residual(hc, b, N, gs_g)
            plus = np.full(1 << N, 2.0 ** (-0.5 * N), dtype=np.float64)
            plus_mapped = apply_X_product(plus, flip_mask(g))
            log = logical_gauged(N, seed, b, gs, gid)
            xq = x0_qaoa_center4(L, 0, None)
            psi_q = log.qaoa(xq, L)
            rec = {
                "seed": int(seed),
                "gid": int(gid),
                "dE0": abs(e - e0),
                "residual": resid,
                "entropy_d": abs(entropy_nats(gs_g, N) - s0),
                "plus_mapped": bool(np.allclose(plus_mapped, plus)),
                "qaoa_center_E": log.energy(psi_q),
                "qaoa_center_F": log.fidelity(psi_q),
            }
            rec["ok"] = bool(
                rec["dE0"] < 1e-8
                and rec["residual"] < 1e-8
                and rec["entropy_d"] < 1e-10
                and rec["plus_mapped"]
            )
            rows.append(rec)
            print(
                f"{'ok' if rec['ok'] else 'FAIL'} gauge-check s={seed} g={gid} "
                f"dE0={rec['dE0']:.2e} resid={rec['residual']:.2e}",
                flush=True,
            )
    # QAOA center energy/F identical across gauges of a seed
    qaoa_dE, qaoa_dF = [], []
    for seed in BAND:
        sub = [r for r in rows if r["seed"] == seed]
        e0 = sub[0]["qaoa_center_E"]
        f0 = sub[0]["qaoa_center_F"]
        qaoa_dE.append(max(abs(r["qaoa_center_E"] - e0) for r in sub))
        qaoa_dF.append(max(abs(r["qaoa_center_F"] - f0) for r in sub))
    payload = {
        "ok": bool(all(r["ok"] for r in rows) and max(qaoa_dE) < 1e-10 and max(qaoa_dF) < 1e-12),
        "n": len(rows),
        "max_dE0": float(max(r["dE0"] for r in rows)),
        "max_residual": float(max(r["residual"] for r in rows)),
        "max_entropy_d": float(max(r["entropy_d"] for r in rows)),
        "qaoa_center_max_dE": float(max(qaoa_dE)),
        "qaoa_center_max_dF": float(max(qaoa_dF)),
        "classical_not_a_quantum_query": True,
        "rows": rows,
    }
    _dump(RES / "GAUGE_CHECK.json", payload)
    if not payload["ok"]:
        raise RuntimeError("N=15 gauge algebra check failed; not fitting")
    print(json.dumps({k: payload[k] for k in payload if k != "rows"}, indent=2), flush=True)


def _one(task: dict) -> dict:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    method_tag, method, mid = task["method_tag"], task["method"], int(task["method_id"])
    seed, gid = int(task["seed"]), int(task["gid"])
    allow(seed)
    path = _path(method_tag, seed, gid)
    if path.exists():
        return {"skipped": True}
    Q = Q_PER_LAYER * L
    t0 = time.perf_counter()
    cpu0 = time.process_time()
    try:
        meta, gs = _meta_gs(seed)
        b = float(meta["Bstar"])
        g = gauge_vector(seed, N, gid)
        logical = logical_gauged(N, seed, b, gs, gid)
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
            "version": "N3INIT_gauge",
            "method_tag": method_tag,
            "method": method,
            "method_id": mid,
            "gid": gid,
            "g": [int(v) for v in g],
            "n_flips": int(np.sum(np.asarray(g) < 0)),
            "n_params": n_params(method, L),
            "N": N,
            "seed": seed,
            "L": L,
            "q": Q_PER_LAYER,
            "Q": Q,
            "procedural_cap": Q,
            "same_query_budget_across_arms": True,
            "rng_omits_gauge": True,
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
        }
    except Exception as exc:
        record = {
            "ok": False,
            "failure": repr(exc),
            "version": "N3INIT_gauge",
            "method_tag": method_tag,
            "method": method,
            "gid": gid,
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
    print(f"{'ok' if record['ok'] else 'FAIL'} gauge {method_tag} g={gid} s={seed}{extra} {record['seconds']:.1f}s", flush=True)
    return record


def run_fits() -> None:
    import multiprocessing as mp

    _conf()
    check_path = RES / "GAUGE_CHECK.json"
    if not check_path.exists() or not json.loads(check_path.read_text()).get("ok"):
        raise RuntimeError("N=15 gauge check missing or failed; not fitting")
    sha = prereg_sha()
    pending = []
    for seed in BAND:
        for gid in range(N_GAUGES):
            for method_tag, method, mid in METHODS:
                if not _path(method_tag, seed, gid).exists():
                    pending.append({
                        "method_tag": method_tag,
                        "method": method,
                        "method_id": mid,
                        "seed": seed,
                        "gid": gid,
                        "prereg_sha256": sha,
                    })
    print(f"gauge pending {len(pending)}", flush=True)
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


def _cell(ok, method_tag, seed, gid):
    hits = [r for r in ok if r["method_tag"] == method_tag and r["seed"] == seed and r["gid"] == gid]
    if len(hits) != 1:
        raise RuntimeError(f"missing cell {method_tag} s={seed} g={gid}")
    return hits[0]


def summarize() -> None:
    rows = [json.loads(p.read_text()) for p in CELLS.glob("*.json")]
    fails = [r for r in rows if not r.get("ok")]
    ok = [r for r in rows if r.get("ok")]
    inst = [json.loads(p.read_text()) for p in INST.glob("N15_s*.json")]

    def boot(values):
        return _boot(np.asarray(values, dtype=np.float64), np.random.default_rng(3334))
    complete = len(ok) == 64 and not fails
    per_s = []
    qaoa_dE = []
    nov3_range = []
    identity_d = []
    by_g = {g: [] for g in range(N_GAUGES)}
    worst = []
    if complete:
        for seed in BAND:
            d_g = []
            e_n, e_q = [], []
            for gid in range(N_GAUGES):
                nrec = _cell(ok, "nov3", seed, gid)
                qrec = _cell(ok, "qaoa", seed, gid)
                d = float(nrec["E"] - qrec["E"])
                d_g.append(d)
                e_n.append(float(nrec["E"]))
                e_q.append(float(qrec["E"]))
                by_g[gid].append(d)
            identity_d.append(d_g[0])
            qaoa_dE.append(max(abs(e - e_q[0]) for e in e_q))
            nov3_range.append(max(e_n) - min(e_n))
            w = int(np.argmax(d_g))
            worst.append({"seed": seed, "gid": w, "dE": d_g[w], "nov3_E": e_n[w], "qaoa_E": e_q[w]})
            per_s.append({
                "seed": seed,
                "dE_g": d_g,
                "mean_dE": float(np.mean(d_g)),
                "min_dE": float(min(d_g)),
                "max_dE": float(max(d_g)),
                "range_dE": float(max(d_g) - min(d_g)),
                "nov3_E": e_n,
                "qaoa_E": e_q,
                "nov3_F": [float(_cell(ok, "nov3", seed, g)["F"]) for g in range(N_GAUGES)],
                "qaoa_F": [float(_cell(ok, "qaoa", seed, g)["F"]) for g in range(N_GAUGES)],
                "worst_gid": w,
            })
    primary = boot([r["mean_dE"] for r in per_s]) if complete else None
    qaoa_ok = bool(complete and max(qaoa_dE) <= QAOA_SANITY)
    label = "GAUGE_IMPL_FAIL" if complete and not qaoa_ok else "GAUGE_EXPLORATORY"
    new_q = int(sum(r["queries"]["total"] for r in ok))
    payload = {
        "version": "N3INIT_gauge",
        "no_gr_patch": True,
        "no_adoption_gate": True,
        "compiler_stream_closed": True,
        "holdout_fail_stands": True,
        "statistical_unit": "8 original instances",
        "n_files": len(rows),
        "n_fail": len(fails),
        "failures": [{"method_tag": r.get("method_tag"), "seed": r.get("seed"), "gid": r.get("gid"), "failure": r.get("failure")} for r in fails],
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
        "qaoa_gauge_covariance_ok": qaoa_ok,
        "qaoa_max_cross_gauge_dE": float(max(qaoa_dE)) if complete else None,
        "nov3_mean_cross_gauge_range": float(np.mean(nov3_range)) if complete else None,
        "primary_instance_gauge_mean_dE": primary,
        "identity_only_dE": boot(identity_d) if complete else None,
        "per_gauge_dE": {str(g): boot(by_g[g]) for g in range(N_GAUGES)} if complete else None,
        "per_instance": per_s,
        "worst_gauge_per_instance": worst,
        "mean_worst_gauge_dE": float(np.mean([w["dE"] for w in worst])) if complete else None,
        "primary_ci_entirely_lt0": bool(primary and primary["mean"] < 0.0 and primary["ci95"][1] < 0.0) if complete else None,
        "not_an_adoption": True,
    }
    _dump(RES / "summary.json", payload)
    lines = [
        "# SK spin-flip gauge robustness N=15 L=1",
        "",
        "Original shared-Uc Nov3 vs QAOA p=1. Frozen v4 center-4. Identity + 3 frozen random local spin-flip gauges. Statistical unit: 8 instances, gauges correlated. Exploratory; no adoption gate; no GR. Compiler stream closed. Hardware W_L unvalidated.",
        "",
        f"**{label}.**",
        "",
        f"New energy queries {new_q}. Historical {HISTORICAL}. Total {HISTORICAL + new_q}. Failures {len(fails)}.",
        f"Classical B* wall {payload['classical_bstar']['wall_seconds']:.1f}s. QAOA cross-gauge max |ΔE| {payload['qaoa_max_cross_gauge_dE']}.",
        "",
    ]
    if not complete:
        lines.append("Panel incomplete.")
    else:
        d = primary
        lines += [
            "## Primary: per-instance gauge-average Nov3−QAOA E (n=8)",
            "",
            f"mean ΔE {d['mean']:+.4f} CI [{d['ci95'][0]:+.3f}, {d['ci95'][1]:+.3f}] median {d['median']:+.4f} min {d['min']:+.4f} max {d['max']:+.4f}.",
            f"Post-hoc F: mean Nov3 {float(np.mean([np.mean(r['nov3_F']) for r in per_s])):.3f} QAOA {float(np.mean([np.mean(r['qaoa_F']) for r in per_s])):.3f} (not selected).",
            "Not an adoption gate. CI entirely <0 is a descriptive fact only.",
            "",
            "## Identity-only (g=0)",
            "",
            f"mean ΔE {payload['identity_only_dE']['mean']:+.4f} CI [{payload['identity_only_dE']['ci95'][0]:+.3f}, {payload['identity_only_dE']['ci95'][1]:+.3f}].",
            "",
            "## Each frozen gauge",
            "",
        ]
        for g in range(N_GAUGES):
            dg = payload["per_gauge_dE"][str(g)]
            lines.append(f"- g={g}: mean ΔE {dg['mean']:+.4f} CI [{dg['ci95'][0]:+.3f}, {dg['ci95'][1]:+.3f}]")
        lines += [
            "",
            "## Per-instance range and worst gauge (largest ΔE, weakest Nov3 edge)",
            "",
            f"Mean Nov3 cross-gauge energy range {payload['nov3_mean_cross_gauge_range']:.4f} (expressivity/labeling; SPSA seed shared).",
            f"Mean worst-gauge ΔE {payload['mean_worst_gauge_dE']:+.4f}.",
            "",
        ]
        for r in per_s:
            lines.append(
                f"- seed {r['seed']}: ΔE g={np.array(r['dE_g']).round(4).tolist()} mean {r['mean_dE']:+.4f} "
                f"range {r['range_dE']:.4f} worst g={r['worst_gid']}"
            )
        lines += [
            "",
            "## QAOA sanity",
            "",
            f"max |E_g−E_0| {payload['qaoa_max_cross_gauge_dE']:.3e} (ok {qaoa_ok}, bar {QAOA_SANITY:.0e}). Implementation check, not assumed.",
            "",
            "F is post hoc. No cherry-picked best gauge. Stop after this panel.",
        ]
    if fails:
        lines += ["", "## Blockers", ""]
        for r in fails:
            lines.append(f"- {r.get('method_tag')} s={r.get('seed')} g={r.get('gid')}: {r.get('failure')}")
    lines.append("")
    lines.append("Fit workers 4, BLAS 1. Each completed ledger equals Q=1024. No hardware claim.")
    (RES / "RESULTS.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", required=True, choices=("bstar", "check", "run", "summary"))
    args = parser.parse_args()
    if not PREREG.exists():
        raise RuntimeError("gauge preregistration missing")
    _conf()
    if args.phase == "bstar":
        build_instances()
    elif args.phase == "check":
        check_gauges()
    elif args.phase == "run":
        run_fits()
    else:
        summarize()


if __name__ == "__main__":
    main()
