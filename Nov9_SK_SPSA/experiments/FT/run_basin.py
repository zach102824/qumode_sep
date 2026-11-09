"""R2 B2 BASIN at per-instance B*. Protocol: PREREG_R2_BASIN.md. SPSA only."""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np

from Nov9_SK_SPSA.experiments.FT.bstar import maximize_entropy_le
from Nov9_SK_SPSA.experiments.FT.instance import make_J
from Nov9_SK_SPSA.experiments.FT.params import (
    METHOD_ID,
    alpha_to_u,
    initial_x,
    make_rng,
    physical,
    shift_physical,
    spec_for,
    unpack,
)
from concurrent.futures import ProcessPoolExecutor, as_completed

from Nov9_SK_SPSA.experiments.FT.run_ft import RESULT, claim_block, evaluate, load
from Nov9_SK_SPSA.experiments.FT.sim import FockTables, get_instance, run_nov9
from Nov9_SK_SPSA.experiments.FT.spsa import spsa_start

OUT = RESULT / "basin_r2"
PREREG = Path(__file__).resolve().parent / "PREREG_R2_BASIN.md"
SEEDS = list(range(10000, 10008))
ARMS = {
    "basin_cold_uz": (128, 1024),
    "basin_wide_uz": (128, 1024),
    "basin_qaoa": (128, 1024),
    "basin_cold_u0": (1024,),
    "basin_wide_u0": (1024,),
}


def prereg_sha() -> str:
    return hashlib.sha256(PREREG.read_bytes()).hexdigest()


def iteration_queries(cap: int) -> list[int]:
    """Exact split from the prereg. Length 4, each value even."""
    remain = int(cap) - 4 * (1 + 8 + 2)
    if remain < 0 or remain % 2:
        raise ValueError(cap)
    base = remain // 4
    if base % 2:
        base -= 1
    leftover = remain - 4 * base
    extra = [0, 0, 0, 0]
    slot = 2
    while leftover:
        extra[slot] += 2
        slot = 2 if slot == 3 else 3
        leftover -= 2
    return [base + e for e in extra]


def wide_x(spec, L: int, start_id: int, rng) -> np.ndarray:
    x = np.zeros(spec.n, dtype=np.float64)
    if start_id == 0:
        x[0] = 0.25
        for ell in range(L):
            sign = -1.0 if ell % 2 else 1.0
            x[spec.index(f"u{ell}")] = alpha_to_u(sign * 0.06)
        x[spec.index("s1_0")] = np.pi / 2
        x[spec.index(f"s1_{L - 1}")] = -np.pi / 2
        return x
    x[0] = float(rng.uniform(-1.0, 1.0))
    for name in spec.features:
        x[spec.index(f"theta_{name}")] = float(rng.uniform(-1.0, 1.0))
    lim = 0.30 * (1.0 - 1e-12)
    for ell in range(L):
        a = float(np.clip(rng.uniform(-0.30, 0.30), -lim, lim))
        x[spec.index(f"u{ell}")] = alpha_to_u(a)
    for ell in range(L):
        for sname in ("s1", "s2", "s3"):
            x[spec.index(f"{sname}_{ell}")] = float(rng.uniform(-np.pi, np.pi))
    return x


def build_bstar() -> dict:
    rows = []
    for N in (6, 9):
        for seed in SEEDS:
            J = make_J(seed, N)
            primary = maximize_entropy_le(J, n_grid=60)
            stable = maximize_entropy_le(J, n_grid=90)
            primary["N"] = N
            primary["seed"] = seed
            primary["J_sha256"] = hashlib.sha256(np.ascontiguousarray(J, dtype="<f8").tobytes()).hexdigest()
            primary["stability_90grid_Bstar"] = stable["Bstar"]
            primary["stability_abs_dB"] = abs(stable["Bstar"] - primary["Bstar"])
            primary["stability_abs_dS_nats"] = abs(stable["S_A_nats"] - primary["S_A_nats"])
            primary.pop("checks", None)
            rows.append(primary)
            print(
                f"B* N{N} s{seed} {primary['Bstar']:.6f} S {primary['S_A_nats']:.4f} nats "
                f"gap {primary['gap_E1_E0']:.4e} dEven {primary['delta_even']:.4f} "
                f"dB90 {primary['stability_abs_dB']:.3e}",
                flush=True,
            )
    return {"prereg_sha256": prereg_sha(), "entropy_unit": "nats", "rows": rows}


def sigma_and_cutoff(rows: list[dict]) -> dict:
    """Leak-step scale and the NF=32 gate, on train seed 10008 only."""
    J = make_J(10008, 9)
    star = maximize_entropy_le(J)
    b = float(star["Bstar"])
    scale = 1.0
    probes = []
    for _ in range(3):
        spec = spec_for("basin_cold_uz", 3, scale)
        inst = get_instance(9, 10008, b, "train")
        x = initial_x(spec, 3, 0, None, inst.theta_K_prior)
        base = evaluate(9, 10008, b, "train", spec, x, 32)
        d = np.zeros(spec.n)
        iu = spec.index("u0")
        d[iu] = spec.sigma[iu]
        dp = abs(evaluate(9, 10008, b, "train", spec, shift_physical(x, d, spec), 32)["p_leak"] - base["p_leak"])
        dm = abs(evaluate(9, 10008, b, "train", spec, shift_physical(x, -d, spec), 32)["p_leak"] - base["p_leak"])
        probes.append({"scale": scale, "plus": dp, "minus": dm, "base_pleak": base["p_leak"]})
        if max(dp, dm) <= 0.01:
            break
        scale *= 0.5
    # Worst-case same-sign displacement, not an optimization.
    tables32 = FockTables(get_instance(9, 10008, b, "train"), 32)
    tables48 = FockTables(get_instance(9, 10008, b, "train"), 48)
    un = {"gamma": 0.25, "theta": {"K": 0.0, "M": 0.0}, "alphas": [0.3, 0.3, 0.3], "shells": np.zeros((3, 3))}
    m32 = run_nov9(tables32, un, "uz", "restricted")
    m48 = run_nov9(tables48, un, "uz", "restricted", return_state=True)
    psi = m48.pop("_psi")
    p_out = float(np.dot(np.abs(psi) ** 2, np.any(tables48.fs >= 32, axis=1)))
    gate = {
        "Bstar_seed_10008_N9": b,
        "dE": abs(m32["E"] - m48["E"]),
        "dF": abs(m32["F"] - m48["F"]),
        "norm32": m32["norm"],
        "norm48": m48["norm"],
        "p_outside_32_at_48": p_out,
        "pass": abs(m32["E"] - m48["E"]) <= 1e-8 and p_out <= 1e-8 and abs(m32["norm"] - 1) <= 1e-9,
    }
    return {"alpha_sigma_scale": scale, "leak_probes": probes, "cutoff_gate": gate}


def _x0(method, spec, L, start_id, rng, prior):
    if method.startswith("basin_wide"):
        return wide_x(spec, L, start_id, rng)
    return initial_x(spec, L, start_id, rng, prior)


def _start_label(method, start_id, L):
    if method.startswith("basin_wide") and start_id == 0:
        return "frame_L1_s1_minus_pi_over_2" if L == 1 else "alpha_plus_phase_frame"
    if method.startswith("basin_cold") and start_id == 0:
        return "echo_s0"
    if "qaoa" in method and start_id == 0:
        return "angles_0.25"
    if method.startswith("basin_wide"):
        return "wide_box"
    if "qaoa" in method:
        return "qaoa_box"
    return "narrow_box"


def run_one(task) -> dict:
    path = Path(task["path"])
    if path.exists() and load(path).get("complete"):
        return {"skipped": True, "path": str(path)}
    t0 = time.perf_counter()
    spec = spec_for(task["method"], task["L"], task["alpha_sigma_scale"])
    inst = get_instance(task["N"], task["seed"], task["b"], "pilot")
    e_plus = -float(task["b"]) * task["N"]
    starts = []
    for s, n_iter_q in enumerate(task["iter_queries"]):
        n_iter = n_iter_q // 2
        rng_i = make_rng(task["seed"], task["N"], task["b"], task["L"], METHOD_ID[task["method"]], s, 2)
        x0 = _x0(task["method"], spec, task["L"], s, rng_i, inst.theta_K_prior)
        init = evaluate(task["N"], task["seed"], task["b"], "pilot", spec, x0, task["nf"])
        rng = make_rng(task["seed"], task["N"], task["b"], task["L"], METHOD_ID[task["method"]], s, 0)

        def objective(x, ctx, _s=s):
            return evaluate(task["N"], task["seed"], task["b"], "pilot", spec, x, task["nf"])

        result = spsa_start(objective, x0, spec, rng, n_iter)
        failed = bool(result["failed"])
        payload = init if failed or result["payload"] is None else result["payload"]
        energy = float(init["E"] if failed or result["E"] is None else result["E"])
        queries = 1 + int(result["queries"])
        starts.append({
            "start_id": s,
            "label": _start_label(task["method"], s, task["L"]),
            "failed": failed,
            "failure_stage": result["failure_stage"],
            "queries": queries,
            "init_queries": 1,
            "spsa_queries": int(result["queries"]),
            "E": energy,
            "F": float(payload["F"]),
            "p_leak": float(payload["p_leak"]),
            "norm": float(payload["norm"]),
            "E_init": float(init["E"]),
            "F_init": float(init["F"]),
            "escape_below_Eplus": bool(energy < e_plus - 1e-3 * task["N"]),
            "E_minus_Eplus": energy - e_plus,
            "x": np.asarray(result["x"], dtype=np.float64).tolist(),
            "physical": physical(result["x"], spec).tolist(),
            "which": result["which"],
        })
    winner = min(range(4), key=lambda i: (starts[i]["E"], i))
    best = starts[winner]
    fp = float(task["F_plus"])
    e0 = float(task["E0"])
    record = {
        **claim_block(),
        "complete": True,
        "prereg_sha256": task["prereg_sha256"],
        "kind": "basin_r2",
        "method": task["method"],
        "N": task["N"],
        "seed": task["seed"],
        "b": task["b"],
        "L": task["L"],
        "cap": task["cap"],
        "nf": task["nf"],
        "alpha_sigma_scale": task["alpha_sigma_scale"],
        "queries_counted": sum(s["queries"] for s in starts),
        "failed_starts": sum(1 for s in starts if s["failed"]),
        "selected_start": winner,
        "selected_label": best["label"],
        "E": best["E"],
        "F": best["F"],
        "F_init": best["F_init"],
        "E_init": best["E_init"],
        "p_leak": best["p_leak"],
        "E_plus": e_plus,
        "E0": e0,
        "F_plus": fp,
        "eta": (best["F"] - fp) / (1.0 - fp),
        "energy_gap_closed": (best["E_init"] - best["E"]) / (best["E_init"] - e0),
        "escape_below_Eplus": best["escape_below_Eplus"],
        "E_minus_Eplus": best["E_minus_Eplus"],
        "S_A_nats": task["S_A_nats"],
        "gap_E1_E0": task["gap_E1_E0"],
        "delta_even": task["delta_even"],
        "starts": starts,
        "runtime_seconds": time.perf_counter() - t0,
        "diagnostic_not_a_performance_claim": True,
        "not_an_expressivity_proof": True,
    }
    if record["failed_starts"] == 0 and record["queries_counted"] != task["cap"]:
        raise RuntimeError(f"queries {record['queries_counted']} != {task['cap']}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2))
    return {"skipped": False, "path": str(path), "E": record["E"], "F": record["F"]}


def _worker(task):
    try:
        return run_one(task)
    except Exception:
        import traceback
        err = traceback.format_exc()
        Path(str(task["path"]) + ".error.txt").parent.mkdir(parents=True, exist_ok=True)
        Path(str(task["path"]) + ".error.txt").write_text(err)
        return {"error": err, "path": task["path"]}


def _pool(tasks, workers: int):
    results = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(_worker, t) for t in tasks]
        done = 0
        for fut in as_completed(futs):
            results.append(fut.result())
            done += 1
            if done % 20 == 0 or done == len(futs):
                print(f"completed {done}/{len(futs)}", flush=True)
    errors = [r for r in results if r.get("error")]
    if errors:
        raise RuntimeError(errors[0]["error"])
    return results


def tasks_from(table, scale, sha) -> list[dict]:
    by = {(r["N"], r["seed"]): r for r in table["rows"]}
    out = []
    for method, caps in ARMS.items():
        for N in (6, 9):
            for seed in SEEDS:
                rec = by[(N, seed)]
                for cap in caps:
                    iters = iteration_queries(cap)
                    for L in (1, 2, 3):
                        out.append({
                            "method": method,
                            "N": N,
                            "seed": seed,
                            "b": float(rec["Bstar"]),
                            "L": L,
                            "cap": cap,
                            "iter_queries": iters,
                            "nf": 32 if "qaoa" not in method else 8,
                            "alpha_sigma_scale": scale,
                            "prereg_sha256": sha,
                            "F_plus": rec["F_plus"],
                            "E0": rec["E0"],
                            "S_A_nats": rec["S_A_nats"],
                            "gap_E1_E0": rec["gap_E1_E0"],
                            "delta_even": rec["delta_even"],
                            "path": str(OUT / "cells" / f"{method}__N{N}__s{seed}__L{L}__c{cap}.json"),
                        })
    return out


def summarize(table):
    cells = [load(p) for p in sorted((OUT / "cells").glob("*.json")) if load(p).get("complete")]
    idx = {(c["method"], c["N"], c["seed"], c["L"], c["cap"]): c for c in cells}
    rng = np.random.default_rng(0)

    def boot(vals):
        vals = np.asarray(vals, float)
        meds = [np.median(rng.choice(vals, size=vals.size, replace=True)) for _ in range(2000)]
        lo, hi = np.quantile(meds, [0.025, 0.975])
        return {"median": float(np.median(vals)), "min": float(vals.min()), "max": float(vals.max()),
                "values": [float(v) for v in vals], "ci95": [float(lo), float(hi)]}

    blocks = []
    for N in (6, 9):
        for L in (1, 2, 3):
            for cap in (128, 1024):
                block = {"N": N, "L": L, "cap": cap, "methods": {}}
                q = [idx[("basin_qaoa", N, s, L, cap)] for s in SEEDS]
                for method in ARMS:
                    if cap not in ARMS[method]:
                        continue
                    rows = [idx[(method, N, s, L, cap)] for s in SEEDS]
                    dF = [r["F"] - qq["F"] for r, qq in zip(rows, q)]
                    dE = [r["E"] - qq["E"] for r, qq in zip(rows, q)]
                    wins = [int(df > 1e-12) - int(df < -1e-12) for df in dF]
                    block["methods"][method] = {
                        "F": boot([r["F"] for r in rows]),
                        "E": boot([r["E"] for r in rows]),
                        "F_init": boot([r["F_init"] for r in rows]),
                        "eta": boot([r["eta"] for r in rows]),
                        "p_leak": boot([r["p_leak"] for r in rows]),
                        "E_minus_Eplus": boot([r["E_minus_Eplus"] for r in rows]),
                        "n_escape": int(sum(r["escape_below_Eplus"] for r in rows)),
                        "failed_starts": int(sum(r["failed_starts"] for r in rows)),
                        "queries": rows[0]["queries_counted"],
                        "paired_dF_vs_qaoa": None if method == "basin_qaoa" else boot(dF),
                        "paired_dE_vs_qaoa": None if method == "basin_qaoa" else boot(dE),
                        "F_wins_ties_losses": None if method == "basin_qaoa" else [
                            sum(w > 0 for w in wins), sum(w == 0 for w in wins), sum(w < 0 for w in wins)],
                    }
                blocks.append(block)
    doc = {"prereg_sha256": prereg_sha(), "n_cells": len(cells), "blocks": blocks, "bstar": table["rows"]}
    (OUT / "summary.json").write_text(json.dumps(doc, indent=2))
    lines = ["# R2 B2 BASIN at B* (diagnostic, not a performance claim)", "",
             "Real-α Nov9 only. No imaginary displacement. Original b=1/1.5 pilot was not repeated. "
             "A missing escape is not an expressivity proof. F is unconditioned.", ""]
    lines.append("| N | seed | B* | S_A nats | S_A bits | gap | Δ_even | E1 parity | F_+ |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in table["rows"]:
        lines.append(
            f"| {r['N']} | {r['seed']} | {r['Bstar']:.4f} | {r['S_A_nats']:.4f} | {r['S_A_bits']:.4f} | "
            f"{r['gap_E1_E0']:.4f} | {r['delta_even']:.4f} | {r['E1_parity']:.3f} | {r['F_plus']:.4f} |"
        )
    lines += ["", "| N | L | cap | method | med F | med E | med η | escapes | med (E-E_+) | med ΔF vs QAOA | wins |",
              "|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---|"]
    for b in blocks:
        for method, st in b["methods"].items():
            dF = "" if st["paired_dF_vs_qaoa"] is None else f"{st['paired_dF_vs_qaoa']['median']:+.4f}"
            w = "" if st["F_wins_ties_losses"] is None else "/".join(str(x) for x in st["F_wins_ties_losses"])
            lines.append(
                f"| {b['N']} | {b['L']} | {b['cap']} | {method} | {st['F']['median']:.4f} | {st['E']['median']:.3f} | "
                f"{st['eta']['median']:.3f} | {st['n_escape']}/8 | {st['E_minus_Eplus']['median']:+.4f} | {dF} | {w} |"
            )
    lines.append("")
    lines.append("Per-seed cells: `results/FT/basin_r2/cells/`. Machine summary: `summary.json`.")
    (OUT / "RESULTS.md").write_text("\n".join(lines) + "\n")
    return doc


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", default="all", choices=["bstar", "run", "summary", "all"])
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    sha = prereg_sha()
    table_path = OUT / "bstar_table.json"
    if args.phase in ("bstar", "all") or not table_path.exists():
        table = build_bstar()
        prep = sigma_and_cutoff(table["rows"])
        if not prep["cutoff_gate"]["pass"]:
            raise SystemExit(f"NF32 gate failed: {prep['cutoff_gate']}")
        table.update(prep)
        table_path.write_text(json.dumps(table, indent=2))
        print("scale", prep["alpha_sigma_scale"], "gate", prep["cutoff_gate"]["pass"], flush=True)
        if args.phase == "bstar":
            return
    else:
        table = load(table_path)
    if args.phase in ("run", "all"):
        tasks = tasks_from(table, float(table["alpha_sigma_scale"]), sha)
        print("tasks", len(tasks), flush=True)
        _pool([t for t in tasks], args.workers)
    if args.phase in ("summary", "all"):
        # NF=48 postpass on selected cap-1024 Nov9 points.
        post = []
        for path in sorted((OUT / "cells").glob("*__c1024.json")):
            doc = load(path)
            if "qaoa" in doc["method"]:
                continue
            spec = spec_for(doc["method"], doc["L"], doc["alpha_sigma_scale"])
            x = np.asarray(doc["starts"][doc["selected_start"]]["x"], dtype=np.float64)
            low = evaluate(doc["N"], doc["seed"], doc["b"], "pilot", spec, x, 32)
            tables = FockTables(get_instance(doc["N"], doc["seed"], doc["b"], "pilot"), 48)
            ext = "u0" if "u0" in doc["method"] else "uz"
            high = run_nov9(tables, unpack(spec, x), ext, "restricted", return_state=True)
            psi = high.pop("_psi")
            p_out = float(np.dot(np.abs(psi) ** 2, np.any(tables.fs >= 32, axis=1)))
            post.append({"method": doc["method"], "N": doc["N"], "seed": doc["seed"], "L": doc["L"],
                         "dE": abs(low["E"] - high["E"]), "dF": abs(low["F"] - high["F"]),
                         "p_out": p_out,
                         "pass": abs(low["E"] - high["E"]) <= 1e-8 and abs(low["F"] - high["F"]) <= 1e-8 and p_out <= 1e-8})
        (OUT / "postpass.json").write_text(json.dumps({"n": len(post), "n_pass": sum(r["pass"] for r in post),
                                                       "max_dE": max(r["dE"] for r in post), "rows": post}, indent=2))
        summarize(table)
        print("summary written", flush=True)


if __name__ == "__main__":
    main()
