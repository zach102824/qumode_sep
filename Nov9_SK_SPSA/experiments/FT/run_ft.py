"""Bounded N9-FT dev screen. Exact energy, SPSA only, dev seeds only.

D2-D5 are the approved simulation definitions (Zach, 18:41 Asia/Shanghai,
2026-11-09). That approval is not a hardware-feasibility claim.
Validation seeds 11000..11031 and final seeds 20000..20999 are not read.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from Nov9_SK_SPSA.experiments.FT import APPROVAL, CLAIM_LEVEL, D2_D5_CONFIRMED
from Nov9_SK_SPSA.experiments.FT.conformance import run_conformance
from Nov9_SK_SPSA.experiments.FT.instance import build_instance
from Nov9_SK_SPSA.experiments.FT.params import (
    METHOD_ID,
    N_TRAIN,
    PILOT_SEEDS,
    TRAIN_SEEDS,
    initial_x,
    make_rng,
    n_iter_for_start_budget,
    physical,
    shift_physical,
    spec_for,
    unpack,
)
from Nov9_SK_SPSA.experiments.FT.sim import FockTables, get_instance, get_tables, run_nov3, run_nov9, run_qaoa
from Nov9_SK_SPSA.experiments.FT.spsa import spsa_start

ROOT = Path(__file__).resolve().parents[2]
RESULT = ROOT / "results" / "FT"
KEEP = (
    "E",
    "F",
    "p_leak",
    "E_zz",
    "E_xpen",
    "parity_code_unnormalized",
    "parity_code_conditional",
    "norm",
    "p_code",
    "ok",
)
COLD_METHODS = (
    "nov9_km_uz",
    "nov9_kmr1_uz",
    "nov9_km_kprior_uz",
    "nov9_km_u0",
    "nov9b_km",
    "nov3_km",
    "qaoa",
)
TRAIN_METHODS = ("train_nov9_km", "train_nov9_kmr1", "train_qaoa")
FT_OF = {"train_nov9_km": "ft_nov9_km", "train_nov9_kmr1": "ft_nov9_kmr1", "train_qaoa": "ft_qaoa"}
CAPS = (128, 256, 1024)
BS = (1.0, 1.5)
LS = (1, 2, 3)
NS = (6, 9)


def _conv(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    raise TypeError(type(o))


def dump(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, default=_conv, indent=2))
    tmp.replace(path)


def load(path: Path):
    return json.loads(path.read_text())


def git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except Exception:
        return "unknown"


def btag(b: float) -> str:
    return f"{int(round(10 * float(b))):02d}"


def clean_metrics(out: dict) -> dict:
    return {k: (bool(out[k]) if k == "ok" else float(out[k])) for k in KEEP}


def claim_block() -> dict:
    return {
        "experiment": "N9-FT",
        "claim_level": CLAIM_LEVEL,
        "D2_D5_approved_simulation_definitions": D2_D5_CONFIRMED,
        "approval": APPROVAL,
        "hardware_feasibility_claim": False,
        "freeze_validation_final_shot_mode": False,
        "fidelity": "unconditioned, no leakage renormalization, no postselection",
        "extensions": {
            "snap_f_ge_8": "phase 0",
            "UcZ_primary": "Uz: a term touching a leaked cavity is off; every other term acts",
            "UcZ_sensitivity": "U0 whole component unphased if any cavity is leaked",
            "energy": "per-term Hext: leaked pair contributes |J|, leaked qubit contributes +b",
            "readout": "ideal logical readout; compiled W_L cost is not modelled",
        },
        "optimizer": "SPSA only",
        "objective": "energy",
        "fidelity_not_used_for_stopping_or_selection": True,
        "ground_state_not_used_as_an_oracle_or_init": True,
    }


def evaluate(N, seed, b, role, spec, x, nf) -> dict:
    un = unpack(spec, x)
    if spec.family == "qaoa":
        out = run_qaoa(get_instance(N, seed, b, role), un)
    elif spec.family == "nov3":
        out = run_nov3(get_instance(N, seed, b, role), un)
    elif spec.family == "nov9b":
        out = run_nov9(get_tables(N, seed, b, 8, role), un, "strict", "strict8")
    else:
        ext = "u0" if "u0" in spec.method else "uz"
        out = run_nov9(get_tables(N, seed, b, nf, role), un, ext, "restricted")
    met = clean_metrics(out)
    met["n_queries"] = 1
    return met


def _categories(result, n_iter: int) -> dict:
    q = int(result["queries"])
    if not result["failed"]:
        return {"calibration": 8, "iterations": 2 * n_iter, "candidates": 2, "failures": 0}
    stage = result["failure_stage"]
    if stage == "calibration":
        return {"calibration": q, "iterations": 0, "candidates": 0, "failures": q}
    if stage == "iteration":
        return {"calibration": 8, "iterations": q - 8, "candidates": 0, "failures": q}
    return {"calibration": 8, "iterations": 2 * n_iter, "candidates": max(q - 8 - 2 * n_iter, 0), "failures": q}


def _run_one_start(task, spec, x0, start_id, objective, probe):
    init = probe(x0)
    rng = make_rng(task["rng_seed"], task["N"], task["b"], task["L"], METHOD_ID[task["method"]], start_id, 0)
    result = spsa_start(objective, x0, spec, rng, task["n_iter"], make_context=task.get("_ctx"))
    scored_as_x0 = False
    if result["failed"] or result["payload"] is None:
        scored_as_x0 = True
        payload = init
        energy = init["E"]
    else:
        payload = result["payload"]
        energy = float(result["E"])
    unpacked = unpack(spec, result["x"])
    phys = physical(result["x"], spec)
    return {
        "start_id": start_id,
        "init_name": "E" if start_id == 0 else "R",
        "failed": bool(result["failed"]),
        "failure_stage": result["failure_stage"],
        "scored_as_x0": scored_as_x0,
        "which": result["which"],
        "queries": int(result["queries"]),
        "categories": _categories(result, task["n_iter"]),
        "a_coeff": result["a_coeff"],
        "E": float(energy),
        "F": float(payload["F"]) if payload.get("F") is not None else None,
        "p_leak": float(payload["p_leak"]) if payload.get("p_leak") is not None else None,
        "parity_code_conditional": payload.get("parity_code_conditional"),
        "norm": payload.get("norm"),
        "ok": bool(payload.get("ok", False)),
        "init_report_only": init,
        "x": np.asarray(result["x"], dtype=np.float64),
        "physical": phys,
        "unpacked_selected": _plain_unpacked(unpacked),
        "trace_E": result["trace_E"],
        "trace_F_oracle_not_used": result["trace_F_oracle_not_used"],
    }


def _plain_unpacked(un: dict) -> dict:
    out = {}
    for k, v in un.items():
        if isinstance(v, np.ndarray):
            out[k] = v.tolist()
        elif isinstance(v, dict):
            out[k] = {kk: float(vv) for kk, vv in v.items()}
        elif isinstance(v, list):
            out[k] = [float(a) for a in v]
        else:
            out[k] = float(v) if isinstance(v, (float, np.floating)) else v
    return out


def _base_record(task, spec, inst_meta, starts, selected):
    qsum = {
        "calibration": 0,
        "iterations": 0,
        "candidates": 0,
        "restarts": 0,
        "monitoring": 0,
        "continuation": 0,
        "failures": 0,
    }
    for s in starts:
        for k, v in s["categories"].items():
            if k == "failures":
                qsum["failures"] += int(s["failed"])
            else:
                qsum[k] += int(v)
    counted = sum(int(s["queries"]) for s in starts)
    return {
        **claim_block(),
        "complete": True,
        "kind": task["kind"],
        "method": task["method"],
        "method_id": METHOD_ID[task["method"]],
        "N": task["N"],
        "seed": task.get("seed"),
        "b": task["b"],
        "L": task["L"],
        "cap": task["cap"],
        "nf": task["nf"],
        "alpha_sigma_scale": task["alpha_sigma_scale"],
        "family": spec.family,
        "features": list(spec.features),
        "kprior": spec.kprior,
        "n_starts": len(starts),
        "n_iter_per_start": task["n_iter"],
        "c_base": 1.0,
        "A": 25,
        "queries_counted": counted,
        "queries_report_only_initial": len(starts),
        "ledger": qsum | {"total_counted": counted, "shots": 0, "mode": "exact_energy"},
        "instance": inst_meta,
        "starts": [
            {kk: vv for kk, vv in s.items() if kk != "x"} | {"x": s["x"].tolist()}
            for s in starts
        ],
        "selected_start": selected,
        "runtime_seconds": None,
    }


def _instance_meta(inst) -> dict:
    return {
        "J_sha256": inst.J_sha256,
        "E0": inst.E0,
        "E1": inst.E1,
        "E0_odd": inst.E0_odd,
        "Delta_odd": inst.Delta_odd,
        "lam_max": inst.lam_max,
        "F_plus": inst.F_plus,
        "residual": inst.residual,
        "theta_K_prior": inst.theta_K_prior,
        "rms_J": inst.rms_J,
        "scale_K": inst.scale_K,
        "scale_M": inst.scale_M,
        "scale_R1": inst.scale_R1,
        "lam_min_J": inst.lam_min_J,
        "feature_rank": inst.feature_diag,
        "classical_feature_seconds": inst.classical_seconds,
        "gs_used_only_for_reporting": True,
    }


def _finish(record, starts, t0):
    winner = min(range(len(starts)), key=lambda i: (starts[i]["E"], starts[i]["start_id"]))
    s = starts[winner]
    inst = record["instance"]
    F = s["F"]
    Fp = inst.get("F_plus")
    E0 = inst.get("E0")
    E_init = s["init_report_only"]["E"]
    record["selected_start"] = s["start_id"]
    record["E"] = s["E"]
    record["F"] = F
    record["F_init"] = s["init_report_only"].get("F")
    record["E_init"] = E_init
    record["p_leak"] = s["p_leak"]
    record["parity_code_conditional"] = s["parity_code_conditional"]
    if F is None or Fp is None or abs(1.0 - float(Fp)) < 1e-15:
        record["eta"] = None
    else:
        record["eta"] = (float(F) - float(Fp)) / (1.0 - float(Fp))
    if E0 is None or abs(float(E_init) - float(E0)) < 1e-15:
        record["energy_gap_closed"] = None
    else:
        record["energy_gap_closed"] = (float(E_init) - float(s["E"])) / (float(E_init) - float(E0))
    record["failed_starts"] = int(sum(1 for st in starts if st["failed"]))
    record["large_parameters"] = _large(s["physical"])
    record["runtime_seconds"] = time.perf_counter() - t0
    record["selected_parameters"] = s["unpacked_selected"]
    return record


def _large(phys) -> bool:
    return bool(np.max(np.abs(phys)) > 20.0)


def run_cold(task) -> dict:
    path = Path(task["path"])
    if path.exists():
        doc = load(path)
        if doc.get("complete"):
            return {"skipped": True, "path": str(path), "E": doc.get("E"), "F": doc.get("F")}
    t0 = time.perf_counter()
    spec = spec_for(task["method"], task["L"], task["alpha_sigma_scale"])
    inst = get_instance(task["N"], task["seed"], task["b"], "pilot")

    def objective(x, ctx):
        return evaluate(task["N"], task["seed"], task["b"], "pilot", spec, x, task["nf"])

    def probe(x):
        return evaluate(task["N"], task["seed"], task["b"], "pilot", spec, x, task["nf"])

    task = dict(task)
    task["_ctx"] = None
    starts = []
    for s in range(4):
        rng_i = make_rng(task["seed"], task["N"], task["b"], task["L"], METHOD_ID[task["method"]], s, 2)
        x0 = initial_x(spec, task["L"], s, rng_i, inst.theta_K_prior)
        starts.append(_run_one_start(task, spec, x0, s, objective, probe))
    counted = sum(st["queries"] for st in starts)
    if all(not st["failed"] for st in starts) and counted != task["cap"]:
        raise RuntimeError(f"counted {counted} != cap {task['cap']}")
    record = _base_record(task, spec, _instance_meta(inst), starts, None)
    record = _finish(record, starts, t0)
    dump(record, path)
    return {"skipped": False, "path": str(path), "E": record["E"], "F": record["F"], "seconds": record["runtime_seconds"]}


class _BatchCursor:
    def __init__(self, seeds, offset):
        self.seeds = list(seeds)
        self.t = int(offset)

    def __call__(self, kind):
        if kind == "cand":
            return list(self.seeds)
        i = self.t % len(self.seeds)
        self.t += 1
        return [self.seeds[(i + k) % len(self.seeds)] for k in range(4)]


def run_train(task) -> dict:
    path = Path(task["path"])
    if path.exists() and load(path).get("complete"):
        return {"skipped": True, "path": str(path)}
    t0 = time.perf_counter()
    spec = spec_for(task["method"], task["L"], task["alpha_sigma_scale"])
    ref = get_instance(N_TRAIN, TRAIN_SEEDS[0], task["b"], "train")

    def objective(x, ctx):
        mets = [evaluate(N_TRAIN, int(seed), task["b"], "train", spec, x, task["nf"]) for seed in ctx]
        return {
            "E": float(np.mean([m["E"] for m in mets])),
            "F": float(np.mean([m["F"] for m in mets])),
            "p_leak": float(np.mean([m["p_leak"] for m in mets])),
            "parity_code_conditional": float(np.nanmean([m["parity_code_conditional"] for m in mets])),
            "norm": float(np.mean([m["norm"] for m in mets])),
            "ok": all(m["ok"] for m in mets),
            "n_queries": len(mets),
            "p_code": float(np.mean([m["p_code"] for m in mets])),
            "E_zz": float(np.mean([m["E_zz"] for m in mets])),
            "E_xpen": float(np.mean([m["E_xpen"] for m in mets])),
        }

    starts = []
    for s in range(4):
        local = dict(task)
        local["_ctx"] = _BatchCursor(TRAIN_SEEDS, offset=17 * s)
        rng_i = make_rng(TRAIN_SEEDS[0], N_TRAIN, task["b"], task["L"], METHOD_ID[task["method"]], s, 2)
        x0 = initial_x(spec, task["L"], s, rng_i, ref.theta_K_prior)
        # Report-only initial mean on the whole train pool. Not counted.
        init = objective(x0, list(TRAIN_SEEDS))
        init["n_queries"] = 0
        starts.append(_run_one_start(local, spec, x0, s, objective, lambda x, _init=init: _init if np.allclose(x, x0) else objective(x, list(TRAIN_SEEDS))))
    counted = sum(st["queries"] for st in starts)
    if all(not st["failed"] for st in starts) and counted != 4096:
        raise RuntimeError(f"train counted {counted} != 4096")
    meta = {
        "train_seeds": TRAIN_SEEDS,
        "N_train": N_TRAIN,
        "held_out_pilot_seeds": PILOT_SEEDS,
        "disjoint": True,
        "minibatch": 4,
        "candidate_pool": 8,
        "n_iter_per_start": 122,
        "cap_instance_queries": 4096,
        "objective": "mean energy",
        "theta_K_prior_of_first_train_seed_not_used_except_kprior_arm": ref.theta_K_prior,
        "classical_feature_seconds_first_seed": ref.classical_seconds,
    }
    record = _base_record(task, spec, meta, starts, None)
    record["seed"] = None
    record["queries_report_only_initial"] = 4 * len(TRAIN_SEEDS)
    record = _finish(record, starts, t0)
    # _finish expects F_plus. Train meta has no F_plus; guard.
    record["eta"] = None
    record["energy_gap_closed"] = None
    record["F_plus_not_defined_for_batch"] = True
    record["x_star"] = starts[record["selected_start"]]["x"].tolist()
    dump(record, path)
    return {"skipped": False, "path": str(path), "E": record["E"], "seconds": record["runtime_seconds"], "queries": counted}


def run_finetune(task) -> dict:
    path = Path(task["path"])
    if path.exists() and load(path).get("complete"):
        return {"skipped": True, "path": str(path), "E": load(path).get("E"), "F": load(path).get("F")}
    t0 = time.perf_counter()
    spec = spec_for(task["method"], task["L"], task["alpha_sigma_scale"])
    x_star = np.asarray(task["x_star"], dtype=np.float64)
    inst = get_instance(task["N"], task["seed"], task["b"], "pilot")
    if task["cap"] == 0:
        met = evaluate(task["N"], task["seed"], task["b"], "pilot", spec, x_star, task["nf"])
        record = {
            **claim_block(),
            "complete": True,
            "kind": "zeroshot",
            "method": task["method"],
            "method_id": METHOD_ID[task["method"]],
            "N": task["N"],
            "seed": task["seed"],
            "b": task["b"],
            "L": task["L"],
            "cap": 0,
            "nf": task["nf"],
            "alpha_sigma_scale": task["alpha_sigma_scale"],
            "train_method": task["train_method"],
            "x_star": x_star.tolist(),
            "queries_counted": 1,
            "queries_report_only_initial": 0,
            "ledger": {"zeroshot": 1, "total_counted": 1, "shots": 0, "mode": "exact_energy",
                       "fresh_shot_check": "not_run_exact_mode"},
            "E": met["E"],
            "F": met["F"],
            "F_init": met["F"],
            "E_init": met["E"],
            "p_leak": met["p_leak"],
            "parity_code_conditional": met["parity_code_conditional"],
            "norm": met["norm"],
            "instance": _instance_meta(inst),
            "failed_starts": 0,
            "selected_parameters": _plain_unpacked(unpack(spec, x_star)),
            "runtime_seconds": time.perf_counter() - t0,
        }
        Fp = inst.F_plus
        record["eta"] = (met["F"] - Fp) / (1.0 - Fp)
        record["energy_gap_closed"] = 0.0
        dump(record, path)
        return {"skipped": False, "path": str(path), "E": record["E"], "F": record["F"], "seconds": record["runtime_seconds"]}

    def objective(x, ctx):
        return evaluate(task["N"], task["seed"], task["b"], "pilot", spec, x, task["nf"])

    local = dict(task)
    local["n_iter"] = n_iter_for_start_budget(task["cap"])
    local["_ctx"] = None
    local["rng_seed"] = task["seed"]
    st = _run_one_start(local, spec, x_star, 0, objective, lambda x: objective(x, None))
    # The probe above is the same function and WAS counted inside spsa only.
    # _run_one_start also calls probe once. That call is report-only if probe is `objective`,
    # but objective does not increment a global counter; the spsa counter is inside spsa via n_queries.
    # The extra probe is a real energy evaluation. Label it report-only and do not add it to counted.
    if st["queries"] != task["cap"] and not st["failed"]:
        raise RuntimeError(f"finetune queries {st['queries']} != {task['cap']}")
    record = _base_record(local, spec, _instance_meta(inst), [st], 0)
    record["kind"] = "finetune"
    record["train_method"] = task["train_method"]
    record["x_star"] = x_star.tolist()
    record = _finish(record, [st], t0)
    dump(record, path)
    return {"skipped": False, "path": str(path), "E": record["E"], "F": record["F"], "seconds": record["runtime_seconds"]}


def _worker(task):
    try:
        kind = task["kind"]
        if kind == "cold":
            return run_cold(task)
        if kind == "train":
            return run_train(task)
        if kind == "finetune" or kind == "zeroshot":
            return run_finetune(task)
        raise ValueError(kind)
    except Exception:
        err = traceback.format_exc()
        path = Path(str(task["path"]) + ".error.txt")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(err)
        return {"error": err, "path": task["path"]}


def _pool(tasks, workers: int) -> list[dict]:
    if not tasks:
        return []
    results = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futs = [pool.submit(_worker, t) for t in tasks]
        done = 0
        for fut in as_completed(futs):
            results.append(fut.result())
            done += 1
            if done % 25 == 0 or done == len(futs):
                print(f"completed {done}/{len(futs)}", flush=True)
    return results


def prepare(nf_request: int | None) -> dict:
    """Cutoff ladder and the one global alpha-perturbation scale. No pilot seed is fit."""
    points = []
    configs = [
        ("worst_echo", [0.30, -0.30, 0.30], 0.25, np.zeros((3, 3)), {"K": 0.0, "M": 0.0}),
        ("init_L3", [0.06, -0.06, 0.06], 0.25, np.zeros((3, 3)), {"K": 0.0, "M": 0.0}),
        ("init_L1", [0.06], 0.25, np.zeros((1, 3)), {"K": 0.2, "M": -0.1}),
    ]
    ladder = [16, 20, 24, 32, 48]
    report = []
    for N in (6, 9):
        for b in BS:
            for name, alphas, gamma, shells, theta in configs:
                L = len(alphas)
                ref = _point(N, 10008, b, 48, alphas, gamma, shells[:L], theta)
                p_out = {}
                for nf in ladder:
                    got = _point(N, 10008, b, nf, alphas, gamma, shells[:L], theta)
                    p_out[nf] = _population_at_least(N, 10008, b, 48, alphas, gamma, shells[:L], theta, nf)
                    report.append({
                        "N": N, "b": b, "point": name, "nf": nf,
                        "dE": abs(got["E"] - ref["E"]),
                        "dF": abs(got["F"] - ref["F"]),
                        "d_pleak": abs(got["p_leak"] - ref["p_leak"]),
                        "norm": got["norm"],
                        "p_leak": got["p_leak"],
                        "p_outside_when_run_at_48": p_out[nf],
                        "alias_f_ge_16_into_code_at_48": ref["alias16"],
                    })
    chosen = None
    for nf in ladder:
        rows = [r for r in report if r["nf"] == nf]
        ok = all(
            r["dE"] <= 1e-8 and r["dF"] <= 1e-8 and r["d_pleak"] <= 1e-8
            and abs(r["norm"] - 1.0) <= 1e-9 and r["p_outside_when_run_at_48"] <= 1e-8
            for r in rows
        )
        if ok:
            chosen = nf
            break
    if nf_request is not None:
        chosen = int(nf_request)
    if chosen is None:
        raise RuntimeError("no NF on the ladder met 1e-8")

    scale = 1.0
    leak_rows = []
    for _try in range(3):
        leak_rows = []
        worst = 0.0
        for L in (1, 3):
            for b in BS:
                spec = spec_for("nov9_km_uz", L, scale)
                inst = get_instance(9, 10008, b, "train")
                x = initial_x(spec, L, 0, None, inst.theta_K_prior)
                base = evaluate(9, 10008, b, "train", spec, x, chosen)["p_leak"]
                d = np.zeros(spec.n)
                iu = spec.index("u0")
                d[iu] = spec.sigma[iu]
                dp = abs(evaluate(9, 10008, b, "train", spec, shift_physical(x, d, spec), chosen)["p_leak"] - base)
                dm = abs(evaluate(9, 10008, b, "train", spec, shift_physical(x, -d, spec), chosen)["p_leak"] - base)
                worst = max(worst, dp, dm)
                leak_rows.append({"L": L, "b": b, "scale": scale, "plus": dp, "minus": dm, "base": base})
        if worst <= 0.01:
            break
        scale *= 0.5
    timing = {}
    for N, L, nq in ((12, 3, 3), (15, 1, 2)):
        timing[f"N{N}_L{L}"] = _time_query(N, L, chosen, nq)
    doc = {
        **claim_block(),
        "nf_sim": chosen,
        "alpha_sigma_scale": scale,
        "alpha_leak_step_at_most_0.01": leak_rows,
        "ladder": report,
        "timing_feasibility_not_a_result": timing,
        "cutoff_seed": 10008,
        "cutoff_seed_role": "train pool, not a held-out inference seed",
    }
    dump(doc, RESULT / "cutoff.json")
    return doc


def _point(N, seed, b, nf, alphas, gamma, shells, theta):
    role = "train"
    tables = FockTables(build_instance(N, seed, b, role=role, with_spectrum=(N <= 12)), nf)
    un = {"gamma": gamma, "theta": theta, "alphas": list(alphas), "shells": np.asarray(shells, dtype=np.float64)}
    out = run_nov9(tables, un, "uz", "restricted", return_state=True)
    alias = _alias_population(tables, out["_psi"])
    met = clean_metrics(out)
    met["alias16"] = alias
    return met


def _population_at_least(N, seed, b, nf_run, alphas, gamma, shells, theta, level):
    role = "train"
    tables = FockTables(build_instance(N, seed, b, role=role, with_spectrum=(N <= 12)), nf_run)
    un = {"gamma": gamma, "theta": theta, "alphas": list(alphas), "shells": np.asarray(shells, dtype=np.float64)}
    out = run_nov9(tables, un, "uz", "restricted", return_state=True)
    mask = np.any(tables.fs >= int(level), axis=1)
    return float(np.dot(np.abs(out["_psi"]) ** 2, mask))


def _alias_population(tables: FockTables, psi: np.ndarray) -> float:
    alias = np.zeros(tables.dim, dtype=bool)
    for c in range(tables.C):
        f = tables.fs[:, c]
        alias |= (f >= 16) & ((f % 16) < 8)
    return float(np.dot(np.abs(psi) ** 2, alias))


def _time_query(N, L, nf, nq) -> dict:
    t0 = time.perf_counter()
    inst = build_instance(N, 10008, 1.0, role="train", with_spectrum=(N <= 12))
    tables = FockTables(inst, nf)
    build_s = time.perf_counter() - t0
    alphas = [0.06 if ell % 2 == 0 else -0.06 for ell in range(L)]
    shells = np.zeros((L, 3))
    un = {"gamma": 0.25, "theta": {"K": 0.0, "M": 0.0}, "alphas": alphas, "shells": shells}
    t1 = time.perf_counter()
    for _ in range(nq):
        run_nov9(tables, un, "uz", "restricted")
    dt = (time.perf_counter() - t1) / nq
    return {
        "seconds_per_query": dt,
        "build_seconds": build_s,
        "dim": tables.dim,
        "queries_timed": nq,
        "estimated_1024_query_seconds": dt * 1024,
        "spectrum": "dense eigh" if N <= 12 else "not computed (timing only)",
    }


def cold_tasks(nf, scale) -> list[dict]:
    tasks = []
    for method in COLD_METHODS:
        for N in NS:
            for b in BS:
                for L in LS:
                    for seed in PILOT_SEEDS:
                        for cap in CAPS:
                            tasks.append({
                                "kind": "cold",
                                "method": method,
                                "N": N,
                                "b": float(b),
                                "L": int(L),
                                "seed": int(seed),
                                "cap": int(cap),
                                "n_iter": n_iter_for_start_budget(cap // 4),
                                "nf": 8 if "nov9b" in method or method.startswith("nov3") or method == "qaoa" else nf,
                                "alpha_sigma_scale": scale,
                                "rng_seed": int(seed),
                                "path": str(RESULT / "cells" / f"cold__{method}__N{N}__b{btag(b)}__L{L}__s{seed}__c{cap}.json"),
                            })
    return tasks


def train_tasks(nf, scale) -> list[dict]:
    tasks = []
    for method in TRAIN_METHODS:
        for b in BS:
            for L in LS:
                tasks.append({
                    "kind": "train",
                    "method": method,
                    "N": N_TRAIN,
                    "b": float(b),
                    "L": int(L),
                    "cap": 4096,
                    "n_iter": 122,
                    "nf": nf if "qaoa" not in method else 8,
                    "alpha_sigma_scale": scale,
                    "rng_seed": TRAIN_SEEDS[0],
                    "path": str(RESULT / "train" / f"train__{method}__b{btag(b)}__L{L}.json"),
                })
    return tasks


def finetune_tasks(nf, scale) -> list[dict]:
    tasks = []
    for train_method, ft_method in FT_OF.items():
        for b in BS:
            for L in LS:
                src = RESULT / "train" / f"train__{train_method}__b{btag(b)}__L{L}.json"
                x_star = load(src)["x_star"]
                for N in NS:
                    for seed in PILOT_SEEDS:
                        for cap in (0, 128, 256):
                            tasks.append({
                                "kind": "zeroshot" if cap == 0 else "finetune",
                                "method": ft_method,
                                "train_method": train_method,
                                "N": N,
                                "b": float(b),
                                "L": int(L),
                                "seed": int(seed),
                                "cap": int(cap),
                                "n_iter": 0 if cap == 0 else n_iter_for_start_budget(cap),
                                "nf": nf if "qaoa" not in ft_method else 8,
                                "alpha_sigma_scale": scale,
                                "rng_seed": int(seed),
                                "x_star": x_star,
                                "path": str(RESULT / "cells" / f"ft__{ft_method}__N{N}__b{btag(b)}__L{L}__s{seed}__c{cap}.json"),
                            })
    return tasks


def postpass(nf: int) -> dict:
    rows = []
    methods = [m for m in COLD_METHODS if m.startswith("nov9_km") or m.startswith("nov9_kmr")]
    # nov9_km_uz, nov9_kmr1_uz, nov9_km_kprior_uz, nov9_km_u0. Not nov9b.
    methods = ["nov9_km_uz", "nov9_kmr1_uz", "nov9_km_kprior_uz", "nov9_km_u0"]
    for method in methods:
        for N in NS:
            for b in BS:
                for L in LS:
                    for seed in PILOT_SEEDS:
                        path = RESULT / "cells" / f"cold__{method}__N{N}__b{btag(b)}__L{L}__s{seed}__c1024.json"
                        if not path.exists():
                            continue
                        doc = load(path)
                        spec = spec_for(method, L, doc["alpha_sigma_scale"])
                        x = np.asarray(doc["starts"][doc["selected_start"]]["x"], dtype=np.float64)
                        low = evaluate(N, seed, b, "pilot", spec, x, nf)
                        tables = get_tables(N, seed, b, 48, "pilot")
                        ext = "u0" if "u0" in method else "uz"
                        high = run_nov9(tables, unpack(spec, x), ext, "restricted", return_state=True)
                        alias = _alias_population(tables, high["_psi"])
                        p_out = float(np.dot(np.abs(high["_psi"]) ** 2, np.any(tables.fs >= nf, axis=1)))
                        hm = clean_metrics(high)
                        rows.append({
                            "method": method, "N": N, "b": b, "L": L, "seed": seed,
                            "dE": abs(low["E"] - hm["E"]),
                            "dF": abs(low["F"] - hm["F"]),
                            "d_pleak": abs(low["p_leak"] - hm["p_leak"]),
                            "norm48": hm["norm"],
                            "p_outside_nf": p_out,
                            "alias16": alias,
                            "pass": abs(low["E"] - hm["E"]) <= 1e-8 and abs(low["F"] - hm["F"]) <= 1e-8 and abs(low["p_leak"] - hm["p_leak"]) <= 1e-8 and p_out <= 1e-8,
                        })
    doc = {"n": len(rows), "n_pass": int(sum(r["pass"] for r in rows)), "rows": rows,
           "max_dE": max((r["dE"] for r in rows), default=None),
           "max_dF": max((r["dF"] for r in rows), default=None),
           "max_alias16": max((r["alias16"] for r in rows), default=None)}
    dump(doc, RESULT / "postpass.json")
    return doc


def _cells():
    return sorted((RESULT / "cells").glob("*.json"))


def summarize() -> dict:
    rows = []
    for path in _cells():
        doc = load(path)
        if not doc.get("complete"):
            continue
        rows.append(doc)
    trains = [load(p) for p in sorted((RESULT / "train").glob("*.json"))]
    # index
    idx = {}
    for doc in rows:
        key = (doc["method"], doc["N"], float(doc["b"]), doc["L"], doc.get("seed"), doc["cap"])
        idx[key] = doc

    def series(method, N, b, L, cap, field):
        vals = []
        seeds = []
        for seed in PILOT_SEEDS:
            doc = idx.get((method, N, float(b), L, seed, cap))
            if doc is None or doc.get(field) is None:
                continue
            vals.append(float(doc[field]))
            seeds.append(seed)
        return seeds, np.asarray(vals, dtype=np.float64)

    rng = np.random.default_rng(0)

    def boot(vals):
        if vals.size == 0:
            return None
        meds = np.empty(2000)
        for i in range(2000):
            meds[i] = np.median(rng.choice(vals, size=vals.size, replace=True))
        lo, hi = np.quantile(meds, [0.025, 0.975])
        return {
            "n": int(vals.size),
            "median": float(np.median(vals)),
            "mean": float(np.mean(vals)),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
            "values": [float(v) for v in vals],
            "boot_median_ci95": [float(lo), float(hi)],
        }

    blocks = []
    for N in NS:
        for b in BS:
            for L in LS:
                for cap in list(CAPS) + [0]:
                    block = {"N": N, "b": b, "L": L, "cap": cap, "methods": {}}
                    present = []
                    for method in list(COLD_METHODS) + list(FT_OF.values()):
                        if cap == 0 and method in COLD_METHODS:
                            continue
                        if cap not in (0, 128, 256) and method in FT_OF.values():
                            continue
                        if cap in CAPS and method in FT_OF.values() and cap == 1024:
                            continue
                        seeds, F = series(method, N, b, L, cap, "F")
                        _, E = series(method, N, b, L, cap, "E")
                        _, eta = series(method, N, b, L, cap, "eta")
                        _, pleak = series(method, N, b, L, cap, "p_leak")
                        _, Fi = series(method, N, b, L, cap, "F_init")
                        _, gap = series(method, N, b, L, cap, "energy_gap_closed")
                        if F.size == 0:
                            continue
                        present.append(method)
                        block["methods"][method] = {
                            "F": boot(F), "E": boot(E), "eta": boot(eta), "p_leak": boot(pleak),
                            "F_init": boot(Fi), "energy_gap_closed": boot(gap), "seeds": seeds,
                        }
                    # paired vs qaoa at the same cap when both exist
                    qkey = "qaoa" if cap in CAPS else None
                    if cap in (128, 256) or cap in CAPS:
                        pass
                    ref_name = "qaoa" if "qaoa" in block["methods"] else None
                    if "ft_qaoa" in block["methods"]:
                        block["paired_vs_ft_qaoa"] = {}
                        for method in present:
                            if method == "ft_qaoa" or not str(method).startswith("ft_"):
                                continue
                            dF, dE, wins = [], [], [0, 0, 0]
                            for seed in PILOT_SEEDS:
                                a = idx.get((method, N, float(b), L, seed, cap))
                                q = idx.get(("ft_qaoa", N, float(b), L, seed, cap))
                                if a is None or q is None:
                                    continue
                                df = float(a["F"] - q["F"])
                                de = float(a["E"] - q["E"])
                                dF.append(df)
                                dE.append(de)
                                if df > 1e-12:
                                    wins[0] += 1
                                elif df < -1e-12:
                                    wins[2] += 1
                                else:
                                    wins[1] += 1
                            block["paired_vs_ft_qaoa"][method] = {
                                "dF": boot(np.asarray(dF)),
                                "dE": boot(np.asarray(dE)),
                                "F_wins_ties_losses": wins,
                            }
                    if ref_name:
                        block["paired_vs_qaoa"] = {}
                        for method in present:
                            if method == "qaoa":
                                continue
                            dF, dE, wins = [], [], [0, 0, 0]
                            for seed in PILOT_SEEDS:
                                a = idx.get((method, N, float(b), L, seed, cap))
                                q = idx.get(("qaoa", N, float(b), L, seed, cap))
                                if a is None or q is None:
                                    continue
                                df = float(a["F"] - q["F"])
                                de = float(a["E"] - q["E"])
                                dF.append(df)
                                dE.append(de)
                                if df > 1e-12:
                                    wins[0] += 1
                                elif df < -1e-12:
                                    wins[2] += 1
                                else:
                                    wins[1] += 1
                            block["paired_vs_qaoa"][method] = {
                                "dF": boot(np.asarray(dF)),
                                "dE": boot(np.asarray(dE)),
                                "F_wins_ties_losses": wins,
                                "E_lower_wins": int(sum(1 for de in dE if de < -1e-12)),
                            }
                    blocks.append(block)

    # transfer amortization
    amort = []
    for train_method, ft_method in FT_OF.items():
        for b in BS:
            for L in LS:
                src = RESULT / "train" / f"train__{train_method}__b{btag(b)}__L{L}.json"
                if not src.exists():
                    continue
                tr = load(src)
                amort.append({
                    "train_method": train_method,
                    "ft_method": ft_method,
                    "b": b,
                    "L": L,
                    "train_instance_queries": tr["queries_counted"],
                    "train_runtime_seconds": tr["runtime_seconds"],
                    "served_instances_both_N": 16,
                    "served_instances_same_N9": 8,
                    "amortized_train_queries_per_served_both_N": tr["queries_counted"] / 16,
                    "amortized_train_queries_per_served_N9_only": tr["queries_counted"] / 8,
                    "finetune_256_plus_zeroshot": 257,
                    "amortized_total_both_N_at_256": tr["queries_counted"] / 16 + 257,
                    "cold_1024_queries": 1024,
                    "note": "Divisor is this milestone only. Cross-N applies the N=9 x* to N=6. Classical feature builds are O(N^3) once per instance and are timed in classical_feature_seconds; they are not energy queries.",
                })

    # start-0 kprior contrast at cap 1024
    kprior_start0 = []
    for N in NS:
        for b in BS:
            for L in LS:
                dF = []
                for seed in PILOT_SEEDS:
                    a = idx.get(("nov9_km_kprior_uz", N, float(b), L, seed, 1024))
                    c = idx.get(("nov9_km_uz", N, float(b), L, seed, 1024))
                    if a is None or c is None:
                        continue
                    # start 0 payloads
                    dF.append(float(a["starts"][0]["F"] - c["starts"][0]["F"]))
                kprior_start0.append({"N": N, "b": b, "L": L, "dF_start0": boot(np.asarray(dF))})

    failures = []
    for doc in rows:
        if doc.get("failed_starts", 0):
            failures.append({"method": doc["method"], "N": doc["N"], "b": doc["b"], "L": doc["L"], "seed": doc.get("seed"), "cap": doc["cap"], "failed_starts": doc["failed_starts"]})

    csv_path = RESULT / "per_seed.csv"
    header = "kind,method,N,b,L,seed,cap,E,F,F_init,F_plus,eta,E0,energy_gap_closed,p_leak,parity,queries,failed_starts,norm\n"
    lines = [header]
    for doc in rows:
        inst = doc.get("instance") or {}
        lines.append(",".join(str(x) for x in [
            doc.get("kind"), doc.get("method"), doc.get("N"), doc.get("b"), doc.get("L"), doc.get("seed"),
            doc.get("cap"), doc.get("E"), doc.get("F"), doc.get("F_init"), inst.get("F_plus"), doc.get("eta"),
            inst.get("E0"), doc.get("energy_gap_closed"), doc.get("p_leak"), doc.get("parity_code_conditional"),
            doc.get("queries_counted"), doc.get("failed_starts"), "",
        ]) + "\n")
    csv_path.write_text("".join(lines))

    summary = {
        **claim_block(),
        "pilot_seeds": PILOT_SEEDS,
        "train_seeds": TRAIN_SEEDS,
        "validation_and_final_not_consumed": True,
        "blocks": blocks,
        "amortization": amort,
        "kprior_start0_dF": kprior_start0,
        "failures": failures,
        "n_cells": len(rows),
        "n_trains": len(trains),
        "per_seed_csv": str(csv_path),
    }
    dump(summary, RESULT / "summary.json")
    _write_md(summary)
    return summary


def _fmt(block_method, key):
    stat = block_method.get(key)
    if not stat:
        return ""
    return f"{stat['median']:.4f}"


def _write_md(summary: dict) -> None:
    lines = []
    lines.append("# N9-FT bounded dev results (provisional mathematical simulation)")
    lines.append("")
    lines.append("D2–D5 are not user-confirmed. These numbers are exact-energy dev simulations under the recommended provisional extensions (SNAP phase 0 above level 7, Uz primary, U0 sensitivity, per-term H^ext, ideal number-resolved readout, ideal W_L). They are not a physical, hardware, FREEZE, validation, or shot-mode claim. The compiled cost of ideal W_L is not modelled. Validation 11000..11031 and final 20000..20999 were not read.")
    lines.append("")
    lines.append("Held-out inference seeds are 10000..10007. Transfer training uses dev seeds 10008..10015 only, so a train instance is never an inference instance. The consultation text trained on 10000..10015; that overlap is not used.")
    lines.append("")
    lines.append("Zach approved D2–D5 at 18:41 Asia/Shanghai on 2026-11-09 as simulation definitions: SNAP phase 0 above level 7, Uz, per-term |J_ij| and +b per leaked qubit, and ideal logical readout. U0 stays a sensitivity. Fidelity is unconditioned. This is not a hardware-feasibility claim.")
    lines.append("")
    lines.append("The local stream reported one preliminary Nov3rd point that FT did not reproduce and that has no QAOA pair: N=18, seed 10000, b=1, L=1, F_+ about 0.21317 to F about 0.61289 at 1024 queries.")
    lines.append("")
    lines.append("Objective is energy. Fidelity, F_+, and the ground state are report-only. SPSA is the only optimizer. QAOA uses the same SPSA rule, the same caps, and the same transfer pipeline. One shared γ and one shared coefficient per feature multiply every layer.")
    lines.append("")
    lines.append("## Medians at cap 1024")
    lines.append("")
    lines.append("| N | b | L | method | med F | min F | max F | med E | med eta | med p_leak | med F_init |")
    lines.append("|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for block in summary["blocks"]:
        if block["cap"] != 1024:
            continue
        for method, stat in block["methods"].items():
            F, E, eta, pl, Fi = stat["F"], stat["E"], stat["eta"], stat["p_leak"], stat["F_init"]
            lines.append(
                f"| {block['N']} | {block['b']} | {block['L']} | {method} | {F['median']:.4f} | {F['min']:.4f} | {F['max']:.4f} | {E['median']:.4f} | {eta['median']:.4f} | {pl['median']:.4f} | {Fi['median']:.4f} |"
            )
    lines.append("")
    lines.append("## Paired ΔF against the matched QAOA pipeline (positive means the row is higher fidelity)")
    lines.append("")
    lines.append("Cold rows are paired with cold QAOA at the same cap. Transfer rows are paired with QAOA that used the same train pool, the same 4096-query training budget, and the same per-instance cap.")
    lines.append("")
    lines.append("| N | b | L | cap | method | med ΔF | 95% boot CI | wins/ties/losses | med ΔE |")
    lines.append("|---|---|---|---:|---|---:|---|---|---:|")
    for block in summary["blocks"]:
        paired = {}
        paired.update(block.get("paired_vs_qaoa") or {})
        paired.update(block.get("paired_vs_ft_qaoa") or {})
        for method, stat in paired.items():
            if stat["dF"] is None:
                continue
            ci = stat["dF"]["boot_median_ci95"]
            w = stat["F_wins_ties_losses"]
            lines.append(
                f"| {block['N']} | {block['b']} | {block['L']} | {block['cap']} | {method} | {stat['dF']['median']:.4f} | [{ci[0]:.4f}, {ci[1]:.4f}] | {w[0]}/{w[1]}/{w[2]} | {stat['dE']['median']:.4f} |"
            )
    lines.append("")
    lines.append("## Paired ΔF against cold QAOA, including transfer rows")
    lines.append("")
    lines.append("| N | b | L | cap | method | med ΔF | 95% boot CI | wins/ties/losses | med ΔE |")
    lines.append("|---|---|---|---:|---|---:|---|---|---:|")
    for block in summary["blocks"]:
        for method, stat in (block.get("paired_vs_qaoa") or {}).items():
            if stat["dF"] is None:
                continue
            ci = stat["dF"]["boot_median_ci95"]
            w = stat["F_wins_ties_losses"]
            lines.append(
                f"| {block['N']} | {block['b']} | {block['L']} | {block['cap']} | {method} | {stat['dF']['median']:.4f} | [{ci[0]:.4f}, {ci[1]:.4f}] | {w[0]}/{w[1]}/{w[2]} | {stat['dE']['median']:.4f} |"
            )
    lines.append("")
    lines.append("## Transfer amortization")
    lines.append("")
    lines.append("The training target is 4096 instance energy queries (4 starts, minibatch of 4, candidate pool of all 8 train seeds). A start that aborts on norm drift is scored as its initial point and spends only the queries it used; the ledger stores that lower total. Each trained vector serves 8 held-out seeds at N=6 and 8 at N=9.")
    lines.append("")
    for row in summary["amortization"]:
        lines.append(
            f"- {row['train_method']} b={row['b']} L={row['L']}: train {row['train_instance_queries']} queries, "
            f"{row['amortized_train_queries_per_served_both_N']:.1f} per served instance across both N, "
            f"plus 256 finetune and 1 zero-shot = {row['amortized_total_both_N_at_256']:.1f} amortized versus cold 1024."
        )
    lines.append("")
    lines.append("Full per-seed rows: `results/FT/per_seed.csv`. Machine-readable medians: `results/FT/summary.json`. Reading of this pilot: `results/FT/FINDINGS.md`.")
    lines.append("")
    (RESULT / "SUMMARY.md").write_text("\n".join(lines) + "\n")


def _save_conformance():
    rows = run_conformance()
    failed = [r for r in rows if not r["ok"]]
    dump({"passed": len(rows) - len(failed), "failed": len(failed), "rows": rows, **claim_block()}, RESULT / "conformance.json")
    if failed:
        raise SystemExit("conformance failed: " + ",".join(r["name"] for r in failed))
    print(f"conformance {len(rows)} passed", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", default="all", choices=["test", "prepare", "cold", "train", "finetune", "postpass", "summary", "all"])
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--nf", type=int, default=0)
    args = parser.parse_args()
    RESULT.mkdir(parents=True, exist_ok=True)
    if args.phase in ("test", "all"):
        _save_conformance()
        if args.phase == "test":
            return
    manifest_path = RESULT / "MANIFEST.json"
    if args.phase in ("prepare", "all") or not manifest_path.exists():
        doc = prepare(None if args.nf == 0 else args.nf)
        manifest = {
            **claim_block(),
            "git_sha": git_sha(),
            "base_consultation": "92e094dde64839d429c751f972871737cd110ac5",
            "nf_sim": doc["nf_sim"],
            "alpha_sigma_scale": doc["alpha_sigma_scale"],
            "pilot_seeds": PILOT_SEEDS,
            "train_seeds": TRAIN_SEEDS,
            "caps": list(CAPS),
            "cold_methods": list(COLD_METHODS),
            "c_base": 1.0,
            "spall": {"A": 25, "alpha": 0.602, "gamma": 0.101, "target_step_over_sigma": 0.5, "clip_over_sigma": 2.0},
        }
        dump(manifest, manifest_path)
        print("prepared", doc["nf_sim"], "scale", doc["alpha_sigma_scale"], flush=True)
        if args.phase == "prepare":
            return
    manifest = load(manifest_path)
    nf, scale = int(manifest["nf_sim"]), float(manifest["alpha_sigma_scale"])
    workers = int(args.workers)
    if args.phase in ("cold", "all"):
        tasks = cold_tasks(nf, scale)
        print(f"cold tasks {len(tasks)}", flush=True)
        _pool(tasks, workers)
    if args.phase in ("train", "all"):
        tasks = train_tasks(nf, scale)
        print(f"train tasks {len(tasks)}", flush=True)
        _pool(tasks, workers)
    if args.phase in ("finetune", "all"):
        tasks = finetune_tasks(nf, scale)
        print(f"finetune tasks {len(tasks)}", flush=True)
        _pool(tasks, workers)
    if args.phase in ("postpass", "all"):
        doc = postpass(nf)
        print("postpass", doc["n_pass"], "/", doc["n"], "max dE", doc["max_dE"], flush=True)
    if args.phase in ("summary", "all"):
        summarize()
        print("summary written", flush=True)


if __name__ == "__main__":
    main()
