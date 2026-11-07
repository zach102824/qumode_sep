"""Second search after the bootstrap ladder.

The bootstrap can sit in a weak basin. This pass adds adiabatic seeds,
extra random local searches, and differential evolution. It keeps a vector
only when its energy is lower. Fidelity is still not an objective.
"""
from __future__ import annotations

import time
from pathlib import Path

import thread_env  # noqa: F401
import numpy as np
from scipy.optimize import differential_evolution

from ansatz import Model, energy_and_grad, n_params
from common import (
    canonicalize_gs,
    dense_H,
    dump_json,
    parity_expectations,
    reflection_expectations,
    bit_reverse_perm,
)
from optimize import Pool, polish_starts
from report import build_verdict, write_figures, write_results_md, write_summary_csv
from run_round1 import RESULTS, SUPPORTS, embed_subspace_into_D, phase_hash, publish

from scipy.linalg import eigh


def symmetry_gaps(n: int, hc: np.ndarray, B: float, n_states: int = 16) -> dict:
    H = dense_H(hc, float(B), n)
    evals, evecs = eigh(H, subset_by_index=[0, n_states - 1])
    perm = bit_reverse_perm(n)
    rows = []
    accessible = None
    for k in range(n_states):
        psi = canonicalize_gs(evecs[:, k])
        P = parity_expectations(psi)["P_plus_weight"]
        R = reflection_expectations(psi, perm)["R_plus_weight"]
        row = {
            "k": k,
            "E": float(evals[k]),
            "gap_from_gs": float(evals[k] - evals[0]),
            "P_plus_weight": P,
            "R_plus_weight": R,
        }
        rows.append(row)
        if k > 0 and accessible is None and P > 0.99 and R > 0.99:
            accessible = row["gap_from_gs"]
    return {
        "n": n,
        "B": float(B),
        "full_space_gap": float(evals[1] - evals[0]),
        "accessible_PR_plus_gap": accessible,
        "lowest_states": rows,
        "note": (
            "The first excited state is the opposite-parity, opposite-reflection partner. "
            "Both QAOA and the shell circuit stay in P=+1, R=+1, which contains the ground state. "
            "The accessible gap is the gap to the next state in that sector."
        ),
    }


def bounds_for(method: str, L: int):
    n = n_params(method, L)
    if method == "qaoa":
        return [(-8.0, 8.0)] * L + [(0.0, 0.5 * np.pi)] * L
    if method == "shared_beta":
        return [(-8.0, 8.0)] + [(0.0, 0.5 * np.pi)] * L
    # Remaining entries are shell angles. Each has period pi/2, so one fundamental domain is enough.
    return [(-8.0, 8.0)] + [(-0.25 * np.pi, 0.25 * np.pi)] * (n - 1)


def adiabatic_starts(method: str, L: int, prev, rng: np.random.Generator):
    s = np.linspace(0.0, 1.0, L)
    starts = []
    g_values = np.linspace(0.2, 3.0, 8)
    b_values = np.linspace(0.05, 1.2, 6)
    if method == "qaoa":
        for gmax in g_values:
            for bmax in b_values:
                starts.append((f"adiab_{gmax:.2f}_{bmax:.2f}", np.concatenate([gmax * s, bmax * (1.0 - s)])))
    elif method == "shared_beta":
        for gmax in g_values:
            for bmax in b_values:
                starts.append((f"adiab_{gmax:.2f}_{bmax:.2f}", np.concatenate([[gmax], bmax * (1.0 - s)])))
    else:
        for gmax in (0.3, 0.8, 1.5, 2.5):
            for bmax in (0.15, 0.4, 0.8):
                for amp in (0.0, 0.15):
                    shell = np.zeros((L, 3))
                    shell[:, 0] = bmax * (1.0 - s)
                    shell[:, 1] = amp * np.sin(np.pi * s)
                    shell[:, 2] = amp * np.sin(2 * np.pi * s)
                    if method == "nov3_D":
                        x = np.concatenate([[gmax], shell.reshape(-1)])
                    elif method == "drop_c2":
                        x = np.concatenate([[gmax], np.column_stack([shell[:, 0], shell[:, 2]]).reshape(-1)])
                    else:
                        x = np.concatenate([[gmax], np.column_stack([shell[:, 0], shell[:, 1]]).reshape(-1)])
                    starts.append((f"adiab_{gmax:.2f}_{bmax:.2f}_{amp:.2f}", x))
    if prev is not None:
        starts.append(("keep_previous_best", np.asarray(prev, dtype=np.float64)))
    for i in range(12):
        width = n_params(method, L)
        starts.append((f"rescue_random_{i}", rng.normal(scale=0.35, size=width)))
    return starts


def adopt(model, gs, target, method, L, current, opt, log, label):
    if not opt.get("ok"):
        log(f"  {label} failed {opt.get('error')}")
        return current
    if current.get("status") == "complete" and float(opt["E_best"]) >= float(current["E"]) - 1e-10:
        current.setdefault("rescue_attempts", []).append(
            {"label": label, "E": opt["E_best"], "adopted": False, "tag": opt.get("best_tag")}
        )
        log(f"  {label} E={opt['E_best']:.6f} did not beat {current['E']:.6f}")
        return current
    record, psi = publish(model, gs, target, method, L, opt, float(current.get("seconds", 0.0)))
    if current.get("starts"):
        record["starts"] = list(current["starts"]) + list(opt.get("starts") or [])
        energies = [s["E"] for s in record["starts"] if s.get("E") is not None]
        record["n_starts"] = len(record["starts"])
        record["n_finished"] = len(energies)
        record["E_median"] = float(np.median(energies)) if energies else record["E_median"]
        record["nfev_sum"] = int(current.get("nfev_sum") or 0) + int(opt.get("nfev_sum") or 0)
    record["best_tag"] = f"{label}:{opt.get('best_tag')}"
    record["seconds"] = float(current.get("seconds", 0.0)) + float(opt.get("seconds", 0.0))
    record["rescue_attempts"] = list(current.get("rescue_attempts") or []) + [
        {"label": label, "E": record["E"], "adopted": True, "tag": record["best_tag"]}
    ]
    record["support"] = SUPPORTS[method]
    from run_round1 import save_record

    save_record(record, psi)
    log(f"  ADOPT {label} E={record['E']:.6f} F={record['F']:.4f} tag={record['best_tag']}")
    return record


def de_search(model, method, L, rng, maxiter, popsize, workers):
    bnds = bounds_for(method, L)

    def fun(x):
        return energy_and_grad(model, method, L, np.asarray(x, dtype=np.float64))[0]

    t0 = time.perf_counter()
    seeds = [int(rng.integers(0, 2**31 - 1)) for _ in range(2)]
    best = None
    nfev = 0
    for seed in seeds:
        res = differential_evolution(
            fun,
            bnds,
            seed=seed,
            maxiter=maxiter,
            popsize=popsize,
            mutation=0.5,
            recombination=0.7,
            atol=0.0,
            tol=1e-8,
            polish=True,
            workers=workers,
            updating="deferred" if workers != 1 else "immediate",
        )
        nfev += int(res.nfev)
        if best is None or res.fun < best.fun:
            best = res
    opt = {
        "ok": True,
        "E_best": float(best.fun),
        "x": np.asarray(best.x, dtype=np.float64).tolist(),
        "best_tag": "differential_evolution",
        "grad_norm": None,
        "n_starts": 2,
        "n_finished": 2,
        "nfev_sum": nfev,
        "E_median": float(best.fun),
        "starts": [
            {
                "tag": "differential_evolution",
                "x": np.asarray(best.x, dtype=np.float64).tolist(),
                "E": float(best.fun),
                "nfev": nfev,
                "success": bool(best.success),
                "grad_norm": None,
                "message": str(best.message),
                "error": None,
                "trace_len": int(best.nit),
                "trace_head": [],
                "trace_tail": [],
            }
        ],
        "best_trace": [],
        "seconds": time.perf_counter() - t0,
    }
    return opt


def main():
    log_path = RESULTS / "rescue.log"

    def log(msg):
        line = f"{time.strftime('%H:%M:%S')} {msg}"
        print(line, flush=True)
        with log_path.open("a") as fh:
            fh.write(line + "\n")

    gaps = {}
    targets = {}
    models = {}
    gs = {}
    for n in (6, 12):
        arrays = np.load(RESULTS / f"target_n{n}_arrays.npz")
        import json

        public = json.loads((RESULTS / f"target_n{n}.json").read_text())
        public["J"] = arrays["J"]
        public["hc_diag"] = arrays["hc"]
        public["gs"] = arrays["gs"]
        targets[n] = public
        models[n] = Model(n, public["J"], float(public["B"]), public["hc_diag"])
        gs[n] = np.asarray(public["gs"], dtype=np.complex128)
        log(f"symmetry gaps N={n}")
        gaps[n] = symmetry_gaps(n, public["hc_diag"], float(public["B"]))
        log(
            f"  full gap {gaps[n]['full_space_gap']:.6e} accessible {gaps[n]['accessible_PR_plus_gap']:.6e}"
        )
    dump_json(RESULTS / "symmetry_gaps.json", gaps)

    plan = []
    for n, methods, Ls, de_iter, de_pop, workers in (
        (6, ("qaoa", "shared_beta", "drop_c2", "drop_c3", "nov3_D"), range(1, 7), 60, 12, 1),
        (12, ("qaoa", "shared_beta", "drop_c2", "drop_c3", "nov3_D"), range(2, 7), 18, 8, 1),
        (6, ("qaoa",), range(7, 9), 40, 10, 1),
        (12, ("qaoa",), range(7, 9), 12, 6, 1),
    ):
        for L in Ls:
            for method in methods:
                plan.append((n, L, method, de_iter, de_pop, workers))

    records = {}
    for path in (RESULTS / "opt").glob("n*_L*_*.json"):
        import json

        row = json.loads(path.read_text())
        records[(row["n"], row["L"], row["method"])] = row

    for n, L, method, de_iter, de_pop, workers in plan:
        key = (n, L, method)
        current = records.get(key)
        if current is None or current.get("status") != "complete":
            log(f"skip missing {key}")
            continue
        log(f"rescue N={n} L={L} {method} current E={current['E']:.6f} F={current['F']:.4f}")
        tag_id = {"qaoa": 1, "shared_beta": 2, "drop_c2": 3, "drop_c3": 4, "nov3_D": 5}[method]
        rng = np.random.default_rng(10_000 + 100 * n + 10 * L + tag_id)
        prev = records.get((n, L - 1, method), {}).get("x_raw") if L > 1 else current["x_raw"]
        pool = Pool(models[n], 4)
        try:
            starts = adiabatic_starts(method, L, current["x_raw"], rng)
            # Subspace repair seeds for D, using the latest ablation vectors.
            if method == "nov3_D":
                for src_method in ("shared_beta", "drop_c2", "drop_c3"):
                    src = records.get((n, L, src_method))
                    if src and src.get("status") == "complete":
                        vec = embed_subspace_into_D(src_method, L, src["x_raw"])
                        if vec is not None:
                            starts.append((f"from_{src_method}", vec))
            t0 = time.perf_counter()
            polished = polish_starts(pool, method, L, starts, maxfun=250 if n == 12 else 400)
            polished["seconds"] = time.perf_counter() - t0
        finally:
            pool.close()
        current = adopt(models[n], gs[n], targets[n], method, L, current, polished, log, "adiabatic_random")
        log(f"  DE maxiter={de_iter} popsize={de_pop}")
        de = de_search(models[n], method, L, rng, de_iter, de_pop, workers)
        current = adopt(models[n], gs[n], targets[n], method, L, current, de, log, "DE")
        records[key] = current

    rows = [records[k] for k in sorted(records)]
    notes = [
        "Rescue pass added adiabatic seeds, random local searches, and differential evolution.",
        "A new vector replaced the published one only when its energy was lower.",
        "Fidelity was not used to accept a vector.",
        gaps[6]["note"],
        f"N=6 full-space gap {gaps[6]['full_space_gap']:.6e}, accessible P=R=+1 gap {gaps[6]['accessible_PR_plus_gap']:.6e}.",
        f"N=12 full-space gap {gaps[12]['full_space_gap']:.6e}, accessible P=R=+1 gap {gaps[12]['accessible_PR_plus_gap']:.6e}.",
        "The tiny full-space gap is the splitting to the inaccessible P=-1, R=-1 partner.",
    ]
    # Preserve the original run notes if present.
    summary_path = RESULTS / "summary.json"
    if summary_path.exists():
        import json

        old = json.loads(summary_path.read_text())
        notes = list(old.get("notes") or []) + notes
    verdict = build_verdict([r for r in rows if r.get("status") == "complete"])
    import json

    from run_round1 import zero_shot

    scale = float(targets[6]["kappa_mean_degree"]) / float(targets[12]["kappa_mean_degree"])
    transfer = []
    for L in range(1, 9):
        for method in ("qaoa", "shared_beta", "nov3_D", "drop_c2", "drop_c3"):
            src = records.get((6, L, method))
            if src is None or src.get("status") != "complete":
                continue
            transfer.extend(
                zero_shot(models[12], gs[12], targets[12], src["x_raw"], method, L, scale)
            )
    dump_json(RESULTS / "transfer.json", transfer)
    dump_json(RESULTS / "summary.json", {"rows": rows, "verdict": verdict, "notes": notes})
    dump_json(RESULTS / "verdict.json", verdict)
    write_summary_csv(RESULTS / "summary.csv", rows)
    write_figures(RESULTS / "figures", rows, {n: json.loads((RESULTS / f"target_n{n}.json").read_text()) for n in (6, 12)})
    write_results_md(RESULTS / "RESULTS.md", rows, {n: json.loads((RESULTS / f"target_n{n}.json").read_text()) for n in (6, 12)}, verdict, transfer, notes)
    log(f"verdict program_success={verdict['program_success']}")
    log(verdict["headline"])


if __name__ == "__main__":
    main()
