"""Run the round-1 A sweep.

Usage, from the repository root:
    python3 -m Nov_3rd.tfim_shared_gamma.round1_A.run_round1 --budget-min 80
"""
from __future__ import annotations

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import argparse
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from .circuit import eta_of, layer_observables, metrics_from_params, state_from_params
from .common import (
    CONTRACT,
    METHOD_ORDER,
    PRIMARY_TRANSFER_SCALE,
    RESULTS,
    log,
    param_count,
    sha256_state,
)
from .hamiltonian import (
    build_problem_from_selection,
    feature_public,
    select_field,
    selection_public,
)
from .optimize import optimize_method
from .report import comparisons, dump_summary, gamma_ramp_section, write_markdown
from .validate import domain_wall_report, gradient_errors, run_validation


def _git_hash() -> str | None:
    repo = Path(__file__).resolve().parents[3]
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _fmt_x(x) -> list[str]:
    return [format(float(v), ".17g") for v in np.asarray(x, dtype=np.float64).reshape(-1)]


def _scale_gamma(method: str, x: np.ndarray, L: int, scale: float) -> np.ndarray:
    y = np.array(x, dtype=np.float64, copy=True)
    if method == "qaoa":
        y[:L] *= scale
    else:
        y[0] *= scale
    return y


def _record(problem, out: dict) -> dict:
    x = np.asarray(out["x"], dtype=np.float64)
    met = metrics_from_params(problem, out["method"], x, out["L"])
    if abs(met["E"] - out["E"]) > 1e-7:
        raise RuntimeError(
            f"replay energy {met['E']} disagrees with optimizer {out['E']} "
            f"for {out['method']} N={problem.n} L={out['L']}"
        )
    psi = met.pop("psi")
    layers = layer_observables(problem, out["method"], x, out["L"])
    gamma = [float(v) for v in out["gamma"]]
    # Shared-gamma methods must not have grown a per-layer gamma.
    if out["method"] != "qaoa" and max(gamma) - min(gamma) > 1e-12:
        raise RuntimeError("shared gamma split across layers")
    rec = {
        "method": out["method"],
        "n": int(problem.n),
        "L": int(out["L"]),
        "param_count": param_count(out["method"], out["L"]),
        "B": problem.B,
        "E0": problem.E0,
        "Emax": problem.Emax,
        "E": met["E"],
        "energy_error": met["E_minus_E0"],
        "F": met["F"],
        "eta": met["eta"],
        "norm": met["norm"],
        "variational_ok": bool(met["E"] >= problem.E0 - 1e-7),
        "x_exact": _fmt_x(x),
        "gamma": gamma,
        "theta": out["theta"],
        "shell": out["shell"],
        "beta": out["beta"],
        "init_label": out["init_label"],
        "n_eval": out["n_eval"],
        "eval_cap": out["eval_cap"],
        "n_starts": out["n_starts"],
        "n_screen": out["n_screen"],
        "seconds": out["seconds"],
        "restart_E_min": out["restart_E_min"],
        "restart_E_median": out["restart_E_median"],
        "restart_E_max": out["restart_E_max"],
        "restart_E_std": out["restart_E_std"],
        "bound_indices": out["bound_indices"],
        "widened_bounds": out["widened_bounds"],
        "best_F_diagnostic": out["best_F_diagnostic"],
        "trace_best_start": out["trace_best_start"],
        "starts": out["starts"],
        "state_sha256_f64": sha256_state(psi),
        "state_sha256_round10": sha256_state(psi, decimals=10),
        "layers": layers,
        "H_hash": problem.H_hash,
        "rescue": bool(out.get("rescue", False)),
        "objective": "energy",
        "fidelity_used_for_selection": False,
    }
    return rec


def _job_path(n: int, L: int, method: str) -> Path:
    return RESULTS / "jobs" / f"n{n}_L{L}_{method}.json"


def _load_job(path: Path, H_hash: str) -> dict | None:
    if not path.exists():
        return None
    import json

    rec = json.loads(path.read_text())
    if rec.get("H_hash") != H_hash:
        return None
    rec["x"] = np.array([float(v) for v in rec["x_exact"]], dtype=np.float64)
    return rec


def _save_target(sel: dict, problem) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    public = selection_public(sel)
    public["features"] = {name: feature_public(feat) for name, feat in problem.features.items()}
    public["rms_J"] = problem.rms_J
    public["J_intra"] = problem.meta["J_intra"]
    public["sum_ei2_identity"] = problem.meta["identity"]
    hc = problem.hc_diag
    fk = problem.features["physical"].fk_diag
    public["Hc_Fk_correlation"] = float(np.corrcoef(hc, fk)[0, 1])
    public["domain_wall"] = domain_wall_report(problem)
    public["kappa_mean"] = problem.kappa_mean
    from .common import json_dump

    json_dump(RESULTS / f"target_n{problem.n}.json", public)
    np.savez_compressed(
        RESULTS / f"target_n{problem.n}.npz",
        J=problem.J,
        hc_diag=problem.hc_diag,
        gs=problem.gs,
        K_physical=problem.features["physical"].K_raw,
        K_physical_norm=problem.features["physical"].K_norm,
        K_random_norm=problem.features["random"].K_norm,
        K_refl_norm=problem.features["refl"].K_norm,
        w1=problem.w1,
        w2=problem.w2,
        w3=problem.w3,
        B=np.array([problem.B]),
    )


def _optimize_and_record(problem, method, L, prev, siblings, periods, **kwargs) -> dict:
    prev_opt = None
    if prev is not None:
        prev_opt = {"x": np.asarray(prev["x"], dtype=np.float64)}
    sib = {}
    for name, rec in (siblings or {}).items():
        if rec is not None:
            sib[name] = {"x": np.asarray(rec["x"], dtype=np.float64)}
    out = optimize_method(
        problem,
        method,
        L,
        prev_opt,
        sib,
        periods,
        **kwargs,
    )
    rec = _record(problem, out)
    return rec


def _transfer(src: dict, dst_problem, method: str, rec: dict) -> list[dict]:
    L = int(rec["L"])
    x = np.asarray(rec["x"], dtype=np.float64)
    scales = {
        "raw": 1.0,
        "kappa": float(src.kappa_mean / dst_problem.kappa_mean),
        "rms": float(src.rms_J / dst_problem.rms_J),
    }
    rows = []
    for name, scale in scales.items():
        y = _scale_gamma(method, x, L, scale)
        met = metrics_from_params(dst_problem, method, y, L)
        psi = met.pop("psi")
        rows.append(
            {
                "L": L,
                "method": method,
                "scale": name,
                "primary": bool(name == PRIMARY_TRANSFER_SCALE),
                "gamma_scale": scale,
                "theta_scale": 1.0,
                "E": met["E"],
                "energy_error": met["E_minus_E0"],
                "F": met["F"],
                "eta": met["eta"],
                "norm": met["norm"],
                "state_sha256_round10": sha256_state(psi, decimals=10),
                "trained_on": src.n,
                "evaluated_on": dst_problem.n,
                "retrained": False,
            }
        )
    return rows


def run(budget_min: float, depths: list[int], ns: list[int], force: bool) -> dict:
    t0 = time.perf_counter()
    deadline = t0 + budget_min * 60.0
    RESULTS.mkdir(parents=True, exist_ok=True)
    failures: list[str] = []
    skipped: list[str] = []

    validation = run_validation()
    if not validation["passed"]:
        raise RuntimeError("validation failed; sweep not started")
    periods = validation["periods"]
    if any(p is None for p in periods):
        log(f"shell periods partially unverified: {validation['period_info']}")
    else:
        log(f"shell periods accepted for interpolation: {periods}")

    problems = {}
    target_public = {}
    for n in ns:
        log(f"selecting entropy-max field at N={n}")
        sel = select_field(n)
        problem = build_problem_from_selection(sel)
        problems[n] = problem
        _save_target(sel, problem)
        target_public[str(n)] = json_target_view(RESULTS / f"target_n{n}.json")
        log(
            f"  N={n} B={problem.B:.4f} E0={problem.E0:.6f} gap={problem.gap:.3e} "
            f"S={problem.S_log2:.6f} P={problem.parity_expectation:.4f} "
            f"R={problem.reflection_expectation:.4f} hash={problem.H_hash[:12]}"
        )
        if abs(problem.parity_expectation - 1) > 1e-6 or abs(problem.reflection_expectation - 1) > 1e-6:
            failures.append(
                f"N={n} full-space GS is outside P=R=+1 "
                f"(P={problem.parity_expectation}, R={problem.reflection_expectation}); "
                "symmetric circuits cannot reach it"
            )
        if problem.gs_negative_mass > 1e-8:
            failures.append(f"N={n} GS negative mass {problem.gs_negative_mass}")
        if problem.residual > 1e-7:
            failures.append(f"N={n} GS residual {problem.residual}")
        if sel["endpoint_after_expansion"]:
            failures.append(f"N={n} entropy max still at an endpoint, B={problem.B}")
        low = sel.get("low_lying") or {}
        if low.get("states"):
            ov = low["states"][0]["overlap_gs"]
            if ov < 1 - 1e-6:
                failures.append(f"N={n} low-lying recomputation overlap {ov}")
        if n >= 12:
            err = gradient_errors(problem, "family_A", 2, seed=7)
            log(f"  N={n} gradient spot check max abs {err:.3e}")
            if err > 5e-4:
                raise RuntimeError(f"N={n} gradient check failed: {err}")

    # In-memory records keyed like the optimizer's previous-depth lookup.
    mem: dict[tuple[int, int, str], dict] = {}

    def remaining() -> float:
        return deadline - time.perf_counter()

    def run_job(n, L, method, **kwargs) -> dict | None:
        path = _job_path(n, L, method)
        if not force:
            loaded = _load_job(path, problems[n].H_hash)
            if loaded is not None:
                log(f"resume N={n} L={L} {method} E={loaded['E']:.6f}")
                mem[(n, L, method)] = loaded
                return loaded
        if L <= 4 and remaining() < 15:
            skipped.append(f"N={n} L={L} {method} (under 15s left)")
            return None
        if L >= 5 and remaining() < 90:
            skipped.append(f"N={n} L={L} {method} (under 90s left before L>=5)")
            return None
        prev = mem.get((n, L - 1, method))
        siblings = {}
        for earlier in METHOD_ORDER:
            if earlier == method:
                break
            if (n, L, earlier) in mem:
                siblings[earlier] = mem[(n, L, earlier)]
        log(f"optimize N={n} L={L} {method} remaining={remaining()/60:.1f} min")
        rec = _optimize_and_record(
            problems[n], method, L, prev, siblings, periods, **kwargs
        )
        from .common import json_dump

        json_dump(path, {k: v for k, v in rec.items() if k != "x"})
        rec["x"] = np.array([float(v) for v in rec["x_exact"]])
        mem[(n, L, method)] = rec
        log(
            f"  E={rec['E']:.6f} err={rec['energy_error']:.3e} F={rec['F']:.4f} "
            f"eta={rec['eta']:.4f} theta={rec['theta']} init={rec['init_label']} "
            f"evals={rec['n_eval']} t={rec['seconds']:.1f}s"
        )
        if not rec["variational_ok"]:
            failures.append(f"variational violation N={n} L={L} {method} E={rec['E']}")
        if rec["bound_indices"]:
            failures.append(
                f"bound saturation N={n} L={L} {method} indices={rec['bound_indices']}"
            )
        return rec

    # Priority: every method at L=1..4, N=6 then N=12, then higher depth.
    for L in depths:
        if L > 4:
            continue
        for n in ns:
            for method in METHOD_ORDER:
                run_job(n, L, method)

    # Theta=0 is a legal family-A point. Missing it is an optimizer failure, not a feature result.
    for L in depths:
        for n in ns:
            A = mem.get((n, L, "family_A"))
            D = mem.get((n, L, "native_D"))
            if A and D and A["E"] > D["E"] + 1e-5:
                failures.append(
                    f"N={n} L={L} family A energy {A['E']:.6f} is worse than native D "
                    f"{D['E']:.6f}; the theta=0 basin was missed"
                )

    # L=1 shared-gamma must reproduce independent QAOA. A miss is an optimizer bug.
    for n in ns:
        a = mem.get((n, 1, "qaoa"))
        b = mem.get((n, 1, "shared_qaoa"))
        if a and b and abs(a["E"] - b["E"]) > 1e-5:
            failures.append(
                f"N={n} L=1 shared QAOA energy {b['E']:.6f} disagrees with "
                f"independent QAOA {a['E']:.6f}"
            )

    def rescue_qaoa(n: int, L: int) -> None:
        if (n, L, "qaoa") not in mem:
            run_job(n, L, "qaoa")
        current = mem.get((n, L, "qaoa"))
        if current is None or remaining() < 60:
            skipped.append(f"rescue QAOA N={n} L={L}")
            return
        log(f"rescue QAOA N={n} L={L}")
        prev = mem.get((n, L - 1, "qaoa"))
        rec = _optimize_and_record(
            problems[n],
            "qaoa",
            L,
            prev,
            {},
            periods,
            eval_cap=2500,
            n_fresh=12,
            seed_tag=90,
            do_screen=False,
        )
        rec["rescue"] = True
        if rec["E"] < current["E"] - 1e-10:
            from .common import json_dump

            json_dump(_job_path(n, L, "qaoa_prerescue"), {k: v for k, v in current.items() if k != "x"})
            current_note = dict(current)
            current_note["superseded_by_rescue"] = True
            rec["prerescue_E"] = current["E"]
            rec["prerescue_F"] = current["F"]
            json_dump(_job_path(n, L, "qaoa"), {k: v for k, v in rec.items() if k != "x"})
            rec["x"] = np.array([float(v) for v in rec["x_exact"]])
            mem[(n, L, "qaoa")] = rec
            log(f"  rescue improved E {current_note['E']:.6f} -> {rec['E']:.6f}")
        else:
            log(f"  rescue did not improve E (best {rec['E']:.6f} vs {current['E']:.6f})")
            mem[(n, L, "qaoa")]["rescue_attempted"] = True
            mem[(n, L, "qaoa")]["rescue_E"] = rec["E"]

    # Apparent energy wins get a stronger QAOA search and an L+1 check.
    for L in [d for d in depths if d <= 4]:
        for n in ns:
            A = mem.get((n, L, "family_A"))
            Q = mem.get((n, L, "qaoa"))
            if A is None or Q is None:
                continue
            if A["E"] < Q["E"] - 1e-8:
                rescue_qaoa(n, L)
            Q = mem.get((n, L, "qaoa"))
            A = mem.get((n, L, "family_A"))
            if A is None or Q is None:
                continue
            if (n, L + 1, "qaoa") not in mem and (L + 1) <= max(depths + [L + 1]):
                if remaining() > 90:
                    run_job(n, L + 1, "qaoa")
            Q1 = mem.get((n, L + 1, "qaoa"))
            if Q1 is not None and A["E"] < Q1["E"] - 1e-8:
                rescue_qaoa(n, L + 1)

    for L in depths:
        if L < 5:
            continue
        for n in ns:
            for method in METHOD_ORDER:
                if (n, L, method) in mem:
                    continue
                run_job(n, L, method)

    # Matched-parameter QAOA where an equal-depth energy-and-fidelity win still
    # stands and the matched depth was not already run. One depth, not a new family.
    for n in ns:
        for L in (1, 2, 3, 4):
            A = mem.get((n, L, "family_A"))
            Q = mem.get((n, L, "qaoa"))
            if A is None or Q is None:
                continue
            if not (A["E"] < Q["E"] - 1e-8 and A["F"] > Q["F"] + 1e-6):
                continue
            from .report import matched_qaoa_depth

            Lm = matched_qaoa_depth(L)
            if (n, Lm, "qaoa") in mem:
                continue
            if remaining() < 180:
                skipped.append(f"matched-param QAOA N={n} L={Lm} for A L={L}")
                continue
            log(f"matched-parameter QAOA check N={n} L={Lm} against family A L={L}")
            # Fill intermediate QAOA depths if the bootstrap chain has a hole.
            for ell in range(1, Lm + 1):
                if (n, ell, "qaoa") not in mem and remaining() > 60:
                    run_job(n, ell, "qaoa")

    for L in depths:
        for n in ns:
            A = mem.get((n, L, "family_A"))
            D = mem.get((n, L, "native_D"))
            if A and D and A["E"] > D["E"] + 1e-5:
                msg = (
                    f"N={n} L={L} family A energy {A['E']:.6f} is worse than native D "
                    f"{D['E']:.6f}; the theta=0 basin was missed"
                )
                if msg not in failures:
                    failures.append(msg)

    transfer_rows = []
    if 6 in problems and 12 in problems:
        for L in sorted({L for (n, L, _m) in mem if n == 6}):
            for method in METHOD_ORDER:
                rec = mem.get((6, L, method))
                if rec is None:
                    continue
                transfer_rows.extend(_transfer(problems[6], problems[12], method, rec))

    jobs = []
    for rec in mem.values():
        jobs.append({k: v for k, v in rec.items() if k != "x"})
    comp = comparisons(jobs)
    summary = {
        "contract": CONTRACT,
        "validation": validation,
        "targets": target_public,
        "jobs": jobs,
        "comparisons": comp,
        "transfer": transfer_rows,
        "transfer_note": {
            "primary_scale": PRIMARY_TRANSFER_SCALE,
            "kappa": {str(n): problems[n].kappa_mean for n in problems},
            "rms_J": {str(n): problems[n].rms_J for n in problems},
            "sK_physical": {
                str(n): problems[n].features["physical"].sK for n in problems
            },
            "theta_policy": "unchanged; F_K is RMS-matched separately at each N",
            "shell_policy": "unchanged; cavity-local controls",
            "gamma_policy": "primary multiplies gamma by kappa_source/kappa_target",
            "distinct_from_retrained": True,
        },
        "gamma_ramp_heuristic": gamma_ramp_section(jobs, target_public),
        "failures": failures,
        "skipped": skipped,
        "provenance": {
            "git": _git_hash(),
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "scipy": __import__("scipy").__version__,
            "platform": platform.platform(),
            "wall_seconds": time.perf_counter() - t0,
            "budget_min": budget_min,
            "depths_requested": depths,
            "ns": ns,
        },
    }
    dump_summary(RESULTS / "summary.json", summary)
    write_markdown(RESULTS / "RESULTS.md", summary)
    log(f"wrote {RESULTS / 'summary.json'} in {summary['provenance']['wall_seconds']:.1f}s")
    log(f"failures={len(failures)} skipped={len(skipped)}")
    return summary


def json_target_view(path: Path) -> dict:
    import json

    data = json.loads(path.read_text())
    # Keep the summary lighter than the full fine curve.
    data.pop("fine_curve", None)
    return data


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--budget-min", type=float, default=80.0)
    p.add_argument("--depths", type=str, default="1,2,3,4,5,6")
    p.add_argument("--ns", type=str, default="6,12")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    depths = [int(v) for v in args.depths.split(",") if v]
    ns = [int(v) for v in args.ns.split(",") if v]
    summary = run(args.budget_min, depths, ns, args.force)
    both = [r for r in summary["comparisons"] if r["both_better_than_qaoa"]]
    log(f"cells with both energy and fidelity above equal-depth QAOA: {len(both)}")


if __name__ == "__main__":
    main()
