#!/usr/bin/env python3
"""CLI: sweep fixed U × L* × Hamiltonians with SPSA Gibbs (noiseless)."""

from __future__ import annotations

# Limit BLAS threads so ProcessPool workers do not oversubscribe.
import os as _os
_os.environ.setdefault("OMP_NUM_THREADS", "1")
_os.environ.setdefault("MKL_NUM_THREADS", "1")
_os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
_os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
import json
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from noiseless.circuit_local_ecd import n_parameters
from noiseless.encoding import (
    EncodingSpec,
    list_four_sat_npz,
    load_four_sat_npz,
    logical_energies_from_terms,
)
from noiseless.spsa_gibbs import (
    ADAM_LR,
    DEFAULT_POLISH_RADIUS,
    DEFAULT_RELAYOUT_ROUNDS,
    RELAYOUT_INITS,
    RELAYOUT_RETURNS,
    RELAYOUT_TARGETS,
    NoiselessSimulator,
    GROW_KICK_SIGMA,
    ground_flat_from_bitstring,
    grow_trial,
    optimize_trial,
    relayout_trial,
    scale_spsa_a,
)
from noiseless.layouts import LAYOUTS, layout_spec
from noiseless.unitaries import U_NAMES, build_fixed_u


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _worker(job: dict) -> dict:
    t0 = time.perf_counter()
    try:
        encoding = str(job.get("encoding", "binary"))
        grow = bool(job.get("grow", False))
        relayout = bool(job.get("relayout", False))
        layout = str(job.get("layout", "identity"))
        if layout != "identity":
            if encoding != "binary":
                raise ValueError("--layout other than identity requires --encoding binary")
            if job.get("layout_perm") is not None:
                encoding = EncodingSpec(tuple(int(v) for v in job["layout_perm"]))
            else:
                gs_true = load_four_sat_npz(job["ham_path"])["ground_bitstring"]
                encoding = layout_spec(layout, gs_true, job["ham_file"])
        inst = load_four_sat_npz(job["ham_path"], encoding=encoding)
        u = build_fixed_u(job["u_name"])
        lambda1 = float(job.get("lambda1", 0.0))
        # Soft-cap λ: prefer "lam", fall back to legacy "lambda3"
        if "lam" in job and job["lam"] is not None:
            lam = float(job["lam"])
        else:
            lam = float(job.get("lambda3", 0.0))
        beta_max = job.get("beta_max", None)
        if beta_max is not None:
            beta_max = float(beta_max)
            if not np.isfinite(beta_max):
                beta_max = None
        adapt_lambda = bool(job.get("adapt_lambda", False))
        rng = np.random.default_rng(int(job["seed"]))
        a = job.get("spsa_a")
        grow_kw = dict(
            start_layers=int(job.get("grow_start", 1)),
            kick_sigma=float(job.get("grow_kick_sigma", GROW_KICK_SIGMA)),
            a=None if a is None else float(a),
            c=float(job.get("spsa_c", 0.15)),
            A=float(job.get("spsa_A", 10.0)),
            optimizer=str(job.get("optimizer", "spsa")),
            steps_per_stage=job.get("grow_steps_per_stage"),
            adam_lr=float(job.get("adam_lr", ADAM_LR)),
            steps_schedule=job.get("grow_steps_schedule"),
            lr_schedule=job.get("grow_lr_schedule"),
            eta_scale_schedule=job.get("grow_eta_scale"),
            c_schedule=job.get("grow_c_schedule"),
            bfgs_eta_mode=str(job.get("bfgs_eta_mode", "callback")),
        )
        if relayout:
            result = relayout_trial(
                u,
                logical_energies_from_terms(inst["terms"], inst["identity"]),
                inst["ground_bitstring"],
                final_layers=int(job["n_layers"]),
                rng=rng,
                base_encoding=encoding,
                relayout_rounds=int(job.get("relayout_rounds", DEFAULT_RELAYOUT_ROUNDS)),
                polish_radius=int(job.get("polish_radius", DEFAULT_POLISH_RADIUS)),
                grow=grow,
                total_steps=int(job["steps"]),
                lambda1=lambda1,
                lam=lam,
                beta_max=beta_max,
                relayout_steps=job.get("relayout_steps"),
                relayout_lr=job.get("relayout_lr"),
                relayout_init=str(job.get("relayout_init", "random")),
                relayout_return=str(job.get("relayout_return", "last")),
                relayout_target=str(job.get("relayout_target", "xor_vacuum")),
                **grow_kw,
            )
            a = (
                result.stages[-1]["spsa_a"]
                if result.stages
                else scale_spsa_a(n_parameters(int(job["n_layers"])))
            )
        elif grow:
            # a=None → per-stage scaling by that stage's n_params.
            sim = NoiselessSimulator(
                u_fixed=u,
                energy_tensor=inst["energy_tensor"],
                n_layers=int(job["n_layers"]),
                ground_bitstring=inst["ground_bitstring"],
                ground_flat_index=ground_flat_from_bitstring(inst["ground_bitstring"], encoding),
                lambda1=lambda1,
                lam=lam,
                beta_max=beta_max,
                encoding=encoding,
            )
            result = grow_trial(
                sim,
                final_layers=int(job["n_layers"]),
                total_steps=int(job["steps"]),
                rng=rng,
                **grow_kw,
            )
            a = result.stages[-1]["spsa_a"]
        else:
            sim = NoiselessSimulator(
                u_fixed=u,
                energy_tensor=inst["energy_tensor"],
                n_layers=int(job["n_layers"]),
                ground_bitstring=inst["ground_bitstring"],
                ground_flat_index=ground_flat_from_bitstring(inst["ground_bitstring"], encoding),
                lambda1=lambda1,
                lam=lam,
                beta_max=beta_max,
                encoding=encoding,
            )
            if a is None:
                a = scale_spsa_a(n_parameters(int(job["n_layers"])))
            result = optimize_trial(
                sim,
                maxiter=int(job["steps"]),
                rng=rng,
                a=float(a),
                c=float(job.get("spsa_c", 0.15)),
                A=float(job.get("spsa_A", 10.0)),
                adapt_lambda=adapt_lambda,
                adapt_warmup_frac=float(job.get("adapt_warmup_frac", 0.25)),
                adapt_every=int(job.get("adapt_every", 25)),
                adapt_f_hi=float(job.get("adapt_f_hi", 0.20)),
                adapt_f_lo=float(job.get("adapt_f_lo", 0.05)),
                adapt_lam_min=float(job.get("adapt_lam_min", 0.5)),
                adapt_lam_max=float(job.get("adapt_lam_max", 5.0)),
                optimizer=str(job.get("optimizer", "spsa")),
                adam_lr=float(job.get("adam_lr", ADAM_LR)),
                bfgs_eta_mode=str(job.get("bfgs_eta_mode", "callback")),
            )
        sel_round = None
        if relayout and result.rounds:
            sel_round = next((r for r in result.rounds if r.get("selected")), result.rounds[-1])
        return {
            "ok": True,
            "ham_file": job["ham_file"],
            "u_name": job["u_name"],
            "n_layers": int(job["n_layers"]),
            "trial": int(job["trial"]),
            "seed": int(job["seed"]),
            "success": bool(result.success),
            "p_gs": float(result.p_gs),
            "most_likely_bitstring": result.most_likely_bitstring,
            "ground_bitstring": result.ground_bitstring,
            "fun": float(result.fun),
            "eta": float(result.eta),
            "energy_mean": float(result.energy_mean),
            "mean_abs_beta": float(result.mean_abs_beta),
            "max_abs_beta": float(result.max_abs_beta),
            "frac_over_beta_max": float(result.frac_over_beta_max),
            "lambda1": lambda1,
            "lam": float(result.final_lam) if adapt_lambda else lam,
            "lambda3": float(result.final_lam) if adapt_lambda else lam,  # legacy mirror
            "final_lam": float(result.final_lam),
            "mean_lam_post_warmup": float(result.mean_lam_post_warmup),
            "adapt_lambda": adapt_lambda,
            "beta_max": beta_max,
            "nfev": int(result.nfev),
            "nit": int(result.nit),
            "optimizer": result.optimizer,
            "opt_status": result.opt_status,
            "opt_message": result.opt_message,
            "spsa_a": float(a),
            "adam_lr": float(job.get("adam_lr", ADAM_LR)) if result.optimizer == "spsa_adam" else None,
            "wall_s": float(time.perf_counter() - t0),
            "x": result.x.tolist(),
            "encoding": sel_round["encoding"] if sel_round is not None else str(encoding),
            "encoding_base": str(encoding),
            "layout": layout,
            "grow": grow,
            "relayout": relayout,
            "relayout_stop": result.relayout_stop,
            "relayout_rounds": int(job.get("relayout_rounds", DEFAULT_RELAYOUT_ROUNDS)) if relayout else 0,
            "polish_radius": int(job.get("polish_radius", DEFAULT_POLISH_RADIUS)) if relayout else 0,
            "rounds": result.rounds,
            "polished_candidate": sel_round["polished_candidate"] if sel_round is not None else None,
            "relayout_init": str(job.get("relayout_init", "random")) if relayout else None,
            "relayout_return": str(job.get("relayout_return", "last")) if relayout else None,
            "relayout_target": str(job.get("relayout_target", "xor_vacuum")) if relayout else None,
            "stages": result.stages,
            "bfgs_eta_mode": str(job.get("bfgs_eta_mode", "callback")) if result.optimizer == "bfgs" else None,
            "opt_info": result.opt_info,
            "error": None,
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "ok": False,
            "ham_file": job.get("ham_file"),
            "u_name": job.get("u_name"),
            "n_layers": job.get("n_layers"),
            "trial": job.get("trial"),
            "seed": job.get("seed"),
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "wall_s": float(time.perf_counter() - t0),
        }


def _aggregate(records: list[dict]) -> dict:
    by_key: dict[tuple, list[dict]] = {}
    for r in records:
        if not r.get("ok"):
            continue
        key = (r["u_name"], int(r["n_layers"]), r["ham_file"])
        by_key.setdefault(key, []).append(r)
    cells = []
    for (u_name, n_layers, ham_file), trials in sorted(by_key.items()):
        n = len(trials)
        n_succ = sum(1 for t in trials if t["success"])
        pgs = [float(t["p_gs"]) for t in trials]
        mabs = [float(t.get("mean_abs_beta", float("nan"))) for t in trials]
        maxb = [float(t.get("max_abs_beta", float("nan"))) for t in trials]
        flams = [float(t.get("final_lam", t.get("lam", float("nan")))) for t in trials]
        mlams = [float(t.get("mean_lam_post_warmup", float("nan"))) for t in trials]
        cells.append(
            {
                "u_name": u_name,
                "n_layers": n_layers,
                "ham_file": ham_file,
                "n_trials": n,
                "n_success": n_succ,
                "success_rate": n_succ / n if n else 0.0,
                "mean_p_gs": float(np.mean(pgs)) if pgs else 0.0,
                "max_p_gs": float(np.max(pgs)) if pgs else 0.0,
                "median_p_gs": float(np.median(pgs)) if pgs else 0.0,
                "mean_abs_beta": float(np.nanmean(mabs)) if mabs else 0.0,
                "median_abs_beta": float(np.nanmedian(mabs)) if mabs else 0.0,
                "mean_max_abs_beta": float(np.nanmean(maxb)) if maxb else 0.0,
                "mean_final_lam": float(np.nanmean(flams)) if flams else 0.0,
                "median_final_lam": float(np.nanmedian(flams)) if flams else 0.0,
                "mean_lam_post_warmup": float(np.nanmean(mlams)) if mlams else 0.0,
            }
        )
    by_ul: dict[tuple, list[dict]] = {}
    for c in cells:
        by_ul.setdefault((c["u_name"], c["n_layers"]), []).append(c)
    ranking = []
    for (u_name, n_layers), group in by_ul.items():
        weights = [g["n_trials"] for g in group]
        tot = sum(weights)
        succ = sum(g["n_success"] for g in group)
        mean_p = float(np.average([g["mean_p_gs"] for g in group], weights=weights)) if tot else 0.0
        mean_b = float(np.average([g["mean_abs_beta"] for g in group], weights=weights)) if tot else 0.0
        med_b = float(np.average([g["median_abs_beta"] for g in group], weights=weights)) if tot else 0.0
        mean_max_b = float(np.average([g["mean_max_abs_beta"] for g in group], weights=weights)) if tot else 0.0
        mean_flam = float(np.average([g.get("mean_final_lam", 0.0) for g in group], weights=weights)) if tot else 0.0
        med_flam = float(np.average([g.get("median_final_lam", 0.0) for g in group], weights=weights)) if tot else 0.0
        mean_lpw = float(np.average([g.get("mean_lam_post_warmup", 0.0) for g in group], weights=weights)) if tot else 0.0
        ranking.append(
            {
                "u_name": u_name,
                "n_layers": n_layers,
                "n_hamiltonians": len(group),
                "n_trials": tot,
                "n_success": succ,
                "success_rate": succ / tot if tot else 0.0,
                "mean_p_gs": mean_p,
                "mean_abs_beta": mean_b,
                "median_abs_beta": med_b,
                "mean_max_abs_beta": mean_max_b,
                "mean_final_lam": mean_flam,
                "median_final_lam": med_flam,
                "mean_lam_post_warmup": mean_lpw,
            }
        )
    ranking.sort(key=lambda r: (r["success_rate"], r["mean_p_gs"]), reverse=True)
    return {"cells": cells, "ranking": ranking}


def build_jobs(args: argparse.Namespace) -> list[dict]:
    paths = list_four_sat_npz(Path(args.ham_dir))
    if args.max_h is not None:
        paths = paths[: int(args.max_h)]
    if not paths:
        raise SystemExit(f"No four_sat_*.npz under {args.ham_dir}")
    for p in paths:
        inst = load_four_sat_npz(p, encoding=args.encoding)
        if inst["num_spins"] != 8:
            raise SystemExit(f"{p.name} has num_spins={inst['num_spins']}, need 8")
        if inst["n_ground"] != 1:
            raise SystemExit(f"{p.name} has n_ground={inst['n_ground']}, need unique GS")
    if args.u_names.strip().lower() == "all":
        u_names = list(U_NAMES)
    else:
        u_names = [s.strip() for s in args.u_names.split(",") if s.strip()]
    layers = [int(x) for x in args.layers.split(",") if x.strip()]
    jobs = []
    seed0 = int(args.seed)
    layout_perms: dict[str, list[int] | None] = {}
    for path in paths:
        if args.layout == "identity":
            layout_perms[path.name] = None
        else:
            gs = load_four_sat_npz(path)["ground_bitstring"]
            layout_perms[path.name] = list(layout_spec(args.layout, gs, path.name).perm)
    for hi, path in enumerate(paths):
        for u_name in u_names:
            for L in layers:
                for t in range(int(args.trials)):
                    seed = seed0 + 100_000 * hi + 1_000 * list(U_NAMES).index(u_name) + 10 * L + t
                    jobs.append(
                        {
                            "ham_path": str(path.resolve()),
                            "ham_file": path.name,
                            "u_name": u_name,
                            "n_layers": L,
                            "trial": t,
                            "seed": seed,
                            "steps": int(args.steps),
                            "spsa_a": args.spsa_a,
                            "spsa_c": float(args.spsa_c),
                            "spsa_A": float(args.spsa_A),
                            "lambda1": float(args.lambda1),
                            "lam": float(args.lam),
                            "lambda3": float(args.lam),  # legacy alias in job payload
                            "beta_max": args.beta_max,
                            "adapt_lambda": bool(args.adapt_lambda),
                            "adapt_warmup_frac": float(args.adapt_warmup_frac),
                            "adapt_every": int(args.adapt_every),
                            "adapt_f_hi": float(args.adapt_f_hi),
                            "adapt_f_lo": float(args.adapt_f_lo),
                            "adapt_lam_min": float(args.adapt_lam_min),
                            "adapt_lam_max": float(args.adapt_lam_max),
                            "optimizer": str(args.optimizer),
                            "adam_lr": float(args.adam_lr),
                            "encoding": str(args.encoding),
                            "grow": bool(args.grow),
                            "grow_start": int(args.grow_start),
                            "grow_kick_sigma": float(args.grow_kick_sigma),
                            "grow_steps_per_stage": args.grow_steps_per_stage,
                            "grow_steps_schedule": args.grow_steps_schedule,
                            "grow_lr_schedule": args.grow_lr_schedule,
                            "grow_eta_scale": args.grow_eta_scale,
                            "grow_c_schedule": args.grow_c_schedule,
                            "bfgs_eta_mode": str(args.bfgs_eta_mode),
                            "relayout": bool(args.relayout),
                            "relayout_rounds": int(args.relayout_rounds),
                            "polish_radius": int(args.polish_radius),
                            "relayout_steps": args.relayout_steps,
                            "relayout_lr": args.relayout_lr,
                            "relayout_init": str(args.relayout_init),
                            "relayout_return": str(args.relayout_return),
                            "relayout_target": str(args.relayout_target),
                            "layout": str(args.layout),
                            "layout_perm": layout_perms[path.name],
                        }
                    )
    return jobs


TUNED_GROW_LR_SCHEDULE = "0.5,0.2,0.05,0.02"
TUNED_STEPS_PER_STAGE = 200
PRESETS = {
    # Tuned growth + SPSA-Adam (GROW_ADAM_TUNING_SUMMARY.md, arm lr_sched): 1604 evals/trial.
    "tuned": {"u_names": "jp", "layers": "4", "optimizer": "spsa_adam", "grow": True},
    # Defaults before 2026-09-30 (reproduces every older run_u_sweep command bit-for-bit).
    "legacy": {"u_names": "all", "layers": "2,3,4", "optimizer": "spsa", "grow": False},
}


def _apply_preset(args: argparse.Namespace, p: argparse.ArgumentParser) -> None:
    """Fill flags left at None from the preset (tuned; legacy under --smoke). Explicit flags win.

    Tuned-only extras, applied only when the user did not set them (or a conflicting flag):
      - growth budget: 200 SPSA steps per stage unless --steps / --grow-steps-per-stage /
        --grow-steps-schedule is given;
      - Adam lr schedule 0.5,0.2,0.05,0.02 when growth + spsa_adam, every requested depth has
        exactly 4 stages, and neither --grow-lr-schedule nor --adam-lr is given.
    """
    if args.preset is None:
        args.preset = "legacy" if args.smoke else "tuned"
    steps_given = args.steps is not None
    adam_lr_given = args.adam_lr is not None
    for k, v in PRESETS[args.preset].items():
        if getattr(args, k) is None:
            setattr(args, k, v)
    if args.steps is None:
        args.steps = 200
    if args.adam_lr is None:
        args.adam_lr = ADAM_LR
    if args.preset != "tuned" or not args.grow:
        return
    if (not steps_given and args.grow_steps_per_stage is None
            and args.grow_steps_schedule is None):
        args.grow_steps_per_stage = TUNED_STEPS_PER_STAGE
    layers = [int(x) for x in args.layers.split(",") if x.strip()]
    four_stages = all(L - int(args.grow_start) + 1 == 4 for L in layers)
    if (args.optimizer == "spsa_adam" and args.grow_lr_schedule is None and not adam_lr_given
            and four_stages):
        args.grow_lr_schedule = TUNED_GROW_LR_SCHEDULE


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ham-dir", type=str, default=str(_REPO / "Hamiltonians" / "four_sat"))
    p.add_argument(
        "--preset",
        choices=tuple(PRESETS),
        default=None,
        help="Default bundle for flags you do not pass. tuned (default; legacy under --smoke): "
        "jp, --layers 4, spsa_adam, --grow L1->4, --grow-lr-schedule 0.5,0.2,0.05,0.02, "
        "200 steps/stage (1604 evals/trial). legacy: the pre-2026-09-30 defaults "
        "(--u-names all --layers 2,3,4 --optimizer spsa, no growth, --steps 200). "
        "Explicit flags always win.",
    )
    p.add_argument("--u-names", type=str, default=None, help="default: jp (tuned) / all (legacy)")
    p.add_argument("--layers", type=str, default=None, help="default: 4 (tuned) / 2,3,4 (legacy)")
    p.add_argument("--trials", type=int, default=5)
    p.add_argument(
        "--steps",
        type=int,
        default=None,
        help="SPSA steps (default 200). With --grow: TOTAL budget split evenly over stages; the "
        "tuned preset instead uses 200 steps per stage unless --steps or a --grow-steps-* is given",
    )
    p.add_argument("--workers", type=int, default=max(1, min(4, os.cpu_count() or 1)))
    p.add_argument("--seed", type=int, default=20260917)
    p.add_argument("--max-h", type=int, default=None)
    p.add_argument("--spsa-a", type=float, default=None)
    p.add_argument("--spsa-c", type=float, default=0.15)
    p.add_argument("--spsa-A", type=float, default=10.0)
    p.add_argument("--outdir", type=str, default=str(_REPO / "noiseless" / "results"))
    p.add_argument("--tag", type=str, default="run")
    p.add_argument("--smoke", action="store_true")
    p.add_argument(
        "--lambda1",
        type=float,
        default=0.0,
        help="L1 weight on sum_i |β_i| (default 0 = Gibbs-only)",
    )
    p.add_argument(
        "--lambda",
        dest="lam_flag",
        type=float,
        default=None,
        help="Soft-cap weight λ on sum_i max(|β_i|-β_max,0)^2 (default 0)",
    )
    p.add_argument(
        "--lambda3",
        type=float,
        default=None,
        help="Alias for --lambda (legacy soft-cap weight)",
    )
    p.add_argument(
        "--beta-max",
        type=float,
        default=None,
        help="Soft |β| cap; omit / None / inf = no cap term",
    )
    p.add_argument(
        "--adapt-lambda",
        action="store_true",
        help="Opt-in adaptive soft-cap λ (warm-up then raise/lower from frac over β_max)",
    )
    p.add_argument("--adapt-warmup-frac", type=float, default=0.25)
    p.add_argument("--adapt-every", type=int, default=25)
    p.add_argument("--adapt-f-hi", type=float, default=0.20)
    p.add_argument("--adapt-f-lo", type=float, default=0.05)
    p.add_argument("--adapt-lam-min", type=float, default=0.5)
    p.add_argument("--adapt-lam-max", type=float, default=5.0)
    p.add_argument(
        "--optimizer",
        choices=("spsa", "bfgs", "spsa_adam"),
        default=None,
        help="spsa_adam (tuned default) / spsa (legacy default), spsa_adam (same SPSA gradient estimate + Adam update, "
        "same eval count) or bfgs (scipy BFGS, finite-difference gradient; same x0)",
    )
    p.add_argument(
        "--adam-lr",
        type=float,
        default=None,
        help="Adam learning rate for --optimizer spsa_adam (default 0.05)",
    )
    p.add_argument(
        "--encoding",
        choices=("binary", "gray"),
        default="binary",
        help="Fock→logical bit map for the two cavities (default binary = legacy results)",
    )
    p.add_argument(
        "--bfgs-eta-mode",
        choices=("callback", "fixed", "restart"),
        default="callback",
        help="With --optimizer bfgs: η handling. callback (legacy default) = refresh every 5 "
        "BFGS iterations inside the run; fixed = refresh once at x0 then hold; restart = BFGS "
        "to convergence at fixed η, refresh, restart (inverse Hessian carried) until η is "
        "stationary (1%% rel.), a restart makes no iteration, or the iteration budget is used. With --grow the per-stage "
        "BFGS maxiter is the stage step count (e.g. --grow-steps-per-stage 500)",
    )
    p.add_argument(
        "--grow",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="(tuned default: on; legacy: off; --no-grow disables) Layer growth: train L=--grow-start, append transparent last layer + kick, "
        "retrain, ... up to each --layers value; --steps is the TOTAL budget split evenly",
    )
    p.add_argument("--grow-start", type=int, default=1)
    p.add_argument("--grow-kick-sigma", type=float, default=GROW_KICK_SIGMA)
    p.add_argument(
        "--grow-steps-per-stage",
        type=int,
        default=None,
        help="With --grow: SPSA steps for EVERY stage (generous budget, total = S x n_stages). "
        "Omit for equal-total budget (--steps split evenly over stages).",
    )
    p.add_argument(
        "--grow-steps-schedule",
        type=str,
        default=None,
        help='With --grow: comma list of SPSA steps per stage, e.g. "100,150,250,300" '
        "(one entry per stage; overrides --steps / --grow-steps-per-stage)",
    )
    p.add_argument(
        "--grow-lr-schedule",
        type=str,
        default=None,
        help='With --grow + spsa_adam: Adam lr per stage, e.g. "0.1,0.1,0.1,0.03"',
    )
    p.add_argument(
        "--grow-eta-scale",
        type=str,
        default=None,
        help="With --grow: per-stage multiplier on the sampled-tail η (η = inverse "
        'temperature: <1 hotter, >1 colder), e.g. "0.5,0.7,1,1"',
    )
    p.add_argument(
        "--grow-c-schedule",
        type=str,
        default=None,
        help='With --grow: SPSA perturbation c per stage, e.g. "0.15,0.15,0.15,0.05"',
    )
    p.add_argument(
        "--relayout",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="After the usual L=1→L* growth on the baseline layout, polish the most-likely "
        "bitstring (Hamming-1 energy lookup), re-encode it to both-cavity Fock 0, and retrain "
        "at L* from a fresh random init. Repeat --relayout-rounds times or until the encoding "
        "stops changing. Off by default so tuned/legacy fleets stay bit-for-bit; this is the "
        "recommended experiment (see noiseless/README.md).",
    )
    p.add_argument(
        "--relayout-rounds",
        type=int,
        default=DEFAULT_RELAYOUT_ROUNDS,
        help="Extra L* retrains after round-0 growth (default 2). 0 = grow only, still records "
        "the polished candidate / would-be next encoding",
    )
    p.add_argument(
        "--polish-radius",
        type=int,
        default=DEFAULT_POLISH_RADIUS,
        help="Hamming radius of the classical polish of the most-likely bitstring (default 1; "
        "0 = use the raw most-likely string). Free; no extra circuit runs",
    )
    p.add_argument(
        "--relayout-steps",
        type=int,
        default=None,
        help="SPSA steps for each extra L* round (default: one growth stage, e.g. 200 under "
        "the tuned preset)",
    )
    p.add_argument(
        "--relayout-lr",
        type=float,
        default=None,
        help="Adam lr for extra L* rounds (default: first-stage / random-init lr, e.g. 0.5 "
        "under the tuned schedule — not the last-stage fine-tune lr)",
    )
    p.add_argument("--relayout-init", choices=RELAYOUT_INITS, default="random",
                   help="x0 of extra relayout rounds: random (|β|~U(0,3), default), small "
                   "(|β|~U(0,0.1)), warm (previous round's x), grow (re-run L=1→L* growth in the "
                   "new layout, --relayout-steps per stage, round-0 lr schedule)")
    p.add_argument("--relayout-return", choices=RELAYOUT_RETURNS, default="last",
                   help="official trial result: last round (default) or best = lowest Gibbs "
                   "cost at a common η over rounds (no GS knowledge)")
    p.add_argument("--relayout-target", choices=RELAYOUT_TARGETS, default="xor_vacuum",
                   help="xor_vacuum (default): XOR cavity maps so the candidate sits at Fock "
                   "(0,0); rule: permutation-only tier-rule layout for the candidate")
    p.add_argument("--layout", choices=LAYOUTS, default="identity",
                   help="variable→slot layout (binary code). rule_best / rule_bad / screen_best "
                   "use the TRUE GS or the screen data (oracles); see noiseless/layouts.py")
    args = p.parse_args(argv)
    if args.smoke:
        args.u_names = "identity"
        args.layers = "2"
        args.trials = 1
        args.steps = 8
        args.max_h = 1
        if args.tag == "run":
            args.tag = "smoke"
    _apply_preset(args, p)
    for name, cast in (("grow_steps_schedule", int), ("grow_lr_schedule", float),
                       ("grow_eta_scale", float), ("grow_c_schedule", float)):
        raw = getattr(args, name)
        if raw is None:
            continue
        if not args.grow:
            p.error(f"--{name.replace('_', '-')} requires --grow")
        vals = [cast(v) for v in raw.split(",") if v.strip()]
        layers = [int(x) for x in args.layers.split(",") if x.strip()]
        for L in layers:
            n_st = L - int(args.grow_start) + 1
            if len(vals) != n_st:
                p.error(f"--{name.replace('_', '-')} has {len(vals)} entries; L={L} from "
                        f"--grow-start {args.grow_start} has {n_st} stages")
        setattr(args, name, vals)
    if args.grow and args.adapt_lambda:
        p.error("--adapt-lambda is not supported with --grow")
    if args.relayout and args.adapt_lambda:
        p.error("--adapt-lambda is not supported with --relayout")
    if args.relayout_rounds < 0:
        p.error("--relayout-rounds must be >= 0")
    if args.polish_radius < 0:
        p.error("--polish-radius must be >= 0")
    if (args.relayout_steps is not None or args.relayout_lr is not None) and not args.relayout:
        p.error("--relayout-steps / --relayout-lr require --relayout")
    if not args.relayout and (args.relayout_init != "random" or args.relayout_return != "last"
                              or args.relayout_target != "xor_vacuum"):
        p.error("--relayout-init / --relayout-return / --relayout-target require --relayout")
    if args.layout != "identity" and args.encoding != "binary":
        p.error("--layout other than identity requires --encoding binary")
    if args.bfgs_eta_mode != "callback" and args.optimizer != "bfgs":
        p.error("--bfgs-eta-mode requires --optimizer bfgs")
    if args.optimizer == "bfgs" and args.adapt_lambda:
        p.error("--adapt-lambda is not supported with --optimizer bfgs")
    # Resolve soft-cap λ: --lambda wins over --lambda3; default 0
    if args.lam_flag is not None:
        args.lam = float(args.lam_flag)
    elif args.lambda3 is not None:
        args.lam = float(args.lambda3)
    else:
        args.lam = 0.0
    args.lambda3 = float(args.lam)  # keep attribute for logging / old code

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    jobs = build_jobs(args)
    print(
        f"[{_now()}] starting {len(jobs)} jobs preset={args.preset} workers={args.workers} steps={args.steps} "
        f"tag={args.tag} lambda1={args.lambda1} lam={args.lam} beta_max={args.beta_max} "
        f"adapt_lambda={args.adapt_lambda} optimizer={args.optimizer} adam_lr={args.adam_lr} "
        f"encoding={args.encoding} grow={args.grow} "
        f"grow_steps_per_stage={args.grow_steps_per_stage} "
        f"relayout={args.relayout} relayout_rounds={args.relayout_rounds}",
        flush=True,
    )
    records: list[dict] = []
    if args.workers <= 1 or len(jobs) == 1:
        for i, job in enumerate(jobs):
            rec = _worker(job)
            records.append(rec)
            print(
                f"  [{i+1}/{len(jobs)}] {'OK' if rec.get('ok') else 'FAIL'} "
                f"{rec.get('u_name')} L={rec.get('n_layers')} {rec.get('ham_file')} "
                f"succ={rec.get('success')} p_gs={rec.get('p_gs')} wall={rec.get('wall_s'):.2f}s",
                flush=True,
            )
    else:
        with ProcessPoolExecutor(max_workers=int(args.workers)) as ex:
            futs = {ex.submit(_worker, job): job for job in jobs}
            done = 0
            for fut in as_completed(futs):
                rec = fut.result()
                records.append(rec)
                done += 1
                print(
                    f"  [{done}/{len(jobs)}] {'OK' if rec.get('ok') else 'FAIL'} "
                    f"{rec.get('u_name')} L={rec.get('n_layers')} {rec.get('ham_file')} "
                    f"succ={rec.get('success')} p_gs={rec.get('p_gs')} wall={rec.get('wall_s'):.2f}s",
                    flush=True,
                )

    agg = _aggregate(records)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = outdir / f"{args.tag}_{stamp}.json"
    summary_path = outdir / f"{args.tag}_{stamp}_summary.json"
    payload = {
        "created_utc": _now(),
        "tag": args.tag,
        "args": {
            "ham_dir": str(args.ham_dir),
            "u_names": args.u_names,
            "layers": args.layers,
            "trials": args.trials,
            "steps": args.steps,
            "workers": args.workers,
            "seed": args.seed,
            "max_h": args.max_h,
            "lambda1": float(args.lambda1),
            "lam": float(args.lam),
            "lambda3": float(args.lam),
            "beta_max": args.beta_max,
            "adapt_lambda": bool(args.adapt_lambda),
            "adapt_warmup_frac": float(args.adapt_warmup_frac),
            "adapt_every": int(args.adapt_every),
            "adapt_f_hi": float(args.adapt_f_hi),
            "adapt_f_lo": float(args.adapt_f_lo),
            "adapt_lam_min": float(args.adapt_lam_min),
            "adapt_lam_max": float(args.adapt_lam_max),
            "optimizer": str(args.optimizer),
            "adam_lr": float(args.adam_lr),
            "encoding": str(args.encoding),
            "grow": bool(args.grow),
            "grow_start": int(args.grow_start),
            "grow_kick_sigma": float(args.grow_kick_sigma),
            "grow_steps_per_stage": args.grow_steps_per_stage,
            "grow_steps_schedule": args.grow_steps_schedule,
            "grow_lr_schedule": args.grow_lr_schedule,
            "grow_eta_scale": args.grow_eta_scale,
            "grow_c_schedule": args.grow_c_schedule,
            "bfgs_eta_mode": str(args.bfgs_eta_mode),
            "relayout": bool(args.relayout),
            "relayout_rounds": int(args.relayout_rounds),
            "polish_radius": int(args.polish_radius),
            "relayout_steps": args.relayout_steps,
            "relayout_lr": args.relayout_lr,
            "relayout_init": str(args.relayout_init),
            "relayout_return": str(args.relayout_return),
            "relayout_target": str(args.relayout_target),
            "layout": str(args.layout),
            "preset": args.preset,
        },
        "n_jobs": len(records),
        "n_ok": sum(1 for r in records if r.get("ok")),
        "n_fail": sum(1 for r in records if not r.get("ok")),
        "aggregate": agg,
        "records": records,
    }
    out_path.write_text(json.dumps(payload, indent=2))
    summary_path.write_text(
        json.dumps(
            {
                "created_utc": payload["created_utc"],
                "tag": args.tag,
                "optimizer": str(args.optimizer),
                "encoding": str(args.encoding),
                "grow": bool(args.grow),
                "relayout": bool(args.relayout),
                "args": payload["args"],
                "n_jobs": payload["n_jobs"],
                "n_ok": payload["n_ok"],
                "n_fail": payload["n_fail"],
                "ranking": agg["ranking"],
                "cells": agg["cells"],
            },
            indent=2,
        )
    )
    print(f"[{_now()}] wrote {out_path}", flush=True)
    print(f"[{_now()}] summary {summary_path}", flush=True)
    if agg["ranking"]:
        best = agg["ranking"][0]
        print(
            f"BEST: U={best['u_name']} L*={best['n_layers']} "
            f"success_rate={best['success_rate']:.3f} mean_p_gs={best['mean_p_gs']:.4f}",
            flush=True,
        )
    return 0 if payload["n_fail"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
