#!/usr/bin/env python3
"""Controlled test of whether the entangling bus gate helps the local-ECD ansatz.

Replaces the retired, uncontrolled jp-vs-identity comparison (JP_GATE_SUMMARY.md and
the jp section of RANKING.md, removed 2026-10-08): there the arms used different random
initialisations (per-trial seeds depended on the gate's U_NAMES index) and mixed budgets.

Design
------
Arms differ ONLY in the frozen bus unitary U on A⊗B; everything else (Hamiltonians,
layer count, optimizer, budget, η controller) is identical, and the per-(H, L, trial)
seed is arm-independent, so x0 AND the SPSA perturbation stream are paired across arms.

  identity    U = I. The circuit factorizes exactly across the (d,A)|(e,B) cut
              (zero cross-pair entanglement by construction).
  jp_local    Best product-gate approximation of jp with the same local phase profile:
              L_A ⊗ L_B with L = diag((-i)^(n mod 2)). Zero entangling power; isolates
              "jp's phases" from "jp's entanglement".
  g<x>        U(γ) = exp(+i γ Π_A Π_B) with γ = x·π, Π = (-1)^n̂ (joint-parity ZZ(2γ)
              dose–response; entangling power increases monotonically on γ ∈ [0, π/4]).
              g0 ≡ identity; g0.25 ≡ jp up to the global phase e^{-iπ/4} (identical
              probabilities, cost and optimizer trajectory). x ∈ [0, 0.5]:
              g0.5 = exp(iπ/2 Π_AΠ_B) = i·Π_A⊗Π_B is a PRODUCT gate (second product
              control: local parity flips, same "angle family" as jp, zero entangling power).
  jp          jp|n,m⟩ = (-i)^{(n+m) mod 2}|n,m⟩ (same as unitaries.joint_parity_ab).

Each record also carries the entanglement-entropy profile of the optimized circuit:
the von Neumann entropy (bits) across the (d,A)|(e,B) cut after every layer, the entropy
just BEFORE the final bus gate (the final U is diagonal in the measured basis, so it cannot
change p(x) or the cost; local gates do not change the cut entropy, so this equals the
entropy after layer L-1), and the "effective peak" = max entropy over all states that can
still influence the output (all layers except the final U).

Protocols (--protocol): ``legacy`` = fixed L, plain SPSA (or --optimizer), --steps steps,
random init (401 evals at 200 steps); ``tuned`` = the run_u_sweep tuned default: growth
L=1→L (warm start, transparent new layer + kick σ=0.05), SPSA-Adam, lr 0.5/0.2/0.05/0.02,
200 steps per stage (1604 evals at L=4); per-stage p(GS) and entropies are logged.

Layouts (--layouts, comma list): identity (legacy class 0), rule_best, rule_bad (tier rule
on the TRUE GS; see noiseless/layouts.py). Seeds are shared across arms AND layouts AND
protocols, so every comparison is paired at the (H, L, trial) level. The
summary adds, per arm: success, p(GS) stats, entropy stats, the within-arm Pearson
correlation of final-layer entropy with p(GS), and paired per-trial deltas against the
reference arm (default: identity).

Example (full fleet, later):
  python -m noiseless.run_entanglement_control --trials 25 --workers 8 \
    --arms identity,jp_local,g0.0625,g0.125,g0.1875,jp --layers 4 --steps 200 \
    --tag ent_control_A

Smoke:
  python -m noiseless.run_entanglement_control --smoke
"""

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

from noiseless.circuit_local_ecd import (
    apply_layer_factored,
    extract_u_ab,
    n_parameters,
    unpack_params,
    vacuum_np,
)
from noiseless.encoding import EncodingSpec, NFOCK, list_four_sat_npz, load_four_sat_npz
from noiseless.layouts import layout_spec
from noiseless.spsa_gibbs import (
    ADAM_LR,
    GROW_KICK_SIGMA,
    NoiselessSimulator,
    ground_flat_from_bitstring,
    grow_trial,
    optimize_trial,
    scale_spsa_a,
)

DEFAULT_ARMS = "identity,jp_local,g0.5,g0.0625,g0.125,g0.1875,jp"
PRODUCT_ARMS = ("identity", "jp_local", "g0.5")
PROTOCOLS = ("legacy", "tuned")
ENT_LAYOUTS = ("identity", "rule_best", "rule_bad")
TUNED_LR_SCHEDULE = (0.5, 0.2, 0.05, 0.02)
TUNED_STEPS_PER_STAGE = 200
NFOCK_SQ = NFOCK * NFOCK


# ----------------------------------------------------------------------------- gates


def _embed_ab_np(u_ab: np.ndarray) -> np.ndarray:
    """I_d ⊗ I_e ⊗ U_AB as a dense 256×256 ndarray (order d, e, A, B)."""
    return np.kron(np.eye(4, dtype=complex), np.asarray(u_ab, dtype=complex))


def gamma_gate_ab(gamma: float) -> np.ndarray:
    """U(γ) = exp(+i γ Π_A Π_B) on A⊗B: e^{+iγ} on even n+m, e^{-iγ} on odd."""
    diag = np.empty(NFOCK_SQ, dtype=complex)
    for n in range(NFOCK):
        for m in range(NFOCK):
            sign = 1.0 if (n + m) % 2 == 0 else -1.0
            diag[n * NFOCK + m] = np.exp(1j * sign * float(gamma))
    return np.diag(diag)


def jp_gate_ab() -> np.ndarray:
    """jp|n,m⟩ = (-i)^{(n+m) mod 2}|n,m⟩ (= e^{-iπ/4}·U(γ=π/4))."""
    diag = np.ones(NFOCK_SQ, dtype=complex)
    for n in range(NFOCK):
        for m in range(NFOCK):
            if (n + m) % 2 == 1:
                diag[n * NFOCK + m] = -1j
    return np.diag(diag)


def jp_local_gate_ab() -> np.ndarray:
    """L_A ⊗ L_B with L = diag((-i)^(n mod 2)): jp's local phase profile, zero entangling power."""
    local = np.diag(np.array([(-1j) ** (n % 2) for n in range(NFOCK)], dtype=complex))
    return np.kron(local, local)


def parse_arm(token: str) -> tuple[str, float | None]:
    """Return (canonical arm name, γ/π or None)."""
    t = token.strip().lower()
    if t == "identity":
        return "identity", 0.0
    if t == "jp":
        return "jp", 0.25
    if t == "jp_local":
        return "jp_local", None
    if t.startswith("g"):
        frac = float(t[1:])
        if not 0.0 <= frac <= 0.5:
            raise ValueError(f"γ/π = {frac} outside [0, 0.5] in arm {token!r}")
        return f"g{frac:g}", frac
    raise ValueError(f"Unknown arm {token!r}; use identity, jp, jp_local, or g<γ/π>")


def build_arm_u(arm: str) -> np.ndarray:
    """256×256 unitary for an arm token."""
    name, frac = parse_arm(arm)
    if name == "identity":
        return _embed_ab_np(np.eye(NFOCK_SQ, dtype=complex))
    if name == "jp":
        return _embed_ab_np(jp_gate_ab())
    if name == "jp_local":
        return _embed_ab_np(jp_local_gate_ab())
    return _embed_ab_np(gamma_gate_ab(frac * np.pi))


# ----------------------------------------------------------------- entanglement


def entropy_dA_eB(psi_flat: np.ndarray) -> float:
    """Von Neumann entanglement entropy (bits) across the (d,A)|(e,B) cut of a pure state."""
    psi = np.asarray(psi_flat, dtype=complex).reshape(2, 2, NFOCK, NFOCK)
    mat = np.transpose(psi, (0, 2, 1, 3)).reshape(2 * NFOCK, 2 * NFOCK)
    s = np.linalg.svd(mat, compute_uv=False)
    p = s * s
    tot = float(p.sum())
    if tot <= 0.0:
        return 0.0
    p = p / tot
    p = p[p > 1e-15]
    return float(-np.sum(p * np.log2(p)))


def entropy_profile(x: np.ndarray, n_layers: int, u_ab: np.ndarray) -> list[float]:
    """Entropy across (d,A)|(e,B) after each layer of the circuit defined by x."""
    return entropy_profiles(x, n_layers, u_ab)[0]


def entropy_profiles(x: np.ndarray, n_layers: int, u_ab: np.ndarray) -> tuple[list[float], list[float]]:
    """(post-U, pre-U) entropy across (d,A)|(e,B) for each layer.

    pre-U[k] is the entropy after layer k's local gates (R, ECD) but before its bus gate;
    local gates cannot change the cut entropy, so pre-U[k] == post-U[k-1] (0 for k=0).
    """
    ket = vacuum_np()
    eye = np.eye(NFOCK_SQ, dtype=complex)
    post, pre = [], []
    for layer in unpack_params(np.asarray(x, dtype=float), n_layers):
        loc = apply_layer_factored(
            ket,
            layer["beta_d"],
            layer["beta_e"],
            layer["theta_d"],
            layer["theta_e"],
            layer["phi_d"],
            layer["phi_e"],
            eye,
        )
        pre.append(entropy_dA_eB(loc))
        ket = np.einsum("ij,dej->dei", u_ab, loc.reshape(2, 2, NFOCK_SQ)).reshape(-1)
        post.append(entropy_dA_eB(ket))
    return post, pre


def entropy_summary(x: np.ndarray, n_layers: int, u_ab: np.ndarray) -> dict:
    """Entropy fields for a record (bits, clipped at 0 to drop -0.0 noise)."""
    post, pre = entropy_profiles(x, n_layers, u_ab)
    post = [max(0.0, float(v)) for v in post]
    pre = [max(0.0, float(v)) for v in pre]
    eff = pre + post[:-1]  # every state that can still influence the measured distribution
    return {
        "entropy_profile": post,
        "entropy_pre_u_profile": pre,
        "final_entropy": post[-1] if post else 0.0,
        "max_entropy": max(post) if post else 0.0,
        "final_entropy_pre_u": pre[-1] if pre else 0.0,
        "peak_entropy_effective": max(eff) if eff else 0.0,
    }


# ------------------------------------------------------------------------ worker


def trial_seed(seed0: int, ham_index: int, n_layers: int, trial: int) -> int:
    """Arm-INDEPENDENT seed: pairs x0 and the SPSA perturbation stream across arms."""
    return int(seed0) + 1_000_000 * int(ham_index) + 1_000 * int(n_layers) + int(trial)


def _worker(job: dict) -> dict:
    t0 = time.perf_counter()
    try:
        encoding = str(job.get("encoding", "binary"))
        layout = str(job.get("layout", "identity"))
        if job.get("layout_perm") is not None:
            encoding = EncodingSpec(tuple(int(v) for v in job["layout_perm"]))
        inst = load_four_sat_npz(job["ham_path"], encoding=encoding)
        u_full = build_arm_u(job["arm"])
        sim = NoiselessSimulator(
            u_fixed=u_full,
            energy_tensor=inst["energy_tensor"],
            n_layers=int(job["n_layers"]),
            ground_bitstring=inst["ground_bitstring"],
            ground_flat_index=ground_flat_from_bitstring(inst["ground_bitstring"], encoding),
            encoding=encoding,
        )
        rng = np.random.default_rng(int(job["seed"]))
        protocol = str(job.get("protocol", "legacy"))
        a = job.get("spsa_a")
        u_ab = extract_u_ab(u_full)
        stage_recs = None
        if protocol == "tuned":
            n_st = int(job["n_layers"])
            lr_sched = list(TUNED_LR_SCHEDULE) if n_st == 4 else None
            result = grow_trial(
                sim,
                final_layers=int(job["n_layers"]),
                rng=rng,
                start_layers=1,
                kick_sigma=float(job.get("grow_kick_sigma", GROW_KICK_SIGMA)),
                a=None if a is None else float(a),
                c=float(job.get("spsa_c", 0.15)),
                A=float(job.get("spsa_A", 10.0)),
                optimizer="spsa_adam",
                steps_per_stage=int(job.get("steps_per_stage", TUNED_STEPS_PER_STAGE)),
                adam_lr=float(job.get("adam_lr", ADAM_LR)),
                lr_schedule=lr_sched,
                record_x=True,
            )
            stage_recs = []
            for st in result.stages:
                xs = np.asarray(st.pop("x"), dtype=float)
                stage_recs.append({**st, **entropy_summary(xs, int(st["n_layers"]), u_ab)})
            a = None
        else:
            if a is None:
                a = scale_spsa_a(n_parameters(int(job["n_layers"])))
            result = optimize_trial(
                sim,
                maxiter=int(job["steps"]),
                rng=rng,
                a=float(a),
                c=float(job.get("spsa_c", 0.15)),
                A=float(job.get("spsa_A", 10.0)),
                optimizer=str(job.get("optimizer", "spsa")),
                adam_lr=float(job.get("adam_lr", ADAM_LR)),
            )
        ent = entropy_summary(result.x, int(job["n_layers"]), u_ab)
        name, gamma_over_pi = parse_arm(job["arm"])
        return {
            "ok": True,
            "arm": name,
            "gamma_over_pi": gamma_over_pi,
            "ham_file": job["ham_file"],
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
            "nfev": int(result.nfev),
            "optimizer": result.optimizer,
            "protocol": protocol,
            "spsa_a": None if a is None else float(a),
            **ent,
            "stages": stage_recs,
            "layout": layout,
            "layout_perm": job.get("layout_perm"),
            "encoding": str(encoding),
            "x": result.x.tolist(),
            "wall_s": float(time.perf_counter() - t0),
            "error": None,
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "ok": False,
            "arm": job.get("arm"),
            "ham_file": job.get("ham_file"),
            "n_layers": job.get("n_layers"),
            "trial": job.get("trial"),
            "seed": job.get("seed"),
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "wall_s": float(time.perf_counter() - t0),
        }


# --------------------------------------------------------------------- aggregate


def _pearson(a: list[float], b: list[float]) -> float | None:
    if len(a) < 3 or np.std(a) == 0.0 or np.std(b) == 0.0:
        return None
    return float(np.corrcoef(a, b)[0, 1])


def aggregate(records: list[dict], reference_arm: str) -> dict:
    ok = [r for r in records if r.get("ok")]
    by_arm: dict[tuple, list[dict]] = {}
    for r in ok:
        by_arm.setdefault((r.get("layout", "identity"), r["arm"], int(r["n_layers"])), []).append(r)

    arms = []
    for (layout, arm, L), recs in sorted(by_arm.items()):
        pgs = [float(r["p_gs"]) for r in recs]
        fent = [float(r["final_entropy"]) for r in recs]
        ment = [float(r["max_entropy"]) for r in recs]
        pent = [float(r.get("final_entropy_pre_u", 0.0)) for r in recs]
        peff = [float(r.get("peak_entropy_effective", 0.0)) for r in recs]
        gamma = recs[0].get("gamma_over_pi")
        arms.append(
            {
                "layout": layout,
                "arm": arm,
                "gamma_over_pi": gamma,
                "n_layers": L,
                "n_trials": len(recs),
                "n_success": sum(1 for r in recs if r["success"]),
                "success_rate": float(np.mean([r["success"] for r in recs])),
                "mean_p_gs": float(np.mean(pgs)),
                "median_p_gs": float(np.median(pgs)),
                "mean_final_entropy": float(np.mean(fent)),
                "mean_max_entropy": float(np.mean(ment)),
                "mean_final_entropy_pre_u": float(np.mean(pent)),
                "mean_peak_entropy_effective": float(np.mean(peff)),
                "corr_final_entropy_p_gs": _pearson(fent, pgs),
                "corr_max_entropy_p_gs": _pearson(ment, pgs),
                "corr_peak_entropy_effective_p_gs": _pearson(peff, pgs),
            }
        )
    arms.sort(key=lambda r: (r["layout"], r["n_layers"], -(r["success_rate"]), -(r["mean_p_gs"])))

    # Paired per-(layout, H, L, trial) deltas vs the reference arm.
    ref = {(r.get("layout", "identity"), r["ham_file"], int(r["n_layers"]), int(r["trial"])): r
           for r in ok if r["arm"] == reference_arm}
    paired = []
    for (layout, arm, L), recs in sorted(by_arm.items()):
        if arm == reference_arm or not ref:
            continue
        d_pgs, d_succ, win, tie, loss = [], [], 0, 0, 0
        for r in recs:
            key = (layout, r["ham_file"], int(r["n_layers"]), int(r["trial"]))
            if key not in ref:
                continue
            rr = ref[key]
            dp = float(r["p_gs"]) - float(rr["p_gs"])
            d_pgs.append(dp)
            d_succ.append(float(r["success"]) - float(rr["success"]))
            win += dp > 1e-12
            loss += dp < -1e-12
            tie += abs(dp) <= 1e-12
        if not d_pgs:
            continue
        paired.append(
            {
                "layout": layout,
                "arm": arm,
                "n_layers": L,
                "reference": reference_arm,
                "n_pairs": len(d_pgs),
                "mean_delta_p_gs": float(np.mean(d_pgs)),
                "median_delta_p_gs": float(np.median(d_pgs)),
                "mean_delta_success": float(np.mean(d_succ)),
                "p_gs_win_tie_loss": [int(win), int(tie), int(loss)],
            }
        )
    return {"arms": arms, "paired_vs_reference": paired}


# -------------------------------------------------------------------------- CLI


def build_jobs(args: argparse.Namespace) -> list[dict]:
    paths = list_four_sat_npz(Path(args.ham_dir))
    if args.max_h is not None:
        paths = paths[: int(args.max_h)]
    if not paths:
        raise SystemExit(f"No four_sat_*.npz under {args.ham_dir}")
    arms = [s.strip() for s in args.arms.split(",") if s.strip()]
    for arm in arms:
        parse_arm(arm)  # validate early
    layers = [int(x) for x in args.layers.split(",") if x.strip()]
    layouts = [s.strip() for s in str(getattr(args, "layouts", "identity")).split(",") if s.strip()]
    protocol = str(getattr(args, "protocol", "legacy"))
    for lay in layouts:
        if lay not in ENT_LAYOUTS:
            raise SystemExit(f"unknown layout {lay!r}; choose from {ENT_LAYOUTS}")
        if lay != "identity" and args.encoding != "binary":
            raise SystemExit("layouts other than identity require --encoding binary")
    jobs = []
    for hi, path in enumerate(paths):
        gs = load_four_sat_npz(path)["ground_bitstring"] if any(
            lay != "identity" for lay in layouts) else None
        perms = {lay: (None if lay == "identity" else list(layout_spec(lay, gs, path.name).perm))
                 for lay in layouts}
        for L in layers:
            for t in range(int(args.trials)):
                seed = trial_seed(args.seed, hi, L, t)
                for lay, arm in [(lay, arm) for lay in layouts for arm in arms]:
                    # same seed for every arm and layout: paired trials
                    jobs.append(
                        {
                            "layout": lay,
                            "layout_perm": perms[lay],
                            "protocol": protocol,
                            "ham_path": str(path.resolve()),
                            "ham_file": path.name,
                            "arm": arm,
                            "n_layers": L,
                            "trial": t,
                            "seed": seed,
                            "steps": int(args.steps),
                            "spsa_a": args.spsa_a,
                            "spsa_c": float(args.spsa_c),
                            "spsa_A": float(args.spsa_A),
                            "optimizer": str(args.optimizer),
                            "adam_lr": float(args.adam_lr),
                            "encoding": str(args.encoding),
                        }
                    )
    return jobs


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ham-dir", type=str, default=str(_REPO / "Hamiltonians" / "four_sat"))
    p.add_argument(
        "--arms",
        type=str,
        default=DEFAULT_ARMS,
        help=f"Comma list of identity, jp, jp_local, g<γ/π> (default {DEFAULT_ARMS})",
    )
    p.add_argument(
        "--reference-arm",
        type=str,
        default="identity",
        help="Arm used for the paired per-trial deltas in the summary (default identity)",
    )
    p.add_argument("--layers", type=str, default="4")
    p.add_argument("--protocol", choices=PROTOCOLS, default="legacy",
                   help="legacy (default): fixed L, --optimizer, --steps; tuned: growth L=1→L, "
                   "SPSA-Adam lr 0.5/0.2/0.05/0.02, 200 steps/stage (1604 evals at L=4)")
    p.add_argument("--layouts", type=str, default="identity",
                   help="comma list of identity, rule_best, rule_bad (true-GS tier rule)")
    p.add_argument("--trials", type=int, default=25)
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--workers", type=int, default=max(1, min(4, os.cpu_count() or 1)))
    p.add_argument("--seed", type=int, default=20260917)
    p.add_argument("--max-h", type=int, default=None)
    p.add_argument("--spsa-a", type=float, default=None,
                   help="SPSA gain (default: scale_spsa_a(n_params))")
    p.add_argument("--spsa-c", type=float, default=0.15)
    p.add_argument("--spsa-A", type=float, default=10.0)
    p.add_argument("--optimizer", choices=("spsa", "spsa_adam"), default="spsa")
    p.add_argument("--adam-lr", type=float, default=ADAM_LR)
    p.add_argument("--encoding", choices=("binary", "gray"), default="binary")
    p.add_argument("--outdir", type=str, default=str(_REPO / "noiseless" / "results"))
    p.add_argument("--tag", type=str, default="ent_control")
    p.add_argument("--smoke", action="store_true")
    args = p.parse_args(argv)
    if args.smoke:
        args.arms = "identity,jp"
        args.layers = "2"
        args.trials = 1
        args.steps = 8
        args.max_h = 1
        if args.tag == "ent_control":
            args.tag = "ent_control_smoke"
    ref_name, _ = parse_arm(args.reference_arm)
    args.reference_arm = ref_name

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    jobs = build_jobs(args)
    print(
        f"[{_now()}] starting {len(jobs)} jobs arms={args.arms} layers={args.layers} "
        f"trials={args.trials} steps={args.steps} optimizer={args.optimizer} "
        f"protocol={args.protocol} layouts={args.layouts} "
        f"seed={args.seed} (arm-independent → paired) tag={args.tag}",
        flush=True,
    )
    records: list[dict] = []
    if args.workers <= 1 or len(jobs) == 1:
        for i, job in enumerate(jobs):
            rec = _worker(job)
            records.append(rec)
            print(
                f"  [{i + 1}/{len(jobs)}] {'OK' if rec.get('ok') else 'FAIL'} "
                f"{rec.get('arm')} L={rec.get('n_layers')} {rec.get('ham_file')} "
                f"succ={rec.get('success')} p_gs={rec.get('p_gs')} "
                f"S_final={rec.get('final_entropy')} wall={rec.get('wall_s'):.2f}s",
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
                    f"{rec.get('arm')} L={rec.get('n_layers')} {rec.get('ham_file')} "
                    f"succ={rec.get('success')} p_gs={rec.get('p_gs')} "
                    f"S_final={rec.get('final_entropy')} wall={rec.get('wall_s'):.2f}s",
                    flush=True,
                )

    agg = aggregate(records, args.reference_arm)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = outdir / f"{args.tag}_{stamp}.json"
    summary_path = outdir / f"{args.tag}_{stamp}_summary.json"
    payload_args = {
        "ham_dir": str(args.ham_dir),
        "arms": args.arms,
        "reference_arm": args.reference_arm,
        "layers": args.layers,
        "trials": int(args.trials),
        "steps": int(args.steps),
        "workers": int(args.workers),
        "seed": int(args.seed),
        "max_h": args.max_h,
        "spsa_a": args.spsa_a,
        "spsa_c": float(args.spsa_c),
        "spsa_A": float(args.spsa_A),
        "optimizer": str(args.optimizer),
        "adam_lr": float(args.adam_lr),
        "encoding": str(args.encoding),
        "protocol": str(args.protocol),
        "layouts": str(args.layouts),
        "seed_scheme": "seed0 + 1e6*ham_index + 1e3*L + trial (arm- and layout-independent)",
    }
    payload = {
        "created_utc": _now(),
        "tag": args.tag,
        "args": payload_args,
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
                "args": payload_args,
                "n_jobs": payload["n_jobs"],
                "n_ok": payload["n_ok"],
                "n_fail": payload["n_fail"],
                "arms": agg["arms"],
                "paired_vs_reference": agg["paired_vs_reference"],
            },
            indent=2,
        )
    )
    print(f"[{_now()}] wrote {out_path}", flush=True)
    print(f"[{_now()}] summary {summary_path}", flush=True)
    for row in agg["arms"]:
        print(
            f"  {row['layout']:>9} ARM {row['arm']:>9} L*={row['n_layers']} "
            f"success={row['success_rate']:.3f} mean_p_gs={row['mean_p_gs']:.4f} "
            f"S_final={row['mean_final_entropy']:.3f} S_max={row['mean_max_entropy']:.3f}",
            flush=True,
        )
    return 0 if payload["n_fail"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
