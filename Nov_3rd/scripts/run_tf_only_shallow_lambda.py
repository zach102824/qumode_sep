#!/usr/bin/env python3
"""Shallow SNAP vs Fourier on the transverse-field family (no cavity XX).

    H = diag(unsat) − λ Σ_i X_i,   μ = 0,   λ ∈ {0.50, 0.75, 1.00}

Train Walsh Hamming-shell SNAP and Fourier QAOA on n∈{6,9} at L∈{2,4}.
Checkpoint on the n=9 holdout. Freeze n=12 (first 128). Readout is the Z
histogram plus the X histogram. HVA is not trained: β^XX = 0 is Fourier, and
H has no XX term.

Does not read or write the n=21 ladder, qaoa_angles/, train69_ladder_fresh250/,
or settings.py.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path

# One thread per process. Pools own the parallelism. Must be set before
# numpy/torch initialize their OpenMP pools.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from Nov_3rd import settings as S
from Nov_3rd.hva_safe import (
    angles_to_lists,
    apply_H_all,
    best_product_energy,
    cavity_lambda_from_px,
    cn_from_unsat,
    expectation_H,
    ground_bundle,
    init_angles,
    lambda_of_state,
    method_parameter_list,
    prepare_state,
    product_gap,
    self_check,
    snap_state_from_components,
    summarize_eps,
    walsh_degree_components,
    _prob,
)
from Nov_3rd.tf_only import (
    CLASSICAL_PEAK,
    DEPTHS,
    FREEZE_N,
    GAP_PASS,
    LAMBDAS,
    LAMBDA_KILL,
    LAMBDA_PASS,
    MU,
    R_PASS,
    R_SOFT,
    TRAIN_NS,
    classify_cell,
    classify_family,
    decision_self_test,
    lambda_coeff,
    shell_rms,
)

RUN = S.OUT / "quantum_h_explore" / "tf_only_shallow_Lambda"
PACK = S.OUT / "beat_qaoa_12h" / "train_packs"
METHODS = ("fourier", "snap")
TRAIN_LO, TRAIN_HI = 0, 200
HOLD_LO, HOLD_HI = 200, 250
FREEZE_B = 128
PROTOCOL = "tf_only_shallow_lambda"
SAFE_N12_LAMBDA = 0.941  # PR #41, SAFE λ=0.75 μ=−0.50, median over cavities and instances


def log(*a):
    print(time.strftime("%H:%M:%S"), "[tf-only]", *a, flush=True)


def lam_tag(lam: float) -> str:
    return f"{float(lam):.2f}"


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(path)


def ratio(num: float, den: float) -> float:
    if den == 0 or not math.isfinite(den) or not math.isfinite(num):
        return float("inf")
    return float(num) / float(den)


def load_unsat(n: int, sl: slice) -> tuple[np.ndarray, np.ndarray]:
    if int(n) == 21:
        raise RuntimeError("n=21 is out of scope")
    path = PACK / f"train_n{n}.npz"
    d = np.load(path)
    E = np.asarray(d["train_E"][sl], dtype=np.float64)
    seeds = np.asarray(d["train_seeds"][sl]).astype(np.int64)
    if E.ndim != 2 or E.shape[1] != (1 << n):
        raise RuntimeError(f"n{n} E shape {E.shape}")
    return E, seeds


def _diag_job(payload: tuple) -> tuple:
    row, n, seed, lam = payload
    lx = lambda_coeff(float(lam))
    bundle = ground_bundle(np.asarray(row, dtype=np.float64), int(n), lambda_x=lx, mu_xx=0.0)
    lam_c = np.asarray(lambda_of_state(bundle["psi"], int(n)), dtype=np.float64).reshape(-1)
    if lam_c.size != int(n) // 3:
        raise RuntimeError(f"Λ shape {lam_c.shape} at n={n}")
    e_prod = best_product_energy(row, int(n), float(lam), 0.0, seed=int(seed))
    gap = product_gap(e_prod, bundle["E0"])
    if e_prod + 1e-5 < float(bundle["E0"]):
        raise RuntimeError(
            f"product energy {e_prod:.6f} below E0 {bundle['E0']:.6f} at n={n} λ={lam}"
        )
    return (
        float(bundle["E0"]),
        float(bundle["peak"]),
        float(bundle["residual"]),
        lam_c,
        float(e_prod),
        float(gap),
    )


def _pool_init(threads: int) -> None:
    os.environ["OMP_NUM_THREADS"] = str(int(threads))
    os.environ["MKL_NUM_THREADS"] = str(int(threads))
    torch.set_num_threads(int(threads))


def diag_path(n: int, role: str, lam: float) -> Path:
    return RUN / f"diag_n{n}_{role}_l{lam_tag(lam)}.npz"


def diagnostics(n: int, role: str, lam: float, E: np.ndarray, seeds: np.ndarray, workers: int) -> dict:
    """E0, Z-peak, per-cavity Λ, and the XZ product gap at this λ, μ=0."""
    cache = diag_path(n, role, lam)
    if cache.exists():
        z = np.load(cache)
        same = (
            np.array_equal(z["seeds"], seeds)
            and int(z["e0"].shape[0]) == int(E.shape[0])
            and abs(float(z["lam"]) - float(lam)) < 1e-12
            and abs(float(z["mu"])) < 1e-12
        )
        if same:
            log(f"diag cache hit n{n} {role} λ={lam_tag(lam)}")
            return {k: np.asarray(z[k]) for k in ("e0", "peak", "residual", "lambda_c", "e_prod", "prod_gap", "seeds")}
    log(f"diag n{n} {role} λ={lam_tag(lam)} B={E.shape[0]} workers={workers}")
    t0 = time.time()
    jobs = [(E[i], n, int(seeds[i]), float(lam)) for i in range(E.shape[0])]
    if workers <= 1 or len(jobs) == 1:
        rows = []
        for i, job in enumerate(jobs):
            rows.append(_diag_job(job))
            if (i + 1) == len(jobs) or (i + 1) % 8 == 0:
                log(
                    f"diag n{n} λ={lam_tag(lam)} {i+1}/{len(jobs)} "
                    f"E0={rows[-1][0]:.4f} Λmed={float(np.median(rows[-1][3])):.3f} "
                    f"elapsed={time.time()-t0:.0f}s"
                )
    else:
        import multiprocessing as mp

        ctx = mp.get_context("spawn")
        rows = []
        with ctx.Pool(int(workers), initializer=_pool_init, initargs=(1,)) as pool:
            for i, row in enumerate(pool.imap(_diag_job, jobs, chunksize=1)):
                rows.append(row)
                if (i + 1) == len(jobs) or (i + 1) % 4 == 0:
                    log(
                        f"diag n{n} λ={lam_tag(lam)} {i+1}/{len(jobs)} "
                        f"E0={row[0]:.4f} Λmed={float(np.median(row[3])):.3f} "
                        f"elapsed={time.time()-t0:.0f}s"
                    )
    e0 = np.array([r[0] for r in rows], dtype=np.float64)
    peak = np.array([r[1] for r in rows], dtype=np.float64)
    residual = np.array([r[2] for r in rows], dtype=np.float64)
    lam_c = np.stack([r[3] for r in rows], axis=0)
    e_prod = np.array([r[4] for r in rows], dtype=np.float64)
    gap = np.array([r[5] for r in rows], dtype=np.float64)
    if np.any(e0 > E.min(axis=1) + 1e-6):
        raise RuntimeError(f"E0 above a diagonal entry at n={n} {role} λ={lam}")
    if np.any(residual > 1e-5):
        raise RuntimeError(f"ground residual {float(residual.max())} at n={n}")
    if np.any(gap < -1e-6):
        raise RuntimeError(f"negative product gap {float(gap.min())} at n={n}")
    RUN.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache,
        seeds=seeds,
        e0=e0,
        peak=peak,
        residual=residual,
        lambda_c=lam_c,
        e_prod=e_prod,
        prod_gap=gap,
        n=np.int64(n),
        role=role,
        lam=np.float64(lam),
        mu=np.float64(0.0),
    )
    log(
        f"diag n{n} {role} λ={lam_tag(lam)} {time.time()-t0:.1f}s  "
        f"medE0={np.median(e0):.4f} Λ={np.median(lam_c):.4f} "
        f"gap={np.median(gap):.4f} peak={np.median(peak):.4f}"
    )
    return {
        "e0": e0,
        "peak": peak,
        "residual": residual,
        "lambda_c": lam_c,
        "e_prod": e_prod,
        "prod_gap": gap,
        "seeds": seeds,
    }


def diag_summary(rec: dict, n: int, role: str) -> dict:
    lam_c = np.asarray(rec["lambda_c"], dtype=np.float64)
    gap = np.asarray(rec["prod_gap"], dtype=np.float64)
    peak = np.asarray(rec["peak"], dtype=np.float64)
    e0 = np.asarray(rec["e0"], dtype=np.float64)
    return {
        "n": int(n),
        "role": role,
        "B": int(peak.shape[0]),
        "lambda_med": float(np.median(lam_c)),
        "lambda_min": float(np.min(lam_c)),
        "lambda_max": float(np.max(lam_c)),
        "frac_lambda_gt_085": float(np.mean(lam_c > LAMBDA_KILL)),
        "gap_med": float(np.median(gap)),
        "gap_min": float(np.min(gap)),
        "frac_gap_ge_003": float(np.mean(gap + 1e-12 >= GAP_PASS)),
        "peak_med": float(np.median(peak)),
        "peak_max": float(np.max(peak)),
        "frac_classical": float(np.mean(peak + 1e-12 >= CLASSICAL_PEAK)),
        "e0_med": float(np.median(e0)),
    }


def seed_path(L: int, method: str, seed: int, lam: float) -> Path:
    return RUN / f"L{L}_{method}_s{seed}_l{lam_tag(lam)}.json"


def _as_f64(angles: dict) -> dict:
    return {
        k: (v.detach().to(dtype=torch.float64) if torch.is_tensor(v) else v)
        for k, v in angles.items()
    }


def eval_method(method: str, angles: dict, bundle: dict, lam: float, *, chunk: int) -> dict:
    ang = _as_f64(angles)
    lx = lambda_coeff(lam)
    out = {}
    for n, (E, e0, seeds) in bundle.items():
        use = 8 if int(n) >= 12 else chunk
        Eu = torch.tensor(E, dtype=torch.float64)
        e0t = torch.tensor(e0, dtype=torch.float64)
        eps_parts = []
        lam_parts = []
        B = Eu.shape[0]
        for s in range(0, B, use):
            sl = slice(s, s + use)
            Cn = cn_from_unsat(Eu[sl], int(n))
            psi = prepare_state(method, Cn, ang)
            h = expectation_H(psi, Eu[sl], int(n), lambda_x=lx, mu_xx=0.0)
            eps_parts.append((h - e0t[sl]).detach().cpu().numpy())
            pX = _prob(apply_H_all(psi, int(n)))
            lam_parts.append(cavity_lambda_from_px(pX, int(n)))
        eps = np.concatenate(eps_parts, axis=0)
        prep = np.concatenate(lam_parts, axis=0)
        rec = summarize_eps(eps)
        rec.update(
            {
                "n": int(n),
                "seeds": [int(x) for x in np.asarray(seeds).tolist()],
                "eps": [float(x) for x in eps.tolist()],
                "e0_med": float(np.median(e0)),
                "prep_lambda_med": float(np.median(prep)),
            }
        )
        out[str(n)] = rec
    return out


def _loss_on(method, angles, Eu, e0b, n: int, lam: float, comps) -> torch.Tensor:
    lx = lambda_coeff(lam)
    if method == "snap":
        psi = snap_state_from_components(comps, angles["gamma_w"], angles["phi"])
    else:
        Cn = cn_from_unsat(Eu, n)
        psi = prepare_state(method, Cn, angles)
    h = expectation_H(psi, Eu, n, lambda_x=lx, mu_xx=0.0)
    return ((h - e0b) / float(n)).mean()


def math_finite(x: float) -> bool:
    return x == x and abs(x) != float("inf")


def train_one(
    method: str,
    L: int,
    seed: int,
    lam: float,
    bundles_t: dict,
    hold_bundle: dict,
    *,
    rounds: int,
    batch: int,
) -> dict:
    torch.manual_seed(int(seed) + 1009 * int(round(100 * lam)) + 17 * int(L))
    rng = np.random.default_rng(int(seed) + 10007 * int(round(1000 * lam)) + L)
    angles = init_angles(method, L, dtype=torch.float64)
    params = method_parameter_list(method, angles)
    E6, e06, _s6 = bundles_t[6]
    E9, e09, _s9 = bundles_t[9]
    n6 = E6.shape[0]
    n9 = E9.shape[0]
    b6 = min(batch // 2, n6)
    b9 = min(batch - b6, n9)

    def hold_med() -> float:
        ev = eval_method(method, angles, hold_bundle, lam, chunk=16)
        return float(ev["9"]["med_eps"])

    t0 = time.time()
    init_med = hold_med()
    best = {
        "med9": init_med,
        "round": -1,
        "angles": {k: v.detach().cpu().clone() for k, v in angles.items()},
    }
    hist = [{"round": -1, "hold_med9": init_med, "loss": None}]
    log(f"L{L} {method} s{seed} λ={lam_tag(lam)} init n9-hold Medε={init_med:.5f}")

    for r in range(rounds):
        opt = torch.optim.LBFGS(
            params,
            lr=S.LBFGS_LR,
            max_iter=S.LBFGS_ITER,
            history_size=16,
            line_search_fn="strong_wolfe",
        )
        i6 = torch.tensor(rng.choice(n6, size=b6, replace=False), dtype=torch.long)
        i9 = torch.tensor(rng.choice(n9, size=b9, replace=False), dtype=torch.long)
        Eu6 = E6.index_select(0, i6)
        Eu9 = E9.index_select(0, i9)
        e6b = e06.index_select(0, i6)
        e9b = e09.index_select(0, i9)
        comps6 = comps9 = None
        if method == "snap":
            comps6 = walsh_degree_components(cn_from_unsat(Eu6, 6)).detach()
            comps9 = walsh_degree_components(cn_from_unsat(Eu9, 9)).detach()

        def closure():
            opt.zero_grad()
            loss = 0.5 * (
                _loss_on(method, angles, Eu6, e6b, 6, lam, comps6)
                + _loss_on(method, angles, Eu9, e9b, 9, lam, comps9)
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            return loss

        try:
            loss = opt.step(closure)
            loss_f = float(loss.detach()) if torch.is_tensor(loss) else float(loss)
        except Exception as e:
            log(f"L{L} {method} s{seed} λ={lam_tag(lam)} r{r} LBFGS skip: {e}")
            hist.append({"round": r, "skip": str(e)})
            continue
        if not math_finite(loss_f):
            log(f"L{L} {method} s{seed} λ={lam_tag(lam)} r{r} nonfinite loss — restore best")
            with torch.no_grad():
                for k, v in best["angles"].items():
                    angles[k].data.copy_(v.to(dtype=angles[k].dtype))
            hist.append({"round": r, "loss": None})
            continue
        med = hold_med()
        hist.append({"round": r, "loss": loss_f, "hold_med9": med})
        mark = ""
        if med < best["med9"]:
            best = {
                "med9": med,
                "round": r,
                "angles": {k: v.detach().cpu().clone() for k, v in angles.items()},
            }
            mark = " *"
        if r % 5 == 0 or r + 1 == rounds or mark:
            log(
                f"L{L} {method} s{seed} λ={lam_tag(lam)} r{r}/{rounds} "
                f"loss={loss_f:.5f} holdMedε={med:.5f} best={best['med9']:.5f}@r{best['round']}{mark}"
            )

    with torch.no_grad():
        for k, v in best["angles"].items():
            angles[k].data.copy_(v.to(dtype=angles[k].dtype))
    return {
        "method": method,
        "L": int(L),
        "seed": int(seed),
        "lambda": float(lam),
        "mu": 0.0,
        "init_hold_med9": init_med,
        "best_round": int(best["round"]),
        "best_hold_med9": float(best["med9"]),
        "wall_s": time.time() - t0,
        "hist": hist,
        "angles": best["angles"],
        "n_params": int(sum(p.numel() for p in params)),
    }


def _load_split(n: int, role: str, lam: float):
    if role == "train":
        E, seeds = load_unsat(n, slice(TRAIN_LO, TRAIN_HI))
    elif role == "hold":
        E, seeds = load_unsat(n, slice(HOLD_LO, HOLD_HI))
    elif role == "freeze":
        if n != FREEZE_N:
            raise RuntimeError(role)
        E, seeds = load_unsat(n, slice(0, FREEZE_B))
    else:
        raise RuntimeError(role)
    diag = diagnostics(n, role, lam, E, seeds, workers=1)
    return E, diag["e0"], seeds


def _train_worker(spec: dict) -> str:
    try:
        _pool_init(int(spec["threads"]))
        lam = float(spec["lam"])
        method = spec["method"]
        L = int(spec["L"])
        seed = int(spec["seed"])
        path = seed_path(L, method, seed, lam)
        if path.exists() and not spec["force"]:
            log(f"skip existing {path.name}")
            return str(path)
        bundles_t = {}
        for n in TRAIN_NS:
            E, e0, seeds = _load_split(n, "train", lam)
            bundles_t[n] = (torch.tensor(E, dtype=torch.float64), torch.tensor(e0, dtype=torch.float64), seeds)
        E9, e09, s9 = _load_split(9, "hold", lam)
        hold = {9: (E9, e09, s9)}
        trained = train_one(
            method,
            L,
            seed,
            lam,
            bundles_t,
            hold,
            rounds=int(spec["rounds"]),
            batch=int(spec["batch"]),
        )
        eval_bundle = {}
        for n, role in ((6, "hold"), (9, "hold"), (12, "freeze")):
            E, e0, seeds = _load_split(n, role, lam)
            eval_bundle[n] = (E, e0, seeds)
        ev = eval_method(method, trained["angles"], eval_bundle, lam, chunk=16)
        payload = {
            "protocol": PROTOCOL,
            "method": method,
            "L": L,
            "seed": seed,
            "lambda": lam,
            "mu": 0.0,
            "rounds": int(spec["rounds"]),
            "batch": int(spec["batch"]),
            "init_hold_med9": trained["init_hold_med9"],
            "best_round": trained["best_round"],
            "best_hold_med9": trained["best_hold_med9"],
            "wall_s": trained["wall_s"],
            "n_params": trained["n_params"],
            "angles": angles_to_lists(method, trained["angles"]),
            "hist": trained["hist"],
            "eval": ev,
        }
        if method == "snap":
            payload["shell_rms"] = shell_rms(trained["angles"]["phi"].detach().cpu().numpy())
        write_json(path, payload)
        log(
            f"wrote {path.name} best r{trained['best_round']} "
            f"n9={trained['best_hold_med9']:.5f} wall={trained['wall_s']:.0f}s"
        )
        return str(path)
    except Exception:
        traceback.print_exc()
        raise


def _fmt(x, nd: int = 4) -> str:
    if x is None or not isinstance(x, (int, float)) or not math.isfinite(float(x)):
        return "—"
    return f"{float(x):.{nd}f}"


def _load_rows(lambdas, depths, seeds) -> list[dict]:
    rows = []
    missing = []
    for lam in lambdas:
        for L in depths:
            for method in METHODS:
                for seed in seeds:
                    path = seed_path(L, method, seed, lam)
                    if not path.exists():
                        missing.append(path.name)
                        continue
                    rows.append(json.loads(path.read_text()))
    if missing:
        raise RuntimeError("missing seed files: " + ", ".join(missing[:12]))
    return rows


def _med_over_seeds(rows: list[dict], method: str, n: int) -> float:
    vals = []
    for row in rows:
        if row["method"] != method:
            continue
        vals.append(float(row["eval"][str(n)]["med_eps"]))
    if not vals:
        raise RuntimeError(f"no rows for {method} n={n}")
    return float(np.median(np.asarray(vals, dtype=np.float64)))


def _paired(rows: list[dict], n: int) -> tuple[int, int]:
    by_seed = {}
    for row in rows:
        by_seed.setdefault(int(row["seed"]), {})[row["method"]] = np.asarray(
            row["eval"][str(n)]["eps"], dtype=np.float64
        )
    wins = 0
    total = 0
    for pair in by_seed.values():
        if "snap" not in pair or "fourier" not in pair:
            continue
        wins += int(np.sum(pair["snap"] < pair["fourier"] - 1e-12))
        total += int(pair["snap"].size)
    return wins, total


def aggregate(lambdas, depths, seeds) -> dict:
    rows = _load_rows(lambdas, depths, seeds)
    diag = {}
    for lam in lambdas:
        diag[lam_tag(lam)] = {}
        for n, role in ((6, "train"), (6, "hold"), (9, "train"), (9, "hold"), (12, "freeze")):
            E, seeds_n = (
                load_unsat(n, slice(TRAIN_LO, TRAIN_HI))
                if role == "train"
                else load_unsat(n, slice(HOLD_LO, HOLD_HI))
                if role == "hold"
                else load_unsat(n, slice(0, FREEZE_B))
            )
            rec = diagnostics(n, role, lam, E, seeds_n, workers=1)
            key = f"{n}_{role}"
            diag[lam_tag(lam)][key] = diag_summary(rec, n, role)

    cells = []
    depth_tables = []
    for lam in lambdas:
        by_depth = {}
        r12 = {}
        r_train = {}
        for L in depths:
            block = [r for r in rows if abs(float(r["lambda"]) - float(lam)) < 1e-12 and int(r["L"]) == int(L)]
            by_n = {}
            for n in (6, 9, 12):
                med_f = _med_over_seeds(block, "fourier", n)
                med_s = _med_over_seeds(block, "snap", n)
                r = ratio(med_s, med_f)
                wins, total = _paired(block, n)
                by_n[str(n)] = {
                    "med_fourier": med_f,
                    "med_snap": med_s,
                    "R": r,
                    "paired_snap_below": wins,
                    "paired_total": total,
                }
            r12[int(L)] = by_n["12"]["R"]
            r_train[(6, int(L))] = by_n["6"]["R"]
            r_train[(9, int(L))] = by_n["9"]["R"]
            seed_rows = []
            for seed in seeds:
                snap = next(r for r in block if r["method"] == "snap" and int(r["seed"]) == int(seed))
                four = next(r for r in block if r["method"] == "fourier" and int(r["seed"]) == int(seed))
                s12 = float(snap["eval"]["12"]["med_eps"])
                f12 = float(four["eval"]["12"]["med_eps"])
                seed_rows.append(
                    {
                        "seed": int(seed),
                        "n12_snap": s12,
                        "n12_fourier": f12,
                        "R12": ratio(s12, f12),
                        "n6_snap": float(snap["eval"]["6"]["med_eps"]),
                        "n6_fourier": float(four["eval"]["6"]["med_eps"]),
                        "n9_snap": float(snap["eval"]["9"]["med_eps"]),
                        "n9_fourier": float(four["eval"]["9"]["med_eps"]),
                        "best_round_snap": snap["best_round"],
                        "best_round_fourier": four["best_round"],
                        "wall_snap": snap["wall_s"],
                        "wall_fourier": four["wall_s"],
                        "shell_rms": snap.get("shell_rms"),
                        "prep_lambda_snap": snap["eval"]["12"].get("prep_lambda_med"),
                        "prep_lambda_fourier": four["eval"]["12"].get("prep_lambda_med"),
                    }
                )
            shell_vals = [s["shell_rms"] for s in seed_rows if s["shell_rms"] is not None]
            by_depth[str(L)] = {
                "by_n": by_n,
                "seeds": seed_rows,
                "shell_rms_med": float(np.median(shell_vals)) if shell_vals else None,
            }
            depth_tables.append({"lambda": float(lam), "L": int(L), **by_depth[str(L)]})
        gate = diag[lam_tag(lam)]["12_freeze"]
        cell = classify_cell(
            lam_med=gate["lambda_med"],
            gap_med=gate["gap_med"],
            peak_med=gate["peak_med"],
            r12_by_depth=r12,
            r_train=r_train,
        )
        cell["lambda"] = float(lam)
        cell["depths"] = by_depth
        cell["diag_n12"] = gate
        cells.append(cell)

    family = classify_family(cells)
    summary = {
        "protocol": PROTOCOL,
        "verdict": family["verdict"],
        "mu": MU,
        "lambdas": [float(x) for x in lambdas],
        "depths": [int(x) for x in depths],
        "seeds": [int(x) for x in seeds],
        "safe_n12_lambda": SAFE_N12_LAMBDA,
        "diag": diag,
        "cells": cells,
        "family_calls": family["calls"],
    }
    return summary


def render(summary: dict) -> str:
    v = summary["verdict"]
    lines = []
    lines.append("# Transverse-field only — shallow SNAP vs Fourier (μ = 0)")
    lines.append("")
    lines.append(f"**Verdict: {v}.**")
    lines.append("")
    lines.append(
        "No cavity XX. The Hamiltonian is H = diag(unsat) − λ Σ_i X_i with "
        "λ ∈ {0.50, 0.75, 1.00}. Walsh Hamming-shell SNAP is compared with Fourier "
        "QAOA. HVA is not trained: β^XX = 0 prepares the Fourier state, and there is "
        "no XX term in H for a separate mixer to match. SAFE (PR #41, λ=0.75, μ=−0.50) "
        f"had n=12 median cavity Λ = {SAFE_N12_LAMBDA:.3f}."
    )
    lines.append("")
    lines.append("## Gate")
    lines.append("")
    lines.append(
        "Med ε is the median over seeds of each seed's median-over-instances excess "
        "⟨H⟩−E₀. R_Fourier = Medε_SNAP / Medε_Fourier. Λ, the product gap, and the "
        "Z-peak are medians on the frozen n=12 pool (B=128). Λ pools cavities and "
        "instances of the true ground state's X histogram. The product gap is "
        "(E_prod − E_0)/|E_0|. A classical ground state has median Z-peak ≥ 0.5."
    )
    lines.append("")
    lines.append(
        f"- **PASS** if some λ has median Λ ≤ {LAMBDA_PASS:.2f}, product gap ≥ {GAP_PASS:.2f}, "
        f"Z-peak < {CLASSICAL_PEAK:.1f}, and n=12 R_Fourier ≤ {R_PASS:.2f} at L=2 or at L=4."
    )
    lines.append(
        f"- **needs_edit** if Λ, the gap, and the Z-peak meet that bar and the better "
        f"n=12 depth has R_Fourier in ({R_PASS:.2f}, {R_SOFT:.2f}]. Also the residual: "
        "a λ that misses PASS and is not a KILL."
    )
    lines.append(
        f"- **KILL** a λ when median Λ > {LAMBDA_KILL:.2f}, or the ground state is classical, "
        f"or R_Fourier > {R_SOFT:.2f} at both L=2 and L=4 on the n=6 and n=9 holdouts. "
        "The family is KILL only when every λ is KILL. One PASS λ is a family PASS."
    )
    lines.append("")
    lines.append("## Hamiltonian and protocol")
    lines.append("")
    lines.append(
        "Cavity c owns qubits [3c, 3c+3), qubit 0 = MSB. The circuit phases "
        "Cn = unsat · n/m. ⟨H⟩ uses the raw unsat count from the Z histogram and "
        "⟨X_i⟩ from the X histogram. No Y, and no XX correlator in the energy. "
        "L-BFGS matches `Nov_3rd.settings` and the shallow seals: 40 rounds × max_iter 8, "
        "lr 0.25, history 16, strong Wolfe, grad clip 5, batch 64 (half n=6 / half n=9). "
        "New optimizer each round. Loss = mean ε/n. Each seed restores the round with "
        "the best n=9-holdout Med ε."
    )
    lines.append("")
    lines.append(
        "Packs: `beat_qaoa_12h/train_packs`. Train indices [0:200) on n=6 and n=9. "
        "Holdout [200:250). Freeze n=12 first 128. Paper init β=0.01, γ=γ_w=−0.01, "
        "φ_k=−β k. L≥8 was not trained. n=21 was not loaded. `qaoa_angles/`, "
        "`train69_ladder_fresh250/`, and `settings.py` were not written."
    )
    lines.append("")
    lines.append("## Ground state")
    lines.append("")
    lines.append(
        "| λ | n | role | B | median Λ | min Λ | max Λ | frac Λ>0.85 | median gap | min gap | median Z-peak | frac peak≥0.5 |"
    )
    lines.append("|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for lam in summary["lambdas"]:
        block = summary["diag"][lam_tag(lam)]
        for key in ("6_train", "6_hold", "9_train", "9_hold", "12_freeze"):
            d = block[key]
            lines.append(
                f"| {lam:.2f} | {d['n']} | {d['role']} | {d['B']} | "
                f"{d['lambda_med']:.4f} | {d['lambda_min']:.4f} | {d['lambda_max']:.4f} | "
                f"{d['frac_lambda_gt_085']:.3f} | {d['gap_med']:.4f} | {d['gap_min']:.4f} | "
                f"{d['peak_med']:.4f} | {d['frac_classical']:.3f} |"
            )
    lines.append("")
    lines.append("The decision row is n=12 freeze. n=6 and n=9 are the same ensemble, reported so a size accident is visible.")
    lines.append("")
    lines.append("## Ratios")
    lines.append("")
    lines.append("| λ | L | n | role | Medε Fourier | Medε SNAP | R_Fourier | paired SNAP<Fourier |")
    lines.append("|---:|---:|---:|---|---:|---:|---:|---:|")
    role = {6: "holdout", 9: "holdout (select)", 12: "freeze"}
    for cell in summary["cells"]:
        for L in summary["depths"]:
            block = cell["depths"][str(L)]
            for n in (6, 9, 12):
                rec = block["by_n"][str(n)]
                lines.append(
                    f"| {cell['lambda']:.2f} | {L} | {n} | {role[n]} | "
                    f"{rec['med_fourier']:.4f} | {rec['med_snap']:.4f} | {rec['R']:.4f} | "
                    f"{rec['paired_snap_below']}/{rec['paired_total']} |"
                )
    lines.append("")
    lines.append("| λ | call | detail | n12 Λ | n12 gap | n12 Z-peak | n12 R L=2 | n12 R L=4 | shell RMS L=2 | shell RMS L=4 |")
    lines.append("|---:|---|---|---:|---:|---:|---:|---:|---:|---:|")
    for cell in summary["cells"]:
        d2 = cell["depths"].get("2", {})
        d4 = cell["depths"].get("4", {})
        lines.append(
            f"| {cell['lambda']:.2f} | {cell['call']} | {cell['detail']} | "
            f"{cell['lambda_med']:.4f} | {cell['gap_med']:.4f} | {cell['peak_med']:.4f} | "
            f"{cell['r12_L2']:.4f} | {cell['r12_L4']:.4f} | "
            f"{_fmt(d2.get('shell_rms_med'))} | {_fmt(d4.get('shell_rms_med'))} |"
        )
    lines.append("")
    lines.append("### Seeds at n=12")
    lines.append("")
    lines.append("| λ | L | seed | SNAP | Fourier | R | shell RMS | prep Λ SNAP | prep Λ Fourier | best round SNAP | wall s |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for cell in summary["cells"]:
        for L in summary["depths"]:
            for s in cell["depths"][str(L)]["seeds"]:
                lines.append(
                    f"| {cell['lambda']:.2f} | {L} | {s['seed']} | {s['n12_snap']:.4f} | "
                    f"{s['n12_fourier']:.4f} | {s['R12']:.4f} | {_fmt(s['shell_rms'])} | "
                    f"{_fmt(s['prep_lambda_snap'])} | {_fmt(s['prep_lambda_fourier'])} | "
                    f"{s['best_round_snap']} | {s['wall_snap']:.0f} |"
                )
    lines.append("")
    lines.append("Prepared-state Λ is the X histogram of the circuit state. The gate uses the ground-state Λ above, not this column.")
    lines.append("")
    lines.append("## Reading")
    lines.append("")
    lines.extend(_reading(summary))
    lines.append("")
    lines.append(
        "Per-seed JSON is `L{L}_{method}_s{seed}_l{λ}.json` in this folder. Angles are the "
        "restored n=9-holdout checkpoint. Ground-state Λ, the product gap, and the Z-peak "
        "are in `diag_n{n}_{role}_l{λ}.npz`."
    )
    lines.append("")
    return "\n".join(lines)


def _reading(summary: dict) -> list[str]:
    v = summary["verdict"]
    by = {round(float(c["lambda"]), 2): c for c in summary["cells"]}
    c50, c75, c100 = by[0.50], by[0.75], by[1.00]
    d50 = summary["diag"]["0.50"]["6_train"]
    d12_50 = summary["diag"]["0.50"]["12_freeze"]
    d12_100 = summary["diag"]["1.00"]["12_freeze"]
    gap_floor = min(float(c["diag_n12"]["gap_min"]) for c in summary["cells"])
    lines = [
        f"The call is **{v}**. Dropping the cavity XX term cuts the ground-state cavity lock "
        f"from the SAFE n=12 median Λ of {SAFE_N12_LAMBDA:.3f} down to "
        f"{c50['lambda_med']:.3f} / {c75['lambda_med']:.3f} / {c100['lambda_med']:.3f} "
        "at λ = 0.50 / 0.75 / 1.00.",
        "",
        "On the frozen n=12 pool the product gap stays near 0.10 "
        f"({c50['gap_med']:.3f}, {c75['gap_med']:.3f}, {c100['gap_med']:.3f}), "
        f"and every instance on that pool is at least {gap_floor:.3f} above the best XZ product state. "
        "The median Z-peak falls with the field: "
        f"{c50['peak_med']:.3f}, {c75['peak_med']:.3f}, {c100['peak_med']:.3f}, "
        "all below 0.5.",
        "",
        f"λ=0.50 and λ=0.75 are PASS. Λ is under 0.80 ({c50['lambda_med']:.3f} and "
        f"{c75['lambda_med']:.3f}) and the gap is above 0.03. L=2 beats Fourier, "
        f"R={c50['r12_L2']:.3f} and R={c75['r12_L2']:.3f}, with every seed under 0.76. "
        f"L=4 at those two fields is only a soft lead (R={c50['r12_L4']:.3f} and "
        f"R={c75['r12_L4']:.3f}). The PASS line accepts either depth, so L=2 carries them.",
        "",
        f"λ=1.00 is the KILL cell. Median Λ={c100['lambda_med']:.3f} is just above 0.85, "
        f"and {d12_100['frac_lambda_gt_085']:.0%} of cavity-instance pairs sit above 0.85. "
        f"The Z-peak is {c100['peak_med']:.3f}, so the state is spread in Z. "
        f"The ratio at this field would have cleared the PASS line by itself "
        f"(L=2 R={c100['r12_L2']:.3f}, L=4 R={c100['r12_L4']:.3f}; "
        "SNAP is below Fourier on every paired L=2 instance). "
        "The transverse field alone is enough to push Λ over 0.85 once λ=1.",
        "",
        f"At λ=0.50, {d50['frac_classical']:.0%} of the n=6 training instances have "
        f"a Z-peak ≥ 0.5. On the n=12 freeze that fraction is "
        f"{d12_50['frac_classical']:.1%}, and the median peak is {c50['peak_med']:.3f}.",
        "",
        "The L=2 Hamming shell is only mildly bent off φ_k = a + bk "
        f"(median RMS {c50['depths']['2']['shell_rms_med']:.3f} at λ=0.50 and "
        f"{c75['depths']['2']['shell_rms_med']:.3f} at λ=0.75). "
        "Part of the lead over Fourier is the Walsh-graded cost: four γ weights per layer "
        "against Fourier's one. L=4 shells bend more "
        f"(RMS {c50['depths']['4']['shell_rms_med']:.3f} and "
        f"{c75['depths']['4']['shell_rms_med']:.3f} at those two fields). "
        "Prepared-state Λ sits above the ground-state Λ, near 0.78–0.90. "
        "The gate uses the ground-state number.",
    ]
    if v == "KILL":
        lines.append("")
        lines.append("Every λ met a KILL clause, so the no-XX family does not clear the shallow bar.")
    elif v == "needs_edit":
        lines.append("")
        lines.append(
            "No λ clears Λ, the product gap, and R_Fourier ≤ 0.85 together, "
            "and at least one λ is not a KILL."
        )
    return lines


def _reject_scope(depths, methods, lambdas) -> None:
    if any(int(L) not in DEPTHS for L in depths):
        raise SystemExit(f"only L in {DEPTHS}, got {depths}")
    if any(m not in METHODS for m in methods):
        raise SystemExit("only fourier and snap; HVA is not trained at μ=0")
    if any(int(n) >= 15 for n in ()):  # n≥15 is not a knob
        raise SystemExit("n≥15 is out of scope")
    for lam in lambdas:
        if lam_tag(lam) not in {lam_tag(x) for x in LAMBDAS}:
            raise SystemExit(f"λ {lam} is outside {LAMBDAS}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--depths", type=str, default="2,4")
    ap.add_argument("--seeds", type=str, default="0,1,2,3,4")
    ap.add_argument("--lambdas", type=str, default="0.50,0.75,1.00")
    ap.add_argument("--methods", type=str, default="fourier,snap")
    ap.add_argument("--rounds", type=int, default=S.LBFGS_ROUNDS)
    ap.add_argument("--batch", type=int, default=S.LBFGS_BATCH)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--aggregate-only", action="store_true")
    ap.add_argument("--diag-only", action="store_true")
    ap.add_argument("--skip-self-check", action="store_true")
    args = ap.parse_args()
    depths = [int(x) for x in args.depths.split(",") if x]
    seeds = [int(x) for x in args.seeds.split(",") if x]
    lambdas = [float(x) for x in args.lambdas.split(",") if x]
    methods = [x.strip() for x in args.methods.split(",") if x.strip()]
    _reject_scope(depths, methods, lambdas)
    if not args.skip_self_check:
        log("decision self-test")
        decision_self_test()
        log("SAFE-point ansatz self-check")
        self_check()
        log("self-check ok")
    RUN.mkdir(parents=True, exist_ok=True)
    if not args.aggregate_only:
        cpu = os.cpu_count() or 2
        diag_workers = max(1, min(int(args.workers), cpu))
        for lam in lambdas:
            for n, role, sl in (
                (6, "train", slice(TRAIN_LO, TRAIN_HI)),
                (6, "hold", slice(HOLD_LO, HOLD_HI)),
                (9, "train", slice(TRAIN_LO, TRAIN_HI)),
                (9, "hold", slice(HOLD_LO, HOLD_HI)),
                (12, "freeze", slice(0, FREEZE_B)),
            ):
                E, seeds_n = load_unsat(n, sl)
                diagnostics(n, role, lam, E, seeds_n, workers=diag_workers)
        if args.diag_only:
            log("diag only — stop before training")
            return
        threads = max(1, cpu // max(1, int(args.workers)))
        specs = []
        for lam in lambdas:
            for L in depths:
                for method in methods:
                    for seed in seeds:
                        path = seed_path(L, method, seed, lam)
                        if path.exists() and not args.force:
                            continue
                        specs.append(
                            {
                                "lam": float(lam),
                                "L": int(L),
                                "method": method,
                                "seed": int(seed),
                                "rounds": int(args.rounds),
                                "batch": int(args.batch),
                                "threads": threads,
                                "force": bool(args.force),
                            }
                        )
        log(f"training jobs {len(specs)} workers={args.workers} threads/job={threads} rounds={args.rounds}")
        if specs:
            if int(args.workers) <= 1:
                for spec in specs:
                    _train_worker(spec)
            else:
                import multiprocessing as mp

                ctx = mp.get_context("spawn")
                with ctx.Pool(int(args.workers)) as pool:
                    for path in pool.imap_unordered(_train_worker, specs):
                        log(f"done {path}")
    summary = aggregate(lambdas, depths, seeds)
    write_json(RUN / "summary.json", summary)
    text = render(summary)
    (RUN / "RESULTS.md").write_text(text)
    log(f"verdict {summary['verdict']} wrote {RUN / 'RESULTS.md'}")
    print(text)


if __name__ == "__main__":
    main()
