#!/usr/bin/env python3
"""Five-seed seal of paper-lock NEW_SAFE through n=18.

    H = diag(unsat) − 0.75 Σ_i X_i

NEW_SAFE is bakeoff #46 cell C: λ=0.75, μ=0. Walsh Hamming-shell SNAP versus
Fourier QAOA. Train on n∈{6,9} at L∈{2,4}. Checkpoint on the n=9 holdout.
Freeze n=12 (B=128). Probe n=15 (B=64) and n=18 (B=32). Report median cavity
Λ from the true ground state's X histogram at n=12, n=15, and n=18.

HVA is not trained. At μ=0, β^XX=0 prepares the Fourier state, and H has no
XX term. L=4 is reported and is not a PASS bar (bakeoff prior ~0.87).

Does not read or write the n=21 ladder, qaoa_angles/, train69_ladder_fresh250/,
or settings.py. Old SAFE (λ=0.75, μ=−0.50) is not re-trained.
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
    dense_H,
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
    walsh_power_by_degree,
    _prob,
)
from Nov_3rd.seal_new_safe import (
    BEAT_NEED,
    GAP_PASS,
    LAM,
    LAMBDA_KILL,
    LAMBDA_PASS,
    LAMBDA_X,
    MU,
    N_SEEDS,
    R_BEAT,
    R_PASS,
    classify_seal,
    decision_self_test,
    shell_rms,
)

RUN = S.OUT / "quantum_h_explore" / "seal_NEW_SAFE_lam075_mu0_n18"
PACK = S.OUT / "beat_qaoa_12h" / "train_packs"
METHODS = ("fourier", "snap")
TRAIN_NS = (6, 9)
TRAIN_LO, TRAIN_HI = 0, 200
HOLD_LO, HOLD_HI = 200, 250
FREEZE_B = {12: 128, 15: 64}
N18_B = 32
N18_BASE = 61_000_000_000
DEPTHS = (2, 4)
SEEDS = (0, 1, 2, 3, 4)
PROTOCOL = "seal_NEW_SAFE_lam075_mu0_n18"
# n=12 anchor at this same point. TF-only (PR #45) used this batch stream.
# Bakeoff #46 rounded the same cell and is the paper-lock source.
ANCHOR = {
    "lambda_med": 0.7477,
    "gap_med": 0.1265,
    "peak_med": 0.0379,
    "R2": 0.6209,
    "R4": 0.8881,
    "bakeoff_lambda12": 0.748,
    "bakeoff_lambda18": 0.749,
    "bakeoff_R2": 0.620,
    "bakeoff_R4": 0.869,
    "old_safe_lambda": 0.941,
}


def log(*a):
    print(time.strftime("%H:%M:%S"), "[new-safe-seal]", *a, flush=True)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(path)


def ratio(num: float, den: float) -> float:
    if den == 0 or not math.isfinite(den) or not math.isfinite(num):
        return float("inf")
    return float(num) / float(den)


def pin_check() -> None:
    """The seal Hamiltonian is diag(unsat) − 0.75 ΣX, with the XX coefficient off."""
    if abs(LAMBDA_X + LAM) > 1e-15 or abs(MU) > 1e-15:
        raise RuntimeError(f"pin LAMBDA_X={LAMBDA_X} MU={MU}")
    rng = np.random.default_rng(0)
    E = rng.integers(0, 5, size=1 << 6).astype(np.float64)
    bundle = ground_bundle(E, 6, lambda_x=LAMBDA_X, mu_xx=MU)
    H = dense_H(E, 6, lambda_x=LAMBDA_X, mu_xx=0.0)
    e_dense = float(np.linalg.eigvalsh(H)[0])
    if abs(bundle["E0"] - e_dense) > 1e-8:
        raise RuntimeError(f"pin E0 {bundle['E0']} != dense {e_dense}")
    H_xx = dense_H(E, 6, lambda_x=LAMBDA_X, mu_xx=-0.50)
    if np.allclose(H, H_xx):
        raise RuntimeError("μ=0 Hamiltonian still contains the SAFE XX term")
    psi = torch.tensor(bundle["psi"], dtype=torch.complex128).reshape(1, -1)
    Et = torch.tensor(E, dtype=torch.float64).reshape(1, -1)
    h = float(expectation_H(psi, Et, 6, lambda_x=LAMBDA_X, mu_xx=0.0))
    if abs(h - bundle["E0"]) > 1e-6:
        raise RuntimeError(f"readout ⟨H⟩ {h} != E0 {bundle['E0']}")


def load_unsat(n: int, sl: slice) -> tuple[np.ndarray, np.ndarray]:
    path = PACK / f"train_n{n}.npz"
    d = np.load(path)
    E = np.asarray(d["train_E"][sl], dtype=np.float64)
    seeds = np.asarray(d["train_seeds"][sl]).astype(np.int64)
    if E.ndim != 2 or E.shape[1] != (1 << n):
        raise RuntimeError(f"n{n} E shape {E.shape}")
    return E, seeds


def build_n18(count: int = N18_B) -> tuple[np.ndarray, np.ndarray]:
    """Satisfiable 4-SAT. Same base as the earlier n=18 probes, so the first rows match."""
    from explore_OCT29.lib.sat_gen import instances

    cache = RUN / f"pack_n18_B{count}.npz"
    if cache.exists():
        z = np.load(cache)
        E = np.asarray(z["E"], dtype=np.float64)
        seeds = np.asarray(z["seeds"]).astype(np.int64)
        if E.shape == (count, 1 << 18) and seeds.shape == (count,) and int(z["base"]) == N18_BASE:
            log(f"n18 pack cache hit B={count}")
            return E, seeds
    log(f"generating n18 sat pack B={count} base={N18_BASE}")
    t0 = time.time()
    recs = instances(18, count, N18_BASE, compute_E=True)
    seeds = np.asarray([int(r["seed"]) for r in recs], dtype=np.int64)
    E = np.stack([np.asarray(r["E"], dtype=np.float64) for r in recs], axis=0)
    if E.shape != (count, 1 << 18):
        raise RuntimeError(f"n18 pack shape {E.shape}")
    if np.any(E.min(axis=1) != 0):
        raise RuntimeError("n18 pack contains an unsatisfiable formula")
    RUN.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, E=E.astype(np.int16), seeds=seeds, base=np.int64(N18_BASE), n=np.int64(18))
    log(f"n18 pack written in {time.time() - t0:.1f}s seeds {int(seeds[0])}..{int(seeds[-1])}")
    return E, seeds


def _diag_job(payload: tuple) -> tuple:
    row, n, seed = payload
    bundle = ground_bundle(np.asarray(row, dtype=np.float64), int(n), lambda_x=LAMBDA_X, mu_xx=0.0)
    lam_c = np.asarray(lambda_of_state(bundle["psi"], int(n)), dtype=np.float64).reshape(-1)
    if lam_c.size != int(n) // 3:
        raise RuntimeError(f"Λ shape {lam_c.shape} at n={n}")
    e_prod = best_product_energy(row, int(n), LAM, 0.0, seed=int(seed))
    gap = product_gap(e_prod, bundle["E0"])
    if e_prod + 1e-5 < float(bundle["E0"]):
        raise RuntimeError(f"product energy {e_prod:.6f} below E0 {bundle['E0']:.6f} at n={n}")
    return (
        float(bundle["E0"]),
        float(bundle["peak"]),
        float(bundle["residual"]),
        lam_c,
        float(e_prod),
        float(gap),
    )


def _pool_init() -> None:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    torch.set_num_threads(1)


def diagnostics(n: int, role: str, E: np.ndarray, seeds: np.ndarray, workers: int) -> dict:
    """E0, Z-peak, per-cavity Λ, and the XZ product gap. Cached."""
    cache = RUN / f"diag_n{n}_{role}.npz"
    if cache.exists():
        z = np.load(cache)
        same = (
            np.array_equal(z["seeds"], seeds)
            and int(z["e0"].shape[0]) == int(E.shape[0])
            and abs(float(z["lam"]) - LAM) < 1e-12
            and abs(float(z["mu"])) < 1e-12
        )
        if same:
            log(f"diag cache hit n{n} {role}")
            return {k: np.asarray(z[k]) for k in ("e0", "peak", "residual", "lambda_c", "e_prod", "prod_gap", "seeds")}
    log(f"diag n{n} {role} B={E.shape[0]} workers={workers}")
    t0 = time.time()
    jobs = [(E[i], n, int(seeds[i])) for i in range(E.shape[0])]
    if workers <= 1 or len(jobs) == 1 or n <= 9:
        rows = []
        for i, job in enumerate(jobs):
            rows.append(_diag_job(job))
            if (i + 1) == len(jobs) or (i + 1) % 8 == 0:
                log(
                    f"diag n{n} {role} {i + 1}/{len(jobs)} "
                    f"E0={rows[-1][0]:.4f} Λmed={float(np.median(rows[-1][3])):.3f} "
                    f"elapsed={time.time() - t0:.0f}s"
                )
    else:
        import multiprocessing as mp

        ctx = mp.get_context("spawn")
        rows = []
        with ctx.Pool(int(workers), initializer=_pool_init) as pool:
            for i, row in enumerate(pool.imap(_diag_job, jobs, chunksize=1)):
                rows.append(row)
                if (i + 1) == len(jobs) or (i + 1) % 4 == 0:
                    log(
                        f"diag n{n} {role} {i + 1}/{len(jobs)} "
                        f"E0={row[0]:.4f} Λmed={float(np.median(row[3])):.3f} "
                        f"elapsed={time.time() - t0:.0f}s"
                    )
    e0 = np.array([r[0] for r in rows], dtype=np.float64)
    peak = np.array([r[1] for r in rows], dtype=np.float64)
    residual = np.array([r[2] for r in rows], dtype=np.float64)
    lam_c = np.stack([r[3] for r in rows], axis=0)
    e_prod = np.array([r[4] for r in rows], dtype=np.float64)
    gap = np.array([r[5] for r in rows], dtype=np.float64)
    if np.any(e0 > E.min(axis=1) + 1e-6):
        raise RuntimeError(f"E0 above a diagonal entry at n={n} {role}")
    if np.any(residual > 1e-5):
        raise RuntimeError(f"ground residual {float(residual.max())} at n={n} {role}")
    if np.any(gap < -1e-6):
        raise RuntimeError(f"negative product gap {float(gap.min())} at n={n} {role}")
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
        lam=np.float64(LAM),
        mu=np.float64(0.0),
    )
    log(
        f"diag n{n} {role} {time.time() - t0:.1f}s  "
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
        "frac_gt_080": float(np.mean(lam_c > LAMBDA_PASS)),
        "frac_ge_090": float(np.mean(lam_c >= LAMBDA_KILL - 1e-12)),
        "gap_med": float(np.median(gap)),
        "gap_min": float(np.min(gap)),
        "frac_gap_ge_003": float(np.mean(gap + 1e-12 >= GAP_PASS)),
        "peak_med": float(np.median(peak)),
        "peak_max": float(np.max(peak)),
        "frac_peak_ge_05": float(np.mean(peak + 1e-12 >= 0.5)),
        "e0_med": float(np.median(e0)),
        "resid_max": float(np.max(rec["residual"])),
    }


def _workers_for(n: int, requested: int) -> int:
    if int(n) >= 18:
        return max(1, min(int(requested), 2))
    if int(n) >= 15:
        return max(1, min(int(requested), 3))
    return max(1, int(requested))


def seed_path(L: int, method: str, seed: int) -> Path:
    return RUN / f"L{L}_{method}_seed{seed}.json"


def _as_f64(angles: dict) -> dict:
    return {k: (v.detach().to(dtype=torch.float64) if torch.is_tensor(v) else v) for k, v in angles.items()}


def _chunk_for(n: int) -> int:
    n = int(n)
    if n >= 18:
        return 1
    if n >= 15:
        return 2
    if n >= 12:
        return 4
    return 16


def eval_method(method: str, angles: dict, bundle: dict, *, with_lambda: bool) -> dict:
    ang = _as_f64(angles)
    out = {}
    for n, (E, e0, seeds) in bundle.items():
        use = _chunk_for(int(n))
        Eu = torch.tensor(E, dtype=torch.float64)
        e0t = torch.tensor(e0, dtype=torch.float64)
        eps_parts = []
        lam_parts = []
        B = int(Eu.shape[0])
        for s in range(0, B, use):
            sl = slice(s, s + use)
            Cn = cn_from_unsat(Eu[sl], int(n))
            psi = prepare_state(method, Cn, ang)
            h = expectation_H(psi, Eu[sl], int(n), lambda_x=LAMBDA_X, mu_xx=0.0)
            eps_parts.append((h - e0t[sl]).detach().cpu().numpy())
            if with_lambda:
                pX = _prob(apply_H_all(psi, int(n)))
                lam_parts.append(cavity_lambda_from_px(pX, int(n)))
        eps = np.concatenate(eps_parts, axis=0)
        rec = summarize_eps(eps)
        rec.update(
            {
                "n": int(n),
                "seeds": [int(x) for x in np.asarray(seeds).tolist()],
                "eps": [float(x) for x in eps.tolist()],
                "e0_med": float(np.median(e0)),
            }
        )
        if with_lambda:
            lam = np.concatenate(lam_parts, axis=0)
            rec["prep_lambda_med"] = float(np.median(lam))
            rec["prep_lambda_mean"] = float(np.mean(lam))
        out[str(n)] = rec
    return out


def _loss_on(method, angles, Eu, e0b, n: int, comps) -> torch.Tensor:
    if method == "snap":
        psi = snap_state_from_components(comps, angles["gamma_w"], angles["phi"])
    else:
        Cn = cn_from_unsat(Eu, n)
        psi = prepare_state(method, Cn, angles)
    h = expectation_H(psi, Eu, n, lambda_x=LAMBDA_X, mu_xx=0.0)
    return ((h - e0b) / float(n)).mean()


def math_finite(x: float) -> bool:
    return x == x and abs(x) != float("inf")


def train_one(method: str, L: int, seed: int, bundles_t: dict, hold_bundle: dict, *, rounds: int, batch: int) -> dict:
    # Same stream as the Round11 TF-only trainer at λ=0.75, so a fresh run
    # follows that batch schedule (and the n=12 anchor from PR #45).
    torch.manual_seed(int(seed) + 1009 * int(round(100 * LAM)) + 17 * int(L))
    rng = np.random.default_rng(int(seed) + 10007 * int(round(1000 * LAM)) + int(L))
    angles = init_angles(method, L, dtype=torch.float64)
    params = method_parameter_list(method, angles)
    E6, e06, _s6 = bundles_t[6]
    E9, e09, _s9 = bundles_t[9]
    n6 = int(E6.shape[0])
    n9 = int(E9.shape[0])
    b6 = min(batch // 2, n6)
    b9 = min(batch - b6, n9)

    def hold_med() -> float:
        ev = eval_method(method, angles, hold_bundle, with_lambda=False)
        return float(ev["9"]["med_eps"])

    t0 = time.time()
    init_med = hold_med()
    best = {
        "med9": init_med,
        "round": -1,
        "angles": {k: v.detach().cpu().clone() for k, v in angles.items()},
    }
    hist = [{"round": -1, "hold_med9": init_med, "loss": None}]
    log(f"L{L} {method} s{seed} init n9-hold Medε={init_med:.5f}")

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
                _loss_on(method, angles, Eu6, e6b, 6, comps6)
                + _loss_on(method, angles, Eu9, e9b, 9, comps9)
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            return loss

        try:
            loss = opt.step(closure)
            loss_f = float(loss.detach()) if torch.is_tensor(loss) else float(loss)
        except Exception as e:
            log(f"L{L} {method} s{seed} r{r} LBFGS skip: {e}")
            hist.append({"round": r, "skip": str(e)})
            continue
        if not math_finite(loss_f):
            log(f"L{L} {method} s{seed} r{r} nonfinite loss — restore best")
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
                f"L{L} {method} s{seed} r{r}/{rounds} loss={loss_f:.5f} "
                f"holdMedε={med:.5f} best={best['med9']:.5f}@r{best['round']}{mark}"
            )

    with torch.no_grad():
        for k, v in best["angles"].items():
            angles[k].data.copy_(v.to(dtype=angles[k].dtype))
    return {
        "method": method,
        "L": int(L),
        "seed": int(seed),
        "init_hold_med9": init_med,
        "best_round": int(best["round"]),
        "best_hold_med9": float(best["med9"]),
        "wall_s": time.time() - t0,
        "hist": hist,
        "angles": best["angles"],
        "n_params": int(sum(p.numel() for p in params)),
    }


def _fmt(x, nd: int = 4) -> str:
    if x is None or not math.isfinite(float(x)):
        return "—"
    return f"{float(x):.{nd}f}"


def _load_rows(depths: list[int], seeds: list[int]) -> list[dict]:
    rows = []
    for L in depths:
        for method in METHODS:
            for seed in seeds:
                path = seed_path(L, method, seed)
                if not path.exists():
                    raise RuntimeError(f"missing {path}")
                rows.append(json.loads(path.read_text()))
    return rows


def _paired(snap_eps, other_eps) -> tuple[int, int]:
    a = np.asarray(snap_eps, dtype=np.float64)
    b = np.asarray(other_eps, dtype=np.float64)
    return int(np.sum(a < b - 1e-12)), int(a.size)


def aggregate(depths: list[int], seeds: list[int], diag_by_n: dict) -> dict:
    rows = _load_rows(depths, seeds)
    by_depth = {}
    for L in depths:
        block = [r for r in rows if int(r["L"]) == int(L)]
        cell = {}
        for method in METHODS:
            sub = [r for r in block if r["method"] == method]
            cell[method] = {}
            for n in ("6", "9", "12", "15", "18"):
                cell[method][n] = float(np.median([float(r["eval"][n]["med_eps"]) for r in sub]))
        rf = {n: ratio(cell["snap"][n], cell["fourier"][n]) for n in ("6", "9", "12", "15", "18")}
        seed_rows = []
        paired = {n: [0, 0] for n in ("12", "15", "18")}
        for seed in seeds:
            snap = next(r for r in block if r["method"] == "snap" and int(r["seed"]) == int(seed))
            four = next(r for r in block if r["method"] == "fourier" and int(r["seed"]) == int(seed))
            r_by_n = {
                n: ratio(snap["eval"][n]["med_eps"], four["eval"][n]["med_eps"]) for n in ("12", "15", "18")
            }
            wins = {}
            for n in ("12", "15", "18"):
                w, b = _paired(snap["eval"][n]["eps"], four["eval"][n]["eps"])
                wins[n] = (w, b)
                paired[n][0] += w
                paired[n][1] += b
            phi = snap["angles"].get("phi")
            seed_rows.append(
                {
                    "seed": int(seed),
                    "snap12": float(snap["eval"]["12"]["med_eps"]),
                    "fourier12": float(four["eval"]["12"]["med_eps"]),
                    "R12": r_by_n["12"],
                    "R15": r_by_n["15"],
                    "R18": r_by_n["18"],
                    "beat12": bool(r_by_n["12"] <= R_BEAT + 1e-12),
                    "paired12": wins["12"][0],
                    "B12": wins["12"][1],
                    "paired18": wins["18"][0],
                    "B18": wins["18"][1],
                    "shell_rms": None if phi is None else shell_rms(phi),
                    "prep_lambda_snap12": snap["eval"]["12"].get("prep_lambda_med"),
                    "prep_lambda_fourier12": four["eval"]["12"].get("prep_lambda_med"),
                    "prep_lambda_snap18": snap["eval"]["18"].get("prep_lambda_med"),
                    "prep_lambda_fourier18": four["eval"]["18"].get("prep_lambda_med"),
                    "best_round_snap": snap["best_round"],
                    "best_round_fourier": four["best_round"],
                    "wall_snap_s": snap["wall_s"],
                    "wall_fourier_s": four["wall_s"],
                }
            )
        shells = [s["shell_rms"] for s in seed_rows if s["shell_rms"] is not None]
        circ = {}
        for method in METHODS:
            sub = [r for r in block if r["method"] == method]
            circ[method] = {
                n: float(np.median([r["eval"][n]["prep_lambda_med"] for r in sub])) for n in ("12", "15", "18")
            }
        by_depth[str(L)] = {
            "med_eps": cell,
            "R_fourier": rf,
            "steps": {
                "12_to_15": rf["15"] - rf["12"],
                "15_to_18": rf["18"] - rf["15"],
            },
            "seeds": seed_rows,
            "n_beat12": int(sum(1 for row in seed_rows if row["beat12"])),
            "paired": {n: {"wins": paired[n][0], "B": paired[n][1]} for n in paired},
            "shell_rms_med": float(np.median(shells)) if shells else None,
            "circuit_lambda_med": circ,
        }

    gs = {str(n): diag_summary(diag_by_n[n], n, "freeze" if n >= 12 else "other") for n in (12, 15, 18)}
    d2 = by_depth["2"]
    gate = classify_seal(
        d2["R_fourier"]["12"],
        d2["R_fourier"]["15"],
        d2["R_fourier"]["18"],
        d2["n_beat12"],
        gs["12"]["lambda_med"],
        gs["18"]["lambda_med"],
        gs["12"]["gap_med"],
        gs["18"]["gap_med"],
        n_seeds=len(seeds),
    )
    if len(seeds) != N_SEEDS or set(seeds) != set(SEEDS) or set(depths) != set(DEPTHS):
        gate = {
            **gate,
            "verdict": "incomplete",
            "reasons": ["partial depths or seeds; not a seal call"],
        }
    l4 = by_depth.get("4")
    l4_note = None
    if l4 is not None:
        r4 = l4["R_fourier"]["12"]
        l4_note = {
            "n12_R": r4,
            "soft": bool(r4 > R_PASS + 1e-12),
            "text": (
                f"n12 L=4 R_Fourier={r4:.4f} is above 0.85, reported as a soft lead and not required for PASS"
                if r4 > R_PASS + 1e-12
                else f"n12 L=4 R_Fourier={r4:.4f} is at or under 0.85; L=4 is still not a PASS bar"
            ),
        }
    summary = {
        "protocol": PROTOCOL,
        "lambda": LAM,
        "mu": MU,
        "lambda_x": LAMBDA_X,
        "verdict": gate["verdict"],
        "reasons": gate["reasons"],
        "kill": gate.get("kill", []),
        "soft": gate.get("soft", []),
        "gate": gate,
        "l4_note": l4_note,
        "ground_state": gs,
        "depths": by_depth,
        "anchor_pr45_n12": ANCHOR,
        "freeze_B": {"12": FREEZE_B[12], "15": FREEZE_B[15], "18": N18_B},
        "n18_base": N18_BASE,
    }
    write_json(RUN / "summary.json", summary)
    text = render(summary)
    (RUN / "RESULTS.md").write_text(text)
    log(f"wrote {RUN / 'RESULTS.md'}")
    return summary


def render(summary: dict) -> str:
    verdict = summary["verdict"]
    reasons = summary["reasons"]
    reason_txt = "; ".join(reasons) if reasons else "All four PASS bars hold"
    gs = summary["ground_state"]
    d2 = summary["depths"]["2"]
    d4 = summary["depths"].get("4")
    q12, q15, q18 = gs["12"], gs["15"], gs["18"]
    lines = [
        "# NEW_SAFE λ=0.75, μ=0 shallow seal — Walsh SNAP vs Fourier through n=18",
        "",
        f"**Verdict: {verdict}.** {reason_txt}.",
        "",
        "Paper-lock from bakeoff #46, cell C: λ=0.75, μ=0, "
        "H = diag(unsat) − 0.75 ΣX. That bakeoff reported L=2 R_Fourier=0.620 (5/5), "
        "Λ12=0.748, Λ18=0.749, with L=4 a soft lead near 0.87. "
        "The TF-only seal (PR #45) at this λ, on the same batch stream as this run, "
        f"had n=12 Λ={ANCHOR['lambda_med']:.4f}, gap={ANCHOR['gap_med']:.4f}, "
        f"L=2 R={ANCHOR['R2']:.4f}, L=4 R={ANCHOR['R4']:.4f}. "
        "Old SAFE (λ=0.75, μ=−0.50) is Λ-KILL appendix-only: median Λ≈0.94 (PR #41). "
        "The question here is whether the NEW_SAFE lead survives the PR #47 five-seed "
        "protocol through n=15 and n=18 without the cavity lock drifting up.",
        "",
        "## Gate",
        "",
        "Med ε is the median over seeds of each seed's median-over-instances excess ⟨H⟩−E₀. "
        "R_Fourier = Medε_SNAP / Medε_Fourier. A seed BEATs when its own n=12 L=2 R_Fourier ≤ 0.98. "
        "Λ_c = ⟨(Σ_{i∈c} X_i / 3)^2⟩ from the X histogram of the true ground state. The reported Λ "
        "is the median over cavities and instances. The product gap is (E_prod − E_0)/|E_0| for the "
        "best real XZ product state.",
        "",
        "- **PASS** when all four hold: (1) n=12 L=2 R_Fourier ≤ 0.85 and at least 4/5 seeds BEAT; "
        "(2) median Λ ≤ 0.80 at n=12 and at n=18; (3) median product gap ≥ 0.03 at n=12 and at n=18; "
        "(4) L=2 R_Fourier rises by at most 0.05 on each +3-qubit step n=12→15 and n=15→18.",
        "- **needs_edit** when nothing kills the point and at least one PASS bar is missed. "
        "Λ drifting above 0.80 (and still under 0.90) is this case.",
        "- **KILL** when n=18 median Λ ≥ 0.90, or when L=2 R_Fourier > 0.98 at n=12, n=15, or n=18.",
        "- **L=4** is reported and is not a PASS bar. A ratio above 0.85 is noted as soft "
        "(bakeoff #46 n=12 L=4 was about 0.87; the TF-only stream was 0.888).",
        "",
        "## Hamiltonian",
        "",
        "\\[",
        "H = \\mathrm{diag}(\\mathrm{unsat}) - 0.75 \\sum_i X_i",
        "\\]",
        "",
        "μ = 0, so there is no intra-cavity XX term. Cavity c owns qubits [3c, 3c+3), qubit 0 = MSB. "
        "The circuit phases Cn = unsat · n/m. ⟨H⟩ uses the raw unsat count from the Z histogram and "
        "⟨X_i⟩ from the X histogram. No Y, and no XX correlator in the energy. The mixer is BinH "
        "Hamming-shell SNAP with a Walsh-graded cost. Fourier is the two-angle QAOA. HVA is not trained: "
        "β^XX = 0 is Fourier.",
        "",
        "## Protocol",
        "",
        "- Packs: `beat_qaoa_12h/train_packs`. Train indices [0:200) on n=6 and n=9. "
        "Checkpoint on n=9 holdout [200:250). Freeze n=12 first 128. Probe n=15 first 64.",
        f"- n=18: {N18_B} satisfiable formulas from `explore_OCT29.lib.sat_gen.instances` at base {N18_BASE}. "
        "There is no n=18 train pack. These instances are not in the loss and not in checkpoint selection.",
        "- L-BFGS matches `Nov_3rd.settings` and the PR #47 shallow seal: 40 rounds × max_iter 8, "
        "lr 0.25, history 16, strong Wolfe, grad clip 5, batch 64 (half n6 / half n9). New optimizer "
        "each round. Loss = mean ε/n. Each seed restores the round with the best n=9-holdout Med ε. "
        "The batch stream matches the Round11 TF-only trainer at λ=0.75.",
        "- Paper init β=0.01, γ=γ_w=−0.01, φ_k=−β k. Fourier and SNAP start on the same state.",
        "- L≥8 was not trained. n=21 was not loaded. `qaoa_angles/`, `train69_ladder_fresh250/`, "
        "and `settings.py` were not written. Old SAFE (0.75, −0.50) was not re-run.",
        "",
        "## Ground state: Λ and product gap",
        "",
        "Λ→1 means every cavity sits in a single X-magnetization sector, so the 3-qubit cavity has "
        "collapsed to an effective spin. A Z-basis product state scores Λ=1/3. The pass line is 0.80 "
        "and the kill line is 0.90 at n=18. Old SAFE μ=−0.50 sat near 0.94.",
        "",
        "| n | role | B | median Λ | min Λ | max Λ | fraction Λ>0.80 | fraction Λ≥0.90 | median product gap | min gap | fraction gap≥0.03 | median Z-peak |",
        "|--:|:-----|--:|--------:|------:|------:|----------------:|----------------:|-------------------:|--------:|------------------:|-------------:|",
    ]
    roles = {"12": "freeze (gate)", "15": "probe", "18": "probe (gate)"}
    for key in ("12", "15", "18"):
        q = gs[key]
        lines.append(
            f"| {key} | {roles[key]} | {q['B']} | {q['lambda_med']:.4f} | {q['lambda_min']:.4f} | "
            f"{q['lambda_max']:.4f} | {q['frac_gt_080']:.3f} | {q['frac_ge_090']:.3f} | "
            f"{q['gap_med']:.4f} | {q['gap_min']:.4f} | {q['frac_gap_ge_003']:.3f} | {q['peak_med']:.4f} |"
        )
    lines += [
        "",
        f"n=12 median Λ = **{q12['lambda_med']:.4f}**. n=15 median Λ = **{q15['lambda_med']:.4f}**. "
        f"n=18 median Λ = **{q18['lambda_med']:.4f}**. "
        f"n=12 median product gap = **{q12['gap_med']:.4f}** "
        f"(n=15 {q15['gap_med']:.4f}, n=18 {q18['gap_med']:.4f}). "
        f"The n=12 Z-basis peak of the ground state is {q12['peak_med']:.4f} "
        f"(n=15 {q15['peak_med']:.4f}, n=18 {q18['peak_med']:.4f}). "
        f"Max ground-state residual is {max(q12['resid_max'], q15['resid_max'], q18['resid_max']):.3e}.",
        "",
    ]
    lines += _depth_section("L=2", d2, gate=True)
    if d4 is not None:
        lines += _depth_section("L=4", d4, gate=False)
        note = summary.get("l4_note") or {}
        lines += [
            f"L=4 note: {note.get('text', '')}.",
            "",
        ]
    lines += [
        "## Reading",
        "",
        f"The call is **{verdict}**. {reason_txt}.",
        "",
        _reading_body(summary),
        "",
        "Per-seed JSON: `L{L}_{method}_seed{s}.json` in this folder. Angles are the restored n=9-holdout "
        "checkpoint. Ground-state Λ, the product gap, and the Z-peak are in `diag_n{n}_{role}.npz`.",
        "",
    ]
    return "\n".join(lines) + "\n"


def _depth_section(title: str, block: dict, *, gate: bool) -> list[str]:
    role = {
        "6": "holdout",
        "9": "holdout (select)",
        "12": "freeze (gate)" if gate else "freeze",
        "15": "probe",
        "18": "probe",
    }
    lines = [
        f"## {title}",
        "",
        "| n | role | Medε Fourier | Medε SNAP | R_Fourier |",
        "|--:|:-----|-------------:|----------:|----------:|",
    ]
    med = block["med_eps"]
    rf = block["R_fourier"]
    for n in ("6", "9", "12", "15", "18"):
        lines.append(
            f"| {n} | {role[n]} | {med['fourier'][n]:.4f} | {med['snap'][n]:.4f} | {rf[n]:.4f} |"
        )
    steps = block["steps"]
    paired = block["paired"]
    lines += [
        "",
        f"**{title} n12 R_Fourier = {rf['12']:.4f}** (SNAP {med['snap']['12']:.4f} / Fourier {med['fourier']['12']:.4f}). "
        f"Seeds BEAT (n12 R≤0.98): {block['n_beat12']}/5. "
        f"Probe R is {rf['15']:.4f} at n=15 and {rf['18']:.4f} at n=18. "
        f"Steps of R: n12→n15 {steps['12_to_15']:+.4f}, n15→n18 {steps['15_to_18']:+.4f}.",
        "",
        f"Paired instances with SNAP ε below Fourier: "
        f"n12 {paired['12']['wins']}/{paired['12']['B']}, "
        f"n15 {paired['15']['wins']}/{paired['15']['B']}, "
        f"n18 {paired['18']['wins']}/{paired['18']['B']}. "
        f"Median shell RMS (distance of φ_k from the best line a+bk) = {_fmt(block['shell_rms_med'])}.",
        "",
        "| seed | n12 SNAP | n12 Fourier | n12 R | BEAT | paired n12 | n15 R | n18 R | shell RMS | SNAP best round | wall s |",
        "|---:|---:|---:|---:|:----:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in block["seeds"]:
        beat = "yes" if row["beat12"] else "no"
        lines.append(
            f"| {row['seed']} | {row['snap12']:.4f} | {row['fourier12']:.4f} | {row['R12']:.4f} | {beat} | "
            f"{row['paired12']}/{row['B12']} | {row['R15']:.4f} | {row['R18']:.4f} | {_fmt(row['shell_rms'])} | "
            f"{row['best_round_snap']} | {row['wall_snap_s']:.0f} |"
        )
    circ = block["circuit_lambda_med"]
    lines += [
        "",
        "Circuit-state median Λ (X histogram of the prepared state, median over seeds of each seed's pooled median) "
        f"at n12: Fourier {_fmt(circ['fourier']['12'], 3)}, SNAP {_fmt(circ['snap']['12'], 3)}. "
        f"At n18: Fourier {_fmt(circ['fourier']['18'], 3)}, SNAP {_fmt(circ['snap']['18'], 3)}. "
        "The gate uses the ground-state Λ, not these.",
        "",
    ]
    return lines


def _anchor_match(q12: dict, r2: float, r4, q18: dict) -> str:
    """Say whether this freeze lands on the TF-only n=12 anchor and the bakeoff n=18 Λ."""
    dlam = abs(float(q12["lambda_med"]) - ANCHOR["lambda_med"])
    dgap = abs(float(q12["gap_med"]) - ANCHOR["gap_med"])
    dr2 = abs(float(r2) - ANCHOR["R2"])
    if dlam <= 0.001 and dgap <= 0.001 and dr2 <= 0.002:
        head = (
            f"The n=12 freeze reproduces that anchor "
            f"(ΔΛ={dlam:.4f}, Δgap={dgap:.4f}, ΔL=2 R={dr2:.4f})."
        )
    else:
        head = (
            f"The n=12 freeze differs from that anchor by "
            f"ΔΛ={float(q12['lambda_med']) - ANCHOR['lambda_med']:+.4f}, "
            f"Δgap={float(q12['gap_med']) - ANCHOR['gap_med']:+.4f}, "
            f"ΔL=2 R={float(r2) - ANCHOR['R2']:+.4f}."
        )
    d18 = float(q18["lambda_med"]) - ANCHOR["bakeoff_lambda18"]
    tail = (
        f" n=18 median Λ is {float(q18['lambda_med']):.4f} "
        f"({d18:+.4f} versus the bakeoff #46 probe of {ANCHOR['bakeoff_lambda18']:.3f})."
    )
    if r4 is not None:
        tail += (
            f" n=12 L=4 R is {float(r4):.4f} against the TF-only anchor {ANCHOR['R4']:.4f} "
            f"and the bakeoff prior {ANCHOR['bakeoff_R4']:.3f}."
        )
    return head + tail


def _reading_body(summary: dict) -> str:
    gs = summary["ground_state"]
    d2 = summary["depths"]["2"]
    d4 = summary["depths"].get("4")
    q12, q15, q18 = gs["12"], gs["15"], gs["18"]
    rf = d2["R_fourier"]
    steps = d2["steps"]
    parts = [
        (
            f"At the n=12 freeze, L=2 R_Fourier={rf['12']:.4f} "
            f"(SNAP {d2['med_eps']['snap']['12']:.4f}, Fourier {d2['med_eps']['fourier']['12']:.4f}) "
            f"with {d2['n_beat12']}/5 seeds at or under the 0.98 BEAT line. "
            f"The same ratio is {rf['15']:.4f} at n=15 and {rf['18']:.4f} at n=18. "
            f"The +3-qubit steps are {steps['12_to_15']:+.4f} and {steps['15_to_18']:+.4f} "
            f"(bar ≤ +0.05 on each step)."
        ),
        (
            f"Ground-state median Λ is {q12['lambda_med']:.4f} / {q15['lambda_med']:.4f} / {q18['lambda_med']:.4f} "
            f"at n=12 / 15 / 18. "
            f"The fraction of cavities above 0.80 is {q12['frac_gt_080']:.3f} / {q15['frac_gt_080']:.3f} / {q18['frac_gt_080']:.3f}. "
            f"The fraction at or above 0.90 is {q12['frac_ge_090']:.3f} / {q15['frac_ge_090']:.3f} / {q18['frac_ge_090']:.3f}. "
            f"The product gap is {q12['gap_med']:.4f} / {q15['gap_med']:.4f} / {q18['gap_med']:.4f} "
            f"(bar ≥ 0.03 at n=12 and n=18). "
            f"Z-basis peaks are {q12['peak_med']:.4f} / {q15['peak_med']:.4f} / {q18['peak_med']:.4f}."
        ),
        (
            "Λ equals 1/3 + (2/9) Σ_{i<j∈c} ⟨X_i X_j⟩, computed from the joint X histogram of the ground state. "
            "Values near 1 mean the intra-cavity space has collapsed to an effective spin, which is what killed "
            f"old SAFE μ=−0.50 (Λ≈{ANCHOR['old_safe_lambda']:.3f}). "
            f"The TF-only n=12 anchor on this batch stream was Λ={ANCHOR['lambda_med']:.4f}, "
            f"gap={ANCHOR['gap_med']:.4f}, Z-peak={ANCHOR['peak_med']:.4f}, "
            f"L=2 R={ANCHOR['R2']:.4f}, L=4 R={ANCHOR['R4']:.4f}. "
            f"Bakeoff #46 reported Λ12={ANCHOR['bakeoff_lambda12']:.3f}, "
            f"Λ18={ANCHOR['bakeoff_lambda18']:.3f}, L=2 R={ANCHOR['bakeoff_R2']:.3f}, "
            f"L=4 R≈{ANCHOR['bakeoff_R4']:.3f}. "
            + _anchor_match(q12, rf['12'], None if d4 is None else d4["R_fourier"]["12"], q18)
        ),
    ]
    if d4 is not None:
        r4 = d4["R_fourier"]
        tail = [
            row
            for row in d4["seeds"]
            if row["R12"] > 0.95 or row["R15"] > 0.95 or row["R18"] > 0.95
        ]
        tail_txt = ""
        if tail:
            bits = [
                f"seed {row['seed']} n12/n15/n18 R={row['R12']:.4f}/{row['R15']:.4f}/{row['R18']:.4f}"
                for row in tail
            ]
            tail_txt = " The soft tail is " + "; ".join(bits) + "."
        parts.append(
            f"L=4 R_Fourier is {r4['12']:.4f} / {r4['15']:.4f} / {r4['18']:.4f} at n=12 / 15 / 18. "
            f"{summary['l4_note']['text']}.{tail_txt}"
        )
    parts.append(
        f"L=2 shell RMS median is {_fmt(d2['shell_rms_med'])}"
        + (f"; L=4 shell RMS median is {_fmt(d4['shell_rms_med'])}." if d4 is not None else ".")
        + " A small L=2 shell RMS means the Hamming shell stays close to a linear φ_k = a + bk, so part of the "
        "L=2 lead over Fourier is the Walsh-graded cost (four γ weights per layer against Fourier's one)."
    )
    return "\n\n".join(parts)


def main():
    decision_self_test()
    if 21 in TRAIN_NS or 21 in FREEZE_B:
        raise SystemExit("n21 is out of scope")
    ap = argparse.ArgumentParser()
    ap.add_argument("--depths", type=str, default="2,4")
    ap.add_argument("--seeds", type=str, default="0,1,2,3,4")
    ap.add_argument("--rounds", type=int, default=S.LBFGS_ROUNDS)
    ap.add_argument("--batch", type=int, default=S.LBFGS_BATCH)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--aggregate-only", action="store_true")
    ap.add_argument("--diag-only", action="store_true")
    ap.add_argument("--skip-self-check", action="store_true")
    args = ap.parse_args()

    depths = [int(x) for x in args.depths.split(",") if x.strip()]
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    bad = [L for L in depths if L not in DEPTHS]
    if bad:
        raise SystemExit(f"refusing depths {bad}: this seal is L in {{2,4}} only")
    if any(s < 0 or s > 4 for s in seeds):
        raise SystemExit("seeds must be in 0..4")
    if args.rounds != S.LBFGS_ROUNDS or args.batch != S.LBFGS_BATCH:
        log(f"non-default budget rounds={args.rounds} batch={args.batch}")
    RUN.mkdir(parents=True, exist_ok=True)

    if not args.skip_self_check and not args.aggregate_only:
        log("self-check")
        self_check()
        pin_check()
        log("self-check ok")

    data_E = {}
    if not args.aggregate_only:
        for n, sl, tag in (
            (6, slice(TRAIN_LO, TRAIN_HI), "train"),
            (9, slice(TRAIN_LO, TRAIN_HI), "train"),
            (6, slice(HOLD_LO, HOLD_HI), "hold"),
            (9, slice(HOLD_LO, HOLD_HI), "hold"),
        ):
            E, sd = load_unsat(n, sl)
            diag = diagnostics(n, tag, E, sd, workers=1)
            data_E[(n, tag)] = (E, diag["e0"], sd)
            log(
                f"n{n} {tag} B={len(E)} medE0={float(np.median(diag['e0'])):.4f} "
                f"Λ={float(np.median(diag['lambda_c'])):.4f} gap={float(np.median(diag['prod_gap'])):.4f}"
            )

    freeze = {}
    for n, b in ((12, FREEZE_B[12]), (15, FREEZE_B[15])):
        E, sd = load_unsat(n, slice(0, b))
        diag = diagnostics(n, "freeze", E, sd, workers=_workers_for(n, args.workers))
        freeze[n] = (E, diag)
    E18, sd18 = build_n18(N18_B)
    diag18 = diagnostics(18, "freeze", E18, sd18, workers=_workers_for(18, args.workers))
    freeze[18] = (E18, diag18)
    diag_by_n = {n: freeze[n][1] for n in (12, 15, 18)}

    if args.aggregate_only or args.diag_only:
        if args.diag_only and not args.aggregate_only:
            for n in (12, 15, 18):
                rec = diag_summary(diag_by_n[n], n, "freeze")
                log(
                    f"GS n{n} Λ={rec['lambda_med']:.4f} gap={rec['gap_med']:.4f} "
                    f"peak={rec['peak_med']:.4f} frac>0.80={rec['frac_gt_080']:.3f}"
                )
            if all(seed_path(L, m, s).exists() for L in depths for m in METHODS for s in seeds):
                summary = aggregate(depths, seeds, diag_by_n)
                log("OVERALL", summary["verdict"])
            return
        summary = aggregate(depths, seeds, diag_by_n)
        log("OVERALL", summary["verdict"])
        return

    pwr = walsh_power_by_degree(freeze[12][0][0])
    hi = float(pwr[5:].sum())
    log(f"Walsh n12 instance0 degree>4 power {hi:.3e}")
    if hi > 1e-8:
        raise RuntimeError(f"n12 cost is not degree ≤4 ({hi})")

    torch.set_num_threads(max(1, min(4, int(args.workers))))
    train_tensors = {}
    for n in TRAIN_NS:
        E, e0, sd = data_E[(n, "train")]
        train_tensors[n] = (
            torch.tensor(E, dtype=torch.float64),
            torch.tensor(e0, dtype=torch.float64),
            sd,
        )
    hold_bundle = {9: data_E[(9, "hold")]}
    eval_bundle = {
        6: data_E[(6, "hold")],
        9: data_E[(9, "hold")],
        12: (freeze[12][0], freeze[12][1]["e0"], freeze[12][1]["seeds"]),
        15: (freeze[15][0], freeze[15][1]["e0"], freeze[15][1]["seeds"]),
        18: (freeze[18][0], freeze[18][1]["e0"], freeze[18][1]["seeds"]),
    }

    for L in depths:
        init_angles_L = {m: init_angles(m, L, dtype=torch.float64) for m in METHODS}
        init_ev = {
            m: eval_method(m, init_angles_L[m], {12: eval_bundle[12]}, with_lambda=False) for m in METHODS
        }
        base = np.asarray(init_ev["fourier"]["12"]["eps"])
        for method in METHODS:
            err = float(np.max(np.abs(np.asarray(init_ev[method]["12"]["eps"]) - base)))
            log(f"L{L} init {method} vs fourier n12 max|Δε|={err:.3e}")
            if err > 1e-6:
                raise RuntimeError(f"init states differ: {method} Δ={err}")

        for method in METHODS:
            for seed in seeds:
                path = seed_path(L, method, seed)
                if path.exists() and not args.force:
                    log(f"skip existing {path.name}")
                    continue
                log(f"=== train L{L} {method} seed{seed} ===")
                trained = train_one(
                    method,
                    L,
                    seed,
                    train_tensors,
                    hold_bundle,
                    rounds=args.rounds,
                    batch=args.batch,
                )
                ev = eval_method(method, trained["angles"], eval_bundle, with_lambda=True)
                phi = trained["angles"].get("phi")
                payload = {
                    "done": True,
                    "protocol": PROTOCOL,
                    "method": method,
                    "L": int(L),
                    "seed": int(seed),
                    "n_params": trained["n_params"],
                    "best_round": trained["best_round"],
                    "init_hold_med9": trained["init_hold_med9"],
                    "best_hold_med9": trained["best_hold_med9"],
                    "wall_s": trained["wall_s"],
                    "rounds": args.rounds,
                    "max_iter": S.LBFGS_ITER,
                    "lr": S.LBFGS_LR,
                    "batch": args.batch,
                    "lambda": LAM,
                    "mu": MU,
                    "lambda_x": LAMBDA_X,
                    "shell_rms": None if phi is None else shell_rms(phi.detach().cpu().numpy()),
                    "angles": angles_to_lists(method, trained["angles"]),
                    "eval": ev,
                    "hist": trained["hist"],
                }
                write_json(path, payload)
                log(
                    f"saved {path.name} best_r={trained['best_round']} "
                    f"n12={ev['12']['med_eps']:.4f} n15={ev['15']['med_eps']:.4f} "
                    f"n18={ev['18']['med_eps']:.4f} wall={trained['wall_s']:.0f}s"
                )

    summary = aggregate(depths, seeds, diag_by_n)
    log("OVERALL", summary["verdict"], summary["reasons"])


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
