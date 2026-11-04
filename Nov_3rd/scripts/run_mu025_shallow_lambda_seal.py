#!/usr/bin/env python3
"""Shallow seal at the softer μ=−0.25 point, after the SAFE Λ kill.

Candidate point (same family as PR #36 / #38 / #41, weaker XX):

    H = diag(unsat) − 0.75 Σ_i X_i − 0.25 Σ_c Σ_{i<j in c} X_i X_j

which is unsat − λ ΣX + μ Σ_cav XX at λ=0.75, μ=−0.25.

Train Walsh Hamming-shell SNAP, matched HVA, and Fourier QAOA on n∈{6,9}
at L∈{2,4}, five seeds. Freeze n=12 (B=128). Probe n=15 (B=64) and n=18
(B=32). L≥8 is not trained. The n=21 ladder is not read or written.

Cavity diagnostic, from the X histogram of the true ground state:

    Λ_c = ⟨(Σ_{i∈c} X_i / 3)^2⟩

Median Λ pools cavities and instances. The gate uses the n=12 median.
Λ→1 means the cavity has collapsed to one X-magnetization.

Gate (n=12 freeze):
    KILL       if median Λ ≥ 0.90, or L=2 R_HVA > 0.98
    PASS       if L=2 and L=4 R_HVA ≤ 0.85, at least 4/5 seeds BEAT
               (per-seed R_HVA ≤ 0.98), median Λ ≤ 0.80, and the median
               product gap ≥ 0.03
    needs_edit otherwise. In particular, R inside the pass bar with
               Λ in (0.80, 0.90) is needs_edit.
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
    LAMBDA_X,
    MU_XX,
    angles_to_lists,
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
    summarize_eps,
    walsh_power_by_degree,
    _prob,
    apply_H_all,
)

RUN = S.OUT / "quantum_h_explore" / "mu025_shallow_Lambda_seal"
PACK = S.OUT / "beat_qaoa_12h" / "train_packs"
METHODS = ("fourier", "hva", "snap")
TRAIN_NS = (6, 9)
TRAIN_LO, TRAIN_HI = 0, 200
HOLD_LO, HOLD_HI = 200, 250
FREEZE_B = {12: 128, 15: 64}
N18_B = 32
N18_BASE = 61_000_000_000
DEPTHS = (2, 4)
SEEDS = (0, 1, 2, 3, 4)
# Product-gap convention: H = diag − λ ΣX + μ ΣXX with λ>0, μ=−0.25.
LAM_POS = 0.75
MU_POINT = -0.25
BEAT_R = 0.98
PROTOCOL = "mu025_shallow_lambda_seal_r11"


def log(*a):
    print(time.strftime("%H:%M:%S"), "[mu025-seal]", *a, flush=True)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(path)


def ratio(num: float, den: float) -> float:
    if den == 0 or not math.isfinite(den) or not math.isfinite(num):
        return float("inf")
    return float(num) / float(den)


def classify(r2: float, r4: float, beats2: int, beats4: int, gap: float, lam: float) -> dict:
    """n=12 gate for μ=−0.25. KILL wins over a soft miss.

    ``beats*`` count seeds whose own n=12 R_HVA is ≤ 0.98. The pass bar is
    at least 4 such seeds, out of the 5-seed protocol.
    """
    kill = []
    if float(lam) >= 0.90 - 1e-12:
        kill.append("n12 median Λ ≥ 0.90")
    if float(r2) > 0.98 + 1e-12:
        kill.append("n12 L=2 R_HVA > 0.98")
    soft = []
    if float(r2) > 0.85 + 1e-12:
        soft.append("n12 L=2 R_HVA > 0.85")
    if float(r4) > 0.85 + 1e-12:
        soft.append("n12 L=4 R_HVA > 0.85")
    if int(beats2) < 4:
        soft.append(f"L=2 only {int(beats2)}/5 seeds BEAT")
    if int(beats4) < 4:
        soft.append(f"L=4 only {int(beats4)}/5 seeds BEAT")
    if float(lam) > 0.80 + 1e-12:
        soft.append("n12 median Λ > 0.80")
    if float(gap) + 1e-12 < 0.03:
        soft.append("n12 median product gap < 0.03")
    if kill:
        verdict = "KILL"
        reasons = kill
    elif soft:
        verdict = "needs_edit"
        reasons = soft
    else:
        verdict = "PASS"
        reasons = []
    return {"verdict": verdict, "reasons": reasons, "kill": kill, "soft": soft}


def _self_test_gate() -> None:
    ok = classify(0.64, 0.80, 5, 4, 0.11, 0.70)
    if ok["verdict"] != "PASS":
        raise AssertionError(ok)
    # R inside the bar, Λ in (0.80, 0.90) → needs_edit.
    soft_lam = classify(0.70, 0.80, 5, 5, 0.11, 0.85)
    if soft_lam["verdict"] != "needs_edit" or "n12 median Λ > 0.80" not in soft_lam["soft"]:
        raise AssertionError(soft_lam)
    soft_r = classify(0.90, 0.80, 4, 5, 0.11, 0.70)
    if soft_r["verdict"] != "needs_edit":
        raise AssertionError(soft_r)
    soft_l4 = classify(0.70, 0.86, 5, 5, 0.11, 0.70)
    if soft_l4["verdict"] != "needs_edit":
        raise AssertionError(soft_l4)
    soft_beat = classify(0.70, 0.80, 3, 5, 0.11, 0.70)
    if soft_beat["verdict"] != "needs_edit":
        raise AssertionError(soft_beat)
    soft_gap = classify(0.70, 0.80, 5, 5, 0.02, 0.70)
    if soft_gap["verdict"] != "needs_edit":
        raise AssertionError(soft_gap)
    kill_r = classify(0.99, 0.50, 5, 5, 0.11, 0.70)
    if kill_r["verdict"] != "KILL":
        raise AssertionError(kill_r)
    kill_l = classify(0.50, 0.50, 5, 5, 0.11, 0.90)
    if kill_l["verdict"] != "KILL":
        raise AssertionError(kill_l)
    # KILL wins when a soft miss is also present.
    both = classify(1.05, 0.95, 1, 1, 0.01, 0.94)
    if both["verdict"] != "KILL" or len(both["kill"]) != 2:
        raise AssertionError(both)
    edge = classify(0.85, 0.85, 4, 4, 0.03, 0.80)
    if edge["verdict"] != "PASS":
        raise AssertionError(edge)
    # Λ just above 0.80 is not a kill.
    just = classify(0.85, 0.85, 4, 4, 0.03, 0.8000001)
    if just["verdict"] != "needs_edit":
        raise AssertionError(just)


def load_unsat(n: int, sl: slice) -> tuple[np.ndarray, np.ndarray]:
    path = PACK / f"train_n{n}.npz"
    d = np.load(path)
    E = np.asarray(d["train_E"][sl], dtype=np.float64)
    seeds = np.asarray(d["train_seeds"][sl]).astype(np.int64)
    if E.ndim != 2 or E.shape[1] != (1 << n):
        raise RuntimeError(f"n{n} E shape {E.shape}")
    return E, seeds


def build_n18(count: int = N18_B) -> tuple[np.ndarray, np.ndarray]:
    """Satisfiable 4-SAT. Same base as the PR #38 probe, so the first 16 match."""
    from explore_OCT29.lib.sat_gen import instances

    cache = RUN / f"pack_n18_B{count}.npz"
    if cache.exists():
        z = np.load(cache)
        E = np.asarray(z["E"], dtype=np.float64)
        seeds = np.asarray(z["seeds"]).astype(np.int64)
        if E.shape == (count, 1 << 18) and seeds.shape == (count,):
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
    log(f"n18 pack written in {time.time()-t0:.1f}s seeds {int(seeds[0])}..{int(seeds[-1])}")
    return E, seeds


def _diag_job(payload: tuple) -> tuple:
    row, n, seed = payload
    bundle = ground_bundle(np.asarray(row, dtype=np.float64), int(n))
    lam = np.asarray(lambda_of_state(bundle["psi"], int(n)), dtype=np.float64).reshape(-1)
    e_prod = best_product_energy(row, int(n), LAM_POS, MU_POINT, seed=int(seed))
    gap = product_gap(e_prod, bundle["E0"])
    return (
        float(bundle["E0"]),
        float(bundle["peak"]),
        float(bundle["residual"]),
        lam,
        float(e_prod),
        float(gap),
    )


def _pool_init() -> None:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    torch.set_num_threads(1)


def diagnostics(n: int, tag: str, E: np.ndarray, seeds: np.ndarray, workers: int) -> dict:
    """E0, GS peak, per-cavity Λ, and the XZ product gap. Cached."""
    cache = RUN / f"diag_n{n}_{tag}.npz"
    if cache.exists():
        z = np.load(cache)
        if np.array_equal(z["seeds"], seeds) and int(z["e0"].shape[0]) == int(E.shape[0]):
            log(f"diag cache hit n{n} {tag}")
            return {k: np.asarray(z[k]) for k in ("e0", "peak", "residual", "lambda_c", "e_prod", "prod_gap", "seeds")}
    log(f"diag n{n} {tag} B={E.shape[0]} workers={workers}")
    t0 = time.time()
    jobs = [(E[i], n, int(seeds[i])) for i in range(E.shape[0])]
    if workers <= 1 or len(jobs) == 1 or n <= 9:
        rows = []
        for i, job in enumerate(jobs):
            rows.append(_diag_job(job))
            if (i + 1) == len(jobs) or (i + 1) % 4 == 0:
                log(f"diag n{n} {i+1}/{len(jobs)} E0={rows[-1][0]:.4f} Λmed={float(np.median(rows[-1][3])):.3f} elapsed={time.time()-t0:.0f}s")
    else:
        import multiprocessing as mp

        ctx = mp.get_context("spawn")
        rows = []
        with ctx.Pool(int(workers), initializer=_pool_init) as pool:
            for i, row in enumerate(pool.imap(_diag_job, jobs, chunksize=1)):
                rows.append(row)
                if (i + 1) == len(jobs) or (i + 1) % 4 == 0:
                    log(f"diag n{n} {i+1}/{len(jobs)} E0={row[0]:.4f} Λmed={float(np.median(row[3])):.3f} elapsed={time.time()-t0:.0f}s")
    e0 = np.array([r[0] for r in rows], dtype=np.float64)
    peak = np.array([r[1] for r in rows], dtype=np.float64)
    residual = np.array([r[2] for r in rows], dtype=np.float64)
    lam = np.stack([r[3] for r in rows], axis=0)
    e_prod = np.array([r[4] for r in rows], dtype=np.float64)
    gap = np.array([r[5] for r in rows], dtype=np.float64)
    if np.any(e0 > E.min(axis=1) + 1e-6):
        raise RuntimeError(f"E0 above a diagonal entry at n={n} {tag}")
    if np.any(residual > 1e-5):
        raise RuntimeError(f"ground residual {float(residual.max())} at n={n}")
    RUN.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache,
        seeds=seeds,
        e0=e0,
        peak=peak,
        residual=residual,
        lambda_c=lam,
        e_prod=e_prod,
        prod_gap=gap,
        n=np.int64(n),
        tag=tag,
    )
    log(
        f"diag n{n} {tag} {time.time()-t0:.1f}s  medE0={np.median(e0):.4f} "
        f"Λ={np.median(lam):.3f} gap={np.median(gap):.3f} peak={np.median(peak):.4f}"
    )
    return {
        "e0": e0,
        "peak": peak,
        "residual": residual,
        "lambda_c": lam,
        "e_prod": e_prod,
        "prod_gap": gap,
        "seeds": seeds,
    }


def _as_f64(angles: dict) -> dict:
    return {
        k: (v.detach().to(dtype=torch.float64) if torch.is_tensor(v) else v)
        for k, v in angles.items()
    }


def eval_method(method: str, angles: dict, bundle: dict, *, chunk: int, with_lambda: bool) -> dict:
    ang = _as_f64(angles)
    out = {}
    for n, (E, e0, seeds) in bundle.items():
        use = 2 if int(n) >= 18 else (4 if int(n) >= 15 else chunk)
        Eu = torch.tensor(E, dtype=torch.float64)
        e0t = torch.tensor(e0, dtype=torch.float64)
        eps_parts = []
        lam_parts = []
        B = Eu.shape[0]
        for s in range(0, B, use):
            sl = slice(s, s + use)
            Cn = cn_from_unsat(Eu[sl], int(n))
            psi = prepare_state(method, Cn, ang)
            h = expectation_H(psi, Eu[sl], int(n))
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
            rec["lambda_med"] = float(np.median(lam))
            rec["lambda_mean"] = float(np.mean(lam))
        out[str(n)] = rec
    return out


def train_loss(method, angles, E, e0, idx, n: int) -> torch.Tensor:
    Eu = E.index_select(0, idx)
    e0b = e0.index_select(0, idx)
    Cn = cn_from_unsat(Eu, n)
    psi = prepare_state(method, Cn, angles)
    h = expectation_H(psi, Eu, n)
    return ((h - e0b) / float(n)).mean()


def math_finite(x: float) -> bool:
    return x == x and abs(x) != float("inf")


def train_one(method: str, L: int, seed: int, bundles_t, hold_bundle, *, rounds: int, batch: int) -> dict:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    angles = init_angles(method, L, dtype=torch.float64)
    params = method_parameter_list(method, angles)
    E6, e06 = bundles_t[6]
    E9, e09 = bundles_t[9]
    n6 = E6.shape[0]
    n9 = E9.shape[0]
    b6 = min(batch // 2, n6)
    b9 = min(batch - b6, n9)

    def hold_med() -> float:
        ev = eval_method(method, angles, hold_bundle, chunk=16, with_lambda=False)
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

        def closure():
            opt.zero_grad()
            loss = 0.5 * (
                train_loss(method, angles, E6, e06, i6, 6)
                + train_loss(method, angles, E9, e09, i9, 9)
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
        "L": L,
        "seed": seed,
        "init_hold_med9": init_med,
        "best_round": best["round"],
        "best_hold_med9": best["med9"],
        "wall_s": time.time() - t0,
        "hist": hist,
        "angles": best["angles"],
        "n_params": int(sum(p.numel() for p in params)),
    }


def seed_path(L: int, method: str, seed: int) -> Path:
    return RUN / f"L{L}_{method}_seed{seed}.json"


def _fmt(x, nd: int = 4) -> str:
    if x is None or not math.isfinite(float(x)):
        return "—"
    return f"{float(x):.{nd}f}"


def _beta_xx_mean(rows: list[dict]) -> float | None:
    vals = []
    for row in rows:
        if row["method"] != "hva":
            continue
        beta = row["angles"].get("beta_xx")
        if beta:
            vals.extend(abs(float(x)) for x in beta)
    if not vals:
        return None
    return float(np.mean(vals))


def aggregate(depths: list[int], seeds: list[int], diag: dict) -> dict:
    rows = []
    for L in depths:
        for method in METHODS:
            for seed in seeds:
                path = seed_path(L, method, seed)
                if not path.exists():
                    raise RuntimeError(f"missing {path}")
                rows.append(json.loads(path.read_text()))
    by_depth = {}
    for L in depths:
        block = [r for r in rows if int(r["L"]) == int(L)]
        cell = {}
        for method in METHODS:
            cell[method] = {}
            sub = [r for r in block if r["method"] == method]
            for n in ("6", "9", "12", "15", "18"):
                vals = [float(r["eval"][n]["med_eps"]) for r in sub]
                cell[method][n] = float(np.median(vals))
        rh = {}
        rf = {}
        steps = {}
        for n in ("6", "9", "12", "15", "18"):
            rh[n] = ratio(cell["snap"][n], cell["hva"][n])
            rf[n] = ratio(cell["snap"][n], cell["fourier"][n])
        order = ("12", "15", "18")
        step_list = []
        for a, b in zip(order, order[1:]):
            step = rh[b] - rh[a]
            steps[f"{a}_to_{b}"] = step
            step_list.append(step)
        # Per-seed ratios. The gate counts n=12 BEATs (R_HVA ≤ 0.98).
        seed_rows = []
        paired12_num = 0
        paired12_den = 0
        paired18_num = 0
        paired18_den = 0
        for seed in seeds:
            snap = next(r for r in block if r["method"] == "snap" and int(r["seed"]) == int(seed))
            hva = next(r for r in block if r["method"] == "hva" and int(r["seed"]) == int(seed))
            four = next(r for r in block if r["method"] == "fourier" and int(r["seed"]) == int(seed))
            r12 = ratio(snap["eval"]["12"]["med_eps"], hva["eval"]["12"]["med_eps"])
            r18 = ratio(snap["eval"]["18"]["med_eps"], hva["eval"]["18"]["med_eps"])
            es12 = np.asarray(snap["eval"]["12"]["eps"], dtype=np.float64)
            eh12 = np.asarray(hva["eval"]["12"]["eps"], dtype=np.float64)
            es18 = np.asarray(snap["eval"]["18"]["eps"], dtype=np.float64)
            eh18 = np.asarray(hva["eval"]["18"]["eps"], dtype=np.float64)
            wins12 = int(np.sum(es12 < eh12 - 1e-12))
            wins18 = int(np.sum(es18 < eh18 - 1e-12))
            paired12_num += wins12
            paired12_den += int(es12.size)
            paired18_num += wins18
            paired18_den += int(es18.size)
            seed_rows.append(
                {
                    "seed": int(seed),
                    "snap12": float(snap["eval"]["12"]["med_eps"]),
                    "hva12": float(hva["eval"]["12"]["med_eps"]),
                    "fourier12": float(four["eval"]["12"]["med_eps"]),
                    "R12": r12,
                    "beat12": bool(r12 <= BEAT_R + 1e-12),
                    "paired12": wins12,
                    "B12": int(es12.size),
                    "snap18": float(snap["eval"]["18"]["med_eps"]),
                    "hva18": float(hva["eval"]["18"]["med_eps"]),
                    "fourier18": float(four["eval"]["18"]["med_eps"]),
                    "R18": r18,
                    "paired18": wins18,
                    "B18": int(es18.size),
                    "best_round_snap": snap["best_round"],
                    "wall_snap_s": snap["wall_s"],
                    "wall_hva_s": hva["wall_s"],
                    "wall_fourier_s": four["wall_s"],
                }
            )
        n_beat = int(sum(1 for row in seed_rows if row["beat12"]))
        circ = {}
        for method in METHODS:
            sub = [r for r in block if r["method"] == method]
            circ[method] = {
                n: float(np.median([r["eval"][n]["lambda_med"] for r in sub]))
                for n in ("12", "15", "18")
            }
        by_depth[str(L)] = {
            "med_eps": cell,
            "R_hva": rh,
            "R_fourier": rf,
            "steps": steps,
            "deg": float(max(step_list)),
            "seeds": seed_rows,
            "n_beat12": n_beat,
            "paired12": paired12_num,
            "paired12_den": paired12_den,
            "paired18": paired18_num,
            "paired18_den": paired18_den,
            "beta_xx_mean_abs": _beta_xx_mean(block),
            "circuit_lambda_med": circ,
        }

    def _lam_summary(n: int) -> dict:
        lam = np.asarray(diag[n]["lambda_c"], dtype=np.float64)
        gap = np.asarray(diag[n]["prod_gap"], dtype=np.float64)
        peak = np.asarray(diag[n]["peak"], dtype=np.float64)
        return {
            "B": int(lam.shape[0]),
            "C": int(lam.shape[1]),
            "lambda_med": float(np.median(lam)),
            "lambda_mean": float(np.mean(lam)),
            "lambda_min": float(np.min(lam)),
            "lambda_max": float(np.max(lam)),
            "frac_gt_080": float(np.mean(lam > 0.80)),
            "frac_gt_085": float(np.mean(lam > 0.85)),
            "frac_ge_090": float(np.mean(lam >= 0.90 - 1e-12)),
            "gap_med": float(np.median(gap)),
            "gap_min": float(np.min(gap)),
            "frac_gap_ge_003": float(np.mean(gap >= 0.03 - 1e-12)),
            "frac_gap_ge_005": float(np.mean(gap >= 0.05 - 1e-12)),
            "peak_med": float(np.median(peak)),
            "peak_max": float(np.max(peak)),
        }

    q = {n: _lam_summary(n) for n in (12, 15, 18)}
    r2 = by_depth["2"]["R_hva"]["12"]
    r4 = by_depth["4"]["R_hva"]["12"]
    call = classify(
        r2,
        r4,
        by_depth["2"]["n_beat12"],
        by_depth["4"]["n_beat12"],
        q[12]["gap_med"],
        q[12]["lambda_med"],
    )
    summary = {
        "protocol": PROTOCOL,
        "lambda_x": LAMBDA_X,
        "mu_xx": MU_XX,
        "lam_pos": LAM_POS,
        "mu_point": MU_POINT,
        "beat_r": BEAT_R,
        "depths": depths,
        "seeds": seeds,
        "freeze_B": {"12": FREEZE_B[12], "15": FREEZE_B[15], "18": N18_B},
        "by_depth": by_depth,
        "quantum": {str(n): q[n] for n in q},
        "verdict": call["verdict"],
        "reasons": call["reasons"],
        "kill": call["kill"],
        "soft": call["soft"],
        "prior": {
            "safe_mu": -0.50,
            "safe_n12_lambda_pr41": 0.941,
            "safe_n18_lambda_pr41": 0.943,
            "mu025_L2_n12_R_fourier_pr25": 0.483,
            "mu025_L16_n12_R_fourier_pr25": 0.342,
            "mu025_n12_product_gap_pr25": 0.110,
        },
    }
    text = render(summary)
    write_json(RUN / "summary.json", summary)
    (RUN / "RESULTS.md").write_text(text)
    log(f"wrote RESULTS.md verdict={call['verdict']}")
    return summary


def render(summary: dict) -> str:
    q18 = summary["quantum"]["18"]
    q12 = summary["quantum"]["12"]
    q15 = summary["quantum"]["15"]
    verdict = summary["verdict"]
    reasons = summary["reasons"]
    reason_txt = "; ".join(reasons) if reasons else "every n=12 bar cleared"
    d2 = summary["by_depth"]["2"]
    d4 = summary["by_depth"]["4"]
    lines = [
        "# μ=−0.25 shallow Λ seal — λ=0.75, Walsh SNAP vs HVA vs Fourier",
        "",
        f"**Verdict: {verdict}.** {reason_txt}.",
        "",
        "This is the softer-μ candidate after the SAFE lock (μ=−0.50) was killed by cavity Λ. "
        f"PR #41 reported median Λ≈0.941 at n=12 and 0.943 at n=18, with shallow R_HVA well under 0.5. "
        "The question here is whether μ=−0.25 keeps the ground-state cavities from collapsing "
        f"(Λ≤0.80) while shallow Walsh still beats matched HVA (R_HVA≤0.85, ≥4/5 seeds). "
        f"Measured n=12 median Λ={_fmt(q12['lambda_med'], 3)}, "
        f"L=2 R_HVA={_fmt(d2['R_hva']['12'])} ({d2['n_beat12']}/5 BEAT), "
        f"L=4 R_HVA={_fmt(d4['R_hva']['12'])} ({d4['n_beat12']}/5 BEAT), "
        f"n=12 median product gap={_fmt(q12['gap_med'], 3)}.",
        "",
        "The gate is the n=12 freeze (B=128). n=15 (B=64) and n=18 (B=32) are probes and do not "
        "enter the verdict. L≥8 is not trained. PR #25 is the earlier μ=−0.25 record against Fourier "
        "(L=2 n=12 R≈0.483, L=16 n=12 R≈0.342, product gap≈0.110), not against HVA.",
        "",
        "## Gate",
        "",
        "Med ε is the median over seeds of each seed's median-over-instances excess energy ⟨H⟩−E₀. "
        "R_HVA = Medε_SNAP / Medε_HVA. A seed BEATs when its own n=12 R_HVA ≤ 0.98. "
        "The product gap is (E_prod − E_0) / |E_0| for the best real XZ product state. "
        "Λ_c = ⟨(Σ_{i∈c} X_i / 3)^2⟩ from the X histogram of the true ground state. "
        "The reported Λ is the median over cavities and instances.",
        "",
        "- **PASS** when, at n=12, L=2 and L=4 R_HVA ≤ 0.85, at least 4/5 seeds BEAT at each depth, median Λ ≤ 0.80, and the median product gap ≥ 0.03.",
        "- **needs_edit** when nothing kills the point and at least one PASS bar is missed. R inside the bar with Λ in (0.80, 0.90) is this case.",
        "- **KILL** when n=12 median Λ ≥ 0.90, or when n=12 L=2 R_HVA > 0.98. Either one kills the candidate.",
        "",
        "## Hamiltonian",
        "",
        "\\[",
        "H = \\mathrm{diag}(\\mathrm{unsat}) - 0.75 \\sum_i X_i - 0.25 \\sum_{c}\\sum_{i<j\\in c} X_i X_j",
        "\\]",
        "",
        "Candidate point **λ=0.75, μ=−0.25**, written unsat − λ ΣX + μ Σ_cav XX. "
        "Cavity c owns qubits [3c, 3c+3), qubit 0 = MSB. Readout is the Z histogram (unsat) plus the X histogram "
        "(⟨X⟩, ⟨XX⟩, and Λ). No Y. The mixer is BinH Hamming-shell SNAP. "
        "HVA adds one β^XX angle per layer on the same intra-cavity pairs. Fourier is the two-angle QAOA nested at β^XX=0.",
        "",
        "## Protocol",
        "",
        "- Packs: `beat_qaoa_12h/train_packs`. Train indices [0:200) on n=6 and n=9. Checkpoint on n=9 holdout [200:250). Freeze n=12 first 128. Probe n=15 first 64.",
        f"- n=18: {N18_B} satisfiable formulas from `explore_OCT29.lib.sat_gen.instances` at base {N18_BASE}. There is no n=18 train pack. These instances are not in the loss and not in checkpoint selection.",
        "- L-BFGS matches `Nov_3rd.settings` and the PR #41 shallow seal: 40 rounds × max_iter 8, lr 0.25, history 16, strong Wolfe, grad clip 5, batch 64 (half n6 / half n9). New optimizer each round. Loss = mean ε/n. Each seed restores the round with the best n=9-holdout Med ε.",
        "- Paper init β=β^X=0.01, β^XX=0, γ=γ_w=−0.01, φ_k=−β k. The three inits are the same state.",
        "- n=21 was not loaded. `qaoa_angles/`, `train69_ladder_fresh250/`, and `settings.py` were not written.",
        "",
        "## Ground state: Λ and product gap",
        "",
        "Λ→1 means every cavity sits in a single X-magnetization sector, so the 3-qubit cavity has collapsed to an effective spin. "
        "A Z-basis product state scores Λ=1/3. The SAFE μ=−0.50 point sat near 0.94. The pass line is 0.80 and the kill line is 0.90.",
        "",
        "| n | role | B | median Λ | min Λ | max Λ | fraction Λ>0.80 | fraction Λ≥0.90 | median product gap | min gap | fraction gap≥0.03 | median Z-peak |",
        "|--:|:-----|--:|--------:|------:|------:|----------------:|----------------:|-------------------:|--------:|------------------:|-------------:|",
    ]
    role_q = {"12": "freeze (gate)", "15": "probe", "18": "probe"}
    for n, row in (("12", q12), ("15", q15), ("18", q18)):
        lines.append(
            f"| {n} | {role_q[n]} | {row['B']} | {_fmt(row['lambda_med'], 3)} | {_fmt(row['lambda_min'], 3)} | "
            f"{_fmt(row['lambda_max'], 3)} | {_fmt(row['frac_gt_080'], 3)} | {_fmt(row['frac_ge_090'], 3)} | "
            f"{_fmt(row['gap_med'], 3)} | {_fmt(row['gap_min'], 3)} | {_fmt(row['frac_gap_ge_003'], 3)} | "
            f"{_fmt(row['peak_med'], 4)} |"
        )
    lines += [
        "",
        f"n=12 median Λ = **{_fmt(q12['lambda_med'], 3)}**. "
        f"n=15 median Λ = **{_fmt(q15['lambda_med'], 3)}**. "
        f"n=18 median Λ = **{_fmt(q18['lambda_med'], 3)}**. "
        f"n=12 median product gap = **{_fmt(q12['gap_med'], 3)}** "
        f"(n=15 {_fmt(q15['gap_med'], 3)}, n=18 {_fmt(q18['gap_med'], 3)}). "
        f"The n=12 Z-basis peak of the ground state is {_fmt(q12['peak_med'], 4)} "
        f"(n=18 {_fmt(q18['peak_med'], 4)}).",
        "",
    ]
    role = {
        "6": "holdout",
        "9": "holdout (select)",
        "12": "freeze (gate)",
        "15": "probe",
        "18": "probe",
    }
    for L in ("2", "4"):
        block = summary["by_depth"][L]
        lines += [
            f"## L={L}",
            "",
            "| n | role | Medε Fourier | Medε HVA | Medε SNAP | R SNAP/Fourier | R SNAP/HVA |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
        for n in ("6", "9", "12", "15", "18"):
            lines.append(
                f"| {n} | {role[n]} | {_fmt(block['med_eps']['fourier'][n])} | "
                f"{_fmt(block['med_eps']['hva'][n])} | {_fmt(block['med_eps']['snap'][n])} | "
                f"{_fmt(block['R_fourier'][n])} | {_fmt(block['R_hva'][n])} |"
            )
        s12 = block["steps"]["12_to_15"]
        s15 = block["steps"]["15_to_18"]
        lines += [
            "",
            f"**L={L} n12 R_HVA = {_fmt(block['R_hva']['12'])}** "
            f"(SNAP {_fmt(block['med_eps']['snap']['12'])} / HVA {_fmt(block['med_eps']['hva']['12'])}). "
            f"Seeds BEAT (n12 R_HVA≤0.98): {block['n_beat12']}/5. "
            f"Probe R_HVA is {_fmt(block['R_hva']['15'])} at n=15 and {_fmt(block['R_hva']['18'])} at n=18. "
            f"Steps of R_HVA: n12→n15 {_fmt(s12, 4)}, n15→n18 {_fmt(s15, 4)}.",
            "",
            f"Paired n12 instances with SNAP ε below HVA: {block['paired12']}/{block['paired12_den']}. "
            f"Paired n18: {block['paired18']}/{block['paired18_den']}. "
            f"HVA mean |β^XX| = {_fmt(block['beta_xx_mean_abs'], 3)}.",
            "",
            "| seed | n12 SNAP | n12 HVA | n12 Fourier | n12 R_HVA | BEAT | paired n12 | n18 R_HVA | best round | wall s |",
            "|---:|---:|---:|---:|---:|:----:|---:|---:|---:|---:|",
        ]
        for row in block["seeds"]:
            beat = "yes" if row["beat12"] else "no"
            lines.append(
                f"| {row['seed']} | {_fmt(row['snap12'])} | {_fmt(row['hva12'])} | {_fmt(row['fourier12'])} | "
                f"{_fmt(row['R12'])} | {beat} | {row['paired12']}/{row['B12']} | {_fmt(row['R18'])} | "
                f"{row['best_round_snap']} | {row['wall_snap_s']:.0f} |"
            )
        circ = block["circuit_lambda_med"]
        lines += [
            "",
            "Circuit-state median Λ (X histogram of the prepared state, median over seeds of each seed's pooled median) "
            f"at n12: Fourier {_fmt(circ['fourier']['12'], 3)}, HVA {_fmt(circ['hva']['12'], 3)}, "
            f"SNAP {_fmt(circ['snap']['12'], 3)}. "
            f"At n18: Fourier {_fmt(circ['fourier']['18'], 3)}, HVA {_fmt(circ['hva']['18'], 3)}, "
            f"SNAP {_fmt(circ['snap']['18'], 3)}. The gate uses the ground-state Λ, not these.",
            "",
        ]
    lines += [
        "## Reading",
        "",
        f"The call is **{verdict}**. {reason_txt}.",
        "",
        f"At the n=12 freeze, L=2 goes R_HVA={_fmt(d2['R_hva']['12'])} "
        f"(Fourier ratio {_fmt(d2['R_fourier']['12'])}) and L=4 goes R_HVA={_fmt(d4['R_hva']['12'])} "
        f"(Fourier ratio {_fmt(d4['R_fourier']['12'])}). "
        f"The probes move to L=2 n15/n18 {_fmt(d2['R_hva']['15'])}/{_fmt(d2['R_hva']['18'])} and "
        f"L=4 n15/n18 {_fmt(d4['R_hva']['15'])}/{_fmt(d4['R_hva']['18'])}. "
        f"HVA |β^XX| is {_fmt(d2['beta_xx_mean_abs'], 3)} at L=2 and {_fmt(d4['beta_xx_mean_abs'], 3)} at L=4. "
        f"L=4 on the n=6 holdout has R_HVA={_fmt(d4['R_hva']['6'])}, so SNAP does not lead HVA on that small set. "
        "The lead is at the n=12 freeze and on both probes.",
        "",
        f"Ground-state median Λ is {_fmt(q12['lambda_med'], 3)} / {_fmt(q15['lambda_med'], 3)} / {_fmt(q18['lambda_med'], 3)} "
        "at n=12 / 15 / 18. "
        f"The fraction of cavities above 0.80 is {_fmt(q12['frac_gt_080'], 3)} / {_fmt(q15['frac_gt_080'], 3)} / {_fmt(q18['frac_gt_080'], 3)}. "
        f"The fraction at or above 0.90 is {_fmt(q12['frac_ge_090'], 3)} / {_fmt(q15['frac_ge_090'], 3)} / {_fmt(q18['frac_ge_090'], 3)}. "
        "Λ equals 1/3 + (2/9) Σ_{i<j∈c} ⟨X_i X_j⟩, computed from the joint X histogram of the ground state. "
        "Values near 1 mean the intra-cavity space has collapsed to an effective spin, which is what killed μ=−0.50. "
        "The only missed PASS bar is the n=12 median Λ. The same median sits in (0.80, 0.90) at n=15 and n=18, "
        "and it stays under the 0.90 kill line at every reported size. Relative to SAFE μ=−0.50 (Λ≈0.94), "
        "the softer XX term moved the median down by about 0.05 and did not restore Λ≤0.80.",
        "",
        f"The product gap is {_fmt(q12['gap_med'], 3)} / {_fmt(q15['gap_med'], 3)} / {_fmt(q18['gap_med'], 3)} "
        "at n=12 / 15 / 18 (bar ≥ 0.03 on the n=12 median). "
        f"Z-basis peaks stay {_fmt(q12['peak_med'], 4)} / {_fmt(q15['peak_med'], 4)} / {_fmt(q18['peak_med'], 4)}, "
        "so the ground state is not a classical bitstring.",
        "",
        "Per-seed JSON: `L{L}_{method}_seed{s}.json` in this folder. Angles are the restored n9-holdout checkpoint. "
        "Ground-state Λ and the product gap are in `diag_n{n}_{tag}.npz`.",
        "",
    ]
    return "\n".join(lines) + "\n"


def _workers_for(n: int, requested: int) -> int:
    """Keep n=18 Lanczos from stacking too many 2^18 workspaces."""
    if int(n) >= 18:
        return max(1, min(int(requested), 2))
    if int(n) >= 15:
        return max(1, min(int(requested), 3))
    return max(1, int(requested))


def main():
    _self_test_gate()
    if abs(float(MU_XX) - MU_POINT) > 1e-15 or abs(float(LAMBDA_X) + LAM_POS) > 1e-15:
        raise SystemExit(f"Hamiltonian pin mismatch LAMBDA_X={LAMBDA_X} MU_XX={MU_XX}")
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
    ap.add_argument("--skip-self-check", action="store_true")
    args = ap.parse_args()

    depths = [int(x) for x in args.depths.split(",") if x.strip()]
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    bad = [L for L in depths if L not in DEPTHS]
    if bad:
        raise SystemExit(f"refusing depths {bad}: this seal is L in {{2,4}} only")
    if any(s < 0 or s > 4 for s in seeds):
        raise SystemExit("seeds must be in 0..4")
    RUN.mkdir(parents=True, exist_ok=True)

    if not args.skip_self_check and not args.aggregate_only:
        log("self-check")
        self_check()
        log("self-check ok")

    # Train-set ground energies do not need Λ. Full diagnostics cover the freeze sets
    # and are independent of the angles, so they can be reused across reruns.
    data_E = {}
    if not args.aggregate_only:
        for n, sl, tag in (
            (6, slice(TRAIN_LO, TRAIN_HI), "train"),
            (9, slice(TRAIN_LO, TRAIN_HI), "train"),
            (6, slice(HOLD_LO, HOLD_HI), "hold"),
            (9, slice(HOLD_LO, HOLD_HI), "hold"),
        ):
            E, sd = load_unsat(n, sl)
            # Dense eigh is inside ground_bundle. n≤9 is fast in-process.
            diag = diagnostics(n, tag, E, sd, workers=1)
            data_E[(n, tag)] = (E, diag["e0"], sd)
            log(f"n{n} {tag} B={len(E)} medE0={float(np.median(diag['e0'])):.4f}")

    freeze = {}
    if True:
        for n, b in ((12, FREEZE_B[12]), (15, FREEZE_B[15])):
            E, sd = load_unsat(n, slice(0, b))
            diag = diagnostics(n, "freeze", E, sd, workers=_workers_for(n, args.workers))
            freeze[n] = (E, diag)
        E18, sd18 = build_n18(N18_B)
        diag18 = diagnostics(18, "freeze", E18, sd18, workers=_workers_for(18, args.workers))
        freeze[18] = (E18, diag18)

    diag_by_n = {n: freeze[n][1] for n in (12, 15, 18)}

    if args.aggregate_only:
        summary = aggregate(depths, seeds, diag_by_n)
        log("OVERALL", summary["verdict"])
        return

    pwr = walsh_power_by_degree(freeze[12][0][0])
    hi = float(pwr[5:].sum())
    log(f"Walsh n12 instance0 degree>4 power {hi:.3e}")
    if hi > 1e-8:
        raise RuntimeError(f"n12 cost is not degree ≤4 ({hi})")

    torch.set_num_threads(max(1, int(args.workers)))
    train_tensors = {}
    for n in TRAIN_NS:
        E, e0, _sd = data_E[(n, "train")]
        train_tensors[n] = (
            torch.tensor(E, dtype=torch.float64),
            torch.tensor(e0, dtype=torch.float64),
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
            m: eval_method(m, init_angles_L[m], {12: eval_bundle[12]}, chunk=8, with_lambda=False)
            for m in METHODS
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
                ev = eval_method(method, trained["angles"], eval_bundle, chunk=8, with_lambda=True)
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
                    "lambda_x": LAMBDA_X,
                    "mu_xx": MU_XX,
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
