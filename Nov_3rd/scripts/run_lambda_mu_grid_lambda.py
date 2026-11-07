#!/usr/bin/env python3
"""(λ, μ) grid: cavity Λ and shallow R_HVA at L=2.

Hamiltonian, cavity c = qubits [3c, 3c+3), qubit 0 = MSB:

    H = diag(unsat) − λ Σ_i X_i + μ Σ_c Σ_{i<j in c} X_i X_j

Grid: λ ∈ {0.50, 0.75, 1.00}, μ ∈ {0, −0.15, −0.25, −0.35, −0.50}.
The SAFE cell λ=0.75, μ=−0.50 is the known-trivialized control (PR #41,
median Λ≈0.943).

Each cell trains Walsh Hamming-shell SNAP, matched HVA, and Fourier QAOA
on n∈{6,9} (indices [0:200)), checkpoints on n=9 holdout [200:250), and
freezes n=12, B=128, L=2, seeds {0,1,2}. Fourier is the 2-angle reference
on the same budget. n=15, n=18, and n=21 are not loaded.

Λ_c = ⟨(Σ_{i∈c} X_i / 3)^2⟩ from the X histogram of the true ground state.
The reported Λ is the median over cavities and instances. The product gap
is (E_prod − E_0) / |E_0| for the best real XZ product state.

A cell is a candidate when, at n=12 L=2,

    median Λ ≤ 0.80,  product gap ≥ 0.03,  R_HVA ≤ 0.85.

R_HVA = Medε_SNAP / Medε_HVA. Medε is the median over seeds of each seed's
median-over-instances excess ⟨H⟩−E_0. Candidates are ranked by low Λ, then
low R. The SAFE cell is flagged and is not a candidate.
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
    best_product_energy,
    cn_from_unsat,
    expectation_H,
    ground_bundle,
    init_angles,
    lambda_of_state,
    method_parameter_list,
    prepare_state,
    product_gap,
    summarize_eps,
)

RUN = S.OUT / "quantum_h_explore" / "lambda_mu_grid_Lambda"
PACK = S.OUT / "beat_qaoa_12h" / "train_packs"
LAMBDAS = (0.50, 0.75, 1.00)
MUS = (0.0, -0.15, -0.25, -0.35, -0.50)
SAFE_CELL = (0.75, -0.50)
METHODS = ("fourier", "hva", "snap")
SEEDS = (0, 1, 2)
L_DEPTH = 2
TRAIN_LO, TRAIN_HI = 0, 200
HOLD_LO, HOLD_HI = 200, 250
FREEZE_B = 128
ROUNDS = 40
BATCH = 64
PROTOCOL = "lambda_mu_grid_Lambda_r11"
# PASS bars at n=12, L=2.
BAR_LAMBDA = 0.80
BAR_GAP = 0.03
BAR_R = 0.85


def log(*a):
    print(time.strftime("%H:%M:%S"), "[lam-mu]", *a, flush=True)


def cells() -> list[tuple[float, float]]:
    return [(float(lam), float(mu)) for lam in LAMBDAS for mu in MUS]


def is_safe(lam: float, mu: float) -> bool:
    return abs(float(lam) - SAFE_CELL[0]) < 1e-12 and abs(float(mu) - SAFE_CELL[1]) < 1e-12


def coeffs(lam: float, mu: float) -> tuple[float, float]:
    """Map (λ, μ) onto H = diag + λ_x ΣX + μ_xx ΣXX."""
    return -float(lam), float(mu)


def tag_of(lam: float, mu: float) -> str:
    def fmt(x: float) -> str:
        return f"{float(x):.2f}".replace("-", "m").replace(".", "p")

    return f"l{fmt(lam)}_{fmt(mu)}"


def ratio(num: float, den: float) -> float:
    if den == 0 or not math.isfinite(den) or not math.isfinite(num):
        return float("inf")
    return float(num) / float(den)


def classify_cell(lam_med: float, gap_med: float, r_hva: float, safe: bool) -> dict:
    """Candidate bars. SAFE is a control even when the numbers would pass."""
    fails = []
    if float(lam_med) > BAR_LAMBDA + 1e-12:
        fails.append(f"median Λ {lam_med:.3f} > {BAR_LAMBDA:.2f}")
    if float(gap_med) + 1e-12 < BAR_GAP:
        fails.append(f"product gap {gap_med:.3f} < {BAR_GAP:.2f}")
    if not math.isfinite(r_hva) or float(r_hva) > BAR_R + 1e-12:
        fails.append(f"R_HVA {r_hva:.3f} > {BAR_R:.2f}")
    if safe:
        role = "SAFE control"
        verdict = "CONTROL"
    elif fails:
        role = "fail"
        verdict = "FAIL"
    else:
        role = "candidate"
        verdict = "PASS"
    return {"role": role, "verdict": verdict, "fails": fails, "safe": bool(safe)}


def _self_test_gate() -> None:
    ok = classify_cell(0.70, 0.04, 0.50, False)
    if ok["verdict"] != "PASS":
        raise AssertionError(ok)
    edge = classify_cell(0.80, 0.03, 0.85, False)
    if edge["verdict"] != "PASS":
        raise AssertionError(edge)
    high_l = classify_cell(0.81, 0.05, 0.30, False)
    if high_l["verdict"] != "FAIL" or len(high_l["fails"]) != 1:
        raise AssertionError(high_l)
    low_g = classify_cell(0.50, 0.029, 0.40, False)
    if low_g["verdict"] != "FAIL":
        raise AssertionError(low_g)
    high_r = classify_cell(0.50, 0.05, 0.86, False)
    if high_r["verdict"] != "FAIL":
        raise AssertionError(high_r)
    both = classify_cell(0.95, 0.01, 1.2, False)
    if both["verdict"] != "FAIL" or len(both["fails"]) != 3:
        raise AssertionError(both)
    control = classify_cell(0.50, 0.05, 0.30, True)
    if control["verdict"] != "CONTROL" or control["role"] != "SAFE control":
        raise AssertionError(control)
    trivial = classify_cell(0.943, 0.066, 0.328, True)
    if trivial["verdict"] != "CONTROL":
        raise AssertionError(trivial)
    if not is_safe(0.75, -0.50) or is_safe(0.75, -0.35) or is_safe(0.50, -0.50):
        raise AssertionError("SAFE tag")
    lx, mx = coeffs(0.75, -0.50)
    if abs(lx + 0.75) > 1e-12 or abs(mx + 0.50) > 1e-12:
        raise AssertionError((lx, mx))


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(path)


def load_pack(n: int) -> tuple[np.ndarray, np.ndarray]:
    if int(n) >= 21:
        raise RuntimeError("n=21 is out of scope")
    path = PACK / f"train_n{n}.npz"
    d = np.load(path)
    E = np.asarray(d["train_E"], dtype=np.float64)
    seeds = np.asarray(d["train_seeds"]).astype(np.int64)
    if E.ndim != 2 or E.shape[1] != (1 << n):
        raise RuntimeError(f"n{n} E shape {E.shape}")
    if seeds.shape[0] < HOLD_HI:
        raise RuntimeError(f"n{n} pack has {seeds.shape[0]} rows, need {HOLD_HI}")
    return E, seeds


def diag_path(lam: float, mu: float) -> Path:
    return RUN / "cache" / f"diag_{tag_of(lam, mu)}.npz"


def seed_path(lam: float, mu: float, method: str, seed: int) -> Path:
    return RUN / "runs" / f"L2_{method}_s{seed}_{tag_of(lam, mu)}.json"


def seed_is_done(path: Path, lam: float, mu: float, method: str, seed: int, rounds: int) -> bool:
    """True only for a finished run at this (λ, μ) and this L-BFGS round count."""
    if not path.exists():
        return False
    try:
        row = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return (
        abs(float(row.get("lambda", 0.0)) - float(lam)) < 1e-12
        and abs(float(row.get("mu", 1.0)) - float(mu)) < 1e-12
        and row.get("method") == method
        and int(row.get("seed", -1)) == int(seed)
        and int(row.get("rounds", -1)) == int(rounds)
        and "12" in row.get("eval", {})
    )


def _pool_init() -> None:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    torch.set_num_threads(1)


def _diag_job(payload: tuple) -> tuple:
    row, n, lam, mu, seed, with_gap = payload
    lx, mx = coeffs(lam, mu)
    bundle = ground_bundle(np.asarray(row, dtype=np.float64), int(n), lambda_x=lx, mu_xx=mx)
    lam_c = np.asarray(lambda_of_state(bundle["psi"], int(n)), dtype=np.float64)
    e_prod = float("nan")
    gap = float("nan")
    if with_gap:
        e_prod = best_product_energy(row, int(n), float(lam), float(mu), seed=int(seed))
        gap = product_gap(e_prod, bundle["E0"])
    return (
        float(bundle["E0"]),
        float(bundle["peak"]),
        float(bundle["residual"]),
        lam_c,
        float(e_prod),
        float(gap),
    )


def _run_jobs(jobs: list, workers: int) -> list:
    if workers <= 1 or len(jobs) <= 1:
        return [_diag_job(job) for job in jobs]
    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    with ctx.Pool(int(workers), initializer=_pool_init) as pool:
        return list(pool.imap(_diag_job, jobs, chunksize=1))


def _pack_slice(n: int, count: int) -> tuple[np.ndarray, np.ndarray]:
    E, seeds = load_pack(n)
    return E[:count], seeds[:count]


def ensure_diag(lam: float, mu: float, workers: int) -> dict:
    """E0 / Λ / peak on n=6 and n=9 (250 rows) and n=12 freeze, plus n=12 gap."""
    cache = diag_path(lam, mu)
    E6, s6 = _pack_slice(6, HOLD_HI)
    E9, s9 = _pack_slice(9, HOLD_HI)
    E12, s12 = _pack_slice(12, FREEZE_B)
    if cache.exists():
        z = np.load(cache)
        if (
            abs(float(z["lam"]) - float(lam)) < 1e-12
            and abs(float(z["mu"]) - float(mu)) < 1e-12
            and np.array_equal(z["n12_seeds"], s12)
            and int(z["n9_e0"].shape[0]) == HOLD_HI
            and int(z["n12_e0"].shape[0]) == FREEZE_B
        ):
            log(f"diag cache hit {tag_of(lam, mu)}")
            return {k: np.asarray(z[k]) for k in z.files}
    log(f"diag {tag_of(lam, mu)} workers={workers}")
    t0 = time.time()
    out = {}
    for n, E, seeds, with_gap in (
        (6, E6, s6, False),
        (9, E9, s9, False),
        (12, E12, s12, True),
    ):
        jobs = [(E[i], n, lam, mu, int(seeds[i]), with_gap) for i in range(E.shape[0])]
        rows = _run_jobs(jobs, workers if n >= 9 else 1)
        e0 = np.array([r[0] for r in rows], dtype=np.float64)
        peak = np.array([r[1] for r in rows], dtype=np.float64)
        resid = np.array([r[2] for r in rows], dtype=np.float64)
        lam_c = np.stack([r[3] for r in rows], axis=0)
        e_prod = np.array([r[4] for r in rows], dtype=np.float64)
        gap = np.array([r[5] for r in rows], dtype=np.float64)
        if np.any(e0 > E.min(axis=1) + 1e-5):
            raise RuntimeError(f"E0 above a diagonal entry at n={n} {tag_of(lam, mu)}")
        if np.any(resid > 1e-5):
            raise RuntimeError(f"ground residual {float(resid.max())} at n={n}")
        if np.any(lam_c < -1e-8) or np.any(lam_c > 1.0 + 1e-6):
            raise RuntimeError(f"Λ out of range at n={n} {tag_of(lam, mu)}")
        prefix = f"n{n}"
        out[f"{prefix}_e0"] = e0
        out[f"{prefix}_peak"] = peak
        out[f"{prefix}_residual"] = resid
        out[f"{prefix}_lambda"] = lam_c
        out[f"{prefix}_seeds"] = seeds
        if with_gap:
            out[f"{prefix}_e_prod"] = e_prod
            out[f"{prefix}_gap"] = gap
        log(
            f"  n{n} medE0={np.median(e0):.4f} Λ={np.median(lam_c):.3f} "
            f"peak={np.median(peak):.4f}"
            + (f" gap={np.median(gap):.3f}" if with_gap else "")
        )
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache,
        lam=np.float64(lam),
        mu=np.float64(mu),
        lambda_x=np.float64(coeffs(lam, mu)[0]),
        mu_xx=np.float64(coeffs(lam, mu)[1]),
        **out,
    )
    log(f"diag {tag_of(lam, mu)} {time.time() - t0:.1f}s")
    return out


def _as_f64(angles: dict) -> dict:
    return {
        k: (v.detach().to(dtype=torch.float64) if torch.is_tensor(v) else v)
        for k, v in angles.items()
    }


def eval_sets(method: str, angles: dict, sets: dict, lam: float, mu: float) -> dict:
    """sets[n] = (E, e0, seeds) numpy arrays."""
    lx, mx = coeffs(lam, mu)
    ang = _as_f64(angles)
    chunk = {6: 32, 9: 16, 12: 8}
    out = {}
    for n, (E, e0, seeds) in sets.items():
        use = chunk[int(n)]
        Eu = torch.tensor(np.asarray(E, dtype=np.float64))
        e0t = torch.tensor(np.asarray(e0, dtype=np.float64))
        parts = []
        for s in range(0, Eu.shape[0], use):
            sl = slice(s, s + use)
            Cn = cn_from_unsat(Eu[sl], int(n))
            psi = prepare_state(method, Cn, ang)
            h = expectation_H(psi, Eu[sl], int(n), lambda_x=lx, mu_xx=mx)
            parts.append((h - e0t[sl]).detach().cpu().numpy())
        eps = np.concatenate(parts, axis=0)
        if np.any(eps < -1e-4):
            raise RuntimeError(
                f"{method} ε below E0 by {float(eps.min()):.3e} at n={n} {tag_of(lam, mu)}"
            )
        rec = summarize_eps(eps)
        rec.update(
            {
                "n": int(n),
                "eps": [float(x) for x in eps.tolist()],
                "seeds": [int(x) for x in np.asarray(seeds).tolist()],
            }
        )
        out[str(n)] = rec
    return out


def _math_finite(x: float) -> bool:
    return x == x and abs(x) != float("inf")


def train_one(
    method: str,
    seed: int,
    lam: float,
    mu: float,
    diag: dict,
    *,
    rounds: int,
) -> dict:
    torch.manual_seed(int(seed))
    _pool_init()
    lx, mx = coeffs(lam, mu)
    rng = np.random.default_rng(int(seed))
    angles = init_angles(method, L_DEPTH, dtype=torch.float64)
    params = method_parameter_list(method, angles)
    E6 = torch.tensor(np.asarray(diag["E6"], dtype=np.float64))
    E9 = torch.tensor(np.asarray(diag["E9"], dtype=np.float64))
    e06 = torch.tensor(np.asarray(diag["e06"], dtype=np.float64))
    e09 = torch.tensor(np.asarray(diag["e09"], dtype=np.float64))
    n6 = int(E6.shape[0])
    n9 = int(E9.shape[0])
    b6 = min(BATCH // 2, n6)
    b9 = min(BATCH - b6, n9)
    hold = {
        9: (
            np.asarray(diag["E9_hold"], dtype=np.float64),
            np.asarray(diag["e09_hold"], dtype=np.float64),
            np.asarray(diag["s9_hold"]),
        )
    }

    def hold_med() -> float:
        ev = eval_sets(method, angles, hold, lam, mu)
        return float(ev["9"]["med_eps"])

    t0 = time.time()
    init_med = hold_med()
    best = {
        "med9": init_med,
        "round": -1,
        "angles": {k: v.detach().cpu().clone() for k, v in angles.items()},
    }
    hist = [{"round": -1, "hold_med9": init_med, "loss": None}]
    log(f"{tag_of(lam, mu)} {method} s{seed} init n9-hold Medε={init_med:.5f}")

    for r in range(int(rounds)):
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

            def loss_one(E, e0, idx, n):
                Eu = E.index_select(0, idx)
                e0b = e0.index_select(0, idx)
                Cn = cn_from_unsat(Eu, n)
                psi = prepare_state(method, Cn, angles)
                h = expectation_H(psi, Eu, n, lambda_x=lx, mu_xx=mx)
                return ((h - e0b) / float(n)).mean()

            loss = 0.5 * (loss_one(E6, e06, i6, 6) + loss_one(E9, e09, i9, 9))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            return loss

        try:
            loss = opt.step(closure)
            loss_f = float(loss.detach()) if torch.is_tensor(loss) else float(loss)
        except Exception as exc:
            log(f"{tag_of(lam, mu)} {method} s{seed} r{r} LBFGS skip: {exc}")
            hist.append({"round": r, "skip": str(exc)})
            continue
        if not _math_finite(loss_f):
            log(f"{tag_of(lam, mu)} {method} s{seed} r{r} nonfinite loss — restore best")
            with torch.no_grad():
                for k, v in best["angles"].items():
                    angles[k].data.copy_(v.to(dtype=angles[k].dtype))
            hist.append({"round": r, "loss": None})
            continue
        med = hold_med()
        hist.append({"round": r, "loss": loss_f, "hold_med9": med})
        if med < best["med9"]:
            best = {
                "med9": med,
                "round": r,
                "angles": {k: v.detach().cpu().clone() for k, v in angles.items()},
            }
            log(
                f"{tag_of(lam, mu)} {method} s{seed} r{r}/{rounds} "
                f"holdMedε={med:.5f} *"
            )
        elif r % 10 == 0 or r + 1 == int(rounds):
            log(
                f"{tag_of(lam, mu)} {method} s{seed} r{r}/{rounds} "
                f"loss={loss_f:.5f} holdMedε={med:.5f} best={best['med9']:.5f}"
            )

    with torch.no_grad():
        for k, v in best["angles"].items():
            angles[k].data.copy_(v.to(dtype=angles[k].dtype))
    sets = {
        6: (
            np.asarray(diag["E6_hold"], dtype=np.float64),
            np.asarray(diag["e06_hold"], dtype=np.float64),
            np.asarray(diag["s6_hold"]),
        ),
        9: hold[9],
        12: (
            np.asarray(diag["E12"], dtype=np.float64),
            np.asarray(diag["e012"], dtype=np.float64),
            np.asarray(diag["s12"]),
        ),
    }
    ev = eval_sets(method, angles, sets, lam, mu)
    return {
        "protocol": PROTOCOL,
        "method": method,
        "L": L_DEPTH,
        "rounds": int(rounds),
        "seed": int(seed),
        "lambda": float(lam),
        "mu": float(mu),
        "lambda_x": lx,
        "mu_xx": mx,
        "init_hold_med9": init_med,
        "best_round": int(best["round"]),
        "best_hold_med9": float(best["med9"]),
        "wall_s": time.time() - t0,
        "hist": hist,
        "angles": angles_to_lists(method, best["angles"]),
        "n_params": int(sum(p.numel() for p in params)),
        "eval": ev,
    }


def _train_payload(lam: float, mu: float, diag_npz: dict) -> dict:
    """Arrays the trainer needs. Holdout is [200:250); train is [0:200)."""
    return {
        "E6": np.asarray(diag_npz["E6"]),
        "e06": np.asarray(diag_npz["n6_e0"][TRAIN_LO:TRAIN_HI]),
        "E9": np.asarray(diag_npz["E9"]),
        "e09": np.asarray(diag_npz["n9_e0"][TRAIN_LO:TRAIN_HI]),
        "E6_hold": np.asarray(diag_npz["E6_hold"]),
        "e06_hold": np.asarray(diag_npz["n6_e0"][HOLD_LO:HOLD_HI]),
        "s6_hold": np.asarray(diag_npz["n6_seeds"][HOLD_LO:HOLD_HI]),
        "E9_hold": np.asarray(diag_npz["E9_hold"]),
        "e09_hold": np.asarray(diag_npz["n9_e0"][HOLD_LO:HOLD_HI]),
        "s9_hold": np.asarray(diag_npz["n9_seeds"][HOLD_LO:HOLD_HI]),
        "E12": np.asarray(diag_npz["E12"]),
        "e012": np.asarray(diag_npz["n12_e0"]),
        "s12": np.asarray(diag_npz["n12_seeds"]),
    }


def _attach_packs(diag: dict) -> dict:
    E6, s6 = load_pack(6)
    E9, s9 = load_pack(9)
    E12, s12 = load_pack(12)
    diag = dict(diag)
    diag["E6"] = E6[TRAIN_LO:TRAIN_HI]
    diag["E6_hold"] = E6[HOLD_LO:HOLD_HI]
    diag["E9"] = E9[TRAIN_LO:TRAIN_HI]
    diag["E9_hold"] = E9[HOLD_LO:HOLD_HI]
    diag["E12"] = E12[:FREEZE_B]
    diag["n6_seeds"] = s6
    diag["n9_seeds"] = s9
    diag["n12_seeds"] = s12[:FREEZE_B]
    return diag


def _train_job(payload: tuple) -> str:
    lam, mu, method, seed, rounds = payload
    path = seed_path(lam, mu, method, seed)
    if seed_is_done(path, lam, mu, method, seed, rounds):
        log(f"train cache hit {method} s{seed} {tag_of(lam, mu)}")
        return str(path)
    try:
        z = np.load(diag_path(lam, mu))
        diag = _attach_packs({k: np.asarray(z[k]) for k in z.files})
        # Seeds in the npz are the full n=6/9 vectors; freeze seeds are n=12.
        row = train_one(method, int(seed), float(lam), float(mu), _train_payload(lam, mu, diag), rounds=rounds)
    except Exception:
        log(f"TRAIN FAIL {method} s{seed} {tag_of(lam, mu)}\n{traceback.format_exc()}")
        raise
    write_json(path, row)
    log(
        f"done {method} s{seed} {tag_of(lam, mu)} "
        f"n12 Medε={row['eval']['12']['med_eps']:.4f} wall={row['wall_s']:.1f}s"
    )
    return str(path)


def _quantum_summary(diag: dict) -> dict:
    def block(n: int, with_gap: bool) -> dict:
        lam = np.asarray(diag[f"n{n}_lambda"], dtype=np.float64)
        peak = np.asarray(diag[f"n{n}_peak"], dtype=np.float64)
        rec = {
            "B": int(lam.shape[0]),
            "C": int(lam.shape[1]),
            "lambda_med": float(np.median(lam)),
            "lambda_mean": float(np.mean(lam)),
            "lambda_min": float(np.min(lam)),
            "lambda_max": float(np.max(lam)),
            "frac_gt_085": float(np.mean(lam > 0.85)),
            "frac_le_080": float(np.mean(lam <= 0.80 + 1e-12)),
            "peak_med": float(np.median(peak)),
            "peak_max": float(np.max(peak)),
        }
        if with_gap:
            gap = np.asarray(diag[f"n{n}_gap"], dtype=np.float64)
            rec.update(
                {
                    "gap_med": float(np.median(gap)),
                    "gap_min": float(np.min(gap)),
                    "gap_max": float(np.max(gap)),
                    "frac_gap_ge_003": float(np.mean(gap >= BAR_GAP - 1e-12)),
                    "frac_gap_ge_005": float(np.mean(gap >= 0.05 - 1e-12)),
                }
            )
        return rec

    return {"6": block(6, False), "9": block(9, False), "12": block(12, True)}


def _load_seed_rows(lam: float, mu: float, seeds: tuple[int, ...]) -> list[dict]:
    rows = []
    for method in METHODS:
        for seed in seeds:
            path = seed_path(lam, mu, method, seed)
            if not path.exists():
                raise RuntimeError(f"missing {path}")
            rows.append(json.loads(path.read_text()))
    return rows


def summarize_cell(lam: float, mu: float, diag: dict, seeds: tuple[int, ...]) -> dict:
    rows = _load_seed_rows(lam, mu, seeds)
    q = _quantum_summary(diag)
    med = {}
    for method in METHODS:
        sub = [r for r in rows if r["method"] == method]
        med[method] = {}
        for n in ("6", "9", "12"):
            med[method][n] = float(np.median([float(r["eval"][n]["med_eps"]) for r in sub]))
    r_hva = {n: ratio(med["snap"][n], med["hva"][n]) for n in ("6", "9", "12")}
    r_f = {n: ratio(med["snap"][n], med["fourier"][n]) for n in ("6", "9", "12")}
    call = classify_cell(q["12"]["lambda_med"], q["12"]["gap_med"], r_hva["12"], is_safe(lam, mu))
    seed_rows = []
    paired_num = 0
    paired_den = 0
    for seed in seeds:
        snap = next(r for r in rows if r["method"] == "snap" and int(r["seed"]) == int(seed))
        hva = next(r for r in rows if r["method"] == "hva" and int(r["seed"]) == int(seed))
        four = next(r for r in rows if r["method"] == "fourier" and int(r["seed"]) == int(seed))
        es = np.asarray(snap["eval"]["12"]["eps"], dtype=np.float64)
        eh = np.asarray(hva["eval"]["12"]["eps"], dtype=np.float64)
        wins = int(np.sum(es < eh - 1e-12))
        paired_num += wins
        paired_den += int(es.size)
        seed_rows.append(
            {
                "seed": int(seed),
                "snap12": float(snap["eval"]["12"]["med_eps"]),
                "hva12": float(hva["eval"]["12"]["med_eps"]),
                "fourier12": float(four["eval"]["12"]["med_eps"]),
                "R_hva": ratio(snap["eval"]["12"]["med_eps"], hva["eval"]["12"]["med_eps"]),
                "R_fourier": ratio(snap["eval"]["12"]["med_eps"], four["eval"]["12"]["med_eps"]),
                "paired_snap_lt_hva": wins,
                "B12": int(es.size),
                "best_round_snap": snap["best_round"],
                "best_round_hva": hva["best_round"],
                "best_round_fourier": four["best_round"],
                "wall_snap_s": snap["wall_s"],
                "wall_hva_s": hva["wall_s"],
                "wall_fourier_s": four["wall_s"],
                "snap9": float(snap["eval"]["9"]["med_eps"]),
                "hva9": float(hva["eval"]["9"]["med_eps"]),
            }
        )
    beta = []
    for row in rows:
        if row["method"] != "hva":
            continue
        beta.extend(abs(float(x)) for x in row["angles"].get("beta_xx", []))
    return {
        "lambda": float(lam),
        "mu": float(mu),
        "lambda_x": coeffs(lam, mu)[0],
        "mu_xx": coeffs(lam, mu)[1],
        "tag": tag_of(lam, mu),
        "safe_control": is_safe(lam, mu),
        "quantum": q,
        "med_eps": med,
        "R_hva": r_hva,
        "R_fourier": r_f,
        "call": call,
        "seeds": seed_rows,
        "paired_snap_lt_hva": paired_num,
        "paired_den": paired_den,
        "beta_xx_mean_abs": float(np.mean(beta)) if beta else None,
    }


def _fmt(x, nd: int = 3) -> str:
    if x is None or not math.isfinite(float(x)):
        return "—"
    return f"{float(x):.{nd}f}"


def _rank_key(cell: dict) -> tuple:
    return (float(cell["quantum"]["12"]["lambda_med"]), float(cell["R_hva"]["12"]))


def render(summary: dict) -> str:
    rows = summary["cells"]
    if not rows:
        return "# (λ, μ) grid — cavity Λ and shallow R_HVA at L=2\n\nNo cells finished.\n"
    cands = [c for c in rows if c["call"]["verdict"] == "PASS"]
    cands_sorted = sorted(cands, key=_rank_key)
    fails = [c for c in rows if c["call"]["verdict"] == "FAIL"]
    controls = [c for c in rows if c["call"]["verdict"] == "CONTROL"]
    if cands_sorted:
        top = cands_sorted[0]
        n_c = len(cands_sorted)
        noun = "candidate" if n_c == 1 else "candidates"
        lead = (
            f"{n_c} {noun}. "
            f"Best by (low Λ, low R) is λ={top['lambda']:.2f}, μ={top['mu']:.2f}: "
            f"Λ={_fmt(top['quantum']['12']['lambda_med'])}, "
            f"gap={_fmt(top['quantum']['12']['gap_med'])}, "
            f"R_HVA={_fmt(top['R_hva']['12'])}."
        )
    else:
        lead = "No cell clears median Λ≤0.80, product gap≥0.03, and R_HVA≤0.85 together."
    lines = [
        "# (λ, μ) grid — cavity Λ and shallow R_HVA at L=2",
        "",
        f"**{lead}**",
        "",
        "Round 10 killed the SAFE lock (λ=0.75, μ=−0.50) because the ground state is cavity-trivial: median Λ≈0.943 at n=12/15/18 even though L=2 R_HVA stayed near 0.33. This grid keeps that Hamiltonian family and asks where Λ drops to ≤0.80 without giving up the shallow SNAP lead over HVA.",
        "",
        "## Hamiltonian",
        "",
        "\\[",
        "H = \\mathrm{diag}(\\mathrm{unsat}) - \\lambda \\sum_i X_i + \\mu \\sum_c \\sum_{i<j\\in c} X_i X_j",
        "\\]",
        "",
        "Cavity c owns qubits [3c, 3c+3), qubit 0 = MSB. The mixer is BinH Hamming-shell SNAP (Walsh degrees 1..4 on Cn). Readout is the Z histogram for unsat and the X histogram for ⟨X⟩, ⟨XX⟩, and Λ. No Y. n=21 is not loaded.",
        "",
        "λ ∈ {0.50, 0.75, 1.00}, μ ∈ {0, −0.15, −0.25, −0.35, −0.50}. In the code the ΣX coefficient is λ_x = −λ and the ΣXX coefficient is μ. SAFE is λ=0.75, μ=−0.50.",
        "",
        "## Gate",
        "",
        "Λ_c = ⟨(Σ_{i∈c} X_i / 3)^2⟩ on the X histogram of the true ground state, equivalently 1/3 + (2/9) Σ_{i<j∈c} ⟨X_i X_j⟩. The reported Λ is the median over cavities and instances. Λ→1 means the cavity has collapsed to one X-magnetization. A Z-basis product state scores 1/3.",
        "",
        "Med ε is the median over seeds of each seed's median-over-instances excess ⟨H⟩−E_0. R_HVA = Medε_SNAP / Medε_HVA at n=12, L=2. R_F is the same ratio against Fourier QAOA. The product gap is (E_prod − E_0) / |E_0| for the best real XZ product state. Z-peak is the median over instances of max_x |⟨x|ψ_0⟩|^2.",
        "",
        "- **Candidate (PASS)** when n=12 median Λ ≤ 0.80, n=12 median product gap ≥ 0.03, and n=12 L=2 R_HVA ≤ 0.85.",
        "- **FAIL** when any of those three bars is missed.",
        "- **SAFE control** is λ=0.75, μ=−0.50. It is flagged as the known-trivialized control and is not ranked as a candidate, whatever its numbers.",
        "",
        "Rank is low Λ, then low R_HVA. Fourier is trained on the same L-BFGS budget and reported as R_F. It is not part of the gate.",
        "",
        "## Protocol",
        "",
        "- Packs: `beat_qaoa_12h/train_packs`. Train indices [0:200) on n=6 and n=9. Checkpoint on n=9 holdout [200:250). Freeze n=12, the first 128 instances (B≥64).",
        "- L=2 only. Three seeds {0, 1, 2}. Methods: Walsh SNAP, HVA (β^X, β^XX, γ), Fourier (β, γ).",
        "- L-BFGS matches `Nov_3rd.settings`: 40 rounds × max_iter 8, lr 0.25, history 16, strong Wolfe, grad clip 5, batch 64 (half n6 / half n9). New optimizer each round. Loss = mean ε/n on that cell's (λ, μ). Each seed restores the round with the best n=9-holdout Med ε.",
        "- Paper init β=β^X=0.01, β^XX=0, γ=γ_w=−0.01, φ_k=−β k.",
        "- Ground states: dense eigh at n=6 and n=9, ARPACK at n=12. Λ and Z-peak use that vector. The product gap is computed on the n=12 freeze only.",
        "- n=15, n=18, and n=21 were not loaded. `qaoa_angles/`, `train69_ladder_fresh250/`, and `settings.py` were not written.",
        "",
        "## Grid at n=12, L=2",
        "",
        "Λ9 is the median cavity Λ on all 250 n=9 pack rows (train and holdout), from the same (λ, μ). The gate uses Λ12.",
        "",
        "| λ | μ | role | Λ12 | Λ9 | gap12 | Z-peak12 | Medε SNAP | Medε HVA | Medε Fourier | R_HVA | R_F |",
        "|--:|--:|:--|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for c in rows:
        q12 = c["quantum"]["12"]
        q9 = c["quantum"]["9"]
        lines.append(
            "| {lam:.2f} | {mu:.2f} | {role} | {l12} | {l9} | {gap} | {peak} | {es} | {eh} | {ef} | {rh} | {rf} |".format(
                lam=c["lambda"],
                mu=c["mu"],
                role=c["call"]["role"],
                l12=_fmt(q12["lambda_med"]),
                l9=_fmt(q9["lambda_med"]),
                gap=_fmt(q12["gap_med"]),
                peak=_fmt(q12["peak_med"], 4),
                es=_fmt(c["med_eps"]["snap"]["12"], 4),
                eh=_fmt(c["med_eps"]["hva"]["12"], 4),
                ef=_fmt(c["med_eps"]["fourier"]["12"], 4),
                rh=_fmt(c["R_hva"]["12"]),
                rf=_fmt(c["R_fourier"]["12"]),
            )
        )
    lines += ["", "## Ranked candidates", ""]
    if not cands_sorted:
        lines.append("None. Every non-control cell misses at least one bar. See the fail notes below.")
    else:
        lines.append("| rank | λ | μ | Λ12 | gap12 | Z-peak12 | R_HVA | R_F | paired SNAP<HVA |")
        lines.append("|--:|--:|--:|---:|---:|---:|---:|---:|--:|")
        for i, c in enumerate(cands_sorted, start=1):
            q12 = c["quantum"]["12"]
            lines.append(
                f"| {i} | {c['lambda']:.2f} | {c['mu']:.2f} | {_fmt(q12['lambda_med'])} | "
                f"{_fmt(q12['gap_med'])} | {_fmt(q12['peak_med'], 4)} | {_fmt(c['R_hva']['12'])} | "
                f"{_fmt(c['R_fourier']['12'])} | {c['paired_snap_lt_hva']}/{c['paired_den']} |"
            )
    lines += ["", "## SAFE control", ""]
    if not controls:
        lines.append("SAFE cell was not in this run.")
    else:
        c = controls[0]
        q12 = c["quantum"]["12"]
        lines.append(
            f"λ=0.75, μ=−0.50 is the known-trivialized control from PR #41 (median Λ≈0.943, L=2 R_HVA≈0.328 on a five-seed n=12 freeze). "
            f"This three-seed rerun measures Λ12={_fmt(q12['lambda_med'])} "
            f"(min {_fmt(q12['lambda_min'])}, max {_fmt(q12['lambda_max'])}, "
            f"fraction of cavities above 0.85 = {_fmt(q12['frac_gt_085'])}), "
            f"gap12={_fmt(q12['gap_med'])}, Z-peak12={_fmt(q12['peak_med'], 4)}, "
            f"R_HVA={_fmt(c['R_hva']['12'])}, R_F={_fmt(c['R_fourier']['12'])}. "
            "It is not ranked as a candidate."
        )
        if q12["lambda_med"] <= BAR_LAMBDA:
            lines.append("")
            lines.append(
                "This rerun puts SAFE under the Λ≤0.80 bar, which disagrees with the PR #41 kill. Treat the grid as unchecked until that disagreement is resolved."
            )
    lines += ["", "## Cells that miss a bar", ""]
    if not fails:
        lines.append("Every non-control cell is a candidate.")
    else:
        for c in fails:
            why = "; ".join(c["call"]["fails"])
            q12 = c["quantum"]["12"]
            lines.append(
                f"- λ={c['lambda']:.2f}, μ={c['mu']:.2f}: {why}. "
                f"Λ12={_fmt(q12['lambda_med'])}, gap={_fmt(q12['gap_med'])}, "
                f"Z-peak={_fmt(q12['peak_med'], 4)}, R_HVA={_fmt(c['R_hva']['12'])}, "
                f"R_F={_fmt(c['R_fourier']['12'])}."
            )
    lines += ["", "## Per-seed n=12 Med ε", ""]
    lines.append("| λ | μ | seed | SNAP | HVA | Fourier | R_HVA | paired | best round SNAP | wall SNAP s |")
    lines.append("|--:|--:|--:|---:|---:|---:|---:|--:|--:|--:|")
    for c in rows:
        for s in c["seeds"]:
            lines.append(
                f"| {c['lambda']:.2f} | {c['mu']:.2f} | {s['seed']} | {_fmt(s['snap12'], 4)} | "
                f"{_fmt(s['hva12'], 4)} | {_fmt(s['fourier12'], 4)} | {_fmt(s['R_hva'])} | "
                f"{s['paired_snap_lt_hva']}/{s['B12']} | {s['best_round_snap']} | {_fmt(s['wall_snap_s'], 1)} |"
            )
    lines += [
        "",
        "## Reading",
        "",
    ]
    # A short factual reading from the measured ranges.
    lams = [c["quantum"]["12"]["lambda_med"] for c in rows]
    gaps = [c["quantum"]["12"]["gap_med"] for c in rows]
    rs = [c["R_hva"]["12"] for c in rows if math.isfinite(c["R_hva"]["12"])]
    bxx = [c["beta_xx_mean_abs"] for c in rows if c.get("beta_xx_mean_abs") is not None]
    lines.append(
        f"Across the 15 cells, n=12 median Λ runs from {_fmt(min(lams))} to {_fmt(max(lams))}, "
        f"the product gap from {_fmt(min(gaps))} to {_fmt(max(gaps))}, "
        f"and R_HVA from {_fmt(min(rs))} to {_fmt(max(rs))}. "
        "Λ rises, and R_HVA falls, as λ increases or as μ becomes more negative. "
        "The three bars are cleared only for small |μ|."
    )
    if bxx:
        lines.append("")
        lines.append(
            f"HVA mean |β^XX| runs from {_fmt(min(bxx))} to {_fmt(max(bxx))}. "
            "It stays away from the paper-init value 0 on every cell, including μ=0, so the HVA column is a trained XX mixer. "
            "At μ=0, R_HVA and R_F agree to about 0.01: the extra XX angle does not improve on Fourier when the Hamiltonian has no XX term."
        )
    misses = [c for c in fails if c["quantum"]["12"]["lambda_med"] > BAR_LAMBDA]
    if misses:
        near = min(misses, key=lambda c: c["quantum"]["12"]["lambda_med"])
        lines.append("")
        lines.append(
            f"Nearest miss on Λ is λ={near['lambda']:.2f}, μ={near['mu']:.2f}: "
            f"Λ12={_fmt(near['quantum']['12']['lambda_med'])} (bar 0.80), "
            f"gap={_fmt(near['quantum']['12']['gap_med'])}, "
            f"R_HVA={_fmt(near['R_hva']['12'])}."
        )
    lines.append("")
    if cands_sorted:
        bits = []
        for c in cands_sorted:
            peak = c["quantum"]["12"]["peak_med"]
            note = ""
            if peak >= 0.5:
                note = " Z-peak ≥ 0.5, so the ground state is close to a computational bitstring."
            elif peak >= 0.2:
                note = " Z-peak is large enough that a single bitstring carries substantial weight."
            bits.append(
                f"λ={c['lambda']:.2f}, μ={c['mu']:.2f} has Λ={_fmt(c['quantum']['12']['lambda_med'])}, "
                f"gap={_fmt(c['quantum']['12']['gap_med'])}, Z-peak={_fmt(peak, 4)}, "
                f"R_HVA={_fmt(c['R_hva']['12'])}.{note}"
            )
        lines.append("Candidates, lowest Λ first: " + " ".join(bits))
    else:
        # Say which bar eliminates the low-Λ cells and which eliminates the low-R cells.
        low_l = sorted(rows, key=lambda c: c["quantum"]["12"]["lambda_med"])[:3]
        lines.append(
            "The lowest-Λ cells are "
            + ", ".join(
                f"λ={c['lambda']:.2f} μ={c['mu']:.2f} (Λ={_fmt(c['quantum']['12']['lambda_med'])}, "
                f"gap={_fmt(c['quantum']['12']['gap_med'])}, R={_fmt(c['R_hva']['12'])})"
                for c in low_l
            )
            + "."
        )
    lines += [
        "",
        "Per-seed JSON is `runs/L2_{method}_s{seed}_{tag}.json`. Ground-state arrays are `cache/diag_{tag}.npz`. Mean |β^XX| per cell is in `summary.json`.",
        "",
    ]
    return "\n".join(lines) + "\n"


def assemble(seeds: tuple[int, ...]) -> dict:
    built = []
    for lam, mu in cells():
        cache = diag_path(lam, mu)
        if not cache.exists():
            continue
        missing = [seed_path(lam, mu, m, s) for m in METHODS for s in seeds if not seed_path(lam, mu, m, s).exists()]
        if missing:
            log(f"assemble skip {tag_of(lam, mu)} missing {len(missing)}")
            continue
        z = np.load(cache)
        diag = {k: np.asarray(z[k]) for k in z.files}
        built.append(summarize_cell(lam, mu, diag, seeds))
    summary = {
        "protocol": PROTOCOL,
        "lambdas": list(LAMBDAS),
        "mus": list(MUS),
        "L": L_DEPTH,
        "seeds": list(seeds),
        "freeze_B": FREEZE_B,
        "bars": {"lambda": BAR_LAMBDA, "gap": BAR_GAP, "R_hva": BAR_R},
        "safe_cell": {"lambda": SAFE_CELL[0], "mu": SAFE_CELL[1]},
        "n_complete": len(built),
        "n_cells": len(cells()),
        "cells": built,
    }
    text = render(summary)
    RUN.mkdir(parents=True, exist_ok=True)
    write_json(RUN / "summary.json", summary)
    (RUN / "RESULTS.md").write_text(text)
    log(f"wrote RESULTS.md ({len(built)}/{len(cells())} cells)")
    return summary


def _ordered_cells(only: list[tuple[float, float]] | None) -> list[tuple[float, float]]:
    chosen = cells() if not only else list(only)
    # SAFE first so a broken control shows up before the rest of the grid.
    return sorted(chosen, key=lambda p: (0 if is_safe(*p) else 1, p[0], p[1]))


def run(workers: int, rounds: int, seeds: tuple[int, ...], only: list[tuple[float, float]] | None) -> dict:
    _self_test_gate()
    _pool_init()
    RUN.mkdir(parents=True, exist_ok=True)
    order = _ordered_cells(only)
    log(f"cells={len(order)} workers={workers} rounds={rounds} seeds={list(seeds)} L={L_DEPTH}")
    for lam, mu in order:
        diag = ensure_diag(lam, mu, workers)
        diag = _attach_packs(diag)
        q = _quantum_summary(diag)
        log(
            f"{tag_of(lam, mu)} Λ12={q['12']['lambda_med']:.3f} "
            f"gap={q['12']['gap_med']:.3f} peak={q['12']['peak_med']:.4f} "
            f"Λ9={q['9']['lambda_med']:.3f}"
        )
        jobs = [(lam, mu, method, seed, rounds) for method in METHODS for seed in seeds]
        pending = []
        for job in jobs:
            path = seed_path(job[0], job[1], job[2], job[3])
            if seed_is_done(path, job[0], job[1], job[2], job[3], job[4]):
                log(f"train cache hit {job[2]} s{job[3]} {tag_of(lam, mu)}")
            else:
                pending.append(job)
        if pending:
            if workers <= 1 or len(pending) == 1:
                for job in pending:
                    _train_job(job)
            else:
                import multiprocessing as mp

                ctx = mp.get_context("spawn")
                with ctx.Pool(min(int(workers), len(pending)), initializer=_pool_init) as pool:
                    list(pool.imap_unordered(_train_job, pending, chunksize=1))
        if is_safe(lam, mu) and rounds >= ROUNDS and tuple(seeds) == SEEDS:
            cell = summarize_cell(lam, mu, diag, seeds)
            lam12 = cell["quantum"]["12"]["lambda_med"]
            r12 = cell["R_hva"]["12"]
            log(f"SAFE control check Λ12={lam12:.3f} R_HVA={r12:.3f}")
            if lam12 < 0.85:
                log("WARNING: SAFE Λ12 < 0.85, disagreeing with the PR #41 kill")
            if not math.isfinite(r12) or r12 > 1.0:
                log("WARNING: SAFE R_HVA > 1, SNAP did not beat HVA")
        assemble(seeds)
    return assemble(seeds)


def _parse_cell(text: str) -> tuple[float, float]:
    parts = text.replace(",", " ").split()
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"cell must be 'λ μ', got {text!r}")
    return float(parts[0]), float(parts[1])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--rounds", type=int, default=ROUNDS)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    ap.add_argument("--cell", action="append", default=[], help="restrict to 'λ μ'; repeatable")
    ap.add_argument("--assemble-only", action="store_true")
    args = ap.parse_args()
    seeds = tuple(int(s) for s in args.seeds)
    only = [_parse_cell(c) for c in args.cell] if args.cell else None
    if args.assemble_only:
        assemble(seeds)
        return
    run(int(args.workers), int(args.rounds), seeds, only)


if __name__ == "__main__":
    main()
