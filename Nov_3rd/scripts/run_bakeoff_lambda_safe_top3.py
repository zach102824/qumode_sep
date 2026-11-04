#!/usr/bin/env python3
"""Matched bakeoff of the three Λ≤0.80 cells.

Walsh Hamming-shell SNAP versus the fair baseline (Fourier when μ=0, HVA
when μ≠0). Same train packs, same five seeds, same L-BFGS budget, L=2 and
L=4. Freeze n=12. Probe n=18 with the ground-state cavity Λ, the product
gap, and the Z-peak. Does not load n=21.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

# One thread per process. Pools own the parallelism. Set before numpy/torch.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from Nov_3rd import settings as S
from Nov_3rd.bakeoff_lambda_safe import (
    BAR_R,
    CANDIDATES,
    N_SEEDS,
    PRIORS,
    assert_n_in_scope,
    decision_self_test,
    family_verdict,
    official_r,
    rank_cells,
    recommend,
    render,
    seed_ratios,
)
from Nov_3rd.bakeoff_lambda_safe import classify_candidate as classify_bars
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
    self_check,
    snap_state_from_components,
    summarize_eps,
    walsh_degree_components,
)

RUN = S.OUT / "quantum_h_explore" / "bakeoff_Lambda_safe_top3"
PACK = S.OUT / "beat_qaoa_12h" / "train_packs"
TRAIN_LO, TRAIN_HI = 0, 200
HOLD_LO, HOLD_HI = 200, 250
FREEZE_B = 128
N18_B = 32
N18_BASE = 61_000_000_000
DEPTHS = (2, 4)
SEEDS = (0, 1, 2, 3, 4)
ROUNDS = 40
BATCH = 64
PROTOCOL = "bakeoff_lambda_safe_top3_r12"


def log(*a):
    print(time.strftime("%H:%M:%S"), "[bakeoff]", *a, flush=True)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(path)


def coeffs(lam: float, mu: float) -> tuple[float, float]:
    return -float(lam), float(mu)


def candidate_by_id(cid: str) -> dict:
    for spec in CANDIDATES:
        if spec["id"] == cid:
            return spec
    raise KeyError(cid)


def load_pack(n: int, count: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    assert_n_in_scope(n)
    if int(n) == 18:
        raise RuntimeError("n=18 is the generated probe, not a train pack")
    path = PACK / f"train_n{n}.npz"
    d = np.load(path)
    E = np.asarray(d["train_E"], dtype=np.float64)
    seeds = np.asarray(d["train_seeds"]).astype(np.int64)
    if E.ndim != 2 or E.shape[1] != (1 << int(n)):
        raise RuntimeError(f"n{n} E shape {E.shape}")
    if count is not None:
        E = E[: int(count)]
        seeds = seeds[: int(count)]
    return E, seeds


def probe_path() -> Path:
    return RUN / "probe_n18.npz"


def ensure_n18_probe(count: int) -> tuple[np.ndarray, np.ndarray]:
    """First `count` satisfiable n=18 formulas at the beat-pack base."""
    assert_n_in_scope(18)
    path = probe_path()
    if path.exists():
        z = np.load(path)
        if (
            int(z["n"]) == 18
            and int(z["base"]) == int(N18_BASE)
            and int(z["train_E"].shape[0]) >= int(count)
            and int(z["train_E"].shape[1]) == (1 << 18)
        ):
            log(f"n18 probe cache hit B={count}")
            return (
                np.asarray(z["train_E"][:count], dtype=np.float64),
                np.asarray(z["train_seeds"][:count]).astype(np.int64),
            )
    from explore_OCT29.lib.sat_gen import instances

    log(f"generating n18 probe B={count} base={N18_BASE}")
    t0 = time.time()
    recs = instances(18, int(count), int(N18_BASE), compute_E=True)
    E = np.stack([np.asarray(r["E"], dtype=np.int16) for r in recs], axis=0)
    seeds = np.asarray([int(r["seed"]) for r in recs], dtype=np.int64)
    if E.shape != (int(count), 1 << 18):
        raise RuntimeError(f"n18 probe shape {E.shape}")
    if np.any(E.min(axis=1) != 0):
        raise RuntimeError("n18 probe contains an unsatisfiable formula")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        n=np.int32(18),
        base=np.int64(N18_BASE),
        train_seeds=seeds,
        train_E=E,
        source="beat_base_satisfiable_prefix",
    )
    log(f"n18 probe {time.time() - t0:.1f}s seeds {int(seeds[0])}..{int(seeds[-1])}")
    return E.astype(np.float64), seeds


def diag_path(cid: str, n: int) -> Path:
    return RUN / "cache" / f"diag_{cid}_n{n}.npz"


def _pool_init() -> None:
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    torch.set_num_threads(1)


def _diag_job(payload: tuple) -> tuple:
    row, n, lam, mu, seed, with_gap = payload
    lx, mx = coeffs(lam, mu)
    bundle = ground_bundle(np.asarray(row, dtype=np.float64), int(n), lambda_x=lx, mu_xx=mx)
    lam_c = np.asarray(lambda_of_state(bundle["psi"], int(n)), dtype=np.float64).reshape(-1)
    if lam_c.size != int(n) // 3:
        raise RuntimeError(f"Λ length {lam_c.size} at n={n}")
    e_prod = float("nan")
    gap = float("nan")
    if with_gap:
        e_prod = best_product_energy(np.asarray(row, dtype=np.float64), int(n), float(lam), float(mu), seed=int(seed))
        gap = product_gap(e_prod, bundle["E0"])
        if e_prod + 1e-5 < float(bundle["E0"]):
            raise RuntimeError(f"product energy {e_prod} below E0 {bundle['E0']} at n={n}")
    return (
        float(bundle["E0"]),
        float(bundle["peak"]),
        float(bundle["residual"]),
        lam_c,
        float(e_prod),
        float(gap),
    )


def _gs_block(rows: list[tuple], with_gap: bool) -> dict:
    e0 = np.array([r[0] for r in rows], dtype=np.float64)
    peak = np.array([r[1] for r in rows], dtype=np.float64)
    resid = np.array([r[2] for r in rows], dtype=np.float64)
    lam_c = np.stack([r[3] for r in rows], axis=0)
    out = {"e0": e0, "peak": peak, "residual": resid, "lambda_c": lam_c}
    if with_gap:
        out["e_prod"] = np.array([r[4] for r in rows], dtype=np.float64)
        out["gap"] = np.array([r[5] for r in rows], dtype=np.float64)
    return out


def _check_diag(block: dict, E: np.ndarray, n: int, cid: str, with_gap: bool) -> None:
    if np.any(block["e0"] > E.min(axis=1) + 1e-5):
        raise RuntimeError(f"E0 above a diagonal entry at n={n} {cid}")
    if np.any(block["residual"] > 1e-5):
        raise RuntimeError(f"ground residual {float(block['residual'].max())} at n={n} {cid}")
    lam_c = block["lambda_c"]
    if np.any(lam_c < -1e-8) or np.any(lam_c > 1.0 + 1e-6):
        raise RuntimeError(f"Λ out of range at n={n} {cid}")
    if with_gap and (np.any(~np.isfinite(block["gap"])) or np.any(block["gap"] < -1e-6)):
        raise RuntimeError(f"product gap invalid at n={n} {cid}")


def _cache_ok(path: Path, lam: float, mu: float, seeds: np.ndarray, with_gap: bool) -> dict | None:
    if not path.exists():
        return None
    z = np.load(path)
    if not (
        abs(float(z["lam"]) - float(lam)) < 1e-12
        and abs(float(z["mu"]) - float(mu)) < 1e-12
        and np.array_equal(z["seeds"], seeds)
        and int(z["e0"].shape[0]) == int(seeds.shape[0])
    ):
        return None
    if with_gap and "gap" not in z.files:
        return None
    return {k: np.asarray(z[k]) for k in z.files}


def ensure_diag(spec: dict, n: int, E: np.ndarray, seeds: np.ndarray, pool, with_gap: bool) -> dict:
    cid = spec["id"]
    lam, mu = float(spec["lambda"]), float(spec["mu"])
    path = diag_path(cid, n)
    hit = _cache_ok(path, lam, mu, seeds, with_gap)
    if hit is not None:
        log(f"diag cache hit {cid} n{n}")
        return hit
    log(f"diag {cid} n{n} B={E.shape[0]} gap={with_gap}")
    t0 = time.time()
    jobs = [
        (np.asarray(E[i], dtype=np.float64), int(n), lam, mu, int(seeds[i]), with_gap)
        for i in range(E.shape[0])
    ]
    if pool is None or len(jobs) == 1 or int(n) <= 6:
        rows = []
        for i, job in enumerate(jobs):
            rows.append(_diag_job(job))
            if (i + 1) == len(jobs) or (i + 1) % 25 == 0:
                log(f"  {cid} n{n} {i+1}/{len(jobs)} E0={rows[-1][0]:.3f} {time.time()-t0:.0f}s")
    else:
        rows = []
        step = 8 if int(n) >= 15 else 25
        for i, rec in enumerate(pool.imap(_diag_job, jobs, chunksize=1)):
            rows.append(rec)
            if (i + 1) == len(jobs) or (i + 1) % step == 0:
                log(f"  {cid} n{n} {i+1}/{len(jobs)} E0={rec[0]:.3f} {time.time()-t0:.0f}s")
    block = _gs_block(rows, with_gap)
    _check_diag(block, E, n, cid, with_gap)
    lx, mx = coeffs(lam, mu)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        lam=np.float64(lam),
        mu=np.float64(mu),
        lambda_x=np.float64(lx),
        mu_xx=np.float64(mx),
        seeds=seeds.astype(np.int64),
        **block,
    )
    med_l = float(np.median(block["lambda_c"]))
    msg = f"diag {cid} n{n} Λ={med_l:.3f} peak={float(np.median(block['peak'])):.4f} {time.time()-t0:.1f}s"
    if with_gap:
        msg += f" gap={float(np.median(block['gap'])):.3f}"
    log(msg)
    block["seeds"] = seeds
    block["lam"] = np.float64(lam)
    block["mu"] = np.float64(mu)
    return block


def quantum_summary(block: dict) -> dict:
    lam = np.asarray(block["lambda_c"], dtype=np.float64)
    peak = np.asarray(block["peak"], dtype=np.float64)
    gap = np.asarray(block["gap"], dtype=np.float64)
    flat = lam.reshape(-1)
    return {
        "B": int(peak.shape[0]),
        "n_cavities": int(lam.shape[1]),
        "lambda_med": float(np.median(flat)),
        "lambda_min": float(np.min(flat)),
        "lambda_max": float(np.max(flat)),
        "frac_gt_080": float(np.mean(flat > 0.80)),
        "frac_gt_085": float(np.mean(flat > 0.85)),
        "gap_med": float(np.median(gap)),
        "gap_min": float(np.min(gap)),
        "peak_med": float(np.median(peak)),
        "frac_peak_ge_05": float(np.mean(peak >= 0.5 - 1e-15)),
    }


def seed_path(L: int, method: str, seed: int, cid: str) -> Path:
    return RUN / "runs" / f"L{int(L)}_{method}_s{int(seed)}_{cid}.json"


def seed_is_done(path: Path, spec: dict, method: str, seed: int, L: int, rounds: int) -> bool:
    if not path.exists():
        return False
    try:
        row = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return (
        row.get("protocol") == PROTOCOL
        and row.get("method") == method
        and row.get("id") == spec["id"]
        and int(row.get("seed", -1)) == int(seed)
        and int(row.get("L", -1)) == int(L)
        and int(row.get("rounds", -1)) == int(rounds)
        and abs(float(row.get("lambda", 0.0)) - float(spec["lambda"])) < 1e-12
        and abs(float(row.get("mu", 1.0)) - float(spec["mu"])) < 1e-12
        and "12" in row.get("eval", {})
    )


def _as_f64(angles: dict) -> dict:
    return {
        k: (v.detach().to(dtype=torch.float64) if torch.is_tensor(v) else v)
        for k, v in angles.items()
    }


def eval_method(method: str, angles: dict, bundle: dict, lam: float, mu: float) -> dict:
    lx, mx = coeffs(lam, mu)
    ang = _as_f64(angles)
    chunk = {6: 32, 9: 16, 12: 8}
    out = {}
    with torch.no_grad():
        for n, (E, e0, seeds) in bundle.items():
            Eu = torch.tensor(np.asarray(E, dtype=np.float64))
            e0t = torch.tensor(np.asarray(e0, dtype=np.float64))
            parts = []
            for s in range(0, Eu.shape[0], chunk[int(n)]):
                sl = slice(s, s + chunk[int(n)])
                Cn = cn_from_unsat(Eu[sl], int(n))
                psi = prepare_state(method, Cn, ang)
                h = expectation_H(psi, Eu[sl], int(n), lambda_x=lx, mu_xx=mx)
                parts.append((h - e0t[sl]).detach().cpu().numpy())
            eps = np.concatenate(parts, axis=0)
            if np.any(eps < -1e-4):
                raise RuntimeError(
                    f"{method} ε below E0 by {float(eps.min()):.3e} at n={n} λ={lam} μ={mu}"
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


def _loss_on(method, angles, Eu, e0b, n: int, lam: float, mu: float, comps) -> torch.Tensor:
    lx, mx = coeffs(lam, mu)
    if method == "snap" and comps is not None:
        psi = snap_state_from_components(comps, angles["gamma_w"], angles["phi"])
    else:
        Cn = cn_from_unsat(Eu, n)
        psi = prepare_state(method, Cn, angles)
    h = expectation_H(psi, Eu, n, lambda_x=lx, mu_xx=mx)
    return ((h - e0b) / float(n)).mean()


def _finite(x: float) -> bool:
    return x == x and abs(x) != float("inf")


def train_one(method: str, L: int, seed: int, spec: dict, packs: dict, rounds: int) -> dict:
    """Matched batch stream: the RNG depends on the seed and not on (λ, μ), method, or L."""
    torch.manual_seed(int(seed))
    rng = np.random.default_rng(int(seed))
    lam, mu = float(spec["lambda"]), float(spec["mu"])
    angles = init_angles(method, int(L), dtype=torch.float64)
    params = method_parameter_list(method, angles)
    E6, e06 = packs["E6"], packs["e06"]
    E9, e09 = packs["E9"], packs["e09"]
    n6, n9 = int(E6.shape[0]), int(E9.shape[0])
    b6 = min(BATCH // 2, n6)
    b9 = min(BATCH - b6, n9)
    hold = {9: (packs["E9_hold"], packs["e09_hold"], packs["s9_hold"])}

    def hold_med() -> float:
        return float(eval_method(method, angles, hold, lam, mu)["9"]["med_eps"])

    t0 = time.time()
    init_med = hold_med()
    best = {
        "med9": init_med,
        "round": -1,
        "angles": {k: v.detach().cpu().clone() for k, v in angles.items()},
    }
    hist = [{"round": -1, "hold_med9": init_med, "loss": None}]
    log(f"{spec['id']} L{L} {method} s{seed} init n9-hold Medε={init_med:.5f}")
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
                _loss_on(method, angles, Eu6, e6b, 6, lam, mu, comps6)
                + _loss_on(method, angles, Eu9, e9b, 9, lam, mu, comps9)
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            return loss

        try:
            loss = opt.step(closure)
            loss_f = float(loss.detach()) if torch.is_tensor(loss) else float(loss)
        except Exception as exc:
            log(f"{spec['id']} L{L} {method} s{seed} r{r} LBFGS skip: {exc}")
            hist.append({"round": r, "skip": str(exc)})
            continue
        if not _finite(loss_f):
            log(f"{spec['id']} L{L} {method} s{seed} r{r} nonfinite — restore best")
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
        if mark or r % 10 == 0 or r + 1 == int(rounds):
            log(
                f"{spec['id']} L{L} {method} s{seed} r{r}/{rounds} "
                f"loss={loss_f:.5f} hold={med:.5f} best={best['med9']:.5f}{mark}"
            )
    with torch.no_grad():
        for k, v in best["angles"].items():
            angles[k].data.copy_(v.to(dtype=angles[k].dtype))
    ev = eval_method(
        method,
        angles,
        {
            6: (packs["E6_hold"], packs["e06_hold"], packs["s6_hold"]),
            9: hold[9],
            12: (packs["E12"], packs["e012"], packs["s12"]),
        },
        lam,
        mu,
    )
    return {
        "protocol": PROTOCOL,
        "id": spec["id"],
        "method": method,
        "baseline": spec["baseline"],
        "L": int(L),
        "seed": int(seed),
        "lambda": lam,
        "mu": mu,
        "lambda_x": coeffs(lam, mu)[0],
        "mu_xx": coeffs(lam, mu)[1],
        "rounds": int(rounds),
        "batch": BATCH,
        "batch_rng": "numpy Generator(seed) shared across cells, methods, and depths",
        "init_hold_med9": init_med,
        "best_round": int(best["round"]),
        "best_hold_med9": float(best["med9"]),
        "wall_s": time.time() - t0,
        "hist": hist,
        "angles": angles_to_lists(method, best["angles"]),
        "n_params": int(sum(p.numel() for p in params)),
        "eval": ev,
    }


def _torch_pack(E: np.ndarray, e0: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.tensor(np.asarray(E, dtype=np.float64)),
        torch.tensor(np.asarray(e0, dtype=np.float64)),
    )


def _train_worker(spec_job: dict) -> str:
    try:
        _pool_init()
        spec = candidate_by_id(spec_job["id"])
        method = spec_job["method"]
        L = int(spec_job["L"])
        seed = int(spec_job["seed"])
        rounds = int(spec_job["rounds"])
        path = seed_path(L, method, seed, spec["id"])
        if seed_is_done(path, spec, method, seed, L, rounds):
            log(f"train cache hit {path.name}")
            return str(path)
        diag6 = np.load(diag_path(spec["id"], 6))
        diag9 = np.load(diag_path(spec["id"], 9))
        diag12 = np.load(diag_path(spec["id"], 12))
        E6, s6 = load_pack(6, HOLD_HI)
        E9, s9 = load_pack(9, HOLD_HI)
        E12, s12 = load_pack(12, FREEZE_B)
        if not np.array_equal(s12, np.asarray(diag12["seeds"])):
            raise RuntimeError(f"n12 seed mismatch for {spec['id']}")
        e06 = np.asarray(diag6["e0"], dtype=np.float64)
        e09 = np.asarray(diag9["e0"], dtype=np.float64)
        E6t, e6t = _torch_pack(E6[TRAIN_LO:TRAIN_HI], e06[TRAIN_LO:TRAIN_HI])
        E9t, e9t = _torch_pack(E9[TRAIN_LO:TRAIN_HI], e09[TRAIN_LO:TRAIN_HI])
        packs = {
            "E6": E6t,
            "e06": e6t,
            "E9": E9t,
            "e09": e9t,
            "E6_hold": E6[HOLD_LO:HOLD_HI],
            "e06_hold": e06[HOLD_LO:HOLD_HI],
            "s6_hold": s6[HOLD_LO:HOLD_HI],
            "E9_hold": E9[HOLD_LO:HOLD_HI],
            "e09_hold": e09[HOLD_LO:HOLD_HI],
            "s9_hold": s9[HOLD_LO:HOLD_HI],
            "E12": E12,
            "e012": np.asarray(diag12["e0"], dtype=np.float64),
            "s12": s12,
        }
        row = train_one(method, L, seed, spec, packs, rounds)
        write_json(path, row)
        log(
            f"done {path.name} n12 Medε={row['eval']['12']['med_eps']:.4f} "
            f"wall={row['wall_s']:.1f}s"
        )
        return str(path)
    except Exception:
        log(f"TRAIN FAIL {spec_job}\n{traceback.format_exc()}")
        raise


def _load_rows(depths: tuple[int, ...], seeds: tuple[int, ...]) -> dict[tuple, dict]:
    rows = {}
    missing = []
    for spec in CANDIDATES:
        for L in depths:
            for method in ("snap", spec["baseline"]):
                for seed in seeds:
                    path = seed_path(L, method, seed, spec["id"])
                    if not seed_is_done(path, spec, method, seed, L, ROUNDS):
                        missing.append(path.name)
                        continue
                    rows[(spec["id"], int(L), method, int(seed))] = json.loads(path.read_text())
    if missing:
        raise RuntimeError("missing seed files: " + ", ".join(missing[:12]))
    return rows


def _paired(snap_rows: list[dict], base_rows: list[dict], n: str) -> tuple[int, int]:
    base_by = {int(r["seed"]): r for r in base_rows}
    win = total = 0
    for srow in snap_rows:
        brow = base_by[int(srow["seed"])]
        es = np.asarray(srow["eval"][n]["eps"], dtype=np.float64)
        eb = np.asarray(brow["eval"][n]["eps"], dtype=np.float64)
        if es.shape != eb.shape:
            raise RuntimeError(f"paired shape {es.shape} vs {eb.shape}")
        win += int(np.sum(es < eb))
        total += int(es.size)
    return win, total


def _depth_block(rows: dict, spec: dict, L: int, seeds: tuple[int, ...]) -> dict:
    snap = [rows[(spec["id"], L, "snap", int(s))] for s in seeds]
    base = [rows[(spec["id"], L, spec["baseline"], int(s))] for s in seeds]
    by_n = {}
    for n in ("6", "9", "12"):
        snap_meds = [float(r["eval"][n]["med_eps"]) for r in snap]
        base_meds = [float(r["eval"][n]["med_eps"]) for r in base]
        win, total = _paired(snap, base, n)
        by_n[n] = {
            "med_snap": float(np.median(np.asarray(snap_meds))),
            "med_base": float(np.median(np.asarray(base_meds))),
            "r": official_r(snap_meds, base_meds),
            "paired_win": win,
            "paired_n": total,
        }
    snap12 = [float(r["eval"]["12"]["med_eps"]) for r in snap]
    base12 = [float(r["eval"]["12"]["med_eps"]) for r in base]
    ratios = seed_ratios(snap12, base12)
    per_seed = []
    for rsnap, rbase, rval in zip(snap, base, ratios):
        per_seed.append(
            {
                "seed": int(rsnap["seed"]),
                "snap": float(rsnap["eval"]["12"]["med_eps"]),
                "base": float(rbase["eval"]["12"]["med_eps"]),
                "r": float(rval),
                "best_round": int(rsnap["best_round"]),
                "wall_s": float(rsnap["wall_s"]),
            }
        )
    n_clear = sum(1 for r in ratios if r <= BAR_R + 1e-12)
    out = {"by_n": by_n, "per_seed": per_seed, "n_clear": int(n_clear), "r_seeds": ratios}
    if spec["baseline"] == "hva":
        abs_b = []
        for rbase in base:
            beta = np.asarray(rbase["angles"]["beta_xx"], dtype=np.float64)
            abs_b.append(float(np.mean(np.abs(beta))))
        out["mean_abs_beta_xx"] = float(np.mean(abs_b))
    return out


def assemble(diags: dict, depths: tuple[int, ...], seeds: tuple[int, ...]) -> dict:
    rows = _load_rows(depths, seeds)
    cells = []
    for spec in CANDIDATES:
        n12 = quantum_summary(diags[(spec["id"], 12)])
        n18 = quantum_summary(diags[(spec["id"], 18)])
        depth_blocks = {}
        for L in depths:
            depth_blocks[f"L{L}"] = _depth_block(rows, spec, int(L), seeds)
        if "L2" not in depth_blocks:
            raise RuntimeError("L=2 is the decision depth")
        decision = classify_bars(
            lam12=n12["lambda_med"],
            lam18=n18["lambda_med"],
            gap12=n12["gap_med"],
            gap18=n18["gap_med"],
            peak12=n12["peak_med"],
            peak18=n18["peak_med"],
            r_official=depth_blocks["L2"]["by_n"]["12"]["r"],
            r_seeds=depth_blocks["L2"]["r_seeds"],
        )
        # n_clear inside classify counts the same seeds. Keep the depth block's count.
        cell = {
            "id": spec["id"],
            "lambda": float(spec["lambda"]),
            "mu": float(spec["mu"]),
            "baseline": spec["baseline"],
            "prior": PRIORS[spec["id"]],
            "n12": n12,
            "n18": n18,
            "decision": decision,
            "lambda_pass": decision["lambda_pass"],
            "call": decision["call"],
            "r": decision["r"],
            "lam12": decision["lam12"],
            "lam18": decision["lam18"],
        }
        cell.update(depth_blocks)
        prior = PRIORS[spec["id"]]
        if abs(n12["lambda_med"] - prior["lam12"]) > 0.01:
            log(
                f"WARNING {spec['id']} n12 Λ {n12['lambda_med']:.4f} "
                f"vs nomination {prior['lam12']:.4f}"
            )
        cells.append(cell)
    ranked = rank_cells(cells)
    rec = recommend(ranked)
    verdict = family_verdict(c["call"] for c in ranked)
    summary = {
        "protocol": PROTOCOL,
        "verdict": verdict,
        "recommendation": rec,
        "ranked": ranked,
        "freeze_b": FREEZE_B,
        "n18_b": int(diags[("A", 18)]["peak"].shape[0]),
        "n18_base": N18_BASE,
        "depths": list(depths),
        "seeds": list(seeds),
        "rounds": ROUNDS,
        "lbfgs_iter": int(S.LBFGS_ITER),
        "lbfgs_lr": float(S.LBFGS_LR),
        "batch": BATCH,
    }
    return summary


def run(workers: int, depths: tuple[int, ...], seeds: tuple[int, ...], n18_b: int, do_self_check: bool) -> dict:
    decision_self_test()
    if do_self_check:
        log("hva_safe self_check")
        self_check()
        log("self_check ok")
    if len(seeds) != N_SEEDS or tuple(seeds) != SEEDS:
        raise SystemExit(f"this bakeoff is seeds {SEEDS}, got {seeds}")
    E18, s18 = ensure_n18_probe(n18_b)
    import multiprocessing as mp

    ctx = mp.get_context("spawn")
    diags = {}
    with ctx.Pool(int(workers), initializer=_pool_init) as pool:
        for spec in CANDIDATES:
            for n, count, gap in ((6, HOLD_HI, False), (9, HOLD_HI, False), (12, FREEZE_B, True)):
                E, seeds_n = load_pack(n, count)
                diags[(spec["id"], n)] = ensure_diag(spec, n, E, seeds_n, pool, gap)
            diags[(spec["id"], 18)] = ensure_diag(spec, 18, E18, s18, pool, True)
    jobs = []
    for spec in CANDIDATES:
        for L in depths:
            for method in ("snap", spec["baseline"]):
                for seed in seeds:
                    jobs.append(
                        {"id": spec["id"], "L": int(L), "method": method, "seed": int(seed), "rounds": ROUNDS}
                    )
    log(f"training {len(jobs)} jobs on {workers} workers")
    t0 = time.time()
    with ctx.Pool(int(workers), initializer=_pool_init) as pool:
        for i, name in enumerate(pool.imap_unordered(_train_worker, jobs, chunksize=1)):
            log(f"train {i+1}/{len(jobs)} {Path(name).name} elapsed={time.time()-t0:.0f}s")
    summary = assemble(diags, depths, seeds)
    text = render(summary)
    RUN.mkdir(parents=True, exist_ok=True)
    (RUN / "RESULTS.md").write_text(text)
    # summary.json without the per-instance eps (those live in the seed files).
    write_json(RUN / "summary.json", _public_summary(summary))
    log(f"verdict {summary['verdict']} lock={summary['recommendation']['lock']} wrote {RUN / 'RESULTS.md'}")
    return summary


def _public_summary(summary: dict) -> dict:
    """Drop nothing large: the ranked cells already store aggregates only."""
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--workers", type=int, default=min(4, os.cpu_count() or 1))
    ap.add_argument("--depths", type=str, default="2,4")
    ap.add_argument("--seeds", type=str, default="0,1,2,3,4")
    ap.add_argument("--n18-b", type=int, default=N18_B)
    ap.add_argument("--skip-self-check", action="store_true")
    args = ap.parse_args()
    depths = tuple(int(x) for x in args.depths.split(",") if x.strip())
    seeds = tuple(int(x) for x in args.seeds.split(",") if x.strip())
    if any(L not in DEPTHS for L in depths):
        raise SystemExit(f"depths must be in {DEPTHS}")
    run(
        workers=max(1, int(args.workers)),
        depths=depths,
        seeds=seeds,
        n18_b=int(args.n18_b),
        do_self_check=not args.skip_self_check,
    )


if __name__ == "__main__":
    main()
