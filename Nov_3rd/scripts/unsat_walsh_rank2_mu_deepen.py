#!/usr/bin/env python3
"""Deepen freeze of unsat Walsh. Default λ=0.75, μ=−0.25 (PR #25).

``--mu -0.50`` reruns that pipeline on the PR #26 winner.
``--mu -1.00`` is the Round 7 seal of the PR #28 edge (λ=0.75), on the same
budget, with an instance quantumness gate: at least 95% of n=12 and of n=15
formulas must have product gap ≥ 0.02.
``--mu -1.00 --lam 1.00`` is the Round 8 seal of the PR #32 operating point.
PASS still needs the L=16 Walsh BEAT on frozen n=12 and n=15 and that same
95% instance gate. A shortfall is needs_edit. The scorer is unchanged:
BinH U_c^X, Hamming-shell SNAP, Z histogram for unsat, X histogram for the
field and intra-cavity XX. Angles are fit on n=6 and n=9. n=12 and n=15 are
the sealed transfer. n=18 is a Lanczos probe and does not enter the loss, the
checkpoint selection, or the PASS gate. n=21 is not loaded.

    H = diag(unsat) − λ Σ_i X_i + μ Σ_cav Σ_{i<j} X_i X_j
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from explore_OCT29.lib.sat_gen import clause_count
from Nov_3rd import settings as S
from Nov_3rd.quantum_h_tf import FockSim
from Nov_3rd.quantum_h_transfer import center_cn, load_split, walsh_mass_fractions
from Nov_3rd.seal_unsat import mark_of
from Nov_3rd.unsat_rank2_xx import (
    GAP_COVERAGE,
    LAM,
    MU,
    PUBLISHED_N12_L16,
    PUBLISHED_POWER_R,
    PUBLISHED_WALSH_R,
    _ground_job,
    bootstrap_ratio,
    decide_deepen,
    paired_seed_table,
    product_gap_tail,
    wilson_interval,
)

_SPEC = importlib.util.spec_from_file_location(
    "unsat_walsh_rank2_mu",
    Path(__file__).with_name("unsat_walsh_rank2_mu.py"),
)
base = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(base)

PARENT = S.OUT / "quantum_h_explore" / "unsat_rank2_mu"
SWEEP_CACHE = S.OUT / "quantum_h_explore" / "unsat_mu_sweep" / "cache"
# PR #28 diagonals at μ=−1.00. A shorter prefix is reused only when seeds match.
NEG_CACHE = S.OUT / "quantum_h_explore" / "unsat_mu_neg" / "cache"
# PR #32 λ=1.00 diagonals at μ=−1.00. Prefixes only; the seal writes its own B.
LAMBDA_CACHE = S.OUT / "quantum_h_explore" / "unsat_lambda_mu1" / "cache"
# PR #23 ratio of medians, 3 seeds, transfer B=64 (n=12) and B=32 (n=15).
# apply_mu(-0.50) replaces these with the PR #26 point estimates.
PRIOR_R = {12: 0.33011169158114184, 15: 0.3554511757416401}
PRIOR_B = {12: 64, 15: 32}
BRANCH = "cursor/unsat-walsh-mu-deepen-5136"
PRIOR_NOTE = "PR #23"
ROUND_NAME = "Round 5 deepen"
LOG_TAG = "unsatXX-r5"
PROTOCOL = "deepen_r5"
BOOT_SEED = 5136
BOOT_DRAWS = 4000


def log(*a):
    print(time.strftime("%H:%M:%S"), f"[{LOG_TAG}]", *a, flush=True)


def _paths_mu(out: Path, L: int, method: str, seed: int):
    """Checkpoint stem. A dot in the μ tag must not be Path.with_suffix."""
    tag = f"{float(MU):+.2f}".replace("+", "").replace(".", "p")
    stem = Path(out) / "runs" / f"mu{tag}_L{int(L)}_{method}_seed{int(seed)}"
    return Path(str(stem) + ".json"), Path(str(stem) + ".pt")


_PATHS_ORIG = base._paths


def apply_mu(mu: float) -> float:
    """Point the imported scorer at one operating point. Default stays −0.25."""
    global MU, PRIOR_R, PRIOR_B, BRANCH, PRIOR_NOTE, ROUND_NAME, LOG_TAG, PROTOCOL
    import Nov_3rd.unsat_rank2_xx as xx

    mu = float(mu)
    if abs(mu - (-0.25)) <= 1e-12:
        MU = -0.25
        PRIOR_R = {12: 0.33011169158114184, 15: 0.3554511757416401}
        PRIOR_B = {12: 64, 15: 32}
        BRANCH = "cursor/unsat-walsh-mu-deepen-5136"
        PRIOR_NOTE = "PR #23"
        ROUND_NAME = "Round 5 deepen"
        LOG_TAG = "unsatXX-r5"
        PROTOCOL = "deepen_r5"
        base._paths = _PATHS_ORIG
    elif abs(mu - (-0.50)) <= 1e-12:
        MU = -0.50
        # PR #26 ratio of medians, 3 seeds, transfer B=64 (n=12) and B=32 (n=15).
        PRIOR_R = {12: 0.15015595428732964, 15: 0.17579095507059755}
        PRIOR_B = {12: 64, 15: 32}
        BRANCH = "cursor/unsat-walsh-mu050-seal-9cbd"
        PRIOR_NOTE = "PR #26"
        ROUND_NAME = "Round 6 seal"
        LOG_TAG = "unsatXX-r6"
        PROTOCOL = "deepen_r5"
        base._paths = _paths_mu
    elif abs(mu - (-1.00)) <= 1e-12:
        MU = -1.00
        # PR #28 ratio of medians, 3 seeds, transfer B=64 (n=12) and B=32 (n=15).
        PRIOR_R = {12: 0.09675474400205866, 15: 0.11904036219488356}
        PRIOR_B = {12: 64, 15: 32}
        BRANCH = "cursor/unsat-walsh-mu100-seal-3d9d"
        PRIOR_NOTE = "PR #28"
        ROUND_NAME = "Round 7 seal"
        LOG_TAG = "unsatXX-r7"
        PROTOCOL = "seal_mu100"
        base._paths = _paths_mu
    else:
        raise SystemExit("deepen μ must be −0.25, −0.50, or −1.00")
    base.MU = MU
    xx.MU = MU
    return MU


def apply_lam(lam: float) -> float:
    """Point the scorer at λ. 0.75 keeps the Round 5–7 seals. 1.00 is Round 8.

    Call after ``apply_mu``. λ=1.00 is only defined on the μ=−1.00 edge.
    """
    global LAM, PRIOR_R, PRIOR_B, BRANCH, PRIOR_NOTE, ROUND_NAME, LOG_TAG, PROTOCOL
    import Nov_3rd.unsat_rank2_xx as xx

    lam = float(lam)
    if abs(lam - 0.75) <= 1e-12:
        LAM = 0.75
    elif abs(lam - 1.00) <= 1e-12:
        if abs(float(MU) - (-1.00)) > 1e-12:
            raise SystemExit("λ=1.00 deepen is only defined at μ=−1.00")
        LAM = 1.00
        # PR #32 ratio of medians, 3 seeds, transfer B=64 (n=12) and B=32 (n=15).
        PRIOR_R = {12: 0.07629004834603922, 15: 0.09580073235382872}
        PRIOR_B = {12: 64, 15: 32}
        BRANCH = "cursor/unsat-walsh-lam1-seal-a846"
        PRIOR_NOTE = "PR #32"
        ROUND_NAME = "Round 8 seal"
        LOG_TAG = "unsatXX-r8"
        PROTOCOL = "seal_lam1_mu100"
    else:
        raise SystemExit("deepen λ must be 0.75 or 1.00")
    base.LAM = LAM
    xx.LAM = LAM
    return LAM


def _tag() -> str:
    return f"m{MU:+.2f}".replace("+", "")


def _lam_tag() -> str:
    return f"{float(LAM):.2f}"


def _cache_name(n: int, role: str, count: int) -> str:
    return f"unsat_xx_n{n}_{role}_B{count}_l{_lam_tag()}_{_tag()}.npz"


def _blob(path: Path) -> dict:
    z = np.load(path)
    return {k: z[k] for k in z.files}


def _cache_dirs(out: Path):
    # PR #26 wrote the μ=−0.50 diagonals under the sweep cache. A prefix there
    # matches this tag only, so the μ=−0.25 files in PARENT stay unused.
    # PR #28 wrote the μ=−1.00 prefixes under NEG_CACHE.
    dirs = [out / "cache", PARENT / "cache", SWEEP_CACHE]
    if abs(float(MU) - (-1.00)) <= 1e-12 and abs(float(LAM) - 0.75) <= 1e-12:
        dirs.append(NEG_CACHE)
    if abs(float(MU) - (-1.00)) <= 1e-12 and abs(float(LAM) - 1.00) <= 1e-12:
        dirs.append(LAMBDA_CACHE)
    return dirs


def _find_exact(out: Path, n: int, role: str, count: int):
    name = _cache_name(n, role, count)
    for folder in _cache_dirs(out):
        path = folder / name
        if path.exists():
            return path
    return None


def _find_prefix(out: Path, n: int, role: str, count: int):
    best = None
    best_b = -1
    prefix = f"unsat_xx_n{n}_{role}_B"
    suffix = f"_l{_lam_tag()}_{_tag()}.npz"
    for folder in _cache_dirs(out):
        if not folder.exists():
            continue
        for path in folder.glob(prefix + "*" + suffix):
            mid = path.name[len(prefix):-len(suffix)]
            if not mid.isdigit():
                continue
            b = int(mid)
            if 0 < b < int(count) and b > best_b:
                best, best_b = path, b
    return best


def _ground_tail(raw: np.ndarray, n: int, seeds: np.ndarray, workers: int) -> dict:
    payloads = [
        (raw[i], int(n), float(LAM), float(MU), int(seeds[i]), True)
        for i in range(raw.shape[0])
    ]
    e0 = np.empty(len(payloads), dtype=np.float64)
    peak = np.empty_like(e0)
    entropy = np.empty_like(e0)
    residual = np.empty_like(e0)
    e_prod = np.empty_like(e0)
    t0 = time.time()
    done = 0

    def take(i, row):
        nonlocal done
        e0[i], peak[i], entropy[i], residual[i], e_prod[i] = row
        done += 1
        if int(n) >= 12 and (done == 1 or done == len(payloads) or done % 8 == 0):
            log(f"  ED n={n} {done}/{len(payloads)} {time.time() - t0:.0f}s")

    if workers <= 1 or len(payloads) < 2:
        for i, payload in enumerate(payloads):
            take(i, _ground_job(payload))
    else:
        with ThreadPoolExecutor(max_workers=int(workers)) as pool:
            for i, row in enumerate(pool.map(_ground_job, payloads)):
                take(i, row)
    return {"E0": e0, "peak": peak, "entropy": entropy, "residual": residual, "E_prod": e_prod}


def _workers_for(n: int, requested: int) -> int:
    if int(n) >= 18:
        return 1
    if int(n) >= 15:
        return min(2, int(requested))
    return int(requested)


def load_diagonal(n: int, role: str, count: int, out: Path, workers: int) -> dict:
    exact = _find_exact(out, n, role, count)
    if exact is not None:
        blob = _blob(exact)
        log(f"cache hit n={n} {role} B={count} ({exact.parent.name})")
        return blob
    kind = "train" if role == "train" else "scale"
    pack = base.TRAIN_PACKS / f"train_n{n}.npz" if role == "train" else None
    t0 = time.time()
    insts, raw = load_split(n, count, kind, pack)
    raw = np.asarray(raw, dtype=np.float64)
    m = int(clause_count(n))
    if role == "train":
        packed_m = int(np.load(pack)["m"])
        if packed_m != m:
            raise RuntimeError(f"n={n} pack m={packed_m} != clause_count {m}")
    seeds = np.asarray([int(ins.seed) for ins in insts], dtype=np.int64)
    e0 = np.empty(count, dtype=np.float64)
    peak = np.empty(count, dtype=np.float64)
    entropy = np.empty(count, dtype=np.float64)
    residual = np.empty(count, dtype=np.float64)
    e_prod = np.empty(count, dtype=np.float64)
    start = 0
    prefix = _find_prefix(out, n, role, count)
    if prefix is not None:
        pre = _blob(prefix)
        pre_seeds = [int(s) for s in np.asarray(pre["seeds"]).reshape(-1)]
        b = len(pre_seeds)
        if pre_seeds != [int(s) for s in seeds[:b]] or not np.array_equal(np.asarray(pre["score"]), raw[:b]):
            raise RuntimeError(f"prefix {prefix.name} does not match the first {b} draws")
        if abs(float(pre["mu"]) - MU) > 1e-12 or abs(float(pre["lam"]) - LAM) > 1e-12:
            raise RuntimeError(f"prefix μ/λ drift {prefix.name}")
        e0[:b] = pre["e0"]
        peak[:b] = pre["peak"]
        entropy[:b] = pre["entropy"]
        residual[:b] = pre["residual"]
        e_prod[:b] = pre["e_prod"]
        start = b
        log(f"prefix n={n} {role} reused {b}/{count} from {prefix.name}")
    if start < count:
        log(f"ED n={n} {role} tail {count - start} unsat λ={LAM} μ={MU}")
        tail = _ground_tail(raw[start:], n, seeds[start:], _workers_for(n, workers))
        e0[start:] = tail["E0"]
        peak[start:] = tail["peak"]
        entropy[start:] = tail["entropy"]
        residual[start:] = tail["residual"]
        e_prod[start:] = tail["E_prod"]
    if float(np.max(residual)) > 1e-5:
        raise RuntimeError(f"n={n} {role} residual {float(np.max(residual)):.2e}")
    pulse = center_cn(raw, n, m)
    denom = np.maximum(np.abs(e0), 1e-8)
    blob = {
        "score": raw,
        "pulse": np.asarray(pulse, dtype=np.float64),
        "e0": e0,
        "peak": peak,
        "entropy": entropy,
        "residual": residual,
        "e_prod": e_prod,
        "prod_gap": (e_prod - e0) / denom,
        "seeds": seeds,
        "m": np.asarray(m, dtype=np.int64),
        "lam": np.asarray(LAM, dtype=np.float64),
        "mu": np.asarray(MU, dtype=np.float64),
    }
    path = out / "cache" / _cache_name(n, role, count)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **blob)
    log(
        f"  built n={n} {role} B={count} in {time.time() - t0:.1f}s "
        f"peak_med={float(np.median(peak)):.3f}"
    )
    return blob


def _meta(blob: dict, n: int, role: str) -> dict:
    if "prod_gap" in blob:
        gap = np.asarray(blob["prod_gap"], dtype=np.float64)
    else:
        denom = np.maximum(np.abs(blob["e0"]), 1e-8)
        gap = (blob["e_prod"] - blob["e0"]) / denom
    tail = product_gap_tail(gap, blob["peak"])
    refs = base._ref_eps(blob["score"], blob["e0"], n)
    return {
        "n": int(n),
        "role": role,
        "B": int(blob["e0"].shape[0]),
        "peak_med": float(np.median(blob["peak"])),
        "entropy_med": float(np.median(blob["entropy"])),
        "residual_max": float(np.max(blob["residual"])),
        "prod_gap_med": float(tail["gap_med"]),
        "seeds": [int(s) for s in np.asarray(blob["seeds"]).reshape(-1)],
        **{k: v for k, v in tail.items() if k != "gap_med"},
        **refs,
    }


def _attach(sims, prepared, quantum, n, role, blob):
    if abs(float(blob["mu"]) - MU) > 1e-12 or abs(float(blob["lam"]) - LAM) > 1e-12:
        raise RuntimeError(f"cache μ/λ drift n={n} {role}")
    if n not in sims:
        sims[n] = FockSim(n, dtype=torch.float64)
    meta = _meta(blob, n, role)
    zx = base._make_zx(sims[n], blob["pulse"], blob["score"], blob["e0"])
    prepared[(n, role)] = {"split": base.Split(n, role, zx, meta["seeds"], meta), "pulse": blob["pulse"], "score": blob["score"]}
    quantum[(n, role)] = meta
    log(
        f"  n={n} {role} B={meta['B']} peak={meta['peak_med']:.3f} "
        f"S={meta['entropy_med']:.3f} gap={meta['prod_gap_med']:.3f} "
        f"frac<0.02={meta['frac_gap_lt_0p02']:.3f} frac<0.05={meta['frac_gap_lt_0p05']:.3f} "
        f"ε+={meta['med_eps_plus']:.3f} εbit={meta['med_eps_bit']:.3f}"
    )


def prepare_core(args, out: Path):
    specs = [
        (6, "train", args.B_train),
        (6, "hold", args.B_hold),
        (9, "train", args.B_train),
        (9, "hold", args.B_hold),
        (12, "transfer", args.B_transfer),
        (15, "transfer", args.B_n15),
    ]
    sims, prepared, quantum = {}, {}, {}
    for n, role, count in specs:
        blob = load_diagonal(n, role, count, out, args.workers)
        _attach(sims, prepared, quantum, n, role, blob)
    for n in (6, 9):
        tr = set(prepared[(n, "train")]["split"].seeds)
        ho = set(prepared[(n, "hold")]["split"].seeds)
        if tr & ho:
            raise RuntimeError(f"n={n} train/hold seed overlap")
        if len(tr) != args.B_train or len(ho) != args.B_hold:
            raise RuntimeError(f"n={n} split size drift")
    return sims, prepared, quantum


def probe_n18(args, out: Path, sims, prepared, quantum) -> dict:
    """Score a frozen transfer at n=18 when one Lanczos-plus-product fit is cheap."""
    if int(args.B_n18) < 4:
        return {"status": "skipped", "detail": "B_n18 < 4"}
    t0 = time.time()
    try:
        insts, raw = load_split(18, 1, "scale")
    except Exception as exc:
        return {"status": "infeasible", "detail": f"instance generation failed: {exc}"}
    raw = np.asarray(raw, dtype=np.float64)
    try:
        _ground_job((raw[0], 18, float(LAM), float(MU), int(insts[0].seed), True))
    except Exception as exc:
        return {"status": "infeasible", "detail": f"Lanczos/product failed: {exc}"}
    dt = time.time() - t0
    log(f"n=18 feasibility probe {dt:.1f}s (limit {args.n18_limit_s:.0f}s)")
    if dt > float(args.n18_limit_s):
        return {
            "status": "infeasible",
            "detail": f"one n=18 ground state took {dt:.0f}s, above the {args.n18_limit_s:.0f}s probe budget",
        }
    try:
        blob = load_diagonal(18, "probe", args.B_n18, out, args.workers)
        _attach(sims, prepared, quantum, 18, "probe", blob)
        zx = prepared[(18, "probe")]["split"].circuit
        with torch.no_grad():
            zx.energy(torch.zeros(2, dtype=torch.float64), torch.zeros(2, 4, dtype=torch.float64), False, LAM)
    except MemoryError:
        prepared.pop((18, "probe"), None)
        quantum.pop((18, "probe"), None)
        return {"status": "infeasible", "detail": "n=18 histogram state did not fit in memory"}
    quantum[(18, "probe")]["probe_s"] = dt
    return {"status": "ok", "B": int(args.B_n18), "probe_s": dt}


def _check_protocol(rec: dict, args) -> None:
    if rec.get("protocol") != PROTOCOL:
        raise RuntimeError(f"resume protocol {rec.get('protocol')}")
    if abs(float(rec.get("lam", -1.0)) - float(LAM)) > 1e-12 or abs(float(rec.get("mu", 99.0)) - float(MU)) > 1e-12:
        raise RuntimeError(f"resume λ/μ {rec.get('lam')}/{rec.get('mu')} != {LAM}/{MU}")
    expect = {
        "B_train": args.B_train,
        "B_hold": args.B_hold,
        "B_transfer": args.B_transfer,
        "B_n15": args.B_n15,
    }
    for key, val in expect.items():
        if int(rec.get(key, -1)) != int(val):
            raise RuntimeError(f"resume {key}={rec.get(key)} != {val}")


def fit_deep(method, L, seed, train, hold, report, args, theta0, ref, out: Path, have_n18: bool):
    cached = None if args.no_resume else base._load_run(out, L, method, seed)
    if cached is not None:
        _check_protocol(cached, args)
        want_n18 = 0 if not have_n18 else int(args.B_n18)
        if int(cached.get("B_n18", 0)) != want_n18:
            raise RuntimeError(f"resume B_n18={cached.get('B_n18')} != {want_n18}")
        missing = [sp for sp in report if sp.n not in cached["eval"]]
        if missing:
            log(f"re-eval {method} L={L} seed={seed} n={[sp.n for sp in missing]}")
            cached["eval"].update(base.evaluate(method, cached["theta"], missing, LAM, L))
            base._save_run(out, cached)
        base._check_select(cached)
        if method != "fourier_qaoa":
            base._assert_warm(method, cached, ref)
        log(f"resume {method} L={L} seed={seed} tag={cached.get('best_tag')}")
        return cached
    rounds = args.rounds_L2 if L <= 2 else args.rounds_L16
    max_iter = args.max_iter_L2 if L <= 2 else args.max_iter_L16
    log(f"train {method} L={L} seed={seed}")
    rec = base.fit_one(method, L, seed, train, hold, LAM, rounds, max_iter, args.lr, theta0, ref)
    rec["eval"] = base.evaluate(method, rec["theta"], report, LAM, L)
    base._check_select(rec)
    rec["protocol"] = PROTOCOL
    rec["B_train"] = int(args.B_train)
    rec["B_hold"] = int(args.B_hold)
    rec["B_transfer"] = int(args.B_transfer)
    rec["B_n15"] = int(args.B_n15)
    rec["B_n18"] = int(args.B_n18) if have_n18 else 0
    base._save_run(out, rec)
    bits = [f"ε{n}={rec['eval'][n]['med_eps']:.4f}" for n in (12, 15) if n in rec["eval"]]
    if 18 in rec["eval"]:
        bits.append(f"ε18={rec['eval'][18]['med_eps']:.4f}")
    log(f"  frozen {' '.join(bits)} wall={rec['wall_s']:.1f}s tag={rec['best_tag']}")
    return rec


def train(prepared, args, out: Path, have_n18: bool):
    train_splits = [prepared[(n, "train")]["split"] for n in (6, 9)]
    hold_splits = [prepared[(n, "hold")]["split"] for n in (6, 9)]
    if any(sp.n >= 12 for sp in train_splits + hold_splits):
        raise RuntimeError("train/hold leaked a transfer n")
    report = [
        prepared[(6, "hold")]["split"],
        prepared[(9, "hold")]["split"],
        prepared[(12, "transfer")]["split"],
        prepared[(15, "transfer")]["split"],
    ]
    if have_n18:
        report.append(prepared[(18, "probe")]["split"])
    runs = []
    for L in args.Ls:
        fourier = {}
        for seed in args.seeds:
            rec = fit_deep("fourier_qaoa", L, seed, train_splits, hold_splits, report, args, None, None, out, have_n18)
            fourier[seed] = rec
            runs.append(rec)
        for method, warm in (("power_p", base.power_theta_from_fourier), ("walsh", base.walsh_theta_from_fourier)):
            for seed in args.seeds:
                ref = {n: fourier[seed]["eval"][n]["med_eps"] for n in (6, 9)}
                theta0 = warm(fourier[seed]["theta"], L)
                rec = fit_deep(method, L, seed, train_splits, hold_splits, report, args, theta0, ref, out, have_n18)
                base._assert_warm(method, rec, ref)
                runs.append(rec)
    return runs


def aggregate(runs, ns):
    cells = {}
    by = {}
    for rec in runs:
        by.setdefault((rec["method"], int(rec["L"])), []).append(rec)
    for (method, L), rows in by.items():
        rows = sorted(rows, key=lambda r: int(r["seed"]))
        for n in ns:
            if any(n not in r["eval"] for r in rows):
                continue
            meds = [float(r["eval"][n]["med_eps"]) for r in rows]
            std = float(np.std(meds, ddof=1)) if len(meds) > 1 else 0.0
            hmed = float(np.median([float(r["eval"][n]["med_H_over_n"]) for r in rows]))
            cells[(method, L, n)] = {
                "method": method,
                "L": int(L),
                "n": int(n),
                "role": rows[0]["eval"][n]["role"],
                "B": int(len(rows[0]["eval"][n]["eps"])),
                "med": float(np.median(meds)),
                "seed_std": std,
                "iqr": float(np.median([float(r["eval"][n]["iqr_eps"]) for r in rows])),
                "med_H_over_n": hmed,
                "per_seed": meds,
                "seeds": [int(r["seed"]) for r in rows],
                "n_params": int(rows[0]["n_params"]),
            }
    for L in sorted({L for (_m, L, _n) in cells}):
        if ("fourier_qaoa", L, ns[0]) not in cells:
            continue
        for n in ns:
            q = cells.get(("fourier_qaoa", L, n))
            if q is None:
                continue
            q_by_seed = {int(s): float(v) for s, v in zip(q["seeds"], q["per_seed"])}
            for method in ("power_p", "walsh"):
                cell = cells.get((method, L, n))
                if cell is None:
                    continue
                if cell["seeds"] != q["seeds"]:
                    raise RuntimeError(f"seed mismatch {method} L={L} n={n}")
                ref = float(q["med"])
                cell["qaoa"] = ref
                cell["R"] = float(cell["med"] / ref) if abs(ref) > 1e-12 else float("nan")
                cell["mark"] = mark_of(cell["R"])
                seed_Rs = []
                for seed, med in zip(cell["seeds"], cell["per_seed"]):
                    base_med = q_by_seed[int(seed)]
                    seed_Rs.append(float(med / base_med) if abs(base_med) > 1e-12 else float("nan"))
                cell["seed_Rs"] = seed_Rs
                published = PUBLISHED_WALSH_R if method == "walsh" else PUBLISHED_POWER_R
                if (int(L), int(n)) in published:
                    cell["published_R"] = float(published[(int(L), int(n))])
                    cell["drop_vs_published"] = float(cell["published_R"] - cell["R"])
    return cells


def _eps(runs, method, L, n):
    rows = sorted(
        [r for r in runs if r["method"] == method and int(r["L"]) == int(L)],
        key=lambda r: int(r["seed"]),
    )
    return [np.asarray(r["eval"][n]["eps"], dtype=np.float64) for r in rows]


def statistics(runs, cells, ns):
    paired, ci, ci_seed = {}, {}, {}
    for L in sorted({int(r["L"]) for r in runs}):
        fourier = {n: _eps(runs, "fourier_qaoa", L, n) for n in ns if ("fourier_qaoa", L, n) in cells}
        for method in ("power_p", "walsh"):
            for n, base_eps in fourier.items():
                if (method, L, n) not in cells:
                    continue
                got = _eps(runs, method, L, n)
                table = paired_seed_table(got, base_eps)
                paired[(method, L, n)] = table
                block = bootstrap_ratio(got, base_eps, n_boot=BOOT_DRAWS, seed=BOOT_SEED, mode="instance")
                point = float(cells[(method, L, n)]["R"])
                if abs(block["R"] - point) > 1e-8:
                    raise RuntimeError(f"bootstrap point {block['R']} != table R {point} at {method} L={L} n={n}")
                ci[(method, L, n)] = block
                if method == "walsh" and int(L) == 16 and int(n) in (12, 15, 18):
                    ci_seed[(method, L, n)] = bootstrap_ratio(
                        got, base_eps, n_boot=BOOT_DRAWS, seed=BOOT_SEED + 1, mode="seed"
                    )
    return paired, ci, ci_seed


def _fmt(x, nd=3):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "—"
    return f"{float(x):.{nd}f}"


def _mu_text(mu: float) -> str:
    mu = float(mu)
    sign = "−" if mu < 0 else "+"
    return f"{sign}{abs(mu):.2f}"


def render(payload: dict) -> str:
    v = payload["verdict"]
    q = payload["quantum"]
    cells = payload["cells"]
    mu_s = _mu_text(payload["mu"])
    lam = float(payload.get("lam", LAM))
    lam_s = f"{lam:.2f}"
    round8 = abs(float(payload["mu"]) + 1.00) < 1e-9 and abs(lam - 1.00) < 1e-9
    round_name = payload.get("round_name", ROUND_NAME)
    prior_note = payload.get("prior_note", PRIOR_NOTE)
    lines = []
    lines.append(f"# RESULTS — {round_name}, unsat λ={lam_s} + μ={mu_s} Walsh freeze")
    lines.append("")
    lines.append(f"**Verdict: {v['verdict']}**")
    lines.append("")
    lines.append(v["reason"])
    lines.append("")
    lines.append("## Setup")
    lines.append("")
    if round8:
        lines.append(
            "Same gate as the PR #30 seal: BinH U_c^X, Hamming-shell SNAP. "
            "The Hamiltonian is the PR #32 operating point, λ=1.00 and μ=−1.00. "
            "Intra-cavity XX is scored from the X histogram. "
            "Angles are fit on n=6 and n=9. The sealed scores at n=12 and n=15, and the n=18 probe, "
            "use those frozen angles. n=21 was not loaded. "
            "PR #32 froze this cell at 3 seeds (n=12 L=16 Walsh R=0.076 on B=64, median product gap 0.021, "
            "19/64 instances under 0.02; n=15 R=0.096 on B=32, 13/32 under 0.02). "
            "PR #30 sealed λ=0.75 at this μ with n=12 R=0.096 and 127/128 instances at gap ≥ 0.02. "
            "This run keeps the PR #30 budget and asks whether λ=1.00 still BEATs and clears that 95% instance gap."
        )
        hold_note = "PR #32 used 128"
        transfer_note = f"PR #32 used {PRIOR_B[12]} and {PRIOR_B[15]}"
        budget_note = "That is the PR #30 optimizer budget on the larger selection holdout."
    elif abs(float(payload["mu"]) - (-1.00)) < 1e-9:
        lines.append(
            "Same gate as the PR #27 seal: BinH U_c^X, Hamming-shell SNAP. "
            "The Hamiltonian is the PR #28 edge, μ=−1.00. "
            "Intra-cavity XX is scored from the X histogram. "
            "Angles are fit on n=6 and n=9. The sealed scores at n=12 and n=15, and the n=18 probe, "
            "use those frozen angles. n=21 was not loaded. "
            "PR #27 sealed μ=−0.50 at n=12 L=16 Walsh R=0.147 with product gap 0.064. "
            "PR #28 moved the same H to μ=−1.00 (n=12 R=0.097 on B=64, median product gap 0.028, "
            "1/64 instances under 0.02). This run keeps that edge and asks the larger sample "
            "to clear a 95% instance gap."
        )
        hold_note = "PR #28 used 128"
        transfer_note = f"PR #28 used {PRIOR_B[12]} and {PRIOR_B[15]}"
        budget_note = "That is the PR #25 and PR #27 optimizer budget on a larger selection holdout."
    elif abs(float(payload["mu"]) - (-0.50)) < 1e-9:
        lines.append(
            "Same gate as the PR #25 deepen: BinH U_c^X, Hamming-shell SNAP. "
            "The Hamiltonian is the PR #26 winner, μ=−0.50. "
            "Intra-cavity XX is scored from the X histogram. "
            "Angles are fit on n=6 and n=9. The sealed scores at n=12 and n=15, and the n=18 probe, "
            "use those frozen angles. n=21 was not loaded."
        )
        hold_note = "PR #26 used 128"
        transfer_note = f"PR #26 used {PRIOR_B[12]} and {PRIOR_B[15]}"
        budget_note = "That is the PR #25 optimizer budget on a larger selection holdout."
    else:
        lines.append(
            "Same gate and Hamiltonian as PR #23. Intra-cavity XX is scored from the X histogram. "
            "Angles are fit on n=6 and n=9. The sealed scores at n=12 and n=15, and the n=18 probe, "
            "use those frozen angles. n=21 was not loaded."
        )
        hold_note = "PR #23 used 128"
        transfer_note = f"PR #23 used {PRIOR_B[12]} and {PRIOR_B[15]}"
        budget_note = "That is the PR #23 optimizer budget on a larger selection holdout."
    lines.append("")
    lines.append("```text")
    lines.append(f"H = diag(unsat) − {lam_s} Σ_i X_i + μ Σ_cav Σ_{{i<j}} X_i X_j ,  μ = {mu_s}")
    lines.append("U_c^X = W exp(−i γ C_n) W ,  W = H^{⊗3} ,  C_n = (n/m) unsat − max")
    lines.append(f"⟨H⟩ = Σ_z p_Z(z) unsat(z) − {lam_s} Σ_i ⟨X_i⟩ + μ Σ_cav Σ_{{i<j}} ⟨X_i X_j⟩")
    lines.append("ε = (⟨H⟩ − E_0) / n")
    lines.append("```")
    lines.append("")
    lines.append(
        f"Train B={payload['B_train']} per n from `beat_qaoa_12h/train_packs` (base 51e9). "
        f"Selection holdout B={payload['B_hold']} per n ({hold_note}). "
        f"Sealed transfer B={payload['B_transfer']} at n=12 and B={payload['B_n15']} at n=15 "
        f"({transfer_note}). "
        f"Seeds {payload['seeds']}. L={payload['Ls']}. "
        f"L-BFGS rounds {payload['rounds_L2']}/{payload['rounds_L16']}, "
        f"max_iter {payload['max_iter_L2']}/{payload['max_iter_L16']}, lr={payload['lr']}. "
        f"{budget_note} "
        "The checkpoint minimizes the worst of the two holdout ratios against that seed's Fourier. "
        "Early stop is 8 rounds without a 10⁻⁶ improvement."
    )
    lines.append("")
    lines.append(
        f"{prior_note} point estimates, 3 seeds: n=12 L=16 Walsh R={PRIOR_R[12]:.3f} on B={PRIOR_B[12]}, "
        f"n=15 L=16 Walsh R={PRIOR_R[15]:.3f} on B={PRIOR_B[15]}. "
        f"μ=0 comparison remains the published PR #19 ratio of medians "
        f"(n=12 L=16 Walsh R={PUBLISHED_N12_L16:.3f}), not a retrain."
    )
    lines.append("")
    n18 = payload["n18"]
    if n18.get("status") == "probe":
        lines.append(
            f"n=18 probe B={n18['B']}. One Lanczos-plus-product ground state took {n18['probe_s']:.1f}s, "
            f"under the {payload['n18_limit_s']:.0f}s feasibility limit. "
            "The probe is reported and is not part of PASS."
        )
    else:
        lines.append(f"n=18 probe: {n18.get('detail', n18.get('status'))}")
    lines.append("")
    lines.append("## Quantumness")
    lines.append("")
    lines.append(
        "E0 is Lanczos on the sparse Hamiltonian. A split is a bitstring when the median "
        "ground-state peak is at least 0.5. The product gap is (E_prod − E_0) / |E_0|."
    )
    lines.append("")
    lines.append(
        f"Worst |ΔE0| Lanczos vs dense = {payload['audit']['worst_e0']:.2e}. "
        f"Histogram vs ⟨ψ|H|ψ⟩ = {payload['audit']['worst_psi']:.2e}. "
        f"μ=0 histogram vs Pauli field = {payload['audit']['mu0_vs_pauli']:.2e}."
    )
    lines.append("")
    lines.append("| n | role | B | peak Med | S_cav Med | prod gap Med | ε |+⟩ Med | ε bit Med | residual max |")
    lines.append("|--:|:-----|--:|---------:|----------:|-------------:|----------:|----------:|-------------:|")
    order_q = [(6, "train"), (6, "hold"), (9, "train"), (9, "hold"), (12, "transfer"), (15, "transfer")]
    if n18.get("status") == "probe":
        order_q.append((18, "probe"))
    for n, role in order_q:
        row = q[f"{n}:{role}"]
        lines.append(
            f"| {n} | {role} | {row['B']} | {row['peak_med']:.3f} | {row['entropy_med']:.3f} | "
            f"{row['prod_gap_med']:.3f} | {row['med_eps_plus']:.3f} | {row['med_eps_bit']:.3f} | "
            f"{row['residual_max']:.1e} |"
        )
    if abs(float(payload["mu"]) - (-1.00)) < 1e-9:
        lines.append("")
        gate_line = (
            "Round 8 keeps the Round 7 instance gate and needs"
            if round8
            else "The Round 7 seal needs"
        )
        lines.append(
            "Instance product-gap tails. A count is the number of formulas with "
            "(E_prod − E_0) / |E_0| strictly below the bar. "
            f"{gate_line} at least 95% of the n=12 instances and at least 95% of the n=15 instances "
            "at or above 0.02. The fraction below 0.05 is reported and does not gate the call. "
            "n=6, n=9, and the n=18 probe are the same count and are not part of that gate."
        )
        lines.append("")
        lines.append("| n | role | B | n < 0.02 | frac < 0.02 | n < 0.05 | frac < 0.05 | min gap |")
        lines.append("|--:|:-----|--:|---------:|------------:|---------:|------------:|--------:|")
        for n, role in order_q:
            row = q[f"{n}:{role}"]
            lines.append(
                f"| {n} | {role} | {row['B']} | {int(row['n_gap_lt_0p02'])} | "
                f"{float(row['frac_gap_lt_0p02']):.3f} | {int(row['n_gap_lt_0p05'])} | "
                f"{float(row['frac_gap_lt_0p05']):.3f} | {float(row['gap_min']):.4f} |"
            )
    mass = payload["walsh_mass"]
    lines.append("")
    lines.append(
        f"Train n=6 C_n Walsh mass in degrees 1–4 is {mass['frac_deg_1_to_4']:.3f} "
        f"(degree 3 is {mass['frac_deg_3']:.3f}, degree ≥5 is {mass['frac_other']:.3e})."
    )
    lines.append("")
    lines.append("## Transfer table")
    lines.append("")
    lines.append(
        "R = (median over seeds of median ε) / (the same number for Fourier QAOA on this H). "
        "BEAT if R≤0.98, TIE if R≤1.02, else UNDER. "
        "The 95% CI resamples instances inside each seed and keeps every seed (4000 draws, seed 5136). "
        "Paired W/T/L counts instances with ε_method < ε_Fourier, exact ties, and losses, summed over seeds. "
        "μ=0 R is the PR #19 unsat seal. n=6 and n=9 are the selection holdout. n=12 and n=15 are not. "
        "n=18 is the probe."
    )
    lines.append("")
    lines.append("| method | L | n | role | B | Med ε | seed std | R | 95% CI | paired W/T/L | mark | μ=0 R | drop |")
    lines.append("|:-------|--:|--:|:-----|--:|------:|---------:|--:|:-------|:-------------|:-----|------:|-----:|")
    keys = []
    for L in payload["Ls"]:
        for method in ("fourier_qaoa", "power_p", "walsh"):
            for n in payload["ns"]:
                keys.append((method, int(L), int(n)))
    for key in keys:
        cell = cells[f"{key[0]}|{key[1]}|{key[2]}"]
        if key[0] == "fourier_qaoa":
            lines.append(
                f"| {key[0]} | {key[1]} | {key[2]} | {cell['role']} | {cell['B']} | "
                f"{cell['med']:.4f} | {cell['seed_std']:.4f} | — | — | — |  | — | — |"
            )
            continue
        stat = payload["stats"][f"{key[0]}|{key[1]}|{key[2]}"]
        ci = stat["ci"]
        p = stat["paired"]
        pub = cell.get("published_R")
        drop = cell.get("drop_vs_published")
        drop_s = "—" if drop is None else f"{float(drop):+.3f}"
        lines.append(
            f"| {key[0]} | {key[1]} | {key[2]} | {cell['role']} | {cell['B']} | "
            f"{cell['med']:.4f} | {cell['seed_std']:.4f} | {cell['R']:.3f} | "
            f"{ci['lo']:.3f}–{ci['hi']:.3f} | {p['win']}/{p['tie']}/{p['loss']} | {cell['mark']} | "
            f"{_fmt(pub)} | {drop_s} |"
        )
    lines.append("")
    lines.append("Per-seed median ε, per-seed R, and per-seed paired wins:")
    lines.append("")
    for key in keys:
        cell = cells[f"{key[0]}|{key[1]}|{key[2]}"]
        meds = ", ".join(f"{x:.4f}" for x in cell["per_seed"])
        if "seed_Rs" not in cell:
            lines.append(f"- `{key[0]}` L={key[1]} n={key[2]} seeds {cell['seeds']}: {meds}")
            continue
        rs = ", ".join(f"{x:.3f}" for x in cell["seed_Rs"])
        med_r = float(np.median(cell["seed_Rs"]))
        wins = payload["stats"][f"{key[0]}|{key[1]}|{key[2]}"]["paired"]["per_seed"]
        paired_s = ", ".join(f"{row['win']}/{row['n']}" for row in wins)
        lines.append(
            f"- `{key[0]}` L={key[1]} n={key[2]} seeds {cell['seeds']}: {meds} R {rs} "
            f"(median of those R = {med_r:.3f}); paired wins {paired_s}"
        )
    lines.append("")
    for n in (12, 15):
        key = f"walsh|16|{n}"
        stat = payload["stats"][key]
        cell = cells[key]
        wlo, whi = stat["wilson"]
        seed_ci = stat.get("ci_seed")
        lines.append(
            f"Walsh L=16 n={n}: R={cell['R']:.3f}, instance CI {stat['ci']['lo']:.3f}–{stat['ci']['hi']:.3f}, "
            f"seed-resample CI {seed_ci['lo']:.3f}–{seed_ci['hi']:.3f}, "
            f"pooled paired wins {stat['paired']['win']}/{stat['paired']['n']} "
            f"(Wilson 95% {wlo:.3f}–{whi:.3f}; pairs repeat instances across seeds). "
            f"{payload.get('prior_note', PRIOR_NOTE)} point estimate {PRIOR_R[n]:.3f}."
        )
    if n18.get("status") == "probe" and "walsh|16|18" in cells:
        stat = payload["stats"]["walsh|16|18"]
        cell = cells["walsh|16|18"]
        wlo, whi = stat["wilson"]
        lines.append(
            f"Walsh L=16 n=18 probe: R={cell['R']:.3f}, instance CI {stat['ci']['lo']:.3f}–{stat['ci']['hi']:.3f}, "
            f"pooled paired wins {stat['paired']['win']}/{stat['paired']['n']} "
            f"(Wilson 95% {wlo:.3f}–{whi:.3f}). Exploratory."
        )
    lines.append("")
    lines.append("## One-parameter cost pulse")
    lines.append("")
    lines.append(
        "SNAP phase 0. XX is not in that pulse. The scan is the minimum median ε over γ ∈ [−2, 2]."
    )
    lines.append("")
    lines.append("| n | role | γ | Med ε | Walsh L=16 Med ε |")
    lines.append("|--:|:-----|--:|------:|------------------:|")
    for n, role in order_q:
        if role == "train":
            continue
        row = payload["cost_scan"][str(n)]
        w = cells[f"walsh|16|{n}"]["med"]
        lines.append(f"| {n} | {role} | {row['gamma']:.4f} | {row['med_eps']:.4f} | {w:.4f} |")
    lines.append("")
    lines.append("## Rule")
    lines.append("")
    rule = (
        "PASS if frozen L=16 Walsh at n=12 and at n=15 both have ratio-of-medians ≤ 0.98, "
        "the instance-bootstrap 95% interval lies inside that BEAT line, every seed beats its own Fourier, "
        "every seed wins a strict majority of instances, the n=12 drop versus PR #19's 0.940 is at least 0.02 "
        "on both the ratio of medians and the median per-seed ratio, and the n=12 and n=15 ground states "
        "have peak < 0.5 and median product gap ≥ 0.02."
    )
    if abs(float(payload["mu"]) - (-1.00)) < 1e-9:
        round_label = (
            "Round 8 keeps that gate and also requires"
            if round8
            else "Round 7 also requires"
        )
        rule += (
            f" {round_label} at least 95% of the n=12 instances and at least 95% of the n=15 instances "
            "to have product gap ≥ 0.02. A shortfall on that fraction is needs_edit and is a quantumness fail, "
            "even when the median gap and the Walsh ratios still look like a BEAT. "
            "The fraction of instances with product gap < 0.05 is reported and does not move the call."
        )
    rule += " power_p does not gate PASS. n=18 does not gate PASS."
    lines.append(rule)
    lines.append("")
    lines.append(
        "A drop under 0.02, with none of the kill conditions, is a null and the call is needs_edit. "
        "needs_edit also covers a BEAT whose interval, seed list, or paired counts do not all sit inside the bar. "
        "KILL if a ground state at n=6, 9, or 12 is a bitstring; if n=12 loses to Fourier and is not below 0.940; "
        "or if the L=16 Walsh advantage reverses (n=6 beats, n=9 is UNDER, n=12 R≥1.05)."
    )
    lines.append("")
    pp = cells["power_p|16|12"]
    lines.append(
        f"power_p control at n=12 L=16: R={pp['R']:.3f} ({pp['mark']}), "
        f"instance CI {payload['stats']['power_p|16|12']['ci']['lo']:.3f}–"
        f"{payload['stats']['power_p|16|12']['ci']['hi']:.3f}, "
        f"published μ=0 power_p R={pp['published_R']:.3f}, drop {pp['drop_vs_published']:+.3f}."
    )
    lines.append("")
    lines.append(
        f"Optimizer time summed over fits {payload['opt_s']:.1f}s. "
        f"Branch `{payload['branch']}`. "
        "`qaoa_angles/`, `train69_ladder_fresh250/`, `settings.py`, and the phase-2 n=21 ladder were not written."
    )
    lines.append("")
    return "\n".join(lines) + "\n"


def _public_cells(cells: dict) -> dict:
    return {f"{m}|{L}|{n}": cell for (m, L, n), cell in cells.items()}


def _public_stats(paired, ci, ci_seed) -> dict:
    out = {}
    for key, table in paired.items():
        name = f"{key[0]}|{key[1]}|{key[2]}"
        wlo, whi = wilson_interval(table["win"], table["n"])
        out[name] = {
            "paired": table,
            "ci": ci[key],
            "wilson": [wlo, whi],
        }
        if key in ci_seed:
            out[name]["ci_seed"] = ci_seed[key]
    return out


def write_all(out: Path, payload: dict):
    out.mkdir(parents=True, exist_ok=True)
    text = render(payload)
    (out / "RESULTS.md").write_text(text)
    (out / "summary.json").write_text(json.dumps(base._jsonable(payload), indent=2) + "\n")
    v = payload["verdict"]
    verdict = {
        "verdict": v["verdict"],
        "null": bool(v.get("null", False)),
        "reason": v["reason"],
        "R12": v.get("R12"),
        "R15": v.get("R15"),
        "ci12": v.get("ci12"),
        "ci15": v.get("ci15"),
        "prior_R12": PRIOR_R[12],
        "prior_R15": PRIOR_R[15],
        "published_R": PUBLISHED_N12_L16,
        "n18": payload["n18"],
        "mu": MU,
        "lam": LAM,
        "seeds": payload["seeds"],
        "B_hold": payload["B_hold"],
        "B_transfer": payload["B_transfer"],
        "B_n15": payload["B_n15"],
        "gap_coverage": payload.get("gap_coverage"),
        "gap_tails": _gap_tails_public(payload),
    }
    (out / "verdict.json").write_text(json.dumps(base._jsonable(verdict), indent=2) + "\n")
    log(f"wrote {out / 'RESULTS.md'} verdict={verdict['verdict']}")


def quantum_for_decide(quantum: dict) -> dict:
    role = {6: "hold", 9: "hold", 12: "transfer", 15: "transfer"}
    tail_keys = (
        "frac_gap_ge_0p02",
        "frac_gap_lt_0p02",
        "frac_gap_lt_0p05",
        "n_gap_lt_0p02",
        "n_gap_lt_0p05",
        "n_gap_ge_0p02",
        "n_inst",
        "gap_min",
    )
    out = {}
    for n in role:
        row = quantum[(n, role[n])]
        item = {
            "peak_med": float(row["peak_med"]),
            "prod_gap_med": float(row["prod_gap_med"]),
        }
        for key in tail_keys:
            if key in row:
                item[key] = row[key]
        out[n] = item
    return out


def _gap_tails_public(payload: dict) -> dict:
    out = {}
    for key, row in payload["quantum"].items():
        if "frac_gap_lt_0p02" not in row:
            continue
        out[key] = {
            "n_inst": int(row["n_inst"]),
            "n_gap_lt_0p02": int(row["n_gap_lt_0p02"]),
            "n_gap_lt_0p05": int(row["n_gap_lt_0p05"]),
            "frac_gap_lt_0p02": float(row["frac_gap_lt_0p02"]),
            "frac_gap_lt_0p05": float(row["frac_gap_lt_0p05"]),
            "frac_gap_ge_0p02": float(row["frac_gap_ge_0p02"]),
            "gap_min": float(row["gap_min"]),
            "prod_gap_med": float(row["prod_gap_med"]),
        }
    return out


def n18_for_decide(cells, quantum, status: dict) -> dict:
    if status.get("status") != "ok":
        return {"status": "infeasible" if status.get("status") != "skipped" else "skipped", "detail": status.get("detail", status.get("status"))}
    cell = cells.get(("walsh", 16, 18))
    meta = quantum[(18, "probe")]
    if cell is None:
        return {"status": "infeasible", "detail": "n=18 angles were not scored"}
    return {
        "status": "probe",
        "R": float(cell["R"]),
        "med": float(cell["med"]),
        "B": int(meta["B"]),
        "peak_med": float(meta["peak_med"]),
        "prod_gap_med": float(meta["prod_gap_med"]),
        "mark": cell["mark"],
        "probe_s": float(status.get("probe_s", meta.get("probe_s", 0.0))),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mu", type=float, default=-0.25)
    ap.add_argument("--lam", type=float, default=0.75)
    ap.add_argument("--out", type=str, default="")
    ap.add_argument("--seeds", type=str, default="0,1,2,3,4")
    ap.add_argument("--Ls", type=str, default="2,16")
    ap.add_argument("--B-train", type=int, default=128)
    ap.add_argument("--B-hold", type=int, default=256)
    ap.add_argument("--B-transfer", type=int, default=128)
    ap.add_argument("--B-n15", type=int, default=64)
    ap.add_argument("--B-n18", type=int, default=16)
    ap.add_argument("--n18-limit-s", type=float, default=90.0)
    ap.add_argument("--rounds-L2", type=int, default=18)
    ap.add_argument("--rounds-L16", type=int, default=48)
    ap.add_argument("--max-iter-L2", type=int, default=8)
    ap.add_argument("--max-iter-L16", type=int, default=8)
    ap.add_argument("--lr", type=float, default=0.35)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()
    apply_mu(float(args.mu))
    apply_lam(float(args.lam))
    if args.out == "":
        if abs(MU - (-1.00)) <= 1e-12 and abs(LAM - 1.00) <= 1e-12:
            args.out = str(S.OUT / "quantum_h_explore" / "unsat_lam100_mu100_seal")
        elif abs(MU - (-1.00)) <= 1e-12:
            args.out = str(S.OUT / "quantum_h_explore" / "unsat_mu100_seal")
        elif abs(MU - (-0.50)) <= 1e-12:
            args.out = str(S.OUT / "quantum_h_explore" / "unsat_mu050_seal")
        else:
            args.out = str(PARENT / "deepen")
    args.seeds = [int(s) for s in args.seeds.split(",") if s != ""]
    args.Ls = [int(s) for s in args.Ls.split(",") if s != ""]
    out = Path(args.out)
    blocked = ("phase2", "train69_ladder", "qaoa_angles")
    if any(part in blocked for part in out.parts) or out.name == "settings.py" or "n21" in out.parts:
        raise SystemExit("refusing to write into the n=21 ladder or frozen angle store")
    if abs(MU - (-0.50)) <= 1e-12 and out.name == "deepen" and "unsat_rank2_mu" in out.parts:
        raise SystemExit("refusing to overwrite the μ=−0.25 deepen directory")
    if abs(MU - (-1.00)) <= 1e-12 and (
        "unsat_mu050_seal" in out.parts or (out.name == "deepen" and "unsat_rank2_mu" in out.parts)
    ):
        raise SystemExit("refusing to overwrite the μ=−0.50 seal or the μ=−0.25 deepen directory")
    if abs(LAM - 1.00) <= 1e-12 and "unsat_mu100_seal" in out.parts:
        raise SystemExit("refusing to write the λ=1.00 seal into the λ=0.75 μ=−1.00 directory")
    if abs(LAM - 0.75) <= 1e-12 and "unsat_lam100_mu100_seal" in out.parts:
        raise SystemExit("refusing to write λ=0.75 into the λ=1.00 seal directory")
    if len(args.seeds) < 5 or len(set(args.seeds)) != len(args.seeds):
        raise SystemExit("deepen needs at least 5 distinct seeds")
    if args.B_train < 128 or args.B_hold < 256 or args.B_transfer < 128 or args.B_n15 < 64:
        raise SystemExit("B_train≥128, B_hold≥256, B_n12≥128, B_n15≥64")
    if sorted(args.Ls) != [2, 16]:
        raise SystemExit("L must be 2 and 16")
    torch.set_num_threads(4)
    log(f"out={out} μ={MU} λ={LAM} seeds={args.seeds} B_hold={args.B_hold}")
    t0 = time.time()
    sims, prepared, quantum = prepare_core(args, out)
    audit_rep = base.audit(prepared)
    n18_status = probe_n18(args, out, sims, prepared, quantum)
    have_n18 = n18_status.get("status") == "ok"
    log(f"n18 status={n18_status.get('status')}")
    runs = train(prepared, args, out, have_n18)
    ns = [6, 9, 12, 15] + ([18] if have_n18 else [])
    cells = aggregate(runs, ns)
    paired, ci, ci_seed = statistics(runs, cells, ns)
    qdec = quantum_for_decide(quantum)
    n18_note = n18_for_decide(cells, quantum, n18_status)
    # Gate reads only the sealed L=16 Walsh rows. power_p and L=2 stay in the table.
    gap_coverage = float(GAP_COVERAGE) if abs(float(MU) - (-1.00)) <= 1e-12 else None
    verdict = decide_deepen(
        cells,
        qdec,
        {k: paired[k] for k in paired if k[0] == "walsh" and k[1] == 16},
        {k: ci[k] for k in ci if k[0] == "walsh" and k[1] == 16},
        n18=n18_note,
        gap_coverage=gap_coverage,
    )
    prior = (
        f"{PRIOR_NOTE} point estimates were n=12 R={PRIOR_R[12]:.3f} and n=15 R={PRIOR_R[15]:.3f}."
    )
    verdict["reason"] = verdict["reason"].rstrip() + " " + prior
    log(f"VERDICT {verdict['verdict']}: {verdict['reason']}")
    scan = {}
    scan_ns = [(6, "hold"), (9, "hold"), (12, "transfer"), (15, "transfer")]
    if have_n18:
        scan_ns.append((18, "probe"))
    for n, role in scan_ns:
        scan[n] = base.cost_scan(prepared[(n, role)]["split"].circuit, n)
        log(f"cost scan n={n} γ={scan[n]['gamma']:.4f} med ε={scan[n]['med_eps']:.4f}")
    mass = walsh_mass_fractions(prepared[(6, "train")]["pulse"])
    stats_pub = _public_stats(paired, ci, ci_seed)
    payload = {
        "verdict": verdict,
        "quantum": {f"{n}:{role}": {k: v for k, v in row.items() if k != "seeds"} | {"n_seeds_formulas": len(row["seeds"])} for (n, role), row in quantum.items()},
        "quantum_decide": qdec,
        "cells": _public_cells(cells),
        "stats": stats_pub,
        "cost_scan": {str(k): v for k, v in scan.items()},
        "walsh_mass": mass,
        "audit": audit_rep,
        "n18": n18_note,
        "ns": ns,
        "B_train": args.B_train,
        "B_hold": args.B_hold,
        "B_transfer": args.B_transfer,
        "B_n15": args.B_n15,
        "B_n18": args.B_n18 if have_n18 else 0,
        "n18_limit_s": args.n18_limit_s,
        "seeds": args.seeds,
        "Ls": args.Ls,
        "rounds_L2": args.rounds_L2,
        "rounds_L16": args.rounds_L16,
        "max_iter_L2": args.max_iter_L2,
        "max_iter_L16": args.max_iter_L16,
        "lr": args.lr,
        "mu": MU,
        "lam": LAM,
        "opt_s": float(sum(float(r["wall_s"]) for r in runs)),
        "wall_s": time.time() - t0,
        "branch": BRANCH,
        "prior_R": {str(k): v for k, v in PRIOR_R.items()},
        "prior_note": PRIOR_NOTE,
        "round_name": ROUND_NAME,
        "gap_coverage": gap_coverage,
    }
    write_all(out, payload)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
